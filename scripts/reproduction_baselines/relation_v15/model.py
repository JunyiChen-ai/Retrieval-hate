import torch
from torch import nn

class EmissionMIL(nn.Module):
 def __init__(self,dims):
  super().__init__();self.heads=nn.ModuleDict({k:nn.Linear(d,1) for k,d in dims.items()})
  for h in self.heads.values():nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
 def forward(self,features):
  logits=[self.heads[k](x).squeeze(-1) for k,x in features.items() if k in self.heads]
  if not logits:raise RuntimeError('all modalities missing')
  return torch.stack(logits).mean(0)

def exact_bag_nll(logits,label):
 """Independent latent frame mixture: negative=all z=0; positive=any z=1."""
 # Cardinality-corrected rare-state prior: with zero emission evidence each
 # frame has event probability 1/(T+1), so bag probability stays O(1) as T
 # varies instead of saturating exponentially with duration.
 z=logits.double()-torch.log(torch.as_tensor(float(logits.numel()),dtype=torch.float64,device=logits.device))
 log_all_bg=torch.nn.functional.logsigmoid(-z).sum(dtype=torch.float64)
 if not label:return -log_all_bg
 # log(1-exp(log_all_bg)), stable and finite.
 return -torch.log(-torch.expm1(torch.clamp(log_all_bg,max=-1e-15)))
