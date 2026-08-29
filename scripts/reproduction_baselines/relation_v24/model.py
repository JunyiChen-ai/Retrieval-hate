#!/usr/bin/env python3
"""Small auditable V24 model; contains no temporal-label data access."""
import torch
from torch import nn

def exact_center(x):
    x=torch.as_tensor(x,dtype=torch.float64)
    return x-x.mean()

def cardinality_stable_lme(x,tau=1.0):
    """tau*log(mean(exp(x/tau))); exact under whole-bag replication."""
    x=torch.as_tensor(x,dtype=torch.float64)
    if x.ndim!=1 or not len(x):raise ValueError('nonempty 1D bag required')
    return tau*(torch.logsumexp(x/tau,dim=0)-torch.log(torch.tensor(float(len(x)),dtype=x.dtype,device=x.device)))

def dedup_mean(channels):
    """Exact duplicate experts inside one family count once."""
    unique=[]
    for x in channels:
        z=torch.as_tensor(x,dtype=torch.float64)
        if not any(torch.equal(z,q) for q in unique):unique.append(z)
    if not unique:raise ValueError('empty family')
    if len({len(x) for x in unique})!=1:raise ValueError('unaligned family channels')
    return torch.stack(unique).mean(0)

class V24(nn.Module):
    def __init__(self,families=('text','multimodal'),tau=1.0):
        super().__init__();self.families=tuple(families);self.tau=float(tau)
        self.family_logits=nn.Parameter(torch.zeros(len(families),dtype=torch.float64))
        # Residual parameterization makes epoch-0 an exact V23-global fallback.
        self.global_delta=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.global_bias=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.gamma=nn.Parameter(torch.zeros((),dtype=torch.float64))
    def local(self,family_channels):
        vals=[]
        for name in self.families:vals.append(exact_center(dedup_mean(family_channels[name])))
        w=torch.softmax(self.family_logits,0);return sum(w[i]*vals[i] for i in range(len(vals)))
    def forward(self,global_score,family_channels):
        g=torch.as_tensor(global_score,dtype=torch.float64)
        loc=self.local(family_channels);bag=cardinality_stable_lme(loc,self.tau)
        video_logit=(1+self.global_delta)*g+self.global_bias+torch.clamp(self.gamma,min=0)*bag
        frame_scores=(1+self.global_delta)*g+self.global_bias+torch.clamp(self.gamma,min=0)*loc
        return video_logit,frame_scores
    def project_(self):
        # Small calibrator guardrail, not a search range.
        with torch.no_grad():self.global_delta.clamp_(-.5,.5);self.global_bias.clamp_(-2,2);self.gamma.clamp_(0,2)
