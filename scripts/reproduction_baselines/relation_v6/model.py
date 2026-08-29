"""Two-level expert reliability: video prior plus centered localization."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def distribution_tokens(scores,valid,topk_divisor=16):
 """Dataset-agnostic per-expert distribution summaries [B,E,6]."""
 tokens=[]
 for b in range(scores.shape[0]):
  x=scores[b,valid[b]]
  if len(x)==0: raise ValueError("empty video")
  k=max(1,len(x)//topk_divisor+1)
  token=torch.stack([x.mean(0),x.std(0,unbiased=False),
                     torch.quantile(x,.1,dim=0),torch.quantile(x,.5,dim=0),
                     torch.quantile(x,.9,dim=0),x.topk(k,dim=0).values.mean(0)],-1)
  tokens.append(token)
 return torch.stack(tokens)


class DistributionPrior(nn.Module):
 def __init__(self,hidden=32,heads=4,dropout=.1):
  super().__init__(); self.token=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.LayerNorm(hidden)); self.relation=nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True); self.norm=nn.LayerNorm(hidden); self.readout=nn.Linear(hidden,1)
 def forward(self,tokens):
  x=self.token(tokens); related,_=self.relation(x,x,x,need_weights=False); return self.readout(self.norm(x+related).mean(1)).squeeze(-1)


class CenteredTransportLocator(nn.Module):
 """Local cross-expert transport with no access to video-level means."""
 def __init__(self,n_experts,hidden=32,window=12,temperature=.2,dropout=.1):
  super().__init__(); self.n_experts=n_experts; self.window=window; self.temperature=temperature; width=n_experts+3*n_experts*n_experts; self.temporal=nn.Sequential(nn.Conv1d(width,hidden,3,padding=1),nn.GELU(),nn.Dropout(dropout),nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU()); self.readout=nn.Conv1d(hidden,1,1)
 def forward(self,centered,valid):
  b,t,e=centered.shape; change=F.pad(centered[:,1:]-centered[:,:-1],(0,0,1,0)).permute(0,2,1); idx=torch.arange(t,device=centered.device); legal=(idx[:,None]-idx[None,:]).abs()<=self.window; legal=legal[None,None,None]&valid[:,None,None,:,None]&valid[:,None,None,None,:]
  affinity=-(change[:,:,None,:,None]-change[:,None,:,None,:]).square()/self.temperature; affinity=affinity.masked_fill(~legal,-torch.inf); transport=torch.softmax(affinity,-1); transport=torch.nan_to_num(transport,nan=0.)*legal; aligned=torch.einsum("bqets,bse->btqe",transport,centered); lag=(idx[None,:]-idx[:,None]).to(centered.dtype); expected_lag=torch.einsum("bqets,ts->btqe",transport,lag)/max(1,self.window); target=centered[:,:,:,None].expand_as(aligned); relation=torch.cat([aligned,target-aligned,expected_lag],-1).reshape(b,t,-1); feature=torch.cat([centered,relation],-1)*valid[...,None]; hidden=self.temporal(feature.transpose(1,2)).transpose(1,2)*valid[...,None]; raw=self.readout(hidden.transpose(1,2)).squeeze(1)*valid
  # Exact centering prevents this branch from carrying video identity/prior.
  mean=(raw.sum(1)/valid.sum(1).clamp_min(1))[:,None]; locator=(raw-mean)*valid
  return locator,transport,aligned


class RelationV6(nn.Module):
 def __init__(self,n_experts,hidden=32,heads=4,window=12,temperature=.2,dropout=.1,topk_divisor=16):
  super().__init__(); self.topk_divisor=topk_divisor; self.prior=DistributionPrior(hidden,heads,dropout); self.locator=CenteredTransportLocator(n_experts,hidden,window,temperature,dropout)
 def forward(self,scores,valid,locator_scale=1.):
  tokens=distribution_tokens(scores,valid,self.topk_divisor); prior=self.prior(tokens); denom=valid.sum(1).clamp_min(1)[...,None]; mean=(scores*valid[...,None]).sum(1)/denom; centered=(scores-mean[:,None])*valid[...,None]; locator,transport,aligned=self.locator(centered,valid); final=(prior[:,None]+float(locator_scale)*locator)*valid
  return {"frame_logit":final,"frame_prob":torch.sigmoid(final)*valid,"prior_logit":prior,"locator_logit":locator,"distribution_tokens":tokens,"centered_experts":centered,"transport":transport,"aligned_centered":aligned}
