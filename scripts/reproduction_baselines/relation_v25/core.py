#!/usr/bin/env python3
"""Frozen mathematical core for Relation-V25."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn

SEEDS=(234,2025,3407); RHO=.20; TAU=.5; EPS=1e-4
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canon_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fold(vid):return int(hashlib.sha256(vid.encode()).hexdigest()[:8],16)%5
def ecdf_logit(x,ref):
 r=np.asarray(ref,dtype=np.float64)
 if r.ndim!=1 or not len(r) or not np.isfinite(r).all():raise ValueError('invalid reference')
 a=np.searchsorted(r,x,'left');b=np.searchsorted(r,x,'right');f=(a+.5*(b-a)+.5)/(len(r)+1);f=np.clip(f,EPS,1-EPS)
 return np.log(f/(1-f))
def fractional_lme(x,rho=RHO,tau=TAU):
 x=torch.as_tensor(x,dtype=torch.float64)
 if x.ndim!=1 or not len(x) or not torch.isfinite(x).all():raise ValueError('invalid bag')
 z=torch.sort(x,descending=True).values;q=rho*len(z);m=int(math.floor(q));a=q-m
 num=torch.exp(z[:m]/tau).sum() if m else z.new_zeros(())
 if a:num=num+a*torch.exp(z[m]/tau)
 return tau*torch.log(num/q)
def reduce_1hz(windows,duration):
 if not math.isfinite(duration) or duration<=0:raise ValueError('duration')
 n=math.ceil(duration);scores=[];mask=[]
 for j in range(n):
  u=min(j+.5,duration-1e-9);hits=[float(w['logit']) for w in windows if w['start']<=u<w['end'] or (w['end']==duration and u==duration)]
  if hits:scores.append(float(torch.sigmoid(torch.tensor(sum(hits)/len(hits))).item()));mask.append(1)
  else:scores.append(float('nan'));mask.append(0)
 return scores,mask
class V25(nn.Module):
 def __init__(self,learn_local=True):
  super().__init__();self.b=nn.Parameter(torch.zeros((),dtype=torch.float64),requires_grad=learn_local);self.sraw=nn.Parameter(torch.tensor(math.log(math.e-1),dtype=torch.float64),requires_grad=learn_local);self.wraw=nn.Parameter(torch.zeros(2,dtype=torch.float64),requires_grad=learn_local);self.delta=nn.Parameter(torch.zeros((),dtype=torch.float64));self.c=nn.Parameter(torch.zeros((),dtype=torch.float64));self.gamma=nn.Parameter(torch.zeros((),dtype=torch.float64))
 def local(self,z):return self.b+torch.nn.functional.softplus(self.sraw)*(torch.softmax(self.wraw,0)[:,None]*z).sum(0)
 def forward(self,g,z):
  ell=self.local(z);return (1+self.delta)*torch.as_tensor(g,dtype=torch.float64)+self.c+torch.clamp(self.gamma,0,2)*fractional_lme(ell),ell
 def project(self):
  with torch.no_grad():self.delta.clamp_(-.5,.5);self.c.clamp_(-2,2);self.gamma.clamp_(0,2)
def loss_one(model,g,z,y):
 L,ell=model(g,z);p=torch.sigmoid(ell);loss=torch.nn.functional.binary_cross_entropy_with_logits(L,torch.tensor(float(y),dtype=torch.float64))
 if not y:loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(ell,torch.zeros_like(ell)).mean()
 else:
  q=RHO*len(ell);m=int(math.floor(q));a=q-m;idx=torch.argsort(ell,descending=True);weights=torch.ones(m+(a>0),dtype=torch.float64);weights[-1]=a if a else weights[-1];r=torch.softmax(ell[idx[:len(weights)]]/TAU+torch.log(weights),0);loss=loss+1e-3*(-(r*torch.log(r+1e-12)).sum())
 if len(p)>1:loss=loss+1e-3*torch.abs(p[1:]-p[:-1]).mean()
 return loss
