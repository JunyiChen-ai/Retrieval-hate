"""Shared hierarchical decomposition and scoring."""
import numpy as np
def hierarchical(scores,prior_weights,residual_weights,amplitude):
 out={}; wp=np.asarray(prior_weights,float); wr=np.asarray(residual_weights,float)
 for vid,x in scores.items():
  mean=x.mean(0); centered=x-mean[None]; prior=float(mean@wp); residual=centered@wr; out[vid]=prior+float(amplitude)*residual
 return out
def simplex(n_experts,step=.1):
 n=round(1./step); out=[]
 def visit(prefix,remaining,slots):
  if slots==1: out.append([(x/n) for x in prefix+[remaining]]); return
  for value in range(remaining+1): visit(prefix+[value],remaining-value,slots-1)
 visit([],n,n_experts); return out
