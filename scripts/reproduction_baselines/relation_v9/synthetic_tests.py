#!/usr/bin/env python3
import numpy as np,torch
from relation_v9.model import DependenceAwareRelation,dependence_weights,estimate_lag,shift_sequence
from relation_v9.loss import weak_loss

def main():
 rng=np.random.default_rng(3);a=rng.normal(size=400);b=rng.normal(size=400);x=np.stack([a,a,b],1);w,clusters,_=dependence_weights(x)
 assert clusters==[[0,1],[2]] and np.allclose(w,[.25,.25,.5])
 base=torch.zeros(30);base[8:14]=1;delayed=shift_sequence(base,4);assert estimate_lag(base,delayed,6)==-4
 scores=torch.tensor(np.stack([base.numpy(),delayed.numpy(),rng.random(30)],1),dtype=torch.float32)[None];valid=torch.ones(1,30,dtype=torch.bool);model=DependenceAwareRelation(3,clusters,hidden=8,window=6)
 out=model(scores,valid);assert torch.equal(out["frame_logit"],out["base_logit"]);assert torch.allclose(out["locator_logit"].mean(1),torch.zeros(1),atol=1e-6)
 keep=torch.tensor([1.,0.]);noise=.01*torch.randn(1,30,2);corrupt=model(scores,valid,keep,noise);loss,parts=weak_loss(out,corrupt,valid,torch.ones(1));assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
 intercept=scores+torch.tensor([.1,.1,.1]);shifted=model(intercept,valid);assert torch.allclose(out["base_locator"],shifted["base_locator"],atol=1e-6)
 # Full-forward structure invariance: duplicating stream 0 changes neither
 # learned token/attention nor locator input because aggregation happens first.
 torch.manual_seed(11);unique=DependenceAwareRelation(2,[[0],[1]],hidden=8,window=6)
 torch.manual_seed(11);duplicate=DependenceAwareRelation(3,[[0,1],[2]],hidden=8,window=6)
 unique.train();duplicate.train();u=torch.stack([base,delayed],-1)[None];d=torch.stack([base,base,delayed],-1)[None]
 uo=unique(u,valid);do=duplicate(d,valid)
 for key in ("frame_logit","prior_logit","locator_logit","prior_delta","locator_delta"):
  assert torch.allclose(uo[key],do[key],atol=1e-7),key
 cluster_noise=.01*torch.randn(1,30,2);cluster_keep=torch.tensor([1.,0.])
 uc=unique(u,valid,cluster_keep,cluster_noise);dc=duplicate(d,valid,cluster_keep,cluster_noise)
 for key in ("frame_logit","prior_logit","locator_logit","prior_delta","locator_delta"):
  assert torch.allclose(uc[key],dc[key],atol=1e-7),"corrupt "+key
 uloss,_=weak_loss(uo,uc,valid,torch.ones(1));dloss,_=weak_loss(do,dc,valid,torch.ones(1));assert torch.allclose(uloss,dloss,atol=1e-7)
 unique.zero_grad();duplicate.zero_grad();uloss.backward();dloss.backward()
 for name in ("prior_scale","locator_scale","token.0.weight","locator.0.weight"):
  ug=dict(unique.named_parameters())[name].grad;dg=dict(duplicate.named_parameters())[name].grad
  assert torch.allclose(ug,dg,atol=1e-7),"gradient "+name
 # A dropped cluster is absent from attention, pooling, locator, and loss.
 dropped=torch.tensor([1.,0.]);noise2=.01*torch.randn(1,30,2);changed=u.clone();changed[...,1]=1-changed[...,1]
 c1=unique(u,valid,dropped,noise2);c2=unique(changed,valid,dropped,noise2)
 for key in ("frame_logit","prior_logit","locator_logit","prior_delta","locator_delta"):
  assert torch.allclose(c1[key],c2[key],atol=1e-7),"dropped leakage "+key
 l1,_=weak_loss(uo,c1,valid,torch.ones(1));l2,_=weak_loss(uo,c2,valid,torch.ones(1));assert torch.allclose(l1,l2,atol=1e-7)
 all_dropped=unique(u,valid,torch.zeros(2),torch.zeros(1,30,2))
 no_dropout=unique(u,valid)
 for key in ("frame_logit","prior_logit","locator_logit"):
  assert torch.allclose(all_dropped[key],no_dropout[key],atol=1e-7),"all-drop fallback "+key
 print("Relation-V9 synthetic tests: PASS")
if __name__=="__main__":main()
