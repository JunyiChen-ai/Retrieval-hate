#!/usr/bin/env python3
import torch
from relation_v6.model import RelationV6,distribution_tokens
def main():
 torch.manual_seed(6); scores=torch.randn(2,11,3); valid=torch.tensor([[1]*11,[1]*7+[0]*4],dtype=torch.bool); model=RelationV6(3,hidden=16,heads=4,window=2,dropout=0.).eval(); out=model(scores,valid)
 assert out["frame_logit"].shape==(2,11); assert out["distribution_tokens"].shape==(2,3,6); assert torch.equal(out["frame_prob"][1,7:],torch.zeros(4))
 # Distribution prior tokens are invariant to temporal permutation.
 perm=torch.randperm(11); assert torch.allclose(distribution_tokens(scores[:1],valid[:1]),distribution_tokens(scores[:1,perm],valid[:1,perm]),atol=1e-6,rtol=1e-6)
 # Locator is exactly invariant to per-video/per-expert constant offsets.
 shifted=scores+torch.tensor([[[3.,-2.,.7]],[[.4,8.,-5.]]]); shifted=shifted*valid[...,None]; so=model(shifted,valid); assert torch.allclose(out["locator_logit"],so["locator_logit"],atol=2e-6,rtol=2e-6)
 assert torch.allclose((out["locator_logit"]*valid).sum(1),torch.zeros(2),atol=1e-6)
 idx=torch.arange(11); illegal=((idx[:,None]-idx[None,:]).abs()>2)[None,None,None]|(~valid[:,None,None,:,None])|(~valid[:,None,None,None,:]); assert torch.equal(out["transport"][illegal.expand_as(out["transport"])],torch.zeros_like(out["transport"][illegal.expand_as(out["transport"])]))
 loss=out["prior_logit"].sum()+out["locator_logit"].square().sum(); loss.backward(); assert model.prior.readout.weight.grad is not None; assert model.locator.readout.weight.grad is not None
 print("Relation-V6 smoke: PASS")
if __name__=="__main__": main()
