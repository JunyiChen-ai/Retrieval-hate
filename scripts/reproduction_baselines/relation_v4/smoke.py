#!/usr/bin/env python3
import torch
from relation_v4.model import AnalyticExpertRelationGate,ExpertRelationGate,masked_rank
def main():
 torch.manual_seed(4); scores=torch.randn(2,9,3); valid=torch.tensor([[1]*9,[1]*6+[0]*3],dtype=torch.bool); m=ExpertRelationGate(3,[.2,.3,.5],hidden=16,window=2).eval(); o=m(scores,valid)
 assert torch.equal(o["frame_score"],o["static_score"])
 expected=(masked_rank(scores,valid)*torch.tensor([.2,.3,.5])).sum(-1)*valid
 assert torch.equal(o["frame_score"],expected)
 assert torch.equal(o["frame_score"][1,6:],torch.zeros(3))
 idx=torch.arange(9); illegal=((idx[:,None]-idx[None,:]).abs()>2)[None,None,None]|(~valid[:,None,None,:,None])|(~valid[:,None,None,None,:])
 assert torch.equal(o["transport"][illegal.expand_as(o["transport"])],torch.zeros_like(o["transport"][illegal.expand_as(o["transport"])]))
 loss=o["frame_score"].sum(); loss.backward(); assert m.correction_scale.grad is not None; assert m.gate.weight.grad is not None; assert torch.equal(m.gate.weight.grad,torch.zeros_like(m.gate.weight.grad))
 # Once scale moves, relation gate receives a usable gradient.
 with torch.no_grad(): m.correction_scale.fill_(.1)
 m.zero_grad(); m(scores,valid)["frame_score"].sum().backward(); assert float(m.gate.weight.grad.abs().sum())>0
 a=AnalyticExpertRelationGate(3,[.2,.3,.5],beta=0.,gamma=0.,window=2).eval(); ao=a(scores,valid)
 assert torch.equal(ao["frame_score"],ao["static_score"])
 b=AnalyticExpertRelationGate(3,[.2,.3,.5],beta=2.,gamma=.2,window=2).eval(); bo=b(scores,valid)
 assert bool((bo["rank_correction"][valid].abs()>0).any())
 print("Relation-V4 smoke: PASS")
if __name__=="__main__": main()
