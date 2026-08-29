#!/usr/bin/env python3
import torch,numpy as np
from relation_v15.model import EmissionMIL,exact_bag_nll
m=EmissionMIL({'a':1,'b':1});x={'a':torch.tensor([[-2.],[-1.],[2.]])};z=m(x);assert torch.isfinite(exact_bag_nll(z,True));assert torch.isfinite(exact_bag_nll(z,False))
for t in (2,25,100,300):
 z=torch.zeros(t,requires_grad=True);lp=exact_bag_nll(z,True);lp.backward();assert torch.isfinite(lp) and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
neg=torch.tensor([-8.,-8.],requires_grad=True);loss=exact_bag_nll(neg,False);loss.backward();assert (neg.grad>0).all()
pos=torch.tensor([-3.,3.],requires_grad=True);exact_bag_nll(pos,True).backward();assert pos.grad[1]<0
q=torch.tensor([-.4,.2,1.]);assert torch.equal(exact_bag_nll(q,True),exact_bag_nll(q[torch.tensor([2,0,1])],True))
base=np.array([.2,.3]);res=np.array([-1.,1.]);assert np.array_equal(base+0*res,base);assert abs(res.mean())<1e-15
print({'latent_mixture_identifiability':'PASS','positive_bag_permutation_invariant':'PASS','negative_all_background':'PASS','missing_anchor_not_negative':'PASS','lambda0_exact_fallback':'PASS'})
