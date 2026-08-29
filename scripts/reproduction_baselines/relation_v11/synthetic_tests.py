#!/usr/bin/env python3
import numpy as np
from relation_v11.robust import fit,huber_barycenter
def main():
 rng=np.random.default_rng(11);x=rng.normal(size=(400,3));base=huber_barycenter(x,fit(x))
 for column in range(3):
  duplicated=np.column_stack([x,x[:,column]])
  assert np.max(np.abs(base-huber_barycenter(duplicated,fit(duplicated))))<1e-10
 constant=np.column_stack([x,np.ones(400),np.ones(400)]);single=np.column_stack([x,np.ones(400)])
 assert np.max(np.abs(huber_barycenter(constant,fit(constant))-huber_barycenter(single,fit(single))))<1e-10
 curve=[]
 for eps in (0,.001,.01,.05,.1,.25):
  contaminated=np.column_stack([x,x[:,0]+eps*rng.normal(size=len(x))]);pred=huber_barycenter(contaminated,fit(contaminated));curve.append({'epsilon':eps,'mean_abs_change':float(np.mean(np.abs(pred-base)))})
 print({'replication_invariance':'PASS','near_duplicate_contamination_curve':curve,'claim':'empirical stability only'})
if __name__=='__main__':main()
