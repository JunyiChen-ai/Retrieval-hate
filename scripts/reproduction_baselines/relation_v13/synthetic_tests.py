#!/usr/bin/env python3
import numpy as np
from relation_v13.diagnostic import reliability,collapse_exact
r=np.random.default_rng(3);signal=r.normal(size=200);x=np.stack([signal+r.normal(scale=.1,size=200) for _ in range(3)],1);y=np.stack([signal+r.normal(scale=2,size=200) for _ in range(3)],1)
assert reliability(x)['icc_mean']>reliability(y)['icc_mean']
vals={'a':{'v':x[:,0]},'copy':{'v':x[:,0].copy()},'b':{'v':y[:,0]}}
assert collapse_exact(vals,['v'])==[['a','copy'],['b']]
print({'known_snr_order':'PASS','exact_duplicate_collapse':'PASS'})
