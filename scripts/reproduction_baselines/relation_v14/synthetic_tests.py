#!/usr/bin/env python3
import numpy as np
from relation_v14.run import add,locator,time_shuffle
values={'a':{'v':np.array([0.,1.,0.])},'copy':{'v':np.array([0.,1.,0.])},'b':{'v':np.array([1.,0.,1.])}}
groups=[['a','copy'],['b']];r=locator(values,groups,np.array([.5,.5]));assert abs(r['v'].mean())<1e-15
base={'v':np.array([.2,.3,.4])};assert np.array_equal(add(base,r,0)['v'],base['v']);assert not np.array_equal(time_shuffle({'v':np.arange(5.)})['v'],np.arange(5.))
print({'zero_mean':'PASS','beta0_exact_identity':'PASS','time_shuffle':'PASS'})
