#!/usr/bin/env python3
import numpy as np
from v22_speech_mask_components import project,calibrate,fuse,shuffle_observed

def row(s,e,m,c):return {'start':s,'end':e,'scores':{'masked_branch_reset':m,'causal_continuous':c}}
def main():
 g,l,m=project([row(1.1,2.7,2,4),row(4.0,4.5,8,6)],6)
 assert g==5 and m.tolist()==[False,True,True,False,True,False]
 assert np.array_equal(l[~m],np.zeros(3)) and abs(l[m].mean())<1e-15
 missing={'available':False,'global_raw':None,'local_raw':np.zeros(6),'speech_mask':np.zeros(6,dtype=bool)}
 observed={'available':True,'global_raw':g,'local_raw':l,'speech_mask':m}
 state=calibrate([observed,missing])
 assert not missing['available'] and np.array_equal(missing['local_calibrated'],np.zeros(6))
 assert abs(observed['local_calibrated'][m].mean())<1e-15 and np.array_equal(observed['local_calibrated'][~m],np.zeros(3))
 # Missing remains explicit metadata, never interpreted as a negative target.
 assert 'available' in missing and missing['available'] is False
 a=fuse(observed['global_calibrated'],observed['local_calibrated'],2,0)
 assert np.array_equal(a,2*observed['global_calibrated']) # beta=0 exact global-only
 for seed in range(10):
  z=shuffle_observed(observed['local_calibrated'],m,seed)
  assert np.array_equal(z[~m],np.zeros(3)) and sorted(z[m])==sorted(observed['local_calibrated'][m])
 print('V22 speech-mask property tests PASS')
if __name__=='__main__':main()
