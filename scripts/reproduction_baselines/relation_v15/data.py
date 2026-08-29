import json
from pathlib import Path
import numpy as np,torch

MODS={'clip':'clip_b16_1fps','vggish':'vggish_1s','i3d':'i3d_rgb_5crop'}
DIMS={'clip':512,'vggish':128,'i3d':1024}
def align(x,t,times=None):
 if x.ndim==3:x=x.mean(1)
 if len(x)==t:return x
 if len(x)==0:return np.zeros((t,x.shape[-1]),np.float32)
 if times is None:raise RuntimeError(f'non-I3D frozen timeline mismatch {len(x)} != {t}')
 centers=np.asarray([(a+b)/2 for a,b in times]);target=np.arange(t)+.5;idx=np.abs(centers[:,None]-target[None]).argmin(0);return x[idx]
class FrozenFeatures:
 def __init__(self,root,corpus):
  self.root=Path(root);self.corpus=corpus;self.index=json.load(open(self.root/'results/reproduction/features/vggish_1s'/corpus/'index.json'))
 def load(self,vid):
  if vid not in self.index:raise RuntimeError(f'{vid} missing frozen 1fps timeline')
  t=int(self.index[vid]['n_frames']);out={};missing=[]
  for name,folder in MODS.items():
   path=self.root/'results/reproduction/features'/folder/self.corpus/f'{vid}.npy'
   if not path.is_file():missing.append(name);continue
   x=np.asarray(np.load(path),np.float32)
   if not np.isfinite(x).all():raise RuntimeError(f'nonfinite {path}')
   times=None
   if name=='i3d':
    tp=path.with_suffix('.times.json')
    if not tp.is_file():raise RuntimeError(f'I3D missing frozen times {tp}')
    times=json.load(open(tp))['times']
    if len(times)!=len(x):raise RuntimeError(f'I3D time/feature mismatch {vid}')
   x=align(x,t,times);norm=np.linalg.norm(x,axis=1,keepdims=True);x=x/np.maximum(norm,1e-6);out[name]=torch.from_numpy(x.astype(np.float32))
  if not out:raise RuntimeError(f'{vid} all modalities missing')
  return out,missing,t
