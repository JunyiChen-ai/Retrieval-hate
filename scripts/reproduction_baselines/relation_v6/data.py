"""Strict split-bound dense expert loading and optional K16 targets."""
import json,os,sys
import numpy as np,torch
from torch.utils.data import Dataset
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits,scoped_labels
from relation_v4.io import sha256
def records(path): return hdata.load_scores_jsonl(path)
def load_dense(manifest,split):
 ids=frozen_splits(manifest["corpus"])[split]; all_experts=[]; provenance=[]
 for expert in manifest["experts"]:
  paths=expert[f"{split}_scores"]; paths=[paths] if isinstance(paths,str) else paths; runs=[records(p) for p in paths]; key=expert["score_key"]
  for run in runs:
   if not set(ids)<=set(run): raise RuntimeError(f"{expert['name']} missing frozen {split} IDs")
  score={v:np.stack([np.asarray(run[v][key],np.float32) for run in runs]).mean(0) for v in ids}; all_experts.append(score); provenance.append({"name":expert["name"],"paths":[os.path.abspath(p) for p in paths],"sha256":[sha256(p) for p in paths],"score_key":key})
 dense={}
 for vid in ids:
  lengths={len(x[vid]) for x in all_experts}
  if len(lengths)!=1: raise RuntimeError(f"expert length mismatch {vid}: {lengths}")
  dense[vid]=np.stack([x[vid] for x in all_experts],-1)
 return ids,dense,provenance
def fit_calibration(dense):
 x=np.concatenate([dense[v] for v in sorted(dense)],0); return [np.sort(x[:,e]).astype(float).tolist() for e in range(x.shape[1])]
def calibrate(x,calibration):
 out=np.empty_like(x,np.float32)
 for e,values in enumerate(calibration):
  ref=np.asarray(values,float); left=np.searchsorted(ref,x[:,e],"left"); right=np.searchsorted(ref,x[:,e],"right"); out[:,e]=np.clip((left+right-1)/(2*max(1,len(ref)-1)),0,1)
 return out
def sparse_target(root,vid,n):
 target=np.zeros(n,np.float32); count=np.zeros(n,np.float32); path=os.path.join(root,vid+".json")
 if not os.path.exists(path): return target,count.astype(bool)
 payload=json.load(open(path))
 if payload.get("video_id")!=vid: raise RuntimeError("K16 video ID mismatch")
 for segment in payload.get("segments",[]):
  start,end,score=float(segment["start"]),float(segment["end"]),float(segment["score"]); index=np.arange(n); use=(index>=start)&(index<end); target[use]+=score; count[use]+=1
 mask=count>0; target[mask]/=count[mask]; return np.clip(target,0,1),mask
class ExpertDataset(Dataset):
 def __init__(self,ids,dense,calibration,labels=None,sparse_root=None,gt=None): self.ids=ids; self.dense=dense; self.calibration=calibration; self.labels=labels; self.sparse_root=sparse_root; self.gt=gt
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  vid=self.ids[i]; x=calibrate(self.dense[vid],self.calibration); n=len(x); teacher,teacher_mask=(sparse_target(self.sparse_root,vid,n) if self.sparse_root else (np.zeros(n,np.float32),np.zeros(n,bool))); label=(-1 if self.labels is None else self.labels[vid]); gold=(np.zeros(n,np.uint8) if self.gt is None else self.gt[vid]); return vid,torch.from_numpy(x),int(label),torch.from_numpy(teacher),torch.from_numpy(teacher_mask),torch.from_numpy(gold)
def collate(batch):
 vids=[x[0] for x in batch]; lengths=torch.tensor([len(x[1]) for x in batch]); width=int(lengths.max()); e=batch[0][1].shape[1]; scores=torch.zeros(len(batch),width,e); teacher=torch.zeros(len(batch),width); tm=torch.zeros(len(batch),width,dtype=torch.bool); gold=torch.zeros(len(batch),width,dtype=torch.uint8); labels=torch.tensor([x[2] for x in batch],dtype=torch.float32)
 for i,(_,x,_,y,m,g) in enumerate(batch): n=len(x); scores[i,:n]=x; teacher[i,:n]=y; tm[i,:n]=m; gold[i,:n]=g
 valid=torch.arange(width)[None]<lengths[:,None]; return vids,scores,valid,lengths,labels,teacher,tm,gold
def build(manifest,split,calibration=None):
 ids,dense,prov=load_dense(manifest,split)
 if calibration is None: calibration=fit_calibration(dense)
 labels=None; sparse=None; gt=None
 if split=="train": labels,_=scoped_labels(manifest["corpus"],"train"); sparse=manifest.get("train_sparse_root")
 else:
  gt=hdata.gt_arrays(manifest["corpus"],split); ids=[v for v in ids if v in gt]
  for v in ids:
   if len(dense[v])!=len(gt[v]): raise RuntimeError(f"GT alignment mismatch {v}")
 return ExpertDataset(ids,dense,calibration,labels,sparse,gt),calibration,prov
