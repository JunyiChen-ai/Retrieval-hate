"""Fail-closed expert score loading for Relation-V4."""
import hashlib,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from hate_common import data as hdata
def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
 return h.hexdigest()
def load_manifest(path):
 m=json.load(open(path)); required={"corpus","experts","static_weights"}
 if not required<=set(m): raise ValueError("manifest missing required fields")
 if len(m["experts"])<2 or len(m["experts"])!=len(m["static_weights"]): raise ValueError("expert/weight size mismatch")
 return m
def load_split(m,split):
 gt=hdata.gt_arrays(m["corpus"],split); experts=[]; provenance=[]
 for e in m["experts"]:
  paths=e[f"{split}_scores"]; paths=[paths] if isinstance(paths,str) else paths; key=e["score_key"]; runs=[]
  for path in paths:
   records=hdata.load_scores_jsonl(path)
   if not set(gt)<=set(records): raise RuntimeError(f"{e['name']} {split} missing frozen-GT videos")
   runs.append({v:np.asarray(records[v][key],dtype=np.float32) for v in gt})
  score={v:np.stack([run[v] for run in runs]).mean(0) for v in gt}
  for v in gt:
   if score[v].shape!=gt[v].shape or not np.isfinite(score[v]).all(): raise RuntimeError(f"{e['name']} alignment/nonfinite {v}")
  experts.append(score); provenance.append({"name":e["name"],"paths":[os.path.abspath(p) for p in paths],"score_key":key,"sha256":[sha256(p) for p in paths],"aggregation":"raw_score_mean"})
 return {v:np.stack([x[v] for x in experts],-1) for v in gt},gt,provenance

def fit_ecdf(scores):
 """Fit one frozen validation empirical CDF per expert."""
 values=np.concatenate([scores[v] for v in sorted(scores)],0)
 return [np.sort(values[:,e]).astype(float).tolist() for e in range(values.shape[1])]

def apply_ecdf(scores,calibration):
 out={}; calibration=[np.asarray(x,float) for x in calibration]
 for vid,value in scores.items():
  rank=np.empty_like(value,dtype=np.float32)
  for e,ref in enumerate(calibration):
   left=np.searchsorted(ref,value[:,e],side="left"); right=np.searchsorted(ref,value[:,e],side="right")
   rank[:,e]=(left+right-1)/(2.*max(1,len(ref)-1))
  out[vid]=np.clip(rank,0.,1.)
 return out
