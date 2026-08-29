"""Fail-closed split, timeline, source, and K16 loading for V6."""
from relation_v6.data import *  # noqa: F401,F403
import hashlib,json,math,os
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256
VGG_INDEX=os.path.join(hdata.REPO_ROOT,"results","reproduction","features","vggish_1s","hateclipseg","index.json")
_base_load_dense=load_dense
def _source_meta(path):
 parent=os.path.dirname(path); seed=os.path.dirname(parent) if os.path.basename(parent) in ("train_infer","val_infer") else parent; return os.path.join(seed,"train_meta.json")
def _verify_source(expert,paths):
 frozen=frozen_splits("hateclipseg"); out=[]
 for path in paths:
  mp=_source_meta(path)
  if not os.path.isfile(mp): raise RuntimeError("expert source lacks train_meta")
  meta=json.load(open(mp)); args=meta.get("args") or {}; train=set(meta.get("train_ids") or []); val=set(meta.get("val_ids") or [])
  if args.get("corpus",meta.get("corpus"))!="hateclipseg" or meta.get("method")!=expert["name"]: raise RuntimeError("expert source corpus/method mismatch")
  if train!=set(frozen["train"]) or val!=set(frozen["val"]) or (train|val)&set(frozen["test"]): raise RuntimeError("expert source split provenance mismatch")
  out.append({"train_meta":os.path.abspath(mp),"sha256":sha256(mp)})
 return out
def load_dense(manifest,split):
 ids,dense,provenance=_base_load_dense(manifest,split); timeline=json.load(open(VGG_INDEX))
 for expert,p in zip(manifest["experts"],provenance):
  paths=expert[f"{split}_scores"]; paths=[paths] if isinstance(paths,str) else paths; p["source"]=_verify_source(expert,paths)
  for path in paths:
   if set(records(path))!=set(ids): raise RuntimeError("expert score IDs not exact frozen split")
 for vid,value in dense.items():
  expected=int(timeline[vid]["n_frames"])
  if len(value)!=expected: raise RuntimeError(f"expert/frozen-1fps length mismatch {vid}: {len(value)} vs {expected}")
 return ids,dense,provenance
def verify_sparse_manifest(path,train_ids):
 payload=json.load(open(path)); cohort=payload.get("selection_cohort_ids",payload.get("ids")); frozen=frozen_splits("hateclipseg"); timeline=json.load(open(VGG_INDEX))
 if payload.get("corpus")!="hateclipseg" or payload.get("split")!="train" or int(payload.get("k",-1))!=16 or not isinstance(cohort,list): raise RuntimeError("invalid K16 manifest identity")
 if not all(payload.get(k) for k in ("prompt","backend","index_rule","root_set_sha256")): raise RuntimeError("K16 manifest lacks generation/index provenance")
 cohort=set(cohort); files=payload.get("files"); root=payload.get("root"); excluded=payload.get("excluded_train_ids",{}); audit_path=payload.get("media_audit"); audit_hash=payload.get("media_audit_sha256")
 if not audit_path or not os.path.isfile(audit_path) or sha256(audit_path)!=audit_hash: raise RuntimeError("K16 media audit missing/hash mismatch")
 audit=json.load(open(audit_path))
 if audit.get("corpus")!="hateclipseg" or audit.get("split")!="train" or audit.get("label_access")!="none" or audit.get("frozen_train_ids")!=frozen["train"] or set(audit.get("records",{}))!=set(train_ids): raise RuntimeError("K16 media audit identity/coverage invalid")
 audited_cohort={v for v,r in audit["records"].items() if r.get("status")=="decodable_visual_stream"}; audited_excluded={v for v,r in audit["records"].items() if r.get("status")=="no_decodable_visual_stream"}
 if audited_cohort|audited_excluded!=set(train_ids) or audited_cohort&audited_excluded: raise RuntimeError("K16 media audit statuses invalid")
 if not cohort<=set(train_ids) or cohort&(set(frozen["val"])|set(frozen["test"])): raise RuntimeError("K16 cohort contamination")
 if not isinstance(excluded,dict) or cohort|set(excluded)!=set(train_ids) or cohort&set(excluded): raise RuntimeError("K16 included/excluded train partition invalid")
 if cohort!=audited_cohort or set(excluded)!=audited_excluded: raise RuntimeError("K16 cohort differs from independent media audit")
 if any((not isinstance(v,dict) or v.get("reason")!="no_decodable_visual_stream") for v in excluded.values()): raise RuntimeError("K16 exclusion provenance invalid")
 if not isinstance(files,dict) or set(files)!=cohort or not root or not os.path.isdir(root): raise RuntimeError("K16 manifest coverage/root invalid")
 if {x[:-5] for x in os.listdir(root) if x.endswith(".json")}!=cohort: raise RuntimeError("K16 root JSON set mismatch")
 aggregate=[]
 for vid in sorted(cohort):
  entry=files[vid]; expected=entry if isinstance(entry,str) else entry.get("sha256"); fp=os.path.join(root,vid+".json")
  actual_hash=sha256(fp)
  if expected!=actual_hash: raise RuntimeError("K16 file hash mismatch")
  aggregate.append(f"{vid}\t{actual_hash}\n")
  row=json.load(open(fp)); duration=float(row["duration"])
  n_frames=int(timeline[vid]["n_frames"])
  if row.get("video_id")!=vid or not math.isfinite(duration) or duration<=0: raise RuntimeError("K16 row identity/media-duration invalid")
  aligned_end=min(duration,float(n_frames)); aligned_length=max(1,int(math.ceil(aligned_end))); expected_count=min(16,aligned_length)
  import numpy as np
  expected_starts=np.unique(np.rint(np.linspace(0,aligned_length-1,expected_count)).astype(int)).astype(float).tolist()
  segments=row.get("segments",[])
  if len(segments)!=len(expected_starts): raise RuntimeError("K16 segment count invalid")
  for seg,expected_start in zip(segments,expected_starts):
   start,end,score=float(seg["start"]),float(seg["end"]),float(seg["score"])
   if not (start==expected_start and 0<=start<end<=aligned_end and 0<=score<=1): raise RuntimeError("K16 segment range/index invalid")
 root_hash=hashlib.sha256("".join(aggregate).encode()).hexdigest()
 if root_hash!=payload["root_set_sha256"]: raise RuntimeError("K16 root aggregate hash mismatch")
 return {"root":root,"cohort":sorted(cohort),"manifest":os.path.abspath(path),"manifest_sha256":sha256(path),"root_set_sha256":root_hash}
def _strict_sparse(info,vid,n):
 if vid not in info["cohort"]:
  import numpy as np
  return np.zeros(n,np.float32),np.zeros(n,bool)
 path=os.path.join(info["root"],vid+".json")
 if not os.path.isfile(path): raise RuntimeError("missing expected K16 file")
 row=json.load(open(path))
 for seg in row.get("segments",[]):
  if not (0<=float(seg["start"])<float(seg["end"])<=n): raise RuntimeError("K16 outside frozen 1fps grid")
 return sparse_target(info["root"],vid,n)
class AuditedDataset(ExpertDataset):
 def __getitem__(self,i):
  old=sparse_target
  if self.sparse_root:
   import relation_v6.data as base
   original=base.sparse_target; base.sparse_target=lambda root,vid,n:_strict_sparse(self.sparse_root,vid,n)
   try: return super().__getitem__(i)
   finally: base.sparse_target=original
  return super().__getitem__(i)
def build(manifest,split,calibration=None):
 ids,dense,prov=load_dense(manifest,split)
 if calibration is None: calibration=fit_calibration(dense)
 labels=sparse=gt=None
 if split=="train":
  from relation_v2.protocol import scoped_labels
  labels,_=scoped_labels(manifest["corpus"],"train"); sparse=verify_sparse_manifest(manifest["train_sparse_manifest"],ids)
 else:
  gt=hdata.gt_arrays(manifest["corpus"],split); ids=[v for v in ids if v in gt]
  for v in ids:
   if len(dense[v])!=len(gt[v]): raise RuntimeError("GT alignment mismatch")
 return AuditedDataset(ids,dense,calibration,labels,sparse,gt),calibration,prov
