#!/usr/bin/env python3
"""Single-corpus train-only prototype; validation selects the checkpoint."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
import frame_eval_common as fec
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits,scoped_labels
from relation_v4.io import sha256
from relation_v9.model import DependenceAwareRelation,dependence_weights
from relation_v9.loss import weak_loss
from relation_v9.build_mhc_manifests import verify_producer

def load(manifest,split):
 frozen=list(frozen_splits(manifest["corpus"])[split])
 ids=frozen if split=="train" else [v for v in frozen if v in hdata.gt_arrays(manifest["corpus"],split)]
 streams=[];prov=[]
 for expert in manifest["experts"]:
  paths=expert[f"{split}_scores"];paths=[paths] if isinstance(paths,str) else paths;runs=[]
  for path in paths:
   resolved=Path(path).resolve()
   if split=="train" and "results/reproduction/relation_v9/train_dense" in str(resolved):
    try: seed=int(resolved.parent.name.removeprefix("seed_"))
    except ValueError as exc:raise RuntimeError(f"invalid producer seed path: {path}") from exc
    verified=verify_producer(Path(hdata.REPO_ROOT),resolved.parent,expert["name"],manifest["corpus"],seed,expert["score_key"])
    if verified.resolve()!=resolved:raise RuntimeError("producer verified a different scores path")
   if split=="train" and manifest.get("require_train_evidence_manifest"):
    path_obj=Path(path);evidence_path=path_obj.parent/"evidence_manifest.json"
    if path_obj.parent.name!="train_infer":raise RuntimeError(f"{expert['name']} train source is not train_infer: {path}")
    if not evidence_path.is_file():raise RuntimeError(f"{expert['name']} missing train evidence manifest")
    evidence=json.load(open(evidence_path))
    if (evidence.get("corpus")!=manifest["corpus"] or evidence.get("split")!="train" or evidence.get("scores_sha256")!=sha256(path_obj) or evidence.get("val_or_test_scores_used_as_train") is not False):raise RuntimeError(f"{expert['name']} invalid train evidence provenance")
   records=hdata.load_scores_jsonl(path);missing=set(ids)-set(records);extras=sorted(set(records)-set(ids))
   if missing:raise RuntimeError(f"{expert['name']} {split} missing IDs")
   if split=="train" and extras:raise RuntimeError(f"{expert['name']} train source has non-frozen extra IDs")
   runs.append(records)
  stream={v:np.mean([np.asarray(r[v][expert["score_key"]],np.float32) for r in runs],0) for v in ids}
  audits=[]
  for path,records in zip(paths,runs):
   extras=sorted(set(records)-set(ids));import hashlib
   audits.append({"extra_count":len(extras),"extra_ids_sorted":extras,"extra_ids_sha256":hashlib.sha256("".join(f"{x}\n" for x in extras).encode()).hexdigest(),"extras_ignored":True})
  streams.append(stream);prov.append({"name":expert["name"],"paths":[str(Path(x).resolve()) for x in paths],"sha256":[sha256(x) for x in paths],"aggregation":"raw_score_mean","cohort":"complete frozen train" if split=="train" else "frozen localization GT intersection","id_audit":audits})
 values={}
 for v in ids:
  lengths={len(x[v]) for x in streams}
  if len(lengths)!=1 or not all(np.isfinite(x[v]).all() for x in streams):raise RuntimeError(f"{split}/{v} alignment/nonfinite")
  values[v]=np.stack([x[v] for x in streams],-1)
  if split=="train" and manifest["corpus"]=="hateclipseg":
   index=json.load(open(Path(hdata.REPO_ROOT)/"results/reproduction/features/vggish_1s/hateclipseg/index.json"));expected=int(index[v]["n_frames"])
   if len(values[v])!=expected:raise RuntimeError(f"train/{v} length {len(values[v])} != frozen VGGish 1fps {expected}")
 return ids,values,prov

def fit_ecdf(values):
 pooled=np.concatenate([values[v] for v in sorted(values)]);return [np.sort(pooled[:,i]) for i in range(pooled.shape[1])]
def calibrate(values,refs):
 out={}
 for v,x in values.items():
  z=np.empty_like(x)
  for e,ref in enumerate(refs):z[:,e]=(np.searchsorted(ref,x[:,e],"left")+np.searchsorted(ref,x[:,e],"right"))/(2*len(ref))
  out[v]=z
 return out
def tensor(value):
 x=torch.from_numpy(value)[None];valid=torch.ones(1,len(value),dtype=torch.bool);return x,valid
def validate(model,val_ids,val,gt,device):
 model.eval();per={}
 with torch.no_grad():
  for vid in val_ids:
   x,valid=tensor(val[vid]);score=torch.sigmoid(model(x.to(device),valid.to(device))["frame_logit"])[0].cpu().numpy();per[vid]=(score,gt[vid])
 metric=fec.evaluate(per);return metric["pr_auc"],metric["roc_auc"]

def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=234);a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
 out=Path(a.out_dir); 
 if out.exists() and any(out.iterdir()):raise RuntimeError("fresh out-dir required")
 out.mkdir(parents=True,exist_ok=True);m=json.load(open(a.manifest));train_ids,train_raw,train_prov=load(m,"train");val_ids,val_raw,val_prov=load(m,"val");labels,_=scoped_labels(m["corpus"],"train");refs=fit_ecdf(train_raw);train=calibrate(train_raw,refs);val=calibrate(val_raw,refs);pooled=np.concatenate([train[v]-train[v].mean(0,keepdims=True) for v in train_ids]);weights,clusters,corr=dependence_weights(pooled)
 model=DependenceAwareRelation(len(m["experts"]),clusters,hidden=32,window=8).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);gt=hdata.gt_arrays(m["corpus"],"val");ap0,roc0=validate(model,val_ids,val,gt,a.device);history=[{"epoch":0,"train_loss":None,"validation_frame_ap":ap0,"validation_frame_roc":roc0,"identity_fallback":True}];best=((ap0,roc0),0,copy.deepcopy(model.state_dict()));torch.save(best[2],out/"model.pth")
 for epoch in range(1,6):
  model.train();order=np.random.permutation(train_ids);losses=[]
  for vid in order:
   x,valid=tensor(train[vid]);x,valid=x.to(a.device),valid.to(a.device);label=torch.tensor([labels[vid]],dtype=torch.float32,device=a.device);clean=model(x,valid);keep=(torch.rand(len(clusters),device=a.device)>.2).float()
   if not keep.any():keep[torch.randint(len(keep),(1,),device=a.device)]=1
   cluster_noise=.01*torch.randn(x.shape[0],x.shape[1],len(clusters),device=a.device)
   corrupt=model(x,valid,keep,cluster_noise);loss,_=weak_loss(clean,corrupt,valid,label);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  ap,roc=validate(model,val_ids,val,gt,a.device);row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_frame_ap":ap,"validation_frame_roc":roc,"identity_fallback":False};history.append(row);key=(row["validation_frame_ap"],row["validation_frame_roc"])
  if best is None or key>best[0]:best=(key,epoch,copy.deepcopy(model.state_dict()));torch.save(best[2],out/"model.pth")
 meta={"method":"relation_v9_dependence_aware_prototype","corpus":m["corpus"],"seed":a.seed,"fixed_pilot":{"epochs":5,"lr":2e-4,"expert_dropout":.2,"score_noise_std":.01,"hidden":32,"window":8},"selected_epoch":best[1],"selected_by":"maximum validation Frame AP; ROC tie-break","history":history,"dependence":{"threshold":.98,"weights":weights.tolist(),"clusters":clusters,"correlation":corr.tolist()},"calibration":{"source":"train only","sorted_values":[x.tolist() for x in refs]},"train_sources":train_prov,"validation_sources":val_prov,"manifest":str(Path(a.manifest).resolve()),"manifest_sha256":sha256(a.manifest),"test_opened":False};json.dump(meta,open(out/"train_meta.json","w"),indent=2);print(json.dumps({"selected_epoch":best[1],"validation":best[0]},indent=2))
if __name__=="__main__":main()
