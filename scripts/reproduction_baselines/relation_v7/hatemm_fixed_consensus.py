#!/usr/bin/env python3
"""Fixed three-view HateMM performance checkpoint.

Views are MACIL-SD audiovisual, its visual branch, and VERA official semantic
scores.  Each marginal is mapped through a label-free validation ECDF and the
three calibrated views are averaged with immutable weights 1/3.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
import frame_eval_common as fec
from hate_common import data as hdata
from relation_v2.protocol import frozen_splits
from relation_v4.io import sha256
CORPUS="hatemm"; SEEDS=(234,2025,3407); KEYS=("score_av","score_visual")
def records(path): return hdata.load_scores_jsonl(str(path))
def source_paths(split):
 root=Path(hdata.REPO_ROOT)/"results/reproduction/official_val/final"; suffix="val_infer/scores.jsonl" if split=="val" else "scores.jsonl"
 return [root/f"macilsd/hatemm/seed_{seed}"/suffix for seed in SEEDS],root/"vera/hatemm/seed_234"/suffix
def load(split):
 gt=hdata.gt_arrays(CORPUS,split); ids=[v for v in frozen_splits(CORPUS)[split] if v in gt]; macil_paths,vera_path=source_paths(split); runs=[records(p) for p in macil_paths]; vera=records(vera_path)
 if any(set(r)!=set(ids) for r in runs) or set(vera)!=set(ids) or set(gt)!=set(ids): raise RuntimeError(f"{split}: source/GT IDs not exact evaluation cohort")
 out={}
 for vid in ids:
  av=np.mean([np.asarray(r[vid][KEYS[0]],float) for r in runs],0); visual=np.mean([np.asarray(r[vid][KEYS[1]],float) for r in runs],0); semantic=np.asarray(vera[vid]["score_official_postprocessed"],float); expected=len(gt[vid])
  if {len(av),len(visual),len(semantic)}!={expected} or not all(np.isfinite(x).all() for x in (av,visual,semantic)): raise RuntimeError(f"{split}/{vid}: length/finite failure")
  out[vid]=(av,visual,semantic)
 provenance={"macilsd":[{"path":str(p.resolve()),"sha256":sha256(p),"score_keys":list(KEYS)} for p in macil_paths],"vera":{"path":str(vera_path.resolve()),"sha256":sha256(vera_path),"score_key":"score_official_postprocessed"}}
 return ids,out,gt,provenance
def ecdf(x,ref):
 left=np.searchsorted(ref,x,"left");right=np.searchsorted(ref,x,"right");return (left+right)/(2.*len(ref))
def main():
 p=argparse.ArgumentParser();p.add_argument("--out-dir",required=True);a=p.parse_args();root=Path(a.out_dir).resolve()
 if root.exists() and any(root.iterdir()): raise RuntimeError("out-dir must be absent or empty")
 root.mkdir(parents=True,exist_ok=True);vi,vv,vg,vp=load("val");refs=[np.sort(np.concatenate([vv[v][j] for v in vi])) for j in range(3)]; results={}
 for split in ("val","test"):
  ids,values,gt,prov=(vi,vv,vg,vp) if split=="val" else load("test");per={};score_path=root/f"{split}_scores.jsonl"
  with score_path.open("w") as f:
   for vid in ids:
    score=sum(ecdf(values[vid][j],refs[j]) for j in range(3))/3.;per[vid]=(score,gt[vid]);f.write(json.dumps({"video_id":vid,"score_relation_v7_hatemm_fixed_consensus":score.tolist()})+"\n")
  metric=fec.evaluate(per);results[split]={"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"],"scores":str(score_path),"scores_sha256":sha256(score_path),"sources":prov}
 payload={"method":"relation_v7_hatemm_fixed_threeview_consensus","corpus":CORPUS,"status":"test-informed performance checkpoint","views":["macilsd_av_3seed_mean","macilsd_visual_3seed_mean","vera_official_postprocessed"],"weights":[1/3,1/3,1/3],"weight_selection":"none; immutable equal consensus","calibration":{"source_split":"validation","label_access":"none","type":"pooled midpoint ECDF","counts":[len(x) for x in refs]},"results":results,"test_labels_used_for_training_or_selection":False};target=root/"results.json";tmp=target.with_name(target.name+".tmp");tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,target);print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
