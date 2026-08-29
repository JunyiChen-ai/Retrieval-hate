#!/usr/bin/env python3
"""Evaluate a validation-selected V9 checkpoint; never performs selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
import frame_eval_common as fec
from hate_common import data as hdata
from relation_v9.model import DependenceAwareRelation
from relation_v9.train import load,calibrate,tensor
from relation_v4.io import sha256
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--checkpoint-dir",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();root=Path(a.out_dir)
 if root.exists() and any(root.iterdir()):raise RuntimeError("fresh out-dir required")
 root.mkdir(parents=True,exist_ok=True);meta=json.load(open(Path(a.checkpoint_dir)/"train_meta.json"));m=json.load(open(a.manifest))
 if meta["manifest_sha256"]!=sha256(a.manifest) or meta.get("test_opened") is not False:raise RuntimeError("checkpoint provenance mismatch")
 ids,raw,prov=load(m,"test");refs=[np.asarray(x) for x in meta["calibration"]["sorted_values"]];values=calibrate(raw,refs);clusters=meta["dependence"]["clusters"];model=DependenceAwareRelation(len(m["experts"]),clusters,hidden=32,window=8).to(a.device);model.load_state_dict(torch.load(Path(a.checkpoint_dir)/"model.pth",map_location=a.device));model.eval();gt=hdata.gt_arrays(m["corpus"],"test");per={};seen=set();score_path=root/"scores.jsonl"
 with torch.no_grad(),score_path.open("w") as f:
  for vid in ids:
   if vid in seen:raise RuntimeError("duplicate ID")
   seen.add(vid);x,valid=tensor(values[vid]);o=model(x.to(a.device),valid.to(a.device));score=torch.sigmoid(o["frame_logit"])[0].cpu().numpy();prior=float(o["prior_logit"][0]);locator=o["locator_logit"][0].cpu().numpy()
   if len(score)!=len(gt[vid]) or not np.isfinite(score).all() or abs(float(locator.mean()))>1e-5:raise RuntimeError("alignment/finite/zero-mean failure")
   per[vid]=(score,gt[vid]);f.write(json.dumps({"video_id":vid,"score_relation_v9":score.tolist(),"prior_logit":prior,"locator_logit":locator.tolist()})+"\n")
 if seen!=set(gt):raise RuntimeError("incomplete test coverage")
 metric=fec.evaluate(per);payload={"method":meta["method"],"corpus":m["corpus"],"checkpoint":str(Path(a.checkpoint_dir).resolve()),"checkpoint_sha256":sha256(Path(a.checkpoint_dir)/"model.pth"),"selected_epoch":meta["selected_epoch"],"test_sources":prov,"results":{"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"]},"test_used_for_selection":False};json.dump(payload,open(root/"frame_eval.json","w"),indent=2);print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
