#!/usr/bin/env python3
"""Apply a frozen V4 validation selection to test and evaluate it."""
import argparse,json,os,sys
import numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf,load_manifest,load_split,sha256
from relation_v4.model import AnalyticExpertRelationGate
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--config",required=True); p.add_argument("--scores-out",required=True); p.add_argument("--eval-out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); c=json.load(open(a.config))
 if c["corpus"]!=m["corpus"] or c["manifest_sha256"]!=sha256(a.manifest): raise RuntimeError("frozen selection/manifest mismatch")
 raw,gt,prov=load_split(m,"test"); scores=apply_ecdf(raw,c["calibration"]["sorted_values"]); model=AnalyticExpertRelationGate(len(c["static_weights"]),c["static_weights"],c["selected_beta"],c.get("selected_gamma",0.),c["window"],c["temperature"],inputs_are_calibrated=True).eval(); per={}; os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)),exist_ok=True)
 with torch.no_grad(),open(a.scores_out,"w") as f:
  for v,x in scores.items():
   o=model(torch.from_numpy(x)[None],torch.ones(1,len(x),dtype=torch.bool)); final=o["frame_score"][0].numpy(); static=o["static_score"][0].numpy(); per[v]=(final,gt[v]); f.write(json.dumps({"video_id":v,"score_relation_v4":final.tolist(),"score_static_fusion":static.tolist()})+"\n")
 metric=fec.evaluate(per); payload={"method":"relation_v4_analytic","corpus":m["corpus"],"config":os.path.abspath(a.config),"config_sha256":sha256(a.config),"test_experts":prov,"results":{"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"]}}
 json.dump(payload,open(a.eval_out+".tmp","w"),indent=2); os.replace(a.eval_out+".tmp",a.eval_out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
