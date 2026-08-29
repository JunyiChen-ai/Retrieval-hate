#!/usr/bin/env python3
"""Apply a frozen hierarchical/fallback selection to test."""
import argparse,json,os,sys
import torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf,load_manifest,load_split,sha256
from relation_v4.model import AnalyticExpertRelationGate
from relation_v5.common import hierarchical
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--config",required=True); p.add_argument("--candidate",default="selected",choices=("selected","best_hierarchical_ablation")); p.add_argument("--scores-out",required=True); p.add_argument("--eval-out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); c=json.load(open(a.config))
 if c["manifest_sha256"]!=sha256(a.manifest): raise RuntimeError("config/manifest mismatch")
 raw,gt,prov=load_split(m,"test"); scores=apply_ecdf(raw,c["calibration"]["sorted_values"]); sel=c[a.candidate]
 if sel["mode"]=="hierarchical": final=hierarchical(scores,sel["prior_weights"],sel["residual_weights"],sel["residual_amplitude"])
 elif sel["mode"]=="static": final=hierarchical(scores,m["static_weights"],m["static_weights"],1.)
 elif sel["mode"]=="expert":
  e=[x["name"] for x in m["experts"]].index(sel["expert"]); final={v:x[:,e] for v,x in scores.items()}
 elif sel["mode"]=="v4_fallback":
  v=c["v4_config"]; vw=v["static_weights"]; model=AnalyticExpertRelationGate(len(vw),vw,v["selected_beta"],v["selected_gamma"],v["window"],v["temperature"],inputs_are_calibrated=True).eval(); final={}
  with torch.no_grad():
   for vid,x in scores.items(): final[vid]=model(torch.from_numpy(x[:,:len(vw)])[None],torch.ones(1,len(x),dtype=torch.bool))["frame_score"][0].numpy()
 else: raise RuntimeError("unknown frozen mode")
 per={v:(s,gt[v]) for v,s in final.items()}; metric=fec.evaluate(per); os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)),exist_ok=True)
 with open(a.scores_out,"w") as f:
  for v,s in final.items(): f.write(json.dumps({"video_id":v,"score_relation_v5":s.tolist()})+"\n")
 payload={"method":"relation_v5_hierarchical","corpus":m["corpus"],"candidate":a.candidate,"selected":sel,"config":os.path.abspath(a.config),"config_sha256":sha256(a.config),"test_experts":prov,"results":{"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"]}}; json.dump(payload,open(a.eval_out+".tmp","w"),indent=2); os.replace(a.eval_out+".tmp",a.eval_out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
