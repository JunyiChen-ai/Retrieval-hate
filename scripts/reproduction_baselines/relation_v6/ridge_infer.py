#!/usr/bin/env python3
"""Apply the frozen ridge-prior sanity model to test."""
import argparse,json,os,sys
import numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import load_manifest,sha256
from relation_v6.audited_data import build
from relation_v6.model import distribution_tokens
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--config",required=True); p.add_argument("--scores-out",required=True); p.add_argument("--eval-out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); c=json.load(open(a.config))
 if c["manifest_sha256"]!=sha256(a.manifest): raise RuntimeError("ridge config/manifest mismatch")
 ds,_,prov=build(m,"test",c["calibration"]["sorted_values"]); mean=np.asarray(c["scaler_mean"]); scale=np.asarray(c["scaler_scale"]); coef=np.asarray(c["coef"]); intercept=float(c["intercept"]); per={}; os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)),exist_ok=True)
 with open(a.scores_out,"w") as f:
  for i in range(len(ds)):
   vid,score,label,teacher,mask,gold=ds[i]; feature=distribution_tokens(score[None],torch.ones(1,len(score),dtype=torch.bool))[0].numpy().reshape(-1); logit=((feature-mean)/scale)@coef+intercept; probability=float(1/(1+np.exp(-logit))); dense=np.full(len(score),probability); per[vid]=(dense,gold.numpy()); f.write(json.dumps({"video_id":vid,"score_relation_v6_ridge_prior":dense.tolist()})+"\n")
 metric=fec.evaluate(per); payload={"method":"relation_v6_ridge_prior_sanity","corpus":m["corpus"],"config":os.path.abspath(a.config),"config_sha256":sha256(a.config),"test_experts":prov,"results":{"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"],"n_videos":metric["n_videos"],"n_frames":metric["n_frames"]}}; json.dump(payload,open(a.eval_out+".tmp","w"),indent=2); os.replace(a.eval_out+".tmp",a.eval_out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
