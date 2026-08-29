#!/usr/bin/env python3
"""Select a two-expert static fusion weight on validation only."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf,fit_ecdf,load_manifest,load_split,sha256
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--out",required=True); a=p.parse_args(); m=load_manifest(a.manifest)
 if len(m["experts"])!=2: raise RuntimeError("weight selector requires exactly two fixed experts")
 raw,gt,prov=load_split(m,"val"); calibration=fit_ecdf(raw); rank=apply_ecdf(raw,calibration); rows=[]
 for i in range(101):
  w=i/100.; per={v:(w*x[:,0]+(1-w)*x[:,1],gt[v]) for v,x in rank.items()}; metric=fec.evaluate(per); rows.append({"weight_expert_0":w,"weight_expert_1":1-w,"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"]})
 best=max(rows,key=lambda r:(r["frame_ap"],r["frame_roc"]))
 payload={"corpus":m["corpus"],"candidate_pool":[x["name"] for x in m["experts"]],"grid":"expert_0=0.00:0.01:1.00; expert_1=1-expert_0","primary":"validation_pooled_frame_ap","tie_break":"validation_pooled_frame_roc","selected":best,"weight_grid":rows,"validation_experts":prov,"manifest":os.path.abspath(a.manifest),"manifest_sha256":sha256(a.manifest),"test_opened":False}
 os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); json.dump(payload,open(a.out+".tmp","w"),indent=2); os.replace(a.out+".tmp",a.out); print(json.dumps(best))
if __name__=="__main__": main()
