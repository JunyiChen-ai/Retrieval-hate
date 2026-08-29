#!/usr/bin/env python3
"""Select only the relation strength on validation; never opens test scores."""
import argparse,json,os,sys
import torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf,fit_ecdf,load_manifest,load_split,sha256
from relation_v4.model import AnalyticExpertRelationGate
def score_all(scores,gt,weights,beta,gamma,window,temp):
 model=AnalyticExpertRelationGate(len(weights),weights,beta,gamma,window,temp,inputs_are_calibrated=True).eval(); out={}
 with torch.no_grad():
  for v,x in scores.items():
   z=torch.from_numpy(x)[None]; valid=torch.ones(1,len(x),dtype=torch.bool); y=model(z,valid)["frame_score"][0].numpy(); out[v]=(y,gt[v])
 return fec.evaluate(out)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); raw,gt,prov=load_split(m,"val"); calibration=fit_ecdf(raw); scores=apply_ecdf(raw,calibration); grid=m.get("beta_grid",[0.,.25,.5,1.,2.,4.,8.]); gamma_grid=m.get("gamma_grid",[0.]); window=int(m.get("window",12)); temp=float(m.get("temperature",.2)); rows=[]
 for beta in grid:
  for gamma in gamma_grid:
   metric=score_all(scores,gt,m["static_weights"],float(beta),float(gamma),window,temp); rows.append({"beta":float(beta),"gamma":float(gamma),"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"]})
 static=next(r for r in rows if r["beta"]==0. and r["gamma"]==0.)
 # A relation correction may replace the reproducible static lower bound only
 # when validation AP and ROC are both non-decreasing.
 eligible=[r for r in rows if r["frame_ap"]>=static["frame_ap"] and r["frame_roc"]>=static["frame_roc"]]
 best=max(eligible,key=lambda r:(r["frame_ap"],r["frame_roc"],-abs(r["beta"]),-abs(r["gamma"])))
 payload={"method":"relation_v4_analytic","corpus":m["corpus"],"manifest":os.path.abspath(a.manifest),"manifest_sha256":sha256(a.manifest),"static_weights":m["static_weights"],"weight_selection":m.get("weight_selection"),"window":window,"temperature":temp,"selected_beta":best["beta"],"selected_gamma":best["gamma"],"selected_by":"validation_pareto_frame_ap_and_roc","fallback":{"beta":0.,"gamma":0.,"validation_frame_ap":static["frame_ap"],"validation_frame_roc":static["frame_roc"]},"validation_grid":rows,"validation_experts":prov,"calibration":{"type":"validation_frozen_ecdf","sorted_values":calibration},"test_opened_during_selection":False}
 os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); json.dump(payload,open(a.out+".tmp","w"),indent=2); os.replace(a.out+".tmp",a.out); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
