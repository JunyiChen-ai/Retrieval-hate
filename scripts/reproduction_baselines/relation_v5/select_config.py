#!/usr/bin/env python3
"""Validation-only hierarchical selection with V4/static fallbacks."""
import argparse,json,os,sys
import numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf,fit_ecdf,load_manifest,load_split,sha256
from relation_v4.model import AnalyticExpertRelationGate
from relation_v5.common import hierarchical,simplex
def metric(score,gt):
 x=fec.evaluate({v:(s,gt[v]) for v,s in score.items()}); return x["pr_auc"],x["roc_auc"]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--v4-config",required=True); p.add_argument("--out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); v4=json.load(open(a.v4_config)); raw,gt,prov=load_split(m,"val"); calibration=fit_ecdf(raw); scores=apply_ecdf(raw,calibration); names=[x["name"] for x in m["experts"]]
 rows=[]; expert=[]
 for e,name in enumerate(names):
  ap,roc=metric({v:x[:,e] for v,x in scores.items()},gt); row={"mode":"expert","expert":name,"frame_ap":ap,"frame_roc":roc}; rows.append(row); expert.append(row)
 best_expert=max(expert,key=lambda r:(r["frame_ap"],r["frame_roc"])); roc_floor=best_expert["frame_roc"]
 static=hierarchical(scores,m["static_weights"],m["static_weights"],1.); ap,roc=metric(static,gt); rows.append({"mode":"static","frame_ap":ap,"frame_roc":roc})
 v4w=v4["static_weights"]; vm=AnalyticExpertRelationGate(len(v4w),v4w,v4["selected_beta"],v4.get("selected_gamma",0.),v4["window"],v4["temperature"],inputs_are_calibrated=True).eval(); vs={}
 with torch.no_grad():
  for vid,x in scores.items(): vs[vid]=vm(torch.from_numpy(x[:,:len(v4w)])[None],torch.ones(1,len(x),dtype=torch.bool))["frame_score"][0].numpy()
 ap,roc=metric(vs,gt); rows.append({"mode":"v4_fallback","frame_ap":ap,"frame_roc":roc})
 weight_grid=simplex(len(names),float(m.get("v5_weight_step",.1))); amplitudes=m.get("v5_amplitude_grid",[0.,.25,.5,.75,1.,1.25,1.5,2.])
 for wp in weight_grid:
  for wr in weight_grid:
   for amp in amplitudes:
    score=hierarchical(scores,wp,wr,amp); ap,roc=metric(score,gt); rows.append({"mode":"hierarchical","prior_weights":wp,"residual_weights":wr,"residual_amplitude":float(amp),"frame_ap":ap,"frame_roc":roc})
 eligible=[r for r in rows if r["frame_roc"]>=roc_floor]; best=max(eligible,key=lambda r:(r["frame_ap"],r["frame_roc"])); best_hierarchical=max((r for r in eligible if r["mode"]=="hierarchical"),key=lambda r:(r["frame_ap"],r["frame_roc"])); payload={"method":"relation_v5_hierarchical","corpus":m["corpus"],"manifest":os.path.abspath(a.manifest),"manifest_sha256":sha256(a.manifest),"selection_rule":"max validation pooled Frame AP subject to ROC >= validation AP-best expert ROC; ROC tie-break","best_expert":best_expert,"roc_floor":roc_floor,"selected":best,"best_hierarchical_ablation":best_hierarchical,"candidate_counts":{"total":len(rows),"eligible":len(eligible)},"fixed_grid":{"prior_weights":weight_grid,"residual_weights":weight_grid,"residual_amplitude":amplitudes},"fallbacks":[r for r in rows if r["mode"]!="hierarchical"],"validation_grid":[r for r in rows if r["mode"]=="hierarchical"],"validation_experts":prov,"calibration":{"type":"validation_frozen_ecdf","sorted_values":calibration},"v4_config":{"path":os.path.abspath(a.v4_config),"sha256":sha256(a.v4_config),"static_weights":v4w,"selected_beta":v4["selected_beta"],"selected_gamma":v4.get("selected_gamma",0.),"window":v4["window"],"temperature":v4["temperature"]},"test_opened_during_selection":False}; os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); json.dump(payload,open(a.out+".tmp","w"),indent=2); os.replace(a.out+".tmp",a.out); print(json.dumps({"best_expert":best_expert,"selected":best,"best_hierarchical_ablation":best_hierarchical,"candidate_counts":payload["candidate_counts"]},indent=2))
if __name__=="__main__": main()
