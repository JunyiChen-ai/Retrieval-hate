#!/usr/bin/env python3
"""Train-only ridge-logistic video-prior sanity lower bound."""
import argparse,json,os,sys
import numpy as np,torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path[:0]=[os.path.dirname(HERE),os.path.join(os.path.dirname(os.path.dirname(HERE)),"duplex")]
import frame_eval_common as fec
from relation_v4.io import load_manifest,sha256
from relation_v6.audited_data import build
from relation_v6.model import distribution_tokens
def features(ds):
 x=[]; y=[]; vids=[]
 for i in range(len(ds)):
  vid,score,label,teacher,mask,gold=ds[i]; token=distribution_tokens(score[None],torch.ones(1,len(score),dtype=torch.bool))[0].numpy().reshape(-1); x.append(token); y.append(label); vids.append((vid,gold.numpy()))
 return np.asarray(x),np.asarray(y),vids
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--out",required=True); a=p.parse_args(); m=load_manifest(a.manifest); tr,cal,trprov=build(m,"train"); va,_,vaprov=build(m,"val",cal); x,y,_=features(tr); xv,_,vids=features(va); scaler=StandardScaler().fit(x); xs=scaler.transform(x); xvs=scaler.transform(xv); rows=[]; models=[]
 for c in (.01,.1,1.):
  model=LogisticRegression(C=c,penalty="l2",solver="lbfgs",max_iter=2000,random_state=0).fit(xs,y); probability=model.predict_proba(xvs)[:,1]; per={vid:(np.full(len(gold),probability[i]),gold) for i,(vid,gold) in enumerate(vids)}; metric=fec.evaluate(per); rows.append({"C":c,"frame_ap":metric["pr_auc"],"frame_roc":metric["roc_auc"]}); models.append(model)
 index=max(range(len(rows)),key=lambda i:(rows[i]["frame_ap"],rows[i]["frame_roc"])); model=models[index]; payload={"method":"relation_v6_ridge_prior_sanity","corpus":m["corpus"],"selected":rows[index],"validation_grid":rows,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"coef":model.coef_[0].tolist(),"intercept":float(model.intercept_[0]),"calibration":{"type":"train_frozen_ecdf","sorted_values":cal},"manifest":os.path.abspath(a.manifest),"manifest_sha256":sha256(a.manifest),"train_experts":trprov,"validation_experts":vaprov,"test_opened":False}; os.makedirs(os.path.dirname(os.path.abspath(a.out)),exist_ok=True); json.dump(payload,open(a.out+".tmp","w"),indent=2); os.replace(a.out+".tmp",a.out); print(json.dumps({"selected":rows[index],"grid":rows},indent=2))
if __name__=="__main__": main()
