#!/usr/bin/env python3
"""Real score-stream robustness benchmark with validation-frozen configs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
from hate_common import data as hdata
from relation_v4.io import fit_ecdf,apply_ecdf,sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.robust import fit,huber_barycenter

def canonical(manifest,split):
 raw,gt,prov=load_split_exact(manifest,split);return raw,gt,prov
def expanded(manifest,split,ids):
 cols=[];names=[];seen=set()
 for expert in manifest['experts']:
  paths=expert[f'{split}_scores'];paths=[paths] if isinstance(paths,str) else paths
  for si,path in enumerate(paths):
   records=hdata.load_scores_jsonl(path);keys=sorted(k for k,v in next(iter(records.values())).items() if isinstance(v,(list,np.ndarray)) and np.asarray(v).ndim==1)
   for key in keys:
    signature=(str(Path(path).resolve()),key)
    if signature in seen:continue
    seen.add(signature);names.append(f"{expert['name']}:seed{si}:{key}");cols.append({v:np.asarray(records[v][key],np.float32) for v in ids})
 return {v:np.stack([x[v] for x in cols],-1) for v in ids},names
def calibrate_pair(val,test):
 refs=fit_ecdf(val);return apply_ecdf(val,refs),apply_ecdf(test,refs),refs
def best_and_worst(val,gt):
 y=np.concatenate([gt[v] for v in sorted(gt)]);aps=[]
 for j in range(next(iter(val.values())).shape[1]):aps.append(average_precision_score(y,np.concatenate([val[v][:,j] for v in sorted(gt)])))
 return int(np.argmax(aps)),int(np.argmin(aps)),aps
def shift(z,k=5):
 out=np.zeros_like(z)
 if len(z)>k:out[k:]=z[:-k]
 return out
def scenarios(clean_val,clean_test,expanded_val,expanded_test,val_gt):
 best,worst,_=best_and_worst(clean_val,val_gt);result={'clean':(clean_val,clean_test)}
 for n in (1,2,5,10,20):result[f'exact_duplicate_{n}']=({v:np.concatenate([x,np.repeat(x[:,best:best+1],n,1)],1) for v,x in clean_val.items()},{v:np.concatenate([x,np.repeat(x[:,best:best+1],n,1)],1) for v,x in clean_test.items()})
 result['near_duplicate_real_seed_branch']=(expanded_val,expanded_test)
 result['weak_baseline']=({v:np.concatenate([x,x[:,worst:worst+1]],1) for v,x in clean_val.items()},{v:np.concatenate([x,x[:,worst:worst+1]],1) for v,x in clean_test.items()})
 result['constant_expert']=({v:np.concatenate([x,np.full((len(x),1),.5)],1) for v,x in clean_val.items()},{v:np.concatenate([x,np.full((len(x),1),.5)],1) for v,x in clean_test.items()})
 result['constant_duplicate_20']=({v:np.concatenate([x,np.full((len(x),20),.5)],1) for v,x in clean_val.items()},{v:np.concatenate([x,np.full((len(x),20),.5)],1) for v,x in clean_test.items()})
 result['reversed_expert']=({v:np.concatenate([x,1-x[:,best:best+1]],1) for v,x in clean_val.items()},{v:np.concatenate([x,1-x[:,best:best+1]],1) for v,x in clean_test.items()})
 keep=[j for j in range(next(iter(clean_val.values())).shape[1]) if j!=best];result['missing_expert']=({v:x[:,keep] for v,x in clean_val.items()},{v:x[:,keep] for v,x in clean_test.items()})
 result['temporal_shift']=({v:np.concatenate([x,shift(x[:,best])[:,None]],1) for v,x in clean_val.items()},{v:np.concatenate([x,shift(x[:,best])[:,None]],1) for v,x in clean_test.items()})
 def corrupt(values,tag):
  out={}
  for v,x in values.items():
   seed=int.from_bytes(hashlib.sha256((tag+v).encode()).digest()[:8],'little');noise=np.random.default_rng(seed).normal(0,.2,len(x));out[v]=np.concatenate([x,np.clip(x[:,best]+noise,0,1)[:,None]],1)
  return out
 result['score_corruption']=(corrupt(clean_val,'val'),corrupt(clean_test,'test'));return result
def trim(x):
 if x.shape[1]<5:return np.median(x,1)
 k=max(1,int(.2*x.shape[1]));return np.sort(x,axis=1)[:,k:-k].mean(1)
def geometric(x):return np.median(x,1)
def fit_eval(name,val,test,val_gt,test_gt):
 ids=sorted(val);xv=np.concatenate([val[v] for v in ids]);yv=np.concatenate([val_gt[v] for v in ids]);state=fit(xv)
 lr=LogisticRegression(C=1.,max_iter=1000,random_state=0).fit(xv,yv)
 methods={}
 for method in ('equal_mean','median','trimmed_mean','huber','geometric_median','v11'):
  pred={}
  for v,x in test.items():
   if method=='equal_mean':z=x.mean(1)
   elif method=='median':z=np.median(x,1)
   elif method=='trimmed_mean':z=trim(x)
   elif method=='geometric_median':z=geometric(x)
   else:z=huber_barycenter(x,state)
   pred[v]=z
  methods[method]=metrics(pred,test_gt)
 pred={v:lr.predict_proba(x)[:,1] for v,x in test.items()};methods['val_stacking']=metrics(pred,test_gt)
 return methods,{'clusters':state['clusters'],'cluster_mass':state['cluster_mass'].tolist(),'stack_coef':lr.coef_[0].tolist(),'stack_intercept':lr.intercept_.tolist()}
def metrics(pred,gt):
 ids=sorted(gt);s=np.concatenate([pred[v] for v in ids]);y=np.concatenate([gt[v] for v in ids]);center=np.concatenate([pred[v]-pred[v].mean() for v in ids]);mixed=[v for v in ids if len(np.unique(gt[v]))==2]
 return {'frame_ap':average_precision_score(y,s),'frame_roc':roc_auc_score(y,s),'within_centered_ap':average_precision_score(y,center),'within_centered_roc':roc_auc_score(y,center),'within_macro_ap':float(np.mean([average_precision_score(gt[v],pred[v]) for v in mixed])),'within_macro_roc':float(np.mean([roc_auc_score(gt[v],pred[v]) for v in mixed])),'mixed_videos':len(mixed)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();m=json.load(open(a.manifest));val_raw,val_gt,val_prov=canonical(m,'val');test_raw,test_gt,test_prov=canonical(m,'test');val,test,refs=calibrate_pair(val_raw,test_raw);ev,en=expanded(m,'val',val_gt);et,_=expanded(m,'test',test_gt);ev,et,expanded_refs=calibrate_pair(ev,et);cases=scenarios(val,test,ev,et,val_gt);rows={};configs={}
 for name,(vv,tt) in cases.items():rows[name],configs[name]=fit_eval(name,vv,tt,val_gt,test_gt)
 methods=next(iter(rows.values()));summary={}
 for method in methods:
  vals=[rows[c][method] for c in rows];summary[method]={'clean':rows['clean'][method],**{f'average_{k}':float(np.mean([x[k] for x in vals])) for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')},**{f'worst_{k}':float(np.min([x[k] for x in vals])) for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')}}
 payload={'method':'relation_v11_real_score_stream_robustness','corpus':m['corpus'],'protocol':'each perturbation configured on validation only, frozen before corresponding test evaluation','manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'canonical_validation_ecdf':refs,'expanded_validation_ecdf':expanded_refs,'expanded_stream_names':en,'validation_sources':val_prov,'test_sources':test_prov,'perturbation_configs':configs,'results':rows,'summary':summary,'test_labels_used_for_configuration':False};atomic_json(a.out,payload);print(json.dumps({'corpus':m['corpus'],'scenarios':len(rows),'streams':len(en),'summary':summary},indent=2))
if __name__=='__main__':main()
