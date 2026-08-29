#!/usr/bin/env python3
"""V14: frozen V10 identity plus a gated V13 zero-mean locator."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/'duplex')]
from relation_v4.io import sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.score_stream_benchmark import metrics
from relation_v12.diagnostic import frozen_v10_identity
from relation_v13.diagnostic import load_repeats,role_stats,collapse_exact,cluster_weights
from relation_v9.train import load as load_v9

BETAS=(0.,.01,.025,.05,.1,.2,.4);N_BOOT=2000

def locator(values,groups,weights):
 out={}
 for v in values[next(iter(values))]:
  z=np.stack([np.mean(np.stack([values[n][v] for n in g]),0) for g in groups],1);r=(z-z.mean(0,keepdims=True))@weights;out[v]=r-r.mean()
 return out
def add(base,residual,beta):return {v:base[v]+beta*residual[v] for v in base}
def mixed_per_video(pred,gt):
 out={}
 for v in sorted(gt):
  if len(np.unique(gt[v]))==2:out[v]=(average_precision_score(gt[v],pred[v]),roc_auc_score(gt[v],pred[v]))
 return out
def bootstrap_gate(candidate,base,seed):
 ids=sorted(base);d=np.asarray([[candidate[v][j]-base[v][j] for j in (0,1)] for v in ids]);rng=np.random.default_rng(seed);means=np.empty((N_BOOT,2))
 for i in range(N_BOOT):means[i]=d[rng.integers(0,len(d),len(d))].mean(0)
 lo=np.quantile(means,.025,axis=0);delta=d.mean(0);return {'delta_macro_ap':float(delta[0]),'delta_macro_roc':float(delta[1]),'lower95_ap':float(lo[0]),'lower95_roc':float(lo[1]),'significant_ap':bool(lo[0]>0),'significant_roc':bool(lo[1]>0)}
def time_shuffle(residual):
 out={}
 for v,x in residual.items():
  if len(x)<2:out[v]=x.copy();continue
  shift=1+int.from_bytes(hashlib.sha256(v.encode()).digest()[:4],'little')%(len(x)-1);out[v]=np.roll(x,shift)
 return out
def noninferior(a,b,tol=1e-12):return a['frame_ap']+tol>=b['frame_ap'] and a['frame_roc']+tol>=b['frame_roc']
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();m=json.load(open(a.manifest));vr,vg,_=load_split_exact(m,'val');tr,tg,_=load_split_exact(m,'test');vids=sorted(vg);tids=sorted(tg)
 use_train=all('train_scores' in e for e in m['experts']);state_ids=load_v9(m,'train')[0] if use_train else vids;state_split='train' if use_train else 'val';values_state={};values_val={};values_test={};stats={};refs_archive={}
 for e in m['experts']:
  sr,refs=load_repeats(m,e,state_split,state_ids);vv,_=load_repeats(m,e,'val',vids,refs);tt,_=load_repeats(m,e,'test',tids,refs);n=e['name'];refs_archive[n]=[x.tolist() for x in refs];values_state[n]={v:np.mean([x[v] for x in sr],0) for v in state_ids};values_val[n]={v:np.mean([x[v] for x in vv],0) for v in vids};values_test[n]={v:np.mean([x[v] for x in tt],0) for v in tids};pr,lo=role_stats(sr,state_ids);stats[n]={'prior':pr,'locator':lo,'repeat_count':len(sr)}
 groups=collapse_exact(values_state,state_ids);wl,detail=cluster_weights(groups,stats,'locator');rv=locator(values_val,groups,wl);rt=locator(values_test,groups,wl);sv=time_shuffle(rv);basev,_=frozen_v10_identity(m,vr);baset,v10_path=frozen_v10_identity(m,tr);basev={v:x[:,0] for v,x in basev.items()};baset={v:x[:,0] for v,x in baset.items()};base_metric=metrics(basev,vg);base_video=mixed_per_video(basev,vg);folds=[vids[i::5] for i in range(5)];rows=[]
 seed=int.from_bytes(hashlib.sha256(m['corpus'].encode()).digest()[:8],'little')
 for beta in BETAS:
  pred=add(basev,rv,beta);met=metrics(pred,vg);per=mixed_per_video(pred,vg);boot=bootstrap_gate(per,base_video,seed);shuffle=metrics(add(basev,sv,beta),vg);fold_gate=[]
  for ids in folds:
   cp={v:pred[v] for v in ids};bp={v:basev[v] for v in ids};cm=metrics(cp,{v:vg[v] for v in ids});bm=metrics(bp,{v:vg[v] for v in ids});cper=mixed_per_video(cp,{v:vg[v] for v in ids});bper=mixed_per_video(bp,{v:vg[v] for v in ids});da=np.mean([cper[v][0]-bper[v][0] for v in cper]) if cper else 0.;dr=np.mean([cper[v][1]-bper[v][1] for v in cper]) if cper else 0.;fold_gate.append({'n':len(ids),'pooled_noninferior':bool(noninferior(cm,bm)),'within_ap_delta':float(da),'within_roc_delta':float(dr),'within_one_positive_other_nonnegative':bool((da>0 and dr>=0) or (dr>0 and da>=0))})
  within_sig=(boot['significant_ap'] and boot['delta_macro_roc']>=-1e-12) or (boot['significant_roc'] and boot['delta_macro_ap']>=-1e-12);shuffle_gate=(boot['significant_ap'] and met['within_macro_ap']>shuffle['within_macro_ap']) or (boot['significant_roc'] and met['within_macro_roc']>shuffle['within_macro_roc']);eligible=beta==0 or (noninferior(met,base_metric) and all(x['pooled_noninferior'] for x in fold_gate) and all(x['within_one_positive_other_nonnegative'] for x in fold_gate) and within_sig and shuffle_gate);rows.append({'beta':beta,'metrics':met,'bootstrap':boot,'time_shuffle_metrics':shuffle,'folds':fold_gate,'within_significance_gate':bool(within_sig),'time_shuffle_gate':bool(shuffle_gate),'eligible':bool(eligible)})
 eligible=[x for x in rows if x['eligible']];selected=max(eligible,key=lambda x:(x['metrics']['frame_ap'],x['metrics']['frame_roc'],x['metrics']['within_macro_ap'],x['metrics']['within_macro_roc'],-x['beta']))
 frozen={'method':'relation_v14_gated_reliable_locator','corpus':m['corpus'],'external_identity':'frozen V10 per-frame scores','external_identity_config':str(v10_path.resolve()),'external_identity_config_sha256':sha256(v10_path),'state_split':state_split,'locator':'V13 family-balanced ICC-odds cluster weights, strict per-video zero mean','clusters':groups,'locator_weights':wl.tolist(),'locator_weight_detail':detail,'ecdf_references':refs_archive,'beta_grid':BETAS,'bootstrap_replicates':N_BOOT,'activation':'pooled AP+ROC noninferior globally and on every video fold; within macro AP or ROC paired-bootstrap significant with other nondecreasing; each fold has one within metric positive and other nonnegative; time-shuffle gate','validation_rows':rows,'selected_beta':selected['beta'],'identity_fallback_beta':0.,'test_opened':False,'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest)};freeze=Path(a.out).with_suffix('.frozen.json');atomic_json(freeze,frozen)
 pred=add(baset,rt,selected['beta']);shuf=add(baset,time_shuffle(rt),selected['beta']);test={'selected':metrics(pred,tg),'identity':metrics(baset,tg),'time_shuffle':metrics(shuf,tg)};delta={k:test['selected'][k]-test['identity'][k] for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')}
 oracle=[]
 for beta in BETAS:
  mm=metrics(add(baset,rt,beta),tg);oracle.append({'beta':beta,'metrics':mm})
 oracle=max(oracle,key=lambda x:(x['metrics']['frame_ap'],x['metrics']['frame_roc'],x['metrics']['within_macro_ap'],x['metrics']['within_macro_roc']))
 payload={'method':frozen['method'],'corpus':m['corpus'],'selected_beta':selected['beta'],'test':test,'delta_selected_minus_v10':delta,'test_informed_checkpoint':{**oracle,'status':'TEST-INFORMED diagnostic only; not selected method'},'frozen_config':str(freeze.resolve()),'frozen_config_sha256':sha256(freeze),'test_labels_used_for_selection':False};atomic_json(a.out,payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
