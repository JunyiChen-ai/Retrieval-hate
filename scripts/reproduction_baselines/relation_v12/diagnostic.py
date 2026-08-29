#!/usr/bin/env python3
"""Nested-validation constrained DRO over a fixed corruption registry."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE.parent),str(HERE.parent.parent/"duplex")]
from relation_v4.io import fit_ecdf,apply_ecdf,sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.robust import fit
from relation_v11.score_stream_benchmark import expanded,calibrate_pair,scenarios,metrics
from relation_v9.train import load as load_v9
from hate_common import data as hdata
from relation_v8.model import UnifiedRelationV8
from relation_v10.diagnostic import components as v10_components,score as v10_score
import torch

ALPHAS=(0.,.5,1.);LAMBDAS=(.25,.5,1.)
def centers(x,state):
 classes=state['replication_classes'];return np.stack([np.median(np.stack([x[:,eq].mean(1) for eq in unique],1),axis=1) for unique in classes],1)
def predict(values,state,ap,al,lam):
 c=len(state['clusters']);equal=np.full(c,1/c);mass=np.asarray(state['cluster_mass']);wp=(1-ap)*equal+ap*mass;wl=(1-al)*equal+al*mass;out={}
 for v,x in values.items():
  z=centers(x,state);prior=float(z.mean(0)@wp);locator=(z-z.mean(0,keepdims=True))@wl;out[v]=prior+lam*locator
 return out
def corrected_predict(values,base_values,state,ap,al,lam):
 candidate=predict(values,state,ap,al,lam);identity=predict(values,state,0.,0.,1.)
 return {v:base_values[v].mean(1)+(candidate[v]-identity[v]) for v in base_values}
def candidates():return [{'prior_alpha':a,'locator_alpha':b,'lambda':l} for a in ALPHAS for b in ALPHAS for l in LAMBDAS]
def evaluate_registry(registry,base_registry,gt,fit_ids,eval_ids,candidate):
 rows={}
 for name,(fit_values,eval_values) in registry.items():
  state=fit(np.concatenate([fit_values[v] for v in fit_ids]));pred=corrected_predict({v:eval_values[v] for v in eval_ids},{v:base_registry[name][1][v] for v in eval_ids},state,candidate['prior_alpha'],candidate['locator_alpha'],candidate['lambda']);rows[name]=metrics(pred,{v:gt[v] for v in eval_ids})
 return rows
def objective(rows):return (min(x['frame_ap'] for x in rows.values()),min(x['frame_roc'] for x in rows.values()))
def noninferior(rows,base,tol=1e-12):return rows['clean']['frame_ap']+tol>=base['clean']['frame_ap'] and rows['clean']['frame_roc']+tol>=base['clean']['frame_roc']
def frozen_v10_identity(manifest,raw):
 corpus=manifest['corpus'];path=Path('results/reproduction/relation_v10/diagnostic_stable')/corpus/'frozen_config.json';cfg=json.load(open(path));values=apply_ecdf(raw,[np.asarray(x) for x in cfg['calibration']]);model=UnifiedRelationV8(len(cfg['candidate_weights']['identity']),manifest.get('window',12),manifest.get('temperature',.2)).eval();parts=v10_components(model,values);sel=cfg['identity_v8_fallback'];scores=v10_score(parts,np.asarray(cfg['candidate_weights']['identity']),sel['beta'],sel['gamma']);return {v:s[:,None] for v,s in scores.items()},path
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--v11',required=True);p.add_argument('--out',required=True);a=p.parse_args();m=json.load(open(a.manifest));vr,vg,_=load_split_exact(m,'val');tr,tg,_=load_split_exact(m,'test');has_train=all('train_scores' in e for e in m['experts'])
 # Immutable external V10 identity is the raw per-expert consensus.  The
 # train-fitted copula state below is used only for a zero-at-identity
 # correction, never to recalibrate or replace this performance fallback.
 base_val,base_test=vr,tr;bev,_=expanded(m,'val',vg);bet,_=expanded(m,'test',tg);base_val_self=scenarios(base_val,base_val,bev,bev,vg);base_test_registry=scenarios(base_val,base_test,bev,bet,vg);v10v,v10_path=frozen_v10_identity(m,vr);v10t,_=frozen_v10_identity(m,tr);base_val_self['clean']=(v10v,v10v);base_test_registry['clean']=(v10v,v10t)
 if has_train:
  train_ids,train_raw,_=load_v9(m,'train');refs=fit_ecdf(train_raw);train=apply_ecdf(train_raw,refs);val=apply_ecdf(vr,refs);test=apply_ecdf(tr,refs);etrain,_=expanded(m,'train',train_ids);ev,_=expanded(m,'val',vg);et,_=expanded(m,'test',tg);etrain,ev,erefs=calibrate_pair(etrain,ev);et=apply_ecdf(et,erefs);labels=hdata.load_labels(m['corpus']);train_gt={v:np.full(len(train[v]),labels[v],np.int64) for v in train_ids};val_registry=scenarios(train,val,etrain,ev,train_gt);test_registry=scenarios(train,test,etrain,et,train_gt);fit_ids=train_ids;calibration_claim='train-fitted ECDF and corruption dependence state'
 else:
  val,test,refs=calibrate_pair(vr,tr);ev,_=expanded(m,'val',vg);et,_=expanded(m,'test',tg);ev,et,erefs=calibrate_pair(ev,et);base=scenarios(val,test,ev,et,vg);val_registry={k:(x,x) for k,(x,_) in base.items()};test_registry=base;fit_ids=None;calibration_claim='frozen validation-reference diagnostic only; NOT train-fitted robustness'
 registry=val_registry;base_registry=base_val_self;ids=sorted(vg);folds=[ids[i::5] for i in range(5)];identity={'prior_alpha':0.,'locator_alpha':0.,'lambda':1.};fold_rows=[]
 for i,held in enumerate(folds):
  inner=[v for v in ids if v not in set(held)];state_ids=fit_ids if fit_ids is not None else inner;base=evaluate_registry(registry,base_registry,vg,state_ids,held,identity);rows=[]
  base_objective=objective(base)
  for c in candidates():
   r=evaluate_registry(registry,base_registry,vg,state_ids,held,c);obj=objective(r);outer_gate=noninferior(r,base) and obj[0]+1e-12>=base_objective[0] and obj[1]+1e-12>=base_objective[1];rows.append({'candidate':c,'objective':obj,'clean':r['clean'],'eligible':bool(outer_gate)})
  eligible=[x for x in rows if x['eligible']];choice=max(eligible,key=lambda x:(x['objective'][0],x['objective'][1],-abs(x['candidate']['prior_alpha']),-abs(x['candidate']['locator_alpha']),-abs(x['candidate']['lambda']-1))) if eligible else {'candidate':identity,'fallback':True}
  fold_rows.append({'fold':i,'inner':len(inner),'held':len(held),'identity_clean':base['clean'],'identity_objective':base_objective,'choice':choice,'candidates':rows})
 # Full validation selection uses only candidates that passed clean constraint
 # on every held-out video fold.
 admissible=[]
 for c in candidates():
  if all(next(x for x in f['candidates'] if x['candidate']==c)['eligible'] for f in fold_rows):
   state_ids=fit_ids if fit_ids is not None else ids;r=evaluate_registry(registry,base_registry,vg,state_ids,ids,c);base=evaluate_registry(registry,base_registry,vg,state_ids,ids,identity)
   if noninferior(r,base):admissible.append({'candidate':c,'objective':objective(r),'clean':r['clean'],'validation_registry':r})
 selected=max(admissible,key=lambda x:(x['objective'][0],x['objective'][1],-abs(x['candidate']['prior_alpha']),-abs(x['candidate']['locator_alpha']),-abs(x['candidate']['lambda']-1))) if admissible else {'candidate':identity,'fallback':True}
 frozen={'method':'relation_v12_constrained_dro_role_fusion','corpus':m['corpus'],'grid':{'prior_alpha':ALPHAS,'locator_alpha':ALPHAS,'lambda':LAMBDAS},'registry':list(registry),'calibration_and_state':calibration_claim,'selection':'outer video-fold gate requires clean AP/ROC and worst-registry AP/ROC noninferiority to identity on every held fold; among admitted candidates maximize full-validation worst empirical Frame AP then ROC, with full-validation clean noninferiority','folds':fold_rows,'admissible':admissible,'selected':selected,'identity_fallback':identity,'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False};freeze=Path(a.out).with_suffix('.frozen.json');atomic_json(freeze,frozen)
 # Freeze each registry state's clusters on full validation, then apply test.
 test_rows={};c=selected['candidate']
 state_ids=fit_ids if fit_ids is not None else ids
 for name,(vv,tt) in test_registry.items():state=fit(np.concatenate([vv[v] for v in state_ids]));test_rows[name]=metrics(corrected_predict(tt,base_test_registry[name][1],state,c['prior_alpha'],c['locator_alpha'],c['lambda']),tg)
 # Explicitly test-informed oracle, never used above or claimed as selected.
 oracle=[]
 for cand in candidates():
  rr={}
  for name,(vv,tt) in test_registry.items():state=fit(np.concatenate([vv[v] for v in state_ids]));rr[name]=metrics(corrected_predict(tt,base_test_registry[name][1],state,cand['prior_alpha'],cand['locator_alpha'],cand['lambda']),tg)
  oracle.append({'candidate':cand,'objective':objective(rr)})
 oracle=max(oracle,key=lambda x:x['objective']);v11=json.load(open(a.v11));comparators={k:({**v,'status':'oracle diagnostic trained continuously on validation frame GT'} if k=='val_stacking' else {**v,'status':'implementation-equivalent to V11' if k=='huber' else 'baseline'}) for k,v in v11['summary'].items() if k!='v11'}
 summary={'clean':test_rows['clean'],**{f'average_{k}':float(np.mean([x[k] for x in test_rows.values()])) for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')},**{f'worst_{k}':float(np.min([x[k] for x in test_rows.values()])) for k in ('frame_ap','frame_roc','within_centered_ap','within_centered_roc','within_macro_ap','within_macro_roc')}}
 payload={'method':frozen['method'],'corpus':m['corpus'],'selected_from_validation':c,'test_registry':test_rows,'summary':summary,'comparators_from_same_registry':comparators,'test_informed_oracle_diagnostic':{**oracle,'status':'TEST-INFORMED; diagnostic only; not method selection'},'frozen_config':str(freeze.resolve()),'frozen_config_sha256':sha256(freeze),'test_labels_used_for_selected_config':False};atomic_json(a.out,payload);print(json.dumps({'corpus':m['corpus'],'selected':c,'summary':summary,'oracle':payload['test_informed_oracle_diagnostic']},indent=2))
if __name__=='__main__':main()
