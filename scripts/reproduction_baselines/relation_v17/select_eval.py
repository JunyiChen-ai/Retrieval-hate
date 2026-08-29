#!/usr/bin/env python3
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from hate_common import data as hdata
from relation_v4.io import sha256
from relation_v8.run import load_split_exact,atomic_json
from relation_v11.score_stream_benchmark import metrics
from relation_v12.diagnostic import frozen_v10_identity
def read_frozen(directory):
 d=Path(directory);m=json.load(open(d/'raw_manifest.json'));p=d/'per_chunk_raw.jsonl'
 if not m['raw_frozen_before_gt'] or m['raw_sha256']!=sha256(p):raise RuntimeError('raw hash/freeze failure')
 return list(map(json.loads,open(p))),m
def raw_roles(rows,ids,lengths,prior_key,locator_key,state=None):
 by={v:[] for v in ids}
 for r in rows:
  if r['video_id'] in by and r.get('temporal_span_valid',True) and r['start'] is not None and r['end'] is not None:by[r['video_id']].append(r)
 raw_prior={};raw_locator={}
 for v in ids:
  q=by[v]
  if not q:raw_prior[v]=None;raw_locator[v]=np.zeros(lengths[v]);continue
  raw_prior[v]=float(np.mean([x['scores'][prior_key] for x in q]));centers=np.asarray([(x['start']+x['end'])/2 for x in q]);vals=np.asarray([x['scores'][locator_key] for x in q]);nearest=np.abs((np.arange(lengths[v])+.5)[:,None]-centers[None]).argmin(1);z=vals[nearest];raw_locator[v]=z-z.mean()
 if state is None:
  pv=np.asarray([x for x in raw_prior.values() if x is not None]);pm=float(pv.mean());ps=float(pv.std()+1e-8);ls=float(np.sqrt(np.mean(np.concatenate([raw_locator[v] for v in ids if raw_prior[v] is not None])**2))+1e-8);state={'prior_mean':pm,'prior_std':ps,'locator_rms':ls,'missing_evidence_policy':'zero correction; identity fallback for that video'}
 prior={v:np.zeros(lengths[v]) if raw_prior[v] is None else np.full(lengths[v],(raw_prior[v]-state['prior_mean'])/state['prior_std']) for v in ids};locator={v:np.zeros(lengths[v]) if raw_prior[v] is None else raw_locator[v]/state['locator_rms'] for v in ids}
 if any(abs(x.mean())>1e-10 for x in locator.values()):raise RuntimeError('locator not centered')
 return prior,locator,state
def add(base,prior,locator,lam):return {v:base[v]+lam*(prior[v]+locator[v]) for v in base}
def shuffled(locator,j):
 out={}
 for v,x in locator.items():
  if len(x)<2:out[v]=x.copy();continue
  k=1+int.from_bytes(hashlib.sha256(f'{j}:{v}'.encode()).digest()[:4],'little')%(len(x)-1);out[v]=np.roll(x,k)
 return out
def paired_ci(base,pred,gt,b,seed=1717):
 ids=sorted(gt);rng=np.random.default_rng(seed);keys=('frame_ap','frame_roc','within_macro_ap','within_macro_roc');rows=[]
 for _ in range(b):
  sample=rng.choice(ids,len(ids),replace=True);bb={};pp={};gg={}
  for j,v in enumerate(sample):k=f'{j}:{v}';bb[k]=base[v];pp[k]=pred[v];gg[k]=gt[v]
  mb=metrics(bb,gg);mp=metrics(pp,gg);rows.append({k:mp[k]-mb[k] for k in keys})
 return {k:{'delta':float(np.mean([x[k] for x in rows])),'lower95':float(np.quantile([x[k] for x in rows],.025)),'upper95':float(np.quantile([x[k] for x in rows],.975))} for k in keys}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--val-dir',required=True);p.add_argument('--test-raw-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);cfg=json.load(open(Path(a.val_dir)/'preregistered_config.json'));vr,vg,_=load_split_exact(json.load(open(a.manifest)),'val');basev,_=frozen_v10_identity(json.load(open(a.manifest)),vr);basev={v:x[:,0] for v,x in basev.items()};ids=sorted(vg);lengths={v:len(vg[v]) for v in ids};rows,rawman=read_frozen(a.val_dir);prior,locator,state=raw_roles(rows,ids,lengths,'causal_continuous','masked_branch_reset');sprior,slocator,sstate=raw_roles(rows,ids,lengths,'masked_branch_reset','causal_continuous');identity=metrics(basev,vg);grid=[]
 for lam in cfg['lambda_grid']:
  pred=add(basev,prior,locator,lam);mm=metrics(pred,vg);ci=paired_ci(basev,pred,vg,cfg['paired_video_bootstrap_B']);sh=[metrics(add(basev,prior,shuffled(locator,j),lam),vg) for j in range(cfg['shuffle_gate_B'])];qap=float(np.quantile([x['within_macro_ap'] for x in sh],.95));qroc=float(np.quantile([x['within_macro_roc'] for x in sh],.95));pooled=ci['frame_ap']['lower95']>=-1e-12 and ci['frame_roc']['lower95']>=-1e-12;within=(ci['within_macro_ap']['lower95']>0 and ci['within_macro_roc']['lower95']>=-1e-12) or (ci['within_macro_roc']['lower95']>0 and ci['within_macro_ap']['lower95']>=-1e-12);sg=(mm['within_macro_ap']>qap and ci['within_macro_roc']['lower95']>=-1e-12) or (mm['within_macro_roc']>qroc and ci['within_macro_ap']['lower95']>=-1e-12);grid.append({'lambda':lam,'metrics':mm,'paired_video_ci':ci,'shuffle_q95':{'within_macro_ap':qap,'within_macro_roc':qroc},'pooled_gate':bool(pooled),'within_gate':bool(within),'shuffle_gate':bool(sg),'eligible':bool(lam==0 or (pooled and within and sg))})
 selected=max([x for x in grid if x['eligible']],key=lambda x:(x['metrics']['frame_ap'],x['metrics']['frame_roc'],x['metrics']['within_macro_ap'],x['metrics']['within_macro_roc'],-x['lambda']));frozen={'method':cfg['method'],'test_informed_design_from_v16':True,'corpus':cfg['corpus'],'roles':cfg['roles'],'normalization_state':state,'swapped_normalization_state':sstate,'lambda_grid':cfg['lambda_grid'],'validation_grid':grid,'selected_lambda':selected['lambda'],'identity_fallback':0.,'val_raw_manifest_sha256':sha256(Path(a.val_dir)/'raw_manifest.json'),'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False};atomic_json(out/'frozen_config.json',frozen)
 # Only after configuration freeze, open the already-hashed V16 test raw and GT.
 tr,tg,_=load_split_exact(json.load(open(a.manifest)),'test');baset,_=frozen_v10_identity(json.load(open(a.manifest)),tr);baset={v:x[:,0] for v,x in baset.items()};tids=sorted(tg);trows,tman=read_frozen(a.test_raw_dir);tl={v:len(tg[v]) for v in tids};tp,tlc,_=raw_roles(trows,tids,tl,'causal_continuous','masked_branch_reset',state);tsp,tsl,_=raw_roles(trows,tids,tl,'masked_branch_reset','causal_continuous',sstate);lam=selected['lambda'];variants={'v8_identity':baset,'causal_prior_only':add(baset,tp,{v:np.zeros_like(x) for v,x in tlc.items()},lam),'masked_locator_only':add(baset,{v:np.zeros_like(x) for v,x in tp.items()},tlc,lam),'dual_full':add(baset,tp,tlc,lam),'swapped_roles':add(baset,tsp,tsl,lam)};results={k:metrics(v,tg) for k,v in variants.items()};sh=[metrics(add(baset,tp,shuffled(tlc,j),lam),tg) for j in range(cfg['test_shuffle_B'])];results['time_shuffle']={k:{'mean':float(np.mean([x[k] for x in sh])),'q025':float(np.quantile([x[k] for x in sh],.025)),'q975':float(np.quantile([x[k] for x in sh],.975))} for k in ('frame_ap','frame_roc','within_macro_ap','within_macro_roc')};payload={'method':cfg['method'],'corpus':cfg['corpus'],'test_informed_design_from_v16':True,'selected_lambda':lam,'test':results,'test_raw_manifest_sha256':sha256(Path(a.test_raw_dir)/'raw_manifest.json'),'test_labels_used_for_selection':False,'frozen_config_sha256':sha256(out/'frozen_config.json')};atomic_json(out/'test_eval.json',payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
