#!/usr/bin/env python3
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score,roc_auc_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from relation_v18.run import read_frozen,raw_components,fuse,load_split_exact,frozen_v10_identity,sha256,atomic_json

def stable(pred,gt):
 ids=sorted(gt);y=np.concatenate([gt[v] for v in ids]);rank={v:rankdata(np.asarray(pred[v],dtype=np.float64),method='average')/len(pred[v]) for v in ids};r=np.concatenate([rank[v] for v in ids]);mixed=[v for v in ids if len(np.unique(gt[v]))==2]
 return {'definition':'tie-aware average per-video ranks; invariant to every per-video additive constant','within_rank_ap':float(average_precision_score(y,r)),'within_rank_roc':float(roc_auc_score(y,r)),'within_macro_ap':float(np.mean([average_precision_score(gt[v],rank[v]) for v in mixed])),'within_macro_roc':float(np.mean([roc_auc_score(gt[v],rank[v]) for v in mixed])),'mixed_videos':len(mixed)}
def perturb(loc,j,kind):
 out={}
 for v,x in loc.items():
  seed=int.from_bytes(hashlib.sha256(f'{kind}:{j}:{v}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
  if len(x)<2:out[v]=x.copy()
  elif kind=='time_permutation':out[v]=x[rng.permutation(len(x))]
  elif kind=='timestamp_circular_shift':out[v]=np.roll(x,1+seed%(len(x)-1))
 return out
def chunk_order_rows(rows,j):
 out=copy.deepcopy(rows);by={}
 for r in out:by.setdefault(r['video_id'],[]).append(r)
 for v,q in by.items():
  if len(q)<2:continue
  seed=int.from_bytes(hashlib.sha256(f'chunk_order:{j}:{v}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed);vals=[x['scores']['masked_branch_reset'] for x in q];perm=rng.permutation(len(q))
  for x,k in zip(q,perm):x['scores']['masked_branch_reset']=vals[k]
 return out
def summarize(rows):
 return {k:{'mean':float(np.mean([x[k] for x in rows])),'q025':float(np.quantile([x[k] for x in rows],.025)),'q975':float(np.quantile([x[k] for x in rows],.975))} for k in ('within_rank_ap','within_rank_roc')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen-config',required=True);p.add_argument('--raw-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args();f=json.load(open(a.frozen_config));m=json.load(open(f['manifest']));_,gt,_=load_split_exact(m,'test');tr,_,_=load_split_exact(m,'test');base,_=frozen_v10_identity(m,tr);base={v:x[:,0] for v,x in base.items()};rows,rm=read_frozen(a.raw_dir);ids=sorted(gt);g,l,_=raw_components(rows,ids,{v:len(gt[v]) for v in ids},base,f['formula_state']);aa=f['selected']['alpha'];bb=f['selected']['beta'];pred=fuse(base,g,l,aa,bb);sb=stable(base,gt);sp=stable(pred,gt)
 if bb==0 and sp!=sb:raise RuntimeError('beta0 stable centered metrics must be bit-identical to identity')
 payload={'corpus':f['corpus'],'method':f['method'],'selected_alpha_beta':[aa,bb],'raw_manifest_sha256':sha256(Path(a.raw_dir)/'raw_manifest.json'),'stable_identity':sb,'stable_selected':sp,'beta0_exact_centered_identity':bool(bb!=0 or sp==sb),'legacy_constant_subtraction_centered_metrics':'deprecated because floating subtraction can perturb ties'}
 if a.controls:
  B=200;ctrl={}
  for kind in ('time_permutation','timestamp_circular_shift'):
   vals=[stable(fuse(base,g,perturb(l,j,kind),aa,bb),gt) for j in range(B)];ctrl[kind]={'B':B,**summarize(vals)}
  vals=[]
  for j in range(B):
   rr=chunk_order_rows(rows,j);_,ll,_=raw_components(rr,ids,{v:len(gt[v]) for v in ids},base,f['formula_state']);vals.append(stable(fuse(base,g,ll,aa,bb),gt))
  ctrl['chunk_order_permutation']={'B':B,**summarize(vals)};payload['controls']=ctrl
 atomic_json(a.out,payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
