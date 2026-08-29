#!/usr/bin/env python3
"""Build label-free V22 speech-mask components from frozen THVL V16 raw."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RAW=ROOT/'results/steward_private/thvl_bench/v16_val_raw_frozen_32_v2'
DEFAULT_QC=ROOT/'results/steward_private/thvl_bench/val32_download_qc.json'

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def ecdf_fit(x):
 x=np.asarray(x,dtype=np.float64)
 if not len(x) or not np.isfinite(x).all():raise ValueError('ECDF reference must be nonempty finite')
 return np.sort(x)
def ecdf_apply(ref,x):
 ref=np.asarray(ref,dtype=np.float64);x=np.asarray(x,dtype=np.float64)
 return (np.searchsorted(ref,x,'left')+np.searchsorted(ref,x,'right'))/(2*len(ref))
def project(rows,duration):
 """Overlap-weight chunk scores onto half-open 1Hz bins; return raw global/local/mask."""
 T=max(1,int(math.ceil(float(duration)))); num=np.zeros(T);den=np.zeros(T)
 causal=[]
 for r in rows:
  s,e=float(r['start']),float(r['end'])
  if not (math.isfinite(s) and math.isfinite(e) and e>s):raise ValueError('invalid frozen chunk span')
  mv=float(r['scores']['masked_branch_reset']);cv=float(r['scores']['causal_continuous'])
  if not (math.isfinite(mv) and math.isfinite(cv)):raise ValueError('nonfinite score')
  causal.append(cv)
  for t in range(max(0,int(math.floor(s))),min(T,int(math.ceil(e)))):
   w=max(0.0,min(e,t+1.0)-max(s,float(t)))
   if w:num[t]+=w*mv;den[t]+=w
 mask=den>0;loc=np.zeros(T)
 if mask.any():
  loc[mask]=num[mask]/den[mask];loc[mask]-=loc[mask].mean()
 return (None if not causal else float(np.mean(causal))),loc,mask
def calibrate(items,state=None):
 available=[x for x in items if x['available']]
 if state is None:
  state={'global_ecdf':ecdf_fit([x['global_raw'] for x in available]).tolist(),
         'local_centered_ecdf':ecdf_fit(np.concatenate([x['local_raw'][x['speech_mask']] for x in available if x['speech_mask'].any()])).tolist(),
         'ecdf_convention':'mid-distribution rank; local rank transformed then re-centered over observed frames only'}
 for x in items:
  T=len(x['local_raw']);g=np.zeros(T);l=np.zeros(T);m=x['speech_mask']
  if x['available']:
   g[:]=float(ecdf_apply(state['global_ecdf'],[x['global_raw']])[0]-.5)
   if m.any():
    l[m]=ecdf_apply(state['local_centered_ecdf'],x['local_raw'][m])-.5;l[m]-=l[m].mean()
  x['global_calibrated']=g;x['local_calibrated']=l
 return state
def fuse(global_c,local_c,alpha,beta):return alpha*np.asarray(global_c)+beta*np.asarray(local_c)
def shuffle_observed(local,mask,seed):
 out=np.asarray(local,dtype=np.float64).copy();m=np.asarray(mask,dtype=bool);rng=np.random.default_rng(seed);out[m]=rng.permutation(out[m]);out[~m]=0.;return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-dir',default=str(DEFAULT_RAW));p.add_argument('--qc',default=str(DEFAULT_QC));p.add_argument('--out-dir',required=True);a=p.parse_args()
 d=Path(a.raw_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
 cfg=json.load(open(d/'preregistered_config.json'));rm=json.load(open(d/'raw_manifest.json'));qc=json.load(open(a.qc))
 if sha256(d/'per_chunk_raw.jsonl')!=rm['raw_sha256'] or sha256(d/'preregistered_config.json')!=rm['config_sha256']:raise RuntimeError('frozen raw/config hash mismatch')
 rows=[json.loads(x) for x in open(d/'per_chunk_raw.jsonl')];by={v:[] for v in cfg['video_ids']}
 for r in rows:
  if r['video_id'] not in by:raise RuntimeError('raw ID outside frozen cohort')
  by[r['video_id']].append(r)
 durations={r['hashed_id']:sum(float(x.get('duration_seconds',0)) for x in r['paths']) for r in qc['rows']}
 if set(durations)!=set(by) or any(not math.isfinite(x) or x<=0 for x in durations.values()):raise RuntimeError('QC duration exact cohort/finite positive required')
 items=[]
 for v in cfg['video_ids']:
  g,l,m=project(by[v],durations[v]);items.append({'video_id':v,'duration_seconds':durations[v],'available':g is not None,'global_raw':g,'local_raw':l,'speech_mask':m})
 state=calibrate(items);op=out/'components.jsonl.tmp'
 with open(op,'x') as f:
  for x in items:
   f.write(json.dumps({'video_id':x['video_id'],'duration_seconds':x['duration_seconds'],'n_frames':len(x['local_raw']),'available':x['available'],'global_raw':x['global_raw'],'global_calibrated':x['global_calibrated'].tolist(),'local_centered_raw':x['local_raw'].tolist(),'local_centered_ecdf':x['local_calibrated'].tolist(),'speech_observed_mask':x['speech_mask'].astype(int).tolist()})+'\n')
 os.replace(op,out/'components.jsonl')
 manifest={'method':'relation_v22_speech_mask_components','label_or_gt_opened':False,'n_videos':len(items),'n_available':sum(x['available'] for x in items),'n_frames':sum(len(x['local_raw']) for x in items),'n_observed_frames':sum(int(x['speech_mask'].sum()) for x in items),'semantics':{'missing':'available=false, corrections numeric zero; missing is not a negative label','uncovered':'speech_observed_mask=false and local exactly zero','centering':'per-video over observed speech frames only','global':'arithmetic mean causal-continuous chunk margin, constant over video after ECDF','local_calibration':'pooled validation-reference ECDF of centered observed margins, then per-video observed-only re-centering','fusion':'alpha*global + beta*local; beta=0 is exact global-only','shuffle':'permute local values only among observed frames; uncovered remain exact zero'},'calibration_state':state,'source':{'raw_sha256':sha256(d/'per_chunk_raw.jsonl'),'raw_manifest_sha256':sha256(d/'raw_manifest.json'),'config_sha256':sha256(d/'preregistered_config.json'),'qc_sha256':sha256(a.qc),'builder_sha256':sha256(__file__)},'components_sha256':sha256(out/'components.jsonl')}
 atomic_json(out/'manifest.json',manifest);print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
