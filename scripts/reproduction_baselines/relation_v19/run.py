#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from relation_v18.run import GRID,KEYS,read_frozen,raw_components,fuse,metrics,paired_ci,shuffled,sha256,load_split_exact,frozen_v10_identity,atomic_json

def delta(m,b):return {k:float(m[k]-b[k]) for k in KEYS}
def report_ci(base,pred,gt,B,seed):
 ids=sorted(gt);rng=np.random.default_rng(seed);vals={k:[] for k in KEYS}
 for _ in range(B):
  sample=rng.choice(ids,len(ids),replace=True);bb={};pp={};gg={}
  for j,v in enumerate(sample):k=f'{j}:{v}';bb[k]=base[v];pp[k]=pred[v];gg[k]=gt[v]
  mb=metrics(bb,gg);mp=metrics(pp,gg)
  for k in KEYS:
   d=mp[k]-mb[k]
   if np.isfinite(d):vals[k].append(float(d))
 out={}
 for k,x in vals.items():
  if not x:out[k]={'n_valid':0,'delta':None,'lower95':None,'upper95':None};continue
  out[k]={'n_valid':len(x),'delta':float(np.mean(x)),'lower95':float(np.quantile(x,.025)),'upper95':float(np.quantile(x,.975))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--val-raw-dir',required=True);p.add_argument('--test-raw-dir');p.add_argument('--selection-only',action='store_true');p.add_argument('--out-dir',required=True);p.add_argument('--method',default='relation_v19_standard_validation_selection');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False);manifest=json.load(open(a.manifest));corpus=manifest['corpus']
 vr,vg,_=load_split_exact(manifest,'val');bv,_=frozen_v10_identity(manifest,vr);bv={v:x[:,0] for v,x in bv.items()};vrows,_=read_frozen(a.val_raw_dir);ids=sorted(vg);gc,lc,state=raw_components(vrows,ids,{v:len(vg[v]) for v in ids},bv);identity=metrics(bv,vg);surface=[]
 for alpha in GRID:
  for beta in GRID:
   mm=metrics(fuse(bv,gc,lc,alpha,beta),vg);eligible=all(mm[k]>=identity[k]-1e-12 for k in KEYS);surface.append({'alpha':alpha,'beta':beta,'metrics':mm,'eligible':bool(eligible)})
 selected=max([x for x in surface if x['eligible']],key=lambda x:(x['metrics']['frame_ap'],x['metrics']['frame_roc'],-abs(x['alpha']),-abs(x['beta'])))
 pred=fuse(bv,gc,lc,selected['alpha'],selected['beta']);ci=report_ci(bv,pred,vg,2000,seed=1919)
 if selected['beta']>0:
  sh=[metrics(fuse(bv,gc,shuffled(lc,j),selected['alpha'],selected['beta']),vg) for j in range(200)];shuffle={'B':200,'q025_q50_q975':{k:[float(np.quantile([x[k] for x in sh],q)) for q in (.025,.5,.975)] for k in ('within_macro_ap','within_macro_roc')}}
 else:shuffle={'B':200,'not_run_reason':'selected beta=0; local shuffle is exactly invariant'}
 frozen={'method':a.method,'status':'FROZEN_BEFORE_TEST','TEST_INFORMED_DEVELOPMENT_VERSION':True,'corpus':corpus,'grid':GRID,'selection_rule':'eligible iff validation point pooled AP/ROC and within macro AP/ROC each >= V10 identity with tolerance 1e-12; select max (AP,ROC,-abs(alpha),-abs(beta)); bootstrap/shuffle report-only','formula_state':state,'validation_identity':identity,'validation_surface':surface,'selected':{'alpha':selected['alpha'],'beta':selected['beta'],'metrics':selected['metrics'],'relative_v10':delta(selected['metrics'],identity),'paired_video_bootstrap_B2000_report_only':ci,'time_shuffle_report_only':shuffle},'val_raw_manifest_sha256':sha256(Path(a.val_raw_dir)/'raw_manifest.json'),'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False};atomic_json(out/'frozen_config.json',frozen)
 if a.selection_only:print(json.dumps(frozen['selected'],indent=2));return
 if not a.test_raw_dir:raise RuntimeError('--test-raw-dir required unless --selection-only')
 # Test is opened only after the standard validation choice is frozen.
 tr,tg,_=load_split_exact(manifest,'test');bt,_=frozen_v10_identity(manifest,tr);bt={v:x[:,0] for v,x in bt.items()};trows,_=read_frozen(a.test_raw_dir);tids=sorted(tg);tgc,tlc,_=raw_components(trows,tids,{v:len(tg[v]) for v in tids},bt,state);identity_t=metrics(bt,tg);formal=metrics(fuse(bt,tgc,tlc,selected['alpha'],selected['beta']),tg);test_ci=report_ci(bt,fuse(bt,tgc,tlc,selected['alpha'],selected['beta']),tg,2000,seed=1920)
 if selected['beta']>0:
  tsh=[metrics(fuse(bt,tgc,shuffled(tlc,j),selected['alpha'],selected['beta']),tg) for j in range(200)];test_shuffle={'B':200,'actual':{k:formal[k] for k in ('within_macro_ap','within_macro_roc')},'q025_q50_q975':{k:[float(np.quantile([x[k] for x in tsh],q)) for q in (.025,.5,.975)] for k in ('within_macro_ap','within_macro_roc')}}
 else:test_shuffle={'B':200,'not_run_reason':'selected beta=0; local shuffle is exactly invariant'}
 payload={'method':frozen['method'],'TEST_INFORMED_DEVELOPMENT_VERSION':True,'corpus':corpus,'selected_alpha':selected['alpha'],'selected_beta':selected['beta'],'test_identity':identity_t,'test_selected':formal,'test_relative_v10':delta(formal,identity_t),'test_paired_video_bootstrap_B2000_report_only':test_ci,'test_time_shuffle_report_only':test_shuffle,'test_labels_used_for_selection':False,'test_raw_manifest_sha256':sha256(Path(a.test_raw_dir)/'raw_manifest.json'),'frozen_config_sha256':sha256(out/'frozen_config.json')};atomic_json(out/'test_eval.json',payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
