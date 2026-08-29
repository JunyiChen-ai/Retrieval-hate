#!/usr/bin/env python3
"""Validation-only MHC V18 selector. It cannot open test raw or test GT."""
import argparse,json,sys,multiprocessing as mp
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from relation_v18.run import *

_CTX={}
def gate_worker(pair):
 alpha,beta=pair;base,gc,lc,gt=_CTX['base'],_CTX['global'],_CTX['local'],_CTX['gt']
 pred=fuse(base,gc,lc,alpha,beta);ci=paired_ci(base,pred,gt,2000,seed=1818);ci_gate=all(ci[k]['lower95']>=-1e-12 for k in KEYS)
 if beta>0:
  sh=[metrics(fuse(base,gc,shuffled(lc,j),alpha,beta),gt) for j in range(200)];q={k:float(np.quantile([x[k] for x in sh],.95)) for k in ('within_macro_ap','within_macro_roc')};sg=all(metrics(pred,gt)[k]>q[k] for k in q)
 else:q=None;sg=True
 return pair,ci,q,bool(ci_gate),bool(sg)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--val-dir',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
 manifest=json.load(open(a.manifest));vr,vg,_=load_split_exact(manifest,'val');bv,_=frozen_v10_identity(manifest,vr);bv={v:x[:,0] for v,x in bv.items()};rows,_=read_frozen(a.val_dir);ids=sorted(vg);gc,lc,state=raw_components(rows,ids,{v:len(vg[v]) for v in ids},bv);identity=metrics(bv,vg);surface=[]
 for alpha in GRID:
  for beta in GRID:
   mm=metrics(fuse(bv,gc,lc,alpha,beta),vg);surface.append({'alpha':alpha,'beta':beta,'metrics':mm,'point_pareto_noninferior':point_pareto(mm,identity)})
 candidates=[(x['alpha'],x['beta']) for x in surface if x['point_pareto_noninferior']]
 _CTX.update({'base':bv,'global':gc,'local':lc,'gt':vg})
 # Fork preserves the frozen in-memory arrays copy-on-write; every worker uses
 # the same seed/rules as the serial evaluator. Keep below the 16-CPU cap.
 with mp.get_context('fork').Pool(processes=min(12,len(candidates))) as pool: gated=dict((pair,(ci,q,cg,sg)) for pair,ci,q,cg,sg in pool.map(gate_worker,candidates))
 for cell in surface:
  pair=(cell['alpha'],cell['beta'])
  if pair not in gated:cell.update({'bootstrap_run':False,'eligible':False});continue
  ci,q,ci_gate,sg=gated[pair];cell.update({'bootstrap_run':True,'paired_video_ci_B2000':ci,'ci_pareto_gate':ci_gate,'local_shuffle_B200_q95':q,'local_shuffle_gate':sg,'eligible':bool(pair==(0,0) or (ci_gate and sg))})
 selected=max([x for x in surface if x['eligible']],key=lambda x:tuple(x['metrics'][k] for k in KEYS)+(-x['alpha'],-x['beta']))
 frozen={'method':'relation_v18_low_cost_dual_pareto_fusion_mhc','test_informed_design_from_v16':True,'corpus':manifest['corpus'],'formula_state':state,'grid':GRID,'selection_rule':'same V18 four-metric point Pareto + B2000 paired CI + beta local B200 shuffle; identity fallback','validation_identity':identity,'validation_surface':surface,'selected':{k:selected[k] for k in ('alpha','beta','metrics')},'val_raw_manifest_sha256':sha256(Path(a.val_dir)/'raw_manifest.json'),'manifest':str(Path(a.manifest).resolve()),'manifest_sha256':sha256(a.manifest),'test_opened':False,'uncovered_policy':'nearest-center extrapolation; zero-valid exact identity'};atomic_json(out/'frozen_config.json',frozen);print(json.dumps(frozen['selected'],indent=2))
if __name__=='__main__':main()
