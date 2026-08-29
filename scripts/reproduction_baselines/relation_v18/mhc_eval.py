#!/usr/bin/env python3
"""Evaluate a frozen MHC V18 selection, plus explicitly unclaimable test oracle."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from relation_v18.run import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen-config',required=True);p.add_argument('--test-raw-dir',required=True);a=p.parse_args();fp=Path(a.frozen_config);f=json.load(open(fp));manifest=json.load(open(f['manifest']));tr,tg,_=load_split_exact(manifest,'test');bt,_=frozen_v10_identity(manifest,tr);bt={v:x[:,0] for v,x in bt.items()};rows,_=read_frozen(a.test_raw_dir);ids=sorted(tg);gc,lc,_=raw_components(rows,ids,{v:len(tg[v]) for v in ids},bt,f['formula_state']);aa=f['selected']['alpha'];bb=f['selected']['beta'];identity=metrics(bt,tg);formal=metrics(fuse(bt,gc,lc,aa,bb),tg);surface=[]
 for alpha in GRID:
  for beta in GRID:
   mm=metrics(fuse(bt,gc,lc,alpha,beta),tg);surface.append({'alpha':alpha,'beta':beta,'metrics':mm,'four_metric_double_positive':all(mm[k]>identity[k] for k in KEYS)})
 best=max(surface,key=lambda x:tuple(x['metrics'][k] for k in KEYS));payload={'method':f['method'],'corpus':f['corpus'],'formal_selected':{'alpha':aa,'beta':bb,'metrics':formal},'identity':identity,'test_labels_used_for_formal_selection':False,'TEST_INFORMED_ORACLE_DIAGNOSTIC_NOT_CLAIMABLE':{'surface':surface,'best_lexicographic':best,'n_four_metric_double_positive':sum(x['four_metric_double_positive'] for x in surface)},'test_raw_manifest_sha256':sha256(Path(a.test_raw_dir)/'raw_manifest.json'),'frozen_config_sha256':sha256(fp)};atomic_json(fp.parent/'test_eval_and_oracle.json',payload);print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
