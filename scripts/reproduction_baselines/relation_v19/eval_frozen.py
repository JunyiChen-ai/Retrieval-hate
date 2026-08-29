#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE.parent),str(ROOT/'scripts/duplex')]
from relation_v19.run import *
from relation_v18.run import read_frozen,raw_components,fuse,shuffled,sha256,load_split_exact,frozen_v10_identity,atomic_json
def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen-config',required=True);p.add_argument('--test-raw-dir',required=True);a=p.parse_args();fp=Path(a.frozen_config);f=json.load(open(fp));m=json.load(open(f['manifest']));tr,tg,_=load_split_exact(m,'test');base,_=frozen_v10_identity(m,tr);base={v:x[:,0] for v,x in base.items()};rows,_=read_frozen(a.test_raw_dir);ids=sorted(tg);g,l,_=raw_components(rows,ids,{v:len(tg[v]) for v in ids},base,f['formula_state']);aa=f['selected']['alpha'];bb=f['selected']['beta'];identity=metrics(base,tg);pred=fuse(base,g,l,aa,bb);formal=metrics(pred,tg);ci=report_ci(base,pred,tg,2000,1920)
 if bb>0:
  sh=[metrics(fuse(base,g,shuffled(l,j),aa,bb),tg) for j in range(200)];sr={'B':200,'actual':{k:formal[k] for k in ('within_macro_ap','within_macro_roc')},'q025_q50_q975':{k:[float(np.quantile([x[k] for x in sh],q)) for q in (.025,.5,.975)] for k in ('within_macro_ap','within_macro_roc')}}
 else:sr={'B':200,'not_run_reason':'selected beta=0; local shuffle exactly invariant'}
 out={'method':f['method'],'TEST_INFORMED_DEVELOPMENT_VERSION':True,'corpus':f['corpus'],'selected_alpha':aa,'selected_beta':bb,'test_identity':identity,'test_selected':formal,'test_relative_v10':delta(formal,identity),'test_paired_video_bootstrap_B2000_report_only':ci,'test_time_shuffle_report_only':sr,'test_labels_used_for_selection':False,'test_raw_manifest_sha256':sha256(Path(a.test_raw_dir)/'raw_manifest.json'),'frozen_config_sha256':sha256(fp)};atomic_json(fp.parent/'test_eval.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
