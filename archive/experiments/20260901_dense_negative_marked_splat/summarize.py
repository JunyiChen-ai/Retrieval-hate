#!/usr/bin/env python3
import argparse,json
from pathlib import Path
SOTA={'hatemm':{'ap':.5938315566,'roc':.8161837922,'within':.6315317180},'hateclipseg':{'ap':.6193710950,'roc':.6050224699,'within':.5619078936}};BASE=Path('/home/jehc223/Retrieval-hate/runs/20260901_validation_selected_marked_splat/formal_seed234')
def metric(path):
 r=json.loads(path.read_text())['results']['score_final'];return {'ap':r['pr_auc'],'roc':r['roc_auc'],'within':r['per_video']['macro_auc']}
p=argparse.ArgumentParser();p.add_argument('--formal-dir',required=True);root=Path(p.parse_args().formal_dir).resolve();out={'corpora':{}}
for c in SOTA:
 b=metric(BASE/c/'metrics.json');x=metric(root/c/'metrics.json');out['corpora'][c]={'validation_selected_base':b,'dense_negative':x,'delta':{k:x[k]-b[k] for k in x},'all_sota':all(x[k]>SOTA[c][k] for k in SOTA[c])}
out['performance_gate']={'both_corpora_all_sota':all(r['all_sota'] for r in out['corpora'].values())};out['decision']='EXPAND' if out['performance_gate']['both_corpora_all_sota'] else 'FAIL_RESET3_METHOD_3_TRIGGER_REVIEW';(root/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
