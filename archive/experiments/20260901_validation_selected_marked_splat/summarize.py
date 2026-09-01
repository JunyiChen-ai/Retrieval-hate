#!/usr/bin/env python3
import argparse,json
from pathlib import Path
SOTA={'hatemm':{'ap':.5938315566,'roc':.8161837922,'within':.6315317180},'hateclipseg':{'ap':.6193710950,'roc':.6050224699,'within':.5619078936}}
OLD=Path('/home/jehc223/Retrieval-hate/runs/20260901_marked_temporal_splat_mil/pilot_seed234')
def metric(path):
    row=json.loads(path.read_text())['results']['score_final']; return {'ap':row['pr_auc'],'roc':row['roc_auc'],'within':row['per_video']['macro_auc']}
p=argparse.ArgumentParser(); p.add_argument('--formal-dir',required=True); root=Path(p.parse_args().formal_dir).resolve(); payload={'corpora':{}}
for corpus in SOTA:
    old=metric(OLD/corpus/'splat/metrics.json'); core=metric(root/corpus/'metrics.json'); payload['corpora'][corpus]={'old_single_config':old,'validation_selected':core,'delta':{k:core[k]-old[k] for k in core},'all_sota':all(core[k]>SOTA[corpus][k] for k in SOTA[corpus])}
payload['performance_gate']={'both_corpora_all_sota':all(r['all_sota'] for r in payload['corpora'].values())}; payload['decision']='EXPAND' if payload['performance_gate']['both_corpora_all_sota'] else 'FAIL_RESET3_METHOD_2'; (root/'summary.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
