#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--corpus-dir',required=True); root=Path(p.parse_args().corpus_dir).resolve(); rows=[]
for path in sorted(root.glob('trials/*/train_log.json')):
    log=json.loads(path.read_text()); rows.append({'config_name':log['config_name'],'selected_validation_video_ap':log['selected_validation_video_ap'],'selected_epoch':log['selected_epoch'],'trial_dir':str(path.parent)})
if not rows: raise RuntimeError('no completed validation trials')
rows.sort(key=lambda r:(-r['selected_validation_video_ap'],r['config_name'])); payload={'selection_rule':'maximum validation video AP; lexical config-name tie break','test_predictions_read_during_selection':False,'selected':rows[0],'all_trials':rows}; (root/'selection.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
