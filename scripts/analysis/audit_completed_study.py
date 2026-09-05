"""Audit a completed study against evaluator files, without recalculating metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--study', type=Path, required=True)
args = p.parse_args()
root = args.study
read = lambda name: json.loads((root / name).read_text())
s = read('study_summary.json')
budget = read('budget.json')['n_trials']
assert s['n_trials'] == len(s['trials']) == budget
assert {t['number'] for t in s['trials']} == set(range(budget))
for t in s['trials']:
    assert t['state'] in ['COMPLETE', 'PRUNED']
    prefix = f"trial{t['number']}"
    run = read(f'{prefix}/summary.json')
    assert len(run['history']) == 50 and 1 <= run['selected_epoch'] <= 50
    assert run['hparams'] == read(f'{prefix}/hparams.json') == t['params']
    assert run['selected_epoch'] == max(run['history'], key=lambda h:
        (h['val']['pooled_ap'] + h['val']['pooled_roc'])/2)['epoch']
    coverage = read(f'{prefix}/coverage.json')['splits']
    for split, file in [('test', 'metrics.json'), ('val', 'metrics_val.json')]:
        m = read(f'{prefix}/{file}')
        assert m['split'] == split and m['corpus'] == s['corpus']
        r = m['results']['score_av']
        assert r['n_videos'] == len(coverage[split])
        values = dict(pooled_ap=r['pr_auc'], pooled_roc=r['roc_auc'], within_roc=r['per_video']['macro_auc'])
        for key, value in values.items():
            assert abs(value - run[split][key]) < 1e-10
            assert abs(value - t['user_attrs'][f'{split}_{key}']) < 1e-10
    for filename in ['model.pth', 'run.log', 'config.json', 'scores_test.jsonl', 'scores_val.jsonl']:
        assert (root / prefix / filename).stat().st_size > 0
complete = [t for t in s['trials'] if t['state'] == 'COMPLETE']
if complete:
    assert s['best']['number'] == max(complete, key=lambda t:t['value'])['number']
else:
    assert s['best'] is None  # Fully audited search can legitimately have no eligible trial.
assert s['validation_selected']['number'] == max(s['trials'], key=lambda t:
    (t['user_attrs']['val_pooled_ap']+t['user_attrs']['val_pooled_roc'])/2)['number']
report = dict(study=str(root), n_trials=budget, states=dict(Counter(t['state'] for t in s['trials'])),
              best=s['best'], validation_selected=s['validation_selected'],
              checks='50 epochs, checkpoint selection, configs, evaluator metrics/coverage, output presence')
(root / 'artifact_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
