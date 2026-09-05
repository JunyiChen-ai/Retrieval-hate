"""Read locked-config module ablations; report evaluator deltas, not new metrics."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
args = p.parse_args()
reference = json.loads((args.reference / 'summary.json').read_text())
arms = ['raw_verdict','ordinary_attention','additive_fusion','full_input_only',
        'four_logits','no_interaction','dempster_fusion','no_block']
report = dict(reference=str(args.reference / 'metrics.json'), full=reference['test'], arms={})
for arm in arms:
    folder = args.root / arm
    s = json.loads((folder / 'summary.json').read_text())
    assert s['ablation'] == arm and s['seed'] == reference['seed']
    assert s['hparams'] == reference['hparams'] and len(s['history']) == 50
    assert s['selected_epoch'] == max(s['history'], key=lambda h:
        (h['val']['pooled_ap'] + h['val']['pooled_roc'])/2)['epoch']
    for split, filename in [('val','metrics_val.json'), ('test','metrics.json')]:
        m = json.loads((folder/filename).read_text())
        assert m['split'] == split and m['corpus'] == reference['corpus']
        r=m['results']['score_av']
        assert r['n_videos'] == reference[split]['n_videos']
        for key, value in [('pooled_ap',r['pr_auc']),('pooled_roc',r['roc_auc']),('within_roc',r['per_video']['macro_auc'])]:
            assert abs(value-s[split][key]) < 1e-10
    assert (folder/'model.pth').stat().st_size > 0
    report['arms'][arm] = dict(metrics=s['test'], source=str(folder/'metrics.json'),
        delta_vs_full={k:s['test'][k]-reference['test'][k] for k in ['pooled_ap','pooled_roc','within_roc']})
(args.root/'artifact_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
