"""Aggregate audited evaluator outputs; no frame metric recomputation."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
seeds = [234, 2025, 3407]
audits = [json.loads((a.root / f'seed{s}' / 'artifact_audit.json').read_text())
          for s in seeds]
keys = ['pooled_ap', 'pooled_roc', 'within_roc']


def read_metrics(path):
    r = json.loads(Path(path).read_text())['results']['score_av']
    return dict(zip(keys, [r['pr_auc'], r['roc_auc'], r['per_video']['macro_auc']]))


full = [read_metrics(x['reference']) for x in audits]
report = dict(seeds=seeds, full_sources=[x['reference'] for x in audits],
              full_mean={k: statistics.mean(x[k] for x in full) for k in keys},
              criterion='Same pooled metric: mean drop >= .01 and every seed drops; this corpus only',
              note='Developmental test-selected configurations; effectiveness is not by itself novelty.',
              arms={})
assert all(set(x['arms']) == set(audits[0]['arms']) for x in audits)
for arm in audits[0]['arms']:
    paths = [x['arms'][arm]['source'] for x in audits]
    metrics = [read_metrics(path) for path in paths]
    drops = {k: [f[k] - m[k] for f, m in zip(full, metrics)] for k in keys}
    mean_drop = {k: statistics.mean(v) for k, v in drops.items()}
    passes = {k: mean_drop[k] >= .01 and all(v > 0 for v in drops[k])
              for k in keys[:2]}
    report['arms'][arm] = dict(sources=paths,
        mean={k: statistics.mean(m[k] for m in metrics) for k in keys},
        std_ddof1={k: statistics.stdev(m[k] for m in metrics) for k in keys},
        full_minus_ablation_by_seed=drops, mean_drop=mean_drop,
        criterion_by_metric=passes, this_corpus_pass=any(passes.values()))
(a.root / 'three_seed_summary.json').write_text(json.dumps(report, indent=2))
for arm, result in report['arms'].items():
    print(arm, 'mean_drop=', result['mean_drop'], 'pass=', result['this_corpus_pass'])
