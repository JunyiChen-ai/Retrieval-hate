"""Validate and summarize existing unified-evaluator artifacts; no reevaluation."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def vector(path):
    m = read(path)
    assert m['split'] == 'test', path
    r = m['results']['score_av']
    return [r['pr_auc'], r['roc_auc'], r['per_video']['macro_auc']]


def stats(rows):
    return {'mean': [statistics.mean(c) for c in zip(*rows)],
            'std_ddof1': [statistics.stdev(c) if len(c) > 1 else 0 for c in zip(*rows)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = {'source': str(args.root), 'metric_order': ['pooled_AP', 'pooled_ROC', 'within_ROC'],
              'note': 'Transcription of evaluator outputs; not a separate evaluator', 'corpora': {}}
    for corpus in ['hatemm', 'hateclipseg']:
        seeds, best_rows, val_rows = [], [], []
        arms = {arm: [] for arm in ['null_token_const', 'masked_no_token', 'full']}
        for seed in [234, 2025, 3407]:
            directory = args.root / corpus / f'seed{seed}'
            summary = read(directory / 'study_summary.json')
            assert summary['n_trials'] == 20 and len(summary['trials']) == 20
            assert {t['number'] for t in summary['trials']} == set(range(20))
            for trial in summary['trials']:
                assert trial['state'] in ['COMPLETE', 'PRUNED'], (directory, trial['number'])
                path = directory / f"trial{trial['number']}" / 'metrics.json'
                values = vector(path)
                attrs = trial['user_attrs']
                expected = [attrs[f'test_{k}'] for k in ['pooled_ap', 'pooled_roc', 'within_roc']]
                assert all(abs(a-b) < 1e-10 for a, b in zip(values, expected)), path
                for file in ['summary.json', 'hparams.json', 'run.log']:
                    assert (path.parent / file).is_file(), path.parent / file
            best = vector(directory / f"trial{summary['best']['number']}" / 'metrics.json')
            val = vector(directory / f"trial{summary['validation_selected']['number']}" / 'metrics.json')
            best_rows.append(best)
            val_rows.append(val)
            arm_values = {}
            for arm in arms:
                path = args.root / 'ablations' / corpus / f'seed{seed}' / arm / 'metrics.json'
                values = vector(path)
                arms[arm].append(values)
                arm_values[arm] = {'metrics': values, 'source': str(path)}
            seeds.append({'seed': seed, 'best_trial': summary['best']['number'], 'best': best,
                          'validation_trial': summary['validation_selected']['number'],
                          'validation_selected': val, 'arms': arm_values})
        result['corpora'][corpus] = {'seeds': seeds, 'best': stats(best_rows),
            'validation_selected': stats(val_rows), 'arms': {a: stats(v) for a, v in arms.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({c: {k:v for k,v in d.items() if k != 'seeds'}
                      for c,d in result['corpora'].items()}, indent=2))


if __name__ == '__main__':
    main()
