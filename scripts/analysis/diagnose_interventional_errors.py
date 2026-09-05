"""Developmental error analysis of saved scores; does not recompute AP/AUC."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reproduction_baselines'))
from hate_common import data as hdata

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
report = dict(purpose='Developmental test error analysis; no training or checkpoint selection', corpora={})
for corpus, seeds in [('hatemm', [234]), ('hateclipseg', [234, 2025, 3407])]:
    gt = hdata.gt_arrays(corpus, 'test')
    labels = hdata.load_labels(corpus)
    split_counts = {}
    for split in ['train', 'val', 'test']:
        ids = hdata.load_split(corpus, split)
        split_counts[split] = dict(n=len(ids), positive=sum(labels[v] for v in ids))
    result = dict(split_video_counts=split_counts, gt_source=str(Path(hdata.GT_ROOT) / f'{corpus}_test.npz'), seeds={})
    for seed in seeds:
        study = a.root / corpus / f'seed{seed}'
        number = json.loads((study / 'study_summary.json').read_text())['best']['number']
        full = study / f'trial{number}'
        abl = a.root / 'ablations' / corpus / f'seed{seed}'
        audit = json.loads((abl / 'artifact_audit.json').read_text())
        arms = {'full': full, **{arm: Path(x['source']).parent for arm, x in audit['arms'].items()}}
        seed_rows = {}
        for arm, folder in arms.items():
            summary = json.loads((folder / 'summary.json').read_text())
            metrics = json.loads((folder / 'metrics.json').read_text())['results']['score_av']
            rows = [json.loads(line) for line in (folder / 'scores_test.jsonl').read_text().splitlines()]
            assert len(rows) == len({r['video_id'] for r in rows}) == len(gt)
            assert {r['video_id'] for r in rows} == set(gt)
            groups = dict(positive_frames=[], background_in_positive_videos=[], negative_videos=[])
            per_video = []
            for row in rows:
                vid = row['video_id']
                score = np.asarray(row['score_av'], dtype=float)
                target = np.asarray(gt[vid], dtype=bool)
                assert len(score) == len(target) == row['n_frames'] and np.isfinite(score).all()
                groups['positive_frames'].extend(score[target].tolist())
                groups['background_in_positive_videos' if target.any() else 'negative_videos'].extend(score[~target].tolist())
                per_video.append(dict(video=vid, n_frames=len(score), positive_fraction=float(target.mean()),
                    score_mean=float(score.mean()), temporal_std=float(score.std()),
                    positive_mean=float(score[target].mean()) if target.any() else None,
                    background_mean=float(score[~target].mean()) if (~target).any() else None,
                    auc_from_evaluator=metrics['per_video']['per_video_auc'].get(vid)))
            mixed = [v for v in per_video if 0 < v['positive_fraction'] < 1]
            seed_rows[arm] = dict(sources=[str(folder / x) for x in ['summary.json','metrics.json','scores_test.jsonl']],
                selected_epoch=summary['selected_epoch'], hparams=summary['hparams'],
                val_first=summary['history'][0]['val'], val_last=summary['history'][-1]['val'],
                train_loss_first=summary['history'][0]['loss'], train_loss_last=summary['history'][-1]['loss'],
                mean_frame_scores={k: dict(n=len(v), mean=float(np.mean(v)) if v else None) for k,v in groups.items()},
                mean_mixed_video_temporal_std=float(np.mean([v['temporal_std'] for v in mixed])),
                n_mixed_video_auc_below_half=sum(v['auc_from_evaluator'] < .5 for v in mixed),
                per_video=per_video)
        result['seeds'][seed] = seed_rows
    report['corpora'][corpus] = result
out = a.root / 'error_analysis'
out.mkdir(exist_ok=True)
(out / 'saved_prediction_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
for corpus, result in report['corpora'].items():
    print(corpus, result['split_video_counts'])
    for seed, arms in result['seeds'].items():
        for arm in ['full','raw_verdict','ordinary_attention','additive_fusion','no_block']:
            r = arms[arm]
            print(seed, arm, 'epoch', r['selected_epoch'], 'scores',
                  {k: round(v['mean'], 4) if v['mean'] is not None else None for k,v in r['mean_frame_scores'].items()},
                  'temporal_std', round(r['mean_mixed_video_temporal_std'], 4),
                  'inverted_videos', r['n_mixed_video_auc_below_half'])
