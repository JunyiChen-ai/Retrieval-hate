"""Describe saved frame predictions; all AP/AUC values come from the evaluator."""
import json
from pathlib import Path

import numpy as np


def describe_run(folder, gt):
    folder = Path(folder)
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
    result = dict(sources=[str(folder / x) for x in ['summary.json', 'metrics.json', 'scores_test.jsonl']],
        selected_epoch=summary['selected_epoch'], hparams=summary['hparams'],
        metrics_from_evaluator={k: metrics[k] for k in ['pr_auc', 'roc_auc', 'video_level']},
        within_from_evaluator=metrics['per_video']['macro_auc'],
        val_first=summary['history'][0]['val'], val_last=summary['history'][-1]['val'],
        train_loss_first=summary['history'][0].get('loss'), train_loss_last=summary['history'][-1].get('loss'),
        mean_frame_scores={k: dict(n=len(v), mean=float(np.mean(v)) if v else None) for k, v in groups.items()},
        n_mixed_videos=len(mixed),
        mean_mixed_video_temporal_std=float(np.mean([v['temporal_std'] for v in mixed])) if mixed else None,
        mean_mixed_positive_background_gap=float(np.mean([v['positive_mean'] - v['background_mean'] for v in mixed])) if mixed else None,
        n_mixed_video_auc_below_half=sum(v['auc_from_evaluator'] < .5 for v in mixed),
        per_video=per_video)
    channel = folder / 'observation_channel.json'
    if channel.exists():
        result['observation_channel'] = json.loads(channel.read_text())
        result['sources'].append(str(channel))
    return result
