"""Developmental test error analysis after the frozen two-corpus pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
BASELINES = REPO / "scripts/reproduction_baselines"
if str(BASELINES) not in sys.path:
    sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


ROOT = REPO / "runs/20260831_factorial_witness_crf/pilot_seed234"
ARMS = ("core", "zero_transition", "collapsed")


def load_records(path):
    out = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row.pop("video_id")
            out[video_id] = {
                key: np.asarray(value, dtype=np.float64) for key, value in row.items()
            }
    return out


def safe_spearman(left, right):
    value = spearmanr(left, right).statistic
    return None if not np.isfinite(value) else float(value)


def analyze_corpus(corpus):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {video_id for video_id in gt if labels[video_id] == 1}
    records = {
        arm: load_records(ROOT / corpus / arm / "scores.jsonl") for arm in ARMS
    }
    result = {"active_posterior_metrics": {}, "bit_posterior": {}}
    per_video_auc = {}
    for arm in ARMS:
        active = {
            video_id: row["active_posterior"]
            for video_id, row in records[arm].items()
        }
        result["active_posterior_metrics"][arm] = evaluate_scores(
            active, gt, hate_ids
        )
        per_video_auc[arm] = {
            video_id: float(roc_auc_score(gt[video_id], score))
            for video_id, score in active.items()
            if video_id in hate_ids and len(np.unique(gt[video_id])) == 2
        }
        if arm != "collapsed":
            all_bits = np.concatenate(
                [row["bit_posterior"] for row in records[arm].values()], axis=0
            )
            corr = np.corrcoef(all_bits.T)
            result["bit_posterior"][arm] = {
                "frame_weighted_mean_audio_visual_text": all_bits.mean(0).tolist(),
                "frame_weighted_argmax_fraction_audio_visual_text": (
                    np.bincount(np.argmax(all_bits, axis=1), minlength=3) / len(all_bits)
                ).tolist(),
                "pairwise_correlation": corr.tolist(),
            }

    eligible = sorted(per_video_auc["core"])
    positive_fraction = np.asarray([gt[v].mean() for v in eligible])
    transition_rate = np.asarray([
        np.count_nonzero(np.diff(gt[v])) / max(len(gt[v]) - 1, 1) for v in eligible
    ])
    result["core_minus_controls"] = {}
    for control in ("zero_transition", "collapsed"):
        delta = np.asarray([
            per_video_auc["core"][v] - per_video_auc[control][v] for v in eligible
        ])
        result["core_minus_controls"][control] = {
            "mean_within_auc_delta": float(delta.mean()),
            "median_within_auc_delta": float(np.median(delta)),
            "fraction_videos_core_better": float(np.mean(delta > 0)),
            "spearman_delta_vs_positive_fraction": safe_spearman(delta, positive_fraction),
            "spearman_delta_vs_gt_transition_rate": safe_spearman(delta, transition_rate),
        }

    scales, video_y = [], []
    for video_id, row in records["core"].items():
        active = row["active_posterior"]
        score = row["score_core"]
        valid = active > 1e-12
        scale = float(np.median(score[valid] / active[valid])) if valid.any() else 0.0
        scales.append(scale)
        video_y.append(int(gt[video_id].max() > 0))
    result["core_video_scale"] = {
        "video_roc_auc": float(roc_auc_score(video_y, scales)),
        "mean_positive": float(np.mean([s for s, y in zip(scales, video_y) if y])),
        "mean_negative": float(np.mean([s for s, y in zip(scales, video_y) if not y])),
    }
    result["n_eligible_within_videos"] = len(eligible)
    return result


def main():
    payload = {
        "status": "iterative/developmental test error analysis",
        "test_artifacts": {
            corpus: {
                arm: str(ROOT / corpus / arm / "scores.jsonl") for arm in ARMS
            }
            for corpus in ("hatemm", "hateclipseg")
        },
        "corpora": {
            corpus: analyze_corpus(corpus) for corpus in ("hatemm", "hateclipseg")
        },
    }
    target = ROOT / "test_error_analysis.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
