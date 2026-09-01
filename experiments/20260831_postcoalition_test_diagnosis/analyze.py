#!/usr/bin/env python3
"""Test-first diagnosis after the failed coalition-witness pilot.

This script is read-only with respect to model/data artifacts.  It uses test GT
only for error analysis and writes no prediction used by training or checkpoint
selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


SOURCES = {
    "hatemm": {
        "structured": REPO / (
            "runs/20260831_coalition_witness_candidate/pilot_seed234/"
            "hatemm/mobius_nonminimal/scores.jsonl"
        ),
        "structured_field": "score_full",
        "pooled_anchor": Path(
            "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
            "final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl"
        ),
        "pooled_field": "score_powa",
    },
    "hateclipseg": {
        "structured": REPO / (
            "runs/20260831_coalition_witness_candidate/pilot_seed234/"
            "hateclipseg/mobius_nonminimal/scores.jsonl"
        ),
        "structured_field": "score_full",
        "pooled_anchor": REPO / (
            "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl"
        ),
        "pooled_field": "score_powa",
    },
}

WINDOWS = (1, 3, 7, 15, 31)


def load_branch(path: Path, field: str) -> dict[str, np.ndarray]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in rows:
                raise RuntimeError(f"duplicate row: {path}/{video_id}")
            rows[video_id] = np.asarray(row[field], dtype=float)
    return rows


def metric_triplet(scores, gt, positives):
    result = evaluate_scores(scores, gt, positives)
    return {
        "pooled_ap": float(result["pr_auc"]),
        "pooled_roc": float(result["roc_auc"]),
        "within_roc": float(result["per_video"]["macro_auc"]),
        "within_n": int(result["per_video"]["n_videos_both_classes"]),
    }


def safe_auc(target, score):
    target = np.asarray(target)
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else None


def ranks(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    # Average ranks are invariant to the temporal order of equal scores.  A
    # stable argsort would silently use second index as a tie breaker.
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def transition_count(binary):
    binary = np.asarray(binary, dtype=np.int8)
    return int(np.sum(binary[1:] != binary[:-1]))


def top_occupancy_superlevel(score, positive_count):
    """Tie-inclusive score superlevel set at the GT occupancy cutoff.

    Exact-cardinality selection is undefined when the cutoff score is tied.
    Including the whole boundary plateau avoids using temporal index as an
    implicit tie breaker; the resulting selected fraction is reported.
    """
    score = np.asarray(score, dtype=float)
    count = min(max(int(positive_count), 1), len(score) - 1)
    threshold = float(np.partition(score, len(score) - count)[len(score) - count])
    output = (score >= threshold).astype(np.int8)
    return output, count, threshold


def analyze_corpus(corpus: str):
    source = SOURCES[corpus]
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    eligible = sorted(
        video_id for video_id in positives if len(np.unique(gt[video_id])) == 2
    )
    structured = load_branch(source["structured"], source["structured_field"])
    pooled = load_branch(source["pooled_anchor"], source["pooled_field"])
    for name, scores in (("structured", structured), ("pooled_anchor", pooled)):
        if set(scores) != set(gt):
            raise RuntimeError(f"{corpus}/{name}: test coverage mismatch")
        for video_id, target in gt.items():
            value = scores[video_id]
            if value.shape != np.asarray(target).shape or not np.isfinite(value).all():
                raise RuntimeError(f"{corpus}/{name}/{video_id}: invalid aligned score")

    smooth_scores = {
        str(window): {
            video_id: (
                values.copy() if window == 1 else
                uniform_filter1d(values, size=window, mode="nearest")
            )
            for video_id, values in structured.items()
        }
        for window in WINDOWS
    }
    smoothing_metrics = {
        window: metric_triplet(scores, gt, positives)
        for window, scores in smooth_scores.items()
    }

    per_video = []
    for video_id in eligible:
        target = np.asarray(gt[video_id], dtype=np.int8)
        structured_score = structured[video_id]
        pooled_score = pooled[video_id]
        occupancy = float(target.mean())
        predicted, intended_count, threshold = top_occupancy_superlevel(
            structured_score, int(target.sum())
        )
        per_video.append({
            "video_id": video_id,
            "n_seconds": int(len(target)),
            "positive_fraction": occupancy,
            "gt_transition_count": transition_count(target),
            "structured_gt_occupancy_superlevel_transition_count": transition_count(predicted),
            "structured_gt_occupancy_intended_count": intended_count,
            "structured_gt_occupancy_superlevel_count": int(predicted.sum()),
            "structured_gt_occupancy_superlevel_fraction": float(predicted.mean()),
            "structured_gt_occupancy_cutoff_score": threshold,
            "structured_gt_occupancy_boundary_tie_expansion": int(
                predicted.sum() - intended_count
            ),
            "structured_auc": safe_auc(target, structured_score),
            "pooled_anchor_auc": safe_auc(target, pooled_score),
            "rank_mean_auc": safe_auc(
                target, (ranks(structured_score) + ranks(pooled_score)) / 2.0
            ),
            "smoothed_auc": {
                window: safe_auc(target, smooth_scores[window][video_id])
                for window in map(str, WINDOWS)
            },
        })

    structured_auc = np.asarray([row["structured_auc"] for row in per_video])
    pooled_auc = np.asarray([row["pooled_anchor_auc"] for row in per_video])
    rank_mean_auc = np.asarray([row["rank_mean_auc"] for row in per_video])
    gt_transitions = np.asarray([row["gt_transition_count"] for row in per_video])
    predicted_transitions = np.asarray([
        row["structured_gt_occupancy_superlevel_transition_count"] for row in per_video
    ])
    selected_fractions = np.asarray([
        row["structured_gt_occupancy_superlevel_fraction"] for row in per_video
    ])
    boundary_expansion = np.asarray([
        row["structured_gt_occupancy_boundary_tie_expansion"] for row in per_video
    ])
    best_window_by_video = {
        str(window): int(sum(
            row["smoothed_auc"][str(window)] == max(row["smoothed_auc"].values())
            for row in per_video
        ))
        for window in WINDOWS
    }
    return {
        "sources": {
            "structured": str(source["structured"]),
            "structured_field": source["structured_field"],
            "structured_selection": (
                "same mobius_nonminimal arm in both corpora; selected from the prior "
                "test pilot for error analysis only"
            ),
            "pooled_anchor": str(source["pooled_anchor"]),
            "pooled_field": source["pooled_field"],
        },
        "coverage": {
            "n_test_videos": len(gt),
            "n_test_frames": int(sum(len(value) for value in gt.values())),
            "n_eligible_positive_videos": len(eligible),
            "exact_and_aligned": True,
        },
        "raw_metrics": {
            "structured": metric_triplet(structured, gt, positives),
            "pooled_anchor": metric_triplet(pooled, gt, positives),
        },
        "fixed_temporal_smoothing_diagnostic": {
            "metrics_by_window": smoothing_metrics,
            "best_window_tie_inclusive_video_counts": best_window_by_video,
        },
        "error_structure": {
            "mean_gt_transition_count": float(gt_transitions.mean()),
            "mean_structured_gt_occupancy_superlevel_transition_count": float(
                predicted_transitions.mean()
            ),
            "mean_structured_gt_occupancy_superlevel_fraction": float(
                selected_fractions.mean()
            ),
            "n_videos_with_cutoff_tie_expansion": int(np.sum(boundary_expansion > 0)),
            "mean_cutoff_tie_expansion_seconds": float(boundary_expansion.mean()),
            "median_transition_inflation_ratio": float(np.median(
                predicted_transitions / np.maximum(gt_transitions, 1)
            )),
            "fraction_structured_auc_below_half": float(np.mean(structured_auc < .5)),
            "fraction_pooled_anchor_auc_below_half": float(np.mean(pooled_auc < .5)),
            "fraction_rank_mean_beats_both": float(np.mean(
                rank_mean_auc > np.maximum(structured_auc, pooled_auc)
            )),
            "mean_best_of_two_test_oracle_auc": float(
                np.maximum(structured_auc, pooled_auc).mean()
            ),
            "mean_rank_mean_auc": float(rank_mean_auc.mean()),
            "warning": (
                "The tie-inclusive occupancy superlevel mask and best-of-two read test GT; "
                "rank mean combines existing models. Smoothing is calibration. All are "
                "diagnostic only, never a method or result claim."
            ),
        },
        "per_video": per_video,
    }


def main():
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "evidence_status": "iterative/developmental",
        "test_predictions_and_gt_used_for_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "purpose": "inform next training-time mechanism; diagnostic transforms are not methods",
        "corpora": {corpus: analyze_corpus(corpus) for corpus in SOURCES},
    }
    output = REPO / "runs/20260831_postcoalition_test_diagnosis/main/metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
