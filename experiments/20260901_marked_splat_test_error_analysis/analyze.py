#!/usr/bin/env python3
"""Focused post-test analysis for the formal marked-splat run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


FORMAL = REPO / "runs/20260901_marked_temporal_splat_mil/pilot_seed234"
ANCHOR = REPO / "runs/20260831_witness_conditional_dgm/pilot_seed234_matched"
OUTPUT = REPO / "runs/20260901_marked_splat_test_error_analysis/main"
CORPORA = ("hatemm", "hateclipseg")
ARMS = ("anchor", "point", "splat")


def load_branch(path: Path, field: str) -> dict[str, np.ndarray]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row["video_id"]
            if video_id in rows:
                raise RuntimeError(f"duplicate prediction row: {path}/{video_id}")
            rows[video_id] = np.asarray(row[field], dtype=float)
    return rows


def load_anchor_modalities(corpus: str) -> dict[str, dict[str, np.ndarray]]:
    path = ANCHOR / corpus / "anchor/scores.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row["video_id"]
            rows[video_id] = {
                name: np.asarray(row[f"score_{name}"], dtype=float)
                for name in ("visual", "audio", "text")
            }
    return rows


def safe_auc(target, score):
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    if len(target) == 0 or len(np.unique(target)) != 2:
        return None
    return float(roc_auc_score(target, score))


def finite_mean(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or len(np.unique(x[keep])) < 2 or len(np.unique(y[keep])) < 2:
        return {"rho": None, "n": int(keep.sum())}
    out = spearmanr(x[keep], y[keep])
    return {"rho": float(out.statistic), "n": int(keep.sum())}


def metric_triplet(scores, gt, positives):
    result = evaluate_scores(scores, gt, positives)
    return {
        "pooled_ap": float(result["pr_auc"]),
        "pooled_roc": float(result["roc_auc"]),
        "within_roc": float(result["per_video"]["macro_auc"]),
        "within_n": int(result["per_video"]["n_videos_both_classes"]),
    }


def positive_runs(target):
    target = np.asarray(target, dtype=np.int8)
    padded = np.r_[0, target, 0]
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return (ends - starts).astype(int)


def interp_profile(values, size=101):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.repeat(values, size)
    return np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(values)), values)


def position_control(scores):
    """Leave-one-video-out common normalized-position profile."""
    ids = sorted(scores)
    profiles = {video_id: interp_profile(scores[video_id]) for video_id in ids}
    total = np.sum([profiles[v] for v in ids], axis=0)
    output = {}
    for video_id in ids:
        loo = (total - profiles[video_id]) / max(len(ids) - 1, 1)
        length = len(scores[video_id])
        output[video_id] = np.interp(
            np.linspace(0, 1, length), np.linspace(0, 1, len(loo)), loo
        )
    return output


def subtract_position_profile(scores, position):
    return {
        video_id: np.asarray(scores[video_id]) - np.asarray(position[video_id])
        for video_id in scores
    }


def time_shift_controls(scores):
    fractions = (0.2, 0.4, 0.6, 0.8)
    return {
        str(fraction): {
            video_id: np.roll(values, max(1, int(round(len(values) * fraction))))
            for video_id, values in scores.items()
        }
        for fraction in fractions
    }


def pairwise_separation(scores, gt, labels):
    positive_frames = []
    benign_in_positive = []
    frames_in_negative = []
    video_label = []
    video_mean = []
    video_max = []
    video_top10 = []
    for video_id in sorted(gt):
        target = np.asarray(gt[video_id], dtype=np.int8)
        value = scores[video_id]
        label = int(labels[video_id])
        video_label.append(label)
        video_mean.append(float(value.mean()))
        video_max.append(float(value.max()))
        count = max(1, int(np.ceil(len(value) * .1)))
        video_top10.append(float(np.partition(value, len(value) - count)[-count:].mean()))
        if label == 1:
            positive_frames.extend(value[target == 1])
            benign_in_positive.extend(value[target == 0])
        else:
            frames_in_negative.extend(value)

    def compare(pos, neg):
        return safe_auc(
            np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)],
            np.r_[np.asarray(pos), np.asarray(neg)],
        )

    return {
        "positive_vs_benign_frames_inside_positive_videos_roc": compare(
            positive_frames, benign_in_positive
        ),
        "positive_frames_vs_all_frames_in_negative_videos_roc": compare(
            positive_frames, frames_in_negative
        ),
        "video_label_roc_from_mean_frame_score": safe_auc(video_label, video_mean),
        "video_label_roc_from_max_frame_score": safe_auc(video_label, video_max),
        "video_label_roc_from_top10pct_mean": safe_auc(video_label, video_top10),
        "mean_score_by_video_label": {
            "positive": finite_mean([v for v, y in zip(video_mean, video_label) if y == 1]),
            "negative": finite_mean([v for v, y in zip(video_mean, video_label) if y == 0]),
        },
    }


def spreading(scores, gt, labels):
    rows = []
    for video_id, values in scores.items():
        target = np.asarray(gt[video_id], dtype=np.int8)
        rows.append({
            "label": int(labels[video_id]),
            "std": float(values.std()),
            "range": float(values.max() - values.min()),
            "above_090": float(np.mean(values >= .9)),
            "below_010": float(np.mean(values <= .1)),
            "positive_mean": float(values[target == 1].mean()) if target.any() else None,
            "benign_mean": float(values[target == 0].mean()) if np.any(target == 0) else None,
        })
    return {
        "mean_temporal_std": finite_mean([r["std"] for r in rows]),
        "mean_temporal_range": finite_mean([r["range"] for r in rows]),
        "mean_fraction_score_ge_090": finite_mean([r["above_090"] for r in rows]),
        "mean_fraction_score_le_010": finite_mean([r["below_010"] for r in rows]),
        "positive_frame_mean": finite_mean([r["positive_mean"] for r in rows]),
        "benign_frame_mean_inside_positive_videos": finite_mean([
            r["benign_mean"] for r in rows if r["label"] == 1
        ]),
    }


def strata_summary(per_video, key):
    groups = {}
    for row in per_video:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "n": len(rows),
            "anchor_auc": finite_mean([r["anchor_auc"] for r in rows]),
            "point_auc": finite_mean([r["point_auc"] for r in rows]),
            "splat_auc": finite_mean([r["splat_auc"] for r in rows]),
            "splat_minus_anchor": finite_mean([
                r["splat_auc"] - r["anchor_auc"] for r in rows
            ]),
            "splat_minus_point": finite_mean([
                r["splat_auc"] - r["point_auc"] for r in rows
            ]),
        }
        for name, rows in groups.items()
    }


def analyze_corpus(corpus):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    scores = {
        "anchor": load_branch(ANCHOR / corpus / "anchor/scores.jsonl", "score_fused"),
        "point": load_branch(FORMAL / corpus / "point/scores.jsonl", "score_final"),
        "splat": load_branch(FORMAL / corpus / "splat/scores.jsonl", "score_final"),
    }
    modalities = load_anchor_modalities(corpus)
    for arm in ARMS:
        if set(scores[arm]) != set(gt):
            raise RuntimeError(f"{corpus}/{arm}: coverage mismatch")
        for video_id, target in gt.items():
            value = scores[arm][video_id]
            if value.shape != np.asarray(target).shape or not np.isfinite(value).all():
                raise RuntimeError(f"{corpus}/{arm}/{video_id}: invalid score grid")

    eligible = sorted(
        video_id for video_id in positives if len(np.unique(gt[video_id])) == 2
    )
    per_video = []
    for video_id in eligible:
        target = np.asarray(gt[video_id], dtype=np.int8)
        runs = positive_runs(target)
        available = {
            name: float(modalities[video_id][name].std()) > 1e-6
            for name in modalities[video_id]
        }
        occupancy = float(target.mean())
        if occupancy <= 1 / 3:
            occupancy_bin = "le_1_3"
        elif occupancy <= 2 / 3:
            occupancy_bin = "1_3_to_2_3"
        else:
            occupancy_bin = "gt_2_3"
        mean_run = float(runs.mean()) if len(runs) else 0.0
        if mean_run <= 5:
            span_bin = "le_5s"
        elif mean_run <= 15:
            span_bin = "6_to_15s"
        else:
            span_bin = "gt_15s"
        row = {
            "video_id": video_id,
            "positive_fraction": occupancy,
            "occupancy_bin": occupancy_bin,
            "mean_positive_run_seconds": mean_run,
            "longest_positive_run_seconds": int(runs.max()) if len(runs) else 0,
            "span_bin": span_bin,
            "variable_carrier_count": sum(available.values()),
            "variable_carriers": [name for name, value in available.items() if value],
        }
        for arm in ARMS:
            row[f"{arm}_auc"] = safe_auc(target, scores[arm][video_id])
        per_video.append(row)

    shifts = time_shift_controls(scores["splat"])
    shift_metrics = {
        fraction: metric_triplet(value, gt, positives)
        for fraction, value in shifts.items()
    }
    positions = {arm: position_control(scores[arm]) for arm in ARMS}
    position_residuals = {
        arm: subtract_position_profile(scores[arm], positions[arm]) for arm in ARMS
    }
    raw_metrics = {arm: metric_triplet(scores[arm], gt, positives) for arm in ARMS}
    splat_delta = np.asarray([
        row["splat_auc"] - row["point_auc"] for row in per_video
    ])
    return {
        "coverage": {
            "n_test_videos": len(gt),
            "n_test_frames": int(sum(len(v) for v in gt.values())),
            "n_eligible_positive_videos": len(eligible),
            "exact_aligned_finite": True,
        },
        "raw_metrics": raw_metrics,
        "pooled_separation_decomposition": {
            arm: pairwise_separation(scores[arm], gt, labels) for arm in ARMS
        },
        "score_spreading": {arm: spreading(scores[arm], gt, labels) for arm in ARMS},
        "duration_gain_strata": {
            "by_positive_fraction": strata_summary(per_video, "occupancy_bin"),
            "by_mean_positive_run": strata_summary(per_video, "span_bin"),
            "by_variable_carrier_count": strata_summary(per_video, "variable_carrier_count"),
            "gain_vs_positive_fraction_spearman": safe_spearman(
                [r["positive_fraction"] for r in per_video], splat_delta
            ),
            "gain_vs_mean_positive_run_spearman": safe_spearman(
                [r["mean_positive_run_seconds"] for r in per_video], splat_delta
            ),
        },
        "matched_controls": {
            "time_circular_shift_metrics": shift_metrics,
            "mean_time_shift_within": finite_mean([
                value["within_roc"] for value in shift_metrics.values()
            ]),
            "raw_splat_minus_mean_shift_within": raw_metrics["splat"]["within_roc"] - finite_mean([
                value["within_roc"] for value in shift_metrics.values()
            ]),
            "leave_one_video_out_position_only": {
                arm: metric_triplet(positions[arm], gt, positives) for arm in ARMS
            },
            "score_minus_leave_one_video_out_position_profile": {
                arm: metric_triplet(position_residuals[arm], gt, positives)
                for arm in ARMS
            },
            "raw_minus_position_only_within": {
                arm: raw_metrics[arm]["within_roc"]
                - metric_triplet(positions[arm], gt, positives)["within_roc"]
                for arm in ARMS
            },
            "carrier_strata_are_diagnostic_not_gates": True,
        },
        "per_video": per_video,
    }


def main():
    payload = {
        "date": "2026-09-01",
        "split": "test",
        "evidence_status": "iterative/developmental post-formal-method analysis",
        "is_premise": False,
        "trains_or_selects_checkpoint": False,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "sources": {
            "formal": str(FORMAL),
            "anchor": str(ANCHOR),
        },
        "corpora": {corpus: analyze_corpus(corpus) for corpus in CORPORA},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "config.json").write_text(json.dumps({
        key: value for key, value in payload.items() if key != "corpora"
    }, indent=2) + "\n")
    target = OUTPUT / "metrics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
