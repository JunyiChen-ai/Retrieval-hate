#!/usr/bin/env python
"""Focused post-test analysis of the frozen active-speaker formal run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "hate_common"))
from hate_common import data as hdata  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402

FORMAL = ROOT / "runs/20260901_active_speaker_bound_utterance_mil/formal_seed234"
CACHE = ROOT / "data/active_speaker_bound"
OUTPUT = ROOT / "runs/20260901_active_speaker_test_error_analysis/main/metrics.json"


def load_arm(corpus: str, arm: str):
    path = FORMAL / "test" / corpus / arm / "scores.jsonl"
    records = hdata.load_scores_jsonl(path)
    return {video_id: row["score_method"] for video_id, row in records.items()}, path


def binary_metrics(y, score):
    y = np.asarray(y, dtype=np.uint8)
    score = np.asarray(score, dtype=float)
    return {
        "n_seconds": int(y.size),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()) if y.size else None,
        "pooled_roc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None,
        "pooled_ap": float(average_precision_score(y, score)) if y.size else None,
    }


def corpus_analysis(corpus: str):
    arms = {}
    inputs = {}
    for arm in ("anchor", "permuted", "core"):
        arms[arm], path = load_arm(corpus, arm)
        inputs[arm] = str(path.relative_to(ROOT))
    gt = hdata.gt_arrays(corpus, "test")
    if not (set(gt) == set(arms["anchor"]) == set(arms["permuted"]) == set(arms["core"])):
        raise RuntimeError(f"{corpus}: prediction/gold cohort mismatch")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels[v] == 1}

    eligible_by_video = {}
    affected_by_video = {}
    for video_id in sorted(gt):
        cache_path = CACHE / corpus / f"{video_id}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            eligible = cache["eligible_multiface"].astype(bool)
            assigned = cache["assigned_track"]
            permuted = cache["permuted_track"]
            core_face = cache["core_face"]
            permuted_face = cache["permuted_face"]
        n = len(gt[video_id])
        lengths = {n, len(eligible), *(len(arms[a][video_id]) for a in arms)}
        if len(lengths) != 1:
            raise RuntimeError(f"{corpus}/{video_id}: time-grid mismatch {sorted(lengths)}")
        expected = eligible & (assigned != permuted)
        feature_diff = np.linalg.norm(core_face - permuted_face, axis=1) > 1e-7
        if not np.array_equal(expected, feature_diff):
            raise RuntimeError(f"{corpus}/{video_id}: eligible/feature control mismatch")
        eligible_by_video[video_id] = expected
        affected_by_video[video_id] = np.abs(
            arms["core"][video_id] - arms["permuted"][video_id]) > 1e-12

    eligible_videos = {v for v, m in eligible_by_video.items() if m.any()}
    affected_videos = {v for v, m in affected_by_video.items() if m.any()}
    total_seconds = sum(len(y) for y in gt.values())
    eligible_seconds = sum(int(m.sum()) for m in eligible_by_video.values())
    affected_seconds = sum(int(m.sum()) for m in affected_by_video.values())

    official = {}
    for arm in arms:
        res = evaluate_scores(arms[arm], gt, hate_ids)
        official[arm] = {
            "pooled_ap": float(res["pr_auc"]),
            "pooled_roc": float(res["roc_auc"]),
            "within_roc": float(res["per_video"]["macro_auc"]),
            "within_n": int(res["per_video"]["n_videos_both_classes"]),
        }

    subgroup = {}
    for group_name, masks in (
        ("eligible_seconds", eligible_by_video),
        ("ineligible_seconds", {v: ~m for v, m in eligible_by_video.items()}),
    ):
        y = np.concatenate([gt[v][masks[v]] for v in sorted(gt)])
        subgroup[group_name] = {}
        for arm in arms:
            score = np.concatenate([arms[arm][v][masks[v]] for v in sorted(gt)])
            subgroup[group_name][arm] = binary_metrics(y, score)

    within_rows = []
    for video_id in sorted(eligible_videos):
        y = np.asarray(gt[video_id])
        if np.unique(y).size != 2:
            continue
        row = {"video_id": video_id,
               "eligible_fraction": float(eligible_by_video[video_id].mean())}
        for arm in arms:
            row[arm] = float(roc_auc_score(y, arms[arm][video_id]))
        row["core_minus_permuted"] = row["core"] - row["permuted"]
        within_rows.append(row)
    deltas = np.asarray([r["core_minus_permuted"] for r in within_rows], dtype=float)
    within_eligible_videos = {
        "n_videos_both_classes": len(within_rows),
        "core_macro_roc": float(np.mean([r["core"] for r in within_rows])) if within_rows else None,
        "permuted_macro_roc": float(np.mean([r["permuted"] for r in within_rows])) if within_rows else None,
        "core_minus_permuted": float(deltas.mean()) if deltas.size else None,
        "n_improved": int((deltas > 0).sum()),
        "n_tied": int((deltas == 0).sum()),
        "n_worse": int((deltas < 0).sum()),
        "delta_median": float(np.median(deltas)) if deltas.size else None,
        "delta_min": float(deltas.min()) if deltas.size else None,
        "delta_max": float(deltas.max()) if deltas.size else None,
    }

    score_effect = {}
    for group_name, masks in (("eligible", eligible_by_video),
                              ("ineligible", {v: ~m for v, m in eligible_by_video.items()})):
        delta = np.concatenate([
            np.abs(arms["core"][v] - arms["permuted"][v])[masks[v]]
            for v in sorted(gt)
        ])
        score_effect[group_name] = {
            "n_seconds": int(delta.size),
            "n_nonzero": int((delta > 1e-12).sum()),
            "mean_absolute_score_change": float(delta.mean()) if delta.size else None,
            "median_absolute_score_change": float(np.median(delta)) if delta.size else None,
            "p95_absolute_score_change": float(np.percentile(delta, 95)) if delta.size else None,
        }

    return {
        "inputs": inputs,
        "cohort": {
            "n_videos": len(gt), "n_seconds": total_seconds,
            "n_eligible_multiface_videos": len(eligible_videos),
            "eligible_video_fraction": len(eligible_videos) / len(gt),
            "n_eligible_multiface_seconds": eligible_seconds,
            "eligible_second_fraction": eligible_seconds / total_seconds,
            "n_output_affected_videos": len(affected_videos),
            "n_output_affected_seconds": affected_seconds,
        },
        "official_full_test": official,
        "subgroup_pooled": subgroup,
        "within_videos_with_eligible_seconds": within_eligible_videos,
        "score_effect": score_effect,
        "per_video_within_rows": within_rows,
    }


def main():
    payload = {
        "analysis": "active-speaker focused post-test error analysis",
        "date": "2026-09-01",
        "test_predictions_and_ground_truth_used": True,
        "training_or_checkpoint_selection_affected": False,
        "corpora": {c: corpus_analysis(c) for c in ("hatemm", "hateclipseg")},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
