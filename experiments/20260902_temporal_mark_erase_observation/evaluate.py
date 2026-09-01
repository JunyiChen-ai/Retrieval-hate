#!/usr/bin/env python3
"""Evaluate complete mark/erase predictions using the shared evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from protocol import (  # noqa: E402
    CORPORA,
    EXPECTED_WITHIN_COUNTS,
    SHIFT_DENOMINATOR,
    SHIFT_FRACTIONS,
)
from run_observation import cohort, load_existing  # noqa: E402


RUN_ROOT = REPO / "runs/20260902_temporal_mark_erase_observation/formal"


def densify(row, arm):
    total = np.zeros(row["length"], dtype=np.float64)
    count = np.zeros(row["length"], dtype=np.float64)
    for window in row["windows"]:
        start, end = window["span"]
        if arm == "contrast":
            value = window["marked"]["score"] - window["erased"]["score"]
        else:
            value = window[arm]["score"]
        total[start:end] += float(value)
        count[start:end] += 1.0
    if np.any(count == 0):
        raise RuntimeError("uncovered second")
    return total / count


def metric(scores, gt, ids):
    result = evaluate_scores(scores, gt, set(ids))
    if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
        raise RuntimeError("shared evaluator coverage mismatch")
    return {
        "pooled_ap_positive_cohort_diagnostic": result["pr_auc"],
        "pooled_roc_positive_cohort_diagnostic": result["roc_auc"],
        "within_roc": result["per_video"]["macro_auc"],
        "within_n": result["per_video"]["n_videos_both_classes"],
    }


def corpus_report(corpus):
    ids = cohort(corpus)
    lengths = {
        video_id: int(np.load(hdata.feature_path(corpus, video_id), mmap_mode="r").shape[0])
        for video_id in ids
    }
    path = RUN_ROOT / corpus / "predictions.jsonl"
    rows = load_existing(path, corpus, ids, lengths)
    if len(rows) != len(ids):
        raise RuntimeError(f"incomplete producer for {corpus}")
    complete_gt = hdata.gt_arrays(corpus, "test")
    gt = {video_id: complete_gt[video_id] for video_id in ids}
    marked = {row["video_id"]: densify(row, "marked") for row in rows}
    erased = {row["video_id"]: densify(row, "erased") for row in rows}
    contrast = {row["video_id"]: densify(row, "contrast") for row in rows}
    for video_id in ids:
        if not (len(marked[video_id]) == len(erased[video_id]) == len(contrast[video_id]) == len(gt[video_id])):
            raise RuntimeError("score/GT alignment mismatch")
    marked_metrics = metric(marked, gt, ids)
    erased_metrics = metric(erased, gt, ids)
    contrast_metrics = metric(contrast, gt, ids)
    if contrast_metrics["within_n"] != EXPECTED_WITHIN_COUNTS[corpus]:
        raise RuntimeError("within cohort count changed")
    shifted = []
    for numerator in SHIFT_FRACTIONS:
        shifted_scores = {}
        for video_id in ids:
            size = len(contrast[video_id])
            offset = max(1, int(round(size * numerator / SHIFT_DENOMINATOR))) % size
            shifted_scores[video_id] = np.roll(contrast[video_id], offset)
        shifted.append(metric(shifted_scores, gt, ids)["within_roc"])
    failures = [
        arm["status"] for row in rows for window in row["windows"]
        for arm in (window["marked"], window["erased"])
    ]
    shift_mean = float(np.mean(shifted))
    contrast_within = contrast_metrics["within_roc"]
    marked_within = marked_metrics["within_roc"]
    return {
        "split": "test_positive",
        "prediction_artifact": str(path.resolve()),
        "n_videos": len(ids),
        "n_calls": len(failures),
        "n_failed_calls": sum(status != "ok" for status in failures),
        "marked": marked_metrics,
        "erased": erased_metrics,
        "contrast": contrast_metrics,
        "fixed_relative_shift_within": shifted,
        "shift_mean_within": shift_mean,
        "contrast_minus_shift_mean": contrast_within - shift_mean,
        "contrast_minus_marked": contrast_within - marked_within,
        "gate_minimum_direction": contrast_within >= 0.52,
        "gate_time_alignment": contrast_within - shift_mean >= 0.02,
    }


def main():
    corpora = {corpus: corpus_report(corpus) for corpus in CORPORA}
    nonnegative_both = all(row["contrast_minus_marked"] >= 0.0 for row in corpora.values())
    load_bearing_one = any(row["contrast_minus_marked"] >= 0.01 for row in corpora.values())
    passed = (
        all(row["gate_minimum_direction"] and row["gate_time_alignment"] for row in corpora.values())
        and nonnegative_both
        and load_bearing_one
    )
    payload = {
        "date": "2026-09-02",
        "split": "test",
        "test_predictions_and_gt_used_for_method_development": True,
        "test_labels_used_for_training_or_checkpoint_selection": False,
        "future_evidence_status": "iterative/developmental",
        "corpora": corpora,
        "gate_contrast_nonnegative_vs_marked_both": nonnegative_both,
        "gate_contrast_plus_point01_one_corpus": load_bearing_one,
        "observation_pass_both": passed,
        "verdict": "ALLOW_FINAL_NOVELTY_BRIEF" if passed else "CLOSE_MARK_ERASE_SOURCE",
    }
    target = RUN_ROOT / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
