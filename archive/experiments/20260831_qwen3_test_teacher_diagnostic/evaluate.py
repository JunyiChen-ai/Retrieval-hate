#!/usr/bin/env python3
"""Evaluate both canonical Qwen3 test artifacts with the shared evaluator."""

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
    CODE_VERSION_DESCRIPTION,
    CORPORA,
    EXPECTED_WITHIN_COUNTS,
    WITHIN_SOTA,
    expected_config,
    positive_test_cohort,
    validate_prediction_row,
)


RUN_ROOT = REPO / "runs/20260831_qwen3_test_teacher_diagnostic/formal"


def expected_ids(corpus):
    labels = hdata.load_labels(corpus)
    return positive_test_cohort(
        corpus, hdata.load_split(corpus, "test"), labels
    )


def read_rows(corpus):
    path = RUN_ROOT / corpus / "predictions.jsonl"
    expected = expected_ids(corpus)
    config_path = path.parent / "config.json"
    version_path = path.parent / "code_version.txt"
    if json.loads(config_path.read_text()) != expected_config(corpus, path):
        raise RuntimeError(f"formal producer config mismatch for {corpus}")
    if version_path.read_text() != CODE_VERSION_DESCRIPTION + "\n":
        raise RuntimeError(f"formal code version description mismatch for {corpus}")

    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank prediction row for {corpus}:{line_number}")
        if len(rows) >= len(expected):
            raise RuntimeError(f"extra prediction row for {corpus}:{line_number}")
        row = json.loads(line)
        video_id = expected[len(rows)]
        feature = np.load(hdata.feature_path(corpus, video_id), mmap_mode="r")
        if len(feature.shape) != 2 or feature.shape[0] <= 0:
            raise RuntimeError(f"invalid current 1 fps feature for {corpus}/{video_id}")
        validate_prediction_row(
            row,
            corpus,
            expected_video_id=video_id,
            expected_length=int(feature.shape[0]),
        )
        rows.append(row)
    ids = [row["video_id"] for row in rows]
    if ids != expected or len(ids) != len(set(ids)):
        raise RuntimeError(f"non-exact prediction coverage for {corpus}")
    return rows, path


def densify(row):
    length = row["length"]
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for window in row["windows"]:
        start, end = window["span"]
        if not (0 <= start < end <= length):
            raise RuntimeError("invalid window span")
        score = window["parsed_score"] / 10.0
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("invalid window score")
        total[start:end] += score
        count[start:end] += 1
    if np.any(count == 0):
        raise RuntimeError("window rasterization left uncovered seconds")
    return total / count


def corpus_report(corpus):
    rows, path = read_rows(corpus)
    complete_gt = hdata.gt_arrays(corpus, "test")
    ids = [row["video_id"] for row in rows]
    if any(video_id not in complete_gt for video_id in ids):
        raise RuntimeError(f"missing GT for canonical {corpus} cohort")
    scores = {row["video_id"]: densify(row) for row in rows}
    gt = {video_id: complete_gt[video_id] for video_id in ids}
    if any(len(scores[video_id]) != len(gt[video_id]) for video_id in ids):
        raise RuntimeError(f"score/GT alignment failure for {corpus}")
    shared = evaluate_scores(scores, gt, set(ids))
    if shared["n_videos_missing_from_scores"] or shared["n_videos_not_in_gold"]:
        raise RuntimeError(f"shared evaluator reported coverage mismatch for {corpus}")
    within = shared["per_video"]["macro_auc"]
    within_n = shared["per_video"]["n_videos_both_classes"]
    if within_n != EXPECTED_WITHIN_COUNTS[corpus]:
        raise RuntimeError(f"within-video cohort count changed for {corpus}: {within_n}")
    fixed_metrics = (shared["pr_auc"], shared["roc_auc"], within)
    if any(value is None or not np.isfinite(value) for value in fixed_metrics):
        raise RuntimeError(f"non-finite fixed metric for {corpus}")
    window_statuses = [window["status"] for row in rows for window in row["windows"]]
    failures = sum(status != "ok" for status in window_statuses)
    result = {
        "split": "test_positive",
        "prediction_artifact": str(path.resolve()),
        "n_videos": len(ids),
        "n_windows": len(window_statuses),
        "n_failed_windows": failures,
        "n_parse_failure_windows": sum(
            status == "parse_failure" for status in window_statuses
        ),
        "n_inference_failure_windows": sum(
            status == "inference_failure" for status in window_statuses
        ),
        "parse_or_inference_failure_rate": failures / max(1, len(window_statuses)),
        "pooled_ap_positive_cohort_diagnostic": shared["pr_auc"],
        "pooled_roc_positive_cohort_diagnostic": shared["roc_auc"],
        "within_video_roc": within,
        "within_n": within_n,
        "within_test_sota_reference": WITHIN_SOTA[corpus],
        "strictly_exceeds_within_sota": within > WITHIN_SOTA[corpus],
    }
    return result


def main():
    corpora = {corpus: corpus_report(corpus) for corpus in CORPORA}
    premise = all(row["strictly_exceeds_within_sota"] for row in corpora.values())
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "test_predictions_and_gt_used_for_method_development": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_evidence_status": "iterative/developmental",
        "corpora": corpora,
        "teacher_premise_pass_both": premise,
        "continue_to_student_design": premise,
        "verdict": "NOVELTY_REVIEW_BEFORE_STUDENT" if premise else "STOP_BEFORE_STUDENT",
    }
    target = RUN_ROOT / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
