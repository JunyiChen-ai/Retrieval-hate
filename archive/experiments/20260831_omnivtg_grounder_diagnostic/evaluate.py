#!/usr/bin/env python3
"""Evaluate a frozen OmniVTG interval file on the positive test cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
DUPLEX = REPO / "scripts/duplex"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(DUPLEX))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402
from protocol import (  # noqa: E402
    CODE_VERSION_DESCRIPTION,
    CONTRACT_VERSION,
    EXPECTED_WITHIN_COUNTS,
    FORMAL_RUNTIME_VERSIONS,
    MODEL_ID,
    QUERY,
    ROW_FIELDS,
    parse_interval,
    positive_test_cohort,
)


CONTROL_WITHIN = {
    "hatemm": 0.633766135171972,
    "hateclipseg": 0.5365185532909721,
}
CANONICAL_RUN_ROOT = REPO / "runs/20260831_omnivtg_grounder_diagnostic/formal"


def validate_run_metadata(corpus: str, predictions: Path) -> None:
    config_path = predictions.parent / "config.json"
    version_path = predictions.parent / "code_version.txt"
    expected = {
        "contract_version": CONTRACT_VERSION,
        "corpus": corpus,
        "split": "test",
        "cohort": "video-level-positive fixed evaluator cohort",
        "model": MODEL_ID,
        "query": QUERY,
        "predictions": str(predictions.resolve()),
        "runtime_versions": FORMAL_RUNTIME_VERSIONS,
        "engine_mode": "vLLM multimodal, enforce_eager=True",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
    }
    if json.loads(config_path.read_text()) != expected:
        raise RuntimeError(f"{corpus}: producer config mismatch")
    if version_path.read_text() != CODE_VERSION_DESCRIPTION + "\n":
        raise RuntimeError(f"{corpus}: code version description mismatch")


def load_rows(path: Path, corpus: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise RuntimeError(f"invalid video_id at {path}:{line_number}")
            if video_id in rows:
                raise RuntimeError(f"duplicate prediction for {video_id}")
            if set(row) != ROW_FIELDS:
                raise RuntimeError(f"invalid prediction schema at {path}:{line_number}")
            if (
                row["contract_version"] != CONTRACT_VERSION
                or row["corpus"] != corpus
                or row["split"] != "test"
                or row["model"] != MODEL_ID
                or row["query"] != QUERY
                or not isinstance(row["source_video"], str)
                or not row["source_video"]
                or not isinstance(row["parse_ok"], bool)
            ):
                raise RuntimeError(f"prediction provenance mismatch at {path}:{line_number}")
            parsed = parse_interval(row["completion"])
            if row["parse_ok"]:
                if (
                    parsed is None
                    or row["interval_seconds"] != parsed
                    or any(row[key] is not None for key in (
                        "error_type", "error_message", "traceback"
                    ))
                ):
                    raise RuntimeError(f"invalid successful prediction at {path}:{line_number}")
            elif row["interval_seconds"] is not None:
                raise RuntimeError(f"failed prediction carries interval at {path}:{line_number}")
            elif row["completion"] is not None:
                if (
                    parsed is not None
                    or row["error_type"] != "ParseFailure"
                    or not isinstance(row["error_message"], str)
                    or not row["error_message"]
                    or row["traceback"] is not None
                ):
                    raise RuntimeError(f"invalid parse failure at {path}:{line_number}")
            elif not all(
                isinstance(row[key], str) and row[key]
                for key in ("error_type", "error_message", "traceback")
            ):
                raise RuntimeError(f"invalid inference failure at {path}:{line_number}")
            rows[video_id] = row
    return rows


def score_from_row(row: dict, target: np.ndarray) -> tuple[np.ndarray, str]:
    if not row.get("parse_ok", False):
        status = "parse_failure" if row.get("completion") is not None else "inference_failure"
        return np.zeros_like(target, dtype=float), status
    interval = row.get("interval_seconds")
    if not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
    ):
        raise RuntimeError(f"parse_ok row has invalid interval: {row.get('video_id')}")
    start, end = map(float, interval)
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
        raise RuntimeError(f"parse_ok row has non-finite/reversed interval: {row.get('video_id')}")
    # The frozen evaluator grid is t=0,1,... with half-open span containment.
    # Using duration=len(target) reproduces exactly that grid length; intervals
    # beyond the annotated clock are clipped by the shared conversion routine.
    score, _ = fec.spans_to_frame_scores(
        [(start, end)], [1.0], duration=len(target), fps=1.0, uncovered=0.0
    )
    status = "outside_grid" if not score.any() else "ok"
    return score, status


def evaluate_corpus(corpus: str, predictions: Path) -> dict:
    validate_run_metadata(corpus, predictions)
    split_ids = hdata.load_split(corpus, "test")
    labels = hdata.load_labels(corpus)
    positive_ids = positive_test_cohort(corpus, split_ids, labels)
    rows = load_rows(predictions, corpus)
    if set(rows) != set(positive_ids):
        missing = sorted(set(positive_ids) - set(rows))
        extra = sorted(set(rows) - set(positive_ids))
        raise RuntimeError(
            f"{corpus}: positive test coverage mismatch: missing={missing}, extra={extra}"
        )

    full_gt = hdata.gt_arrays(corpus, "test")
    missing_gold = sorted(set(positive_ids) - set(full_gt))
    if missing_gold:
        raise RuntimeError(f"{corpus}: fixed positive cohort missing GT: {missing_gold}")
    gt = {video_id: full_gt[video_id] for video_id in positive_ids}
    scores: dict[str, np.ndarray] = {}
    statuses: dict[str, str] = {}
    for video_id in positive_ids:
        scores[video_id], statuses[video_id] = score_from_row(rows[video_id], gt[video_id])

    result = evaluate_scores(scores, gt, set(positive_ids))
    if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
        raise RuntimeError("shared evaluator reported a coverage mismatch")
    within_n = result["per_video"]["n_videos_both_classes"]
    if within_n != EXPECTED_WITHIN_COUNTS[corpus]:
        raise RuntimeError(f"{corpus}: within-video cohort count changed: {within_n}")
    within = result["per_video"]["macro_auc"]
    gate = CONTROL_WITHIN[corpus]
    return {
        "corpus": corpus,
        "split": "test",
        "evidence_status": "iterative/developmental test teacher premise",
        "cohort": "all video-level-positive videos in the fixed evaluator test cohort",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "prediction_file": str(predictions.resolve()),
        "n_positive_test_videos": len(positive_ids),
        "n_parse_ok": sum(row["parse_ok"] for row in rows.values()),
        "n_parse_failure": sum(status == "parse_failure" for status in statuses.values()),
        "n_inference_failure": sum(
            status == "inference_failure" for status in statuses.values()
        ),
        "n_interval_outside_grid": sum(status == "outside_grid" for status in statuses.values()),
        "metrics_positive_test_cohort": {
            "pooled_ap": result["pr_auc"],
            "pooled_roc": result["roc_auc"],
            "within_roc": within,
            "within_n": within_n,
        },
        "fixed_structured_control_within_roc": gate,
        "teacher_premise_pass": bool(within is not None and within > gate),
        "gate_note": (
            "Only within-video ROC is comparable to the full-test structured control. "
            "Pooled AP/ROC above are reported on the positive cohort for diagnosis and "
            "are not full-test SOTA claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    corpora = {
        "hatemm": evaluate_corpus(
            "hatemm", CANONICAL_RUN_ROOT / "hatemm/predictions.jsonl"
        ),
        "hateclipseg": evaluate_corpus(
            "hateclipseg",
            CANONICAL_RUN_ROOT / "hateclipseg/predictions.jsonl",
        ),
    }
    premise_pass_both = all(
        row["teacher_premise_pass"] for row in corpora.values()
    )
    payload = {
        "split": "test",
        "evidence_status": "iterative/developmental test teacher premise",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "same_model_query_and_protocol_both_corpora": True,
        "corpus_routing_allowed": False,
        "corpora": corpora,
        "teacher_premise_pass_both": premise_pass_both,
        "continue_to_student_design": premise_pass_both,
        "verdict": (
            "PASS_BOTH_PENDING_SEPARATE_STUDENT_REVIEW"
            if premise_pass_both else "STOP_BEFORE_STUDENT"
        ),
    }
    output = CANONICAL_RUN_ROOT / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
