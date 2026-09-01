#!/usr/bin/env python3
"""Evaluate one frozen witness prediction artifact with the shared evaluator."""

from __future__ import annotations

import argparse
import json
import os
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
from protocol import TEST_SOTA, evaluator_test_ids  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("corpus") != args.corpus:
        raise RuntimeError("config corpus mismatch")
    completion = json.loads((run_dir / "training_complete.json").read_text())
    if completion.get("status") != "prediction_complete":
        raise RuntimeError("test prediction is not marked complete")
    if config.get("test_labels_used_for_gradient_or_checkpoint_selection") is not False:
        raise RuntimeError("run config does not certify the test-label isolation boundary")
    branches = (
        "score",
        "score_independent_transport_control",
        "score_nearest_normal_control",
    )
    rows = []
    for line_number, line in enumerate(
        (run_dir / "predictions.jsonl").read_text().splitlines(), 1
    ):
        if not line.strip():
            raise RuntimeError(f"blank prediction line {line_number}")
        row = json.loads(line)
        if set(row) != {"video_id", *branches}:
            raise RuntimeError("prediction schema mismatch")
        rows.append(row)
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate prediction video")

    gold = hdata.gt_arrays(args.corpus, "test")
    expected = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    if set(expected) != set(gold):
        raise RuntimeError("fixed evaluator-test cohort no longer matches test gold")
    if ids != expected:
        raise RuntimeError("predictions are not the exact ordered evaluator-test cohort")
    scores = {branch: {} for branch in branches}
    for row in rows:
        video_id = row["video_id"]
        for branch in branches:
            value = np.asarray(row[branch], dtype=np.float64)
            if value.ndim != 1 or len(value) != len(gold[video_id]):
                raise RuntimeError(f"{branch} score/GT length mismatch for {video_id}")
            if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
                raise RuntimeError(f"invalid {branch} score range for {video_id}")
            scores[branch][video_id] = value

    reports = {}
    for branch in branches:
        metrics = evaluate_scores(scores[branch], gold, set(expected))
        if metrics["n_videos_missing_from_scores"] or metrics["n_videos_not_in_gold"]:
            raise RuntimeError(f"shared evaluator reported {branch} coverage mismatch")
        reports[branch] = {
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "within_video_roc": metrics["per_video"]["macro_auc"],
            "within_n": metrics["per_video"]["n_videos_both_classes"],
            "shared_evaluator": metrics,
        }
        reports[branch]["strictly_exceeds_test_sota"] = {
            metric: reports[branch][metric] > reference
            for metric, reference in TEST_SOTA[args.corpus].items()
        }
        reports[branch]["all_fixed_metrics_sota"] = all(
            reports[branch]["strictly_exceeds_test_sota"].values()
        )
    payload = {
        "date": "2026-08-31",
        "corpus": args.corpus,
        "split": "test",
        "test_predictions_and_gt_used_for_method_development": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_sota_reference": TEST_SOTA[args.corpus],
        "branches": reports,
        "core_all_fixed_metrics_sota": reports["score"]["all_fixed_metrics_sota"],
        "core_exceeds_both_attribution_controls_within": all(
            reports["score"]["within_video_roc"]
            > reports[branch]["within_video_roc"]
            for branch in branches[1:]
        ),
    }
    target = run_dir / "metrics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
