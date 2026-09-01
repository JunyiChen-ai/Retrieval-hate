#!/usr/bin/env python3
"""Evaluate typed REBA branches with the repository shared evaluator."""

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
from protocol import TEST_SOTA  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids  # noqa: E402


BRANCHES = ("score", "score_scale1_control")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    completion = json.loads((run_dir / "training_complete.json").read_text())
    if config.get("corpus") != args.corpus:
        raise RuntimeError("config corpus mismatch")
    if config.get("test_labels_used_for_gradient_or_checkpoint_selection") is not False:
        raise RuntimeError("test-label isolation statement missing")
    if completion.get("status") != "prediction_complete":
        raise RuntimeError("prediction is not marked complete")

    rows = []
    for line_number, line in enumerate(
        (run_dir / "predictions.jsonl").read_text().splitlines(), 1
    ):
        if not line.strip():
            raise RuntimeError(f"blank prediction row {line_number}")
        row = json.loads(line)
        if set(row) != {"video_id", *BRANCHES}:
            raise RuntimeError("prediction schema mismatch")
        rows.append(row)
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate prediction id")
    gold = hdata.gt_arrays(args.corpus, "test")
    expected = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    if ids != expected or set(expected) != set(gold):
        raise RuntimeError("prediction/evaluator-test cohort mismatch")

    branch_scores = {branch: {} for branch in BRANCHES}
    for row in rows:
        video_id = row["video_id"]
        for branch in BRANCHES:
            score = np.asarray(row[branch], dtype=np.float64)
            if score.shape != gold[video_id].shape:
                raise RuntimeError(f"{branch} score/GT shape mismatch for {video_id}")
            if not np.isfinite(score).all() or np.any(score < 0) or np.any(score > 1):
                raise RuntimeError(f"invalid {branch} values for {video_id}")
            branch_scores[branch][video_id] = score

    reports = {}
    for branch in BRANCHES:
        shared = evaluate_scores(branch_scores[branch], gold, set(expected))
        if shared["n_videos_missing_from_scores"] or shared["n_videos_not_in_gold"]:
            raise RuntimeError(f"shared evaluator rejected {branch} coverage")
        report = {
            "pr_auc": shared["pr_auc"],
            "roc_auc": shared["roc_auc"],
            "within_video_roc": shared["per_video"]["macro_auc"],
            "within_n": shared["per_video"]["n_videos_both_classes"],
            "shared_evaluator": shared,
        }
        report["strictly_exceeds_test_sota"] = {
            name: report[name] > reference
            for name, reference in TEST_SOTA[args.corpus].items()
        }
        report["all_fixed_metrics_sota"] = all(
            report["strictly_exceeds_test_sota"].values()
        )
        reports[branch] = report

    payload = {
        "date": "2026-08-31",
        "corpus": args.corpus,
        "split": "test",
        "test_predictions_and_gt_used_for_method_development": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_sota_reference": TEST_SOTA[args.corpus],
        "branches": reports,
        "core_all_fixed_metrics_sota": reports["score"]["all_fixed_metrics_sota"],
        "core_exceeds_scale1_control_within": (
            reports["score"]["within_video_roc"]
            > reports["score_scale1_control"]["within_video_roc"]
        ),
    }
    target = run_dir / "metrics.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
