#!/usr/bin/env python
"""Focused developmental test analysis for sequence-crowd controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
from hate_common import data as hdata  # noqa: E402


ARMS = ("core", "token_ds", "unconstrained_bsc")


def load_arm(root, arm):
    records = hdata.load_scores_jsonl(str(root / arm / "scores.jsonl"))
    return {video_id: np.asarray(row["score_method"], dtype=float)
            for video_id, row in records.items()}


def quartile_rows(rows, comparison):
    occupancy = np.asarray([row["positive_fraction"] for row in rows])
    edges = np.quantile(occupancy, [.25, .5, .75])
    groups = []
    bins = np.digitize(occupancy, edges, right=True)
    for group in range(4):
        selected = [row for row, value in zip(rows, bins) if value == group]
        delta = [row[comparison] for row in selected]
        groups.append({
            "quartile": group + 1,
            "n_videos": len(selected),
            "positive_fraction_min": min(row["positive_fraction"] for row in selected),
            "positive_fraction_max": max(row["positive_fraction"] for row in selected),
            "mean_within_delta": float(np.mean(delta)),
            "improve_tie_worse": [
                sum(value > 1e-12 for value in delta),
                sum(abs(value) <= 1e-12 for value in delta),
                sum(value < -1e-12 for value in delta),
            ],
        })
    return {"occupancy_edges": edges.tolist(), "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    test_root = Path(args.test_root)
    scores = {arm: load_arm(test_root, arm) for arm in ARMS}
    gt = hdata.gt_arrays(args.corpus, "test")
    for arm in ARMS:
        if set(scores[arm]) != set(gt):
            raise RuntimeError(f"{arm} does not exactly cover frozen test GT")
    rows = []
    for video_id, y in gt.items():
        y = np.asarray(y)
        if len(np.unique(y)) != 2:
            continue
        auc = {arm: float(roc_auc_score(y, scores[arm][video_id]))
               for arm in ARMS}
        rows.append({
            "video_id": video_id,
            "n_seconds": len(y),
            "positive_fraction": float(y.mean()),
            "core_within": auc["core"],
            "token_ds_within": auc["token_ds"],
            "unconstrained_bsc_within": auc["unconstrained_bsc"],
            "core_minus_token_ds": auc["core"] - auc["token_ds"],
            "core_minus_unconstrained_bsc": (
                auc["core"] - auc["unconstrained_bsc"]),
        })
    comparisons = {}
    for key in ("core_minus_token_ds", "core_minus_unconstrained_bsc"):
        values = np.asarray([row[key] for row in rows])
        comparisons[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "improve_tie_worse": [
                int((values > 1e-12).sum()), int((np.abs(values) <= 1e-12).sum()),
                int((values < -1e-12).sum()),
            ],
            "by_positive_fraction_quartile": quartile_rows(rows, key),
        }
    payload = {
        "date": "2026-09-01",
        "corpus": args.corpus,
        "split": "test",
        "test_artifacts": {arm: str((test_root / arm / "scores.jsonl").resolve())
                           for arm in ARMS},
        "test_predictions_and_gt_used_for_developmental_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "n_eligible_positive_videos": len(rows),
        "comparisons": comparisons,
        "per_video": rows,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"corpus": args.corpus, "n": len(rows),
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
