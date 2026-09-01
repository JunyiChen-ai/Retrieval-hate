#!/usr/bin/env python
"""Evaluate the minimum evidence gate with the canonical metrics code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


def load(path):
    records = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            records[row["video_id"]] = {
                key: np.asarray(value, dtype=float)
                for key, value in row.items() if key != "video_id"}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    result = {"developmental_test_evidence": True, "corpora": {}}
    for corpus in ("hatemm", "hateclipseg"):
        records = load(run_dir / corpus / "scores.jsonl")
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        if set(records) != set(gt):
            raise RuntimeError(f"{corpus}: incomplete test score coverage")
        aligned_scores = {v: row["score_aligned"] for v, row in records.items()}
        shifted_scores = {v: row["score_shifted"] for v, row in records.items()}
        aligned = evaluate_scores(aligned_scores, gt, hate_ids)
        shifted = evaluate_scores(shifted_scores, gt, hate_ids)
        eligible_std = [float(np.std(aligned_scores[v])) for v in sorted(hate_ids)
                        if len(np.unique(gt[v])) == 2]
        aw = float(aligned["per_video"]["macro_auc"])
        sw = float(shifted["per_video"]["macro_auc"])
        gate = {
            "local_change": float(np.median(eligible_std)) > 0.0,
            "aligned_within_above_chance": aw > .5,
            "aligned_beats_shifted": aw > sw,
        }
        gate["pass"] = all(gate.values())
        result["corpora"][corpus] = {
            "aligned": aligned, "shifted_control": shifted,
            "eligible_score_std_median": float(np.median(eligible_std)),
            "aligned_minus_shifted_within": aw - sw, "gate": gate,
        }
        print(corpus, "within", aw, "shifted", sw,
              "delta", aw - sw, "PASS", gate["pass"], flush=True)
    result["joint_pass"] = all(
        row["gate"]["pass"] for row in result["corpora"].values())
    result["decision"] = ("PROCEED_TO_NOVELTY" if result["joint_pass"]
                          else "STOP_INFORMATION_SOURCE")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
