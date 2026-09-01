#!/usr/bin/env python
"""Post-test mechanism analysis for the completed dual-corpus pilot."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


MODALITIES = ("visual", "audio", "text")
ARMS = ("anchor", "source_dgm", "witness_dgm")
CORPORA = ("hatemm", "hateclipseg")


def load_scores(path):
    records = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row.pop("video_id")
            records[video_id] = {
                key: np.asarray(value, dtype=float) for key, value in row.items()}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_dir).resolve()
    output = {"developmental_test_error_analysis": True, "corpora": {}}
    for corpus in CORPORA:
        corpus_out = {"arms": {}}
        loaded = {}
        for arm in ARMS:
            run = root / corpus / arm
            log = json.loads((run / "train_log.json").read_text())
            branch = json.loads((run / "branch_metrics.json").read_text())
            videos = log["test_video_diagnostics"]
            weights = np.asarray([videos[v]["weights"] for v in sorted(videos)])
            selected = Counter(MODALITIES[int(index)]
                               for index in np.argmax(weights, axis=1))
            history = log["history"]
            corpus_out["arms"][arm] = {
                "selected_epoch": log["selected_epoch"],
                "mean_dms_weights": dict(zip(MODALITIES,
                                              weights.mean(axis=0).tolist())),
                "dms_argmax_counts": {name: int(selected[name])
                                      for name in MODALITIES},
                "mean_training_coefficients": {
                    name: float(np.mean([row["coefficient_" + name]
                                         for row in history]))
                    for name in MODALITIES},
                "branch_within_roc": {
                    name.removeprefix("score_"):
                        value["per_video"]["macro_auc"]
                    for name, value in branch["results"].items()},
            }
            loaded[arm] = load_scores(run / "scores.jsonl")
        common = sorted(set(loaded["anchor"]) & set(loaded["witness_dgm"]))
        correlations, absolute = [], []
        for video_id in common:
            anchor = loaded["anchor"][video_id]["score_fused"]
            core = loaded["witness_dgm"][video_id]["score_fused"]
            if np.std(anchor) > 0 and np.std(core) > 0:
                correlations.append(float(spearmanr(anchor, core).statistic))
            absolute.append(float(np.mean(np.abs(anchor - core))))
        corpus_out["core_vs_anchor_final_score"] = {
            "n_videos": len(common),
            "mean_per_video_spearman": float(np.mean(correlations)),
            "median_per_video_spearman": float(np.median(correlations)),
            "mean_absolute_difference": float(np.mean(absolute)),
        }
        output["corpora"][corpus] = corpus_out
    (root / "mechanism_analysis.json").write_text(
        json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
