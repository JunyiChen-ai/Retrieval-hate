#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/reproduction_baselines"),
                str(ROOT / "scripts/duplex")]
from hate_common import data as hdata  # noqa: E402
from frame_eval_common import rank_roc_auc  # noqa: E402

SOTA = {
    "hatemm": {"ap": .5938315566, "roc": .8161837922, "within": .6315317180},
    "hateclipseg": {"ap": .6193710950, "roc": .6050224699, "within": .5619078936},
}
ANCHOR = {"hatemm": .6284561854116418,
          "hateclipseg": .5237011979483558}


def read_metrics(path):
    row = json.loads(path.read_text())["results"]["score_fused"]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"],
            "within": row["per_video"]["macro_auc"]}


def read_scores(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = np.asarray(row["score_fused"], float)
    return rows


def occupancy_deltas(root, corpus):
    gold = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    control = read_scores(root / corpus / "fixed_topk_control/scores.jsonl")
    core = read_scores(root / corpus / "sparse_scan/scores.jsonl")
    strata = {"low_le_1_3": [], "middle": [], "high_gt_2_3": []}
    for video_id, target in gold.items():
        if labels[video_id] != 1 or len(np.unique(target)) < 2:
            continue
        fraction = float(np.mean(target))
        delta = rank_roc_auc(core[video_id], target) - rank_roc_auc(
            control[video_id], target)
        key = ("low_le_1_3" if fraction <= 1 / 3 else
               "high_gt_2_3" if fraction > 2 / 3 else "middle")
        strata[key].append(delta)
    return {key: {"n": len(values),
                  "mean_core_minus_control_within": float(np.mean(values))
                  if values else None}
            for key, values in strata.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    out = {"corpora": {}}
    for corpus, sota in SOTA.items():
        arms = {arm: read_metrics(root / corpus / arm / "metrics.json")
                for arm in ("fixed_topk_control", "sparse_scan")}
        core = arms["sparse_scan"]
        gains = {"vs_control": core["within"] - arms["fixed_topk_control"]["within"],
                 "vs_multihateloc_anchor": core["within"] - ANCHOR[corpus]}
        strata = occupancy_deltas(root, corpus)
        out["corpora"][corpus] = {
            "arms": arms, "core_within_gains": gains,
            "occupancy_strata": strata,
            "core_all_sota": all(core[key] > sota[key] for key in sota)}
    gains = [row["core_within_gains"] for row in out["corpora"].values()]
    out["mechanism_gate"] = {
        "core_beats_control_and_anchor_on_both": all(
            row["vs_control"] > 0 and row["vs_multihateloc_anchor"] > 0
            for row in gains),
        "at_least_one_anchor_gain_ge_020": max(
            row["vs_multihateloc_anchor"] for row in gains) >= .020,
        "low_occupancy_gain_exceeds_high_on_both": all(
            row["occupancy_strata"]["low_le_1_3"]["mean_core_minus_control_within"] >
            row["occupancy_strata"]["high_gt_2_3"]["mean_core_minus_control_within"]
            for row in out["corpora"].values()),
    }
    out["performance_gate"] = {"both_corpora_all_three_sota": all(
        row["core_all_sota"] for row in out["corpora"].values())}
    out["decision"] = ("EXPAND" if all(out["mechanism_gate"].values()) and
                       out["performance_gate"]["both_corpora_all_three_sota"]
                       else "STOP_OR_ONE_POST_TEST_CORRECTION")
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
