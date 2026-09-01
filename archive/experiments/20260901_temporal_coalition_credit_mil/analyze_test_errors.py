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


def read_scores(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = {
                key: np.asarray(value, dtype=float)
                for key, value in row.items() if key.startswith("score_")}
    return rows


def group(values):
    return {"n": len(values), "mean": float(np.mean(values)) if values else None}


def analyze(root, corpus):
    gold = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    arms = {arm: read_scores(root / corpus / arm / "scores.jsonl")
            for arm in ("anchor", "aligned", "shifted")}
    rows = []
    for video_id, target in gold.items():
        if labels[video_id] != 1 or len(np.unique(target)) < 2:
            continue
        branch_auc = {
            modality: rank_roc_auc(
                arms["anchor"][video_id]["score_" + modality], target)
            for modality in ("visual", "audio", "text")}
        best = max(branch_auc, key=branch_auc.get)
        anchor_auc = rank_roc_auc(
            arms["anchor"][video_id]["score_fused"], target)
        aligned_auc = rank_roc_auc(
            arms["aligned"][video_id]["score_fused"], target)
        shifted_auc = rank_roc_auc(
            arms["shifted"][video_id]["score_fused"], target)
        rows.append({
            "video_id": video_id, "best_anchor_branch": best,
            "oracle_gap": branch_auc[best] - anchor_auc,
            "aligned_minus_anchor": aligned_auc - anchor_auc,
            "aligned_minus_shifted": aligned_auc - shifted_auc,
        })
    gap_median = float(np.median([row["oracle_gap"] for row in rows]))
    strata = {}
    for key, predicate in {
        "visual_best": lambda row: row["best_anchor_branch"] == "visual",
        "nonvisual_best": lambda row: row["best_anchor_branch"] != "visual",
        "low_oracle_gap": lambda row: row["oracle_gap"] <= gap_median,
        "high_oracle_gap": lambda row: row["oracle_gap"] > gap_median,
    }.items():
        selected = [row for row in rows if predicate(row)]
        strata[key] = {
            "aligned_minus_anchor": group(
                [row["aligned_minus_anchor"] for row in selected]),
            "aligned_minus_shifted": group(
                [row["aligned_minus_shifted"] for row in selected]),
        }
    return {"n_eligible": len(rows), "oracle_gap_median": gap_median,
            "strata": strata, "per_video": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    payload = {
        "date": "2026-09-01",
        "test_exposure": (
            "developmental error analysis of formal predictions and test GT; "
            "not checkpoint selection or confirmatory evidence"),
        "corpora": {corpus: analyze(root, corpus)
                    for corpus in ("hatemm", "hateclipseg")},
    }
    out = root / "test_error_analysis.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({corpus: row["strata"]
                      for corpus, row in payload["corpora"].items()}, indent=2))


if __name__ == "__main__":
    main()
