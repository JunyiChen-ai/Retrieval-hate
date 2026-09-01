#!/usr/bin/env python3
"""Exploratory validation-only empirical-null/BH premise probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "scripts/reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
VIEWS = ("audio", "visual", "text", "concat")
Q = 0.10


def load(path):
    with path.open() as handle:
        return {row["video_id"]: row for row in map(json.loads, handle)}


def bh_mask(pvalues, q):
    order = np.argsort(pvalues, kind="stable")
    accepted = np.flatnonzero(
        pvalues[order] <= q * np.arange(1, len(order) + 1) / len(order)
    )
    mask = np.zeros(len(order), dtype=bool)
    if len(accepted):
        mask[order[:accepted[-1] + 1]] = True
    return mask


def analyze(corpus):
    path = REPO / f"runs/20260831_negative_density_probe/{corpus}/scores.jsonl"
    rows = load(path)
    labels = hdata.load_labels(corpus)
    gt = hdata.gt_arrays(corpus, "val")
    eligible = [
        video_id for video_id in sorted(gt)
        if labels[video_id] == 1 and len(np.unique(gt[video_id])) == 2
    ]
    result = {}
    for view in VIEWS:
        key = f"score_probe_{view}"
        null = np.sort(np.concatenate([
            np.asarray(rows[video_id][key], dtype=float)
            for video_id in rows if labels[video_id] == 0
        ]))
        precision, coverage = [], []
        for video_id in eligible:
            score = np.asarray(rows[video_id][key], dtype=float)
            pvalue = (
                len(null) - np.searchsorted(null, score, side="left") + 1
            ) / (len(null) + 1)
            selected = bh_mask(pvalue, Q)
            if selected.any():
                precision.append(float(gt[video_id][selected].mean()))
                coverage.append(float(selected.mean()))
        result[view] = {
            "eligible_videos": len(eligible),
            "videos_with_discoveries": len(precision),
            "macro_precision_when_discovered": (
                float(np.mean(precision)) if precision else None
            ),
            "macro_selected_fraction_when_discovered": (
                float(np.mean(coverage)) if coverage else None
            ),
            "eligible_video_positive_fraction_macro": float(np.mean([
                gt[video_id].mean() for video_id in eligible
            ])),
        }
    return {"score_artifact": str(path.resolve()), "views": result}


def main():
    corpora = {corpus: analyze(corpus) for corpus in CORPORA}
    hcs = corpora["hateclipseg"]["views"]["concat"]
    payload = {
        "date": "2026-08-31",
        "split": "val",
        "test_used": False,
        "status": "exploratory_premise_probe",
        "bh_q": Q,
        "corpora": corpora,
        "verdict": "STOP_BEFORE_NOVELTY",
        "reason": (
            "fixed concat rule has insufficient HCS coverage and precision "
            "below the eligible-video positive-fraction baseline"
        ),
        "hcs_concat_failed": (
            hcs["videos_with_discoveries"] < 10
            or hcs["macro_precision_when_discovered"]
            <= hcs["eligible_video_positive_fraction_macro"]
        ),
    }
    out = REPO / "runs/20260831_negative_conformal_probe/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
