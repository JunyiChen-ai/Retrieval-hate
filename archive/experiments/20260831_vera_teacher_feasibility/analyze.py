#!/usr/bin/env python3
"""Measure VERA-order / POWA-measure complementarity on validation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


ANCHORS = {
    "hatemm": REPO / "runs/20260831_powa_span_marginal_pilot/hatemm_span_marginal_seed234/val_scores_epoch0.jsonl",
    "hateclipseg": REPO / "runs/20260831_powa_span_marginal_pilot/hateclipseg_span_marginal_seed234/val_scores_epoch0.jsonl",
}
VERA = {
    corpus: REPO / f"results/reproduction/official_val/final/vera/{corpus}/seed_234/val_infer/scores.jsonl"
    for corpus in ANCHORS
}
VERA_BRANCHES = ("score_raw", "score_neighbor", "score_official_postprocessed")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def transport(anchor, order):
    output = np.empty_like(anchor)
    output[np.argsort(order, kind="stable")] = np.sort(anchor, kind="stable")
    return output


def summarize(report):
    return {
        "pooled_ap": report["pr_auc"], "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def analyze(corpus):
    anchor_rows, vera_rows = load(ANCHORS[corpus]), load(VERA[corpus])
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    ids = sorted(gt)
    if set(ids) != set(anchor_rows) or set(ids) != set(vera_rows):
        raise RuntimeError(f"coverage mismatch for {corpus}")
    branches = {"score_powa": {}, "score_position_center": {}}
    for name in VERA_BRANCHES:
        branches[name] = {}
        branches[f"transport_{name}"] = {}
    errors = []
    for video_id in ids:
        anchor = np.asarray(anchor_rows[video_id]["score_powa"], dtype=float)
        if len(anchor) != len(gt[video_id]):
            raise RuntimeError(f"anchor alignment mismatch {corpus}/{video_id}")
        branches["score_powa"][video_id] = anchor
        center = -np.abs(np.arange(len(anchor)) - (len(anchor) - 1) / 2)
        branches["score_position_center"][video_id] = transport(anchor, center)
        for name in VERA_BRANCHES:
            teacher = np.asarray(vera_rows[video_id][name], dtype=float)
            if teacher.shape != anchor.shape:
                raise RuntimeError(f"VERA alignment mismatch {corpus}/{video_id}/{name}")
            branches[name][video_id] = teacher
            moved = transport(anchor, teacher)
            branches[f"transport_{name}"][video_id] = moved
            errors.append(float(np.max(np.abs(np.sort(anchor) - np.sort(moved)))))
    positives = {video_id for video_id in ids if labels[video_id] == 1}
    reports = {name: evaluate_scores(scores, gt, positives)
               for name, scores in branches.items()}
    return {
        "anchor_path": str(ANCHORS[corpus]),
        "anchor_sha256": sha256(ANCHORS[corpus]),
        "vera_path": str(VERA[corpus]), "vera_sha256": sha256(VERA[corpus]),
        "max_multiset_error": max(errors),
        "metrics": {name: summarize(report) for name, report in reports.items()},
    }


def main():
    result = {
        "date": "2026-08-31", "split": "val",
        "status": "upper_bound_diagnostic", "test_used": False,
        "corpora": {corpus: analyze(corpus) for corpus in ANCHORS},
    }
    out = REPO / "runs/20260831_vera_teacher_feasibility/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
