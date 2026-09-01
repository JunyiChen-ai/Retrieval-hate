#!/usr/bin/env python3
"""Known graph-propagation ordering upper bound on validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")


def load(path):
    with path.open() as handle:
        return {row["video_id"]: row for row in map(json.loads, handle)}


def transport(anchor, ordering):
    moved = np.empty_like(anchor)
    moved[np.argsort(ordering, kind="stable")] = np.sort(anchor, kind="stable")
    return moved


def propagate(score, visual):
    visual = visual / np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    similarity = visual @ visual.T
    count = max(1, int(.15 * len(score)))
    output = np.empty(len(score), dtype=float)
    for index in range(len(score)):
        neighbor = np.argsort(similarity[index])[-count:]
        logits = similarity[index, neighbor] * 10.0
        weight = np.exp(logits - logits.max())
        weight /= weight.sum()
        output[index] = weight @ score[neighbor]
    return output


def summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def analyze(corpus):
    score_path = REPO / f"runs/20260831_negative_density_probe/{corpus}/scores.jsonl"
    rows = load(score_path)
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    branches = {name: {} for name in (
        "score_powa", "transport_concat", "transport_neighbor",
        "transport_neighbor_smooth",
    )}
    for video_id, target in gt.items():
        anchor = np.asarray(rows[video_id]["score_powa"], dtype=float)
        score = np.asarray(rows[video_id]["score_probe_concat"], dtype=float)
        snippets = align.snippet_bounds(corpus, video_id)
        index = align.snippet_index_for_seconds(snippets, len(target))
        visual_file = np.load(align.visual_path(corpus, video_id), mmap_mode="r")
        # Match VERA's released float32 CLIP-neighbor computation exactly;
        # near-tied similarities make the top-15% set dtype-sensitive.
        visual = np.asarray(visual_file).mean(1)[index]
        neighbor = propagate(score, visual)
        smooth = gaussian_filter1d(
            neighbor, sigma=10, radius=7, mode="nearest"
        )
        branches["score_powa"][video_id] = anchor
        branches["transport_concat"][video_id] = transport(anchor, score)
        branches["transport_neighbor"][video_id] = transport(anchor, neighbor)
        branches["transport_neighbor_smooth"][video_id] = transport(anchor, smooth)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    metrics = {
        name: summary(evaluate_scores(scores, gt, positives))
        for name, scores in branches.items()
    }
    anchor = metrics["score_powa"]
    core = metrics["transport_neighbor_smooth"]
    gates = {
        "within_gain_at_least_0.020": core["within_roc"] >= anchor["within_roc"] + .020,
        "pooled_ap_feasible": core["pooled_ap"] >= anchor["pooled_ap"] - .010,
        "pooled_roc_feasible": core["pooled_roc"] >= anchor["pooled_roc"] - .010,
    }
    return {
        "score_artifact": str(score_path.resolve()),
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
    }


def main():
    corpora = {corpus: analyze(corpus) for corpus in CORPORA}
    payload = {
        "date": "2026-08-31",
        "split": "val",
        "test_used": False,
        "status": "known_calibration_ordering_upper_bound_only",
        "fixed_rule": {
            "source": "train-bag-label concat local probe",
            "visual_neighbor_fraction": .15,
            "softmax_temperature": 10.0,
            "gaussian_sigma": 10,
            "gaussian_radius": 7,
        },
        "corpora": corpora,
        "pass": all(row["pass"] for row in corpora.values()),
        "verdict": "TRAINING_TARGET_FEASIBLE_NOT_A_METHOD",
    }
    out = REPO / "runs/20260831_semantic_neighbor_probe/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
