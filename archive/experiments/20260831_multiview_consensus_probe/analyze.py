#!/usr/bin/env python3
"""Parameter-free local-view consensus feasibility on validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
DENSITY = {
    corpus: REPO / f"runs/20260831_negative_density_probe/{corpus}/scores.jsonl"
    for corpus in CORPORA
}
VERA = {
    corpus: REPO / f"results/reproduction/official_val/final/vera/{corpus}/seed_234/val_infer/scores.jsonl"
    for corpus in CORPORA
}
VIEWS = {
    "all_view": ("audio", "visual", "text", "vera"),
    "density_view": ("audio", "visual", "text"),
    "audio_vera": ("audio", "vera"),
    "visual_vera": ("visual", "vera"),
    "concat_vera": ("concat", "vera"),
}


def load(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def percentile(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.zeros(1)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def transport(anchor, order):
    output = np.empty_like(anchor)
    output[np.argsort(order, kind="stable")] = np.sort(anchor, kind="stable")
    return output


def summary(report):
    return {
        "pooled_ap": report["pr_auc"], "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def analyze(corpus):
    density, vera = load(DENSITY[corpus]), load(VERA[corpus])
    gt, labels = hdata.gt_arrays(corpus, "val"), hdata.load_labels(corpus)
    if set(gt) != set(density) or set(gt) != set(vera):
        raise RuntimeError(f"coverage mismatch: {corpus}")
    branches = {"score_powa": {}}
    for view in ("audio", "visual", "text", "concat", "vera"):
        branches[f"transport_{view}"] = {}
    for name in VIEWS:
        branches[f"transport_{name}"] = {}
    errors = []
    for video_id in sorted(gt):
        anchor = np.asarray(density[video_id]["score_powa"], dtype=float)
        local = {
            "audio": np.asarray(density[video_id]["score_probe_audio"]),
            "visual": np.asarray(density[video_id]["score_probe_visual"]),
            "text": np.asarray(density[video_id]["score_probe_text"]),
            "concat": np.asarray(density[video_id]["score_probe_concat"]),
            "vera": np.asarray(vera[video_id]["score_neighbor"]),
        }
        branches["score_powa"][video_id] = anchor
        ranked = {name: percentile(value) for name, value in local.items()}
        for name, value in ranked.items():
            moved = transport(anchor, value)
            branches[f"transport_{name}"][video_id] = moved
            errors.append(float(np.max(np.abs(np.sort(anchor)-np.sort(moved)))))
        for name, members in VIEWS.items():
            consensus = np.mean([ranked[member] for member in members], axis=0)
            moved = transport(anchor, consensus)
            branches[f"transport_{name}"][video_id] = moved
            errors.append(float(np.max(np.abs(np.sort(anchor)-np.sort(moved)))))
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {name: evaluate_scores(scores, gt, positives)
               for name, scores in branches.items()}
    metrics = {name: summary(report) for name, report in reports.items()}
    anchor, core = metrics["score_powa"], metrics["transport_all_view"]
    gates = {
        "within_gain_at_least_0.020":
        core["within_roc"] >= anchor["within_roc"] + .020,
        "pooled_ap_feasible": core["pooled_ap"] >= anchor["pooled_ap"] - .010,
        "pooled_roc_feasible": core["pooled_roc"] >= anchor["pooled_roc"] - .010,
    }
    return {"metrics": metrics, "max_multiset_error": max(errors),
            "gates": gates, "pass": all(gates.values())}


def main():
    corpora = {corpus: analyze(corpus) for corpus in CORPORA}
    payload = {
        "date": "2026-08-31", "split": "val", "test_used": False,
        "status": "ensemble_upper_bound_only",
        "corpora": corpora, "pass": all(row["pass"] for row in corpora.values()),
        "verdict": ("MULTIVIEW_STUDENT_FEASIBLE" if all(row["pass"] for row in corpora.values())
                    else "STOP_MULTIVIEW_STUDENT"),
    }
    out = REPO / "runs/20260831_multiview_consensus_probe/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
