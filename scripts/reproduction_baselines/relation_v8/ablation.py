#!/usr/bin/env python3
"""Validation-select the preregistered V8 four-arm mechanism ablation.

The complete per-arm configuration is atomically frozen before test is opened.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "duplex")]
import frame_eval_common as fec
from relation_v4.io import apply_ecdf, fit_ecdf, load_manifest, sha256
from relation_v8.model import UnifiedRelationV8
from relation_v8.run import atomic_json, load_split_exact

MODES = ("prior_only", "locator_only", "uncentered", "full")


def forward(model, values, beta, gamma, mode):
    rows = {}
    with torch.no_grad():
        for vid in sorted(values):
            x = torch.from_numpy(values[vid])[None]
            valid = torch.ones(1, len(x[0]), dtype=torch.bool)
            out = model.forward_ablation(x, valid, beta, gamma, mode)
            rows[vid] = out["frame_score"][0].numpy()
    return rows


def metric(scores, gt):
    return fec.evaluate({v: (scores[v], gt[v]) for v in gt})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("fresh out-dir required")
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(a.manifest)
    n = len(manifest["experts"])
    expected = np.full(n, 1 / n)
    if not np.allclose(manifest["static_weights"], expected, atol=1e-12, rtol=0):
        raise RuntimeError("ablation requires immutable equal consensus")

    val_raw, val_gt, val_prov = load_split_exact(manifest, "val")
    calibration = fit_ecdf(val_raw)
    val = apply_ecdf(val_raw, calibration)
    model = UnifiedRelationV8(n, manifest.get("window", 12),
                              manifest.get("temperature", .2)).eval()
    beta = [float(x) for x in manifest["beta_grid"]]
    gamma = [float(x) for x in manifest["gamma_grid"]]
    grid, selected = {}, {}
    for mode in MODES:
        candidates = [(0., 0.)] if mode == "prior_only" else [
            (b, g) for b in beta for g in gamma]
        rows = []
        for b, g in candidates:
            m = metric(forward(model, val, b, g, mode), val_gt)
            rows.append({"mode": mode, "beta": b, "gamma": g,
                         "frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"]})
        grid[mode] = rows
        selected[mode] = max(rows, key=lambda x: (x["frame_ap"], x["frame_roc"],
                                                   -abs(x["beta"]), -abs(x["gamma"])))

    config = {
        "method": "relation_v8_four_arm_ablation", "corpus": manifest["corpus"],
        "manifest": str(Path(a.manifest).resolve()),
        "manifest_sha256": sha256(a.manifest), "modes": list(MODES),
        "selection_split": "validation", "selection_metric":
        "pooled Frame AP; ROC tie-break; then minimum |beta|,|gamma|",
        "selected": selected, "validation_grid": grid,
        "calibration": {"type": "validation-frozen pooled midpoint ECDF",
                        "sorted_values": calibration},
        "validation_experts": val_prov,
        "test_opened_during_selection": False,
    }
    config_path = out / "frozen_config.json"
    atomic_json(config_path, config)

    # Only the already-frozen four configurations cross this boundary.
    test_raw, test_gt, test_prov = load_split_exact(manifest, "test")
    test = apply_ecdf(test_raw, calibration)
    results, all_scores = {}, {}
    for mode in MODES:
        choice = selected[mode]
        scores = forward(model, test, choice["beta"], choice["gamma"], mode)
        m = metric(scores, test_gt)
        results[mode] = {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
                         "beta": choice["beta"], "gamma": choice["gamma"]}
        all_scores[mode] = scores
    score_path = out / "test_scores.jsonl"
    with score_path.open("w") as handle:
        for vid in sorted(test_gt):
            row = {"video_id": vid}
            for mode in MODES:
                value = all_scores[mode][vid]
                if len(value) != len(test_gt[vid]) or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid {mode} output: {vid}")
                row[mode] = value.tolist()
            handle.write(json.dumps(row) + "\n")
    payload = {"method": config["method"], "corpus": manifest["corpus"],
               "config": str(config_path), "config_sha256": sha256(config_path),
               "scores": str(score_path), "scores_sha256": sha256(score_path),
               "results": results, "test_experts": test_prov,
               "test_labels_used_for_selection": False}
    atomic_json(out / "frame_eval.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
