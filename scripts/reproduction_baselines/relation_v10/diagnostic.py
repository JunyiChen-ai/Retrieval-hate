#!/usr/bin/env python3
"""Validation-freeze V10 consensus; test is diagnostic only after freezing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frame_eval_common import evaluate
from relation_v4.io import apply_ecdf, fit_ecdf, load_manifest, sha256
from relation_v8.model import UnifiedRelationV8
from relation_v8.run import atomic_json, load_split_exact, select_candidate
from relation_v10.copula import fit


def components(model, values):
    result = {}
    with torch.no_grad():
        for vid in sorted(values):
            x = torch.from_numpy(values[vid])[None]
            valid = torch.ones(x.shape[:2], dtype=torch.bool)
            base = model(x, valid, 0., 0.)
            result[vid] = {"relation": base["relation_residual"][0].numpy(),
                           "evidence": values[vid]}
    return result


def score(parts, weights, beta, gamma):
    out = {}
    for vid, item in parts.items():
        consensus = item["evidence"] @ weights
        prior = consensus.mean(); locator = consensus - prior
        out[vid] = prior + beta * locator + gamma * item["relation"]
    return out


def metric(scores, gt):
    value = evaluate({v: (scores[v], gt[v]) for v in gt})
    return value["pr_auc"], value["roc_auc"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True); a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise RuntimeError("fresh out-dir required")
    manifest = load_manifest(a.manifest); n = len(manifest["experts"])
    val_raw, val_gt, val_prov = load_split_exact(manifest, "val")
    refs = fit_ecdf(val_raw); val = apply_ecdf(val_raw, refs)
    centered = np.concatenate([val[v] - val[v].mean(0, keepdims=True) for v in sorted(val)])
    dependence = fit(centered)
    identity = np.full(n, 1/n)
    candidates = {"identity": identity,
                  "cluster_equal": dependence["cluster_equal"],
                  "cluster_robust": dependence["cluster_robust"]}
    # Conservative shrinkage provides robust choices without forcing a jump.
    for alpha in (.25, .5, .75):
        candidates[f"robust_shrink_{alpha}"] = (1-alpha)*identity + alpha*dependence["cluster_robust"]
    model = UnifiedRelationV8(n, manifest.get("window", 12), manifest.get("temperature", .2)).eval()
    val_parts = components(model, val); grids = {}
    for name, weights in candidates.items():
        rows = []
        for beta in map(float, manifest["beta_grid"]):
            for gamma in map(float, manifest["gamma_grid"]):
                ap, roc = metric(score(val_parts, weights, beta, gamma), val_gt)
                rows.append({"aggregation": name, "beta": beta, "gamma": gamma,
                             "frame_ap": ap, "frame_roc": roc})
        grids[name] = rows
    identity_selected, _, _ = select_candidate(grids["identity"])
    all_rows = sum(grids.values(), [])
    eligible = [x for x in all_rows if x["frame_ap"] >= identity_selected["frame_ap"]
                and x["frame_roc"] >= identity_selected["frame_roc"]]
    # A global validation gain can be driven by a few videos.  Admit at most
    # the best candidate from each aggregation, and require both metrics to be
    # nondecreasing against identity on every deterministic video fold.
    representatives = []
    for name in candidates:
        rows = [x for x in eligible if x["aggregation"] == name]
        if rows:
            representatives.append(max(rows, key=lambda x: (x["frame_ap"], x["frame_roc"],
                                                             -abs(x["beta"]), -abs(x["gamma"]))))
    folds = [sorted(val_gt)[i::5] for i in range(5)]
    identity_scores = score(val_parts, identity, identity_selected["beta"], identity_selected["gamma"])
    stability = []
    for row in representatives:
        candidate_scores = score(val_parts, candidates[row["aggregation"]], row["beta"], row["gamma"])
        fold_rows = []
        for index, ids in enumerate(folds):
            cap, croc = metric({v:candidate_scores[v] for v in ids}, {v:val_gt[v] for v in ids})
            iap, iroc = metric({v:identity_scores[v] for v in ids}, {v:val_gt[v] for v in ids})
            fold_rows.append({"fold": index, "n_videos": len(ids),
                              "ap_delta": cap-iap, "roc_delta": croc-iroc})
        stable = all(x["ap_delta"] >= 0 and x["roc_delta"] >= 0 for x in fold_rows)
        stability.append({"candidate": row, "all_folds_nondecrease": stable,
                          "fold_deltas": fold_rows})
    stable_rows = [x["candidate"] for x in stability if x["all_folds_nondecrease"]]
    if not stable_rows: raise RuntimeError("identity candidate unexpectedly failed stability gate")
    selected = max(stable_rows, key=lambda x: (x["frame_ap"], x["frame_roc"],
                                                x["aggregation"] == "identity",
                                                -abs(x["beta"]), -abs(x["gamma"])))
    frozen = {"method": "relation_v10_performance_preserving_copula",
              "corpus": manifest["corpus"], "manifest": str(Path(a.manifest).resolve()),
              "manifest_sha256": sha256(a.manifest), "dependence_source": "validation scores; labels unused",
              "ecdf_reference": "validation frozen", "clusters": dependence["clusters"],
              "correlation": dependence["correlation"].tolist(),
              "cluster_quality": dependence["cluster_quality"].tolist(),
              "candidate_weights": {k: v.tolist() for k, v in candidates.items()},
              "identity_v8_fallback": identity_selected, "selected": selected,
              "selection_rule": "global validation AP/ROC and each of five deterministic video-fold AP/ROC must all not decrease versus exact identity V8 fallback",
              "validation_video_fold_stability": stability,
              "validation_grid": grids, "validation_sources": val_prov,
              "calibration": [x for x in refs], "test_opened_during_selection": False}
    config = out / "frozen_config.json"; atomic_json(config, frozen)
    test_raw, test_gt, test_prov = load_split_exact(manifest, "test")
    test = apply_ecdf(test_raw, refs); test_parts = components(model, test)
    chosen = candidates[selected["aggregation"]]
    selected_scores = score(test_parts, chosen, selected["beta"], selected["gamma"])
    fallback_scores = score(test_parts, identity, identity_selected["beta"], identity_selected["gamma"])
    sap, sroc = metric(selected_scores, test_gt); fap, froc = metric(fallback_scores, test_gt)
    payload = {"method": frozen["method"], "corpus": manifest["corpus"],
               "selected": selected, "test": {"frame_ap": sap, "frame_roc": sroc},
               "identity_fallback_test": {"frame_ap": fap, "frame_roc": froc},
               "test_delta": {"frame_ap": sap-fap, "frame_roc": sroc-froc},
               "test_sources": test_prov, "test_labels_used_for_selection": False,
               "config": str(config.resolve()), "config_sha256": sha256(config)}
    atomic_json(out / "diagnostic.json", payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
