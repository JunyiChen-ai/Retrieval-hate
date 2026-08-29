#!/usr/bin/env python3
"""Validation-select and infer unified Relation-V8 without test selection."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "duplex")]
import frame_eval_common as fec
from hate_common import data as hdata
from relation_v4.io import apply_ecdf, fit_ecdf, load_manifest, load_split, sha256
from relation_v8.model import UnifiedRelationV8


def forward_all(model, values, beta, gamma):
    result = {}
    # Process videos independently: local transport is O(T^2), and padding all
    # videos to the longest clip would multiply that allocation by the corpus.
    with torch.no_grad():
        for vid in sorted(values):
            score = torch.from_numpy(values[vid])[None]
            valid = torch.ones(1, len(score[0]), dtype=torch.bool)
            out = model(score, valid, beta, gamma)
            result[vid] = {key: value[0].cpu().numpy() for key, value in out.items()
                           if key in ("frame_score", "static_prior", "static_locator",
                                      "transported_locator", "relation_residual",
                                      "locator_correction", "correction")}
            result[vid]["video_prior"] = float(out["video_prior"][0])
    return result


def evaluate(outputs, gt):
    return fec.evaluate({v: (outputs[v]["frame_score"], gt[v]) for v in gt})


def atomic_json(path, payload):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def audit_id_set(observed, expected, source="source"):
    """Require frozen-GT coverage and deterministically archive ignored extras."""
    observed, expected = set(observed), set(expected)
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"{source} missing frozen-GT IDs: {missing[:3]}")
    extras = sorted(observed - expected)
    digest = hashlib.sha256("".join(f"{item}\n" for item in extras).encode()).hexdigest()
    return {"coverage": "all_frozen_gt_ids", "missing_count": 0,
            "extra_count": len(extras), "extra_ids_sorted": extras,
            "extra_ids_sha256": digest, "extras_ignored_for_alignment_and_evaluation": True}


def load_split_exact(manifest, split):
    """Require GT coverage; archive and ignore legal non-cohort source IDs."""
    gt = hdata.gt_arrays(manifest["corpus"], split)
    expected = set(gt)
    audits = []
    for expert in manifest["experts"]:
        paths = expert[f"{split}_scores"]
        paths = [paths] if isinstance(paths, str) else paths
        source_audits = []
        for path in paths:
            observed = set(hdata.load_scores_jsonl(path))
            source_audits.append({"path": str(Path(path).resolve()),
                                  **audit_id_set(observed, expected,
                                                 f"{expert['name']} {split} {path}")})
        audits.append(source_audits)
    raw, aligned_gt, provenance = load_split(manifest, split)
    for item, source_audits in zip(provenance, audits):
        item["id_audit"] = source_audits
    return raw, aligned_gt, provenance


def select_candidate(rows):
    fallback = next(x for x in rows if x["beta"] == 0. and x["gamma"] == 0.)
    eligible = [x for x in rows if x["frame_ap"] >= fallback["frame_ap"]
                and x["frame_roc"] >= fallback["frame_roc"]]
    selected = max(eligible, key=lambda x: (x["frame_ap"], x["frame_roc"],
                                            -abs(x["beta"]), -abs(x["gamma"])))
    return selected, fallback, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("fresh out-dir required")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    n_experts = len(manifest["experts"])
    expected = np.full(n_experts, 1. / n_experts)
    if not np.allclose(manifest["static_weights"], expected, atol=1e-12, rtol=0):
        raise RuntimeError("V8 requires immutable equal-expert consensus")

    val_raw, val_gt, val_provenance = load_split_exact(manifest, "val")
    calibration = fit_ecdf(val_raw)
    val = apply_ecdf(val_raw, calibration)
    model = UnifiedRelationV8(n_experts, manifest.get("window", 12),
                              manifest.get("temperature", .2)).eval()
    beta_grid = [float(x) for x in manifest["beta_grid"]]
    gamma_grid = [float(x) for x in manifest["gamma_grid"]]
    if 0. not in beta_grid or 0. not in gamma_grid:
        raise RuntimeError("grid must contain exact static-prior fallback")
    rows = []
    for beta in beta_grid:
        for gamma in gamma_grid:
            outputs = forward_all(model, val, beta, gamma)
            metric = evaluate(outputs, val_gt)
            rows.append({"beta": beta, "gamma": gamma,
                         "frame_ap": metric["pr_auc"], "frame_roc": metric["roc_auc"]})
    selected, fallback, eligible = select_candidate(rows)
    config = {
        "method": "relation_v8_unified_hierarchical",
        "corpus": manifest["corpus"],
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256(args.manifest),
        "experts": [x["name"] for x in manifest["experts"]],
        "expert_weights": expected.tolist(),
        "selection_rule": "among candidates with validation Frame AP and ROC both >= (beta=0,gamma=0) fallback: max pooled Frame AP; ROC tie-break; then minimum |beta| and minimum |gamma|",
        "selected": selected,
        "fallback": fallback,
        "eligible_count": len(eligible),
        "validation_grid": rows,
        "validation_experts": val_provenance,
        "calibration": {"type": "validation-frozen pooled midpoint ECDF",
                        "sorted_values": calibration},
        "assertions": {"equal_symmetric_consensus": True,
                       "transport_only_in_zero_mean_locator": True,
                       "beta_gamma_zero_exact_static_prior": True},
        "test_opened_during_selection": False,
    }
    config_path = out_dir / "frozen_config.json"
    atomic_json(config_path, config)

    # Test is opened only after the complete frozen validation config exists.
    test_raw, test_gt, test_provenance = load_split_exact(manifest, "test")
    test = apply_ecdf(test_raw, calibration)
    outputs = forward_all(model, test, selected["beta"], selected["gamma"])
    seen = set()
    score_path = out_dir / "test_scores.jsonl"
    with score_path.open("w") as handle:
        for vid in sorted(outputs):
            if vid in seen: raise RuntimeError("duplicate output video")
            seen.add(vid); row = {"video_id": vid, "video_prior": outputs[vid]["video_prior"]}
            for key, value in outputs[vid].items():
                if key != "video_prior":
                    if len(value) != len(test_gt[vid]) or not np.isfinite(value).all():
                        raise RuntimeError(f"output alignment/nonfinite: {vid}/{key}")
                    row[key] = value.tolist()
            handle.write(json.dumps(row) + "\n")
    if seen != set(test_gt): raise RuntimeError("incomplete test output")
    metric = evaluate(outputs, test_gt)
    result = {
        "method": config["method"], "corpus": manifest["corpus"],
        "config": str(config_path), "config_sha256": sha256(config_path),
        "scores": str(score_path), "scores_sha256": sha256(score_path),
        "selected": selected, "test_experts": test_provenance,
        "results": {"frame_ap": metric["pr_auc"], "frame_roc": metric["roc_auc"],
                    "n_videos": metric["n_videos"], "n_frames": metric["n_frames"]},
        "test_labels_used_for_training_or_selection": False,
    }
    atomic_json(out_dir / "frame_eval.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
