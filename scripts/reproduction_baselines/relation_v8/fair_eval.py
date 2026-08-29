#!/usr/bin/env python3
"""Matched-seed, video-clustered evaluation for Relation-V8.

Every trainable expert must expose exactly seeds 234/2025/3407.  A one-path
expert is treated as frozen and shared, explicitly recorded in provenance.
Hyperparameters and ECDF references are fitted independently on validation for
each matched seed.  Test is opened only after all three configs are frozen.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch

from relation_v4.io import apply_ecdf, fit_ecdf, load_manifest, sha256
from relation_v8.model import UnifiedRelationV8
from relation_v8.run import atomic_json, load_split_exact, select_candidate
from frame_eval_common import average_precision, rank_roc_auc

SEEDS = (234, 2025, 3407)
SEED_RE = re.compile(r"(?:^|/)seed_(\d+)(?:/|$)")


def _paths(value):
    return [value] if isinstance(value, str) else list(value)


def seed_map(paths, expert, split, fixed_across_seeds=False):
    """Classify an expert as frozen/shared or exact three-seed trainable."""
    paths = _paths(paths)
    if len(paths) == 1:
        if fixed_across_seeds is not True:
            raise RuntimeError(f"{expert} {split}: single path requires explicit fixed_across_seeds:true")
        return {seed: paths[0] for seed in SEEDS}, "fixed_shared"
    if fixed_across_seeds:
        raise RuntimeError(f"{expert} {split}: fixed_across_seeds forbids multi-path scores")
    found = {}
    for path in paths:
        match = SEED_RE.search(str(path))
        if match is None:
            raise RuntimeError(f"{expert} {split}: multi-run path lacks seed_N: {path}")
        seed = int(match.group(1))
        if seed in found:
            raise RuntimeError(f"{expert} {split}: duplicate seed {seed}")
        found[seed] = path
    missing, extra = sorted(set(SEEDS) - set(found)), sorted(set(found) - set(SEEDS))
    if missing or extra:
        raise RuntimeError(f"{expert} {split}: expected {SEEDS}; missing={missing} extra={extra}")
    return found, "trainable_matched"


def validate_fixed_provenance(expert):
    required = {"val_checkpoint_identity", "test_checkpoint_identity",
                "identity_source", "identity_source_sha256"}
    provenance = expert.get("fixed_provenance")
    if not isinstance(provenance, dict) or not required <= set(provenance):
        raise RuntimeError(f"{expert['name']}: fixed expert requires explicit fixed_provenance {sorted(required)}")
    if not provenance["val_checkpoint_identity"] or not provenance["test_checkpoint_identity"]:
        raise RuntimeError(f"{expert['name']}: empty checkpoint identity")
    if provenance["val_checkpoint_identity"] != provenance["test_checkpoint_identity"]:
        raise RuntimeError(f"{expert['name']}: val/test checkpoint identity mismatch")
    source = Path(provenance["identity_source"])
    if not source.is_file():
        raise RuntimeError(f"{expert['name']}: missing identity source {source}")
    observed = sha256(source)
    if observed != provenance["identity_source_sha256"]:
        raise RuntimeError(f"{expert['name']}: identity source hash mismatch")
    return {**provenance, "identity_source": str(source.resolve()),
            "identity_source_hash_verified": True}


def matched_manifests(manifest):
    maps, audit = {}, []
    for expert in manifest["experts"]:
        fixed = expert.get("fixed_across_seeds", False)
        if not isinstance(fixed, bool):
            raise RuntimeError(f"{expert['name']}: fixed_across_seeds must be boolean")
        val, val_kind = seed_map(expert["val_scores"], expert["name"], "val", fixed)
        test, test_kind = seed_map(expert["test_scores"], expert["name"], "test", fixed)
        if val_kind != test_kind:
            raise RuntimeError(f"{expert['name']}: val/test fixed-vs-trainable mismatch")
        fixed_provenance = validate_fixed_provenance(expert) if fixed else None
        maps[expert["name"]] = (val, test)
        audit.append({"name": expert["name"], "kind": val_kind,
                      "seeds": list(SEEDS) if val_kind == "trainable_matched" else [],
                      "shared_path_across_seeds": val_kind == "fixed_shared",
                      "fixed_provenance": fixed_provenance})
    out = {}
    for seed in SEEDS:
        item = copy.deepcopy(manifest)
        for expert in item["experts"]:
            val, test = maps[expert["name"]]
            expert["val_scores"], expert["test_scores"] = val[seed], test[seed]
        out[seed] = item
    return out, audit


def forward(model, values, beta, gamma):
    result = {}
    with torch.no_grad():
        for vid in sorted(values):
            x = torch.from_numpy(values[vid])[None]
            valid = torch.ones(x.shape[:2], dtype=torch.bool)
            result[vid] = model(x, valid, beta, gamma)["frame_score"][0].numpy()
    return result


def pooled(scores, gt, ids=None):
    ids = sorted(gt) if ids is None else list(ids)
    s = np.concatenate([np.asarray(scores[v]) for v in ids])
    y = np.concatenate([np.asarray(gt[v]) for v in ids])
    return {"frame_ap": average_precision(s, y), "frame_roc": rank_roc_auc(s, y)}


def extended_metrics(scores, gt):
    out = pooled(scores, gt)
    eligible = [v for v in sorted(gt) if np.any(gt[v] == 1) and np.any(gt[v] == 0)]
    aucs = [rank_roc_auc(scores[v], gt[v]) for v in eligible]
    hateful = [v for v in sorted(gt) if np.any(gt[v] == 1)]
    hateful_roc = pooled(scores, gt, hateful)["frame_roc"] if hateful else None
    out.update({
        "within_video_macro_roc": float(np.mean(aucs)) if aucs else None,
        "within_video_macro_roc_std": (float(np.std(aucs, ddof=1))
                                        if len(aucs) > 1 else None),
        "within_video_macro_roc_eligible_videos": len(eligible),
        "hateful_video_only_pooled_roc": hateful_roc,
        "hateful_video_count": len(hateful),
        "n_videos": len(gt),
        "n_frames": int(sum(len(x) for x in gt.values())),
    })
    return out


def summarize_seed_metrics(rows):
    result = {}
    keys = ("frame_ap", "frame_roc", "within_video_macro_roc",
            "hateful_video_only_pooled_roc")
    for key in keys:
        values = np.asarray([row[key] for row in rows if row[key] is not None], float)
        result[key] = {"mean": float(values.mean()) if len(values) else None,
                       "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                       "n_seeds": int(len(values))}
    result["within_video_macro_roc_eligible_videos"] = [
        int(x["within_video_macro_roc_eligible_videos"]) for x in rows]
    return result


def select_shared(seed_grids):
    """Select one configuration from mean validation metrics over three seeds."""
    if set(seed_grids) != set(SEEDS):
        raise ValueError("shared selection requires all matched validation seeds")
    indexed = {}
    for seed, rows in seed_grids.items():
        current = {(float(x["beta"]), float(x["gamma"])): x for x in rows}
        if len(current) != len(rows):
            raise RuntimeError(f"seed {seed}: duplicate validation candidate")
        indexed[seed] = current
    candidates = set(indexed[SEEDS[0]])
    if any(set(indexed[s]) != candidates for s in SEEDS[1:]):
        raise RuntimeError("validation candidate grids differ across seeds")
    averaged = []
    for beta, gamma in sorted(candidates):
        averaged.append({"beta": beta, "gamma": gamma,
                         "frame_ap": float(np.mean([
                             indexed[s][(beta, gamma)]["frame_ap"] for s in SEEDS])),
                         "frame_roc": float(np.mean([
                             indexed[s][(beta, gamma)]["frame_roc"] for s in SEEDS]))})
    selected, fallback, eligible = select_candidate(averaged)
    return selected, fallback, eligible, averaged


def cluster_bootstrap(method_runs, baseline_runs, gt, n_boot=2000, rng_seed=20260829):
    """Jointly resample video clusters; report mean matched-seed differences."""
    if set(method_runs) != set(SEEDS) or set(baseline_runs) != set(SEEDS):
        raise ValueError("bootstrap requires all matched seeds")
    ids = sorted(gt)
    rng = np.random.default_rng(rng_seed)
    point, draws = {}, {"frame_ap": [], "frame_roc": []}
    for metric in draws:
        point[metric] = float(np.mean([
            pooled(method_runs[s], gt)[metric] - pooled(baseline_runs[s], gt)[metric]
            for s in SEEDS]))
    for _ in range(int(n_boot)):
        sampled = rng.choice(ids, size=len(ids), replace=True).tolist()
        for metric in draws:
            pairs = [(pooled(method_runs[s], gt, sampled)[metric],
                      pooled(baseline_runs[s], gt, sampled)[metric]) for s in SEEDS]
            # A cluster resample can contain only one frame class.  ROC and the
            # frozen AP protocol are undefined there; exclude that draw rather
            # than inventing a value.
            if any(a is None or b is None for a, b in pairs):
                continue
            delta = [a - b for a, b in pairs]
            draws[metric].append(float(np.mean(delta)))
    if any(not values for values in draws.values()):
        raise RuntimeError("no valid two-class video-cluster bootstrap draws")
    return {metric: {"point_difference": point[metric],
                     "ci95": [float(np.percentile(values, 2.5)),
                              float(np.percentile(values, 97.5))],
                     "n_bootstrap_requested": int(n_boot),
                     "n_bootstrap_valid": len(values), "cluster": "video",
                     "matched_seed_average": True, "rng_seed": int(rng_seed)}
            for metric, values in draws.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-expert", required=True,
                        help="pre-registered strongest expert/branch name")
    parser.add_argument("--selection-mode", choices=("shared", "per-seed"),
                        default="shared", help="per-seed is an explicit sensitivity mode")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("fresh out-dir required")
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    per_seed, seed_audit = matched_manifests(manifest)
    names = [x["name"] for x in manifest["experts"]]
    baseline = args.baseline_expert
    if baseline not in names:
        raise RuntimeError(f"unknown baseline expert {baseline}; available={names}")
    baseline_index = names.index(baseline)
    baseline_branch = {"expert": baseline,
                       "score_key": manifest["experts"][baseline_index]["score_key"],
                       "source": "explicit --baseline-expert resolved against frozen manifest"}

    frozen, validation_cache, seed_grids = {}, {}, {}
    for seed in SEEDS:
        val_raw, val_gt, provenance = load_split_exact(per_seed[seed], "val")
        calibration = fit_ecdf(val_raw)
        val = apply_ecdf(val_raw, calibration)
        model = UnifiedRelationV8(len(names), manifest.get("window", 12),
                                  manifest.get("temperature", .2)).eval()
        rows = []
        for beta in map(float, manifest["beta_grid"]):
            for gamma in map(float, manifest["gamma_grid"]):
                scores = forward(model, val, beta, gamma)
                metric = pooled(scores, val_gt)
                rows.append({"beta": beta, "gamma": gamma, **metric})
        seed_grids[seed] = rows
        frozen[str(seed)] = {"validation_grid": rows, "calibration": calibration,
                             "validation_experts": provenance}
        validation_cache[seed] = model
    shared = None
    if args.selection_mode == "shared":
        selected, fallback, eligible, mean_grid = select_shared(seed_grids)
        shared = {"selected": selected, "fallback": fallback,
                  "eligible_count": len(eligible),
                  "three_seed_validation_mean_grid": mean_grid}
        for seed in SEEDS:
            frozen[str(seed)]["selected"] = selected
    else:
        for seed in SEEDS:
            selected, fallback, eligible = select_candidate(seed_grids[seed])
            frozen[str(seed)].update({"selected": selected, "fallback": fallback,
                                      "eligible_count": len(eligible)})
    config = {"method": "relation_v8_matched_seed_evaluation",
              "corpus": manifest["corpus"], "seeds": list(SEEDS),
              "seed_protocol": seed_audit, "baseline_branch": baseline_branch,
              "selection_mode": args.selection_mode,
              "selection": ("one shared beta/gamma selected by three-seed mean validation AP, "
                            "subject to mean validation ROC nondecrease"
                            if args.selection_mode == "shared" else
                            "EXPLICIT SENSITIVITY: independent per-seed validation-only selection"),
              "shared_validation_selection": shared,
              "manifest": str(Path(args.manifest).resolve()),
              "manifest_sha256": sha256(args.manifest), "frozen": frozen,
              "test_opened_during_selection": False}
    config_path = out / "frozen_config.json"
    atomic_json(config_path, config)

    method_runs, baseline_runs, seed_results, common_gt = {}, {}, [], None
    for seed in SEEDS:
        test_raw, test_gt, provenance = load_split_exact(per_seed[seed], "test")
        if common_gt is None:
            common_gt = test_gt
        elif any(not np.array_equal(common_gt[v], test_gt[v]) for v in common_gt):
            raise RuntimeError("test GT differs across matched seeds")
        calibrated = apply_ecdf(test_raw, frozen[str(seed)]["calibration"])
        choice = frozen[str(seed)]["selected"]
        method_scores = forward(validation_cache[seed], calibrated,
                                choice["beta"], choice["gamma"])
        baseline_scores = {v: calibrated[v][:, baseline_index] for v in calibrated}
        method_runs[seed], baseline_runs[seed] = method_scores, baseline_scores
        seed_results.append({"seed": seed, "method": extended_metrics(method_scores, test_gt),
                             "baseline": extended_metrics(baseline_scores, test_gt),
                             "test_experts": provenance})
    result = {"method": config["method"], "corpus": manifest["corpus"],
              "config": str(config_path), "config_sha256": sha256(config_path),
              "baseline_branch": baseline_branch, "per_seed": seed_results,
              "method_mean_std": summarize_seed_metrics([x["method"] for x in seed_results]),
              "baseline_mean_std": summarize_seed_metrics([x["baseline"] for x in seed_results]),
              "method_minus_baseline_video_cluster_bootstrap": cluster_bootstrap(
                  method_runs, baseline_runs, common_gt, args.bootstrap, args.bootstrap_seed),
              "test_labels_used_for_training_or_selection": False}
    atomic_json(out / "fair_eval.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
