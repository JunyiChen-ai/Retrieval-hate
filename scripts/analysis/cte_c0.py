#!/usr/bin/env python
"""CTE C0: freeze, vectorized full-bank audit, and independent decision."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from cte_common import (  # noqa: E402
    DATASETS, SUPERVISION, acquire_lock, add_payload_hash, atomic_write_json,
    base_manifest, build_model, canonical_json, choose_anchors_and_support,
    fold_path, git_head, gpu_metadata, implementation_hashes, input_records,
    interval_cost, load_config, load_fold_contract, load_segment_cache,
    load_train_cache, normalize_checked, output_records, read_json, require_slurm,
    resolve, select_rows, set_seed, sha256_file, sha256_obj, state_dict_sha256,
    subclip_cache_path, tangent_values, train_cache_path, validate_manifest_common,
    vectorized_margin, verify_config_freeze, verify_payload,
)


C0_RUNS = {
    "MHC": "CTE-C0-MICRO-MHC-S0-v1",
    "MHC_zh": "CTE-C0-MICRO-MHC_zh-S0-v1",
}


def fixture_paths(cfg, dataset):
    root = resolve(cfg, "ssr_artifacts") / "oof" / dataset
    paths = []
    for fold in range(5):
        manifest_path = root / f"fold{fold}" / "manifest.json"
        manifest = read_json(manifest_path)
        checkpoint_names = [name for name in manifest.get("outputs", {})
                            if name.startswith("checkpoint_epoch") and name.endswith(".pt")]
        if len(checkpoint_names) != 1:
            raise RuntimeError(f"{dataset}/fold{fold}: unique checkpoint missing")
        checkpoint = manifest_path.parent / checkpoint_names[0]
        if sha256_file(checkpoint) != manifest["outputs"][checkpoint_names[0]]:
            raise RuntimeError("SSR checkpoint hash mismatch")
        if manifest.get("only_gold_supervision") != "video_level_binary_label" or \
                manifest.get("segment_gold_exists") is not False:
            raise RuntimeError("SSR fixture supervision contract mismatch")
        paths.append((manifest_path, checkpoint))
    return paths


def scalar_margin(query, label, query_id, bank, bank_labels, bank_ids, tau):
    values_same, values_other = [], []
    q = query.double() / torch.linalg.vector_norm(query.double())
    for index, bank_id in enumerate(bank_ids):
        if bank_id == query_id:
            continue
        k = bank[index].double() / torch.linalg.vector_norm(bank[index].double())
        value = float(torch.dot(q, k) / float(tau))
        if int(bank_labels[index]) == int(label):
            values_same.append(value)
        else:
            values_other.append(value)
    if not values_same or not values_other:
        raise RuntimeError("scalar margin missing a class")
    same = torch.logsumexp(torch.tensor(values_same, dtype=torch.float64), dim=0)
    other = torch.logsumexp(torch.tensor(values_other, dtype=torch.float64), dim=0)
    return float(float(tau) * (same - other))


def synthetic_numerics(cfg):
    set_seed(20260710)
    device = "cuda"
    n, d = 17, 31
    bank = F.normalize(torch.randn(n, d, device=device), dim=1)
    query = F.normalize(torch.randn(9, d, device=device), dim=1)
    bank_labels = torch.as_tensor([0, 1, 0, 1, 0, 1, 1, 0, 1,
                                   0, 0, 1, 0, 1, 1, 0, 1], device=device)
    query_labels = torch.as_tensor([0, 1, 0, 1, 1, 0, 1, 0, 1], device=device)
    bank_ids = [f"bank-{3 + 7 * index}" for index in range(n)]
    query_ids = [bank_ids[2], "query-101", bank_ids[7], "query-305",
                 bank_ids[4], "query-509", bank_ids[12], "query-701", bank_ids[15]]
    rows = []
    margin_error = 0.0
    for tau in cfg["cte"]["tau_grid"]:
        vector = vectorized_margin(query, query_labels, query_ids, bank,
                                   bank_labels, bank_ids, tau)
        scalar = [scalar_margin(query[row], query_labels[row], query_ids[row],
                                bank, bank_labels, bank_ids, tau)
                  for row in range(len(query_ids))]
        errors = [abs(float(vector[row]) - scalar[row]) for row in range(len(scalar))]
        margin_error = max(margin_error, max(errors))
        rows.append({"tau": float(tau), "vector": [float(x) for x in vector.cpu()],
                     "scalar_double": scalar, "absolute_errors": errors})

    # The same class-wise constant shift must cancel in the LSE difference.
    logits = torch.randn(13, device=device, dtype=torch.float64)
    mask = torch.as_tensor([index % 2 == 0 for index in range(13)], device=device)
    base = torch.logsumexp(logits[mask], 0) - torch.logsumexp(logits[~mask], 0)
    shift_errors = {}
    for shift in (-100.0, 100.0):
        moved = torch.logsumexp((logits + shift)[mask], 0) - \
            torch.logsumexp((logits + shift)[~mask], 0)
        shift_errors[str(int(shift))] = abs(float(moved - base))

    # Double-precision directional finite difference of the exact vectorized kernel.
    q = F.normalize(torch.randn(6, 11, device=device, dtype=torch.float64), dim=1)
    k = F.normalize(torch.randn(14, 11, device=device, dtype=torch.float64), dim=1)
    q.requires_grad_(True)
    qlabels = torch.as_tensor([0, 1, 0, 1, 0, 1], device=device)
    klabels = torch.as_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              device=device)
    qids = ["q-2", "q-9", "q-25", "q-44", "q-80", "q-121"]
    kids = ["k-1", "q-9", "k-8", "k-12", "q-44", "k-33", "k-41",
            "k-55", "q-2", "k-89", "k-144", "k-233", "k-377", "k-610"]
    objective = vectorized_margin(q, qlabels, qids, k, klabels, kids, 0.1).mean()
    grad = torch.autograd.grad(objective, q)[0]
    direction = F.normalize(torch.randn_like(q), dim=1)
    analytic = float((grad * direction).sum())
    epsilon = 1e-5
    with torch.no_grad():
        plus = vectorized_margin(q + epsilon * direction, qlabels, qids,
                                 k, klabels, kids, 0.1).mean()
        minus = vectorized_margin(q - epsilon * direction, qlabels, qids,
                                  k, klabels, kids, 0.1).mean()
    finite_difference = float((plus - minus) / (2 * epsilon))
    gradient_relative_error = abs(analytic - finite_difference) / max(
        abs(analytic), abs(finite_difference), 1e-12)
    return {"cases": rows, "max_absolute_margin_error": margin_error,
            "stable_lse_shift_errors": shift_errors,
            "autograd_all_finite": bool(torch.isfinite(grad).all()),
            "analytic_directional_derivative": analytic,
            "finite_difference_directional_derivative": finite_difference,
            "relative_gradient_error": gradient_relative_error}


def actual_numerics(cfg, model, encoded, support, subset):
    pair = support["selected_pair"]
    if pair is None:
        return {"skipped": True, "reason": "no_supported_adjacent_pair",
                "max_absolute_margin_error": None,
                "max_absolute_t_error": None,
                "max_absolute_cost_error": None}
    take = min(32, len(subset["ids"]))
    ids, labels = subset["ids"][:take], subset["labels"][:take].cuda()
    full = encoded["z"][:take]
    from cte_common import encode_parts
    max_margin, max_t, max_cost, max_gradient = 0.0, 0.0, 0.0, 0.0
    cases = []
    for tau in cfg["cte"]["tau_grid"]:
        vector_full = vectorized_margin(full, labels, ids, encoded["z"],
                                        subset["labels"].cuda(), subset["ids"], tau)
        scalar_full = [scalar_margin(full[row], labels[row], ids[row], encoded["z"],
                                     subset["labels"].cuda(), subset["ids"], tau)
                       for row in range(take)]
        full_error = max(abs(float(vector_full[row]) - scalar_full[row])
                         for row in range(take))
        max_margin = max(max_margin, full_error)
        for modality in ("V", "L"):
            for radius in cfg["cte"]["radii"]:
                path = encode_parts(
                    model=model, img=subset["img"][:take].cuda(),
                    txt=subset["txt"][:take].cuda(), cfg=cfg,
                    anchor_parts=support["anchors"], radius=float(radius),
                    modality=modality)["z"]
                support_mask = support["masks"][f"{modality}:{float(radius):.2f}"][:take]
                for smin in cfg["cte"]["smin_grid"]:
                    tangent, mad, _, vector_path = tangent_values(
                        full, path, labels, ids, encoded["z"], subset["labels"].cuda(),
                        subset["ids"], tau, radius, smin, support=support_mask)
                    scalar_path = [scalar_margin(
                        path[row], labels[row], ids[row], encoded["z"],
                        subset["labels"].cuda(), subset["ids"], tau)
                        for row in range(take)]
                    scalar_t = [math.tanh((scalar_path[row] - scalar_full[row]) /
                                          (radius * max(mad, smin) + 1e-6))
                                for row in range(take)]
                    lower = torch.full_like(tangent, -0.5)
                    upper = torch.full_like(tangent, -0.2)
                    vector_cost = interval_cost(tangent, lower, upper)
                    scalar_cost = [max(-0.5 - value, 0.0, value + 0.2) ** 2 / 4.0
                                   for value in scalar_t]
                    t_errors = [abs(float(tangent[row]) - scalar_t[row])
                                for row in range(take)]
                    cost_errors = [abs(float(vector_cost[row]) - scalar_cost[row])
                                   for row in range(take)]
                    path_error = max(abs(float(vector_path[row]) - scalar_path[row])
                                     for row in range(take))

                    # Directional derivative of the complete T -> interval-cost kernel.
                    q_path = path.detach().double().requires_grad_(True)
                    q_full = full.detach().double()
                    bank_double = encoded["z"].detach().double()
                    tangent_double, _, _, _ = tangent_values(
                        q_full, q_path, labels, ids, bank_double,
                        subset["labels"].cuda(), subset["ids"], tau, radius,
                        smin, mad=mad)
                    objective = interval_cost(
                        tangent_double, torch.full_like(tangent_double, -0.5),
                        torch.full_like(tangent_double, -0.2)).mean()
                    gradient = torch.autograd.grad(objective, q_path)[0]
                    direction = F.normalize(torch.randn_like(q_path), dim=1)
                    analytic = float((gradient * direction).sum())
                    epsilon = 1e-5
                    with torch.no_grad():
                        plus_t, _, _, _ = tangent_values(
                            q_full, q_path + epsilon * direction, labels, ids,
                            bank_double, subset["labels"].cuda(), subset["ids"],
                            tau, radius, smin, mad=mad)
                        minus_t, _, _, _ = tangent_values(
                            q_full, q_path - epsilon * direction, labels, ids,
                            bank_double, subset["labels"].cuda(), subset["ids"],
                            tau, radius, smin, mad=mad)
                        plus = interval_cost(
                            plus_t, torch.full_like(plus_t, -0.5),
                            torch.full_like(plus_t, -0.2)).mean()
                        minus = interval_cost(
                            minus_t, torch.full_like(minus_t, -0.5),
                            torch.full_like(minus_t, -0.2)).mean()
                    finite_difference = float((plus - minus) / (2 * epsilon))
                    relative = abs(analytic - finite_difference) / max(
                        abs(analytic), abs(finite_difference), 1e-12)
                    max_margin = max(max_margin, path_error)
                    max_t, max_cost = max(max_t, max(t_errors)), max(max_cost, max(cost_errors))
                    max_gradient = max(max_gradient, relative)
                    cases.append({"tau": float(tau), "smin": float(smin),
                                  "modality": modality, "radius": float(radius),
                                  "mad": mad, "max_margin_full_error": full_error,
                                  "max_margin_path_error": path_error,
                                  "max_t_error": max(t_errors),
                                  "max_cost_error": max(cost_errors),
                                  "gradient_all_finite": bool(torch.isfinite(gradient).all()),
                                  "analytic_directional_derivative": analytic,
                                  "finite_difference_directional_derivative": finite_difference,
                                  "relative_gradient_error": relative})
    return {"skipped": False, "max_absolute_margin_error": max_margin,
            "max_absolute_t_error": max_t, "max_absolute_cost_error": max_cost,
            "max_relative_gradient_error": max_gradient,
            "all_gradients_finite": all(item["gradient_all_finite"] for item in cases),
            "cases": cases}


def benchmark(cfg, model, subset, support):
    pair = support["selected_pair"]
    if pair is None:
        return {"oom": False, "skipped": True, "reason": "no_supported_pair",
                "median_ms": None, "p95_ms": None,
                "peak_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2**30}
    from cte_common import encode_parts
    batch = min(cfg["model"]["cte_microbatch_size"], len(subset["ids"]))
    img, txt = subset["img"][:batch].cuda(), subset["txt"][:batch].cuda()
    labels, ids = subset["labels"][:batch].cuda(), subset["ids"][:batch]
    bank = support["encoded"]["z"]
    bank_labels = subset["labels"].cuda()
    warmup = int(cfg["cte"]["c0_warmup_iterations"])
    timed = int(cfg["cte"]["c0_timed_iterations"])
    torch.cuda.reset_peak_memory_stats()

    def one():
        model.zero_grad(set_to_none=True)
        full = encode_parts(model, img, txt, cfg, grad=True)["z"]
        losses = []
        for modality in ("V", "L"):
            path = encode_parts(model, img, txt, cfg, support["anchors"],
                                pair["a1"], modality, grad=True)["z"]
            tangent, _, _, _ = tangent_values(
                full, path, labels, ids, bank, bank_labels, subset["ids"],
                tau=0.1, radius=pair["a1"], smin=0.05, mad=0.05)
            losses.append(interval_cost(tangent,
                                        torch.full_like(tangent, -0.5),
                                        torch.full_like(tangent, -0.2)).mean())
        sum(losses).backward()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(timed):
        start = time.perf_counter()
        one()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    steps_per_epoch = math.ceil(len(subset["ids"]) / cfg["model"]["batch_size"])
    epochs = cfg["datasets"][benchmark.dataset]["epoch_index"] + 1
    estimate = (np.median(elapsed) / 1000.0) * 2 * steps_per_epoch * epochs * 9 * 3 / 3600.0
    return {"oom": False, "skipped": False, "batch_size": batch,
            "warmup_iterations": warmup, "timed_iterations": timed,
            "median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.quantile(elapsed, 0.95, method="linear")),
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated()) / 2**30,
            "estimated_c1_gpu_hours": float(estimate),
            "estimate_definition": "median_kernel_x_2microbatches_x_steps_x_epochs_x_9arms_x_3inner_plus_outer"}


def phase_sanity(cfg):
    require_slurm(gpu=False)
    for dataset in DATASETS:
        bundle = load_train_cache(cfg, dataset)
        subset = select_rows(bundle, sorted(bundle["ids"]))
        _, _, audit = load_segment_cache(cfg, dataset, bundle, subset)
        if audit["label_source"] != "inherited_parent_video_label_not_segment_gold":
            raise RuntimeError("K4 inheritance declaration mismatch")
        fixture_paths(cfg, dataset)
    impl, digest = implementation_hashes()
    print(canonical_json({"status": "SANITY_GO", "implementation_sha256": digest,
                          "implementation_files": impl, **SUPERVISION}))


def phase_freeze(cfg, run_id):
    require_slurm(gpu=False)
    if run_id != "CTE-CONFIG-FREEZE-v1":
        raise RuntimeError("freeze run ID mismatch")
    artifact_root = resolve(cfg, "artifact_root")
    review_path = ROOT / "refine-logs/cte/CTE_C0_C1_CODE_REVIEW.md"
    if not review_path.exists():
        raise RuntimeError("independent code review record missing")
    text = review_path.read_text(encoding="utf-8")
    if "HIGH: 0" not in text or "CRITICAL: 0" not in text:
        raise RuntimeError("independent review has unresolved HIGH/CRITICAL")
    sources = [ROOT / "configs/cte/cte_v1.json", review_path,
               ROOT / "refine-logs/cte/EXPERIMENT_PLAN.md",
               ROOT / "refine-logs/cte/FINAL_PROPOSAL.md"]
    fold_hashes = {}
    for dataset in DATASETS:
        bundle = load_train_cache(cfg, dataset)
        subset = select_rows(bundle, sorted(bundle["ids"]))
        seg_path, _, _ = load_segment_cache(cfg, dataset, bundle, subset)
        paths = fixture_paths(cfg, dataset)
        sources.extend([bundle["fold_path"], bundle["cache_path"], seg_path])
        for manifest, checkpoint in paths:
            sources.extend([manifest, checkpoint])
        fold_hashes[dataset] = sha256_obj(sorted(
            (vid, row["label"], row["fold"]) for vid, row in bundle["by_id"].items()))
    impl, impl_sha = implementation_hashes()
    acquire_lock(artifact_root / ".CONFIG_FREEZE.lock", run_id, "M0_CONFIG_FREEZE")
    meta = gpu_metadata()
    payload = add_payload_hash({
        "schema_version": 1, "run_id": run_id, "stage": "M0_CONFIG_FREEZE",
        "status": "GO", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_head": git_head(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": meta["cuda_version"], "gpu_name": meta["gpu_name"],
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": impl_sha, "implementation_files": impl,
        "input_files": input_records(sorted(set(sources))),
        "fold_ids_sha256": sha256_obj(fold_hashes), "checkpoint_sha256": None,
        "output_files": [], "independent_review_sha256": sha256_file(review_path),
        "C2_C3_C4_locked": True, **SUPERVISION,
    })
    atomic_write_json(artifact_root / "CONFIG_FREEZE.json", payload)
    print(canonical_json(payload))


def phase_micro(cfg, dataset, run_id):
    require_slurm(gpu=True)
    if run_id != C0_RUNS[dataset]:
        raise RuntimeError("C0 micro run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    out_dir = resolve(cfg, "artifact_root") / "c0" / dataset
    acquire_lock(out_dir / ".micro.lock", run_id, "C0_MICRO")
    set_seed(cfg["cte"]["seed"])
    bundle = load_train_cache(cfg, dataset)
    input_paths = [freeze_path, bundle["fold_path"], bundle["cache_path"]]
    synthetic = synthetic_numerics(cfg)
    support_rows, actual_rows = [], []
    checkpoint_hashes = {}
    bench = None
    minimum_norm = float("inf")
    for outer in range(5):
        train_ids = sorted(vid for vid, row in bundle["by_id"].items()
                           if row["fold"] != outer)
        subset = select_rows(bundle, train_ids)
        manifest_path, checkpoint_path = fixture_paths(cfg, dataset)[outer]
        input_paths.extend([manifest_path, checkpoint_path])
        model = build_model(cfg, dataset, subset["img"].shape[1],
                            subset["txt"].shape[1]).cuda()
        state = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(state, strict=True)
        checkpoint_hashes[str(outer)] = sha256_file(checkpoint_path)
        audit = choose_anchors_and_support(model, subset, cfg)
        minimum_norm = min(minimum_norm, audit["minimum_pre_normalization_norm"])
        support_rows.append({"outer_fold": outer,
                             "anchor_ids": audit["anchor_ids"],
                             "thresholds": audit["thresholds"],
                             "details": audit["details"],
                             "adjacent_candidates": audit["candidates"],
                             "selected_pair": audit["selected_pair"],
                             "minimum_pre_normalization_norm": audit["minimum_pre_normalization_norm"]})
        actual = actual_numerics(cfg, model, audit["encoded"], audit, subset)
        actual["outer_fold"] = outer
        actual_rows.append(actual)
        if outer == 0:
            benchmark.dataset = dataset
            bench = benchmark(cfg, model, subset, audit)
        del model, state, audit
        torch.cuda.empty_cache()
    numerical = {
        "schema_version": 1, "run_id": run_id, "dataset": dataset,
        "synthetic": synthetic, "actual_folds": actual_rows,
        "minimum_pre_normalization_norm": minimum_norm,
        "all_finite": bool(math.isfinite(minimum_norm) and
                           synthetic["autograd_all_finite"] and
                           all(item["max_absolute_margin_error"] is not None and
                               math.isfinite(item["max_absolute_margin_error"])
                               and item.get("all_gradients_finite") is True
                               for item in actual_rows)),
    }
    support_payload = {"schema_version": 1, "run_id": run_id,
                       "dataset": dataset, "folds": support_rows}
    micro_payload = {"schema_version": 1, "run_id": run_id,
                     "dataset": dataset, **bench}
    numerics_path, support_path, micro_path = (
        out_dir / "numerics.json", out_dir / "support.json",
        out_dir / "microbenchmark.json")
    atomic_write_json(numerics_path, numerical)
    atomic_write_json(support_path, support_payload)
    atomic_write_json(micro_path, micro_payload)
    fold_sha = sha256_obj({dataset: sorted(
        (vid, row["label"], row["fold"]) for vid, row in bundle["by_id"].items())})
    manifest = base_manifest(
        cfg, run_id, "C0_MICRO", "COMPLETED", input_paths,
        [numerics_path, support_path, micro_path], fold_sha,
        checkpoint_sha256=sha256_obj(checkpoint_hashes),
        extra={"dataset": dataset, "outer_folds": list(range(5)),
               "checkpoint_hashes": checkpoint_hashes,
               "exact_params": cfg["cte"], "C1_started": False,
               "C2_C3_C4_locked": True})
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": run_id,
                          "support_pairs": [row["selected_pair"] for row in support_rows],
                          "microbenchmark": bench}))


def phase_decide(cfg, run_id):
    require_slurm(gpu=False)
    if run_id != "CTE-C0-DECISION-v1":
        raise RuntimeError("C0 decision run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    root = resolve(cfg, "artifact_root")
    cells, inputs = [], [freeze_path]
    all_pass = True
    for dataset in DATASETS:
        directory = root / "c0" / dataset
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path)
        expected_fold_sha = sha256_obj({dataset: sorted(
            (str(row["id"]), int(row["label"]), int(row["fold"]))
            for row in read_json(fold_path(cfg, dataset))["records"])})
        expected_checkpoint_sha = sha256_obj({str(index): sha256_file(checkpoint)
                                              for index, (_, checkpoint) in enumerate(
                                                  fixture_paths(cfg, dataset))})
        required_outputs = [directory / "numerics.json", directory / "support.json",
                            directory / "microbenchmark.json"]
        validate_manifest_common(manifest, C0_RUNS[dataset], "C0_MICRO", cfg,
                                 expected_fold_sha, expected_checkpoint_sha,
                                 required_outputs,
                                 [freeze_path, fold_path(cfg, dataset),
                                  train_cache_path(cfg, dataset)])
        numerics_path, support_path, micro_path = (
            directory / "numerics.json", directory / "support.json",
            directory / "microbenchmark.json")
        numerical, support, micro = map(read_json,
                                        (numerics_path, support_path, micro_path))
        inputs.extend([manifest_path, numerics_path, support_path, micro_path])
        syn = numerical["synthetic"]
        expected_case_keys = {(float(tau), modality, float(radius), float(smin))
                              for tau in cfg["cte"]["tau_grid"]
                              for modality in ("V", "L")
                              for radius in cfg["cte"]["radii"]
                              for smin in cfg["cte"]["smin_grid"]}
        actual_folds = numerical.get("actual_folds", [])
        fold_indices = [item.get("outer_fold") for item in actual_folds]
        actual_complete = len(actual_folds) == 5 and set(fold_indices) == set(range(5)) and \
            len(set(fold_indices)) == 5
        margin_values = []
        t_values, cost_values, gradient_values = [], [], []
        all_case_finite = True
        for fold_item in actual_folds:
            cases = fold_item.get("cases", [])
            try:
                observed_keys = [(float(item.get("tau")), item.get("modality"),
                                  float(item.get("radius")), float(item.get("smin")))
                                 for item in cases]
            except (TypeError, ValueError):
                actual_complete = False
                continue
            if len(cases) != len(expected_case_keys) or \
                    set(observed_keys) != expected_case_keys or \
                    len(set(observed_keys)) != len(expected_case_keys):
                actual_complete = False
                continue
            for item in cases:
                values = [item.get("max_margin_full_error"),
                          item.get("max_margin_path_error"), item.get("max_t_error"),
                          item.get("max_cost_error"), item.get("relative_gradient_error")]
                if any(value is None or not math.isfinite(float(value)) for value in values) or \
                        item.get("gradient_all_finite") is not True:
                    all_case_finite = False
                    continue
                margin_values.extend([float(values[0]), float(values[1])])
                t_values.append(float(values[2]))
                cost_values.append(float(values[3]))
                gradient_values.append(float(values[4]))
        synthetic_cases = syn.get("cases", [])
        try:
            synthetic_taus = {float(item.get("tau")) for item in synthetic_cases}
        except (TypeError, ValueError):
            synthetic_taus = set()
        synthetic_complete = (len(synthetic_cases) == len(cfg["cte"]["tau_grid"]) and
                              synthetic_taus ==
                              {float(value) for value in cfg["cte"]["tau_grid"]})
        for item in synthetic_cases:
            margin_values.extend(float(value) for value in item.get("absolute_errors", []))
        actual_complete = actual_complete and synthetic_complete and all_case_finite and \
            bool(margin_values) and bool(t_values) and bool(cost_values) and bool(gradient_values)
        margin_max = max(margin_values) if actual_complete else None
        t_max = max(t_values) if actual_complete else None
        cost_max = max(cost_values) if actual_complete else None
        gradient_max = max([float(syn.get("relative_gradient_error", float("inf")))] +
                           gradient_values) if actual_complete else None
        support_folds = support.get("folds", [])
        expected_support_keys = {(modality, float(radius)) for modality in ("V", "L")
                                 for radius in cfg["cte"]["radii"]}
        support_pass = len(support_folds) == 5 and \
            {item.get("outer_fold") for item in support_folds} == set(range(5))
        for fold_item in support_folds:
            try:
                detail_keys = [(item.get("modality"), float(item.get("radius")))
                               for item in fold_item.get("details", [])]
            except (TypeError, ValueError):
                support_pass = False
                continue
            support_pass = support_pass and len(detail_keys) == len(expected_support_keys) and \
                set(detail_keys) == expected_support_keys and \
                len(set(detail_keys)) == len(expected_support_keys) and \
                fold_item.get("selected_pair") is not None
        minimum_norm = min((float(item.get("minimum_pre_normalization_norm", 0.0))
                            for item in support_folds), default=0.0)
        gates = {
            "margin_error": actual_complete and margin_max <= cfg["cte"]["c0_margin_error_max"],
            "t_error": actual_complete and t_max <= cfg["cte"]["c0_t_cost_error_max"],
            "cost_error": actual_complete and cost_max <= cfg["cte"]["c0_t_cost_error_max"],
            "gradient_error": actual_complete and gradient_max <=
                cfg["cte"]["c0_relative_gradient_error_max"],
            "complete_case_cardinality": actual_complete,
            "finite": all_case_finite and syn.get("autograd_all_finite") is True,
            "minimum_norm": minimum_norm >=
                cfg["cte"]["minimum_pre_normalization_norm"],
            "stable_shift": max(syn["stable_lse_shift_errors"].values()) <= 1e-10,
            "all_folds_supported": support_pass,
            "batch32_no_oom": micro.get("oom") is False and micro.get("skipped") is False,
            "peak_memory": float(micro["peak_allocated_gib"]) <=
                cfg["cte"]["c0_peak_memory_gib_max"],
            "iteration_count": micro.get("warmup_iterations") ==
                cfg["cte"]["c0_warmup_iterations"] and
                micro.get("timed_iterations") == cfg["cte"]["c0_timed_iterations"],
        }
        passed = all(gates.values())
        all_pass = all_pass and passed
        cells.append({"dataset": dataset, "gates": gates, "passed": passed,
                      "observed": {"max_margin_error": margin_max,
                                   "max_t_error": t_max, "max_cost_error": cost_max,
                                   "relative_gradient_error": gradient_max,
                                   "minimum_norm": minimum_norm,
                                   "median_ms": micro.get("median_ms"),
                                   "p95_ms": micro.get("p95_ms"),
                                   "peak_memory_gib": micro.get("peak_allocated_gib"),
                                   "estimated_c1_gpu_hours": micro.get("estimated_c1_gpu_hours")}})
    decision = "GO" if all_pass else "STOP"
    impl, impl_sha = implementation_hashes()
    meta = gpu_metadata()
    fold_hashes = {dataset: read_json(fold_path(cfg, dataset))["id_to_fold_sha256"]
                   for dataset in DATASETS}
    payload = add_payload_hash({
        "schema_version": 1, "run_id": run_id, "stage": "C0_DECISION",
        "status": decision, "C0_DECISION": decision,
        "reason": "all_C0_numerics_support_resource_gates_pass" if all_pass
                  else "one_or_more_C0_gates_failed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_head": git_head(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": meta["cuda_version"], "gpu_name": meta["gpu_name"],
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": impl_sha, "implementation_files": impl,
        "input_files": input_records(inputs),
        "fold_ids_sha256": sha256_obj(fold_hashes), "checkpoint_sha256": None,
        "output_files": [], "cells": cells,
        "C1_unlocked": bool(all_pass), "C2_C3_C4_locked": True,
        "interpretation": "implementation_and_supported-path_feasibility_only_not_MLLM_evidence",
        **SUPERVISION,
    })
    acquire_lock(root / ".C0_DECISION.lock", run_id, "C0_DECISION")
    atomic_write_json(root / "C0_DECISION.json", payload)
    print(canonical_json(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True,
                        choices=("sanity", "freeze", "micro", "decide"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.phase == "sanity":
        if args.dataset or args.run_id:
            raise RuntimeError("sanity takes no dataset/run-id")
        phase_sanity(cfg)
    elif args.phase == "freeze":
        if args.dataset or not args.run_id:
            raise RuntimeError("freeze requires only run-id")
        phase_freeze(cfg, args.run_id)
    elif args.phase == "micro":
        if not args.dataset or not args.run_id:
            raise RuntimeError("micro requires dataset and run-id")
        phase_micro(cfg, args.dataset, args.run_id)
    else:
        if args.dataset or not args.run_id:
            raise RuntimeError("decide requires only run-id")
        phase_decide(cfg, args.run_id)


if __name__ == "__main__":
    main()
