#!/usr/bin/env python
"""CTE C1: strict nested train-OOF zero-teacher action-capacity screen.

Only parent-video binary labels are consumed.  The code never opens validation
or test content and has no MLLM/OCR/teacher-cache path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from data_loader.rac_dataloader import RACDataset  # noqa: E402
from model.loss import compute_loss  # noqa: E402
from cte_common import (  # noqa: E402
    ARMS, DATASETS, SUPERVISION, acquire_lock, add_payload_hash,
    atomic_torch_save, atomic_write_json, atomic_write_jsonl, base_manifest,
    binary_metrics, build_model, canonical_json, choose_anchors_and_support,
    clone_state, deterministic_derangement, encode_dataset, encode_parts,
    fold_path, git_head, gpu_metadata, implementation_hashes, input_records,
    interval_cost, jaccard_churn, load_config, load_segment_cache,
    load_train_cache, ordinary_knn, percentile_linear, read_json, read_jsonl,
    require_slurm, resolve, robust_mad, select_rows, set_seed, sha256_file,
    sha256_obj, state_dict_sha256, tangent_values, train_args,
    validate_manifest_common, verify_config_freeze, verify_payload,
)


def tuple_grid(cfg):
    return [{"tau": float(tau), "lambda": float(lam), "smin": float(smin)}
            for tau, lam, smin in product(cfg["cte"]["tau_grid"],
                                         cfg["cte"]["lambda_grid"],
                                         cfg["cte"]["smin_grid"])]


def tuple_key(item):
    return f"tau={item['tau']:.2f},lambda={item['lambda']:.2f},smin={item['smin']:.2f}"


def expected_inner_run(dataset, fold):
    return f"CTE-C1-INNER-{dataset}-F{fold}-S0-v1"


def expected_outer_run(dataset, fold):
    return f"CTE-C1-OUTER-{dataset}-F{fold}-S0-v1"


def require_c0_go(cfg):
    path = resolve(cfg, "artifact_root") / "C0_DECISION.json"
    payload = read_json(path)
    verify_payload(payload)
    if payload.get("C0_DECISION") != "GO" or payload.get("C1_unlocked") is not True:
        raise RuntimeError("C1 remains locked because C0 is not verified GO")
    if payload.get("config_canonical_sha256") != cfg["computed_config_sha256"]:
        raise RuntimeError("C0 decision config provenance mismatch")
    _, current_impl = implementation_hashes()
    if payload.get("implementation_sha256") != current_impl:
        raise RuntimeError("C0 decision implementation provenance mismatch")
    for key, value in SUPERVISION.items():
        if payload.get(key) != value:
            raise RuntimeError("C0 zero-call/supervision declaration mismatch")
    for item in payload.get("input_files", []):
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("C0 decision input changed")
    return path, payload


def subset_dataset(subset):
    return RACDataset((subset["img"], subset["txt"]), subset["ids"], subset["labels"])


def make_loader(subset, batch_size, seed, shuffle=True):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(subset_dataset(subset), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, generator=generator)


def baseline_train(cfg, dataset, subset, segment_cache, seed):
    """Create a train-only baseline initialization without touching held-out IDs."""
    set_seed(seed)
    model = build_model(cfg, dataset, subset["img"].shape[1], subset["txt"].shape[1])
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"]["adamw_weight_decay"],
        betas=tuple(cfg["model"]["adamw_betas"]), eps=cfg["model"]["adamw_epsilon"],
        amsgrad=cfg["model"]["adamw_amsgrad"])
    train_dl = make_loader(subset, cfg["model"]["batch_size"], seed, True)
    train_set = train_dl.dataset
    args = train_args(cfg, dataset)
    train_feats = train_labels = None
    history = []
    epochs = int(cfg["datasets"][dataset]["epoch_index"]) + 1
    for epoch in range(epochs):
        losses, ids_seen = [], []
        for batch in train_dl:
            ids_seen.extend(str(item) for item in batch["ids"])
            output = compute_loss(
                batch, train_dl, model, args, train_set=train_set,
                sparse_retrieval_dictionary=None, train_feats=train_feats,
                train_labels=train_labels, segment_cache=segment_cache,
                aux_pack=None, cf_pack=None)
            loss, train_feats, train_labels = output[0], output[-2], output[-1]
            if torch.is_tensor(train_feats):
                train_feats = train_feats.detach()
            if torch.is_tensor(train_labels):
                train_labels = train_labels.detach()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite baseline initialization loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if sorted(ids_seen) != sorted(subset["ids"]) or len(ids_seen) != len(subset["ids"]):
            raise RuntimeError("baseline epoch did not cover each train video exactly once")
        history.append({"epoch_index": epoch, "mean_base_loss": float(np.mean(losses)),
                        "optimizer_steps": len(losses),
                        "query_ids_sha256": sha256_obj(ids_seen)})
    state = clone_state(model.state_dict())
    return state, {"random_initialization_sha256": initial_hash,
                   "baseline_checkpoint_sha256": state_dict_sha256(state),
                   "epochs": epochs, "history": history}


def ssr_initialization(cfg, dataset, outer_fold, model):
    directory = resolve(cfg, "ssr_artifacts") / "oof" / dataset / f"fold{outer_fold}"
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    names = [name for name in manifest["outputs"] if name.startswith("checkpoint_epoch")]
    if len(names) != 1:
        raise RuntimeError("unique SSR outer initialization missing")
    checkpoint_path = directory / names[0]
    if sha256_file(checkpoint_path) != manifest["outputs"][names[0]]:
        raise RuntimeError("SSR initialization hash mismatch")
    if manifest.get("outer_fold") != outer_fold or manifest.get("dataset") != dataset:
        raise RuntimeError("SSR initialization fold mismatch")
    if manifest.get("only_gold_supervision") != "video_level_binary_label" or \
            manifest.get("segment_gold_exists") is not False:
        raise RuntimeError("SSR initialization supervision mismatch")
    state = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state, strict=True)
    return clone_state(model.state_dict()), checkpoint_path, manifest_path


def stratified_three(ids, labels, seed):
    ids = list(ids)
    labels = np.asarray(labels, dtype=np.int64)
    order = sorted(range(len(ids)), key=lambda idx: ids[idx].encode("utf-8"))
    ordered_y = labels[order]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    folds = []
    dummy = np.zeros((len(order), 1), dtype=np.float32)
    for _, held in splitter.split(dummy, ordered_y):
        folds.append([order[index] for index in held.tolist()])
    flattened = [item for fold in folds for item in fold]
    if sorted(flattened) != list(range(len(ids))) or len(set(flattened)) != len(ids):
        raise RuntimeError("stratified three-way partition invalid")
    return folds


def fit_probe(x_a, y_a, x_b, y_b, cfg):
    candidates = []
    for value in cfg["cte"]["probe_C_grid"]:
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                C=float(value), penalty="l2", solver=cfg["cte"]["probe_solver"],
                max_iter=cfg["cte"]["probe_max_iter"],
                tol=cfg["cte"]["probe_tol"], random_state=0)),
        ])
        pipeline.fit(x_a, y_a)
        probability = pipeline.predict_proba(x_b)[:, 1]
        score = float(log_loss(y_b, probability, labels=[0, 1]))
        candidates.append((score, float(value)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0], candidates


def strict_probe_targets(cfg, model, subset, support, seed, namespace):
    """Three rotations: C never participates in fitting or C selection."""
    pair = support.get("selected_pair")
    if pair is None:
        raise RuntimeError("probe target requires a supported adjacent radius pair")
    encoded = support["encoded"]
    path_g = {}
    for modality in ("V", "L"):
        path_g[modality] = encode_parts(
            model, subset["img"].cuda(), subset["txt"].cuda(), cfg,
            support["anchors"], pair["a1"], modality)["g"].detach().cpu().numpy()
    full_g = encoded["g"].detach().cpu().numpy()
    labels = subset["labels"].cpu().numpy().astype(np.int64)
    folds = stratified_three(subset["ids"], labels, seed)
    full_score = np.zeros(len(subset["ids"]), dtype=np.float64)
    path_scores = {modality: np.zeros(len(subset["ids"]), dtype=np.float64)
                   for modality in ("V", "L")}
    selected_by_row, rotation_records = {}, []
    for c_index in range(3):
        a_index, b_index = (c_index + 1) % 3, (c_index + 2) % 3
        a, b, c = folds[a_index], folds[b_index], folds[c_index]
        selected, candidates = fit_probe(full_g[a], labels[a], full_g[b], labels[b], cfg)
        chosen_c = selected[1]
        refit = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                C=chosen_c, penalty="l2", solver=cfg["cte"]["probe_solver"],
                max_iter=cfg["cte"]["probe_max_iter"],
                tol=cfg["cte"]["probe_tol"], random_state=0)),
        ])
        train = a + b
        refit.fit(full_g[train], labels[train])
        full_score[c] = refit.decision_function(full_g[c])
        for modality in ("V", "L"):
            path_scores[modality][c] = refit.decision_function(path_g[modality][c])
        for row in c:
            selected_by_row[row] = chosen_c
        rotation_records.append({
            "rotation": c_index, "A_ids_sha256": sha256_obj([subset["ids"][i] for i in a]),
            "B_ids_sha256": sha256_obj([subset["ids"][i] for i in b]),
            "C_ids_sha256": sha256_obj([subset["ids"][i] for i in c]),
            "A_n": len(a), "B_n": len(b), "C_n": len(c),
            "candidate_validation_logloss": [{"C": value, "logloss": loss}
                                               for loss, value in candidates],
            "selected_C": chosen_c,
            "C_label_used_for_fit_or_selection": False,
        })
    if set(selected_by_row) != set(range(len(subset["ids"]))):
        raise RuntimeError("strict probe did not cover every training video OOF")
    q = 2 * labels - 1
    targets, target_map, mads = [], {}, {}
    for modality in ("V", "L"):
        delta = q * (path_scores[modality] - full_score)
        median = float(np.median(delta))
        mad = float(np.median(np.abs(delta - median)))
        mads[modality] = mad
        b = np.minimum(0.0, np.tanh(delta / max(mad, 0.05)))
        support_mask = support["masks"][f"{modality}:{pair['a1']:.2f}"].cpu().numpy()
        for row, vid in enumerate(subset["ids"]):
            record = target_map.setdefault(vid, {"video_id": vid,
                                                 "label": int(labels[row]),
                                                 "namespace": namespace})
            record[modality] = {
                "lower": float(max(-1.0, b[row] - cfg["cte"]["target_half_width"])),
                "upper": float(min(0.0, b[row] + cfg["cte"]["target_half_width"])),
                "b": float(b[row]), "active": bool(support_mask[row]),
                "probe_C": float(selected_by_row[row]),
            }
    targets = [target_map[vid] for vid in sorted(target_map)]
    return target_map, targets, {"seed": int(seed), "folds": rotation_records,
                                 "probe_MAD": mads,
                                 "target_ids_sha256": sha256_obj(sorted(target_map)),
                                 "target_payload_sha256": sha256_obj(targets)}


def randomize_targets(target_map, seed):
    ids = sorted(target_map)
    permutation = deterministic_derangement(len(ids), seed)
    output = {}
    for row, vid in enumerate(ids):
        donor = ids[permutation[row]]
        copied = copy.deepcopy(target_map[donor])
        copied["video_id"] = vid
        copied["donor_video_id"] = donor
        copied["label"] = target_map[vid]["label"]
        output[vid] = copied
    if any(output[vid]["donor_video_id"] == vid for vid in ids):
        raise RuntimeError("random whole-record permutation has a fixed point")
    return output


def fixed_path_identity(model, subset, support, cfg):
    pair = support["selected_pair"]
    full = support["encoded"]["z"]
    directions = {}
    for modality in ("V", "L"):
        path = encode_parts(model, subset["img"].cuda(), subset["txt"].cuda(), cfg,
                            support["anchors"], pair["a1"], modality)["z"]
        direction = path - full
        directions[modality] = F.normalize(direction, dim=1, eps=1e-12).detach()
    return directions


def refresh_context(cfg, model, subset, fixed, frozen_directions, tuple_spec):
    audit = choose_anchors_and_support(model, subset, cfg, fixed=fixed)
    pair = fixed["selected_pair"]
    if pair is None:
        raise RuntimeError("fixed support pair missing")
    selected_masks = [audit["masks"][f"{m}:{float(a):.2f}"]
                      for m in ("V", "L") for a in (pair["a1"], pair["a2"])]
    joint = torch.stack(selected_masks).all(dim=0)
    direction_stats = {}
    current_directions = {}
    for modality in ("V", "L"):
        path = encode_parts(model, subset["img"].cuda(), subset["txt"].cuda(), cfg,
                            audit["anchors"], pair["a1"], modality)["z"]
        current = F.normalize(path - audit["encoded"]["z"], dim=1, eps=1e-12)
        cosine = (current * frozen_directions[modality]).sum(dim=1)
        current_directions[modality] = current
        direction_stats[modality] = {
            "median": float(torch.quantile(cosine, 0.5).cpu()),
            "p10": float(torch.quantile(cosine, 0.1).cpu()),
        }
    masks = {modality: audit["masks"][f"{modality}:{pair['a1']:.2f}"]
             for modality in ("V", "L")}
    mads = {}
    for modality in ("V", "L"):
        path = encode_parts(model, subset["img"].cuda(), subset["txt"].cuda(), cfg,
                            audit["anchors"], pair["a1"], modality)["z"]
        _, mad, _, _ = tangent_values(
            audit["encoded"]["z"], path, subset["labels"].cuda(), subset["ids"],
            audit["encoded"]["z"], subset["labels"].cuda(), subset["ids"],
            tuple_spec["tau"], pair["a1"], tuple_spec["smin"], support=masks[modality])
        mads[modality] = mad
    gate = {
        "joint_support": float(joint.float().mean().cpu()) >=
            cfg["cte"]["refresh_support_min"],
        "direction_median": all(value["median"] >=
                                cfg["cte"]["direction_cosine_median_min"]
                                for value in direction_stats.values()),
        "direction_p10": all(value["p10"] >=
                             cfg["cte"]["direction_cosine_p10_min"]
                             for value in direction_stats.values()),
        "minimum_norm": audit["minimum_pre_normalization_norm"] >=
            cfg["cte"]["minimum_pre_normalization_norm"],
    }
    return {"audit": audit, "bank_z": audit["encoded"]["z"].detach(),
            "masks": masks, "mads": mads, "direction_stats": direction_stats,
            "joint_support": float(joint.float().mean().cpu()), "gate": gate}


def target_interval(batch_ids, target_map, arm, modality, device):
    if arm == "MULTIVIEW":
        n = len(batch_ids)
        return (torch.full((n,), -0.05, device=device),
                torch.full((n,), 0.05, device=device),
                torch.ones(n, dtype=torch.bool, device=device))
    if arm not in ("LABEL_ONLY", "RANDOM"):
        raise RuntimeError("interval requested for non-auxiliary arm")
    records = [target_map[vid][modality] for vid in batch_ids]
    return (torch.as_tensor([item["lower"] for item in records], device=device),
            torch.as_tensor([item["upper"] for item in records], device=device),
            torch.as_tensor([item["active"] for item in records], dtype=torch.bool,
                            device=device))


def cte_loss_for_batch(cfg, model, subset, batch, context, fixed, tuple_spec,
                       target_map, arm):
    if arm == "REMOVE":
        return torch.zeros((), device="cuda"), {"active_rows": 0, "cost": 0.0}
    pair = fixed["selected_pair"]
    id_to_row = {vid: index for index, vid in enumerate(subset["ids"])}
    total, denom, per_modality = torch.zeros((), device="cuda"), 0, {}
    for start in range(0, len(batch["ids"]), cfg["model"]["cte_microbatch_size"]):
        stop = start + cfg["model"]["cte_microbatch_size"]
        ids = [str(value) for value in batch["ids"][start:stop]]
        labels = batch["labels"][start:stop].cuda()
        img, txt = batch["image_feats"][start:stop].cuda(), batch["text_feats"][start:stop].cuda()
        full = encode_parts(model, img, txt, cfg, grad=True)["z"]
        anchor_rows = {m: id_to_row[fixed["anchor_ids"][m]] for m in ("V", "L")}
        anchor_v = encode_parts(
            model, subset["img"][anchor_rows["V"]:anchor_rows["V"] + 1].cuda(),
            subset["txt"][anchor_rows["V"]:anchor_rows["V"] + 1].cuda(), cfg,
            grad=True)["V"]
        anchor_l = encode_parts(
            model, subset["img"][anchor_rows["L"]:anchor_rows["L"] + 1].cuda(),
            subset["txt"][anchor_rows["L"]:anchor_rows["L"] + 1].cuda(), cfg,
            grad=True)["L"]
        anchors = {"V": anchor_v, "L": anchor_l}
        rows = torch.as_tensor([id_to_row[vid] for vid in ids], device="cuda")
        for modality in ("V", "L"):
            path = encode_parts(model, img, txt, cfg, anchors, pair["a1"],
                                modality, grad=True)["z"]
            tangent, _, _, _ = tangent_values(
                full, path, labels, ids, context["bank_z"], subset["labels"].cuda(),
                subset["ids"], tuple_spec["tau"], pair["a1"], tuple_spec["smin"],
                mad=context["mads"][modality])
            lower, upper, target_active = target_interval(ids, target_map, arm,
                                                           modality, "cuda")
            support_active = context["masks"][modality].index_select(0, rows)
            active = support_active & target_active
            cost = interval_cost(tangent, lower, upper)
            total = total + (cost * active.float()).sum()
            count = int(active.sum().item())
            denom += count
            per_modality[modality] = per_modality.get(modality, 0) + count
    if denom == 0:
        raise RuntimeError("auxiliary arm has zero active CTE rows")
    loss = total / float(denom)
    return loss, {"active_rows": denom, "active_by_modality": per_modality,
                  "cost": float(loss.detach().cpu())}


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True,
                                    allow_unused=True)
    squared = torch.zeros((), device="cuda")
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().square().sum()
    return float(torch.sqrt(squared).cpu())


def aggregate_cte_gradient_norm(cfg, model, subset, context, fixed, tuple_spec,
                                target_map, arm):
    """Norm of the clean-training-fold aggregate auxiliary gradient at step zero."""
    loader = make_loader(subset, cfg["model"]["batch_size"], seed=0, shuffle=False)
    weighted = torch.zeros((), device="cuda")
    active_total = 0
    covered_ids = []
    for batch in loader:
        covered_ids.extend(str(value) for value in batch["ids"])
        loss, diagnostics = cte_loss_for_batch(
            cfg, model, subset, batch, context, fixed, tuple_spec,
            target_map, arm)
        active = int(diagnostics["active_rows"])
        weighted = weighted + loss * active
        active_total += active
    if active_total <= 0:
        raise RuntimeError("aggregate CTE gradient has no active training-fold rows")
    if covered_ids != subset["ids"] or len(set(covered_ids)) != len(subset["ids"]):
        raise RuntimeError("aggregate strength match did not cover the full clean fold")
    aggregate = weighted / float(active_total)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return gradient_norm(aggregate, parameters), {
        "active_modality_rows": active_total,
        "covered_video_n": len(covered_ids),
        "covered_video_ids_sha256": sha256_obj(covered_ids),
    }


def bank_drift(start_z, end_z):
    cosine = (F.normalize(start_z, dim=1) * F.normalize(end_z, dim=1)).sum(dim=1)
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    return {"median_same_id_cosine": float(torch.quantile(cosine, 0.5).cpu()),
            "p95_angular_drift_rad": float(torch.quantile(angle, 0.95).cpu())}


def train_arm(cfg, dataset, subset, segment_cache, initialization, fixed,
              frozen_directions, tuple_spec, target_map, arm, strength, recipe,
              seed):
    set_seed(seed)
    model = build_model(cfg, dataset, subset["img"].shape[1], subset["txt"].shape[1])
    model.load_state_dict(initialization, strict=True)
    if state_dict_sha256(model.state_dict()) != state_dict_sha256(initialization):
        raise RuntimeError("arm initialization hash mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"]["adamw_weight_decay"],
        betas=tuple(cfg["model"]["adamw_betas"]), eps=cfg["model"]["adamw_epsilon"],
        amsgrad=cfg["model"]["adamw_amsgrad"])
    loader = make_loader(subset, cfg["model"]["batch_size"], seed, True)
    args = train_args(cfg, dataset)
    train_feats = train_labels = None
    history, refresh_ledger = [], []
    first_grad = None
    first_epoch_drift_failed = False
    total_steps = 0
    epochs = int(cfg["datasets"][dataset]["epoch_index"]) + 1
    for epoch in range(epochs):
        context = refresh_context(cfg, model, subset, fixed, frozen_directions,
                                  tuple_spec)
        if not all(context["gate"].values()):
            raise RuntimeError(f"support/direction/norm STOP at epoch refresh: {context['gate']}")
        interval_start = context["bank_z"].detach().clone()
        seen, losses = [], []
        half_step = math.ceil(len(loader) / 2)
        for step, batch in enumerate(loader):
            if recipe == "half_epoch" and step == half_step:
                before = encode_dataset(model, subset, cfg)["z"]
                drift = bank_drift(interval_start, before)
                drift_pass = (drift["median_same_id_cosine"] >=
                              cfg["cte"]["bank_same_id_cosine_median_min"] and
                              drift["p95_angular_drift_rad"] <=
                              cfg["cte"]["bank_angular_drift_p95_max"])
                refresh_ledger.append({"epoch": epoch, "position": "half",
                                       "drift": drift, "drift_pass": drift_pass,
                                       "support": context["joint_support"],
                                       "direction": context["direction_stats"]})
                if not drift_pass:
                    raise RuntimeError("half-epoch bank interval violates drift gate")
                context = refresh_context(cfg, model, subset, fixed,
                                          frozen_directions, tuple_spec)
                if not all(context["gate"].values()):
                    raise RuntimeError("half-epoch support/direction STOP")
                interval_start = context["bank_z"].detach().clone()
            ids = [str(value) for value in batch["ids"]]
            seen.extend(ids)
            output = compute_loss(
                batch, loader, model, args, train_set=loader.dataset,
                sparse_retrieval_dictionary=None, train_feats=train_feats,
                train_labels=train_labels, segment_cache=segment_cache,
                aux_pack=None, cf_pack=None)
            base_loss, train_feats, train_labels = output[0], output[-2], output[-1]
            if torch.is_tensor(train_feats):
                train_feats = train_feats.detach()
            if torch.is_tensor(train_labels):
                train_labels = train_labels.detach()
            aux_loss, aux_diag = cte_loss_for_batch(
                cfg, model, subset, batch, context, fixed, tuple_spec,
                target_map, arm)
            total = base_loss + tuple_spec["lambda"] * float(strength) * aux_loss
            if not bool(torch.isfinite(total)):
                raise RuntimeError("nonfinite arm training loss")
            if first_grad is None:
                parameters = [parameter for parameter in model.parameters()
                              if parameter.requires_grad]
                first_grad = {
                    "base_gradient_norm": gradient_norm(base_loss, parameters),
                    "raw_cte_gradient_norm": gradient_norm(aux_loss, parameters)
                    if arm != "REMOVE" else 0.0,
                    "strength_scalar": float(strength),
                    "weighted_cte_gradient_norm":
                        (gradient_norm(aux_loss, parameters) * tuple_spec["lambda"] *
                         float(strength)) if arm != "REMOVE" else 0.0,
                }
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["model"]["grad_clip"])
            optimizer.step()
            total_steps += 1
            losses.append({"base": float(base_loss.detach().cpu()),
                           "aux": float(aux_loss.detach().cpu()),
                           "total": float(total.detach().cpu()), **aux_diag})
        if sorted(seen) != sorted(subset["ids"]) or len(seen) != len(subset["ids"]):
            raise RuntimeError("each train video must be exactly one CTE query per epoch")
        end_z = encode_dataset(model, subset, cfg)["z"]
        drift = bank_drift(interval_start, end_z)
        drift_pass = (drift["median_same_id_cosine"] >=
                      cfg["cte"]["bank_same_id_cosine_median_min"] and
                      drift["p95_angular_drift_rad"] <=
                      cfg["cte"]["bank_angular_drift_p95_max"])
        if epoch == 0 and not drift_pass:
            first_epoch_drift_failed = True
        refresh_ledger.append({"epoch": epoch, "position": "end",
                               "drift": drift, "drift_pass": drift_pass,
                               "support": context["joint_support"],
                               "direction": context["direction_stats"]})
        if recipe == "half_epoch" and not drift_pass:
            raise RuntimeError("half-epoch refresh still violates bank drift gate")
        if recipe == "epoch" and epoch > 0 and not drift_pass:
            raise RuntimeError("post-first-epoch bank drift gate failed")
        history.append({"epoch_index": epoch, "steps": len(losses),
                        "mean_base_loss": float(np.mean([x["base"] for x in losses])),
                        "mean_cte_loss": float(np.mean([x["aux"] for x in losses])),
                        "mean_total_loss": float(np.mean([x["total"] for x in losses])),
                        "query_ids_sha256": sha256_obj(seen)})
    final_context = refresh_context(cfg, model, subset, fixed, frozen_directions,
                                    tuple_spec)
    if not all(final_context["gate"].values()):
        raise RuntimeError("final support/direction/norm gate failed")
    return {"state": clone_state(model.state_dict()), "model": model,
            "history": history, "refresh_ledger": refresh_ledger,
            "first_gradient": first_grad, "first_epoch_drift_failed": first_epoch_drift_failed,
            "optimizer_steps": total_steps, "epochs": epochs,
            "final_support": final_context["joint_support"],
            "final_direction": final_context["direction_stats"],
            "final_checkpoint_sha256": state_dict_sha256(model.state_dict())}


def support_freeze_dict(support):
    return {"anchor_ids": support["anchor_ids"],
            "thresholds": support["thresholds"],
            "selected_pair": support["selected_pair"],
            "initial_support_details": support["details"],
            "initial_adjacent_candidates": support["candidates"]}


def train_and_evaluate(cfg, dataset, train_subset, query_subset, segment_cache,
                       initialization, target_map, tuple_spec, arms, recipe, seed,
                       outer_fold=None):
    init_model = build_model(cfg, dataset, train_subset["img"].shape[1],
                             train_subset["txt"].shape[1])
    init_model.load_state_dict(initialization, strict=True)
    support = choose_anchors_and_support(init_model, train_subset, cfg)
    if support["selected_pair"] is None:
        raise RuntimeError("C1 namespace has no supported adjacent pair")
    fixed = support_freeze_dict(support)
    frozen_directions = fixed_path_identity(init_model, train_subset, support, cfg)
    random_targets = randomize_targets(target_map, seed + 1701)
    target_for_arm = {"REMOVE": target_map, "LABEL_ONLY": target_map,
                      "MULTIVIEW": target_map, "RANDOM": random_targets}

    # Exact clean-training-fold aggregate gradient strength match at common initialization.
    strengths = {"REMOVE": 0.0, "LABEL_ONLY": 1.0}
    context = refresh_context(cfg, init_model, train_subset, fixed,
                              frozen_directions, tuple_spec)
    label_norm, label_coverage = aggregate_cte_gradient_norm(
        cfg, init_model, train_subset, context, fixed, tuple_spec,
        target_map, "LABEL_ONLY")
    if label_norm <= 0 or not math.isfinite(label_norm):
        raise RuntimeError("LABEL_ONLY aggregate first-step CTE gradient norm invalid")
    aggregate_coverage = {"LABEL_ONLY": label_coverage}
    aggregate_norms = {"LABEL_ONLY": label_norm}
    for control in ("MULTIVIEW", "RANDOM"):
        if control not in arms:
            continue
        norm, coverage = aggregate_cte_gradient_norm(
            cfg, init_model, train_subset, context, fixed, tuple_spec,
            target_for_arm[control], control)
        if norm <= 0 or not math.isfinite(norm):
            raise RuntimeError(f"{control} aggregate first-step CTE gradient norm invalid")
        strengths[control] = label_norm / norm
        aggregate_coverage[control] = coverage
        aggregate_norms[control] = norm
    del init_model

    results = {}
    any_epoch_drift_failure = False
    for arm in arms:
        result = train_arm(
            cfg, dataset, train_subset, segment_cache, initialization, fixed,
            frozen_directions, tuple_spec, target_for_arm[arm], arm,
            strengths[arm], recipe, seed)
        memory = encode_dataset(result["model"], train_subset, cfg)["z"]
        query = encode_dataset(result["model"], query_subset, cfg)["z"]
        neighbors, predictions, metrics = ordinary_knn(
            train_subset["ids"], memory, train_subset["labels"],
            query_subset["ids"], query, query_subset["labels"],
            cfg["model"]["topk"])
        result.update({"neighbors": neighbors, "predictions": predictions,
                       "metrics": metrics})
        any_epoch_drift_failure = any_epoch_drift_failure or \
            result["first_epoch_drift_failed"]
        del result["model"]
        results[arm] = result
        torch.cuda.empty_cache()
    return {"results": results, "fixed": fixed,
            "frozen_direction_sha256": sha256_obj({m: hashlib.sha256(
                frozen_directions[m].detach().cpu().numpy().tobytes()).hexdigest()
                for m in ("V", "L")}),
            "strengths": strengths, "label_first_step_gradient_norm": label_norm,
            "aggregate_strength_match_coverage": aggregate_coverage,
            "aggregate_first_step_gradient_norms": aggregate_norms,
            "any_first_epoch_drift_failure": any_epoch_drift_failure,
            "recipe": recipe, "random_target_payload_sha256":
                sha256_obj([random_targets[vid] for vid in sorted(random_targets)])}


def inner_splits(bundle, outer_fold, cfg):
    outer_train_ids = sorted(vid for vid, row in bundle["by_id"].items()
                             if row["fold"] != outer_fold)
    labels = [bundle["by_id"][vid]["label"] for vid in outer_train_ids]
    fold_rows = stratified_three(outer_train_ids, labels,
                                 cfg["cte"]["inner_random_state_base"] + outer_fold)
    output = []
    for inner_fold, held_rows in enumerate(fold_rows):
        held = sorted(outer_train_ids[index] for index in held_rows)
        train = sorted(set(outer_train_ids) - set(held))
        if set(train) & set(held) or set(train) | set(held) != set(outer_train_ids):
            raise RuntimeError("inner train/validation partition invalid")
        output.append({"inner_fold": inner_fold, "P_ids": train, "J_ids": held,
                       "P_ids_sha256": sha256_obj(train),
                       "J_ids_sha256": sha256_obj(held),
                       "J_label_used_for_target_fit_or_selection": False})
    return outer_train_ids, output


def phase_inner(cfg, dataset, outer_fold, run_id):
    require_slurm(gpu=True)
    if run_id != expected_inner_run(dataset, outer_fold):
        raise RuntimeError("C1 inner run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    c0_path, _ = require_c0_go(cfg)
    root = resolve(cfg, "artifact_root")
    out_dir = root / "c1" / "inner" / dataset / f"fold{outer_fold}"
    acquire_lock(out_dir / ".inner.lock", run_id, "C1_INNER")
    bundle = load_train_cache(cfg, dataset)
    outer_train_ids, splits = inner_splits(bundle, outer_fold, cfg)
    all_target_rows, grid_rows, initialization_records = [], [], []
    input_paths = [freeze_path, c0_path, bundle["fold_path"], bundle["cache_path"]]
    recipes = []
    for split in splits:
        p = select_rows(bundle, split["P_ids"])
        j = select_rows(bundle, split["J_ids"])
        seg_path, seg_cache, seg_audit = load_segment_cache(cfg, dataset, bundle, p)
        if seg_path not in input_paths:
            input_paths.append(seg_path)
        initialization, init_record = baseline_train(
            cfg, dataset, p, seg_cache,
            cfg["cte"]["seed"] + outer_fold * 10 + split["inner_fold"])
        initialization_records.append({"inner_fold": split["inner_fold"],
                                       "segment_audit": seg_audit, **init_record})
        init_model = build_model(cfg, dataset, p["img"].shape[1], p["txt"].shape[1])
        init_model.load_state_dict(initialization, strict=True)
        support = choose_anchors_and_support(init_model, p, cfg)
        if support["selected_pair"] is None:
            raise RuntimeError("inner split unsupported at baseline initialization")
        targets, target_rows, probe = strict_probe_targets(
            cfg, init_model, p, support,
            cfg["cte"]["inner_random_state_base"] + outer_fold * 100 +
            split["inner_fold"], f"inner{split['inner_fold']}")
        for row in target_rows:
            row["inner_fold"] = split["inner_fold"]
        all_target_rows.extend(target_rows)
        split["probe"] = probe
        del init_model
        recipe = "epoch"
        split_rows = []
        for spec in tuple_grid(cfg):
            package = train_and_evaluate(
                cfg, dataset, p, j, seg_cache, initialization, targets, spec,
                ("REMOVE", "LABEL_ONLY"), recipe,
                cfg["cte"]["seed"] + outer_fold * 10 + split["inner_fold"])
            if package["any_first_epoch_drift_failure"]:
                recipe = "half_epoch"
                split_rows = []
                break
            remove = package["results"]["REMOVE"]
            label = package["results"]["LABEL_ONLY"]
            split_rows.append({
                "outer_fold": outer_fold, "inner_fold": split["inner_fold"],
                "tuple": spec, "tuple_key": tuple_key(spec), "recipe": recipe,
                "REMOVE_metrics": remove["metrics"],
                "LABEL_ONLY_metrics": label["metrics"],
                "accuracy_gain": label["metrics"]["accuracy"] - remove["metrics"]["accuracy"],
                "macro_f1_gain": label["metrics"]["macro_f1"] - remove["metrics"]["macro_f1"],
                "query_ids": j["ids"], "query_labels": [int(x) for x in j["labels"]],
                "REMOVE_predictions": [x["prediction"] for x in remove["predictions"]],
                "LABEL_ONLY_predictions": [x["prediction"] for x in label["predictions"]],
                "initialization_sha256": init_record["baseline_checkpoint_sha256"],
                "fixed": package["fixed"], "strengths": package["strengths"],
                "aggregate_strength_match_coverage":
                    package["aggregate_strength_match_coverage"],
                "aggregate_first_step_gradient_norms":
                    package["aggregate_first_step_gradient_norms"],
                "label_aggregate_first_step_gradient_norm":
                    package["label_first_step_gradient_norm"],
                "REMOVE_diagnostics": {k: remove[k] for k in
                    ("first_gradient", "refresh_ledger", "optimizer_steps", "epochs",
                     "final_support", "final_direction", "final_checkpoint_sha256")},
                "LABEL_ONLY_diagnostics": {k: label[k] for k in
                    ("first_gradient", "refresh_ledger", "optimizer_steps", "epochs",
                     "final_support", "final_direction", "final_checkpoint_sha256")},
            })
        if recipe == "half_epoch":
            for spec in tuple_grid(cfg):
                package = train_and_evaluate(
                    cfg, dataset, p, j, seg_cache, initialization, targets, spec,
                    ("REMOVE", "LABEL_ONLY"), recipe,
                    cfg["cte"]["seed"] + outer_fold * 10 + split["inner_fold"])
                if package["any_first_epoch_drift_failure"]:
                    raise RuntimeError("half-epoch fallback still fails first-epoch drift")
                remove, label = package["results"]["REMOVE"], package["results"]["LABEL_ONLY"]
                split_rows.append({
                    "outer_fold": outer_fold, "inner_fold": split["inner_fold"],
                    "tuple": spec, "tuple_key": tuple_key(spec), "recipe": recipe,
                    "REMOVE_metrics": remove["metrics"], "LABEL_ONLY_metrics": label["metrics"],
                    "accuracy_gain": label["metrics"]["accuracy"] - remove["metrics"]["accuracy"],
                    "macro_f1_gain": label["metrics"]["macro_f1"] - remove["metrics"]["macro_f1"],
                    "query_ids": j["ids"], "query_labels": [int(x) for x in j["labels"]],
                    "REMOVE_predictions": [x["prediction"] for x in remove["predictions"]],
                    "LABEL_ONLY_predictions": [x["prediction"] for x in label["predictions"]],
                    "initialization_sha256": init_record["baseline_checkpoint_sha256"],
                    "fixed": package["fixed"], "strengths": package["strengths"],
                    "aggregate_strength_match_coverage":
                        package["aggregate_strength_match_coverage"],
                    "aggregate_first_step_gradient_norms":
                        package["aggregate_first_step_gradient_norms"],
                    "label_aggregate_first_step_gradient_norm":
                        package["label_first_step_gradient_norm"],
                    "REMOVE_diagnostics": {k: remove[k] for k in
                        ("first_gradient", "refresh_ledger", "optimizer_steps", "epochs",
                         "final_support", "final_direction", "final_checkpoint_sha256")},
                    "LABEL_ONLY_diagnostics": {k: label[k] for k in
                        ("first_gradient", "refresh_ledger", "optimizer_steps", "epochs",
                         "final_support", "final_direction", "final_checkpoint_sha256")},
                })
        recipes.append(recipe)
        grid_rows.extend(split_rows)
    if len(grid_rows) != 3 * len(tuple_grid(cfg)):
        raise RuntimeError("inner grid result cardinality mismatch")
    split_path, target_path, grid_path = (out_dir / "inner_splits.json",
                                          out_dir / "probe_targets.jsonl",
                                          out_dir / "grid_metrics.jsonl")
    atomic_write_json(split_path, {"schema_version": 1, "run_id": run_id,
                                  "dataset": dataset, "outer_fold": outer_fold,
                                  "outer_train_ids_sha256": sha256_obj(outer_train_ids),
                                  "splits": splits})
    atomic_write_jsonl(target_path, all_target_rows)
    atomic_write_jsonl(grid_path, grid_rows)
    fold_sha = sha256_obj(sorted((vid, row["label"], row["fold"])
                                 for vid, row in bundle["by_id"].items()))
    manifest = base_manifest(
        cfg, run_id, "C1_INNER", "COMPLETED", input_paths,
        [split_path, target_path, grid_path], fold_sha,
        checkpoint_sha256=sha256_obj([x["baseline_checkpoint_sha256"]
                                      for x in initialization_records]),
        extra={"dataset": dataset, "outer_fold": outer_fold,
               "outer_train_ids_sha256": sha256_obj(outer_train_ids),
               "initializations": initialization_records, "recipes": recipes,
               "exact_params": cfg["cte"], "grid_rows": len(grid_rows),
               "C2_C3_C4_locked": True})
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": run_id,
                          "grid_rows": len(grid_rows), "recipes": recipes}))


def verify_grid_row(row, expected_outer, expected_query_ids, by_id):
    if row.get("outer_fold") != expected_outer:
        raise RuntimeError("inner grid outer fold mismatch")
    if row.get("query_ids") != expected_query_ids:
        raise RuntimeError("inner grid query IDs differ from authoritative J partition")
    expected_labels = [by_id[vid]["label"] for vid in expected_query_ids]
    if [int(value) for value in row.get("query_labels", [])] != expected_labels:
        raise RuntimeError("inner grid labels differ from authoritative video labels")
    n = len(row["query_ids"])
    if not (n == len(row["query_labels"]) == len(row["REMOVE_predictions"]) ==
            len(row["LABEL_ONLY_predictions"])):
        raise RuntimeError("inner prediction ledger length mismatch")
    remove = binary_metrics(row["query_labels"], row["REMOVE_predictions"])
    label = binary_metrics(row["query_labels"], row["LABEL_ONLY_predictions"])
    if remove != row["REMOVE_metrics"] or label != row["LABEL_ONLY_metrics"]:
        raise RuntimeError("inner producer metrics disagree with independent recomputation")
    if abs((label["accuracy"] - remove["accuracy"]) - row["accuracy_gain"]) > 1e-12 or \
            abs((label["macro_f1"] - remove["macro_f1"]) - row["macro_f1_gain"]) > 1e-12:
        raise RuntimeError("inner gain mismatch")


def selection_sort_key(item):
    spec = item["tuple"]
    return (-item["min_gain"], spec["lambda"], -spec["smin"],
            spec["tau"], spec["lambda"], spec["smin"])


def phase_select(cfg, run_id):
    require_slurm(gpu=False)
    if run_id != "CTE-C1-SELECT-v1":
        raise RuntimeError("selection run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    c0_path, _ = require_c0_go(cfg)
    root = resolve(cfg, "artifact_root")
    out_dir = root / "c1" / "selection"
    source_paths = [freeze_path, c0_path]
    by_dataset_fold = {}
    for dataset in DATASETS:
        bundle = load_train_cache(cfg, dataset)
        fold_sha = sha256_obj(sorted((vid, item["label"], item["fold"])
                                     for vid, item in bundle["by_id"].items()))
        for outer in range(5):
            directory = root / "c1" / "inner" / dataset / f"fold{outer}"
            manifest_path, grid_path = directory / "manifest.json", directory / "grid_metrics.jsonl"
            split_path, target_path = directory / "inner_splits.json", directory / "probe_targets.jsonl"
            manifest = read_json(manifest_path)
            expected_checkpoint = sha256_obj([
                item["baseline_checkpoint_sha256"]
                for item in manifest.get("initializations", [])])
            validate_manifest_common(
                manifest, expected_inner_run(dataset, outer), "C1_INNER", cfg,
                fold_sha, expected_checkpoint,
                [split_path, target_path, grid_path],
                [freeze_path, c0_path, bundle["fold_path"], bundle["cache_path"]])
            authoritative_outer, authoritative_splits = inner_splits(bundle, outer, cfg)
            split_payload = read_json(split_path)
            if split_payload.get("run_id") != expected_inner_run(dataset, outer) or \
                    split_payload.get("dataset") != dataset or \
                    split_payload.get("outer_fold") != outer or \
                    split_payload.get("outer_train_ids_sha256") != sha256_obj(authoritative_outer):
                raise RuntimeError("inner split provenance mismatch")
            produced_splits = split_payload.get("splits", [])
            if len(produced_splits) != 3:
                raise RuntimeError("inner split artifact must contain exactly three folds")
            for expected, produced in zip(authoritative_splits, produced_splits):
                for key in ("inner_fold", "P_ids", "J_ids", "P_ids_sha256", "J_ids_sha256"):
                    if produced.get(key) != expected.get(key):
                        raise RuntimeError(f"inner authoritative partition mismatch: {key}")
                if produced.get("J_label_used_for_target_fit_or_selection") is not False:
                    raise RuntimeError("inner J-label leakage declaration mismatch")
            target_rows = read_jsonl(target_path)
            targets_by_inner = {index: [] for index in range(3)}
            for target in target_rows:
                inner = target.get("inner_fold")
                if inner not in targets_by_inner:
                    raise RuntimeError("probe target has invalid inner fold")
                vid = target.get("video_id")
                if vid not in bundle["by_id"] or int(target.get("label")) != \
                        bundle["by_id"][vid]["label"]:
                    raise RuntimeError("probe target ID/label differs from video gold")
                if any(name not in target for name in ("V", "L")):
                    raise RuntimeError("probe target missing a whole-modality record")
                targets_by_inner[inner].append(vid)
            for split in authoritative_splits:
                inner = split["inner_fold"]
                if sorted(targets_by_inner[inner]) != split["P_ids"] or \
                        len(set(targets_by_inner[inner])) != len(split["P_ids"]):
                    raise RuntimeError("probe targets do not exactly cover authoritative P")
                if set(targets_by_inner[inner]) & set(split["J_ids"]):
                    raise RuntimeError("inner J ID appears in candidate training targets")
            rows = read_jsonl(grid_path)
            for row in rows:
                inner = row.get("inner_fold")
                if inner not in range(3):
                    raise RuntimeError("grid row inner fold invalid")
                verify_grid_row(row, outer, authoritative_splits[inner]["J_ids"],
                                bundle["by_id"])
            expected_keys = {(inner, tuple_key(spec)) for inner in range(3)
                             for spec in tuple_grid(cfg)}
            observed_keys = [(row["inner_fold"], row["tuple_key"]) for row in rows]
            if set(observed_keys) != expected_keys or len(observed_keys) != len(expected_keys):
                raise RuntimeError("inner grid does not exactly cover 3 folds x 8 tuples")
            expected_tuple_by_key = {tuple_key(spec): spec for spec in tuple_grid(cfg)}
            if any(row.get("tuple") != expected_tuple_by_key.get(row.get("tuple_key"))
                   for row in rows):
                raise RuntimeError("inner tuple payload differs from frozen grid key")
            by_dataset_fold[(dataset, outer)] = rows
            source_paths.extend([manifest_path, grid_path, split_path, target_path])
    outputs = []
    selections = []
    pending_fold_payloads = []
    for outer in range(5):
        candidate_rows = []
        for spec in tuple_grid(cfg):
            key = tuple_key(spec)
            cells = {}
            for dataset in DATASETS:
                rows = [row for row in by_dataset_fold[(dataset, outer)]
                        if row["tuple_key"] == key]
                if len(rows) != 3:
                    raise RuntimeError("selection missing three inner-fold rows")
                labels = sum((row["query_labels"] for row in rows), [])
                remove_pred = sum((row["REMOVE_predictions"] for row in rows), [])
                label_pred = sum((row["LABEL_ONLY_predictions"] for row in rows), [])
                if len(set(sum((row["query_ids"] for row in rows), []))) != len(labels):
                    raise RuntimeError("inner held-out IDs overlap or duplicate")
                remove = binary_metrics(labels, remove_pred)
                label = binary_metrics(labels, label_pred)
                cells[dataset] = {"REMOVE": remove, "LABEL_ONLY": label,
                                  "accuracy_gain": label["accuracy"] - remove["accuracy"],
                                  "macro_f1_gain": label["macro_f1"] - remove["macro_f1"]}
            gains = [cells[d][metric] for d in DATASETS
                     for metric in ("accuracy_gain", "macro_f1_gain")]
            candidate_rows.append({"tuple": spec, "tuple_key": key, "cells": cells,
                                   "min_gain": min(gains)})
        candidate_rows.sort(key=selection_sort_key)
        selected = candidate_rows[0]
        fold_payload = add_payload_hash({
            "schema_version": 1, "run_id": run_id, "outer_fold": outer,
            "selected_tuple": selected["tuple"], "selected_tuple_key": selected["tuple_key"],
            "selection_objective_min_gain": selected["min_gain"],
            "cells": selected["cells"], "all_candidates": candidate_rows,
            "outer_dev_test_teacher_used_for_selection": False, **SUPERVISION,
        })
        path = out_dir / f"fold{outer}.json"
        pending_fold_payloads.append((path, fold_payload))
        outputs.append(path)
        selections.append(selected["tuple"])
    fold_hashes = {dataset: read_json(fold_path(cfg, dataset))["id_to_fold_sha256"]
                   for dataset in DATASETS}
    acquire_lock(out_dir / ".selection.lock", run_id, "C1_SELECT")
    for path, fold_payload in pending_fold_payloads:
        atomic_write_json(path, fold_payload)
    manifest = base_manifest(
        cfg, run_id, "C1_SELECT", "COMPLETED", source_paths, outputs,
        sha256_obj(fold_hashes), checkpoint_sha256=None,
        extra={"selected_tuples": selections,
               "selection_rule": "maximize_min_four_dataset_metric_gains_then_smaller_lambda_larger_smin_lexicographic",
               "C2_C3_C4_locked": True})
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": run_id,
                          "selected_tuples": selections}))


def phase_outer(cfg, dataset, outer_fold, run_id):
    require_slurm(gpu=True)
    if run_id != expected_outer_run(dataset, outer_fold):
        raise RuntimeError("C1 outer run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    c0_path, _ = require_c0_go(cfg)
    root = resolve(cfg, "artifact_root")
    selection_manifest = root / "c1" / "selection" / "manifest.json"
    selection = read_json(selection_manifest)
    combined_fold_sha = sha256_obj({
        name: read_json(fold_path(cfg, name))["id_to_fold_sha256"]
        for name in DATASETS})
    validate_manifest_common(selection, "CTE-C1-SELECT-v1", "C1_SELECT", cfg,
                             combined_fold_sha, None,
                             [root / "c1" / "selection" / f"fold{index}.json"
                              for index in range(5)], [freeze_path, c0_path])
    selected_path = root / "c1" / "selection" / f"fold{outer_fold}.json"
    selected = read_json(selected_path)
    verify_payload(selected)
    spec = selected["selected_tuple"]
    out_dir = root / "c1" / "outer" / dataset / f"fold{outer_fold}"
    acquire_lock(out_dir / ".outer.lock", run_id, "C1_OUTER")
    bundle = load_train_cache(cfg, dataset)
    train_ids = sorted(vid for vid, row in bundle["by_id"].items()
                       if row["fold"] != outer_fold)
    query_ids = sorted(vid for vid, row in bundle["by_id"].items()
                       if row["fold"] == outer_fold)
    train_subset, query_subset = select_rows(bundle, train_ids), select_rows(bundle, query_ids)
    seg_path, seg_cache, seg_audit = load_segment_cache(cfg, dataset, bundle, train_subset)
    init_model = build_model(cfg, dataset, train_subset["img"].shape[1],
                             train_subset["txt"].shape[1])
    initialization, checkpoint_path, checkpoint_manifest = ssr_initialization(
        cfg, dataset, outer_fold, init_model)
    support = choose_anchors_and_support(init_model, train_subset, cfg)
    if support["selected_pair"] is None:
        raise RuntimeError("outer initialization has no supported adjacent pair")
    target_map, target_rows, probe = strict_probe_targets(
        cfg, init_model, train_subset, support,
        cfg["cte"]["inner_random_state_base"] + 1000 + outer_fold,
        f"outer{outer_fold}")
    del init_model
    recipe = "epoch"
    package = train_and_evaluate(
        cfg, dataset, train_subset, query_subset, seg_cache, initialization,
        target_map, spec, ARMS, recipe, cfg["cte"]["seed"], outer_fold)
    if package["any_first_epoch_drift_failure"]:
        recipe = "half_epoch"
        package = train_and_evaluate(
            cfg, dataset, train_subset, query_subset, seg_cache, initialization,
            target_map, spec, ARMS, recipe, cfg["cte"]["seed"], outer_fold)
        if package["any_first_epoch_drift_failure"]:
            raise RuntimeError("half-epoch fallback still fails outer drift")
    anchors_path, targets_path, diagnostics_path = (
        out_dir / "anchors.json", out_dir / "probe_targets.jsonl",
        out_dir / "diagnostics.jsonl")
    atomic_write_json(anchors_path, add_payload_hash({
        "schema_version": 1, "run_id": run_id, "dataset": dataset,
        "outer_fold": outer_fold, "fixed": package["fixed"],
        "frozen_direction_sha256": package["frozen_direction_sha256"],
        "anchor_reselection_after_initialization": False,
    }))
    atomic_write_jsonl(targets_path, target_rows)
    diagnostics_rows, output_paths = [], [anchors_path, targets_path]
    checkpoint_hashes = {}
    for arm in ARMS:
        result = package["results"][arm]
        diagnostics_rows.append({
            "arm": arm, "recipe": recipe, "tuple": spec,
            "strength": package["strengths"][arm], "metrics": result["metrics"],
            "first_gradient": result["first_gradient"],
            "optimizer_steps": result["optimizer_steps"], "epochs": result["epochs"],
            "refresh_ledger": result["refresh_ledger"],
            "final_support": result["final_support"],
            "final_direction": result["final_direction"],
            "final_checkpoint_sha256": result["final_checkpoint_sha256"],
        })
        checkpoint_hashes[arm] = result["final_checkpoint_sha256"]
        pred_path = out_dir / f"predictions_{arm}.json"
        neighbor_path = out_dir / f"neighbors_{arm}.jsonl"
        atomic_write_json(pred_path, result["predictions"])
        atomic_write_jsonl(neighbor_path, result["neighbors"])
        output_paths.extend([pred_path, neighbor_path])
    atomic_write_jsonl(diagnostics_path, diagnostics_rows)
    output_paths.append(diagnostics_path)
    fold_sha = sha256_obj(sorted((vid, row["label"], row["fold"])
                                 for vid, row in bundle["by_id"].items()))
    input_paths = [freeze_path, c0_path, selection_manifest, selected_path,
                   bundle["fold_path"], bundle["cache_path"], seg_path,
                   checkpoint_path, checkpoint_manifest]
    manifest = base_manifest(
        cfg, run_id, "C1_OUTER", "COMPLETED", input_paths, output_paths,
        fold_sha, checkpoint_sha256=sha256_obj(checkpoint_hashes),
        extra={"dataset": dataset, "outer_fold": outer_fold,
               "train_ids_sha256": sha256_obj(train_ids),
               "query_ids_sha256": sha256_obj(query_ids),
               "train_query_overlap": [], "selected_tuple": spec,
               "initialization_hash": sha256_file(checkpoint_path),
               "batch_size": cfg["model"]["batch_size"],
               "cte_microbatch_size": cfg["model"]["cte_microbatch_size"],
               "optimizer_step_counts": {arm: package["results"][arm]["optimizer_steps"]
                                         for arm in ARMS},
               "epoch_refresh_recipe": recipe,
               "anchor_ids": package["fixed"]["anchor_ids"],
               "radii": package["fixed"]["selected_pair"],
               "support_drift_norm_gradient_diagnostics":
                   str(diagnostics_path.relative_to(ROOT)),
               "probe": probe, "segment_cache_audit": seg_audit,
               "control_strength_scalars": package["strengths"],
               "aggregate_strength_match_coverage":
                   package["aggregate_strength_match_coverage"],
               "aggregate_first_step_gradient_norms":
                   package["aggregate_first_step_gradient_norms"],
               "label_aggregate_first_step_gradient_norm":
                   package["label_first_step_gradient_norm"],
               "random_target_payload_sha256": package["random_target_payload_sha256"],
               "C2_C3_C4_locked": True})
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": run_id,
                          "recipe": recipe,
                          "metrics": {arm: package["results"][arm]["metrics"]
                                      for arm in ARMS}}))


def recompute_outer_arm(cfg, directory, arm, expected_query_ids,
                        expected_memory_ids, by_id):
    predictions = read_json(directory / f"predictions_{arm}.json")
    neighbors = read_jsonl(directory / f"neighbors_{arm}.jsonl")
    if len(predictions) != len(neighbors) or not predictions:
        raise RuntimeError("outer prediction/neighbor ledger cardinality mismatch")
    by_neighbor = {row["query_id"]: row for row in neighbors}
    if len(by_neighbor) != len(neighbors):
        raise RuntimeError("duplicate neighbor query IDs")
    if [row.get("query_id") for row in predictions] != expected_query_ids or \
            [row.get("query_id") for row in neighbors] != expected_query_ids:
        raise RuntimeError("outer ledger query IDs differ from authoritative held-out fold")
    recomputed, neighbor_sets = [], {}
    weights = list(range(cfg["model"]["topk"], 0, -1))
    for row in predictions:
        nrow = by_neighbor.get(row["query_id"])
        if nrow is None or int(nrow["query_label"]) != int(row["query_label"]):
            raise RuntimeError("prediction-neighbor label/ID mismatch")
        if int(row["query_label"]) != by_id[row["query_id"]]["label"]:
            raise RuntimeError("outer ledger label differs from authoritative video label")
        top = nrow["neighbors"]
        if len(top) != cfg["model"]["topk"]:
            raise RuntimeError("neighbor ledger is not exact top20")
        if [item["rank"] for item in top] != list(range(1, cfg["model"]["topk"] + 1)):
            raise RuntimeError("neighbor ranks malformed")
        neighbor_ids = [item["id"] for item in top]
        if len(set(neighbor_ids)) != len(neighbor_ids) or \
                not set(neighbor_ids).issubset(expected_memory_ids) or \
                row["query_id"] in neighbor_ids:
            raise RuntimeError("outer neighbors violate authoritative memory/self exclusion")
        if any(item["id"] not in by_id or int(item["label"]) !=
               by_id[item["id"]]["label"] for item in top):
            raise RuntimeError("neighbor label differs from authoritative video label")
        score = sum(weight * float(item["cosine"]) * (2 * int(item["label"]) - 1)
                    for weight, item in zip(weights, top)) / sum(weights)
        prediction = int(score >= 0.0)
        if prediction != int(row["prediction"]) or \
                abs(score - float(row["arithmetic_cosine_score"])) > 1e-7:
            raise RuntimeError("ordinary kNN producer ledger mismatch")
        recomputed.append({"query_id": row["query_id"],
                           "label": int(row["query_label"]),
                           "prediction": prediction})
        neighbor_sets[row["query_id"]] = [item["id"] for item in top]
    return recomputed, neighbor_sets


def bootstrap_churn_difference(label_churn, random_churn, seed, replicates):
    label_churn = np.asarray(label_churn, dtype=np.float64)
    random_churn = np.asarray(random_churn, dtype=np.float64)
    if label_churn.shape != random_churn.shape or not label_churn.size:
        raise RuntimeError("churn bootstrap input mismatch")
    rng = np.random.default_rng(int(seed))
    n = len(label_churn)
    samples = np.empty(int(replicates), dtype=np.float64)
    difference = label_churn - random_churn
    for index in range(int(replicates)):
        draw = rng.integers(0, n, n)
        samples[index] = difference[draw].mean()
    return {"observed": float(difference.mean()),
            "lower_95": percentile_linear(samples, 0.025),
            "upper_95": percentile_linear(samples, 0.975),
            "replicates": int(replicates), "seed": int(seed)}


def deterministic_mode(tuples):
    counts = Counter(tuple_key(item) for item in tuples)
    maximum = max(counts.values())
    candidates = [item for item in tuples if counts[tuple_key(item)] == maximum]
    candidates.sort(key=lambda item: (item["lambda"], -item["smin"],
                                      item["tau"], item["lambda"], item["smin"]))
    return candidates[0], dict(counts)


def phase_decide(cfg, run_id):
    require_slurm(gpu=False)
    if run_id != "CTE-C1-DECISION-v1":
        raise RuntimeError("C1 decision run ID mismatch")
    freeze_path, _ = verify_config_freeze(cfg)
    c0_path, _ = require_c0_go(cfg)
    root = resolve(cfg, "artifact_root")
    selection_manifest = root / "c1" / "selection" / "manifest.json"
    selection = read_json(selection_manifest)
    combined_fold_sha = sha256_obj({
        name: read_json(fold_path(cfg, name))["id_to_fold_sha256"]
        for name in DATASETS})
    validate_manifest_common(selection, "CTE-C1-SELECT-v1", "C1_SELECT", cfg,
                             combined_fold_sha, None,
                             [root / "c1" / "selection" / f"fold{index}.json"
                              for index in range(5)], [freeze_path, c0_path])
    selected_tuples, inputs, dataset_results = [], [freeze_path, c0_path, selection_manifest], {}
    for outer in range(5):
        selected_path = root / "c1" / "selection" / f"fold{outer}.json"
        payload = read_json(selected_path)
        verify_payload(payload)
        selected_tuples.append(payload["selected_tuple"])
        inputs.append(selected_path)
    modal_tuple, mode_counts = deterministic_mode(selected_tuples)
    for dataset in DATASETS:
        bundle = load_train_cache(cfg, dataset)
        fold_sha = sha256_obj(sorted((vid, item["label"], item["fold"])
                                     for vid, item in bundle["by_id"].items()))
        arm_rows = {arm: [] for arm in ARMS}
        neighbor_sets = {arm: {} for arm in ARMS}
        diagnostics = []
        for outer in range(5):
            directory = root / "c1" / "outer" / dataset / f"fold{outer}"
            manifest_path = directory / "manifest.json"
            manifest = read_json(manifest_path)
            diagnostics_path = directory / "diagnostics.jsonl"
            fold_diagnostics = read_jsonl(diagnostics_path)
            if len(fold_diagnostics) != len(ARMS) or \
                    {row.get("arm") for row in fold_diagnostics} != set(ARMS):
                raise RuntimeError("outer diagnostics do not contain exactly four arms")
            expected_checkpoint = sha256_obj({row["arm"]: row["final_checkpoint_sha256"]
                                              for row in fold_diagnostics})
            validate_manifest_common(
                manifest, expected_outer_run(dataset, outer), "C1_OUTER", cfg,
                fold_sha, expected_checkpoint,
                [directory / "anchors.json", directory / "probe_targets.jsonl",
                 directory / "diagnostics.jsonl"] +
                [directory / f"predictions_{arm}.json" for arm in ARMS] +
                [directory / f"neighbors_{arm}.jsonl" for arm in ARMS],
                [freeze_path, c0_path, selection_manifest,
                 root / "c1" / "selection" / f"fold{outer}.json",
                 bundle["fold_path"], bundle["cache_path"]])
            if manifest.get("selected_tuple") != selected_tuples[outer]:
                raise RuntimeError("outer selected tuple differs from frozen selection")
            expected_query_ids = sorted(vid for vid, item in bundle["by_id"].items()
                                        if item["fold"] == outer)
            expected_memory_ids = {vid for vid, item in bundle["by_id"].items()
                                   if item["fold"] != outer}
            if manifest.get("query_ids_sha256") != sha256_obj(expected_query_ids) or \
                    manifest.get("train_ids_sha256") != sha256_obj(sorted(expected_memory_ids)) or \
                    manifest.get("train_query_overlap") != []:
                raise RuntimeError("outer manifest partition provenance mismatch")
            strengths = manifest.get("control_strength_scalars", {})
            norms = manifest.get("aggregate_first_step_gradient_norms", {})
            coverage = manifest.get("aggregate_strength_match_coverage", {})
            ordered_memory = sorted(expected_memory_ids)
            for arm in ("LABEL_ONLY", "MULTIVIEW", "RANDOM"):
                item = coverage.get(arm, {})
                if item.get("covered_video_n") != len(ordered_memory) or \
                        item.get("covered_video_ids_sha256") != sha256_obj(ordered_memory) or \
                        int(item.get("active_modality_rows", 0)) <= 0 or \
                        not math.isfinite(float(norms.get(arm, float("nan")))) or \
                        float(norms.get(arm, 0.0)) <= 0:
                    raise RuntimeError("control aggregate gradient coverage/norm invalid")
            if float(strengths.get("LABEL_ONLY", float("nan"))) != 1.0:
                raise RuntimeError("LABEL_ONLY aggregate strength must be one")
            for arm in ("MULTIVIEW", "RANDOM"):
                expected_strength = float(norms["LABEL_ONLY"]) / float(norms[arm])
                if not math.isclose(float(strengths.get(arm, float("nan"))),
                                    expected_strength, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError("control scalar does not exactly match aggregate norm ratio")
            anchors_path = directory / "anchors.json"
            anchors = read_json(anchors_path)
            verify_payload(anchors)
            fixed = anchors.get("fixed", {})
            if anchors.get("dataset") != dataset or anchors.get("outer_fold") != outer or \
                    anchors.get("run_id") != expected_outer_run(dataset, outer):
                raise RuntimeError("outer anchor artifact provenance mismatch")
            if fixed.get("anchor_ids") != manifest.get("anchor_ids") or \
                    fixed.get("selected_pair") != manifest.get("radii") or \
                    any(value not in expected_memory_ids
                        for value in fixed.get("anchor_ids", {}).values()):
                raise RuntimeError("outer fixed anchor/radius mismatch")
            passing = [item for item in fixed.get("initial_adjacent_candidates", [])
                       if item.get("passed") is True]
            if not passing or fixed.get("selected_pair") != \
                    max(passing, key=lambda item: item["a2"]):
                raise RuntimeError("outer adjacent radius was not independently reproducible")
            source_checkpoint = resolve(cfg, "ssr_artifacts") / "oof" / dataset / \
                f"fold{outer}" / f"checkpoint_epoch{cfg['datasets'][dataset]['epoch_index']}.pt"
            if manifest.get("initialization_hash") != sha256_file(source_checkpoint):
                raise RuntimeError("outer initialization checkpoint provenance mismatch")
            inputs.append(manifest_path)
            diagnostics.extend(fold_diagnostics)
            target_path = directory / "probe_targets.jsonl"
            target_rows = read_jsonl(target_path)
            target_ids = [row.get("video_id") for row in target_rows]
            if sorted(target_ids) != sorted(expected_memory_ids) or \
                    len(set(target_ids)) != len(expected_memory_ids):
                raise RuntimeError("outer probe targets do not exactly cover outer train IDs")
            if any(vid not in bundle["by_id"] or int(row.get("label")) !=
                   bundle["by_id"][vid]["label"] or "V" not in row or "L" not in row
                   for vid, row in zip(target_ids, target_rows)):
                raise RuntimeError("outer probe target ID/label/modality mismatch")
            if manifest.get("probe", {}).get("target_payload_sha256") != \
                    sha256_obj(target_rows):
                raise RuntimeError("outer probe target payload provenance mismatch")
            inputs.extend([anchors_path, target_path,
                           diagnostics_path])
            for arm in ARMS:
                rows, sets = recompute_outer_arm(
                    cfg, directory, arm, expected_query_ids,
                    expected_memory_ids, bundle["by_id"])
                if any(item["query_id"] in neighbor_sets[arm] for item in rows):
                    raise RuntimeError("outer query ID appears in multiple folds")
                arm_rows[arm].extend(rows)
                neighbor_sets[arm].update(sets)
                inputs.extend([directory / f"predictions_{arm}.json",
                               directory / f"neighbors_{arm}.jsonl"])
        expected_ids = set(bundle["by_id"])
        if any(set(row["query_id"] for row in arm_rows[arm]) != expected_ids for arm in ARMS):
            raise RuntimeError("five outer ledgers do not form exact train OOF coverage")
        metrics = {arm: binary_metrics([row["label"] for row in arm_rows[arm]],
                                       [row["prediction"] for row in arm_rows[arm]])
                   for arm in ARMS}
        remove_by_id = {row["query_id"]: row for row in arm_rows["REMOVE"]}
        label_by_id = {row["query_id"]: row for row in arm_rows["LABEL_ONLY"]}
        corrected, broken, per_class = 0, 0, {}
        for cls in (0, 1):
            c = b = 0
            for vid in sorted(expected_ids):
                base, label = remove_by_id[vid], label_by_id[vid]
                if base["label"] != cls:
                    continue
                c += int(base["prediction"] != cls and label["prediction"] == cls)
                b += int(base["prediction"] == cls and label["prediction"] != cls)
            per_class[str(cls)] = {"corrected": c, "broken": b,
                                   "net": c - b, "net_positive": c - b > 0}
            corrected += c
            broken += b
        label_churn, random_churn = [], []
        for vid in sorted(expected_ids):
            base = neighbor_sets["REMOVE"][vid]
            label_churn.append(jaccard_churn(base, neighbor_sets["LABEL_ONLY"][vid]))
            random_churn.append(jaccard_churn(base, neighbor_sets["RANDOM"][vid]))
        bootstrap = bootstrap_churn_difference(
            label_churn, random_churn,
            cfg["cte"]["bootstrap_seed"] + (0 if dataset == "MHC" else 1),
            cfg["cte"]["bootstrap_replicates"])
        support_drift_gates = {
            "all_diagnostics_present": len(diagnostics) == 5 * len(ARMS),
            "all_initial_final_support": all(row["final_support"] >=
                cfg["cte"]["refresh_support_min"] for row in diagnostics),
            "all_direction_median": all(all(value["median"] >=
                cfg["cte"]["direction_cosine_median_min"]
                for value in row["final_direction"].values()) for row in diagnostics),
            "all_direction_p10": all(all(value["p10"] >=
                cfg["cte"]["direction_cosine_p10_min"]
                for value in row["final_direction"].values()) for row in diagnostics),
            "all_bank_drift": all(bool(row.get("refresh_ledger")) and
                all(item.get("drift_pass") is True
                    for item in row["refresh_ledger"])
                for row in diagnostics),
            "finite_gradients": all(all(math.isfinite(float(value)) for value in
                row["first_gradient"].values()) for row in diagnostics),
            "same_optimizer_steps": len(set(row["optimizer_steps"] for row in diagnostics)) == 1,
        }
        gains = {metric: metrics["LABEL_ONLY"][metric] - metrics["REMOVE"][metric]
                 for metric in ("accuracy", "macro_f1")}
        gates = {
            "support_direction_bank_numerics": all(support_drift_gates.values()),
            "accuracy_gain": gains["accuracy"] >= cfg["cte"]["c1_gain_min"],
            "macro_f1_gain": gains["macro_f1"] >= cfg["cte"]["c1_gain_min"],
            "corrected_errors": corrected >= cfg["cte"]["corrected_errors_min"][dataset],
            "net_correction_each_class": all(value["net_positive"]
                                             for value in per_class.values()),
            "churn_delta": bootstrap["observed"] >= cfg["cte"]["churn_delta_min"],
            "churn_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
            "beats_multiview_both": all(metrics["LABEL_ONLY"][metric] >
                                        metrics["MULTIVIEW"][metric]
                                        for metric in ("accuracy", "macro_f1")),
            "beats_random_both": all(metrics["LABEL_ONLY"][metric] >
                                     metrics["RANDOM"][metric]
                                     for metric in ("accuracy", "macro_f1")),
        }
        dataset_results[dataset] = {
            "metrics": metrics, "LABEL_ONLY_minus_REMOVE": gains,
            "corrected_baseline_errors": corrected, "broken_baseline_correct": broken,
            "per_class_correction": per_class,
            "churn_LABEL_ONLY_minus_RANDOM": bootstrap,
            "support_drift_gates": support_drift_gates, "gates": gates,
            "passed": all(gates.values()),
        }
    mode_gate = len(selected_tuples) == 5 and mode_counts.get(tuple_key(modal_tuple), 0) > 0
    all_pass = mode_gate and all(value["passed"] for value in dataset_results.values())
    decision = "GO" if all_pass else "STOP"
    impl, impl_sha = implementation_hashes()
    meta = gpu_metadata()
    fold_hashes = {dataset: read_json(fold_path(cfg, dataset))["id_to_fold_sha256"]
                   for dataset in DATASETS}
    payload = add_payload_hash({
        "schema_version": 1, "run_id": run_id, "stage": "C1_DECISION",
        "status": decision, "C1_DECISION": decision,
        "reason": "all_binding_dual_dataset_C1_gates_pass" if all_pass
                  else "one_or_more_binding_C1_gates_failed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "git_head": git_head(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": meta["cuda_version"], "gpu_name": meta["gpu_name"],
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": impl_sha, "implementation_files": impl,
        "input_files": input_records(inputs), "fold_ids_sha256": sha256_obj(fold_hashes),
        "checkpoint_sha256": None, "output_files": [],
        "outer_selected_tuples": selected_tuples, "mode_counts": mode_counts,
        "modal_frozen_tuple": modal_tuple if mode_gate else None,
        "mode_gate": mode_gate, "datasets": dataset_results,
        "C2_planning_unlocked": bool(all_pass), "C2_teacher_calls_authorized": False,
        "C3_C4_locked": True,
        "interpretation": "zero_teacher_label_only_action_capacity_only_not_MLLM_success",
        **SUPERVISION,
    })
    acquire_lock(root / ".C1_DECISION.lock", run_id, "C1_DECISION")
    atomic_write_json(root / "C1_DECISION.json", payload)
    print(canonical_json(payload))


def phase_full_freeze(cfg, dataset, run_id):
    require_slurm(gpu=True)
    expected = f"CTE-C1-FULLFREEZE-{dataset}-S0-v1"
    if run_id != expected:
        raise RuntimeError("full-freeze run ID mismatch")
    root = resolve(cfg, "artifact_root")
    decision_path = root / "C1_DECISION.json"
    decision = read_json(decision_path)
    verify_payload(decision)
    if decision.get("C1_DECISION") != "GO":
        raise RuntimeError("full-freeze is locked unless C1 GO")
    out_dir = root / "c1" / "fullfreeze" / dataset
    acquire_lock(out_dir / ".fullfreeze.lock", run_id, "C1_FULLFREEZE")
    bundle = load_train_cache(cfg, dataset)
    subset = select_rows(bundle, sorted(bundle["ids"]))
    seg_path, seg_cache, seg_audit = load_segment_cache(cfg, dataset, bundle, subset)
    initialization, init_record = baseline_train(cfg, dataset, subset, seg_cache, 0)
    init_model = build_model(cfg, dataset, subset["img"].shape[1], subset["txt"].shape[1])
    init_model.load_state_dict(initialization, strict=True)
    support = choose_anchors_and_support(init_model, subset, cfg)
    target_map, target_rows, probe = strict_probe_targets(
        cfg, init_model, subset, support,
        cfg["cte"]["inner_random_state_base"] + 2000, "fulltrain")
    del init_model
    package = train_and_evaluate(
        cfg, dataset, subset, subset, seg_cache, initialization, target_map,
        decision["modal_frozen_tuple"], ("LABEL_ONLY",), "epoch", 0)
    if package["any_first_epoch_drift_failure"]:
        package = train_and_evaluate(
            cfg, dataset, subset, subset, seg_cache, initialization, target_map,
            decision["modal_frozen_tuple"], ("LABEL_ONLY",), "half_epoch", 0)
        if package["any_first_epoch_drift_failure"]:
            raise RuntimeError("full-freeze half-epoch fallback failed")
    result = package["results"]["LABEL_ONLY"]
    # The teacher-before-call identity is selected on the post-C1 checkpoint,
    # not copied from the pre-C1 action-capacity initialization.
    post_model = build_model(cfg, dataset, subset["img"].shape[1], subset["txt"].shape[1])
    post_model.load_state_dict(result["state"], strict=True)
    post_support = choose_anchors_and_support(post_model, subset, cfg)
    if post_support["selected_pair"] is None:
        raise RuntimeError("post-C1 full-train checkpoint has no supported adjacent pair")
    post_directions = fixed_path_identity(post_model, subset, post_support, cfg)
    post_direction_sha = sha256_obj({m: hashlib.sha256(
        post_directions[m].detach().cpu().numpy().tobytes()).hexdigest()
        for m in ("V", "L")})
    post_fixed = support_freeze_dict(post_support)
    del post_model
    checkpoint_path, anchors_path = out_dir / "checkpoint.pt", out_dir / "anchors.json"
    atomic_torch_save(checkpoint_path, result["state"])
    atomic_write_json(anchors_path, add_payload_hash({
        "schema_version": 1, "run_id": run_id, "dataset": dataset,
        "anchor_ids": post_fixed["anchor_ids"],
        "thresholds": post_fixed["thresholds"],
        "selected_pair": post_fixed["selected_pair"],
        "initial_support_details": post_fixed["initial_support_details"],
        "initial_adjacent_candidates": post_fixed["initial_adjacent_candidates"],
        "frozen_direction_sha256": post_direction_sha,
        "anchor_and_radius_selection_checkpoint_sha256": sha256_file(checkpoint_path),
        "target_payload_sha256": sha256_obj(target_rows),
        "teacher_calls_before_freeze": 0,
    }))
    fold_sha = sha256_obj(sorted((vid, row["label"], row["fold"])
                                 for vid, row in bundle["by_id"].items()))
    freeze_path, _ = verify_config_freeze(cfg)
    manifest = base_manifest(
        cfg, run_id, "C1_FULLFREEZE", "COMPLETED",
        [freeze_path, decision_path, bundle["fold_path"], bundle["cache_path"], seg_path],
        [checkpoint_path, anchors_path], fold_sha, sha256_file(checkpoint_path),
        extra={"dataset": dataset, "modal_tuple": decision["modal_frozen_tuple"],
               "anchor_ids": post_fixed["anchor_ids"],
               "radii": post_fixed["selected_pair"],
               "post_C1_anchor_selection": True,
               "initialization": init_record, "segment_cache_audit": seg_audit,
               "probe": probe, "optimizer_steps": result["optimizer_steps"],
               "control_strength_scalars": package["strengths"],
               "aggregate_strength_match_coverage":
                   package["aggregate_strength_match_coverage"],
               "aggregate_first_step_gradient_norms":
                   package["aggregate_first_step_gradient_norms"],
               "label_aggregate_first_step_gradient_norm":
                   package["label_first_step_gradient_norm"],
               "epoch_refresh_ledger": result["refresh_ledger"],
               "support_drift_norm_gradient_diagnostics": {
                   "final_support": result["final_support"],
                   "final_direction": result["final_direction"],
                   "first_gradient": result["first_gradient"]},
               "C2_teacher_calls_authorized": False, "C3_C4_locked": True})
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": run_id,
                          "checkpoint_sha256": sha256_file(checkpoint_path)}))


def phase_verify_freeze(cfg, run_id):
    require_slurm(gpu=False)
    if run_id != "CTE-C1-FREEZE-VERIFY-v1":
        raise RuntimeError("freeze verify run ID mismatch")
    root = resolve(cfg, "artifact_root")
    decision_path = root / "C1_DECISION.json"
    decision = read_json(decision_path)
    verify_payload(decision)
    if decision.get("C1_DECISION") != "GO":
        raise RuntimeError("freeze verification locked unless C1 GO")
    inputs, datasets = [decision_path], {}
    for dataset in DATASETS:
        directory = root / "c1" / "fullfreeze" / dataset
        manifest_path, anchors_path, checkpoint_path = (
            directory / "manifest.json", directory / "anchors.json",
            directory / "checkpoint.pt")
        manifest = read_json(manifest_path)
        fold_records = read_json(fold_path(cfg, dataset))["records"]
        expected_fold_sha = sha256_obj(sorted(
            (str(row["id"]), int(row["label"]), int(row["fold"]))
            for row in fold_records))
        validate_manifest_common(
            manifest, f"CTE-C1-FULLFREEZE-{dataset}-S0-v1", "C1_FULLFREEZE",
            cfg, expected_fold_sha, sha256_file(checkpoint_path),
            [checkpoint_path, anchors_path], [decision_path])
        anchors = read_json(anchors_path)
        verify_payload(anchors)
        if anchors.get("teacher_calls_before_freeze") != 0 or \
                manifest.get("mllm_call_count") != 0:
            raise RuntimeError("teacher-before-freeze violation")
        if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise RuntimeError("full-freeze checkpoint hash mismatch")
        if anchors.get("anchor_and_radius_selection_checkpoint_sha256") != \
                manifest.get("checkpoint_sha256") or \
                anchors.get("anchor_ids") != manifest.get("anchor_ids") or \
                anchors.get("selected_pair") != manifest.get("radii"):
            raise RuntimeError("post-C1 anchor/radius freeze provenance mismatch")
        bundle = load_train_cache(cfg, dataset)
        subset = select_rows(bundle, sorted(bundle["ids"]))
        verify_model = build_model(
            cfg, dataset, subset["img"].shape[1], subset["txt"].shape[1],
            device="cpu")
        verify_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"),
                                     strict=True)
        recomputed = choose_anchors_and_support(
            verify_model, subset, cfg, fixed=None, device="cpu")
        if recomputed["anchor_ids"] != anchors["anchor_ids"] or \
                recomputed["selected_pair"] != anchors["selected_pair"]:
            raise RuntimeError("CPU independent post-C1 anchor/radius recomputation mismatch")
        if recomputed["selected_pair"] is None:
            raise RuntimeError("independent post-C1 geometry has no supported pair")
        inputs.extend([manifest_path, anchors_path, checkpoint_path,
                       bundle["fold_path"], bundle["cache_path"]])
        datasets[dataset] = {"checkpoint_sha256": sha256_file(checkpoint_path),
                             "anchor_ids": anchors["anchor_ids"],
                             "selected_pair": anchors["selected_pair"],
                             "independently_recomputed_on_cpu": True}
    impl, impl_sha = implementation_hashes()
    meta = gpu_metadata()
    fold_hashes = {dataset: read_json(fold_path(cfg, dataset))["id_to_fold_sha256"]
                   for dataset in DATASETS}
    payload = add_payload_hash({
        "schema_version": 1, "run_id": run_id, "stage": "C1_FREEZE_VERIFY",
        "status": "GO", "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_head": git_head(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_version": meta["cuda_version"], "gpu_name": meta["gpu_name"],
        "config_canonical_sha256": cfg["computed_config_sha256"],
        "implementation_sha256": impl_sha, "implementation_files": impl,
        "input_files": input_records(inputs), "fold_ids_sha256": sha256_obj(fold_hashes),
        "checkpoint_sha256": sha256_obj({d: x["checkpoint_sha256"]
                                         for d, x in datasets.items()}),
        "output_files": [], "datasets": datasets,
        "C2_planning_permitted": True, "C2_teacher_calls_authorized": False,
        "C3_C4_locked": True, **SUPERVISION,
    })
    acquire_lock(root / ".C1_FREEZE_VERIFY.lock", run_id, "C1_FREEZE_VERIFY")
    atomic_write_json(root / "C1_FREEZE_VERIFY.json", payload)
    print(canonical_json(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", required=True,
                        choices=("inner", "select", "outer", "decide",
                                 "full-freeze", "verify-freeze"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.phase in ("inner", "outer"):
        if args.dataset is None or args.outer_fold is None:
            raise RuntimeError(f"{args.phase} requires dataset and outer-fold")
        if args.phase == "inner":
            phase_inner(cfg, args.dataset, args.outer_fold, args.run_id)
        else:
            phase_outer(cfg, args.dataset, args.outer_fold, args.run_id)
    elif args.phase == "full-freeze":
        if args.dataset is None or args.outer_fold is not None:
            raise RuntimeError("full-freeze requires dataset only")
        phase_full_freeze(cfg, args.dataset, args.run_id)
    else:
        if args.dataset is not None or args.outer_fold is not None:
            raise RuntimeError(f"{args.phase} takes no dataset/fold")
        if args.phase == "select":
            phase_select(cfg, args.run_id)
        elif args.phase == "decide":
            phase_decide(cfg, args.run_id)
        else:
            phase_verify_freeze(cfg, args.run_id)


if __name__ == "__main__":
    main()
