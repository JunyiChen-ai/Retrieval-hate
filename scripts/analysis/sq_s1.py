#!/usr/bin/env python
"""Learned strict-OOF SQ-0 screen, guarded by the verified S0 decision.

Only outer-train parent-video labels and train-only q records enter training.
Outer-held queries are full-video endpoint-only and their q is never read.
The evaluator signature has no archive/q/teacher argument.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from sq_common import (  # noqa: E402
    acquire_namespace, base_cluster_q, base_manifest, canonical_json,
    exact_ranking, exclusive_write_json, exclusive_write_jsonl, input_record,
    load_config, make_shuffle_q, metrics_from_predictions, output_records,
    random_matched_q, read_json, read_jsonl, require_runtime, resolve,
    sha256_file, sha256_obj, sha256_text, sq_loss_for_batch, sq_sampling_plan,
)

ARMS = ("REMOVE", "BASE_CLUSTER", "LABEL_ONLY", "FULL", "SHUFFLE", "RANDOM")
DATASETS = ("MHC", "MHC_zh")


def expected_outer_id(dataset, fold):
    return "SQ-S1-OOF-{}-F{}-S0-v1".format(dataset, fold)


def load_train_q(cfg, dataset, memory_ids):
    """Read only q rows for outer-train IDs; reject any outer-held request."""
    path = resolve(cfg, "artifacts") / "s0/qproxy" / dataset / "posterior.jsonl"
    rows = read_jsonl(path)
    by = {str(x["id"]): x for x in rows}
    if len(by) != len(rows):
        raise RuntimeError("duplicate q IDs")
    missing = [x for x in memory_ids if x not in by]
    if missing:
        raise RuntimeError("outer-train q missing: {}".format(missing[:10]))
    # Deliberately do not project any non-memory row.
    q = np.asarray([by[x]["q"] for x in memory_ids], dtype=np.float64)
    r = np.asarray([by[x]["r"] for x in memory_ids], dtype=np.float64)
    return path, q, r


def evaluator_full_video_only(memory_ids, memory_z, memory_y,
                              query_ids, query_z, query_y, fold, topk):
    """Ordinary exact cosine kNN. No q/archive/teacher parameter exists."""
    predictions, neighbors = [], []
    for i, qid in enumerate(query_ids):
        rows, vote, pred, denom = exact_ranking(
            memory_ids, memory_z, memory_y, query_z[i], topk=topk)
        predictions.append({"query_id": qid, "query_label": int(query_y[i]),
                            "outer_fold": fold, "prediction": pred,
                            "vote": vote, "vote_abs_denom": denom,
                            "baseline_error": int(pred != int(query_y[i]))})
        neighbors.append({"query_id": qid, "query_label": int(query_y[i]),
                          "outer_fold": fold, "top20": rows})
    return predictions, neighbors


def task_outer(cfg, dataset, fold, run_id):
    if run_id != expected_outer_id(dataset, fold):
        raise RuntimeError("nonfrozen run-id")
    s0_path = resolve(cfg, "artifacts") / "S0_DECISION.json"
    if not s0_path.is_file():
        raise RuntimeError("S0 decision missing")
    s0 = read_json(s0_path)
    if s0.get("status") != "GO" or not s0.get("S1_unlocked"):
        raise RuntimeError("S1 fail-closed: verified S0 GO absent")
    lambda_q = s0.get("lambda_Q")
    if lambda_q not in cfg["sq"]["lambda_candidates"]:
        raise RuntimeError("S0 did not freeze legal lambda_Q")

    import torch
    from torch.utils.data import DataLoader
    from data_loader.rac_dataloader import RACDataset
    from model.classifier import classifier_hateClipper
    from model.loss import compute_loss
    from ssr_oof import load_train_cache, make_segment_cache, take_dataset, train_args

    folds_path = resolve(cfg, "ssr_artifacts") / "folds" / (dataset + ".json")
    folds = read_json(folds_path); records = folds["records"]
    query_ids = sorted(str(x["id"]) for x in records if int(x["fold"]) == fold)
    memory_ids = sorted(str(x["id"]) for x in records if int(x["fold"]) != fold)
    if set(query_ids) & set(memory_ids) or set(query_ids) | set(memory_ids) != {
            str(x["id"]) for x in records}:
        raise RuntimeError("invalid outer partition")
    out_root = resolve(cfg, "artifacts") / "s1/oof" / dataset / ("fold{}".format(fold))
    acquire_namespace(out_root, run_id)

    ssr_cfg = dict(cfg); ssr_cfg["comparator"] = cfg["training"]
    cache_path, full_ids, img, txt, labels = load_train_cache(ssr_cfg, dataset)
    frozen = folds["split_assertions"]["clip_cache"]["train"]
    if sha256_file(cache_path) != frozen["sha256"]:
        raise RuntimeError("train cache changed")
    label_by = {str(x["id"]): int(x["label"]) for x in records}
    if set(full_ids) != set(label_by) or any(int(labels[i]) != label_by[x]
                                             for i, x in enumerate(full_ids)):
        raise RuntimeError("frozen parent-video labels disagree")
    memory, memory_idx = take_dataset(full_ids, img, txt, labels, memory_ids)
    query, _ = take_dataset(full_ids, img, txt, labels, query_ids)
    seg_path, segment_cache, seg_manifest = make_segment_cache(
        ssr_cfg, dataset, full_ids, labels, memory, memory_idx)
    if seg_manifest["label_source"] != "inherited_parent_video_label_not_segment_gold":
        raise RuntimeError("unexpected subclip supervision contract")
    if sha256_file(seg_path) != folds["split_assertions"]["subclip_cache"]["sha256"]:
        raise RuntimeError("subclip cache changed")
    qpath, full_q, full_r = load_train_q(cfg, dataset, memory_ids)
    y_memory = np.asarray(memory[3], dtype=np.int64)

    args = train_args(ssr_cfg, dataset)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    template = classifier_hateClipper(
        int(img.shape[1]), int(txt.shape[1]), cfg["training"]["num_layers"],
        cfg["training"]["proj_dim"], cfg["training"]["map_dim"],
        cfg["training"]["fusion_mode"], dropout=cfg["training"]["dropout"],
        batch_norm=cfg["training"]["batch_norm"], args=args).cuda()
    initial_state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    init_hash = sha256_obj({k: sha256_text(v.numpy().tobytes().hex())
                            for k, v in initial_state.items()})
    del template

    # Arm records are derived only inside this outer-train partition.
    base_init_z = np.load(resolve(cfg, "ssr_artifacts") / "oof" / dataset /
                          ("fold{}".format(fold)) / "embeddings.npz")["memory_z"]
    bc_q, bc_r, _ = base_cluster_q(base_init_z)
    sh_q, sh_r, sh_perm = make_shuffle_q(
        memory_ids, y_memory, full_q, full_r, cfg["sq"]["shuffle_seed"])
    rd_q, rd_r, rd_diag = random_matched_q(
        memory_ids, full_q, full_r, cfg["sq"]["random_seed"])
    arm_records = {
        "BASE_CLUSTER": (bc_q, bc_r, False),
        "LABEL_ONLY": (full_q, full_r, True),
        "FULL": (full_q, full_r, False),
        "SHUFFLE": (sh_q, sh_r, False),
        "RANDOM": (rd_q, rd_r, False),
    }
    micro = read_json(resolve(cfg, "artifacts") / "s0/micro" / dataset / "timings.json")
    fold_strength = micro["control_folds"][fold]["strength_scalars"]
    arm_strength = {"FULL": 1.0, "REMOVE": 0.0,
                    **{k: float(v) for k, v in fold_strength.items()}}

    @torch.no_grad()
    def project(model, data):
        model.eval(); out = []
        for start in range(0, len(data[0]), 256):
            _, z = model(data[1][start:start + 256].cuda(),
                         data[2][start:start + 256].cuda(), return_embed=True)
            out.append(z.detach().cpu())
        return torch.cat(out).numpy().astype(np.float32)

    arm_metrics = {}
    for arm in ARMS:
        arm_dir = out_root / arm
        acquire_namespace(arm_dir, run_id + "-" + arm)
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = classifier_hateClipper(
            int(img.shape[1]), int(txt.shape[1]), cfg["training"]["num_layers"],
            cfg["training"]["proj_dim"], cfg["training"]["map_dim"],
            cfg["training"]["fusion_mode"], dropout=cfg["training"]["dropout"],
            batch_norm=cfg["training"]["batch_norm"], args=args).cuda()
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
        generator = torch.Generator(); generator.manual_seed(0)
        train_set = RACDataset((memory[1], memory[2]), memory[0], memory[3])
        train_dl = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=0, generator=generator)
        history = []; epoch_bank_hashes = []; q_ids_read = set()
        epochs = cfg["datasets"][dataset]["epoch_index"] + 1
        for epoch in range(epochs):
            bank_z_np = project(model, memory)
            bank_z = torch.as_tensor(bank_z_np, device="cuda")
            epoch_bank_hashes.append(sha256_text(bank_z_np.tobytes().hex()))
            plans = None; plan_stats = {"active_anchors": 0, "total_anchors": len(memory_ids)}
            if arm != "REMOVE":
                aq, ar, label_only = arm_records[arm]
                plans, plan_stats = sq_sampling_plan(
                    memory_ids, y_memory, bank_z_np, aq, ar,
                    cfg["computed_config_sha256"], cfg["sq"]["seed"], epoch,
                    cfg["sq"]["triplets_per_anchor"], cfg["sq"]["min_kish_ess"],
                    cfg["evaluator"]["topk"], label_only=label_only)
                if arm in {"FULL", "SHUFFLE", "RANDOM"}:
                    q_ids_read.update(memory_ids)
            train_feats = train_labels = None; order = []; losses = []; aux_losses = []
            for batch in train_dl:
                ids_batch = [str(x) for x in batch["ids"]]; order.extend(ids_batch)
                result = compute_loss(
                    batch, train_dl, model, args, train_set=train_set,
                    train_feats=train_feats, train_labels=train_labels,
                    segment_cache=segment_cache, aux_pack=None, cf_pack=None)
                base_loss, train_feats, train_labels = result[0], result[-2], result[-1]
                if torch.is_tensor(train_feats): train_feats = train_feats.detach()
                if torch.is_tensor(train_labels): train_labels = train_labels.detach()
                aux_value = base_loss * 0.0
                if arm != "REMOVE":
                    _, anchor_z = model(batch["image_feats"].cuda(),
                                        batch["text_feats"].cuda(), return_embed=True)
                    aux_value, _ = sq_loss_for_batch(
                        ids_batch, anchor_z, bank_z, plans, cfg["sq"]["margin"],
                        cfg["sq"]["temperature"])
                total = base_loss + float(lambda_q) * arm_strength[arm] * aux_value
                if not torch.isfinite(total):
                    raise RuntimeError("nonfinite loss")
                optimizer.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                optimizer.step()
                losses.append(float(total.detach().cpu())); aux_losses.append(float(aux_value.detach().cpu()))
            history.append({"epoch_index": epoch, "steps": len(losses),
                            "mean_total_loss": float(np.mean(losses)),
                            "mean_aux_loss": float(np.mean(aux_losses)),
                            "batch_order_sha256": sha256_obj(order),
                            "epoch_bank_sha256": epoch_bank_hashes[-1],
                            "active_anchors": plan_stats["active_anchors"],
                            "fallback_anchors": plan_stats["total_anchors"] - plan_stats["active_anchors"],
                            "triplet_plan_sha256": plan_stats.get("triplet_plan_sha256")})
        ckpt = arm_dir / "checkpoint.pt"; torch.save(model.state_dict(), ckpt)
        mem_z = project(model, memory); qry_z = project(model, query)
        predictions, neighbors = evaluator_full_video_only(
            memory_ids, mem_z, np.asarray(memory[3]), query_ids, qry_z,
            np.asarray(query[3]), fold, cfg["evaluator"]["topk"])
        ppath = arm_dir / "predictions.json"; npath = arm_dir / "neighbors.jsonl"
        tpath = arm_dir / "training.jsonl"
        exclusive_write_json(ppath, predictions); exclusive_write_jsonl(npath, neighbors)
        exclusive_write_jsonl(tpath, history)
        met = metrics_from_predictions([x["query_label"] for x in predictions],
                                       [x["prediction"] for x in predictions])
        arm_metrics[arm] = met
        outputs = [ckpt, ppath, npath, tpath]
        manifest = base_manifest(
            cfg, run_id + "-" + arm, "S1_OUTER_ARM", "COMPLETED",
            inputs=[input_record(cache_path), input_record(seg_path),
                    input_record(folds_path), input_record(s0_path)] +
                   ([input_record(qpath)] if arm != "REMOVE" else []),
            outputs=output_records(outputs), gpu_name=torch.cuda.get_device_name(0),
            extra={"dataset": dataset, "outer_fold": fold, "arm": arm,
                   "initialization_sha256": init_hash, "fixed_epoch_index": epochs - 1,
                   "epochs_trained": epochs, "optimizer_steps": sum(x["steps"] for x in history),
                   "epoch_bank_hashes": epoch_bank_hashes,
                   "q_ids_read_sha256": sha256_obj(sorted(q_ids_read)),
                   "q_id_read_count": len(q_ids_read), "outer_held_q_read_count": 0,
                   "query_n": len(query_ids), "memory_n": len(memory_ids),
                   "query_memory_overlap": [], "metrics": met,
                   "evaluator_reject_log": {"q_argument_supported": False,
                                            "archive_argument_supported": False,
                                            "teacher_argument_supported": False},
                   "segment_gold_exists": False, "segment_gold_used": False})
        exclusive_write_json(arm_dir / "manifest.json", manifest)
        del model; torch.cuda.empty_cache()
    summary = {"run_id": run_id, "status": "COMPLETED", "dataset": dataset,
               "outer_fold": fold, "arm_order": list(ARMS), "arm_metrics": arm_metrics,
               "identical_initialization_sha256": init_hash,
               "outer_held_q_read_count": 0, "new_teacher_call_count": 0,
               "segment_gold_exists": False, "segment_gold_used": False,
               "shuffle_permutation_sha256": sha256_obj(sh_perm.tolist()),
               "random_calibration": rd_diag}
    exclusive_write_json(out_root / "fold_summary.json", summary)
    print(canonical_json({"run_id": run_id, "status": "COMPLETED", "arm_metrics": arm_metrics}))


def bootstrap_metric_delta(rows_a, rows_b, metric, reps, seed):
    by_a = {x["query_id"]: x for x in rows_a}; by_b = {x["query_id"]: x for x in rows_b}
    ids = sorted(by_a)
    if set(ids) != set(by_b): raise RuntimeError("prediction ID mismatch")
    strata = defaultdict(list)
    for vid in ids:
        x = by_a[vid]; strata[(x["outer_fold"], x["query_label"])].append(vid)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        sample = []
        for v in strata.values(): sample.extend(rng.choice(v, len(v), replace=True).tolist())
        ya = [by_a[x]["query_label"] for x in sample]
        pa = [by_a[x]["prediction"] for x in sample]; pb = [by_b[x]["prediction"] for x in sample]
        vals.append(metrics_from_predictions(ya, pa)[metric] - metrics_from_predictions(ya, pb)[metric])
    return {"lower_95": float(np.percentile(vals, 2.5)),
            "upper_95": float(np.percentile(vals, 97.5)),
            "p_one_sided": float((1 + sum(x <= 0 for x in vals)) / (reps + 1))}


def task_decide(cfg, run_id):
    if run_id != "SQ-S1-DECISION-v1": raise RuntimeError("nonfrozen run-id")
    s0 = read_json(resolve(cfg, "artifacts") / "S0_DECISION.json")
    if s0.get("status") != "GO" or not s0.get("S1_unlocked"):
        raise RuntimeError("S1 decision fail-closed: S0 GO absent")
    out = resolve(cfg, "artifacts") / "S1_DECISION.json"
    if out.exists(): raise RuntimeError("refusing overwrite {}".format(out))
    datasets = {}; all_inputs = []; primary_p = []
    for d in DATASETS:
        arm_rows = {a: [] for a in ARMS}; fold_metrics = {a: [] for a in ARMS}
        for f in range(5):
            for a in ARMS:
                ad = resolve(cfg, "artifacts") / "s1/oof" / d / ("fold{}".format(f)) / a
                man = read_json(ad / "manifest.json")
                if (man["initialization_sha256"] != read_json(resolve(cfg, "artifacts") / "s1/oof" / d / ("fold{}".format(f)) / "REMOVE/manifest.json")["initialization_sha256"] or
                        man["outer_held_q_read_count"] != 0 or man["new_teacher_call_count"] != 0):
                    raise RuntimeError("fold manifest audit failed")
                rows = read_json(ad / "predictions.json"); arm_rows[a].extend(rows)
                fold_metrics[a].append(metrics_from_predictions(
                    [x["query_label"] for x in rows], [x["prediction"] for x in rows]))
                all_inputs.extend([input_record(ad / "manifest.json"), input_record(ad / "predictions.json"),
                                   input_record(ad / "neighbors.jsonl")])
        metrics = {a: metrics_from_predictions([x["query_label"] for x in arm_rows[a]],
                                                [x["prediction"] for x in arm_rows[a]]) for a in ARMS}
        gates = {}; boots = {}
        for metric in ("accuracy", "macro_f1"):
            comparator = max(("REMOVE", "BASE_CLUSTER"), key=lambda a: metrics[a][metric])
            delta = metrics["FULL"][metric] - metrics[comparator][metric]
            fold_signs = [fold_metrics["FULL"][f][metric] - fold_metrics[comparator][f][metric] for f in range(5)]
            boot = bootstrap_metric_delta(arm_rows["FULL"], arm_rows[comparator], metric,
                                          cfg["evaluator"]["bootstrap_replicates"],
                                          cfg["evaluator"]["bootstrap_seed"] + (0 if metric == "accuracy" else 1))
            primary_p.append(boot["p_one_sided"])
            controls = {a: metrics["FULL"][metric] - metrics[a][metric]
                        for a in ("LABEL_ONLY", "SHUFFLE", "RANDOM")}
            control_boot = {a: bootstrap_metric_delta(
                arm_rows["FULL"], arm_rows[a], metric,
                cfg["evaluator"]["bootstrap_replicates"],
                cfg["evaluator"]["bootstrap_seed"] + 10 + i)
                for i, a in enumerate(("LABEL_ONLY", "SHUFFLE", "RANDOM"))}
            gates[metric] = {"moving_comparator": comparator, "delta": delta,
                             "fold_deltas": fold_signs, "control_deltas": controls,
                             "pass": bool(delta >= 0.05 and all(x > 0 for x in fold_signs) and
                                          boot["lower_95"] > 0 and
                                          all(x >= 0.01 for x in controls.values()) and
                                          all(x["lower_95"] > 0 for x in control_boot.values()))}
            boots[metric] = {"moving": boot, "controls": control_boot}
        # Wrong-class positive signed mass and corrected/broken by video class.
        comparator = gates["accuracy"]["moving_comparator"]
        def signed_mass_rows(arm):
            rows = []
            for f in range(5):
                p = resolve(cfg, "artifacts") / "s1/oof" / d / ("fold{}".format(f)) / arm / "neighbors.jsonl"
                rows.extend(read_jsonl(p))
            return {x["query_id"]: sum(n["weight"] * max(n["cosine"], 0.0)
                    for n in x["top20"] if n["label"] != x["query_label"]) for x in rows}
        mf = signed_mass_rows("FULL"); mc = signed_mass_rows(comparator)
        mass_reduction = float(np.mean([mc[x] - mf[x] for x in mf]))
        pred_full = {x["query_id"]: x for x in arm_rows["FULL"]}
        pred_comp = {x["query_id"]: x for x in arm_rows[comparator]}
        net = {}
        corrected_ids = []
        for c in (0, 1):
            ids = [x for x in pred_full if pred_full[x]["query_label"] == c]
            corrected = [x for x in ids if pred_comp[x]["baseline_error"] and not pred_full[x]["baseline_error"]]
            broken = [x for x in ids if not pred_comp[x]["baseline_error"] and pred_full[x]["baseline_error"]]
            net[str(c)] = len(corrected) - len(broken); corrected_ids.extend(corrected)
        # Fail closed if the old-universe diagnostic cannot be independently established.
        reach_beyond = {"status": "MISSING_STOP", "outside_union_corrected_minus_broken": None}
        dataset_pass = all(x["pass"] for x in gates.values()) and mass_reduction > 0 and all(x > 0 for x in net.values()) and reach_beyond["status"] == "PASS"
        datasets[d] = {"metrics": metrics, "gates": gates, "bootstrap": boots,
                       "wrong_class_signed_mass_reduction": mass_reduction,
                       "net_corrected_minus_broken_by_class": net,
                       "corrected_ids": corrected_ids, "reach_beyond": reach_beyond,
                       "pass": dataset_pass}
    # Holm FWER on four primary tests.
    order = sorted(range(4), key=lambda i: primary_p[i]); holm = [False] * 4
    for rank, idx in enumerate(order):
        if primary_p[idx] <= 0.05 / (4 - rank): holm[idx] = True
        else: break
    go = all(x["pass"] for x in datasets.values()) and all(holm)
    decision = base_manifest(
        cfg, run_id, "S1_DECISION", "GO" if go else "STOP", inputs=all_inputs,
        extra={"datasets": datasets, "primary_p_values": primary_p,
               "holm_rejections": holm, "SQ-0": "GO" if go else "STOP",
               "S2_unlocked": bool(go), "S2_S4_execution_authorized": False,
               "q_signal_status": s0["q_signal_status"],
               "claim_boundary": "action-capacity only" if s0["q_signal_status"] == "PROXY_ONLY_CHEAP_FORMAT" else "archive-weak-MLLM",
               "segment_gold_exists": False, "segment_gold_used": False})
    exclusive_write_json(out, decision)
    print(canonical_json({"run_id": run_id, "status": decision["status"],
                          "S2_unlocked": decision["S2_unlocked"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=["outer", "decide"])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--outer-fold", type=int, choices=range(5))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(); require_runtime(gpu=args.task == "outer")
    cfg = load_config(args.config)
    if args.task == "outer":
        if args.dataset is None or args.outer_fold is None: ap.error("outer needs dataset/fold")
        task_outer(cfg, args.dataset, args.outer_fold, args.run_id)
    else: task_decide(cfg, args.run_id)


if __name__ == "__main__":
    main()
