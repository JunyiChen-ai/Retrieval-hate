#!/usr/bin/env python3
"""Train the preregistered POWA frozen-score temporal assignment pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
DUPLEX = REPO / "scripts" / "duplex"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(DUPLEX))

import frame_eval_common as fec  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402
from src.weak_supervision.same_corpus_insertion import (  # noqa: E402
    INSERTION_ARMS,
    SameCorpusInsertionDataset,
)

from model import FrozenPowaTemporalAssignment  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    p.add_argument("--anchor-checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--arm", required=True, choices=sorted(INSERTION_ARMS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--stability-weight", type=float, default=0.5)
    p.add_argument("--topk-divisor", type=int, default=16)
    p.add_argument("--min-donor-rows", type=int, default=12)
    p.add_argument("--max-donor-rows", type=int, default=36)
    p.add_argument("--boundary-buffer", type=int, default=3)
    p.add_argument("--pooled-tolerance", type=float, default=0.002)
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_from_lengths(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths[:, None]


def load_anchor(checkpoint: Path, corpus: str, device: str):
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    cfg = SimpleNamespace(**meta["args"])
    if list(getattr(cfg, "corpora", [corpus])) != [corpus]:
        raise ValueError("anchor checkpoint is not corpus-specific")
    powa = POWAMACIL(cfg, policy=corpus).to(device)
    state_path = checkpoint / "model.pth"
    state = torch.load(state_path, map_location=device)
    legacy_typed_only = "policy_residual_gate" not in state
    powa.load_state_dict(state, strict=not legacy_typed_only)
    powa.use_policy_residual = (
        not legacy_typed_only and not getattr(cfg, "typed_only", False)
    )
    powa.eval()
    return powa, cfg, meta, sha256(state_path)


def build_data(args, cfg):
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    train_ids = usable_text_ids(args.corpus, train_ids)
    val_ids = usable_text_ids(args.corpus, val_ids)
    train_set = SameCorpusInsertionDataset(
        args.corpus,
        train_ids,
        labels,
        cfg.max_seqlen,
        cfg.grid,
        "av",
        cfg.crop_repeat,
        arm=args.arm,
        seed=args.seed,
        min_donor_rows=args.min_donor_rows,
        max_donor_rows=args.max_donor_rows,
        boundary_buffer=args.boundary_buffer,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_set = PowaTestDataset(
        args.corpus, val_ids, cfg.max_seqlen, cfg.grid, "av"
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    return train_set, train_loader, val_loader, {
        "corpus": args.corpus,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "negative_train_ids": train_set.negative_ids,
        "positive_train_ids": train_set.positive_ids,
        "negative_donor_ids": train_set.negative_donor_ids,
        "positive_donor_ids": train_set.positive_donor_ids,
    }


def _nonzero_interval(mask: torch.Tensor):
    positions = torch.nonzero(mask > 0, as_tuple=False).flatten()
    if not len(positions):
        return None
    return [int(positions[0]), int(positions[-1]) + 1]


def write_manifest(handle, batch, corpus, arm):
    for index, recipient in enumerate(batch["recipient_id"]):
        if not bool(batch["has_insertion"][index]):
            continue
        handle.write(
            json.dumps(
                {
                    "corpus": corpus,
                    "split": "train",
                    "recipient_id": recipient,
                    "donor_id": batch["donor_id"][index],
                    "donor_crop": int(batch["donor_crop"][index]),
                    "epoch": int(batch["epoch"][index]),
                    "crop": int(batch["crop"][index]),
                    "insert_at": int(batch["insert_at"][index]),
                    "donor_start": int(batch["donor_start"][index]),
                    "donor_rows": int(batch["donor_rows"][index]),
                    "donor_interior_aug": _nonzero_interval(
                        batch["donor_mask"][index]
                    ),
                    "supervision_interval_aug": _nonzero_interval(
                        batch["supervision_mask"][index]
                    ),
                    "stability_excluded_interval_recipient": [
                        max(
                            0,
                            int(batch["insert_at"][index])
                            - int(batch["boundary_buffer"][index]),
                        ),
                        min(
                            int(batch["orig_length"][index]),
                            int(batch["insert_at"][index])
                            + int(batch["boundary_buffer"][index]),
                        ),
                    ],
                    "has_stability_support": bool(
                        batch["has_stability_support"][index]
                    ),
                    "recipient_map_aug": batch["recipient_map"][index][
                        : int(batch["orig_length"][index])
                    ].tolist(),
                    "arm": arm,
                }
            )
            + "\n"
        )


def topk_mean(values: torch.Tensor, divisor: int):
    k = max(1, values.numel() // divisor + 1)
    return values.topk(k).values.mean()


def train_epoch(model, dataset, loader, optimizer, args, epoch, manifest):
    model.train()
    dataset.set_epoch(epoch)
    totals = {"loss": 0.0, "order": 0.0, "stability": 0.0,
              "batches": 0, "insertions": 0,
              "skipped_no_recipient_support": 0,
              "stability_exceptions": 0}
    for batch in loader:
        write_manifest(manifest, batch, args.corpus, args.arm)
        has_insertion = batch["has_insertion"].bool().to(args.device)
        if not has_insertion.any():
            continue
        orig_lengths = batch["orig_length"].long().to(args.device)
        orig_width = int(orig_lengths.max())
        orig_valid = mask_from_lengths(orig_lengths, orig_width, args.device)
        original = model(
            batch["orig_a"][:, :orig_width].float().to(args.device),
            batch["orig_v"][:, :orig_width].float().to(args.device),
            batch["orig_t"][:, :orig_width].float().to(args.device),
            orig_lengths,
            orig_valid,
            args.corpus,
        )
        aug_lengths = batch["aug_length"].long().to(args.device)
        aug_width = int(aug_lengths.max())
        aug_valid = mask_from_lengths(aug_lengths, aug_width, args.device)
        augmented = model(
            batch["aug_a"][:, :aug_width].float().to(args.device),
            batch["aug_v"][:, :aug_width].float().to(args.device),
            batch["aug_t"][:, :aug_width].float().to(args.device),
            aug_lengths,
            aug_valid,
            args.corpus,
        )
        recipient_map = batch["recipient_map"][:, :orig_width].long().to(
            args.device
        )
        stability_mask = (
            batch["stability_mask"][:, :orig_width].to(args.device) > 0
        ) & orig_valid
        supervision_mask = (
            batch["supervision_mask"][:, :aug_width].to(args.device) > 0
        ) & aug_valid

        order_terms = []
        stability_terms = []
        for item in torch.nonzero(has_insertion, as_tuple=False).flatten():
            item = int(item)
            recipient_keep = orig_valid[item]
            recipient_indices = torch.nonzero(
                recipient_keep, as_tuple=False
            ).flatten()
            if recipient_indices.numel() == 0 or not supervision_mask[item].any():
                totals["skipped_no_recipient_support"] += 1
                continue
            frozen_order = original["anchor_logit"][item, recipient_indices]
            k_recipient = max(
                1, recipient_indices.numel() // args.topk_divisor + 1
            )
            candidate_local = frozen_order.topk(k_recipient).indices
            candidate_orig = recipient_indices[candidate_local]
            candidate_aug = recipient_map[item, candidate_orig]
            recipient_order = augmented["order_logit"][item, candidate_aug].mean()
            donor_order = topk_mean(
                augmented["order_logit"][item, supervision_mask[item]],
                args.topk_divisor,
            )
            order_terms.append(
                F.softplus(args.margin - recipient_order + donor_order)
            )

            stable_keep = stability_mask[item]
            if stable_keep.any():
                mapped_order = augmented["order_logit"][item].gather(
                    0, recipient_map[item]
                )[stable_keep]
                original_order = original["order_logit"][item, stable_keep]
                mapped_centered = mapped_order - mapped_order.mean()
                original_centered = original_order - original_order.mean()
                stability_terms.append(
                    F.smooth_l1_loss(mapped_centered, original_centered)
                )
            else:
                totals["stability_exceptions"] += 1

        if not order_terms:
            continue
        order_loss = torch.stack(order_terms).mean()
        stability_loss = (
            torch.stack(stability_terms).mean()
            if stability_terms
            else torch.zeros((), device=args.device)
        )
        loss = order_loss + args.stability_weight * stability_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite temporal assignment loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.order_head.parameters(), 5.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["order"] += float(order_loss.detach())
        totals["stability"] += float(stability_loss.detach())
        totals["batches"] += 1
        totals["insertions"] += len(order_terms)
    batches = max(1, totals["batches"])
    return {
        key: (
            value
            if key in (
                "batches", "insertions", "skipped_no_recipient_support",
                "stability_exceptions",
            )
            else value / batches
        )
        for key, value in totals.items()
    }


def stable_transport(values, order, tie_secondary=None):
    values = np.asarray(values, dtype=np.float64)
    order = np.asarray(order, dtype=np.float64)
    if values.ndim != 1 or values.shape != order.shape:
        raise ValueError("transport inputs must be aligned 1-D arrays")
    if tie_secondary is None:
        temporal_order = np.argsort(order, kind="stable")
    else:
        temporal_order = np.lexsort((np.asarray(tie_secondary), order))
    output = np.empty_like(values)
    output[temporal_order] = np.sort(values, kind="stable")
    return output


def _fixed_orders(length, video_id):
    index = np.arange(length, dtype=np.float64)
    center = (length - 1) / 2.0
    seed = int.from_bytes(
        hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    return {
        "random_permutation": rng.random(length),
        "chronological": index,
        "reverse_chronological": -index,
        "edge_first": np.abs(index - center),
        "center_first": -np.abs(index - center),
    }


def _metric_summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_median": report["per_video"]["macro_auc_median"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def _evaluate_branches(branches, corpus):
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    positive_ids = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {
        name: evaluate_scores(scores, gt, positive_ids)
        for name, scores in branches.items()
    }
    for name, report in reports.items():
        if report["n_videos_missing_from_scores"] or report["n_videos_not_in_gold"]:
            raise RuntimeError(f"validation coverage mismatch in {name}")
    anchor_auc = reports["score_powa"]["per_video"]["per_video_auc"]
    candidate_auc = reports["score_rank_transport"]["per_video"]["per_video_auc"]
    shared_ids = sorted(set(anchor_auc) & set(candidate_auc))
    deltas = np.asarray(
        [candidate_auc[v] - anchor_auc[v] for v in shared_ids], dtype=float
    )
    strata = {}
    for name, low, high in (
        ("le_0.2", 0.0, 0.2),
        ("0.2_to_0.6", 0.2, 0.6),
        ("gt_0.6", 0.6, 1.0),
    ):
        ids = []
        for video_id in shared_ids:
            fraction = float(np.asarray(gt[video_id]).mean())
            if (name == "le_0.2" and fraction <= high) or (
                name == "0.2_to_0.6" and low < fraction <= high
            ) or (name == "gt_0.6" and fraction > low):
                ids.append(video_id)
        strata[name] = {
            "n": len(ids),
            "anchor_within": (
                float(np.mean([anchor_auc[v] for v in ids])) if ids else None
            ),
            "candidate_within": (
                float(np.mean([candidate_auc[v] for v in ids])) if ids else None
            ),
        }
    ordering_diagnostics = {}
    for branch_name in ("score_powa", "score_rank_transport",
                        "score_direct_additive"):
        scores = branches[branch_name]
        hateful_frames = []
        positive_benign_frames = []
        negative_video_frames = []
        edge_auc = []
        interior_auc = []
        for video_id, target in gt.items():
            target = np.asarray(target)
            score = np.asarray(scores[video_id])
            if labels[video_id] == 1:
                hateful_frames.append(score[target.astype(bool)])
                positive_benign_frames.append(score[~target.astype(bool)])
                count = len(target)
                boundary = max(1, int(np.ceil(0.2 * count)))
                edge = np.zeros(count, dtype=bool)
                edge[:boundary] = True
                edge[-boundary:] = True
                for region, sink in ((edge, edge_auc), (~edge, interior_auc)):
                    auc = fec.rank_roc_auc(score[region], target[region])
                    if auc is not None:
                        sink.append(auc)
            else:
                negative_video_frames.append(score)
        hateful = np.concatenate(hateful_frames)
        positive_benign = np.concatenate(positive_benign_frames)
        negative = np.concatenate(negative_video_frames)
        ordering_diagnostics[branch_name] = {
            "hate_vs_positive_video_benign_auc": fec.rank_auc(
                hateful, positive_benign
            ),
            "hate_vs_negative_video_frames_auc": fec.rank_auc(
                hateful, negative
            ),
            "edge_macro_auc": float(np.mean(edge_auc)) if edge_auc else None,
            "edge_n": len(edge_auc),
            "interior_macro_auc": (
                float(np.mean(interior_auc)) if interior_auc else None
            ),
            "interior_n": len(interior_auc),
        }
    return {
        "metrics": {name: _metric_summary(report) for name, report in reports.items()},
        "paired_within": {
            "n": len(shared_ids),
            "mean_delta": float(deltas.mean()) if len(deltas) else None,
            "median_delta": float(np.median(deltas)) if len(deltas) else None,
            "improvement_ratio": float(np.mean(deltas > 0)) if len(deltas) else None,
            "strata": strata,
        },
        "ordering_diagnostics": ordering_diagnostics,
    }


@torch.no_grad()
def validate(model, loader, corpus, device):
    model.eval()
    branches = {
        "score_powa": {},
        "score_rank_transport": {},
        "score_direct_additive": {},
        "score_order_raw": {},
        "score_tie_reverse": {},
        "score_tie_random": {},
        "score_random_permutation": {},
        "score_chronological": {},
        "score_reverse_chronological": {},
        "score_edge_first": {},
        "score_center_first": {},
    }
    invariant_errors = []
    zero_residual_identity_errors = []
    zero_residual_videos = 0
    unique_ratios = []
    spearman = []
    kendall = []
    score_rows = []
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        lengths = torch.full(
            (f_v.shape[0],), f_v.shape[1], dtype=torch.long, device=device
        )
        valid = torch.ones(
            (f_v.shape[0], f_v.shape[1]), dtype=torch.bool, device=device
        )
        output = model(f_a, f_v, f_t, lengths, valid, corpus)
        index = index_map[0].numpy()
        anchor_tensor = output["anchor_prob"].mean(0)
        residual_tensor = output["order_residual"].mean(0)
        order_tensor = (
            torch.log(anchor_tensor.clamp(1e-5, 1.0 - 1e-5))
            - torch.log1p(-anchor_tensor.clamp(1e-5, 1.0 - 1e-5))
            + residual_tensor
        )
        anchor = anchor_tensor.cpu().numpy()[index]
        order = order_tensor.cpu().numpy()[index]
        direct = torch.sigmoid(order_tensor).cpu().numpy()[index]
        if len(anchor) != int(n_seconds):
            raise RuntimeError(f"validation alignment mismatch {video_id}")
        residual_second = residual_tensor.cpu().numpy()[index]
        if np.count_nonzero(residual_second) == 0:
            # Make the preregistered epoch-0 identity bit-exact instead of
            # relying on finite-precision logit monotonicity for near ties.
            transported = anchor.copy()
            zero_residual_videos += 1
            zero_residual_identity_errors.append(
                float(np.max(np.abs(transported - anchor)))
            )
        else:
            transported = stable_transport(anchor, order)
        branches["score_powa"][video_id] = anchor
        branches["score_rank_transport"][video_id] = transported
        branches["score_direct_additive"][video_id] = direct
        branches["score_order_raw"][video_id] = order
        reverse_secondary = -np.arange(len(order), dtype=np.float64)
        random_secondary = _fixed_orders(len(order), video_id)["random_permutation"]
        branches["score_tie_reverse"][video_id] = stable_transport(
            anchor, order, reverse_secondary
        )
        branches["score_tie_random"][video_id] = stable_transport(
            anchor, order, random_secondary
        )
        for name, fixed_order in _fixed_orders(len(order), video_id).items():
            branches[f"score_{name}"][video_id] = stable_transport(
                anchor, fixed_order
            )
        invariant_errors.append(
            float(np.max(np.abs(np.sort(anchor) - np.sort(transported))))
        )
        unique_ratios.append(float(len(np.unique(order)) / len(order)))
        spearman.append(float(stats.spearmanr(anchor, order).statistic))
        kendall.append(float(stats.kendalltau(anchor, order).statistic))
        score_rows.append({
            "video_id": video_id,
            "n_frames": len(anchor),
            **{
                name: np.asarray(values[video_id]).tolist()
                for name, values in branches.items()
            },
        })
    report = _evaluate_branches(branches, corpus)
    report["invariants"] = {
        "per_video_second_score_multiset_max_abs_error": max(invariant_errors),
        "exact_float64": max(invariant_errors) == 0.0,
        "mean_order_unique_ratio": float(np.mean(unique_ratios)),
        "mean_anchor_order_spearman": float(np.nanmean(spearman)),
        "mean_anchor_order_kendall": float(np.nanmean(kendall)),
        "zero_residual_videos": zero_residual_videos,
        "zero_residual_pointwise_identity_max_abs_error": (
            max(zero_residual_identity_errors)
            if zero_residual_identity_errors else None
        ),
    }
    report["score_rows"] = score_rows
    return report


def _save_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    temporary.replace(path)


def _write_scores(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.anchor_checkpoint = args.anchor_checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    plan_path = HERE / "PILOT_PLAN.md"
    powa, cfg, anchor_meta, anchor_sha = load_anchor(
        args.anchor_checkpoint, args.corpus, args.device
    )
    model = FrozenPowaTemporalAssignment(
        powa, text_dim=cfg.text_feature_size, hidden=cfg.hid_dim
    ).to(args.device)
    dataset, loader, val_loader, split_manifest = build_data(args, cfg)
    optimizer = torch.optim.AdamW(
        model.order_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )

    baseline_report = validate(model, val_loader, args.corpus, args.device)
    if (
        baseline_report["invariants"]["zero_residual_videos"]
        != len(val_loader.dataset)
        or baseline_report["invariants"][
            "zero_residual_pointwise_identity_max_abs_error"
        ] != 0.0
    ):
        raise RuntimeError("epoch-0 assignment is not exact POWA identity")
    _write_scores(
        args.out_dir / "val_scores_epoch0.jsonl", baseline_report.pop("score_rows")
    )
    baseline = baseline_report["metrics"]["score_powa"]
    history = [{"epoch": 0, "validation": baseline_report, "feasible": True}]
    best_key = None
    best_epoch = None
    best_state = None
    best_rows = None
    manifest_path = args.out_dir / "augmentation_manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            losses = train_epoch(
                model, dataset, loader, optimizer, args, epoch, manifest
            )
            scheduler.step()
            validation = validate(model, val_loader, args.corpus, args.device)
            score_rows = validation.pop("score_rows")
            candidate = validation["metrics"]["score_rank_transport"]
            feasible = (
                candidate["pooled_ap"] >= baseline["pooled_ap"] - args.pooled_tolerance
                and candidate["pooled_roc"]
                >= baseline["pooled_roc"] - args.pooled_tolerance
            )
            record = {
                "epoch": epoch,
                "train": losses,
                "validation": validation,
                "feasible": feasible,
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            print(json.dumps(record, default=float), flush=True)
            if feasible:
                key = (
                    candidate["within_roc"],
                    candidate["pooled_ap"],
                    -epoch,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.order_head.state_dict())
                    best_rows = score_rows

    if best_state is not None:
        temporary = args.out_dir / "rank_head.pth.tmp"
        torch.save(best_state, temporary)
        temporary.replace(args.out_dir / "rank_head.pth")
        _write_scores(args.out_dir / "val_scores.jsonl", best_rows)
        selected_validation = next(
            row["validation"] for row in history if row["epoch"] == best_epoch
        )
        _save_json(args.out_dir / "val_metrics.json", selected_validation)
    else:
        selected_validation = None

    meta = {
        "method": "powa_frozen_score_temporal_assignment",
        "arm": args.arm,
        "corpus": args.corpus,
        "seed": args.seed,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "anchor_checkpoint": str(args.anchor_checkpoint),
        "anchor_model_sha256": anchor_sha,
        "anchor_selected_epoch": anchor_meta.get("selected_epoch"),
        "anchor_corpus_specific": True,
        "splits": split_manifest,
        "selected_epoch": best_epoch,
        "selected_key": list(best_key) if best_key is not None else None,
        "selection": (
            "val pooled AP/ROC each >= epoch0 - 0.002; then max val within ROC, "
            "AP tie-break, earlier epoch"
        ),
        "selected_validation": selected_validation,
        "history": history,
        "pilot_plan": str(plan_path),
        "pilot_plan_sha256": sha256(plan_path),
        "test_labels_used_for_training_or_selection": False,
        "cross_corpus_training": False,
        "powa_parameters_trainable": 0,
        "rank_head_parameters": sum(
            parameter.numel() for parameter in model.order_head.parameters()
        ),
    }
    _save_json(args.out_dir / "train_meta.json", meta)
    print(
        json.dumps({"selected_epoch": best_epoch, "selected_key": best_key}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
