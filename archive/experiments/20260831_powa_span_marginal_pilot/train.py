#!/usr/bin/env python3
"""Train a frozen-POWA context-quotient span-marginal residual."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from powa_macil.dataset import (  # noqa: E402
    PowaTestDataset,
    PowaTrainDataset,
    usable_text_ids,
)
from macilsd.utils import uniform_extract  # noqa: E402
from src.powa_residual import (  # noqa: E402
    FrozenPowaContextResidual,
    load_corpus_powa,
    safe_logit,
    sha256,
)


ARMS = ("span_marginal", "singleton", "shuffled_span")
SPAN_LENGTHS = (3, 5, 9, 17, 33)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   choices=("hatemm", "hateclipseg"))
    p.add_argument("--anchor-checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--negative-dense-weight", type=float, default=1.0)
    p.add_argument("--residual-l2-weight", type=float, default=0.01)
    p.add_argument("--pooled-tolerance", type=float, default=0.002)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-train-videos", type=int, default=0,
                   help="Development smoke only; formal runs keep zero")
    p.add_argument("--limit-val-videos", type=int, default=0,
                   help="Development smoke only; formal runs keep zero")
    return p


def mask_from_lengths(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths[:, None]


def normalized_span_marginal(values, arm, temperature, generator=None):
    if values.ndim != 1 or not values.numel():
        raise ValueError("span marginal requires a non-empty 1-D sequence")
    if arm == "shuffled_span":
        permutation = torch.randperm(
            values.numel(), generator=generator, device="cpu"
        ).to(values.device)
        values = values[permutation]
    lengths = (1,) if arm == "singleton" else SPAN_LENGTHS
    lengths = sorted({min(int(length), values.numel()) for length in lengths})
    candidates = []
    sequence = values[None, None]
    for length in lengths:
        means = F.avg_pool1d(sequence, kernel_size=length, stride=1).flatten()
        candidates.append(means)
    candidates = torch.cat(candidates)
    return temperature * (
        torch.logsumexp(candidates / temperature, dim=0)
        - math.log(candidates.numel())
    )


def centered_local_logit(anchor_logit, residual):
    """Local ordering axis used by the weak video-label objective."""
    return anchor_logit - anchor_logit.mean() + residual


class IdentifiedPowaTrainDataset(PowaTrainDataset):
    """POWA item plus stable identity for auditable shuffle controls."""

    def __getitem__(self, index):
        item = super().__getitem__(index)
        return (*item, self.video_ids[index // self.crop_repeat],
                index % self.crop_repeat)


def shuffle_seed(seed, epoch, video_id, crop):
    payload = f"{seed}|{epoch}|{video_id}|{crop}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def uniform_model_grid(f_v, f_a, f_t, max_seqlen):
    """Apply the exact training-time uniform/pad rule to a dense test video."""
    native_length = int(f_v.shape[1])
    if native_length > max_seqlen:
        index = torch.from_numpy(
            np.linspace(0, native_length - 1, max_seqlen,
                        dtype=np.uint16).astype(np.int64)
        ).to(f_v.device)
        return (f_v.index_select(1, index), f_a.index_select(1, index),
                f_t.index_select(1, index), max_seqlen, index.cpu().numpy())
    padding = max_seqlen - native_length
    if padding:
        f_v = F.pad(f_v, (0, 0, 0, padding))
        f_a = F.pad(f_a, (0, 0, 0, padding))
        f_t = F.pad(f_t, (0, 0, 0, padding))
    return (f_v, f_a, f_t, native_length,
            np.arange(native_length, dtype=np.int64))


def residual_to_native(residual, sample_index, native_length):
    """Interpolate a coarse residual to the native grid and re-quotient it."""
    rows = []
    target = np.arange(native_length, dtype=np.float64)
    for crop in residual.detach().cpu().numpy():
        if len(sample_index) == native_length:
            dense = crop[:native_length].astype(np.float64, copy=True)
        else:
            dense = np.interp(target, sample_index, crop[:len(sample_index)])
        dense -= dense.mean()
        rows.append(dense)
    return np.stack(rows)


def build_data(args, cfg):
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    train_ids = usable_text_ids(args.corpus, train_ids)
    val_ids = usable_text_ids(args.corpus, val_ids)
    if args.limit_train_videos:
        train_ids = train_ids[:args.limit_train_videos]
    if args.limit_val_videos:
        val_ids = val_ids[:args.limit_val_videos]
    dataset = IdentifiedPowaTrainDataset(
        args.corpus, train_ids, labels, cfg.max_seqlen, cfg.grid,
        "av", cfg.crop_repeat,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator,
    )
    val_dataset = PowaTestDataset(
        args.corpus, val_ids, cfg.max_seqlen, cfg.grid, "av"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    return loader, val_loader, {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_positive": sum(labels[video_id] == 1 for video_id in train_ids),
        "train_negative": sum(labels[video_id] == 0 for video_id in train_ids),
    }


def train_epoch(model, loader, optimizer, args, epoch):
    model.train()
    totals = {"loss": 0.0, "bag": 0.0, "negative_dense": 0.0,
              "l2": 0.0, "batches": 0}
    for batch_index, (f_v, f_a, f_t, label, video_ids, crops) in enumerate(loader):
        lengths = _seq_len_of(f_v).long().to(args.device)
        width = int(lengths.max())
        valid = mask_from_lengths(lengths, width, args.device)
        output = model(
            f_a[:, :width].float().to(args.device),
            f_v[:, :width].float().to(args.device),
            f_t[:, :width].float().to(args.device),
            lengths, valid, args.corpus,
        )
        labels = label.float().to(args.device)
        bag_terms = []
        dense_terms = []
        l2_terms = []
        for item in range(len(labels)):
            count = int(lengths[item])
            residual = output["residual"][item, :count]
            local_logit = centered_local_logit(
                output["anchor_logit"][item, :count], residual
            )
            rng = torch.Generator(device="cpu").manual_seed(shuffle_seed(
                args.seed, epoch, video_ids[item], int(crops[item])
            ))
            bag_logit = normalized_span_marginal(
                local_logit, args.arm, args.temperature, rng
            )
            bag_terms.append(F.binary_cross_entropy_with_logits(
                bag_logit, labels[item]
            ))
            if labels[item] < 0.5:
                dense_terms.append(F.binary_cross_entropy_with_logits(
                    local_logit, torch.zeros_like(local_logit)
                ))
            l2_terms.append(residual.square().mean())
        bag_loss = torch.stack(bag_terms).mean()
        dense_loss = (
            torch.stack(dense_terms).mean() if dense_terms
            else torch.zeros((), device=args.device)
        )
        l2_loss = torch.stack(l2_terms).mean()
        loss = (
            bag_loss + args.negative_dense_weight * dense_loss
            + args.residual_l2_weight * l2_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite span marginal loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.residual_head.parameters(), 5.0
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite residual gradient")
        optimizer.step()
        for key, value in (("loss", loss), ("bag", bag_loss),
                           ("negative_dense", dense_loss), ("l2", l2_loss)):
            totals[key] += float(value.detach())
        totals["batches"] += 1
        totals.setdefault("grad_norm", 0.0)
        totals.setdefault("nonzero_grad_batches", 0)
        totals["grad_norm"] += float(grad_norm)
        totals["nonzero_grad_batches"] += int(float(grad_norm) > 0)
    n = max(1, totals["batches"])
    if totals.get("nonzero_grad_batches", 0) == 0:
        raise RuntimeError("residual head received zero gradient for the epoch")
    return {
        key: value if key in ("batches", "nonzero_grad_batches") else value / n
        for key, value in totals.items()
    }


def fixed_orders(length):
    index = np.arange(length, dtype=np.float64)
    center = (length - 1) / 2
    return {
        "chronological": index,
        "reverse_chronological": -index,
        "edge_first": np.abs(index - center),
        "center_first": -np.abs(index - center),
    }


def assign_values(values, order):
    output = np.empty_like(values)
    output[np.argsort(order, kind="stable")] = np.sort(values, kind="stable")
    return output


def summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


@torch.no_grad()
def validate(model, loader, corpus, device, split="val"):
    model.eval()
    branches = {"score_powa": {}, "score_candidate": {},
                "score_residual_only": {}}
    for name in fixed_orders(2):
        branches[f"score_{name}"] = {}
    score_rows = []
    grid_mean_errors = []
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        native_length = int(f_v.shape[1])
        dense_lengths = torch.full(
            (f_v.shape[0],), native_length, dtype=torch.long, device=device
        )
        dense_valid = torch.ones(
            (f_v.shape[0], native_length), dtype=torch.bool, device=device
        )
        with torch.no_grad():
            dense_raw = model.powa(
                f_a, f_v, f_t, dense_lengths, dense_valid, policy=corpus
            )
        anchor_tensor = dense_raw["frame_prob"].mean(0)
        c_v, c_a, c_t, coarse_length, sample_index = uniform_model_grid(
            f_v, f_a, f_t, model.max_seqlen
        )
        lengths = torch.full(
            (c_v.shape[0],), coarse_length, dtype=torch.long, device=device
        )
        valid = mask_from_lengths(lengths, c_v.shape[1], device)
        output = model(c_a, c_v, c_t, lengths, valid, corpus)
        dense_residual = residual_to_native(
            output["residual"][:, :coarse_length], sample_index, native_length
        )
        grid_mean_errors.append(float(np.abs(dense_residual.mean(1)).max()))
        residual_tensor = torch.from_numpy(dense_residual.mean(0)).to(
            anchor_tensor.device, anchor_tensor.dtype
        )
        index = index_map[0].numpy()
        anchor = anchor_tensor.cpu().numpy()[index]
        residual = residual_tensor.cpu().numpy()[index]
        if np.count_nonzero(residual) == 0:
            candidate = anchor.copy()
        else:
            candidate = torch.sigmoid(
                safe_logit(anchor_tensor) + residual_tensor
            ).cpu().numpy()[index]
        if len(candidate) != int(n_seconds):
            raise RuntimeError(f"validation alignment mismatch {video_id}")
        branches["score_powa"][video_id] = anchor
        branches["score_candidate"][video_id] = candidate
        branches["score_residual_only"][video_id] = 1 / (1 + np.exp(-residual))
        anchor_logit = np.log(np.clip(anchor, 1e-5, 1 - 1e-5)) - np.log1p(
            -np.clip(anchor, 1e-5, 1 - 1e-5)
        )
        for name, order in fixed_orders(len(residual)).items():
            assigned = assign_values(residual, order)
            branches[f"score_{name}"][video_id] = 1 / (
                1 + np.exp(-(anchor_logit + assigned))
            )
        score_rows.append({
            "video_id": video_id, "n_frames": len(anchor),
            **{name: values[video_id].tolist()
               for name, values in branches.items()},
        })
    gt = hdata.gt_arrays(corpus, split)
    labels = hdata.load_labels(corpus)
    positives = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {
        name: evaluate_scores(scores, gt, positives)
        for name, scores in branches.items()
    }
    anchor_auc = reports["score_powa"]["per_video"]["per_video_auc"]
    candidate_auc = reports["score_candidate"]["per_video"]["per_video_auc"]
    shared = sorted(set(anchor_auc) & set(candidate_auc))
    deltas = np.asarray([candidate_auc[v] - anchor_auc[v] for v in shared])
    high = [v for v in shared if np.asarray(gt[v]).mean() > 0.6]
    return {
        "metrics": {name: summary(report) for name, report in reports.items()},
        "paired_within": {
            "n": len(shared),
            "mean_delta": float(deltas.mean()),
            "improvement_ratio": float(np.mean(deltas > 0)),
            "high_pos": {
                "n": len(high),
                "anchor": float(np.mean([anchor_auc[v] for v in high]))
                if high else None,
                "candidate": float(np.mean([candidate_auc[v] for v in high]))
                if high else None,
            },
        },
        "max_abs_grid_residual_mean": max(grid_mean_errors),
        "score_rows": score_rows,
    }


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_scores(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def main(argv=None):
    args = parser().parse_args(argv)
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.device = runtime.resolve_device(args.device)
    save_json(args.out_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    runtime.setup_seed(args.seed)
    powa, cfg, anchor_meta, anchor_hash = load_corpus_powa(
        args.anchor_checkpoint, args.corpus, args.device
    )
    model = FrozenPowaContextResidual(
        powa, cfg.text_feature_size, cfg.hid_dim, cfg.max_seqlen
    ).to(args.device)
    loader, val_loader, splits = build_data(args, cfg)
    optimizer = torch.optim.AdamW(
        model.residual_head.parameters(), lr=args.lr,
        weight_decay=args.weight_decay,
    )
    baseline = validate(model, val_loader, args.corpus, args.device)
    baseline_rows = baseline.pop("score_rows")
    if baseline["metrics"]["score_candidate"] != baseline["metrics"]["score_powa"]:
        raise RuntimeError("epoch-0 candidate is not exact POWA identity")
    write_scores(args.out_dir / "val_scores_epoch0.jsonl", baseline_rows)
    history = [{"epoch": 0, "validation": baseline, "feasible": True}]
    anchor_metric = baseline["metrics"]["score_powa"]
    best_key = best_epoch = best_state = best_rows = None
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        training = train_epoch(model, loader, optimizer, args, epoch)
        validation = validate(model, val_loader, args.corpus, args.device)
        rows = validation.pop("score_rows")
        metric = validation["metrics"]["score_candidate"]
        feasible = (
            metric["pooled_ap"] >= anchor_metric["pooled_ap"] - args.pooled_tolerance
            and metric["pooled_roc"]
            >= anchor_metric["pooled_roc"] - args.pooled_tolerance
        )
        record = {"epoch": epoch, "train": training,
                  "validation": validation, "feasible": feasible,
                  "seconds": time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if feasible:
            key = (metric["within_roc"], metric["pooled_ap"], -epoch)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                best_state = copy.deepcopy(model.residual_head.state_dict())
                best_rows = rows
    selected = None
    if best_state is not None:
        torch.save(best_state, args.out_dir / "residual_head.pth")
        write_scores(args.out_dir / "val_scores.jsonl", best_rows)
        selected = next(
            row["validation"] for row in history if row["epoch"] == best_epoch
        )
    meta = {
        "method": "powa_context_quotient_span_marginal",
        "corpus": args.corpus, "arm": args.arm, "seed": args.seed,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "anchor_checkpoint": str(Path(args.anchor_checkpoint).resolve()),
        "anchor_model_sha256": anchor_hash,
        "anchor_train_meta_sha256": sha256(
            Path(args.anchor_checkpoint).resolve() / "train_meta.json"
        ),
        "anchor_selected_epoch": anchor_meta.get("selected_epoch"),
        "powa_parameters_trainable": 0,
        "cross_corpus_training": False,
        "test_labels_used": False,
        "shuffle_seed_rule": "sha256(seed|epoch|video_id|crop)[:8]",
        "splits": splits,
        "pilot_plan": str((HERE / "PILOT_PLAN.md").resolve()),
        "pilot_plan_sha256": sha256(HERE / "PILOT_PLAN.md"),
        "history": history,
        "selected_epoch": best_epoch,
        "selected_key": list(best_key) if best_key is not None else None,
        "selected_validation": selected,
    }
    save_json(args.out_dir / "train_meta.json", meta)
    print(json.dumps({"selected_epoch": best_epoch,
                      "selected_key": best_key}), flush=True)


if __name__ == "__main__":
    main()
