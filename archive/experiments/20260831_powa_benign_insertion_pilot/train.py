#!/usr/bin/env python3
"""Train the preregistered benign-insertion POWA pilot."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from powa_macil.dataset import (PowaTestDataset, load_teacher_jsonl,
                                usable_text_ids)  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402
from powa_macil.train import (compute_powa_loss, parser as powa_parser)  # noqa: E402

from dataset import BenignInsertionDataset, INSERTION_ARMS  # noqa: E402


ARMS = ("matched_powa", "splice_only", "original_negative_dense",
        "insertion_benign", "full", "positive_donor")


def parser():
    p = powa_parser()
    p.description = __doc__
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--min-donor-rows", type=int, default=12)
    p.add_argument("--max-donor-rows", type=int, default=36)
    p.add_argument("--boundary-buffer", type=int, default=3)
    p.add_argument("--donor-bce-weight", type=float, default=1.0)
    p.add_argument("--composite-mil-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=0.5)
    return p


def _mask(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths[:, None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data(args, corpus):
    labels = hdata.load_labels(corpus)
    train_ids, val_ids = hdata.load_train_val(corpus, labels)
    train_ids = usable_text_ids(corpus, train_ids)
    val_ids = usable_text_ids(corpus, val_ids)
    teacher = load_teacher_jsonl(args.teacher_file) if args.teacher_file else None
    train_set = BenignInsertionDataset(
        corpus, train_ids, labels, args.max_seqlen, args.grid, "av",
        args.crop_repeat, teacher_records=teacher,
        permute_teacher_channels=args.ablation == "teacher_permutation",
        arm=args.arm, seed=args.seed,
        min_donor_rows=args.min_donor_rows,
        max_donor_rows=args.max_donor_rows,
        boundary_buffer=args.boundary_buffer)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False)
    val_set = PowaTestDataset(corpus, val_ids, args.max_seqlen,
                              args.grid, "av")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    return train_set, train_loader, val_loader, {
        "corpus": corpus,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "negative_train_ids": train_set.negative_ids,
    }


def _write_augmentation_records(handle, batch, corpus):
    if handle is None:
        return
    count = len(batch["recipient_id"])
    for i in range(count):
        if not bool(batch["has_insertion"][i]):
            continue
        row = {
            "corpus": corpus,
            "split": "train",
            "recipient_id": batch["recipient_id"][i],
            "donor_id": batch["donor_id"][i],
            "donor_crop": int(batch["donor_crop"][i]),
            "epoch": int(batch["epoch"][i]),
            "crop": int(batch["crop"][i]),
            "insert_at": int(batch["insert_at"][i]),
            "donor_start": int(batch["donor_start"][i]),
            "donor_rows": int(batch["donor_rows"][i]),
            "donor_supervision_interval_aug": [
                int(batch["insert_at"][i] + batch["boundary_buffer"][i]),
                int(batch["insert_at"][i] + batch["donor_rows"][i] -
                    batch["boundary_buffer"][i]),
            ],
            "consistency_excluded_interval_recipient": [
                max(0, int(batch["insert_at"][i] -
                           batch["boundary_buffer"][i])),
                min(int(batch["orig_length"][i]),
                    int(batch["insert_at"][i] +
                        batch["boundary_buffer"][i])),
            ],
        }
        handle.write(json.dumps(row) + "\n")


def train_epoch(model, dataset, loader, optimizer, args, corpus, epoch,
                manifest_handle):
    model.train()
    if args.freeze_macil:
        model.macil.eval()
    dataset.set_epoch(epoch)
    names = ("loss", "original", "composite_mil", "donor_bce",
             "consistency", "original_negative_dense")
    totals = {name: 0.0 for name in names}
    totals["batches"] = 0
    totals["insertions"] = 0
    for batch in loader:
        _write_augmentation_records(manifest_handle, batch, corpus)
        y = batch["label"].float().to(args.device)
        orig_lengths = batch["orig_length"].long().to(args.device)
        orig_keep = int(orig_lengths.max())
        orig_valid = _mask(orig_lengths, orig_keep, args.device)
        orig_v = batch["orig_v"][:, :orig_keep].float().to(args.device)
        orig_a = batch["orig_a"][:, :orig_keep].float().to(args.device)
        orig_t = batch["orig_t"][:, :orig_keep].float().to(args.device)
        teacher_target = batch["teacher_target"][:, :orig_keep].to(args.device)
        teacher_mask = batch["teacher_mask"][:, :orig_keep].to(args.device)
        original = model(orig_a, orig_v, orig_t, orig_lengths, orig_valid,
                         policy=corpus)
        original_loss, _ = compute_powa_loss(
            original, y, orig_valid, args, teacher_target, teacher_mask)
        total = original_loss
        composite_mil = torch.zeros((), device=args.device)
        donor_bce = torch.zeros((), device=args.device)
        consistency = torch.zeros((), device=args.device)
        original_negative_dense = torch.zeros((), device=args.device)

        if args.arm == "original_negative_dense":
            negative = (y == 0)[:, None] & orig_valid
            if negative.any():
                original_negative_dense = F.binary_cross_entropy(
                    original["frame_prob"][negative],
                    torch.zeros_like(original["frame_prob"][negative]))
                total = total + args.donor_bce_weight * original_negative_dense

        has_insertion = batch["has_insertion"].bool().to(args.device)
        if args.arm in INSERTION_ARMS and has_insertion.any():
            aug_lengths = batch["aug_length"].long().to(args.device)
            aug_keep = int(aug_lengths.max())
            aug_valid = _mask(aug_lengths, aug_keep, args.device)
            aug_v = batch["aug_v"][:, :aug_keep].float().to(args.device)
            aug_a = batch["aug_a"][:, :aug_keep].float().to(args.device)
            aug_t = batch["aug_t"][:, :aug_keep].float().to(args.device)
            augmented = model(aug_a, aug_v, aug_t, aug_lengths, aug_valid,
                              policy=corpus)
            composite_mil = F.binary_cross_entropy(
                augmented["bag_prob"][has_insertion],
                torch.ones_like(augmented["bag_prob"][has_insertion]))
            total = total + args.composite_mil_weight * composite_mil

            if args.arm in ("insertion_benign", "full", "positive_donor"):
                donor_mask = (batch["donor_mask"][:, :aug_keep].to(
                    args.device) > 0) & has_insertion[:, None]
                if donor_mask.any():
                    donor_bce = F.binary_cross_entropy(
                        augmented["frame_prob"][donor_mask],
                        torch.zeros_like(augmented["frame_prob"][donor_mask]))
                    total = total + args.donor_bce_weight * donor_bce

            if args.arm in ("full", "positive_donor"):
                recipient_map = batch["recipient_map"][:, :orig_keep].long().to(
                    args.device)
                mapped = torch.gather(augmented["frame_prob"], 1,
                                      recipient_map)
                keep = (batch["consistency_mask"][:, :orig_keep].to(
                    args.device) > 0)
                keep = keep & orig_valid & has_insertion[:, None]
                if keep.any():
                    consistency = F.mse_loss(
                        mapped[keep], original["frame_prob"].detach()[keep])
                    total = total + args.consistency_weight * consistency

        if not torch.isfinite(total):
            raise FloatingPointError("non-finite candidate loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        values = {
            "loss": total,
            "original": original_loss,
            "composite_mil": composite_mil,
            "donor_bce": donor_bce,
            "consistency": consistency,
            "original_negative_dense": original_negative_dense,
        }
        for name, value in values.items():
            totals[name] += float(value.detach())
        totals["batches"] += 1
        totals["insertions"] += int(has_insertion.sum())
    batches = max(1, totals["batches"])
    return {name: (value if name in ("batches", "insertions")
                   else value / batches)
            for name, value in totals.items()}


@torch.no_grad()
def validate(model, loader, corpus, device):
    model.eval()
    scores = {}
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long,
                             device=device)
        valid = torch.ones((f_v.shape[0], f_v.shape[1]), dtype=torch.bool,
                           device=device)
        out = model(f_a, f_v, f_t, lengths, valid, policy=corpus)
        dense = out["frame_prob"].mean(0).cpu().numpy()
        dense = dense[index_map[0].numpy()]
        if len(dense) != int(n_seconds):
            raise RuntimeError(f"validation alignment mismatch {video_id}")
        scores[video_id] = dense
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    hate_ids = {video_id for video_id in gt if labels[video_id] == 1}
    report = evaluate_scores(scores, gt, hate_ids)
    if report["n_videos_missing_from_scores"] or report["n_videos_not_in_gold"]:
        raise RuntimeError("validation score coverage mismatch")
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if len(args.corpora) != 1:
        raise SystemExit("exactly one --corpora value is required")
    corpus = args.corpora[0]
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = HERE / "PILOT_PLAN.md"

    dataset, loader, val_loader, split_manifest = build_data(args, corpus)
    model = POWAMACIL(args, policy=corpus).to(args.device)
    model.use_policy_residual = not args.typed_only
    if not args.macil_init:
        raise ValueError("pilot requires a corpus-specific --macil-init")
    model.macil.load_state_dict(torch.load(args.macil_init,
                                           map_location=args.device))
    if args.freeze_macil:
        for parameter in model.macil.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_epoch)
    best_key = None
    best_epoch = None
    history = []
    manifest_path = out_dir / "augmentation_manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for epoch in range(1, args.max_epoch + 1):
            started = time.time()
            losses = train_epoch(model, dataset, loader, optimizer, args,
                                 corpus, epoch, manifest)
            scheduler.step()
            metrics = validate(model, val_loader, corpus, args.device)
            record = {
                "epoch": epoch,
                "train": losses,
                "validation": metrics,
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            key = (metrics["within_roc"], metrics["pooled_ap"], -epoch)
            improves_within = (best_key is None or
                               key[0] > best_key[0] + 1e-6)
            within_tie = (best_key is not None and
                          abs(key[0] - best_key[0]) <= 1e-6)
            improves_tie = within_tie and (key[1], key[2]) > (
                best_key[1], best_key[2])
            if improves_within or improves_tie:
                best_key = key
                best_epoch = epoch
                temporary = out_dir / "model.pth.tmp"
                torch.save(copy.deepcopy(model.state_dict()), temporary)
                temporary.replace(out_dir / "model.pth")

    meta = {
        "method": "powa_negative_bag_certified_benign_insertion",
        "arm": args.arm,
        "corpus": corpus,
        "seed": args.seed,
        "args": vars(args),
        "splits": split_manifest,
        "selected_epoch": best_epoch,
        "selected_key": list(best_key),
        "selected_metric": "validation within-video ROC; AP tie-break",
        "history": history,
        "pilot_plan": str(plan_path),
        "pilot_plan_sha256": _sha256(plan_path),
        "test_labels_used_for_training_or_selection": False,
        "cross_corpus_training": False,
    }
    temporary = out_dir / "train_meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    temporary.replace(out_dir / "train_meta.json")
    print(json.dumps({"selected_epoch": best_epoch,
                      "selected_key": best_key}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
