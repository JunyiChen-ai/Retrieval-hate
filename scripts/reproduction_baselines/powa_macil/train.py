#!/usr/bin/env python3
"""Joint official-validation training for POWA-MACIL.

The default is a deliberately cheap no-MLLM feasibility run.  It trains one
 shared model across corpora and never reads test gold. It can select on the
official validation frame annotations; those annotations are used only for
model selection, never as a training target. MLLM primitive distillation will only be
enabled after this structural pilot justifies its cost.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd.train import _seq_len_of, _stratified_head  # noqa: E402
from powa_macil.dataset import (PowaTestDataset, PowaTrainDataset,
                                load_teacher_jsonl, usable_text_ids)  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402

ALL_CORPORA = tuple(hdata.CORPORA)


def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    runtime.add_common_args(p)
    p.add_argument("--corpora", nargs="+", default=list(ALL_CORPORA),
                   choices=list(ALL_CORPORA))
    p.add_argument("--out-dir", default="results/reproduction/powa_macil/pilot")
    p.add_argument("--max-epoch", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--max-seqlen", type=int, default=200)
    p.add_argument("--crop-repeat", type=int, default=1,
                   help="pilot=1; confirmation should use MACIL's five crops")
    p.add_argument("--grid", default="snippet", choices=["snippet", "second"])
    p.add_argument("--hid-dim", type=int, default=128)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--a-feature-size", type=int, default=128)
    p.add_argument("--v-feature-size", type=int, default=1024)
    p.add_argument("--text-feature-size", type=int, default=768)
    p.add_argument("--binding-window", type=int, default=12)
    p.add_argument("--binding-temperature", type=float, default=0.2)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--base-loss-weight", type=float, default=0.25)
    p.add_argument("--sparsity-weight", type=float, default=0.002)
    p.add_argument("--teacher-file", default=None)
    p.add_argument("--teacher-weight", type=float, default=0.5)
    p.add_argument("--semantic-grounding", action="store_true")
    p.add_argument("--semantic-prototype-file", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "results",
        "reproduction", "powa_macil", "semantic_prototypes.npz"))
    p.add_argument("--semantic-strength", type=float, default=0.5)
    p.add_argument("--semantic-temperature", type=float, default=0.07)
    p.add_argument("--grounding-weight", type=float, default=0.1)
    p.add_argument("--macil-init", default=None,
                   help="official MACIL-SD AV state_dict used as the backbone")
    p.add_argument("--multi-backbone", action="store_true",
                   help="one corpus-specific MACIL backbone with shared POWA modules")
    p.add_argument("--macil-init-root", default=None,
                   help="root containing <corpus>/seed_<seed>/model.pth for multi-backbone training")
    p.add_argument("--freeze-macil", action="store_true")
    p.add_argument("--typed-only", action="store_true",
                   help="use the compiled typed witness as the dense score, without a MACIL-logit residual")
    p.add_argument("--residual-mode", default="signed",
                   choices=["signed", "positive_evidence"],
                   help="signed log-odds or open-world positive-only policy evidence")
    p.add_argument("--ablation", default="full", choices=[
        "full", "flat_fusion", "pointwise", "same_time",
        "anonymous_head", "policy_permutation", "semantic_permutation",
        "teacher_permutation"])
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--selection", default="mean_frame_ap",
                   choices=["mean_frame_ap", "mean_video_ap"])
    p.add_argument("--val-gt-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
        "results", "reproduction", "gt"))
    return p


def _mask(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths.to(device)[:, None]


def build_loaders(args):
    train, val = {}, {}
    split_manifest = {}
    teacher = load_teacher_jsonl(args.teacher_file) if args.teacher_file else None
    for corpus in args.corpora:
        labels = hdata.load_labels(corpus)
        train_ids, val_ids = hdata.load_train_val(corpus, labels)
        train_ids = usable_text_ids(corpus, train_ids)
        val_ids = usable_text_ids(corpus, val_ids)
        if args.limit_videos:
            train_ids = _stratified_head(train_ids, labels, args.limit_videos)
            val_ids = _stratified_head(val_ids, labels,
                                       max(2, args.limit_videos // 4))
        ds = PowaTrainDataset(corpus, train_ids, labels, args.max_seqlen,
                              args.grid, "av", args.crop_repeat,
                              teacher_records=teacher,
                              permute_teacher_channels=(
                                  args.ablation == "teacher_permutation"))
        train[corpus] = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=False)
        vds = PowaTestDataset(corpus, val_ids, args.max_seqlen, args.grid, "av")
        val[corpus] = DataLoader(vds, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers)
        split_manifest[corpus] = {"train_ids": train_ids, "val_ids": val_ids}
    return train, val, split_manifest


def train_epoch(model, loaders, optimizer, args, device):
    model.train()
    if args.freeze_macil:
        (model.macils if args.multi_backbone else model.macil).eval()
    totals = {"loss": 0.0, "witness_bce": 0.0, "base_bce": 0.0,
              "sparsity": 0.0, "teacher": 0.0, "grounding": 0.0,
              "batches": 0}
    # Round-robin prevents the largest corpus from monopolising the tail of an
    # epoch and makes every optimiser cycle see all moderation policies.
    iters = {c: iter(x) for c, x in loaders.items()}
    active = list(loaders)
    while active:
        for corpus in list(active):
            try:
                batch = next(iters[corpus])
            except StopIteration:
                active.remove(corpus)
                continue
            f_v, f_a, f_t, label = batch[:4]
            teacher_target, teacher_mask = (batch[4:6] if len(batch) == 6
                                             else (None, None))
            lengths = _seq_len_of(f_v)
            keep = int(lengths.max())
            f_v = f_v[:, :keep].float().to(device)
            f_a = f_a[:, :keep].float().to(device)
            f_t = f_t[:, :keep].float().to(device)
            y = label.float().to(device)
            valid = _mask(lengths, keep, device)
            out = model(f_a, f_v, f_t, lengths, valid, policy=corpus)
            witness_bce = torch.nn.functional.binary_cross_entropy(out["bag_prob"], y)
            base = out["base_bag_prob"].reshape(-1).clamp(1e-5, 1 - 1e-5)
            base_bce = torch.nn.functional.binary_cross_entropy(base, y)
            # A weak negative-video constraint discourages every primitive
            # from saturating while leaving positive videos to MIL selection.
            neg = (1.0 - y)[:, None] * valid
            sparsity = ((out["primitive_prob"].mean(-1) * neg).sum() /
                        neg.sum().clamp_min(1))
            teacher_loss = torch.zeros((), device=device)
            if teacher_target is not None:
                teacher_target = teacher_target[:, :keep].float().to(device)
                teacher_mask = teacher_mask[:, :keep].float().to(device) * valid
                elem = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["primitive_logits"], teacher_target, reduction="none")
                teacher_loss = ((elem.mean(-1) * teacher_mask).sum() /
                                teacher_mask.sum().clamp_min(1))
            grounding_loss = torch.zeros((), device=device)
            if out["semantic_logits"] is not None:
                semantic_target = torch.sigmoid(out["semantic_logits"].detach())
                semantic_mask = out["semantic_text_mask"] * valid
                elem = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["primitive_logits"], semantic_target, reduction="none")
                grounding_loss = ((elem.mean(-1) * semantic_mask).sum() /
                                  semantic_mask.sum().clamp_min(1))
            loss = (witness_bce + args.base_loss_weight * base_bce +
                    args.sparsity_weight * sparsity +
                    args.teacher_weight * teacher_loss +
                    args.grounding_weight * grounding_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss in %s" % corpus)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for k, value in (("loss", loss), ("witness_bce", witness_bce),
                             ("base_bce", base_bce), ("sparsity", sparsity),
                             ("teacher", teacher_loss),
                             ("grounding", grounding_loss)):
                totals[k] += float(value.detach())
            totals["batches"] += 1
    n = max(1, totals["batches"])
    return {k: (v if k == "batches" else v / n) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loaders, device, val_gt_root):
    model.eval()
    metrics = {}
    for corpus, loader in loaders.items():
        scores, gold = [], []
        frame_scores, frame_gold = [], []
        labels = hdata.load_labels(corpus)
        gt_path = os.path.join(val_gt_root, corpus + "_val.npz")
        frame_gt = np.load(gt_path) if os.path.exists(gt_path) else None
        for f_v, f_a, f_t, index_map, n_seconds, vid in loader:
            vid = vid[0]
            f_v, f_a, f_t = f_v[0].to(device), f_a[0].to(device), f_t[0].to(device)
            length = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
            out = model(f_a, f_v, f_t, length, policy=corpus)
            scores.append(float(out["bag_prob"].mean().cpu()))
            gold.append(int(labels[vid]))
            if frame_gt is not None and vid in frame_gt.files:
                second_score = out["frame_prob"].mean(0).cpu().numpy()
                second_score = second_score[index_map[0].numpy()]
                target = frame_gt[vid]
                if len(second_score) != len(target):
                    raise RuntimeError("validation alignment mismatch %s/%s" %
                                       (corpus, vid))
                frame_scores.append(second_score)
                frame_gold.append(target)
        y, s = np.asarray(gold), np.asarray(scores)
        fy = np.concatenate(frame_gold) if frame_gold else None
        fs = np.concatenate(frame_scores) if frame_scores else None
        metrics[corpus] = {
            "video_ap": float(average_precision_score(y, s)),
            "video_roc_auc": (float(roc_auc_score(y, s))
                              if len(np.unique(y)) == 2 else None),
            "frame_ap": (float(average_precision_score(fy, fs))
                         if fy is not None else None),
            "frame_roc_auc": (float(roc_auc_score(fy, fs))
                              if fy is not None and len(np.unique(fy)) == 2
                              else None),
            "n_videos": int(len(y)), "n_positive": int(y.sum())}
        if frame_gt is not None:
            frame_gt.close()
    metrics["mean_video_ap"] = float(np.mean(
        [v["video_ap"] for v in metrics.values()
         if isinstance(v, dict)]))
    metrics["mean_frame_ap"] = float(np.mean(
        [v["frame_ap"] for v in metrics.values()
         if isinstance(v, dict) and v["frame_ap"] is not None]))
    return metrics


def main(argv=None):
    args = parser().parse_args(argv)
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    loaders, val_loaders, splits = build_loaders(args)
    model = POWAMACIL(args).to(args.device)
    model.use_policy_residual = not args.typed_only
    if args.multi_backbone:
        if not args.macil_init_root:
            raise ValueError("--multi-backbone requires --macil-init-root")
        for corpus in args.corpora:
            path = os.path.join(args.macil_init_root, corpus,
                                "seed_%d" % args.seed, "model.pth")
            model.macils[corpus].load_state_dict(torch.load(
                path, map_location=args.device))
    elif args.macil_init:
        model.macil.load_state_dict(torch.load(args.macil_init,
                                               map_location=args.device))
    if args.freeze_macil:
        backbone = model.macils if args.multi_backbone else model.macil
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
    best, best_epoch, best_state, history = -1.0, -1, None, []
    for epoch in range(1, args.max_epoch + 1):
        start = time.time()
        loss = train_epoch(model, loaders, optimizer, args, args.device)
        scheduler.step()
        val = validate(model, val_loaders, args.device, args.val_gt_root) if epoch % args.eval_every == 0 else None
        record = {"epoch": epoch, "train": loss, "validation": val,
                  "seconds": round(time.time() - start, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val and val[args.selection] > best:
            best, best_epoch = val[args.selection], epoch
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = os.path.join(args.out_dir, "model.pth")
            temporary = checkpoint + ".tmp"
            torch.save(best_state, temporary)
            os.replace(temporary, checkpoint)
    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    # The best state is already persisted immediately on improvement. Saving
    # again here keeps the normal completion path explicit.
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    meta = {"method": "powa_macil", "stage": "validation_selected_candidate",
            "api_calls": 0, "estimated_api_cost": 0.0,
            "args": vars(args), "splits": splits, "selected_epoch": best_epoch,
            "selected_value": best, "history": history,
            "selected_metric": args.selection,
            "selection_uses_frame_labels": args.selection == "mean_frame_ap",
            "test_labels_used_for_training_or_selection": False}
    with open(os.path.join(args.out_dir, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("selected epoch %d %s %.6f" % (best_epoch, args.selection, best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
