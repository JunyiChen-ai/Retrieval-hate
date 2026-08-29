#!/usr/bin/env python3
"""Fail-closed single-corpus Relation-V2 training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(os.path.dirname(PARENT), "duplex"))

from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from powa_macil.dataset import PowaTestDataset, PowaTrainDataset, usable_text_ids  # noqa: E402
import frame_eval_common as fec  # noqa: E402
from relation_v2.model import RelationV2  # noqa: E402
from relation_v2.protocol import (frozen_splits, scoped_labels, sha256_file,
                                  sha256_ids, verify_macil_init)  # noqa: E402


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--macil-init", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epoch", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--crop-repeat", type=int, default=5)
    p.add_argument("--max-seqlen", type=int, default=200)
    p.add_argument("--grid", default="snippet", choices=("snippet", "second"))
    p.add_argument("--hid-dim", type=int, default=128)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--a-feature-size", type=int, default=128)
    p.add_argument("--v-feature-size", type=int, default=1024)
    p.add_argument("--text-feature-size", type=int, default=768)
    p.add_argument("--n-relations", type=int, default=4)
    p.add_argument("--relation-dim", type=int, default=32)
    p.add_argument("--binding-window", type=int, default=24)
    p.add_argument("--binding-temperature", type=float, default=.2)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--topk-divisor", type=int, default=16)
    p.add_argument("--base-loss-weight", type=float, default=.25)
    p.add_argument("--sparsity-weight", type=float, default=.002)
    p.add_argument("--diversity-weight", type=float, default=.01)
    p.add_argument("--freeze-macil", action="store_true")
    return p


def mask(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths.to(device)[:, None]


def build(args):
    splits = frozen_splits(args.corpus)
    train_labels, train_label_path = scoped_labels(args.corpus, "train")
    train_ids = usable_text_ids(args.corpus, splits["train"])
    val_ids = usable_text_ids(args.corpus, splits["val"])
    if set(train_ids) != set(splits["train"]):
        raise RuntimeError("training features do not exactly cover frozen train split")
    train_ds = PowaTrainDataset(args.corpus, train_ids, train_labels,
                                args.max_seqlen, args.grid, "av",
                                args.crop_repeat)
    val_ds = PowaTestDataset(args.corpus, val_ids, args.max_seqlen,
                             args.grid, "av")
    return (DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False),
            DataLoader(val_ds, batch_size=1, shuffle=False,
                       num_workers=args.num_workers),
            splits, train_label_path)


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    if args.freeze_macil:
        model.macil.eval()
    totals = {"loss": 0., "bag": 0., "base": 0., "sparsity": 0.,
              "diversity": 0., "batches": 0}
    for f_v, f_a, f_t, label in loader:
        lengths = _seq_len_of(f_v)
        keep = int(lengths.max())
        valid = mask(lengths, keep, device)
        f_v = f_v[:, :keep].float().to(device)
        f_a = f_a[:, :keep].float().to(device)
        f_t = f_t[:, :keep].float().to(device)
        y = label.float().to(device)
        out = model(f_a, f_v, f_t, lengths, valid)
        bag = torch.nn.functional.binary_cross_entropy(out["bag_prob"], y)
        base = out["base_bag_prob"].reshape(-1).clamp(1e-5, 1 - 1e-5)
        base_loss = torch.nn.functional.binary_cross_entropy(base, y)
        negative = (1 - y)[:, None] * valid
        graph_prob = torch.sigmoid(out["graph_logit"])
        sparsity = ((graph_prob * negative).sum() / negative.sum().clamp_min(1))
        role = torch.cat([out["relation_source"], out["relation_target"]], 1)
        role_mask = torch.cat([valid, valid], 1).to(role.dtype)
        role = role * role_mask[..., None]
        role = role.permute(0, 2, 1)
        role = torch.nn.functional.normalize(role, dim=-1, eps=1e-6)
        gram = torch.einsum("brt,bst->brs", role, role)
        eye = torch.eye(gram.shape[-1], device=device)[None]
        diversity = ((gram - eye).square().sum() /
                     max(1, gram.shape[0] * gram.shape[1] * (gram.shape[1] - 1)))
        loss = (bag + args.base_loss_weight * base_loss +
                args.sparsity_weight * sparsity +
                args.diversity_weight * diversity)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Relation-V2 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        for name, value in (("loss", loss), ("bag", bag),
                            ("base", base_loss), ("sparsity", sparsity),
                            ("diversity", diversity)):
            totals[name] += float(value.detach())
        totals["batches"] += 1
    n = max(1, totals["batches"])
    return {k: (v if k == "batches" else v / n) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, corpus, device):
    model.eval()
    gt = hdata.gt_arrays(corpus, "val")
    per_video, seen = {}, set()
    for f_v, f_a, f_t, index_map, n_seconds, vid in loader:
        vid = vid[0]
        if vid not in gt:
            continue
        if vid in seen:
            raise RuntimeError("duplicate validation video %s" % vid)
        seen.add(vid)
        f_v, f_a, f_t = f_v[0].to(device), f_a[0].to(device), f_t[0].to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
        out = model(f_a, f_v, f_t, lengths)
        score = out["frame_prob"].mean(0).cpu().numpy()[index_map[0].numpy()]
        if len(score) != len(gt[vid]) or not np.isfinite(score).all():
            raise RuntimeError("validation alignment/nonfinite %s" % vid)
        per_video[vid] = (score, gt[vid])
    if seen != set(gt):
        raise RuntimeError("validation GT coverage mismatch: missing %s" %
                           sorted(set(gt) - seen)[:5])
    metric = fec.evaluate(per_video)
    return {"frame_ap": metric["pr_auc"], "frame_roc": metric["roc_auc"],
            "n_eval_videos": metric["n_videos"]}


def main(argv=None):
    args = parser().parse_args(argv)
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise RuntimeError("out-dir must be absent or empty: %s" % args.out_dir)
    init_provenance = verify_macil_init(args.corpus, args.macil_init)
    loader, val_loader, splits, label_path = build(args)
    model = RelationV2(args).to(args.device)
    model.macil.load_state_dict(torch.load(args.macil_init,
                                           map_location=args.device))
    if args.freeze_macil:
        for p in model.macil.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            args.max_epoch)
    os.makedirs(args.out_dir, exist_ok=True)
    best, best_epoch, history = -1., -1, []
    for epoch in range(1, args.max_epoch + 1):
        start = time.time()
        losses = train_epoch(model, loader, optimizer, args, args.device)
        scheduler.step()
        val = validate(model, val_loader, args.corpus, args.device)
        row = {"epoch": epoch, "train": losses, "validation": val,
               "seconds": round(time.time() - start, 2)}
        history.append(row); print(json.dumps(row), flush=True)
        if val["frame_ap"] > best:
            best, best_epoch = val["frame_ap"], epoch
            temporary = os.path.join(args.out_dir, "model.pth.tmp")
            torch.save(copy.deepcopy(model.state_dict()), temporary)
            os.replace(temporary, os.path.join(args.out_dir, "model.pth"))
    if best_epoch < 0:
        raise RuntimeError("no validation checkpoint")
    meta = {
        "method": "relation_v2_performance", "corpus": args.corpus,
        "args": vars(args), "selected_epoch": best_epoch,
        "selected_metric": "validation_pooled_frame_ap",
        "selected_value": best, "history": history,
        "train_ids": sorted(splits["train"]),
        "val_ids": sorted(splits["val"]),
        "test_manifest_ids": sorted(splits["test"]),
        "split_hashes": {k: sha256_ids(v) for k, v in splits.items()},
        "scoped_train_labels": {"path": label_path,
                                "sha256": sha256_file(label_path)},
        "macil_init": init_provenance,
        "test_labels_used_in_gradient_training": False,
        "cross_corpus_data_or_parameters": False,
    }
    model_path = os.path.join(args.out_dir, "model.pth")
    meta["model_sha256"] = sha256_file(model_path)
    meta_path = os.path.join(args.out_dir, "train_meta.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w") as fh:
        json.dump(meta, fh, indent=2); fh.write("\n")
    os.replace(meta_tmp, meta_path)
    complete = {"corpus": args.corpus,
                "model_sha256": meta["model_sha256"],
                "meta_sha256": sha256_file(meta_path)}
    complete_path = os.path.join(args.out_dir, "COMPLETE.json")
    complete_tmp = complete_path + ".tmp"
    with open(complete_tmp, "w") as fh:
        json.dump(complete, fh, indent=2); fh.write("\n")
    os.replace(complete_tmp, complete_path)
    print("selected epoch %d val Frame AP %.6f" % (best_epoch, best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
