#!/usr/bin/env python
"""Full POWA training with training-only policy cluster transport."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from method import PolicyClusterTransport  # noqa: E402
from powa_macil.dataset import (PowaTestDataset, PowaTrainDataset,
                                load_teacher_jsonl, usable_text_ids)  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402
from powa_macil.train import compute_powa_loss, parser as powa_parser  # noqa: E402


def parser():
    p = powa_parser()
    p.description = __doc__
    p.add_argument("--arm", choices=("anchor", "policy", "binary", "permuted"),
                   required=True)
    p.add_argument("--transport-weight", type=float, required=True)
    p.add_argument("--transport-temperature", type=float, required=True)
    p.add_argument("--min-harmful-mass", type=float, default=.10)
    return p


def mask(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths.to(device)[:, None]


def build_loaders(args):
    if len(args.corpora) != 1:
        raise ValueError("one independently trained corpus is required")
    corpus = args.corpora[0]
    labels = hdata.load_labels(corpus)
    train_ids, val_ids = hdata.load_train_val(corpus, labels)
    train_ids = usable_text_ids(corpus, train_ids)
    val_ids = usable_text_ids(corpus, val_ids)
    if args.limit_videos:
        raise ValueError("limit-videos is forbidden for formal training")
    teacher = load_teacher_jsonl(args.teacher_file) if args.teacher_file else None
    train_ds = PowaTrainDataset(
        corpus, train_ids, labels, args.max_seqlen, args.grid, "av",
        args.crop_repeat, teacher_records=teacher,
        permute_teacher_channels=False)
    val_ds = PowaTestDataset(corpus, val_ids, args.max_seqlen, args.grid, "av")
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    val = DataLoader(val_ds, batch_size=1, shuffle=False,
                     num_workers=args.num_workers)
    return corpus, train, val, train_ids, val_ids


def train_epoch(model, transport, loader, optimizer, args, device, corpus):
    model.train()
    totals = {"loss": 0., "powa": 0., "transport": 0., "cluster": 0.,
              "semantic": 0., "harmful_mass": 0., "abstain_mass": 0.,
              "batches": 0}
    for batch in loader:
        f_v, f_a, f_t, label = batch[:4]
        teacher, teacher_mask = (batch[4:6] if len(batch) == 6 else (None, None))
        lengths = _seq_len_of(f_v)
        keep = int(lengths.max())
        f_v = f_v[:, :keep].float().to(device)
        f_a = f_a[:, :keep].float().to(device)
        f_t = f_t[:, :keep].float().to(device)
        y = label.float().to(device)
        valid = mask(lengths, keep, device)
        out = model(f_a, f_v, f_t, lengths, valid, policy=corpus)
        if teacher is not None:
            teacher = teacher[:, :keep].float().to(device)
            teacher_mask = teacher_mask[:, :keep].float().to(device)
        powa_loss, _ = compute_powa_loss(
            out, y, valid, args, teacher, teacher_mask)
        if args.arm == "anchor" or args.transport_weight == 0:
            extra = out["shared_rep"].sum() * 0.
            stats = {"cluster": 0., "semantic": 0., "harmful_mass": 0.,
                     "abstain_mass": 0.}
        else:
            extra, stats = transport(
                out["shared_rep"], out["primitive_logits"],
                out["frame_prob"], valid, y, corpus, args.arm)
        loss = powa_loss + args.transport_weight * extra
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(transport.parameters()), 5.)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["powa"] += float(powa_loss.detach())
        totals["transport"] += float(extra.detach())
        for key in ("cluster", "semantic", "harmful_mass", "abstain_mass"):
            totals[key] += float(stats[key])
        totals["batches"] += 1
    n = max(1, totals["batches"])
    for key in totals:
        if key != "batches":
            totals[key] /= n
    return totals


@torch.no_grad()
def validate(model, loader, device, corpus, val_ids):
    model.eval()
    scores = {}
    for f_v, f_a, f_t, index_map, n_seconds, vid in loader:
        name = vid[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
        out = model(f_a, f_v, f_t, lengths, policy=corpus)
        frame = out["frame_prob"].mean(0).cpu().numpy()
        frame = frame[index_map[0].numpy()]
        if len(frame) != int(n_seconds):
            raise RuntimeError(f"validation alignment mismatch {corpus}/{name}")
        scores[name] = frame
    gt_all = hdata.gt_arrays(corpus, "val")
    gt = {vid: gt_all[vid] for vid in val_ids if vid in gt_all}
    labels = hdata.load_labels(corpus)
    hate_ids = {vid for vid in gt if labels[vid] == 1}
    result = evaluate_scores(scores, gt, hate_ids)
    if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
        raise RuntimeError("validation score cohort mismatch")
    return {"pooled_ap": float(result["pr_auc"]),
            "pooled_roc": float(result["roc_auc"]),
            "within_roc": float(result["per_video"]["macro_auc"]),
            "within_n": int(result["per_video"]["n_videos_both_classes"])}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.arm == "anchor" and args.transport_weight != 0:
        raise ValueError("anchor requires transport-weight 0")
    if not 0 < args.transport_temperature:
        raise ValueError("transport temperature must be positive")
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, train_loader, val_loader, train_ids, val_ids = build_loaders(args)
    model = POWAMACIL(args).to(args.device)
    model.use_policy_residual = not args.typed_only
    if not args.macil_init:
        raise ValueError("macil-init is required")
    model.macil.load_state_dict(torch.load(args.macil_init,
                                           map_location=args.device))
    transport = PolicyClusterTransport(
        args.hid_dim, args.transport_temperature,
        args.min_harmful_mass).to(args.device)
    parameters = list(model.parameters())
    if args.arm != "anchor":
        parameters += list(transport.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_epoch)
    best_tuple = best_state = best_epoch = None
    history = []
    for epoch in range(1, args.max_epoch + 1):
        started = time.time()
        train_metrics = train_epoch(
            model, transport, train_loader, optimizer, args, args.device, corpus)
        scheduler.step()
        val_metrics = validate(model, val_loader, args.device, corpus, val_ids)
        record = {"epoch": epoch, "train": train_metrics,
                  "validation": val_metrics,
                  "seconds": round(time.time() - started, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        rank = (val_metrics["within_roc"], val_metrics["pooled_ap"],
                val_metrics["pooled_roc"], -epoch)
        if best_tuple is None or rank > best_tuple:
            best_tuple, best_epoch = rank, epoch
            best_state = copy.deepcopy(model.state_dict())
            temporary = out_dir / "model.pt.tmp"
            torch.save(best_state, temporary)
            os.replace(temporary, out_dir / "model.pt")
    config = vars(args).copy()
    meta = {"method": "policy_cluster_transport", "corpus": corpus,
            "arm": args.arm, "config": config, "train_ids": train_ids,
            "val_ids": val_ids,
            "selection_rule": "within_roc, then pooled_ap, then pooled_roc",
            "selected_epoch": best_epoch,
            "selected_validation": history[best_epoch - 1]["validation"],
            "history": history,
            "test_labels_used_for_training_or_selection": False}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best_epoch,
                      "selected_validation": meta["selected_validation"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
