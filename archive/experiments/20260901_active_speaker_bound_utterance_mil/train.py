#!/usr/bin/env python
"""Full validation-selected training for active-speaker-bound POWA."""
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

from dataset import SourceTestDataset, SourceTrainDataset  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from method import ActiveSpeakerBoundPOWA  # noqa: E402
from powa_macil.dataset import load_teacher_jsonl, usable_text_ids  # noqa: E402
from powa_macil.train import compute_powa_loss, parser as powa_parser  # noqa: E402


def parser():
    ap = powa_parser()
    ap.description = __doc__
    ap.add_argument("--arm", required=True,
                    choices=("anchor", "core", "permuted"))
    ap.add_argument("--relation-weight", type=float, required=True)
    ap.add_argument("--face-feature-size", type=int, default=512)
    return ap


def mask(lengths, width, device):
    return torch.arange(width, device=device)[None] < lengths.to(device)[:, None]


def build_loaders(args):
    if len(args.corpora) != 1:
        raise ValueError("one independently trained corpus is required")
    if args.limit_videos:
        raise ValueError("limit-videos is forbidden for formal training")
    corpus = args.corpora[0]
    labels = hdata.load_labels(corpus)
    train_ids, val_ids = hdata.load_train_val(corpus, labels)
    train_ids = usable_text_ids(corpus, train_ids)
    val_ids = usable_text_ids(corpus, val_ids)
    teacher = load_teacher_jsonl(args.teacher_file) if args.teacher_file else None
    data_arm = "permuted" if args.arm == "permuted" else "core"
    train_ds = SourceTrainDataset(
        corpus, train_ids, labels, args.max_seqlen, args.grid, "av",
        args.crop_repeat, teacher_records=teacher,
        permute_teacher_channels=False, arm=data_arm)
    val_ds = SourceTestDataset(
        corpus, val_ids, args.max_seqlen, args.grid, "av", arm=data_arm)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=False)
    val = DataLoader(val_ds, batch_size=1, shuffle=False,
                     num_workers=args.num_workers)
    return corpus, train, val, train_ids, val_ids


def unpack_train(batch):
    if len(batch) == 9:
        (f_v, f_a, f_t, label, teacher, teacher_mask, face, state,
         source_utterance) = batch
    elif len(batch) == 7:
        f_v, f_a, f_t, label, face, state, source_utterance = batch
        teacher = teacher_mask = None
    else:
        raise RuntimeError(f"unexpected training batch of length {len(batch)}")
    return (f_v, f_a, f_t, label, teacher, teacher_mask, face, state,
            source_utterance)


def train_epoch(model, loader, optimizer, args, device, corpus):
    model.train()
    totals = {"loss": 0., "batches": 0}
    for batch in loader:
        (f_v, f_a, f_t, label, teacher, teacher_mask, face, state,
         source_utterance) = unpack_train(batch)
        lengths = _seq_len_of(f_v)
        keep = int(lengths.max())
        f_v = f_v[:, :keep].float().to(device)
        f_a = f_a[:, :keep].float().to(device)
        f_t = f_t[:, :keep].float().to(device)
        face = face[:, :keep].float().to(device)
        state = state[:, :keep].long().to(device)
        source_utterance = source_utterance[:, :keep].float().to(device)
        y = label.float().to(device)
        valid = mask(lengths, keep, device)
        out = model(f_a, f_v, f_t, face, state, source_utterance,
                    lengths, valid, policy=corpus)
        if teacher is not None:
            teacher = teacher[:, :keep].float().to(device)
            teacher_mask = teacher_mask[:, :keep].float().to(device)
        loss, _ = compute_powa_loss(
            out, y, valid, args, teacher, teacher_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["batches"] += 1
    totals["loss"] /= max(1, totals["batches"])
    return totals


@torch.no_grad()
def validate(model, loader, device, corpus, val_ids):
    model.eval()
    scores = {}
    for batch in loader:
        (f_v, f_a, f_t, index_map, n_seconds, vid, face, state,
         source_utterance) = batch
        name = vid[0]
        f_v, f_a, f_t = (f_v[0].float().to(device),
                          f_a[0].float().to(device),
                          f_t[0].float().to(device))
        face = face[0].float().to(device)
        state = state[0].long().to(device)
        source_utterance = source_utterance[0].float().to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
        out = model(f_a, f_v, f_t, face, state, source_utterance,
                    lengths, policy=corpus)
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
    if args.arm == "anchor" and args.relation_weight != 0:
        raise ValueError("anchor requires relation-weight 0")
    if args.arm != "anchor" and args.relation_weight <= 0:
        raise ValueError("relation arm requires positive relation-weight")
    args.device = runtime.resolve_device(args.device)
    runtime.setup_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus, train_loader, val_loader, train_ids, val_ids = build_loaders(args)
    model = ActiveSpeakerBoundPOWA(args).to(args.device)
    model.use_policy_residual = not args.typed_only
    if not args.macil_init:
        raise ValueError("macil-init is required")
    model.macil.load_state_dict(torch.load(args.macil_init,
                                           map_location=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_epoch)
    best_rank = best_epoch = None
    history = []
    for epoch in range(1, args.max_epoch + 1):
        started = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, args, args.device, corpus)
        scheduler.step()
        val_metrics = validate(
            model, val_loader, args.device, corpus, val_ids)
        record = {"epoch": epoch, "train": train_metrics,
                  "validation": val_metrics,
                  "seconds": round(time.time() - started, 2)}
        history.append(record)
        print(json.dumps(record), flush=True)
        rank = (val_metrics["within_roc"], val_metrics["pooled_ap"],
                val_metrics["pooled_roc"], -epoch)
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch = rank, epoch
            state = copy.deepcopy(model.state_dict())
            temporary = out_dir / "model.pt.tmp"
            torch.save(state, temporary)
            os.replace(temporary, out_dir / "model.pt")
    config = vars(args).copy()
    meta = {"method": "active_speaker_bound_utterance_mil",
            "corpus": corpus, "arm": args.arm, "config": config,
            "train_ids": train_ids, "val_ids": val_ids,
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
