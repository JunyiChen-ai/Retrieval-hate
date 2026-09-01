#!/usr/bin/env python3
"""Train on train and select a checkpoint by validation video AP; never reads test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tdata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "scripts/reproduction_baselines"
MM = BASE / "multihateloc"
sys.path[:0] = [str(HERE), str(ROOT), str(BASE), str(MM)]

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import MassConservingMarkedSplatMIL  # noqa: E402
from src.scoped_video_protocol import scoped_video_labels  # noqa: E402


def average_precision(target, score):
    target = np.asarray(target, dtype=float)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    ordered = target[order]
    if ordered.sum() == 0:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    recall = np.cumsum(ordered) / ordered.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=mdata.collate,
        num_workers=workers,
        generator=generator,
        drop_last=False,
    )


def run_epoch(model, batches, device, optimizer, args):
    model.train()
    totals, seen = {}, 0
    for features, labels, lengths, mask, _ in batches:
        features = {key: value.to(device, non_blocking=True) for key, value in features.items()}
        labels = labels.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        output = model(features, mask)
        mil, _ = model.mil_loss(output["prob"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["prob"], mask)
        contrast = model.contrastive_loss(output["embeds"], mask)
        loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = {
            "loss": loss,
            "mil": mil,
            "smooth": smooth,
            "contrast": contrast,
            "mean_probability": output["prob"].sum() / mask.sum(),
        }
        batch_size = len(labels)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        seen += batch_size
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def validation_video_scores(model, batches, device):
    model.eval()
    scores, labels = {}, {}
    for features, target, lengths, mask, video_ids in batches:
        features = {key: value.to(device) for key, value in features.items()}
        output = model(features, mask.to(device))
        video = model.video_score(output["prob"], mask.to(device), lengths.to(device))
        for index, video_id in enumerate(video_ids):
            scores[video_id] = float(video[index])
            labels[video_id] = int(target[index])
    return scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, required=True)
    parser.add_argument("--k-proportion", type=int, required=True)
    parser.add_argument("--lambda-smooth", type=float, required=True)
    parser.add_argument("--lambda-contrast", type=float, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--embed", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "date": "2026-09-01",
        "method": "mass_conserving_marked_temporal_splat",
        "stage": "validation_selection_only",
        "test_split_read": False,
        "args": vars(args),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/validation overlap")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    generator = torch.Generator().manual_seed(args.seed)
    train_batches = loader(
        args.corpus, train_ids, train_labels, args.batch_size, True, 4, generator
    )
    val_batches = loader(args.corpus, val_ids, val_labels, args.batch_size, False, 2)
    model = MassConservingMarkedSplatMIL(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        args.hidden,
        args.embed,
        args.dropout,
        args.k_proportion,
        args.temperature,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, best_epoch, best_state = -1.0, None, None
    history, started = [], time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_batches, args.device, optimizer, args)
        scores, labels = validation_video_scores(model, val_batches, args.device)
        ordered = sorted(scores)
        val_ap = average_precision([labels[v] for v in ordered], [scores[v] for v in ordered])
        stats.update({"epoch": epoch, "validation_video_ap": val_ap})
        history.append(stats)
        if val_ap == val_ap and val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{args.corpus}/{args.config_name} epoch {epoch:03d} "
                f"loss={stats['loss']:.4f} val_ap={val_ap:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    torch.save(best_state, output / "checkpoint.pt")
    (output / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus,
        "config_name": args.config_name,
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "test_prediction_generated": False,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; validation AP {best_ap:.6f}; no test read", flush=True)


if __name__ == "__main__":
    main()
