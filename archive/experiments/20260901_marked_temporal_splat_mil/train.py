#!/usr/bin/env python
"""Train one marked-splat arm, select on validation, then predict test."""
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
BASE = ROOT / "scripts" / "reproduction_baselines"
MM = BASE / "multihateloc"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MM))
sys.path.insert(0, str(HERE))
import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import ARMS, MarkedTemporalSplatMIL  # noqa: E402
from src.scoped_video_protocol import (evaluator_test_ids,
                                       scoped_video_labels)  # noqa: E402


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    recall = np.cumsum(y) / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, collate_fn=mdata.collate, num_workers=workers,
        generator=generator, drop_last=False)


def run_epoch(model, batches, device, optimizer, args):
    model.train()
    totals, seen = {}, 0
    for feats, labels, lengths, mask, _ in batches:
        feats = {key: value.to(device, non_blocking=True)
                 for key, value in feats.items()}
        labels, lengths, mask = labels.to(device), lengths.to(device), mask.to(device)
        output = model(feats, mask)
        mil, _ = model.mil_loss(output["prob"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["prob"], mask)
        contrast = model.contrastive_loss(output["embeds"], mask)
        loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast,
                 "mean_probability": output["prob"].sum() / mask.sum()}
        for index, name in enumerate(model.modalities):
            terms["amplitude_" + name] = (
                output["amplitudes"][:, index].sum() / mask.sum())
            expected = sum(width * output["mixtures"][:, index, :, j]
                           for j, width in enumerate(model.widths))
            terms["duration_" + name] = expected.sum() / mask.sum()
        batch_size = len(labels)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        seen += batch_size
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    records, video, label_rows = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in batches:
        feats = {key: value.to(device) for key, value in feats.items()}
        lengths_d, mask_d = lengths.to(device), mask.to(device)
        output = model(feats, mask_d)
        video_score = model.video_score(output["prob"], mask_d, lengths_d)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            records[video_id] = output["prob"][index, :length].cpu().numpy()
            video[video_id] = {
                "score": float(video_score[index]),
                "mean_probability": float(output["prob"][index, :length].mean()),
                "mean_amplitude": {
                    name: float(output["amplitudes"][index, mod, :length].mean())
                    for mod, name in enumerate(model.modalities)},
                "mean_duration": {
                    name: float(sum(width * output["mixtures"][
                        index, mod, :length, scale].mean()
                        for scale, width in enumerate(model.widths)))
                    for mod, name in enumerate(model.modalities)},
            }
            label_rows[video_id] = int(labels[index])
    return records, video, label_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output-dir", required=True)
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "marked_temporal_splat_mil",
        "evaluation_split": "test", "args": vars(args)}, indent=2) + "\n")
    train_ids, val_ids = (hdata.load_split(args.corpus, "train"),
                          hdata.load_split(args.corpus, "val"))
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/validation overlap")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(args.corpus, train_ids, train_labels,
                          args.batch_size, True, 4, generator)
    val_loader = loader(args.corpus, val_ids, val_labels,
                        args.batch_size, False, 2)
    model = MarkedTemporalSplatMIL(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        args.arm, args.hidden, args.embed, args.dropout,
        args.k_proportion, args.temperature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, best_epoch, best_state = -1.0, None, None
    history, started = [], time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, args.device, optimizer, args)
        _, val_video, val_y = predict(model, val_loader, args.device)
        ordered = sorted(val_video)
        val_ap = average_precision([val_y[v] for v in ordered],
                                   [val_video[v]["score"] for v in ordered])
        stats.update({"epoch": epoch, "val_video_ap": val_ap,
                      "elapsed_seconds": round(time.time() - started, 1)})
        history.append(stats)
        if val_ap == val_ap and val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{args.corpus}/{args.arm} epoch {epoch:03d} "
                  f"loss={stats['loss']:.4f} val_ap={val_ap:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    test_loader = loader(args.corpus, test_ids, {v: 0 for v in test_ids},
                         args.batch_size, False, 2)
    records, diagnostics, _ = predict(model, test_loader, args.device)
    with (output_dir / "scores.jsonl").open("w") as handle:
        for video_id in test_ids:
            handle.write(json.dumps({"video_id": video_id,
                                     "score_final": [float(x) for x in records[video_id]]}) + "\n")
    torch.save(best_state, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus, "arm": args.arm,
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "test_prediction_generated_after_checkpoint_selection": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_video_diagnostics": diagnostics, "history": history,
        "elapsed_seconds": round(time.time() - started, 1)}, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; wrote {len(test_ids)} test videos", flush=True)


if __name__ == "__main__":
    main()
