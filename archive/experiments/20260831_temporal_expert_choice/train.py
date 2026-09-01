#!/usr/bin/env python
"""Train one temporal routing arm and immediately predict blind test IDs."""
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
BASE = ROOT / "scripts" / "reproduction_baselines" / "multihateloc"
COMMON = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HERE))

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from src.scoped_video_protocol import (evaluator_test_ids,
                                       scoped_video_labels)  # noqa: E402

from model import ARMS, TemporalExpertChoice  # noqa: E402


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    if y.sum() == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def make_loader(corpus, ids, labels, batch_size, shuffle, workers,
                generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, collate_fn=mdata.collate, num_workers=workers,
        drop_last=False, generator=generator)


def run_epoch(model, loader, device, optimizer, args):
    model.train()
    sums, seen = {}, 0
    for feats, labels, lengths, mask, _ in loader:
        feats = {key: value.to(device, non_blocking=True)
                 for key, value in feats.items()}
        labels, lengths, mask = labels.to(device), lengths.to(device), mask.to(device)
        out = model(feats, mask)
        mil, _ = model.mil_loss(out["prob"], mask, lengths, labels)
        smooth = model.smoothness_loss(out["prob"], mask)
        contrast = model.contrastive_loss(out["embeds"], mask)
        loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast}
        for index, name in enumerate(model.modalities):
            terms["assignment_rate_" + name] = (
                out["selected"][:, index].sum() / mask.sum()).detach()
            terms["mean_abs_contribution_" + name] = (
                out["contribution"][:, index].abs().sum() /
                mask.sum()).detach()
        batch = len(labels)
        for key, value in terms.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch
        seen += batch
    return {key: value / seen for key, value in sums.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, videos, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in loader:
        feats = {key: value.to(device) for key, value in feats.items()}
        lengths_d, mask_d = lengths.to(device), mask.to(device)
        out = model(feats, mask_d)
        video_score = model.video_score(out["prob"], mask_d, lengths_d)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            record = {"score_final": out["prob"][index, :length].cpu().numpy()}
            for mod_index, name in enumerate(model.modalities):
                record["assignment_" + name] = out["selected"][
                    index, mod_index, :length].float().cpu().numpy()
                record["contribution_" + name] = out["contribution"][
                    index, mod_index, :length].cpu().numpy()
            scores[video_id] = record
            videos[video_id] = {
                "score": float(video_score[index]),
                "assignment_count": {
                    name: int(out["selected"][index, mod_index, :length].sum())
                    for mod_index, name in enumerate(model.modalities)},
                "signed_contribution_mean": {
                    name: float(out["contribution"][
                        index, mod_index, :length].mean())
                    for mod_index, name in enumerate(model.modalities)},
            }
            labels_out[video_id] = int(labels[index])
    return scores, videos, labels_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "hateclipseg"))
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "temporal_expert_choice_mil",
        "evaluation_split": "test", "args": vars(args)}, indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/validation manifests overlap")
    test_ids = evaluator_test_ids(
        args.corpus, hdata.load_split(args.corpus, "test"))
    test_dummy = {video_id: 0 for video_id in test_ids}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(args.corpus, train_ids, train_labels,
                               args.batch_size, True, 4, generator)
    val_loader = make_loader(args.corpus, val_ids, val_labels,
                             args.batch_size, False, 2)

    model = TemporalExpertChoice(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        arm=args.arm, hidden=args.hidden, embed=args.embed,
        dropout=args.dropout, k_proportion=args.k_proportion,
        temperature=args.temperature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, best_epoch, best_state = -1.0, None, None
    history, started = [], time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, args.device, optimizer, args)
        _, video, video_labels = predict(model, val_loader, args.device)
        ids = sorted(video)
        val_ap = average_precision([video_labels[v] for v in ids],
                                   [video[v]["score"] for v in ids])
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
    test_loader = make_loader(args.corpus, test_ids, test_dummy,
                              args.batch_size, False, 2)
    records, diagnostics, _ = predict(model, test_loader, args.device)
    with (output_dir / "scores.jsonl").open("w") as handle:
        for video_id in test_ids:
            row = {"video_id": video_id}
            row.update({key: [float(x) for x in value]
                        for key, value in records[video_id].items()})
            handle.write(json.dumps(row) + "\n")
    torch.save(best_state, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus, "arm": args.arm,
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "test_prediction_generated_after_checkpoint_selection": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_video_diagnostics": diagnostics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; wrote {len(test_ids)} test videos",
          flush=True)


if __name__ == "__main__":
    main()
