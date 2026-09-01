#!/usr/bin/env python
"""Train one MultiHateLoc/DGM arm and immediately predict the test split."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines" / "multihateloc"
COMMON = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(COMMON))

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import MultiHateLoc  # noqa: E402
from src.scoped_video_protocol import (evaluator_test_ids,
                                       scoped_video_labels)  # noqa: E402

from method import (ARMS, apply_gradient_modulation, competence,
                    modulation_coefficients)  # noqa: E402


def average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    npos = y.sum()
    if npos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / npos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def run_epoch(model, loader, device, optimizer, args):
    model.train()
    totals = {}
    seen = 0
    for feats, labels, lengths, mask, _ in loader:
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        labels = labels.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        output = model(feats, mask)
        mil, per_branch = model.mil_loss(
            output["probs"], mask, lengths, labels)
        smooth = model.smoothness_loss(output["probs"], mask)
        contrast = model.contrastive_loss(output["embeds"], mask)
        loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast

        comp = competence(output["probs"], mask, lengths, labels,
                          model.modalities, args.arm, args.k_proportion)
        coeff = modulation_coefficients(comp, args.arm, args.gamma)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        apply_gradient_modulation(model, coeff)
        optimizer.step()

        terms = {"loss": loss, "mil": mil, "smooth": smooth,
                 "contrast": contrast}
        terms.update({"mil_" + k: v for k, v in per_branch.items()})
        for index, name in enumerate(model.modalities):
            terms["competence_" + name] = comp[index]
            terms["coefficient_" + name] = coeff[index]
        batch = len(labels)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch
        seen += batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    frame_records, video_records, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in loader:
        feats = {key: value.to(device) for key, value in feats.items()}
        lengths_device = lengths.to(device)
        mask = mask.to(device)
        output = model(feats, mask)
        video_scores = model.video_scores(
            output["probs"], mask, lengths_device)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            frame_records[video_id] = {
                "score_" + name:
                    output["probs"][name][index, :length].cpu().numpy()
                for name in output["probs"]}
            video_records[video_id] = {
                name: float(value[index]) for name, value in video_scores.items()}
            video_records[video_id]["weights"] = [
                float(x) for x in output["weights"][index]]
            labels_out[video_id] = int(labels[index])
    return frame_records, video_records, labels_out


def loader(corpus, ids, labels, batch_size, shuffle, workers, generator=None):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels), batch_size=batch_size,
        shuffle=shuffle, drop_last=False, collate_fn=mdata.collate,
        num_workers=workers, generator=generator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--k-proportion", type=int, default=3)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({"date": "2026-08-31", "method":
                    "witness_conditional_dynamic_gradient_modulation",
                    "evaluation_split": "test", "args": vars(args)},
                   indent=2) + "\n")

    train_ids = hdata.load_split(args.corpus, "train")
    val_ids = hdata.load_split(args.corpus, "val")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train and validation manifests overlap")
    test_ids = evaluator_test_ids(
        args.corpus, hdata.load_split(args.corpus, "test"))
    test_dummy_labels = {video_id: 0 for video_id in test_ids}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(args.corpus, train_ids, train_labels, args.batch_size,
                          True, 4, generator)
    val_loader = loader(args.corpus, val_ids, val_labels, args.batch_size,
                        False, 2)
    test_loader = None

    model = MultiHateLoc(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        k_proportion=args.k_proportion,
        temperature=args.temperature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ap, best_epoch, best_state = -1.0, None, None
    history = []
    started = time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, args.device, optimizer, args)
        _, video, video_labels = predict(model, val_loader, args.device)
        ids = sorted(video)
        val_ap = average_precision(
            [video_labels[v] for v in ids], [video[v]["fused"] for v in ids])
        stats.update({"epoch": epoch, "val_video_ap": val_ap,
                      "elapsed_seconds": round(time.time() - started, 1)})
        history.append(stats)
        if val_ap == val_ap and val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            coeff = ",".join(
                f"{name}:{stats['coefficient_' + name]:.3f}"
                for name in model.modalities)
            print(f"{args.corpus}/{args.arm} epoch {epoch:03d} "
                  f"loss={stats['loss']:.4f} val_ap={val_ap:.4f} "
                  f"coeff={coeff}", flush=True)
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no state")
    model.load_state_dict(best_state)
    # Construct the label-blind evaluator cohort only after checkpoint freeze.
    test_loader = loader(args.corpus, test_ids, test_dummy_labels,
                         args.batch_size, False, 2)
    frames, test_video, _ = predict(model, test_loader, args.device)
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            record = {"video_id": video_id}
            record.update({key: [float(x) for x in value]
                           for key, value in frames[video_id].items()})
            handle.write(json.dumps(record) + "\n")
    torch.save(best_state, output_dir / "checkpoint.pt")
    (output_dir / "train_log.json").write_text(json.dumps({
        "corpus": args.corpus, "arm": args.arm,
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "test_prediction_generated_immediately_after_selection": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_video_diagnostics": test_video,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2) + "\n")
    print(f"selected epoch {best_epoch}; wrote {len(test_ids)} test videos",
          flush=True)


if __name__ == "__main__":
    main()
