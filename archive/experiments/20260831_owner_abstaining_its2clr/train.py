"""Train one fixed deletion-carrier ItS2CLR attribution arm."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from imports import base_data
from model import ABSTAIN, BACKGROUND, CARRIER, CarrierItS2CLR
from protocol import supervised_split
from states import batch_states


def loader(corpus, ids, labels, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        base_data.MultiModalDataset(corpus, ids, labels),
        batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=base_data.collate, generator=generator, drop_last=False,
    )


@torch.no_grad()
def validation_predictions(model, val_loader, device):
    model.eval()
    truth, scores = [], []
    for feats, labels, lengths, mask, _ in val_loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        mask = mask.to(device)
        output = model(feats, mask)
        score = model.video_scores(output, mask, lengths.to(device))
        truth.extend(labels.tolist())
        scores.extend(score.cpu().tolist())
    return truth, scores


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=CarrierItS2CLR.ARMS)
    parser.add_argument("--oof")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--k-proportion", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-instances", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-contrast", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (out_dir / "code_version.txt").write_text(
        "2026-08-31 deletion-carrier-abstaining ItS2CLR working-tree implementation; "
        "imports.py, protocol.py, oof.py, states.py, model.py, train.py, predict.py, evaluate.py\n"
    )
    train_ids, train_labels = supervised_split(args.corpus, "train")
    val_ids, val_labels = supervised_split(args.corpus, "val")
    cache = None
    if args.arm != "anchor":
        if not args.oof:
            raise RuntimeError("non-anchor arms require their own iterative OOF cache")
        cache = torch.load(args.oof, map_location="cpu", weights_only=True)
        if cache.get("corpus") != args.corpus or cache.get("arm") != args.arm:
            raise RuntimeError("OOF corpus/arm mismatch")
        if tuple(cache.get("modalities", ())) != tuple(base_data.MODALITIES):
            raise RuntimeError("OOF modality order mismatch")
        if cache.get("train_ids") != train_ids or set(cache["rows"]) != set(train_ids):
            raise RuntimeError("OOF cache does not exactly match official train split")
    train_loader = loader(
        args.corpus, train_ids, train_labels, args.batch_size, args.workers,
        True, args.seed,
    )
    val_loader = loader(
        args.corpus, val_ids, val_labels, args.batch_size, args.workers,
        False, args.seed,
    )
    model_args = {
        "dims": {name: base_data.FEATURE_DIMS[name] for name in base_data.MODALITIES},
        "arm": args.arm, "hidden": args.hidden, "embed": args.embed,
        "dropout": args.dropout, "k_proportion": args.k_proportion,
        "temperature": args.temperature, "max_instances": args.max_instances,
    }
    model = CarrierItS2CLR(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_ap, best_epoch, best_state = -math.inf, None, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals, n_items = {}, 0
        state_counts = {"background": 0, "carrier": 0, "abstain": 0}
        for feats, labels, lengths, mask, video_ids in train_loader:
            if args.arm == "anchor":
                states = torch.full(
                    (len(video_ids), mask.shape[1], len(base_data.MODALITIES)),
                    ABSTAIN, dtype=torch.long,
                )
            else:
                states = batch_states(
                    video_ids, labels, lengths, mask, cache, args.arm, epoch,
                    args.epochs, args.k_proportion,
                )
            state_counts["background"] += int((states == BACKGROUND).sum())
            state_counts["carrier"] += int((states == CARRIER).sum())
            state_counts["abstain"] += int(
                ((states == ABSTAIN) & mask[:, :, None]).sum()
            )
            feats = {name: value.to(device) for name, value in feats.items()}
            labels = labels.to(device)
            lengths_device = lengths.to(device)
            mask_device = mask.to(device)
            output = model(feats, mask_device)
            loss, terms = model.training_loss(
                output, states.to(device), mask_device, lengths_device, labels,
                args.lambda_smooth, args.lambda_contrast,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(labels)
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
            n_items += count
        val_truth, val_score = validation_predictions(model, val_loader, device)
        val_ap = float(average_precision_score(val_truth, val_score))
        record = {key: value / n_items for key, value in totals.items()}
        record.update({"epoch": epoch, "val_video_ap": val_ap,
                       "pseudo_state_counts": state_counts})
        history.append(record)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(), "model_args": model_args,
        "corpus": args.corpus, "selected_epoch": best_epoch,
    }, out_dir / "model.pt")
    train_log = {
        "corpus": args.corpus, "arm": args.arm, "seed": args.seed,
        "n_train": len(train_ids), "n_val": len(val_ids),
        "selected_epoch": best_epoch,
        "selected_by": "validation video-level AP within this fixed training only",
        "selected_val_video_ap": best_ap,
        "test_used_for_gradient_or_checkpoint_selection": False,
        "wall_seconds": time.time() - started, "history": history,
    }
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best_epoch,
                      "selected_val_video_ap": best_ap}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
