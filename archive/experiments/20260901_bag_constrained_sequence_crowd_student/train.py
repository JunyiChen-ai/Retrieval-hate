#!/usr/bin/env python
"""Full student training with validation-only checkpoint selection."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from hate_common import data as hdata  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402
from src.multimodal_video_data import multimodal_loader  # noqa: E402
from dataset import target_loader  # noqa: E402
from model import SequenceCrowdStudent, topk_bag_probability  # noqa: E402


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--arm", required=True,
                    choices=("core", "token_ds", "unconstrained_bsc"))
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--bag-weight", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pooled-tolerance", type=float, default=.01)
    return ap


@torch.no_grad()
def validation_metrics(model, loader, corpus, val_ids, device):
    model.eval(); scores = {}
    for features, _, lengths, mask, video_ids in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features, mask.to(device))
        for row, video_id in enumerate(video_ids):
            scores[video_id] = torch.sigmoid(logits[row, :int(lengths[row])]).cpu().numpy()
    gt_all = hdata.gt_arrays(corpus, "val")
    gt = {v: gt_all[v] for v in val_ids}
    labels = hdata.load_labels(corpus)
    result = evaluate_scores(scores, gt, {v for v in gt if labels[v] == 1})
    return {"pooled_ap": float(result["pr_auc"]),
            "pooled_roc": float(result["roc_auc"]),
            "within_roc": float(result["per_video"]["macro_auc"]),
            "within_n": int(result["per_video"]["n_videos_both_classes"])}


def select_epoch(rows, tolerance):
    best_ap = max(x["validation"]["pooled_ap"] for x in rows)
    best_roc = max(x["validation"]["pooled_roc"] for x in rows)
    eligible = [x for x in rows
                if x["validation"]["pooled_ap"] >= best_ap - tolerance
                and x["validation"]["pooled_roc"] >= best_roc - tolerance]
    return max(eligible, key=lambda x: (x["validation"]["within_roc"],
                                        x["validation"]["pooled_ap"],
                                        x["validation"]["pooled_roc"]))


def main(argv=None):
    args = parser().parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels,
                                               val_frac=.1, seed=args.seed)
    train = target_loader(args.corpus, train_ids, labels, args.targets,
                          args.batch_size, args.workers, True, args.seed)
    val = multimodal_loader(args.corpus, val_ids, labels, args.batch_size,
                            args.workers, False, args.seed)
    model = SequenceCrowdStudent(args.width, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []; states = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for features, bag_labels, targets, lengths, mask, _ in train:
            features = {k: v.to(device) for k, v in features.items()}
            bag_labels = bag_labels.to(device); targets = targets.to(device)
            mask_d = mask.to(device); lengths_d = lengths.to(device)
            logits = model(features, mask_d)
            posterior_loss = F.binary_cross_entropy_with_logits(
                logits[mask_d], targets[mask_d])
            bag = topk_bag_probability(logits, lengths_d)
            bag_loss = F.binary_cross_entropy(bag, bag_labels)
            loss = posterior_loss + args.bag_weight * bag_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(bag_labels); count += len(bag_labels)
        metrics = validation_metrics(model, val, args.corpus, val_ids, device)
        rows.append({"epoch": epoch, "train_loss": total / count,
                     "validation": metrics})
        states.append({k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()})
        print(json.dumps(rows[-1]), flush=True)
    selected = select_epoch(rows, args.pooled_tolerance)
    model.load_state_dict(states[selected["epoch"] - 1])
    torch.save(model.state_dict(), out / "model.pt")
    payload = {"method": "bag_constrained_sequence_crowd_student",
               "arm": args.arm, "corpus": args.corpus, "args": vars(args),
               "selection_split": "validation",
               "selection_rule": "AP and ROC within .01 of respective best epoch, then max within ROC",
               "selected_epoch": selected["epoch"],
               "selected_validation": selected["validation"],
               "history": rows, "train_ids": train_ids, "val_ids": val_ids,
               "test_predictions_seen": False,
               "test_labels_used_for_training_or_selection": False,
               "wall_seconds": time.time() - started}
    temporary = out / "train_meta.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, out / "train_meta.json")


if __name__ == "__main__":
    main()
