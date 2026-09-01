"""Train one independently scoped factorial-witness arm."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import FactorialWitnessCRF  # noqa: E402
from protocol import supervised_split  # noqa: E402
from src.multimodal_video_data import multimodal_loader  # noqa: E402


@torch.no_grad()
def video_predictions(model, loader, device):
    model.eval()
    labels_out, logits_out = [], []
    for feats, labels, lengths, _, _ in loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        output = model(feats, lengths)
        labels_out.extend(labels.tolist())
        logits_out.extend(output["bag_logits"].cpu().tolist())
    return labels_out, logits_out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=FactorialWitnessCRF.ARMS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
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
        "2026-08-31 factorial witness CRF working-tree implementation; "
        "model.py, protocol.py, train.py, predict.py, evaluate.py\n"
    )

    train_ids, train_labels = supervised_split(args.corpus, "train")
    val_ids, val_labels = supervised_split(args.corpus, "val")
    train_loader = multimodal_loader(args.corpus, train_ids, train_labels, args.batch_size, args.workers, True, args.seed)
    val_loader = multimodal_loader(args.corpus, val_ids, val_labels, args.batch_size, args.workers, False, args.seed)
    n_positive = sum(train_labels.values())
    n_negative = len(train_labels) - n_positive
    if not n_positive or not n_negative:
        raise RuntimeError("training split must contain both video classes")
    positive_weight = torch.tensor(n_negative / n_positive, device=device)

    model = FactorialWitnessCRF(args.arm, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap, best_epoch, best_state = -math.inf, None, None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for feats, labels, lengths, _, _ in train_loader:
            feats = {name: value.to(device) for name, value in feats.items()}
            labels = labels.to(device)
            output = model(feats, lengths)
            loss = F.binary_cross_entropy_with_logits(
                output["bag_logits"], labels, pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total_items += len(labels)
        val_y, val_logits = video_predictions(model, val_loader, device)
        val_ap = float(average_precision_score(val_y, val_logits))
        record = {"epoch": epoch, "train_loss": total_loss / total_items, "val_video_ap": val_ap}
        history.append(record)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps(record), flush=True)

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "model_args": {"arm": args.arm, "hidden": args.hidden, "dropout": args.dropout},
        "corpus": args.corpus,
        "selected_epoch": best_epoch,
    }, out_dir / "model.pt")
    train_log = {
        "corpus": args.corpus,
        "arm": args.arm,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "selected_epoch": best_epoch,
        "selected_by": "validation video-level AP within this fixed training only",
        "selected_val_video_ap": best_ap,
        "test_used_for_gradient_or_checkpoint_selection": False,
        "wall_seconds": time.time() - started,
        "history": history,
    }
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2) + "\n")
    print(json.dumps({"selected_epoch": best_epoch, "selected_val_video_ap": best_ap}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
