#!/usr/bin/env python3
"""Train one same-corpus typed REBA model and emit frozen test scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from model import TypedREBA  # noqa: E402
from src.multimodal_video_data import DIMS, multimodal_loader  # noqa: E402
from src.scoped_video_protocol import (  # noqa: E402
    evaluator_test_ids,
    scoped_video_labels,
)


CODE_VERSION_DESCRIPTION = "2026-08-31 typed REBA cross-task adaptation pilot"


def atomic_text(path: Path, value: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_torch(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def move(feats, labels, mask, device):
    return (
        {name: value.to(device, non_blocking=True) for name, value in feats.items()},
        labels.to(device),
        mask.to(device),
    )


@torch.no_grad()
def predict(model, data, device, include_control=False):
    model.eval()
    scores, bags, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in data:
        feats, labels_device, mask_device = move(feats, labels, mask, device)
        output = model(feats, mask_device, include_control=include_control)
        branches = [name for name in output if name.startswith("score")]
        for branch in branches:
            scores.setdefault(branch, {})
        for row, video_id in enumerate(video_ids):
            length = int(lengths[row])
            for branch in branches:
                value = output[branch][row, :length].detach().cpu().numpy()
                if value.shape != (length,) or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid {branch} for {video_id}")
                scores[branch][video_id] = value
            bags[video_id] = float(output["bag_probability"][row])
            labels_out[video_id] = int(labels_device[row])
    return scores, bags, labels_out


def train_epoch(model, data, optimizer, device, args, class_weights):
    model.train()
    totals = {"loss": 0.0, "bag": 0.0, "align": 0.0}
    count = 0
    for feats, labels, _, mask, _ in data:
        feats, labels, mask = move(feats, labels, mask, device)
        output = model(feats, mask)
        weights = torch.where(labels > 0.5, class_weights[1], class_weights[0])
        loss, terms = model.loss(
            output, labels, mask, args.lambda_align, weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        size = len(labels)
        totals["loss"] += float(loss.detach()) * size
        for name, value in terms.items():
            totals[name] += float(value) * size
        count += size
    return {name: value / max(count, 1) for name, value in totals.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--residual-alpha", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool-temperature", type=float, default=0.25)
    parser.add_argument("--alignment-temperature", type=float, default=0.10)
    parser.add_argument("--lambda-align", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a formal run")
    device = "cuda"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    version_path = output_dir / "code_version.txt"
    state_path = output_dir / "training_state.pt"
    completion_path = output_dir / "training_complete.json"
    predictions_path = output_dir / "predictions.jsonl"
    canonical_arguments = vars(args).copy()
    canonical_arguments.pop("resume")
    config = {
        "corpus": args.corpus,
        "seed": args.seed,
        "method": "modality-typed residual multi-scale experts with class-aware multi-positive bidirectional alignment",
        "official_reba_reference": str(REPO / "third_party/REBA-WSVAD"),
        "official_test-selected_checkpoint_logic_used": False,
        "split_protocol": "frozen same-corpus train; frozen validation only selects checkpoint; test prediction after training_complete",
        "features": "aligned 1 fps audio/visual/text from src.hate_local_features",
        "arguments": canonical_arguments,
        "input_provenance": {
            "train_split": str(Path(hdata.SPLIT_ROOT) / f"{args.corpus}_train.txt"),
            "validation_split": str(Path(hdata.SPLIT_ROOT) / f"{args.corpus}_val.txt"),
            "test_split": str(Path(hdata.SPLIT_ROOT) / f"{args.corpus}_test.txt"),
            "scoped_train_video_labels": str(
                REPO / "results/reproduction/splits/scoped_labels"
                / f"{args.corpus}_train.json"
            ),
            "scoped_validation_video_labels": str(
                REPO / "results/reproduction/splits/scoped_labels"
                / f"{args.corpus}_val.json"
            ),
            "feature_producer": str(REPO / "src/hate_local_features.py"),
            "shared_dataset": str(REPO / "src/multimodal_video_data.py"),
            "shared_protocol": str(REPO / "src/scoped_video_protocol.py"),
        },
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_predictions_and_gt_may_inform_later_development": True,
    }
    if args.resume:
        if not all(path.is_file() for path in (config_path, version_path, state_path)):
            raise RuntimeError("resume metadata or state missing")
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("resume config mismatch")
        if version_path.read_text() != CODE_VERSION_DESCRIPTION + "\n":
            raise RuntimeError("resume code-version description mismatch")
        if predictions_path.exists():
            raise RuntimeError("prediction already complete")
    else:
        occupied = [
            path.name for path in
            (config_path, version_path, state_path, completion_path, predictions_path)
            if path.exists()
        ]
        if occupied:
            raise RuntimeError("formal run directory is occupied: " + ", ".join(occupied))
        atomic_text(config_path, json.dumps(config, indent=2) + "\n")
        atomic_text(version_path, CODE_VERSION_DESCRIPTION + "\n")

    train_ids, val_ids = hdata.load_train_val(args.corpus, None, 0.1, args.seed)
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    train_data = multimodal_loader(
        args.corpus, train_ids, train_labels, args.batch_size, args.workers, True,
        args.seed,
    )
    val_data = multimodal_loader(
        args.corpus, val_ids, val_labels, args.batch_size, args.workers, False,
        args.seed,
    )
    counts = torch.tensor(
        [sum(value == cls for value in train_labels.values()) for cls in (0, 1)],
        dtype=torch.float32, device=device,
    )
    class_weights = len(train_ids) / (2.0 * counts)
    model = TypedREBA(
        DIMS, width=args.width, residual_alpha=args.residual_alpha,
        dropout=args.dropout, pool_temperature=args.pool_temperature,
        alignment_temperature=args.alignment_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_ap, best_epoch, best_state = -1.0, None, None
    history, start_epoch = [], 1
    if args.resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_ap, best_epoch, best_state = (
            state["best_ap"], state["best_epoch"], state["best_state"]
        )
        history = state["history"]
        start_epoch = state["completed_epoch"] + 1
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        train_data.generator.set_state(state["loader_generator_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        print(f"resuming_after_epoch={start_epoch - 1}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        stats = train_epoch(model, train_data, optimizer, device, args, class_weights)
        _, val_bags, val_targets = predict(model, val_data, device)
        ordered = sorted(val_bags)
        val_ap = float(average_precision_score(
            [val_targets[item] for item in ordered],
            [val_bags[item] for item in ordered],
        ))
        stats.update({"epoch": epoch, "validation_video_ap": val_ap})
        history.append(stats)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        atomic_torch(state_path, {
            "completed_epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_ap": best_ap,
            "best_epoch": best_epoch, "best_state": best_state,
            "history": history, "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "loader_generator_state": train_data.generator.get_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        })
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} loss={stats['loss']:.6f} "
                f"validation_video_ap={val_ap:.6f}", flush=True,
            )
    if best_state is None:
        raise RuntimeError("checkpoint selection produced no state")
    model.load_state_dict(best_state)
    atomic_torch(output_dir / "model.pt", model.state_dict())
    atomic_text(output_dir / "train_history.json", json.dumps({
        "selected_epoch": best_epoch,
        "selected_validation_video_ap": best_ap,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }, indent=2) + "\n")
    atomic_text(completion_path, json.dumps({
        "status": "training_complete", "selected_epoch": best_epoch,
        "epochs_completed": args.epochs,
    }, indent=2) + "\n")
    print(f"training_complete corpus={args.corpus} selected_epoch={best_epoch}", flush=True)

    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    placeholders = {video_id: 0 for video_id in test_ids}
    test_data = multimodal_loader(
        args.corpus, test_ids, placeholders, args.batch_size, args.workers,
        False, args.seed,
    )
    scores, _, _ = predict(model, test_data, device, include_control=True)
    rows = []
    for video_id in test_ids:
        rows.append(json.dumps({
            "video_id": video_id,
            **{
                branch: [float(value) for value in values[video_id]]
                for branch, values in scores.items()
            },
        }))
    atomic_text(predictions_path, "\n".join(rows) + "\n")
    atomic_text(completion_path, json.dumps({
        "status": "prediction_complete", "selected_epoch": best_epoch,
        "epochs_completed": args.epochs, "test_videos": len(test_ids),
    }, indent=2) + "\n")
    print(
        f"complete corpus={args.corpus} selected_epoch={best_epoch} "
        f"test_videos={len(test_ids)} elapsed_seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
