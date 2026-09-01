#!/usr/bin/env python3
"""Train one same-corpus normal-reference witness and emit test scores."""

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
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from model import MODALITIES, NormalReferenceWitness  # noqa: E402
from protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402


DIMS = {"audio": 128, "visual": 1024, "text": 768}
CODE_VERSION_DESCRIPTION = "2026-08-31 typed shared-capacity normal-reference witness pilot"


class VideoDataset(tdata.Dataset):
    def __init__(self, corpus: str, video_ids, labels):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        parts = aligned_local_features(self.corpus, video_id)
        feats = {name: torch.from_numpy(parts[name]) for name in MODALITIES}
        lengths = {len(value) for value in feats.values()}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise RuntimeError(f"unaligned local features for {self.corpus}/{video_id}")
        return feats, float(self.labels[video_id]), video_id


def collate(items):
    lengths = torch.tensor([len(item[0]["audio"]) for item in items], dtype=torch.long)
    longest = int(lengths.max())
    feats = {
        name: torch.zeros(len(items), longest, DIMS[name], dtype=torch.float32)
        for name in MODALITIES
    }
    for row, (parts, _, _) in enumerate(items):
        length = int(lengths[row])
        for name in MODALITIES:
            feats[name][row, :length] = parts[name]
    mask = torch.arange(longest)[None, :] < lengths[:, None]
    labels = torch.tensor([item[1] for item in items], dtype=torch.float32)
    return feats, labels, lengths, mask, [item[2] for item in items]


def loader(corpus, ids, labels, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return tdata.DataLoader(
        VideoDataset(corpus, ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )


def move(feats, labels, mask, device):
    return (
        {name: value.to(device, non_blocking=True) for name, value in feats.items()},
        labels.to(device),
        mask.to(device),
    )


def train_epoch(model, data, optimizer, device, args):
    model.train()
    totals = {"loss": 0.0, "bag": 0.0, "temporal": 0.0, "negative_cost": 0.0}
    count = 0
    for feats, labels, _, mask, _ in data:
        feats, labels, mask = move(feats, labels, mask, device)
        output = model(feats, mask, reference_gradient_mask=(labels == 0))
        loss, terms = model.loss(
            output, labels, mask, args.lambda_temporal, args.lambda_negative_cost
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


@torch.no_grad()
def predict(model, data, device, include_controls=False):
    model.eval()
    scores, bags, labels_out = {}, {}, {}
    for feats, labels, lengths, mask, video_ids in data:
        feats, labels_device, mask_device = move(feats, labels, mask, device)
        output = model(feats, mask_device, include_controls=include_controls)
        score_keys = [name for name in output if name.startswith("score")]
        for name in score_keys:
            scores.setdefault(name, {})
        for row, video_id in enumerate(video_ids):
            length = int(lengths[row])
            for name in score_keys:
                score = output[name][row, :length].detach().cpu().numpy()
                if score.shape != (length,) or not np.isfinite(score).all():
                    raise RuntimeError(f"invalid {name} for {video_id}")
                scores[name][video_id] = score
            bags[video_id] = float(output["bag_probability"][row])
            labels_out[video_id] = int(labels_device[row])
    return scores, bags, labels_out


def atomic_text(path: Path, value: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--reject-cost", type=float, default=1.0)
    parser.add_argument("--transport-steps", type=int, default=8)
    parser.add_argument("--pool-power", type=float, default=8.0)
    parser.add_argument("--lambda-temporal", type=float, default=0.10)
    parser.add_argument("--lambda-negative-cost", type=float, default=0.20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted training loop in the same run directory",
    )
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
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "training_state.pt"
    complete_path = output_dir / "training_complete.json"
    if predictions_path.exists():
        raise RuntimeError("formal predictions already exist; use a new run directory")
    canonical_arguments = vars(args).copy()
    canonical_arguments.pop("resume")
    config = {
        "corpus": args.corpus,
        "seed": args.seed,
        "split_protocol": "same-corpus frozen train manifest; frozen validation manifest used only for checkpoint selection; test prediction only after training completion",
        "features": "aligned 1 fps audio/visual/text from src.hate_local_features",
        "model": "typed shared-capacity reject-option witness with one shared latent normal atom bank",
        "arguments": canonical_arguments,
        "input_provenance": {
            "train_split": str(hdata.SPLIT_ROOT + f"/{args.corpus}_train.txt"),
            "validation_split": str(hdata.SPLIT_ROOT + f"/{args.corpus}_val.txt"),
            "test_split": str(hdata.SPLIT_ROOT + f"/{args.corpus}_test.txt"),
            "scoped_train_video_labels": str(
                REPO / "results/reproduction/splits/scoped_labels" / f"{args.corpus}_train.json"
            ),
            "scoped_validation_video_labels": str(
                REPO / "results/reproduction/splits/scoped_labels" / f"{args.corpus}_val.json"
            ),
            "feature_producer": str(REPO / "src/hate_local_features.py"),
        },
        "resume_supported": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_predictions_and_gt_may_inform_later_development": True,
    }
    if args.resume:
        if complete_path.exists():
            prior_status = json.loads(complete_path.read_text()).get("status")
            if prior_status == "prediction_complete":
                raise RuntimeError("test prediction is already complete; do not resume this run")
            if prior_status != "training_complete":
                raise RuntimeError("unrecognized completion marker state")
        if not config_path.exists() or not version_path.exists() or not progress_path.exists():
            raise RuntimeError("resume requested but run metadata or training state is missing")
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("resume arguments or input provenance differ from the run config")
        if version_path.read_text() != CODE_VERSION_DESCRIPTION + "\n":
            raise RuntimeError("resume code-version description mismatch")
    else:
        occupied = [
            path.name
            for path in (config_path, version_path, progress_path, complete_path)
            if path.exists()
        ]
        if occupied:
            raise RuntimeError(
                "run directory already contains formal artifacts; use --resume or a new directory: "
                + ", ".join(occupied)
            )
        atomic_text(config_path, json.dumps(config, indent=2) + "\n")
        atomic_text(version_path, CODE_VERSION_DESCRIPTION + "\n")

    train_ids, val_ids = hdata.load_train_val(args.corpus, None, 0.1, args.seed)
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    val_labels = scoped_video_labels(args.corpus, "val", val_ids)
    train_data = loader(
        args.corpus, train_ids, train_labels, args.batch_size, args.workers, True, args.seed
    )
    val_data = loader(
        args.corpus, val_ids, val_labels, args.batch_size, args.workers, False, args.seed
    )
    model = NormalReferenceWitness(
        DIMS,
        hidden=args.hidden,
        embed=args.embed,
        atoms=args.atoms,
        temperature=args.temperature,
        reject_cost=args.reject_cost,
        transport_steps=args.transport_steps,
        pool_power=args.pool_power,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_ap = -1.0
    best_epoch = None
    best_state = None
    history = []
    start_epoch = 1
    started = time.time()
    if args.resume:
        state = torch.load(progress_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_ap = float(state["best_ap"])
        best_epoch = state["best_epoch"]
        best_state = state["best_state"]
        history = state["history"]
        start_epoch = int(state["completed_epoch"]) + 1
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        train_data.generator.set_state(state["loader_generator_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        print(f"resuming_after_epoch={start_epoch - 1}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        stats = train_epoch(model, train_data, optimizer, device, args)
        _, val_bags, val_labels = predict(model, val_data, device)
        ordered = sorted(val_bags)
        val_ap = float(
            average_precision_score(
                [val_labels[item] for item in ordered],
                [val_bags[item] for item in ordered],
            )
        )
        stats.update({"epoch": epoch, "validation_video_ap": val_ap})
        history.append(stats)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        atomic_torch_save(
            progress_path,
            {
                "completed_epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "loader_generator_state": train_data.generator.get_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} loss={stats['loss']:.6f} "
                f"validation_video_ap={val_ap:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("validation checkpoint selection produced no checkpoint")
    model.load_state_dict(best_state)
    atomic_torch_save(output_dir / "model.pt", model.state_dict())
    atomic_text(
        output_dir / "train_history.json",
        json.dumps(
            {
                "selected_epoch": best_epoch,
                "selected_validation_video_ap": best_ap,
                "history": history,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n",
    )
    atomic_text(
        complete_path,
        json.dumps(
            {
                "status": "training_complete",
                "selected_epoch": best_epoch,
                "epochs_completed": args.epochs,
                "test_prediction_started": False,
            },
            indent=2,
        )
        + "\n",
    )
    print(
        f"training_complete corpus={args.corpus} selected_epoch={best_epoch}",
        flush=True,
    )

    # This point is the sole transition from training/checkpoint selection to
    # test production.  No test labels or temporal gold are loaded here.
    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    test_placeholders = {video_id: 0 for video_id in test_ids}
    test_data = loader(
        args.corpus, test_ids, test_placeholders, args.batch_size, args.workers, False, args.seed
    )
    scores, _, _ = predict(model, test_data, device, include_controls=True)
    lines = []
    for video_id in test_ids:
        lines.append(
            json.dumps(
                {
                    "video_id": video_id,
                    **{
                        name: [float(value) for value in branch[video_id]]
                        for name, branch in scores.items()
                    },
                }
            )
        )
    atomic_text(predictions_path, "\n".join(lines) + "\n")
    atomic_text(
        complete_path,
        json.dumps(
            {
                "status": "prediction_complete",
                "selected_epoch": best_epoch,
                "epochs_completed": args.epochs,
                "test_prediction_started": True,
                "test_videos": len(test_ids),
            },
            indent=2,
        )
        + "\n",
    )
    print(
        f"complete corpus={args.corpus} selected_epoch={best_epoch} "
        f"test_videos={len(test_ids)} elapsed_seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
