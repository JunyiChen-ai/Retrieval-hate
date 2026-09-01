"""Train train-only fold seeds and emit deletion-sensitivity pseudo evidence."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from imports import base_data, base_model
from model import CarrierItS2CLR
from protocol import supervised_split
from states import batch_states


def loader(corpus, ids, labels, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        base_data.MultiModalDataset(corpus, ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=base_data.collate,
        generator=generator,
        drop_last=False,
    )


def negative_centroids(corpus, train_ids, labels):
    sums = {name: np.zeros(base_data.FEATURE_DIMS[name], dtype=np.float64)
            for name in base_data.MODALITIES}
    counts = {name: 0 for name in base_data.MODALITIES}
    for video_id in train_ids:
        if labels[video_id] != 0:
            continue
        for name in base_data.MODALITIES:
            rows = np.load(base_data.feature_path(name, corpus, video_id), mmap_mode="r")
            sums[name] += np.asarray(rows, dtype=np.float64).sum(0)
            counts[name] += len(rows)
    if any(counts[name] == 0 for name in base_data.MODALITIES):
        raise RuntimeError("negative train frames are required for every modality")
    return {name: torch.tensor(sums[name] / counts[name], dtype=torch.float32)
            for name in base_data.MODALITIES}


def train_fold(corpus, ids, labels, args, fold, device):
    fold_loader = loader(
        corpus, ids, labels, args.batch_size, args.workers, True,
        args.seed + fold,
    )
    model = CarrierItS2CLR(
        {name: base_data.FEATURE_DIMS[name] for name in base_data.MODALITIES},
        args.arm, hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        k_proportion=args.k_proportion, temperature=args.temperature,
        max_instances=args.max_instances,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.seed_epochs + 1):
        model.train()
        sums, count = {"loss": 0.0, "mil": 0.0, "smooth": 0.0,
                       "contrast": 0.0}, 0
        for feats, labels_batch, lengths, mask, _ in fold_loader:
            feats = {key: value.to(device) for key, value in feats.items()}
            labels_batch = labels_batch.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            output = model(feats, mask)
            mil, _ = model.backbone.mil_loss(
                output["probs"], mask, lengths, labels_batch
            )
            smooth = model.backbone.smoothness_loss(output["probs"], mask)
            contrast = model.backbone.contrastive_loss(output["embeds"], mask)
            loss = mil + args.lambda_smooth * smooth + args.lambda_contrast * contrast
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(labels_batch)
            for key, value in (("loss", loss), ("mil", mil),
                               ("smooth", smooth), ("contrast", contrast)):
                sums[key] += float(value.detach()) * n
            count += n
        record = {key: value / count for key, value in sums.items()}
        record["epoch"] = epoch
        history.append(record)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.seed_epochs:
            print(json.dumps({"fold": fold, **record}), flush=True)
    return model, history


def refine_fold(model, corpus, ids, labels, cache, args, fold, device,
                start_epoch):
    fold_loader = loader(
        corpus, ids, labels, args.batch_size, args.workers, True,
        args.seed + 100 + fold + start_epoch,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for offset in range(args.refresh_every):
        epoch = start_epoch + offset + 1
        model.train()
        sums, count = {"loss": 0.0, "mil": 0.0, "smooth": 0.0,
                       "contrast": 0.0}, 0
        for feats, labels_batch, lengths, mask, video_ids in fold_loader:
            states = batch_states(
                video_ids, labels_batch, lengths, mask, cache, args.arm,
                epoch, args.refine_epochs, args.k_proportion,
            )
            feats = {key: value.to(device) for key, value in feats.items()}
            labels_batch = labels_batch.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)
            output = model(feats, mask)
            loss, terms = model.training_loss(
                output, states.to(device), mask, lengths, labels_batch,
                args.lambda_smooth, args.lambda_contrast,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = len(labels_batch)
            sums["loss"] += float(loss.detach()) * n
            for key in ("mil", "smooth", "contrast"):
                sums[key] += float(terms[key].detach()) * n
            count += n
        record = {key: value / count for key, value in sums.items()}
        record["epoch"] = epoch
        history.append(record)
        print(json.dumps({"fold": fold, "refine": True, **record}), flush=True)
    return history


def neighbor_replacement(rows):
    length = len(rows)
    if length == 1:
        return rows.clone()
    output = torch.empty_like(rows)
    output[0] = rows[1]
    output[-1] = rows[-2]
    if length > 2:
        output[1:-1] = 0.5 * (rows[:-2] + rows[2:])
    return output


@torch.no_grad()
def local_logits_with_replacement(model, output, feats, name, replacement):
    """Recompute only each second's fused head at frozen original DMS weights."""
    modality_index = model.modalities.index(name)
    replaced_embed, _ = model.branches[name](replacement)
    parts = []
    for index, original in enumerate(output["embeds"]):
        chosen = replaced_embed if index == modality_index else original
        scale = output["weights"][:, index] * len(model.modalities)
        parts.append(chosen * scale[:, None, None])
    fused = model.fuse(torch.cat(parts, dim=-1))
    return model.fuse_head(fused).squeeze(-1)


@torch.no_grad()
def pseudo_for_video(model, feats, length, centroids, device):
    feats = {name: value[:, :length].to(device) for name, value in feats.items()}
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    model.eval()
    output = model(feats, mask)
    original_logit = model.fuse_head(output["fused_embed"]).squeeze(-1)
    deletion_centroid, deletion_neighbor = [], []
    branch_score = []
    for name in model.modalities:
        center = centroids[name].to(device)[None, None, :].expand(1, length, -1)
        neighbor = neighbor_replacement(feats[name][0])[None]
        centroid_logit = local_logits_with_replacement(
                model, output, feats, name, center
        )
        neighbor_logit = local_logits_with_replacement(
            model, output, feats, name, neighbor
        )
        deletion_centroid.append((original_logit - centroid_logit)[0].cpu())
        deletion_neighbor.append((original_logit - neighbor_logit)[0].cpu())
        branch_score.append(output["probs"][name][0].cpu())
    return {
        "fused_score": output["probs"]["fused"][0].cpu(),
        "branch_score": torch.stack(branch_score, dim=1),
        "deletion_centroid": torch.stack(deletion_centroid, dim=1),
        "deletion_neighbor": torch.stack(deletion_neighbor, dim=1),
    }


def refresh_change(before, after):
    score_error, score_entries, carrier_changed, entries = 0.0, 0, 0, 0
    for video_id in before:
        old = before[video_id]
        new = after[video_id]
        score_error += float((old["fused_score"] - new["fused_score"]).abs().sum())
        score_entries += len(old["fused_score"])
        old_carrier = (old["deletion_centroid"] > 0) & (old["deletion_neighbor"] > 0)
        new_carrier = (new["deletion_centroid"] > 0) & (new["deletion_neighbor"] > 0)
        carrier_changed += int((old_carrier != new_carrier).sum())
        entries += int(old_carrier.numel())
    return {
        "mean_absolute_fused_score_change": score_error / max(score_entries, 1),
        "carrier_mask_changed_fraction": carrier_changed / max(entries, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True,
                        choices=CarrierItS2CLR.ARMS[1:])
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed-epochs", type=int, default=40)
    parser.add_argument("--refine-epochs", type=int, default=15)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
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
    if args.refine_epochs <= 0 or args.refresh_every <= 0 \
            or args.refine_epochs % args.refresh_every:
        raise SystemExit("refine-epochs must be a positive multiple of refresh-every")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device(args.device)
    train_ids, labels = supervised_split(args.corpus, "train")
    y = np.asarray([labels[video_id] for video_id in train_ids])
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed)
    rows, fold_logs, fold_states = {}, [], []
    started = time.time()
    indices = np.arange(len(train_ids))
    for fold, (fit_index, held_index) in enumerate(splitter.split(indices, y)):
        fit_ids = [train_ids[index] for index in fit_index]
        held_ids = [train_ids[index] for index in held_index]
        fold_centroids = negative_centroids(args.corpus, fit_ids, labels)
        model, history = train_fold(
            args.corpus, fit_ids, labels, args, fold, device
        )
        held_loader = loader(
            args.corpus, held_ids, labels, 1, args.workers, False, args.seed
        )
        for feats, _, lengths, _, video_ids in held_loader:
            video_id = video_ids[0]
            rows[video_id] = pseudo_for_video(
                model.backbone, feats, int(lengths[0]), fold_centroids, device
            )
            rows[video_id]["fold"] = fold
        fold_logs.append({"fold": fold, "n_fit": len(fit_ids),
                          "n_held": len(held_ids), "seed_history": history,
                          "refine_history": []})
        fold_states.append({"model": model, "fit_ids": fit_ids,
                            "held_ids": held_ids, "centroids": fold_centroids})
    if set(rows) != set(train_ids):
        raise RuntimeError("OOF pseudo evidence does not exactly cover train split")
    rng = np.random.default_rng(args.seed)
    for video_id in train_ids:
        length = len(rows[video_id]["fused_score"])
        rows[video_id]["shuffle_key"] = torch.from_numpy(
            rng.random((length, len(base_data.MODALITIES))).astype(np.float32)
        )
    cache = {"corpus": args.corpus, "modalities": tuple(base_data.MODALITIES),
             "train_ids": train_ids, "rows": rows}
    refresh_diagnostics = []
    for start_epoch in range(0, args.refine_epochs, args.refresh_every):
        for fold, state in enumerate(fold_states):
            history = refine_fold(
                state["model"], args.corpus, state["fit_ids"], labels,
                cache, args, fold, device, start_epoch,
            )
            fold_logs[fold]["refine_history"].extend(history)
        refreshed = {}
        for fold, state in enumerate(fold_states):
            held_loader = loader(
                args.corpus, state["held_ids"], labels, 1, args.workers,
                False, args.seed,
            )
            for feats, _, lengths, _, video_ids in held_loader:
                video_id = video_ids[0]
                refreshed[video_id] = pseudo_for_video(
                    state["model"].backbone, feats, int(lengths[0]),
                    state["centroids"], device,
                )
                refreshed[video_id]["fold"] = fold
                refreshed[video_id]["shuffle_key"] = rows[video_id]["shuffle_key"]
        if set(refreshed) != set(train_ids):
            raise RuntimeError("iterative OOF refresh lost train videos")
        refresh_diagnostics.append({
            "after_refine_epoch": start_epoch + args.refresh_every,
            **refresh_change(rows, refreshed),
        })
        rows = refreshed
        cache["rows"] = rows
    payload = {
        "corpus": args.corpus,
        "arm": args.arm,
        "modalities": tuple(base_data.MODALITIES),
        "train_ids": train_ids,
        "config": vars(args),
        "replacement_definitions": {
            "centroid": "per-modality mean over this corpus's negative train frames",
            "neighbor": "per-second mean of immediate same-video temporal neighbors",
            "locality": "original DMS weights frozen; only replaced second enters fused head",
        },
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    log = {
        "corpus": args.corpus,
        "train_only_cross_fitting": True,
        "validation_used": False,
        "test_used": False,
        "n_train": len(train_ids),
        "n_rows": len(rows),
        "wall_seconds": time.time() - started,
        "refresh_diagnostics": refresh_diagnostics,
        "folds": fold_logs,
    }
    (output.parent / "oof_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps({"out": str(output), "n_train": len(rows),
                      "wall_seconds": log["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
