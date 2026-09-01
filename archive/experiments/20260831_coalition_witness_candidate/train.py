#!/usr/bin/env python3
"""Train one fixed coalition-witness arm and immediately infer test scores."""

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
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
MHL = BASE / "multihateloc"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MHL))

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
sys.path.insert(0, str(HERE))
from model import CoalitionModel, MODALITIES  # noqa: E402


ARMS = ("all_subset_mil", "synib", "mobius_nonminimal", "coalition_witness")


def make_loader(corpus, ids, labels, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=mdata.collate,
        num_workers=workers,
        generator=generator if shuffle else None,
    )


def run_epoch(model, loader, optimizer, device, args):
    model.train()
    totals = {"loss": 0.0, "primary": 0.0, "smooth": 0.0}
    count = 0
    for feats, labels, lengths, mask, _ in loader:
        feats = {name: value.to(device, non_blocking=True) for name, value in feats.items()}
        labels = labels.to(device)
        lengths = lengths.to(device)
        mask = mask.to(device)
        output = model(feats, mask, lengths, args.arm)
        loss, terms = model.loss(
            output, labels, mask, lengths, args.arm, lambda_smooth=args.lambda_smooth
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        size = len(labels)
        totals["loss"] += float(loss.detach()) * size
        totals["primary"] += float(terms["primary"].detach()) * size
        totals["smooth"] += float(terms["smooth"].detach()) * size
        count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device, arm, include_diagnostics=False):
    model.eval()
    scores, video_scores, diagnostics = {}, {}, {}
    reconstruction_residual = 0.0
    for feats, _, lengths, mask, video_ids in loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        lengths_device = lengths.to(device)
        mask_device = mask.to(device)
        output = model(feats, mask_device, lengths_device, arm)
        for index, video_id in enumerate(video_ids):
            length = int(lengths[index])
            frame = output["frame_scores"][index, :length].cpu().numpy()
            scores[video_id] = frame
            if arm == "coalition_witness":
                video_scores[video_id] = float(torch.sigmoid(output["video_logits"][index]).cpu())
            else:
                video_scores[video_id] = float(output["video_scores"][index].cpu())
            if arm in ("mobius_nonminimal", "coalition_witness"):
                atoms = output["coalition_logits"][index, :length]
                tau = model.temperature
                centered_log_sum = tau * (
                    torch.logsumexp(atoms / tau, dim=-1) - np.log(7.0)
                )
                residual = torch.max(torch.abs(centered_log_sum - output["frame_logits"][index, :length]))
                reconstruction_residual = max(reconstruction_residual, float(residual.cpu()))
                if include_diagnostics:
                    diagnostics[video_id] = model.posterior_summary(atoms, length)
    return scores, video_scores, diagnostics, reconstruction_residual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--k-proportion", type=int, default=3)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--synib-margin", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA unavailable")
    device = torch.device(args.device)

    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    gold = hdata.gt_arrays(args.corpus, "test")
    test_ids = [video_id for video_id in hdata.load_split(args.corpus, "test") if video_id in gold]
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise RuntimeError("train/validation/test split overlap")

    train_loader = make_loader(
        args.corpus, train_ids, labels, args.batch_size, True, 4, args.seed
    )
    val_loader = make_loader(
        args.corpus, val_ids, labels, args.batch_size, False, 2, args.seed
    )
    test_loader = make_loader(
        args.corpus, test_ids, labels, args.batch_size, False, 2, args.seed
    )

    dims = {name: mdata.FEATURE_DIMS[name] for name in MODALITIES}
    model = CoalitionModel(
        dims,
        hidden=args.hidden,
        embed=args.embed,
        dropout=args.dropout,
        temperature=args.temperature,
        k_proportion=args.k_proportion,
        synib_margin=args.synib_margin,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_count = sum(value.numel() for value in model.parameters())

    history = []
    best_ap = -1.0
    best_epoch = None
    best_state = None
    started = time.time()
    for epoch in range(1, args.max_epoch + 1):
        stats = run_epoch(model, train_loader, optimizer, device, args)
        _, val_video, _, _ = predict(model, val_loader, device, args.arm)
        ordered = sorted(val_video)
        val_ap = float(average_precision_score(
            [labels[video_id] for video_id in ordered],
            [val_video[video_id] for video_id in ordered],
        ))
        stats.update({"epoch": epoch, "checkpoint_selection_video_ap": val_ap})
        history.append(stats)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{args.corpus}/{args.arm} epoch={epoch} loss={stats['loss']:.5f} "
                f"selection_ap={val_ap:.5f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out_dir / "model.pt")

    test_scores, _, diagnostics, reconstruction_residual = predict(
        model, test_loader, device, args.arm, include_diagnostics=True
    )
    with (args.out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            handle.write(json.dumps({
                "video_id": video_id,
                "score_full": [round(float(value), 7) for value in test_scores[video_id]],
            }) + "\n")

    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update({
        "split_policy": "official train; official validation only selects checkpoint; immediate test",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
        "modalities": list(MODALITIES),
        "coalition_subsets_bitmask": list(range(1, 8)),
        "parameter_count": int(parameter_count),
    })
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out_dir / "code_version.txt").write_text(
        "Working-tree snapshot dated 2026-08-31; coalition witness pilot entrypoint\n"
    )
    train_record = {
        "corpus": args.corpus,
        "arm": args.arm,
        "n_train": len(train_ids),
        "n_validation": len(val_ids),
        "n_test": len(test_ids),
        "selected_epoch": best_epoch,
        "checkpoint_selection_video_ap": best_ap,
        "history": history,
        "full_score_reconstruction_max_abs_residual": reconstruction_residual,
        "test_coalition_diagnostics": diagnostics,
        "wall_seconds": time.time() - started,
    }
    (args.out_dir / "train_record.json").write_text(
        json.dumps(train_record, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "complete",
        "corpus": args.corpus,
        "arm": args.arm,
        "selected_epoch": best_epoch,
        "scores": str(args.out_dir / "scores.jsonl"),
    }), flush=True)


if __name__ == "__main__":
    main()
