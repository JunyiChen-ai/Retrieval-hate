#!/usr/bin/env python3
"""Audit learned primitive channels against the retained sparse MLLM teacher.

This audit uses train videos only and never reads frame-localization labels.  It
compares each learned primitive probability with every teacher channel at the
sparse teacher-covered locations.  The declared diagonal must outperform a
cyclically wrong channel mapping for the semantic names to be identifiable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hate_common import data as hdata  # noqa: E402
from macilsd.train import _seq_len_of  # noqa: E402
from powa_macil.dataset import (PowaTrainDataset, load_teacher_jsonl,
                                usable_text_ids)  # noqa: E402
from powa_macil.model import POWAMACIL, PRIMITIVES  # noqa: E402


def _binary_cross_entropy(probability, target):
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -(target * np.log(probability) +
             (1.0 - target) * np.log(1.0 - probability)).mean()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--teacher-file", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(os.path.join(args.checkpoint_dir, "train_meta.json")) as fh:
        meta = json.load(fh)
    cfg = SimpleNamespace(**meta["args"])
    teacher = load_teacher_jsonl(args.teacher_file)
    model = POWAMACIL(cfg).to(args.device)
    state = torch.load(os.path.join(args.checkpoint_dir, "model.pth"),
                       map_location=args.device)
    model.load_state_dict(state)
    model.use_policy_residual = not getattr(cfg, "typed_only", False)
    model.eval()

    predicted, targets = [], []
    counts = {}
    with torch.no_grad():
        for corpus in hdata.CORPORA:
            labels = hdata.load_labels(corpus)
            ids = usable_text_ids(corpus, hdata.load_split(corpus, "train"))
            dataset = PowaTrainDataset(
                corpus, ids, labels, cfg.max_seqlen, cfg.grid, "av", 1,
                teacher_records=teacher, permute_teacher_channels=False)
            loader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
            before = len(predicted)
            for f_v, f_a, f_t, _, teacher_target, teacher_mask in loader:
                lengths = _seq_len_of(f_v)
                keep = int(lengths.max())
                valid = (torch.arange(keep)[None] < lengths[:, None])
                f_v = f_v[:, :keep].float().to(args.device)
                f_a = f_a[:, :keep].float().to(args.device)
                f_t = f_t[:, :keep].float().to(args.device)
                out = model(f_a, f_v, f_t, lengths,
                            valid.to(args.device), policy=corpus)
                mask = (teacher_mask[:, :keep] > 0) & valid
                if mask.any():
                    predicted.append(out["primitive_prob"][mask.to(args.device)]
                                     .cpu().numpy())
                    targets.append(teacher_target[:, :keep][mask].numpy())
            counts[corpus] = int(sum(len(x) for x in predicted[before:]))

    if not predicted:
        raise RuntimeError("no sparse teacher-covered locations were found")
    predicted = np.concatenate(predicted)
    targets = np.concatenate(targets)
    n = len(PRIMITIVES)
    correlation = np.asarray([
        [spearmanr(predicted[:, i], targets[:, j]).statistic
         for j in range(n)] for i in range(n)])
    bce = np.asarray([
        [_binary_cross_entropy(predicted[:, i], targets[:, j])
         for j in range(n)] for i in range(n)])
    diagonal = np.arange(n)
    cyclic = (diagonal + 1) % n
    result = {
        "audit_split": "train",
        "uses_frame_localization_labels": False,
        "channels": list(PRIMITIVES),
        "n_teacher_locations": int(len(predicted)),
        "locations_by_corpus": counts,
        "spearman_matrix": correlation.tolist(),
        "bce_matrix": bce.tolist(),
        "declared_mean_spearman": float(correlation[diagonal, diagonal].mean()),
        "cyclic_wrong_mean_spearman": float(correlation[diagonal, cyclic].mean()),
        "declared_mean_bce": float(bce[diagonal, diagonal].mean()),
        "cyclic_wrong_mean_bce": float(bce[diagonal, cyclic].mean()),
    }
    result["declared_minus_cyclic_spearman"] = (
        result["declared_mean_spearman"] -
        result["cyclic_wrong_mean_spearman"])
    result["cyclic_minus_declared_bce"] = (
        result["cyclic_wrong_mean_bce"] - result["declared_mean_bce"])
    out = args.out or os.path.join(args.checkpoint_dir,
                                    "teacher_alignment_audit.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
