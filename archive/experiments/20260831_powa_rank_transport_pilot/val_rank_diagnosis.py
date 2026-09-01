#!/usr/bin/env python3
"""Validation-only diagnosis: assign frozen POWA values by another model's rank.

This is a failure-analysis probe, not the candidate method.  It never reads the
test split and does not train or select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, choices=list(hdata.CORPORA))
    p.add_argument("--value-checkpoint", required=True, type=Path)
    p.add_argument("--rank-checkpoint", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, type=Path)
    return p


def load_model(checkpoint: Path, device: str):
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    cfg = SimpleNamespace(**meta["args"])
    model = POWAMACIL(cfg, policy=meta.get("corpus")).to(device)
    state = torch.load(checkpoint / "model.pth", map_location=device)
    legacy_typed_only = "policy_residual_gate" not in state
    model.load_state_dict(state, strict=not legacy_typed_only)
    model.use_policy_residual = (
        not legacy_typed_only and not getattr(cfg, "typed_only", False)
    )
    model.eval()
    return model, cfg, meta


def stable_rank_transport(values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Return an exact permutation of values, ascending with ranks."""
    if values.ndim != 1 or ranks.shape != values.shape:
        raise ValueError("values and ranks must be aligned 1-D arrays")
    slot_order = np.argsort(ranks, kind="stable")
    sorted_values = np.sort(values, kind="stable")
    transported = np.empty_like(values)
    transported[slot_order] = sorted_values
    return transported


@torch.no_grad()
def infer(model, corpus: str, loader, device: str):
    scores = {}
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].float().to(device)
        f_a = f_a[0].float().to(device)
        f_t = f_t[0].float().to(device)
        lengths = torch.full(
            (f_v.shape[0],), f_v.shape[1], dtype=torch.long, device=device
        )
        valid = torch.ones(
            (f_v.shape[0], f_v.shape[1]), dtype=torch.bool, device=device
        )
        output = model(f_a, f_v, f_t, lengths, valid, policy=corpus)
        dense = output["frame_prob"].mean(0).cpu().numpy()
        dense = dense[index_map[0].numpy()]
        if len(dense) != int(n_seconds):
            raise RuntimeError(f"alignment mismatch for {video_id}")
        scores[video_id] = dense.astype(np.float64, copy=False)
    return scores


def metrics(scores, corpus: str):
    gt = hdata.gt_arrays(corpus, "val")
    labels = hdata.load_labels(corpus)
    positive_ids = {video_id for video_id in gt if labels[video_id] == 1}
    report = evaluate_scores(scores, gt, positive_ids)
    if report["n_videos_missing_from_scores"] or report["n_videos_not_in_gold"]:
        raise RuntimeError("validation score coverage mismatch")
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    value_model, value_cfg, value_meta = load_model(
        args.value_checkpoint.resolve(), args.device
    )
    rank_model, rank_cfg, rank_meta = load_model(
        args.rank_checkpoint.resolve(), args.device
    )
    if value_cfg.grid != rank_cfg.grid:
        raise ValueError("checkpoint temporal grids differ")
    ids = usable_text_ids(
        args.corpus, hdata.load_split(args.corpus, "val")
    )
    gt = hdata.gt_arrays(args.corpus, "val")
    ids = [video_id for video_id in ids if video_id in gt]
    dataset = PowaTestDataset(
        args.corpus, ids, value_cfg.max_seqlen, value_cfg.grid, "av"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    value_scores = infer(value_model, args.corpus, loader, args.device)
    rank_scores = infer(rank_model, args.corpus, loader, args.device)
    transported = {
        video_id: stable_rank_transport(value_scores[video_id], rank_scores[video_id])
        for video_id in ids
    }
    multiset_max_abs_error = max(
        float(np.max(np.abs(np.sort(value_scores[v]) - np.sort(transported[v]))))
        for v in ids
    )
    output = {
        "purpose": "validation-only failure diagnosis; not method selection",
        "corpus": args.corpus,
        "split": "val",
        "value_checkpoint": str(args.value_checkpoint.resolve()),
        "rank_checkpoint": str(args.rank_checkpoint.resolve()),
        "value_checkpoint_selected_epoch": value_meta.get("selected_epoch"),
        "rank_checkpoint_selected_epoch": rank_meta.get("selected_epoch"),
        "test_labels_read": False,
        "trained_or_selected_here": False,
        "metrics": {
            "powa_values_original_order": metrics(value_scores, args.corpus),
            "failed_model_original_scores": metrics(rank_scores, args.corpus),
            "powa_values_failed_model_rank": metrics(transported, args.corpus),
        },
        "invariants": {
            "videos": len(ids),
            "per_video_second_score_multiset_max_abs_error": multiset_max_abs_error,
            "exact_within_float64": multiset_max_abs_error == 0.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
