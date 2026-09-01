#!/usr/bin/env python3
"""Frozen POWA AWB edge-to-path ordering feasibility on validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from src.powa_residual import load_corpus_powa  # noqa: E402


CHECKPOINTS = {
    "hatemm": REPO / "results/reproduction/powa_macil/final_maskfix_finetune_hatemm_seed234_e5",
    "hateclipseg": REPO / "runs/20260831_powa_starting_point/hcs_maskfix_seed234",
}


def path_occupancy(plan, length_normalized=False):
    """Sum directed edge mass over every timestamp on its inclusive path."""
    crops, width, _ = plan.shape
    out = torch.zeros((crops, width), device=plan.device, dtype=plan.dtype)
    ii = torch.arange(width, device=plan.device)[:, None].expand(width, width)
    jj = torch.arange(width, device=plan.device)[None, :].expand(width, width)
    start = torch.minimum(ii, jj).reshape(-1)
    end = torch.maximum(ii, jj).reshape(-1)
    distance = (end - start + 1).to(plan.dtype)
    for crop in range(crops):
        weight = plan[crop].reshape(-1)
        if length_normalized:
            weight = weight / distance
        delta = torch.zeros(width + 1, device=plan.device, dtype=plan.dtype)
        delta.scatter_add_(0, start, weight)
        delta.scatter_add_(0, end + 1, -weight)
        out[crop] = delta[:-1].cumsum(0)
    return out


def transport(anchor, ordering):
    moved = np.empty_like(anchor)
    moved[np.argsort(ordering, kind="stable")] = np.sort(anchor, kind="stable")
    return moved


def summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


@torch.no_grad()
def analyze(corpus, device="cuda"):
    model, cfg, _, checkpoint_hash = load_corpus_powa(
        CHECKPOINTS[corpus], corpus, device
    )
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    val_ids = usable_text_ids(corpus, val_ids)
    gt = hdata.gt_arrays(corpus, "val")
    val_ids = [video_id for video_id in val_ids if video_id in gt]
    dataset = PowaTestDataset(corpus, val_ids, cfg.max_seqlen, cfg.grid, "av")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=getattr(cfg, "num_workers", 0))
    branches = {name: {} for name in (
        "score_powa", "transport_endpoint", "transport_path_mass",
        "transport_path_length_normalized", "transport_center_first",
        "transport_edge_first",
    )}
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v = f_v[0].to(device)
        f_a = f_a[0].to(device)
        f_t = f_t[0].to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
        raw = model(f_a, f_v, f_t, lengths, policy=corpus)
        if raw["transport"] is None:
            raise RuntimeError("POWA checkpoint did not return AWB transport")
        index = index_map[0].numpy()
        anchor = raw["frame_prob"].mean(0).cpu().numpy()[index]
        endpoint = raw["targeted_hate"].mean(0).cpu().numpy()[index]
        path = path_occupancy(raw["transport"]).mean(0).cpu().numpy()[index]
        normalized = path_occupancy(
            raw["transport"], length_normalized=True
        ).mean(0).cpu().numpy()[index]
        if len(anchor) != int(n_seconds) or len(anchor) != len(gt[video_id]):
            raise RuntimeError(f"alignment mismatch {corpus}/{video_id}")
        branches["score_powa"][video_id] = anchor
        branches["transport_endpoint"][video_id] = transport(anchor, endpoint)
        branches["transport_path_mass"][video_id] = transport(anchor, path)
        branches["transport_path_length_normalized"][video_id] = transport(
            anchor, normalized
        )
        position = np.arange(len(anchor), dtype=float)
        center_order = -np.abs(position - (len(anchor) - 1) / 2.0)
        branches["transport_center_first"][video_id] = transport(
            anchor, center_order
        )
        branches["transport_edge_first"][video_id] = transport(
            anchor, -center_order
        )
    positives = {video_id for video_id in val_ids if labels[video_id] == 1}
    reports = {
        name: evaluate_scores(scores, gt, positives)
        for name, scores in branches.items()
    }
    metrics = {name: summary(report) for name, report in reports.items()}
    anchor = metrics["score_powa"]["within_roc"]
    endpoint = metrics["transport_endpoint"]["within_roc"]
    core = metrics["transport_path_mass"]["within_roc"]
    center = metrics["transport_center_first"]["within_roc"]
    gates = {
        "path_gain_over_powa_at_least_0.020": core >= anchor + .020,
        "path_gain_over_endpoint_at_least_0.010": core >= endpoint + .010,
        "path_beats_fixed_center": core >= center,
    }
    return {
        "checkpoint": str(CHECKPOINTS[corpus].resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
    }


def main():
    corpora = {corpus: analyze(corpus) for corpus in CHECKPOINTS}
    passed = all(row["pass"] for row in corpora.values())
    payload = {
        "date": "2026-08-31",
        "split": "val",
        "test_used": False,
        "status": "ordering_upper_bound_only",
        "corpora": corpora,
        "pass": passed,
        "verdict": "GO_NOVELTY_REVIEW" if passed else "STOP_BEFORE_NOVELTY",
    }
    out = REPO / "runs/20260831_powa_witness_path_probe/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
