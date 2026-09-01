#!/usr/bin/env python3
"""Does typed-policy similarity improve semantic recurrence propagation?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from src.powa_residual import load_corpus_powa  # noqa: E402


CHECKPOINTS = {
    "hatemm": REPO / "results/reproduction/powa_macil/final_maskfix_finetune_hatemm_seed234_e5",
    "hateclipseg": REPO / "runs/20260831_powa_starting_point/hcs_maskfix_seed234",
}


def load(path):
    with path.open() as handle:
        return {row["video_id"]: row for row in map(json.loads, handle)}


def normalize(rows):
    rows = np.asarray(rows)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def propagate(score, affinity):
    count = max(1, int(.15 * len(score)))
    output = np.empty(len(score), dtype=float)
    for index in range(len(score)):
        neighbor = np.argsort(affinity[index])[-count:]
        logits = affinity[index, neighbor] * 10.0
        weight = np.exp(logits - logits.max())
        weight /= weight.sum()
        output[index] = weight @ score[neighbor]
    return gaussian_filter1d(output, sigma=10, radius=7, mode="nearest")


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
    density_path = REPO / f"runs/20260831_negative_density_probe/{corpus}/scores.jsonl"
    density = load(density_path)
    branches = {name: {} for name in (
        "score_powa", "transport_visual_neighbor_smooth",
        "transport_primitive_neighbor_smooth",
        "transport_policy_gated_neighbor_smooth",
        "transport_shuffled_policy_neighbor_smooth",
    )}
    for f_v, f_a, f_t, index_map, n_seconds, video_id in loader:
        video_id = video_id[0]
        f_v, f_a, f_t = f_v[0].to(device), f_a[0].to(device), f_t[0].to(device)
        lengths = torch.full((f_v.shape[0],), f_v.shape[1], dtype=torch.long)
        raw = model(f_a, f_v, f_t, lengths, policy=corpus)
        index = index_map[0].numpy()
        primitive = raw["primitive_prob"].mean(0).cpu().numpy()[index]
        anchor = np.asarray(density[video_id]["score_powa"], dtype=float)
        seed = np.asarray(density[video_id]["score_probe_concat"], dtype=float)
        snippets = align.snippet_bounds(corpus, video_id)
        seconds = align.snippet_index_for_seconds(snippets, int(n_seconds))
        visual_file = np.load(align.visual_path(corpus, video_id), mmap_mode="r")
        visual = np.asarray(visual_file).mean(1)[seconds]
        if len(anchor) != len(primitive) or len(anchor) != len(visual):
            raise RuntimeError(f"alignment mismatch {corpus}/{video_id}")
        visual_affinity = normalize(visual) @ normalize(visual).T
        primitive_affinity = normalize(primitive) @ normalize(primitive).T
        policy_affinity = .5 * (visual_affinity + primitive_affinity)
        shuffled = np.roll(primitive_affinity, 1, axis=0)
        shuffled = np.roll(shuffled, 1, axis=1)
        signals = {
            "transport_visual_neighbor_smooth": propagate(seed, visual_affinity),
            "transport_primitive_neighbor_smooth": propagate(seed, primitive_affinity),
            "transport_policy_gated_neighbor_smooth": propagate(seed, policy_affinity),
            "transport_shuffled_policy_neighbor_smooth": propagate(
                seed, .5 * (visual_affinity + shuffled)
            ),
        }
        branches["score_powa"][video_id] = anchor
        for name, signal in signals.items():
            branches[name][video_id] = transport(anchor, signal)
    positives = {video_id for video_id in val_ids if labels[video_id] == 1}
    metrics = {
        name: summary(evaluate_scores(scores, gt, positives))
        for name, scores in branches.items()
    }
    anchor = metrics["score_powa"]["within_roc"]
    visual = metrics["transport_visual_neighbor_smooth"]["within_roc"]
    core = metrics["transport_policy_gated_neighbor_smooth"]["within_roc"]
    gates = {
        "gain_over_powa_at_least_0.020": core >= anchor + .020,
        "gain_over_visual_only_at_least_0.010": core >= visual + .010,
    }
    return {
        "checkpoint": str(CHECKPOINTS[corpus].resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "density_artifact": str(density_path.resolve()),
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
    out = REPO / "runs/20260831_policy_recurrence_probe/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
