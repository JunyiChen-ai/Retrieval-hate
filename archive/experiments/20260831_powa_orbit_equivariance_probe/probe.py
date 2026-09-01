#!/usr/bin/env python3
"""Validation-only temporal-origin equivariance upper bound for POWA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from powa_macil.dataset import PowaTestDataset, usable_text_ids  # noqa: E402
from powa_macil.model import POWAMACIL  # noqa: E402


CHECKPOINTS = {
    "hatemm": Path("/home/jehc223/Hate-follow-up/results/reproduction/"
                   "powa_macil/final_maskfix_finetune_hatemm_seed234_e5"),
    "hateclipseg": REPO / "runs/20260831_powa_starting_point/"
                          "hcs_maskfix_seed234",
}
CORPORA = tuple(CHECKPOINTS)


def stable_seed(corpus, video_id):
    raw = hashlib.sha256(f"20260831:{corpus}:{video_id}".encode()).digest()
    return int.from_bytes(raw[:8], "little")


def load_model(run_dir, corpus, device):
    meta = json.loads((run_dir / "train_meta.json").read_text())
    args = Namespace(**meta["args"])
    # These archived paths are not read when semantic_grounding is disabled.
    model = POWAMACIL(args, policy=corpus).to(device)
    state = torch.load(run_dir / "model.pth", map_location=device)
    model.load_state_dict(state)
    model.use_policy_residual = not args.typed_only
    return model.eval(), args, meta


@torch.inference_mode()
def predict_order(model, tensors, corpus, order):
    f_v, f_a, f_t = tensors
    order_t = torch.as_tensor(order, device=f_v.device, dtype=torch.long)
    inv_t = torch.argsort(order_t)
    transformed = tuple(x.index_select(1, order_t) for x in (f_v, f_a, f_t))
    length = torch.full(
        (f_v.shape[0],), f_v.shape[1], dtype=torch.long, device=f_v.device
    )
    transformed_v, transformed_a, transformed_t = transformed
    out = model(transformed_a, transformed_v, transformed_t, length,
                policy=corpus)
    return out["frame_prob"].mean(0).index_select(0, inv_t).cpu().numpy()


def metrics(scores, gt):
    ids = set(scores)
    report = evaluate_scores(scores, {video_id: gt[video_id] for video_id in ids}, ids)
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def run_corpus(corpus, device):
    run_dir = CHECKPOINTS[corpus]
    model, args, meta = load_model(run_dir, corpus, device)
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    val_ids = usable_text_ids(corpus, val_ids)
    dataset = PowaTestDataset(corpus, val_ids, args.max_seqlen, args.grid, "av")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    gt = hdata.gt_arrays(corpus, "val")
    branches = {name: {} for name in ("original", "cyclic_mean", "random_mean")}
    cyclic_mae = []
    for item in loader:
        f_v, f_a, f_t, index_map, _, video_id = item
        video_id = video_id[0]
        tensors = (f_v[0].float().to(device), f_a[0].float().to(device),
                   f_t[0].float().to(device))
        width = tensors[0].shape[1]
        identity = np.arange(width)
        original = predict_order(model, tensors, corpus, identity)
        cyclic = [original]
        for fraction in (0.25, 0.5, 0.75):
            shift = int(round(width * fraction)) % width
            order = np.roll(identity, shift)
            restored = predict_order(model, tensors, corpus, order)
            cyclic.append(restored)
            cyclic_mae.append(float(np.mean(np.abs(restored - original))))
        rng = np.random.default_rng(stable_seed(corpus, video_id))
        random_views = [original]
        for _ in range(3):
            random_views.append(
                predict_order(model, tensors, corpus, rng.permutation(width))
            )
        second_index = index_map[0].numpy()
        branches["original"][video_id] = original[second_index]
        branches["cyclic_mean"][video_id] = np.mean(cyclic, axis=0)[second_index]
        branches["random_mean"][video_id] = np.mean(random_views, axis=0)[second_index]
        print(f"PROGRESS {corpus} {len(branches['original'])}/{len(val_ids)} "
              f"{video_id}", flush=True)
    result = {
        "checkpoint": str(run_dir.resolve()),
        "checkpoint_selected_epoch": meta.get("selected_epoch"),
        "videos": len(val_ids),
        "mean_cyclic_equivariance_mae": float(np.mean(cyclic_mae)),
        "metrics": {name: metrics(scores, gt) for name, scores in branches.items()},
    }
    base = result["metrics"]["original"]
    cyclic = result["metrics"]["cyclic_mean"]
    random_control = result["metrics"]["random_mean"]
    result["deltas"] = {
        "cyclic_within_minus_original": cyclic["within_roc"] - base["within_roc"],
        "cyclic_ap_minus_original": cyclic["pooled_ap"] - base["pooled_ap"],
        "cyclic_roc_minus_original": cyclic["pooled_roc"] - base["pooled_roc"],
        "cyclic_within_minus_random": cyclic["within_roc"] - random_control["within_roc"],
    }
    result["checks"] = {
        "within_gain_at_least_0.020": result["deltas"]["cyclic_within_minus_original"] >= .020,
        "pooled_ap_drop_at_most_0.005": result["deltas"]["cyclic_ap_minus_original"] >= -.005,
        "pooled_roc_drop_at_most_0.005": result["deltas"]["cyclic_roc_minus_original"] >= -.005,
        "cyclic_beats_random_by_0.010": result["deltas"]["cyclic_within_minus_random"] >= .010,
    }
    result["pass"] = all(result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    corpora = {corpus: run_corpus(corpus, args.device) for corpus in CORPORA}
    payload = {
        "date": "2026-08-31",
        "stage": "powa_temporal_orbit_equivariance_upper_bound",
        "split": "validation",
        "test_used": False,
        "method_claimed": False,
        "ensemble_upper_bound_only": True,
        "corpora": corpora,
        "pass": all(result["pass"] for result in corpora.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
