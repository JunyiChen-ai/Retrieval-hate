#!/usr/bin/env python3
"""Same-corpus train-span feature ceiling; diagnostic only, never a method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
DATASET_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg"}
CURRENT_DIRS = ("clip_b16_1fps", "vggish_1s", "bert_sentence_1fps")
CURRENT_DIM = 512 + 128 + 768
IB_DIM = 1024
SEED = 234


class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)


def hold_to_seconds(array, seconds, rate):
    index = np.floor((np.arange(seconds) + .5) * rate).astype(np.int64)
    index = np.clip(index, 0, len(array) - 1)
    return np.asarray(array[index], dtype=np.float32)


def current_features(corpus, video_id, seconds):
    parts = []
    for name in CURRENT_DIRS:
        path = REPO / "results/reproduction/features" / name / corpus / f"{video_id}.npy"
        value = np.load(path).astype(np.float32)
        if len(value) < seconds:
            value = np.concatenate(
                (value, np.repeat(value[-1:], seconds - len(value), axis=0)), axis=0
            )
        parts.append(value[:seconds])
    return np.concatenate(parts, axis=1)


def imagebind_features(corpus, video_id, seconds, missing):
    dataset = DATASET_DIR[corpus]
    parts = []
    for channel, rate in (("image", 4.0), ("video", .5), ("audio", .5)):
        path = REPO / f"data/CLIP_Embedding/{dataset}/imagebind_{channel}/{video_id}.npy"
        if not path.exists():
            missing[channel].append(video_id)
            parts.append(np.zeros((seconds, IB_DIM), dtype=np.float32))
            continue
        value = np.load(path).astype(np.float32)
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)
        parts.append(hold_to_seconds(value, seconds, rate))
    return np.concatenate(parts, axis=1)


def pack(corpus, gt, arm, missing):
    packed = []
    for video_id, target in gt.items():
        target = np.asarray(target, dtype=np.float32)
        current = current_features(corpus, video_id, len(target))
        feature = (current if arm == "current" else np.concatenate(
            (current, imagebind_features(corpus, video_id, len(target), missing)), axis=1
        ))
        packed.append((video_id, torch.from_numpy(feature), torch.from_numpy(target)))
    return packed


def run_arm(corpus, arm, epochs, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_gt_path = (REPO / "runs/20260830_powa_within_diagnosis/"
                            "gt_train_diagnosis_only" / f"{corpus}_train.npz")
    with np.load(train_gt_path) as archive:
        train_gt = {key: archive[key] for key in archive.files}
    test_gt = hdata.gt_arrays(corpus, "test")
    missing = {channel: [] for channel in ("image", "video", "audio")}
    train = pack(corpus, train_gt, arm, missing)
    test = pack(corpus, test_gt, arm, missing)
    dimension = train[0][1].shape[1]
    model = Linear(dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positive_rate = float(np.mean([row[2].mean().item() for row in train]))
    pos_weight = torch.tensor(
        (1.0 - positive_rate) / max(positive_rate, 1e-6), device=device
    )
    for epoch in range(epochs):
        order = np.random.permutation(len(train))
        total = 0.0
        model.train()
        for index in order:
            _, feature, target = train[index]
            feature, target = feature.to(device), target.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(feature), target, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"PROGRESS {corpus} {arm} epoch={epoch + 1}/{epochs} "
              f"loss={total / len(train):.6f}", flush=True)
    model.eval()
    scores = {}
    with torch.inference_mode():
        for video_id, feature, _ in test:
            scores[video_id] = model(feature.to(device)).sigmoid().cpu().numpy()
    labels = hdata.load_labels(corpus)
    positive_ids = {video_id for video_id in test_gt if labels[video_id] == 1}
    report = evaluate_scores(scores, test_gt, positive_ids)
    return {
        "feature_dimension": dimension,
        "train_videos": len(train),
        "test_videos": len(test),
        "missing_stream_video_ids": {key: sorted(set(value)) for key, value in missing.items()},
        "metrics": {
            "pooled_ap": report["pr_auc"],
            "pooled_roc": report["roc_auc"],
            "within_roc": report["per_video"]["macro_auc"],
            "within_n": report["per_video"]["n_videos_both_classes"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    corpora = {}
    for corpus in CORPORA:
        arms = {
            arm: run_arm(corpus, arm, args.epochs, args.device)
            for arm in ("current", "current_plus_imagebind_all")
        }
        base, candidate = arms["current"]["metrics"], arms["current_plus_imagebind_all"]["metrics"]
        delta = {key: candidate[key] - base[key]
                 for key in ("pooled_ap", "pooled_roc", "within_roc")}
        checks = {
            "pooled_ap_drop_at_most_0.020": delta["pooled_ap"] >= -.020,
            "pooled_roc_drop_at_most_0.020": delta["pooled_roc"] >= -.020,
            ("within_drop_at_most_0.010" if corpus == "hatemm"
             else "within_gain_at_least_0.020"):
                (delta["within_roc"] >= -.010 if corpus == "hatemm"
                 else delta["within_roc"] >= .020),
        }
        corpora[corpus] = {
            "arms": arms, "deltas_candidate_minus_current": delta,
            "checks": checks, "pass": all(checks.values()),
        }
    payload = {
        "date": "2026-08-31",
        "stage": "imagebind_feature_supervised_ceiling_diagnostic",
        "supervision": "same-corpus train span rasterization; diagnostic only",
        "evaluation_split": "test",
        "test_used_for_error_analysis": True,
        "weak_method_claimed": False,
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
