#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train import (build_model, hdata, make_loader,
                   scoped_video_labels)  # noqa: E402

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/duplex"))
from frame_eval_common import rank_roc_auc  # noqa: E402


class Args:
    def __init__(self, values):
        self.__dict__.update(values)


def load_model(config, checkpoint, n_video_ids):
    model = build_model(Args(config), n_video_ids)
    state = torch.load(checkpoint, map_location=config["device"], weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.set_grl(0.0, 0.0)
    return model


def subsample(indices, maximum=32):
    if len(indices) <= maximum:
        return indices
    chosen = np.linspace(0, len(indices) - 1, maximum).round().astype(int)
    return indices[chosen]


@torch.no_grad()
def extract_probe_split(model, loader, video_index, position_bins, device):
    parts = {"train": {"x": [], "video": [], "position": []},
             "eval": {"x": [], "video": [], "position": []}}
    for feats, _, lengths, mask, video_ids in loader:
        feats = {name: value.to(device) for name, value in feats.items()}
        output = model(feats, mask.to(device))
        local = output["local_rep"].cpu().numpy()
        for row, video_id in enumerate(video_ids):
            length = int(lengths[row])
            if length < 2:
                raise RuntimeError(f"video cannot enter disjoint probe split: {video_id}")
            all_indices = np.arange(length)
            for split, indices in (("train", all_indices[::2]),
                                   ("eval", all_indices[1::2])):
                if len(indices) == 0:
                    raise RuntimeError(f"empty {split} probe split: {video_id}")
                indices = subsample(indices)
                parts[split]["x"].append(local[row, indices])
                parts[split]["video"].append(np.full(
                    len(indices), video_index[video_id], dtype=np.int64))
                position = np.minimum(
                    (indices * position_bins // length), position_bins - 1)
                parts[split]["position"].append(position.astype(np.int64))
    for split in parts.values():
        for key in split:
            split[key] = np.concatenate(split[key], axis=0)
    train_classes = set(parts["train"]["video"].tolist())
    eval_classes = set(parts["eval"]["video"].tolist())
    if train_classes != eval_classes or len(train_classes) != len(video_index):
        raise RuntimeError("video-ID probe does not contain every ID in both splits")
    return parts


def centroid_accuracy(train_x, train_y, eval_x, eval_y, n_classes):
    x = F.normalize(torch.from_numpy(train_x).float(), dim=1)
    y = torch.from_numpy(train_y).long()
    sums = torch.zeros(n_classes, x.shape[1])
    sums.index_add_(0, y, x)
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    centroids = F.normalize(sums / counts[:, None], dim=1)
    query = F.normalize(torch.from_numpy(eval_x).float(), dim=1)
    prediction = (query @ centroids.t()).argmax(1).numpy()
    return float(np.mean(prediction == eval_y))


def probe_metrics(model, loader, video_index, position_bins, device):
    split = extract_probe_split(model, loader, video_index, position_bins, device)
    return {
        "video_id_accuracy": centroid_accuracy(
            split["train"]["x"], split["train"]["video"],
            split["eval"]["x"], split["eval"]["video"], len(video_index)),
        "position_bin_accuracy": centroid_accuracy(
            split["train"]["x"], split["train"]["position"],
            split["eval"]["x"], split["eval"]["position"], position_bins),
        "n_probe_train": int(len(split["train"]["video"])),
        "n_probe_eval": int(len(split["eval"]["video"])),
        "video_id_chance": 1.0 / len(video_index),
        "position_bin_chance": 1.0 / position_bins,
    }


def read_scores(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = np.asarray(row["score_fused"], float)
    return rows


def normalized_profile(values, size=201):
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, size)
    return np.interp(target, source, np.asarray(values, float))


def position_risk(corpus, run_root):
    gold = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    eligible = [video_id for video_id, target in gold.items()
                if labels[video_id] == 1 and len(np.unique(target)) == 2]
    profiles = {video_id: normalized_profile(gold[video_id]) for video_id in eligible}
    control = read_scores(run_root / corpus / "local_control/scores.jsonl")
    core = read_scores(run_root / corpus / "local_adversarial/scores.jsonl")
    rows = []
    for video_id in eligible:
        others = [profile for other_id, profile in profiles.items()
                  if other_id != video_id]
        if not others:
            raise RuntimeError("position-risk needs at least two eligible videos")
        mean_profile = np.mean(others, axis=0)
        score = np.interp(np.linspace(0.0, 1.0, len(gold[video_id])),
                          np.linspace(0.0, 1.0, len(mean_profile)), mean_profile)
        risk = rank_roc_auc(score, gold[video_id])
        delta = (rank_roc_auc(core[video_id], gold[video_id]) -
                 rank_roc_auc(control[video_id], gold[video_id]))
        rows.append((video_id, risk, delta))
    threshold = float(np.median([row[1] for row in rows]))
    low = [row[2] for row in rows if row[1] < threshold]
    high = [row[2] for row in rows if row[1] >= threshold]
    return {"definition": "leave-one-positive-video-out normalized-position GT profile",
            "median_risk": threshold,
            "low": {"n": len(low), "mean_core_minus_control_within": float(np.mean(low))},
            "high": {"n": len(high), "mean_core_minus_control_within": float(np.mean(high))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())["selected"]
    core_config = selection["args"]
    core_checkpoint = selection_path.parent / selection["trial"] / "checkpoint.pt"
    control_dir = run_root / args.corpus / "local_control"
    control_config = json.loads((control_dir / "config.json").read_text())["args"]
    control_checkpoint = control_dir / "checkpoint.pt"
    train_ids = hdata.load_split(args.corpus, "train")
    train_labels = scoped_video_labels(args.corpus, "train", train_ids)
    video_index = {video_id: index for index, video_id in enumerate(sorted(train_ids))}
    loader = make_loader(args.corpus, train_ids, train_labels,
                         core_config["batch_size"], False, 2)
    core = load_model(core_config, core_checkpoint, len(video_index))
    control = load_model(control_config, control_checkpoint, len(video_index))
    payload = {
        "corpus": args.corpus,
        "code_version_description": "disjoint-within-training-video nearest-centroid nuisance probes and LOO normalized-position test risk",
        "probe_split": "even seconds train / odd seconds eval; max 32 each per video",
        "arms": {
            "local_control": probe_metrics(
                control, loader, video_index, core_config["position_bins"],
                core_config["device"]),
            "local_adversarial": probe_metrics(
                core, loader, video_index, core_config["position_bins"],
                core_config["device"]),
        },
        "position_risk": position_risk(args.corpus, run_root),
        "test_labels_used_for_training_or_checkpoint_selection": False,
    }
    output = run_root / args.corpus / "mechanism.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
