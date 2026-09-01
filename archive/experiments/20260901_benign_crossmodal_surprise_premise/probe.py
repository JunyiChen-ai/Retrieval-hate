#!/usr/bin/env python
"""Train negative-only cross-modal projections and emit label-blind test scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts" / "reproduction_baselines"
MM = BASE / "multihateloc"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MM))
import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from src.scoped_video_protocol import (evaluator_test_ids,
                                       scoped_video_labels)  # noqa: E402

CORPORA = ("hatemm", "hateclipseg")
MODALITIES = mdata.MODALITIES


class Projectors(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.layers = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(mdata.FEATURE_DIMS[name]),
                                nn.Linear(mdata.FEATURE_DIMS[name], width))
            for name in MODALITIES})

    def forward(self, batch):
        return {name: F.normalize(self.layers[name](batch[name]), dim=-1)
                for name in MODALITIES}


def load_negative_frames(corpus, rng, max_per_video=32):
    ids = hdata.load_split(corpus, "train")
    labels = scoped_video_labels(corpus, "train", ids)
    chunks = {name: [] for name in MODALITIES}
    for video_id in ids:
        if labels[video_id] != 0:
            continue
        arrays = {name: np.load(mdata.feature_path(name, corpus, video_id))
                  for name in MODALITIES}
        length = len(arrays["visual"])
        take = rng.choice(length, size=min(length, max_per_video), replace=False)
        for name in MODALITIES:
            chunks[name].append(arrays[name][take].astype(np.float32))
    return {name: torch.from_numpy(np.concatenate(chunks[name], axis=0))
            for name in MODALITIES}


def train_projectors(corpus, device, seed):
    rng = np.random.default_rng(seed)
    frames = load_negative_frames(corpus, rng)
    model = Projectors().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed)
    count = len(frames["visual"])
    batch_size = min(512, count)
    for step in range(300):
        index = torch.randint(count, (batch_size,), generator=generator)
        batch = {name: frames[name][index].to(device) for name in MODALITIES}
        projected = model(batch)
        target = torch.arange(batch_size, device=device)
        loss = 0.0
        pairs = 0
        for left in range(len(MODALITIES)):
            for right in range(left + 1, len(MODALITIES)):
                logits = (projected[MODALITIES[left]] @
                          projected[MODALITIES[right]].t()) / .1
                loss = loss + .5 * (F.cross_entropy(logits, target) +
                                     F.cross_entropy(logits.t(), target))
                pairs += 1
        loss = loss / pairs
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), count, float(loss.detach())


def disagreement(projected):
    values = []
    for left in range(len(MODALITIES)):
        for right in range(left + 1, len(MODALITIES)):
            values.append(1.0 - (projected[MODALITIES[left]] *
                                 projected[MODALITIES[right]]).sum(-1))
    return torch.stack(values).mean(0)


@torch.no_grad()
def score_video(model, corpus, video_id, device):
    batch = {name: torch.from_numpy(np.load(
        mdata.feature_path(name, corpus, video_id)).astype(np.float32)).to(device)
        for name in MODALITIES}
    projected = model(batch)
    aligned = disagreement(projected)
    length = len(aligned)
    shifted = dict(projected)
    if length > 1:
        shifted["audio"] = torch.roll(shifted["audio"], max(1, length // 3), 0)
        shifted["text"] = torch.roll(shifted["text"], max(1, 2 * length // 3), 0)
    control = disagreement(shifted)
    return aligned.cpu().numpy(), control.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=234)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {"test_gt_read_by_producer": False, "corpora": {}}
    for corpus in CORPORA:
        model, n_frames, final_loss = train_projectors(
            corpus, args.device, args.seed)
        test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
        out_dir = run_dir / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "scores.jsonl").open("w") as handle:
            for video_id in test_ids:
                aligned, shifted = score_video(model, corpus, video_id,
                                               args.device)
                handle.write(json.dumps({
                    "video_id": video_id,
                    "score_aligned": aligned.astype(float).tolist(),
                    "score_shifted": shifted.astype(float).tolist(),
                }) + "\n")
        report["corpora"][corpus] = {
            "negative_train_frames": n_frames,
            "final_train_loss": final_loss,
            "test_videos": len(test_ids),
        }
    (run_dir / "producer_report.json").write_text(
        json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
