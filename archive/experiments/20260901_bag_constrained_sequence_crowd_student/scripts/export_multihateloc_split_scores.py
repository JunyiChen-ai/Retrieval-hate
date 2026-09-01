#!/usr/bin/env python
"""Export a frozen MultiHateLoc checkpoint on a readable split ID list."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/reproduction_baselines"
MLH = BASE / "multihateloc"
sys.path[:0] = [str(MLH), str(BASE)]
import data as mdata  # noqa: E402
from model import MultiHateLoc  # noqa: E402
from hate_common import data as hdata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint_dir)
    meta = json.loads((checkpoint / "train_log.json").read_text())
    cfg = meta["args"]
    if cfg["corpus"] != args.corpus:
        raise RuntimeError("checkpoint/corpus mismatch")
    ids = hdata.load_split(args.corpus, args.split)
    labels = hdata.load_labels(args.corpus)
    loader = DataLoader(mdata.MultiModalDataset(args.corpus, ids, labels),
                        batch_size=args.batch_size, shuffle=False,
                        collate_fn=mdata.collate, num_workers=4)
    model = MultiHateLoc({m: mdata.FEATURE_DIMS[m] for m in mdata.MODALITIES},
                         hidden=cfg["hidden"], embed=cfg["embed"],
                         dropout=cfg["dropout"],
                         k_proportion=cfg["k_proportion"],
                         temperature=cfg["temperature"]).to(args.device)
    model.load_state_dict(torch.load(checkpoint / "model.pt",
                                     map_location=args.device))
    model.eval(); target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), target.open("w") as handle:
        for features, _, lengths, mask, video_ids in loader:
            features = {k: v.to(args.device) for k, v in features.items()}
            out = model(features, mask.to(args.device))
            fused = out["probs"]["fused"].cpu()
            for row, video_id in enumerate(video_ids):
                score = fused[row, :int(lengths[row])].tolist()
                handle.write(json.dumps({"video_id": video_id,
                                         "score_multihateloc": score}) + "\n")
    provenance = {"corpus": args.corpus, "split": args.split,
                  "checkpoint": str(checkpoint.resolve()),
                  "model": "MultiHateLoc reimplementation",
                  "output": str(target.resolve()), "n_videos": len(ids),
                  "verification": "parsed checkpoint; full readable split IDs; 1fps lengths"}
    (target.parent / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
