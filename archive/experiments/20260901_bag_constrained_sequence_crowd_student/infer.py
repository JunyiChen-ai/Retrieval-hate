#!/usr/bin/env python
"""Teacher-free validation/test inference for the selected student."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE))
from hate_common import data as hdata  # noqa: E402
from src.multimodal_video_data import multimodal_loader  # noqa: E402
from model import SequenceCrowdStudent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint_dir)
    meta = json.loads((checkpoint / "train_meta.json").read_text())
    if meta["corpus"] != args.corpus:
        raise RuntimeError("checkpoint/corpus mismatch")
    ids = hdata.load_split(args.corpus, args.split)
    if args.split == "test":
        gt_ids = set(hdata.gt_arrays(args.corpus, "test"))
        ids = [video_id for video_id in ids if video_id in gt_ids]
        if set(ids) != gt_ids:
            raise RuntimeError("frozen test manifest/GT cohort mismatch")
    labels = hdata.load_labels(args.corpus)
    loader = multimodal_loader(args.corpus, ids, labels,
                               meta["args"]["batch_size"],
                               meta["args"]["workers"], False,
                               meta["args"]["seed"])
    model = SequenceCrowdStudent(meta["args"]["width"],
                                 meta["args"]["dropout"]).to(args.device)
    model.load_state_dict(torch.load(checkpoint / "model.pt",
                                     map_location=args.device))
    model.eval(); output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), output.open("w") as handle:
        for features, _, lengths, mask, video_ids in loader:
            features = {k: v.to(args.device) for k, v in features.items()}
            score = torch.sigmoid(model(features, mask.to(args.device))).cpu()
            for row, video_id in enumerate(video_ids):
                values = score[row, :int(lengths[row])].tolist()
                handle.write(json.dumps({"video_id": video_id,
                                         "score_method": values}) + "\n")


if __name__ == "__main__":
    main()
