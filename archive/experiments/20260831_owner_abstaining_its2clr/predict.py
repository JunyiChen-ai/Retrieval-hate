"""Blind test inference; this module never imports localization ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from imports import base_data
from model import CarrierItS2CLR
from protocol import blind_test_split


def loader(corpus, ids, labels, batch_size, workers):
    return torch.utils.data.DataLoader(
        base_data.MultiModalDataset(corpus, ids, labels),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=base_data.collate, drop_last=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = CarrierItS2CLR(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ids, placeholders = blind_test_split(checkpoint["corpus"])
    test_loader = loader(
        checkpoint["corpus"], ids, placeholders, args.batch_size, args.workers
    )
    output_path = Path(args.scores_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = []
    with output_path.open("w") as handle, torch.no_grad():
        for feats, _, lengths, mask, video_ids in test_loader:
            feats = {name: value.to(device) for name, value in feats.items()}
            mask_device = mask.to(device)
            output = model(feats, mask_device)
            scores = model.frame_scores(output).cpu()
            for index, video_id in enumerate(video_ids):
                length = int(lengths[index])
                handle.write(json.dumps({
                    "video_id": video_id,
                    "score_core": scores[index, :length].tolist(),
                }) + "\n")
                seen.append(video_id)
    if seen != ids:
        raise RuntimeError("blind prediction cohort/order mismatch")
    print(json.dumps({"corpus": checkpoint["corpus"],
                      "n_videos": len(seen), "scores": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

