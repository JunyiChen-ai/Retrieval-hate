"""Blind test feature inference. This module never imports localization GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import FactorialWitnessCRF  # noqa: E402
from protocol import blind_test_split  # noqa: E402
from src.multimodal_video_data import multimodal_loader  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = FactorialWitnessCRF(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ids, placeholders = blind_test_split(checkpoint["corpus"])
    loader = multimodal_loader(checkpoint["corpus"], ids, placeholders, args.batch_size, args.workers, False, 0)
    output_path = Path(args.scores_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = []
    with output_path.open("w") as handle, torch.no_grad():
        for feats, _, lengths, _, video_ids in loader:
            feats = {name: value.to(device) for name, value in feats.items()}
            output = model(feats, lengths)
            for row, video_id in enumerate(video_ids):
                handle.write(json.dumps({
                    "video_id": video_id,
                    "score_core": output["frame_scores"][row].cpu().tolist(),
                    "active_posterior": output["active_posteriors"][row].cpu().tolist(),
                    "bit_posterior": output["bit_posteriors"][row].cpu().tolist(),
                }) + "\n")
                seen.append(video_id)
    if seen != ids:
        raise RuntimeError("blind prediction cohort/order mismatch")
    print(json.dumps({"corpus": checkpoint["corpus"], "n_videos": len(seen), "scores": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
