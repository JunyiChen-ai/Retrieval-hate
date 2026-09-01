#!/usr/bin/env python3
"""Generate test predictions only after validation configuration selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.utils.data as tdata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "scripts/reproduction_baselines"
MM = BASE / "multihateloc"
sys.path[:0] = [str(HERE), str(ROOT), str(BASE), str(MM)]

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import MassConservingMarkedSplatMIL  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids  # noqa: E402


def loader(corpus, ids, batch_size):
    return tdata.DataLoader(
        mdata.MultiModalDataset(corpus, ids, {video_id: 0 for video_id in ids}),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mdata.collate,
        num_workers=2,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.corpus_dir).resolve()
    selection = json.loads((output / "selection.json").read_text())
    trial = Path(selection["selected"]["trial_dir"])
    config = json.loads((trial / "config.json").read_text())["args"]
    model = MassConservingMarkedSplatMIL(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        config["hidden"],
        config["embed"],
        config["dropout"],
        config["k_proportion"],
        config["temperature"],
    ).to(args.device)
    state = torch.load(trial / "checkpoint.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    test_ids = evaluator_test_ids(args.corpus, hdata.load_split(args.corpus, "test"))
    records = {}
    with torch.no_grad():
        for features, _, lengths, mask, video_ids in loader(args.corpus, test_ids, config["batch_size"]):
            features = {key: value.to(args.device) for key, value in features.items()}
            probability = model(features, mask.to(args.device))["prob"]
            for index, video_id in enumerate(video_ids):
                length = int(lengths[index])
                records[video_id] = probability[index, :length].cpu().tolist()
    with (output / "scores.jsonl").open("w") as handle:
        for video_id in test_ids:
            handle.write(json.dumps({"video_id": video_id, "score_final": records[video_id]}) + "\n")
    (output / "prediction_record.json").write_text(json.dumps({
        "split": "test",
        "selected_trial": str(trial),
        "selected_by_validation_before_test_prediction": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "n_videos": len(records),
    }, indent=2) + "\n")
    print(f"wrote {len(records)} test videos", flush=True)


if __name__ == "__main__":
    main()
