#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train import (MultiHateLoc, evaluator_test_ids, hdata, make_loader,
                   mdata, predict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    chosen = selection["selected"]
    config = chosen["args"]
    if config["arm"] != "sparse_scan" or config["run_test"] is not False:
        raise RuntimeError("selection is not a validation-only sparse_scan trial")
    checkpoint = selection_path.parent / chosen["trial"] / "checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing selected checkpoint: {checkpoint}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config["device"]
    model = MultiHateLoc(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        hidden=config["hidden"], embed=config["embed"],
        dropout=config["dropout"], k_proportion=config["k_proportion"],
        temperature=config["temperature"]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    corpus = config["corpus"]
    test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    dummy_labels = {video_id: 0 for video_id in test_ids}
    loader = make_loader(corpus, test_ids, dummy_labels,
                         config["batch_size"], False, 2)
    frames, test_video, _ = predict(model, loader, device)
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            record = {"video_id": video_id}
            record.update({name: [float(x) for x in values]
                           for name, values in frames[video_id].items()})
            handle.write(json.dumps(record) + "\n")
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "sparse_mixture_scan_mil",
        "arm": "sparse_scan", "evaluation_split": "test",
        "selected_validation_trial": chosen,
        "selected_checkpoint": str(checkpoint),
        "test_labels_used_for_training_or_checkpoint_selection": False,
        "test_video_diagnostics": test_video}, indent=2) + "\n")
    print(f"loaded {checkpoint}; wrote {len(test_ids)} test videos")


if __name__ == "__main__":
    main()
