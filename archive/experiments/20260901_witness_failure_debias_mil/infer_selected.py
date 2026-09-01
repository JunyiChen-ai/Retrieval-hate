#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines/multihateloc"
COMMON = ROOT / "scripts/reproduction_baselines"
sys.path[:0] = [str(BASE), str(COMMON), str(ROOT)]

import data as mdata  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from model import MultiHateLoc  # noqa: E402
from src.scoped_video_protocol import evaluator_test_ids  # noqa: E402

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "witness_failure_train", Path(__file__).with_name("train.py"))
if _TRAIN_SPEC is None or _TRAIN_SPEC.loader is None:
    raise ImportError("cannot load this experiment's train.py")
_TRAIN_MODULE = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN_MODULE)
loader = _TRAIN_MODULE.loader
predict = _TRAIN_MODULE.predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--role", required=True,
                        choices=("anchor", "uniform", "relative"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).resolve().read_text())
    row = selection["selected"][args.role]
    config = json.loads(Path(row["config"]).read_text())
    trial_args = config["args"]
    fixed = config["official_fixed_config"]
    corpus = trial_args["corpus"]
    model = MultiHateLoc(
        {name: mdata.FEATURE_DIMS[name] for name in mdata.MODALITIES},
        hidden=fixed["hidden"], embed=fixed["embed"],
        dropout=fixed["dropout"], k_proportion=fixed["k_proportion"],
        temperature=fixed["temperature"]).to(args.device)
    state = torch.load(row["checkpoint"], map_location=args.device,
                       weights_only=True)
    model.load_state_dict(state, strict=True)
    test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    all_labels = hdata.load_labels(corpus)
    test_labels = {video_id: all_labels[video_id] for video_id in test_ids}
    test_data = loader(corpus, test_ids, test_labels,
                       trial_args["batch_size"], False, 2)
    frames, _, _ = predict(model, test_data, args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            record = {"video_id": video_id}
            record.update({name: [float(value) for value in values]
                           for name, values in frames[video_id].items()})
            handle.write(json.dumps(record) + "\n")
    (output_dir / "config.json").write_text(json.dumps({
        "date": "2026-09-01", "method": "witness_failure_debias_mil",
        "code_version_description": config["code_version_description"],
        "evaluation_split": "test", "role": args.role,
        "selected_validation_trial": row,
        "test_predictions_do_not_condition_on_labels": True,
    }, indent=2) + "\n")
    print(f"wrote {len(test_ids)} {corpus} test predictions for {args.role}")


if __name__ == "__main__":
    main()
