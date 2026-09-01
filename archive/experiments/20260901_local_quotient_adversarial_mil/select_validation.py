#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = ("lr", "batch_size", "max_epoch", "k_proportion", "lambda_smooth",
        "lambda_contrast", "lambda_video", "lambda_position", "local_scale",
        "position_bins", "hidden", "embed", "dropout", "temperature")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--print-args", action="store_true")
    args = parser.parse_args()
    root = Path(args.search_root).resolve()
    expected = ["trial_%02d" % index for index in range(1, 13)]
    actual = sorted(path.name for path in root.glob("trial_*") if path.is_dir())
    if actual != expected:
        raise SystemExit(f"expected exactly {expected}, found {actual}")
    rows, corpus = [], None
    for trial in expected:
        log_path = root / trial / "train_log.json"
        config_path = root / trial / "config.json"
        checkpoint = root / trial / "checkpoint.pt"
        if not log_path.is_file() or not config_path.is_file() or not checkpoint.is_file():
            raise SystemExit(f"incomplete validation trial: {trial}")
        log = json.loads(log_path.read_text())
        config = json.loads(config_path.read_text())
        trial_args = config["args"]
        if (trial_args.get("arm") != "local_adversarial" or
                trial_args.get("run_test") is not False or
                config.get("evaluation_split") != "validation_only"):
            raise SystemExit(f"non-validation core trial: {trial}")
        if corpus is None:
            corpus = trial_args.get("corpus")
        elif trial_args.get("corpus") != corpus:
            raise SystemExit("mixed corpora in validation search")
        value = log.get("selected_validation_metric")
        if value is None:
            raise SystemExit(f"missing validation selection metric: {trial}")
        rows.append({"trial": trial, "selected_epoch": log["selected_epoch"],
                     "validation_metric": value, "args": trial_args})
    rows.sort(key=lambda row: (-row["validation_metric"], row["trial"]))
    payload = {"selection_metric": "within_roc", "n_trials": len(rows),
               "selected": rows[0], "ranking": rows}
    (root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    if args.print_args:
        chosen = rows[0]["args"]
        for key in KEYS:
            print("--" + key.replace("_", "-"))
            print(chosen[key])
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

