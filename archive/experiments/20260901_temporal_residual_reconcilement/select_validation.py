#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--print-args", action="store_true")
    args = parser.parse_args()
    root = Path(args.search_root).resolve()
    rows = []
    for log_path in sorted(root.glob("trial_*/train_log.json")):
        log = json.loads(log_path.read_text())
        config = json.loads((log_path.parent / "config.json").read_text())
        rows.append({"trial": log_path.parent.name,
                     "selected_epoch": log["selected_epoch"],
                     "validation_metric": log["selected_validation_metric"],
                     "args": config["args"]})
    if not rows:
        raise SystemExit("no completed validation trials")
    rows.sort(key=lambda row: (-row["validation_metric"], row["trial"]))
    payload = {"selection_metric": "within_roc", "n_trials": len(rows),
               "selected": rows[0], "ranking": rows}
    (root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    if args.print_args:
        chosen = rows[0]["args"]
        keys = ("lr", "max_epoch", "k_proportion", "lambda_smooth",
                "lambda_contrast", "lambda_residual", "hidden", "embed",
                "dropout", "temperature")
        for key in keys:
            print("--" + key.replace("_", "-"))
            print(chosen[key])
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
