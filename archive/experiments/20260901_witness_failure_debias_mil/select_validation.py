#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_for_log(rows, path):
    target = Path(path).resolve()
    for row in rows:
        if Path(row["train_log"]).resolve() == target:
            return row
    raise ValueError(f"reference absent from search: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", required=True)
    args = parser.parse_args()
    root = Path(args.search_root).resolve()
    trial_dirs = sorted(path for path in root.glob("trial_*") if path.is_dir())
    if len(trial_dirs) != 14:
        raise SystemExit(f"expected exactly 14 validation trials, found {len(trial_dirs)}")
    rows = []
    for trial_dir in trial_dirs:
        config_path = trial_dir / "config.json"
        log_path = trial_dir / "train_log.json"
        checkpoint_path = trial_dir / "checkpoint.pt"
        if not all(path.is_file() for path in
                   (config_path, log_path, checkpoint_path)):
            raise SystemExit(f"incomplete validation trial: {trial_dir.name}")
        config = json.loads(config_path.read_text())
        log = json.loads(log_path.read_text())
        if config.get("evaluation_split") != "validation_only":
            raise SystemExit(f"non-validation artifact: {trial_dir.name}")
        rows.append({
            "trial": trial_dir.name, "arm": log["arm"],
            "lambda_failure": log["lambda_failure"], "lr": log["lr"],
            "selected_epoch": log["selected_epoch"],
            "selection_key": log["selected_validation_key"],
            "validation_metrics": log["selected_validation_metrics"],
            "reference_log": log["reference_log"],
            "config": str(config_path), "train_log": str(log_path),
            "checkpoint": str(checkpoint_path),
        })
    counts = {arm: sum(row["arm"] == arm for row in rows)
              for arm in ("anchor", "uniform", "relative")}
    if counts != {"anchor": 2, "uniform": 6, "relative": 6}:
        raise SystemExit(f"unexpected arm counts: {counts}")
    relative_rows = [row for row in rows if row["arm"] == "relative"]
    relative_rows.sort(
        key=lambda row: (tuple(row["selection_key"]), row["trial"]),
        reverse=True)
    relative = relative_rows[0]
    uniform = row_for_log(rows, relative["reference_log"])
    if uniform["arm"] != "uniform":
        raise ValueError("relative reference is not uniform control")
    anchor = row_for_log(rows, uniform["reference_log"])
    if anchor["arm"] != "anchor":
        raise ValueError("uniform reference is not exact anchor")
    if relative["lr"] != uniform["lr"] or relative["lr"] != anchor["lr"]:
        raise ValueError("selected matched chain has inconsistent learning rate")
    if relative["lambda_failure"] != uniform["lambda_failure"]:
        raise ValueError("selected matched chain has inconsistent failure strength")
    payload = {
        "selection_rule": (
            "select relative by validation key; report its same-lr same-lambda "
            "uniform control and same-lr lambda=0 anchor"),
        "n_trials": len(rows), "arm_counts": counts,
        "selected": {"anchor": anchor, "uniform": uniform,
                     "relative": relative},
        "trials": rows,
    }
    (root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selected"], indent=2))


if __name__ == "__main__":
    main()
