#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        if not config_path.is_file() or not log_path.is_file() or not checkpoint_path.is_file():
            raise SystemExit(f"incomplete validation trial: {trial_dir.name}")
        config = json.loads(config_path.read_text())
        log = json.loads(log_path.read_text())
        if config.get("evaluation_split") != "validation_only":
            raise SystemExit(f"non-validation artifact in search: {trial_dir.name}")
        row = {
            "trial": trial_dir.name,
            "arm": log["arm"], "alpha": log["alpha"], "lr": log["lr"],
            "selected_epoch": log["selected_epoch"],
            "selection_key": log["selected_validation_key"],
            "validation_metrics": log["selected_validation_metrics"],
            "reference_log": log["reference_log"],
            "config": str(config_path), "train_log": str(log_path),
            "checkpoint": str(checkpoint_path),
        }
        rows.append(row)
    counts = {arm: sum(row["arm"] == arm for row in rows)
              for arm in ("anchor", "aligned", "shifted")}
    if counts != {"anchor": 2, "aligned": 6, "shifted": 6}:
        raise SystemExit(f"unexpected arm counts: {counts}")
    selected = {}
    for arm in ("anchor", "aligned", "shifted"):
        eligible = [row for row in rows if row["arm"] == arm]
        eligible.sort(key=lambda row: (tuple(row["selection_key"]), row["trial"]),
                      reverse=True)
        selected[arm] = eligible[0]
    payload = {
        "selection_rule": (
            "per arm: validation within primary; pooled AP/ROC max drop .01 "
            "against same-lr alpha=0 anchor; infeasible configs minimize max "
            "violation then maximize within"),
        "n_trials": len(rows), "arm_counts": counts,
        "selected": selected, "trials": rows,
    }
    (root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selected"], indent=2))


if __name__ == "__main__":
    main()

