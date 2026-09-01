#!/usr/bin/env python
"""Select one complete method configuration strictly from validation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-trials", type=int, default=6)
    ap.add_argument("--pooled-tolerance", type=float, default=.01)
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.corpus_dir).glob("core_*/train_meta.json")):
        meta = json.loads(path.read_text())
        if meta["arm"] != "core" or meta["selection_split"] != "validation":
            raise RuntimeError(f"unexpected trial metadata: {path}")
        rows.append({
            "trial": path.parent.name,
            "path": str(path.parent.resolve()),
            "lr": float(meta["args"]["lr"]),
            "bag_weight": float(meta["args"]["bag_weight"]),
            "selected_epoch": int(meta["selected_epoch"]),
            "validation": meta["selected_validation"],
        })
    if len(rows) != args.expected_trials:
        raise RuntimeError(
            f"incomplete validation search: {len(rows)} != {args.expected_trials}")

    best_ap = max(row["validation"]["pooled_ap"] for row in rows)
    best_roc = max(row["validation"]["pooled_roc"] for row in rows)
    feasible = [
        row for row in rows
        if row["validation"]["pooled_ap"] >= best_ap - args.pooled_tolerance
        and row["validation"]["pooled_roc"] >= best_roc - args.pooled_tolerance
    ]
    selected = max(
        feasible,
        key=lambda row: (
            row["validation"]["within_roc"],
            row["validation"]["pooled_ap"],
            row["validation"]["pooled_roc"],
            -row["bag_weight"],
            -row["lr"],
        ),
    )
    payload = {
        "selection_split": "validation",
        "selection_rule": (
            "AP and ROC within tolerance of their best trial, then maximum "
            "within-video ROC; deterministic lower-complexity tie break"
        ),
        "pooled_tolerance": args.pooled_tolerance,
        "selected": selected,
        "trials": rows,
        "test_predictions_seen": False,
        "test_labels_used": False,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
