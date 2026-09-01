#!/usr/bin/env python3
"""Write an atomic completion record for one span-marginal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import sha256


HERE = Path(__file__).resolve().parent


def maybe_hash(path):
    return sha256(path) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    meta_path = run / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    selected = meta.get("selected_epoch") is not None
    for name in ("config.json", "run.log", "run.pid", "source_snapshot.sha256"):
        if not (run / name).is_file():
            raise RuntimeError(f"missing completion input: {name}")
    if selected:
        for name in ("residual_head.pth", "val_scores.jsonl", "metrics.json"):
            if not (run / name).is_file():
                raise RuntimeError(f"missing selected artifact: {name}")
    payload = {
        "corpus": meta["corpus"], "arm": meta["arm"], "seed": meta["seed"],
        "selected_epoch": meta.get("selected_epoch"),
        "train_meta_sha256": sha256(meta_path),
        "config_sha256": sha256(run / "config.json"),
        "pilot_plan_sha256": sha256(HERE / "PILOT_PLAN.md"),
        "pre_run_review_sha256": sha256(HERE / "PRE_RUN_REVIEW.md"),
        "source_snapshot_manifest_sha256": sha256(
            run / "source_snapshot.sha256"
        ),
        "residual_head_sha256": maybe_hash(run / "residual_head.pth"),
        "val_scores_sha256": maybe_hash(run / "val_scores.jsonl"),
        "metrics_sha256": maybe_hash(run / "metrics.json"),
    }
    temporary = run / "completion.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(run / "completion.json")


if __name__ == "__main__":
    main()
