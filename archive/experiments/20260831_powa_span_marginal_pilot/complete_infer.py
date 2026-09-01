#!/usr/bin/env python3
"""Atomically seal one authorized test inference directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    required = (
        "config.json", "test_scores.jsonl", "test_infer_meta.json",
        "metrics.json", "run.log", "run.pid", "source_snapshot.sha256",
        "test_claim.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise RuntimeError(f"missing inference artifacts: {missing}")
    meta = json.loads((run / "test_infer_meta.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics.get("scores_sha256") != sha256(run / "test_scores.jsonl"):
        raise RuntimeError("evaluator score hash mismatch")
    payload = {
        "corpus": meta["corpus"], "split": meta["split"],
        "authorization_sha256": meta["authorization_sha256"],
        **{f"{name}_sha256": sha256(run / name) for name in required},
    }
    temporary = run / "completion.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(run / "completion.json")


if __name__ == "__main__":
    main()
