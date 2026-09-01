#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
from eval_baseline_scores import main as evaluate_main  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    evaluate_main(["--corpus", args.corpus,
                   "--scores", str(run_dir / "scores.jsonl"),
                   "--split", "test", "--branch", "score_fused",
                   "--json-out", str(run_dir / "metrics.json"),
                   "--require-full-coverage"])


if __name__ == "__main__":
    main()
