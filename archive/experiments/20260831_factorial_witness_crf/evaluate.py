"""Invoke the repository's single shared evaluator on frozen test scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVALUATOR_DIR = REPO / "scripts/reproduction_baselines"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from eval_baseline_scores import main as evaluator_main  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args(argv)
    return evaluator_main([
        "--corpus", args.corpus,
        "--split", "test",
        "--scores", args.scores,
        "--branch", "score_core",
        "--json-out", args.metrics_out,
        "--require-full-coverage",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
