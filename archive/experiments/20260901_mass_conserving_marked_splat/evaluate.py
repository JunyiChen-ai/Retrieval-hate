#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
from eval_baseline_scores import main as canonical_evaluate  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
parser.add_argument("--corpus-dir", required=True)
args = parser.parse_args()
run = Path(args.corpus_dir).resolve()
canonical_evaluate([
    "--corpus", args.corpus,
    "--scores", str(run / "scores.jsonl"),
    "--split", "test",
    "--branch", "score_final",
    "--json-out", str(run / "metrics.json"),
    "--require-full-coverage",
])
