#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
EXP_DIR="$ROOT/experiments/20260831_pair_alignment_attribution"
RUN_DIR="$ROOT/runs/20260831_pair_alignment_attribution/main"
PYTHON=/home/jehc223/miniconda3/envs/HateVideo/bin/python

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing formal run directory: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
echo "$$" > "$RUN_DIR/run.pid"
{
  echo "Working-tree source snapshot dated 2026-08-31."
  echo "Canonical analysis: $EXP_DIR/analyze.py"
  echo "Shared evaluator: $ROOT/scripts/reproduction_baselines/eval_baseline_scores.py"
} > "$RUN_DIR/code_version.txt"

exec > "$RUN_DIR/run.log" 2>&1
exec "$PYTHON" "$EXP_DIR/analyze.py" --out "$RUN_DIR/metrics.json"
