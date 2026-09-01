#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_DIR=${1:-$ROOT/runs/20260831_local_paralinguistic_alignment/premise}
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/run.pid"
printf '%s\n' "2026-08-31 working tree; readable experiment path recorded in config" > "$RUN_DIR/code_version.txt"
"$PY" "$ROOT/experiments/20260831_local_paralinguistic_alignment/produce.py" --run-dir "$RUN_DIR"
"$PY" "$ROOT/experiments/20260831_local_paralinguistic_alignment/evaluate.py" --run-dir "$RUN_DIR"
