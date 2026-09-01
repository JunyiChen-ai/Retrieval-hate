#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
RUN_DIR="$ROOT/runs/20260901_benign_crossmodal_surprise_premise/main"
mkdir -p "$RUN_DIR"
echo $$ > "$RUN_DIR/run.pid"
"$PY" "$ROOT/experiments/20260901_benign_crossmodal_surprise_premise/probe.py" \
  --run-dir "$RUN_DIR"
"$PY" "$ROOT/experiments/20260901_benign_crossmodal_surprise_premise/evaluate.py" \
  --run-dir "$RUN_DIR"
