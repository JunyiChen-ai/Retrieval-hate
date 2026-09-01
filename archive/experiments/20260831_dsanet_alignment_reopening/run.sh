#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_ROOT="$ROOT/runs/20260831_dsanet_alignment_reopening/main"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/run.pid"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

"$PY" "$ROOT/experiments/20260831_dsanet_alignment_reopening/produce_scores.py" \
  --corpus hatemm
"$PY" "$ROOT/experiments/20260831_dsanet_alignment_reopening/produce_scores.py" \
  --corpus hateclipseg
"$PY" "$ROOT/experiments/20260831_dsanet_alignment_reopening/produce_controls.py"
"$PY" "$ROOT/experiments/20260831_dsanet_alignment_reopening/evaluate.py"
