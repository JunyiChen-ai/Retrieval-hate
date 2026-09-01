#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_ROOT="$ROOT/runs/20260831_omnivtg_reopening_controls/main"
OMNI_PY=/home/jehc223/miniconda3/envs/OmniVTG/bin/python
EVAL_PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/run.pid"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

"$OMNI_PY" "$ROOT/experiments/20260831_omnivtg_reopening_controls/produce_corruption.py" \
  --corpus hatemm
"$OMNI_PY" "$ROOT/experiments/20260831_omnivtg_reopening_controls/produce_corruption.py" \
  --corpus hateclipseg
"$EVAL_PY" "$ROOT/experiments/20260831_omnivtg_reopening_controls/produce_controls.py"
"$EVAL_PY" "$ROOT/experiments/20260831_omnivtg_reopening_controls/evaluate.py"
