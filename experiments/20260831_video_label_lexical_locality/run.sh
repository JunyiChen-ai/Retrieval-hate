#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$ROOT/runs/20260831_video_label_lexical_locality/premise"
ENV_PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
mkdir -p "$RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/run.pid"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

"$ENV_PY" "$ROOT/experiments/20260831_video_label_lexical_locality/produce.py" \
  --run-dir "$RUN_DIR"
"$ENV_PY" "$ROOT/experiments/20260831_video_label_lexical_locality/evaluate.py" \
  --run-dir "$RUN_DIR"
