#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$ROOT/runs/20260831_clip_policy_reopening/main"
mkdir -p "$RUN_DIR"
echo $$ > "$RUN_DIR/run.pid"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1
cd "$ROOT"
/home/jehc223/miniconda3/bin/conda run --no-capture-output -n HateVideo python experiments/20260831_clip_policy_reopening/produce.py
/home/jehc223/miniconda3/bin/conda run --no-capture-output -n HateVideo python experiments/20260831_clip_policy_reopening/evaluate.py
