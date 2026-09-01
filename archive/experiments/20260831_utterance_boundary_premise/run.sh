#!/usr/bin/env bash
set -euo pipefail

cd /home/jehc223/Retrieval-hate
run_dir="runs/20260831_utterance_boundary_premise/main"
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/run.pid"
/home/jehc223/miniconda3/bin/conda run --no-capture-output -n HateVideo \
  python experiments/20260831_utterance_boundary_premise/analyze.py \
  > "$run_dir/run.log" 2>&1
