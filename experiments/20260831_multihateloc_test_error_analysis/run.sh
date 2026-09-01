#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
run_dir="$repo/runs/20260831_multihateloc_test_error_analysis/main"
mkdir -p "$run_dir"
echo "$$" > "$run_dir/run.pid"

cd "$repo"
exec /home/jehc223/miniconda3/envs/HateVideo/bin/python \
  experiments/20260831_multihateloc_test_error_analysis/analyze.py \
  > "$run_dir/run.log" 2>&1

