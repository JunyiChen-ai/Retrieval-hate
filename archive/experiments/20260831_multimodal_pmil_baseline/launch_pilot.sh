#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
run_dir="$repo/runs/20260831_multimodal_pmil_baseline/pilot_seed234"
mkdir -p "$run_dir"
setsid bash "$repo/experiments/20260831_multimodal_pmil_baseline/run_pilot.sh" \
  >"$run_dir/run.log" 2>&1 < /dev/null &
echo $! >"$run_dir/run.pid"
echo "launched PID $(tr -d '[:space:]' < "$run_dir/run.pid"); log $run_dir/run.log"
