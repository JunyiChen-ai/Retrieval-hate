#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
run_dir="$repo/runs/20260831_factorial_witness_crf/pilot_seed234"
mkdir -p "$run_dir"
setsid bash "$repo/experiments/20260831_factorial_witness_crf/run_pilot.sh" \
  >"$run_dir/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$run_dir/run.pid"
echo "launched PID $pid; log $run_dir/run.log"
