#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
run_dir="$repo/runs/20260831_target_conditioned_normal_proposal_mil/premise"
mkdir -p "$run_dir"
setsid bash "$repo/experiments/20260831_target_conditioned_normal_proposal_mil/run_probe.sh" \
  >"$run_dir/run.log" 2>&1 < /dev/null &
echo $! >"$run_dir/run.pid"
echo "launched PID $(cat "$run_dir/run.pid"); log $run_dir/run.log"
