#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
run="$repo/runs/20260831_coalition_witness_candidate/pilot_seed234"
mkdir -p "$run"
setsid nohup bash "$repo/experiments/20260831_coalition_witness_candidate/supervise.sh" \
  > "$run/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run/run.pid"
echo "$pid"

