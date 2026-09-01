#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
root="$repo/runs/20260831_owner_abstaining_its2clr/pilot_seed234"
setsid bash "$repo/experiments/20260831_owner_abstaining_its2clr/run_rejection_completion.sh" \
  >"$root/rejection_completion.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$root/run.pid"
echo "launched PID $pid; log $root/rejection_completion.log"

