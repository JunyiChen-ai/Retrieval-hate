#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
HERE="$ROOT/experiments/20260901_active_speaker_bound_utterance_mil"
RUN_ROOT="$ROOT/runs/20260901_active_speaker_bound_utterance_mil"
mkdir -p "$RUN_ROOT"
setsid nohup bash "$HERE/prepare_and_run_parallel.sh" \
  > "$RUN_ROOT/run_parallel.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RUN_ROOT/run.pid"
echo "launched parallel pid $pid"
