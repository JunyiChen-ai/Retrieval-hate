#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$ROOT/runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234"
mkdir -p "$RUN_DIR"
setsid "$ROOT/experiments/20260901_bag_constrained_sequence_crowd_student/run_formal.sh" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_DIR/run.pid"
printf 'launched PID %s; log %s\n' "$!" "$RUN_DIR/run.log"
