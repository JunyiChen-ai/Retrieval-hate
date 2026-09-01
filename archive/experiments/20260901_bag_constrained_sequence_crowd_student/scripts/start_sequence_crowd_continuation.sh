#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$PROJECT_ROOT/runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234"
mkdir -p "$RUN_DIR"
setsid "$PROJECT_ROOT/scripts/continue_sequence_crowd_after_hatemm_vera.sh" \
  > "$RUN_DIR/run_hatemm_continuation.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_DIR/run_hatemm_continuation.pid"
printf 'launched PID %s; log %s\n' "$!" "$RUN_DIR/run_hatemm_continuation.log"
