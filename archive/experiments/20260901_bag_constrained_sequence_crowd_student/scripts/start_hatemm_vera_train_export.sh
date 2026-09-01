#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$PROJECT_ROOT/runs/20260901_sequence_crowd_teacher_export/hatemm_vera_train"
mkdir -p "$RUN_DIR"
setsid "$PROJECT_ROOT/scripts/launch_hatemm_vera_train_export.sh" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_DIR/run.pid"
printf 'launched PID %s; log %s\n' "$!" "$RUN_DIR/run.log"
