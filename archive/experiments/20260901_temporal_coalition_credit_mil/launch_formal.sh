#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
EXP="$ROOT/experiments/20260901_temporal_coalition_credit_mil"
RUN="$ROOT/runs/20260901_temporal_coalition_credit_mil/formal_val_selected_seed234"
mkdir -p "$RUN"
setsid bash "$EXP/run_formal.sh" > "$RUN/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RUN/run.pid"
echo "launched PID $pid; log $RUN/run.log"
