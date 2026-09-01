#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate
RUN_DIR="$ROOT/runs/20260901_policy_cluster_transport/formal_seed234"
mkdir -p "$RUN_DIR"
setsid nohup bash "$ROOT/archive/experiments/20260901_policy_cluster_transport/run_formal.sh" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN_DIR/run.pid"
printf 'launched pid=%s log=%s\n' "$pid" "$RUN_DIR/run.log"
