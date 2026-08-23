#!/usr/bin/env bash
# TERA Gate-0 detached-run wrapper (appendix sec 0.3).
#
#   scripts/tera_gate0/run_detached.sh <task> <command...>
#
# writes logging/runs/<task>/run.log and logging/runs/<task>/run.pid, exactly as
# the registered execution discipline requires.  Progress:
#   tail -f logging/runs/<task>/run.log
#   ps -p $(cat logging/runs/<task>/run.pid)
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <task> <command...>" >&2
  exit 2
fi

TASK="$1"; shift
ROOT="${TERA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOGDIR="$ROOT/logging/runs/$TASK"
mkdir -p "$LOGDIR"

export TERA_REPO_ROOT="$ROOT"
export TERA_LOG_PATH="$LOGDIR/run.log"
export TERA_PID_FILE="$LOGDIR/run.pid"

cd "$ROOT"
nohup "$@" > "$LOGDIR/run.log" 2>&1 &
echo $! > "$LOGDIR/run.pid"
echo "[tera-gate0] task=$TASK pid=$(cat "$LOGDIR/run.pid") log=$LOGDIR/run.log"
