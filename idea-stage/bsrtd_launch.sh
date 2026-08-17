#!/usr/bin/env bash
# B-SRTD stage launcher.
#
# Replaces the `mkdir && ... && setsid nohup ... & echo $! > run.pid` pattern that raced the
# pid write three times in this project (IDEA_REPORT.md sec 8.12).  Here the *child* writes
# its own pid file, so the pid is always the pid of the process that is actually running.
#
#   bash idea-stage/bsrtd_launch.sh <stage> [extra args passed through]
#
# stages: teacher | embed | gates | smoke-planted | smoke-nosignal | primary
# logs:   logging/runs/bsrtd_<stage>/run.log , run.pid
#
# follow:  tail -f logging/runs/bsrtd_<stage>/run.log
# alive?:  ps -p "$(cat logging/runs/bsrtd_<stage>/run.pid)"
# stop:    kill "$(cat logging/runs/bsrtd_<stage>/run.pid)"
set -euo pipefail

STAGE="${1:?usage: bsrtd_launch.sh <teacher|embed|gates|smoke-planted|smoke-nosignal|primary> [args]}"
shift || true

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNDIR="$REPO/logging/runs/bsrtd_${STAGE}"
mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"
PIDF="$RUNDIR/run.pid"

if [[ -f "$PIDF" ]] && ps -p "$(cat "$PIDF")" >/dev/null 2>&1; then
  echo "REFUSING: stage '$STAGE' already running as pid $(cat "$PIDF")" >&2
  exit 1
fi

PY="python"
case "$STAGE" in
  teacher)        CMD=("$PY" "$REPO/idea-stage/bsrtd_teacher_score.py") ;;
  embed)          CMD=("$PY" "$REPO/idea-stage/bsrtd_embed_cells.py") ;;
  gates)          CMD=("$PY" "$REPO/idea-stage/bsrtd_pilot.py" --mode gates) ;;
  smoke-planted)  CMD=("$PY" "$REPO/idea-stage/bsrtd_pilot.py" --mode smoke-planted \
                        --holdout val --skip-secondary \
                        --out "$REPO/idea-stage/bsrtd_smoke_planted.json") ;;
  smoke-nosignal) CMD=("$PY" "$REPO/idea-stage/bsrtd_pilot.py" --mode smoke-nosignal \
                        --holdout val --skip-secondary \
                        --out "$REPO/idea-stage/bsrtd_smoke_nosignal.json") ;;
  primary)        CMD=("$PY" "$REPO/idea-stage/bsrtd_pilot.py" --mode primary) ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

cd "$REPO"
setsid bash -c '
  echo $$ > "'"$PIDF"'"
  exec "$@"
' _ "${CMD[@]}" "$@" >>"$LOG" 2>&1 < /dev/null &

sleep 1
echo "stage=$STAGE pid=$(cat "$PIDF" 2>/dev/null || echo '?')"
echo "log=$LOG"
echo "tail -f $LOG"
