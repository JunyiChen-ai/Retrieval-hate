#!/usr/bin/env bash
# REPRO campaign — one GPU, one big model at a time.
#
#   scripts/repro_campaign/gpu_queue.sh <owner-name> <command...>
#
# Blocks until the lock is free, runs the command, releases the lock even on
# failure or SIGTERM.  `mkdir` is the atomic primitive.  A lock whose owning PID
# is gone is reclaimed after a warning, so a killed job cannot wedge the queue.
set -u
LOCK=/home/jehc223/Retrieval-hate/logging/runs/GPU.lock
OWNER=${1:?owner name required}; shift

while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    echo "$OWNER $$" > "$LOCK/owner"
    break
  fi
  held=$(cat "$LOCK/owner" 2>/dev/null || echo "unknown 0")
  pid=${held##* }
  if [ -n "$pid" ] && [ "$pid" != "0" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "[gpu_queue] stale lock from '$held' (pid gone) -> reclaiming" >&2
    rm -rf "$LOCK"
    continue
  fi
  echo "[gpu_queue] $OWNER waiting, held by '$held' $(date -Is)" >&2
  sleep 60
done

cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT INT TERM

echo "[gpu_queue] $OWNER acquired $(date -Is)" >&2
"$@"
rc=$?
echo "[gpu_queue] $OWNER released rc=$rc $(date -Is)" >&2
exit $rc
