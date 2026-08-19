#!/usr/bin/env bash
# REPRO campaign — one GPU, one big model at a time.
#
#   scripts/repro_campaign/gpu_queue.sh <owner-name> <command...>
#
# Blocks until the lock is free, runs the command, releases the lock even on
# failure or SIGTERM.  `mkdir` is the atomic primitive.  A lock whose owning PID
# is gone is reclaimed after a warning, so a killed job cannot wedge the queue.
set -u

# Re-exec once at startup so the whole script is parsed from the current file in
# one go.  bash otherwise reads a script lazily, by byte offset: editing this file
# while a waiter is parked in the loop below can make that waiter execute garbage,
# and it certainly makes it keep enforcing whatever policy it parsed at launch.
# (Measured 2026-08-19: the reservation guard was added at e2b63b4 18:30:38, and
# the one waiter launched at 18:29 walked straight past it and took a reserved
# card.)  Re-exec does not make a parked waiter see later edits -- nothing can --
# so policy lives in files this loop re-reads every iteration, never in the code.
if [ "${GPU_QUEUE_REEXEC:-}" != "1" ]; then
  export GPU_QUEUE_REEXEC=1
  exec bash "$0" "$@"
fi

LOCK=/home/jehc223/Retrieval-hate/logging/runs/GPU.lock
OWNER=${1:?owner name required}; shift

RESERVE=/home/jehc223/Retrieval-hate/logging/runs/GPU.reservation
# A reservation outranks the queue. While the file exists, only the owners it
# names may take the card; everyone else waits. Written by the coordinator's
# ruling of 2026-08-19, which gave the night to the LAVAD + URF run because it is
# the largest job and the one that cannot be split.
while [ -f "$RESERVE" ] && ! grep -qx "allow:$OWNER" "$RESERVE"; do
  echo "[gpu_queue] $OWNER blocked by reservation $(date -Is)" >&2
  sleep 120
done

while true; do
  if mkdir "$LOCK" 2>/dev/null; then
    # Re-check the reservation with the lock in hand: it may have been written
    # between our last poll and this mkdir, and taking a reserved card is worse
    # than waiting for one.
    if [ -f "$RESERVE" ] && ! grep -qx "allow:$OWNER" "$RESERVE"; then
      rm -rf "$LOCK"
      echo "[gpu_queue] $OWNER dropped the lock: reservation appeared $(date -Is)" >&2
      sleep 60
      continue
    fi
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
