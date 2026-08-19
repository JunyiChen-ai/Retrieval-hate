#!/usr/bin/env bash
# Release the GPU reservation once its owners are finished, so the parked
# UniTime and AV2A jobs resume without anyone remembering to do it by hand.
#
# The reservation names the LAVAD/URF owners; while it exists, gpu_queue.sh
# blocks everyone else. Nothing removed it automatically, which made "someone
# remembers" a single point of failure for ~20 h of queued work.
#
# Debounced on purpose: LAVAD legitimately has no waiter alive for short gaps
# between its stages, so the all-clear must hold for QUIET consecutive checks
# before the file is removed.
set -u
R=/home/jehc223/Retrieval-hate
RES=$R/logging/runs/GPU.reservation
LOG=$R/logging/runs/reservation_autorelease.log
QUIET=${QUIET:-10}      # consecutive clear checks (x 60 s) before releasing
clear_count=0
echo "[autorelease] armed $(date -Is), needs $QUIET consecutive clear minutes" >> "$LOG"
while [ -f "$RES" ]; do
  # any queue process whose owner is named in the reservation's allow-list?
  busy=0
  for p in $(pgrep -f "gpu_queue.sh" 2>/dev/null); do
    own=$(ps -o args= -p "$p" 2>/dev/null | sed 's/.*gpu_queue.sh //; s/ .*//')
    grep -qx "allow:$own" "$RES" 2>/dev/null && busy=1 && break
  done
  if [ "$busy" -eq 1 ]; then
    clear_count=0
  else
    clear_count=$((clear_count + 1))
    echo "[autorelease] no reserved owner alive ($clear_count/$QUIET) $(date -Is)" >> "$LOG"
    if [ "$clear_count" -ge "$QUIET" ]; then
      mv "$RES" "$RES.released.$(date +%H%M)"
      echo "[autorelease] RELEASED $(date -Is) - unitime/av2a may now take the card" >> "$LOG"
      break
    fi
  fi
  sleep 60
done
echo "[autorelease] exiting $(date -Is)" >> "$LOG"
