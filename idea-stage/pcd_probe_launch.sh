#!/usr/bin/env bash
# Launcher for the PCD paper-and-pencil feasibility probes (train/val only, no test file).
# Written as a script rather than an && chain: the R3-1/R4-1 launcher defect class
# (pid write racing a backgrounded chain) is recorded in IDEA_REPORT.md 8.12.
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate
RUNDIR="$ROOT/logging/runs/pcd_spec_probe"
mkdir -p "$RUNDIR"
cd "$ROOT/idea-stage"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate HateVideo
export HF_HUB_OFFLINE=1
{
  for s in pcd_space_probe pcd_space_probe2 pcd_space_probe3 pcd_space_probe4; do
    echo "########## $s  ($(date -Is))"
    python "$s.py" 2>&1 | grep -viE "warn|futurew" || true
  done
  echo "########## done ($(date -Is))"
} > "$RUNDIR/run.log" 2>&1 &
echo $! > "$RUNDIR/run.pid"
echo "launched pid $(cat "$RUNDIR/run.pid") -> $RUNDIR/run.log"
