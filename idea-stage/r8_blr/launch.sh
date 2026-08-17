#!/usr/bin/env bash
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
RUNDIR=logging/runs/r8_blr
mkdir -p "$RUNDIR"
python idea-stage/r8_blr/blr.py > "$RUNDIR/run.log" 2>&1
echo "EXIT=$?" >> "$RUNDIR/run.log"
