#!/bin/bash
# Reaper-resilient driver for the LSMI $0 gate (CPU-only, no SLURM, no GPU).
# The login node reaps long non-SLURM processes; lsmi_gate.py checkpoints per cell in
# refine-logs/.lsmi_ckpt/, so re-invoking simply resumes. Retries until exit 0 or MAX tries.
set +u
cd /data/jehc223/RGCL
source ~/.bashrc >/dev/null 2>&1
conda activate HateVideo
export OMP_NUM_THREADS=${OMP:-8}
STAGE=${1:-main}
MAX=${2:-60}
LOG=refine-logs/LSMI_GATE_run_${STAGE}.log
for i in $(seq 1 "$MAX"); do
  echo "===== attempt $i  stage=$STAGE  $(date '+%F %T') =====" >> "$LOG"
  python -u scripts/analysis/lsmi_gate.py --stage "$STAGE" >> "$LOG" 2>&1
  rc=$?
  echo "===== attempt $i exit=$rc =====" >> "$LOG"
  [ $rc -eq 0 ] && { echo "STAGE $STAGE COMPLETE" >> "$LOG"; exit 0; }
  sleep 20
done
echo "STAGE $STAGE EXHAUSTED RETRIES" >> "$LOG"
exit 1
