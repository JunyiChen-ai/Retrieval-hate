#!/usr/bin/env bash
# R10 Task B leg 1 -- MHC_zh, 5 arms x 30 seeds (500-529).
# Uses idea-stage/reaudit/run_grid.sh UNCHANGED (byte-identical hyperparameters to
# r6_confirm/run_confirm.sh -> r6_readout/run_arms.sh).  Frozen: R10_TOKPOS_FREEZE.md 2.4.
set -uo pipefail
cd /home/jehc223/Retrieval-hate

RUNDIR=logging/runs/r10_leg1
SEEDS=$(seq -s, 500 529)

echo "=== R10 leg1 start $(date -Is) seeds=$SEEDS ==="
bash idea-stage/reaudit/run_grid.sh "$RUNDIR" \
  "A0:R10TP-A0,TXT:R10TP-TXT,CAT:R10TP-CAT,RAND:R10TP-RAND,SEG:R10TP-SEG" \
  MHC_zh "$SEEDS" R10TP

echo "=== R10 leg1 grid done, analysing $(date -Is) ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python idea-stage/reaudit/analyze_grid.py \
  --logdir "$RUNDIR/logs" --dataset MHC_zh \
  --arms A0,TXT,CAT,RAND,SEG \
  --seeds "$SEEDS" \
  --contrasts "TXT-A0,CAT-A0,CAT-RAND,RAND-A0,SEG-A0" \
  --out idea-stage/r10_tokpos/leg1.json

echo "=== R10 LEG1 ALLDONE $(date -Is) ==="
