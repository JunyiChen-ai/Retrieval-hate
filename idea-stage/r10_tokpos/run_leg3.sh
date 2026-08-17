#!/usr/bin/env bash
# R10 leg 3 -- HateMM confirmation, winning arm (CAT) vs A0, 15 seeds 500-514.
# Frozen: R10_TOKPOS_FREEZE.md 2.6 leg 3.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
RUNDIR=logging/runs/r10_leg3
SEEDS=$(seq -s, 500 514)
echo "=== R10 leg3 build arms $(date -Is) ==="
python idea-stage/r10_tokpos/build_arms.py --dataset HateMM --base "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
echo "=== R10 leg3 grid start $(date -Is) ==="
bash idea-stage/reaudit/run_grid.sh "$RUNDIR" \
  "A0:R10TP-A0,TXT:R10TP-TXT,CAT:R10TP-CAT,RAND:R10TP-RAND,SEG:R10TP-SEG" \
  HateMM "$SEEDS" R10TP3
echo "=== R10 leg3 analysing $(date -Is) ==="
python idea-stage/reaudit/analyze_grid.py --logdir "$RUNDIR/logs" --dataset HateMM \
  --arms A0,TXT,CAT,RAND,SEG --seeds "$SEEDS" \
  --contrasts "CAT-A0,CAT-RAND,TXT-A0,RAND-A0,SEG-A0" \
  --out idea-stage/r10_tokpos/leg3.json
echo "=== R10 LEG3 ALLDONE $(date -Is) ==="
