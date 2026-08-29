#!/usr/bin/env bash
set -uo pipefail
cd /home/jehc223/Retrieval-hate
RUNDIR=logging/runs/r10_leg2
SEEDS=$(seq -s, 500 529)
echo "=== R10 leg2 start $(date -Is) W=CAT (dev 0.8499 vs TXT 0.7925) ==="
bash idea-stage/reaudit/run_grid.sh "$RUNDIR" "C0:R10L2-C0,C1:R10L2-C1" MHC_zh "$SEEDS" R10L2
source ~/miniconda3/etc/profile.d/conda.sh; conda activate HateVideo
python idea-stage/reaudit/analyze_grid.py --logdir "$RUNDIR/logs" --dataset MHC_zh \
  --arms C0,C1 --seeds "$SEEDS" --contrasts "C1-C0" --out idea-stage/r10_tokpos/leg2.json
echo "=== R10 LEG2 ALLDONE $(date -Is) ==="
