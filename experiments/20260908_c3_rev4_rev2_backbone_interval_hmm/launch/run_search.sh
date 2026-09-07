#!/usr/bin/env bash
# usage: bash experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/launch/run_search.sh <corpus> <seed>
# candidate 3 revision 4 (revision-2 backbone + interval evidence HMM),
# declared 5-scalar search space, no within pruning. Output:
# runs/20260908_c3_rev4_rev2_backbone_interval_hmm/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/search.py --corpus "$1" --seed "$2" --out-root runs/20260908_c3_rev4_rev2_backbone_interval_hmm --num-workers 4
