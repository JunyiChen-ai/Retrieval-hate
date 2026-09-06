#!/usr/bin/env bash
# usage: bash experiments/20260907_c3_rev3_interval_evidence/launch/run_search.sh <corpus> <seed>
# candidate 3 revision 3 (evidence-routed attention + interval evidence HMM),
# declared 5-scalar search space, no within pruning. Output:
# runs/20260907_c3_rev3_interval_evidence/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260907_c3_rev3_interval_evidence/search.py --corpus "$1" --seed "$2" --out-root runs/20260907_c3_rev3_interval_evidence --num-workers 4
