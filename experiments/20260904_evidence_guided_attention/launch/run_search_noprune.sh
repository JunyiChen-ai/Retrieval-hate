#!/usr/bin/env bash
# usage: bash experiments/20260904_evidence_guided_attention/launch/run_search_noprune.sh <corpus> <seed>
# candidate 3 revision-2 model and its declared 6-scalar search space, unchanged;
# search without within pruning (rule 7 as amended 2026-09-06). Output:
# runs/20260904_evidence_guided_attention_rev2_noprune/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260904_evidence_guided_attention/search.py --corpus "$1" --seed "$2" --out-root runs/20260904_evidence_guided_attention_rev2_noprune --no-within-prune --num-workers 4
