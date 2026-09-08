#!/usr/bin/env bash
# usage: bash experiments/20260908_adaptive_vlm_query/launch/run_search.sh <corpus> <seed>
# adaptive VLM query module (evidence dropout + eoc acquisition),
# declared 5-scalar search space, no within pruning. Output:
# runs/20260908_adaptive_vlm_query/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# usage: run_search.sh <corpus> <seed> [suffix]; EXTRA_CONFIG='{"policy_start":"coarse"}' for iteration >= 1.
# Output root runs/20260908_adaptive_vlm_query<suffix> (suffix _it1, _it2, ...; iteration 0 = no suffix).
suffix="${3:-}"
extra=()
if [[ -n "${EXTRA_CONFIG:-}" ]]; then extra=(--extra-config "$EXTRA_CONFIG"); fi
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260908_adaptive_vlm_query/search.py --corpus "$1" --seed "$2" --out-root "runs/20260908_adaptive_vlm_query${suffix}" --num-workers 4 "${extra[@]}"
