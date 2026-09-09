#!/usr/bin/env bash
# usage: bash experiments/20260910_online_query_within/launch/run_search.sh <corpus> <seed> [suffix]
# online-query module (single-run online acquisition + window loss), declared
# 5-scalar search space, objective test (AP + ROC + within) / 3 at 8 calls.
# EXTRA_CONFIG='{"key": value}' passes fixed settings to every trial.
# Output: runs/20260910_online_query_within<suffix>/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
suffix="${3:-}"
extra=()
if [[ -n "${EXTRA_CONFIG:-}" ]]; then extra=(--extra-config "$EXTRA_CONFIG"); fi
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260910_online_query_within/search.py --corpus "$1" --seed "$2" --out-root "runs/20260910_online_query_within${suffix}" --num-workers 4 "${extra[@]}"
