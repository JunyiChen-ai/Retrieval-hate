#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u archive/experiments/20260906_context_witness/search.py --corpus "$1" --seed "$2" --out-root runs/20260906_context_witness --num-workers 4
