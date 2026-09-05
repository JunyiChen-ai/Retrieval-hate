#!/usr/bin/env bash
# usage: bash experiments/20260906_hier_evidence_clean/launch/run_search.sh <corpus> <seed>
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260906_hier_evidence_clean/search.py --corpus "$1" --seed "$2" --out-root runs/20260906_hier_evidence_clean --num-workers 4
