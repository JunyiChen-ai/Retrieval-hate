#!/usr/bin/env bash
# usage: bash experiments/20260906_hier_evidence_clean/launch/run_search.sh <corpus> <seed> [v1|v2]
# v1 -> runs/20260906_hier_evidence_clean ; v2 -> runs/20260906_hier_evidence_clean_v2
set -euo pipefail
cd "$HOME/Retrieval-hate"
space="${3:-v1}"
out="runs/20260906_hier_evidence_clean"
[[ "$space" == v1 ]] || out="${out}_${space}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260906_hier_evidence_clean/search.py --corpus "$1" --seed "$2" --space "$space" --out-root "$out" --num-workers 4
