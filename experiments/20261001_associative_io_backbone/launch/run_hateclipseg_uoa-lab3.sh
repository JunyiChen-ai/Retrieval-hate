#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  experiments/20261001_associative_io_backbone/launch/run_search.py --corpus hateclipseg --seed "${1:-234}" "${@:2}"
