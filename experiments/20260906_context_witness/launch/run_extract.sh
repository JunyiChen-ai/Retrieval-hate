#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/analysis/extract_context_witness.py --corpus "$1" --run-dir "runs/20260906_context_witness/${2:-extract_$1}"
