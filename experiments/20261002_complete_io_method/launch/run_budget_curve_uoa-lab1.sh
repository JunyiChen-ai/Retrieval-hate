#!/usr/bin/env bash
# CPU only: budget curves for the checkpoints evaluated on this host (HateClipSeg; DeHate seed2025).
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/miniconda3/envs/HateVideo/bin/python"; R=runs/20261002_complete_io_method
echo "host $(hostname) | $(date -Iseconds)"
"$PY" experiments/20261002_complete_io_method/budget_curve.py --corpus hateclipseg --results-dir $R/hateclipseg/integrated/results
"$PY" experiments/20261002_complete_io_method/budget_curve.py --corpus dehate --results-dir $R/dehate/seed2025/results
echo "DONE $(date -Iseconds)"
