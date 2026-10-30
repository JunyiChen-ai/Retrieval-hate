#!/usr/bin/env bash
# CPU only: budget curves for the checkpoints evaluated on this host (HateMM; DeHate seed234/3407).
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="$HOME/miniconda3/envs/HateVideo/bin/python"; R=runs/20261002_complete_io_method
echo "host $(hostname) | $(date -Iseconds)"
"$PY" experiments/20261002_complete_io_method/budget_curve.py --corpus hatemm --results-dir $R/hatemm/integrated/results
"$PY" experiments/20261002_complete_io_method/budget_curve.py --corpus dehate --results-dir $R/dehate/seed234/results $R/dehate/seed3407/results
echo "DONE $(date -Iseconds)"
