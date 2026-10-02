#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  archive/experiments/20261002_residual_io_backbone/launch/run_diagnostics.py \
  --corpus hatemm --arm "${1:?full, nooutside or noresidual required}" \
  --config runs/20261002_residual_io_backbone/hatemm/seed234/trial10/config.json \
  --source-trial runs/20261002_residual_io_backbone/hatemm/seed234/trial10
