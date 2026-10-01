#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  experiments/20261001_inside_outside_backbone/launch/run_diagnostics.py \
  --corpus hatemm --arm "${1:?full or nooutside required}" \
  --config experiments/20261001_inside_outside_backbone/configs/diagnostic_hatemm_seed234.json \
  --source-trial runs/20261001_inside_outside_backbone/hatemm/seed234/trial17
