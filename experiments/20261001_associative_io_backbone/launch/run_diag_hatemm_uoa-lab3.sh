#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  experiments/20261001_associative_io_backbone/launch/run_diagnostics.py \
  --corpus hatemm --arm "${1:?full, nooutside or noattention required}" \
  --config experiments/20261001_associative_io_backbone/configs/diagnostic_hatemm_seed234.json \
  --source-trial runs/20261001_associative_io_backbone/hatemm/seed234/trial10
