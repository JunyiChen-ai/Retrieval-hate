#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus=$1
seed=$2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260905_latent_evidence_sequence/search.py --corpus "$corpus" --seed "$seed" --out-root runs/20260905_latent_evidence_sequence --num-workers 4
