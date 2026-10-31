#!/usr/bin/env bash
# Full Optuna search for one corpus and seed. Submit on the target host:
#   ssh <host> 'cd ~/Retrieval-hate && sbatch -p local-<node> -J sacma_<corpus>_<seed> \
#     -o runs/20261031_scale_aware_cma/slurm_<corpus>_<seed>_%j.out \
#     experiments/20261031_scale_aware_cma/launch/sbatch_search.sh <corpus> <seed>'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpus $1 seed $2"
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  experiments/20261031_scale_aware_cma/launch/run_search.py --corpus "$1" --seed "$2" "${@:3}"
