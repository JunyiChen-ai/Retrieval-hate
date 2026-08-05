#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=c01_zero_probe
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/c01_zero_probe_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/c01_zero_probe_%j.err
# NOTE: intentionally NO --time and NO GPU request.

set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

python scripts/analysis/c01_zero_contract_probe.py
