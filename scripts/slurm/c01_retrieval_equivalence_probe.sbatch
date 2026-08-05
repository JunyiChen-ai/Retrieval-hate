#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=c01_retrieval_equiv_probe
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/c01_retrieval_equiv_probe_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/c01_retrieval_equiv_probe_%j.err
# NOTE: intentionally NO --time and NO GPU request.

set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

python scripts/analysis/c01_retrieval_equivalence_probe.py \
  --config configs/c01/c01_retrieval_equivalence_probe_v1.json
