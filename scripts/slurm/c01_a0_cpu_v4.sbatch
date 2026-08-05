#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=c01_a0_v4
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/c01_a0_v4_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/c01_a0_v4_%j.err
# NOTE: intentionally NO --time and NO GPU request.

set -euo pipefail

cd /data/jehc223/RGCL
sha256sum --check --strict <<'SHA256'
3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514  scripts/analysis/c01_policy_contrast_a0_v4.py
2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6  configs/c01/c01_a0_v4.json
40b35eee2fb6fdbdb21fe9b4acfdcebf003c121c76492b898fbd2ea9b8c34dfb  scripts/analysis/c01_policy_contrast_a0_v3.py
4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5  configs/c01/c01_a0_v3.json
SHA256

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

python scripts/analysis/c01_policy_contrast_a0_v4.py \
  --config configs/c01/c01_a0_v4.json
