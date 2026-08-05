#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=c04_a0t_small_v1_v6_preflight
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no time directive; CPU-only, no array/dependency/submission chain.
set -euo pipefail

cd /data/jehc223/RGCL
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

bash scripts/wrappers/c04_a0t_small_v1_v6_preflight.sh
