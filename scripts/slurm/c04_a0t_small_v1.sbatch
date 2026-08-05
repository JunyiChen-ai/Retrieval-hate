#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=c04_a0t_small_v1
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no --time. Exactly one GPU; HateMM then MHC_zh in one serial process.
# No array, dependency, chained submission, force-release, or automatic resubmission.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

RUN_ID=C04-A0T-SMALL-v1
CONFIG=${CONFIG:-configs/c04/c04_a0t_small_v1.json}
RUN_ID="$RUN_ID" CONFIG="$CONFIG" bash scripts/wrappers/c04_a0t_small_v1.sh
