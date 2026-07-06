#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p8gen
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P8 (research-wiki/EXP_p8_semantic_compression.md): Qwen ≤60-word summaries of Title+
# Transcript for ALL videos (MHC/MHC_zh/HateMM, all splits) + single-chunk CLIP text caches
# (p8sum/p8trunc/p8concat). ~2.7k summaries + ~8k single-chunk encodes.
#
# Usage:  smoke : LIMIT=3 sbatch scripts/slurm/p8_generate_summaries.sbatch
#         full  : sbatch scripts/slurm/p8_generate_summaries.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-MHC,MHC_zh,HateMM}

echo "[p8gen] DATASETS=$DATASETS LIMIT=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/p8_generate_summaries.py \
    --datasets "$DATASETS" --limit "$LIMIT" --device cuda

echo "[p8gen] done"
