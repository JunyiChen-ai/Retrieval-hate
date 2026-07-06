#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p2_judge
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
# P2 MLLM neighbor-comparability judge. Text-only Qwen2.5-VL-7B, greedy.
# Submit ONE per language (2-GPU cap): DATASETS=MHC and DATASETS=MHC_zh.
#   smoke : LIMIT=8 DATASETS=MHC sbatch scripts/slurm/judge_neighbor_compat.sbatch
#   full  : DATASETS=MHC sbatch scripts/slurm/judge_neighbor_compat.sbatch
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-MHC}
LIMIT=${LIMIT:-0}
echo "[p2_judge] DATASETS=$DATASETS LIMIT=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/judge_neighbor_compat.py \
    --datasets "$DATASETS" \
    --limit "$LIMIT" \
    --device cuda \
    --cache_dir /data/jehc223/RGCL/scripts/analysis/p2_out

echo "[p2_judge] done DATASETS=$DATASETS"
