#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Usage: sbatch scripts/slurm/gen_embed_mllm.sbatch [DATASET]
#   DATASET defaults to HateMM.
#   Extracts frozen Qwen2.5-VL-7B hidden-state embeddings (Config MLLM-A) for all
#   three splits, writes {split}_Qwen2.5-VL-7B-Instruct_HF.pt into
#   data/CLIP_Embedding/<DATASET>/, then pushes to B2 and prunes.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${1:-HateMM}
echo "[mllm_embed] DATASET=${DATASET} NUM_FRAMES=${NUM_FRAMES:-8}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/utils/generate_VideoMLLM_embedding_HF.py \
    --dataset "$DATASET" \
    --num_frames "${NUM_FRAMES:-8}" \
    --device cuda

# On success, push MLLM embeddings to B2 (copy, keep local for training).
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET"

echo "[mllm_embed] done for DATASET=${DATASET}"
