#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed_16f
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# FRAME16_PREREG stage-1 extraction. Frozen Qwen2.5-VL-7B hidden-state embeddings at
# 16 frames (vs the banked 8f cache). EXACTLY one changed variable vs the 8f extractor:
#   --num_frames 16   (default 8 in generate_VideoMLLM_embedding_HF.py)
# and a DISTINCT out-tag so the banked 8f cache is NEVER clobbered:
#   --out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f
# Writes {split}_Qwen2.5-VL-7B-Instruct_HF-16f.pt into data/CLIP_Embedding/<DATASET>/
# (banked 8f is {split}_Qwen2.5-VL-7B-Instruct_HF.pt — untouched), then pushes to B2.
# No code edit to the extractor: both --num_frames and --out_model_tag are pre-existing
# argparse args (generate_VideoMLLM_embedding_HF.py:84-95).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${1:-HateMM}
echo "[mllm_embed_16f] DATASET=${DATASET} NUM_FRAMES=16 OUT_TAG=Qwen2.5-VL-7B-Instruct_HF-16f"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/utils/generate_VideoMLLM_embedding_HF.py \
    --dataset "$DATASET" \
    --num_frames 16 \
    --out_model_tag Qwen2.5-VL-7B-Instruct_HF-16f \
    --device cuda

# On success, push MLLM-16f embeddings to B2 (copy, keep local for training).
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET"

echo "[mllm_embed_16f] done for DATASET=${DATASET}"
