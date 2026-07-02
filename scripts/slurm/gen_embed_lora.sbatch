#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lora_embed
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Usage: sbatch scripts/slurm/gen_embed_lora.sbatch <DATASET> <LORA_DIR> [OUT_MODEL_TAG]
#   e.g.  sbatch scripts/slurm/gen_embed_lora.sbatch MHC    logging/lora/MHC
#         sbatch scripts/slurm/gen_embed_lora.sbatch MHC_zh logging/lora/MHC_zh
#   DATASET defaults to MHC; LORA_DIR is REQUIRED (the peft adapter directory saved
#   by the Stage-1 LoRA-SFT, i.e. logging/lora/<DATASET>/).
#   Loads the frozen base Qwen2.5-VL-7B + merges the LoRA adapter, then extracts the
#   two-stream hidden-state embeddings (8-frame dual-stream pooling) for all three
#   splits, writing {split}_<OUT_MODEL_TAG>.pt into data/CLIP_Embedding/<DATASET>/,
#   then pushes to B2 and prunes. The LoRA cache is DISTINCT from the frozen cache
#   (default tag Qwen2.5-VL-7B-Instruct-LoRA_HF) and never clobbers it.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${1:-MHC}
LORA_DIR=${2:-}
OUT_MODEL_TAG=${3:-Qwen2.5-VL-7B-Instruct-LoRA_HF}

if [ -z "${LORA_DIR}" ]; then
    echo "[lora_embed] ERROR: LORA_DIR (arg 2) is required, e.g. logging/lora/${DATASET}" >&2
    exit 2
fi
if [ ! -d "${LORA_DIR}" ]; then
    echo "[lora_embed] ERROR: LORA_DIR '${LORA_DIR}' is not a directory (expected peft adapter dir)." >&2
    exit 2
fi

echo "[lora_embed] DATASET=${DATASET} LORA_DIR=${LORA_DIR} OUT_MODEL_TAG=${OUT_MODEL_TAG} NUM_FRAMES=${NUM_FRAMES:-8}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/utils/generate_VideoMLLM_embedding_lora_HF.py \
    --dataset "$DATASET" \
    --lora_dir "$LORA_DIR" \
    --out_model_tag "$OUT_MODEL_TAG" \
    --num_frames "${NUM_FRAMES:-8}" \
    --device cuda

# On success, push the LoRA embeddings to B2 (copy, keep local for training).
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET"

echo "[lora_embed] done for DATASET=${DATASET} (LORA_DIR=${LORA_DIR})"
