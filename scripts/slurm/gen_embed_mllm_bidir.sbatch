#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed_bidir
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# BIDIR_STAGE1_PREREG stage-1 extraction. Bidirectional-attention LoRA-Qwen2.5-VL
# hidden-state embeddings (mask flipped via src/utils/bidir_patch.apply_bidir_mask),
# 8 frames, baseline (prefix-mean) readout, NOTHING else changed vs the causal LoRA
# arm. Two datasets in one job, EACH reusing its OWN banked adapter (no re-SFT), so
# the ONLY difference from the banked causal arm is the attention mask:
#   (1) MHC_zh   adapter logging/lora/MHC_zh        (== the 13150 generic-LoRA arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF
#   (2) HateMM   adapter logging/lora/HateMM_curric (== the 13241 curric arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF
# DISTINCT "-bidir" out-tags => the banked causal caches (…-LoRA_HF / …-LoRA-curric_HF)
# are NEVER clobbered. Writes {split}_<OUT_TAG>.pt into data/CLIP_Embedding/<DS>/,
# then pushes to B2. The extractor is generate_VideoMLLM_embedding_bidir_HF.py, a thin
# fork that imports the causal LoRA extractor's pooling operator VERBATIM.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# Rows: "DATASET  LORA_DIR  OUT_MODEL_TAG"  (adapters are the SAME banked ones the
# causal paired arms used; only the attention mask differs).
ROWS=(
  "MHC_zh  logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF"
  "HateMM  logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF"
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for row in "${ROWS[@]}"; do
  read -r DS LORA_DIR OUT_TAG <<< "$row"
  echo "[mllm_embed_bidir] DATASET=${DS} LORA_DIR=${LORA_DIR} OUT_TAG=${OUT_TAG} NUM_FRAMES=8 (BIDIR)"
  python src/utils/generate_VideoMLLM_embedding_bidir_HF.py \
      --dataset "$DS" \
      --lora_dir "$LORA_DIR" \
      --out_model_tag "$OUT_TAG" \
      --num_frames 8 \
      --device cuda
  bash scripts/b2_push.sh "data/CLIP_Embedding/$DS" "embeddings/$DS"
  echo "[mllm_embed_bidir] done DATASET=${DS} (OUT_TAG=${OUT_TAG})"
done

echo "[mllm_embed_bidir] ALL DONE"
