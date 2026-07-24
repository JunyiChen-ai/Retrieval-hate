#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=gen_embed_readout
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# READOUT-GRID re-extraction (prereg refine-logs/READOUT_PREREG.md).
# Runs the readout extractor for BOTH datasets sequentially in ONE job (8 CPU / 64 G /
# 1 GPU throughout -- never two concurrent CPU jobs; respects the submit-time aggregate
# cap). Each dataset merges its DEPLOYED adapter and writes the four grid caches
# (R0/R1/R2/R3) as {split}_<BASE_TAG>-ro_{L28,L24,ow_L28,ow_L24}.pt. The deployed cache
# {split}_<BASE_TAG>.pt is NEVER written (distinct -ro_* suffix -> no clobber).
#
# DATASET / LORA_DIR / BASE_TAG are HARDCODED per frame16 discipline so no submit-time
# typo can point at the wrong adapter or clobber the wrong base:
#   ZH     : MHC_zh  logging/lora/MHC_zh         Qwen2.5-VL-7B-Instruct-LoRA_HF
#   HateMM : HateMM  logging/lora/HateMM_curric  Qwen2.5-VL-7B-Instruct-LoRA-curric_HF
# ~1 GPU-h per dataset (4 forwards/item, 8 frames); ~2 GPU-h total.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

# config list: "DATASET LORA_DIR BASE_TAG"
CONFIGS=(
  "MHC_zh logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA_HF"
  "HateMM logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
)

run_one () {
  local DATASET="$1" LORA_DIR="$2" BASE_TAG="$3"
  if [ ! -d "${LORA_DIR}" ]; then
    echo "[readout] ERROR: LORA_DIR '${LORA_DIR}' is not a directory." >&2
    exit 2
  fi
  echo ""
  echo "########## READOUT EXTRACT: DATASET=$DATASET LORA_DIR=$LORA_DIR BASE_TAG=$BASE_TAG ##########"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python src/utils/generate_VideoMLLM_embedding_readout_HF.py \
      --dataset "$DATASET" \
      --lora_dir "$LORA_DIR" \
      --out_model_base_tag "$BASE_TAG" \
      --num_frames 8 \
      --device cuda

  # push derived .pt caches to B2 (videos NEVER leave the node); keep local for the screen.
  bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET" || true
}

for cfg in "${CONFIGS[@]}"; do
  read -r D LD BT <<< "$cfg"
  run_one "$D" "$LD" "$BT"
done

echo ""
echo "======== gen_embed_readout ALL DONE ($SLURM_JOB_ID) ========"
