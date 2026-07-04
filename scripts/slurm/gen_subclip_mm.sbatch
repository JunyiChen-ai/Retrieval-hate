#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mhc_submm
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Multimodal sub-clip cache = existing frame-CLIP sub-clip cache + per-window
# Whisper-ASR CLIP-text embeddings (EXP_mm_segment_keys phase 2).
# Writes ONLY NEW files *_subclipK<K>_mm_*.pt (never overwrites the visual cache).
# Requires the ASR jsonl from gen_segment_asr.sbatch.
#
# Usage: sbatch scripts/slurm/gen_subclip_mm.sbatch
#        DATASETS=MHC sbatch scripts/slurm/gen_subclip_mm.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-MHC,MHC_zh}
SPLITS=${SPLITS:-train,val,test}

IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DS in "${DS_ARR[@]}"; do
    echo "[submm] DATASET=$DS SPLITS=$SPLITS"
    python src/utils/generate_subclip_mm_embedding_HF.py \
        --dataset "$DS" \
        --splits "$SPLITS" \
        --device cuda
    bash scripts/b2_push.sh "data/CLIP_Embedding/$DS" "embeddings/$DS" || true
done

echo "[submm] done DATASETS=$DATASETS"
