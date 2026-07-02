#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_subclip
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Sub-clip (fine-granularity) CLIP embedding extraction for the
# multi-granularity annotation-free temporal retrieval method (Delta 1).
#
# Usage: sbatch scripts/slurm/gen_subclip.sbatch [DATASET]
#   DATASET defaults to MHC.
#   Env overrides: NUM_FRAMES (default 16), NUM_SUBCLIPS (default 4), BATCH (default 32).
set -euo pipefail

cd /data/jehc223/RGCL

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

DATASET=${1:-MHC}

echo "[gen_subclip] DATASET=${DATASET} NUM_FRAMES=${NUM_FRAMES:-16} NUM_SUBCLIPS=${NUM_SUBCLIPS:-4} BATCH=${BATCH:-32}"

python src/utils/generate_subclip_embedding_HF.py \
    --dataset "$DATASET" \
    --num_frames "${NUM_FRAMES:-16}" \
    --num_subclips "${NUM_SUBCLIPS:-4}" \
    --batch_size "${BATCH:-32}" \
    --device cuda

# On success, push sub-clip embeddings to B2 (they live alongside the whole-video cache).
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET" || true

echo "[gen_subclip] done for DATASET=${DATASET}"
