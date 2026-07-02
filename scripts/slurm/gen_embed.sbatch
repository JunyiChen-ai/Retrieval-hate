#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_embed
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Usage: sbatch scripts/slurm/gen_embed.sbatch [DATASET]
#   DATASET defaults to MHC.
#   Env overrides: NUM_FRAMES (default 8), BATCH (default 32).
set -euo pipefail

cd /data/jehc223/RGCL

# Activate the HateVideo conda environment.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# Disk quota watchdog: reclaim space (B2-verified prune) before the run.
# Run as a subprocess so it can never abort this job; it is a no-op below the
# threshold. See scripts/disk_guard.sh and research-wiki/DISK_POLICY.md.
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

# CLIP model is pre-cached; stay offline. Disable wandb.
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

DATASET=${1:-MHC}

echo "[gen_embed] DATASET=${DATASET} NUM_FRAMES=${NUM_FRAMES:-8} BATCH=${BATCH:-32}"

python src/utils/generate_VideoCLIP_embedding_HF.py \
    --dataset "$DATASET" \
    --num_frames "${NUM_FRAMES:-8}" \
    --batch_size "${BATCH:-32}" \
    --device cuda

# On success, push embeddings to B2.
bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET"

echo "[gen_embed] done for DATASET=${DATASET}"
