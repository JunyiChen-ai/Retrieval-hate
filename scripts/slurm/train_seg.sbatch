#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_train_seg
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Multi-granularity ablation trainer: whole-video baseline (LAMBDA_SEG=0) vs
# +sub-clip retrieval (LAMBDA_SEG=0.5), for a given DATASET/MODEL.
#
# Usage: LAMBDA_SEG=0.5 DATASET=MHC sbatch scripts/slurm/train_seg.sbatch
#   Env overrides:
#     DATASET   (default MHC)
#     MODEL     (default openai_clip-vit-large-patch14-336_HF)
#     EPOCHS    (default 30)
#     LAMBDA_SEG(default 0)     -> 0 baseline, 0.5 multi-granularity
#     SEG_MODE  (default full)  -> full | driftneg | milmax (segment-loss variant)
#     NUM_SUBCLIPS (default 4)
#     WARMUP    (default 5)
#     GROUP_NAME(default RAC_video_seg)
set -euo pipefail

cd /data/jehc223/RGCL

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export WANDB_MODE=disabled HF_HUB_OFFLINE=1

export DATASET=${DATASET:-MHC}
export MODEL=${MODEL:-openai_clip-vit-large-patch14-336_HF}
export EPOCHS=${EPOCHS:-30}
export LAMBDA_SEG=${LAMBDA_SEG:-0}
export SEG_MODE=${SEG_MODE:-full}
export NUM_SUBCLIPS=${NUM_SUBCLIPS:-4}
export WARMUP=${WARMUP:-5}
export FAISS_GPU=${FAISS_GPU:-False}
export GROUP_NAME=${GROUP_NAME:-RAC_video_seg}

echo "[train_seg] DATASET=$DATASET MODEL=$MODEL EPOCHS=$EPOCHS LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE K=$NUM_SUBCLIPS"

bash scripts/experiments_video.sh

bash scripts/b2_push.sh logging "logs/$DATASET" || true

echo "[train_seg] done for DATASET=$DATASET LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE"
