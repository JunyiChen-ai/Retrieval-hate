#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_train_cons
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Retrieval-consensus segment denoising ablation (DESIGN_iter3 SS2 / E1):
#   SEG_MODE=consensus  -> retrieval-as-annotator pseudo-labels + EM
#   SEG_MODE=selfscore  -> MIST/C2FPL-style self-scored pseudo-labels + EM
#   SEG_MODE=full       -> inherited labels (Phase-3 reproduction)
# All runs use a FRESH group dir and FORCE=False: existing checkpoints can
# never be overwritten (the run aborts instead).
#
# Usage: DATASET=MHC SEG_MODE=consensus sbatch scripts/slurm/train_consensus.sbatch
set -euo pipefail

cd /data/jehc223/RGCL

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export WANDB_MODE=disabled HF_HUB_OFFLINE=1

export DATASET=${DATASET:-MHC}
export MODEL=${MODEL:-openai_clip-vit-large-patch14-336_HF}
export EPOCHS=${EPOCHS:-30}
export LAMBDA_SEG=${LAMBDA_SEG:-0.5}
export SEG_MODE=${SEG_MODE:-consensus}
export NUM_SUBCLIPS=${NUM_SUBCLIPS:-4}
export WARMUP=${WARMUP:-5}
export FAISS_GPU=${FAISS_GPU:-False}
export GROUP_NAME=${GROUP_NAME:-RAC_video_consensus}
export EM_ROUNDS=${EM_ROUNDS:-2}
export CONS_TOPK=${CONS_TOPK:-10}
export CONS_MARGIN=${CONS_MARGIN:-0.2}
export CONS_USE_DRIFT=${CONS_USE_DRIFT:-True}
export CONS_CONFLICT=${CONS_CONFLICT:-ignore}
export FORCE=${FORCE:-False}

echo "[train_cons] DATASET=$DATASET MODEL=$MODEL EPOCHS=$EPOCHS LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE K=$NUM_SUBCLIPS EM_ROUNDS=$EM_ROUNDS TOPK=$CONS_TOPK TAU=$CONS_MARGIN GROUP=$GROUP_NAME"

bash scripts/experiments_video.sh

bash scripts/b2_push.sh logging "logs/$DATASET" || true

echo "[train_cons] done for DATASET=$DATASET SEG_MODE=$SEG_MODE"
