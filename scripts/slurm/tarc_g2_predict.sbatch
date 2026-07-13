#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tarc_g2pred
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. NO b2_push / rclone tail.
#
# TARC G2 (exp-tarc-t0.md §6-G2): Qwen2.5-VL-7B target-community prediction on
# HateMM. Writes ONLY data/gt/HateMM/target_pred_<tag>.{jsonl,json}. GT target
# is NEVER read here (gold-annotation isolation, §5); scoring is a separate step.
# Resume-safe (skips already-done ids).
#
# Usage:
#   LIMIT=20 sbatch scripts/slurm/tarc_g2_predict.sbatch     # smoke (20/split)
#   sbatch scripts/slurm/tarc_g2_predict.sbatch              # full 1066 videos
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-HateMM}
SPLITS=${SPLITS:-train,val,test}
LIMIT=${LIMIT:-0}
TAG=${TAG:-qwen7b}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}

echo "[tarc_g2pred] DATASET=$DATASET SPLITS=$SPLITS LIMIT=$LIMIT TAG=$TAG MODEL=$MODEL"
python scripts/analysis/predict_target_qwen.py \
    --dataset "$DATASET" \
    --splits "$SPLITS" \
    --tag "$TAG" \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --device cuda

echo "[tarc_g2pred] done SPLITS=$SPLITS LIMIT=$LIMIT"
