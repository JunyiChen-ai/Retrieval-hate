#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p3_segscore
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P3 (EXP_p3_evidence_pooling §0.1): MLLM evidence-density segment scoring with
# frozen Qwen2.5-VL-7B-Instruct. Writes ONLY data/MLLM_scores/<DS>/*.jsonl —
# no existing cache is touched. Resume-safe (skips already-done ids).
#
# Usage:
#   DATASET=MHC SPLITS=train sbatch scripts/slurm/score_segments_mllm.sbatch   # probe input
#   DATASET=MHC SPLITS=train,val,test sbatch scripts/slurm/score_segments_mllm.sbatch
#   LIMIT=10 DATASET=MHC SPLITS=train sbatch scripts/slurm/score_segments_mllm.sbatch  # smoke
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC}
SPLITS=${SPLITS:-train}
LIMIT=${LIMIT:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}

echo "[p3_segscore] DATASET=$DATASET SPLITS=$SPLITS LIMIT=$LIMIT MODEL=$MODEL"
python scripts/analysis/score_segments_mllm.py \
    --dataset "$DATASET" \
    --splits "$SPLITS" \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --device cuda

echo "[p3_segscore] done DATASET=$DATASET SPLITS=$SPLITS"
