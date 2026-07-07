#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p8b_vsum
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P8b (EXP_p8 extra arm): VISION-GROUNDED evidence summaries — Qwen2.5-VL reads
# 8 frames + Title+Transcript and writes the ≤60-word summary, transcribing
# on-screen text verbatim. Writes ONLY data/Summaries_vision/<DS>/*.jsonl
# (parallel to P8's data/Summaries/; never clobbers the text-only arm).
# Resume-safe. FIRE ONLY after the team-lead confirms P8's ZH probe gated closed.
#
# Usage:
#   DATASET=MHC_zh sbatch scripts/slurm/generate_vision_summary.sbatch        # primary
#   LIMIT=10 DATASET=MHC_zh SPLITS=train sbatch ...                           # smoke
#   DATASET=MHC sbatch ... ; DATASET=HateMM sbatch ...                        # optional secondary
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC_zh}
SPLITS=${SPLITS:-train,val,test}
LIMIT=${LIMIT:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
NUM_FRAMES=${NUM_FRAMES:-8}
# SUMMARY_LANG=en -> P8b (data/Summaries_vision); zh -> P8c (data/Summaries_vision_zh)
SUMMARY_LANG=${SUMMARY_LANG:-en}
if [ "$SUMMARY_LANG" = "zh" ]; then OUT_DIR=${OUT_DIR:-./data/Summaries_vision_zh}; else OUT_DIR=${OUT_DIR:-./data/Summaries_vision}; fi

echo "[p8b_vsum] DATASET=$DATASET SPLITS=$SPLITS LIMIT=$LIMIT MODEL=$MODEL FRAMES=$NUM_FRAMES LANG=$SUMMARY_LANG OUT=$OUT_DIR"
python scripts/analysis/generate_vision_summary.py \
    --dataset "$DATASET" \
    --splits "$SPLITS" \
    --model "$MODEL" \
    --num_frames "$NUM_FRAMES" \
    --summary_lang "$SUMMARY_LANG" \
    --out_dir "$OUT_DIR" \
    --limit "$LIMIT" \
    --device cuda

echo "[p8b_vsum] done DATASET=$DATASET SPLITS=$SPLITS"
