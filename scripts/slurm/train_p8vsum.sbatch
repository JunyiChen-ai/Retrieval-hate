#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p8vsum_tr
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P8b (EXP_p8b): train the B_vision arm (vision-grounded summary text channel).
# Flags matched EXACTLY to p1-prior's P8 recipe so B_vision is apples-to-apples
# with B_text; A(floor) and C(p8trunc) are REUSED from p1-prior (not retrained).
# Trainlog named p8vsum_<ds>_Bvis_s<seed>.trainlog so p8_collect.py can fold it.
# ONLY submit if the p8b probe gate OPENED (B_vision >= A AND >= C on ZH train).
#
# Usage: DATASET=MHC_zh SEED=0 sbatch scripts/slurm/train_p8vsum.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC_zh}
SEED=${SEED:-0}
export WARMUP=${WARMUP:-5}
GROUP_NAME=${GROUP_NAME:-RAC_video_p8vsum}
RUNLOG="slurm/logs/p8vsum_${DATASET}_Bvis_s${SEED}.trainlog"

echo "[p8vsum] DATASET=$DATASET SEED=$SEED GROUP=$GROUP_NAME -> $RUNLOG"
python ./src/run_rac.py --batch_size 64 \
    --lr 0.0001 --epochs 30 --topk 20 --dataset "$DATASET" \
    --model p8vsum_HF \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode align \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
    --metric cos --loss triplet --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --exp_comment _p8Bvis \
    --majority_voting arithmetic --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

echo "[p8vsum] done DATASET=$DATASET SEED=$SEED -> $RUNLOG"
