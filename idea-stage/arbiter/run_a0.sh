#!/usr/bin/env bash
# ARBITER step 1 -- reproduce the LORA/HateMM/L1/I1 baseline head, 3 seeds, keeping every
# per-epoch checkpoint so the val-selected epoch's per-video probabilities can be dumped.
# Identical command line to idea-stage/desc_channel/run_arms.sh arm A0.
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR=logging/runs/arbiter
LOGDIR="$RUNDIR/logs"
GROUP=ARBITER_20260813
MODEL=Qwen2.5-VL-7B-Instruct-LoRA-curric_HF
mkdir -p "$LOGDIR"

START_TS=$(date +%s)
echo "GRID_START total=3 group=$GROUP at $(date -Is)"
for SEED in 0 1 2; do
  T0=$(date +%s)
  echo "PROGRESS_START arm=A0 seed=$SEED at $(date -Is)"
  python ./src/run_rac.py \
      --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
      --dataset HateMM --model "$MODEL" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode "align" \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "$GROUP" \
      --metric "cos" --loss "triplet" --batch_norm False \
      --hybrid_loss True --warmup 5 \
      --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
      --lambda_seg 0 \
      --contrast_mode none \
      --exp_comment "_LORA_A0" \
      --Faiss_GPU False --force False --keep_epoch_ckpts True \
      > "$LOGDIR/A0_s${SEED}.trainlog" 2>&1
  RC=$?
  T1=$(date +%s)
  echo "PROGRESS arm=A0 seed=$SEED rc=$RC secs=$((T1-T0))"
done
echo "GRID_DONE at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
