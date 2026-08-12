#!/usr/bin/env bash
# DESC_CHANNEL step 2 -- 7 arms x 3 seeds = 21 head-level runs, single submission.
# Frozen design: idea-stage/DESC_CHANNEL_FREEZE.md section 5/6.
#
# Usage:  run_arms.sh <FEATSDIR> <LOGDIR> [GROUP]
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

FEATSDIR="${1:-idea-stage/desc_channel/feats}"
RUNDIR="${2:-logging/runs/desc_channel}"
GROUP="${3:-DESC_CHANNEL_20260813}"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

MODEL=Qwen2.5-VL-7B-Instruct-LoRA-curric_HF
ARMS=(A0 T B G Bmis Gmis N)

TOTAL=$(( ${#ARMS[@]} * 3 ))
DONE=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL feats=$FEATSDIR group=$GROUP at $(date -Is)"

for ARM in "${ARMS[@]}"; do
  for SEED in 0 1 2; do
    TAG="${ARM}_s${SEED}"
    RUNLOG="$LOGDIR/${TAG}.trainlog"
    if [ "$ARM" = "A0" ]; then
      ARCHIVE_ARGS=()
    else
      ARCHIVE_ARGS=(--archive_feats "${FEATSDIR}/{split}_${ARM}.pt" --archive_mode stream)
    fi
    T0=$(date +%s)
    echo "PROGRESS_START arm=$ARM seed=$SEED ($((DONE+1))/$TOTAL) at $(date -Is)"
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
        --exp_comment "_LORA_${ARM}" \
        --Faiss_GPU False --force False --keep_epoch_ckpts True \
        "${ARCHIVE_ARGS[@]}" > "$RUNLOG" 2>&1
    RC=$?
    T1=$(date +%s)
    DONE=$((DONE+1))
    if [ $RC -ne 0 ]; then
      echo "PROGRESS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC secs=$((T1-T0))"
    else
      echo "PROGRESS arm=$ARM seed=$SEED STATUS=OK secs=$((T1-T0))"
    fi
    echo "PROGRESS_TOTAL done=$DONE/$TOTAL elapsed=$((T1-START_TS))s"
  done
done

echo "GRID_DONE $DONE/$TOTAL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
