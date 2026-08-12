#!/usr/bin/env bash
# TEXT_MERGE step 2 -- 4 arms x 3 seeds = 12 head-level runs, single submission.
# Frozen design: idea-stage/TEXT_MERGE_FREEZE.md sections 4/5.
#
# Usage:  run_arms.sh [LOGDIR] [GROUP] [TAGPREFIX]
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="${1:-logging/runs/text_merge}"
GROUP="${2:-TEXT_MERGE_20260813}"
PREFIX="${3:-TEXTMERGE}"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

ARMS=(A0 TMt TMall TMshuf)

TOTAL=$(( ${#ARMS[@]} * 3 ))
DONE=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP prefix=$PREFIX at $(date -Is)"

for ARM in "${ARMS[@]}"; do
  for SEED in 0 1 2; do
    TAG="${ARM}_s${SEED}"
    RUNLOG="$LOGDIR/${TAG}.trainlog"
    T0=$(date +%s)
    echo "PROGRESS_START arm=$ARM seed=$SEED ($((DONE+1))/$TOTAL) at $(date -Is)"
    python ./src/run_rac.py \
        --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
        --dataset HateMM --model "${PREFIX}-${ARM}" \
        --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
        --fusion_mode "align" \
        --hard_negatives_loss True --no_hard_negatives 1 \
        --final_eval False --seed "$SEED" --group_name "$GROUP" \
        --metric "cos" --loss "triplet" --batch_norm False \
        --hybrid_loss True --warmup 5 \
        --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
        --lambda_seg 0 \
        --contrast_mode none \
        --exp_comment "_TM_${ARM}" \
        --Faiss_GPU False --force False --keep_epoch_ckpts True \
        > "$RUNLOG" 2>&1
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
