#!/usr/bin/env bash
# R6-1 step 2 -- 2 datasets x 4 arms x 3 seeds = 24 head-level runs, single submission.
# Frozen design: idea-stage/R6_PILOT_FREEZE_2026-08-17.md (pilot R6-1).
# Modelled on idea-stage/text_merge/run_arms.sh; identical hyperparameters.
#
# Usage:  run_arms.sh [LOGDIR] [GROUP] [TAGPREFIX] [ARMS_CSV] [DATASETS_CSV] [SEEDS_CSV]
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="${1:-logging/runs/r6_readout}"
GROUP="${2:-R6_READOUT_20260817}"
PREFIX="${3:-R6RO}"
IFS=',' read -r -a ARMS <<< "${4:-A0,L24,CAT,RANDCAT}"
IFS=',' read -r -a DSS  <<< "${5:-HateMM,MHC_zh}"
IFS=',' read -r -a SEEDS <<< "${6:-0,1,2}"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#ARMS[@]} * ${#DSS[@]} * ${#SEEDS[@]} ))
DONE=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP prefix=$PREFIX at $(date -Is)"

for DS in "${DSS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      TAG="${DS}_${ARM}_s${SEED}"
      RUNLOG="$LOGDIR/${TAG}.trainlog"
      T0=$(date +%s)
      echo "PROGRESS_START ds=$DS arm=$ARM seed=$SEED ($((DONE+1))/$TOTAL) at $(date -Is)"
      python ./src/run_rac.py \
          --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
          --dataset "$DS" --model "${PREFIX}-${ARM}" \
          --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
          --fusion_mode "align" \
          --hard_negatives_loss True --no_hard_negatives 1 \
          --final_eval False --seed "$SEED" --group_name "$GROUP" \
          --metric "cos" --loss "triplet" --batch_norm False \
          --hybrid_loss True --warmup 5 \
          --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
          --lambda_seg 0 \
          --contrast_mode none \
          --exp_comment "_R6RO_${ARM}" \
          --Faiss_GPU False --force False \
          > "$RUNLOG" 2>&1
      RC=$?
      T1=$(date +%s)
      DONE=$((DONE+1))
      if [ $RC -ne 0 ]; then
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC secs=$((T1-T0))"
      else
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=OK secs=$((T1-T0))"
      fi
      echo "PROGRESS_TOTAL done=$DONE/$TOTAL elapsed=$((T1-START_TS))s"
    done
  done
done

echo "GRID_DONE $DONE/$TOTAL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
