#!/usr/bin/env bash
# R6-AUDIT -- measurement-protocol certification.
# 2 datasets x 3 arms x 30 seeds = 180 head-level runs, single background job.
# Hyperparameters are byte-identical to idea-stage/r6_readout/run_arms.sh except:
#   --group_name / --exp_comment (so the r6_readout artefacts are not clobbered)
# Per-run output dirs (ckpts) are deleted immediately after the run, once the
# trainlog -- which carries every metric this audit reads -- has been written.
#
# Usage: run_audit.sh [RUNDIR] [ARMS_CSV] [DATASETS_CSV] [SEEDS_CSV] [GROUP]
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="${1:-logging/runs/r6_audit}"
IFS=',' read -r -a ARMS <<< "${2:-A0,CAT,RANDCAT}"
IFS=',' read -r -a DSS  <<< "${3:-HateMM,MHC_zh}"
IFS=',' read -r -a SEEDS <<< "${4:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}"
GROUP="${5:-R6_AUDIT_20260817}"
PREFIX="R6RO"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#ARMS[@]} * ${#DSS[@]} * ${#SEEDS[@]} ))
DONE=0
NFAIL=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP prefix=$PREFIX at $(date -Is)"

for DS in "${DSS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      TAG="${DS}_${ARM}_s${SEED}"
      RUNLOG="$LOGDIR/${TAG}.trainlog"
      EXPROOT="logging/Retrieval/${DS}/${GROUP}"
      rm -rf "$EXPROOT"
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
          --exp_comment "_R6AUD_${ARM}" \
          --Faiss_GPU False --force False \
          > "$RUNLOG" 2>&1
      RC=$?
      NEPOCH=$(grep -c "^test Epoch" "$RUNLOG")
      rm -rf "$EXPROOT"
      T1=$(date +%s)
      DONE=$((DONE+1))
      if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 30 ]; then
        NFAIL=$((NFAIL+1))
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC nepoch=$NEPOCH secs=$((T1-T0))"
      else
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=OK nepoch=$NEPOCH secs=$((T1-T0))"
      fi
      echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL elapsed=$((T1-START_TS))s"
    done
  done
done

echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
