#!/usr/bin/env bash
# R12-ANCHOR -- fork of idea-stage/r11_union/run_union_grid.sh.
# ONLY changes: the ARMSPEC gains two optional fields (anchor teacher json,
# lambda_anchor) so the churn-anchored arms can be driven from the same loop.
# When the teacher field is "-" the two flags are not passed at all, so the run
# is byte-identical to the R10-COMBO command line.  Verify with:
#   diff idea-stage/r10_combo/run_combo_grid.sh idea-stage/r11_union/run_union_grid.sh
#
# Hyperparameters are otherwise byte-identical to idea-stage/r10_combo/run_combo_grid.sh.
# Per-run checkpoint dirs are deleted before AND after each run.
#
# Usage: run_union_grid.sh RUNDIR ARMSPEC_CSV DATASET SEEDS_CSV GROUP
#   ARMSPEC entry = TAG:MODEL:FUSION:TEACHER:LAMBDA:WEIGHTS
#     TAG      names the arm in the trainlog filename
#     MODEL    the --model string (cache suffix)
#     FUSION   fusion mode, "align"
#     TEACHER  path to the anchor teacher json, or "-" for none
#     LAMBDA   --lambda_anchor, ignored when TEACHER is "-"
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="$1"
IFS=',' read -r -a ARMSPECS <<< "$2"
IFS=',' read -r -a DSS  <<< "$3"
IFS=',' read -r -a SEEDS <<< "$4"
GROUP="$5"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#ARMSPECS[@]} * ${#DSS[@]} * ${#SEEDS[@]} ))
DONE=0
NFAIL=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP arms=${ARMSPECS[*]} ds=${DSS[*]} nseeds=${#SEEDS[@]} at $(date -Is)"

for DS in "${DSS[@]}"; do
  for SPEC in "${ARMSPECS[@]}"; do
    IFS=':' read -r ARM MODEL FUSION TEACHER LAMBDA WEIGHTS <<< "$SPEC"
    FUSION="${FUSION:-align}"
    TEACHER="${TEACHER:--}"
    LAMBDA="${LAMBDA:-0}"
    ANCHOR_ARGS=()
    if [ "$TEACHER" != "-" ]; then
      ANCHOR_ARGS=(--anchor_logits "$TEACHER" --lambda_anchor "$LAMBDA")
      if [ "${WEIGHTS:--}" != "-" ]; then ANCHOR_ARGS+=(--anchor_weights "$WEIGHTS"); fi
    fi
    for SEED in "${SEEDS[@]}"; do
      TAG="${DS}_${ARM}_s${SEED}"
      RUNLOG="$LOGDIR/${TAG}.trainlog"
      EXPROOT="logging/Retrieval/${DS}/${GROUP}"
      rm -rf "$EXPROOT"
      T0=$(date +%s)
      python ./src/run_rac.py \
          --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
          --dataset "$DS" --model "$MODEL" \
          --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
          --fusion_mode "$FUSION" \
          --hard_negatives_loss True --no_hard_negatives 1 \
          --final_eval False --seed "$SEED" --group_name "$GROUP" \
          --metric "cos" --loss "triplet" --batch_norm False \
          --hybrid_loss True --warmup 5 \
          --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
          --lambda_seg 0 \
          --contrast_mode none \
          --exp_comment "_R12AN_${ARM}" \
          --Faiss_GPU False --force False \
          --dump_head_scores "$LOGDIR/${TAG}.scores.jsonl" \
          "${ANCHOR_ARGS[@]}" \
          > "$RUNLOG" 2>&1
      RC=$?
      NEPOCH=$(grep -c "^test Epoch" "$RUNLOG")
      rm -rf "$EXPROOT"
      T1=$(date +%s)
      DONE=$((DONE+1))
      if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 30 ]; then
        NFAIL=$((NFAIL+1))
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC nepoch=$NEPOCH secs=$((T1-T0))"
      fi
      if [ $((DONE % 20)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
        AVAIL=$(df -m / | tail -1 | awk '{print $4}')
        echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL last=$TAG elapsed=$((T1-START_TS))s eta=$(( (T1-START_TS)*(TOTAL-DONE)/DONE ))s free_mb=$AVAIL"
      fi
    done
  done
done

echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
