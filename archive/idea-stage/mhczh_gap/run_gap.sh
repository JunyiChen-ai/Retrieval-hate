#!/usr/bin/env bash
# MHC-ZH gap diagnostic: same head training code, same seeds (30..89) as
# idea-stage/r6_confirm, ONLY the feature cache differs.
#   CURRIC = Qwen2.5-VL-7B-Instruct-LoRA-curric_HF  (cache the 0.7821 contrast line used)
#   PLAIN  = Qwen2.5-VL-7B-Instruct-LoRA_HF         (== ro_L28 == R6RO-A0, bit-exact)
# Hyperparameters byte-identical to idea-stage/r6_confirm/run_confirm.sh.
set -uo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

RUNDIR=logging/runs/mhczh_gap
LOGDIR="$RUNDIR/logs"; mkdir -p "$LOGDIR"
GROUP=MHCZH_GAP_20260817
DS=MHC_zh
declare -A MODELS=( ["CURRIC"]="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF" ["PLAIN"]="Qwen2.5-VL-7B-Instruct-LoRA_HF" )
IFS=',' read -r -a ARMS <<< "${1:-CURRIC,PLAIN}"
DEFAULT_SEEDS=$(seq -s, 30 89)
IFS=',' read -r -a SEEDS <<< "${2:-$DEFAULT_SEEDS}"

TOTAL=$(( ${#ARMS[@]} * ${#SEEDS[@]} )); DONE=0; NFAIL=0; START_TS=$(date +%s)
echo "GRID_START total=$TOTAL arms=${ARMS[*]} nseeds=${#SEEDS[@]} at $(date -Is)"
for ARM in "${ARMS[@]}"; do
  M="${MODELS[$ARM]}"
  for SEED in "${SEEDS[@]}"; do
    TAG="${DS}_${ARM}_s${SEED}"
    RUNLOG="$LOGDIR/${TAG}.trainlog"
    EXPROOT="logging/Retrieval/${DS}/${GROUP}"
    rm -rf "$EXPROOT"; T0=$(date +%s)
    python ./src/run_rac.py \
        --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
        --dataset "$DS" --model "$M" \
        --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
        --fusion_mode "align" \
        --hard_negatives_loss True --no_hard_negatives 1 \
        --final_eval False --seed "$SEED" --group_name "$GROUP" \
        --metric "cos" --loss "triplet" --batch_norm False \
        --hybrid_loss True --warmup 5 \
        --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
        --lambda_seg 0 \
        --contrast_mode none \
        --exp_comment "_GAP_${ARM}" \
        --Faiss_GPU False --force False \
        > "$RUNLOG" 2>&1
    RC=$?; NEPOCH=$(grep -c "^test Epoch" "$RUNLOG"); rm -rf "$EXPROOT"; T1=$(date +%s)
    DONE=$((DONE+1))
    if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 30 ]; then NFAIL=$((NFAIL+1))
      echo "PROGRESS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC nepoch=$NEPOCH secs=$((T1-T0))"; fi
    if [ $((DONE % 10)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
      echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL last=$TAG elapsed=$((T1-START_TS))s"; fi
  done
done
echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
