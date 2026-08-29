#!/usr/bin/env bash
# R6-2 step 1 -- train the contrast-line head on each of the 4 datasets x 3 seeds,
# keeping every per-epoch checkpoint so the val-selected epoch can be reloaded and
# its per-item probabilities + fused embeddings dumped.
#
# Hyperparameters are byte-identical to idea-stage/text_merge/run_arms.sh.
# Usage: run_heads.sh [DATASET ...]   (default: all four)
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR=logging/runs/r6_trans
LOGDIR="$RUNDIR/logs"
GROUP=R6TRANS_20260817
mkdir -p "$LOGDIR"

declare -A ENC
ENC[HateMM]="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
ENC[MHC]="Qwen2.5-VL-7B-Instruct_HF"
ENC[MHC_zh]="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
ENC[ImpliHateVid]="openai_clip-vit-large-patch14-336_HF"

if [ $# -gt 0 ]; then DATASETS=("$@"); else DATASETS=(HateMM MHC MHC_zh ImpliHateVid); fi
read -r -a SEEDS <<< "${R6T_SEEDS:-0 1 2}"

TOTAL=$(( ${#DATASETS[@]} * ${#SEEDS[@]} ))
DONE=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP at $(date -Is)"

for DS in "${DATASETS[@]}"; do
  M="${ENC[$DS]}"
  for SEED in "${SEEDS[@]}"; do
    TAG="${DS}_s${SEED}"
    RUNLOG="$LOGDIR/${TAG}.trainlog"
    T0=$(date +%s)
    echo "PROGRESS_START dataset=$DS seed=$SEED ($((DONE+1))/$TOTAL) at $(date -Is)"
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
        --exp_comment "_R6T" \
        --Faiss_GPU False --force False --keep_epoch_ckpts True \
        > "$RUNLOG" 2>&1
    RC=$?
    T1=$(date +%s)
    DONE=$((DONE+1))
    if [ $RC -ne 0 ]; then
      echo "PROGRESS dataset=$DS seed=$SEED STATUS=FAIL rc=$RC secs=$((T1-T0))"
    else
      echo "PROGRESS dataset=$DS seed=$SEED STATUS=OK secs=$((T1-T0))"
    fi
    echo "PROGRESS_TOTAL done=$DONE/$TOTAL elapsed=$((T1-START_TS))s"
  done
done

echo "GRID_DONE $DONE/$TOTAL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
