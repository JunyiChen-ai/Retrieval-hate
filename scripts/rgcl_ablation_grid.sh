#!/usr/bin/env bash
# RGCL component-attribution ablation grid.
# Frozen design: idea-stage/RGCL_ABLATION_FREEZE.md  (frozen 2026-08-09, before this ran)
#
# 3 loss rungs (contrast_mode none/random/retrieval) x 11 (encoder,dataset) cells x 3 seeds
# = 99 head-level runs, single submission. Both inference readouts (I1 classifier head,
# I2 kNN vote) are read out of the SAME run's log, so inference is not a run dimension.
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR=logging/runs/rgcl_ablation
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

GROUP_NAME=RGCL_ABLATION_20260809
WARMUP=5

CLIP=openai_clip-vit-large-patch14-336_HF
QWEN=Qwen2.5-VL-7B-Instruct_HF

# (encoder_tag, dataset, model_string) -- LoRA name is per-dataset, see FREEZE section 2.
CELLS=(
  "CLIP HateMM      $CLIP"
  "CLIP MHC         $CLIP"
  "CLIP MHC_zh      $CLIP"
  "CLIP ImpliHateVid $CLIP"
  "QWEN HateMM      $QWEN"
  "QWEN MHC         $QWEN"
  "QWEN MHC_zh      $QWEN"
  "QWEN ImpliHateVid $QWEN"
  "LORA HateMM      Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
  "LORA MHC         Qwen2.5-VL-7B-Instruct-LoRA_HF"
  "LORA MHC_zh      Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
)
# NOTE: LORA x ImpliHateVid is intentionally absent -- no LoRA feature cache exists
# locally for that dataset (reported as a MISSING cell, never back-filled).

# contrast_mode -> loss rung label
declare -A RUNG=( ["none"]="L1" ["random"]="L2" ["retrieval"]="L3" )

TOTAL=$(( ${#CELLS[@]} * 3 * 3 ))
DONE=0
START_TS=$(date +%s)

echo "GRID_START total=$TOTAL at $(date -Is)"

for CM in none random retrieval; do
  L="${RUNG[$CM]}"
  for cell in "${CELLS[@]}"; do
    read -r ENC DS MODEL <<< "$cell"
    for SEED in 0 1 2; do
      TAG="${ENC}_${DS}_${L}_s${SEED}"
      RUNLOG="$LOGDIR/${TAG}.trainlog"
      T0=$(date +%s)
      echo "PROGRESS_START enc=$ENC ds=$DS loss=$L inf=I1+I2 seed=$SEED ($((DONE+1))/$TOTAL)"
      python ./src/run_rac.py \
          --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
          --dataset "$DS" --model "$MODEL" \
          --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
          --fusion_mode "align" \
          --hard_negatives_loss True --no_hard_negatives 1 \
          --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
          --metric "cos" --loss "triplet" --batch_norm False \
          --hybrid_loss True --warmup "$WARMUP" \
          --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
          --lambda_seg 0 \
          --contrast_mode "$CM" \
          --exp_comment "_${ENC}" \
          --Faiss_GPU False --force False > "$RUNLOG" 2>&1
      RC=$?
      T1=$(date +%s)
      DONE=$((DONE+1))
      if [ $RC -ne 0 ]; then
        echo "PROGRESS enc=$ENC ds=$DS loss=$L inf=I1 seed=$SEED STATUS=FAIL rc=$RC"
        echo "PROGRESS enc=$ENC ds=$DS loss=$L inf=I2 seed=$SEED STATUS=FAIL rc=$RC"
      else
        echo "PROGRESS enc=$ENC ds=$DS loss=$L inf=I1 seed=$SEED STATUS=OK secs=$((T1-T0))"
        echo "PROGRESS enc=$ENC ds=$DS loss=$L inf=I2 seed=$SEED STATUS=OK secs=$((T1-T0))"
      fi
      echo "PROGRESS_TOTAL done=$DONE/$TOTAL elapsed=$((T1-START_TS))s"
    done
  done
done

echo "GRID_DONE $DONE/$TOTAL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
