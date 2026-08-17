#!/usr/bin/env bash
# R7-1 -- annotator-vote soft-label training.
# 2 datasets x 5 arms x 30 seeds (100..129) = 300 head runs, single background job.
# Frozen design: idea-stage/R7_SOFTVOTE_FREEZE.md.
# Hyperparameters identical to idea-stage/r6_confirm/run_confirm.sh; the ONLY
# difference between arms is the BCE training target.
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="${1:-logging/runs/r7_softvote}"
GROUP="R7_SOFTVOTE_20260817"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

ARMS=(A0 SOFT10 SOFT05 LS10 LS05)
DSS=(MHC_zh MHC)
SEEDS=$(seq 100 129)

# per-dataset feature cache
cache_for() { case "$1" in MHC_zh) echo "R6RO-A0";; MHC) echo "Qwen2.5-VL-7B-Instruct-LoRA_HF";; esac; }
# entropy-matched label-smoothing epsilons (idea-stage/r7_softvote/build_meta.json)
eps_for() {
  case "$1/$2" in
    MHC_zh/LS10) echo 0.02445;; MHC_zh/LS05) echo 0.05823;;
    MHC/LS10)    echo 0.01143;; MHC/LS05)    echo 0.04810;;
    *) echo 0;;
  esac
}

TOTAL=$(( ${#ARMS[@]} * ${#DSS[@]} * $(echo "$SEEDS" | wc -w) ))
DONE=0; NFAIL=0; START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP arms=${ARMS[*]} ds=${DSS[*]} at $(date -Is)"

for DS in "${DSS[@]}"; do
  MODEL=$(cache_for "$DS")
  for ARM in "${ARMS[@]}"; do
    EXTRA=()
    case "$ARM" in
      SOFT10) EXTRA=(--soft_target_json "idea-stage/r7_softvote/targets_${DS}_SOFT10.json");;
      SOFT05) EXTRA=(--soft_target_json "idea-stage/r7_softvote/targets_${DS}_SOFT05.json");;
      LS10|LS05) EXTRA=(--label_smoothing "$(eps_for "$DS" "$ARM")");;
    esac
    for SEED in $SEEDS; do
      TAG="${DS}_${ARM}_s${SEED}"
      RUNLOG="$LOGDIR/${TAG}.trainlog"
      EXPROOT="logging/Retrieval/${DS}/${GROUP}"
      rm -rf "$EXPROOT"
      T0=$(date +%s)
      python ./src/run_rac.py \
          --batch_size 64 --lr 0.0001 --epochs 30 --topk 20 \
          --dataset "$DS" --model "$MODEL" \
          --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
          --fusion_mode "align" \
          --hard_negatives_loss True --no_hard_negatives 1 \
          --final_eval False --seed "$SEED" --group_name "$GROUP" \
          --metric "cos" --loss "triplet" --batch_norm False \
          --hybrid_loss True --warmup 5 \
          --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
          --lambda_seg 0 \
          --contrast_mode none \
          --exp_comment "_R7SV_${ARM}" \
          --Faiss_GPU False --force False \
          "${EXTRA[@]}" \
          > "$RUNLOG" 2>&1
      RC=$?
      NEPOCH=$(grep -c "^test Epoch" "$RUNLOG")
      rm -rf "$EXPROOT"
      T1=$(date +%s); DONE=$((DONE+1))
      if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 30 ]; then
        NFAIL=$((NFAIL+1))
        echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC nepoch=$NEPOCH secs=$((T1-T0))"
      fi
      if [ $((DONE % 20)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
        echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL last=$TAG elapsed=$((T1-START_TS))s eta=$(( (T1-START_TS)*(TOTAL-DONE)/DONE ))s"
      fi
    done
  done
done

echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
