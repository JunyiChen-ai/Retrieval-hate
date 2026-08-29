#!/usr/bin/env bash
# R7-2 -- on-screen-text provenance rule channel at the decision layer.
# HateMM, ONE arm (A0), 30 seeds (100..129) = 30 head runs.
# The head is NOT touched: the only change vs idea-stage/r6_confirm/run_confirm.sh
# is --dump_head_scores, which appends per-epoch per-item dev/test head logits and
# has no effect on training (verified: byte-identical trainlog to the no-flag run).
# All four combiner arms are derived OFFLINE from these same 30 runs.
# Frozen design: idea-stage/R7_OCRPROV_FREEZE.md.
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="${1:-logging/runs/r7_ocrprov}"
GROUP="R7_OCRPROV_20260817"
LOGDIR="$RUNDIR/logs"
DUMPDIR="$RUNDIR/scores"
mkdir -p "$LOGDIR" "$DUMPDIR"

DS=HateMM
MODEL="R6RO-A0"
SEEDS=$(seq 100 129)

TOTAL=$(echo "$SEEDS" | wc -w); DONE=0; NFAIL=0; START_TS=$(date +%s)
echo "GRID_START total=$TOTAL group=$GROUP ds=$DS at $(date -Is)"

for SEED in $SEEDS; do
  TAG="${DS}_A0_s${SEED}"
  RUNLOG="$LOGDIR/${TAG}.trainlog"
  DUMP="$DUMPDIR/${TAG}.jsonl"
  EXPROOT="logging/Retrieval/${DS}/${GROUP}"
  rm -rf "$EXPROOT"; rm -f "$DUMP"     # dump is opened in append mode
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
      --exp_comment "_R7OP_A0" \
      --Faiss_GPU False --force False \
      --dump_head_scores "$DUMP" \
      > "$RUNLOG" 2>&1
  RC=$?
  NEPOCH=$(grep -c "^test Epoch" "$RUNLOG")
  NDUMP=$(wc -l < "$DUMP" 2>/dev/null || echo 0)
  rm -rf "$EXPROOT"
  T1=$(date +%s); DONE=$((DONE+1))
  if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 30 ] || [ "$NDUMP" -ne 60 ]; then
    NFAIL=$((NFAIL+1))
    echo "PROGRESS seed=$SEED STATUS=FAIL rc=$RC nepoch=$NEPOCH ndump=$NDUMP secs=$((T1-T0))"
  fi
  echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL last=$TAG elapsed=$((T1-START_TS))s"
done

echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
