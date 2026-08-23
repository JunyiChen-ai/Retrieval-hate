#!/usr/bin/env bash
# A0 +- OCR end-to-end, per idea-stage/A0_OCR_E2E_FREEZE.md.
# 2 arms x 3 seeds, serial, single invocation. Val-only (test firewall ON).
#
#   DATA_ROOT   data root passed to --path            (default ./data/)
#   OUT_ROOT    --output_path                          (default ./logging/)
#   EPOCHS      --epochs                               (default 30)
#   SEEDS       space-separated seed list              (default "0 1 2")
#   GROUP       --group_name                           (default A0_OCR_E2E)
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

DATA_ROOT="${DATA_ROOT:-./data/}"
OUT_ROOT="${OUT_ROOT:-./logging/}"
EPOCHS="${EPOCHS:-30}"
SEEDS="${SEEDS:-0 1 2}"
GROUP="${GROUP:-A0_OCR_E2E}"
LOGDIR="${LOGDIR:-$REPO/logging/runs/a0_ocr_e2e}"
# NOTE: '{split}' must not sit inside a ${VAR:-default} expansion -- its closing
# brace would terminate the expansion early. Assign in two steps.
if [ -z "${OCR_TMPL:-}" ]; then
  OCR_TMPL="${DATA_ROOT}/OCR/HateMM/rac_ocrmean30_"'{split}'".pt"
fi

mkdir -p "$LOGDIR/trainlogs"

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MODEL=openai_clip-vit-large-patch14-336_HF

run_one () {
  local ARM="$1" SEED="$2"
  local TL="$LOGDIR/trainlogs/arm${ARM}_seed${SEED}.trainlog"
  local EXTRA=()
  if [ "$ARM" = "B" ]; then
    EXTRA=(--archive_feats "$OCR_TMPL" --archive_mode stream)
  fi
  echo "PROGRESS arm=$ARM seed=$SEED epoch=start $(date -Is)"
  echo "########## ARM=$ARM SEED=$SEED -> $TL ##########"
  python "$REPO/src/run_rac.py" \
      --path "$DATA_ROOT" --output_path "$OUT_ROOT" \
      --batch_size 64 --lr 0.0001 --epochs "$EPOCHS" --topk 20 \
      --dataset HateMM --model "$MODEL" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode align \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "$GROUP" \
      --metric cos --loss triplet --batch_norm False \
      --hybrid_loss True --warmup 5 \
      --majority_voting arithmetic --no_pseudo_gold_positives 1 \
      --lambda_seg 0 --seg_mode full --num_subclips 4 \
      --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
      --exp_comment "_a0ocre2e_arm${ARM}" \
      --val_only_eval True \
      --Faiss_GPU False --force True \
      "${EXTRA[@]}" 2>&1 | tee "$TL" | awk -v arm="$ARM" -v seed="$SEED" '
        # BLIND CONSOLE: only structural lines and a metric-free epoch counter
        # reach run.log. No candidate number is echoed while the run is live.
        /^Val_Retrieval Epoch/ { print "PROGRESS arm=" arm " seed=" seed " epoch=" $3; next }
        /^(\[val_only_eval\]|\[archive\]|Image feature|Text feature|Traceback|ValueError|RuntimeError|AssertionError|FileNotFoundError)/ { print; next }
        /Error|error:/ { print; next }
      ' || true
  echo "PROGRESS arm=$ARM seed=$SEED epoch=done $(date -Is)"
}

for SEED in $SEEDS; do
  for ARM in A B; do
    run_one "$ARM" "$SEED"
  done
done

echo "======== A0_OCR_E2E ALL DONE $(date -Is) ========"
