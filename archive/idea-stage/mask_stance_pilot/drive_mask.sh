#!/usr/bin/env bash
# MASK STANCE PILOT driver: extract(batch) -> mask -> stance(batch) -> score.
# Idempotent: each stage is skipped if its output already exists.
set -u
TAG="${1:-m1}"
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
D=idea-stage/mask_stance_pilot
R=run_mask.py

echo "[drive] === STEP 1 extraction (Batch API, transcript only) ==="
if [ ! -f "$D/extract_${TAG}.jsonl" ]; then
  if [ ! -f "$D/batch_meta_ext_${TAG}.json" ]; then
    python $D/$R extract --split eval --tag "$TAG" --batch || exit 1
  fi
  python $D/$R poll  --tag "$TAG" --stage ext || exit 1
  python $D/$R fetch --tag "$TAG" --stage ext || exit 1
else
  echo "[drive] extract cached"
fi

echo "[drive] === STEP 1.5 masking (programmatic) ==="
python $D/$R mask --split eval --tag "$TAG" || exit 1

echo "[drive] === STEP 2 masked stance (Batch API, 8 frames + masked transcript) ==="
if [ ! -f "$D/pred_${TAG}.jsonl" ]; then
  if [ ! -f "$D/batch_meta_stn_${TAG}.json" ]; then
    python $D/$R stance --split eval --tag "$TAG" --batch || exit 1
  fi
  python $D/$R poll  --tag "$TAG" --stage stn || exit 1
  python $D/$R fetch --tag "$TAG" --stage stn || exit 1
else
  echo "[drive] stance cached"
fi

echo "[drive] === SCORE ==="
python $D/score_mask.py --tag "$TAG" || exit 1
echo "[drive] done"
