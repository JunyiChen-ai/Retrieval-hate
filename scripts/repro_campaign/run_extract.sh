#!/usr/bin/env bash
# REPRO campaign Phase A step 3: dense 4 fps CLIP-L/336 + wav2vec2-emotion for
# HateMM (1083) + MHC-EN (792) + MHC-ZH (814).  Single GPU, strictly serial so a
# second agent keeps headroom on the card.  Idempotent + restartable.
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
export PYTHONUNBUFFERED=1
echo "[run] start $(date -Is)"
for DS in HateMM MHC MHC_zh; do
  echo "[run] === $DS ==="
  $PY scripts/repro_campaign/extract_dense.py --dataset "$DS" --channels visual,audio --batch 48
  echo "[run] $DS exit=$?"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  df -h /home/jehc223 | tail -1
done
echo "[run] done $(date -Is)"
