#!/usr/bin/env bash
# Build the K=30 OCR cache for HateMM (train+val whitelist) and HateClipSeg (395).
# Background-safe: nohup'd by the caller, writes logging/runs/ocr_cache/run.{log,pid}.
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/venvs/ocr_paddle/bin/python
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "[run] start $(date -Is)"
for DS in HateMM HateClipSeg; do
  echo "[run] === dataset $DS ==="
  $PY scripts/ocr_cache/extract_ocr_windows.py --dataset "$DS" --engine paddleocr --lang en
  echo "[run] $DS exit=$?"
done
echo "[run] done $(date -Is)"
