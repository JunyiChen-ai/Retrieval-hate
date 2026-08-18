#!/usr/bin/env bash
# REPRO campaign Phase A step 4: backfill MHC + MHC_zh train/dev OCR (K=30),
# matching the already-built data/OCR/{MHC,MHC_zh}_test caches exactly.
# Decision recorded in REPRO_CAMPAIGN_FREEZE.md §13 (K=30, not K=4).
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/venvs/ocr_paddle/bin/python
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
echo "[run] start $(date -Is)"
$PY scripts/ocr_cache/extract_ocr_windows.py --dataset MHC --engine paddleocr --lang en
echo "[run] MHC exit=$?"
$PY scripts/ocr_cache/extract_ocr_windows.py --dataset MHC_zh --engine paddleocr --lang ch
echo "[run] MHC_zh exit=$?"
echo "[run] done $(date -Is)"
