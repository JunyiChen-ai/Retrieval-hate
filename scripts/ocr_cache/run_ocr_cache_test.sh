#!/usr/bin/env bash
# Build the K=30 OCR cache for the HateMM TEST split (215 videos).
# Same builder / engine / params as the train+val build (run_ocr_cache.sh);
# only the input id list differs. Test *inputs* unsealed by user 2026-08-09.
# Background-safe: writes logging/runs/test_ocr_cache/run.{log,pid}.
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/venvs/ocr_paddle/bin/python
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "[run] start $(date -Is)"
$PY scripts/ocr_cache/extract_ocr_windows.py --dataset HateMM_test --engine paddleocr \
    --lang en --out-dir data/OCR/HateMM_test_stage
echo "[run] HateMM_test exit=$?"
echo "[run] done $(date -Is)"
