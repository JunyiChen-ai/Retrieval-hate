#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
OUT="${CVOI_VISUAL_OUT:-artifacts/cvoi_acq/premetric-v2/visual-v7}"
DUR=artifacts/cvoi_acq/premetric-v2/durations/hatemm_train_val.json
mkdir -p "$OUT"
PYTHONPATH=. "$PY" -m scripts.cvoi_acq.start_manifest \
  --out "$OUT/start_manifest.json" --role train --role val \
  --code scripts/cvoi_acq/visual_assets.py --code scripts/cvoi_acq/actions.py \
  --code scripts/cvoi_acq/common.py --code scripts/cvoi_acq/run_visual_assets.sh \
  --code "$DUR"
echo "[visual-assets] start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for role in train val; do
  echo "[visual-assets] role=$role start $(date -Is)"
  PYTHONPATH=. "$PY" -m scripts.cvoi_acq.visual_assets --role "$role" --duration-registry "$DUR" --out-dir "$OUT"
  echo "[visual-assets] role=$role done $(date -Is)"
done
echo "[visual-assets] done $(date -Is)"
