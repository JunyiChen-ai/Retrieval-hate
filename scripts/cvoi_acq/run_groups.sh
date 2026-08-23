#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
OUT="${CVOI_GROUP_OUT:-artifacts/cvoi_acq/premetric-v2/groups-v4}"
DUR=artifacts/cvoi_acq/premetric-v2/durations/hatemm_train_val.json
mkdir -p "$OUT"
PYTHONPATH=. "$PY" -m scripts.cvoi_acq.start_manifest \
  --out "$OUT/start_manifest.json" --role train \
  --code scripts/cvoi_acq/groups.py --code scripts/cvoi_acq/actions.py \
  --code scripts/cvoi_acq/common.py --code scripts/cvoi_acq/run_groups.sh \
  --code "$DUR"
PYTHONPATH=. "$PY" -m scripts.cvoi_acq.groups --out-dir "$OUT"
