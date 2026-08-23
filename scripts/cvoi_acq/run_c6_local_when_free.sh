#!/usr/bin/env bash
set -euo pipefail
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
OUT=artifacts/cvoi_acq/premetric-v2/c6-cost-v1
mkdir -p "$OUT" logging/cvoi_c6
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]+'; do
  date --iso-8601=seconds >> logging/cvoi_c6/gpu_wait.log
  sleep 600
done
python -m scripts.cvoi_acq.cost_driver --type ocr \
 --train-actions artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl \
 --val-actions artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl \
 --out "$OUT/ocr_cost_actions.jsonl" > logging/cvoi_c6/ocr.log 2>&1
python -m scripts.cvoi_acq.cost_driver --type dense \
 --train-actions artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl \
 --val-actions artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl \
 --out "$OUT/dense_cost_actions.jsonl" > logging/cvoi_c6/dense.log 2>&1
python -m scripts.cvoi_acq.cost_overhead_driver --out "$OUT/overhead_costs.jsonl" > logging/cvoi_c6/overhead.log 2>&1
python -m scripts.cvoi_acq.cost_audit --ocr "$OUT/ocr_cost_actions.jsonl" --dense "$OUT/dense_cost_actions.jsonl" \
 --train-actions artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl \
 --val-actions artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl \
 --components artifacts/cvoi_acq/premetric-v2/groups-v4/group_components.json \
 --outer artifacts/cvoi_acq/premetric-v2/groups-v4/outer_folds.json --out "$OUT/independent_audit_v1.json" \
 > logging/cvoi_c6/audit.log 2>&1
