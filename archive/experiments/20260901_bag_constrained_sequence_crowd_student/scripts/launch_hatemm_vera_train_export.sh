#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/jehc223/Retrieval-hate
PYTHON_BIN=/home/jehc223/miniconda3/envs/HateVideo/bin/python
RUN_DIR="$PROJECT_ROOT/runs/20260901_sequence_crowd_teacher_export/hatemm_vera_train"
RAW_DIR="$PROJECT_ROOT/data/sequence_crowd_sources/vera_sparse/hatemm/raw"
PROMPT=/home/jehc223/Hate-follow-up/results/reproduction/official_val/tuning/vera/hatemm/selected_prompt.json

mkdir -p "$RUN_DIR" "$RAW_DIR"
printf '%s\n' "$$" > "$RUN_DIR/run.pid"
printf 'Waiting for GPU availability before loading the frozen VERA producer.\n'
while :; do
  USED_MIB=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
    | awk '{sum += $1} END {print sum + 0}')
  if [ "$USED_MIB" -lt 4000 ]; then
    break
  fi
  printf 'GPU busy: %s MiB in use; checking again in 30 seconds.\n' "$USED_MIB"
  sleep 30
done

exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_sparse_vera_train_scores.py" \
  --corpus hatemm --raw-root "$RAW_DIR" --prompt-json "$PROMPT" \
  --out "$RUN_DIR/scores.jsonl" --generate-missing
