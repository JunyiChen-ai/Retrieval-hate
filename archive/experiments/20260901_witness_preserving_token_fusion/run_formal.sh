#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate
EXP="$ROOT/experiments/20260901_witness_preserving_token_fusion"
RUN="$ROOT/runs/20260901_witness_preserving_token_fusion/formal_val_selected_seed234"
mkdir -p "$RUN"; echo $$ > "$RUN/run.pid"
bash "$EXP/search_and_test.sh" hatemm
bash "$EXP/search_and_test.sh" hateclipseg
/home/jehc223/miniconda3/envs/HateVideo/bin/python "$EXP/summarize.py" \
  --run-root "$RUN" > "$RUN/summary.log" 2>&1
