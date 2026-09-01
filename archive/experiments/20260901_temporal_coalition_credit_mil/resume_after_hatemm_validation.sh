#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_temporal_coalition_credit_mil"
SEARCH="$ROOT/runs/20260901_temporal_coalition_credit_mil/val_search/hatemm"
RUN="$ROOT/runs/20260901_temporal_coalition_credit_mil/formal_val_selected_seed234"
echo $$ > "$RUN/run.pid"

for role in anchor aligned shifted; do
  out="$RUN/hatemm/$role"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/infer_selected.py" --selection "$SEARCH/selection.json" \
    --role "$role" --output-dir "$out" > "$out/run.log" 2>&1
  "$PY" "$EXP/evaluate.py" --corpus hatemm --run-dir "$out" \
    >> "$out/run.log" 2>&1
done

bash "$EXP/search_and_test.sh" hateclipseg
"$PY" "$EXP/summarize.py" --run-root "$RUN" > "$RUN/summary.log" 2>&1
