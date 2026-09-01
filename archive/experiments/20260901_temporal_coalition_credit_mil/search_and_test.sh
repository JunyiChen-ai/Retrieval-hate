#!/usr/bin/env bash
set -euo pipefail

corpus="$1"
ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_temporal_coalition_credit_mil"
SEARCH="$ROOT/runs/20260901_temporal_coalition_credit_mil/val_search/$corpus"
FORMAL="$ROOT/runs/20260901_temporal_coalition_credit_mil/formal_val_selected_seed234/$corpus"
mkdir -p "$SEARCH" "$FORMAL"
echo $$ > "$SEARCH/run.pid"

if [[ "$corpus" == "hatemm" ]]; then
  lr_low=9.24576114238049e-06
  lr_high=1.849152228476098e-05
else
  lr_low=9.095411152325318e-05
  lr_high=.00018190822304650636
fi

trial=0
declare -a anchor_logs
for lr in "$lr_low" "$lr_high"; do
  trial=$((trial + 1))
  out="$SEARCH/$(printf 'trial_%02d' "$trial")"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/train.py" --corpus "$corpus" --arm anchor --alpha 0 \
    --lr "$lr" --output-dir "$out" > "$out/run.log" 2>&1
  anchor_logs+=("$out/train_log.json")
done

for arm in aligned shifted; do
  lr_index=0
  for lr in "$lr_low" "$lr_high"; do
    reference="${anchor_logs[$lr_index]}"
    lr_index=$((lr_index + 1))
    for alpha in .25 .5 .75; do
      trial=$((trial + 1))
      out="$SEARCH/$(printf 'trial_%02d' "$trial")"
      mkdir -p "$out"
      echo $$ > "$out/run.pid"
      "$PY" "$EXP/train.py" --corpus "$corpus" --arm "$arm" --alpha "$alpha" \
        --lr "$lr" --reference-log "$reference" --output-dir "$out" \
        > "$out/run.log" 2>&1
    done
  done
done

"$PY" "$EXP/select_validation.py" --search-root "$SEARCH" \
  > "$SEARCH/selection.log" 2>&1

for role in anchor aligned shifted; do
  out="$FORMAL/$role"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/infer_selected.py" --selection "$SEARCH/selection.json" \
    --role "$role" --output-dir "$out" > "$out/run.log" 2>&1
  "$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$out" \
    >> "$out/run.log" 2>&1
done

