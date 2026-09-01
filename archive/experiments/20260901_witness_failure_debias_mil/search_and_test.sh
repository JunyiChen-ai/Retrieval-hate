#!/usr/bin/env bash
set -euo pipefail

corpus="$1"
ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_witness_failure_debias_mil"
SEARCH="$ROOT/runs/20260901_witness_failure_debias_mil/val_search/$corpus"
FORMAL="$ROOT/runs/20260901_witness_failure_debias_mil/formal_val_selected_seed234/$corpus"
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
  "$PY" "$EXP/train.py" --corpus "$corpus" --arm anchor \
    --lambda-failure 0 --lr "$lr" --output-dir "$out" \
    > "$out/run.log" 2>&1
  anchor_logs+=("$out/train_log.json")
done

declare -A uniform_logs
lr_index=0
for lr in "$lr_low" "$lr_high"; do
  anchor_reference="${anchor_logs[$lr_index]}"
  lr_index=$((lr_index + 1))
  for strength in .25 .5 1.0; do
    trial=$((trial + 1))
    out="$SEARCH/$(printf 'trial_%02d' "$trial")"
    mkdir -p "$out"
    echo $$ > "$out/run.pid"
    "$PY" "$EXP/train.py" --corpus "$corpus" --arm uniform \
      --lambda-failure "$strength" --lr "$lr" \
      --reference-log "$anchor_reference" --output-dir "$out" \
      > "$out/run.log" 2>&1
    uniform_logs["$lr,$strength"]="$out/train_log.json"
  done
done

for lr in "$lr_low" "$lr_high"; do
  for strength in .25 .5 1.0; do
    trial=$((trial + 1))
    out="$SEARCH/$(printf 'trial_%02d' "$trial")"
    mkdir -p "$out"
    echo $$ > "$out/run.pid"
    "$PY" "$EXP/train.py" --corpus "$corpus" --arm relative \
      --lambda-failure "$strength" --lr "$lr" \
      --reference-log "${uniform_logs[$lr,$strength]}" --output-dir "$out" \
      > "$out/run.log" 2>&1
  done
done

"$PY" "$EXP/select_validation.py" --search-root "$SEARCH" \
  > "$SEARCH/selection.log" 2>&1

for role in anchor uniform relative; do
  out="$FORMAL/$role"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/infer_selected.py" --selection "$SEARCH/selection.json" \
    --role "$role" --output-dir "$out" > "$out/run.log" 2>&1
  "$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$out" \
    >> "$out/run.log" 2>&1
done
