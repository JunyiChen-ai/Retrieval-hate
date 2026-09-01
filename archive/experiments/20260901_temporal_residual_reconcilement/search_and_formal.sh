#!/usr/bin/env bash
set -euo pipefail

corpus="$1"
ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_temporal_residual_reconcilement"
SEARCH="$ROOT/runs/20260901_temporal_residual_reconcilement/val_search/$corpus"
FORMAL="$ROOT/runs/20260901_temporal_residual_reconcilement/formal_val_selected_seed234/$corpus"
mkdir -p "$SEARCH" "$FORMAL"
echo $$ > "$SEARCH/run.pid"

if [[ "$corpus" == "hatemm" ]]; then
  base_lr=1.849152228476098e-05
  fixed=(--lambda-smooth .01420807210603241 --lambda-contrast .18733857665415116 \
    --hidden 512 --embed 64 --dropout .05 --temperature .07)
  specs=(
    "$base_lr .05 8" "$base_lr .1 8" "$base_lr .25 8"
    "$base_lr .5 8" "$base_lr 1 8"
    "9.24576114238049e-06 .25 8" "3.698304456952196e-05 .25 8"
    "3.698304456952196e-05 .5 8"
    "$base_lr .1 4" "$base_lr .25 4"
    "$base_lr .1 12" "3.698304456952196e-05 .1 12")
elif [[ "$corpus" == "hateclipseg" ]]; then
  base_lr=.00018190822304650636
  fixed=(--lambda-smooth .10337306075094418 --lambda-contrast .03728675834293724 \
    --hidden 512 --embed 256 --dropout .05 --temperature .03)
  specs=(
    "$base_lr .05 3" "$base_lr .1 3" "$base_lr .25 3"
    "$base_lr .5 3" "$base_lr 1 3"
    "9.095411152325318e-05 .25 3" "3.638164460930127e-04 .25 3"
    "3.638164460930127e-04 .5 3"
    "$base_lr .1 2" "$base_lr .25 2"
    "$base_lr .1 5" "3.638164460930127e-04 .1 5")
else
  echo "unsupported corpus: $corpus" >&2
  exit 2
fi

index=0
for spec in "${specs[@]}"; do
  index=$((index + 1))
  read -r lr residual kprop <<< "$spec"
  trial=$(printf 'trial_%02d' "$index")
  out="$SEARCH/$trial"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/train.py" --corpus "$corpus" --arm temporal_residual \
    --output-dir "$out" --lr "$lr" --lambda-residual "$residual" \
    --k-proportion "$kprop" --max-epoch 90 "${fixed[@]}" \
    > "$out/run.log" 2>&1
done

"$PY" "$EXP/select_validation.py" --search-root "$SEARCH" \
  > "$SEARCH/selection.log" 2>&1
mapfile -t chosen < <("$PY" "$EXP/select_validation.py" \
  --search-root "$SEARCH" --print-args)

for arm in cyclic_control temporal_residual; do
  out="$FORMAL/$arm"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/train.py" --corpus "$corpus" --arm "$arm" \
    --output-dir "$out" --run-test "${chosen[@]}" > "$out/run.log" 2>&1
  "$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$out" \
    >> "$out/run.log" 2>&1
done
