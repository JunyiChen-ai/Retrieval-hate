#!/usr/bin/env bash
set -euo pipefail

corpus="$1"
ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_sparse_mixture_scan_mil"
SEARCH="$ROOT/runs/20260901_sparse_mixture_scan_mil/val_search/$corpus"
FORMAL="$ROOT/runs/20260901_sparse_mixture_scan_mil/formal_val_selected_seed234/$corpus"
mkdir -p "$SEARCH" "$FORMAL"
echo $$ > "$SEARCH/run.pid"

if [[ "$corpus" == "hatemm" ]]; then
  base_lr=1.849152228476098e-05
  half_lr=9.24576114238049e-06
  double_lr=3.698304456952196e-05
  fixed=(--max-epoch 60 --k-proportion 8 \
    --lambda-smooth .01420807210603241 --lambda-contrast .18733857665415116 \
    --hidden 512 --embed 64 --dropout .05 --temperature .07)
else
  base_lr=.00018190822304650636
  half_lr=9.095411152325318e-05
  double_lr=3.638164460930127e-04
  fixed=(--max-epoch 100 --k-proportion 3 \
    --lambda-smooth .10337306075094418 --lambda-contrast .03728675834293724 \
    --hidden 512 --embed 256 --dropout .05 --temperature .03)
fi

# lr, lambda_scan, margin, max-rank fraction, dense-negative weight, scan temperature
specs=(
  "$base_lr .1 .1 .5 .05 1"
  "$base_lr .25 .1 .5 .05 1"
  "$base_lr .5 .1 .5 .05 1"
  "$base_lr 1 .1 .5 .05 1"
  "$base_lr .25 .25 .5 .05 .5"
  "$base_lr .25 .25 .5 .05 2"
  "$base_lr .25 .25 .25 .05 1"
  "$base_lr .25 .25 .1 .05 1"
  "$double_lr .25 .25 .5 .05 1"
  "$half_lr .25 .25 .5 .05 1"
  "$base_lr .5 .25 .5 .1 1"
  "$base_lr .5 .25 .5 .25 1")

index=0
for spec in "${specs[@]}"; do
  index=$((index + 1))
  read -r lr scan margin max_fraction dense scan_temperature <<< "$spec"
  trial=$(printf 'trial_%02d' "$index")
  out="$SEARCH/$trial"
  mkdir -p "$out"
  echo $$ > "$out/run.pid"
  "$PY" "$EXP/train.py" --corpus "$corpus" --arm sparse_scan \
    --output-dir "$out" --lr "$lr" --lambda-scan "$scan" \
    --scan-margin "$margin" --rank-max-fraction "$max_fraction" \
    --lambda-dense "$dense" --n-ranks 8 --null-momentum .9 \
    --scan-temperature "$scan_temperature" \
    "${fixed[@]}" > "$out/run.log" 2>&1
done

"$PY" "$EXP/select_validation.py" --search-root "$SEARCH" \
  > "$SEARCH/selection.log" 2>&1
mapfile -t chosen < <("$PY" "$EXP/select_validation.py" \
  --search-root "$SEARCH" --print-args)

core_out="$FORMAL/sparse_scan"
mkdir -p "$core_out"
echo $$ > "$core_out/run.pid"
"$PY" "$EXP/infer_selected.py" --selection "$SEARCH/selection.json" \
  --output-dir "$core_out" > "$core_out/run.log" 2>&1
"$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$core_out" \
  >> "$core_out/run.log" 2>&1

control_out="$FORMAL/fixed_topk_control"
mkdir -p "$control_out"
echo $$ > "$control_out/run.pid"
"$PY" "$EXP/train.py" --corpus "$corpus" --arm fixed_topk_control \
  --output-dir "$control_out" --run-test "${chosen[@]}" \
  > "$control_out/run.log" 2>&1
"$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$control_out" \
  >> "$control_out/run.log" 2>&1
