#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_temporal_residual_reconcilement"
OUT="${RUN_ROOT:-$ROOT/runs/20260901_temporal_residual_reconcilement/formal_seed234}"
mkdir -p "$OUT"
echo $$ > "$OUT/run.pid"

for corpus in hatemm hateclipseg; do
  if [[ "$corpus" == "hatemm" ]]; then
    corpus_args=(--lr 1.849152228476098e-05 --max-epoch 90 --k-proportion 8 \
      --lambda-smooth .01420807210603241 --lambda-contrast .18733857665415116 \
      --hidden 512 --embed 64 --dropout .05 --temperature .07)
  else
    corpus_args=(--lr .00018190822304650636 --max-epoch 90 \
      --k-proportion 3 --lambda-smooth .10337306075094418 \
      --lambda-contrast .03728675834293724 --hidden 512 --embed 256 \
      --dropout .05 --temperature .03)
  fi
  for arm in cyclic_control temporal_residual; do
    run_dir="$OUT/$corpus/$arm"
    mkdir -p "$run_dir"
    echo $$ > "$run_dir/run.pid"
    "$PY" "$EXP/train.py" --corpus "$corpus" --arm "$arm" \
      --output-dir "$run_dir" --run-test "${corpus_args[@]}" > "$run_dir/run.log" 2>&1
    "$PY" "$EXP/evaluate.py" --corpus "$corpus" --run-dir "$run_dir" \
      >> "$run_dir/run.log" 2>&1
  done
done

"$PY" "$EXP/summarize.py" --run-root "$OUT" > "$OUT/summary.log" 2>&1
