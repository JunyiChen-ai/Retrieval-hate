#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate;PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python;EXP="$ROOT/experiments/20260901_dense_negative_marked_splat";OUT="${FORMAL_DIR:-$ROOT/runs/20260901_dense_negative_marked_splat/formal_seed234}";mkdir -p "$OUT"
mkdir -p "$OUT/hatemm" "$OUT/hateclipseg"
"$PY" "$EXP/train_select.py" --corpus hatemm --output-dir "$OUT/hatemm" --lr .00005 --max-epoch 80 --k-proportion 8 --lambda-smooth .01 --lambda-contrast .05 --hidden 512 --embed 64 --dropout .05 --temperature .07 > "$OUT/hatemm/run.log" 2>&1
"$PY" "$EXP/train_select.py" --corpus hateclipseg --output-dir "$OUT/hateclipseg" --lr .0002 --max-epoch 100 --k-proportion 3 --lambda-smooth .01 --lambda-contrast 0 --hidden 512 --embed 256 --dropout .05 --temperature .03 > "$OUT/hateclipseg/run.log" 2>&1
for c in hatemm hateclipseg;do "$PY" "$EXP/predict_test.py" --corpus "$c" --run-dir "$OUT/$c" > "$OUT/$c/predict.log" 2>&1;"$PY" "$EXP/evaluate.py" --corpus "$c" --run-dir "$OUT/$c" > "$OUT/$c/evaluate.log" 2>&1;done
"$PY" "$EXP/summarize.py" --formal-dir "$OUT" > "$OUT/summary.log" 2>&1
