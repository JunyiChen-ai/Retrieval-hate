#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_validation_selected_marked_splat"
OUT="${FORMAL_DIR:-$ROOT/runs/20260901_validation_selected_marked_splat/formal_seed234}"
mkdir -p "$OUT"
run_trial(){ local corpus="$1" name="$2"; shift 2; local trial="$OUT/$corpus/trials/$name"; mkdir -p "$trial"; "$PY" "$EXP/train_select.py" --corpus "$corpus" --output-dir "$trial" --config-name "$name" "$@" > "$trial/run.log" 2>&1; }
run_trial hatemm base --lr 1.849152228476098e-05 --max-epoch 80 --k-proportion 8 --lambda-smooth .01420807210603241 --lambda-contrast .18733857665415116 --hidden 512 --embed 64 --dropout .05 --temperature .07
run_trial hatemm mid_lr --lr .00005 --max-epoch 80 --k-proportion 8 --lambda-smooth .01420807210603241 --lambda-contrast .18733857665415116 --hidden 512 --embed 64 --dropout .05 --temperature .07
run_trial hatemm low_regularization --lr .00005 --max-epoch 80 --k-proportion 8 --lambda-smooth .01 --lambda-contrast .05 --hidden 512 --embed 64 --dropout .05 --temperature .07
run_trial hatemm bag_focus --lr .0001 --max-epoch 80 --k-proportion 8 --lambda-smooth .005 --lambda-contrast 0 --hidden 512 --embed 64 --dropout .05 --temperature .07
run_trial hateclipseg base --lr .00018190822304650636 --max-epoch 100 --k-proportion 3 --lambda-smooth .10337306075094418 --lambda-contrast .03728675834293724 --hidden 512 --embed 256 --dropout .05 --temperature .03
run_trial hateclipseg low_lr --lr .00005 --max-epoch 100 --k-proportion 3 --lambda-smooth .10337306075094418 --lambda-contrast .03728675834293724 --hidden 512 --embed 256 --dropout .05 --temperature .03
run_trial hateclipseg low_regularization --lr .0001 --max-epoch 100 --k-proportion 3 --lambda-smooth .03 --lambda-contrast .01 --hidden 512 --embed 256 --dropout .05 --temperature .03
run_trial hateclipseg bag_focus --lr .0002 --max-epoch 100 --k-proportion 3 --lambda-smooth .01 --lambda-contrast 0 --hidden 512 --embed 256 --dropout .05 --temperature .03
for corpus in hatemm hateclipseg; do "$PY" "$EXP/select_config.py" --corpus-dir "$OUT/$corpus" > "$OUT/$corpus/selection.log" 2>&1; done
for corpus in hatemm hateclipseg; do "$PY" "$EXP/predict_test.py" --corpus "$corpus" --corpus-dir "$OUT/$corpus" > "$OUT/$corpus/predict.log" 2>&1; "$PY" "$EXP/evaluate.py" --corpus "$corpus" --corpus-dir "$OUT/$corpus" > "$OUT/$corpus/evaluate.log" 2>&1; done
"$PY" "$EXP/summarize.py" --formal-dir "$OUT" > "$OUT/summary.log" 2>&1
