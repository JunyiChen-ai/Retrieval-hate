#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
PILOT="${PILOT_DIR:-$ROOT/runs/20260831_witness_conditional_dgm/pilot_seed234_matched}"
mkdir -p "$PILOT"

for corpus in hatemm hateclipseg; do
  if [[ "$corpus" == "hatemm" ]]; then
    corpus_args=(--lr 1.849152228476098e-05 --max-epoch 50 \
      --k-proportion 8 --lambda-smooth .01420807210603241 \
      --lambda-contrast .18733857665415116 --hidden 512 --embed 64 \
      --dropout .05 --temperature .07)
  else
    corpus_args=(--lr .00018190822304650636 --max-epoch 100 \
      --k-proportion 3 --lambda-smooth .10337306075094418 \
      --lambda-contrast .03728675834293724 --hidden 512 --embed 256 \
      --dropout .05 --temperature .03)
  fi
  for arm in anchor source_dgm witness_dgm; do
    run_dir="$PILOT/$corpus/$arm"
    mkdir -p "$run_dir"
    echo $$ > "$run_dir/run.pid"
    "$PY" "$ROOT/experiments/20260831_witness_conditional_dgm/train.py" \
      --corpus "$corpus" --arm "$arm" --output-dir "$run_dir" \
      "${corpus_args[@]}" \
      > "$run_dir/run.log" 2>&1
    "$PY" "$ROOT/experiments/20260831_witness_conditional_dgm/evaluate.py" \
      --corpus "$corpus" --run-dir "$run_dir" >> "$run_dir/run.log" 2>&1
  done
done

"$PY" "$ROOT/experiments/20260831_witness_conditional_dgm/summarize.py" \
  --pilot-dir "$PILOT" > "$PILOT/summary.log" 2>&1
