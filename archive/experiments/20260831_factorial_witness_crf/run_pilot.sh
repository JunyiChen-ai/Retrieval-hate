#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_factorial_witness_crf"
root="$repo/runs/20260831_factorial_witness_crf/pilot_seed234"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python

if [[ ! -x "$python" ]]; then
  echo "HateVideo Python is unavailable: $python" >&2
  exit 1
fi

for corpus in hatemm hateclipseg; do
  for arm in core zero_transition collapsed; do
    run_dir="$root/$corpus/$arm"
    mkdir -p "$run_dir"
    "$python" "$exp/train.py" --corpus "$corpus" --arm "$arm" --out-dir "$run_dir" \
      --seed 234 --epochs 40 --batch-size 16 --hidden 128 --device cuda
    "$python" "$exp/predict.py" --checkpoint "$run_dir/model.pt" \
      --scores-out "$run_dir/scores.jsonl" --device cuda
    "$python" "$exp/evaluate.py" --corpus "$corpus" --scores "$run_dir/scores.jsonl" \
      --metrics-out "$run_dir/metrics.json"
  done
done
