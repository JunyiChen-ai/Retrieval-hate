#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run_one.sh CORPUS ARM" >&2
  exit 2
fi
corpus=$1
arm=$2
repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_factorial_witness_crf"
root="$repo/runs/20260831_factorial_witness_crf/pilot_seed234"
run_dir="$root/$corpus/$arm"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python

mkdir -p "$run_dir"
"$python" "$exp/train.py" --corpus "$corpus" --arm "$arm" --out-dir "$run_dir" \
  --seed 234 --epochs 40 --batch-size 16 --hidden 128 --device cuda
"$python" "$exp/predict.py" --checkpoint "$run_dir/model.pt" \
  --scores-out "$run_dir/scores.jsonl" --device cuda
"$python" "$exp/evaluate.py" --corpus "$corpus" --scores "$run_dir/scores.jsonl" \
  --metrics-out "$run_dir/metrics.json"
