#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: postprocess_existing_train.sh TRAIN_PID CORPUS ARM" >&2
  exit 2
fi
train_pid=$1
corpus=$2
arm=$3
repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_factorial_witness_crf"
run_dir="$repo/runs/20260831_factorial_witness_crf/pilot_seed234/$corpus/$arm"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python

while kill -0 "$train_pid" 2>/dev/null; do
  sleep 20
done
[[ -s "$run_dir/model.pt" && -s "$run_dir/train_log.json" ]]
"$python" "$exp/predict.py" --checkpoint "$run_dir/model.pt" \
  --scores-out "$run_dir/scores.jsonl" --device cuda
"$python" "$exp/evaluate.py" --corpus "$corpus" --scores "$run_dir/scores.jsonl" \
  --metrics-out "$run_dir/metrics.json"
