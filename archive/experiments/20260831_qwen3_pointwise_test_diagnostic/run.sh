#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_qwen3_pointwise_test_diagnostic"
run_root="$repo/runs/20260831_qwen3_pointwise_test_diagnostic"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python

for corpus in hatemm hateclipseg; do
  out="$run_root/$corpus"
  mkdir -p "$out"
  "$python" "$exp/generate.py" \
    --corpus "$corpus" --out-dir "$out" --device cuda
  "$python" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
    --corpus "$corpus" --scores "$out/scores.jsonl" \
    --split test --branch score_qwen3_pointwise \
    --json-out "$out/metrics.json" --require-full-coverage
done
