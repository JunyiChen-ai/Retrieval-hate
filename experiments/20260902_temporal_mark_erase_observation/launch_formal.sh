#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
exp_dir="$repo/experiments/20260902_temporal_mark_erase_observation"
run_root="$repo/runs/20260902_temporal_mark_erase_observation/formal"
mkdir -p "$run_root"

stable_free=0
while (( stable_free < 3 )); do
  read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  printf '%s gpu_memory_used_mb=%s gpu_utilization_pct=%s\n' "$(date -Iseconds)" "$used" "$util"
  if (( used < 2000 && util < 10 )); then
    stable_free=$((stable_free + 1))
  else
    stable_free=0
  fi
  if (( stable_free < 3 )); then
    sleep 30
  fi
done

for corpus in hatemm hateclipseg; do
  corpus_dir="$run_root/$corpus"
  mkdir -p "$corpus_dir"
  "$python_bin" "$exp_dir/run_observation.py" --corpus "$corpus" \
    > "$corpus_dir/run.log" 2>&1 &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$corpus_dir/run.pid"
  wait "$child_pid"
done

"$python_bin" "$exp_dir/evaluate.py" > "$run_root/evaluate.log" 2>&1 &
eval_pid=$!
printf '%s\n' "$eval_pid" > "$run_root/evaluate.pid"
wait "$eval_pid"
printf '%s formal_chain_complete\n' "$(date -Iseconds)"
