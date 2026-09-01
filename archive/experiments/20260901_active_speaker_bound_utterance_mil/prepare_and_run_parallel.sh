#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
HERE="$ROOT/experiments/20260901_active_speaker_bound_utterance_mil"
RUN_ROOT="$ROOT/runs/20260901_active_speaker_bound_utterance_mil"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
NUM_SHARDS=4

run_cache_shards() {
  local corpus="$1"
  local log_root="$RUN_ROOT/cache_${corpus}_parallel"
  mkdir -p "$log_root"
  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    "$PY" "$ROOT/scripts/build_active_speaker_bound_cache.py" \
      --corpus "$corpus" --device cuda --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" --face-batch-size 128 \
      > "$log_root/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  printf '%s\n' "${pids[@]}" > "$log_root/run.pids"
  local failed=0
  for worker_pid in "${pids[@]}"; do
    if ! wait "$worker_pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    return 1
  fi
}

mkdir -p "$RUN_ROOT/formal_seed234"
run_cache_shards hatemm
run_cache_shards hateclipseg
bash "$HERE/run_formal.sh" > "$RUN_ROOT/formal_seed234/run.log" 2>&1
