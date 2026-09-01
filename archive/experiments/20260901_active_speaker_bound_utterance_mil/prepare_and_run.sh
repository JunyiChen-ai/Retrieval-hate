#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
HERE="$ROOT/experiments/20260901_active_speaker_bound_utterance_mil"
RUN_ROOT="$ROOT/runs/20260901_active_speaker_bound_utterance_mil"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python

mkdir -p "$RUN_ROOT/cache_hatemm" "$RUN_ROOT/cache_hateclipseg" \
  "$RUN_ROOT/formal_seed234"
"$PY" "$ROOT/scripts/build_active_speaker_bound_cache.py" \
  --corpus hatemm --device cuda \
  > "$RUN_ROOT/cache_hatemm/run.log" 2>&1
"$PY" "$ROOT/scripts/build_active_speaker_bound_cache.py" \
  --corpus hateclipseg --device cuda \
  > "$RUN_ROOT/cache_hateclipseg/run.log" 2>&1
bash "$HERE/run_formal.sh" > "$RUN_ROOT/formal_seed234/run.log" 2>&1
