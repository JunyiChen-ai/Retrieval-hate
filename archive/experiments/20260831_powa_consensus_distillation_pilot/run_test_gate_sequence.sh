#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
experiment="$repo/experiments/20260831_powa_consensus_distillation_pilot"
cache_root="$repo/runs/20260831_powa_consensus_distillation_pilot/teacher_cache"
hmm_dir="$cache_root/hatemm_test"
hcs_dir="$cache_root/hateclipseg_test"
diagnostic_dir="$repo/runs/20260831_powa_consensus_distillation_pilot/test_teacher_diagnostic"

hmm_pid=$(tr -d '\n' < "$hmm_dir/run.pid")
while [[ ! -f "$hmm_dir/manifest.json" ]]; do
    if ! kill -0 "$hmm_pid" 2>/dev/null; then
        printf 'HMM test cache process ended without a manifest\n' >&2
        exit 1
    fi
    sleep 30
done

mkdir -p "$hcs_dir/raw"
printf '%s\n' "$$" > "$hcs_dir/run.pid"
"$python_bin" "$experiment/prepare_vera_k16.py" \
    --corpus hateclipseg --split test --raw-root "$hcs_dir/raw" \
    > "$hcs_dir/run.log" 2>&1
test -f "$hcs_dir/manifest.json"

mkdir -p "$diagnostic_dir"
printf '%s\n' "$$" > "$diagnostic_dir/run.pid"
"$python_bin" "$experiment/test_teacher_diagnostic.py" \
    > "$diagnostic_dir/run.log" 2>&1
test -f "$diagnostic_dir/analysis.json"
printf 'test gate sequence complete\n'
