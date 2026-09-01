#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_target_conditioned_normal_proposal_mil"
root="$repo/runs/20260831_target_conditioned_normal_proposal_mil/premise"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python
evaluator="$repo/scripts/reproduction_baselines/eval_baseline_scores.py"

if [[ ! -x "$python" ]]; then
  echo "HateVideo Python is unavailable: $python" >&2
  exit 1
fi

for corpus in hatemm hateclipseg; do
  run_dir="$root/$corpus"
  mkdir -p "$run_dir"
  "$python" "$exp/probe.py" --corpus "$corpus" --out-dir "$run_dir" --seed 234
  "$python" "$evaluator" --corpus "$corpus" --split test \
    --scores "$run_dir/scores.jsonl" --json-out "$run_dir/metrics.json" \
    --require-full-coverage
done

"$python" "$exp/decide_premise.py" --run-root "$root"
