#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp="$repo/experiments/20260831_owner_abstaining_its2clr"
root="$repo/runs/20260831_owner_abstaining_its2clr/pilot_seed234"
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python
corpus=hateclipseg

for arm in anchor broadcast core; do
  run_dir="$root/$corpus/$arm"
  mkdir -p "$run_dir"
  oof_args=()
  if [[ "$arm" != anchor ]]; then
    oof_dir="$run_dir/oof"
    mkdir -p "$oof_dir"
    "$python" "$exp/oof.py" --corpus "$corpus" --arm "$arm" \
      --out "$oof_dir/pseudo.pt" --seed 234 --folds 3 --seed-epochs 40 \
      --refine-epochs 15 --refresh-every 5 --batch-size 32 --device cuda
    "$python" "$exp/analyze_oof.py" --cache "$oof_dir/pseudo.pt" \
      --out "$oof_dir/diagnostics.json"
    oof_args=(--oof "$oof_dir/pseudo.pt")
  fi
  "$python" "$exp/train.py" --corpus "$corpus" --arm "$arm" \
    "${oof_args[@]}" --out-dir "$run_dir" --seed 234 --epochs 60 \
    --batch-size 24 --device cuda
  "$python" "$exp/predict.py" --checkpoint "$run_dir/model.pt" \
    --scores-out "$run_dir/scores.jsonl" --device cuda
  "$python" "$exp/evaluate.py" --corpus "$corpus" \
    --scores "$run_dir/scores.jsonl" --metrics-out "$run_dir/metrics.json"
done

"$python" "$exp/summarize_rejection.py" --root "$root"

