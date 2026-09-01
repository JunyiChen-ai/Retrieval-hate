#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
python=/home/jehc223/miniconda3/envs/HateVideo/bin/python
run="$repo/runs/20260831_coalition_witness_candidate/pilot_seed234"
mkdir -p "$run"

for corpus in hatemm hateclipseg; do
  if [[ "$corpus" == "hatemm" ]]; then
    lr=0.00001849152228476098
    max_epoch=50
    k_proportion=8
    lambda_smooth=0.01420807210603241
    hidden=512
    embed=64
    dropout=0.05
    temperature=0.07
  else
    lr=0.00018190822304650636
    max_epoch=100
    k_proportion=3
    lambda_smooth=0.10337306075094418
    hidden=512
    embed=256
    dropout=0.05
    temperature=0.03
  fi

  out="$run/$corpus/no_infonce"
  mkdir -p "$out"
  if ! "$python" "$repo/experiments/20260831_coalition_witness_candidate/validate_arm.py" \
      --corpus "$corpus" --arm no_infonce --out-dir "$out" >/dev/null 2>&1; then
    echo "$$" > "$out/run.pid"
    "$python" "$repo/scripts/reproduction_baselines/multihateloc/train.py" \
      --corpus "$corpus" --out-root "$out/producer" --seed 234 \
      --lr "$lr" --batch-size 32 --max-epoch "$max_epoch" \
      --k-proportion "$k_proportion" --lambda-smooth "$lambda_smooth" \
      --lambda-contrast 0 --hidden "$hidden" --embed "$embed" \
      --dropout "$dropout" --temperature "$temperature" \
      --run-test --device cuda \
      > "$out/run.log" 2>&1
    "$python" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
      --corpus "$corpus" --split test \
      --scores "$out/producer/$corpus/scores.jsonl" --branch score_fused \
      --require-full-coverage --json-out "$out/metrics.json" \
      >> "$out/run.log" 2>&1
  fi

  for arm in all_subset_mil synib mobius_nonminimal coalition_witness; do
    out="$run/$corpus/$arm"
    mkdir -p "$out"
    if ! "$python" "$repo/experiments/20260831_coalition_witness_candidate/validate_arm.py" \
        --corpus "$corpus" --arm "$arm" --out-dir "$out" >/dev/null 2>&1; then
      echo "$$" > "$out/run.pid"
      "$python" "$repo/experiments/20260831_coalition_witness_candidate/train.py" \
        --corpus "$corpus" --arm "$arm" --out-dir "$out" --seed 234 \
        --lr "$lr" --batch-size 32 --max-epoch "$max_epoch" \
        --k-proportion "$k_proportion" --lambda-smooth "$lambda_smooth" \
        --hidden "$hidden" --embed "$embed" --dropout "$dropout" \
        --temperature "$temperature" --device cuda \
        > "$out/run.log" 2>&1
      "$python" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
        --corpus "$corpus" --split test --scores "$out/scores.jsonl" \
        --branch score_full --require-full-coverage --json-out "$out/metrics.json" \
        >> "$out/run.log" 2>&1
    fi
  done
done

"$python" "$repo/experiments/20260831_coalition_witness_candidate/summarize.py" \
  > "$run/summary.log" 2>&1
