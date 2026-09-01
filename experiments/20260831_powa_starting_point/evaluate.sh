#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
archive=/home/jehc223/Hate-follow-up/results/reproduction/powa_macil
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
out_root="$repo/runs/20260831_powa_starting_point"
evaluator="$repo/scripts/reproduction_baselines/eval_baseline_scores.py"

evaluate_one() {
    local corpus="$1"
    local seed="$2"
    local variant="$3"
    local source_dir="$archive/$variant"
    local run_dir="$out_root/${corpus}_seed${seed}"

    mkdir -p "$run_dir"
    printf '%s\n' "$$" > "$run_dir/run.pid"
    git -C "$repo" rev-parse HEAD > "$run_dir/code_commit.txt"
    cp "$source_dir/train_meta.json" "$run_dir/config.json"
    printf '%s\n' "$source_dir/$corpus/scores.jsonl" > "$run_dir/scores_source.txt"
    "$python_bin" "$evaluator" \
        --corpus "$corpus" \
        --scores "$source_dir/$corpus/scores.jsonl" \
        --branch score_powa \
        --require-full-coverage \
        --json-out "$run_dir/metrics.json" \
        > "$run_dir/run.log" 2>&1
}

for seed in 234 2025 3407; do
    evaluate_one hatemm "$seed" "final_maskfix_finetune_hatemm_seed${seed}_e5"
    evaluate_one mhclip_en "$seed" "final_maskfix_finetune_mhclip_en_seed${seed}_e5"
    evaluate_one mhclip_zh "$seed" "final_maskfix_frozen_positive_mhclip_zh_seed${seed}_e5"
    evaluate_one hateclipseg "$seed" "final5crop_teacher005_finetune_hateclipseg_seed${seed}_e5"
done

"$python_bin" "$repo/experiments/20260831_powa_starting_point/summarize.py" \
    --root "$out_root" \
    --out "$out_root/summary.json"

