#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
out_root="$repo/runs/20260831_powa_starting_point"
teacher="$repo/results/reproduction/powa_macil/teacher_qwen2vl7b_train_2chunks.jsonl"
evaluator="$repo/scripts/reproduction_baselines/eval_baseline_scores.py"

for seed in 234 2025 3407; do
    run_dir="$out_root/hcs_maskfix_seed${seed}"
    if [[ -f "$run_dir/metrics.json" ]]; then
        continue
    fi
    mkdir -p "$run_dir"
    printf '%s\n' "$$" > "$run_dir/run.pid"
    git -C "$repo" rev-parse HEAD > "$run_dir/code_commit.txt"
    {
        "$python_bin" "$repo/scripts/reproduction_baselines/powa_macil/train.py" \
            --corpora hateclipseg \
            --out-dir "$run_dir" \
            --seed "$seed" \
            --device cuda \
            --num-workers 4 \
            --max-epoch 5 \
            --batch-size 24 \
            --lr 0.0002 \
            --max-seqlen 200 \
            --crop-repeat 5 \
            --grid snippet \
            --teacher-file "$teacher" \
            --teacher-weight 0.05 \
            --macil-init "$repo/results/reproduction/official_val/final/macilsd/hateclipseg/seed_${seed}/model.pth" \
            --typed-only \
            --selection mean_frame_ap
        "$python_bin" "$repo/scripts/reproduction_baselines/powa_macil/infer.py" \
            --checkpoint-dir "$run_dir" \
            --corpus hateclipseg \
            --split test \
            --device cuda \
            --out "$run_dir/scores.jsonl"
        "$python_bin" "$evaluator" \
            --corpus hateclipseg \
            --scores "$run_dir/scores.jsonl" \
            --branch score_powa \
            --require-full-coverage \
            --json-out "$run_dir/metrics.json"
    } > "$run_dir/run.log" 2>&1
done
