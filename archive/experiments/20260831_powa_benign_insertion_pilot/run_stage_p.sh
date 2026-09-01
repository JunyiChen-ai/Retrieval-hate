#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
exp_id=20260831_powa_benign_insertion_pilot
out_root="$repo/runs/$exp_id"
teacher="$repo/results/reproduction/powa_macil/teacher_qwen2vl7b_train_2chunks.jsonl"
evaluator="$repo/scripts/reproduction_baselines/eval_baseline_scores.py"

run_one() {
    local corpus="$1"
    local arm="$2"
    local seed=234
    local run_name="${corpus}_${arm}_seed${seed}"
    local run_dir="$out_root/$run_name"
    if [[ -f "$run_dir/metrics.json" ]]; then
        return
    fi
    mkdir -p "$run_dir"
    printf '%s\n' "$$" > "$run_dir/run.pid"
    git -C "$repo" rev-parse HEAD > "$run_dir/code_commit.txt"
    mkdir -p "$run_dir/source_snapshot"
    cp "$repo/experiments/$exp_id/"*.py \
       "$repo/experiments/$exp_id/"*.sh \
       "$repo/experiments/$exp_id/"*.md \
       "$run_dir/source_snapshot/"
    mkdir -p "$run_dir/source_snapshot/macilsd" \
             "$run_dir/source_snapshot/powa_macil"
    cp "$repo/scripts/reproduction_baselines/macilsd/Transformer.py" \
       "$repo/scripts/reproduction_baselines/macilsd/avce_network.py" \
       "$run_dir/source_snapshot/macilsd/"
    cp "$repo/scripts/reproduction_baselines/powa_macil/model.py" \
       "$repo/scripts/reproduction_baselines/powa_macil/train.py" \
       "$run_dir/source_snapshot/powa_macil/"
    find "$run_dir/source_snapshot" -type f -print0 | sort -z | \
        xargs -0 sha256sum > "$run_dir/source_snapshot.sha256"
    git -C "$repo" diff -- \
        scripts/reproduction_baselines/macilsd/Transformer.py \
        scripts/reproduction_baselines/macilsd/avce_network.py \
        scripts/reproduction_baselines/powa_macil/model.py \
        scripts/reproduction_baselines/powa_macil/train.py \
        > "$run_dir/tracked_code.patch"
    sha256sum "$repo/experiments/$exp_id/PILOT_PLAN.md" \
        > "$run_dir/pilot_plan.sha256"
    {
        "$python_bin" "$repo/experiments/$exp_id/train.py" \
            --corpora "$corpus" \
            --arm "$arm" \
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
            --macil-init "$repo/results/reproduction/official_val/final/macilsd/$corpus/seed_${seed}/model.pth" \
            --typed-only \
            --min-donor-rows 12 \
            --max-donor-rows 36 \
            --boundary-buffer 3 \
            --donor-bce-weight 1.0 \
            --composite-mil-weight 1.0 \
            --consistency-weight 0.5
        "$python_bin" "$repo/scripts/reproduction_baselines/powa_macil/infer.py" \
            --checkpoint-dir "$run_dir" \
            --corpus "$corpus" \
            --split test \
            --device cuda \
            --out "$run_dir/scores.jsonl"
        "$python_bin" "$evaluator" \
            --corpus "$corpus" \
            --scores "$run_dir/scores.jsonl" \
            --branch score_powa \
            --require-full-coverage \
            --json-out "$run_dir/metrics.json"
    } > "$run_dir/run.log" 2>&1
}

mkdir -p "$out_root"
for corpus in hatemm hateclipseg; do
    run_one "$corpus" matched_powa
    run_one "$corpus" full
done

"$python_bin" "$repo/experiments/$exp_id/summarize_stage_p.py" \
    --root "$out_root" \
    --out "$out_root/stage_p_summary.json"
