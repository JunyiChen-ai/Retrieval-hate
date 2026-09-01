#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp_id=20260831_powa_rank_transport_pilot
run_root="$repo/runs/$exp_id"
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python
seed=234

anchor_for() {
    case "$1" in
        hatemm)
            printf '%s\n' "$repo/results/reproduction/powa_macil/final_maskfix_finetune_hatemm_seed234_e5"
            ;;
        hateclipseg)
            printf '%s\n' "$repo/runs/20260831_powa_starting_point/hcs_maskfix_seed234"
            ;;
        *)
            return 2
            ;;
    esac
}

snapshot_source() {
    local run_dir=$1
    mkdir -p "$run_dir/source_snapshot/experiment" \
             "$run_dir/source_snapshot/shared" \
             "$run_dir/source_snapshot/powa_macil" \
             "$run_dir/source_snapshot/macilsd" \
             "$run_dir/source_snapshot/evaluator"
    cp "$repo/experiments/$exp_id/README.md" \
       "$repo/experiments/$exp_id/PILOT_PLAN.md" \
       "$repo/experiments/$exp_id/NOVELTY_SCOUT.md" \
       "$repo/experiments/$exp_id/model.py" \
       "$repo/experiments/$exp_id/train.py" \
       "$repo/experiments/$exp_id/infer.py" \
       "$repo/experiments/$exp_id/complete_run.py" \
       "$repo/experiments/$exp_id/audit_interventions.py" \
       "$repo/experiments/$exp_id/summarize_stage_v.py" \
       "$repo/experiments/$exp_id/run_stage_v.sh" \
       "$repo/experiments/$exp_id/test_rank_transport.py" \
       "$repo/experiments/$exp_id/PRE_RUN_REVIEW.md" \
       "$run_dir/source_snapshot/experiment/"
    cp "$repo/src/weak_supervision/__init__.py" \
       "$repo/src/weak_supervision/same_corpus_insertion.py" \
       "$run_dir/source_snapshot/shared/"
    cp "$repo/scripts/reproduction_baselines/powa_macil/model.py" \
       "$repo/scripts/reproduction_baselines/powa_macil/dataset.py" \
       "$run_dir/source_snapshot/powa_macil/"
    cp "$repo/scripts/reproduction_baselines/macilsd/Transformer.py" \
       "$repo/scripts/reproduction_baselines/macilsd/avce_network.py" \
       "$run_dir/source_snapshot/macilsd/"
    cp "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
       "$repo/scripts/duplex/frame_eval_common.py" \
       "$run_dir/source_snapshot/evaluator/"
    find "$run_dir/source_snapshot" -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum > "$run_dir/source_snapshot.sha256"
    git -C "$repo" rev-parse HEAD > "$run_dir/code_commit.txt"
    git -C "$repo" diff --binary -- \
        scripts/reproduction_baselines/macilsd/Transformer.py \
        scripts/reproduction_baselines/macilsd/avce_network.py \
        scripts/reproduction_baselines/powa_macil/model.py \
        scripts/reproduction_baselines/powa_macil/train.py \
        src/weak_supervision \
        "experiments/$exp_id" > "$run_dir/tracked_code.patch"
}

run_one() {
    local corpus=$1
    local arm=$2
    local run_dir="$run_root/${corpus}_${arm}_seed${seed}"
    if [[ -e "$run_dir/train_meta.json" || -e "$run_dir/completion.json" ]]; then
        if [[ -f "$run_dir/train_meta.json" && -f "$run_dir/completion.json" ]]; then
            printf 'SKIP recorded run; summary will verify %s\n' "$run_dir"
            return
        fi
        printf 'ABORT partial or stale run directory %s\n' "$run_dir" >&2
        return 3
    fi
    mkdir -p "$run_dir"
    snapshot_source "$run_dir"
    printf '%s\n' "$$" > "$run_dir/run.pid"
    "$python_bin" "$repo/experiments/$exp_id/train.py" \
        --corpus "$corpus" \
        --anchor-checkpoint "$(anchor_for "$corpus")" \
        --out-dir "$run_dir" \
        --arm "$arm" \
        --device cuda \
        --seed "$seed" \
        --num-workers 4 \
        --epochs 5 \
        --batch-size 24 \
        --lr 0.0002 \
        --weight-decay 0.0001 \
        --margin 1.0 \
        --stability-weight 0.5 \
        --topk-divisor 16 \
        --min-donor-rows 12 \
        --max-donor-rows 36 \
        --boundary-buffer 3 \
        --pooled-tolerance 0.002 \
        >> "$run_dir/run.log" 2>&1
    if [[ -f "$run_dir/val_scores.jsonl" ]]; then
        "$python_bin" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
            --corpus "$corpus" \
            --scores "$run_dir/val_scores.jsonl" \
            --split val \
            --json-out "$run_dir/metrics.json" \
            --require-full-coverage \
            >> "$run_dir/run.log" 2>&1
    fi
    "$python_bin" "$repo/experiments/$exp_id/complete_run.py" \
        --run-dir "$run_dir" >> "$run_dir/run.log" 2>&1
}

mkdir -p "$run_root"
for corpus in hatemm hateclipseg; do
    for arm in negative_donor positive_donor shifted_mask; do
        run_one "$corpus" "$arm"
    done
done

if "$python_bin" "$repo/experiments/$exp_id/summarize_stage_v.py" \
    --run-root "$run_root" \
    --out "$run_root/stage_v_summary.json" \
    > "$run_root/stage_v_supervisor_summary.log" 2>&1; then
    printf '%s\n' 'ADVANCE_TO_STAGE_P'
else
    printf '%s\n' 'KILL_BEFORE_TEST'
fi
