#!/usr/bin/env bash
set -euo pipefail

repo=/home/jehc223/Retrieval-hate
exp_id=20260831_powa_span_marginal_pilot
run_root="$repo/runs/$exp_id"
python_bin=/home/jehc223/miniconda3/envs/HateVideo/bin/python

anchor_for() {
    case "$1" in
        hatemm) printf '%s\n' "$repo/results/reproduction/powa_macil/final_maskfix_finetune_hatemm_seed234_e5" ;;
        hateclipseg) printf '%s\n' "$repo/runs/20260831_powa_starting_point/hcs_maskfix_seed234" ;;
        *) return 2 ;;
    esac
}

snapshot_source() {
    local run=$1
    mkdir -p "$run/source_snapshot/experiment" "$run/source_snapshot/shared" \
             "$run/source_snapshot/powa_macil" "$run/source_snapshot/macilsd" \
             "$run/source_snapshot/hate_common" "$run/source_snapshot/evaluator"
    cp "$repo/experiments/$exp_id/"{README.md,PILOT_PLAN.md,PRE_RUN_REVIEW.md,train.py,infer.py,complete_run.py,complete_infer.py,summarize_stage_v.py,run_stage_v.sh,test_span_marginal.py} "$run/source_snapshot/experiment/"
    cp "$repo/src/powa_residual.py" "$run/source_snapshot/shared/"
    cp "$repo/scripts/reproduction_baselines/powa_macil/"*.py "$run/source_snapshot/powa_macil/"
    cp "$repo/scripts/reproduction_baselines/macilsd/"*.py "$run/source_snapshot/macilsd/"
    cp "$repo/scripts/reproduction_baselines/hate_common/"*.py "$run/source_snapshot/hate_common/"
    cp "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" "$repo/scripts/duplex/frame_eval_common.py" "$run/source_snapshot/evaluator/"
    find "$run/source_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum > "$run/source_snapshot.sha256"
    git -C "$repo" rev-parse HEAD > "$run/code_commit.txt"
    git -C "$repo" diff --binary -- src/powa_residual.py "experiments/$exp_id" > "$run/tracked_code.patch"
}

run_one() {
    local corpus=$1
    local arm=$2
    local run="$run_root/${corpus}_${arm}_seed234"
    if [[ -e "$run" ]]; then
        printf 'ABORT stale run %s\n' "$run" >&2
        return 3
    fi
    mkdir -p "$run"
    snapshot_source "$run"
    printf '%s\n' "$$" > "$run/run.pid"
    "$python_bin" "$repo/experiments/$exp_id/train.py" \
        --corpus "$corpus" --anchor-checkpoint "$(anchor_for "$corpus")" \
        --out-dir "$run" --arm "$arm" --device cuda --seed 234 \
        --epochs 5 --batch-size 24 --lr .0002 --weight-decay .0001 \
        --temperature .5 --negative-dense-weight 1 --residual-l2-weight .01 \
        --pooled-tolerance .002 --num-workers 4 >> "$run/run.log" 2>&1
    if [[ -f "$run/val_scores.jsonl" ]]; then
        "$python_bin" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
            --corpus "$corpus" --scores "$run/val_scores.jsonl" --split val \
            --json-out "$run/metrics.json" --require-full-coverage >> "$run/run.log" 2>&1
    fi
    "$python_bin" "$repo/experiments/$exp_id/complete_run.py" --run-dir "$run" >> "$run/run.log" 2>&1
}

infer_one() {
    local corpus=$1
    local train_run="$run_root/${corpus}_span_marginal_seed234"
    local run="$run_root/test_${corpus}_seed234"
    if [[ -e "$run" ]]; then
        printf 'ABORT stale inference %s\n' "$run" >&2
        return 3
    fi
    mkdir -p "$run"
    cp -a "$train_run/source_snapshot" "$run/source_snapshot"
    cp "$train_run/source_snapshot.sha256" "$run/source_snapshot.sha256"
    cp "$train_run/code_commit.txt" "$run/code_commit.txt"
    cp "$train_run/tracked_code.patch" "$run/tracked_code.patch"
    printf '%s\n' "$$" > "$run/run.pid"
    "$python_bin" "$repo/experiments/$exp_id/infer.py" \
        --corpus "$corpus" --anchor-checkpoint "$(anchor_for "$corpus")" \
        --train-run "$train_run" \
        --authorization "$run_root/stage_v_summary.json" \
        --out-dir "$run" --split test --device cuda --num-workers 4 \
        >> "$run/run.log" 2>&1
    "$python_bin" "$repo/scripts/reproduction_baselines/eval_baseline_scores.py" \
        --corpus "$corpus" --scores "$run/test_scores.jsonl" --split test \
        --json-out "$run/metrics.json" --require-full-coverage \
        >> "$run/run.log" 2>&1
    "$python_bin" "$repo/experiments/$exp_id/complete_infer.py" \
        --run-dir "$run" >> "$run/run.log" 2>&1
}

mkdir -p "$run_root"
for corpus in hatemm hateclipseg; do
    for arm in span_marginal singleton shuffled_span; do
        run_one "$corpus" "$arm"
    done
done

if "$python_bin" "$repo/experiments/$exp_id/summarize_stage_v.py" \
    --run-root "$run_root" --out "$run_root/stage_v_summary.json" \
    > "$run_root/stage_v_supervisor_summary.log" 2>&1; then
    printf '%s\n' ADVANCE_TO_STAGE_P
    infer_one hatemm
    infer_one hateclipseg
else
    printf '%s\n' KILL_BEFORE_TEST
fi
