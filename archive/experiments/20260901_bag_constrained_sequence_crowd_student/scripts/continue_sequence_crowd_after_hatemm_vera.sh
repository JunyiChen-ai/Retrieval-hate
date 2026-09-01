#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=/home/jehc223/Retrieval-hate
METHOD_DIR="$PROJECT_ROOT/experiments/20260901_bag_constrained_sequence_crowd_student"
FORMAL_DIR="$PROJECT_ROOT/runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234"
VERA_DIR="$PROJECT_ROOT/runs/20260901_sequence_crowd_teacher_export/hatemm_vera_train"
PYTHON_BIN=/home/jehc223/miniconda3/envs/HateVideo/bin/python
VERA_PID=$(tr -d '[:space:]' < "$VERA_DIR/run.pid")

while kill -0 "$VERA_PID" 2>/dev/null; do
  printf 'Waiting for HateMM VERA producer PID %s.\n' "$VERA_PID"
  sleep 30
done

test -s "$VERA_DIR/scores.jsonl"
test -s "$VERA_DIR/PROVENANCE.json"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_sequence_crowd_targets.py" \
  --corpus hatemm \
  --lexical-npz "$PROJECT_ROOT/runs/20260831_lexical_posterior_regularization/stage_a_fix2/evidence/hatemm/train_evidence.npz" \
  --powa "$PROJECT_ROOT/runs/20260901_sequence_crowd_teacher_export/hatemm_powa_train/scores.jsonl" \
  --vera "$VERA_DIR/scores.jsonl" \
  --multihateloc "$PROJECT_ROOT/runs/20260901_sequence_crowd_teacher_export/hatemm_multihateloc_train/scores.jsonl" \
  --out-root "$PROJECT_ROOT/data/sequence_crowd_targets" \
  --n-bins 5 --iterations 20

TARGET="$PROJECT_ROOT/data/sequence_crowd_targets/hatemm"
run_trial() {
  local arm="$1" lr="$2" weight="$3" name="$4"
  local out="$FORMAL_DIR/val_search/hatemm/$name"
  mkdir -p "$out"
  "$PYTHON_BIN" "$METHOD_DIR/train.py" --corpus hatemm --arm "$arm" \
    --targets "$TARGET/$arm.npz" --out-dir "$out" --lr "$lr" \
    --bag-weight "$weight" --epochs 30 --batch-size 8 --width 128 \
    --dropout .1 --workers 4 --seed 234 --device cuda \
    --pooled-tolerance .01 2>&1 | tee "$out/train.log"
}

for lr in 0.00003 0.0001 0.0003; do
  for weight in .25 1.0; do
    run_trial core "$lr" "$weight" "core_lr${lr}_bw${weight}"
  done
done

"$PYTHON_BIN" "$METHOD_DIR/select_validation.py" \
  --corpus-dir "$FORMAL_DIR/val_search/hatemm" --expected-trials 6 \
  --pooled-tolerance .01 \
  --out "$FORMAL_DIR/val_search/hatemm/selection.json"

selection="$FORMAL_DIR/val_search/hatemm/selection.json"
lr=$(jq -r '.selected.lr' "$selection")
weight=$(jq -r '.selected.bag_weight' "$selection")
run_trial token_ds "$lr" "$weight" selected_token_ds
run_trial unconstrained_bsc "$lr" "$weight" selected_unconstrained_bsc

for arm in core token_ds unconstrained_bsc; do
  case "$arm" in
    core) checkpoint=$(jq -r '.selected.path' "$selection") ;;
    token_ds) checkpoint="$FORMAL_DIR/val_search/hatemm/selected_token_ds" ;;
    unconstrained_bsc) checkpoint="$FORMAL_DIR/val_search/hatemm/selected_unconstrained_bsc" ;;
  esac
  test_dir="$FORMAL_DIR/test/hatemm/$arm"
  mkdir -p "$test_dir"
  "$PYTHON_BIN" "$METHOD_DIR/infer.py" --checkpoint-dir "$checkpoint" \
    --corpus hatemm --split test --device cuda \
    --out "$test_dir/scores.jsonl"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/reproduction_baselines/eval_baseline_scores.py" \
    --corpus hatemm --split test --scores "$test_dir/scores.jsonl" \
    --branch score_method --require-full-coverage \
    --json-out "$test_dir/metrics.json"
done

printf 'complete\n' > "$FORMAL_DIR/hatemm.complete"
"$PYTHON_BIN" "$METHOD_DIR/summarize.py" --run-dir "$FORMAL_DIR" \
  --out "$FORMAL_DIR/summary.json"
