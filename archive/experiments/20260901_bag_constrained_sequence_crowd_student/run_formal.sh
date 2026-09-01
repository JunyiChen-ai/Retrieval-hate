#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
HERE="$ROOT/experiments/20260901_bag_constrained_sequence_crowd_student"
RUN_DIR="$ROOT/runs/20260901_bag_constrained_sequence_crowd_student/formal_seed234"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
TARGET_ROOT="$ROOT/data/sequence_crowd_targets"

mkdir -p "$RUN_DIR/val_search" "$RUN_DIR/test"
cp "$HERE/formal_config.json" "$RUN_DIR/config.json"
mkdir -p "$RUN_DIR/source_snapshot"
cp "$HERE"/*.py "$HERE"/*.sh "$HERE"/*.json "$HERE"/*.md \
  "$RUN_DIR/source_snapshot/"
printf '%s\n' "$$" > "$RUN_DIR/run.pid"

train_trial() {
  local corpus="$1"
  local arm="$2"
  local lr="$3"
  local weight="$4"
  local name="$5"
  local out="$RUN_DIR/val_search/$corpus/$name"
  mkdir -p "$out"
  "$PY" "$HERE/train.py" \
    --corpus "$corpus" --arm "$arm" \
    --targets "$TARGET_ROOT/$corpus/$arm.npz" --out-dir "$out" \
    --lr "$lr" --bag-weight "$weight" --epochs 30 --batch-size 8 \
    --width 128 --dropout .1 --workers 4 --seed 234 --device cuda \
    --pooled-tolerance .01 2>&1 | tee "$out/train.log"
}

for corpus in hatemm hateclipseg; do
  for lr in 0.00003 0.0001 0.0003; do
    for weight in .25 1.0; do
      train_trial "$corpus" core "$lr" "$weight" "core_lr${lr}_bw${weight}"
    done
  done
  "$PY" "$HERE/select_validation.py" \
    --corpus-dir "$RUN_DIR/val_search/$corpus" \
    --expected-trials 6 --pooled-tolerance .01 \
    --out "$RUN_DIR/val_search/$corpus/selection.json"
done

for corpus in hatemm hateclipseg; do
  selection="$RUN_DIR/val_search/$corpus/selection.json"
  lr=$(jq -r '.selected.lr' "$selection")
  weight=$(jq -r '.selected.bag_weight' "$selection")
  train_trial "$corpus" token_ds "$lr" "$weight" "selected_token_ds"
  train_trial "$corpus" unconstrained_bsc "$lr" "$weight" "selected_unconstrained_bsc"
done

for corpus in hatemm hateclipseg; do
  selection="$RUN_DIR/val_search/$corpus/selection.json"
  for arm in core token_ds unconstrained_bsc; do
    case "$arm" in
      core) checkpoint=$(jq -r '.selected.path' "$selection") ;;
      token_ds) checkpoint="$RUN_DIR/val_search/$corpus/selected_token_ds" ;;
      unconstrained_bsc) checkpoint="$RUN_DIR/val_search/$corpus/selected_unconstrained_bsc" ;;
    esac
    test_dir="$RUN_DIR/test/$corpus/$arm"
    mkdir -p "$test_dir"
    "$PY" "$HERE/infer.py" --checkpoint-dir "$checkpoint" \
      --corpus "$corpus" --split test --device cuda \
      --out "$test_dir/scores.jsonl"
    "$PY" "$ROOT/scripts/reproduction_baselines/eval_baseline_scores.py" \
      --corpus "$corpus" --split test --scores "$test_dir/scores.jsonl" \
      --branch score_method --require-full-coverage \
      --json-out "$test_dir/metrics.json"
  done
done

"$PY" "$HERE/summarize.py" --run-dir "$RUN_DIR" \
  --out "$RUN_DIR/summary.json"
