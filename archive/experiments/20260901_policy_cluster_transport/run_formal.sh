#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
HERE="$ROOT/archive/experiments/20260901_policy_cluster_transport"
RUN_DIR="$ROOT/runs/20260901_policy_cluster_transport/formal_seed234"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
TEACHER="$ROOT/results/reproduction/powa_macil/teacher_qwen2vl7b_train_2chunks.jsonl"

mkdir -p "$RUN_DIR/val_search"
cp "$HERE/formal_config.json" "$RUN_DIR/config.json"
cp "$HERE/code_version.txt" "$RUN_DIR/code_version.txt"

run_trial() {
  local corpus="$1"
  local arm="$2"
  local lr="$3"
  local weight="$4"
  local temperature="$5"
  local name="$6"
  local out="$RUN_DIR/val_search/$corpus/$name"
  local init="$ROOT/results/reproduction/official_val/final/macilsd/$corpus/seed_234/model.pth"
  mkdir -p "$out"
  "$PY" "$HERE/train.py" \
    --corpus "$corpus" --corpora "$corpus" --arm "$arm" \
    --out-dir "$out" --transport-weight "$weight" \
    --transport-temperature "$temperature" --min-harmful-mass .10 \
    --lr "$lr" --max-epoch 5 --batch-size 24 --max-seqlen 200 \
    --crop-repeat 5 --grid snippet --hid-dim 128 --ffn-dim 128 \
    --nhead 4 --dropout .1 --binding-window 12 \
    --binding-temperature .2 --sinkhorn-iters 8 \
    --base-loss-weight .25 --sparsity-weight .002 \
    --teacher-file "$TEACHER" --teacher-weight .05 \
    --macil-init "$init" --typed-only --device cuda --seed 234 \
    --num-workers 4 2>&1 | tee "$out/train.log"
}

for corpus in hatemm hateclipseg; do
  for lr in 0.0001 0.0002; do
    run_trial "$corpus" anchor "$lr" 0 .05 "anchor_lr${lr}"
  done
  for lr in 0.0001 0.0002; do
    for weight in .05 .2 .5; do
      for temperature in .05 .2; do
        run_trial "$corpus" policy "$lr" "$weight" "$temperature" \
          "policy_lr${lr}_w${weight}_t${temperature}"
      done
    done
  done
  "$PY" "$HERE/select_validation.py" \
    --corpus-dir "$RUN_DIR/val_search/$corpus" \
    --out "$RUN_DIR/val_search/$corpus/selection.json"
done

for corpus in hatemm hateclipseg; do
  selection="$RUN_DIR/val_search/$corpus/selection.json"
  lr=$(jq -r '.selected.lr' "$selection")
  weight=$(jq -r '.selected.transport_weight' "$selection")
  temperature=$(jq -r '.selected.transport_temperature' "$selection")
  run_trial "$corpus" binary "$lr" "$weight" "$temperature" "selected_binary"
  run_trial "$corpus" permuted "$lr" "$weight" "$temperature" "selected_permuted"
done

for corpus in hatemm hateclipseg; do
  selection="$RUN_DIR/val_search/$corpus/selection.json"
  policy_dir=$(jq -r '.selected.path' "$selection")
  anchor_dir=$(jq -r '.matched_anchor.path' "$selection")
  for arm in anchor binary permuted policy; do
    case "$arm" in
      anchor) checkpoint="$anchor_dir" ;;
      binary) checkpoint="$RUN_DIR/val_search/$corpus/selected_binary" ;;
      permuted) checkpoint="$RUN_DIR/val_search/$corpus/selected_permuted" ;;
      policy) checkpoint="$policy_dir" ;;
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
