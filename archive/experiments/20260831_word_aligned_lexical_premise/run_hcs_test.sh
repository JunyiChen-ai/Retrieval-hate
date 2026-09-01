#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/jehc223/Retrieval-hate"
run_dir="$repo_dir/runs/20260831_word_aligned_lexical_premise/hateclipseg_test"
python_bin="/home/jehc223/miniconda3/envs/HateVideo/bin/python"
model_name="openai/whisper-large-v3"
mkdir -p "$run_dir"
for pair in \
  "README.md:README.snapshot.md" \
  "generate_word_asr.py:generate_word_asr.snapshot.py" \
  "evaluate_corpus.py:evaluate_corpus.snapshot.py"; do
  source_name="${pair%%:*}"
  snapshot_name="${pair##*:}"
  source_path="$repo_dir/experiments/20260831_word_aligned_lexical_premise/$source_name"
  snapshot_path="$run_dir/$snapshot_name"
  if [[ -e "$snapshot_path" ]]; then
    cmp -s "$source_path" "$snapshot_path" || {
      echo "refusing resume: source differs from existing readable snapshot" >&2
      exit 2
    }
  else
    cp "$source_path" "$snapshot_path"
  fi
done
cd "$repo_dir"
setsid nohup env PYTHONPATH="$repo_dir" bash -c '
  set -euo pipefail
  "$1" experiments/20260831_word_aligned_lexical_premise/generate_word_asr.py \
    --corpus hateclipseg --split test --model "$2" \
    --output "$3/word_asr.jsonl"
  "$1" experiments/20260831_word_aligned_lexical_premise/evaluate_corpus.py \
    --corpus hateclipseg --word-asr "$3/word_asr.jsonl" --run-dir "$3"
' worker "$python_bin" "$model_name" "$run_dir" > "$run_dir/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run_dir/run.pid"
echo "started PID $pid; log: $run_dir/run.log"
