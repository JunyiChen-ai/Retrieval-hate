#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/jehc223/Retrieval-hate"
run_dir="$repo_dir/runs/20260831_word_aligned_lexical_premise/smoke"
python_bin="/home/jehc223/miniconda3/envs/HateVideo/bin/python"
model_name="openai/whisper-large-v3"
video_path="/home/jehc223/data/HateClipSeg/videos/bit_Y4NcS9xwARDO.mp4"

mkdir -p "$run_dir"
cp "$repo_dir/experiments/20260831_word_aligned_lexical_premise/README.md" "$run_dir/README.snapshot.md"
cp "$repo_dir/experiments/20260831_word_aligned_lexical_premise/smoke.py" "$run_dir/smoke.snapshot.py"

cd "$repo_dir"
setsid nohup env PYTHONPATH="$repo_dir" "$python_bin" \
  experiments/20260831_word_aligned_lexical_premise/smoke.py \
  --video "$video_path" \
  --model "$model_name" \
  --output-dir "$run_dir" \
  > "$run_dir/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run_dir/run.pid"
echo "started PID $pid; log: $run_dir/run.log"
