#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/jehc223/Retrieval-hate"
run_dir="$repo_dir/runs/20260831_universal_teacher_simplex_diagnostic/main"
python_bin="/home/jehc223/miniconda3/envs/HateVideo/bin/python"
if [[ -e "$run_dir" ]]; then
  echo "refusing non-fresh run directory: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
cp "$repo_dir/experiments/20260831_universal_teacher_simplex_diagnostic/README.md" "$run_dir/README.snapshot.md"
cp "$repo_dir/experiments/20260831_universal_teacher_simplex_diagnostic/analyze.py" "$run_dir/analyze.snapshot.py"
cp "$repo_dir/src/score_diagnostics.py" "$run_dir/score_diagnostics.snapshot.py"
printf '%s\n' 'universal-teacher-simplex-v1' > "$run_dir/code_version.txt"
cd "$repo_dir"
setsid nohup env PYTHONPATH="$repo_dir/scripts/reproduction_baselines" \
  "$python_bin" experiments/20260831_universal_teacher_simplex_diagnostic/analyze.py \
  > "$run_dir/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run_dir/run.pid"
echo "started PID $pid; log: $run_dir/run.log"
