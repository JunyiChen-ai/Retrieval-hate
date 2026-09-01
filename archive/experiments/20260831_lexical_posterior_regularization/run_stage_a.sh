#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
RUN_ROOT="$ROOT/runs/20260831_lexical_posterior_regularization/stage_a_fix2"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260831_lexical_posterior_regularization"
if [ -e "$RUN_ROOT" ]; then
  echo "ABORT: formal run directory already exists; refusing mixed or stale artifacts" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/run.pid"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

"$PY" "$EXP/prepare_evidence.py" --out-dir "$RUN_ROOT/evidence"
for corpus in hatemm hateclipseg; do
  for arm in anchor core; do
    out_dir="$RUN_ROOT/$corpus/$arm"
    "$PY" "$EXP/train.py" --corpus "$corpus" --arm "$arm" \
      --evidence-dir "$RUN_ROOT/evidence" --out-dir "$out_dir"
    "$PY" "$EXP/evaluate.py" --corpus "$corpus" --arm "$arm" \
      --run-dir "$out_dir"
  done
done
"$PY" "$EXP/summarize.py" --run-root "$RUN_ROOT"
