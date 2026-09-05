#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
run=runs/20260905_interventional_evidence/hateclipseg/seed234
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is)"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python_bin="$HOME/miniconda3/envs/HateVideo/bin/python"
"$python_bin" scripts/analysis/audit_interventional_inputs.py --corpus hateclipseg --require-complete --out "$run/input_audit.json"
exec "$python_bin" -u experiments/20260905_interventional_evidence/search.py --corpus hateclipseg --seed 234 --out-root runs/20260905_interventional_evidence
