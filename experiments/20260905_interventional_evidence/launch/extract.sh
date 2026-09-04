#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus=$1
run="runs/20260905_interventional_evidence/extract_${corpus}"
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is)"
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/analysis/extract_interventional_evidence.py --corpus "$corpus" --run-dir "$run"
