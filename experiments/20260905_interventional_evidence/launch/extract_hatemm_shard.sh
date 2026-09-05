#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
shard=$1
[[ "$shard" == 0 || "$shard" == 1 ]]
run="runs/20260905_interventional_evidence/extract_hatemm_v2_shard${shard}"
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is) shard=$shard/2"
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/analysis/extract_interventional_evidence.py --corpus hatemm --shards 2 --shard "$shard" --run-dir "$run"
