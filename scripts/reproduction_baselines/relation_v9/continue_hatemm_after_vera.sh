#!/usr/bin/env bash
set -euo pipefail
pid="${1:?VERA producer PID required}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines:$PWD/scripts/duplex"
while kill -0 "$pid" 2>/dev/null; do
  count="$(find results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer/raw -maxdepth 1 -name '*.json' | wc -l)"
  printf '%s VERA raw %s/744 pid=%s\n' "$(date --iso-8601=seconds)" "$count" "$pid"
  sleep 60
done
scores=results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer/scores.jsonl
if [[ ! -s "$scores" ]]; then
  echo "VERA process ended without aggregate scores" >&2
  exit 2
fi
python scripts/reproduction_baselines/relation_v9/finalize_hatemm_vera_train.py
python scripts/reproduction_baselines/relation_v9/hatemm_preflight.py --require-complete
python scripts/reproduction_baselines/relation_v9/hatemm_weak_residual_pilot.py \
  --out-dir results/reproduction/relation_v9/hatemm_weak_residual_seed234 \
  --device cuda --seed 234
