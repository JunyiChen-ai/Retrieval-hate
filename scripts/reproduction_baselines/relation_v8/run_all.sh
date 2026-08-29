#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$repo_root/scripts/reproduction_baselines"
python scripts/reproduction_baselines/relation_v8/smoke.py
for corpus in hatemm mhclip_en mhclip_zh hateclipseg; do
  manifest="results/reproduction/relation_v8/manifests/${corpus}_equal.json"
  out="results/reproduction/relation_v8/unified/${corpus}"
  python scripts/reproduction_baselines/relation_v8/run.py --manifest "$manifest" --out-dir "$out"
done
