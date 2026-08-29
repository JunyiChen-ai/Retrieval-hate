#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$repo_root/scripts/reproduction_baselines:$repo_root/scripts/duplex"
python scripts/reproduction_baselines/relation_v8/fair_eval_synthetic_test.py
declare -A baseline=(
  [hatemm]="macilsd_av"
  [mhclip_en]="macilsd_audio"
  [mhclip_zh]="fed_wsvad_1client"
  [hateclipseg]="vera"
)
for corpus in hatemm mhclip_en mhclip_zh hateclipseg; do
  python scripts/reproduction_baselines/relation_v8/fair_eval.py \
    --manifest "results/reproduction/relation_v8/manifests/${corpus}_equal.json" \
    --out-dir "results/reproduction/relation_v8/fair_eval/${corpus}" \
    --baseline-expert "${baseline[$corpus]}" \
    --selection-mode shared
done
