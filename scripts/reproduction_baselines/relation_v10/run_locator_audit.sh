#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PWD/scripts/reproduction_baselines:$PWD/scripts/duplex"
mkdir -p results/reproduction/relation_v10/locator_audit
for corpus in hatemm mhclip_en mhclip_zh hateclipseg; do
 python scripts/reproduction_baselines/relation_v10/locator_audit.py \
  --manifest "results/reproduction/relation_v8/manifests/${corpus}_equal.json" \
  --out "results/reproduction/relation_v10/locator_audit/${corpus}.json"
done
