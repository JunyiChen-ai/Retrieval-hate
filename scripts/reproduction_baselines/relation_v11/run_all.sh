#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$PWD/scripts/reproduction_baselines:$PWD/scripts/duplex"
python scripts/reproduction_baselines/relation_v11/synthetic_tests.py
mkdir -p results/reproduction/relation_v11
for c in hatemm mhclip_en mhclip_zh hateclipseg;do python scripts/reproduction_baselines/relation_v11/diagnostic.py --manifest results/reproduction/relation_v8/manifests/${c}_equal.json --out results/reproduction/relation_v11/${c}.json;done
