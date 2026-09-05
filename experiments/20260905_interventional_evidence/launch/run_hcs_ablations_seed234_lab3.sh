#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
exec bash experiments/20260905_interventional_evidence/launch/run_module_ablations.sh hateclipseg 234
