#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
run=runs/20260905_interventional_evidence/ablations/hateclipseg/confirmation_chain
mkdir -p "$run"
echo "host=$(hostname) date=$(date -Is)"
echo "$$" > "$run/run.pid"
for seed in 2025 3407; do
  bash experiments/20260905_interventional_evidence/launch/run_module_ablations.sh hateclipseg "$seed"
done
"$HOME/miniconda3/envs/HateVideo/bin/python" -c 'import json,pathlib; pathlib.Path("runs/20260905_interventional_evidence/ablations/hateclipseg/confirmation_chain/completion.json").write_text(json.dumps({"state":"CONFIRMATION_ABLATIONS_FINISHED","expected_runs":16}))'
