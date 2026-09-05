#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
exec bash experiments/20260905_interventional_evidence/launch/run_search.sh hateclipseg 234
