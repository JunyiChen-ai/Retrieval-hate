#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
export ABLATION_CHAIN_NAME=auxiliary_chain
exec bash scripts/run_locked_ablations.sh 20260906_interval_evidence_transport hateclipseg 234 no_observation_loss categorical_noise
