#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
exec bash scripts/run_locked_ablations.sh 20260905_latent_evidence_sequence "$1" "$2" diagonal_emission static_transition event_to_topk full_input_emission raw_verdict no_temporal_content independent_state no_observation_likelihood
