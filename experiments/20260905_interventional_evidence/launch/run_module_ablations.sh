#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
exec bash scripts/run_locked_ablations.sh 20260905_interventional_evidence "$1" "$2" raw_verdict ordinary_attention additive_fusion full_input_only four_logits no_interaction dempster_fusion no_block
