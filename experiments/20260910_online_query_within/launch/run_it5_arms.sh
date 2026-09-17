#!/usr/bin/env bash
# usage: bash experiments/20260910_online_query_within/launch/run_it5_arms.sh <corpus> "<seeds>" <tag>:<ablation>[:'{json}'] ...
# Iteration 5 diagnostic arms for several seeds in sequence on one GPU. The best-trial hparams file
# runs/20260910_online_query_within_it5/diag/<corpus>/seed<seed>/hparams_trial*.json must already exist
# (written by run_it5_chain.sh or copied from the main machine).
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seeds="$2"; shift 2
for seed in $seeds; do
  diag="runs/20260910_online_query_within_it5/diag/$corpus/seed$seed"
  hp=$(ls "$diag"/hparams_trial*.json | head -1)
  bash experiments/20260910_online_query_within/launch/run_diag.sh "$corpus" "$seed" "$hp" _it5 "$@"
done
