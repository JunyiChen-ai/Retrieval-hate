#!/usr/bin/env bash
# usage: bash experiments/20260910_online_query_within/launch/run_it5_chain.sh <corpus> <seed> <tag>:<ablation>[:'{json}'] ...
# Iteration 5, one corpus/seed on one GPU: (1) run or resume the search (run_it5.sh; budget.json in the
# study directory governs how many trials remain), (2) copy the best trial's hparams into
# runs/20260910_online_query_within_it5/diag/<corpus>/seed<seed>/, (3) run the given diagnostic arms
# with run_diag.sh. Chain log: runs/20260910_online_query_within_it5/<corpus>/seed<seed>/chain.log
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seed="$2"; shift 2
exp=experiments/20260910_online_query_within
root="runs/20260910_online_query_within_it5/$corpus/seed$seed"
mkdir -p "$root"
echo "$(date) $(hostname) search start" >> "$root/chain.log"
bash "$exp/launch/run_it5.sh" "$corpus" "$seed" >> "$root/search.out" 2>&1 || echo "$(date) search FAILED" >> "$root/chain.log"
b=$("$HOME/miniconda3/envs/HateVideo/bin/python" -c "import json;print(json.load(open('$root/study_summary.json'))['best']['number'])")
echo "$(date) search done, best trial $b" >> "$root/chain.log"
diag="runs/20260910_online_query_within_it5/diag/$corpus/seed$seed"
mkdir -p "$diag"
cp "$root/trial$b/hparams.json" "$diag/hparams_trial$b.json"
if [ "$#" -gt 0 ]; then
  bash "$exp/launch/run_diag.sh" "$corpus" "$seed" "$diag/hparams_trial$b.json" _it5 "$@"
fi
echo "$(date) DONE" >> "$root/chain.log"
