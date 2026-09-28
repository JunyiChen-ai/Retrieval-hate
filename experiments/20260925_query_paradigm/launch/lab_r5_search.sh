#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5 && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_r5_search.sh <extra_config_json> <corpus> [corpus ...] \
#     > runs/20260929_query_paradigm_r5/launch_search_$(hostname).out 2>&1 < /dev/null &
# Revision 5 (README section 17.1): fixed Optuna search of every (corpus, seed) for the given corpora, the three seeds
# of one corpus concurrently, corpora one after another; launch/run_search.sh each (search.DONE / search.FAILED in
# <out_root>/<corpus>/seed<seed>/). Budget 20 trials, pre-written as in revision 4 (budget.json), so a slow first
# trial on a shared GPU cannot cut the search to 5 trials.
set -uo pipefail
cd "$HOME/Retrieval-hate"
OUT="${OUT_ROOT:-runs/20260929_query_paradigm_r5}"   # OUT_ROOT=<dir> for the ablation searches (soft_levels 4 / 16)
extra="$1"; shift
hostname; echo $$ > "$OUT/launch_search_$(hostname).pid"
for c in "$@"; do
  pids=()
  for s in 234 2025 3407; do
    mkdir -p "$OUT/$c/seed$s"
    [ -f "$OUT/$c/seed$s/budget.json" ] || echo '{"n_trials": 20, "first_trial_seconds": null}' > "$OUT/$c/seed$s/budget.json"
    bash experiments/20260925_query_paradigm/launch/run_search.sh "$c" "$s" "$OUT" "$extra" > "$OUT/launch_${c}_seed${s}.out" 2>&1 &
    pids+=($!)
  done
  echo "$(date) $c seeds started: ${pids[*]}"
  wait "${pids[@]}"
  echo "$(date) $c seeds finished"
done
echo "$(date) ALL_DONE"
