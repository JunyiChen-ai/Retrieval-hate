#!/usr/bin/env bash
# usage (repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/run_ablations_r4.sh <corpus> <seed> [parallel] \
#            > runs/20260927_query_paradigm_r4/ablations/launch_<corpus>_seed<seed>.out 2>&1 &
# README section 15.2 ablations of revision 4: every arm trains once with the best trial's hyperparameters of the
# revision-4 search of this (corpus, seed) plus the search's fixed settings (answer_source words, node_prior true),
# no search. Arms that switch to a model form without node potentials (label-only training, independent prior,
# coupled chain) also set node_prior false (the node potentials need the answer likelihood and the zero-inflated
# chain). Output: runs/20260927_query_paradigm_r4/ablations/<corpus>/seed<seed>/<arm>/ ; DONE / FAILED in the log.
set -uo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seed="$2"; par="${3:-2}"
PY="$HOME/miniconda3/envs/HateVideo/bin/python"
study="runs/20260927_query_paradigm_r4/$corpus/seed$seed/study_summary.json"
[ -f "$study" ] || { echo "$(date) FAILED no study summary $study"; exit 1; }
root="runs/20260927_query_paradigm_r4/ablations/$corpus/seed$seed"
mkdir -p "$root"
arms=(
  'full_rerun={}'
  'no_node={"node_prior": false}'
  'k30={"answer_source": "k30"}'
  'a_hate_only={"categories": [0]}'
  'b_flat={"fusion": "flat"}'
  'c_label={"objective": "label", "node_prior": false}'
  'd_bfs={"order": "bfs"}'
  'e_independent={"prior": "independent", "node_prior": false}'
  'f_joint={"answer_model": "joint"}'
  'g_coupled={"chain_form": "coupled", "node_prior": false}'
)
run_arm() {
  local tag="${1%%=*}" arm="${1#*=}" out="$root/${1%%=*}"
  mkdir -p "$out"
  "$PY" -c "import json,sys; s=json.load(open('$study')); b=dict(s['extra']); b.update(s['best']['params']); b.update(json.loads(sys.argv[1])); json.dump(b, open('$out/hparams.json','w'), indent=2)" "$arm"
  echo "$(date) $(hostname) start $corpus seed $seed $tag $(tr -d '\n ' < "$out/hparams.json")"
  if "$PY" experiments/20260925_query_paradigm/train.py --corpus "$corpus" --seed "$seed" --out-dir "$out" \
       --config "$out/hparams.json" > "$out/stdout.log" 2>&1; then
    echo "$(date) DONE $tag: $(grep -E '^fixed  8' "$out/run.log")"
  else
    echo "$(date) FAILED $tag"
  fi
}
i=0
for spec in "${arms[@]}"; do
  run_arm "$spec" &
  i=$((i + 1))
  if [ $((i % par)) -eq 0 ]; then wait; fi
done
wait
echo "$(date) ALL DONE"
