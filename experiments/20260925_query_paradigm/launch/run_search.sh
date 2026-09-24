#!/usr/bin/env bash
# usage (repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/run_search.sh <corpus> <seed> [out_root] [extra_config_json] \
#            > runs/20260925_query_paradigm/launch_<corpus>_seed<seed>.out 2>&1 &
# README section 4: fixed Optuna search of one (corpus, seed). DONE / FAILED markers in <out_root>/<corpus>/seed<seed>/.
set -uo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seed="$2"; out_root="${3:-runs/20260925_query_paradigm}"; extra="${4:-}"
root="$out_root/$corpus/seed$seed"
mkdir -p "$root"
echo "$(date) $(hostname) search $corpus seed $seed extra=${extra:-none}"
args=(--corpus "$corpus" --seed "$seed" --out-root "$out_root")
[ -n "$extra" ] && args+=(--extra-config "$extra")
if "$HOME/miniconda3/envs/HateVideo/bin/python" experiments/20260925_query_paradigm/search.py "${args[@]}"; then
  echo "$(date) DONE" | tee "$root/search.DONE"
else
  echo "$(date) FAILED rc=$?" | tee "$root/search.FAILED"
fi
