#!/usr/bin/env bash
# usage: bash experiments/20260910_online_query_within/launch/run_diag.sh <corpus> <seed> <hparams.json> <suffix> <tag>:<ablation>[:'{"key":value}'] ...
# Diagnostic single runs (user 2026-09-10: ablations only for diagnosing the design).
# Each spec = output tag, --ablation value, optional JSON overrides merged into the hparams.
# Runs sequentially on one GPU. Output: runs/20260910_online_query_within<suffix>/diag/<corpus>/seed<seed>/<tag>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
corpus="$1"; seed="$2"; hp="$3"; suffix="$4"; shift 4
py="$HOME/miniconda3/envs/HateVideo/bin/python"
root="runs/20260910_online_query_within${suffix}/diag/${corpus}/seed${seed}"
mkdir -p "$root"
for spec in "$@"; do
  IFS=: read -r tag abl extra <<< "$spec"
  out="$root/$tag"; mkdir -p "$out"
  "$py" - "$hp" "$out/config_in.json" "${extra:-{\}}" <<'PY'
import json, sys
h = json.load(open(sys.argv[1])); h.update(json.loads(sys.argv[3]))
json.dump(h, open(sys.argv[2], "w"), indent=2)
PY
  echo "$(date) start $tag ablation=$abl extra=${extra:-}" >> "$root/chain.log"
  "$py" -u experiments/20260910_online_query_within/train.py --corpus "$corpus" --seed "$seed" --ablation "${abl:-full}" --config "$out/config_in.json" --out-dir "$out" --num-workers 4 > "$out/stdout.log" 2>&1 || echo "$(date) FAILED $tag" >> "$root/chain.log"
  echo "$(date) done $tag" >> "$root/chain.log"
done
echo "$(date) DONE" >> "$root/chain.log"
