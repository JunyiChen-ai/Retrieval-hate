#!/usr/bin/env bash
# usage (repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/run_diag.sh <corpus> <seed> "<tag>=<json>" ["<tag>=<json>" ...] \
#            > runs/20260925_query_paradigm_diag/launch_<corpus>_<host>.out 2>&1 &
# README section 8: diagnostic runs (one fixed configuration each, no search), run one after another.
# Output: runs/20260925_query_paradigm_diag/<corpus>/seed<seed>/<tag>/ ; DONE / FAILED lines in the launch log.
set -uo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seed="$2"; shift 2
for spec in "$@"; do
  tag="${spec%%=*}"; json="${spec#*=}"
  out="runs/20260925_query_paradigm_diag/$corpus/seed$seed/$tag"
  mkdir -p "$out"
  echo "$json" > "$out/hparams.json"
  echo "$(date) $(hostname) start $corpus $tag $json"
  if "$HOME/miniconda3/envs/HateVideo/bin/python" experiments/20260925_query_paradigm/train.py --corpus "$corpus" --seed "$seed" \
       --out-dir "$out" --config "$out/hparams.json" > "$out/stdout.log" 2>&1; then
    echo "$(date) DONE $tag: $(grep -E '^fixed  8' "$out/run.log")"
  else
    echo "$(date) FAILED $tag"
  fi
done
echo "$(date) ALL DONE"
