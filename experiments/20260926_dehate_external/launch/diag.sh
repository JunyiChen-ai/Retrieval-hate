#!/usr/bin/env bash
# DeHate diagnostic arms of the query-tree method (README section 4; the arms of
# experiments/20260925_query_paradigm/README.md section 12: abl_full, abl_noback, abl_lvl{4,8,16}), default
# hyperparameters, no search, one configuration after another:
#   setsid nohup bash experiments/20260926_dehate_external/launch/diag.sh <seed> "<tag>=<json>" [...] \
#       > runs/20260926_dehate_external/diag/launch_seed<seed>_<host>.out 2>&1 &
# Output: runs/20260926_dehate_external/diag/dehate/seed<seed>/<tag>/ ; DONE / FAILED lines in the launch log.
set -uo pipefail
cd "$HOME/Retrieval-hate"
seed="$1"; shift
for spec in "$@"; do
  tag="${spec%%=*}"; json="${spec#*=}"
  out="runs/20260926_dehate_external/diag/dehate/seed$seed/$tag"
  mkdir -p "$out"
  echo "$json" > "$out/hparams.json"
  echo "$(date) $(hostname) start dehate $tag $json"
  if "$HOME/miniconda3/envs/HateVideo/bin/python" experiments/20260925_query_paradigm/train.py --corpus dehate --seed "$seed" \
       --out-dir "$out" --config "$out/hparams.json" > "$out/stdout.log" 2>&1; then
    echo "$(date) DONE $tag: $(grep -E '^fixed  8' "$out/run.log")"
  else
    echo "$(date) FAILED $tag"
  fi
done
echo "$(date) ALL DONE"
