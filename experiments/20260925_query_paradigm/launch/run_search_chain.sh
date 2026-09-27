#!/usr/bin/env bash
# usage (repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/run_search_chain.sh <out_root> <extra_config_json> <corpus:seed> ... \
#            > <out_root>/launch_chain_<name>.out 2>&1 &
# Revision 4 (README section 15.4): several fixed Optuna searches one after another in one process chain
# (launch/run_search.sh each; search.DONE / search.FAILED markers in <out_root>/<corpus>/seed<seed>/).
set -uo pipefail
cd "$HOME/Retrieval-hate"
out_root="$1"; extra="$2"; shift 2
for job in "$@"; do
  c="${job%%:*}"; s="${job#*:}"
  bash experiments/20260925_query_paradigm/launch/run_search.sh "$c" "$s" "$out_root" "$extra"
done
echo "$(date) CHAIN DONE"
