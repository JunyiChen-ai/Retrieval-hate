#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5/stop_check && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_stop_check.sh [device] [out_root] [out_suffix] [copy_pi] \
#     > runs/20260929_query_paradigm_r5/stop_check/run_$(hostname)${out_suffix}.log 2>&1 < /dev/null &
# README section 17.4: stopping rules (stop_check.py) on the best trials of <out_root> (revision 4 by default: the
# decoded-answer reference; later the revision 5 soft-answer trials), three corpora, no training.
set -euo pipefail
cd "$HOME/Retrieval-hate"
DEV="${1:-cpu}"
R="${2:-runs/20260927_query_paradigm_r4}"
SUFFIX="${3:-}"
COPY="${4:-}"
hostname; echo $$ > "runs/20260929_query_paradigm_r5/stop_check/run_$(hostname)${SUFFIX}.pid"
PY="$HOME/miniconda3/envs/HateVideo/bin/python"
extra=()
[ -n "$COPY" ] && extra+=(--copy-pi "$COPY")
best() {  # corpus -> the three best trial dirs (test-selected, summary.json of the search root)
  "$PY" - "$R" "$1" <<'PYEOF'
import json, os, sys
root, corpus = sys.argv[1], sys.argv[2]
for seed in (234, 2025, 3407):
    s = json.load(open(os.path.join(root, corpus, "seed%d" % seed, "study_summary.json")))
    print(os.path.join(root, corpus, "seed%d" % seed, "trial%d" % s["best"]["number"]))
PYEOF
}
for c in hatemm hateclipseg dehate; do
  "$PY" -u experiments/20260925_query_paradigm/stop_check.py --corpus "$c" --device "$DEV" --out-suffix "$SUFFIX" "${extra[@]}" \
    --trials $(best "$c")
done
echo ALL_DONE
