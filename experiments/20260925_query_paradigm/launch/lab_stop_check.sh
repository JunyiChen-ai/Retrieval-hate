#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5/stop_check && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_stop_check.sh [device] [out_root] [out_suffix] [copy_pi] \
#     > runs/20260929_query_paradigm_r5/stop_check/run_$(hostname)${out_suffix}.log 2>&1 < /dev/null &
# README section 17.4: stopping rules (stop_check.py) on the best trials of <out_root> (revision 4 by default: the
# decoded-answer reference; later the revision 5 soft-answer trials), three corpora, no training.
# env: CORPORA="hatemm hateclipseg" restricts the corpora; DUMP=1 saves the recorded runs (runs.pkl); RULES="hG hT" restricts the rules; FROM_DUMP=_r5state re-scores the saved runs of that earlier suffix instead of re-running the policy.
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
[ -n "${DUMP:-}" ] && extra+=(--dump)
[ -n "${RULES:-}" ] && extra+=(--rules $RULES)
[ -n "${FROM_DUMP:-}" ] && extra+=(--from-dump "$FROM_DUMP")
[ -n "${ERM:-}" ] && extra+=(--erm "$ERM")
[ -n "${SELECT_VAL:-}" ] && extra+=(--select-val)
best() {  # corpus -> the three best trial dirs: study_summary.json of the search root when present (revision 5 on the
          # machine that ran the search), else the revision-4 numbers (lab machines hold the trial dirs only)
  local c="$1" out=""
  for seed in 234 2025 3407; do
    if [ -f "$R/$c/seed$seed/study_summary.json" ]; then
      out="$out $R/$c/seed$seed/trial$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best']['number'])" "$R/$c/seed$seed/study_summary.json")"
    fi
  done
  if [ "$(echo $out | wc -w)" -eq 3 ]; then echo "$out"; return; fi   # all three seeds searched here; else the fixed list
  case "$c" in
    hatemm)      echo "$R/hatemm/seed234/trial5 $R/hatemm/seed2025/trial11 $R/hatemm/seed3407/trial4" ;;
    hateclipseg) echo "$R/hateclipseg/seed234/trial13 $R/hateclipseg/seed2025/trial18 $R/hateclipseg/seed3407/trial16" ;;
    dehate)      echo "$R/dehate/seed234/trial16 $R/dehate/seed2025/trial19 $R/dehate/seed3407/trial2" ;;
  esac
}
CORPORA="${CORPORA:-hatemm hateclipseg dehate}"
for c in $CORPORA; do
  "$PY" -u experiments/20260925_query_paradigm/stop_check.py --corpus "$c" --device "$DEV" --out-suffix "$SUFFIX" "${extra[@]}" \
    --trials $(best "$c")
done
echo ALL_DONE
