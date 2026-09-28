#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5/copy_check && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_copy_check.sh [device] [variants] [out_suffix] [search_root]   (CORPORA="hatemm hateclipseg" to restrict) \
#     > runs/20260929_query_paradigm_r5/copy_check/run_$(hostname)${out_suffix}.log 2>&1 &
# README section 17.2: copy-type nested-answer likelihood on the revision-4 best trials (copy_check.py), three corpora,
# no training; device cpu by default (the GPU is busy with the soft-answer extraction).
# Second run (README 17.2 variants): variants=copyeig_neg,copylik_neg,copyeig_0.5,copylik_0.5 out_suffix=_v2
set -euo pipefail
cd "$HOME/Retrieval-hate"
DEV="${1:-cpu}"
VARIANTS="${2:-tree,copy_neg,copy_0.25,copy_0.5,copy_0.75}"
SUFFIX="${3:-}"
hostname; echo $$ > "runs/20260929_query_paradigm_r5/copy_check/run_$(hostname)${SUFFIX}.pid"
R="${4:-runs/20260927_query_paradigm_r4}"      # search root (revision 5: runs/20260929_query_paradigm_r5)
CORPORA="${CORPORA:-hatemm hateclipseg dehate}"
PY="$HOME/miniconda3/envs/HateVideo/bin/python"
best() {  # as in lab_stop_check.sh: study_summary.json when present, else the revision-4 numbers
  local c="$1" out=""
  for seed in 234 2025 3407; do
    if [ -f "$R/$c/seed$seed/study_summary.json" ]; then
      out="$out $R/$c/seed$seed/trial$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best']['number'])" "$R/$c/seed$seed/study_summary.json")"
    fi
  done
  if [ -n "$out" ]; then echo "$out"; return; fi
  case "$c" in
    hatemm)      echo "$R/hatemm/seed234/trial5 $R/hatemm/seed2025/trial11 $R/hatemm/seed3407/trial4" ;;
    hateclipseg) echo "$R/hateclipseg/seed234/trial13 $R/hateclipseg/seed2025/trial18 $R/hateclipseg/seed3407/trial16" ;;
    dehate)      echo "$R/dehate/seed234/trial16 $R/dehate/seed2025/trial19 $R/dehate/seed3407/trial2" ;;
  esac
}
for c in $CORPORA; do
  "$PY" -u experiments/20260925_query_paradigm/copy_check.py --corpus "$c" --device "$DEV" --variants "$VARIANTS" --out-suffix "$SUFFIX" \
    --trials $(best "$c")
done
echo ALL_DONE
