#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5/copy_check && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_copy_check.sh [device] \
#     > runs/20260929_query_paradigm_r5/copy_check/run_$(hostname).log 2>&1 &
# README section 17.2: copy-type nested-answer likelihood on the revision-4 best trials (copy_check.py), three corpora,
# no training; device cpu by default (the GPU is busy with the soft-answer extraction).
set -euo pipefail
cd "$HOME/Retrieval-hate"
DEV="${1:-cpu}"
hostname; echo $$ > "runs/20260929_query_paradigm_r5/copy_check/run_$(hostname).pid"
R4=runs/20260927_query_paradigm_r4
PY="$HOME/miniconda3/envs/HateVideo/bin/python"
"$PY" -u experiments/20260925_query_paradigm/copy_check.py --corpus hatemm --device "$DEV" \
  --trials $R4/hatemm/seed234/trial5 $R4/hatemm/seed2025/trial11 $R4/hatemm/seed3407/trial4
"$PY" -u experiments/20260925_query_paradigm/copy_check.py --corpus hateclipseg --device "$DEV" \
  --trials $R4/hateclipseg/seed234/trial13 $R4/hateclipseg/seed2025/trial18 $R4/hateclipseg/seed3407/trial16
"$PY" -u experiments/20260925_query_paradigm/copy_check.py --corpus dehate --device "$DEV" \
  --trials $R4/dehate/seed234/trial16 $R4/dehate/seed2025/trial19 $R4/dehate/seed3407/trial2
echo ALL_DONE
