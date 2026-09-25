#!/usr/bin/env bash
# One baseline on DeHate (README section 2): the 40-trial validation search at seed 234, then the three-seed
# retraining of the winner scored on test once. Resumable (Optuna study and completed seeds are kept).
#   setsid nohup bash experiments/20260926_dehate_external/launch/baseline.sh <method> \
#       > runs/20260926_dehate_external/baselines/<method>_<host>.out 2>&1 &
# method: macilsd | multihateloc | dsanet | fed_wsvad_3client. DONE / FAILED markers next to the .out file.
set -uo pipefail
cd "$HOME/Retrieval-hate"
m="$1"
B=runs/20260926_dehate_external/baselines
mkdir -p "$B"
PY=$HOME/miniconda3/envs/HateVideo/bin/python
echo "$(date -Is) $(hostname) baseline $m"
if $PY -u scripts/reproduction_baselines/tune_official_val.py --method "$m" --corpus dehate --trials 40 \
       --root "$B/tuning" \
   && $PY -u experiments/20260926_dehate_external/confirm_baselines.py --method "$m" --corpus dehate \
       --tuning-root "$B/tuning" --final-root "$B/final"; then
  echo "$(date -Is) DONE" | tee "$B/$m.DONE"
else
  echo "$(date -Is) FAILED rc=$?" | tee "$B/$m.FAILED"
fi
