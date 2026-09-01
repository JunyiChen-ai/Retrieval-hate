#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXP="$ROOT/experiments/20260901_local_quotient_adversarial_mil"
RUN="$ROOT/runs/20260901_local_quotient_adversarial_mil"
mkdir -p "$RUN"
echo $$ > "$RUN/run.pid"

bash "$EXP/search_and_formal.sh" hatemm > "$RUN/hatemm.log" 2>&1 &
hatemm_pid=$!
bash "$EXP/search_and_formal.sh" hateclipseg > "$RUN/hateclipseg.log" 2>&1 &
hcs_pid=$!
echo "$hatemm_pid" > "$RUN/hatemm.pid"
echo "$hcs_pid" > "$RUN/hateclipseg.pid"
wait "$hatemm_pid"
wait "$hcs_pid"

"$PY" "$EXP/summarize.py" \
  --run-root "$RUN/formal_val_selected_seed234" > "$RUN/summary.log" 2>&1
