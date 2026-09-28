#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260928_query_paradigm_r5_analysis && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_r5_analysis_dehate.sh \
#     > runs/20260928_query_paradigm_r5_analysis/dehate_run_$(hostname).log 2>&1 &
# README section 16.4 (3) and (6) on DeHate: question placement (tree / disjoint / single level) and the
# validation-fitted three-state answer model, on the revision-4 DeHate best trials (seed 234 trial 16,
# seed 2025 trial 19, seed 3407 trial 2; summary_dehate.json). No training. Trial dirs copied from uoa-lab2 / uoa-lab3.
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
OUT=runs/20260928_query_paradigm_r5_analysis
echo $$ > "$OUT/dehate_run_$(hostname).pid"
R4=runs/20260927_query_paradigm_r4/dehate
TRIALS="$R4/seed234/trial16 $R4/seed2025/trial19 $R4/seed3407/trial2"
PY="$HOME/miniconda3/envs/HateVideo/bin/python"
"$PY" -u experiments/20260925_query_paradigm/asking_check.py --corpus dehate --trials $TRIALS
"$PY" -u experiments/20260925_query_paradigm/val_calibration_check.py --corpus dehate --trials $TRIALS
echo ALL_DONE
