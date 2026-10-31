#!/usr/bin/env bash
# CPU policy-order diagnostic with the validation-fitted three-state answer model, r5 checkpoints (uoa-lab1).
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_policy_lab_val3.sh \
#   > runs/20261031_two_goal_diagnostics/policy_lab/launch_val3.out 2>&1 &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261031_two_goal_diagnostics/policy_lab.py
R5=runs/20260929_query_paradigm_r5
O=runs/20261031_two_goal_diagnostics/policy_lab
echo "host $(hostname) | $(date -Iseconds) | val3"
echo $$ > $O/run_val3.pid
"$P" $S --corpus hatemm --tag r5_val3 --answer-model val3 --threads 3 --trial-dirs $R5/hatemm/seed234/trial9 $R5/hatemm/seed2025/trial4 $R5/hatemm/seed3407/trial5 > $O/r5_val3_hatemm.log 2>&1 &
"$P" $S --corpus hateclipseg --tag r5_val3 --answer-model val3 --threads 3 --trial-dirs $R5/hateclipseg/seed234/trial13 $R5/hateclipseg/seed2025/trial11 $R5/hateclipseg/seed3407/trial19 > $O/r5_val3_hateclipseg.log 2>&1 &
wait
grep -l Traceback $O/r5_val3_*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
