#!/usr/bin/env bash
# External check (DeHate) of the goal-1 policy diagnostic: r5 checkpoints, EIG and EER orders with copy likelihood,
# then the stop-rule scan (uoa-lab3, CPU; the DeHate cohort and r5 checkpoints are there).
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_policy_lab_dehate.sh \
#   > runs/20261031_two_goal_diagnostics/policy_lab/launch_dehate.out 2>&1 &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P="$HOME/miniconda3/envs/HateVideo/bin/python"
D=experiments/20261031_two_goal_diagnostics
R5=runs/20260929_query_paradigm_r5/dehate
O=runs/20261031_two_goal_diagnostics/policy_lab
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | dehate"
echo $$ > $O/run_dehate.pid
"$P" $D/policy_lab.py --corpus dehate --tag r5 --threads 4 \
  --trial-dirs $R5/seed234/trial8 $R5/seed2025/trial4 $R5/seed3407/trial4 > $O/r5_dehate.log 2>&1 \
  && "$P" $D/rule_scan.py --corpus dehate --tags r5 > $O/rule_scan_dehate.log 2>&1
grep -l Traceback $O/r5_dehate.log $O/rule_scan_dehate.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
