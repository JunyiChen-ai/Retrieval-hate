#!/usr/bin/env bash
# Robustness check of the goal-1 stop-rule candidates on more r5 checkpoints: the 2nd and 3rd best trials of each
# seed by test (AP + ROC) / 2 at fixed 8 (uoa-lab1, CPU), then the rule scan on r5 and r5_more.
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_policy_lab_more.sh \
#   > runs/20261031_two_goal_diagnostics/policy_lab/launch_more.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="$HOME/miniconda3/envs/HateVideo/bin/python"
D=experiments/20261031_two_goal_diagnostics
R5=runs/20260929_query_paradigm_r5
O=runs/20261031_two_goal_diagnostics/policy_lab
echo "host $(hostname) | $(date -Iseconds) | more"
echo $$ > $O/run_more.pid
"$P" $D/policy_lab.py --corpus hatemm --tag r5_more --threads 3 --trial-dirs \
  $R5/hatemm/seed234/trial17 $R5/hatemm/seed234/trial16 $R5/hatemm/seed2025/trial5 $R5/hatemm/seed2025/trial2 \
  $R5/hatemm/seed3407/trial10 $R5/hatemm/seed3407/trial4 > $O/r5_more_hatemm.log 2>&1 \
  && "$P" $D/rule_scan.py --corpus hatemm --tags r5 r5_more > $O/rule_scan_hatemm.log 2>&1 &
"$P" $D/policy_lab.py --corpus hateclipseg --tag r5_more --threads 3 --trial-dirs \
  $R5/hateclipseg/seed234/trial1 $R5/hateclipseg/seed234/trial15 $R5/hateclipseg/seed2025/trial10 \
  $R5/hateclipseg/seed2025/trial6 $R5/hateclipseg/seed3407/trial16 $R5/hateclipseg/seed3407/trial5 \
  > $O/r5_more_hateclipseg.log 2>&1 \
  && "$P" $D/rule_scan.py --corpus hateclipseg --tags r5 r5_more > $O/rule_scan_hateclipseg.log 2>&1 &
wait
grep -l Traceback $O/r5_more_*.log $O/rule_scan_h*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
