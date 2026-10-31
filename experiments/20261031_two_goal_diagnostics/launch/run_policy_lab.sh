#!/usr/bin/env bash
# CPU policy-order diagnostic on the host holding the checkpoints and cohort.
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_policy_lab.sh <uoa-lab1|uoa-lab3> \
#   > runs/20261031_two_goal_diagnostics/policy_lab/launch_<host>.out 2>&1 &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261031_two_goal_diagnostics/policy_lab.py
R5=runs/20260929_query_paradigm_r5
IO=runs/20261001_inside_outside_backbone
O=runs/20261031_two_goal_diagnostics/policy_lab
echo "host $(hostname) | $(date -Iseconds) | $1"
echo $$ > $O/run_$1.pid
case "$1" in
  uoa-lab1)
    "$P" $S --corpus hatemm --tag r5 --threads 3 --trial-dirs $R5/hatemm/seed234/trial9 $R5/hatemm/seed2025/trial4 $R5/hatemm/seed3407/trial5 > $O/r5_hatemm.log 2>&1 &
    "$P" $S --corpus hateclipseg --tag r5 --threads 3 --trial-dirs $R5/hateclipseg/seed234/trial13 $R5/hateclipseg/seed2025/trial11 $R5/hateclipseg/seed3407/trial19 > $O/r5_hateclipseg.log 2>&1 &
    "$P" $S --corpus hateclipseg --tag io --threads 3 --trial-dirs $IO/hateclipseg/seed234/trial11 $IO/hateclipseg/seed2025/trial2 $IO/hateclipseg/seed3407/trial10 > $O/io_hateclipseg.log 2>&1 &
    wait ;;
  uoa-lab3)
    "$P" $S --corpus hatemm --tag io --threads 3 --trial-dirs $IO/hatemm/seed234/trial17 $IO/hatemm/seed2025/trial11 $IO/hatemm/seed3407/trial11 > $O/io_hatemm.log 2>&1 ;;
  *) echo "unknown host tag $1"; exit 1 ;;
esac
grep -l Traceback $O/*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
