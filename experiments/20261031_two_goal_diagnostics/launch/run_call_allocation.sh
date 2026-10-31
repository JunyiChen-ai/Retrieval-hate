#!/usr/bin/env bash
# CPU diagnostic on the host holding the cohort and the complete-method trajectories.
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_call_allocation.sh <uoa-lab1|uoa-lab3> \
#   > runs/20261031_two_goal_diagnostics/call_allocation/launch_<host>.out 2>&1 &
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261031_two_goal_diagnostics/call_allocation.py
R=runs/20261002_complete_io_method
echo "host $(hostname) | $(date -Iseconds) | $1"
echo $$ > runs/20261031_two_goal_diagnostics/call_allocation/run_$1.pid
case "$1" in
  uoa-lab3)
    "$P" $S --corpus hatemm --results-dir $R/hatemm/integrated/results
    "$P" $S --corpus dehate --results-dir $R/dehate/seed234/results $R/dehate/seed3407/results ;;
  uoa-lab1)
    "$P" $S --corpus hateclipseg --results-dir $R/hateclipseg/integrated/results
    "$P" $S --corpus dehate --results-dir $R/dehate/seed2025/results ;;
  *) echo "unknown host tag $1"; exit 1 ;;
esac
echo "DONE $(date -Iseconds)"
