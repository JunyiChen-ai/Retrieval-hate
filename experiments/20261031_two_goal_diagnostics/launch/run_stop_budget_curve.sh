#!/usr/bin/env bash
# Goal 1 option 1 (user ruling 2026-11-01): qmixE1 curve over user budgets 1-7 against fixed k on the stored r5
# policy_lab runs of the three main corpora (uoa-lab1, CPU).
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_stop_budget_curve.sh \
#   > runs/20261031_two_goal_diagnostics/stop_budget_curve/launch.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
D=experiments/20261031_two_goal_diagnostics
O=runs/20261031_two_goal_diagnostics/stop_budget_curve
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | stop_budget_curve"
echo $$ > $O/run.pid
for c in hatemm hateclipseg dehate; do
  "$P" $D/stop_budget_curve.py --corpus $c > $O/$c.log 2>&1 &
done
wait
grep -l Traceback $O/*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
