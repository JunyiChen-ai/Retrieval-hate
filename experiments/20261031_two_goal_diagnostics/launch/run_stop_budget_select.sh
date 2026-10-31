#!/usr/bin/env bash
# Goal 1 option 2: qmixE1 threshold chosen on validation (argmax and one-standard-error rule, no call target) on the
# stored r5 policy_lab runs of the three main corpora (uoa-lab1, CPU).
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_two_goal_diagnostics/launch/run_stop_budget_select.sh \
#   > runs/20261031_two_goal_diagnostics/stop_budget_select/launch.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
D=experiments/20261031_two_goal_diagnostics
O=runs/20261031_two_goal_diagnostics/stop_budget_select
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | stop_budget_select"
echo $$ > $O/run.pid
for c in hatemm hateclipseg dehate; do
  "$P" $D/stop_budget_select.py --corpus $c > $O/$c.log 2>&1 &
done
wait
grep -l Traceback $O/*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
