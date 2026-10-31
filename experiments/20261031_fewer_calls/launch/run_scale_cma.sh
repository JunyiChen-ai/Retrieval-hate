#!/usr/bin/env bash
# Goal-1 stopping rule on the scale-aware CMA checkpoints (goal-2 candidate, test-selected trial per seed), CPU.
# Main rule and the component ablation (README section 5) in one run: the policy runs are shared.
# Arguments: corpus, then one trial directory per seed. Example (uoa-lab1):
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_fewer_calls/launch/run_scale_cma.sh hateclipseg \
#   runs/20261031_scale_aware_cma/hateclipseg/seed234/trial7 ... > runs/20261031_fewer_calls/launch_scale_cma_hateclipseg.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261031_fewer_calls/evaluate_stop.py
O=runs/20261031_fewer_calls
C=$1; shift
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | scale_cma $C | $*"
echo $$ > $O/run_scale_cma_$C.pid
"$P" $S --corpus $C --tag scale_cma --rules qmixE1 qmixE3 dlsum eig --threads 3 --trial-dirs "$@" > $O/scale_cma_$C.log 2>&1
grep -l Traceback $O/scale_cma_$C.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
