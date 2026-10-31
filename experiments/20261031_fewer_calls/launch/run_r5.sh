#!/usr/bin/env bash
# Goal-1 stopping rule on the r5 checkpoints (test-selected trial per seed), CPU.
# Main corpora on uoa-lab1, DeHate (external check) on uoa-lab3:
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_fewer_calls/launch/run_r5.sh main \
#   > runs/20261031_fewer_calls/launch_r5_main.out 2>&1 < /dev/null &
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261031_fewer_calls/launch/run_r5.sh dehate \
#   > runs/20261031_fewer_calls/launch_r5_dehate.out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261031_fewer_calls/evaluate_stop.py
R5=runs/20260929_query_paradigm_r5
O=runs/20261031_fewer_calls
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | r5 $1"
echo $$ > $O/run_r5_$1.pid
if [ "$1" = main ]; then
  "$P" $S --corpus hatemm --tag r5 --threads 3 --trial-dirs $R5/hatemm/seed234/trial9 $R5/hatemm/seed2025/trial4 \
    $R5/hatemm/seed3407/trial5 > $O/r5_hatemm.log 2>&1 &
  "$P" $S --corpus hateclipseg --tag r5 --threads 3 --trial-dirs $R5/hateclipseg/seed234/trial13 \
    $R5/hateclipseg/seed2025/trial11 $R5/hateclipseg/seed3407/trial19 > $O/r5_hateclipseg.log 2>&1 &
  wait
  L="$O/r5_hatemm.log $O/r5_hateclipseg.log"
else
  "$P" $S --corpus dehate --tag r5 --threads 4 --trial-dirs $R5/dehate/seed234/trial8 $R5/dehate/seed2025/trial4 \
    $R5/dehate/seed3407/trial4 > $O/r5_dehate.log 2>&1
  L="$O/r5_dehate.log"
fi
grep -l Traceback $L && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
