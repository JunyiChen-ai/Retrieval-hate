#!/usr/bin/env bash
# Test-time check of the per-video answer offset on the r5 checkpoints (README section 3.1), CPU, one process per trial.
# cd ~/Retrieval-hate && setsid nohup bash experiments/20261101_r5_completion/launch/run_offset_check.sh <corpus...> \
#   > runs/20261101_r5_completion/offset_check/launch_$(hostname).out 2>&1 < /dev/null &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=experiments/20261101_r5_completion/offset_check.py
R5=runs/20260929_query_paradigm_r5
O=runs/20261101_r5_completion/offset_check
mkdir -p $O
echo "host $(hostname) | $(date -Iseconds) | $*"
echo $$ > $O/run_$(hostname).pid
for c in "$@"; do
  case $c in
    hatemm) T="seed234/trial9 seed2025/trial4 seed3407/trial5" ;;
    hateclipseg) T="seed234/trial13 seed2025/trial11 seed3407/trial19" ;;
    dehate) T="seed234/trial8 seed2025/trial4 seed3407/trial4" ;;
    *) echo "unknown corpus $c"; exit 1 ;;
  esac
  for t in $T; do
    "$P" $S --corpus $c --trial-dir $R5/$c/$t ${OFF_ARGS:-} > $O/${c}_${t%%/*}.log 2>&1 &
  done
done
wait
grep -l Traceback $O/*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
