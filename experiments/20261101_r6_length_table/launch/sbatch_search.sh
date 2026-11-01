#!/usr/bin/env bash
# Revision 6 search of one corpus, seeds 234 / 2025 / 3407 concurrently on one GPU (as the r5 search). Submit on the
# target host:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261101_r6_length_table && sbatch -p local-<node> -J r6_<corpus> \
#     -o runs/20261101_r6_length_table/slurm_<corpus>_%j.out experiments/20261101_r6_length_table/launch/sbatch_search.sh <corpus>'
# Budget 20 trials per seed written before the search (README: same cost as r5, whose searches pre-wrote 20). A search
# cut by the 1-day job limit resumes from its optuna.db when the same command is submitted again.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G
#SBATCH --time=1-00:00:00
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261101_r6_length_table
c="$1"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpus $c | code $(git log -1 --format='%s (%ad)')"
pids=()
for s in 234 2025 3407; do
  d=$O/$c/seed$s
  mkdir -p $d
  [ -f $d/budget.json ] || echo '{"n_trials": 20, "first_trial_seconds": null}' > $d/budget.json
  rm -f $d/search.DONE $d/search.FAILED
  ( "$P" experiments/20261101_r6_length_table/search.py --corpus $c --seed $s --out-root $O --num-workers 4 \
      >> $d/search_stdout.log 2>&1 && touch $d/search.DONE || touch $d/search.FAILED ) &
  pids+=($!)
done
wait "${pids[@]}"
ls $O/$c/seed*/search.FAILED 2>/dev/null && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
