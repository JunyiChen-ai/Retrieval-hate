#!/usr/bin/env bash
# Revision 7 with a validated first level (README section 10), search of one corpus, the given seeds (default
# 234 2025 3407) concurrently on one GPU. Submit on the
# target host:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261110_prior_weight/search_design && sbatch -p local-<node> \
#     -J r7d_<corpus> -o runs/20261110_prior_weight/search_design/slurm_<corpus>_%j.out \
#     experiments/20261110_prior_weight/launch/sbatch_search_design.sh <corpus> ["<seed> ..."]'
# The trial budget is set by src/qtl/search.py from the measured time of each seed's first trial (rule 7: 20 trials
# when it took at most 1 hour, else 5) and written to budget.json; DeHate runs at most two seeds per GPU so that the
# measured time is not inflated by three processes sharing one GPU (revision 6: 3308-3407 s per DeHate trial with
# three). A search cut by the 1-day job limit resumes from its optuna.db when the same command is submitted again.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G
#SBATCH --time=1-00:00:00
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261110_prior_weight/search_design
c="$1"
seeds="${2:-234 2025 3407}"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpus $c | seeds $seeds | code $(git log -1 --format='%s (%ad)')"
pids=()
for s in $seeds; do
  d=$O/$c/seed$s
  mkdir -p $d
  rm -f $d/search.DONE $d/search.FAILED
  ( "$P" experiments/20261110_prior_weight/search_design.py --corpus $c --seed $s --out-root $O --num-workers 4 \
      >> $d/search_stdout.log 2>&1 && touch $d/search.DONE || touch $d/search.FAILED ) &
  pids+=($!)
done
wait "${pids[@]}"
fail=0
for s in $seeds; do [ -f $O/$c/seed$s/search.FAILED ] && fail=1; done
[ $fail -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $(date -Iseconds)"
