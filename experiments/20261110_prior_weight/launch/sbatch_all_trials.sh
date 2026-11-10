#!/usr/bin/env bash
# Stage A over all revision-6 trials (stage_a_all_trials.py), the three seeds of each given corpus concurrently on one
# GPU. Submit on the target host:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261110_prior_weight/all_trials && sbatch -p local-<node> \
#     -J a_all -o runs/20261110_prior_weight/all_trials/slurm_%j.out \
#     experiments/20261110_prior_weight/launch/sbatch_all_trials.sh "<corpus> [<corpus> ...]" ["<variant> ..."]'
# Finished trials are skipped, so the same command resumes or adds variants.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G
#SBATCH --time=1-00:00:00
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261110_prior_weight/all_trials
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpora $1 | code $(git log -1 --format='%s (%ad)')"
pids=()
for c in $1; do
  for s in 234 2025 3407; do
    mkdir -p $O/$c/seed$s
    if [ -n "${2:-}" ]; then
      "$P" experiments/20261110_prior_weight/stage_a_all_trials.py --corpus $c --seed $s --threads 2 --variants $2 \
        >> $O/$c/seed$s/stdout.log 2>&1 &
    else
      "$P" experiments/20261110_prior_weight/stage_a_all_trials.py --corpus $c --seed $s --threads 2 \
        >> $O/$c/seed$s/stdout.log 2>&1 &
    fi
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $(date -Iseconds)"
