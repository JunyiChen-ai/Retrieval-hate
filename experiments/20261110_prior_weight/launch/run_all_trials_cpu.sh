#!/usr/bin/env bash
# Stage A over all revision-6 trials on the CPU (no Slurm; for variants that are cheap without a GPU, e.g. the
# validation-split replays "<variant>@val"). The corpora run one after another, the three seeds of a corpus
# concurrently with 2 threads each. Start on the target host, detached:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261110_prior_weight/all_trials && setsid nohup bash \
#     experiments/20261110_prior_weight/launch/run_all_trials_cpu.sh "<corpus> [<corpus> ...]" "<variant> ..." \
#     > runs/20261110_prior_weight/all_trials/cpu_<tag>.out 2>&1 &'
# Finished variants are skipped (same trial files as the GPU launcher). PID: runs/.../all_trials/cpu_run.pid.
# The code is that of the checkout this script lies in (the main repository, or a branch checkout
# ~/Retrieval-hate-<branch> whose data / results / runs link to the main repository).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261110_prior_weight/all_trials
echo $$ > $O/cpu_run.pid
echo "host $(hostname) | $(date -Iseconds) | corpora $1 | variants $2 | code $(git log -1 --format='%s (%ad)')"
fail=0
for c in $1; do
  pids=()
  for s in 234 2025 3407; do
    mkdir -p $O/$c/seed$s
    "$P" experiments/20261110_prior_weight/stage_a_all_trials.py --corpus $c --seed $s --threads 2 --device cpu \
      --variants $2 >> $O/$c/seed$s/stdout_cpu.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  echo "corpus $c finished $(date -Iseconds) fail=$fail"
done
[ $fail -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $(date -Iseconds)"
