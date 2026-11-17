#!/usr/bin/env bash
# CMHKF on DeHate (README section 2): the 5-trial validation search at seed 234, then the three-seed retraining of
# the winner scored on test once; the same protocol as the four DeHate baselines of 20260926_dehate_external.
# Resumable: the Optuna study and completed seeds are kept, so a job stopped by the 1-day limit is resubmitted as is.
# Submit on uoa-lab1 (features in results/reproduction/features, third_party/cmhkf at 3b07707):
#   ssh uoa-lab1 'cd ~/Retrieval-hate && mkdir -p runs/20261118_dehate_vera_cmhkf/cmhkf && \
#       sbatch experiments/20261118_dehate_vera_cmhkf/launch/cmhkf_dehate.sbatch'
#SBATCH -J rh-cmhkf-dehate
#SBATCH -p local-sc474397
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/20261118_dehate_vera_cmhkf/cmhkf/slurm_%j.out
set -uo pipefail
cd "$HOME/Retrieval-hate"
B=runs/20261118_dehate_vera_cmhkf/cmhkf
mkdir -p "$B"
PY=$HOME/miniconda3/envs/HateVideo/bin/python
echo "$(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none} cmhkf dehate"
rm -f "$B/cmhkf.DONE" "$B/cmhkf.FAILED"
# --max-new-attempts 40: about half of the DeHate temporal choices (128-step) are pruned at once by the CMHKF guard,
# so the default 10 attempts can stop before 5 trials complete; pruned attempts train nothing.
if $PY -u scripts/reproduction_baselines/tune_official_val.py --method cmhkf --corpus dehate --trials 5 \
       --max-new-attempts 40 --root "$B/tuning" \
   && $PY -u experiments/20260926_dehate_external/confirm_baselines.py --method cmhkf --corpus dehate \
       --tuning-root "$B/tuning" --final-root "$B/final"; then
  echo "$(date -Is) DONE" | tee "$B/cmhkf.DONE"
else
  echo "$(date -Is) FAILED rc=$?" | tee "$B/cmhkf.FAILED"
fi
