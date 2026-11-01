#!/usr/bin/env bash
# Reported curves of the revision 6 search (CPU, no Slurm): for one corpus, the 3 seeds' trial chosen on test (rule 7,
# study_summary "best") and the trial chosen on validation (study_summary "validation_selected"), each through
# experiments/20261101_r5_completion/final_curves.py (fixed 0-32 calls, qmixE1 budgets 1-8, unified evaluator).
# Run on the host that holds the search output:
#   cd ~/Retrieval-hate && setsid nohup bash experiments/20261101_r6_length_table/launch/run_final_curves.sh <corpus> \
#     > runs/20261101_r6_length_table/final_curves/launch_<corpus>.out 2>&1 &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261101_r6_length_table
c="$1"
echo "host $(hostname) | $(date -Iseconds) | corpus $c | code $(git log -1 --format='%s (%ad)')"
mkdir -p $O/final_curves && echo $$ > $O/final_curves/run_$c.pid
fail=0
for sel in best validation_selected; do
  dirs=()
  for s in 234 2025 3407; do
    n=$("$P" -c "import json; print(json.load(open('$O/$c/seed$s/study_summary.json'))['$sel']['number'])") || fail=1
    dirs+=("$O/$c/seed$s/trial$n")
  done
  tag=$([ $sel = best ] && echo test_selected || echo val_selected)
  echo "$tag ${dirs[*]}"
  "$P" experiments/20261101_r5_completion/final_curves.py --corpus $c --trial-dirs "${dirs[@]}" --tag $tag \
      --out-root $O/final_curves || fail=1
done
[ $fail -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $(date -Iseconds)"
