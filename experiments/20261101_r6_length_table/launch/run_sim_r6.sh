#!/usr/bin/env bash
# Test-time variants and the answer-dependence diagnosis on the revision 6 test-selected trials (CPU, no Slurm):
# experiments/20261101_r5_completion/sim_monotone.py with sources real_indep (revision 6 as trained), real_copy (plus the
# copy likelihood at test), sim_indep (answers drawn from the model's own table), shuf_anch (real answers shuffled across
# videos within true state x length bucket). The real_indep runs pickles feed baseline_plus_vlm.py (P7).
#   cd ~/Retrieval-hate && setsid nohup bash experiments/20261101_r6_length_table/launch/run_sim_r6.sh <corpus> \
#     > runs/20261101_r6_length_table/sim_monotone/launch_<corpus>.out 2>&1 &
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
O=runs/20261101_r6_length_table
c="$1"
mkdir -p $O/sim_monotone
echo "host $(hostname) | $(date -Iseconds) | corpus $c | code $(git log -1 --format='%s (%ad)')"
echo $$ > $O/sim_monotone/run_$c.pid
for s in 234 2025 3407; do
  n=$("$P" -c "import json; print(json.load(open('$O/$c/seed$s/study_summary.json'))['best']['number'])")
  "$P" experiments/20261101_r5_completion/sim_monotone.py --corpus $c --trial-dir $O/$c/seed$s/trial$n \
      --sources real_indep real_copy sim_indep shuf_anch --out-root $O/sim_monotone > $O/sim_monotone/${c}_seed$s.log 2>&1 &
done
wait
grep -l Traceback $O/sim_monotone/${c}_seed*.log && echo "FAILED $(date -Iseconds)" || echo "DONE $(date -Iseconds)"
