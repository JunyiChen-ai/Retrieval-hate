#!/usr/bin/env bash
# Ablation arms and controls of the adopted method (experiments/20261101_r5_completion/README.md P6 / P8), one
# corpus, seeds 234 / 2025 / 3407 concurrently on one GPU, arms one after another per seed. Every arm trains once with
# the best trial's hyperparameters of that (corpus, seed) search plus the search's fixed settings, then the arm's
# change; no search. Submit on the target host:
#   ssh <host> 'cd ~/Retrieval-hate && mkdir -p runs/20261101_r6_length_table/arms && sbatch -p local-<node> -J r6arms_<corpus> \
#     -o runs/20261101_r6_length_table/arms/slurm_<corpus>_%j.out experiments/20261101_r6_length_table/launch/sbatch_arms.sh <corpus> [arm ...]'
# Arms already finished (summary.json present) are skipped, so a job cut by the time limit resumes when resubmitted.
# b_flat / d_bfs change only how the answers are combined / which node is asked (also at validation epoch selection), as
# the r5 arms (b) and (d); the copy likelihood is a test-time variant (sim_monotone.py source real_copy), no training.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=54G
#SBATCH --time=1-00:00:00
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$HOME/miniconda3/envs/HateVideo/bin/python"
S=runs/20261101_r6_length_table
c="$1"; shift
declare -A ARM=(
  [no_node]='{"node_prior": false}'
  [no_length]='{"answer_model": "anchored", "length_term": false}'
  [per_state_len]='{"answer_model": "anchored", "length_term": true}'
  [words]='{"answer_source": "words"}'
  [b_flat]='{"fusion": "flat"}'
  [d_bfs]='{"order": "bfs"}'
  [c_label]='{"objective": "label", "node_prior": false}'
  [e_independent]='{"prior": "independent", "node_prior": false}'
  [f_joint]='{"answer_model": "joint", "length_term": false}'
  [g_coupled]='{"chain_form": "coupled", "node_prior": false}'
  [lvl4]='{"query_level": 4}'
  [lvl8]='{"query_level": 8}'
  [lvl16]='{"query_level": 16}'
  [noback]='{"backbone": "const", "lamda_cma": 0, "node_prior": false}'
)
ORDER=(no_node no_length per_state_len words b_flat d_bfs c_label e_independent f_joint g_coupled lvl4 lvl8 lvl16 noback)
arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=("${ORDER[@]}")
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpus $c | arms ${arms[*]} | code $(git log -1 --format='%s (%ad)')"
run_seed() {
  local s=$1 study=$S/$c/seed$1/study_summary.json
  [ -f $S/$c/seed$s/search.DONE ] && [ -f $study ] || { echo "FAILED search of $c seed $s not finished"; return 1; }
  for a in "${arms[@]}"; do
    local out=$S/arms/$c/seed$s/$a
    [ -f $out/summary.json ] && { echo "skip $c seed $s $a (done)"; continue; }
    mkdir -p $out; rm -f $S/arms/$c/seed$s/$a.FAILED
    "$P" -c "import json,sys; s=json.load(open('$study')); b=dict(s['extra']); b.update(s['best']['params']); b.update(json.loads(sys.argv[1])); json.dump(b, open('$out/hparams.json','w'), indent=2)" "${ARM[$a]}"
    if "$P" experiments/20261101_r6_length_table/train.py --corpus $c --seed $s --out-dir $out --config $out/hparams.json \
        --num-workers 4 > $out/stdout.log 2>&1; then
      echo "$(date -Iseconds) done $c seed $s $a: $(grep -E '^fixed  8' $out/run.log)"
    else
      echo "$(date -Iseconds) FAILED $c seed $s $a"; touch $S/arms/$c/seed$s/$a.FAILED
    fi
  done
}
for s in 234 2025 3407; do run_seed $s & done
wait
fails=$(ls $S/arms/$c/seed*/*.FAILED 2>/dev/null | wc -l)
for s in 234 2025 3407; do [ -f $S/$c/seed$s/search.DONE ] || fails=$((fails + 1)); done
[ $fails -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $fails $(date -Iseconds)"

