#!/usr/bin/env bash
# Review round 1 CPU controls for one corpus (README): policy replays, E3, E4, baseline re-scoring and E1. No GPU.
#   cd ~/Retrieval-hate && setsid nohup bash experiments/20261108_astra_review/launch/run_round1_cpu.sh <corpus> \
#     > runs/20261108_astra_review/launch_<corpus>.out 2>&1 &
# Trials: the revision-6 test-selected and validation-selected trials of each seed (final_curves summaries).
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
X=experiments/20261108_astra_review
S=runs/20261108_astra_review
c="$1"
echo $$ > $S/run_$c.pid
echo "host $(hostname) | $(date -Iseconds) | corpus $c | code $(git log -1 --format='%s (%ad)')"
trials() { "$P" -c "import json,sys; s=json.load(open('runs/20261101_r6_length_table/final_curves/$1/$c/summary.json'));
print(' '.join(r['trial'] for r in sorted(s['trials'].values(), key=lambda r: [234, 2025, 3407].index(int(r['seed'])))))"; }
TS=($(trials test_selected)); VS=($(trials val_selected))
rdir() { echo "$S/replay/$c/seed$(basename $(dirname $1) | sed 's/seed//')_$(basename $1)"; }
fails=0
step() { echo "$(date -Iseconds) start $*"; "$@" || { echo "$(date -Iseconds) FAILED step: $*"; fails=$((fails + 1)); return 1; }; }
# baseline re-scoring (E1 step 1), in parallel with the replays
case $c in
  hateclipseg) BM=(fed_wsvad_3client dsanet); MD=$S/baseline_models ;;
  dehate) BM=(fed_wsvad_3client); MD=$S/baseline_models ;;
  *) BM=() ;;
esac
pids=()
for m in "${BM[@]}"; do for s in 234 2025 3407; do
  ( step "$P" $X/baseline_infer.py --method $m --corpus $c --seed $s --model-dir $MD/$m/$c/seed_$s \
      --ref $MD/$m/$c/seed_$s/scores.jsonl --threads 2 > $S/baseline_infer_${m}_${c}_$s.log 2>&1 ) & pids+=($!)
done; done
# replays: test-selected trials on train / val / test (method) and test (one level), validation-selected on val / test
for i in 0 1 2; do
  ( step "$P" $X/policy_replay.py --corpus $c --trial-dirs ${TS[$i]} --splits train val test --levels 0 &&
    step "$P" $X/policy_replay.py --corpus $c --trial-dirs ${TS[$i]} --splits test --levels 4 8 16 &&
    step "$P" $X/policy_replay.py --corpus $c --trial-dirs ${VS[$i]} --splits val test --levels 0 ) \
    > $S/replay_${c}_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
grep -h FAILED $S/replay_${c}_*.log $S/baseline_infer_*_${c}_*.log 2>/dev/null && fails=$((fails + 1))
RT=(); for t in "${TS[@]}"; do RT+=($(rdir $t)); done
RV=(); for t in "${VS[@]}"; do RV+=($(rdir $t)); done
step "$P" $X/e3_level_matched.py --corpus $c --replay-dirs "${RT[@]}"
step "$P" $X/e4_budget_audit.py --corpus $c --tag test_selected --replay-dirs "${RT[@]}"
step "$P" $X/e4_budget_audit.py --corpus $c --tag val_selected --replay-dirs "${RV[@]}"
# E1: one process per seed and baseline
case $c in hatemm) EM=(macilsd) ;; hateclipseg) EM=(fed_wsvad_3client dsanet) ;; dehate) EM=(fed_wsvad_3client) ;; esac
pids=()
for m in "${EM[@]}"; do for s in 234 2025 3407; do
  ( step "$P" $X/e1_fusion.py --corpus $c --method $m --replay-dirs "${RT[@]}" --seeds $s --threads 2 ) \
    > $S/e1_${c}_${m}_$s.log 2>&1 & pids+=($!)
done; done
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
grep -h FAILED $S/e1_${c}_*.log 2>/dev/null && fails=$((fails + 1))
for m in "${EM[@]}"; do [ -f $S/e1/$c/$m/summary.json ] || { echo "no E1 summary for $m"; fails=$((fails + 1)); }; done
[ $fails -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $fails $(date -Iseconds)"
