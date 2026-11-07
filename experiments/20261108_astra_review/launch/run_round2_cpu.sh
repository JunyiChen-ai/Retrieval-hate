#!/usr/bin/env bash
# Review round 2 CPU control E1b for one corpus (README): the frozen baselines' hidden features, then the fusion head
# trained on them jointly with all answers (e1_fusion.py --variants hid). No GPU. Needs the round-1 outputs (replays,
# baseline scores) and the official checkpoints in runs/20261108_astra_review/baseline_models/.
#   cd ~/Retrieval-hate && setsid nohup bash experiments/20261108_astra_review/launch/run_round2_cpu.sh <corpus> [hid|hid_val] \
#     > runs/20261108_astra_review/launch_r2_<corpus>[_hid_val].out 2>&1 &
# Variant hid (default) writes runs/20261108_astra_review/e1b, hid_val (head trained on the validation videos)
# writes runs/20261108_astra_review/e1b_val; the hidden features are extracted once (skipped when present).
set -uo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/miniconda3/envs/HateVideo/bin/python"
X=experiments/20261108_astra_review
S=runs/20261108_astra_review
c="$1"
V="${2:-hid}"
case $V in hid) O=$S/e1b; L=e1b ;; hid_val) O=$S/e1b_val; L=e1b_val ;; *) echo "unknown variant $V"; exit 1 ;; esac
echo $$ > $S/run_r2_${c}_$V.pid
echo "host $(hostname) | $(date -Iseconds) | corpus $c | variant $V | code $(git log -1 --format='%s (%ad)')"
trials() { "$P" -c "import json,sys; s=json.load(open('runs/20261101_r6_length_table/final_curves/$1/$c/summary.json'));
print(' '.join(r['trial'] for r in sorted(s['trials'].values(), key=lambda r: [234, 2025, 3407].index(int(r['seed'])))))"; }
TS=($(trials test_selected))
rdir() { echo "$S/replay/$c/seed$(basename $(dirname $1) | sed 's/seed//')_$(basename $1)"; }
fails=0
step() { echo "$(date -Iseconds) start $*"; "$@" || { echo "$(date -Iseconds) FAILED step: $*"; fails=$((fails + 1)); return 1; }; }
case $c in hatemm) EM=(macilsd) ;; hateclipseg) EM=(fed_wsvad_3client dsanet) ;; dehate) EM=(fed_wsvad_3client) ;; esac
pids=()
for m in "${EM[@]}"; do for s in 234 2025 3407; do
  ( step "$P" $X/baseline_infer.py --hidden --method $m --corpus $c --seed $s \
      --model-dir $S/baseline_models/$m/$c/seed_$s --threads 2 ) > $S/baseline_hidden_${m}_${c}_$s.log 2>&1 & pids+=($!)
done; done
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
grep -h FAILED $S/baseline_hidden_*_${c}_*.log 2>/dev/null && fails=$((fails + 1))
[ $fails -eq 0 ] || { echo "FAILED $fails $(date -Iseconds)"; exit 1; }
RT=(); for t in "${TS[@]}"; do RT+=($(rdir $t)); done
pids=()
for m in "${EM[@]}"; do for s in 234 2025 3407; do
  ( step "$P" $X/e1_fusion.py --corpus $c --method $m --replay-dirs "${RT[@]}" --seeds $s --threads 2 \
      --variants $V --out-root $O ) > $S/${L}_${c}_${m}_$s.log 2>&1 & pids+=($!)
done; done
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
grep -h FAILED $S/${L}_${c}_*.log 2>/dev/null && fails=$((fails + 1))
for m in "${EM[@]}"; do [ -f $O/$c/$m/summary.json ] || { echo "no $L summary for $m"; fails=$((fails + 1)); }; done
[ $fails -eq 0 ] && echo "DONE $(date -Iseconds)" || echo "FAILED $fails $(date -Iseconds)"
