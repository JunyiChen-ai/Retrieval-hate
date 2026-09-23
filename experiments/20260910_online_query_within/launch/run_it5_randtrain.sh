#!/usr/bin/env bash
# usage: cd ~/Retrieval-hate && mkdir -p runs/20260910_online_query_within_it5_randtrain && setsid nohup bash experiments/20260910_online_query_within/launch/run_it5_randtrain.sh <seed> \
#            > runs/20260910_online_query_within_it5_randtrain/launch_seed<seed>.out 2>&1 &
# README section 14 (2026-09-23): arm random_train with the iteration-5 search protocol, both corpora of one seed in
# parallel on this machine. Trials 0..7 use exactly the hparams of trials 0..7 of the iteration-5 study (trial 0 = the
# iteration-4b warm start, 1..7 = its TPE startup samples), budget 8 trials per seed: the first 8 trials of the
# iteration-5 study are the paired default. Test policies restricted to eoc / uniform (does not change the eoc operating point).
# Output runs/20260910_online_query_within_it5_randtrain/<corpus>/seed<seed>/; DONE marker randtrain_seed<seed>.DONE
set -euo pipefail
cd "$HOME/Retrieval-hate"
seed="$1"
out="runs/20260910_online_query_within_it5_randtrain"
mkdir -p "$out"
echo $$ > "$out/randtrain_seed$seed.pid"
case "$seed" in 234|2025|3407) ;; *) echo "unknown seed $seed" >&2; exit 2 ;; esac
# trials 0..7 = exactly the hparams of trials 0..7 of the iteration-5 study (launch/randtrain_enqueue.json, written
# from runs/20260910_online_query_within_it5/<corpus>/seed<seed>/trial<k>/hparams.json), so the pairing does not
# depend on the sampler reproducing them and survives a restart
enq_of() {
  "$HOME/miniconda3/envs/HateVideo/bin/python" -c "import json,sys; print(json.dumps(json.load(open('experiments/20260910_online_query_within/launch/randtrain_enqueue.json'))[sys.argv[1]]))" "$1:$2"
}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py="$HOME/miniconda3/envs/HateVideo/bin/python"
pids=()
for corpus in hatemm hateclipseg; do
  root="$out/$corpus/seed$seed"; mkdir -p "$root"
  [ -f "$root/budget.json" ] || echo '{"n_trials": 8, "first_trial_seconds": null, "note": "section 14 pre-set budget"}' > "$root/budget.json"
  "$py" -u experiments/20260910_online_query_within/search.py --corpus "$corpus" --seed "$seed" --out-root "$out" \
    --num-workers 4 --ablation random_train --enqueue-json "$(enq_of "$corpus" "$seed")" \
    --extra-config '{"policies": ["eoc", "uniform"]}' > "$root/search.out" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date) DONE rc=$rc" | tee "$out/randtrain_seed$seed.DONE"
