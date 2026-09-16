#!/usr/bin/env bash
# usage: bash experiments/20260910_online_query_within/launch/run_it5.sh <corpus> <seed>
# Module-1 iteration 5 (README section 11): warm start from the iteration-4b best trial of the same
# seed, budget HateMM 12 / HCS 15 trials, output runs/20260910_online_query_within_it5/<corpus>/seed<seed>/
set -euo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; seed="$2"
root="runs/20260910_online_query_within_it5/$corpus/seed$seed"
mkdir -p "$root"
case "$corpus:$seed" in
  hatemm:234)      enq='{"lr": 0.0006502668508138526, "max_seqlen": 300, "lamda_cma": 1.984429561771332, "prior_scale": 0.6452966052619578, "lambda_block": 0.2748876745915947}'; n=12 ;;
  hatemm:2025)     enq='{"lr": 0.0003335398420162505, "max_seqlen": 200, "lamda_cma": 0.9676170139917135, "prior_scale": 0.6208915817770003, "lambda_block": 1.590355849092756}'; n=12 ;;
  hatemm:3407)     enq='{"lr": 0.00024202125311385616, "max_seqlen": 200, "lamda_cma": 1.6782932252972214, "prior_scale": 1.356845735725023, "lambda_block": 0.3399502890267682}'; n=12 ;;
  hateclipseg:234) enq='{"lr": 0.00018256101272761348, "max_seqlen": 300, "lamda_cma": 0.8787626624611624, "prior_scale": 5.309170664597484, "lambda_block": 0.14930992714184946}'; n=15 ;;
  hateclipseg:2025) enq='{"lr": 0.00010070085047170324, "max_seqlen": 300, "lamda_cma": 0.5072311493448376, "prior_scale": 2.7657908846890615, "lambda_block": 0.19328944700770595}'; n=15 ;;
  hateclipseg:3407) enq='{"lr": 0.0001391334590778541, "max_seqlen": 200, "lamda_cma": 0.8275215385476362, "prior_scale": 3.6479203634129167, "lambda_block": 0.4200747308657341}'; n=15 ;;
  *) echo "unknown corpus/seed $corpus $seed" >&2; exit 2 ;;
esac
[ -f "$root/budget.json" ] || echo "{\"n_trials\": $n, \"first_trial_seconds\": null, \"note\": \"iteration 5 pre-set budget\"}" > "$root/budget.json"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$HOME/miniconda3/envs/HateVideo/bin/python" -u experiments/20260910_online_query_within/search.py --corpus "$corpus" --seed "$seed" \
  --out-root "runs/20260910_online_query_within_it5" --num-workers 4 --enqueue-json "$enq"
