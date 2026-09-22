#!/usr/bin/env bash
# usage: cd ~/Retrieval-hate && setsid nohup bash experiments/20260910_online_query_within/launch/run_it5_ablations.sh <seed> \
#            > runs/20260910_online_query_within_it5/launch_ablations_seed<seed>.out 2>&1 &
# README section 13 (2026-09-23): ablation table of the default method (iteration 5), both corpora, one seed per
# machine, 3 runs in parallel on the GPU. No search: every arm uses the seed's best-trial hparams
# (runs/20260910_online_query_within_it5/diag/<corpus>/seed<seed>/hparams_trial*.json) plus the arm's overrides.
# Only the eoc and uniform test policies are evaluated (the other policies are curves only, evaluated after
# training; they do not change the eoc operating point). Output per arm:
#   runs/20260910_online_query_within_it5/diag/<corpus>/seed<seed>/<arm>/{summary.json,run.log,stdout.log,...}
# Chain logs: runs/20260910_online_query_within_it5/diag/<corpus>/seed<seed>/chain.log (FAILED lines on errors);
# DONE marker: runs/20260910_online_query_within_it5/ablations_seed<seed>.DONE
set -euo pipefail
cd "$HOME/Retrieval-hate"
PARALLEL=${PARALLEL:-3}
POL='"policies": ["eoc", "uniform"]'
# tag:ablation:{json overrides}   (HateMM first: its runs are the longest)
SPECS=(
  "full_rerun:full:{$POL}"
  "no_verdict:no_verdict:{$POL}"
  "coarse_only:coarse_only:{$POL}"
  "fixed_uniform_train:fixed_uniform_train:{$POL}"
  "no_dropout:no_dropout:{$POL}"
  "no_missing_state:no_missing_state:{$POL}"
  "no_hmm:no_hmm:{$POL}"
  "seconds_time:full:{$POL, \"normalized_time\": false}"
  "no_constraint:full:{$POL, \"positive_constraint\": false}"
  "no_decomp:no_decomp:{$POL}"
  "no_video_term:no_video_term:{$POL}"
  "no_text_term:no_text_term:{$POL}"
  "avce:avce:{$POL}"
  "no_qk_enc:no_qk_enc:{$POL}"
  "no_cell:no_cell:{$POL}"
  "no_bias:full:{$POL, \"bias_mode\": \"none\"}"
  "no_context:full:{$POL, \"ctx_mode\": \"none\"}"
  "no_cmal:full:{$POL, \"lamda_cma\": 0.0}"
  "no_block:no_block:{$POL}"
  "no_window_loss:no_window_loss:{$POL}"
  "no_prior:no_prior:{$POL}"
)
CORPORA=(hatemm hateclipseg)

if [ "${1:-}" = "--job" ]; then          # worker: --job <index> <seed>
  i="$2"; seed="$3"
  n=${#SPECS[@]}
  corpus=${CORPORA[$((i / n))]}
  spec=${SPECS[$((i % n))]}
  diag="runs/20260910_online_query_within_it5/diag/$corpus/seed$seed"
  hp=$(ls "$diag"/hparams_trial*.json 2>/dev/null || true)
  [ -n "$hp" ] && [ "$(echo "$hp" | wc -l)" -eq 1 ] || { echo "$(date) FAILED $corpus seed$seed: need exactly one hparams_trial*.json" >> "$diag/chain.log"; exit 0; }
  bash experiments/20260910_online_query_within/launch/run_diag.sh "$corpus" "$seed" "$hp" _it5 "$spec"
  exit 0
fi

seed="$1"
echo "$(date) $(hostname) seed $seed: ${#SPECS[@]} arms x ${#CORPORA[@]} corpora, $PARALLEL in parallel"
echo $$ > "runs/20260910_online_query_within_it5/ablations_seed$seed.pid"
total=$(( ${#SPECS[@]} * ${#CORPORA[@]} ))
seq 0 $((total - 1)) | xargs -P "$PARALLEL" -I{} bash "$0" --job {} "$seed"
echo "$(date) DONE" | tee "runs/20260910_online_query_within_it5/ablations_seed$seed.DONE"
