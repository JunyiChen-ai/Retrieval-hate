#!/usr/bin/env bash
# usage (repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/run_controls_r4.sh <corpus> <seed> ... \
#            > runs/20260927_query_paradigm_r4/diag/launch_controls_<corpus>_<host>.out 2>&1 &
# README section 15.2 controls (default hyperparameters, no search), seeds one after another, via run_diag_r4.sh:
# abl_full (revision 4), abl_no_node (no node potentials), abl_noback (no content backbone), abl_lvl{4,8,16} (questions
# from one tree depth only); all with the word-transcript answers.
set -uo pipefail
cd "$HOME/Retrieval-hate"
corpus="$1"; shift
for seed in "$@"; do
  bash experiments/20260925_query_paradigm/launch/run_diag_r4.sh "$corpus" "$seed" \
    'abl_full={"answer_source": "words", "node_prior": true}' \
    'abl_no_node={"answer_source": "words"}' \
    'abl_noback={"answer_source": "words", "backbone": "const", "lamda_cma": 0}' \
    'abl_lvl4={"answer_source": "words", "node_prior": true, "query_level": 4}' \
    'abl_lvl8={"answer_source": "words", "node_prior": true, "query_level": 8}' \
    'abl_lvl16={"answer_source": "words", "node_prior": true, "query_level": 16}'
done
echo "$(date) CONTROLS DONE $corpus"
