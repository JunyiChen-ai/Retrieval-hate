#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_r4_inputs.sh <shard i/n> \
#            > runs/20260927_query_paradigm_r4/launch_inputs_$(hostname).out 2>&1 &
# README section 14: (1) test answers with word-timestamp transcripts for the three corpora, then (2) word-level ASR
# of val and train. Needs data/vlm_tree/<C>/manifest_words.jsonl (built on uoa-lab2 from the test words).
set -euo pipefail
cd "$HOME/Retrieval-hate"
s="$1"
bash experiments/20260925_query_paradigm/launch/lab_extract_words.sh hatemm:$s:test hateclipseg:$s:test dehate:$s:test
bash experiments/20260925_query_paradigm/launch/lab_asr_words.sh hatemm:$s:val,train hateclipseg:$s:val,train dehate:$s:val,train
echo INPUTS_DONE
