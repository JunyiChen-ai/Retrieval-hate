#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_asr_words.sh <corpus:shard:splits> ... \
#            > runs/20260927_query_paradigm_r4/asr/launch_$(hostname).out 2>&1 &
# README section 14: word-level Whisper transcripts (scripts/asr_words.py), one job after another, e.g.
#   hatemm:0/2:test hateclipseg:0/2:test dehate:0/2:test hatemm:0/2:val,train ...
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260927_query_paradigm_r4/asr/launch_$(hostname).pid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for job in "$@"; do
  IFS=: read -r c s sp <<< "$job"
  "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/asr_words.py --corpus "$c" --shard "$s" --splits "$sp" \
    >> "runs/20260927_query_paradigm_r4/asr/${c}_${s//\//of}_$(hostname).log" 2>&1
done
echo ALL_DONE
