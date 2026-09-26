#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_asr_words.sh <corpus:shard> ... \
#            > runs/20260927_query_paradigm_r4/asr/launch_$(hostname).out 2>&1 &
# README section 14: word-level Whisper transcripts (scripts/asr_words.py), one corpus shard after another.
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260927_query_paradigm_r4/asr/launch_$(hostname).pid"
for job in "$@"; do
  c="${job%%:*}"; s="${job#*:}"
  "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/asr_words.py --corpus "$c" --shard "$s" \
    > "runs/20260927_query_paradigm_r4/asr/${c}_${s//\//of}_$(hostname).log" 2>&1
done
echo ALL_DONE
