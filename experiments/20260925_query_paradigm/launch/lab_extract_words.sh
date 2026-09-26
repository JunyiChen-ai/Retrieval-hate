#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_extract_words.sh <corpus:shard:splits> ... \
#            > runs/20260927_query_paradigm_r4/extract/launch_$(hostname).out 2>&1 &
# README section 14: the revision-3 questions asked again with word-timestamp transcripts
# (data/vlm_tree/<C>/manifest_words.jsonl -> answers_words_qwen7b_mod5[.shard<i>of<n>].jsonl), one job after another.
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260927_query_paradigm_r4/extract/launch_$(hostname).pid"
for job in "$@"; do
  IFS=: read -r c s sp <<< "$job"
  "$HOME/miniconda3/envs/vlm/bin/python" -u experiments/20260925_query_paradigm/extract_tree_answers.py \
    --corpus "$c" --splits "$sp" --shard "$s" --gpu-mem "${GPU_MEM:-0.75}" \
    --manifest manifest_words.jsonl --out-prefix answers_words_qwen7b_mod5 \
    >> "runs/20260927_query_paradigm_r4/extract/${c}_${s//\//of}_$(hostname).log" 2>&1
done
echo ALL_DONE
