#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_extract.sh <corpus> <splits> <shard i/n> \
#            > runs/20260925_query_paradigm/extract/<corpus>_<shard>_<host>.log 2>&1 &
# README section 2.1: query-tree answers with Qwen2.5-VL-7B (vLLM, env ~/miniconda3/envs/vlm)
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260925_query_paradigm/extract/$1_${3//\//of}_$(hostname).pid"
"$HOME/miniconda3/envs/vlm/bin/python" experiments/20260925_query_paradigm/extract_tree_answers.py \
  --corpus "$1" --splits "$2" --shard "$3" --gpu-mem 0.85
