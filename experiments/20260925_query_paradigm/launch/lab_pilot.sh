#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_pilot.sh <corpus> \
#            > runs/20260925_query_paradigm/pilot_context/<corpus>_run.log 2>&1 &
# README section 11 pilot: context question on sampled short test nodes (vLLM, env ~/miniconda3/envs/vlm)
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260925_query_paradigm/pilot_context/$1_run.pid"
"$HOME/miniconda3/envs/vlm/bin/python" experiments/20260925_query_paradigm/pilot_context_prompt.py run --corpus "$1" \
  --gpu-mem "${GPU_MEM:-0.75}"
