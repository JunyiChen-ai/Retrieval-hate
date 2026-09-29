#!/usr/bin/env bash
# usage (lab machine, repo root): mkdir -p runs/20260929_query_paradigm_r5/extract && \
#   setsid nohup bash experiments/20260925_query_paradigm/launch/lab_extract_soft.sh <corpus:shard:splits:view> ... \
#     > runs/20260929_query_paradigm_r5/extract/launch_$(hostname).out 2>&1 &
# README section 17.1 / 17.3: soft first-token Yes/No answers under per-dataset definitions (extract_tree_soft.py),
# one job after another; view = both | frames | text. Output data/vlm_tree/<C>/soft_<view>_p1[.shard].jsonl.
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260929_query_paradigm_r5/extract/launch_$(hostname).pid"
for job in "$@"; do
  IFS=: read -r c s sp v <<< "$job"
  "$HOME/miniconda3/envs/vlm/bin/python" -u experiments/20260925_query_paradigm/extract_tree_soft.py \
    --corpus "$c" --splits "$sp" --shard "$s" --view "$v" --gpu-mem "${GPU_MEM:-0.75}" --prompt "${PROMPT:-p1}" \
    >> "runs/20260929_query_paradigm_r5/extract/${c}_${v}_${s//\//of}_$(hostname).log" 2>&1
done
echo ALL_DONE
