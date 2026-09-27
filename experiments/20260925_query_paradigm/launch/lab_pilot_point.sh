#!/usr/bin/env bash
# usage (lab machine, repo root): setsid nohup bash experiments/20260925_query_paradigm/launch/lab_pilot_point.sh \
#            > runs/20260928_query_paradigm_r5_analysis/pilot_point/run_$(hostname).log 2>&1 &
# README section 16.4: "point to the harmful lines / frames" question on sampled test nodes of HateMM and
# HateClipSeg (vLLM, env ~/miniconda3/envs/vlm); jobs built on uoa-lab2 and copied here.
set -euo pipefail
cd "$HOME/Retrieval-hate"
hostname; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
echo $$ > "runs/20260928_query_paradigm_r5_analysis/pilot_point/run_$(hostname).pid"
for c in hatemm hateclipseg; do
  "$HOME/miniconda3/envs/vlm/bin/python" -u experiments/20260925_query_paradigm/pilot_point_prompt.py run --corpus "$c" \
    --gpu-mem "${GPU_MEM:-0.6}"
done
echo ALL_DONE
