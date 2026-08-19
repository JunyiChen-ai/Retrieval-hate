#!/usr/bin/env bash
# REPRO campaign Wave 1 — the whole URF-HVAA chain in one GPU-queue slot.
#
#   bash scripts/repro_campaign/run_urf_wave1.sh [SPLIT] [STEP_SECONDS]
#
# STEP is the center spacing.  URF's own step is 16 native frames (0.533 s at
# 30 fps) with a 10 s window; at 10 s the windows tile the video exactly once,
# which is the coarsest faithful choice and the only one that fits one GPU for
# 84 h of video.  `native_rate` in the result table is 1/STEP.
# Every stage is idempotent, so re-running this script resumes.
set -u
# A stage that fails must stop the chain; see the 19:51 collapse, where an
# OOM in stage 04a was followed by stages that ran on 6 videos and exited 0.
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT=${1:-test}
STEP=${2:-10}
DS=HateMM,MHC,MHC_zh,HateClipSeg

echo "=== [$(date -Is)] URF 1 caption (VideoLLaMA3-7B, ${STEP}s centers, 10s window)"
$PY scripts/repro_campaign/urf_chain.py caption --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || { rc=$?; echo "!! caption rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] URF 2 round-1 scores (Llama-3.1-8B-Instruct)"
$PY scripts/repro_campaign/urf_chain.py score --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" --batch-size 32 || { rc=$?; echo "!! score rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] URF 3 sliding-window highest/lowest intervals (CPU)"
$PY scripts/repro_campaign/urf_chain.py filter --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || { rc=$?; echo "!! filter rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] URF 4 suspicious-phrase tags (VideoLLaMA3-7B)"
$PY scripts/repro_campaign/urf_chain.py tags --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || { rc=$?; echo "!! tags rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] URF 5 gated refinement (Llama-3.1-8B-Instruct)"
$PY scripts/repro_campaign/urf_chain.py refine --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" --batch-size 32 || { rc=$?; echo "!! refine rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] curves"
$PY scripts/repro_campaign/urf_chain.py curves --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || { rc=$?; echo "!! curves rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] URF chain finished"
