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
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
SPLIT=${1:-test}
STEP=${2:-10}
DS=HateMM,MHC,MHC_zh,HateClipSeg

echo "=== [$(date -Is)] URF 1 caption (VideoLLaMA3-7B, ${STEP}s centers, 10s window)"
$PY scripts/repro_campaign/urf_chain.py caption --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || echo "!! caption rc=$?"

echo "=== [$(date -Is)] URF 2 round-1 scores (Llama-3.1-8B-Instruct)"
$PY scripts/repro_campaign/urf_chain.py score --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" --batch-size 32 || echo "!! score rc=$?"

echo "=== [$(date -Is)] URF 3 sliding-window highest/lowest intervals (CPU)"
$PY scripts/repro_campaign/urf_chain.py filter --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || echo "!! filter rc=$?"

echo "=== [$(date -Is)] URF 4 suspicious-phrase tags (VideoLLaMA3-7B)"
$PY scripts/repro_campaign/urf_chain.py tags --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || echo "!! tags rc=$?"

echo "=== [$(date -Is)] URF 5 gated refinement (Llama-3.1-8B-Instruct)"
$PY scripts/repro_campaign/urf_chain.py refine --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" --batch-size 32 || echo "!! refine rc=$?"

echo "=== [$(date -Is)] curves"
$PY scripts/repro_campaign/urf_chain.py curves --datasets "$DS" --split "$SPLIT" \
    --step "$STEP" || echo "!! curves rc=$?"

echo "=== [$(date -Is)] URF chain finished"
