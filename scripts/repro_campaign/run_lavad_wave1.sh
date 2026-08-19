#!/usr/bin/env bash
# REPRO campaign Wave 1 — the whole LAVAD chain in one GPU-queue slot.
#
# Held under a single `gpu_queue.sh blip2_caption` acquisition on purpose: the
# stages are strictly sequential and each loads a different large model, so
# releasing the lock between them only invites another job in and stalls the run.
# Every stage is idempotent, so re-running this script resumes.
#
#   bash scripts/repro_campaign/run_lavad_wave1.sh [SPLIT] [CENTER_STEP]
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
SPLIT=${1:-test}
STEP=${2:-1}
DS=HateMM,MHC,MHC_zh,HateClipSeg

echo "=== [$(date -Is)] stage 01 caption (BLIP-2 blip2-opt-6.7b-coco, 1 fps)"
$PY scripts/repro_campaign/blip2_caption.py --datasets "$DS" --split "$SPLIT" \
    --batch-size 64 || echo "!! caption rc=$?"

echo "=== [$(date -Is)] stage 02+03 index + clean captions (ImageBind)"
$PY scripts/repro_campaign/lavad_chain.py clean --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || echo "!! clean rc=$?"

echo "=== [$(date -Is)] stage 04a temporal summaries (Llama-2-13b-chat NF4)"
$PY scripts/repro_campaign/lavad_chain.py summarize --datasets "$DS" \
    --split "$SPLIT" --center-step "$STEP" --batch-size 64 || echo "!! summarize rc=$?"

echo "=== [$(date -Is)] stage 04b anomaly scores, verbatim prompt"
$PY scripts/repro_campaign/lavad_chain.py score --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" --batch-size 64 --prompt verbatim || echo "!! score rc=$?"

echo "=== [$(date -Is)] stage 05+06 summary index + refined scores (ImageBind)"
$PY scripts/repro_campaign/lavad_chain.py refine --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || echo "!! refine rc=$?"

echo "=== [$(date -Is)] curves"
$PY scripts/repro_campaign/lavad_chain.py curves --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || echo "!! curves rc=$?"

echo "=== [$(date -Is)] LAVAD chain finished"
