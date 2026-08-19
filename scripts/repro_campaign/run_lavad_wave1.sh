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
# A stage that fails must stop the chain; see the 19:51 collapse, where an
# OOM in stage 04a was followed by stages that ran on 6 videos and exited 0.
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
# The batch-64 OOM at 19:56 reported 6.35 GiB reserved-but-unallocated, i.e.
# fragmentation rather than a real shortage; expandable segments remove it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT=${1:-test}
STEP=${2:-1}
DS=HateMM,MHC,MHC_zh,HateClipSeg

echo "=== [$(date -Is)] stage 01 caption (BLIP-2 blip2-opt-6.7b-coco, 1 fps)"
$PY scripts/repro_campaign/blip2_caption.py --datasets "$DS" --split "$SPLIT" \
    --batch-size 48 || { rc=$?; echo "!! caption rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] stage 02+03 index + clean captions (ImageBind)"
$PY scripts/repro_campaign/lavad_chain.py clean --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || { rc=$?; echo "!! clean rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] stage 04a temporal summaries (Llama-2-13b-chat NF4)"
$PY scripts/repro_campaign/lavad_chain.py summarize --datasets "$DS" \
    --split "$SPLIT" --center-step "$STEP" --batch-size 48 || { rc=$?; echo "!! summarize rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] stage 04b anomaly scores, verbatim prompt"
$PY scripts/repro_campaign/lavad_chain.py score --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" --batch-size 48 --prompt verbatim || { rc=$?; echo "!! score rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] stage 05+06 summary index + refined scores (ImageBind)"
$PY scripts/repro_campaign/lavad_chain.py refine --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || { rc=$?; echo "!! refine rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] curves"
$PY scripts/repro_campaign/lavad_chain.py curves --datasets "$DS" --split "$SPLIT" \
    --center-step "$STEP" || { rc=$?; echo "!! curves rc=$rc -- stopping the chain"; exit "$rc"; }

echo "=== [$(date -Is)] LAVAD chain finished"
