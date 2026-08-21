#!/usr/bin/env bash
# REPRO campaign Wave 1 — UniTime corpus run (resume-safe).
#
# Restarted 2026-08-21 after the 2026-08-20 attempt died at model load with
# CUDA OOM: the AV2A worker was still holding 18.6 GiB when UniTime's queue slot
# opened.  `expandable_segments` is added here because the OOM message asked for
# it; the real fix is that the card is now empty.
#
# The driver resumes from its own JSONL (`done_ids`), so re-running is idempotent.
set -u
cd /home/jehc223/Retrieval-hate
LOG=logging/runs/repro_unitime/run.log
mkdir -p "$(dirname "$LOG")"

source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

exec scripts/repro_campaign/gpu_queue.sh unitime \
  python scripts/repro_campaign/run_unitime.py \
    --datasets HateMM,MHC,MHC_zh,HateClipSeg \
    --hcs-classes
