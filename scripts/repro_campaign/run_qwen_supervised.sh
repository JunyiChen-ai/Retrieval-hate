#!/bin/bash
# Supervisor for the Qwen2.5-VL grounding corpus run.
#
# A handful of released files are truncated mid-stream and the C video decoder does
# not raise on them -- it takes the process down with no Python traceback.  The
# driver is idempotent and records an in-flight marker, so the correct response is
# to restart it: the marker tells the next attempt which id killed the previous one,
# that id is retired as a decode failure, and the run continues.  The loop exits as
# soon as the driver exits cleanly.
set -u
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
cd /home/jehc223/Retrieval-hate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_RESTARTS=${MAX_RESTARTS:-60}
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== attempt $attempt $(date -Is) ==="
  python -u scripts/repro_campaign/run_qwen_grounding.py \
      --datasets HateMM,MHC,MHC_zh,HateClipSeg --hcs-classes --load-4bit
  rc=$?
  echo "=== driver exited rc=$rc $(date -Is) ==="
  if [ "$rc" -eq 0 ]; then echo "=== RUN COMPLETE ==="; exit 0; fi
  sleep 10
done
echo "=== GAVE UP after $MAX_RESTARTS attempts ==="
exit 1
