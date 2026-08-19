#!/bin/bash
# Supervisor for the AV²A corpus run (REPRO campaign Wave 1).
#
# Same reason as the Qwen2.5-VL run: a few released containers are truncated
# mid-stream and the C video decoder does not raise on them, it takes the process
# down with no Python traceback.  The driver is idempotent and writes an in-flight
# marker, so a restart retires exactly the id that killed the previous attempt as a
# decode failure and carries on.  Exits as soon as the driver exits cleanly.
set -u
cd /home/jehc223/Retrieval-hate
PY=third_party/_venv/av2a/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smoke pass first: two videos per dataset, one PROGRESS line each, inside the
# same GPU-lock acquisition as the corpus run so the smoke does not cost a whole
# extra turn of a queue whose other jobs run for ten hours.  The driver is
# idempotent, so these eight videos are simply the first eight of the corpus run.
echo "=== smoke (2 videos per dataset) $(date -Is) ==="
$PY -u scripts/repro_campaign/run_av2a.py \
    --datasets HateMM,MHC,MHC_zh,HateClipSeg --limit 2 --progress-every 1
echo "=== smoke done rc=$? $(date -Is) ==="

MAX_RESTARTS=${MAX_RESTARTS:-80}
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== attempt $attempt $(date -Is) ==="
  $PY -u scripts/repro_campaign/run_av2a.py \
      --datasets HateMM,MHC,MHC_zh,HateClipSeg --progress-every 25
  rc=$?
  echo "=== driver exited rc=$rc $(date -Is) ==="
  if [ "$rc" -eq 0 ]; then
    $PY -u scripts/repro_campaign/run_av2a.py --build-intervals
    echo "=== RUN COMPLETE ==="
    exit 0
  fi
  sleep 10
done
echo "=== GAVE UP after $MAX_RESTARTS attempts ==="
exit 1
