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
smoke_rc=$?
echo "=== smoke done rc=$smoke_rc $(date -Is) ==="
# Check it, do not just print it (campaign ruling 2026-08-19): a swallowed stage
# failure is what let the LAVAD chain exit 0 having written 5 curves.
if [ "$smoke_rc" -ne 0 ]; then
  echo "!! smoke failed rc=$smoke_rc - refusing to start the corpus run"; exit 1
fi

MAX_RESTARTS=${MAX_RESTARTS:-80}
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  echo "=== attempt $attempt $(date -Is) ==="
  $PY -u scripts/repro_campaign/run_av2a.py \
      --datasets HateMM,MHC,MHC_zh,HateClipSeg --progress-every 25
  rc=$?
  echo "=== driver exited rc=$rc $(date -Is) ==="
  if [ "$rc" -eq 0 ]; then
    $PY -u scripts/repro_campaign/run_av2a.py --build-intervals
    # Completeness guard: rc=0 is not evidence the corpus was processed.  The
    # eval watcher triggers on "RUN COMPLETE", so an empty run would otherwise
    # be scored and written up as a result.  Expected drops are the 17 HateMM
    # videos with no audio or no video stream and the 1 truncated HateClipSeg
    # container; anything beyond a 2% margin on top of those is a truncation.
    if ! /home/jehc223/miniconda3/envs/HateVideo/bin/python - <<'GUARD'
import glob, sys, numpy as np
ALLOW = {"HateMM": 17, "MHC": 0, "MHC_zh": 0, "HateClipSeg": 1}
bad = []
for ds, drop in ALLOW.items():
    exp = len(np.load(f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)["video_ids"]) - drop
    got = len(glob.glob(f"idea-stage/repro_av2a/curves/{ds}/*.npz"))
    print(f"  {ds:<12} curves={got}/{exp}")
    if got < exp * 0.98:
        bad.append(f"{ds}: {got}/{exp}")
if bad:
    sys.exit("FAIL - short of curves: " + "; ".join(bad))
print("  artifact counts OK")
GUARD
    then
      echo "!! completeness guard failed - NOT emitting RUN COMPLETE"; exit 1
    fi
    echo "=== RUN COMPLETE ==="
    exit 0
  fi
  sleep 10
done
echo "=== GAVE UP after $MAX_RESTARTS attempts ==="
exit 1
