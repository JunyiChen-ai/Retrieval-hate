#!/usr/bin/env bash
# LaGoVAD: wait for the feature extraction to finish, then inference, then evaluate.
#
# fail-fast on purpose (campaign ruling 2026-08-19).  The LAVAD chain exited rc=0
# having written 5 curves for one dataset because a stage failure was swallowed
# and the chain marched on.  A chain that reports success it did not earn is
# worse than one that stops, so: -e aborts on any failing stage, -o pipefail
# stops a failure being hidden behind a pipe, and the guard below refuses to
# finish while any dataset has produced no curves at all.
set -euo pipefail
R=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXTRACT_PID=$(cat $R/logging/runs/repro_lagovad_extract/run.pid | head -1)
while kill -0 "$EXTRACT_PID" 2>/dev/null; do sleep 60; done
echo "[chain] extraction finished $(date -Is)"
# Deliberately outside the GPU queue: the LaGoVAD head is a 2-layer temporal
# transformer over cached 512-d features, ~1.5 GiB and a few minutes for the whole
# corpus. Queueing it behind a multi-hour captioning job would hold up the whole
# Wave 1 table for no memory benefit.
$PY $R/scripts/repro_campaign/run_lagovad.py --stage infer
echo "[chain] inference finished $(date -Is)"
for SP in test all; do
  $PY $R/scripts/repro_campaign/eval_frame.py --method curves \
    --curve-dir $R/idea-stage/repro_lagovad/curves --method-name LaGoVAD \
    --variants main,sens_short,sens_vad,normal,main_pair,main_vsnormal,sens_short_pair,sens_short_vsnormal,sens_vad_pair,sens_vad_vsnormal,bin \
    --wave 1 --supervision aux-temporal-pretrain --split $SP \
    --out $R/idea-stage/repro_campaign/eval_lagovad_$SP.json
  $PY $R/scripts/repro_campaign/eval_frame.py --method curves \
    --curve-dir $R/idea-stage/repro_lagovad/curves --method-name LaGoVAD \
    --variants c0_normal,c1_hateful,c2_insulting,c3_sexual,c4_violence,c5_harm \
    --datasets HateClipSeg --wave 1 --supervision aux-temporal-pretrain --split $SP \
    --out $R/idea-stage/repro_campaign/eval_lagovad_hcs6_$SP.json
done
echo "[chain] done $(date -Is)"

# --- completeness guard: never report success on an empty or near-empty run ---
echo "[chain] verifying artifact counts"
$PY - <<'GUARD'
import glob, sys, numpy as np
bad = []
for ds in ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]:
    exp = len(np.load(f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)["video_ids"])
    got = len(glob.glob(f"idea-stage/repro_lagovad/curves/{ds}/*.npz"))
    print(f"  {ds:<12} curves={got}/{exp}")
    # >2% of a corpus missing is a truncation, not the handful of genuinely
    # undecodable containers (3 across all four datasets as of 2026-08-19)
    if got < exp * 0.98:
        bad.append(f"{ds}: {got}/{exp}")
if bad:
    sys.exit("[chain] FAIL - datasets short of curves: " + "; ".join(bad))
print("[chain] artifact counts OK")
GUARD
echo "[chain] done $(date -Is)"
