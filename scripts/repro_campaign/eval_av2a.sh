#!/bin/bash
# Score the AV²A run through the shared evaluator (freeze §2) — CPU only.
#
# Six variants, three channels x {continuous per-second similarity, rasterised
# AV²A event decisions}, on both the test split (headline) and the full corpus.
# The evt_* variants also carry an interval file, so F1@tIoU is reported for them
# and reads n/a for the sim_* curves, exactly as freeze §2 requires.
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
VARIANTS=sim_video,sim_audio,sim_combined,evt_video,evt_audio,evt_combined

# intervals are built from the raw event jsonl; harmless to repeat
third_party/_venv/av2a/bin/python -u scripts/repro_campaign/run_av2a.py --build-intervals

for split in test all; do
  $PY -u scripts/repro_campaign/eval_frame.py \
      --method curves \
      --curve-dir idea-stage/repro_av2a/curves \
      --method-name "AV2A" \
      --variants "$VARIANTS" \
      --wave 1 --supervision label-free --split "$split" \
      --out "idea-stage/repro_campaign/eval_AV2A_${split}.json"
done
echo "=== EVAL COMPLETE $(date -Is) ==="
