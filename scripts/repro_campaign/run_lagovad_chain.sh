#!/usr/bin/env bash
# LaGoVAD: wait for the feature extraction to finish, then inference, then evaluate.
set -u
R=/home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
EXTRACT_PID=$(cat $R/logging/runs/repro_lagovad_extract/run.pid | head -1)
while kill -0 "$EXTRACT_PID" 2>/dev/null; do sleep 60; done
echo "[chain] extraction finished $(date -Is)"
bash $R/scripts/repro_campaign/gpu_queue.sh lagovad_infer \
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
