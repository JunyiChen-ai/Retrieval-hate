#!/usr/bin/env bash
# REPRO campaign Phase A step 4: backfill MHC-EN dev/test ASR (whisper-large-v3,
# K=4 sub-clip windows, M=16 -- same contract as the existing MHC train file).
# Waits for the whisper download and for the dense-feature extraction to finish
# so the two never share the GPU.
set -u
cd /home/jehc223/Retrieval-hate
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
export PYTHONUNBUFFERED=1
echo "[run] start $(date -Is)"
for tag in repro_whisper_dl repro_extract; do
  p=$(cat "logging/runs/$tag/run.pid" 2>/dev/null || echo "")
  if [ -n "$p" ]; then
    echo "[wait] $tag pid=$p"
    while kill -0 "$p" 2>/dev/null; do sleep 30; done
    echo "[wait] $tag finished $(date -Is)"
  fi
done
$PY src/utils/generate_segment_asr_HF.py --dataset MHC --splits val,test \
    --num_frames 16 --num_subclips 4 --model openai/whisper-large-v3 \
    --language auto --timestamps word --batch_size 8
echo "[run] MHC asr exit=$?"
echo "[run] done $(date -Is)"
