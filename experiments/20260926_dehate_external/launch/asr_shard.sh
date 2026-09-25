#!/usr/bin/env bash
# Sentence-level Whisper large-v3 transcripts of DeHate (README section 3), one shard per machine:
#   setsid nohup bash experiments/20260926_dehate_external/launch/asr_shard.sh <i/n> \
#       > runs/20260926_dehate_external/prep/asr_shard<i>of<n>_<host>.log 2>&1 &
# Needs data/AV2A_wav/DeHate and results/reproduction/splits/dehate_*.txt on the machine. Output:
# results/reproduction/asr/dehate_all/timestamped_chunks.shard<i>of<n>.jsonl (concatenated on uoa-lab2 afterwards).
set -uo pipefail
cd "$HOME/Retrieval-hate"
export TOKENIZERS_PARALLELISM=false
echo "$(date -Is) $(hostname) asr shard $1"
tag="${1//\//of}"
rc=1
for attempt in 1 2 3; do
  "$HOME/miniconda3/envs/HateVideo/bin/python" -u scripts/duplex/interleaved_timeline_asr.py \
      --corpus dehate_all --out-root results/reproduction/asr --shard "$1"
  rc=$?; [ "$rc" -eq 0 ] && break
  echo "attempt $attempt rc=$rc; resuming in 20 s"; sleep 20
done
if [ "$rc" -eq 0 ]; then echo "$(date -Is) DONE" | tee "runs/20260926_dehate_external/prep/asr_shard${tag}.DONE"
else echo "$(date -Is) FAILED rc=$rc" | tee "runs/20260926_dehate_external/prep/asr_shard${tag}.FAILED"; fi
