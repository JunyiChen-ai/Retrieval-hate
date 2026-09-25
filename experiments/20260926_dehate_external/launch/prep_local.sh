#!/usr/bin/env bash
# DeHate preparation stages on uoa-lab2 (README section 2). One stage per call, resumable:
#   setsid nohup bash experiments/20260926_dehate_external/launch/prep_local.sh <stage> \
#       > runs/20260926_dehate_external/prep/<stage>.log 2>&1 &
# stages: asr_k30 (data/ASR/DeHate, word-level Whisper for the VLM questions), asr_chunks (results/reproduction/asr/
# dehate_all, the timestamped chunks behind the BERT rows and the 1-fps durations), clip | vit | i3d | vggish | bert
# (results/reproduction/features/<type>/dehate), gt (results/reproduction/gt/dehate_{val,test}.npz).
set -uo pipefail
cd "$HOME/Retrieval-hate"
export HVD_DATA_ROOT=/home/jehc223/data
export TOKENIZERS_PARALLELISM=false
PY=$HOME/miniconda3/envs/HateVideo/bin/python
stage="$1"
echo "$(date -Is) $(hostname) stage $stage"
case "$stage" in
  asr_k30)
    $PY -u src/utils/generate_segment_asr_HF.py --dataset DeHate --num_subclips 30 --splits train,val,test \
        --gt_dir ./data/gt --video_dir ./data/video --out_dir ./data/ASR ;;
  asr_chunks)
    rc=1
    for attempt in 1 2 3; do
      $PY -u scripts/duplex/interleaved_timeline_asr.py --corpus dehate_all --out-root results/reproduction/asr
      rc=$?; [ "$rc" -eq 0 ] && break
      echo "attempt $attempt rc=$rc; resuming in 20 s"; sleep 20
    done
    [ "$rc" -eq 0 ] || exit "$rc" ;;
  clip|vit|i3d|vggish|bert)
    bash scripts/duplex/run_reproduction_features.sh "$stage" dehate ;;
  gt)
    $PY -u scripts/dehate/prepare_dehate.py gt ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
rc=$?
if [ "$rc" -le 1 ]; then echo "$(date -Is) DONE $stage rc=$rc" | tee "runs/20260926_dehate_external/prep/$stage.DONE"
else echo "$(date -Is) FAILED $stage rc=$rc" | tee "runs/20260926_dehate_external/prep/$stage.FAILED"; fi
