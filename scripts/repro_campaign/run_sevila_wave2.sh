#!/usr/bin/env bash
# REPRO campaign Wave 2 — SeViLA Localizer, test split, all four datasets.
#
# Frozen before launch (freeze §5, §10):
#   * question texts and prompt template  : run_sevila.py module constants
#   * sampling rate                       : 1.0 fps, the campaign's frozen 1 fps grid
#   * read-out                            : repo `yes_score`; `*_margin` extra row
#   * free knobs                          : none (nothing is chosen on val)
#   * budget rule                         : test split runs at 1 fps unconditionally
#       (67,647 frames of vision, 237k prompt passes -> a low single-digit GPU-hour
#        job); the optional full-corpus block is a separate later launch and is
#        skipped if the measured throughput puts it over 12 GPU-hours.
#
# Takes the campaign GPU lock, runs the GPU self-test first (shapes / score range /
# equality against the unmodified published `generate`), then the corpus.
set -u
cd /home/jehc223/Retrieval-hate

export PYTHONPATH=/home/jehc223/Retrieval-hate/third_party/SeViLA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/jehc223/Retrieval-hate/third_party/_venv/sevila/bin/python

echo "=== SeViLA Wave 2 launch $(date -Is) ==="
scripts/repro_campaign/gpu_queue.sh sevila bash -c "
  set -u
  echo '--- gpu selftest ---'
  $PY scripts/repro_campaign/run_sevila.py --selftest --device cuda --chunk 32 || exit 1
  echo '--- corpus: test split ---'
  $PY scripts/repro_campaign/run_sevila.py --split test --chunk 32 \
      --datasets HateMM,MHC,MHC_zh,HateClipSeg
"
echo "=== SeViLA Wave 2 exit rc=$? $(date -Is) ==="
