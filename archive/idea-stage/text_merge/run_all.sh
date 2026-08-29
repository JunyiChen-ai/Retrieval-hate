#!/usr/bin/env bash
# TEXT_MERGE end-to-end driver: extract -> assemble -> 12 head runs -> analyze.
# Single background submission (CLAUDE.md single-GPU branch; freeze section 5).
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNDIR=logging/runs/text_merge
mkdir -p "$RUNDIR/logs"

echo "STAGE=extract START $(date -Is)"
python idea-stage/text_merge/extract_text_feats.py --offload_gib "${OFFLOAD_GIB:-9}"
RC=$?; echo "STAGE=extract RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC

echo "STAGE=assemble START $(date -Is)"
python idea-stage/text_merge/extract_text_feats.py --assemble
RC=$?; echo "STAGE=assemble RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC

echo "STAGE=train START $(date -Is)"
bash idea-stage/text_merge/run_arms.sh "$RUNDIR" TEXT_MERGE_20260813 TEXTMERGE
echo "STAGE=train RC=$? $(date -Is)"

echo "STAGE=analyze START $(date -Is)"
python idea-stage/text_merge/analyze_arms.py \
    --logdir "$RUNDIR/logs" --out idea-stage/text_merge/results.json
echo "STAGE=analyze RC=$? $(date -Is)"
echo "ALL_DONE $(date -Is)"
