#!/usr/bin/env bash
# CAD end-to-end driver: generate -> gate -> encode -> assemble -> 9 head runs -> analyze.
# Single background submission (CLAUDE.md single-GPU branch; CAD_FREEZE.md section 5).
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNDIR=logging/runs/cad
mkdir -p "$RUNDIR/logs"

echo "STAGE=generate START $(date -Is)"
python idea-stage/cad/cadgen.py realtime --prompt V2 --workers "${WORKERS:-8}"
RC=$?; echo "STAGE=generate RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC
python idea-stage/cad/cadgen.py report

echo "STAGE=gates START $(date -Is)"
python idea-stage/cad/gates.py
RC=$?; echo "STAGE=gates RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC

echo "STAGE=encode START $(date -Is)"
python idea-stage/cad/build_cad_feats.py --offload_gib "${OFFLOAD_GIB:-9}"
RC=$?; echo "STAGE=encode RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC

echo "STAGE=assemble START $(date -Is)"
python idea-stage/cad/build_cad_feats.py --assemble
RC=$?; echo "STAGE=assemble RC=$RC $(date -Is)"
[ $RC -ne 0 ] && exit $RC

echo "STAGE=train START $(date -Is)"
bash idea-stage/cad/run_arms.sh "$RUNDIR" CAD_20260813 CAD
echo "STAGE=train RC=$? $(date -Is)"

echo "STAGE=analyze START $(date -Is)"
python idea-stage/cad/analyze.py --logdir "$RUNDIR/logs" --out idea-stage/cad/results.json
echo "STAGE=analyze RC=$? $(date -Is)"
echo "ALL_DONE $(date -Is)"
