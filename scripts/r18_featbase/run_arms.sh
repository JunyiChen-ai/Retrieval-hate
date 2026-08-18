#!/usr/bin/env bash
# R18-FEATBASE: the five frozen arms, single submission, in the order of the freeze.
# Primary family A1/A2/A3 first, then the descriptive visual-only family B1/B2.
set -euo pipefail
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

EMB=data/CLIP_Embedding/HateClipSeg
OUT=idea-stage/r18_featbase/out
SEEDS="7300 7301 7302"
mkdir -p "$OUT"

# Guard: the ActionFormer dataset adapter silently drops videos whose feature file is
# missing, which would make the arms non-comparable.  Assert all 395 are present first.
for d in dense4fps_vat dense4fps_mat dense4fps_mvat dense4fps_clipL336 dense4fps_vmaev2g; do
  n=$(ls "$EMB/$d" | wc -l)
  echo "[guard] $d: $n files"
  [ "$n" -eq 395 ] || { echo "ABORT: $d has $n != 395 feature files"; exit 1; }
done

run () {  # tag  featdir  dim
  echo "=================== ARM $1 ($3-d) ==================="
  python scripts/r16_detbase/run_af.py \
    --gt rawseg --feat "$EMB/$2" --input-dim "$3" \
    --seeds $SEEDS --touch-test --dump-preds \
    --tag "$1" --out "$OUT"
}

run clip_vat     dense4fps_vat        2816
run mae_vat      dense4fps_mat        3200
run maeclip_vat  dense4fps_mvat       4224
run clip_v       dense4fps_clipL336   1024
run mae_v        dense4fps_vmaev2g    1408
echo "ALL ARMS DONE"
