#!/usr/bin/env bash
# CONTRAST STANCE PILOT driver. Idempotent: every stage skips if its output exists.
set -u
TAG="${1:-c1}"
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
D=idea-stage/contrast_stance
for P in 0 1 2 3 4; do
  echo "[drive] === pair $P submit ==="
  python $D/run_contrast.py submit --tag "$TAG" --pair $P || exit 1
done
for P in 0 1 2 3 4; do
  echo "[drive] === pair $P poll+fetch ==="
  python $D/run_contrast.py poll  --tag "$TAG" --pair $P || exit 1
  python $D/run_contrast.py fetch --tag "$TAG" --pair $P || exit 1
done
python $D/run_contrast.py merge --tag "$TAG" || exit 1
echo "[drive] done"
