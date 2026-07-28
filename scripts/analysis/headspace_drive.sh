#!/usr/bin/env bash
# HEADSPACE-TRANSFER driver.  One process per unit of work so a login-node reap costs at
# most one unit (the VSW §3.10 / LSMI precedent).  DET-1: the thread environment is
# exported HERE, before any python process starts.
set -uo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONUNBUFFERED=1

SC="$1"          # scratch root
DS="$2"          # hatemm | zh
SEEDS="${3:-0 1 2}"
mkdir -p "$SC"

for seed in $SEEDS; do
  for fold in -1 0 1 2 3 4; do
    tagf=$([ "$fold" -lt 0 ] && echo full || echo "$fold")
    out="$SC/mint_${DS}_s${seed}_f${tagf}.npz"
    for try in 1 2 3 4 5; do
      [ -f "$out" ] && break
      echo "### mint ${DS} seed=$seed fold=$fold try=$try"
      python scripts/analysis/headspace_mint.py --dataset "$DS" --seed "$seed" \
        --fold "$fold" --out "$out" --scratch "$SC" >> "$SC/mint_${DS}.log" 2>&1
    done
    [ -f "$out" ] || { echo "MINT FAILED: $out"; exit 1; }
  done
done
echo "ALL MINTS DONE ($DS)"
