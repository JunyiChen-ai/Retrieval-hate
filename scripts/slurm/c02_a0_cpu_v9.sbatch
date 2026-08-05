#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=c02_a0_v9
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time limit. CPU ONLY, no --gres. No dependency, no array,
# no singleton, no requeue, no chain, no force, no release. ONE submission only.
#
# C02 A0 -- Stage-0 density-orbit reachability oracle.
# Prereg: refine-logs/C02_A0_V9_RECORD.md.  Config: configs/c02/c02_a0_v9.json.
#
# Order of work, fail-closed at every step (set -e):
#   1. DET-1: thread environment exported BEFORE any python process starts.
#   2. 36 CPU head mints = 2 datasets x 3 seeds x {5 fitting-pool folds + 1 deployed-
#      configuration head}.  Each mint asserts fold parity against the banked
#      scripts/analysis/vsw_ckpt/<ds>/f{0..4}.npz and stores the head keys of every
#      C02 view.
#   3. GATE-FID via the FROZEN, UNMODIFIED scripts/analysis/headspace_fidelity.py
#      (dev-only reader), then the declared STOP RULE: if the 3-seed instrument band
#      B_fid >= the Stage-0 accuracy bar 0.050, the measurement is inadmissible and
#      the job halts before the arena runs.
#   4. The arena, which renders the verdict.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# DET-1 (refine-logs/PREGATE_DETERMINISM_CLAUSE.md): exported before python starts.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ID=C02-A0-v9
SC=/data/jehc223/RGCL/artifacts/c02_edq/v1/a0/${RUN_ID}/scratch
OUTDIR=/data/jehc223/RGCL/artifacts/c02_edq/v1/a0/${RUN_ID}
CFG=/data/jehc223/RGCL/configs/c02/c02_a0_v9.json
mkdir -p "$SC"

echo "[c02_a0] job=${SLURM_JOB_ID:-NA} run_id=${RUN_ID}"
python - <<'PY'
import torch, numpy, faiss, sys
print("[c02_a0] python", sys.version.split()[0], "numpy", numpy.__version__,
      "torch", torch.__version__, "faiss", faiss.__version__)
PY

# I13: the oracle self-test runs BEFORE the 36 mints, so a numerical-contract break
# costs seconds for the JOB, not just for the arena process.
echo "########## oracle self-test (synthetic arrays, no project data) ##########"
python - <<'PYSELF'
import sys
sys.path.insert(0, "/data/jehc223/RGCL/scripts/analysis")
sys.path.insert(0, "/data/jehc223/RGCL/src/utils")
import c02_a0_arena_v9 as A
print("[c02_a0] oracle self-test PASS:", ", ".join(A.oracle_self_test()))
PYSELF

# I4: the GATE-FID reader is used UNMODIFIED -- enforce that, do not merely assert it.
# Every FROZEN imported module is verified HERE, before the 36 mints, so a changed
# module costs seconds instead of ~25 minutes.  The arena and the mint re-assert their
# own subsets in-process; this is the earliest and most complete check.
check_sha () {
  local want="$1" path="$2"
  local got; got=$(sha256sum "$path" | cut -d" " -f1)
  if [ "$got" != "$want" ]; then
    echo "HALT_C02_A0_FROZEN_MODULE_CHANGED: $path $got != $want" >&2
    exit 4
  fi
  echo "[c02_a0] frozen module verified: $path"
}
check_sha 72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598 scripts/analysis/headspace_fidelity.py
check_sha 635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d scripts/analysis/mechfix_ops.py
check_sha 77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d scripts/analysis/mechnov_pairverify.py
check_sha cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612 scripts/analysis/headspace_mint.py

for DS in hatemm zh; do
  for SEED in 0 1 2; do
    for FOLD in -1 0 1 2 3 4; do
      if [ "$FOLD" -lt 0 ]; then TAG=full; else TAG=$FOLD; fi
      OUT="$SC/mint_${DS}_s${SEED}_f${TAG}.npz"
      if [ -f "$OUT" ]; then echo "[c02_a0] have $OUT"; continue; fi
      echo "### mint ds=$DS seed=$SEED fold=$FOLD"
      python scripts/analysis/c02_a0_mint.py --dataset "$DS" --seed "$SEED" \
        --fold "$FOLD" --out "$OUT" --scratch "$SC"
    done
  done
done
echo "[c02_a0] all 36 mints done"

for DS in hatemm zh; do
  python scripts/analysis/headspace_fidelity.py --dataset "$DS" --mintdir "$SC" \
    --out "$OUTDIR/C02_A0_FIDELITY_${DS}.json" --seeds 0,1,2
done

python - "$OUTDIR" <<'PY'
import json, sys, os
outdir = sys.argv[1]
bar = 0.050
for ds in ("hatemm", "zh"):
    g = json.load(open(os.path.join(outdir, "C02_A0_FIDELITY_%s.json" % ds)))["gate"]
    b = float(g["B_fid_abs_3seedmean"])
    print("[GATE-FID] %s B_fid=%.4f bar=%.3f" % (ds, b, bar))
    assert b < bar, ("HALT_C02_A0_INSTRUMENT_BAND: %s B_fid %.4f >= Stage-0 bar %.3f"
                     % (ds, b, bar))
print("[GATE-FID] PASS on both datasets")
PY

python scripts/analysis/c02_a0_arena_v9.py --mintdir "$SC" --outdir "$OUTDIR" --config "$CFG"

echo "[c02_a0] done"
