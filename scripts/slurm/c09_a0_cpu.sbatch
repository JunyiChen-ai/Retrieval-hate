#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=c09_a0
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time limit. CPU ONLY, no --gres. No dependency, no array,
# no singleton, no requeue, no chain, no force, no release. ONE submission only.
#
# C09 A0 -- Stage-0 for Stable-Inversion Topology Surgery.
# Design of record: refine-logs/C09_A0_RECORD.md (text: C09_A0_V17_RECORD.md,
# design review GO 0C/0H/0I at round 17).  Config: configs/c09/c09_a0.json.
#
# Order of work, fail-closed at every step (set -e):
#   1. DET-1: thread environment exported BEFORE any python process starts.
#   2. 36 CPU head mints = 2 datasets x 3 seeds x {5 fitting-pool folds + 1
#      deployed-configuration head}, via the FROZEN, UNMODIFIED
#      scripts/analysis/headspace_mint.py.  Each mint asserts fold parity against the
#      banked scripts/analysis/vsw_ckpt/<ds>/f{0..4}.npz and refuses on mismatch.
#      Layout follows headspace_drive.sh:20/24-25 exactly: the .npz files at the
#      scratch ROOT, run_rac output trees under <scratch>/mint/<tag>/.
#      Resume: headspace_mint.py:192-194 skips any --out that already exists.
#   3. GATE-DEVFID via the FROZEN, UNMODIFIED scripts/analysis/headspace_fidelity.py
#      (Val_Retrieval-only hard filter).  REPORTING ONLY -- it does not gate (§8.2).
#   4. The arena, which runs the battery and renders the verdict.
#
# ZERO GPU.  Train split + dev_seen features only.  No test-like cache is opened:
# headspace_mint.py:106-116 installs a torch.load guard, and scripts/analysis/
# c09_guard/sitecustomize.py installs an open()-level guard at interpreter startup
# in every python process of this job (mints, fidelity, arena), which torch.load,
# np.load and json.load all bottom out in.  Verified against the working tree: the
# predicate is COMPONENT-WISE (a directory named test_seen counts, not just a
# basename) and its coverage is MEASURED at run time by c09guard.verify_predicate(),
# whose result is emitted into GATE_LEDGER rather than asserted in this comment.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# DET-1 (refine-logs/PREGATE_DETERMINISM_CLAUSE.md): exported before python starts.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
# Job-wide split guard: sitecustomize is imported by EVERY python below (mints,
# fidelity, arena), so the test-split perimeter is not confined to the analysis
# process, and each process appends its measured open()-counts to C09_LEDGER_DIR
# for GATE-LEDGER to aggregate.  No hash-frozen module is edited.
export PYTHONPATH=/data/jehc223/RGCL/scripts/analysis/c09_guard${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_ID=C09-A0-v1
BASE=/data/jehc223/RGCL/artifacts/c09_topo/v1/a0/${RUN_ID}
SC=${BASE}/scratch
export C09_LEDGER_DIR=${BASE}/ledger
OUTDIR=${BASE}
CFG=/data/jehc223/RGCL/configs/c09/c09_a0.json
mkdir -p "$SC" "$OUTDIR" "$C09_LEDGER_DIR"

echo "[c09_a0] job=${SLURM_JOB_ID:-NA} run_id=${RUN_ID} host=$(hostname)"
python - <<'PY'
import builtins, torch, numpy, faiss, sys, sklearn, scipy
assert builtins.open.__name__ == "_guarded_open", "C09 startup guard NOT installed"
print("[c09_a0] python", sys.version.split()[0], "numpy", numpy.__version__,
      "scipy", scipy.__version__, "sklearn", sklearn.__version__,
      "torch", torch.__version__, "faiss", faiss.__version__)
assert torch.cuda.device_count() == 0, "ZERO-GPU VIOLATION: a device is visible"
PY

# --- frozen-module sha256 verified HERE, before the 36 mints, so a changed module
#     costs seconds instead of ~25 minutes.  The mint and the arena re-assert too.
check_sha () {
  want="$1"; f="$2"
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "[c09_a0] FROZEN MODULE CHANGED: $f"; echo "  want $want"; echo "  got  $got"
    exit 4
  fi
  echo "[c09_a0] sha OK $f"
}
check_sha cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612 scripts/analysis/headspace_mint.py
check_sha 635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d scripts/analysis/mechfix_ops.py
check_sha 77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d scripts/analysis/mechnov_pairverify.py
check_sha 72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598 scripts/analysis/headspace_fidelity.py

# ------------------------------------------------------------------ 2. 36 head mints
for DS in hatemm zh; do
  for SEED in 0 1 2; do
    for FOLD in 0 1 2 3 4 -1; do
      if [ "$FOLD" -lt 0 ]; then TAG=full; else TAG=$FOLD; fi
      OUT="$SC/mint_${DS}_s${SEED}_f${TAG}.npz"
      if [ -f "$OUT" ]; then echo "[c09_a0] have $OUT"; continue; fi
      echo "### mint ds=$DS seed=$SEED fold=$FOLD"
      python scripts/analysis/headspace_mint.py --dataset "$DS" --seed "$SEED" \
        --fold "$FOLD" --out "$OUT" --scratch "$SC"
    done
  done
done
echo "[c09_a0] all 36 mints present"

# ------------------------------------------------- 3. GATE-DEVFID (reporting only)
# REPORTING ONLY (design 8.2): GATE-DEVFID does not gate, so under `set -e` it must
# not be able to abort the job before a verdict is ever computed.  A failure here is
# recorded and the arena's os.path.exists branch reports the instrument as absent.
for DS in hatemm zh; do
  if ! python scripts/analysis/headspace_fidelity.py --dataset "$DS" --mintdir "$SC" \
       --out "$OUTDIR/C09_FIDELITY_${DS}.json"; then
    echo "[c09_a0] GATE-DEVFID FAILED for ${DS} -- reporting instrument only, continuing"
  fi
done
echo "[c09_a0] GATE-DEVFID written (reporting instrument, does NOT gate -- design 8.2)"

# ------------------------------------------------------------------- 4. the arena
python scripts/analysis/c09_a0_arena.py --config "$CFG" --mintdir "$SC" \
  --outdir "$OUTDIR" --threads 8

echo "[c09_a0] COMPLETE -> $OUTDIR/C09_A0_OUT.json"
python - <<PY
import json
d = json.load(open("$OUTDIR/C09_A0_DECISION.json"))
print("[c09_a0] VERDICT:", d["DECISION"]["verdict"])
PY
