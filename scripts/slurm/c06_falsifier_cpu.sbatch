#!/bin/bash
#SBATCH --job-name=c06_falsifier
#SBATCH --output=slurm/logs/c06_falsifier_%j.out
#SBATCH --error=slurm/logs/c06_falsifier_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NO --gres, NO --time, NO --array, NO --dependency, NO --requeue  (§13)
set -euo pipefail

# =============================================================================
# C06 $0 CPU falsifier -- the single submission (§9, §13).
# Frozen design: refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md (GO at round 15,
#   + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2).
#
#   74 processes, in this order:  1 GATE-SHA driver leg -> 66 mints -> 6 fidelity
#                                 -> 1 arena
#   GATE-SHA runs in this driver before any of them, and again inside the arena at
#   the point of use (TOCTOU): two passes, §6 and §13.
#   The progress file is created here, before the first python process starts.
#
# ZERO GPU.  CUDA_VISIBLE_DEVICES is exported empty and the C01 algebra's own
# import_compute_modules() dies if CUDA is visible despite the guard.
# =============================================================================

REPO=/data/jehc223/RGCL
cd "$REPO"

# ---- the environment.  CODE-R1 M-3 recorded that this sbatch activated no environment
#      and referred it onward; it was never wired, and job 13987 died in 1 s at
#      load_frozen() with ModuleNotFoundError: No module named 'torch'.  GATE-SHA had
#      already passed 38/38, so the failure was environment-only, not a design defect.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# ---- DET-1: the thread environment must exist BEFORE any python starts (§6) ----
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

# ---- §13.1 item 28: layer 3 of the test guard only loads if c09_guard is on
#      PYTHONPATH.  Without this export the frozen sitecustomize is importable but
#      never imported, and GATE-LEDGER's test_path_opens would be unguarded.
export PYTHONPATH="$REPO/scripts/analysis/c09_guard:${PYTHONPATH:-}"

CONFIG="$REPO/configs/c06/c06_falsifier.json"

# ---- ERRATUM 2 §7: the heartbeat denominator's single source is the config, exported so
#      every one of the 68 ratio-computing processes divides by the same number.
#
#      CODE-R2 C-2: this read runs BEFORE C09_LEDGER_DIR is exported, and that ordering is
#      load-bearing.  c09_guard's sitecustomize installs in every interpreter on PYTHONPATH
#      and registers an atexit flush; with C09_LEDGER_DIR already set, this helper wrote a
#      74th ledger file, the arena computed processes_reporting = len(procs) + 1 = 75, and
#      GATE-LEDGER HALTed against the declared 74 on EVERY clean run.  Running the read
#      first leaves _ledger_path() returning None for this interpreter, so it reports
#      nothing and the battery's process inventory stays exactly 1 + 66 + 6 + 1 = 74.
#      DO NOT move this below the C09_LEDGER_DIR export.
C06_PROJECTED_SECONDS="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['execution']['projected_seconds'])" "$CONFIG")"
export C06_PROJECTED_SECONDS
echo "[c06] projected_seconds single source: $C06_PROJECTED_SECONDS"

# ---- CODE-R1 H-2: c09_guard writes each process's MEASURED ledger here at exit, and the
#      arena aggregates them into GATE-LEDGER.  Without this export _ledger_path() returns
#      None and every process's counts are silently discarded.
export C09_LEDGER_DIR="$REPO/artifacts/c06_falsifier/ledger"
OUT_ROOT="$REPO/artifacts/c06_falsifier"
MINTDIR="$OUT_ROOT/mints"
FIDDIR="$OUT_ROOT/fidelity"
SCRATCH="$OUT_ROOT/scratch"
PROGRESS="$OUT_ROOT/progress/C06_PROGRESS.txt"
VERDICT="$OUT_ROOT/C06_VERDICT.json"

mkdir -p "$MINTDIR" "$FIDDIR" "$SCRATCH" "$(dirname "$PROGRESS")" \
         "$C09_LEDGER_DIR" slurm/logs

# ---- the progress file exists before the first python process (§9) -------------
: > /dev/null
printf '%s | DRIVER | - | 0.0s | 0.000x | job=%s node=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "${SLURM_JOB_ID:-none}" "$(hostname)" \
    >> "$PROGRESS"

echo "[c06] driver start $(date -u +%FT%TZ)"

# ============================ 1 GATE-SHA driver leg ============================
# GATE-SHA, first of two passes, before any other process (§6, §13).
python "$REPO/scripts/analysis/c06_falsifier_arena.py" \
    --progress "$PROGRESS" --gate-sha-only
echo "[c06] GATE-SHA passed"

# ============================ 66 mints =========================================
# Head-N : 2 ds x 3 seeds x (5 folds + 1 full) = 36   (no --train-cache: native)
# Head-R : 2 ds x 3 seeds x 5 folds            = 30   (--train-cache: ro_L24)
# ONE driver, --train-cache the only lineage-varying argument (§3.3).
declare -A RO_CACHE=(
  [hatemm]="$REPO/data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_L24.pt"
  [zh]="$REPO/data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_L24.pt"
)

MINT_N=0
# ERRATUM 2 round-7 H-2: MINT_N counts ATTEMPTS and is 66 on every run, including a
# resume.  GATE-LEDGER binds dev_path_opens == mints_executed + expected_sha_dev_opens,
# and c06_falsifier_mint.py:218-220 returns at MINT-SKIP *before* the dev load whenever
# --out already exists.  Exporting MINT_N would therefore HALT a legitimate resume --
# the self-defeating gate §12's "Why mints_executed and not 66" warrant exists to
# prevent.  MINTS_EXECUTED is incremented on exactly the mint driver's own skip
# condition (the .npz absent at call time), so it counts EXECUTED mints, never attempts.
MINTS_EXECUTED=0
for DS in hatemm zh; do
  for SEED in 0 1 2; do
    for FOLD in 0 1 2 3 4 -1; do
      OUT="$MINTDIR/mint_${DS}_N_s${SEED}_f$([ "$FOLD" -lt 0 ] && echo full || echo "$FOLD").npz"
      MINT_N=$((MINT_N+1))
      [ -f "$OUT" ] || MINTS_EXECUTED=$((MINTS_EXECUTED+1))
      echo "[c06] mint ${MINT_N}/66  Head-N ${DS} s${SEED} f${FOLD}"
      python "$REPO/scripts/analysis/c06_falsifier_mint.py" \
          --dataset "$DS" --seed "$SEED" --fold "$FOLD" --lineage N \
          --out "$OUT" --scratch "$SCRATCH" --threads 8 --progress "$PROGRESS"
    done
    for FOLD in 0 1 2 3 4; do
      OUT="$MINTDIR/mint_${DS}_R_s${SEED}_f${FOLD}.npz"
      MINT_N=$((MINT_N+1))
      [ -f "$OUT" ] || MINTS_EXECUTED=$((MINTS_EXECUTED+1))
      echo "[c06] mint ${MINT_N}/66  Head-R ${DS} s${SEED} f${FOLD}"
      python "$REPO/scripts/analysis/c06_falsifier_mint.py" \
          --dataset "$DS" --seed "$SEED" --fold "$FOLD" --lineage R \
          --train-cache "${RO_CACHE[$DS]}" \
          --out "$OUT" --scratch "$SCRATCH" --threads 8 --progress "$PROGRESS"
    done
  done
done
echo "[c06] mints complete: $MINT_N attempted, $MINTS_EXECUTED executed"
test "$MINT_N" -eq 66 || { echo "[c06] FATAL: mint count $MINT_N != 66"; exit 1; }
export C06_MINTS_EXECUTED="$MINTS_EXECUTED"

# ============================ 6 fidelity =======================================
# GATE-DEVFID (reporting only): headspace_fidelity.py on Head-N's 6 full heads.
# It reads only `meta` out of the banked mint .npz and opens no dev_seen file (§12).
#
# IMPLEMENTATION NOTE (b) -- REPORTED TO THE DESIGN LINEAGE, NOT IMPROVISED.
#   §9 requires each of the 66 mint .npz to name its (dataset, lineage, seed, fold)
#   quadruple.  headspace_fidelity.py:66 -- sha-frozen in §11 and therefore run
#   UNMODIFIED -- hard-codes "mint_{dataset}_s{seed}_ffull.npz", which has no lineage
#   slot.  The two requirements cannot both be met by one filename.
#   Resolution used here: the mints keep their quadruple names, and this driver builds a
#   read-only ALIAS directory of symlinks in the shape the frozen reader expects, pointing
#   at Head-N's six full mints.  Nothing frozen is edited and no mint is duplicated; only
#   the --mintdir the fidelity processes are given differs.
FIDVIEW="$OUT_ROOT/fidelity_view"
mkdir -p "$FIDVIEW"
for DS in hatemm zh; do
  for SEED in 0 1 2; do
    ln -sfn "$MINTDIR/mint_${DS}_N_s${SEED}_ffull.npz" \
            "$FIDVIEW/mint_${DS}_s${SEED}_ffull.npz"
  done
done

FID_N=0
for DS in hatemm zh; do
  for SEED in 0 1 2; do
    FID_N=$((FID_N+1))
    echo "[c06] fidelity ${FID_N}/6  ${DS} s${SEED}"
    printf '%s | FIDELITY | %d/6 | - | - | %s s%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$FID_N" "$DS" "$SEED" >> "$PROGRESS"
    python "$REPO/scripts/analysis/headspace_fidelity.py" \
        --dataset "$DS" --mintdir "$FIDVIEW" \
        --out "$FIDDIR/devfid_${DS}_s${SEED}.json" --seeds "$SEED"
  done
done
echo "[c06] fidelity complete: $FID_N"
test "$FID_N" -eq 6 || { echo "[c06] FATAL: fidelity count $FID_N != 6"; exit 1; }

# ============================ 1 arena ==========================================
echo "[c06] arena start"
# CODE-R1 M-3: `set -e` aborts before `RC=$?` can run, so the old trailer was dead code and
# the HALT path never printed a completion line.  `|| RC=$?` keeps the status without
# tripping errexit, and a GateFailure HALT (exit 2) is a legitimate published outcome, not a crash; so is a
#   §5.6 rule-3 HALT (no lineage passed its per-lineage gates), which is published by the
#   NORMAL emitter and exits 0 (CODE-R2 M-4).  Do not key on the exit code to tell a
#   published HALT from a clean run: read "verdict" on the face.
RC=0
python "$REPO/scripts/analysis/c06_falsifier_arena.py" \
    --config "$CONFIG" \
    --mintdir "$MINTDIR" --out "$VERDICT" --progress "$PROGRESS" --threads 8 || RC=$?

echo "[c06] driver done rc=$RC $(date -u +%FT%TZ)  verdict -> $VERDICT"
printf '%s | DRIVER-DONE | - | - | - | rc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$RC" >> "$PROGRESS"
exit $RC
