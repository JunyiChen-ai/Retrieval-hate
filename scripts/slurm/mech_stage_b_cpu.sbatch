#!/bin/bash
#SBATCH --job-name=mech_stage_b
#SBATCH --output=slurm/logs/mech_stage_b_%j.out
#SBATCH --error=slurm/logs/mech_stage_b_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NO --gres, NO --time, NO --array, NO --dependency  (CLAUDE.md GPU/资源 section)
set -euo pipefail

# =============================================================================
# MECH-STAGE-B -- the single submission.
# Frozen design: refine-logs/MECH_STAGE_B_PREREG.md
#
#   1 instrument process   (banked-mint floor replay; HALT gate, no training)
# + 6 parity processes     (3 arms x 2 datasets, mechanism OFF, must be bit-exact)
# + 90 B1 mint processes   (3 lambda x 5 folds x 3 seeds x 2 datasets)
# + 30 B2 mint processes
# + 30 B3 mint processes
# + 1 b4 process           (trust radii over the banked mints, no training)
# + 1 collect process
# = 159 processes, ~7481 s projected.
#
# ZERO GPU: CUDA_VISIBLE_DEVICES is exported empty.
# ZERO TEST CONTACT: headspace_mint's torch.load guard + its train/dev-only load_split
# + c09_guard on PYTHONPATH.
# =============================================================================

REPO=/data/jehc223/RGCL
cd "$REPO"

# ---- the environment.  The C06 lesson (job 13987 died in 1 s with
#      ModuleNotFoundError: No module named 'torch'): this sbatch MUST activate conda.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# ---- DET-1: the thread environment must exist BEFORE any python starts.
#      headspace_mint.det1_assert() hard-fails without it.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

# ---- test-split guard layer 3 (frozen sitecustomize)
export PYTHONPATH="$REPO/scripts/analysis/c09_guard:${PYTHONPATH:-}"

# ---- heartbeat denominator, prereg §6
export MSB_PROJECTED_SECONDS=7481

OUT_ROOT="$REPO/artifacts/mech_stage_b"
SCRATCH="$OUT_ROOT/scratch"
PROGRESS="$OUT_ROOT/progress/MSB_PROGRESS.txt"
DRIVER="$REPO/scripts/analysis/mech_stage_b.py"

mkdir -p "$OUT_ROOT/heads" "$SCRATCH" "$(dirname "$PROGRESS")" slurm/logs

printf '%s | DRIVER | - | 0.0s | 0.000x | job=%s node=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "${SLURM_JOB_ID:-none}" "$(hostname)" \
    >> "$PROGRESS"
echo "[msb] driver start $(date -u +%FT%TZ) on $(hostname)"

run() { python "$DRIVER" "$@" --outroot "$OUT_ROOT" --scratch "$SCRATCH" --progress "$PROGRESS"; }

# ===================== instrument: floor replay (HALT gate) ====================
run instrument

# ===================== mechanism-OFF parity heads (HALT gate) ==================
for ARM in b1 b2 b3; do
  for DS in hatemm zh; do
    echo "[msb] parity ${ARM} ${DS}"
    run head --arm "$ARM" --dataset "$DS" --seed 0 --fold 0 --parity --threads 8
  done
done

# ===================== B1: 3 lambda x 30 heads =================================
N=0
for LAM in 0.1 0.3 1.0; do
  for DS in hatemm zh; do
    for SEED in 0 1 2; do
      for FOLD in 0 1 2 3 4; do
        N=$((N+1))
        echo "[msb] b1 ${N}/90  lam=${LAM} ${DS} s${SEED} f${FOLD}"
        run head --arm b1 --dataset "$DS" --seed "$SEED" --fold "$FOLD" \
            --lam "$LAM" --threads 8
      done
    done
  done
done
test "$N" -eq 90 || { echo "[msb] FATAL: b1 count $N != 90"; exit 1; }

# ===================== B2 and B3: 30 heads each ================================
for ARM in b2 b3; do
  M=0
  for DS in hatemm zh; do
    for SEED in 0 1 2; do
      for FOLD in 0 1 2 3 4; do
        M=$((M+1))
        echo "[msb] ${ARM} ${M}/30  ${DS} s${SEED} f${FOLD}"
        run head --arm "$ARM" --dataset "$DS" --seed "$SEED" --fold "$FOLD" --threads 8
      done
    done
  done
  test "$M" -eq 30 || { echo "[msb] FATAL: ${ARM} count $M != 30"; exit 1; }
done

# ===================== B4 (no training) + collect ==============================
run b4
run collect

echo "[msb] driver done $(date -u +%FT%TZ) -> $OUT_ROOT/MECH_STAGE_B_RESULT.json"
printf '%s | DRIVER-DONE | - | - | - | ok\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$PROGRESS"
