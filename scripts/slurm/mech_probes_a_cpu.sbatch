#!/bin/bash
#SBATCH --job-name=mech_probes_a
#SBATCH --output=slurm/logs/mech_probes_a_%j.out
#SBATCH --error=slurm/logs/mech_probes_a_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NO --gres, NO --time, NO --array, NO --dependency  (CLAUDE.md GPU/资源 section)
set -euo pipefail

# =============================================================================
# MECH-PROBES-A -- the single submission.
# Frozen design: refine-logs/MECH_PROBES_A_PREREG.md
#
#   1 probe12 process  (probes 1 and 2, banked mints, zero training)
# + 30 mint processes  (5 folds x 3 seeds x 2 datasets, ~52 s each, CPU)
# + 1 probe3 process   (fold arena over the epoch snapshots)
# + 1 report process
#
# ZERO GPU: CUDA_VISIBLE_DEVICES is exported empty.
# ZERO TEST CONTACT: headspace_mint's torch.load guard + the frozen train/dev-only
# load_split + c09_guard on PYTHONPATH.
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

# ---- test-split guard layer 3 (frozen sitecustomize; importable only from here)
export PYTHONPATH="$REPO/scripts/analysis/c09_guard:${PYTHONPATH:-}"

# ---- heartbeat denominator, prereg §2
export MPA_PROJECTED_SECONDS=1210

OUT_ROOT="$REPO/artifacts/mech_probes_a"
SNAPDIR="$OUT_ROOT/snaps"
SCRATCH="$OUT_ROOT/scratch"
PROGRESS="$OUT_ROOT/progress/MPA_PROGRESS.txt"
DRIVER="$REPO/scripts/analysis/mech_probes_a.py"

mkdir -p "$SNAPDIR" "$SCRATCH" "$(dirname "$PROGRESS")" slurm/logs

printf '%s | DRIVER | - | 0.0s | 0.000x | job=%s node=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "${SLURM_JOB_ID:-none}" "$(hostname)" \
    >> "$PROGRESS"
echo "[mpa] driver start $(date -u +%FT%TZ) on $(hostname)"

# ============================ probes 1 + 2 (no training) =======================
python "$DRIVER" probe12 --outdir "$OUT_ROOT" --progress "$PROGRESS"

# ============================ 30 snapshot mints ================================
N=0
for DS in hatemm zh; do
  for SEED in 0 1 2; do
    for FOLD in 0 1 2 3 4; do
      N=$((N+1))
      echo "[mpa] mint ${N}/30  ${DS} s${SEED} f${FOLD}"
      python "$DRIVER" mint --dataset "$DS" --seed "$SEED" --fold "$FOLD" \
          --snapdir "$SNAPDIR" --scratch "$SCRATCH" --threads 8 --progress "$PROGRESS"
    done
  done
done
test "$N" -eq 30 || { echo "[mpa] FATAL: mint count $N != 30"; exit 1; }
echo "[mpa] mints complete: $N"

# ============================ probe 3 + report =================================
python "$DRIVER" probe3 --outdir "$OUT_ROOT" --snapdir "$SNAPDIR" --progress "$PROGRESS"
python "$DRIVER" report --outdir "$OUT_ROOT" --progress "$PROGRESS"

echo "[mpa] driver done $(date -u +%FT%TZ) -> $OUT_ROOT/MECH_PROBES_A_RESULT.json"
printf '%s | DRIVER-DONE | - | - | - | ok\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$PROGRESS"
