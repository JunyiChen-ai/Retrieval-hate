#!/bin/bash
#SBATCH --job-name=mech_stage_b_b4
#SBATCH --output=slurm/logs/mech_stage_b_b4_%j.out
#SBATCH --error=slurm/logs/mech_stage_b_b4_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NO --gres, NO --time, NO --array, NO --dependency  (CLAUDE.md GPU/资源 section)
set -euo pipefail

# =============================================================================
# MECH-STAGE-B, B4 leg only -- the resume of job 13998.
# Frozen design: refine-logs/MECH_STAGE_B_PREREG.md (+ amendments A1, A2).
#
# Job 13998 completed all 156 head trains (B1 90, B2 30, B3 30, 6 parity, all
# cached under artifacts/mech_stage_b/heads/) and then HALTed in run_b4 on its OWN
# instrument check: admit-always TRA did not reproduce mechfix_ops.deployed_vote.
# Cause + fix = prereg amendment A2 (faiss/argsort tie-order); re-verified at
# max|Dvote| = 0 on all 30 (ds, seed, fold) cells before this submission.
#
# NO head is retrained here -- B1/B2/B3 verdicts were already aggregated from the
# cached predictions of job 13998.  This runs the zero-training B4 leg and re-folds
# everything into MECH_STAGE_B_RESULT.json.
#
# ZERO GPU, ZERO TEST CONTACT (B4 reads only the banked C06 train-split mints).
# Projected ~1000 s (measured: 9.43 s/cell HateMM, 2.82 s/cell ZH, 75 cells each).
# =============================================================================

REPO=/data/jehc223/RGCL
cd "$REPO"

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$REPO/scripts/analysis/c09_guard:${PYTHONPATH:-}"
export MSB_PROJECTED_SECONDS=1000

OUT_ROOT="$REPO/artifacts/mech_stage_b"
SCRATCH="$OUT_ROOT/scratch"
PROGRESS="$OUT_ROOT/progress/MSB_PROGRESS.txt"
DRIVER="$REPO/scripts/analysis/mech_stage_b.py"

mkdir -p "$OUT_ROOT/heads" "$SCRATCH" "$(dirname "$PROGRESS")" slurm/logs

printf '%s | DRIVER-B4 | - | 0.0s | 0.000x | job=%s node=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "${SLURM_JOB_ID:-none}" "$(hostname)" \
    >> "$PROGRESS"
echo "[msb-b4] start $(date -u +%FT%TZ) on $(hostname)"

run() { python "$DRIVER" "$@" --outroot "$OUT_ROOT" --scratch "$SCRATCH" --progress "$PROGRESS"; }

# the floor replay is cheap and is the battery's HALT gate -- re-assert it here
run instrument
run b4
run collect

echo "[msb-b4] done $(date -u +%FT%TZ) -> $OUT_ROOT/MECH_STAGE_B_RESULT.json"
printf '%s | DRIVER-B4-DONE | - | - | - | ok\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> "$PROGRESS"
