#!/bin/bash
# vsw_drive.sh -- ORCHESTRATION ONLY for the VSW $0 pregate.
#
# The login node SIGTERMs sustained non-SLURM CPU processes (~2 min of process life was
# measured for this workload; F95 §3 and LSMI_GATE §2.7 document the same failure).  The
# frozen script `vsw_pregate.py` checkpoints every unit of work -- one fold's arena, one
# (fold x permutation draw) verifier fit, one draw's evaluation -- the instant it
# completes, so a reap costs at most one unit.  This driver simply re-invokes the frozen
# script until the stage reports success.
#
# IT CHANGES NO ARM.  Same constants, same seeds, same fold assignment, same permutation
# draw sequence (every draw is seeded from (PERM_SEED, draw, fold)).  Only the process
# boundary differs -- the mechnov_pairverify_runner.py / LSMI per-draw precedent.
#
# Usage: bash scripts/analysis/vsw_drive.sh <stage> <dataset> [max_attempts]
set -u
STAGE="$1"; DS="$2"; MAXA="${3:-4000}"
REPO=/data/jehc223/RGCL
# the frozen script's tier-2 parity gate reads the anchor at this exact path
if [ "$STAGE" = "anchor" ]; then
  OUT="$REPO/scripts/analysis/vsw_f95anchor_${DS}_OUT.json"
else
  OUT="$REPO/scripts/analysis/vsw_${STAGE}_${DS}_OUT.json"
fi
cd "$REPO" || exit 1
for i in $(seq 1 "$MAXA"); do
  if python scripts/analysis/vsw_pregate.py --stage "$STAGE" --dataset "$DS" \
        --out "$OUT" >/dev/null 2>&1; then
    echo "$(date -Is) $STAGE $DS COMPLETE after $i process attempt(s)"
    exit 0
  fi
done
echo "$(date -Is) $STAGE $DS NOT COMPLETE after $MAXA attempts"
exit 1
