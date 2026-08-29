#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
evidence=scripts/reproduction_baselines/relation_v9/hatemm_train_evidence.sbatch
macil_job="$(sbatch --parsable --export=ALL,EVIDENCE_MODE=macil "$evidence")"
vera_job="$(sbatch --parsable --export=ALL,EVIDENCE_MODE=vera "$evidence")"
preflight_job="$(sbatch --parsable --dependency="afterok:${macil_job}:${vera_job}" \
  scripts/reproduction_baselines/relation_v9/hatemm_preflight.sbatch)"
pilot_job="$(sbatch --parsable --dependency="afterok:${preflight_job}" \
  scripts/reproduction_baselines/relation_v9/hatemm_pilot.sbatch)"
echo "macil_evidence=$macil_job"
echo "vera_evidence=$vera_job"
echo "preflight_afterok=$preflight_job"
echo "pilot_afterok=$pilot_job"
