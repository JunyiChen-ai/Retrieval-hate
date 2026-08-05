#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

readonly RUN_ID=C04-A0T-SMALL-v1
readonly CONFIG=configs/c04/c04_a0t_small_v1_v8.json
readonly PYTHON_BIN=/data/jehc223/miniconda3/envs/HateVideo/bin/python
readonly PREFLIGHT_SCRIPT=scripts/analysis/c04_a0t_small_v1_v8_preflight.py

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "C04 preflight requires SLURM" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 arrays and dependencies are forbidden" >&2
  exit 2
fi
if [[ "$(jq -r '.authorization.preflight_materialization_authorized' "$CONFIG")" != "true" ]]; then
  echo "Prospective config blocks preflight pending code/resource GO" >&2
  exit 2
fi
# The v8 preflight decodes video and runs the Qwen2.5-VL image_processor to
# measure per-item visual geometry.  Neither needs a GPU, and the preflight
# refuses to run if one is visible, so make the CPU-only condition explicit
# rather than relying on the sbatch not having requested a gres.
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|-1|NoDevFiles) ;;
  *)
    echo "C04 preflight must be CPU-only" >&2
    exit 2
    ;;
esac
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "0" ]]; then
  echo "C04 preflight received a GPU allocation" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=""
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Pinned HateVideo Python is unavailable" >&2
  exit 2
fi

"$PYTHON_BIN" "$PREFLIGHT_SCRIPT" --mode self-test
"$PYTHON_BIN" "$PREFLIGHT_SCRIPT" --mode freeze

echo "C04 impl-v8 CPU preflight complete; the measured projection gate passed."
echo "GPU remains blocked pending a fresh payload review and GPU-execution GO."
