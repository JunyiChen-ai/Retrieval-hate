#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
# CPU-ONLY durable driver: NO --gres (zero GPU). The probe COMPUTE runs on Modal cloud; this
# SLURM job only hosts the reap-proof chunk-loop client (login=compute node reaps non-SLURM
# background processes — the recurring failure mode; a SLURM job is not reaped).
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=w2a_probe_loop
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/w2a_probe_loop_%j.log
# NOTE: intentionally NO --time (project policy). PENDING (JobHeldUser) -> wait for auto-release.
#
# W2-A Stage-P' probe — path-2 chunked resumable loop. Root cause: Modal clamps the effective
# function timeout to ~3600s server-side (verified: MODAL_PROBE_TIMEOUT=43200 reaches the decorator
# yet the container is killed at ~3600s), so the multi-hour K9 perm-null cannot finish in one
# container. Each chunk runs the FROZEN probe (af4a2f9f, via the adapter shim) NON-detached under a
# soft budget < cap; _execute commits the resumable --ci_ckpt to the volume each chunk; this loop
# relaunches to RESUME until /W2A_PROBE_RESULTS.md appears on the volume. Frozen probe byte-unchanged.
set -u

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export MODAL_PROBE_APP_NAME=rgcl-w2a-probe-e
export MODAL_PROBE_TIMEOUT=3500
BUDGET=3200
MAXCHUNKS=40
LEDGER=/data/jehc223/RGCL/refine-logs/W2A_CHUNK_LOG.md
CKTMP=/data/jehc223/RGCL/slurm/logs/w2a_ck_probe_${SLURM_JOB_ID:-local}.json

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "[loop] app=$MODAL_PROBE_APP_NAME requested_timeout=${MODAL_PROBE_TIMEOUT}s soft_budget=${BUDGET}s"
echo "[loop] https_proxy=${https_proxy:-<unset>} http_proxy=${http_proxy:-<unset>}"
# socks preflight — fail LOUD if the squid proxy deps are missing (gRPC control + volume upload planes)
python -c "import python_socks, aiohttp_socks; print('[loop] socks deps OK (python_socks + aiohttp_socks)')" || {
    echo "[loop] FATAL: missing python-socks / aiohttp-socks — modal cannot reach api.modal.com over squid"; exit 3; }
python -c "import modal; print('[loop] modal client import OK')" || { echo "[loop] FATAL: modal import failed"; exit 3; }
echo "[loop] frozen probe sha256:"; sha256sum scripts/analysis/w2a_probe.py
echo "================================================================================"

prev_total=-1; stall=0
for i in $(seq 1 $MAXCHUNKS); do
  ts=$(date -u +%FT%TZ)
  echo "[loop] === chunk $i launch $ts (budget=${BUDGET}s) ==="
  CLOG=/data/jehc223/RGCL/slurm/logs/w2a_probe_chunk_${SLURM_JOB_ID:-local}_$i.log
  # NON-detached: the client stays attached for the whole chunk; the soft budget makes the function
  # exit cleanly at ~${BUDGET}s and modal run returns. The SLURM job keeps the client durable.
  modal run scripts/cloud/modal_probe_runner.py::run \
    --script scripts/analysis/w2a_probe_cloud_adapter.py --chunk-budget "$BUDGET" \
    > "$CLOG" 2>&1 || echo "[loop] chunk $i modal-run rc=$? (transient; continuing)"
  appid=$(grep -oE "ap-[A-Za-z0-9]+" "$CLOG" | head -1)
  echo "[loop] chunk $i app=${appid:-?} finished $(date -u +%FT%TZ)"
  # completion?
  if modal volume ls rgcl-features / 2>/dev/null | grep -q "W2A_PROBE_RESULTS.md"; then
    echo "| $i | ${appid:-?} | $ts | COMPLETE |" >> "$LEDGER"
    echo "[loop] COMPLETE — W2A_PROBE_RESULTS.md present on volume after chunk $i"
    break
  fi
  # progress / stall guard
  modal volume get --force rgcl-features /w2a_ci_ckpt.json "$CKTMP" >/dev/null 2>&1
  total=$(python -c "
import json
try:
    d=json.load(open('$CKTMP'))
    print(sum(len(v.get('perm',{}).get('maxk',[])) for k,v in d.items() if k!='_meta' and isinstance(v,dict)))
except Exception:
    print(-1)
" 2>/dev/null)
  echo "| $i | ${appid:-?} | $ts | ${total:-?} |" >> "$LEDGER"
  echo "[loop] chunk $i perms-after=$total (prev=$prev_total)"
  if [ "${total:-0}" -le "$prev_total" ]; then stall=$((stall+1)); else stall=0; fi
  if [ "$stall" -ge 3 ]; then echo "[loop] STALL: no perm progress for 3 chunks -> ABORT for debug"; exit 4; fi
  prev_total=${total:-0}
done
echo "[loop] loop finished $(date -u +%FT%TZ)"
