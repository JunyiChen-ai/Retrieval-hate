#!/usr/bin/env bash
# REPRO campaign Wave 2 -- CLAP corpus run.
# Two chains in parallel; inside a chain everything is sequential so no two
# processes write the same cluster pickles.  The GPU is shared with another
# campaign method's long run, so each CLAP process is hard-capped at 10% of the
# card (`--gpu_memory_fraction 0.10`); CLAP's scorer is a 4-layer MLP over
# cached feature vectors and uses a few hundred MB.
set -u
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
cd /home/jehc223/Retrieval-hate

LOG=logging/runs/repro_clap
mkdir -p "$LOG"

export CLAP_THREADS=2

run_chain () {
  for DS in "$@"; do
    echo "[chain] $DS start $(date -Is)"
    python scripts/repro_campaign/run_clap.py --stage train --datasets "$DS" \
      && echo "[chain] $DS train ok $(date -Is)" \
      || echo "[chain] $DS train FAILED $(date -Is)"
    python scripts/repro_campaign/run_clap.py --stage normality --datasets "$DS" \
      && echo "[chain] $DS normality ok $(date -Is)" \
      || echo "[chain] $DS normality FAILED $(date -Is)"
    echo "[chain] $DS done $(date -Is)"
  done
}

run_chain HateMM MHC     >> "$LOG/run.log" 2>&1 &
run_chain HateClipSeg MHC_zh >> "$LOG/run.log" 2>&1 &
wait
echo "[chain] ALL DONE $(date -Is)" >> "$LOG/run.log"
