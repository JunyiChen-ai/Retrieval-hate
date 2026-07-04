#!/bin/bash
# Poll until feature extraction (job 12233) finishes or produces ~all npz.
JOB=12233
FEATDIR=/data/jehc223/RGCL/data/multihateloc_feats/HateMM
for i in $(seq 1 720); do   # up to ~24h at 120s
  N=$(ls $FEATDIR/*.npz 2>/dev/null | wc -l)
  ST=$(squeue -j $JOB -h -o "%t" 2>/dev/null)
  echo "$(date +%H:%M:%S) npz=$N/1083 jobstate='${ST:-GONE}'"
  if [ "$N" -ge 1080 ]; then echo "DONE_ENOUGH"; exit 0; fi
  if [ -z "$ST" ]; then echo "JOB_GONE npz=$N"; exit 0; fi
  sleep 120
done
echo "TIMEOUT"; exit 0
