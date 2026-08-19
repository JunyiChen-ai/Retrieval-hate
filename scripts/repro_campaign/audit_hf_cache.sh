#!/usr/bin/env bash
# REPRO campaign asset integrity: every HF-cache blob is stored under a filename
# equal to its sha256 (the LFS oid), so re-hashing every blob is a complete
# corruption check.  Motivated by openai/clip-vit-base-patch16's pytorch_model.bin,
# which downloaded with the right byte count but wrong content (max |w| = 3.7e19,
# every CLIP image feature collapsing to a constant vector).
set -u
OUT=${1:-/home/jehc223/Retrieval-hate/idea-stage/repro_campaign/hf_cache_audit.txt}
: > "$OUT"
find ~/.cache/huggingface/hub -type f -path '*/blobs/*' -printf '%s\t%p\n' | sort -rn |
while IFS=$'\t' read -r sz f; do
  b=$(basename "$f")
  [[ "$b" =~ ^[0-9a-f]{64}$ ]] || { echo -e "SKIP\t$sz\t$f" >> "$OUT"; continue; }
  h=$(sha256sum "$f" | cut -d' ' -f1)
  if [[ "$h" == "$b" ]]; then echo -e "OK\t$sz\t$f" >> "$OUT"
  else echo -e "CORRUPT\t$sz\t$f\tgot=$h" >> "$OUT"; fi
done
echo "DONE $(grep -c '^OK' "$OUT") ok, $(grep -c '^CORRUPT' "$OUT") corrupt, $(grep -c '^SKIP' "$OUT") unhashable" >> "$OUT"
