#!/usr/bin/env bash
# SAV (C2) F-G0 -> guard -> F-G1 chain wrapper (fail-closed jq gating between stages).
# Authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED).
#
# Stage order (single serial lineage; the guard's PRIMARY gates the probe):
#   1. sav_f0_extract.py  : frozen per-head + pooled extraction (train+val, 3 datasets)
#   2. sav_f0_guard.py    : two-tier reproduction guard; PRIMARY min-cosine >= 0.999
#   3. sav_f0_probe.py    : F-G1 statistics engine -> machine-readable verdict.json
#
# All intermediate/temp files live IN-REPO (artifacts/sav_f0/, slurm/tmp/) — never
# $TMPDIR (realbank $TMPDIR burn lesson). Extraction is resumable/idempotent
# (skip-if-exists), so a re-run continues; nothing is force-deleted on failure.
set -euo pipefail

cd /data/jehc223/RGCL

RUN_ID=${RUN_ID:?RUN_ID required}
EXPECTED=${EXPECTED:-SAV-F0-FG0-FG1}
DATASETS=${DATASETS:-HateMM,MHC,MHC_zh}
LIMIT=${LIMIT:-0}

if [[ "$RUN_ID" != "$EXPECTED" ]]; then
  echo "Refusing unauthorized RUN_ID=$RUN_ID (expected $EXPECTED)" >&2
  exit 2
fi

command -v jq >/dev/null 2>&1 || { echo "jq not found" >&2; exit 2; }
mkdir -p slurm/tmp

ART=artifacts/sav_f0

echo "########## [SAV F-G0] extraction (datasets=$DATASETS limit=$LIMIT) ##########"
python scripts/analysis/sav_f0_extract.py --datasets "$DATASETS" --splits train,val --limit "$LIMIT"

# gate 1: every (dataset,split) manifest complete and full-count
IFS=',' read -ra DS <<< "$DATASETS"
for ds in "${DS[@]}"; do
  for sp in train val; do
    MF="$ART/extract/$ds/$sp/_manifest.json"
    jq -e '.complete == true and (.n == .n_expected)' "$MF" >/dev/null \
      || { echo "FAIL: extraction manifest not complete: $MF" >&2; exit 3; }
  done
done
echo "[gate1] all extraction manifests complete."

echo "########## [SAV F-G0(b)] two-tier reproduction guard ##########"
python scripts/analysis/sav_f0_guard.py --datasets "$DATASETS"

# gate 2: PRIMARY guard must pass for every dataset (hard fail = drift)
for ds in "${DS[@]}"; do
  GF="$ART/guard/$ds/guard.json"
  jq -e '.pass == true' "$GF" >/dev/null \
    || { echo "FAIL: reproduction guard PRIMARY failed: $GF" >&2; exit 4; }
done
echo "[gate2] reproduction guard PRIMARY passed on all datasets."

echo "########## [SAV F-G1] statistics engine ##########"
python scripts/analysis/sav_f0_probe.py --datasets "$DATASETS"

# gate 3: verdict emitted and complete (fail-closed on missing arm)
VF="$ART/probe/verdict.json"
jq -e '.status == "COMPLETE"' "$VF" >/dev/null \
  || { echo "FAIL: probe verdict not COMPLETE: $VF" >&2; exit 5; }
echo "[gate3] F-G1 verdict: $(jq -r '.verdict' "$VF")"
echo "########## SAV F-G0/F-G1 chain COMPLETE ##########"
