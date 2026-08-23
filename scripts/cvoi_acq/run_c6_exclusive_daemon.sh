#!/usr/bin/env bash
# C6 measured-cost daemon (deviation D2, 2026-08-09).
#
# Replaces the foreground poller `run_c6_local_when_free.sh`, which violated the
# repository rule that local long jobs must be detached from the SSH session and
# externally inspectable. This script:
#   1. verifies the four pinned C6 source files still match preflight_review_v2.json
#      (v1 superseded by deviation D3, 2026-08-09: its digests predated a 10:58 code edit);
#   2. waits for a stable GPU-exclusive window (no foreign compute process);
#   3. samples co-tenancy for the whole timing run;
#   4. runs the registered C6 chain unchanged into a per-attempt directory;
#   5. voids the attempt if any foreign compute process appeared, and retries;
#   6. promotes only an EXCLUSIVE_OK attempt to the canonical C6 artifact names.
#
# It changes no pinned module, no protocol constant, and computes no candidate metric.
set -uo pipefail

ROOT=/home/jehc223/Retrieval-hate
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

RUN=logging/runs/cvoi_c6_cost
OUT=artifacts/cvoi_acq/premetric-v2/c6-cost-v1
ACT=artifacts/cvoi_acq/premetric-v2/actions
GRP=artifacts/cvoi_acq/premetric-v2/groups-v4
POLL_S=${POLL_S:-600}
CONFIRM_N=${CONFIRM_N:-3}
CONFIRM_S=${CONFIRM_S:-20}
SAMPLE_S=${SAMPLE_S:-15}

mkdir -p "$RUN" "$OUT"
echo $$ > "$RUN/run.pid"
OWN_PGID=$(ps -o pgid= -p $$ | tr -d ' ')

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "$RUN/run.log"; }
probe() { python -m scripts.cvoi_acq.c6_exclusivity probe --own-pgid "$OWN_PGID" >/dev/null 2>>"$RUN/run.log"; }

log "DAEMON_START pid=$$ pgid=$OWN_PGID poll_s=$POLL_S sample_s=$SAMPLE_S nproc=$(nproc) omp=${OMP_NUM_THREADS:-unset}"

# --- pinned-source integrity ------------------------------------------------
python - <<'PY' >> "$RUN/run.log" 2>&1
import hashlib, json, sys
from pathlib import Path
ref = json.load(open("artifacts/cvoi_acq/premetric-v2/c6-cost-v1/preflight_review_v2.json"))
bad = []
for rel, want in sorted(ref["source_sha256"].items()):
    got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    print(("OK  " if got == want else "DRIFT ") + rel + " " + got)
    if got != want:
        bad.append(rel)
sys.exit(9 if bad else 0)
PY
if [ $? -ne 0 ]; then
  log "HALT_C6_PINNED_SOURCE_DRIFT"
  echo HALT_C6_PINNED_SOURCE_DRIFT > "$RUN/status"
  exit 9
fi
log "PINNED_SOURCE_OK"

attempt=0
while true; do
  if ! probe; then
    log "WAIT gpu_not_exclusive"
    sleep "$POLL_S"
    continue
  fi
  stable=1
  for _ in $(seq 1 "$CONFIRM_N"); do
    sleep "$CONFIRM_S"
    probe || { stable=0; break; }
  done
  if [ "$stable" -ne 1 ]; then
    log "WAIT exclusivity_not_stable"
    sleep "$POLL_S"
    continue
  fi

  attempt=$((attempt + 1))
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  ADIR="$OUT/attempt-$STAMP"
  mkdir -p "$ADIR"
  log "ATTEMPT_START n=$attempt dir=$ADIR"

  python -m scripts.cvoi_acq.c6_exclusivity watch --own-pgid "$OWN_PGID" \
    --interval "$SAMPLE_S" --out "$ADIR/cotenancy.jsonl" >/dev/null 2>&1 &
  WPID=$!
  START_UTC=$(date -u --iso-8601=seconds)

  # heartbeat: external progress rows, no pinned module is modified
  ( while true; do
      sleep 600
      log "HEARTBEAT attempt=$attempt bytes=$(du -sb "$ADIR" 2>/dev/null | cut -f1)"
    done ) &
  HPID=$!

  rc=0
  python -m scripts.cvoi_acq.cost_driver --type ocr \
    --train-actions "$ACT/train_ocr_actions.jsonl" \
    --val-actions "$ACT/val_ocr_actions.jsonl" \
    --out "$ADIR/ocr_cost_actions.jsonl" > "$ADIR/ocr.log" 2>&1 || rc=$?
  log "STAGE ocr rc=$rc"

  if [ $rc -eq 0 ]; then
    python -m scripts.cvoi_acq.cost_driver --type dense \
      --train-actions "$ACT/train_ocr_actions.jsonl" \
      --val-actions "$ACT/val_ocr_actions.jsonl" \
      --out "$ADIR/dense_cost_actions.jsonl" > "$ADIR/dense.log" 2>&1 || rc=$?
    log "STAGE dense rc=$rc"
  fi

  if [ $rc -eq 0 ]; then
    python -m scripts.cvoi_acq.cost_overhead_driver \
      --out "$ADIR/overhead_costs.jsonl" > "$ADIR/overhead.log" 2>&1 || rc=$?
    log "STAGE overhead rc=$rc"
  fi

  if [ $rc -eq 0 ]; then
    python -m scripts.cvoi_acq.cost_audit \
      --ocr "$ADIR/ocr_cost_actions.jsonl" --dense "$ADIR/dense_cost_actions.jsonl" \
      --train-actions "$ACT/train_ocr_actions.jsonl" --val-actions "$ACT/val_ocr_actions.jsonl" \
      --components "$GRP/group_components.json" --outer "$GRP/outer_folds.json" \
      --out "$ADIR/independent_audit_v1.json" > "$ADIR/audit.log" 2>&1 || rc=$?
    log "STAGE audit rc=$rc"
  fi

  END_UTC=$(date -u --iso-8601=seconds)
  kill "$HPID" 2>/dev/null; wait "$HPID" 2>/dev/null
  kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null

  vrc=0
  python -m scripts.cvoi_acq.c6_exclusivity verify --samples "$ADIR/cotenancy.jsonl" \
    --start-utc "$START_UTC" --end-utc "$END_UTC" --interval "$SAMPLE_S" \
    --out "$ADIR/cotenancy_verdict.json" >> "$RUN/run.log" 2>&1 || vrc=$?
  log "VERIFY rc=$vrc"

  if [ $rc -eq 0 ] && [ $vrc -eq 0 ]; then
    for f in ocr_cost_actions.jsonl ocr_cost_actions.meta.json ocr_cost_actions.start.json \
             dense_cost_actions.jsonl dense_cost_actions.meta.json dense_cost_actions.start.json \
             overhead_costs.jsonl independent_audit_v1.json cotenancy.jsonl cotenancy_verdict.json; do
      [ -f "$ADIR/$f" ] && cp -p "$ADIR/$f" "$OUT/$f"
    done
    python - "$ADIR" "$START_UTC" "$END_UTC" <<'PY' >> "$RUN/run.log" 2>&1
import hashlib, json, os, sys
from pathlib import Path
adir, start, end = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out = adir.parent
rec = {"schema": "cvoi-c6-bound-attempt/1", "deviation": "D2",
       "attempt_dir": str(adir), "run_start_utc": start, "run_end_utc": end,
       "cotenancy_status": json.load(open(adir / "cotenancy_verdict.json"))["status"],
       "candidate_metric_computed": False, "test_contact_count": 0, "sha256": {}}
for p in sorted(out.glob("*")):
    if p.is_file():
        rec["sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
tmp = out / "bound_attempt.json.tmp"
tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
os.replace(tmp, out / "bound_attempt.json")
print("PROMOTED " + str(adir))
PY
    log "ATTEMPT_BOUND dir=$ADIR"
    echo BOUND > "$RUN/status"
    exit 0
  fi

  : > "$ADIR/VOID"
  log "ATTEMPT_VOID n=$attempt rc=$rc verify_rc=$vrc dir=$ADIR (timings discarded, not promoted)"
  echo "VOID attempt=$attempt rc=$rc verify_rc=$vrc" > "$RUN/status"
  sleep "$POLL_S"
done
