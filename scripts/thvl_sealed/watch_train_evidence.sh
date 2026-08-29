#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/jehc223/miniconda3/envs/HateVideo/bin/python
BASE="$ROOT/results/steward_private/thvl_bench/train314"
LOG="$BASE/watcher.log"
STATUS="$BASE/watcher_status.json"
ASR_PID="$1"
cd "$ROOT" || exit 90
exec >>"$LOG" 2>&1
fail(){ code="$1"; stage="$2"; printf '{"status":"FAILED","stage":"%s","exit_code":%s,"utc":"%s"}\n' "$stage" "$code" "$(date -u +%FT%TZ)" >"$STATUS.tmp"; sync "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; exit "$code"; }
mark(){ printf '{"status":"RUNNING","stage":"%s","utc":"%s","watcher_pid":%s,"asr_pid":%s}\n' "$1" "$(date -u +%FT%TZ)" "$$" "$ASR_PID" >"$STATUS.tmp"; sync "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; }
run(){ stage="$1";shift;mark "$stage";"$@"; rc=$?; [ "$rc" -eq 0 ] || fail "$rc" "$stage"; }
mark monitor_asr
while kill -0 "$ASR_PID" 2>/dev/null; do n=$(find "$BASE/asr_records" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l); echo "$(date -u +%FT%TZ) ASR $n/314 pid=$ASR_PID"; sleep 60; done
n=$(find "$BASE/asr_records" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
[ "$n" -eq 314 ] || fail 71 asr_coverage_after_process_exit
if [ -f "$BASE/final_manifest.json" ]; then
 run verify_finalized_asr "$PY" -c "import json,hashlib;from pathlib import Path;p=Path('$BASE');m=json.load(open(p/'final_manifest.json'));h=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest();assert m['n_videos']==314 and m['timestamped_chunks_sha256']==h(p/'timestamped_chunks.jsonl') and m['windows_sha256']==h(p/'windows30s_full_coverage.jsonl')"
else
 run finalize_asr "$PY" scripts/thvl_sealed/train_media_asr.py --stage finalize
fi
V16="$BASE/v16_train_causal_frozen"
if [ -f "$V16/raw_manifest.json" ]; then
 run verify_v16 "$PY" -c "import json,hashlib;from pathlib import Path;p=Path('$V16');m=json.load(open(p/'raw_manifest.json'));c=json.load(open(p/'preregistered_config.json'));h=lambda x:hashlib.sha256(open(x,'rb').read()).hexdigest();assert c['n_videos']==314 and m['n_rows']==c['n_valid_chunks'] and m['raw_sha256']==h(p/'per_chunk_raw.jsonl') and m['config_sha256']==h(p/'preregistered_config.json')"
else
 [ ! -e "$V16" ] || fail 72 v16_incomplete_existing_output
 run preregister_v16 "$PY" scripts/thvl_sealed/preregister_train_v16.py --asr "$BASE/timestamped_chunks.jsonl" --qc-dir "$BASE/qc_records" --out-dir "$V16"
 run forward_v16 env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reproduction_baselines/relation_v16/forward.py --out-dir "$V16"
fi
[ -f "$V16/raw_manifest.json" ] || fail 73 v16_manifest_missing
nv=$($PY -c "import json;print(json.load(open('$V16/preregistered_config.json'))['n_videos'])")
[ "$nv" -eq 314 ] || fail 74 v16_coverage
EVID="$BASE/v24_train_evidence_frozen"
if [ -f "$EVID/evidence_manifest.json" ]; then
 run verify_train_evidence "$PY" -c "import json,hashlib;from pathlib import Path;p=Path('$EVID');m=json.load(open(p/'evidence_manifest.json'));assert m['n_videos']==314 and len(m['records'])==314 and all(hashlib.sha256(open(p/'records'/(v+'.json'),'rb').read()).hexdigest()==h for v,h in m['records'].items())"
else
 [ ! -e "$EVID" ] || fail 75 v24_incomplete_existing_output
 run v24_prepare "$PY" scripts/reproduction_baselines/relation_v24/evidence_producer.py --qc-dir "$BASE/qc_records" --asr-dir "$BASE/asr_records" --v16-dir "$V16" --split train --out-dir "$EVID" --prepare-only
 run v24_forward env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reproduction_baselines/relation_v24/evidence_producer.py --qc-dir "$BASE/qc_records" --asr-dir "$BASE/asr_records" --v16-dir "$V16" --split train --out-dir "$EVID"
fi
ne=$($PY -c "import json;print(json.load(open('$EVID/evidence_manifest.json'))['n_videos'])")
[ "$ne" -eq 314 ] || fail 76 evidence_coverage
JOIN="$BASE/v24_train_bags_frozen"
[ ! -e "$JOIN" ] || fail 77 join_fresh_output
run steward_join "$PY" scripts/reproduction_baselines/relation_v24/steward_join.py --evidence-dir "$EVID" --weak-manifest results/reproduction/thvl_sealed/train_media_manifest.json --out-dir "$JOIN"
nb=$(wc -l <"$JOIN/bags.jsonl")
[ "$nb" -eq 314 ] || fail 78 bags_coverage
VAL_ATOMIC="$ROOT/results/steward_private/thvl_bench/val32_v24_atomic"
[ ! -e "$VAL_ATOMIC" ] || fail 79 val_atomic_fresh_output
run val_atomic_prepare "$PY" scripts/thvl_sealed/prepare_val_atomic.py
VAL_EVID="$ROOT/results/steward_private/thvl_bench/v24_val_evidence_frozen"
[ ! -e "$VAL_EVID" ] || fail 80 val_evidence_fresh_output
VAL_V16="$ROOT/results/steward_private/thvl_bench/v16_val_raw_frozen_32_v2"
run v24_val_prepare "$PY" scripts/reproduction_baselines/relation_v24/evidence_producer.py --qc-dir "$VAL_ATOMIC/qc_records" --asr-dir "$VAL_ATOMIC/asr_records" --v16-dir "$VAL_V16" --split val --out-dir "$VAL_EVID" --prepare-only
run v24_val_forward env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reproduction_baselines/relation_v24/evidence_producer.py --qc-dir "$VAL_ATOMIC/qc_records" --asr-dir "$VAL_ATOMIC/asr_records" --v16-dir "$VAL_V16" --split val --out-dir "$VAL_EVID"
vn=$($PY -c "import json;print(json.load(open('$VAL_EVID/evidence_manifest.json'))['n_videos'])")
[ "$vn" -eq 32 ] || fail 81 val_evidence_coverage
printf '{"status":"COMPLETE_TRAIN_AND_VAL_EVIDENCE_ONLY_NO_TRAINING","stage":"done","utc":"%s","train_videos":314,"val_videos":32}\n' "$(date -u +%FT%TZ)" >"$STATUS.tmp";sync "$STATUS.tmp";mv "$STATUS.tmp" "$STATUS"
