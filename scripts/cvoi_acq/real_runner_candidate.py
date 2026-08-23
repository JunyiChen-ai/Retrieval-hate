"""Schema-only candidate wiring for a future real CVoI executor.

This file is deliberately independent of the signed C14 runner.  It cannot
compute or load predictions/metrics and defaults to fail-closed until a full
materialized ledger and independent signature are supplied.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from .common import ContactLedger, atomic_json, sha256_file
from .lock import COMPLETENESS

REQUIRED_REAL_ASSET_GATES=("C1","C5","C6")

def preflight(completeness:Path,signature:Path,schedule:Path)->dict:
    ledger_contact=ContactLedger()
    for path,role in ((completeness,"full_completeness_ledger"),(signature,"independent_signature"),(schedule,"signed_schedule")):
        ledger_contact.register(path,role)
        if not path.exists():raise RuntimeError("HALT_REAL_PREFLIGHT_MISSING:"+role)
    obj=json.loads(completeness.read_text())
    if obj.get("schema")!="cvoi-completeness/1" or set(obj.get("gates",{}))!=set(COMPLETENESS):
        raise RuntimeError("HALT_REAL_PREFLIGHT_FULL_LEDGER_SCHEMA")
    missing_assets=[g for g in REQUIRED_REAL_ASSET_GATES if obj["gates"][g].get("status")!="PASS"]
    if missing_assets:raise RuntimeError("HALT_REAL_PREFLIGHT_ASSETS:"+",".join(missing_assets))
    pending=[g for g in COMPLETENESS if obj["gates"][g].get("status")!="PASS"]
    if pending:raise RuntimeError("HALT_REAL_PREFLIGHT_LEDGER:"+",".join(pending))
    sig=json.loads(signature.read_text())
    required={"schema","reviewer","completeness_sha256","schedule_sha256","signed"}
    if not required<=set(sig) or sig["schema"]!="cvoi-independent-real-run-signature/1" or sig["signed"] is not True:
        raise RuntimeError("HALT_REAL_PREFLIGHT_SIGNATURE_SCHEMA")
    if sig["completeness_sha256"]!=sha256_file(completeness) or sig["schedule_sha256"]!=sha256_file(schedule):
        raise RuntimeError("HALT_REAL_PREFLIGHT_SIGNATURE_HASH")
    sched=json.loads(schedule.read_text())
    if sched.get("n_runs")!=45 or len(sched.get("runs",[]))!=45 or sched.get("metric_locked") is not True:
        raise RuntimeError("HALT_REAL_PREFLIGHT_SCHEDULE")
    return {"schema":"cvoi-real-runner-candidate-preflight/1","status":"SCHEMA_WIRING_PASS_NO_METRICS",
            "execution_authorized":False,"candidate_metric_computed":False,"test_contact_count":ledger_contact.test_contact_count,
            "inputs":{"completeness_sha256":sha256_file(completeness),"signature_sha256":sha256_file(signature),"schedule_sha256":sha256_file(schedule)}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--completeness",type=Path,required=True);ap.add_argument("--signature",type=Path,required=True);ap.add_argument("--schedule",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    report=preflight(a.completeness,a.signature,a.schedule)
    atomic_json(a.out,report)
    raise RuntimeError("HALT_REAL_EXECUTION_BODY_NOT_SIGNED")

if __name__=="__main__":main()
