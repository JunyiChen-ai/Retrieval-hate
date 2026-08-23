from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .common import atomic_json,canonical_bytes,sha256_file
from .lock import COMPLETENESS,load_ledger

def build(completeness,prereg,appendix,sources,out,signature_file=None):
    state=load_ledger(completeness);pending=[k for k in COMPLETENESS if state["gates"][k]["status"]!="PASS"]
    if pending:raise RuntimeError("HALT_FREEZE_INCOMPLETE:"+",".join(pending))
    file_map={str(p.resolve()):sha256_file(p) for p in sorted(sources,key=lambda p:str(p.resolve()))}
    body={"study_id":"CVOI-ACQ-v1","prereg_sha256":sha256_file(prereg),"appendix_sha256":sha256_file(appendix),
          "completeness_sha256":sha256_file(completeness),"source_file_sha256":file_map}
    review_payload=hashlib.sha256(canonical_bytes(body)).hexdigest()
    if signature_file is None:
        body.update({"review_payload_sha256":review_payload,"status":"UNSIGNED"});atomic_json(out,body);return
    sig=json.loads(signature_file.read_text())
    if sig.get("review_payload_sha256")!=review_payload or not sig.get("reviewer_id") or not sig.get("signed_utc"):
        raise RuntimeError("HALT_BAD_INDEPENDENT_SIGNATURE")
    body.update({"review_payload_sha256":review_payload,"independent_signature":sig,"status":"FROZEN"})
    body["payload_sha256"]=hashlib.sha256(canonical_bytes(body)).hexdigest();atomic_json(out,body)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--completeness",type=Path,required=True);ap.add_argument("--prereg",type=Path,required=True);ap.add_argument("--appendix",type=Path,required=True);ap.add_argument("--source",type=Path,action="append",default=[]);ap.add_argument("--signature-file",type=Path);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();build(a.completeness,a.prereg,a.appendix,a.source,a.out,a.signature_file)
if __name__=="__main__":main()
