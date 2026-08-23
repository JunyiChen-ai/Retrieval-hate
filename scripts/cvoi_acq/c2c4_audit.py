from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path

from .common import ContactLedger,atomic_json,sha256_file,sorted_id_bytes
from .lineage import AUDIT,SAMPLE,STRICT_HASH,BROAD_HASH


def lineage_check(ledger):
    ledger.register(AUDIT,"diagnostic_gate_c_audit");ledger.register(SAMPLE,"diagnostic_gate_c_sample")
    rows=[json.loads(x) for x in AUDIT.open() if x.strip()];sample=json.loads(SAMPLE.read_text())
    final={r["video_id"]:r for r in rows if r["coder_id"].endswith("c1")};final.update({r["video_id"]:r for r in rows if r["coder_id"].endswith("adj")})
    strict=[];broad=[]
    for vid in sorted(set(sample["audit_fn"])):
        req=set(final[vid]["required_modalities"])
        if "on_screen_text" in req and "speech" not in req:
            broad.append(vid)
            if "transcript" not in req:strict.append(vid)
    return {"source_audit_sha256":sha256_file(AUDIT),"source_sample_sha256":sha256_file(SAMPLE),
            "strict":{"count":len(strict),"sha256":hashlib.sha256(sorted_id_bytes(strict)).hexdigest(),"expected":STRICT_HASH},
            "broad":{"count":len(broad),"sha256":hashlib.sha256(sorted_id_bytes(broad)).hexdigest(),"expected":BROAD_HASH}}

def action_check(path,role,expected_videos,ledger):
    ledger.register(path,"canonical_ocr_actions");rows=[json.loads(x) for x in path.open() if x.strip()]
    required={"schema","video_id","action_id","window_id"};ids=set();keys=set()
    for r in rows:
        if not required<=set(r):raise RuntimeError("HALT_OCR_SCHEMA")
        if r["action_id"]!=r["video_id"]+":ocr:"+format(int(r["window_id"]),"02d"):raise RuntimeError("HALT_OCR_ACTION_ID")
        key=(r["video_id"],int(r["window_id"]));
        if key in keys:raise RuntimeError("HALT_OCR_DUPLICATE")
        keys.add(key);ids.add(r["video_id"])
    if len(ids)!=expected_videos or len(rows)!=expected_videos*30 or any((v,k) not in keys for v in ids for k in range(30)):raise RuntimeError("HALT_OCR_CARDINALITY")
    return {"role":role,"path":str(path.resolve()),"sha256":sha256_file(path),"n_videos":len(ids),"n_actions":len(rows),"schema":sorted({r["schema"] for r in rows})}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();ledger=ContactLedger()
    engine=[]
    for d in (Path("/home/jehc223/.paddlex/official_models/PP-OCRv6_medium_det"),Path("/home/jehc223/.paddlex/official_models/PP-OCRv6_medium_rec")):
        for name in ("inference.json","inference.yml","inference.pdiparams"):
            p=d/name;ledger.register(p,"ocr_engine_bytes");engine.append({"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    payload={"schema":"cvoi-c2-c4-audit/1","C2":lineage_check(ledger),
      "C4":{"actions":[action_check(Path("artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl"),"train",744,ledger),action_check(Path("artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl"),"val",107,ledger)],"engine_bytes":engine},
      "contact":ledger.snapshot(),"candidate_metrics_read":False}
    if payload["C2"]["strict"]["sha256"]!=STRICT_HASH or payload["C2"]["broad"]["sha256"]!=BROAD_HASH:raise RuntimeError("HALT_LINEAGE_REPLAY")
    atomic_json(a.out,payload)

if __name__=="__main__":main()
