from __future__ import annotations

import argparse, json
from pathlib import Path
from .common import atomic_json, sha256_file
from .lock import COMPLETENESS, load_ledger, publish_ledger

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--init",action="store_true");ap.add_argument("--pass-gate",choices=COMPLETENESS)
    ap.add_argument("--evidence",type=Path,action="append",default=[]);a=ap.parse_args()
    state=load_ledger(a.input) if a.input else load_ledger(Path("/__cvoi_missing_ledger__"))
    if a.pass_gate:
        if not a.evidence: raise RuntimeError("PASS requires evidence")
        state["gates"][a.pass_gate]={"status":"PASS","evidence":[{"path":str(p.resolve()),"sha256":sha256_file(p)} for p in a.evidence]}
    if a.out.exists(): raise FileExistsError("ledger is append-versioned; output exists")
    publish_ledger(a.out,state);print(json.dumps(state,sort_keys=True))
if __name__=="__main__":main()
