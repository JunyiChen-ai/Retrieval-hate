from __future__ import annotations
import argparse,json
from pathlib import Path
from .common import ContactLedger,ROOT,atomic_json

def build(out):
    ledger=ContactLedger();p=ROOT/"data/gt/HateMM/hate_spans.json";ledger.register(p,"duration_only_steward")
    raw=json.loads(p.read_text()); allowed={}
    for role in ("train","val"):
        sp=ROOT/("data/gt/HateMM/"+role+".jsonl");ledger.register(sp,role+"_duration_ids")
        for line in sp.open():
            if not line.strip():continue
            vid=str(json.loads(line)["id"]);entry=raw.get(vid) or {};D=entry.get("duration")
            if D is None or float(D)<=0:raise RuntimeError("HALT_DURATION:"+vid)
            allowed[vid]={"video_id":vid,"split_role":role,"duration_s":float(D)}
    obj={"schema":"cvoi-sanitized-durations/1","records":[allowed[k] for k in sorted(allowed)],"contact":ledger.snapshot(),
         "forbidden_source_keys_not_serialized":["spans","label"]}
    atomic_json(out,obj);return {"n":len(allowed)}
def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True);x=a.parse_args();print(json.dumps(build(x.out),sort_keys=True))
if __name__=="__main__":main()
