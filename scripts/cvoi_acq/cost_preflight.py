"""CPU-only C6 preflight; writes an independent review package, never PASS."""
from __future__ import annotations
import argparse,json,platform
from pathlib import Path
from .common import ContactLedger,atomic_json,sha256_file

def load(path,ledger,role):
    ledger.register(path,role);return [json.loads(x) for x in path.open() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[2]
    ledger=ContactLedger();ta=load(root/"artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl",ledger,"train_actions")
    va=load(root/"artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl",ledger,"val_actions")
    dense=load(root/"artifacts/cvoi_acq/premetric-v2/visual-v10/train_dense_sidecar.jsonl",ledger,"train_dense_sidecar")+load(root/"artifacts/cvoi_acq/premetric-v2/visual-v10/val_dense_sidecar.jsonl",ledger,"val_dense_sidecar")
    errs=[]
    if len(ta)!=744*30 or len(va)!=107*30:errs.append("ocr_coordinate_coverage")
    if len(dense)!=(744+107)*30*4:errs.append("dense_frame_coverage")
    if len({x["action_id"] for x in ta+va})!=(744+107)*30:errs.append("ocr_action_uniqueness")
    if len({(x["video_id"],x["window_id"],x["frame_slot"]) for x in dense})!=(744+107)*30*4:errs.append("dense_slot_uniqueness")
    code=[root/"scripts/cvoi_acq/costs.py",root/"scripts/cvoi_acq/cost_driver.py",root/"scripts/cvoi_acq/cost_audit.py",root/"scripts/cvoi_acq/cost_overhead_driver.py"]
    out={"schema":"cvoi-c6-preflight-review/1","status":"HALT" if errs else "PENDING_REAL_MEASUREMENT_AND_INDEPENDENT_REVIEW",
      "errors":errs,"coverage":{"train_videos":len({x["video_id"] for x in ta}),"val_videos":len({x["video_id"] for x in va}),
      "ocr_actions":len(ta)+len(va),"dense_actions":len(dense)//4,"dense_frames":len(dense)},
      "protocol":{"warmups":100,"repetitions":5,"binding_repetitions":[2,3,4,5],"seed":20260814,"batch_size":1},
      "source_sha256":{str(p.relative_to(root)):sha256_file(p) for p in code},"python":platform.python_version(),
      "contact":ledger.snapshot(),"candidate_metric_computed":False,"gate_promoted":False}
    atomic_json(a.out,out)
if __name__=="__main__":main()
