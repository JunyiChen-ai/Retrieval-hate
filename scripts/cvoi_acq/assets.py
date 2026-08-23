from __future__ import annotations
import argparse,json
from pathlib import Path
from .common import ROOT,atomic_json,sha256_file

OCR_DIRS=[Path("/home/jehc223/.paddlex/official_models/PP-OCRv6_medium_det"),Path("/home/jehc223/.paddlex/official_models/PP-OCRv6_medium_rec")]
def record(path,role):
    return {"path":str(path.resolve()),"role":role,"bytes":path.stat().st_size,"sha256":sha256_file(path)}
def build(out,extra):
    paths=[(ROOT/"data/gt/HateMM/train.jsonl","train_labels_text"),(ROOT/"data/gt/HateMM/val.jsonl","val_sealed_labels_text"),
           (ROOT/"data/OCR/HateMM/ocr_windows_K30.jsonl","ocr_source"),(ROOT/"data/OCR/HateMM/meta.json","ocr_metadata")]
    for d in OCR_DIRS:
        for name in ("inference.json","inference.yml","inference.pdiparams"):paths.append((d/name,"ocr_engine_bytes"))
    paths.extend((p,"registered_extra") for p in extra)
    missing=[str(p) for p,_ in paths if not p.exists()]
    if missing:raise RuntimeError("HALT_ASSET_MISSING:"+",".join(missing))
    rows=[record(p,r) for p,r in paths];obj={"schema":"cvoi-asset-registry/1","assets":rows}
    atomic_json(out,obj);return {"n":len(rows)}
def main():
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True);a.add_argument("--extra",type=Path,action="append",default=[]);x=a.parse_args();print(json.dumps(build(x.out,x.extra),sort_keys=True))
if __name__=="__main__":main()
