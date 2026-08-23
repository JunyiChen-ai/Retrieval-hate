from __future__ import annotations

import argparse, hashlib, json, time, unicodedata
from pathlib import Path
import numpy as np
from .common import ContactLedger, ROOT, atomic_json, atomic_write, canonical_bytes, sha256_file

K = 30

def video_path(vid):
    root = ROOT / "data/video/HateMM/All"
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = root / (vid + ext)
        if p.exists(): return p.resolve()
    return None

def split_rows(role, ledger):
    p = ROOT / ("data/gt/HateMM/" + ("train" if role == "train" else "val") + ".jsonl")
    ledger.register(p, role + "_ids")
    return [json.loads(x) for x in p.open() if x.strip()]

def durations(ledger, registry):
    if registry is None: raise RuntimeError("HALT_SANITIZED_DURATION_REQUIRED")
    ledger.register(registry,"sanitized_duration_registry")
    obj=json.loads(registry.read_text())
    if obj.get("schema")!="cvoi-sanitized-durations/1":raise RuntimeError("HALT_DURATION_SCHEMA")
    return {r["video_id"]:float(r["duration_s"]) for r in obj["records"]}

def canonicalize_ocr(role, out, duration_registry):
    ledger = ContactLedger(); ids = {str(r["id"]) for r in split_rows(role, ledger)}; dur = durations(ledger,duration_registry)
    src = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"; ledger.register(src, "ocr_action_source")
    rows, seen = [], set()
    for line in src.open():
        r = json.loads(line); vid = str(r["video_id"])
        if vid not in ids: continue
        k = int(r["window_k"]); D = dur.get(vid)
        if D is None: raise RuntimeError("HALT_MISSING_DURATION:" + vid)
        dets = []
        for q, d in enumerate(r.get("texts", [])):
            text = " ".join(unicodedata.normalize("NFKC", str(d.get("text", ""))).split()).strip()
            conf = float(d.get("conf", 0)); box = d.get("bbox") or []
            if conf < .5 or len(text) < 2: continue
            ys = [float(p[1]) for p in box] if box else [float("inf")]
            xs = [float(p[0]) for p in box] if box else [float("inf")]
            dets.append((min(ys), min(xs), q, {"text": text, "conf": conf, "bbox": box}))
        dets.sort(key=lambda x: (x[0], x[1], x[2])); unique, used = [], set()
        for item in dets:
            d = item[3]
            if d["text"] not in used: used.add(d["text"]); unique.append(d)
        joined = " [SEP] ".join(d["text"] for d in unique)
        outcome={"texts":unique,"normalized_text":joined,"engine":r.get("engine"),
                 "engine_version":r.get("engine_version"),
                 "engine_status":r.get("engine_status","ok")}
        rows.append({"schema":"cvoi-ocr-action/1", "video_id":vid,
                     "action_id":vid + ":ocr:" + format(k, "02d"), "window_id":k,
                     "window_start_s":k*D/K, "window_end_s":(k+1)*D/K,
                     "sample_t_s":float(r["t_mid"]), **outcome,
                     "output_sha256":hashlib.sha256(canonical_bytes(outcome)).hexdigest()})
        seen.add((vid, k))
    missing = [(v,k) for v in ids for k in range(K) if (v,k) not in seen]
    if missing: raise RuntimeError("HALT_OCR_COVERAGE:" + str(len(missing)))
    rows.sort(key=lambda r:(r["video_id"],r["window_id"]))
    atomic_write(out, b"".join(canonical_bytes(r) for r in rows))
    return {"role":role,"n_videos":len(ids),"n_actions":len(rows),"sha256":sha256_file(out),"contact":ledger.snapshot()}

def read_frame(path, timestamp):
    from decord import VideoReader,cpu
    try:
        vr=VideoReader(str(path),ctx=cpu(0),num_threads=1);fps=float(vr.get_avg_fps() or 0);n=len(vr)
        if fps<=0 or n<=0:return None,None
        idx=max(0,min(n-1,int(np.floor(timestamp*fps+0.5))))
        return vr[idx].asnumpy(),idx/fps
    except Exception:return None,None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["ocr"]); ap.add_argument("--role",choices=["train","val"],required=True); ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--duration-registry",type=Path,required=True); args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps(canonicalize_ocr(args.role,args.out_dir/(args.role+"_ocr_actions.jsonl"),args.duration_registry),sort_keys=True))

if __name__=="__main__": main()
