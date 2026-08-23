from __future__ import annotations
import argparse,json,random,statistics,subprocess,datetime,os
from collections import OrderedDict
from pathlib import Path
from .actions import read_frame,video_path
from .common import ContactLedger,atomic_json,atomic_write,canonical_bytes,sha256_file
from .costs import benchmark_phased,hardware_software_lock

def load_actions(paths,ledger):
    rows=[]
    for p in paths:
        ledger.register(p,"cost_action_registry");rows.extend(json.loads(x) for x in p.open() if x.strip())
    return rows

def dense_factory(rows):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,CLIPVisionModel
    model_id="openai/clip-vit-large-patch14-336";proc=AutoImageProcessor.from_pretrained(model_id,local_files_only=True)
    model=CLIPVisionModel.from_pretrained(model_id,local_files_only=True).eval().cuda();by={r["action_id"]:r for r in rows}
    def factory(aid):
        r=by[aid];path=video_path(r["video_id"]);times=[r["window_start_s"]+(m+.5)*(r["window_end_s"]-r["window_start_s"])/4 for m in range(4)]
        def decode(_):
            from decord import VideoReader,cpu
            import numpy as np
            frames=[];retries=0
            try:
                vr=VideoReader(str(path),ctx=cpu(0),num_threads=1);fps=float(vr.get_avg_fps() or 0);n=len(vr)
                if fps<=0 or n<=0:raise RuntimeError("empty video")
                idx=[max(0,min(n-1,int(np.floor(t*fps+.5)))) for t in times]
                batch=vr.get_batch(idx).asnumpy();frames=[batch[j] for j in range(4)]
            except Exception:
                # The registered action still consumes the failed attempt and exposes a
                # missing outcome. retries counts four failed slots; there is no free retry.
                frames=[None]*4;retries=4
            return {"frames":frames,"retries":retries}
        def preprocess(v):
            good=[x for x in v["frames"] if x is not None]
            if not good:return {**v,"pixels":None}
            # Nearest successful slot is the frozen dense4 fallback rule.
            imgs=[]
            for j,x in enumerate(v["frames"]):
                if x is None:
                    q=min((z for z,y in enumerate(v["frames"]) if y is not None),key=lambda z:(abs(z-j),z));x=v["frames"][q]
                imgs.append(Image.fromarray(x))
            return {**v,"pixels":proc(images=imgs,return_tensors="pt")["pixel_values"].cuda()}
        def inference(v):
            if v["pixels"] is not None:
                with torch.inference_mode():v["features"]=model(pixel_values=v["pixels"]).pooler_output
            return v
        def post(v):
            if "features" in v:v["outcome"]=v["features"].mean(0).float().cpu()
            return v
        return OrderedDict((("decode",decode),("preprocess",preprocess),("inference",inference),("postprocess",post)))
    return factory

def ocr_factory(rows):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ocr_cache"))
    from extract_ocr_windows import PaddleOCREngine
    engine=PaddleOCREngine(lang="en",gpu=True);by={r["action_id"]:r for r in rows}
    def factory(aid):
        r=by[aid];path=video_path(r["video_id"]);t=r["sample_t_s"]
        def decode(_):
            rgb,_actual=read_frame(path,t);return {"frame":None if rgb is None else rgb[:,:,::-1].copy(),"retries":int(rgb is None)}
        def preprocess(v):return v
        def inference(v):
            v["detections"]=[] if v["frame"] is None else engine.run([v["frame"]])[0];return v
        def post(v):v["normalized"]=[d for d in v["detections"] if d["conf"]>=.5 and len(d["text"].strip())>=2];return v
        return OrderedDict((("decode",decode),("preprocess",preprocess),("inference",inference),("postprocess",post)))
    return factory

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--type",choices=["ocr","dense"],required=True);ap.add_argument("--train-actions",type=Path,required=True);ap.add_argument("--val-actions",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    ledger=ContactLedger();train=load_actions([a.train_actions],ledger);val=load_actions([a.val_actions],ledger)
    for vid in sorted({r["video_id"] for r in train+val}):
        p=video_path(vid)
        if p is None:raise RuntimeError("HALT_COST_VIDEO_MISSING:"+vid)
        ledger.register(p,"raw_video_cost_benchmark")
    atomic_json(a.out.with_suffix(".start.json"),{"schema":"cvoi-cost-start/1","type":a.type,
      "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"pid":os.getpid(),
      "seed":20260814,"warmups":100,"repetitions":5,"n_train_actions":len(train),"n_val_actions":len(val),
      "train_actions_sha256":sha256_file(a.train_actions),"val_actions_sha256":sha256_file(a.val_actions),
      "hardware_software_lock":hardware_software_lock(),"contact":ledger.snapshot(),
      "candidate_metric_computed":False,"test_contact_count":ledger.test_contact_count})
    # Dense actions derive their timing coordinates from OCR's identical K30 boundaries.
    rows=[]
    for r in train+val:
        q=dict(r);q["action_id"]=r["video_id"]+(":ocr:" if a.type=="ocr" else ":dense4:")+format(r["window_id"],"02d");rows.append(q)
    headers={}
    for vid in sorted({r["video_id"] for r in rows}):
        cmd=["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height","-of","json",str(video_path(vid))]
        obj=json.loads(subprocess.run(cmd,check=True,text=True,capture_output=True).stdout);stream=obj["streams"][0]
        headers[vid]={"source_width":int(stream["width"]),"source_height":int(stream["height"])}
    factory=ocr_factory(rows) if a.type=="ocr" else dense_factory(rows)
    ids=[r["action_id"] for r in rows];by_coord={r["action_id"]:r for r in rows};warm=[r["video_id"]+(":ocr:" if a.type=="ocr" else ":dense4:")+format(r["window_id"],"02d") for r in train]
    if len(ids)!=len(set(ids)):raise RuntimeError("HALT_COST_DUPLICATE_ACTION")
    rng=random.Random(20260814)
    # Exactly 100 train-only warmups. They are deliberately not included in raw costs.
    for _ in range(100):
        value=None
        for fn in factory(warm[rng.randrange(len(warm))]).values():value=fn(value)
    order=[aid for aid in ids for _ in range(5)];rng.shuffle(order)
    raw=[]
    for seq,aid in enumerate(order):
        q=benchmark_phased(aid,factory(aid),repetitions=1,use_cuda=True)["raw_repetitions"][0]
        q.update({"sequence_index":seq,"action_id":aid})
        raw.append(q)
    by={aid:[] for aid in ids}
    for q in raw:by[q["action_id"]].append(q)
    result=[]
    for aid in ids:
        rr=by[aid]
        for rep,q in enumerate(rr,1):q["repetition"]=rep
        result.append({"schema":"cvoi-action-cost/1","action_id":aid,"action_type":a.type,
          "cheap_cost_covariates":{"duration_s":float(by_coord[aid]["window_end_s"])*30.0,
            **headers[aid.split(":")[0]],"window_index":int(aid.rsplit(":",1)[1])},
          "raw_repetitions":rr,"binding_wall_ns":int(statistics.median(x["wall_ns"] for x in rr[1:])),
          "binding_cuda_ms":float(statistics.median(x["cuda_ms"] for x in rr[1:]))})
    atomic_write(a.out,b"".join(canonical_bytes(r) for r in result))
    atomic_json(a.out.with_suffix(".meta.json"),{"schema":"cvoi-cost-run/1","type":a.type,
      "seed":20260814,"warmups":100,"repetitions":5,"binding_repetitions":[2,3,4,5],
      "batch_size":1,"n_actions":len(result),"n_train_actions":len(train),"n_val_actions":len(val),
      "cost_sha256":sha256_file(a.out),"hardware_software_lock":hardware_software_lock(),
      "header_probe":"ffprobe stream width,height only; no frame decode","contact":ledger.snapshot(),"candidate_metric_computed":False})
if __name__=="__main__":main()
