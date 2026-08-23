from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, canonical_bytes, sha256_file


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--video-root",type=Path,default=Path("data/video/HateMM/All"))
    ap.add_argument("--max-videos",type=int)
    ap.add_argument("--tolerance",type=float,default=5e-5)
    a=ap.parse_args()
    from transformers import CLIPProcessor,CLIPVisionModel
    from src.utils.generate_subclip_embedding_HF import load_video_frames,encode_frames_pooled,_window_bounds
    old=torch.load(a.cache,map_location="cpu",weights_only=False)
    ids=list(old["video_ids"]);K=int(old["num_subclips"]);M=int(old["num_frames"])
    if K!=30 or M!=120:raise RuntimeError("HALT_OLD_DENSE_SCHEMA")
    if a.max_videos is not None:ids=ids[:a.max_videos]
    model_id="openai/clip-vit-large-patch14-336";model=CLIPVisionModel.from_pretrained(model_id,local_files_only=True).cuda().eval()
    proc=CLIPProcessor.from_pretrained(model_id,local_files_only=True)
    diffs=[];rows=[]
    for i,vid in enumerate(ids):
        path=a.video_root/(vid+".mp4");frames,ok=load_video_frames(str(path),M)
        if ok:
            pf=encode_frames_pooled(frames,proc,model,336,torch.device("cuda"),32)
            replay=torch.stack([pf[s:e].mean(0) for s,e in _window_bounds(M,K)])
        else:replay=torch.zeros((K,model.config.hidden_size))
        ref=old["subclip_img_feats"][i*K:(i+1)*K]
        d=float(torch.max(torch.abs(replay.cpu()-ref)).item());diffs.append(d)
        rows.append({"video_id":vid,"max_abs":d,"video_sha256":sha256_file(path)})
    payload={"schema":"cvoi-old-dense-parity/1","cache":str(a.cache.resolve()),"cache_sha256":sha256_file(a.cache),
             "producer_sha256":sha256_file(Path("src/utils/generate_subclip_embedding_HF.py")),
             "n_checked":len(ids),"full_cache":len(ids)==len(old["video_ids"]),"tolerance":a.tolerance,
             "max_abs":max(diffs),"passed":bool(max(diffs)<=a.tolerance),"rows_sha256":hashlib.sha256(canonical_bytes(rows)).hexdigest(),"rows":rows}
    atomic_json(a.out,payload)
    if not payload["passed"]:raise RuntimeError("HALT_OLD_DENSE_PARITY")

if __name__=="__main__":main()
