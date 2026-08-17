#!/usr/bin/env python
"""R11-SEG: CLIP text embeddings for the K=30 ASR and OCR window text of HateClipSeg.

Uses the same frozen encoder as the visual channel
(openai/clip-vit-large-patch14-336, text tower, 768-d) so that all channels come
from one model family.  Empty window text -> zero vector (an honest "no text"),
recorded in the mask arrays.

Output: idea-stage/r11_seg/out/text_feats.npz
  video_ids (395,), asr_feat (395,30,768), ocr_feat (395,30,768),
  asr_mask (395,30) bool, ocr_mask (395,30) bool
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path("/home/jehc223/Retrieval-hate")
K = 30
MODEL = "openai/clip-vit-large-patch14-336"
OUT = ROOT / "idea-stage/r11_seg/out/text_feats.npz"


def main() -> None:
    from transformers import CLIPModel, CLIPTokenizerFast

    clipt = torch.load(
        ROOT / "data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
        map_location="cpu",
    )
    vids = list(clipt["video_ids"])
    idx = {v: i for i, v in enumerate(vids)}

    asr_txt = [["" for _ in range(K)] for _ in vids]
    with open(ROOT / "data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl") as fh:
        for ln in fh:
            d = json.loads(ln)
            if d["id"] in idx:
                asr_txt[idx[d["id"]]] = [t.strip() for t in d["window_text"]]

    ocr_txt = [["" for _ in range(K)] for _ in vids]
    with open(ROOT / "data/OCR/HateClipSeg/ocr_windows_K30.jsonl") as fh:
        for ln in fh:
            e = json.loads(ln)
            if e["video_id"] not in idx:
                continue
            toks = [t["text"] for t in e["texts"] if t.get("conf", 0) >= 0.5]
            ocr_txt[idx[e["video_id"]]][e["window_k"]] = " ".join(toks).strip()

    tok = CLIPTokenizerFast.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval().cuda()
    dim = model.config.projection_dim

    def embed(texts):
        feats = np.zeros((len(texts), dim), dtype=np.float32)
        mask = np.array([bool(t) for t in texts])
        nz = np.where(mask)[0]
        B = 256
        with torch.no_grad():
            for s in range(0, len(nz), B):
                sel = nz[s : s + B]
                enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                          max_length=77, return_tensors="pt").to("cuda")
                out = model.get_text_features(**enc)
                feats[sel] = out.float().cpu().numpy()
        return feats, mask

    flat_asr = [t for v in asr_txt for t in v]
    flat_ocr = [t for v in ocr_txt for t in v]
    fa, ma = embed(flat_asr)
    fo, mo = embed(flat_ocr)
    print(f"asr nonempty {ma.sum()}/{len(ma)}  ocr nonempty {mo.sum()}/{len(mo)}  dim={dim}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        video_ids=np.array(vids),
        asr_feat=fa.reshape(len(vids), K, dim),
        ocr_feat=fo.reshape(len(vids), K, dim),
        asr_mask=ma.reshape(len(vids), K),
        ocr_mask=mo.reshape(len(vids), K),
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
