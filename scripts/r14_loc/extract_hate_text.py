#!/usr/bin/env python
"""R14-WVD factor B1: frozen hate-tuned text encoder over the SAME per-window ASR / OCR strings
that `scripts/r11_seg/extract_text_feats.py` fed to the CLIP text tower.

Encoder: cardiffnlp/twitter-roberta-base-hate-latest, mean-pooled last hidden state (768-d),
attention-masked. The checkpoint's `pooler` is randomly initialised and is never used.

Output: idea-stage/r14_loc/out/hate_text_feats.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
K = 30
MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"
OUT = ROOT / "idea-stage/r14_loc/out/hate_text_feats.npz"


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

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

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval().cuda()
    dim = model.config.hidden_size

    def embed(texts):
        feats = np.zeros((len(texts), dim), dtype=np.float32)
        mask = np.array([bool(t) for t in texts])
        nz = np.where(mask)[0]
        B = 128
        with torch.no_grad():
            for s in range(0, len(nz), B):
                sel = nz[s : s + B]
                enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to("cuda")
                out = model(**enc).last_hidden_state
                am = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * am).sum(1) / am.sum(1).clamp(min=1.0)
                feats[sel] = pooled.float().cpu().numpy()
        return feats, mask

    fa, ma = embed([t for v in asr_txt for t in v])
    fo, mo = embed([t for v in ocr_txt for t in v])
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
