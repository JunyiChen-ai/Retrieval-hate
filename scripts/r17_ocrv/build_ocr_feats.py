#!/usr/bin/env python
"""R17-OCRV: build the dense 4-FPS ON-SCREEN-TEXT channel and the two fused feature sets.

Frozen in `idea-stage/R17_OCRV_FREEZE.md` §2 (commit 1e268c6), written after it.

Source: the project's existing PaddleOCR cache
`data/OCR/HateClipSeg/ocr_windows_K30.jsonl` — one frame sampled at the midpoint of each of
K = 30 equal windows per video.  No new OCR is run in this round.

For each 4-FPS index i (t = i / 4 s) the window is k = clip(floor(t / (duration/30)), 0, 29).
Its texts with confidence >= 0.5 are lower-cased, joined in reading order and encoded once per
unique string with the frozen `bert-base-uncased` CLS vector -- the same encoder and the same
read-out the ASR channel `dense4fps_bertbase` already uses.  Empty windows are the zero vector,
matching that channel's convention for empty ASR.

Outputs:
  dense4fps_ocrbert/<vid>.npy        (T, 768)
  dense4fps_vato/<vid>.npy           (T, 3584)  = V | A | T | O
  dense4fps_vato_shuf/<vid>.npy      (T, 3584)  = V | A | T | O[perm], perm drawn once per
                                                  video from seed 6280 and applied to the O
                                                  rows only (width, per-video OCR content and
                                                  marginal statistics preserved; timing destroyed)
"""
from __future__ import annotations

import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
EMB = ROOT / "data/CLIP_Embedding/HateClipSeg"
VIS_DIR = EMB / "dense4fps_clipL336"
AUD_DIR = EMB / "dense4fps_w2vemo"
TXT_DIR = EMB / "dense4fps_bertbase"
OCR_DIR = EMB / "dense4fps_ocrbert"
VATO_DIR = EMB / "dense4fps_vato"
SHUF_DIR = EMB / "dense4fps_vato_shuf"
OCR_JSONL = ROOT / "data/OCR/HateClipSeg/ocr_windows_K30.jsonl"
BERT = "bert-base-uncased"
FPS = 4.0
K = 30
CONF = 0.5
SHUF_SEED = 6280


def main() -> None:
    for d in (OCR_DIR, VATO_DIR, SHUF_DIR):
        d.mkdir(parents=True, exist_ok=True)

    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    win_text: dict[str, dict[int, str]] = {}
    with open(OCR_JSONL) as fh:
        for ln in fh:
            d = json.loads(ln)
            t = " ".join(x["text"].strip() for x in d["texts"]
                         if float(x.get("conf", 0.0)) >= CONF and x["text"].strip())
            win_text.setdefault(d["video_id"], {})[int(d["window_k"])] = t.lower().strip()

    # ---- encode every unique OCR string once
    uniq = sorted({s for wd in win_text.values() for s in wd.values() if s})
    print(f"[ocr] videos with cache {len(win_text)}  unique non-empty strings {len(uniq)}",
          flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BERT)
    tm = AutoModel.from_pretrained(BERT, torch_dtype=torch.float16).to("cuda").eval()
    cache: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i in range(0, len(uniq), 64):
        bt = uniq[i:i + 64]
        enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to("cuda")
        with torch.no_grad():
            o = tm(**enc).last_hidden_state[:, 0].float().cpu().numpy()
        for j, s in enumerate(bt):
            cache[s] = o[j]
    print(f"[ocr] encoded in {time.time()-t0:.0f}s", flush=True)
    del tm
    torch.cuda.empty_cache()

    n_empty_vid = 0
    for n, vid in enumerate(sorted(gold)):
        vpath = VIS_DIR / f"{vid}.npy"
        if not vpath.exists():
            print(f"[SKIP-novis] {vid}", flush=True)
            continue
        V = np.load(vpath).astype(np.float32)
        A = np.load(AUD_DIR / f"{vid}.npy").astype(np.float32)
        T_ = np.load(TXT_DIR / f"{vid}.npy").astype(np.float32)
        T = V.shape[0]
        assert A.shape[0] == T and T_.shape[0] == T, (vid, V.shape, A.shape, T_.shape)

        dur = float(gold[vid]["duration"])
        W = max(dur / K, 1e-6)
        wd = win_text.get(vid, {})
        O = np.zeros((T, 768), dtype=np.float32)
        idx = np.clip((np.arange(T) / FPS / W).astype(int), 0, K - 1)
        if not any(wd.get(k) for k in range(K)):
            n_empty_vid += 1
        else:
            for k in range(K):
                s = wd.get(k, "")
                if s:
                    O[idx == k] = cache[s]

        np.save(OCR_DIR / f".{vid}.tmp.npy", O)
        os.replace(OCR_DIR / f".{vid}.tmp.npy", OCR_DIR / f"{vid}.npy")

        VATO = np.concatenate([V, A, T_, O], axis=1)
        np.save(VATO_DIR / f".{vid}.tmp.npy", VATO)
        os.replace(VATO_DIR / f".{vid}.tmp.npy", VATO_DIR / f"{vid}.npy")

        rng = np.random.default_rng(SHUF_SEED + zlib.crc32(vid.encode()))
        VATOS = np.concatenate([V, A, T_, O[rng.permutation(T)]], axis=1)
        np.save(SHUF_DIR / f".{vid}.tmp.npy", VATOS)
        os.replace(SHUF_DIR / f".{vid}.tmp.npy", SHUF_DIR / f"{vid}.npy")

        if (n + 1) % 50 == 0:
            print(f"[{n+1}/{len(gold)}] {vid} T={T} dim={VATO.shape[1]} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    print(f"[done] videos with no OCR text at all: {n_empty_vid}  wall {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
