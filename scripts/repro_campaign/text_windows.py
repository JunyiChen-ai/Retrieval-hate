#!/usr/bin/env python
"""REPRO campaign — freeze §8 `+text (ours)` window strings, for any native window.

Freeze §8: "The injected string for a method's native window is the
concatenation, in time order, of the ASR chunks and OCR window texts that
**overlap** that window, ASR first then OCR, separated by `" | "`.  Empty when
neither channel has text."

The caches are the frozen ones and are never rebuilt here:
  data/ASR/<DS>/*_asrK4_whisper-large-v3.jsonl
  data/OCR/<DS>/ocr_windows_K30.jsonl       (K = 30 midpoint grid, freeze §13)

`zs_clip.py` already reads both; its loaders are reused so the two methods see
byte-identical text.  The only difference is the window: ZS-CLIP scores on the
OCR cache's own K=30 grid, while a method with a free-running window (LAVAD's
10 s clip, URF's 10 s clip) asks for an arbitrary `[t0, t1)` here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/repro_campaign"))

_WS = re.compile(r"\s+")
K_OCR = 30


class TextWindows:
    def __init__(self, ds: str):
        from zs_clip import load_asr, load_ocr

        self.asr = load_asr(ds)
        self.ocr = load_ocr(ds)
        z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
        self.dur = {str(v): float(d) for v, d in zip(z["video_ids"], z["duration"])}

    def has(self, vid: str) -> bool:
        return bool(self.asr.get(vid)) or bool(self.ocr.get(vid))

    def get(self, vid: str, t0: float, t1: float) -> str:
        """ASR chunks then OCR windows overlapping `[t0, t1)`, joined by ' | '."""
        a = [t for s, e, t in sorted(self.asr.get(vid, []))
             if max(s, t0) < min(max(e, s), t1) or (s == e and t0 <= s < t1)]
        parts = []
        a_txt = _WS.sub(" ", "".join(a)).strip()
        if a_txt:
            parts.append(a_txt)
        D = max(self.dur.get(vid, 0.0), 1e-6)
        ow = self.ocr.get(vid, {})
        if ow:
            w = D / K_OCR
            k0 = max(int(np.floor(t0 / w)), 0)
            k1 = min(int(np.ceil(t1 / w)), K_OCR)
            o = [ow.get(k, "") for k in range(k0, k1)]
            o_txt = _WS.sub(" ", " ".join(x for x in o if x)).strip()
            if o_txt:
                parts.append(o_txt)
        return " | ".join(parts)
