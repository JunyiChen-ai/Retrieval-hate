#!/usr/bin/env python
"""Coverage / language statistics + sha256 for the K=30 OCR cache."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
LATIN = re.compile(r"[A-Za-z]")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(ds, min_conf=0.5, min_len=2):
    d = ROOT / f"data/OCR/{ds}"
    win = d / "ocr_windows_K30.jsonl"
    per_video = {}
    for line in open(win):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        v = r["video_id"]
        e = per_video.setdefault(v, {"win_raw": 0, "win_f": 0, "dets_raw": 0,
                                     "dets_f": 0, "chars_raw": 0, "chars_f": 0,
                                     "cjk": 0, "latin": 0, "windows": 0})
        e["windows"] += 1
        raw = [t for t in r["texts"] if t["text"].strip()]
        flt = [t for t in raw if t["conf"] >= min_conf and len(t["text"].strip()) >= min_len]
        e["dets_raw"] += len(raw)
        e["dets_f"] += len(flt)
        e["chars_raw"] += sum(len(t["text"].strip()) for t in raw)
        e["chars_f"] += sum(len(t["text"].strip()) for t in flt)
        e["win_raw"] += 1 if raw else 0
        e["win_f"] += 1 if flt else 0
        for t in flt:
            s = t["text"]
            e["cjk"] += len(CJK.findall(s))
            e["latin"] += len(LATIN.findall(s))
    V = len(per_video)
    a = lambda k: np.array([per_video[v][k] for v in per_video], dtype=float)
    chars_f, chars_raw = a("chars_f"), a("chars_raw")
    winf, winraw = a("win_f"), a("win_raw")
    cjk, latin = a("cjk").sum(), a("latin").sum()
    out = {
        "dataset": ds,
        "n_videos": V,
        "n_windows": int(a("windows").sum()),
        "filter": {"min_conf": min_conf, "min_text_len": min_len},
        "frac_videos_with_any_text_raw": round(float((chars_raw > 0).mean()), 4),
        "frac_videos_with_any_text_filtered": round(float((chars_f > 0).mean()), 4),
        "frac_videos_with_ge20_chars_filtered": round(float((chars_f >= 20).mean()), 4),
        "mean_chars_per_video_raw": round(float(chars_raw.mean()), 1),
        "mean_chars_per_video_filtered": round(float(chars_f.mean()), 1),
        "median_chars_per_video_filtered": round(float(np.median(chars_f)), 1),
        "mean_windows_with_text_per_video_raw": round(float(winraw.mean()), 2),
        "mean_windows_with_text_per_video_filtered": round(float(winf.mean()), 2),
        "frac_windows_with_text_filtered": round(float(winf.sum() / a("windows").sum()), 4),
        "mean_dets_per_video_filtered": round(float(a("dets_f").mean()), 2),
        "script_mix_filtered": {
            "latin_chars": int(latin),
            "cjk_chars": int(cjk),
            "latin_frac": round(float(latin / max(latin + cjk, 1)), 4),
            "cjk_frac": round(float(cjk / max(latin + cjk, 1)), 4),
        },
        "sha256": {},
    }
    for f in sorted(d.glob("*.jsonl")):
        out["sha256"][f.name] = sha256(f)
    for f in sorted(d.glob("meta.json")):
        out["sha256"][f.name] = sha256(f)
    return out


if __name__ == "__main__":
    all_out = {}
    for ds in (sys.argv[1:] or ["HateMM", "HateClipSeg"]):
        all_out[ds] = stats(ds)
    json.dump(all_out, open(ROOT / "data/OCR/ocr_cache_stats.json", "w"), indent=1)
    print(json.dumps(all_out, indent=1))
