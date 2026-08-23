#!/usr/bin/env python
"""Build the REAL HateMM test-split OCR-mean-30 archive cache + window-vec cache.

Background: data/OCR/HateMM/rac_ocrmean30_test_seen.pt was written by
build_ocr_rac_cache.py as a PLACEHOLDER carrying the 107 dev_seen rows, because
HateMM test OCR had never been extracted. The user unsealed test *inputs* on
2026-08-09 (OCR is an input feature, no labels involved), the test OCR cache was
built with the identical extractor, and this script replaces the placeholder
with genuine test features.

Recipe: imported verbatim from build_ocr_rac_cache.py (K=30, conf>=0.5,
len(text)>=2, CLIP text tower pooler_output 768-d, per-window L2, mean over
non-empty windows, L2 again; videos with no usable OCR text -> all-zero row).
No constant is redefined here.

Outputs (data/OCR/HateMM/):
  test_ocr_window_vecs.npz        {vecs, texts, sha} -- same layout as
                                  pilot_ocr_window_vecs.npz, for the test split
  rac_ocrmean30_test_seen.pt      real 215-row archive cache (placeholder is
                                  saved aside as rac_ocrmean30_test_seen_PLACEHOLDER.pt)

Usage:
  python scripts/ocr_cache/build_ocr_rac_cache_test.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ocr_rac_cache as B  # frozen recipe: constants + functions

ROOT = Path(__file__).resolve().parents[2]
OCRW_TEST = ROOT / "data/OCR/HateMM/ocr_windows_K30_test.jsonl"
PILOT_VECS = ROOT / "data/OCR/HateMM/pilot_ocr_window_vecs.npz"
OUT_VECS = ROOT / "data/OCR/HateMM/test_ocr_window_vecs.npz"
OUT_PT = ROOT / "data/OCR/HateMM/rac_ocrmean30_test_seen.pt"
PLACEHOLDER_BAK = ROOT / "data/OCR/HateMM/rac_ocrmean30_test_seen_PLACEHOLDER.pt"

log = B.log


def main():
    sha = B.sha256_file(OCRW_TEST)
    log("ocr_windows_K30_test.jsonl sha256=%s" % sha)

    # ---- id order comes from the CLIP whole-video archive for test_seen ----
    ids = B.split_ids("test_seen")
    log("test_seen ids: %d" % len(ids))

    # ---- window texts, read through the frozen loader (retargeted file) ----
    B.OCRW = OCRW_TEST
    wt = B.window_texts(ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vecs = B.encode_texts([t for v in ids for t in wt[v]], device)

    # ---- recipe check: our encoder must reproduce the pilot's cached vectors ----
    if PILOT_VECS.exists():
        z = np.load(PILOT_VECS, allow_pickle=True)
        p_texts = [str(t) for t in z["texts"].tolist()]
        p_vecs = z["vecs"]
        idx = [i for i, t in enumerate(p_texts) if t in vecs][:16]
        if idx:
            d = max(float(np.abs(vecs[p_texts[i]] - p_vecs[i]).max()) for i in idx)
            log("pilot-encoder reproduction check on %d shared texts: max|delta|=%.3e"
                % (len(idx), d))
            if d > 1e-3:
                raise SystemExit("HALT_ENCODER_MISMATCH %.3e" % d)
        else:
            log("pilot-encoder check skipped: no shared window texts")

    blk, n_empty = B.ocr_block(ids, wt, vecs)
    log("test_seen block: %d/%d videos with NO usable OCR text (all-zero row)"
        % (n_empty, len(ids)))

    # ---- window-vec cache, mirroring pilot_ocr_window_vecs.npz ----
    uniq = sorted(vecs)
    np.savez(OUT_VECS,
             vecs=np.stack([vecs[t] for t in uniq]) if uniq else np.zeros((0, 768)),
             texts=np.array(uniq, dtype=object),
             sha=np.array(sha))
    log("wrote %s  (%d unique window texts)" % (OUT_VECS, len(uniq)))

    # ---- replace the placeholder archive cache ----
    if OUT_PT.exists() and not PLACEHOLDER_BAK.exists():
        old = torch.load(OUT_PT, map_location="cpu")
        if "PLACEHOLDER" in str((old.get("meta") or {}).get("note", "")):
            shutil.copy2(OUT_PT, PLACEHOLDER_BAK)
            log("saved old placeholder aside -> %s" % PLACEHOLDER_BAK.name)

    torch.save({
        "ids": list(ids),
        "text_feats": torch.as_tensor(blk, dtype=torch.float32),
        "meta": {
            "kind": "ocr_mean30_clip_text",
            "K": B.K, "min_conf": B.MIN_CONF, "min_text_len": B.MIN_TEXT_LEN,
            "clip_model": B.CLIP_MODEL,
            "ocr_windows_sha256": sha,
            "ocr_windows_file": "data/OCR/HateMM/ocr_windows_K30_test.jsonl",
            "source": "encoded by build_ocr_rac_cache_test.py",
            "note": ("REAL test-split OCR features (215 rows). Replaces the "
                     "dev-row placeholder written 2026-08-09 by "
                     "build_ocr_rac_cache.py; that file is kept as "
                     "rac_ocrmean30_test_seen_PLACEHOLDER.pt. Inputs only -- "
                     "no test label was read at any stage."),
        },
    }, OUT_PT)
    log("wrote %s  (%d x %d)" % (OUT_PT, blk.shape[0], blk.shape[1]))
    log(json.dumps({"n_ids": len(ids), "n_all_zero_rows": n_empty,
                    "n_unique_window_texts": len(uniq)}))


if __name__ == "__main__":
    sys.exit(main())
