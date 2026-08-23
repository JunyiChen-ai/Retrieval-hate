#!/usr/bin/env python
"""Build the per-split OCR-mean-30 archive caches consumed by src/run_rac.py.

Produces {ids, text_feats} .pt files in the same format as the MLLM structured
archive, so the OCR stream can be injected through the pipeline's existing
`--archive_feats ... --archive_mode stream` third-stream path with no new model
code.

Vector definition is IDENTICAL to arm 2 ("OCR-30") of the frozen-space pilot
(idea-stage/OCR_FUSION_PILOT_FREEZE.md): for each video, take the 30 K-windows,
keep OCR boxes with conf >= 0.5 and len(text) >= 2, join per window, encode each
non-empty window text with the CLIP text tower (pooler_output, 768-d),
L2-normalise each window vector, average over non-empty windows, L2-normalise
the mean. Videos with no usable OCR text get the all-zero vector.

TRAIN rows are REUSED verbatim from data/OCR/HateMM/pilot_ocr_blocks.npz['o30']
(no re-encoding). DEV_SEEN rows are encoded here, since the pilot only built the
train block.

TEST: this script writes a `test_seen` PLACEHOLDER carrying the DEV rows, valid
only under `--val_only_eval True`.
SUPERSEDED 2026-08-09: the user unsealed test *inputs* (OCR is an input feature,
no labels), the test OCR cache was built, and
scripts/ocr_cache/build_ocr_rac_cache_test.py overwrote
data/OCR/HateMM/rac_ocrmean30_test_seen.pt with REAL 215-row test features
(the old placeholder is kept as rac_ocrmean30_test_seen_PLACEHOLDER.pt).
=> Re-running this script would clobber the real file with the placeholder
again; run build_ocr_rac_cache_test.py afterwards if you do.

Usage:
  python scripts/ocr_cache/build_ocr_rac_cache.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

WHOLE = ROOT / "data/CLIP_Embedding/HateMM/{split}_openai_clip-vit-large-patch14-336_HF.pt"
OCRW = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
SHASUMS = ROOT / "data/OCR/SHA256SUMS.json"
BLOCKS = ROOT / "data/OCR/HateMM/pilot_ocr_blocks.npz"
OUT = ROOT / "data/OCR/HateMM/rac_ocrmean30_{split}.pt"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"

# ---- frozen constants, copied from OCR_FUSION_PILOT_FREEZE.md ----
K = 30
MIN_CONF = 0.5
MIN_TEXT_LEN = 2


def log(m):
    print(m, flush=True)


def l2np(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_ids(split):
    d = torch.load(str(WHOLE).format(split=split), map_location="cpu")
    raw = d["ids"]
    ids = [i for sub in raw for i in sub] if isinstance(raw[0], list) else list(raw)
    return ids


def window_texts(ids):
    want = set(ids)
    got = {}
    with open(OCRW, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r["video_id"]
            if v not in want:
                continue
            keep = []
            for d in r.get("texts") or []:
                t = (d.get("text") or "").strip()
                if float(d.get("conf", 0.0)) >= MIN_CONF and len(t) >= MIN_TEXT_LEN:
                    keep.append(t)
            got.setdefault(v, {})[int(r["window_k"])] = " ".join(keep).strip()
    missing = want - set(got)
    if missing:
        raise SystemExit("HALT_OCR_MISSING_VIDEOS:%d %s" % (
            len(missing), sorted(missing)[:5]))
    out = {}
    for v in ids:
        wk = got[v]
        if sorted(wk) != list(range(K)):
            raise SystemExit("HALT_OCR_WINDOW_COUNT:" + v)
        out[v] = [wk[k] for k in range(K)]
    return out


def encode_texts(texts, device):
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval().to(device)
    uniq = sorted({t for t in texts if t})
    log("encoding %d unique OCR window texts on %s" % (len(uniq), device))
    vecs, B = {}, 64
    with torch.no_grad():
        for i in range(0, len(uniq), B):
            batch = uniq[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc).pooler_output.float().cpu().numpy().astype(np.float64)
            for t, v in zip(batch, out):
                vecs[t] = v
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return vecs


def ocr_block(ids, wtexts, vecs):
    out = np.zeros((len(ids), 768), dtype=np.float64)
    n_empty = 0
    for i, v in enumerate(ids):
        rows = [vecs[wtexts[v][k]] for k in range(K) if wtexts[v][k]]
        if not rows:
            n_empty += 1
            continue
        m = l2np(np.stack(rows), axis=1).mean(axis=0)
        out[i] = m / max(float(np.linalg.norm(m)), 1e-8)
    return out, n_empty


def main():
    sha = sha256_file(OCRW)
    ref = json.load(open(SHASUMS))
    want = ref.get("ocr_windows_K30.jsonl") or ref.get(
        "data/OCR/HateMM/ocr_windows_K30.jsonl")
    if isinstance(want, dict):
        want = want.get("sha256")
    if want and want != sha:
        raise SystemExit("HALT_OCR_SHA_MISMATCH %s != %s" % (sha, want))
    log("ocr_windows_K30.jsonl sha256=%s (manifest %s)" % (sha, "OK" if want else "n/a"))

    tr_ids = split_ids("train")
    dv_ids = split_ids("dev_seen")
    log("train=%d dev_seen=%d" % (len(tr_ids), len(dv_ids)))

    # ---- TRAIN: reuse the pilot block verbatim (no re-encoding) ----
    npz = np.load(BLOCKS, allow_pickle=True)
    npz_ids = [str(x) for x in npz["ids"].tolist()]
    if sorted(npz_ids) != sorted(tr_ids):
        raise SystemExit("HALT_PILOT_BLOCK_ID_SET_MISMATCH")
    row = {v: i for i, v in enumerate(npz_ids)}
    tr_block = npz["o30"][[row[v] for v in tr_ids]].astype(np.float64)
    log("train block reused from %s (o30) sha=%s" % (BLOCKS.name, str(npz["sha"])))

    # ---- DEV: encode here with the identical recipe ----
    dv_wt = window_texts(dv_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vecs = encode_texts([t for v in dv_ids for t in dv_wt[v]], device)
    dv_block, dv_empty = ocr_block(dv_ids, dv_wt, vecs)
    log("dev_seen block: %d/%d videos with NO usable OCR text (all-zero row)"
        % (dv_empty, len(dv_ids)))

    # ---- self-check: re-encode 8 train videos and compare to the reused block ----
    probe = tr_ids[:8]
    pr_wt = window_texts(probe)
    pr_vecs = encode_texts([t for v in probe for t in pr_wt[v]], device)
    pr_block, _ = ocr_block(probe, pr_wt, pr_vecs)
    delta = float(np.abs(pr_block - tr_block[:8]).max())
    log("train-block reproduction check on 8 videos: max|delta| = %.3e" % delta)
    if delta > 1e-3:
        raise SystemExit("HALT_TRAIN_BLOCK_MISMATCH %.3e" % delta)

    payload = {
        "train": (tr_ids, tr_block),
        "dev_seen": (dv_ids, dv_block),
        # placeholder; guarded by --val_only_eval (see module docstring)
        "test_seen": (dv_ids, dv_block),
    }
    for split, (ids, blk) in payload.items():
        p = str(OUT).format(split=split)
        torch.save({
            "ids": list(ids),
            "text_feats": torch.as_tensor(blk, dtype=torch.float32),
            "meta": {
                "kind": "ocr_mean30_clip_text",
                "K": K, "min_conf": MIN_CONF, "min_text_len": MIN_TEXT_LEN,
                "clip_model": CLIP_MODEL,
                "ocr_windows_sha256": sha,
                "source": ("pilot_ocr_blocks.npz[o30]" if split == "train"
                           else "encoded by build_ocr_rac_cache.py"),
                "note": ("PLACEHOLDER: carries dev_seen rows. HateMM test OCR "
                         "was never encoded. Only valid under --val_only_eval "
                         "True." if split == "test_seen" else ""),
            },
        }, p)
        log("wrote %s  (%d x %d)" % (p, blk.shape[0], blk.shape[1]))


if __name__ == "__main__":
    sys.exit(main())
