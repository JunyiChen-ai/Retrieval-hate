#!/usr/bin/env python
"""P-C — on-screen-text provenance separability (frozen: idea-stage/PILOT_FREEZE_2026-08-09.md).

Four arms, HateMM-train only (744 videos), CPU. dev_seen / test are never opened.
Decision rule is frozen in the freeze document and is NOT edited after results.

  arm 0   baseline  [l2(img) || l2(txt)]                              1792-d
  arm 1   untyped   baseline || ocr30                                 2560-d   (reproduces +0.0094)
  arm 1c  control   baseline || ocr30 || ocr30 (duplicated)           3328-d
  arm 2   typed     baseline || overlay_mean || scene_mean            3328-d

Gating quantity = seed-mean (arm2 - arm1c).

Everything about folds / head / optimiser / epoch+threshold selection / seeds / OCR filter /
aggregation is copied verbatim from scripts/ocr_cache/ocr_fusion_pilot.py so the numbers are
comparable with idea-stage/OCR_FUSION_PILOT_RESULT.md.

Usage:
  python idea-stage/pilot_c_ocr_provenance.py --smoke synthetic
  python idea-stage/pilot_c_ocr_provenance.py --smoke permuted --out /tmp/null.json
  python idea-stage/pilot_c_ocr_provenance.py --out idea-stage/pilot_c.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tera_gate0.common import select_threshold  # noqa: E402

RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
OCRW = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
SHASUMS = ROOT / "data/OCR/SHA256SUMS.json"
BLOCKS = ROOT / "data/OCR/HateMM/pilot_ocr_blocks.npz"          # untyped o30 (prior pilot)
WVECS = ROOT / "data/OCR/HateMM/pilot_ocr_window_vecs.npz"      # text -> 768-d CLIP text vec
TYPED_CACHE = ROOT / "data/OCR/HateMM/pilot_c_typed_blocks.npz"
DIMS_CACHE = ROOT / "data/OCR/HateMM/frame_dims_train.json"
VIDEO_DIR = ROOT / "data/video/HateMM/All"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"

# ---- frozen constants inherited from OCR_FUSION_PILOT_FREEZE.md ----
K = 30
MIN_CONF = 0.5
MIN_TEXT_LEN = 2
ARM2_WINDOWS = tuple(range(K))
SEEDS = (20260810, 20260811, 20260812)
INNER_FOLD_SEED = 20260808
N_INNER = 4
LR = 1e-3
WD = 1e-2
BATCH_SIZE = 64
E_MAX = 200
PATIENCE = 40
MIN_DELTA = 1e-4
TORCH_THREADS = 8

# ---- frozen constants of the P-C typing rule (PILOT_FREEZE_2026-08-09.md, section P-C) ----
CENTRE_TOL = 0.05        # normalised box-centre proximity, both axes
JACCARD_TOL = 0.6        # token Jaccard for track continuation
PERSIST_FRAC = 0.50      # >= 50% of the windows that contain any text
STD_TOL = 0.05           # box-centre standard deviation, both axes

# ---- frozen decision rule (P-C, primary = seed-mean arm2 - arm1c) ----
GO_T = 0.010
AMBIG_T = 0.003

# ---- RE-AUDIT read-out constants (idea-stage/REAUDIT_FREEZE.md) ----
REAUDIT_BAR = 0.005
N_BOOT = 20000
BOOT_SEED = 20260817

# ---- null control, declared BEFORE the permuted run was executed ----
NULL_F1_RANGE = (0.40, 0.60)
NULL_MAX_ABS_DELTA = 0.010

ARMS = ("0", "1", "1c", "1r", "2")
# Seed-scope tag per arm. Arms "0" and "1" reuse the prior pilot's tags (int 0 and int 2) so the
# head init / batch-shuffle streams are bit-identical to ocr_fusion_pilot.py arm0 and arm2 and the
# reproduction check is exact. The two new arms get distinct tags.
ARM_SEED_TAG = {"0": 0, "1": 2, "1c": "1c", "1r": "1r", "2": "2t"}
# RE-AUDIT addition: RAND_SEED fixes the one Gaussian projection used to build arm "1r",
# the dimension-matched content-free control for arm 1 (R6-1C RANDCAT analogue).
RAND_SEED = 20260817


class Halt(RuntimeError):
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------- guards --
FORBIDDEN_PATH = ("dev_seen", "test")
FORBIDDEN_ID = ("test", "dev")


def guard_path(p):
    s = str(p).lower()
    for bad in FORBIDDEN_PATH:
        if bad in s:
            raise Halt("HALT_FORBIDDEN_PATH:%s" % p)
    return p


def guard_ids(ids, where):
    for v in ids:
        lv = str(v).lower()
        for bad in FORBIDDEN_ID:
            if bad in lv:
                raise Halt("HALT_FORBIDDEN_ID:%s:%s" % (where, v))
    return ids


def l2np(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(guard_path(p), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_f1(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    fs = []
    for c in (0, 1):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


# --------------------------------------------------------------------- data --
def load_base():
    who = torch.load(guard_path(WHOLE), map_location="cpu")
    raw = who["ids"]
    ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    ids = guard_ids(list(ids), "whole_cache")
    img = who["img_feats"].numpy().astype(np.float64)
    txt = who["text_feats"].numpy().astype(np.float64)
    y = who["labels"].numpy().astype(np.int64)
    if not (len(ids) == img.shape[0] == txt.shape[0] == y.shape[0]):
        raise Halt("HALT_CACHE_SHAPE")
    idx = {v: i for i, v in enumerate(ids)}
    folds = []
    for f in range(5):
        tr = json.load(open(guard_path(RUN / ("folds/fold_%d/train_ids.json" % f))))
        qu = json.load(open(guard_path(RUN / ("folds/fold_%d/query_ids.json" % f))))
        guard_ids(tr + qu, "folds")
        for v in tr + qu:
            if v not in idx:
                raise Halt("HALT_FOLD_ID_NOT_IN_CACHE:" + v)
        folds.append((sorted(tr), sorted(qu)))
    covered = sorted({v for _, qu in folds for v in qu})
    if covered != sorted(ids):
        raise Halt("HALT_FOLD_COVERAGE")
    return ids, idx, img, txt, y, folds


def read_detections(ids):
    """{video_id: [30 lists of kept detections]} with the frozen conf>=0.5, len>=2 filter.

    A kept detection is {"text": str, "conf": float, "bbox": [[x,y] x4]} in file order.
    """
    want = set(ids)
    got = {}
    with open(guard_path(OCRW), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r["video_id"]
            if v not in want:          # val rows discarded at load time
                continue
            k = int(r["window_k"])
            keep = []
            for d in r.get("texts") or []:
                t = (d.get("text") or "").strip()
                if float(d.get("conf", 0.0)) >= MIN_CONF and len(t) >= MIN_TEXT_LEN:
                    keep.append({"text": t, "conf": float(d["conf"]), "bbox": d["bbox"]})
            got.setdefault(v, {})[k] = keep
    if set(got) != want:
        raise Halt("HALT_OCR_MISSING_VIDEOS:%d" % len(want - set(got)))
    out = {}
    for v, wk in got.items():
        if sorted(wk) != list(range(K)):
            raise Halt("HALT_OCR_WINDOW_COUNT:" + v)
        out[v] = [wk[k] for k in range(K)]
    return out


def frame_dims(ids, dets):
    """(W, H) per video. Primary = ffprobe on the source file; fallback = per-video max bbox
    extent over ALL detections (documented deviation). Returns (dims, source_counts)."""
    if DIMS_CACHE.exists():
        cached = json.load(open(guard_path(DIMS_CACHE)))
    else:
        cached = {}
    src = {}
    dims = {}
    changed = False
    for v in ids:
        if v in cached:
            w, h, s = cached[v]
        else:
            w = h = None
            s = "ffprobe"
            path = None
            for ext in (".mp4", ".mkv", ".webm", ".avi"):
                p = VIDEO_DIR / (v + ext)
                if p.exists():
                    path = guard_path(p)
                    break
            if path is not None:
                try:
                    out = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height",
                         "-of", "csv=p=0:s=x", str(path)],
                        capture_output=True, text=True, timeout=60).stdout.strip()
                    a, b = out.splitlines()[0].split("x")
                    w, h = int(a), int(b)
                except Exception:
                    w = h = None
            if not w or not h:
                s = "bbox_max_extent"
                mx = my = 0.0
                for k in range(K):
                    for d in dets[v][k]:
                        for (x, yy) in d["bbox"]:
                            mx = max(mx, float(x))
                            my = max(my, float(yy))
                w, h = max(mx, 1.0), max(my, 1.0)
            cached[v] = [w, h, s]
            changed = True
        dims[v] = (float(w), float(h))
        src[v] = s
    if changed:
        json.dump(cached, open(DIMS_CACHE, "w"))
    counts = {}
    for v in ids:
        counts[src[v]] = counts.get(src[v], 0) + 1
    return dims, counts


# ------------------------------------------------------- frozen typing rule --
_TOK = re.compile(r"\w+", re.UNICODE)


def tokens(t):
    return set(_TOK.findall(t.lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def centre(bbox, w, h):
    pts = np.asarray(bbox, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    return float(c[0] / w), float(c[1] / h)


def type_video(dets_v, w, h):
    """Apply the frozen provenance typing rule to one video.

    Tracks: greedy, deterministic, windows in ascending k, detections in file order. A detection
    joins the existing track whose LAST member satisfies |dcx|<=0.05, |dcy|<=0.05 and token
    Jaccard >= 0.6 and which has no member in the current window; among candidates the smallest
    Chebyshev centre distance wins, ties broken by lowest track index. Otherwise it opens a track.

    A track is overlay-like iff it appears in >= 50% of the windows that contain any kept text and
    its member centre std (ddof=0) is <= 0.05 on both axes. Every other detection is scene-like.

    Returns (overlay_texts[30], scene_texts[30], stats).
    """
    n_text_windows = sum(1 for k in range(K) if dets_v[k])
    tracks = []          # each: dict(cx=[], cy=[], last_tok=set, last_k=int, members=[(k, i)])
    for k in range(K):
        for i, d in enumerate(dets_v[k]):
            cx, cy = centre(d["bbox"], w, h)
            tk = tokens(d["text"])
            best = None
            for ti, t in enumerate(tracks):
                if t["last_k"] == k:
                    continue
                dx = abs(cx - t["cx"][-1])
                dy = abs(cy - t["cy"][-1])
                if dx > CENTRE_TOL or dy > CENTRE_TOL:
                    continue
                if jaccard(tk, t["last_tok"]) < JACCARD_TOL:
                    continue
                cand = (max(dx, dy), ti)
                if best is None or cand < best:
                    best = cand
            if best is None:
                tracks.append({"cx": [cx], "cy": [cy], "last_tok": tk,
                               "last_k": k, "members": [(k, i)]})
            else:
                t = tracks[best[1]]
                t["cx"].append(cx)
                t["cy"].append(cy)
                t["last_tok"] = tk
                t["last_k"] = k
                t["members"].append((k, i))

    overlay_flag = {}    # (k, i) -> True/False
    n_overlay_tracks = 0
    for t in tracks:
        persists = (n_text_windows > 0
                    and len(t["members"]) >= PERSIST_FRAC * n_text_windows)
        sx = float(np.std(np.asarray(t["cx"]), ddof=0))
        sy = float(np.std(np.asarray(t["cy"]), ddof=0))
        ov = bool(persists and sx <= STD_TOL and sy <= STD_TOL)
        n_overlay_tracks += int(ov)
        for m in t["members"]:
            overlay_flag[m] = ov

    ov_texts, sc_texts = [], []
    n_ov_det = n_sc_det = 0
    ov_chars = sc_chars = 0
    for k in range(K):
        o, s = [], []
        for i, d in enumerate(dets_v[k]):
            if overlay_flag[(k, i)]:
                o.append(d["text"])
                n_ov_det += 1
                ov_chars += len(d["text"])
            else:
                s.append(d["text"])
                n_sc_det += 1
                sc_chars += len(d["text"])
        ov_texts.append(" ".join(o).strip())
        sc_texts.append(" ".join(s).strip())
    stats = {"n_text_windows": n_text_windows, "n_tracks": len(tracks),
             "n_overlay_tracks": n_overlay_tracks,
             "n_overlay_det": n_ov_det, "n_scene_det": n_sc_det,
             "overlay_chars": ov_chars, "scene_chars": sc_chars,
             "n_overlay_windows": sum(1 for t in ov_texts if t),
             "n_scene_windows": sum(1 for t in sc_texts if t)}
    return ov_texts, sc_texts, stats


# ---------------------------------------------------------------- encoding --
def load_vec_cache(sha):
    if not WVECS.exists():
        return {}
    z = np.load(guard_path(WVECS), allow_pickle=True)
    if str(z["sha"]) != sha:
        log("WARN window-vec cache sha mismatch; ignoring cache")
        return {}
    return {str(t): v for t, v in zip(z["texts"], z["vecs"].astype(np.float64))}


def encode_missing(need, have):
    todo = sorted(t for t in need if t and t not in have)
    if not todo:
        return have, 0
    from transformers import CLIPTextModel, CLIPTokenizer
    log("encoding %d NEW unique typed window texts on cpu" % len(todo))
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval()
    B = 64
    with torch.no_grad():
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            out = mdl(**enc).pooler_output.float().numpy().astype(np.float64)
            for t, v in zip(batch, out):
                have[t] = v
            if (i // B) % 10 == 0:
                log("PROGRESS encode %d/%d" % (i, len(todo)))
    del mdl
    return have, len(todo)


def mean_block(per_video_texts, ids, vecs):
    """Frozen aggregation: L2-normalise each non-empty window embedding, unweighted mean over the
    non-empty windows in {0..29}, L2-normalise. No text anywhere -> all-zero 768-d."""
    out = np.zeros((len(ids), 768), dtype=np.float64)
    n_zero = 0
    for i, v in enumerate(ids):
        rows = [vecs[per_video_texts[v][k]] for k in ARM2_WINDOWS if per_video_texts[v][k]]
        if not rows:
            n_zero += 1
            continue
        m = l2np(np.stack(rows), axis=1).mean(axis=0)
        out[i] = m / max(float(np.linalg.norm(m)), 1e-8)
    return out, n_zero


# ------------------------------------------------------------------- model --
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)
        nn.init.normal_(self.head.weight, 0.0, 0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(x).squeeze(-1)


def derive_seed(*parts):
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def train_epochs(model, X, y, rows, opt, lossfn, scope, e_from, e_to):
    for epoch in range(e_from, e_to):
        gen = torch.Generator()
        gen.manual_seed((derive_seed(scope) + epoch) % (2 ** 31 - 1))
        perm = torch.randperm(len(rows), generator=gen)
        sh = rows[perm]
        model.train()
        for s in range(0, len(sh), BATCH_SIZE):
            b = sh[s:s + BATCH_SIZE]
            opt.zero_grad()
            loss = lossfn(model(X[b]), y[b])
            loss.backward()
            opt.step()


@torch.no_grad()
def score(model, X, rows):
    model.eval()
    return torch.sigmoid(model(X[rows])).double().numpy()


def run_fold(Xnp, ynp, tr_ids, qu_ids, tag, seed, outer, idx):
    X = torch.as_tensor(Xnp, dtype=torch.float32)
    yt = torch.as_tensor(ynp, dtype=torch.float32)
    d = X.shape[1]
    lossfn = nn.BCEWithLogitsLoss()

    order = sorted(tr_ids)
    yo = np.array([ynp[idx[v]] for v in order], dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=INNER_FOLD_SEED)
    arr = np.array(order)
    inner = [(sorted(arr[a].tolist()), sorted(arr[b].tolist()))
             for a, b in skf.split(np.zeros(len(order)), yo)]

    models, opts, itr, iva = [], [], [], []
    for j, (a, b) in enumerate(inner):
        torch.manual_seed(derive_seed(seed, tag, outer, j) % (2 ** 31 - 1))
        m = Head(d)
        models.append(m)
        opts.append(torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD,
                                      betas=(0.9, 0.999), eps=1e-8, amsgrad=False))
        itr.append(torch.as_tensor([idx[v] for v in a], dtype=torch.long))
        iva.append(torch.as_tensor([idx[v] for v in b], dtype=torch.long))

    val_rows = torch.cat(iva)
    val_y = ynp[val_rows.numpy()]
    best_f1, best_epoch, best_theta, since = -1.0, 1, 0.5, 0
    for epoch in range(E_MAX):
        for j in range(N_INNER):
            train_epochs(models[j], X, yt, itr[j], opts[j], lossfn,
                         (seed, tag, outer, j), epoch, epoch + 1)
        pooled = np.concatenate([score(models[j], X, iva[j]) for j in range(N_INNER)])
        theta, f1 = select_threshold(pooled, val_y)
        if f1 > best_f1 + MIN_DELTA:
            best_f1, best_epoch, best_theta, since = float(f1), epoch + 1, float(theta), 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    torch.manual_seed(derive_seed(seed, tag, outer, "refit") % (2 ** 31 - 1))
    m = Head(d)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD,
                            betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
    rows = torch.as_tensor([idx[v] for v in order], dtype=torch.long)
    train_epochs(m, X, yt, rows, opt, lossfn, (seed, tag, outer, "refit"), 0, best_epoch)
    qrows = torch.as_tensor([idx[v] for v in sorted(qu_ids)], dtype=torch.long)
    s = score(m, X, qrows)
    return qrows.numpy(), (s >= best_theta).astype(np.int64), best_epoch, best_theta, best_f1


def run_arm(Xnp, ynp, ids, folds, arm, seed, idx):
    tag = ARM_SEED_TAG[arm]
    pred = np.full(len(ids), -1, dtype=np.int64)
    info = []
    for outer, (tr_ids, qu_ids) in enumerate(folds):
        t0 = time.time()
        qr, p, ep, th, inf1 = run_fold(Xnp, ynp, tr_ids, qu_ids, tag, seed, outer, idx)
        pred[qr] = p
        info.append({"outer": outer, "epoch": ep, "theta": th, "inner_macro_f1": inf1,
                     "seconds": round(time.time() - t0, 1)})
        log("PROGRESS arm=%s seed=%d fold=%d epoch=%d theta=%.4f dt=%.1fs"
            % (arm, seed, outer, ep, th, time.time() - t0))
    if (pred < 0).any():
        raise Halt("HALT_INCOMPLETE_OOF")
    return macro_f1(ynp, pred), info


# ------------------------------------------------------------------ features --
def rand_block(base, d_out):
    """Dimension-matched, content-free control block: l2norm(base @ R), R fixed Gaussian.

    RE-AUDIT addition (not part of the 2026-08-09 P-C design).  Same construction as the
    RANDCAT / RANDA-RANDB control of idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md: it carries
    exactly the added dimensionality of arm 1 and none of the OCR content, so arm1 - arm1r
    isolates on-screen-text content from head capacity.  R is drawn once from
    numpy.random.default_rng(RAND_SEED) and depends on no label.
    """
    rng = np.random.default_rng(RAND_SEED)
    R = rng.normal(0.0, 1.0 / np.sqrt(base.shape[1]), size=(base.shape[1], d_out))
    return l2np(base @ R)


def build_features(smoke):
    if smoke == "synthetic":
        rng = np.random.default_rng(0)
        n = 200
        ids = ["v%03d" % i for i in range(n)]
        y = (rng.random(n) < 0.4).astype(np.int64)
        img = rng.normal(size=(n, 1024)) + 0.3 * y[:, None]
        txt = rng.normal(size=(n, 768)) + 0.3 * y[:, None]
        o30 = rng.normal(size=(n, 768)) + 0.5 * y[:, None]
        ov = rng.normal(size=(n, 768)) + 0.6 * y[:, None]
        sc = rng.normal(size=(n, 768))
        idx = {v: i for i, v in enumerate(ids)}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        arr = np.array(ids)
        folds = [(sorted(arr[a].tolist()), sorted(arr[b].tolist()))
                 for a, b in skf.split(np.zeros(n), y)]
        base = np.hstack([l2np(img), l2np(txt)])
        X = {"0": base,
             "1": np.hstack([base, l2np(o30)]),
             "1c": np.hstack([base, l2np(o30), l2np(o30)]),
             "1r": np.hstack([base, rand_block(base, 768)]),
             "2": np.hstack([base, l2np(ov), l2np(sc)])}
        return ids, idx, y, folds, X, {"mode": "synthetic"}, {}

    ids, idx, img, txt, y, folds = load_base()
    have = json.load(open(guard_path(SHASUMS)))["data/OCR/HateMM/ocr_windows_K30.jsonl"]
    got = sha256_file(OCRW)
    if got != have:
        raise Halt("HALT_OCR_CACHE_SHA:%s" % got)
    log("OCR cache sha256 verified: %s" % got)

    # untyped OCR-30 block, byte-identical to the prior pilot
    if not BLOCKS.exists():
        raise Halt("HALT_MISSING_UNTYPED_BLOCKS")
    z = np.load(guard_path(BLOCKS), allow_pickle=True)
    if str(z["sha"]) != got or list(z["ids"]) != list(ids):
        raise Halt("HALT_OCR_BLOCK_CACHE_STALE")
    o30 = z["o30"].astype(np.float64)
    e30 = int(z["e30"])
    log("reused untyped OCR-30 block from %s (all-zero videos=%d/%d)" % (BLOCKS.name, e30, len(ids)))

    # typed blocks
    if TYPED_CACHE.exists():
        tz = np.load(guard_path(TYPED_CACHE), allow_pickle=True)
        if str(tz["sha"]) == got and list(tz["ids"]) == list(ids):
            ov, sc = tz["ov"].astype(np.float64), tz["sc"].astype(np.float64)
            typing = json.loads(str(tz["typing"]))
            log("reused typed block cache %s" % TYPED_CACHE.name)
        else:
            raise Halt("HALT_TYPED_CACHE_STALE")
    else:
        dets = read_detections(ids)
        dims, dim_src = frame_dims(ids, dets)
        log("frame dims: %s" % json.dumps(dim_src))
        ovt, sct, per_video = {}, {}, {}
        for v in ids:
            w, h = dims[v]
            a, b, st = type_video(dets[v], w, h)
            ovt[v], sct[v], per_video[v] = a, b, st
        need = set()
        for v in ids:
            need.update(t for t in ovt[v] if t)
            need.update(t for t in sct[v] if t)
        vecs = load_vec_cache(got)
        n_cached = len(vecs)
        vecs, n_new = encode_missing(need, vecs)
        ov, z_ov = mean_block(ovt, ids, vecs)
        sc, z_sc = mean_block(sct, ids, vecs)
        typing = {"per_video": per_video, "frame_dim_source": dim_src,
                  "n_unique_typed_window_texts": len(need),
                  "n_reused_from_cache": len(need) - n_new,
                  "n_newly_encoded": n_new, "vec_cache_size": n_cached,
                  "zero_overlay_block_videos": z_ov, "zero_scene_block_videos": z_sc}
        np.savez(TYPED_CACHE, ov=ov, sc=sc, sha=got,
                 ids=np.array(ids, dtype=object), typing=json.dumps(typing))
        log("typed blocks built (new texts encoded=%d)" % n_new)

    o1 = descriptive_o1(ids, typing["per_video"])
    o2 = descriptive_o2(ids, typing["per_video"], y)

    if smoke == "permuted":
        rng = np.random.default_rng(12345)
        y = y[rng.permutation(len(y))]
        log("SMOKE: labels permuted (null control)")

    base = np.hstack([l2np(img), l2np(txt)])
    X = {"0": base,
         "1": np.hstack([base, o30]),
         "1c": np.hstack([base, o30, o30]),
         "1r": np.hstack([base, rand_block(base, o30.shape[1])]),
         "2": np.hstack([base, ov, sc])}
    meta = {"mode": "real" if smoke is None else "permuted",
            "n_videos": len(ids), "n_pos": int(y.sum()),
            "ocr_windows_sha256": got,
            "untyped_zero_ocr_videos": e30,
            "typing_summary": {k: v for k, v in typing.items() if k != "per_video"},
            "dims": {a: int(X[a].shape[1]) for a in ARMS}}
    return ids, idx, y, folds, X, meta, {"O1": o1, "O2": o2}


def descriptive_o1(ids, per_video):
    n_ov = sum(1 for v in ids if per_video[v]["n_overlay_det"] > 0)
    n_sc = sum(1 for v in ids if per_video[v]["n_scene_det"] > 0)
    n_both = sum(1 for v in ids
                 if per_video[v]["n_overlay_det"] > 0 and per_video[v]["n_scene_det"] > 0)
    n_none = sum(1 for v in ids
                 if per_video[v]["n_overlay_det"] == 0 and per_video[v]["n_scene_det"] == 0)
    tot_ov = sum(per_video[v]["n_overlay_det"] for v in ids)
    tot_sc = sum(per_video[v]["n_scene_det"] for v in ids)
    ch_ov = sum(per_video[v]["overlay_chars"] for v in ids)
    ch_sc = sum(per_video[v]["scene_chars"] for v in ids)
    return {"n_videos": len(ids),
            "videos_with_overlay": n_ov, "videos_with_scene": n_sc,
            "videos_with_both": n_both, "videos_with_neither": n_none,
            "videos_overlay_only": n_ov - n_both, "videos_scene_only": n_sc - n_both,
            "detections_overlay": tot_ov, "detections_scene": tot_sc,
            "det_share_overlay": tot_ov / max(tot_ov + tot_sc, 1),
            "chars_overlay": ch_ov, "chars_scene": ch_sc,
            "char_share_overlay": ch_ov / max(ch_ov + ch_sc, 1),
            "tracks_total": sum(per_video[v]["n_tracks"] for v in ids),
            "tracks_overlay": sum(per_video[v]["n_overlay_tracks"] for v in ids)}


def descriptive_o2(ids, per_video, y):
    ov = np.array([1.0 if per_video[v]["n_overlay_det"] > 0 else 0.0 for v in ids])
    sc = np.array([1.0 if per_video[v]["n_scene_det"] > 0 else 0.0 for v in ids])
    any_ = np.array([1.0 if (per_video[v]["n_overlay_det"] + per_video[v]["n_scene_det"]) > 0
                     else 0.0 for v in ids])
    out = {}
    for name, x in (("overlay_presence", ov), ("scene_presence", sc), ("any_text_presence", any_)):
        out[name] = (float(roc_auc_score(y, x)) if len(np.unique(x)) > 1 else None)
        out[name + "_rate"] = float(x.mean())
    return out


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args()
    torch.set_num_threads(TORCH_THREADS)
    log("GUARD ARMED: paths containing %s and ids containing %s HALT the run"
        % (list(FORBIDDEN_PATH), list(FORBIDDEN_ID)))

    ids, idx, y, folds, X, meta, desc = build_features(a.smoke)
    log("features ready: %s" % json.dumps(meta.get("dims", {})))
    if desc:
        log("O1 %s" % json.dumps(desc["O1"]))
        log("O2 %s" % json.dumps(desc["O2"]))

    res = {"meta": meta, "seeds": a.seeds, "descriptive": desc, "arms": {}}
    per = {a_: [] for a_ in ARMS}
    for arm in ARMS:
        rows = []
        for seed in a.seeds:
            f1, info = run_arm(X[arm], y, ids, folds, arm, seed, idx)
            per[arm].append(f1)
            rows.append({"seed": seed, "oof_macro_f1": f1, "folds": info})
            log("RESULT arm=%s seed=%d oof_macro_f1=%.4f" % (arm, seed, f1))
        res["arms"][arm] = {"seed_tag": str(ARM_SEED_TAG[arm]), "per_seed": rows,
                            "mean": float(np.mean(per[arm])),
                            "std": float(np.std(per[arm], ddof=1)) if len(per[arm]) > 1 else 0.0}

    boot_rng = np.random.default_rng(BOOT_SEED)

    def contrast(x, b):
        p = [u - v for u, v in zip(per[x], per[b])]
        d = np.asarray(p, dtype=np.float64)
        out = {"per_seed": p, "mean": float(np.mean(p)),
               "std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
               "n_positive": int(sum(1 for u in p if u > 0)), "n_seeds": len(p)}
        if len(d) > 1:
            idxb = boot_rng.integers(0, len(d), size=(N_BOOT, len(d)))
            means = d[idxb].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            out["ci95"] = [float(lo), float(hi)]
            out["se"] = float(means.std(ddof=1))
            out["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
        return out

    res["contrasts"] = {"arm2_minus_arm1c": contrast("2", "1c"),
                        "arm2_minus_arm1": contrast("2", "1"),
                        "arm1_minus_arm0": contrast("1", "0"),
                        "arm1_minus_arm1r": contrast("1", "1r"),
                        "arm1r_minus_arm0": contrast("1r", "0"),
                        "arm2_minus_arm1r": contrast("2", "1r"),
                        "arm1c_minus_arm1": contrast("1c", "1"),
                        "arm1c_minus_arm0": contrast("1c", "0"),
                        "arm2_minus_arm0": contrast("2", "0")}

    # ---- RE-AUDIT decision rule (frozen in idea-stage/REAUDIT_FREEZE.md) ----
    def passes(c):
        return bool(c["mean"] >= REAUDIT_BAR and c.get("ci_excludes_zero") and c["mean"] > 0)

    ca = res["contrasts"]
    res["reaudit"] = {
        "bar": REAUDIT_BAR, "n_seeds": len(a.seeds), "n_boot": N_BOOT,
        "R1_ocr_fusion": {
            "primary": "arm1_minus_arm0", "control": "arm1_minus_arm1r",
            "primary_pass": passes(ca["arm1_minus_arm0"]),
            "control_pass": passes(ca["arm1_minus_arm1r"]),
            "verdict": ("REVIVED" if (passes(ca["arm1_minus_arm0"])
                                      and passes(ca["arm1_minus_arm1r"])) else "NOT REVIVED"),
        },
        "R2_ocr_typing": {
            "primary": "arm2_minus_arm1c", "control": "arm2_minus_arm1",
            "primary_pass": passes(ca["arm2_minus_arm1c"]),
            "control_pass": passes(ca["arm2_minus_arm1"]),
            "verdict": ("REVIVED" if (passes(ca["arm2_minus_arm1c"])
                                      and passes(ca["arm2_minus_arm1"])) else "NOT REVIVED"),
        },
    }
    g = res["contrasts"]["arm2_minus_arm1c"]
    if g["mean"] >= GO_T and g["n_positive"] == g["n_seeds"]:
        verdict = "GO"
    elif g["mean"] > AMBIG_T:
        verdict = "AMBIGUOUS"
    else:
        verdict = "NO-GO"
    res["verdict"] = verdict
    res["verdict_rule"] = ("primary = seed-mean arm2-arm1c; GO if >= +0.010 AND positive on 3/3 "
                           "seeds; AMBIGUOUS if +0.003..+0.010 or mixed sign with positive mean; "
                           "NO-GO if <= +0.003")
    res["null_control_expectation"] = {"macro_f1_range": list(NULL_F1_RANGE),
                                       "max_abs_arm2_minus_arm1c": NULL_MAX_ABS_DELTA}
    res["arm1_reproduction_reference"] = {"prior_arm0_mean": 0.8104, "prior_ocr30_mean": 0.8198,
                                          "prior_delta": 0.0094}
    log("VERDICT arm0=%.4f arm1=%.4f arm1c=%.4f arm2=%.4f | 2-1c=%+.4f 2-1=%+.4f 1-0=%+.4f -> %s"
        % (np.mean(per["0"]), np.mean(per["1"]), np.mean(per["1c"]), np.mean(per["2"]),
           g["mean"], res["contrasts"]["arm2_minus_arm1"]["mean"],
           res["contrasts"]["arm1_minus_arm0"]["mean"], verdict))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
