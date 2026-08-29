#!/usr/bin/env python
"""R3-2 — C12 "Proposition-mass firewall": formatting mass vs informative-window dilution.

Decision rules are FROZEN in idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section "Pilot R3-2")
and are NOT edited after any result is seen. Every ambiguity in the frozen text was resolved
by the most conservative reading available and is recorded verbatim in the output JSON under
"interpretations".

Zero test-set contact: only data/OCR/HateMM/{ocr_windows_K30.jsonl,frame_dims_train.json}
and data/gt/HateMM/{train,val}.jsonl are opened. An explicit path guard HALTs on any path
whose name contains "test".

Usage:
  python idea-stage/r3_pilot2_proposition_mass.py --smoke synthetic
  python idea-stage/r3_pilot2_proposition_mass.py --smoke permuted
  python idea-stage/r3_pilot2_proposition_mass.py --out idea-stage/r3_pilot2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path("/home/jehc223/Retrieval-hate")

# ------------------------------------------------------------ frozen constants --
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]            # 5 fixed seeds (shared protocol)
N_PERM_PER_SEED = 20               # 20 permutations per seed = 100 null replicates
MIN_CONF = 0.5                     # project OCR stats filter
MIN_TEXT_LEN = 2                   # project OCR stats filter
# frozen decision thresholds (R3_PILOT_FREEZE_2026-08-09.md, Pilot R3-2)
GO_RHO = 0.24
GO_A = 0.30
GO_NULL_MULT = 3.0
GO_CORR_CR = 0.80
# probe hyper-parameters (not frozen in the document; recorded in interpretations)
TFIDF_MAX_FEATURES = 100000
TFIDF_MIN_DF = 2
LR_C = 1.0
# attack parameters (not frozen in the document; recorded in interpretations)
ATK_REPEAT_FACTOR = 3              # token repetition: each token emitted 3x
ATK_BOX_SCALE_LINEAR = 3.0         # box-area scaling: linear x3 (area x9) on even-index dets
ATK_DUP_REL_AREA = 1.0             # cross-channel duplicate rendered at full frame area

_WS = re.compile(r"\s+")


class Halt(RuntimeError):
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------- guards --
_GUARD_ARMED = False
_TOUCHED = []


def arm_guard():
    global _GUARD_ARMED
    _GUARD_ARMED = True
    log("GUARD ARMED: any path whose name contains 'test' HALTs; "
        "allowed split tokens = {train, val, valid, dev_seen}")


def guard_path(p):
    if not _GUARD_ARMED:
        raise Halt("HALT_GUARD_NOT_ARMED")
    p = Path(p)
    low = str(p).lower()
    for part in p.parts:
        if "test" in part.lower():
            raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    if "test_seen" in low:
        raise Halt("HALT_TEST_CONTACT:path=%s" % p)
    _TOUCHED.append(str(p))
    return p


def guard_open(p, **kw):
    return open(guard_path(p), **kw)


def guard_torch_load(p):  # unused here; kept for parity with the pilot-A guard set
    import torch
    return torch.load(guard_path(p), map_location="cpu", weights_only=False)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------- math --
def normalise(s):
    """Unicode NFKC + lowercase + whitespace collapse (frozen in the pilot text)."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", s).lower()).strip()


def poly_area(bbox):
    """Shoelace area of the 4-point OCR box."""
    x = np.array([p[0] for p in bbox], dtype=np.float64)
    y = np.array([p[1] for p in bbox], dtype=np.float64)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def partial_corr(a, b, Z):
    """Pearson correlation of a and b after OLS-residualising both on [1 | Z]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    D = np.column_stack([np.ones(len(a)), Z])
    coef_a, *_ = np.linalg.lstsq(D, a, rcond=None)
    coef_b, *_ = np.linalg.lstsq(D, b, rcond=None)
    return pearson(a - D @ coef_a, b - D @ coef_b)


# --------------------------------------------------------------------- data --
def load_labels():
    lab = {}
    files = {}
    for split in ("train", "val"):
        p = ROOT / "data/gt/HateMM" / ("%s.jsonl" % split)
        n = 0
        with guard_open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                vid = str(o["id"])
                if vid in lab:
                    raise Halt("HALT_DUP_LABEL_ID:%s" % vid)
                lab[vid] = int(o["label"])
                n += 1
        files[str(p)] = {"rows": n, "sha256": sha256_file(p)}
    return lab, files


def load_ocr(lab):
    """Returns (records, files, counts). records = list of dicts per video."""
    files = {}
    p_dim = ROOT / "data/OCR/HateMM/frame_dims_train.json"
    with guard_open(p_dim, encoding="utf-8") as fh:
        dims_raw = json.load(fh)
    files[str(p_dim)] = {"entries": len(dims_raw), "sha256": sha256_file(p_dim)}

    p_ocr = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
    files[str(p_ocr)] = {"sha256": sha256_file(p_ocr)}

    per_video = {}          # vid -> list of (window_k, text_norm, conf, area_px)
    ext = {}                # vid -> [max_x, max_y] over ALL raw detections (fallback frame)
    n_lines = 0
    n_det_raw = 0
    n_det_filter = 0
    n_det_empty_after_norm = 0
    vids_seen = set()
    with guard_open(p_ocr, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            o = json.loads(line)
            n_lines += 1
            vid = str(o["video_id"])
            vids_seen.add(vid)
            per_video.setdefault(vid, [])
            e = ext.setdefault(vid, [0.0, 0.0])
            for t in o["texts"]:
                n_det_raw += 1
                xs = [float(q[0]) for q in t["bbox"]]
                ys = [float(q[1]) for q in t["bbox"]]
                e[0] = max(e[0], max(xs))
                e[1] = max(e[1], max(ys))
                if float(t["conf"]) < MIN_CONF or len(str(t["text"]).strip()) < MIN_TEXT_LEN:
                    continue
                n_det_filter += 1
                s = normalise(str(t["text"]))
                if not s:
                    n_det_empty_after_norm += 1
                    continue
                per_video[vid].append((int(o["window_k"]), s, float(t["conf"]),
                                       poly_area(t["bbox"])))

    if vids_seen - set(lab):
        raise Halt("HALT_OCR_VIDEO_NOT_IN_TRAIN_VAL:%d" % len(vids_seen - set(lab)))

    # frame area: recorded dims where available, else the per-video max observed
    # bbox extent (frame_dims_train.json covers the 744 train videos only).
    n_dim_fallback = 0
    frame_area = {}
    for vid in vids_seen:
        if vid in dims_raw:
            w, h = float(dims_raw[vid][0]), float(dims_raw[vid][1])
        else:
            w, h = ext[vid]
            n_dim_fallback += 1
        if w <= 0 or h <= 0:
            w, h = max(w, 1.0), max(h, 1.0)
        frame_area[vid] = w * h
    counts = {
        "n_ocr_lines": n_lines,
        "n_videos_in_ocr_cache": len(vids_seen),
        "n_detections_before_filter": n_det_raw,
        "n_detections_after_filter": n_det_filter,
        "n_detections_dropped_empty_after_normalisation": n_det_empty_after_norm,
        "n_detections_used": n_det_filter - n_det_empty_after_norm,
        "n_videos_frame_dims_fallback_to_bbox_extent": n_dim_fallback,
    }
    return per_video, frame_area, files, counts


# ------------------------------------------------------- structure building ---
class Struct:
    """Flat arrays for clean detections plus every attacked detection multiset."""
    pass


def build_struct(per_video, frame_area, lab):
    vids = sorted(v for v in per_video if len(per_video[v]) > 0)
    vidx = {v: i for i, v in enumerate(vids)}
    y = np.array([lab[v] for v in vids], dtype=np.int64)

    str_ids = {}

    def sid(s):
        i = str_ids.get(s)
        if i is None:
            i = len(str_ids)
            str_ids[s] = i
        return i

    det_v, det_w, det_s, det_win = [], [], [], []
    for v in vids:
        fa = frame_area[v]
        for (k, s, conf, area_px) in per_video[v]:
            det_v.append(vidx[v])
            det_win.append(k)
            det_s.append(sid(s))
            det_w.append(conf * np.sqrt(max(area_px, 0.0) / fa))
    det_v = np.array(det_v, dtype=np.int64)
    det_w = np.array(det_w, dtype=np.float64)
    det_s = np.array(det_s, dtype=np.int64)
    det_win = np.array(det_win, dtype=np.int64)

    # ---- attack variants (deterministic, label-independent) ----
    attacks = {}
    # A1 token repetition: each whitespace token emitted ATK_REPEAT_FACTOR times.
    rep_of = np.empty(len(str_ids), dtype=np.int64)
    strings = [None] * len(str_ids)
    for s, i in str_ids.items():
        strings[i] = s
    for i, s in enumerate(list(strings)):
        toks = s.split(" ")
        rep_of[i] = sid(" ".join(t for t in toks for _ in range(ATK_REPEAT_FACTOR)))
    attacks["token_repetition"] = (det_v, rep_of[det_s], det_w.copy())

    # A2 single-box splitting: every detection with >=2 tokens becomes one detection
    #    per token; the box is cut into equal-area pieces (area/ntok => w/sqrt(ntok)).
    tok_sids = {}
    for i, s in enumerate(list(strings)):
        toks = [t for t in s.split(" ") if t]
        tok_sids[i] = [sid(t) for t in toks] if len(toks) >= 2 else None
    sv, ss, sw = [], [], []
    for j in range(len(det_v)):
        tl = tok_sids[det_s[j]]
        if tl is None:
            sv.append(det_v[j]); ss.append(det_s[j]); sw.append(det_w[j])
        else:
            f = det_w[j] / np.sqrt(len(tl))
            for t in tl:
                sv.append(det_v[j]); ss.append(t); sw.append(f)
    attacks["single_box_split"] = (np.array(sv, dtype=np.int64),
                                   np.array(ss, dtype=np.int64),
                                   np.array(sw, dtype=np.float64))

    # A3 box reorder: reverse the reading order of every video's detection list.
    #    (r and c are both permutation-invariant -> an exact invariance check.)
    order = np.lexsort((-np.arange(len(det_v)), det_v))
    attacks["box_reorder"] = (det_v[order], det_s[order], det_w[order])

    # A4 box-area scaling: every even-index detection (reading order, within video)
    #    has its box scaled by ATK_BOX_SCALE_LINEAR linearly => weight x that factor.
    pos = np.zeros(len(det_v), dtype=np.int64)
    seen = {}
    for j in range(len(det_v)):
        v = int(det_v[j])
        pos[j] = seen.get(v, 0)
        seen[v] = pos[j] + 1
    w4 = det_w * np.where(pos % 2 == 0, ATK_BOX_SCALE_LINEAR, 1.0)
    attacks["box_area_scaling"] = (det_v, det_s, w4)

    # A5 cross-channel duplication: every detection is additionally rendered in a
    #    second channel at full frame area (same string, same conf).
    conf_arr = []
    for v in vids:
        fa = frame_area[v]
        for (k, s, conf, area_px) in per_video[v]:
            conf_arr.append(conf)
    conf_arr = np.array(conf_arr, dtype=np.float64)
    attacks["cross_channel_dup"] = (
        np.concatenate([det_v, det_v]),
        np.concatenate([det_s, det_s]),
        np.concatenate([det_w, conf_arr * np.sqrt(ATK_DUP_REL_AREA)]))

    strings = [None] * len(str_ids)
    for s, i in str_ids.items():
        strings[i] = s

    st = Struct()
    st.vids = vids
    st.y = y
    st.strings = strings
    st.det_v, st.det_s, st.det_w, st.det_win = det_v, det_s, det_w, det_win
    st.attacks = attacks
    st.n_vid = len(vids)

    # per-(video,window) group index for the window-concentration control
    key = det_v * 100 + det_win
    uk, st.win_g = np.unique(key, return_inverse=True)
    st.win_owner = (uk // 100).astype(np.int64)
    st.n_win_groups = len(uk)
    st.nwin = np.bincount(st.win_owner, minlength=st.n_vid).astype(np.float64)
    # (video, slot) address of every window group, precomputed once
    st.win_col = np.zeros(st.n_win_groups, dtype=np.int64)
    _slot = np.zeros(st.n_vid, dtype=np.int64)
    for _g in range(st.n_win_groups):
        _v = int(st.win_owner[_g])
        st.win_col[_g] = _slot[_v]
        _slot[_v] += 1

    # clean N_i / U_i and unique (video,string) pairs for c_i, per attack too
    st.N = np.bincount(det_v, minlength=st.n_vid).astype(np.float64)
    st.uniq = {}
    for name, (av, as_, aw) in [("__clean__", (det_v, det_s, det_w))] + list(attacks.items()):
        pairs = np.unique(av.astype(np.int64) * (len(str_ids) + 1) + as_.astype(np.int64))
        uv = (pairs // (len(str_ids) + 1)).astype(np.int64)
        us = (pairs % (len(str_ids) + 1)).astype(np.int64)
        st.uniq[name] = (uv, us, np.bincount(uv, minlength=st.n_vid).astype(np.float64))
    st.U = st.uniq["__clean__"][2]
    return st


# ----------------------------------------------------------------- scoring ---
def score_video(st, name, E):
    """(r_i, c_i) for the clean set (name='__clean__') or an attacked set."""
    if name == "__clean__":
        av, as_, aw = st.det_v, st.det_s, st.det_w
    else:
        av, as_, aw = st.attacks[name]
    e = E[as_]
    num = np.bincount(av, weights=aw * e, minlength=st.n_vid)
    den = np.bincount(av, weights=aw, minlength=st.n_vid)
    r = num / np.maximum(den, 1e-12)
    uv, us, uc = st.uniq[name]
    c = np.bincount(uv, weights=E[us], minlength=st.n_vid) / np.maximum(uc, 1e-12)
    return r, c


def window_control(st, E):
    """k_i = mean(top-3 window logits) - mean(remaining non-empty window logits)."""
    e = E[st.det_s]
    s = np.bincount(st.win_g, weights=e, minlength=st.n_win_groups)
    n = np.bincount(st.win_g, minlength=st.n_win_groups).astype(np.float64)
    wl = s / np.maximum(n, 1e-12)
    K = int(st.win_col.max()) + 1 if st.n_win_groups else 1
    M = np.full((st.n_vid, K), -np.inf)
    M[st.win_owner, st.win_col] = wl
    Ms = -np.sort(-M, axis=1)
    valid = np.isfinite(Ms)
    Mz = np.where(valid, Ms, 0.0)
    nwin = st.nwin
    topk = np.minimum(3.0, nwin)
    top_sum = Mz[:, :3].sum(axis=1)
    tot = Mz.sum(axis=1)
    rest_n = nwin - topk
    k = np.where(rest_n > 0,
                 top_sum / np.maximum(topk, 1e-12)
                 - (tot - top_sum) / np.maximum(rest_n, 1e-12),
                 0.0)
    return k, int((rest_n <= 0).sum())


def replicate(st, folds, models, y_use):
    """Compute rho and A for one label assignment, given per-fold fitted models."""
    E_by_fold = models["E_by_fold"]          # [n_folds, n_strings] standardised logits
    fold_of_vid = folds                       # [n_vid]
    n_s = E_by_fold.shape[1]
    # gather the fold-local logit for each string as used by each video
    # -> vector per video is not possible directly, so score per fold and stitch.
    r = np.zeros(st.n_vid); c = np.zeros(st.n_vid)
    ra = {a: np.zeros(st.n_vid) for a in st.attacks}
    ca = {a: np.zeros(st.n_vid) for a in st.attacks}
    kk = np.zeros(st.n_vid)
    n_short = 0
    for f in range(N_FOLDS):
        m = fold_of_vid == f
        if not m.any():
            continue
        E = E_by_fold[f]
        rf, cf = score_video(st, "__clean__", E)
        r[m] = rf[m]; c[m] = cf[m]
        for a in st.attacks:
            raf, caf = score_video(st, a, E)
            ra[a][m] = raf[m]; ca[a][m] = caf[m]
        kf, ns = window_control(st, E)
        kk[m] = kf[m]
        n_short += int(ns)
    z = (2.0 * y_use - 1.0) * (c - r)
    g = np.log(st.N) - np.log(st.U)
    Z = np.column_stack([kk, np.log(st.U), st.nwin, y_use.astype(np.float64)])
    rho = partial_corr(z, g, Z)
    sd_r = float(np.std(r, ddof=1))
    dr = np.max(np.stack([np.abs(ra[a] - r) for a in st.attacks]), axis=0) / max(sd_r, 1e-12)
    dc = np.max(np.stack([np.abs(ca[a] - c) for a in st.attacks]), axis=0) / max(sd_r, 1e-12)
    A = float(np.median(dr - dc))
    out = {
        "rho": float(rho), "A": A, "corr_c_r": pearson(c, r), "sd_r": sd_r,
        "mean_dr": float(np.mean(dr)), "mean_dc": float(np.mean(dc)),
        "per_attack_mean_abs_dr": {a: float(np.mean(np.abs(ra[a] - r)) / max(sd_r, 1e-12))
                                   for a in st.attacks},
        "per_attack_mean_abs_dc": {a: float(np.mean(np.abs(ca[a] - c)) / max(sd_r, 1e-12))
                                   for a in st.attacks},
        "n_videos_with_le3_nonempty_windows": n_short,
        "mean_z": float(np.mean(z)), "mean_g": float(np.mean(g)),
    }
    return out


# ------------------------------------------------------------------ fitting --
def make_folds(st, seed):
    """Label-independent shuffled 5-fold over videos (groups = videos by construction)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(st.n_vid)
    fold = np.zeros(st.n_vid, dtype=np.int64)
    fold[perm] = np.arange(st.n_vid) % N_FOLDS
    return fold


def fit_vectorizers(st, fold_of_vid, max_features):
    """Fit the fold-local TF-IDF (label-independent) and transform every string once."""
    per_fold = []
    for f in range(N_FOLDS):
        tr = fold_of_vid[st.det_v] != f
        docs = [st.strings[i] for i in st.det_s[tr]]
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=TFIDF_MIN_DF,
                              max_features=max_features, lowercase=False)
        Xtr = vec.fit_transform(docs)
        Xall = vec.transform(st.strings)
        per_fold.append({"Xtr": Xtr, "Xall": Xall, "tr_mask": tr,
                         "n_feat": Xtr.shape[1], "n_train_docs": Xtr.shape[0]})
    return per_fold


def fit_models(st, fold_of_vid, per_fold, y_use):
    """Refit only the logistic probe (the TF-IDF above is unsupervised => reusable)."""
    E = np.zeros((N_FOLDS, len(st.strings)), dtype=np.float64)
    for f in range(N_FOLDS):
        pf = per_fold[f]
        ytr = y_use[st.det_v[pf["tr_mask"]]]
        if len(np.unique(ytr)) < 2:
            raise Halt("HALT_DEGENERATE_FOLD_LABELS:fold=%d" % f)
        lr = LogisticRegression(max_iter=1000, C=LR_C, solver="liblinear")
        lr.fit(pf["Xtr"], ytr)
        d_tr = lr.decision_function(pf["Xtr"])
        m, s = float(d_tr.mean()), float(d_tr.std())
        if s < 1e-9:
            raise Halt("HALT_DEGENERATE_LOGIT_SCALE:fold=%d" % f)
        E[f] = (lr.decision_function(pf["Xall"]) - m) / s
    return {"E_by_fold": E}


def run_all(st, seeds, n_perm, max_features, tag):
    per_seed = []
    null_rho, null_A = [], []
    for si, seed in enumerate(seeds):
        t0 = time.time()
        fold = make_folds(st, seed)
        per_fold = fit_vectorizers(st, fold, max_features)
        models = fit_models(st, fold, per_fold, st.y)
        obs = replicate(st, fold, models, st.y)
        obs["seed"] = int(seed)
        obs["n_features_per_fold"] = [pf["n_feat"] for pf in per_fold]
        per_seed.append(obs)
        log("[progress] %s seed=%d obs rho=%.4f A=%.4f corr_cr=%.4f (%.1fs)"
            % (tag, seed, obs["rho"], obs["A"], obs["corr_c_r"], time.time() - t0))
        rng = np.random.default_rng(100000 + seed)
        for p in range(n_perm):
            yp = st.y[rng.permutation(st.n_vid)]
            mp = fit_models(st, fold, per_fold, yp)
            nr = replicate(st, fold, mp, yp)
            null_rho.append(nr["rho"]); null_A.append(nr["A"])
            if (p + 1) % 5 == 0 or p == n_perm - 1:
                log("[progress] %s seed=%d null %d/%d rho=%.4f A=%.4f elapsed=%.1fs"
                    % (tag, seed, p + 1, n_perm, nr["rho"], nr["A"], time.time() - t0))
        del per_fold
    return per_seed, np.array(null_rho), np.array(null_A)


def adjudicate(per_seed, null_rho, null_A):
    rhos = np.array([s["rho"] for s in per_seed])
    As = np.array([s["A"] for s in per_seed])
    ccr = np.array([s["corr_c_r"] for s in per_seed])
    rho_obs = float(np.mean(rhos))
    A_obs = float(np.mean(As))
    corr_cr = float(np.mean(ccr))
    n95_rho = float(np.percentile(null_rho, 95))
    n95_A = float(np.percentile(null_A, 95))
    cond = {
        "1_rho_obs_ge_0.24": bool(rho_obs >= GO_RHO),
        "2_A_obs_ge_0.30": bool(A_obs >= GO_A),
        "3_rho_obs_ge_3xN95rho": bool(rho_obs >= GO_NULL_MULT * n95_rho),
        "4_A_obs_ge_3xN95A": bool(A_obs >= GO_NULL_MULT * n95_A),
        "5_corr_c_r_ge_0.80": bool(corr_cr >= GO_CORR_CR),
    }
    verdict = "GO" if all(cond.values()) else "KILL"
    # sensitivity of the verdict to the seed-aggregation reading (mean vs min)
    rho_min, A_min, ccr_min = float(rhos.min()), float(As.min()), float(ccr.min())
    cond_min = {
        "1_rho_obs_ge_0.24": bool(rho_min >= GO_RHO),
        "2_A_obs_ge_0.30": bool(A_min >= GO_A),
        "3_rho_obs_ge_3xN95rho": bool(rho_min >= GO_NULL_MULT * n95_rho),
        "4_A_obs_ge_3xN95A": bool(A_min >= GO_NULL_MULT * n95_A),
        "5_corr_c_r_ge_0.80": bool(ccr_min >= GO_CORR_CR),
    }
    # If N95 is negative, "3 x N95" is a LOWER bar than N95 itself. The frozen rule is
    # applied literally; the strict variant 3 x max(N95, 0) is reported alongside so the
    # reader can re-adjudicate. (It can only matter when the absolute threshold already
    # fails, since conditions 1/2 force the observed statistic above 0.24 / 0.30 anyway.)
    cond_strictnull = {
        "3_rho_obs_ge_3xmax(N95rho,0)": bool(rho_obs >= GO_NULL_MULT * max(n95_rho, 0.0)),
        "4_A_obs_ge_3xmax(N95A,0)": bool(A_obs >= GO_NULL_MULT * max(n95_A, 0.0)),
    }
    return {
        "rho_obs": rho_obs, "A_obs": A_obs, "corr_c_r": corr_cr,
        "N95_rho": n95_rho, "N95_A": n95_A,
        "3xN95_rho": GO_NULL_MULT * n95_rho, "3xN95_A": GO_NULL_MULT * n95_A,
        "conditions_strict_null_variant": cond_strictnull,
        "rho_per_seed": [float(v) for v in rhos],
        "A_per_seed": [float(v) for v in As],
        "corr_c_r_per_seed": [float(v) for v in ccr],
        "rho_min_over_seeds": rho_min, "A_min_over_seeds": A_min,
        "corr_c_r_min_over_seeds": ccr_min,
        "null_n_replicates": int(len(null_rho)),
        "null_rho_mean": float(np.mean(null_rho)), "null_rho_max": float(np.max(null_rho)),
        "null_A_mean": float(np.mean(null_A)), "null_A_max": float(np.max(null_A)),
        "conditions": cond,
        "verdict": verdict,
        "conditions_under_min_over_seeds": cond_min,
        "verdict_under_min_over_seeds": "GO" if all(cond_min.values()) else "KILL",
    }


# --------------------------------------------------------------- synthetic ---
def synthetic_data(n_vid=200, seed=7, plant=True):
    """Synthetic OCR cache with a PLANTED positive proposition-mass effect.

    Each video repeats one label-neutral boilerplate string a video-specific number of
    times (the same distribution in both classes, so the probe learns nothing from it).
    That inflates N_i relative to U_i (large g_i) and shrinks the mass-weighted raw score
    r_i toward zero, while the unique-string score c_i is barely affected, so
    z_i = (2y-1)(c_i - r_i) rises with g_i => rho should come out clearly positive.
    Used only to prove the estimator can recover a real effect; never gates anything.
    """
    rng = np.random.default_rng(seed)
    hate = ["kill them all", "vermin filth", "go back home", "subhuman scum"]
    benign = ["subscribe now", "like and share", "chapter one", "breaking news"]
    filler = ["www example com", "omegle", "talk to strangers", "zions bank"]
    lab, per_video, frame_area = {}, {}, {}
    for i in range(n_vid):
        v = "syn_video_%03d" % i
        y = int(i % 2 == 0)
        lab[v] = y
        frame_area[v] = 854.0 * 480.0
        dets = []
        pool = hate if y else benign
        n_rep = int(rng.integers(0, 25)) if plant else 0
        for k in range(30):
            if rng.random() < 0.35:
                continue
            for _ in range(int(rng.integers(1, 4))):
                s = pool[int(rng.integers(0, len(pool)))] if rng.random() < 0.5 \
                    else filler[int(rng.integers(0, len(filler)))]
                area = float(rng.uniform(500, 40000))
                dets.append((k, normalise(s), float(rng.uniform(0.5, 1.0)), area))
        # the repeats go into windows that are ALREADY non-empty, so that the planted
        # format mass is NOT confounded with the non-empty-window count that the frozen
        # partial correlation conditions on.
        neutral_s = normalise("copyright disclaimer fair use")
        used = sorted({d[0] for d in dets}) or [0]
        for j in range(n_rep):
            dets.append((used[j % len(used)], neutral_s, 0.95, 60000.0))
        if not dets:
            dets.append((0, filler[0], 0.9, 1000.0))
        per_video[v] = dets
    return lab, per_video, frame_area


# ------------------------------------------------------------------- main ----
INTERPRETATIONS = {
    "seed_aggregation": (
        "The freeze fixes 5 seeds and 5-fold video-level OOF estimation but not how the "
        "5 per-seed statistics combine into the single gated value. Primary reading = mean "
        "across the 5 seeds. The verdict under the stricter min-across-seeds reading is also "
        "computed and reported ('verdict_under_min_over_seeds'); if the two disagree the "
        "conservative (min) reading is the one that governs the write-up."),
    "detection_filter": (
        "min_conf=0.5 and min_text_len=2 applied to the RAW detection exactly as in "
        "scripts/ocr_cache/ocr_stats.py (conf >= 0.5 and len(text.strip()) >= 2), then NFKC + "
        "lowercase + whitespace-collapse normalisation; detections whose normalised string is "
        "empty are additionally dropped and counted."),
    "probe": (
        "Fold-local TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2, "
        "max_features=%d, lowercase=False) + LogisticRegression(C=%.1f, solver='liblinear'). "
        "Documents = individual detections (with multiplicity) from the 4 training folds only. "
        "Detection label = its video's label. e_ij = decision_function, standardised by the "
        "mean/SD of the same fold's TRAINING detection logits (fold-local, no held-out "
        "statistics). Only continuous unbounded logits are used anywhere."
        % (TFIDF_MAX_FEATURES, LR_C)),
    "folds": (
        "5 folds assigned by shuffling the video list with the seed and taking index %% 5. "
        "Label-independent by construction, so the identical partition is used for the "
        "observed run and for every label-permuted null replicate of that seed, and no "
        "video's detections can straddle a fold boundary."),
    "tfidf_reuse_in_null": (
        "The TF-IDF vectoriser is unsupervised: its fit depends only on the training-fold "
        "detection strings, never on labels. It is therefore fitted once per (seed, fold) and "
        "reused across that seed's 20 permutations; only the logistic probe is refitted. This "
        "is an exact-equivalence speedup, not an approximation."),
    "box_area": (
        "Box area = shoelace area of the 4-point bbox, divided by the frame area, then sqrt. "
        "frame_dims_train.json covers only the 744 train videos; for the 107 val videos the "
        "frame area falls back to the per-video maximum observed bbox extent (max x times "
        "max y over all raw detections of that video). Because r_i is a weighted MEAN, any "
        "per-video constant rescaling of the weights cancels exactly, so this fallback cannot "
        "affect r_i, c_i, rho or corr(c,r); it only sets the reference area used by the "
        "cross-channel-duplication attack."),
    "window_control_k": (
        "Window logit = unweighted mean of e_ij over the detections in that (video, window). "
        "k_i = mean of the top-3 window logits minus the mean of the remaining non-empty "
        "window logits. If a video has <= 3 non-empty windows the 'remaining' set is empty "
        "and k_i is set to 0 (count reported)."),
    "excluded_videos": (
        "Videos with zero surviving detections have undefined r_i / c_i / g_i and are excluded "
        "from every video-level statistic; the count is reported. No video is dropped for any "
        "other reason."),
    "rho_definition": (
        "rho_obs = Pearson correlation between z and g after OLS-residualising BOTH on "
        "[1, k, log U, non-empty-window count, y]. One-sided: the frozen rule requires "
        "rho_obs >= +0.24, so a negative partial correlation fails condition 1."),
    "A_definition": (
        "A_obs = median_i [ max_a |r_i^(a) - r_i| - max_a |c_i^(a) - c_i| ], with both terms "
        "divided by the SD (ddof=1) of the CLEAN raw score r over the retained videos of that "
        "seed ('in clean raw-score standard deviations'). The max runs over all five frozen "
        "attacks. Attacks are applied to the detection multiset and both scorers are "
        "recomputed from the identical attacked multiset with the identical frozen fold model "
        "(no refitting under attack)."),
    "attack_token_repetition": (
        "Each detection's normalised text is rewritten so that every whitespace token appears "
        "%d times consecutively ('the cat' -> 'the the the cat cat cat'). Confidences, boxes, "
        "detection count and window assignment unchanged. Changes e_ij for both scorers."
        % ATK_REPEAT_FACTOR),
    "attack_single_box_split": (
        "Every detection whose normalised text has >= 2 whitespace tokens is replaced by one "
        "detection per token, keeping the confidence and giving each piece an equal share of "
        "the original box area (area/n_tokens, i.e. weight/sqrt(n_tokens)). Single-token "
        "detections are untouched. Raises N_i without adding propositions."),
    "attack_box_reorder": (
        "The reading order of each video's detection list is reversed. Both r_i and c_i are "
        "permutation-invariant by construction, so this attack must produce exactly zero "
        "displacement; it is an implementation invariance check and is reported per attack."),
    "attack_box_area_scaling": (
        "Every even-index detection in each video's reading order has its box scaled by %.1f "
        "linearly (area x%.1f), i.e. its weight multiplied by %.1f; odd-index detections are "
        "untouched. Deterministic and label-agnostic (an attacker cannot see the probe). "
        "Uniform scaling of ALL boxes would be a no-op because r_i is a weighted mean, so a "
        "non-uniform but content-independent choice is the only non-degenerate reading. "
        "c_i is invariant by construction."
        % (ATK_BOX_SCALE_LINEAR, ATK_BOX_SCALE_LINEAR ** 2, ATK_BOX_SCALE_LINEAR)),
    "attack_cross_channel_dup": (
        "Every detection is additionally emitted in a second channel with the same string and "
        "confidence but rendered at full frame area (relative area 1.0). N_i doubles, U_i is "
        "unchanged, so c_i is EXACTLY invariant while r_i is re-weighted. Note that an exact "
        "duplication that preserved the box would move neither score, because r_i is a "
        "weighted mean of a multiset; the second-channel rendering size is what makes the "
        "attack non-degenerate."),
    "negative_N95": (
        "The frozen conditions 3/4 read 'observed >= 3 x N95'. If a null 95th percentile is "
        "negative, 3 x N95 is a weaker bar than N95 itself. The frozen rule is applied "
        "literally and unchanged; the stricter variant 3 x max(N95, 0) is additionally "
        "reported under 'conditions_strict_null_variant'. It cannot change the verdict, "
        "because conditions 1 and 2 already require the observed statistic to exceed "
        "+0.24 / +0.30."),
    "null": (
        "20 permutations of the video label vector per seed x 5 seeds = 100 null replicates. "
        "Each replicate refits the logistic probe on the permuted detection labels and "
        "recomputes BOTH statistics end to end, including z_i = (2y-1)(c-r) and the y column "
        "of the conditioning set with the PERMUTED labels. N95 = one-sided (upper) 95th "
        "percentile over the 100 replicates."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-features", type=int, default=TFIDF_MAX_FEATURES)
    args = ap.parse_args()

    arm_guard()
    t0 = time.time()
    out = {
        "pilot": "R3-2 C12 proposition-mass firewall",
        "freeze": "idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section 'Pilot R3-2')",
        "mode": args.smoke or "real",
        "guard": "armed: any path containing 'test' HALTs",
        "config": {
            "n_folds": N_FOLDS, "seeds": SEEDS, "n_perm_per_seed": N_PERM_PER_SEED,
            "min_conf": MIN_CONF, "min_text_len": MIN_TEXT_LEN,
            "tfidf_max_features": args.max_features, "tfidf_min_df": TFIDF_MIN_DF,
            "lr_C": LR_C, "attack_repeat_factor": ATK_REPEAT_FACTOR,
            "attack_box_scale_linear": ATK_BOX_SCALE_LINEAR,
            "attack_dup_rel_area": ATK_DUP_REL_AREA,
            "thresholds": {"rho": GO_RHO, "A": GO_A, "null_mult": GO_NULL_MULT,
                           "corr_c_r": GO_CORR_CR},
        },
        "interpretations": INTERPRETATIONS,
    }

    if args.smoke == "synthetic":
        lab, per_video, frame_area = synthetic_data()
        st = build_struct(per_video, frame_area, lab)
        log("SMOKE synthetic: %d videos, %d detections, %d unique strings"
            % (st.n_vid, len(st.det_v), len(st.strings)))
        per_seed, nr, na = run_all(st, SEEDS[:2], 3, 20000, "synth")
        out["adjudication"] = adjudicate(per_seed, nr, na)
        out["per_seed"] = per_seed
        log("SMOKE synthetic verdict=%s rho=%.4f A=%.4f corr=%.4f N95rho=%.4f N95A=%.4f"
            % (out["adjudication"]["verdict"], out["adjudication"]["rho_obs"],
               out["adjudication"]["A_obs"], out["adjudication"]["corr_c_r"],
               out["adjudication"]["N95_rho"], out["adjudication"]["N95_A"]))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    lab, lab_files = load_labels()
    per_video, frame_area, ocr_files, counts = load_ocr(lab)
    files = {}
    files.update(lab_files)
    files.update(ocr_files)
    st = build_struct(per_video, frame_area, lab)
    counts["n_videos_retained"] = st.n_vid
    counts["n_videos_zero_surviving_detections"] = len(per_video) - st.n_vid
    counts["n_unique_normalised_strings_clean_plus_attacks"] = len(st.strings)
    counts["mean_N_i"] = float(np.mean(st.N))
    counts["mean_U_i"] = float(np.mean(st.U))
    counts["median_N_i"] = float(np.median(st.N))
    counts["median_U_i"] = float(np.median(st.U))
    counts["mean_nonempty_windows"] = float(np.mean(st.nwin))
    counts["label_positive_rate"] = float(np.mean(st.y))
    out["data"] = counts
    out["files"] = files
    log("data: %d videos retained (%d with zero surviving detections), %d detections "
        "before filter -> %d used, mean N_i=%.2f mean U_i=%.2f"
        % (st.n_vid, counts["n_videos_zero_surviving_detections"],
           counts["n_detections_before_filter"], counts["n_detections_used"],
           counts["mean_N_i"], counts["mean_U_i"]))

    if args.smoke == "permuted":
        rng = np.random.default_rng(999)
        st.y = st.y[rng.permutation(st.n_vid)]
        per_seed, nr, na = run_all(st, SEEDS[:1], 2, args.max_features, "permsmoke")
        out["adjudication"] = adjudicate(per_seed, nr, na)
        out["per_seed"] = per_seed
        log("SMOKE permuted verdict=%s rho=%.4f A=%.4f corr=%.4f"
            % (out["adjudication"]["verdict"], out["adjudication"]["rho_obs"],
               out["adjudication"]["A_obs"], out["adjudication"]["corr_c_r"]))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ---------------------------- REAL RUN (single submission) ----------------
    per_seed, nr, na = run_all(st, SEEDS, N_PERM_PER_SEED, args.max_features, "real")
    out["per_seed"] = per_seed
    out["adjudication"] = adjudicate(per_seed, nr, na)
    out["null_rho_all"] = [float(v) for v in nr]
    out["null_A_all"] = [float(v) for v in na]
    out["paths_touched"] = sorted(set(_TOUCHED))
    out["elapsed_sec"] = time.time() - t0
    a = out["adjudication"]
    log("RESULT rho_obs=%.4f (N95=%.4f, 3xN95=%.4f) A_obs=%.4f (N95=%.4f, 3xN95=%.4f) "
        "corr(c,r)=%.4f" % (a["rho_obs"], a["N95_rho"], a["3xN95_rho"], a["A_obs"],
                            a["N95_A"], a["3xN95_A"], a["corr_c_r"]))
    for k, v in a["conditions"].items():
        log("  cond %s: %s" % (k, "PASS" if v else "FAIL"))
    log("VERDICT: %s (under min-over-seeds: %s)"
        % (a["verdict"], a["verdict_under_min_over_seeds"]))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
