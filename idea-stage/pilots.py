#!/usr/bin/env python
"""ARIS idea-creator Phase 5 pilots (P1/P2/P3).

Decision rules are frozen in idea-stage/PILOT_FREEZE.md and are NOT edited after results.
HateMM-train only. dev_seen / test are never opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/home/jehc223/Retrieval-hate")
RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
SEG = ROOT / "data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
SPANS = ROOT / "data/gt/HateMM/hate_spans.json"
K = 30
BOOT = 2000
SEED = 20260808


def l2(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


def load():
    seg = torch.load(SEG, map_location="cpu")
    who = torch.load(WHOLE, map_location="cpu")
    vids = list(seg["video_ids"])
    idx = {v: i for i, v in enumerate(vids)}
    S = seg["subclip_img_feats"].numpy().astype(np.float64).reshape(len(vids), K, -1)
    W_img = who["img_feats"].numpy().astype(np.float64)
    W_txt = who["text_feats"].numpy().astype(np.float64)
    y = who["labels"].numpy().astype(int)
    # whole-video cache order must match the segment cache order
    assert W_img.shape[0] == len(vids)
    wids = who["ids"][0] if (len(who["ids"]) == 1 and isinstance(who["ids"][0], list)) else who["ids"]
    assert list(wids) == vids, "whole-video / segment cache id order mismatch"
    folds = []
    for f in range(5):
        tr = json.load(open(RUN / f"folds/fold_{f}/train_ids.json"))
        qu = json.load(open(RUN / f"folds/fold_{f}/query_ids.json"))
        folds.append((np.array([idx[v] for v in tr]), np.array([idx[v] for v in qu])))
    spans = json.load(open(SPANS))
    return vids, idx, S, W_img, W_txt, y, folds, spans


def gold_seg_mask(vids, spans):
    """[V,K] bool: does segment k overlap any gold hateful span."""
    V = len(vids)
    M = np.zeros((V, K), dtype=bool)
    have = np.zeros(V, dtype=bool)
    for i, v in enumerate(vids):
        e = spans.get(v)
        if not e:
            continue
        D = float(e["duration"])
        sp = e.get("spans") or []
        if D <= 0 or not sp:
            continue
        have[i] = True
        for a, b in sp:
            k0 = int(np.floor(max(0.0, a) / D * K))
            k1 = int(np.ceil(min(D, b) / D * K))
            M[i, max(0, k0):min(K, max(k1, k0 + 1))] = True
    return M, have


def fit_predict(X, y, folds):
    """OOF predictions + per-fold macro-F1-maximizing threshold picked on the train side."""
    oof = np.full(len(y), np.nan)
    for tr, qu in folds:
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        ptr = clf.predict_proba(X[tr])[:, 1]
        grid = np.unique(np.round(ptr, 3))
        best, bt = -1.0, 0.5
        for t in grid:
            s = f1_score(y[tr], (ptr >= t).astype(int), average="macro")
            if s > best:
                best, bt = s, t
        pq = clf.predict_proba(X[qu])[:, 1]
        oof[qu] = (pq >= bt).astype(float)
    assert not np.isnan(oof).any()
    return oof.astype(int)


def mf1(y, p):
    return f1_score(y, p, average="macro")


def paired_boot(y, pa, pb, rng):
    """bootstrap distribution of macroF1(pb) - macroF1(pa)"""
    n = len(y)
    out = np.empty(BOOT)
    for b in range(BOOT):
        ii = rng.integers(0, n, n)
        out[b] = mf1(y[ii], pb[ii]) - mf1(y[ii], pa[ii])
    return out


# ------------------------------------------------------------------ P1
def pilot1(vids, S, W_img, W_txt, y, folds, spans, res):
    rng = np.random.default_rng(SEED)
    M, have = gold_seg_mask(vids, spans)
    V = len(vids)
    pos = np.where(y == 1)[0]
    # empirical window-length distribution from hateful videos that have gold spans
    lens = M[pos].sum(axis=1)
    lens = lens[lens > 0]
    res["p1_n_pos"] = int(len(pos))
    res["p1_n_pos_with_spans"] = int(have[pos].sum())
    res["p1_len_dist"] = {"median": float(np.median(lens)), "mean": float(lens.mean()),
                          "min": int(lens.min()), "max": int(lens.max())}
    # view B window per video (both classes)
    m = rng.choice(lens, size=V, replace=True)
    start = np.array([rng.integers(0, K - mi + 1) for mi in m])
    Bmask = np.zeros((V, K), dtype=bool)
    for i in range(V):
        Bmask[i, start[i]:start[i] + m[i]] = True
    Cmask = Bmask.copy()
    for i in pos:
        if have[i] and M[i].any():
            Cmask[i] = M[i]

    def view(mask):
        w = mask.astype(np.float64)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
        vis = np.einsum("vk,vkd->vd", w, S)
        return np.hstack([l2(vis), l2(W_txt)])

    XA = np.hstack([l2(S.mean(axis=1)), l2(W_txt)])
    XB, XC = view(Bmask), view(Cmask)
    pA, pB, pC = (fit_predict(X, y, folds) for X in (XA, XB, XC))
    fA, fB, fC = mf1(y, pA), mf1(y, pB), mf1(y, pC)
    d_generic = 100 * (fB - fA)
    d_oracle = 100 * (fC - fB)
    bo = 100 * paired_boot(y, pB, pC, np.random.default_rng(SEED))
    lo, hi = np.percentile(bo, [2.5, 97.5])
    res["P1"] = {
        "macroF1_full_A": fA, "macroF1_randwin_B": fB, "macroF1_goldwin_C": fC,
        "generic_trim_pts": d_generic, "oracle_alignment_pts": d_oracle,
        "oracle_boot_ci95": [float(lo), float(hi)],
    }
    if d_oracle >= 1.5 and lo > 0:
        v = "GO"
    elif hi <= 0.5:
        v = "GO-AS-NEGATIVE"
    else:
        v = "AMBIGUOUS"
    res["P1"]["verdict"] = v
    print(f"[P1] full={fA:.4f} randwin={fB:.4f} goldwin={fC:.4f} | "
          f"generic={d_generic:+.2f}pt oracle={d_oracle:+.2f}pt CI95=[{lo:+.2f},{hi:+.2f}] -> {v}")


# ------------------------------------------------------------------ P2
def pilot2(vids, S, W_img, W_txt, y, folds, spans, res):
    M, have = gold_seg_mask(vids, spans)
    V = len(vids)
    Sn = l2(S, axis=2)
    jstar = np.full(V, -1, dtype=int)
    for tr, qu in folds:
        mem = Sn[tr].reshape(-1, Sn.shape[2])            # [n_tr*30, d]
        mem_lab = np.repeat(y[tr], K)
        for i in qu:
            sim = Sn[i] @ mem.T                          # [30, n_mem]
            nb = np.argpartition(-sim, 20, axis=1)[:, :20]
            jstar[i] = int(np.argmax(mem_lab[nb].mean(axis=1)))
    assert (jstar >= 0).all()
    # metric 1: hit rate on hateful videos with gold spans
    ev = np.where((y == 1) & have & M.any(axis=1))[0]
    hits = M[ev, jstar[ev]].astype(float)
    chance = M[ev].sum(axis=1) / K
    rng = np.random.default_rng(SEED)
    bh = np.array([hits[rng.integers(0, len(ev), len(ev))].mean() for _ in range(BOOT)])
    hit, ch = float(hits.mean()), float(chance.mean())
    lo = float(np.percentile(bh, 2.5))
    # metric 2: classification
    Xb = np.hstack([l2(W_img), l2(W_txt)])
    Xs = np.hstack([l2(W_img), l2(W_txt), l2(S[np.arange(V), jstar])])
    pb, ps = fit_predict(Xb, y, folds), fit_predict(Xs, y, folds)
    d = 100 * (mf1(y, ps) - mf1(y, pb))
    bo = 100 * paired_boot(y, pb, ps, np.random.default_rng(SEED))
    res["P2"] = {
        "n_eval_videos": int(len(ev)), "hit_rate": hit, "chance_rate": ch,
        "ratio": hit / ch if ch else None, "hit_boot_lo95": lo,
        "macroF1_base": mf1(y, pb), "macroF1_with_selected_segment": mf1(y, ps),
        "delta_pts": d, "delta_boot_ci95": [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
    }
    m1 = "GO" if (hit >= 2 * ch and hit >= 0.35 and lo > ch) else ("NO-GO" if hit < 1.3 * ch else "AMBIGUOUS")
    m2 = "GO" if d >= 1.5 else ("NO-GO" if d < 0.5 else "AMBIGUOUS")
    res["P2"]["verdict_selection"], res["P2"]["verdict_classification"] = m1, m2
    res["P2"]["verdict"] = "GO" if (m1 == "GO" and m2 == "GO") else "NO-GO"
    print(f"[P2] hit={hit:.4f} chance={ch:.4f} ratio={hit/ch:.2f} lo95={lo:.4f} -> {m1} | "
          f"macroF1 {mf1(y,pb):.4f}->{mf1(y,ps):.4f} delta={d:+.2f}pt -> {m2}")


# ------------------------------------------------------------------ P3
def pilot3(vids, idx, S, W_img, W_txt, y, folds, res):
    rows = [json.loads(l) for l in open(RUN / "gate_c_audit.jsonl")]
    final = {}
    for r in rows:
        if r["coder_id"].endswith("c1"):
            final[r["video_id"]] = r
    for r in rows:
        if r["coder_id"].endswith("adj"):
            final[r["video_id"]] = r
    aud = [v for v in final if v in idx]
    ai = np.array([idx[v] for v in aud])
    t = np.array([1 if "on_screen_text" in final[v]["required_modalities"] else 0 for v in aud])
    fold_of = np.full(len(vids), -1, dtype=int)
    for f, (_, qu) in enumerate(folds):
        fold_of[qu] = f
    X = np.hstack([l2(W_img), l2(W_txt), l2(S.max(axis=1))])
    # (a) probe, grouped by frozen outer fold
    oof_t = np.full(len(aud), np.nan)
    for f in range(5):
        te = np.where(fold_of[ai] == f)[0]
        tr = np.where(fold_of[ai] != f)[0]
        if len(te) == 0 or len(np.unique(t[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[ai[tr]], t[tr])
        oof_t[te] = clf.predict_proba(X[ai[te]])[:, 1]
    ok = ~np.isnan(oof_t)
    auc = roc_auc_score(t[ok], oof_t[ok])
    rng = np.random.default_rng(SEED)
    tt, pp = t[ok], oof_t[ok]
    ba = []
    for _ in range(BOOT):
        ii = rng.integers(0, len(tt), len(tt))
        if len(np.unique(tt[ii])) < 2:
            continue
        ba.append(roc_auc_score(tt[ii], pp[ii]))
    lo = float(np.percentile(ba, 2.5))
    # (b) typed routing: predicted type bit for every train video (OOF w.r.t. the query fold)
    type_bit = np.zeros(len(vids), dtype=int)
    for f in range(5):
        tr = np.where(fold_of[ai] != f)[0]
        if len(np.unique(t[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X[ai[tr]], t[tr])
        qu = folds[f][1]
        type_bit[qu] = (clf.predict_proba(X[qu])[:, 1] >= 0.5).astype(int)
    # memory-side bits: for fold f, memory videos are in other folds -> use their own predicted bit
    Xn = l2(np.hstack([l2(W_img), l2(W_txt)]))
    per_fold = []
    for f, (tr, qu) in enumerate(folds):
        qs = [i for i in qu if i in set(ai.tolist())]
        if not qs:
            continue
        un, ty = [], []
        for i in qs:
            sim = Xn[i] @ Xn[tr].T
            nb = tr[np.argsort(-sim)[:20]]
            un.append((y[nb] == y[i]).mean())
            same = tr[type_bit[tr] == type_bit[i]]
            if len(same) >= 20:
                sim2 = Xn[i] @ Xn[same].T
                nb2 = same[np.argsort(-sim2)[:20]]
                ty.append((y[nb2] == y[i]).mean())
            else:
                ty.append(un[-1])
        per_fold.append({"fold": f, "n": len(qs), "purity_unrestricted": float(np.mean(un)),
                         "purity_typed": float(np.mean(ty)),
                         "delta": float(np.mean(ty) - np.mean(un))})
    nwin = sum(1 for d in per_fold if d["delta"] >= 0.05)
    va = "GO" if (auc >= 0.68 and lo > 0.55) else ("NO-GO" if auc <= 0.60 else "AMBIGUOUS")
    vb = "GO" if nwin >= 4 else "NO-GO"
    res["P3"] = {"n_audited": int(ok.sum()), "pos_rate": float(t[ok].mean()),
                 "probe_auroc": float(auc), "probe_boot_lo95": lo,
                 "typed_routing_per_fold": per_fold, "folds_with_delta_ge_0.05": nwin,
                 "verdict_probe": va, "verdict_routing": vb,
                 "verdict": "GO" if (va == "GO" and vb == "GO") else "NO-GO"}
    print(f"[P3] probe AUROC={auc:.4f} lo95={lo:.4f} (n={ok.sum()}, pos={t[ok].mean():.3f}) -> {va} | "
          f"typed routing folds>=+0.05: {nwin}/5 -> {vb}")


def main():
    vids, idx, S, W_img, W_txt, y, folds, spans = load()
    print(f"loaded V={len(vids)} pos={int(y.sum())} neg={int((1-y).sum())} K={K}")
    res = {}
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which:
        pilot1(vids, S, W_img, W_txt, y, folds, spans, res)
    if "2" in which:
        pilot2(vids, S, W_img, W_txt, y, folds, spans, res)
    if "3" in which:
        pilot3(vids, idx, S, W_img, W_txt, y, folds, res)
    out = ROOT / "idea-stage/pilot_results.json"
    prev = json.load(open(out)) if out.exists() else {}
    prev.update(res)
    json.dump(prev, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
