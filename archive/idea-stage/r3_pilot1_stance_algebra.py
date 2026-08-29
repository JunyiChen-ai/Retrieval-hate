#!/usr/bin/env python
"""R3-1 — C1 "Target-conditioned attack/defence algebra": algebraic double dissociation.

Decision rules are FROZEN in idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section "Pilot R3-1").
Nothing below changes a threshold, a statistic, or a rule. Every ambiguity that had to be
resolved is resolved to the most conservative (least GO-favouring) reading and is written into
the output JSON under "interpretations".

Zero test-set contact: PA's path guard is armed; every file open / torch.load goes through it,
so any path containing "test" HALTs. MHC "valid"/"dev_seen" is the validation split and is
allowed.

Usage:
  python idea-stage/r3_pilot1_stance_algebra.py --smoke synthetic
  python idea-stage/r3_pilot1_stance_algebra.py --smoke permuted
  python idea-stage/r3_pilot1_stance_algebra.py --out idea-stage/r3_pilot1.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "idea-stage"))
import pilot_a_disagreement_retrievability as PA  # noqa: E402

Halt = PA.Halt
log = PA.log

# ------------------------------------------------------------- frozen knobs --
# From the freeze: 5 fixed seeds, 5-fold video-level OOF, 128 PCA dims inside each fold,
# 20 permutations per seed (= 100 null replicates), N95 = one-sided 95th pct of the null.
SEEDS = [20260901, 20260902, 20260903, 20260904, 20260905]
NFOLD = 5
NPCA = 128
NPERM = 20
LOGREG_C = 1.0
MAX_ITER = 5000

# Frozen decision rule thresholds (R3_PILOT_FREEZE_2026-08-09.md, Pilot R3-1)
GO_MIN_MATCHED = 40          # cond 1
GO_D_CONTENT = 0.60          # cond 2
GO_G_CONTENT = 0.35          # cond 3 (upper bound)
GO_D_STANCE = 0.60           # cond 4
GO_PER_LANG = 0.35           # cond 5
GO_NULL_MULT = 3.0           # cond 6

CLIP = PA.CLIP
LANGS = PA.LANGS             # {"EN": {"ds": "MHC", "tsv": "English"}, "ZH": {...}}

# raw vote tokens used by the group definition (canonicalised by PA.parse_votes)
V_HATEFUL = "Hateful"
V_COUNTER = "Counter Narrative"
V_NORMAL = "Normal"

FILE_SHAS: dict[str, str] = {}


def _sha(p):
    p = str(p)
    if p not in FILE_SHAS:
        FILE_SHAS[p] = PA.sha256_file(p)
    return FILE_SHAS[p]


# ------------------------------------------------------------------- loaders --
def load_mpnet(ds):
    """{video_id: 768-d MPNet transcript embedding} for train + dev_seen."""
    out = {}
    for split in ("train", "dev_seen"):
        p = ROOT / "data/CLIP_Embedding" / ds / ("%s_transcript_mpnet512_HF.pt" % split)
        d = PA.guard_torch_load(p)
        _sha(p)
        raw = d["ids"]
        ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
        f = d["text_feats"].numpy().astype(np.float64)
        if len(ids) != f.shape[0]:
            raise Halt("HALT_MPNET_SHAPE:%s" % p)
        for i, v in enumerate(ids):
            out[str(v)] = f[i]
    return out


def load_whisper(ds):
    """{video_id: pooled Whisper-large-v3 audio-encoder embedding} (train+val cache).

    NOT part of the frozen feature spec (the freeze names CLAP, which does not exist for MHC).
    Used only for the clearly-labelled non-gating sensitivity variant.
    """
    p = ROOT / "data/audio" / ds / "whisper_whisper-large-v3_trainval.pt"
    if not p.exists():
        return None
    d = PA.guard_torch_load(p)
    _sha(p)
    raw = d["ids"]
    ids = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    f = d["emb"].numpy().astype(np.float64)
    if len(ids) != f.shape[0]:
        raise Halt("HALT_WHISPER_SHAPE:%s" % p)
    return {str(v): f[i] for i, v in enumerate(ids)}


def load_targets(tsv_lang):
    """{video_id: tuple(sorted(set(Target_Victim)))} from the train + valid vote TSVs."""
    out = {}
    for split in ("train", "valid"):
        p = ROOT / "data/gt/mhc_votes" / ("mhc_%s_%s.tsv" % (tsv_lang, split))
        _sha(p)
        with PA.guard_open(p, encoding="utf-8") as fh:
            hdr = fh.readline().rstrip("\r\n").split("\t")
            if hdr[:4] != ["Video_ID", "Majority_Voting", "Label", "Target_Victim"]:
                raise Halt("HALT_TSV_HEADER_TARGET:%s:%r" % (p, hdr))
            for line in fh:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                vid = parts[0].strip()
                tv = ast.literal_eval(parts[3])
                toks = tuple(sorted({("<NONE>" if x is None else str(x).strip()) for x in tv}))
                if vid in out and out[vid] != toks:
                    raise Halt("HALT_DUP_TARGET_ROW:%s" % vid)
                out[vid] = toks
    return out


def groups_from_votes(votes):
    """H / C / N / other, per the frozen group definition."""
    s = set(votes)
    if V_HATEFUL in s and V_COUNTER not in s:
        return "H"
    if V_COUNTER in s and V_HATEFUL not in s:
        return "C"
    if s == {V_NORMAL}:
        return "N"
    return "other"


# ------------------------------------------------------------------- matching --
def matched_mask(lang, grp, tgt_exact, tgt_key_mode):
    """Target matching: keep H/C items whose (language x Target_Victim stratum) contains at
    least one H AND at least one C.  Returns a boolean mask over all items."""
    keys = stratum_keys(lang, tgt_exact, tgt_key_mode)
    ok = set()
    have_h, have_c = {}, {}
    for i, k in enumerate(keys):
        if grp[i] == "H":
            have_h[k] = have_h.get(k, 0) + 1
        elif grp[i] == "C":
            have_c[k] = have_c.get(k, 0) + 1
    for k in have_h:
        if have_c.get(k, 0) > 0:
            ok.add(k)
    m = np.zeros(len(grp), dtype=bool)
    for i, k in enumerate(keys):
        if grp[i] in ("H", "C") and k in ok:
            m[i] = True
    return m, keys


def stratum_keys(lang, tgt_exact, mode):
    if mode == "exact":
        return [(lang[i], tgt_exact[i]) for i in range(len(lang))]
    if mode == "primary":
        return [(lang[i], tgt_exact[i][0] if tgt_exact[i] else "<EMPTY>")
                for i in range(len(lang))]
    raise Halt("HALT_BAD_STRATUM_MODE:%s" % mode)


def inv_freq_weights(grp, keys, idx):
    """Inverse-frequency weights over (stratum x class) cells, computed on `idx` only."""
    cnt = {}
    for i in idx:
        cnt[(keys[i], grp[i])] = cnt.get((keys[i], grp[i]), 0) + 1
    w = np.array([1.0 / cnt[(keys[i], grp[i])] for i in idx], dtype=np.float64)
    return w * (len(idx) / w.sum())


# ------------------------------------------------------------------ pipeline --
def _folds(strat, seed):
    """5-fold, stratified on (language x group). One row = one video, so video-level grouping
    is automatic (no video contributes rows to two folds)."""
    _, y = np.unique(np.array(strat, dtype=object).astype(str), return_inverse=True)
    # collapse strata with < NFOLD members so StratifiedKFold does not error
    cnt = np.bincount(y)
    y2 = np.where(cnt[y] < NFOLD, -1, y)
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=int(seed))
    return list(skf.split(np.zeros(len(y2)), y2))


def _zfit(X, idx):
    mu = X[idx].mean(axis=0)
    sd = np.maximum(X[idx].std(axis=0), 1e-8)
    return mu, sd


def run_pipeline(X, lang, grp, tgt_exact, seed, tgt_key_mode):
    """One complete replicate.  Returns the statistics dict (or NaNs where undefined)."""
    n = len(grp)
    grp = np.asarray(grp, dtype=object)
    isH = grp == "H"
    isC = grp == "C"
    isN = grp == "N"

    mmask, keys = matched_mask(lang, grp, tgt_exact, tgt_key_mode)
    n_mH = int((mmask & isH).sum())
    n_mC = int((mmask & isC).sum())

    q = np.full(n, np.nan)
    s = np.full(n, np.nan)
    q_tr_mu, q_tr_sd = [], []
    s_tr_mu, s_tr_sd = [], []

    strat = ["%s|%s" % (lang[i], grp[i]) for i in range(n)]
    for itr, ite in _folds(strat, seed):
        # ---- fold-local standardisation + PCA(128), fit on training-fold items only ----
        mu, sd = _zfit(X, itr)
        Z = (X - mu) / sd
        ncomp = min(NPCA, len(itr) - 1, X.shape[1])
        pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=int(seed))
        pca.fit(Z[itr])
        P = pca.transform(Z)
        pmu, psd = _zfit(P, itr)
        P = (P - pmu) / psd

        # ---- step 1: H-vs-N probe, C never used ----
        tr_hn = np.array([i for i in itr if isH[i] or isN[i]], dtype=int)
        if len(tr_hn) < 2 or len(set(grp[tr_hn])) < 2:
            continue
        yhn = isH[tr_hn].astype(int)
        m1 = LogisticRegression(C=LOGREG_C, penalty="l2", solver="lbfgs",
                                max_iter=MAX_ITER, class_weight="balanced")
        m1.fit(P[tr_hn], yhn)
        qall = m1.decision_function(P)
        te = np.array([i for i in ite if isH[i] or isC[i] or isN[i]], dtype=int)
        q[te] = qall[te]
        q_tr_mu.append(float(qall[itr].mean()))
        q_tr_sd.append(float(qall[itr].std()))

        # ---- step 2: remove the probe direction (fold-local projection) ----
        w = m1.coef_.ravel()
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            raise Halt("HALT_ZERO_PROBE_DIRECTION")
        u = w / nw
        Pr = P - np.outer(P @ u, u)

        # ---- step 3: target-matched H-vs-C probe with inverse-frequency weights ----
        tr_hc = np.array([i for i in itr if mmask[i]], dtype=int)
        te_hc = np.array([i for i in ite if mmask[i]], dtype=int)
        if len(tr_hc) < 2 or len(set(grp[tr_hc])) < 2 or len(te_hc) == 0:
            continue
        yhc = isH[tr_hc].astype(int)
        sw = inv_freq_weights(grp, keys, tr_hc)
        m2 = LogisticRegression(C=LOGREG_C, penalty="l2", solver="lbfgs", max_iter=MAX_ITER)
        m2.fit(Pr[tr_hc], yhc, sample_weight=sw)
        sall = m2.decision_function(Pr)
        s[te_hc] = sall[te_hc]
        s_tr_mu.append(float(sall[tr_hc].mean()))
        s_tr_sd.append(float(sall[tr_hc].std()))

    # ---- standardised OOF logits (pooled per replicate; see interpretations) ----
    okq = ~np.isnan(q)
    oks = ~np.isnan(s)
    qz = np.full(n, np.nan)
    sz = np.full(n, np.nan)
    if okq.sum() > 1 and np.std(q[okq]) > 1e-12:
        qz[okq] = (q[okq] - q[okq].mean()) / q[okq].std()
    if oks.sum() > 1 and np.std(s[oks]) > 1e-12:
        sz[oks] = (s[oks] - s[oks].mean()) / s[oks].std()

    def mean_of(vec, mask):
        m = mask & ~np.isnan(vec)
        return float(vec[m].mean()) if m.sum() > 0 else float("nan")

    def stats(sub):
        """sub = boolean mask selecting the population (all items, or one language)."""
        qH, qC, qN = (mean_of(qz, sub & isH), mean_of(qz, sub & isC), mean_of(qz, sub & isN))
        sH = mean_of(sz, sub & isH & mmask)
        sC = mean_of(sz, sub & isC & mmask)
        d_content = qC - qN
        g_content = abs(qH - qC)
        d_stance = sH - sC
        return {
            "D_content": d_content,
            "G_content": g_content,
            "D_stance": d_stance,
            "T": float(np.nanmin([d_content, d_stance]))
                 if not (np.isnan(d_content) or np.isnan(d_stance)) else float("nan"),
            "mean_q_H": qH, "mean_q_C": qC, "mean_q_N": qN,
            "mean_s_H": sH, "mean_s_C": sC,
            "n_H": int((sub & isH).sum()), "n_C": int((sub & isC).sum()),
            "n_N": int((sub & isN).sum()),
            "n_matched_H": int((sub & isH & mmask).sum()),
            "n_matched_C": int((sub & isC & mmask).sum()),
            "n_s_available": int((sub & mmask & ~np.isnan(sz)).sum()),
        }

    allm = np.ones(n, dtype=bool)
    out = stats(allm)
    out["per_language"] = {}
    for L in sorted(set(lang)):
        out["per_language"][L] = stats(np.array([x == L for x in lang]))
    out["n_matched_H_total"] = n_mH
    out["n_matched_C_total"] = n_mC
    out["n_folds_q_fitted"] = len(q_tr_mu)
    out["n_folds_s_fitted"] = len(s_tr_mu)
    # non-gating diagnostic: what the contrasts look like if the logits are instead scaled by
    # the fold-local TRAINING-fold logit SD (strictest fold-local reading of "standardised")
    if q_tr_sd and s_tr_sd:
        qs = float(np.mean(q_tr_sd))
        ss = float(np.mean(s_tr_sd))
        if qs > 1e-12 and ss > 1e-12:
            qq = q / qs
            ssv = s / ss
            out["diag_foldlocal_scaling"] = {
                "D_content": mean_of(qq, isC) - mean_of(qq, isN),
                "G_content": abs(mean_of(qq, isH) - mean_of(qq, isC)),
                "D_stance": mean_of(ssv, isH & mmask) - mean_of(ssv, isC & mmask),
            }
    return out


# ---------------------------------------------------------------------- null --
def permute_groups(lang, grp, tgt_exact, rng):
    """Permute the H/C/N assignment within language x target-present strata."""
    grp = list(grp)
    out = list(grp)
    buckets = {}
    for i in range(len(grp)):
        k = (lang[i], bool(len(tgt_exact[i]) > 0))
        buckets.setdefault(k, []).append(i)
    for k, idxs in buckets.items():
        vals = [grp[i] for i in idxs]
        perm = rng.permutation(len(vals))
        for j, i in enumerate(idxs):
            out[i] = vals[perm[j]]
    return np.array(out, dtype=object)


# ------------------------------------------------------------------- variant --
def run_variant(name, X, lang, grp, tgt_exact, nperm, tgt_key_mode, t0):
    obs_seeds, null_T = [], []
    null_D, null_S = [], []
    for si, sd in enumerate(SEEDS):
        r = run_pipeline(X, lang, grp, tgt_exact, sd, tgt_key_mode)
        obs_seeds.append(r)
        log("PROGRESS variant=%s seed=%d obs D_content=%.4f G_content=%.4f D_stance=%.4f "
            "T=%.4f elapsed=%.1fs"
            % (name, sd, r["D_content"], r["G_content"], r["D_stance"], r["T"],
               time.time() - t0))
        rng = np.random.default_rng(sd)
        for k in range(nperm):
            gp = permute_groups(lang, grp, tgt_exact, rng)
            rn = run_pipeline(X, lang, gp, tgt_exact, sd, tgt_key_mode)
            null_T.append(rn["T"])
            null_D.append(rn["D_content"])
            null_S.append(rn["D_stance"])
            if (k + 1) % 5 == 0:
                log("PROGRESS variant=%s seed=%d null=%d/%d elapsed=%.1fs"
                    % (name, sd, k + 1, nperm, time.time() - t0))

    def agg(key, sub=None):
        vals = [(o["per_language"][sub][key] if sub else o[key]) for o in obs_seeds]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    def n95(arr):
        a = np.array([v for v in arr if not np.isnan(v)], dtype=np.float64)
        return (float(np.percentile(a, 95)), int(a.size))

    D_content = agg("D_content")
    G_content = agg("G_content")
    D_stance = agg("D_stance")
    T_obs = float(min(D_content, D_stance)) if not (
        np.isnan(D_content) or np.isnan(D_stance)) else float("nan")
    N95, n_valid = n95(null_T)
    N95_D, _ = n95(null_D)
    N95_S, _ = n95(null_S)

    langs = sorted(obs_seeds[0]["per_language"].keys())
    per_lang = {}
    for L in langs:
        per_lang[L] = {
            "D_content": agg("D_content", L),
            "G_content": agg("G_content", L),
            "D_stance": agg("D_stance", L),
            "n_H": obs_seeds[0]["per_language"][L]["n_H"],
            "n_C": obs_seeds[0]["per_language"][L]["n_C"],
            "n_N": obs_seeds[0]["per_language"][L]["n_N"],
            "n_matched_H": obs_seeds[0]["per_language"][L]["n_matched_H"],
            "n_matched_C": obs_seeds[0]["per_language"][L]["n_matched_C"],
        }

    mH = obs_seeds[0]["n_matched_H_total"]
    mC = obs_seeds[0]["n_matched_C_total"]

    def ge(a, b):
        return bool((not np.isnan(a)) and a >= b)

    def le(a, b):
        return bool((not np.isnan(a)) and a <= b)

    cond = {
        "1_matched_counts_ge_40": {
            "pass": bool(mH >= GO_MIN_MATCHED and mC >= GO_MIN_MATCHED),
            "rule": ">= 40 target-matched H and >= 40 target-matched C",
            "matched_H": mH, "matched_C": mC},
        "2_D_content_ge_0.60": {
            "pass": ge(D_content, GO_D_CONTENT), "value": D_content, "threshold": GO_D_CONTENT},
        "3_G_content_le_0.35": {
            "pass": le(G_content, GO_G_CONTENT), "value": G_content, "threshold": GO_G_CONTENT},
        "4_D_stance_ge_0.60": {
            "pass": ge(D_stance, GO_D_STANCE), "value": D_stance, "threshold": GO_D_STANCE},
        "5_per_language_ge_0.35": {
            "pass": bool(all(ge(per_lang[L]["D_content"], GO_PER_LANG)
                             and ge(per_lang[L]["D_stance"], GO_PER_LANG) for L in langs)),
            "threshold": GO_PER_LANG,
            "values": {L: {"D_content": per_lang[L]["D_content"],
                           "D_stance": per_lang[L]["D_stance"]} for L in langs}},
        "6_T_obs_ge_3xN95": {
            "pass": ge(T_obs, GO_NULL_MULT * N95) if not np.isnan(N95) else False,
            "T_obs": T_obs, "N95": N95, "3xN95": GO_NULL_MULT * N95},
    }
    verdict = "GO" if all(c["pass"] for c in cond.values()) else "KILL"

    return {
        "variant": name,
        "stratum_mode": tgt_key_mode,
        "D_content": D_content,
        "G_content": G_content,
        "D_stance": D_stance,
        "T_obs": T_obs,
        "N95": N95,
        "N95_n_valid_replicates": n_valid,
        "N95_total_replicates": len(null_T),
        "N95_D_content_only": N95_D,
        "N95_D_stance_only": N95_S,
        "conditions": cond,
        "verdict": verdict,
        "per_language": per_lang,
        "group_sizes": {"H": obs_seeds[0]["n_H"], "C": obs_seeds[0]["n_C"],
                        "N": obs_seeds[0]["n_N"],
                        "matched_H": mH, "matched_C": mC},
        "per_seed": [{"seed": SEEDS[i],
                      "D_content": obs_seeds[i]["D_content"],
                      "G_content": obs_seeds[i]["G_content"],
                      "D_stance": obs_seeds[i]["D_stance"],
                      "T": obs_seeds[i]["T"],
                      "n_s_available": obs_seeds[i]["n_s_available"],
                      "n_folds_q_fitted": obs_seeds[i]["n_folds_q_fitted"],
                      "n_folds_s_fitted": obs_seeds[i]["n_folds_s_fitted"],
                      "diag_foldlocal_scaling": obs_seeds[i].get("diag_foldlocal_scaling"),
                      "per_language": {L: {k: obs_seeds[i]["per_language"][L][k]
                                           for k in ("D_content", "G_content", "D_stance")}
                                       for L in langs}}
                     for i in range(len(SEEDS))],
        "null_T_distribution": {
            "n": len(null_T),
            "n_valid": n_valid,
            "mean": float(np.nanmean(null_T)) if n_valid else float("nan"),
            "sd": float(np.nanstd(null_T)) if n_valid else float("nan"),
            "p50": float(np.nanpercentile(null_T, 50)) if n_valid else float("nan"),
            "p95": N95,
            "max": float(np.nanmax(null_T)) if n_valid else float("nan"),
        },
    }


# ---------------------------------------------------------------------- data --
def build_real_data(use_whisper):
    lang, ids_all, grp, tgt, blocks = [], [], [], [], []
    meta = {"per_language_raw": {}}
    per_file = {}
    Xparts = {"mpnet": [], "clip_img": [], "clip_txt": [], "whisper": []}
    for L, cfg in LANGS.items():
        ds = cfg["ds"]
        ids, img, txt, y, origin = PA.load_lang(L)
        for split in ("train", "dev_seen"):
            _sha(ROOT / "data/CLIP_Embedding" / ds / ("%s_%s.pt" % (split, CLIP)))
        for split in ("train", "val"):
            _sha(ROOT / "data/gt" / ds / ("%s.jsonl" % split))
        votes, n_alias, pf = PA.load_votes(cfg["tsv"])
        per_file.update(pf)
        targets = load_targets(cfg["tsv"])
        mp = load_mpnet(ds)
        wh = load_whisper(ds) if use_whisper else None

        miss_mp = [v for v in ids if v not in mp]
        miss_tg = [v for v in ids if v not in targets]
        if miss_mp:
            raise Halt("HALT_MPNET_JOIN:%d missing e.g. %r" % (len(miss_mp), miss_mp[:3]))
        if miss_tg:
            raise Halt("HALT_TARGET_JOIN:%d missing e.g. %r" % (len(miss_tg), miss_tg[:3]))
        miss_wh = [v for v in ids if wh is not None and v not in wh]
        if wh is not None and miss_wh:
            raise Halt("HALT_WHISPER_JOIN:%d missing" % len(miss_wh))

        g = [groups_from_votes(votes[v]) for v in ids]
        keep = [i for i, x in enumerate(g) if x in ("H", "C", "N")]
        meta["per_language_raw"][L] = {
            "n_cached_trainval": len(ids),
            "n_train": origin.count("train"), "n_val": origin.count("val"),
            "n_votes_rows_in_tsv": len(votes),
            "vote_alias_No_to_Normal": n_alias,
            "group_counts_all_cached": {k: int(sum(1 for x in g if x == k))
                                        for k in ("H", "C", "N", "other")},
        }
        for i in keep:
            lang.append(L)
            ids_all.append(ids[i])
            grp.append(g[i])
            tgt.append(targets[ids[i]])
            Xparts["mpnet"].append(mp[ids[i]])
            Xparts["clip_img"].append(img[i])
            Xparts["clip_txt"].append(txt[i])
            if wh is not None:
                Xparts["whisper"].append(wh[ids[i]])

    mats = [np.array(Xparts["mpnet"]), np.array(Xparts["clip_img"]),
            np.array(Xparts["clip_txt"])]
    names = ["mpnet_transcript", "clip_vitL14_336_image", "clip_vitL14_336_text"]
    if use_whisper and Xparts["whisper"]:
        mats.append(np.array(Xparts["whisper"]))
        names.append("whisper_large_v3_audio_NON_GATING")
    blocks = ["%s_d%d" % (names[i], mats[i].shape[1]) for i in range(len(mats))]
    X = np.concatenate(mats, axis=1)
    meta["files"] = per_file
    n_whisper = mats[-1].shape[1] if names[-1].startswith("whisper") else 0
    return X, lang, grp, tgt, ids_all, blocks, meta, n_whisper


def synthetic_data(n=420, d=180, seed=3):
    rng = np.random.default_rng(seed)
    lang, grp, tgt = [], [], []
    tset = [("Woman",), ("Man",), ("LGBTQ",), ()]
    for i in range(n):
        lang.append("EN" if i % 2 == 0 else "ZH")
        grp.append(["H", "C", "N"][i % 3])
        tgt.append(tset[i % 4])
    X = rng.normal(size=(n, d))
    content = np.array([1.0 if g in ("H", "C") else 0.0 for g in grp])
    stance = np.array([1.0 if g == "H" else (-1.0 if g == "C" else 0.0) for g in grp])
    X[:, 0] += 3.0 * content
    X[:, 1] += 3.0 * stance
    return X, lang, grp, tgt


# ---------------------------------------------------------------------- main --
INTERPRETATIONS = [
    {"id": "I1", "issue": "CLAP audio block",
     "text": "The freeze names cached CLAP audio embeddings as one of the three feature blocks. "
             "CLAP caches exist on disk ONLY for HateMM (data/audio/HateMM/clap_larger_clap_"
             "general_*), not for MHC or MHC_zh; data/audio/{MHC,MHC_zh}/ contain only Whisper "
             "ASR-encoder caches. The block is therefore genuinely unavailable for this pilot's "
             "data. It was NOT faked and NOT silently substituted: the GATING analysis uses only "
             "the two blocks that exist as specified (MPNet transcript + CLIP ViT-L/14-336 "
             "image and text). A clearly-labelled NON-GATING sensitivity variant "
             "'with_whisper_audio' adds the Whisper-large-v3 pooled audio embedding so the "
             "reader can see whether an audio channel would change anything."},
    {"id": "I2", "issue": "Corpus size",
     "text": "The freeze says 801 EN / 800 ZH. Those are the full vote-TSV row counts including "
             "the held-out test split. Only train + dev_seen embedding caches may be touched "
             "(zero test contact), giving 629 EN / 657 ZH cached videos before the H/C/N group "
             "filter. Analysis is run on those."},
    {"id": "I3", "issue": "Population entering the pipeline",
     "text": "Items whose votes fall outside H/C/N (e.g. Offensive-only, or Hateful+Counter "
             "Narrative mixtures) are dropped entirely: they are used neither for the fold-local "
             "standardisation/PCA nor for any probe, and they take no part in the null "
             "permutation. This is the reading that keeps the permutation null exactly a "
             "relabelling of H/C/N."},
    {"id": "I4", "issue": "Where PCA is fitted",
     "text": "Standardisation and PCA(128) are fitted fold-locally on ALL training-fold items "
             "(H, C and N together). Both are unsupervised, so no label information crosses the "
             "fold boundary, and a single shared basis is required for the step-2 direction "
             "removal to be meaningful for the step-3 probe. The H-vs-N probe itself is fitted "
             "on training-fold H and N only, i.e. C is never used to fit it, as frozen. PCA "
             "scores are re-standardised on the training fold before the logistic fit."},
    {"id": "I5", "issue": "Logit standardisation",
     "text": "'Standardised decision_function logit' is implemented as: assemble the OOF logit "
             "vector across the 5 folds, then z-score it by its own mean/SD within the "
             "replicate, so D/G are in pooled-logit SD units. A stricter fold-local alternative "
             "(scale by the mean training-fold logit SD) is reported per seed as the non-gating "
             "field 'diag_foldlocal_scaling'. No labels enter either scaling."},
    {"id": "I6", "issue": "Target_Victim stratum definition",
     "text": "Target_Victim is a list. The gating run uses the strictest reading: a stratum is "
             "(language, exact sorted set of Target_Victim labels). H/C items survive matching "
             "only if their stratum contains at least one H AND at least one C. This is the "
             "conservative choice (it retains the fewest items, so condition 1 is hardest to "
             "pass). A non-gating sensitivity variant 'permissive_stratum' instead keys on "
             "(language, alphabetically-first target label, or '<EMPTY>')."},
    {"id": "I7", "issue": "Inverse-frequency weights",
     "text": "Applied as sklearn sample_weight over (stratum x class) cells, computed on the "
             "training fold only, normalised to sum to the training-fold size. The H-vs-C "
             "logistic therefore uses no class_weight='balanced' on top (that would double-count "
             "the balancing). The H-vs-N probe, which has no matching step, uses "
             "class_weight='balanced'."},
    {"id": "I8", "issue": "Population for D_stance",
     "text": "D_stance is computed over the target-matched H and C items only, i.e. the same "
             "population the matched probe is defined on. Condition 1's counts are the same "
             "matched counts."},
    {"id": "I9", "issue": "Pooled vs per-language fitting",
     "text": "EN and ZH are pooled into one model (one fold split, one PCA, one probe pair); the "
             "per-language numbers required by condition 5 are the same pooled OOF logits "
             "restricted to each language's items. Refitting separately per language would be a "
             "different estimator than the one frozen, and the freeze's 7/10 clause ('not an "
             "EN/ZH pooling artifact') presumes a pooled main analysis."},
    {"id": "I10", "issue": "Aggregation over seeds and the null",
     "text": "Each of the 5 seeds gives one complete OOF replicate; the reported D_content, "
             "G_content, D_stance are the means over the 5 seeds, and T_obs = min(D_content, "
             "D_stance) of those means. The null is 20 within-(language x target-present) group "
             "permutations per seed = 100 replicates; N95 is the one-sided 95th percentile of "
             "the 100 null T values. Null replicates whose T is undefined (matched population "
             "too small in that permutation to fit any H-vs-C fold) are dropped before the "
             "percentile; dropping them raises N95 and so is the conservative choice. The count "
             "of valid replicates is reported."},
    {"id": "I11", "issue": "Seed values and unspecified hyper-parameters",
     "text": "The freeze fixes '5 fixed seeds' but not their values: 20260901-20260905 were "
             "chosen before any result was seen and drive fold splits, PCA and permutations. "
             "The freeze does not specify the logistic regularisation: C=1.0, L2, lbfgs, "
             "max_iter=5000 is used everywhere, with no tuning of any kind (no inner search), "
             "so nothing can be tuned toward a threshold."},
    {"id": "I12", "issue": "Video-level folds",
     "text": "One MHC row = one video, so a video-level grouped split is exactly a per-row "
             "split. StratifiedKFold on (language x group) is used so that every fold contains "
             "H, C and N where the counts allow; strata with fewer than 5 members are pooled "
             "into a single residual stratum so the splitter does not error."},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["synthetic", "permuted"], default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    PA.arm_guard()
    t0 = time.time()
    out = {
        "pilot": "R3-1 C1 target-conditioned attack/defence algebra",
        "freeze": "idea-stage/R3_PILOT_FREEZE_2026-08-09.md (section 'Pilot R3-1')",
        "mode": args.smoke or "real",
        "seeds": SEEDS, "nfold": NFOLD, "npca": NPCA, "nperm_per_seed": NPERM,
        "guard": "armed: any path containing 'test' HALTs",
        "interpretations": INTERPRETATIONS,
    }

    if args.smoke == "synthetic":
        X, lang, grp, tgt = synthetic_data()
        r = run_variant("synthetic", X, lang, grp, tgt, 3, "exact", t0)
        log("SMOKE synthetic verdict=%s D_content=%.4f G_content=%.4f D_stance=%.4f "
            "T=%.4f N95=%.4f matched H/C=%d/%d"
            % (r["verdict"], r["D_content"], r["G_content"], r["D_stance"], r["T_obs"],
               r["N95"], r["group_sizes"]["matched_H"], r["group_sizes"]["matched_C"]))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    X, lang, grp, tgt, ids_all, blocks, meta, nW = build_real_data(use_whisper=True)
    Xg = X[:, :X.shape[1] - nW] if nW else X
    log("data: n=%d items in H/C/N, feature dim gating=%d full=%d, blocks=%s"
        % (len(grp), Xg.shape[1], X.shape[1], blocks))
    log("group counts: %s" % {k: int(sum(1 for g in grp if g == k)) for k in ("H", "C", "N")})

    if args.smoke == "permuted":
        rng = np.random.default_rng(999)
        perm = rng.permutation(len(grp))
        gp = [grp[i] for i in perm]
        r = run_variant("permuted_smoke", Xg, lang, gp, tgt, 3, "exact", t0)
        log("SMOKE permuted verdict=%s D_content=%.4f G_content=%.4f D_stance=%.4f "
            "T=%.4f N95=%.4f matched H/C=%d/%d"
            % (r["verdict"], r["D_content"], r["G_content"], r["D_stance"], r["T_obs"],
               r["N95"], r["group_sizes"]["matched_H"], r["group_sizes"]["matched_C"]))
        log("elapsed %.1fs" % (time.time() - t0))
        return

    # ------------------------------- REAL RUN (single submission) -------------------------
    out["feature_blocks_gating"] = [b for b in blocks if "whisper" not in b]
    out["feature_blocks_available"] = blocks
    out["feature_dim_gating"] = int(Xg.shape[1])
    out["n_items_analysed"] = int(len(grp))
    out["data_meta"] = meta

    primary = run_variant("PRIMARY", Xg, lang, grp, tgt, NPERM, "exact", t0)
    out["primary"] = primary
    out["verdict"] = primary["verdict"]
    out["D_content"] = primary["D_content"]
    out["G_content"] = primary["G_content"]
    out["D_stance"] = primary["D_stance"]
    out["T_obs"] = primary["T_obs"]
    out["N95"] = primary["N95"]
    out["group_sizes"] = primary["group_sizes"]
    out["per_language"] = primary["per_language"]
    out["conditions"] = primary["conditions"]
    log("PRIMARY VERDICT=%s D_content=%.4f G_content=%.4f D_stance=%.4f T_obs=%.4f N95=%.4f"
        % (primary["verdict"], primary["D_content"], primary["G_content"],
           primary["D_stance"], primary["T_obs"], primary["N95"]))

    sens = {}
    sens["with_whisper_audio"] = run_variant(
        "with_whisper_audio", X, lang, grp, tgt, NPERM, "exact", t0)
    sens["permissive_stratum"] = run_variant(
        "permissive_stratum", Xg, lang, grp, tgt, NPERM, "primary", t0)
    out["sensitivity_non_gating"] = sens
    for k, v in sens.items():
        log("SENSITIVITY %s (NON-GATING) verdict=%s D_content=%.4f G_content=%.4f "
            "D_stance=%.4f T=%.4f N95=%.4f matched H/C=%d/%d"
            % (k, v["verdict"], v["D_content"], v["G_content"], v["D_stance"],
               v["T_obs"], v["N95"], v["group_sizes"]["matched_H"],
               v["group_sizes"]["matched_C"]))

    out["input_file_sha256"] = dict(sorted(FILE_SHAS.items()))
    out["paths_touched"] = sorted(set(PA._TOUCHED))
    out["elapsed_sec"] = time.time() - t0
    log("FINAL VERDICT: %s" % out["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                             default=lambda o: None))
        log("wrote %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except Halt as e:
        log("HALT %s" % e)
        sys.exit(3)
