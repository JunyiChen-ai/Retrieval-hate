#!/usr/bin/env python
"""
restrans_pregate.py -- FROZEN implementation of the $0 CPU pregate for membank
candidate C1, the RESIDUAL-TRANSPORT VOTE.

RECORD: refine-logs/RESTRANS_PREGATE_RECORD.md
DESIGN SOURCE (binding): refine-logs/LITSWEEP6_MEMBANK.md section 1 (a)-(f), in
particular section 1(e), whose frozen bars are quoted verbatim in the record before
any number in this file was computed.

THE IDEA UNDER TEST
    The deployed decision is  v = SUM_i (2*lab_i - 1) * cos_i * w_i / SUM_i w_i  over
    the top-20 own-train neighbours, w = [20..1] (src/utils/metrics.py:262-301,
    src/model/evaluate_rac.py:405-465; replayed by the F89-frozen
    mechfix_ops.deployed_vote). C1 replaces the LABEL SUMMAND

        s_i = 2*lab_i - 1        ->        r_i = s_i - (2*phat_i - 1)

    where phat_i = P^(hate | nuisance covariate of bank item i), fitted on the
    FITTING FOLDS ONLY, leave-one-out. Retrieval, k, rank weights, threshold and the
    key space are all untouched: the identical 20 neighbours in the identical order
    are retrieved and only the transported quantity changes.

ARENA (F95 harness verbatim)
    Banked RAW encoder key spaces (seed-independent), TRAIN SPLIT ONLY, K=5
    StratifiedKFold(shuffle=True, random_state=0) over train items, item-disjoint.
    PRIMARY space = fused; text and img are SECONDARY. This is the arena
    mechnov_pairverify.py (F95) froze, and its recorded per-fold and pooled deployed
    numbers are the parity anchors asserted below. The trained head is NOT the arena:
    it memorises its own train split (LOO train acc 0.998, F47), so a train-side
    screen in head space would be measuring memorisation.

    A SUPPLEMENTARY head-space read is deliberately NOT run: the head saw every train
    item, so a fold-disjoint bank in head space still leaks. Stated in the record.

TEST-SPLIT CONTACT: NONE. Only <cache>/train_<model>.pt and data/gt/<DS>/train.jsonl
(plus data/gt/HateMM/hate_spans.json for the HateMM-only B-c duration feature) are
opened. dev_seen / test_seen are never loaded.

COST: CPU only, <= 8 threads. Zero GPU, zero SLURM, zero Modal, zero training of any
deployed object.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechfix_ops as M          # noqa: E402  F89-frozen, 15/15 floor parity
import mechnov_pairverify as PV  # noqa: E402  F95-frozen harness (spaces, loaders)

# ------------------------------------------------------- FROZEN OPERATOR CONTRACT
MECHFIX_OPS_SHA = "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d"

# --------------------------------------------------------------- FROZEN CONSTANTS
K_FOLDS = 5                  # == PV.K_FOLDS
FOLD_SEED = 0                # == PV.FOLD_SEED  (StratifiedKFold shuffle=True)
TOPK = 20                    # deployed budget, unchanged
PATHOLOGY_RANK = 5           # ERRPAT/F95 pathology population definition
SPACES = ("fused", "text", "img")          # PRIMARY = fused
PRIMARY_SPACE = "fused"

# base-model arms (LITSWEEP6 section 1(d))
BA_LOGIT_C = 1.0             # sklearn default L2 strength
BA_MAXITER = 1000
BB_N_BINS = 10               # equal-count bins of the covariate
BB_FDS_KS = 5                # FDS kernel support, +/-2 bins
BB_FDS_SIGMA = 2.0           # FDS Gaussian sigma over bin index

# controls
NULL_SEED = 20260727         # frozen RNG for both shuffled-covariate nulls
D1_STANDARDISE = True        # fit-fold mean/sd standardisation for the dead relative

DATASETS = {
    "hatemm": dict(ds="HateMM", model="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
                   cache_dir=os.path.join(REPO, "data/CLIP_Embedding/HateMM"),
                   gt=os.path.join(REPO, "data/gt/HateMM/train.jsonl"),
                   vol="words",
                   dur=os.path.join(REPO, "data/gt/HateMM/hate_spans.json")),
    "zh": dict(ds="MHC_zh", model="Qwen2.5-VL-7B-Instruct-LoRA_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC_zh"),
               gt=os.path.join(REPO, "data/gt/MHC_zh/train.jsonl"),
               vol="chars", dur=None),
    "en": dict(ds="MHC", model="Qwen2.5-VL-7B-Instruct_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC"),
               gt=os.path.join(REPO, "data/gt/MHC/train.jsonl"),
               vol="words", dur=None),
}

BASE_ARMS = ("Ba", "Bb", "Bc")       # Bc runs only where a duration source exists
TREATMENTS = ("C1_Ba", "C1_Bb", "C1_Bc")
CONTROLS = ("D1_lenlogit", "N1_permcov", "N2_permphat")


# --------------------------------------------------------------------- helpers
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def load_volume(gt_path, ids, mode):
    """Transcript volume per id, EXACTLY the F89-T3 frozen definition
    (mechfix_run.volume_scalar): whitespace tokens for HateMM/MHC-EN, characters of
    the composed gt text for MHC-ZH, read from data/gt/<DS>/train.jsonl "text"."""
    txt = {}
    for line in open(gt_path):
        r = json.loads(line)
        txt[r["id"]] = r["text"]
    miss = [i for i in ids if i not in txt]
    assert not miss, ("gt text missing", miss[:5])
    return np.asarray([len(txt[i].split()) if mode == "words" else len(txt[i])
                       for i in ids], dtype="float64")


def load_duration(dur_path, ids):
    if dur_path is None:
        return None
    sp = json.load(open(dur_path))
    out = []
    for i in ids:
        d = sp.get(i, {}).get("duration", None)
        if d is None or not (float(d) > 0):
            return None                      # all-or-nothing: no partial coverage
        out.append(float(d))
    return np.asarray(out, dtype="float64")


# ---------------------------------------------------------------- the C1 operator
def residual_vote(bank_keys, bank_lab, query_keys, r_bank, topk=TOPK):
    """The deployed vote with the label summand s_i replaced by the residual r_i.

    Retrieval engine is bit-identical to mechfix_ops.deployed_vote: same _norm32,
    same faiss.IndexFlatIP, same k, same rank weights, same threshold at 0. The ONLY
    difference is the quantity being summed. With r_bank = 2*lab - 1 this function is
    the deployed vote, and that identity is asserted as an implementation gate.
    """
    b = M._norm32(bank_keys)
    q = M._norm32(query_keys)
    D, I = M._flat_ip(b, q, topk)
    r = np.asarray(r_bank, dtype="float64")[I]
    sim = D.astype("float64")
    w = M._rank_weights(topk)
    votes = (r * sim * w).sum(1) / w.sum()
    return votes, (votes >= 0).astype(int), I, sim


# ------------------------------------------------------------- base models for phat
def phat_logistic_loo(feat, lab):
    """B-a / B-c: logistic on the covariate(s), genuine leave-one-out over the
    fitting-fold items. feat is (n, d) with d = 1 (B-a) or 2 (B-c)."""
    n = len(lab)
    out = np.empty(n, dtype="float64")
    F = np.asarray(feat, dtype="float64").reshape(n, -1)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        yi = lab[m]
        if yi.min() == yi.max():                       # degenerate fold (never here)
            out[i] = float(yi.mean())
            continue
        clf = LogisticRegression(penalty="l2", C=BA_LOGIT_C, solver="lbfgs",
                                 max_iter=BA_MAXITER, n_jobs=1)
        clf.fit(F[m], yi)
        out[i] = float(clf.predict_proba(F[i:i + 1])[0, 1])
    return out


def _fds_kernel():
    half = BB_FDS_KS // 2
    off = np.arange(-half, half + 1, dtype="float64")
    k = np.exp(-0.5 * (off / BB_FDS_SIGMA) ** 2)
    return k / k.sum()


def phat_binned_fds_loo(v, lab):
    """B-b: ordered equal-count bins of the covariate, bin hate-rate smoothed across
    NEIGHBOURING bins with an FDS-style Gaussian kernel (arXiv:2102.09554), genuine
    leave-one-out. Counts (numerator and denominator) are each convolved with the
    kernel and the rate is their ratio, which is the rate-valued analogue of FDS'
    statistics smoothing and is what makes 5-20-item bins usable."""
    n = len(lab)
    edges = np.quantile(v, np.linspace(0.0, 1.0, BB_N_BINS + 1))
    edges = np.unique(edges)
    nb = max(len(edges) - 1, 1)
    b = np.clip(np.digitize(v, edges[1:-1], right=False), 0, nb - 1) if nb > 1 \
        else np.zeros(n, dtype=int)
    tot = np.bincount(b, minlength=nb).astype("float64")
    hit = np.bincount(b, weights=lab.astype("float64"), minlength=nb)
    ker = _fds_kernel()
    half = BB_FDS_KS // 2

    def smooth(x):
        out = np.empty(nb, dtype="float64")
        for j in range(nb):
            lo, hi = max(0, j - half), min(nb - 1, j + half)
            wk = ker[(lo - j + half):(hi - j + half + 1)]
            out[j] = float((x[lo:hi + 1] * wk).sum() / wk.sum())
        return out

    out = np.empty(n, dtype="float64")
    for j in range(nb):                      # only 2 distinct LOO states per bin
        for y in (0, 1):
            sel = (b == j) & (lab == y)
            if not sel.any():
                continue
            t = tot.copy(); h = hit.copy()
            t[j] -= 1.0; h[j] -= float(y)
            st, sh = smooth(t), smooth(h)
            out[sel] = sh[j] / st[j] if st[j] > 0 else float(lab.mean())
    return out, nb, b


# ------------------------------------------------------- the dead relative (D1)
def d1_length_logit(vote_fit, vol_fit, lab_fit, vote_ho, vol_ho):
    """The DEAD score-level length de-bias (ERRPAT-HateMM 4.3/6.2, train-LOO fit
    -0.0016 acc): logistic(deployed vote, log(1+volume)) -> gold, a monotone
    reweighting of the FINAL SCALAR applied AFTER retrieval. C1 must beat it."""
    Xf = np.column_stack([vote_fit, np.log1p(vol_fit)])
    Xh = np.column_stack([vote_ho, np.log1p(vol_ho)])
    if D1_STANDARDISE:
        mu, sd = Xf.mean(0), Xf.std(0)
        sd[sd == 0] = 1.0
        Xf = (Xf - mu) / sd
        Xh = (Xh - mu) / sd
    clf = LogisticRegression(penalty="l2", C=BA_LOGIT_C, solver="lbfgs",
                             max_iter=BA_MAXITER, n_jobs=1)
    clf.fit(Xf, lab_fit)
    p = clf.predict_proba(Xh)[:, 1]
    return (p >= 0.5).astype(int)


# ----------------------------------------------------------- ERRPAT worked example
def worked_example():
    """LITSWEEP6 1(c): correct analogue at rank 1, nineteen wrong-class neighbours at
    ranks 2-20, rank weights [20..1]. The record states the normalised vote moves
    from -0.81 to -0.10. Reproduced closed-form under the uniform-cosine limit the
    configuration implies (the space is cone-collapsed, top-1 cos 0.9439-0.9686)."""
    w = np.arange(1, 21)[::-1].astype("float64")
    s = np.full(20, -1.0); s[0] = +1.0
    cos = np.ones(20)
    v_dep = float((s * cos * w).sum() / w.sum())
    out = {"config": "rank-1 same-class, ranks 2-20 wrong-class, w=[20..1], cos=1",
           "v_deployed": round(v_dep, 4)}
    for nm, ph in (("phat_errpat_0_1word_bin_0.1096", 0.1096),
                   ("phat_2_50word_bin_0.2926", 0.2926)):
        r = s - (2 * ph - 1)
        out[nm] = round(float((r * cos * w).sum() / w.sum()), 4)
    # the phat that reproduces the record's -0.10 exactly
    out["phat_reproducing_minus_0.10"] = round(float((1.0 + (v_dep + 0.10)) / 2.0), 4)
    out["identity_note"] = ("with uniform cosines and a uniform-phat neighbourhood, "
                            "v_res = v_dep - (2*phat-1) EXACTLY, i.e. a constant shift")
    return out


# --------------------------------------------------------------------- one dataset
def run_dataset(key, log):
    cfg = DATASETS[key]
    ids, img, txt, lab = PV.load_cache(cfg["cache_dir"], "train", cfg["model"])
    n = len(ids)
    vol = load_volume(cfg["gt"], ids, cfg["vol"])
    dur = load_duration(cfg["dur"], ids)
    lv = np.log1p(vol)
    log(f"[{key}] n={n} pos-rate={lab.mean():.4f} vol({cfg['vol']}) "
        f"min/med/max {vol.min():.0f}/{np.median(vol):.0f}/{vol.max():.0f} "
        f"duration={'yes' if dur is not None else 'NO -> B-c NOT RUN'}")

    anchors = json.load(open(os.path.join(
        REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")))

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(skf.split(np.zeros((n, 1)), lab))

    # ------------------------------------------------ phat: fold-level, space-free
    rng = np.random.RandomState(NULL_SEED)
    phat = {}          # fold -> arm -> vector over fit_idx
    phat_stats = {}
    for f, (fit_idx, ho_idx) in enumerate(folds):
        yf = lab[fit_idx]
        pa = phat_logistic_loo(lv[fit_idx], yf)
        pb, nb, _ = phat_binned_fds_loo(lv[fit_idx], yf)
        d = {"Ba": pa, "Bb": pb}
        if dur is not None:
            d["Bc"] = phat_logistic_loo(
                np.column_stack([lv[fit_idx], np.log(dur[fit_idx])]), yf)
        # N1: permute the covariate across bank items, THEN refit (destroys both the
        #     spread's source and the item correspondence)
        perm1 = rng.permutation(len(fit_idx))
        d["N1"] = phat_logistic_loo(lv[fit_idx][perm1], yf)
        # N2: fit on the TRUE covariate, then permute the assignment (PRESERVES the
        #     marginal spread of phat, destroys only the item correspondence). This
        #     is the sharper null: it asks whether the effect needs the RIGHT nuisance
        #     or merely a spread of summand magnitudes.
        perm2 = rng.permutation(len(fit_idx))
        d["N2"] = pa[perm2]
        phat[f] = d
        st = {}
        for k2, v2 in d.items():
            st[f"sd_{k2}"] = round(float(np.std(v2)), 4)
            st[f"mean_{k2}"] = round(float(np.mean(v2)), 4)
            st[f"min_{k2}"] = round(float(np.min(v2)), 4)
            st[f"max_{k2}"] = round(float(np.max(v2)), 4)
            st[f"auc_{k2}"] = round(float(roc_auc_score(yf, v2)), 4) \
                if yf.min() != yf.max() else None
        st["n_bins_Bb"] = int(nb)
        phat_stats[f] = st
        log(f"  [{key}] fold {f} phat  sd(Ba)={st['sd_Ba']:.4f} auc(Ba)={st['auc_Ba']} "
            f"sd(Bb)={st['sd_Bb']:.4f} auc(Bb)={st['auc_Bb']} "
            f"sd(N1)={st['sd_N1']:.4f} sd(N2)={st['sd_N2']:.4f}")

    arms = [a for a in TREATMENTS if a.split("_")[1] in phat[0]] + \
           ["D1_lenlogit"] + \
           [a for a in ("N1_permcov", "N2_permphat")]

    out = {"n_items": n, "pos_rate": round(float(lab.mean()), 4),
           "vol_mode": cfg["vol"], "has_duration": dur is not None,
           "phat_stats": phat_stats, "arms": arms, "spaces": {}}

    for space in SPACES:
        X = PV.build_space(img, txt, space)
        coll = {"dep": np.full(n, -1, dtype=int), "sc_rank": np.full(n, -1, dtype=int)}
        for a in arms:
            coll[a] = np.full(n, -1, dtype=int)
        per_fold = []
        for f, (fit_idx, ho_idx) in enumerate(folds):
            t0 = time.time()
            fit_idx = np.asarray(fit_idx); ho_idx = np.asarray(ho_idx)
            Xb, yb = X[fit_idx], lab[fit_idx]

            # ---------- FLOOR: the F89-frozen deployed vote on the fitting-fold bank
            dv, dp, dI, dS = M.deployed_vote(Xb, yb, X[ho_idx], topk=TOPK)
            coll["dep"][ho_idx] = dp

            # ---------- IMPLEMENTATION GATE: residual_vote with r = s is the floor
            rv0, rp0, rI0, rS0 = residual_vote(Xb, yb, X[ho_idx], 2.0 * yb - 1.0)
            assert np.array_equal(rp0, dp) and np.allclose(rv0, dv, atol=0, rtol=0) \
                and np.array_equal(rI0, dI), "residual_vote != deployed_vote at r=s"

            # ---------- treatments and nulls: identical retrieval, new summand
            for a in arms:
                if a.startswith("C1_") or a.startswith("N1") or a.startswith("N2"):
                    tag = {"C1_Ba": "Ba", "C1_Bb": "Bb", "C1_Bc": "Bc",
                           "N1_permcov": "N1", "N2_permphat": "N2"}[a]
                    ph = phat[f][tag]
                    r = (2.0 * yb - 1.0) - (2.0 * ph - 1.0)
                    _, pp, _, _ = residual_vote(Xb, yb, X[ho_idx], r)
                    coll[a][ho_idx] = pp

            # ---------- D1 dead relative: needs the bank items' own LOO votes
            vfit, _, _, _ = M.deployed_vote(Xb, yb, Xb, topk=TOPK, exclude_self=True)
            coll["D1_lenlogit"][ho_idx] = d1_length_logit(
                vfit, vol[fit_idx], yb, dv, vol[ho_idx])

            # ---------- pathology rank (F95 definition, full-space cosine)
            S = X[ho_idx] @ Xb.T
            order = np.argsort(-S, axis=1, kind="stable")
            bl = yb[order]
            for r_, q_ in enumerate(ho_idx):
                hit = np.flatnonzero(bl[r_] == lab[q_])
                coll["sc_rank"][q_] = int(hit[0]) + 1 if len(hit) else 10 ** 6

            rec = {"fold": f, "n_fit": int(len(fit_idx)), "n_ho": int(len(ho_idx)),
                   "acc_deployed": round(acc(lab[ho_idx], dp), 4),
                   "secs": round(time.time() - t0, 1)}
            for a in arms:
                rec[f"acc_{a}"] = round(acc(lab[ho_idx], coll[a][ho_idx]), 4)
                rec[f"d_{a}"] = round(rec[f"acc_{a}"] - rec["acc_deployed"], 4)
            per_fold.append(rec)
            log(f"    [{key}/{space}] fold {f} dep {rec['acc_deployed']:.4f}  " +
                "  ".join(f"{a}:{rec['d_' + a]:+.4f}" for a in arms) +
                f"  ({rec['secs']}s)")

        # ---------------------------------------------------------- PARITY GATE
        A = anchors["spaces"][space]
        par = {"anchor_pooled_acc": A["pooled"]["acc_deployed"],
               "anchor_pooled_mF1": A["pooled"]["mF1_deployed"],
               "got_pooled_acc": round(acc(lab, coll["dep"]), 4),
               "got_pooled_mF1": round(M.macro_f1(lab, coll["dep"]), 4),
               "anchor_fold_acc": [r["acc_deployed"] for r in A["per_fold"]],
               "got_fold_acc": [r["acc_deployed"] for r in per_fold],
               "anchor_n_deployed_wrong": A["control3_mechanism"]["n_deployed_wrong"],
               "got_n_deployed_wrong": int((coll["dep"] != lab).sum()),
               "anchor_n_pathology": A["control3_mechanism"]["n_pathology_pop"],
               "got_n_pathology": int(((coll["dep"] != lab) &
                                       (coll["sc_rank"] <= PATHOLOGY_RANK) &
                                       (coll["sc_rank"] > 0)).sum())}
        par["PASS"] = bool(par["got_pooled_acc"] == par["anchor_pooled_acc"]
                           and par["got_pooled_mF1"] == par["anchor_pooled_mF1"]
                           and par["got_fold_acc"] == par["anchor_fold_acc"]
                           and par["got_n_deployed_wrong"] == par["anchor_n_deployed_wrong"]
                           and par["got_n_pathology"] == par["anchor_n_pathology"])
        assert par["PASS"], f"PARITY FAIL {key}/{space}: {par}"
        log(f"  [{key}/{space}] PARITY 4dp PASS  pooled {par['got_pooled_acc']:.4f}/"
            f"{par['got_pooled_mF1']:.4f}  folds {par['got_fold_acc']}")

        # ------------------------------------------------------------- pooled read
        dep_wrong = coll["dep"] != lab
        patho = dep_wrong & (coll["sc_rank"] <= PATHOLOGY_RANK) & (coll["sc_rank"] > 0)
        med_fitvol = {}
        for f, (fit_idx, ho_idx) in enumerate(folds):
            med_fitvol[f] = float(np.median(lv[fit_idx]))
        short = np.zeros(n, dtype=bool)
        for f, (fit_idx, ho_idx) in enumerate(folds):
            short[ho_idx] = lv[ho_idx] <= med_fitvol[f]

        pooled = {"acc_deployed": par["got_pooled_acc"],
                  "mF1_deployed": par["got_pooled_mF1"],
                  "posrate_bank": round(float(lab.mean()), 4),
                  "posrate_deployed": round(float(coll["dep"].mean()), 4),
                  "n_deployed_wrong": int(dep_wrong.sum()),
                  "n_pathology": int(patho.sum()),
                  "n_short": int(short.sum()), "n_long": int((~short).sum())}
        for a in arms:
            p = coll[a]
            assert (p >= 0).all()
            fixed = dep_wrong & (p == lab)
            broke = (~dep_wrong) & (p != lab)
            fs = [r[f"d_{a}"] for r in per_fold]
            e = {"acc": round(acc(lab, p), 4), "mF1": round(M.macro_f1(lab, p), 4)}
            e["d_acc"] = round(e["acc"] - pooled["acc_deployed"], 4)
            e["d_mF1"] = round(e["mF1"] - pooled["mF1_deployed"], 4)
            e["posrate"] = round(float(p.mean()), 4)
            e["fold_deltas"] = fs
            e["fold_signs"] = "".join("+" if x > 0 else ("-" if x < 0 else "0")
                                      for x in fs)
            e["n_folds_ge0"] = int(sum(1 for x in fs if x >= 0))
            e["n_folds_pos"] = int(sum(1 for x in fs if x > 0))
            e["n_changed"] = int((p != coll["dep"]).sum())
            e["n_fixed"] = int(fixed.sum())
            e["n_broken"] = int(broke.sum())
            e["net"] = e["n_fixed"] - e["n_broken"]
            e["exchange_rate"] = (round(e["n_fixed"] / e["n_broken"], 4)
                                  if e["n_broken"] else None)
            e["pathology_fixed"] = int((patho & (p == lab)).sum())
            e["pathology_frac_fixed"] = (round(float((patho & (p == lab)).sum())
                                               / float(patho.sum()), 4)
                                         if patho.sum() else None)
            # effect concentration: ranks 1-5 vs the tail
            near = (coll["sc_rank"] <= PATHOLOGY_RANK) & (coll["sc_rank"] > 0)
            e["changed_rank1to5"] = int(((p != coll["dep"]) & near).sum())
            e["changed_rank6plus"] = int(((p != coll["dep"]) & ~near).sum())
            e["fixed_rank1to5"] = int((fixed & near).sum())
            e["broken_rank1to5"] = int((broke & near).sum())
            e["fixed_rank6plus"] = int((fixed & ~near).sum())
            e["broken_rank6plus"] = int((broke & ~near).sum())
            # stratum honesty (bar 4)
            e["d_acc_short"] = round(acc(lab[short], p[short])
                                     - acc(lab[short], coll["dep"][short]), 4)
            e["d_acc_long"] = round(acc(lab[~short], p[~short])
                                    - acc(lab[~short], coll["dep"][~short]), 4)
            e["posrate_short"] = round(float(p[short].mean()), 4)
            e["posrate_short_deployed"] = round(float(coll["dep"][short].mean()), 4)
            pooled[a] = e
        out["spaces"][space] = {"per_fold": per_fold, "parity": par, "pooled": pooled}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hatemm,zh,en")
    ap.add_argument("--out", default=os.path.join(
        REPO, "scripts/analysis/restrans_pregate_OUT.json"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    import faiss
    faiss.omp_set_num_threads(a.threads)
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", \
        "CPU-only pregate: run with CUDA_VISIBLE_DEVICES=''"

    logf = open(a.out.replace(".json", ".log"), "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    me = os.path.abspath(__file__)
    ops_sha = sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py"))
    assert ops_sha == MECHFIX_OPS_SHA, ("mechfix_ops.py is not the F89-frozen file",
                                        ops_sha)
    OUT = {"meta": {
        "script": me, "script_sha256": sha256_of(me),
        "mechfix_ops_sha256": ops_sha,
        "mechnov_pairverify_sha256": sha256_of(os.path.join(
            REPO, "scripts/analysis/mechnov_pairverify.py")),
        "cpu_only": True, "gpu_jobs": 0, "slurm_jobs": 0, "modal_jobs": 0,
        "training": 0,
        "test_contact": "NONE -- only train_*.pt, data/gt/*/train.jsonl and "
                        "data/gt/HateMM/hate_spans.json are opened",
        "frozen": dict(K_FOLDS=K_FOLDS, FOLD_SEED=FOLD_SEED, TOPK=TOPK,
                       PATHOLOGY_RANK=PATHOLOGY_RANK, SPACES=list(SPACES),
                       PRIMARY_SPACE=PRIMARY_SPACE, BA_LOGIT_C=BA_LOGIT_C,
                       BA_MAXITER=BA_MAXITER, BB_N_BINS=BB_N_BINS,
                       BB_FDS_KS=BB_FDS_KS, BB_FDS_SIGMA=BB_FDS_SIGMA,
                       NULL_SEED=NULL_SEED, D1_STANDARDISE=D1_STANDARDISE),
    }, "worked_example": worked_example(), "datasets": {}}
    log(f"worked example: {json.dumps(OUT['worked_example'])}")

    for k in a.datasets.split(","):
        OUT["datasets"][k] = run_dataset(k, log)
        json.dump(OUT, open(a.out, "w"), indent=1)
    json.dump(OUT, open(a.out, "w"), indent=1)
    log(f"DONE -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
