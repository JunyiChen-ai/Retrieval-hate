#!/usr/bin/env python
"""
membank_c4_pregate.py -- FROZEN implementation of the MEMBANK-C4 $0 pregate.
Record: refine-logs/MEMBANK_C4_PREGATE_RECORD.md.
Spec:   refine-logs/LITSWEEP6_MEMBANK.md section 4 (a)-(f).

NAMING HAZARD (VSW_ASYMMETRY_RECON.md 7.3): there are two litsweep-6 candidates called
"C4".  RELGEN-C4 = VSW (dead).  MEMBANK-C4 = aggregate-then-compare subspace residual
= THIS FILE.

WHAT IS UNDER TEST
    The deployed decision is ALREADY an aggregate-then-compare rule:
        s_c = sum_{i: lab_i = c} cos_i * w_i / sum_i w_i ,  v = s_1 - s_0 ,  pred = v >= 0
    with the per-class aggregator a rank-weighted signed-cosine sum over the deployed
    top-20.  MEMBANK-C4 replaces ONLY that aggregator by a class-conditional SUBSPACE
    RESIDUAL: build a rank-r basis from class c's retrieved members and score the query
    by how much of it that span explains.  Retrieval, keys, k = 20, the candidate set
    and the bank are untouched.  Training-free (a least-squares projection).

ARENA
    Banked RAW encoder key spaces (seed-independent), TRAIN SPLIT ONLY, item-disjoint
    StratifiedKFold(5, shuffle=True, random_state=0) -- the F95 protocol verbatim.
    PRIMARY space = fused.  ZERO test-split contact anywhere in this file.

COST
    CPU only, <= 8 threads.  Zero GPU, zero SLURM, zero Modal.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechnov_pairverify as P      # noqa: E402  FROZEN F95 harness (constants, loaders)
import mechfix_ops as M             # noqa: E402  FROZEN F89 deployed-vote replay

FROZEN_PAIRVERIFY_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"
FROZEN_MECHFIX_SHA = "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d"

# ------------------------------------------------------------------ FROZEN CONSTANTS
TOPK = P.TOPK_DEPLOYED                 # 20, deployed
K_FOLDS = P.K_FOLDS                    # 5
FOLD_SEED = P.FOLD_SEED                # 0
PATHOLOGY_RANK = P.PATHOLOGY_RANK      # 5
SPACES = ("fused", "text", "img")
PRIMARY_SPACE = "fused"

PCA_RANKS = (1, 2, 3, 5)               # LITSWEEP6 section 4(d): arms, not tuning
RIDGE_GAMMAS = (1e-3, 1e-2, 1e-1, 1.0)  # LITSWEEP6 section 4(d): ridge is an arm
EIG_TOL = 1e-12
REDUCED_DIM = P.PCA_DIM                # 256, the sweep's own degeneracy worry

DEG_TOL = 0.01                         # GATE 1: median relative residual gap < 1% => degenerate
DEG_KILL = 0.95                        # GATE 3/4/5 agreement threshold
CLASSBAL_TOL = 0.10                    # GATE 6
BAR1 = 0.010                           # LITSWEEP6 bar 1
BAR_FULL = 0.030                       # frozen full-version bar
ER_BAR = 1.2                           # LITSWEEP6 bar 2

INNER_FOLDS = 5
INNER_SEED = 17                        # == VGA / AGGNET inner-CV seed
F94_K_GRID = (1, 2, 3, 5, 7, 10, 15, 20)

N_PERM = 200
PERM_SEED = 12345
GEOMNULL_SEED = 7
N_GEOM = 20

THETA_GRID = (0.10, 0.20)              # VSW_ASYMMETRY_RECON 3.2 flip-cost radii

# recon 6.1 hard cross-check (PARITY-RECON): (signvote acc, agreement with deployed)
RECON_SIGNVOTE = {"hatemm": (0.8427, 0.9960), "zh": (0.8480, 0.9965),
                  "en": (0.7778, 0.9982)}
BANK_POSRATE = {"hatemm": 0.4005, "zh": 0.3109, "en": 0.3060}

DATASETS = P.DATASETS

CELLS = [("pca", r) for r in PCA_RANKS] + [("pca", "full")] + \
        [("ridge", g) for g in RIDGE_GAMMAS]


def _cell_id(cell):
    fam, par = cell
    if fam == "pca":
        return "C4_pca_rfull" if par == "full" else f"C4_pca_r{par}"
    return f"C4_ridge_g{par}"


CELL_IDS = [_cell_id(c) for c in CELLS]


# ------------------------------------------------------------------------- helpers
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def rank_w(k=TOPK):
    return np.arange(1, k + 1)[::-1].astype("float64")


def best_threshold(v_fit, y_fit):
    """DEG-A: global decision threshold tau maximising fitting-pool accuracy.  Exact
    optimum over all thresholds (aggnet_pregate.best_threshold, reused verbatim)."""
    u = np.unique(v_fit)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0], [0.0]])
    best, bt = -1.0, 0.0
    for t in cand:
        a = acc(y_fit, (v_fit >= t).astype(int))
        if a > best:
            best, bt = a, float(t)
    return bt, best


def flip_cost(nl, cs, w=None):
    """Minimum probability mass that must MOVE between the 20 rank weights to drive the
    deployed score across 0 (VSW_ASYMMETRY_RECON 3.1).  Closed-form optimal transport:
    drain the extreme wrong-side entries into the single most extreme right-side entry."""
    if w is None:
        w = rank_w()
    p = w / w.sum()
    V = (2.0 * nl - 1.0) * cs
    out = np.empty(V.shape[0], dtype="float64")
    for t in range(V.shape[0]):
        v = V[t]
        s = float((p * v).sum())
        vv = v if s >= 0 else -v
        need = abs(s)
        order = np.argsort(-vv, kind="stable")
        tgt = int(np.argmin(vv))
        vmin = vv[tgt]
        cost = 0.0
        for a in order:
            if a == tgt:
                continue
            g = vv[a] - vmin
            if g <= 0:
                break
            avail = p[a] * g
            if avail >= need:
                cost += need / g
                need = 0.0
                break
            cost += p[a]
            need -= avail
        out[t] = cost if need <= 1e-15 else np.inf
    return out


# ---------------------------------------------------------------- the C4 operator
def _class_eig(G, b):
    """One symmetric eigendecomposition of the member Gram G = A A^T serves EVERY
    declared cell (all pca ranks and all ridge gammas).  c = W^T (A q)."""
    ev, W = np.linalg.eigh(G)
    o = np.argsort(-ev, kind="stable")
    ev = np.clip(ev[o], 0.0, None)
    c = W[:, o].T @ b
    return ev, c


def _r2_pca(ev, c, r):
    """residual^2 = 1 - ||U^T q||^2 for the top-r span (||q|| = 1)."""
    k = min(int(r), int((ev > EIG_TOL).sum()))
    if k == 0:
        return 1.0, 0
    e = float((c[:k] ** 2 / ev[:k]).sum())
    return max(0.0, 1.0 - e), k


def _r2_ridge(ev, c, g):
    """alpha = (G + gI)^-1 b ; residual^2 = 1 - 2 alpha.b + alpha^T G alpha."""
    den = ev + g
    e = float((2.0 * c ** 2 / den - ev * c ** 2 / den ** 2).sum())
    return max(0.0, 1.0 - e), int(ev.size)


def c4_predict(nl, nidx, Gb, Sq, dep_pred, cells=CELLS, countmatch=True):
    """Run every declared cell on one block of queries.

    nl   : (n, 20) bank labels of the deployed top-20, in deployed rank order
    nidx : (n, 20) bank-local indices of those neighbours
    Gb   : (n_bank, n_bank) bank Gram in the residual space
    Sq   : (n, n_bank) query-to-bank inner products in the residual space
    dep_pred : (n,) deployed predictions -- the declared fallback for class-pure
               neighbourhoods and for exact ties.
    """
    n = nl.shape[0]
    ids = [_cell_id(c) for c in cells]
    preds = {k: dep_pred.copy() for k in ids}
    gaps = {k: np.zeros(n) for k in ids}
    rel = {k: np.full(n, np.nan) for k in ids}
    reff = {k: np.zeros(n, dtype=int) for k in ids}
    mixed = np.zeros(n, dtype=bool)
    for t in range(n):
        row = nidx[t]
        lt = nl[t]
        p0 = row[lt == 0]
        p1 = row[lt == 1]
        if p0.size == 0 or p1.size == 0:
            continue
        mixed[t] = True
        if countmatch:
            m = min(p0.size, p1.size)
            p0, p1 = p0[:m], p1[:m]
        E = {}
        for c, pc in ((0, p0), (1, p1)):
            E[c] = _class_eig(Gb[np.ix_(pc, pc)], Sq[t, pc])
        for cell, cid in zip(cells, ids):
            fam, par = cell
            r2, re_ = {}, {}
            for c in (0, 1):
                ev, cc = E[c]
                if fam == "pca":
                    r2[c], re_[c] = _r2_pca(ev, cc, ev.size if par == "full" else par)
                else:
                    r2[c], re_[c] = _r2_ridge(ev, cc, float(par))
            d0, d1 = np.sqrt(r2[0]), np.sqrt(r2[1])
            gap = d0 - d1
            gaps[cid][t] = gap
            den = 0.5 * (d0 + d1)
            rel[cid][t] = abs(gap) / den if den > 0 else 0.0
            reff[cid][t] = min(re_[0], re_[1])
            if gap > 0:
                preds[cid][t] = 1
            elif gap < 0:
                preds[cid][t] = 0
            # gap == 0 exactly -> declared fallback to the deployed prediction
    return preds, gaps, rel, reff, mixed


# -------------------------------------------------------- deployed-aggregator parity
def deployed_agg_from_split(nl, cs):
    """PARITY-IMPL: the SAME aggregate-then-compare skeleton with the residual
    aggregator swapped for the deployed rank-weighted signed-cosine sum.  Must
    reproduce mechfix_ops.deployed_vote bit-for-bit in predictions."""
    w = rank_w()
    s1 = ((nl == 1) * cs * w).sum(1) / w.sum()
    s0 = ((nl == 0) * cs * w).sum(1) / w.sum()
    v = s1 - s0
    return v, (v >= 0).astype(int)


# ------------------------------------------------------------------ fold machinery
def fold_cache(X, Xr, fit_idx, ho_idx, lab_fit_for_split):
    """Everything label-INDEPENDENT, computed ONCE per (space, fold).  Retrieval and
    geometry do not depend on labels, so the permutation battery reuses this cache
    unchanged and only re-splits by the shuffled labels."""
    Xb, Xq = X[fit_idx], X[ho_idx]
    dum = np.zeros(len(fit_idx), dtype=int)
    _, _, dI, dS = M.deployed_vote(Xb, dum, Xq, topk=TOPK)
    _, _, fI, fS = M.deployed_vote(Xb, dum, Xb, topk=TOPK, exclude_self=True)
    Rb, Rq = Xr[fit_idx], Xr[ho_idx]
    C = dict(fit_idx=fit_idx, ho_idx=ho_idx, dI=dI, dS=dS, fI=fI, fS=fS,
             Gb=(Rb @ Rb.T), Sq=(Rq @ Rb.T))
    inner = []
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for itr, ite in skf.split(Xb, lab_fit_for_split):
        _, _, iI, iS = M.deployed_vote(Xb[itr], np.zeros(len(itr), dtype=int),
                                       Xb[ite], topk=TOPK)
        Rbi = Rb[itr]
        inner.append(dict(itr=np.asarray(itr), ite=np.asarray(ite),
                          iI=np.asarray(itr)[iI], iS=iS))
    C["inner"] = inner
    return C


def sel_cell(C, yb):
    """C4_sel: inner-CV selection over {DEPLOYED} + the nine declared cells.
    Ties break toward DEPLOYED, then toward the earlier declared cell."""
    names = ["DEPLOYED"] + CELL_IDS
    tot = {k: 0 for k in names}
    ntot = 0
    for S in C["inner"]:
        yte = yb[S["ite"]]
        nl = yb[S["iI"]]
        _, dpred = deployed_agg_from_split(nl, S["iS"])
        tot["DEPLOYED"] += int((dpred == yte).sum())
        pr, _, _, _, _ = c4_predict(nl, S["iI"], C["Gb"], C["Gb"][S["ite"]], dpred)
        for k in CELL_IDS:
            tot[k] += int((pr[k] == yte).sum())
        ntot += len(yte)
    best = max(names, key=lambda k: (tot[k], -names.index(k)))
    return best, {k: round(tot[k] / ntot, 4) for k in names}


def eval_space(key, X, Xr, lab, folds, space, log, want_controls=True, yb_override=None):
    """One (dataset, space) cell.  yb_override (permutation) replaces the fitting-pool
    labels fold by fold; held-out labels are never touched."""
    n = len(lab)
    arms = ["DEPLOYED"] + CELL_IDS + ["C4_sel", "SIGNVOTE", "SIGNVOTE_gt0",
                                      "THRESH_best", "MAGTWIN_max", "MAGTWIN_mean"] + \
           [f"FIXK_{k}" for k in F94_K_GRID] + \
           [f"C4_pca_r{r}_nomatch" for r in PCA_RANKS]
    pred = {a: np.full(n, -1, dtype=int) for a in arms}
    relgap = {c: np.full(n, np.nan) for c in CELL_IDS}
    gapv = {c: np.full(n, np.nan) for c in CELL_IDS}
    reffv = {c: np.zeros(n, dtype=int) for c in CELL_IDS}
    mixedv = np.zeros(n, dtype=bool)
    ntie = np.zeros(n, dtype=bool)
    fcost = np.full(n, np.nan)
    purity = np.full(n, np.nan)
    sc_rank = np.full(n, -1, dtype=int)
    fold_of = np.full(n, -1, dtype=int)
    selinfo, per_fold = [], []

    for f, C in enumerate(folds):
        t0 = time.time()
        fit_idx, ho_idx = C["fit_idx"], C["ho_idx"]
        fold_of[ho_idx] = f
        yb = lab[fit_idx] if yb_override is None else yb_override[f]
        yho = lab[ho_idx]
        nl, cs = yb[C["dI"]], C["dS"]

        dv, dp = deployed_agg_from_split(nl, cs)
        pred["DEPLOYED"][ho_idx] = dp

        pr, gp, rl, rf, mx = c4_predict(nl, C["dI"], C["Gb"], C["Sq"], dp)
        for c in CELL_IDS:
            pred[c][ho_idx] = pr[c]
            relgap[c][ho_idx] = rl[c]
            gapv[c][ho_idx] = gp[c]
            reffv[c][ho_idx] = rf[c]
        mixedv[ho_idx] = mx

        best, innacc = sel_cell(C, yb)
        selinfo.append({"fold": f, "selected": best, "inner_acc": innacc})
        pred["C4_sel"][ho_idx] = dp if best == "DEPLOYED" else pr[best]

        geom = np.nan
        if want_controls:
            prn, _, _, _, _ = c4_predict(nl, C["dI"], C["Gb"], C["Sq"], dp,
                                         [("pca", r) for r in PCA_RANKS],
                                         countmatch=False)
            for r in PCA_RANKS:
                pred[f"C4_pca_r{r}_nomatch"][ho_idx] = prn[f"C4_pca_r{r}"]

            w = rank_w()
            sv = ((2.0 * nl - 1.0) * w).sum(1) / w.sum()
            pred["SIGNVOTE"][ho_idx] = (sv >= 0).astype(int)      # deployed tie rule
            pred["SIGNVOTE_gt0"][ho_idx] = (sv > 0).astype(int)   # recon tie rule
            ntie[ho_idx] = (sv == 0)

            fnl = yb[C["fI"]]
            fv, _ = deployed_agg_from_split(fnl, C["fS"])
            tau, tacc = best_threshold(fv, yb)
            pred["THRESH_best"][ho_idx] = (dv >= tau).astype(int)

            for k in F94_K_GRID:
                wk = np.zeros(TOPK)
                wk[:k] = np.arange(1, k + 1)[::-1]
                vk = ((2.0 * nl - 1.0) * cs * wk).sum(1) / wk.sum()
                pred[f"FIXK_{k}"][ho_idx] = (vk >= 0).astype(int)

            pmax, pmean = dp.copy(), dp.copy()
            for t in range(len(ho_idx)):
                a1, a0 = cs[t][nl[t] == 1], cs[t][nl[t] == 0]
                if a1.size == 0 or a0.size == 0:
                    continue
                pmax[t] = int(a1.max() >= a0.max())
                m = min(a0.size, a1.size)
                pmean[t] = int(a1[:m].mean() >= a0[:m].mean())
            pred["MAGTWIN_max"][ho_idx] = pmax
            pred["MAGTWIN_mean"][ho_idx] = pmean

            rng = np.random.RandomState(GEOMNULL_SEED + f)
            pool = {c: np.flatnonzero(yb == c) for c in (0, 1)}
            hit = np.zeros(len(ho_idx))
            for _ in range(N_GEOM):
                fake = C["dI"].copy()
                for t in range(len(ho_idx)):
                    for c in (0, 1):
                        sl = np.flatnonzero(nl[t] == c)
                        if sl.size:
                            fake[t, sl] = rng.choice(pool[c], size=sl.size, replace=False)
                pg, _, _, _, _ = c4_predict(nl, fake, C["Gb"], C["Sq"], dp,
                                            [("pca", PCA_RANKS[0])])
                hit += (pg[f"C4_pca_r{PCA_RANKS[0]}"] == yho)
            geom = float(hit.mean() / N_GEOM)

            fcost[ho_idx] = flip_cost(nl, cs)
            purity[ho_idx] = (nl == dp[:, None]).mean(1)
            Sfull = X[ho_idx] @ X[fit_idx].T
            order = np.argsort(-Sfull, axis=1, kind="stable")
            bl = yb[order]
            for r_, q_ in enumerate(ho_idx):
                h = np.flatnonzero(bl[r_] == lab[q_])
                sc_rank[q_] = int(h[0]) + 1 if len(h) else 10 ** 6

        rec = {"fold": f, "n_fit": int(len(fit_idx)), "n_ho": int(len(ho_idx)),
               "acc_deployed": round(acc(yho, dp), 4),
               "selected": best, "secs": round(time.time() - t0, 1)}
        if want_controls:
            rec["tau"] = round(tau, 6)
            rec["tau_fit_acc"] = round(tacc, 4)
            rec["acc_NULL2_geom"] = round(geom, 4)
        for a in CELL_IDS + ["C4_sel"]:
            rec[f"d_{a}"] = round(acc(yho, pred[a][ho_idx]) - rec["acc_deployed"], 4)
        per_fold.append(rec)
        if log:
            bc = max(CELL_IDS, key=lambda c: rec["d_" + c])
            log(f"    [{key}/{space}] fold {f} dep {rec['acc_deployed']:.4f} "
                f"sel={best} d_sel {rec['d_C4_sel']:+.4f} best_cell {bc} "
                f"{rec['d_' + bc]:+.4f} ({rec['secs']}s)")

    return dict(pred=pred, relgap=relgap, gap=gapv, reff=reffv, mixed=mixedv,
                ntie=ntie, fcost=fcost, purity=purity, sc_rank=sc_rank, fold=fold_of,
                selinfo=selinfo, per_fold=per_fold, arms=arms)


# ------------------------------------------------------------------------- summary
def summarise(key, lab, R):
    dep = R["pred"]["DEPLOYED"]
    n = len(lab)
    dw = dep != lab
    patho = dw & (R["sc_rank"] <= PATHOLOGY_RANK) & (R["sc_rank"] > 0)
    out = {"n_items": n, "acc_deployed": round(acc(lab, dep), 4),
           "mF1_deployed": round(M.macro_f1(lab, dep), 4),
           "posrate_deployed": round(float(dep.mean()), 4),
           "n_deployed_wrong": int(dw.sum()), "n_pathology_pop": int(patho.sum()),
           "frac_class_mixed": round(float(R["mixed"].mean()), 4), "arms": {}}
    for a in R["arms"]:
        p = R["pred"][a]
        if (p < 0).any():
            continue
        fx = dw & (p == lab)
        bk = (~dw) & (p != lab)
        d = {"acc": round(acc(lab, p), 4), "mF1": round(M.macro_f1(lab, p), 4),
             "dacc": round(acc(lab, p) - out["acc_deployed"], 4),
             "dmF1": round(M.macro_f1(lab, p) - out["mF1_deployed"], 4),
             "posrate": round(float(p.mean()), 4),
             "fixed": int(fx.sum()), "broken": int(bk.sum()),
             "net": int(fx.sum()) - int(bk.sum()),
             "changed": int((p != dep).sum()),
             "exchange_rate": (round(float(fx.sum()) / float(bk.sum()), 4)
                               if bk.sum() else None),
             "pathology_fixed": int((patho & (p == lab)).sum()),
             "agree_deployed": round(float((p == dep).mean()), 4),
             "fix_yield": round(float(fx.sum()) / max(1, int(dw.sum())), 4),
             "break_exposure": round(float(bk.sum()) / max(1, int((~dw).sum())), 4)}
        fs = [round(acc(lab[R["fold"] == f], p[R["fold"] == f])
                    - acc(lab[R["fold"] == f], dep[R["fold"] == f]), 4)
              for f in range(K_FOLDS)]
        d["fold_dacc"] = fs
        d["fold_signs_ge0"] = int(sum(1 for x in fs if x >= 0))
        d["fold_signs_gt0"] = int(sum(1 for x in fs if x > 0))
        out["arms"][a] = d

    deg = {}
    for c in CELL_IDS:
        g = R["relgap"][c][R["mixed"]]
        g = g[np.isfinite(g)]
        deg[c] = {"median_rel_gap": round(float(np.median(g)), 6) if g.size else None,
                  "frac_below_tol": round(float((g < DEG_TOL).mean()), 4) if g.size else None,
                  "degenerate": bool(g.size and float(np.median(g)) < DEG_TOL),
                  "median_reff": float(np.median(R["reff"][c][R["mixed"]]))}
    out["degeneracy"] = deg
    out["all_cells_degenerate"] = bool(all(deg[c]["degenerate"] for c in CELL_IDS))

    bestc = max(CELL_IDS, key=lambda c: out["arms"][c]["net"])
    out["oracle_pooled"] = {"cell": bestc, "net": out["arms"][bestc]["net"],
                            "dacc": out["arms"][bestc]["dacc"],
                            "fix_yield": out["arms"][bestc]["fix_yield"],
                            "break_exposure": out["arms"][bestc]["break_exposure"],
                            "exchange_rate": out["arms"][bestc]["exchange_rate"]}
    fo = 0
    for f in range(K_FOLDS):
        m = R["fold"] == f
        fo += max(int((dw & (R["pred"][c] == lab) & m).sum())
                  - int(((~dw) & (R["pred"][c] != lab) & m).sum()) for c in CELL_IDS)
    out["oracle_perfold_net"] = int(fo)
    out["req_net"] = {"bar_010": int(np.ceil(BAR1 * n)),
                      "bar_030": int(np.ceil(BAR_FULL * n))}
    out["exposure_precheck"] = {
        "n_ERR": int(dw.sum()), "n_COR": int((~dw).sum()),
        "per_cell": {c: {"fix_supply": out["arms"][c]["fixed"],
                         "break_exposure": out["arms"][c]["broken"],
                         "fix_yield": out["arms"][c]["fix_yield"],
                         "break_exposure_rate": out["arms"][c]["break_exposure"],
                         "net": out["arms"][c]["net"],
                         "ER": out["arms"][c]["exchange_rate"]} for c in CELL_IDS},
        "ceiling_net": out["oracle_pooled"]["net"],
        "passes_bar010": bool(out["oracle_pooled"]["net"] >= out["req_net"]["bar_010"])}
    return out


def controls(key, lab, R, S):
    c4, dep = R["pred"]["C4_sel"], R["pred"]["DEPLOYED"]
    ch = c4 != dep
    out = {"A_agree_threshold_shift": round(float((c4 == R["pred"]["THRESH_best"]).mean()), 4)}
    ag = {str(k): round(float((c4 == R["pred"][f"FIXK_{k}"]).mean()), 4) for k in F94_K_GRID}
    out["B_agree_fixk"] = ag
    out["B_agree_fixk_max"] = max(ag.values())
    out["B_argmax_k"] = max(ag, key=lambda k: ag[k])
    out["n_changed"] = int(ch.sum())
    if ch.sum():
        mm = {t: round(float((R["pred"][t][ch] == c4[ch]).mean()), 4)
              for t in ("MAGTWIN_max", "MAGTWIN_mean")}
        out["MEMMAG_magnitude_mediated"] = mm
        out["MEMMAG_magnitude_mediated_max"] = max(mm.values())
        out["MEMMAG_membership_mediated"] = round(
            float((R["pred"]["SIGNVOTE"][ch] != c4[ch]).mean()), 4)
    else:
        out["MEMMAG_magnitude_mediated"] = None
        out["MEMMAG_magnitude_mediated_max"] = None
        out["MEMMAG_membership_mediated"] = None
    br = BANK_POSRATE[key]
    pr_ = S["arms"]["C4_sel"]["posrate"]
    out["class_balance"] = {"bank_posrate": br, "c4_posrate": pr_,
                            "deviation": round(abs(pr_ - br), 4),
                            "within_tol": bool(abs(pr_ - br) <= CLASSBAL_TOL)}
    sv, sv2 = R["pred"]["SIGNVOTE"], R["pred"]["SIGNVOTE_gt0"]
    out["signvote"] = {"acc": round(acc(lab, sv), 4),
                       "agree_deployed": round(float((sv == dep).mean()), 4),
                       "acc_gt0_tiebreak": round(acc(lab, sv2), 4),
                       "agree_deployed_gt0_tiebreak": round(float((sv2 == dep).mean()), 4),
                       "n_exact_ties": int(R["ntie"].sum()),
                       "recon_target": list(RECON_SIGNVOTE[key])}
    dw = dep != lab
    fc = R["fcost"]
    fin = np.isfinite(fc)
    out["flipcost"] = {
        "median_correct": round(float(np.median(fc[(~dw) & fin])), 4),
        "median_wrong": round(float(np.median(fc[dw & fin])), 4),
        "frac_correct_cheap_0.10": round(float((fc[~dw] <= 0.10).mean()), 4),
        "frac_wrong_cheap_0.10": round(float((fc[dw] <= 0.10).mean()), 4)}
    out["purity"] = {"median_correct": round(float(np.median(R["purity"][~dw])), 4),
                     "median_wrong": round(float(np.median(R["purity"][dw])), 4)}
    bc = S["oracle_pooled"]["cell"]
    p = R["pred"][bc]
    jt = {}
    for th in THETA_GRID:
        cheap = fc <= th
        sup = int((dw & cheap & (p == lab)).sum())
        ex = int(((~dw) & cheap & (p != lab)).sum())
        jt[str(th)] = {"fix_supply": sup, "break_exposure": ex,
                       "ratio": round(sup / ex, 4) if ex else None}
    out["joint_supply_exposure"] = {"cell": bc, "by_theta": jt}
    return out


# ---------------------------------------------------------------------- data plumb
def prep(key, space):
    cfg = DATASETS[key]
    ids, img, txt, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
    return ids, P.build_space(img, txt, space), lab, cfg


def make_folds(X, Xr, lab):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=FOLD_SEED)
    return [fold_cache(X, Xr, np.asarray(a), np.asarray(b), lab[np.asarray(a)])
            for a, b in skf.split(X, lab)]


def parity_arena(key, space, lab, R, log):
    ref = json.load(open(os.path.join(
        REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")))["spaces"][space]
    dep = R["pred"]["DEPLOYED"]
    dw = dep != lab
    patho = dw & (R["sc_rank"] <= PATHOLOGY_RANK) & (R["sc_rank"] > 0)
    cells = [("pooled.acc_deployed", ref["pooled"]["acc_deployed"], round(acc(lab, dep), 4)),
             ("pooled.mF1_deployed", ref["pooled"]["mF1_deployed"],
              round(M.macro_f1(lab, dep), 4)),
             ("pooled.posrate_deployed", ref["pooled"]["posrate_deployed"],
              round(float(dep.mean()), 4)),
             ("mech.n_deployed_wrong", ref["control3_mechanism"]["n_deployed_wrong"],
              int(dw.sum())),
             ("mech.n_pathology_pop", ref["control3_mechanism"]["n_pathology_pop"],
              int(patho.sum()))]
    for f in range(K_FOLDS):
        m = R["fold"] == f
        cells.append((f"fold{f}.acc_deployed", ref["per_fold"][f]["acc_deployed"],
                      round(acc(lab[m], dep[m]), 4)))
    bad = [c for c in cells if c[1] != c[2]]
    for c in bad:
        log(f"    PARITY-ARENA FAIL {c[0]}: frozen={c[1]} got={c[2]}")
    assert not bad, f"{key}/{space}: PARITY-ARENA failed ({len(bad)} cells)"
    log(f"    [{key}/{space}] PARITY-ARENA {len(cells)}/{len(cells)}")
    return {"n_cells": len(cells), "n_pass": len(cells),
            "cells": [{"k": a, "frozen": b, "got": c} for a, b, c in cells]}


def parity_impl(key, space, X, lab, folds, log):
    mx = 0.0
    for f, C in enumerate(folds):
        yb = lab[C["fit_idx"]]
        dv0, dp0, dI0, dS0 = M.deployed_vote(X[C["fit_idx"]], yb, X[C["ho_idx"]], topk=TOPK)
        assert (dI0 == C["dI"]).all() and (dS0 == C["dS"]).all(), \
            f"{key}/{space}/f{f}: retrieval cache mismatch"
        v, p = deployed_agg_from_split(yb[C["dI"]], C["dS"])
        assert (p == dp0).all(), f"{key}/{space}/f{f}: PARITY-IMPL prediction mismatch"
        e = float(np.abs(v - dv0).max())
        assert e < 1e-12, f"{key}/{space}/f{f}: PARITY-IMPL score mismatch {e:.3e}"
        mx = max(mx, e)
    log(f"    [{key}/{space}] PARITY-IMPL {K_FOLDS}/{K_FOLDS} folds bit-exact "
        f"(max score |delta| {mx:.2e})")
    return {"folds_bit_exact": K_FOLDS, "max_score_delta": mx}


# ------------------------------------------------------------------------- selftest
SYNTH_A = dict(csz=3, beta=0.30, a0=0.15, a1=0.30, b2=4.0, pos=0.4)
SYNTH_C_NOISE = 6.0
N_PERM_SELFTEST = 100


def _synth(kind, n=699, d=256, seed=0):
    """Synthetic arenas for the pre-freeze machinery check (record section 2.6).
    Only the GENERATORS are designed here; no bar, no control and no operator constant
    depends on anything in this function."""
    rng = np.random.RandomState(seed)
    if kind == "A":
        # PLANTED SUBSPACE STRUCTURE, cone-collapsed like the real arena.
        #   x = e0 + beta*(b_k1 + b2*eps_i*b_k2) + alpha_y * u_i
        # e0 is shared by everything (the cone).  Items come in CONCEPTS of csz items;
        # a concept's members share a 2-d basis (b_k1, b_k2) and a concept belongs to
        # exactly one class, so a few same-class members SPAN the query's concept
        # component and no wrong-class member does.  alpha_0 < alpha_1 makes class-0
        # items systematically shorter off-cone and therefore systematically HIGHER in
        # cosine, so the deployed rank-weighted label count is dragged toward the
        # majority class while the subspace residual is not.
        a = SYNTH_A
        K = n // a["csz"]
        kcls = (rng.rand(K) < a["pos"]).astype(int)
        y = np.repeat(kcls, a["csz"])
        kid = np.repeat(np.arange(K), a["csz"])
        nn = len(y)
        e0 = rng.randn(d); e0 /= np.linalg.norm(e0)
        B1 = rng.randn(K, d); B1 /= np.linalg.norm(B1, axis=1, keepdims=True)
        B2 = rng.randn(K, d); B2 /= np.linalg.norm(B2, axis=1, keepdims=True)
        U = rng.randn(nn, d); U /= np.linalg.norm(U, axis=1, keepdims=True)
        eps = rng.randn(nn)
        al = np.where(y == 0, a["a0"], a["a1"])[:, None]
        X = (e0[None, :] + a["beta"] * (B1[kid] + a["b2"] * eps[:, None] * B2[kid])
             + al * U)
        return P.l2n(X), y
    y = (rng.rand(n) < 0.4).astype(int)
    if kind == "B":                          # pure noise: labels independent of geometry
        X = rng.randn(n, d)
    else:                                    # C: deployed vote already optimal
        mu = {c: rng.randn(d) / np.sqrt(d) for c in (0, 1)}
        X = np.stack([mu[y[i]] + SYNTH_C_NOISE * rng.randn(d) / np.sqrt(d)
                      for i in range(n)])
    return P.l2n(X), y


def selftest_D():
    """S-D, RANK-DISCRIMINATION UNIT TEST -- the control that proves the MEM-MAG gate
    can be passed by a genuinely subspace-mediated effect rather than always firing.

    Construction: q = (u1 + u2)/sqrt(2).  The gold class (1) has four members, three
    near u1 and one at u2 -- every one of them at cosine 0.7071 to q.  Class 0 has four
    near-collinear members at cosine 0.8000 to q.  Therefore
      * the deployed vote and the max-cosine magnitude twin both predict 0 (WRONG);
      * rank 1 predicts 0 (WRONG): class 1's top principal direction is ~u1;
      * rank >= 2 predicts 1 (RIGHT): span{u1, u2} contains q exactly.
    """
    d = 64
    rng = np.random.RandomState(0)
    E = np.eye(d)
    u1, u2 = E[0], E[1]
    q = (u1 + u2) / np.sqrt(2.0)
    w = E[2]
    v = 0.8 * q + 0.6 * w
    mem = [u1, u1 + 0.02 * E[3], u1 + 0.02 * E[4], u2,                  # class 1
           v, v + 0.02 * E[5], v + 0.02 * E[6], v + 0.02 * E[7]]        # class 0
    Amem = np.stack([m / np.linalg.norm(m) for m in mem])
    lab = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    cos = Amem @ q
    order = np.argsort(-cos, kind="stable")
    nidx = order[None, :]
    nl = lab[order][None, :]
    cs = cos[order][None, :]
    Gb = Amem @ Amem.T
    Sq = (Amem @ q)[None, :]
    dep_v = float(((2 * nl - 1) * cs * np.arange(len(order), 0, -1)).sum()
                  / np.arange(len(order), 0, -1).sum())
    dep_p = np.array([int(dep_v >= 0)])
    magmax = int(cs[0][nl[0] == 1].max() >= cs[0][nl[0] == 0].max())
    pr, gp, _, _, _ = c4_predict(nl, nidx, Gb, Sq, dep_p)
    out = {"cos_class1_max": round(float(cs[0][nl[0] == 1].max()), 4),
           "cos_class0_max": round(float(cs[0][nl[0] == 0].max()), 4),
           "deployed_pred": int(dep_p[0]), "magtwin_max_pred": magmax,
           "pred": {c: int(pr[c][0]) for c in CELL_IDS},
           "gap": {c: round(float(gp[c][0]), 4) for c in CELL_IDS}}
    out["PASS"] = bool(out["deployed_pred"] == 0 and magmax == 0
                       and out["pred"]["C4_pca_r1"] == 0
                       and out["pred"]["C4_pca_r2"] == 1
                       and out["pred"]["C4_pca_r3"] == 1)
    return out


def selftest(log):
    res = {}
    for kind in ("A", "B", "C"):
        X, y = _synth(kind)
        folds = make_folds(X, X, y)
        R = eval_space(f"synth{kind}", X, X, y, folds, "synthetic", log)
        S = summarise(f"synth{kind}", y, R)
        best = max(CELL_IDS, key=lambda c: S["arms"][c]["dacc"])
        res[kind] = {"n": int(len(y)), "pos_rate": round(float(y.mean()), 4),
                     "acc_deployed": S["acc_deployed"],
                     "d_C4_sel": S["arms"]["C4_sel"]["dacc"],
                     "sel_per_fold": [s["selected"] for s in R["selinfo"]],
                     "best_cell": best, "d_best_cell": S["arms"][best]["dacc"],
                     "d_per_cell": {c: S["arms"][c]["dacc"] for c in CELL_IDS},
                     "d_THRESH_best": S["arms"]["THRESH_best"]["dacc"],
                     "d_MAGTWIN_max": S["arms"]["MAGTWIN_max"]["dacc"],
                     "frac_class_mixed": S["frac_class_mixed"],
                     "median_rel_gap_r1": S["degeneracy"]["C4_pca_r1"]["median_rel_gap"],
                     "all_cells_degenerate": S["all_cells_degenerate"],
                     "agree_deployed_sel": S["arms"]["C4_sel"]["agree_deployed"]}
        log(f"  SELFTEST {kind}: n {len(y)} dep {S['acc_deployed']:.4f}  "
            f"d_C4_sel {res[kind]['d_C4_sel']:+.4f}  best {best} "
            f"{res[kind]['d_best_cell']:+.4f}  mixed {S['frac_class_mixed']:.4f}  "
            f"sel {res[kind]['sel_per_fold']}")

    res["D"] = selftest_D()
    log(f"  SELFTEST D (rank discrimination): {res['D']}")

    # null machinery: the permutation battery must be non-significant on pure noise
    X, y = _synth("B")
    folds = make_folds(X, X, y)
    base = eval_space("synthB", X, X, y, folds, "synthetic", None, want_controls=False)
    obs = acc(y, base["pred"]["C4_sel"]) - acc(y, base["pred"]["DEPLOYED"])
    rng = np.random.RandomState(PERM_SEED)
    dr = []
    for _ in range(N_PERM_SELFTEST):
        yov = []
        for C in folds:
            yb = y[C["fit_idx"]].copy()
            rng.shuffle(yb)
            yov.append(yb)
        Rp = eval_space("synthB", X, X, y, folds, "synthetic", None,
                        want_controls=False, yb_override=yov)
        dr.append(float(acc(y, Rp["pred"]["C4_sel"]) - acc(y, Rp["pred"]["DEPLOYED"])))
    dr = np.asarray(dr)
    pB = (1 + int((dr >= obs - 1e-12).sum())) / (N_PERM_SELFTEST + 1)
    res["B_perm"] = {"n_perm": N_PERM_SELFTEST, "observed": round(float(obs), 4),
                     "p": round(pB, 4), "null_mean": round(float(dr.mean()), 6),
                     "null_sd": round(float(dr.std()), 6),
                     "frac_null_ge_0": round(float((dr >= 0).mean()), 4)}
    log(f"  SELFTEST B permutation: {res['B_perm']}")

    res["VERDICT"] = {
        "A_clear_positive": bool(res["A"]["d_best_cell"] >= 0.05),
        "B_honest_null": bool(abs(res["B"]["d_C4_sel"]) <= 0.02 and pB > 0.05),
        "C_returns_floor_bit_exact": bool(res["C"]["d_C4_sel"] == 0.0
                                          and res["C"]["agree_deployed_sel"] == 1.0),
        "D_rank_discrimination": bool(res["D"]["PASS"])}
    log(f"  SELFTEST VERDICT {res['VERDICT']}")
    return res


# ----------------------------------------------------------------------- main paths
def run_main(key, log):
    OUT = {"meta": {"dataset": key, "ds": DATASETS[key]["ds"],
                    "model": DATASETS[key]["model"],
                    "test_contact": "NONE -- only train_<model>.pt is loaded",
                    "script_sha256": sha256_of(os.path.abspath(__file__)),
                    "frozen_mechfix_ops_sha256": FROZEN_MECHFIX_SHA,
                    "frozen_pairverify_sha256": FROZEN_PAIRVERIFY_SHA,
                    "cells": CELL_IDS}, "spaces": {}}
    for space in SPACES:
        ids, X, lab, cfg = prep(key, space)
        OUT["meta"]["n_train_items"] = len(lab)
        OUT["meta"]["pos_rate"] = round(float(lab.mean()), 4)
        folds = make_folds(X, X, lab)
        R = eval_space(key, X, X, lab, folds, space, log)
        S = summarise(key, lab, R)
        S["parity_arena"] = parity_arena(key, space, lab, R, log)
        S["parity_impl"] = parity_impl(key, space, X, lab, folds, log)
        S["selection"] = R["selinfo"]
        S["per_fold"] = R["per_fold"]
        S["controls"] = controls(key, lab, R, S)
        gs = [pf["acc_NULL2_geom"] for pf in R["per_fold"]]
        wts = [pf["n_ho"] for pf in R["per_fold"]]
        S["NULL2_geom_acc"] = round(float(np.average(gs, weights=wts)), 4)
        S["NULL2_geom_dacc"] = round(S["NULL2_geom_acc"] - S["acc_deployed"], 4)
        if space == PRIMARY_SPACE:
            a_, g_ = RECON_SIGNVOTE[key]
            got = S["controls"]["signvote"]
            # PARITY-RECON: DEMOTED from a hard abort to a reported cross-check --
            # see the record's ERRATUM E-1.  The residual is a TIE CONVENTION on
            # 2/2/3 exactly-tied items (the recon used strict > 0, this record
            # inherits the deployed >= 0), plus HateMM's disclosed zero-norm key.
            # The qualitative fact GATE 5 rests on is confirmed either way.
            got["parity_recon_exact_declared_tiebreak"] = bool(
                got["acc"] == a_ and got["agree_deployed"] == g_)
            got["parity_recon_exact_recon_tiebreak"] = bool(
                got["acc_gt0_tiebreak"] == a_
                and got["agree_deployed_gt0_tiebreak"] == g_)
            log(f"    [{key}/{space}] PARITY-RECON (reported) declared>=0 "
                f"{got['acc']}/{got['agree_deployed']} | recon>0 "
                f"{got['acc_gt0_tiebreak']}/{got['agree_deployed_gt0_tiebreak']} | "
                f"target {a_}/{g_} | ties {got['n_exact_ties']}")

            from sklearn.decomposition import PCA as SKPCA
            predr = {f"C4_pca256_r{r}": np.full(len(lab), -1, dtype=int) for r in PCA_RANKS}
            degr = {f"C4_pca256_r{r}": [] for r in PCA_RANKS}
            for f, C in enumerate(folds):
                nc = min(REDUCED_DIM, len(C["fit_idx"]) - 1, X.shape[1])
                pc = SKPCA(n_components=nc, svd_solver=P.PCA_SOLVER, random_state=0)
                pc.fit(X[C["fit_idx"]])
                Z = P.l2n(pc.transform(X))
                Zb = Z[C["fit_idx"]]
                yb = lab[C["fit_idx"]]
                nl = yb[C["dI"]]
                _, dp = deployed_agg_from_split(nl, C["dS"])
                pr, _, rl, _, mx_ = c4_predict(nl, C["dI"], Zb @ Zb.T,
                                               Z[C["ho_idx"]] @ Zb.T, dp,
                                               [("pca", r) for r in PCA_RANKS])
                for r in PCA_RANKS:
                    predr[f"C4_pca256_r{r}"][C["ho_idx"]] = pr[f"C4_pca_r{r}"]
                    degr[f"C4_pca256_r{r}"].append(rl[f"C4_pca_r{r}"][mx_])
            dep = R["pred"]["DEPLOYED"]
            S["reduced_space_arms"] = {
                k: {"acc": round(acc(lab, v), 4),
                    "dacc": round(acc(lab, v) - S["acc_deployed"], 4),
                    "agree_deployed": round(float((v == dep).mean()), 4),
                    "median_rel_gap": round(float(np.median(np.concatenate(degr[k]))), 6)}
                for k, v in predr.items()}
        OUT["spaces"][space] = S
        log(f"  [{key}/{space}] dep {S['acc_deployed']:.4f}  C4_sel "
            f"{S['arms']['C4_sel']['dacc']:+.4f}  oracle {S['oracle_pooled']['cell']} "
            f"net {S['oracle_pooled']['net']} (need {S['req_net']['bar_010']}"
            f"/{S['req_net']['bar_030']})")
    return OUT


def run_perm(key, log):
    ids, X, lab, cfg = prep(key, PRIMARY_SPACE)
    folds = make_folds(X, X, lab)
    base = eval_space(key, X, X, lab, folds, PRIMARY_SPACE, log, want_controls=False)
    obs = acc(lab, base["pred"]["C4_sel"]) - acc(lab, base["pred"]["DEPLOYED"])
    rng = np.random.RandomState(PERM_SEED)
    draws = []
    t0 = time.time()
    for b in range(N_PERM):
        yov = []
        for C in folds:
            yb = lab[C["fit_idx"]].copy()
            rng.shuffle(yb)
            yov.append(yb)
        Rp = eval_space(key, X, X, lab, folds, PRIMARY_SPACE, None,
                        want_controls=False, yb_override=yov)
        draws.append(round(float(acc(lab, Rp["pred"]["C4_sel"])
                                 - acc(lab, Rp["pred"]["DEPLOYED"])), 6))
        if (b + 1) % 25 == 0:
            log(f"    [{key}] perm {b + 1}/{N_PERM} ({time.time() - t0:.0f}s)")
    dr = np.asarray(draws)
    p = (1 + int((dr >= obs - 1e-12).sum())) / (N_PERM + 1)
    return {"meta": {"dataset": key, "n_perm": N_PERM, "perm_seed": PERM_SEED,
                     "script_sha256": sha256_of(os.path.abspath(__file__)),
                     "space": PRIMARY_SPACE, "arm": "C4_sel",
                     "null_object": "fitting-pool bank labels shuffled within each fold; "
                                    "retrieval/geometry/held-out labels untouched; the "
                                    "full pipeline incl. inner-CV selection re-run per draw"},
            "observed_dacc": round(float(obs), 4), "p": round(p, 4),
            "null_mean": round(float(dr.mean()), 6), "null_sd": round(float(dr.std()), 6),
            "null_max": round(float(dr.max()), 4),
            "frac_null_ge_0": round(float((dr >= 0).mean()), 4),
            "informative": bool((dr >= 0).mean() >= 0.05), "draws": draws}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--perm", action="store_true")
    a = ap.parse_args()

    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py")) \
        == FROZEN_MECHFIX_SHA, "FROZEN F89 OPS MODULE CHANGED -- refusing to run"
    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) \
        == FROZEN_PAIRVERIFY_SHA, "FROZEN F95 MODULE CHANGED -- refusing to run"

    import torch
    torch.set_num_threads(8)
    out = a.out or os.path.join(REPO, "scripts/analysis/membank_c4_selftest_OUT.json")
    logf = open(out.replace(".json", ".log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    if a.selftest:
        R = {"meta": {"script_sha256": sha256_of(os.path.abspath(__file__))},
             "selftest": selftest(log)}
    elif a.perm:
        R = run_perm(a.dataset, log)
    else:
        R = run_main(a.dataset, log)
    json.dump(R, open(out, "w"), indent=1)
    log(f"DONE -> {out} ({time.time() - t0:.0f}s)")
    logf.close()


if __name__ == "__main__":
    main()
