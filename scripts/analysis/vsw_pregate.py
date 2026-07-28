#!/usr/bin/env python
"""
vsw_pregate.py -- FROZEN implementation of the VSW ($0) pregate.
Record: refine-logs/VSW_PREGATE_RECORD.md.  Spec: refine-logs/LITSWEEP6_RELGEN.md §2 C4.

THE IDEA UNDER TEST (LITSWEEP6_RELGEN §2 C4, verbatim)
    "Keep the deployed rank-weighted sum over the top-20 exactly as it is, and multiply
    each neighbour's rank weight by a monotone function of its verifier score, with an
    interpolation coefficient lambda such that lambda=0 reproduces the deployed vote
    bit-exactly."

        v(lam) = SUM_i (2*lab_i - 1) * cos_i * w_i * m_i(lam) / SUM_i w_i * m_i(lam)
        m_i(0) == 1     w = [20,19,...,1]     predict 1 iff v >= 0

    m_i is a monotone non-decreasing function of p_i, the FROZEN F95 pair verifier's
    P(same-class) score for (query, i-th deployed neighbour).

WHY IT IS RUN (this is a DOOR-CLOSER, not a goal bet)
    The sweep record prices P(clear K-VSW-1 on >=2 datasets) at ~2 % and recommends it
    "as analysis, never as a lever".  Its value is K-VSW-2, "a diagnostic that cannot
    fail": the exchange rate as a function of aggregation sharpness, which turns the
    paper's law-I datum from the two-point read F95 carries (max / mean-top-3) into a
    full curve.

WHAT IS FROZEN
    Every constant below was fixed before any real-data treatment number was computed.
    The F95 arms module and the F89 ops module are IMPORTED UNMODIFIED and their sha256
    are asserted at run time.  There is no tuning, no early stopping, no post-hoc arm
    and no post-hoc multiplier family.

REAP RESILIENCE (process boundaries only -- changes NO arm)
    The login node SIGTERMs sustained non-SLURM CPU processes (F95 §3 lost a run this
    way; LSMI_GATE §2.7 documents the same).  Every unit of work here -- one fold's
    arena, one (fold x permutation draw) verifier fit, one draw's evaluation -- is
    serialised to <ckpt>/ the moment it completes, via a tmp-file + os.replace so a
    reap cannot leave a half-written file.  A re-run skips completed units.  This is
    the mechnov_pairverify_runner.py / LSMI per-draw checkpoint precedent: same arms,
    same constants, same seeds, same fold assignment, same permutation draw sequence --
    only the process boundary differs.  Every draw's RNG is seeded from (PERM_SEED,
    draw, fold), so the draw sequence is identical to an uninterrupted run.

ARENA / COST
    Banked RAW fused encoder key space, TRAIN SPLIT ONLY, item-disjoint 5-fold LOO --
    the F95 protocol verbatim.  ZERO test contact, ZERO GPU, ZERO SLURM, ZERO Modal,
    CPU <= 8 threads.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechnov_pairverify as P      # noqa: E402  FROZEN F95 arms -- imported, never edited
import mechfix_ops as M             # noqa: E402  FROZEN F89 deployed-vote replay

FROZEN_PAIRVERIFY_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"
FROZEN_MECHFIX_SHA = "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d"

# ------------------------------------------------------------- FROZEN VSW CONSTANTS
SPACE = "fused"              # F95 PRIMARY space
VERIFIER = "mlp"             # F95 PRIMARY model (its logistic arm fired F95 control 4)
INNER_FOLDS = 5              # lambda selection, inner CV inside the fitting pool
INNER_SEED = 17              # VGA §2.4 / F98 §1.4 constant
N_PERM = 200                 # permutation null budget (VGA constant)
PERM_SEED = 12345            # VGA constant
N_PERM_SELFTEST = 60         # self-test only; p-resolution 1/61 (VGA self-test precedent)
DEG_KILL = 0.95              # DEG-A / DEG-B agreement threshold (F98 constant)
CLASSBAL_TOL = 0.10          # class-balance tolerance vs the bank rate (F95 control 4)
MIN_CHANGED_FOR_RATE = 20    # K-VSW-2 outcome (a) guard: >=20 changed decisions
P_FLOOR = 1e-12              # numerical floor on verifier scores / cosines

F94_K_GRID = (1, 2, 3, 5, 7, 10, 15, 20)     # DEG-B: the F94 fixed-k profiles

# multiplier families.  m_i(0) == 1.0 EXACTLY in every family -- that identity is what
# makes the PARITY-lambda0 gate an exact-equality assert rather than a tolerance.
#   pow : (p_i / max_j p_j)^lam            PRIMARY -- spans deployed (lam=0) -> verifier
#                                          argmax (lam=inf), i.e. the whole sharpness
#                                          continuum, and contains the spec's lam in [0,1]
#   exp : exp(lam * (p_i - max_j p_j))     SECONDARY -- monotone in p, not in log p.
#                                          Written against the row max rather than the row
#                                          mean purely for overflow safety; v's SIGN is
#                                          invariant to a positive global rescaling of the
#                                          weights, so the two forms give identical decisions.
#   lin : (1-lam) + lam*(p_i / mean_j p_j) SECONDARY -- the literal "interpolation
#                                          coefficient" reading, lam in [0,1]
LAM_POW = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
           48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 1024.0, 4096.0, np.inf)
LAM_EXP = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
LAM_LIN = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
FAMILIES = {"pow": LAM_POW, "exp": LAM_EXP, "lin": LAM_LIN}
PRIMARY_FAMILY = "pow"


# --------------------------------------------------------------------------- helpers
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def atomic_savez(path, **kw):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def rank_weights(k=P.TOPK_DEPLOYED):
    """w = [20, 19, ..., 1] -- mechfix_ops._rank_weights, float64."""
    return np.arange(1, k + 1)[::-1].astype("float64")


def vote_with_weights(S, C, G):
    """v = SUM_i s_i cos_i g_i / SUM_i g_i ; predict 1 iff v >= 0.

    With G = tile([20..1]) the elementwise product order and the float64 dtype are
    identical to mechfix_ops.deployed_vote and SUM(G_row) == w.sum() == 210.0 exactly,
    so this is bit-for-bit the deployed vote.  Asserted every fold (PARITY-lambda0).
    """
    num = (S * C * G).sum(1)
    den = G.sum(1)
    v = num / den
    return v, (v >= 0).astype(int)


def multiplier(fam, lam, Pm):
    """m_i(lam), monotone non-decreasing in Pm; m_i(0) == 1.0 exactly."""
    if fam == "pow":
        if not np.isfinite(lam):
            mx = Pm.max(1, keepdims=True)
            return (Pm >= mx).astype("float64")
        return np.exp(lam * (np.log(Pm) - np.log(Pm.max(1, keepdims=True))))
    if fam == "exp":
        return np.exp(lam * (Pm - Pm.max(1, keepdims=True)))
    if fam == "lin":
        return (1.0 - lam) + lam * (Pm / Pm.mean(1, keepdims=True))
    raise ValueError(fam)


def decide(fam, lam, Sm, Cm, Pm, w):
    G = w[None, :] * multiplier(fam, lam, Pm)
    return vote_with_weights(Sm, Cm, G)


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def best_threshold(v_fit, y_fit):
    """DEG-A: the global decision threshold tau maximising fitting-pool accuracy.
    Candidates are midpoints between consecutive distinct fit votes plus the deployed
    tau=0 and both open ends -- the exact optimum over all thresholds (F98 §5.3)."""
    u = np.unique(v_fit)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0], [0.0]])
    best, bt = -1.0, 0.0
    for t in cand:
        a = acc(y_fit, (v_fit >= t).astype(int))
        if a > best:
            best, bt = a, float(t)
    return bt, best


def foldsigns(deltas):
    return "".join("+" if v > 1e-12 else ("-" if v < -1e-12 else "0") for v in deltas)


# ---------------------------------------------------------------------- the emitter
def _fold_setup(X, lab, fit_idx, fold):
    """The frozen F95 per-fold verifier inputs: PCA on FITTING-fold items only, the
    seeded pair subsample, the standardised fit matrix.  Every constant frozen."""
    ncomp = min(P.PCA_DIM, len(fit_idx) - 1, X.shape[1])
    pca = PCA(n_components=ncomp, svd_solver=P.PCA_SOLVER, random_state=0)
    pca.fit(X[fit_idx])
    Zn = P.l2n(pca.transform(X))
    evr = float(pca.explained_variance_ratio_.sum())
    rng = np.random.RandomState(P.PAIR_SUBSAMPLE_SEED + fold)
    pi, pj, tot_pairs = P.all_unordered_pairs(fit_idx, rng, P.PAIR_FIT_CAP)
    Phi = P.pair_features(Zn, pi, pj)
    mu = Phi.mean(0)
    sd = Phi.std(0)
    sd[sd == 0] = 1.0
    Phi -= mu
    Phi /= sd
    return Zn, evr, ncomp, pi, pj, tot_pairs, Phi, mu, sd


def emit_arena(X, lab, ck, log, perm_draws=None, tag=""):
    """Replay the frozen F95 fused primary cell fold by fold and emit, per TRAIN item:
    its deployed top-20 (bank indices, cosines, neighbour labels), the deployed vote and
    decision, the 20 FROZEN verifier scores on those neighbours, plus the F95 parity
    quantities (cosine-shape control, mlp_max, mlp_mean3, same-class-analogue rank).

    perm_draws: for each draw the FITTING-fold ITEM LABELS are permuted and the verifier
    is refitted at the identical frozen budget; the bank labels used in the vote, the
    retrieval, the cosines and every gold label are untouched.  One file per
    (fold, draw); a reap costs at most one fit.
    """
    os.makedirs(ck, exist_ok=True)
    n = X.shape[0]
    K = P.TOPK_DEPLOYED
    skf = StratifiedKFold(n_splits=P.K_FOLDS, shuffle=True, random_state=P.FOLD_SEED)
    splits = list(skf.split(X, lab))

    for fold, (fit_idx, ho_idx) in enumerate(splits):
        fit_idx = np.asarray(fit_idx)
        ho_idx = np.asarray(ho_idx)
        af = os.path.join(ck, f"f{fold}.npz")
        need_arena = not os.path.exists(af)
        need_draws = ([d for d in perm_draws
                       if not os.path.exists(os.path.join(ck, f"f{fold}_d{d}.npy"))]
                      if perm_draws else [])
        if not need_arena and not need_draws:
            continue

        t0 = time.time()
        Zn, evr, ncomp, pi, pj, tot_pairs, Phi_fit, mu, sd = _fold_setup(
            X, lab, fit_idx, fold)

        dep_v, dep_p, dep_I, dep_sim = M.deployed_vote(
            X[fit_idx], lab[fit_idx], X[ho_idx], topk=K)
        qq_d = np.repeat(ho_idx, K)
        bb_d = fit_idx[dep_I].ravel()
        Phi_d = P.pair_features(Zn, qq_d, bb_d)
        Phi_d -= mu
        Phi_d /= sd
        log(f"    {tag}fold {fold} setup {time.time() - t0:.1f}s "
            f"(arena={need_arena} draws={len(need_draws)})")

        if need_arena:
            t1 = time.time()
            S = X[ho_idx] @ X[fit_idx].T
            cls_pos = {c: np.flatnonzero(lab[fit_idx] == c) for c in (0, 1)}
            for c in (0, 1):
                assert len(cls_pos[c]) >= P.M_PER_CLASS, (c, len(cls_pos[c]))
            nom = {}
            for c in (0, 1):
                sub = S[:, cls_pos[c]]
                top = np.argsort(-sub, axis=1, kind="stable")[:, :P.M_PER_CLASS]
                nom[c] = cls_pos[c][top]
            cs0 = np.take_along_axis(S, nom[0], axis=1).max(1)
            cs1 = np.take_along_axis(S, nom[1], axis=1).max(1)
            cos_shape = (cs1 >= cs0).astype(int)
            nom_all = np.concatenate([nom[0], nom[1]], axis=1)
            Phi_n = P.pair_features(Zn, np.repeat(ho_idx, nom_all.shape[1]),
                                    fit_idx[nom_all].ravel())
            Phi_n -= mu
            Phi_n /= sd
            mdl = P.fit_mlp(Phi_fit, (lab[pi] == lab[pj]).astype(int))
            nbr_p = P.predict_mlp(mdl, Phi_d).reshape(len(ho_idx), K)
            scn = P.predict_mlp(mdl, Phi_n).reshape(len(ho_idx), nom_all.shape[1])
            del mdl, Phi_n
            v0, v1 = scn[:, :P.M_PER_CLASS], scn[:, P.M_PER_CLASS:]
            adj_max = (v1.max(1) >= v0.max(1)).astype(int)
            m0 = (-np.sort(-v0, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
            m1 = (-np.sort(-v1, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
            adj_mean3 = (m1 >= m0).astype(int)
            order = np.argsort(-S, axis=1, kind="stable")
            bl = lab[fit_idx][order]
            scr = np.empty(len(ho_idx), dtype=int)
            for r, q in enumerate(ho_idx):
                hit = np.flatnonzero(bl[r] == lab[q])
                scr[r] = int(hit[0]) + 1 if len(hit) else 10 ** 6
            atomic_savez(af, ho_idx=ho_idx, dep_v=dep_v, dep_p=dep_p,
                         nbr_cos=dep_sim, nbr_lab=lab[fit_idx][dep_I].astype("float64"),
                         nbr_glb=fit_idx[dep_I], nbr_p=nbr_p, cos_shape=cos_shape,
                         adj_max=adj_max, adj_mean3=adj_mean3, sc_rank=scr,
                         meta=np.array([fold, len(fit_idx), len(ho_idx), ncomp,
                                        round(evr, 6), tot_pairs, len(pi),
                                        round(time.time() - t1, 1)], dtype="float64"))
            log(f"    {tag}fold {fold} ARENA done {time.time() - t1:.1f}s "
                f"dep {acc(lab[ho_idx], dep_p):.4f}")

        if need_draws:
            loc = np.full(n, -1, dtype=int)
            loc[fit_idx] = np.arange(len(fit_idx))
            pil, pjl = loc[pi], loc[pj]
            assert (pil >= 0).all() and (pjl >= 0).all()
            for d in need_draws:
                t1 = time.time()
                r2 = np.random.RandomState(PERM_SEED + 1000 * int(d) + fold)
                lab_perm = lab[fit_idx][r2.permutation(len(fit_idx))]
                yp = (lab_perm[pil] == lab_perm[pjl]).astype(int)
                mp = P.fit_mlp(Phi_fit, yp)
                sc = P.predict_mlp(mp, Phi_d).reshape(len(ho_idx), K)
                del mp
                tmp = os.path.join(ck, f".tmp_f{fold}_d{d}.npy")
                np.save(tmp, sc)
                os.replace(tmp, os.path.join(ck, f"f{fold}_d{d}.npy"))
                log(f"    {tag}fold {fold} draw {d} {time.time() - t1:.1f}s")
        del Phi_fit, Phi_d, Zn

    # ---- assemble from the checkpoints
    A = {"n": n, "lab": lab, "fold": np.full(n, -1, dtype=int),
         "nbr_cos": np.full((n, K), np.nan), "nbr_lab": np.full((n, K), np.nan),
         "nbr_glb": np.full((n, K), -1, dtype=int), "nbr_p": np.full((n, K), np.nan),
         "dep_vote": np.full(n, np.nan), "dep_pred": np.full(n, -1, dtype=int),
         "cos_shape_pred": np.full(n, -1, dtype=int),
         "adj_max_pred": np.full(n, -1, dtype=int),
         "adj_mean3_pred": np.full(n, -1, dtype=int),
         "sc_rank": np.full(n, -1, dtype=int), "per_fold": []}
    for fold in range(P.K_FOLDS):
        z = np.load(os.path.join(ck, f"f{fold}.npz"))
        h = z["ho_idx"]
        A["fold"][h] = fold
        A["nbr_cos"][h] = z["nbr_cos"]
        A["nbr_lab"][h] = z["nbr_lab"]
        A["nbr_glb"][h] = z["nbr_glb"]
        A["nbr_p"][h] = z["nbr_p"]
        A["dep_vote"][h] = z["dep_v"]
        A["dep_pred"][h] = z["dep_p"]
        A["cos_shape_pred"][h] = z["cos_shape"]
        A["adj_max_pred"][h] = z["adj_max"]
        A["adj_mean3_pred"][h] = z["adj_mean3"]
        A["sc_rank"][h] = z["sc_rank"]
        m = z["meta"]
        A["per_fold"].append({"fold": int(m[0]), "n_fit_items": int(m[1]),
                              "n_ho_items": int(m[2]), "pca_dim": int(m[3]),
                              "pca_explained_var": round(float(m[4]), 4),
                              "n_pairs_total": int(m[5]), "n_pairs_fitted": int(m[6]),
                              "secs": float(m[7])})
    assert (A["dep_pred"] >= 0).all() and (A["fold"] >= 0).all()
    assert np.isfinite(A["nbr_p"]).all() and np.isfinite(A["nbr_cos"]).all()
    return A


def load_perm_table(ck, A, d):
    """Assemble one permutation draw's n x 20 verifier score table from its per-fold
    checkpoints.  Missing file -> None (draw not yet computed)."""
    Pn = np.full((A["n"], P.TOPK_DEPLOYED), np.nan)
    for fold in range(P.K_FOLDS):
        f = os.path.join(ck, f"f{fold}_d{d}.npy")
        if not os.path.exists(f):
            return None
        h = np.load(os.path.join(ck, f"f{fold}.npz"))["ho_idx"]
        Pn[h] = np.load(f)
    assert np.isfinite(Pn).all()
    return Pn


# --------------------------------------------------------------------------- gates
def parity_lambda0(A, log, tag=""):
    """PARITY-lambda0 (hard assert).  The VSW vote engine at lambda = 0 must reproduce
    mechfix_ops.deployed_vote BIT-EXACTLY, every family, every fold."""
    w = rank_weights()
    Sm = 2.0 * A["nbr_lab"] - 1.0
    Cm = A["nbr_cos"]
    Pm = np.clip(A["nbr_p"], P_FLOOR, 1.0)
    gates = []
    for fam in FAMILIES:
        v, p = decide(fam, 0.0, Sm, Cm, Pm, w)
        for f in range(P.K_FOLDS):
            m = A["fold"] == f
            gates.append((f"{fam}_fold{f}_bitexact",
                          bool(np.array_equal(v[m], A["dep_vote"][m]) and
                               np.array_equal(p[m], A["dep_pred"][m]))))
        gates.append((f"{fam}_pooled_acc_4dp",
                      round(acc(A["lab"], p), 4) == round(acc(A["lab"], A["dep_pred"]), 4)))
    npass = sum(1 for _, ok in gates if ok)
    for k, ok in gates:
        if not ok:
            log(f"    PARITY-lambda0 FAIL {k}")
    log(f"  {tag}PARITY-lambda0 {npass}/{len(gates)}")
    assert npass == len(gates), f"PARITY-lambda0 failed ({npass}/{len(gates)})"
    return {"n_gates": len(gates), "n_pass": npass,
            "gates": [{"key": k, "pass": ok} for k, ok in gates]}


# The 26 F95 quantities VGA asserted 78/78 on, split by WHAT THEY DEPEND ON.
# INVARIANT = closed-form (retrieval, the deployed vote, the cosine-shape control, the
# ERRPAT rank statistics) -- these reproduce the RECORDED F95 numbers bit-exactly.
# TRAINED   = quantities that depend on the torch-fitted MLP verifier -- these are
# measured NOT to be bit-reproducible across sessions (record §4.1), so they are asserted
# against a FRESH SAME-SESSION re-run of the frozen F95 module itself (--stage anchor).
INVARIANT_POOLED = ("acc_deployed", "mF1_deployed", "posrate_deployed",
                    "acc_cos_shape", "mF1_cos_shape", "posrate_cos_shape")
INVARIANT_MECH = ("n_deployed_wrong", "n_pathology_pop", "median_sc_rank_all",
                  "median_sc_rank_deployed_wrong")
TRAINED_POOLED = ("acc_mlp_max", "mF1_mlp_max", "posrate_mlp_max",
                  "acc_mlp_mean3", "mF1_mlp_mean3", "posrate_mlp_mean3")
TRAINED_MECH = tuple(f"mlp_{ag}_{s}" for ag in ("max", "mean3")
                     for s in ("fixed", "broke", "net", "exchange_rate",
                               "pathology_fixed"))


def parity_f95(key, A, log):
    """F95 PARITY (hard assert), TWO TIERS, 26 quantities per dataset at 4 dp.

    TIER 1 -- the 10 INVARIANT quantities against the RECORDED F95 cell
    (`mechnov_pairverify_{ds}_OUT.json`).  This is the claim "the regenerated arena IS
    the F95 arena".

    TIER 2 -- the 16 TRAINED quantities against `vsw_f95anchor_{ds}_OUT.json`, produced
    by `--stage anchor`, which calls the FROZEN F95 module's own `run_space` UNMODIFIED
    in this session.  This is the claim "scoring only the nominated 20 pairs reproduces
    the frozen module's full-eval-matrix scoring exactly".

    The recorded-vs-anchor difference on the TRAINED quantities is REPORTED as a measured
    drift table, never asserted away.
    """
    ref = json.load(open(os.path.join(
        REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")))
    po = ref["spaces"][SPACE]["pooled"]
    me = ref["spaces"][SPACE]["control3_mechanism"]
    apath = os.path.join(REPO, f"scripts/analysis/vsw_f95anchor_{key}_OUT.json")
    assert os.path.exists(apath), (
        f"missing tier-2 anchor {apath}; run --stage anchor --dataset {key} first")
    anc = json.load(open(apath))
    assert anc["meta"]["frozen_pairverify_sha256"] == FROZEN_PAIRVERIFY_SHA
    apo = anc["run_space"]["pooled"]
    ame = anc["run_space"]["control3_mechanism"]
    lab = A["lab"]
    got = {
        "acc_deployed": round(acc(lab, A["dep_pred"]), 4),
        "mF1_deployed": round(M.macro_f1(lab, A["dep_pred"]), 4),
        "posrate_deployed": round(float(A["dep_pred"].mean()), 4),
        "acc_cos_shape": round(acc(lab, A["cos_shape_pred"]), 4),
        "mF1_cos_shape": round(M.macro_f1(lab, A["cos_shape_pred"]), 4),
        "posrate_cos_shape": round(float(A["cos_shape_pred"].mean()), 4),
        "acc_mlp_max": round(acc(lab, A["adj_max_pred"]), 4),
        "mF1_mlp_max": round(M.macro_f1(lab, A["adj_max_pred"]), 4),
        "posrate_mlp_max": round(float(A["adj_max_pred"].mean()), 4),
        "acc_mlp_mean3": round(acc(lab, A["adj_mean3_pred"]), 4),
        "mF1_mlp_mean3": round(M.macro_f1(lab, A["adj_mean3_pred"]), 4),
        "posrate_mlp_mean3": round(float(A["adj_mean3_pred"].mean()), 4),
    }
    dw = A["dep_pred"] != lab
    patho = dw & (A["sc_rank"] <= P.PATHOLOGY_RANK) & (A["sc_rank"] > 0)
    gotm = {"n_deployed_wrong": int(dw.sum()), "n_pathology_pop": int(patho.sum()),
            "median_sc_rank_all": float(np.median(A["sc_rank"])),
            "median_sc_rank_deployed_wrong": float(np.median(A["sc_rank"][dw]))}
    for ag, pk in (("max", "adj_max_pred"), ("mean3", "adj_mean3_pred")):
        p = A[pk]
        fx = dw & (p == lab)
        bk = (~dw) & (p != lab)
        gotm[f"mlp_{ag}_fixed"] = int(fx.sum())
        gotm[f"mlp_{ag}_broke"] = int(bk.sum())
        gotm[f"mlp_{ag}_net"] = int(fx.sum()) - int(bk.sum())
        gotm[f"mlp_{ag}_exchange_rate"] = round(float(fx.sum()) / float(bk.sum()), 4)
        gotm[f"mlp_{ag}_pathology_fixed"] = int((patho & (p == lab)).sum())

    gates, drift = [], []
    for k in INVARIANT_POOLED:
        gates.append(("T1", k, po[k], got[k], bool(po[k] == got[k])))
    for k in INVARIANT_MECH:
        gates.append(("T1", k, me[k], gotm[k], bool(me[k] == gotm[k])))
    for k in TRAINED_POOLED:
        gates.append(("T2", k, apo[k], got[k], bool(apo[k] == got[k])))
        drift.append({"key": k, "f95_recorded": po[k], "f95_anchor_this_session": apo[k],
                      "same": bool(po[k] == apo[k])})
    for k in TRAINED_MECH:
        gates.append(("T2", k, ame[k], gotm[k], bool(ame[k] == gotm[k])))
        drift.append({"key": k, "f95_recorded": me[k], "f95_anchor_this_session": ame[k],
                      "same": bool(me[k] == ame[k])})
    npass = sum(1 for g in gates if g[4])
    for t, k, e, o, ok in gates:
        if not ok:
            log(f"    F95-PARITY {t} FAIL {k}: reference={e} emitted={o}")
    n1 = sum(1 for g in gates if g[0] == "T1")
    p1 = sum(1 for g in gates if g[0] == "T1" and g[4])
    n2 = len(gates) - n1
    p2 = npass - p1
    nd = sum(1 for d in drift if not d["same"])
    log(f"  [{key}] F95-PARITY {npass}/{len(gates)}  (TIER-1 vs recorded {p1}/{n1}, "
        f"TIER-2 vs same-session anchor {p2}/{n2});  recorded-vs-anchor DRIFT on "
        f"{nd}/{len(drift)} trained quantities")
    assert npass == len(gates), f"{key}: F95 parity failed ({npass}/{len(gates)})"
    return {"n_gates": len(gates), "n_pass": npass,
            "tier1": {"n": n1, "pass": p1}, "tier2": {"n": n2, "pass": p2},
            "n_trained_quantities_drifted": nd, "n_trained_quantities": len(drift),
            "drift_recorded_vs_anchor": drift,
            "gates": [{"tier": t, "key": k, "reference": e, "emitted": o, "pass": ok}
                      for t, k, e, o, ok in gates]}


# ------------------------------------------------------------------- the VSW battery
def mech(lab, dep_pred, pred):
    dw = dep_pred != lab
    fx = int((dw & (pred == lab)).sum())
    bk = int(((~dw) & (pred != lab)).sum())
    return {"fixed": fx, "broke": bk, "net": fx - bk,
            "exchange_rate": (round(fx / bk, 4) if bk else None),
            "changed": int((pred != dep_pred).sum())}


def curve(A, Pm, w, fam):
    """K-VSW-2: exchange rate and Delta-acc as a function of aggregation sharpness.
    FIXED lambda, no selection -- a property of the operator, not of a selector."""
    lab, dp, fold = A["lab"], A["dep_pred"], A["fold"]
    Sm = 2.0 * A["nbr_lab"] - 1.0
    Cm = A["nbr_cos"]
    a_dep = acc(lab, dp)
    fdep = [acc(lab[fold == f], dp[fold == f]) for f in range(P.K_FOLDS)]
    rows = []
    for lam in FAMILIES[fam]:
        _, p = decide(fam, lam, Sm, Cm, Pm, w)
        d = [acc(lab[fold == f], p[fold == f]) - fdep[f] for f in range(P.K_FOLDS)]
        r = {"lam": (None if not np.isfinite(lam) else float(lam)),
             "lam_is_inf": bool(not np.isfinite(lam)),
             "acc": round(acc(lab, p), 4), "dacc": round(acc(lab, p) - a_dep, 4),
             "mF1": round(M.macro_f1(lab, p), 4), "posrate": round(float(p.mean()), 4),
             "foldsigns": foldsigns(d), "folddeltas": [round(x, 4) for x in d]}
        r.update(mech(lab, dp, p))
        rows.append(r)
    return rows


def select_lambda(fam, A, Pm, w, fit_items):
    """lambda chosen on INNER folds inside the fitting pool; ties -> smallest lambda
    (toward the deployed rule).  Never sees the evaluated fold."""
    lab = A["lab"]
    Sm = 2.0 * A["nbr_lab"] - 1.0
    Cm = A["nbr_cos"]
    grid = FAMILIES[fam]
    tot = np.zeros(len(grid))
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for _, iho in inner.split(np.zeros(len(fit_items)), lab[fit_items]):
        idx = fit_items[iho]
        for li, lam in enumerate(grid):
            _, p = decide(fam, lam, Sm[idx], Cm[idx], Pm[idx], w)
            tot[li] += acc(lab[idx], p)
    tot /= INNER_FOLDS
    b = int(np.argmax(tot))          # grid ascending, argmax -> smallest lam on ties
    return grid[b], tot


def selected_arm(fam, A, Pm, w):
    lab, fold, n = A["lab"], A["fold"], A["n"]
    Sm = 2.0 * A["nbr_lab"] - 1.0
    Cm = A["nbr_cos"]
    pred = np.full(n, -1, dtype=int)
    lams = []
    for f in range(P.K_FOLDS):
        fit_items = np.flatnonzero(fold != f)
        ho = np.flatnonzero(fold == f)
        lam, ia = select_lambda(fam, A, Pm, w, fit_items)
        lams.append({"fold": f, "lam": (None if not np.isfinite(lam) else float(lam)),
                     "lam_is_inf": bool(not np.isfinite(lam)),
                     "inner_acc_max": round(float(ia.max()), 4),
                     "inner_acc_at_lam0": round(float(ia[0]), 4)})
        _, p = decide(fam, lam, Sm[ho], Cm[ho], Pm[ho], w)
        pred[ho] = p
    assert (pred >= 0).all()
    return pred, lams


def summarise(A, pred, extra=None):
    lab, dp, fold = A["lab"], A["dep_pred"], A["fold"]
    a_dep = acc(lab, dp)
    fdep = [acc(lab[fold == f], dp[fold == f]) for f in range(P.K_FOLDS)]
    d = [acc(lab[fold == f], pred[fold == f]) - fdep[f] for f in range(P.K_FOLDS)]
    e = {"acc": round(acc(lab, pred), 4), "dacc": round(acc(lab, pred) - a_dep, 4),
         "mF1": round(M.macro_f1(lab, pred), 4), "posrate": round(float(pred.mean()), 4),
         "foldsigns": foldsigns(d), "folddeltas": [round(x, 4) for x in d]}
    e.update(mech(lab, dp, pred))
    if extra:
        e.update(extra)
    return e


def run_arms(A, Pm=None, perm_mode=False):
    """All declared arms + the K-VSW-2 curves + DEG-A/B/D.  perm_mode=True computes only
    the quantities the permutation null consumes (the label-independent controls -- the
    threshold twin, the fixed-k twin and the oracle -- do not depend on the verifier and
    are identical in every draw)."""
    lab, dp, fold, n = A["lab"], A["dep_pred"], A["fold"], A["n"]
    w = rank_weights()
    if Pm is None:
        Pm = np.clip(A["nbr_p"], P_FLOOR, 1.0)
    Cm, Sm = A["nbr_cos"], 2.0 * A["nbr_lab"] - 1.0
    R = {"acc_deployed": round(acc(lab, dp), 4),
         "mF1_deployed": round(M.macro_f1(lab, dp), 4),
         "posrate_deployed": round(float(dp.mean()), 4),
         "posrate_bank": round(float(lab.mean()), 4),
         "fold_acc_deployed": [round(acc(lab[fold == f], dp[fold == f]), 4)
                               for f in range(P.K_FOLDS)],
         "arms": {}}
    R["curve"] = {fam: curve(A, Pm, w, fam) for fam in FAMILIES}
    sel = {}
    for fam in FAMILIES:
        pred, lams = selected_arm(fam, A, Pm, w)
        sel[fam] = pred
        R["arms"][f"VSW_{fam}"] = summarise(A, pred, {"lambda_per_fold": lams})
    if perm_mode:
        return R

    # ---- DEG-D: identical machinery driven by the COSINE instead of the verifier
    Cclip = np.clip(Cm, P_FLOOR, None)
    R["curve_ctrl_cos"] = {PRIMARY_FAMILY: curve(A, Cclip, w, PRIMARY_FAMILY)}
    pred, lams = selected_arm(PRIMARY_FAMILY, A, Cclip, w)
    sel["ctrl_cos"] = pred
    R["arms"]["CTRL_cos_pow"] = summarise(A, pred, {"lambda_per_fold": lams})

    # ---- ORACLE ceiling: best lambda chosen on the HELD-OUT fold (report only)
    pred = np.full(n, -1, dtype=int)
    for f in range(P.K_FOLDS):
        ho = np.flatnonzero(fold == f)
        best, bp = -1.0, None
        for lam in FAMILIES[PRIMARY_FAMILY]:
            _, p = decide(PRIMARY_FAMILY, lam, Sm[ho], Cm[ho], Pm[ho], w)
            a = acc(lab[ho], p)
            if a > best:
                best, bp = a, p
        pred[ho] = bp
    R["arms"]["ORACLE_lambda_pow"] = summarise(A, pred)

    # ---- DEG-A: global threshold shift fitted on the fitting pool
    tp = np.full(n, -1, dtype=int)
    taus = []
    for f in range(P.K_FOLDS):
        fit_items = np.flatnonzero(fold != f)
        ho = np.flatnonzero(fold == f)
        tau, ta = best_threshold(A["dep_vote"][fit_items], lab[fit_items])
        taus.append({"fold": f, "tau": round(tau, 6), "fit_acc": round(ta, 4)})
        tp[ho] = (A["dep_vote"][ho] >= tau).astype(int)
    R["arms"]["THRESH_best"] = summarise(A, tp, {"tau_per_fold": taus})

    # ---- DEG-B: the eight F94 fixed-k profiles over the identical deployed top-20
    fixk = {}
    for k in F94_K_GRID:
        wk = np.zeros(P.TOPK_DEPLOYED)
        wk[:k] = np.arange(1, k + 1)[::-1]
        _, p = vote_with_weights(Sm, Cm, np.tile(wk, (n, 1)))
        fixk[k] = p
        R["arms"][f"FIXK_{k}"] = summarise(A, p)

    prim = sel[PRIMARY_FAMILY]
    R["degeneracy"] = {
        "A_agree_threshold_shift": round(float((prim == tp).mean()), 4),
        "B_agree_fixk": {str(k): round(float((prim == fixk[k]).mean()), 4)
                         for k in F94_K_GRID},
        "B_agree_fixk_max": round(max(float((prim == fixk[k]).mean())
                                      for k in F94_K_GRID), 4),
        "B_argmax_k": int(max(F94_K_GRID,
                              key=lambda k: float((prim == fixk[k]).mean()))),
        "D_agree_ctrl_cos": round(float((prim == sel["ctrl_cos"]).mean()), 4),
        "agree_deployed": round(float((prim == dp).mean()), 4), "DEG_KILL": DEG_KILL}
    R["degeneracy"]["A_FIRES"] = bool(
        R["degeneracy"]["A_agree_threshold_shift"] >= DEG_KILL)
    R["degeneracy"]["B_FIRES"] = bool(R["degeneracy"]["B_agree_fixk_max"] >= DEG_KILL)
    R["class_balance"] = {
        "bank_posrate": round(float(lab.mean()), 4), "tol": CLASSBAL_TOL,
        "arm_posrate": {k: R["arms"][k]["posrate"] for k in R["arms"]},
        "primary_deviation": round(
            abs(R["arms"][f"VSW_{PRIMARY_FAMILY}"]["posrate"] - float(lab.mean())), 4)}
    R["class_balance"]["PRIMARY_PASS"] = bool(
        R["class_balance"]["primary_deviation"] <= CLASSBAL_TOL)
    return R


# ------------------------------------------------------------------------- selftest
def synth(arm, seed=0, n=350, d_noise=100, n_factor=5, fac_sd=10.0, rate=0.4, sig=1.5):
    """SYNTHETIC ONLY (the F95 §2.5 construction shape).

    Geometry, identical in both arms.  `n_factor` shared nuisance factors of standard
    deviation `fac_sd` DOMINATE the cosine: their contribution to the inner product has
    sd sqrt(n_factor)*fac_sd^2 ~ 224, against a planted class term of only sig^2 = 2.25,
    so the deployed top-20 retrieval is essentially class-blind (the tail-enrichment
    ratio is exp(2*sig^2*t/sigma^2) ~ 1.03).  The class signal lives in ONE NOISELESS
    coordinate whose variance (2.25) still exceeds the isotropic noise coordinates' (1),
    so the PCA keeps it as its own component and a relation function on
    `[|z-z'|, z*z']` can read its SIGNED PRODUCT -- which separates same-class from
    cross-class pairs almost perfectly.

    Arm A -- the planted coordinate carries the TRUE label, so verifier re-weighting of
    the deployed top-20 should repair the vote.  A harness that cannot show a positive
    here cannot show one anywhere.

    Arms B0 and B -- NOTHING LEARNABLE.  The planted coordinate carries a
    label-INDEPENDENT grouping and the labels are drawn independently of every feature,
    so the verifier's fit target `1[lab_i == lab_j]` is unlearnable.  They differ only in
    the class rate, and that difference is the point:

      B0 (rate 0.4, IMBALANCED) is the F98 §2.5 arm-B lesson made concrete.  A
      re-weighting arm's function class contains a drift toward the neighbourhood LABEL
      RATIO -- the verifier memorises its FITTING-fold items, so p(q, bank_j) drifts
      toward P(class of q == lab_j) ~ the class PRIOR, which upweights majority-class
      neighbours.  Where the deployed vote sits BELOW the majority rate that drift buys
      free accuracy with nothing learnable present, and the permutation null does NOT
      absorb it (shuffling the fit targets destroys the correspondence with the real bank
      labels the vote consumes).  B0 therefore returns a SPURIOUS positive with a
      significant p -- and the FROZEN CLASS-BALANCE control catches it, because the drift
      is exactly a collapse away from the bank rate.  B0 exists to demonstrate that.

      B (rate 0.5, BALANCED) removes the prior asymmetry the drift feeds on, so it is the
      arm on which an honest null is REQUIRED.
    """
    rng = np.random.RandomState(seed)
    lab = (rng.rand(n) < rate).astype(int)
    F = fac_sd * rng.randn(n, n_factor)             # dominant nuisance -> drives cosine
    N = rng.randn(n, d_noise)
    c = sig * (2 * lab - 1)                         # noiseless class coordinate
    if arm != "A":
        r = rate if arm == "B0" else 0.5
        c = sig * (2 * (rng.rand(n) < r).astype(int) - 1)      # unrelated grouping
        lab = (rng.rand(n) < r).astype(int)                    # labels independent of X
    return P.l2n(np.concatenate([c[:, None], F, N], axis=1).astype("float64")), lab


def check_multipliers(log):
    pp = np.sort(np.random.RandomState(0).rand(4, 20), axis=1)
    ncell = 0
    for fam, grid in FAMILIES.items():
        for lam in grid:
            m = multiplier(fam, lam, np.clip(pp, P_FLOOR, 1.0))
            assert (np.diff(m, axis=1) >= -1e-12).all(), (fam, lam)
            if lam == 0.0:
                assert (m == 1.0).all(), (fam, "m(0) != 1 exactly")
            ncell += 1
    log(f"  multiplier check: monotone in p on 4x20 sorted probes and m(0)==1 EXACT, "
        f"{ncell} (family x lambda) cells")
    return ncell


def selftest(ckroot, log):
    from sklearn.metrics import roc_auc_score
    ncell = check_multipliers(log)
    out = {"n_multiplier_cells": ncell}
    for arm in ("A", "B0", "B"):
        ck = os.path.join(ckroot, f"st_{arm}")
        X, lab = synth(arm)
        draws = list(range(N_PERM_SELFTEST))
        A = emit_arena(X, lab, ck, log, perm_draws=draws, tag=f"[st{arm}] ")
        pg = parity_lambda0(A, log, tag=f"[st{arm}] ")
        R = run_arms(A)
        obs = R["arms"][f"VSW_{PRIMARY_FAMILY}"]["dacc"]
        nulls = []
        for d_ in draws:
            ef = os.path.join(ck, f"eval_d{d_}.json")
            if os.path.exists(ef):
                nulls.append(json.load(open(ef))["dacc"])
                continue
            Pn = np.clip(load_perm_table(ck, A, d_), P_FLOOR, 1.0)
            v = run_arms(A, Pm=Pn, perm_mode=True)["arms"][f"VSW_{PRIMARY_FAMILY}"]["dacc"]
            atomic_json(ef, {"dacc": v})
            nulls.append(v)
        nulls = np.asarray(nulls)
        pval = round(float((1 + int((nulls >= obs).sum())) / (len(nulls) + 1)), 4)
        same = (A["nbr_lab"] == A["lab"][:, None]).astype(int).ravel()
        out[arm] = {"parity_lambda0": pg, "acc_deployed": R["acc_deployed"],
                    "dacc": obs, "arm": R["arms"][f"VSW_{PRIMARY_FAMILY}"],
                    "perm_p": pval, "n_perm": int(len(nulls)),
                    "null_mean": round(float(nulls.mean()), 4),
                    "null_sd": round(float(nulls.std()), 4),
                    "null_max": round(float(nulls.max()), 4),
                    "null_frac_ge_zero": round(float((nulls >= 0).mean()), 4),
                    "nbrscore_auc_verifier": round(
                        float(roc_auc_score(same, A["nbr_p"].ravel())), 4),
                    "nbrscore_auc_cosine": round(
                        float(roc_auc_score(same, A["nbr_cos"].ravel())), 4),
                    "curve_pow": R["curve"]["pow"],
                    "degeneracy": R["degeneracy"],
                    "class_balance": R["class_balance"]}
        log(f"  [selftest {arm}] dep {R['acc_deployed']:.4f} dacc {obs:+.4f} p={pval} "
            f"nbr-AUC cos {out[arm]['nbrscore_auc_cosine']} -> ver "
            f"{out[arm]['nbrscore_auc_verifier']} | null mean {out[arm]['null_mean']:+.4f} "
            f"max {out[arm]['null_max']:+.4f} frac>=0 {out[arm]['null_frac_ge_zero']}")
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["selftest", "anchor", "main", "perm"])
    ap.add_argument("--dataset", choices=sorted(P.DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(REPO, "scripts/analysis/vsw_ckpt"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) \
        == FROZEN_PAIRVERIFY_SHA, "FROZEN F95 ARMS MODULE CHANGED -- refusing to run"
    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py")) \
        == FROZEN_MECHFIX_SHA, "FROZEN F89 OPS MODULE CHANGED -- refusing to run"
    assert a.threads <= 8, "house cap: <= 8 threads"
    torch.set_num_threads(a.threads)

    logf = open(a.out.replace(".json", ".log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    meta = {"script": os.path.abspath(__file__),
            "script_sha256": sha256_of(os.path.abspath(__file__)),
            "frozen_pairverify_sha256": FROZEN_PAIRVERIFY_SHA,
            "frozen_mechfix_ops_sha256": FROZEN_MECHFIX_SHA,
            "stage": a.stage, "space": SPACE, "verifier": VERIFIER,
            "threads": a.threads,
            "frozen": dict(K_FOLDS=P.K_FOLDS, FOLD_SEED=P.FOLD_SEED,
                           TOPK_DEPLOYED=P.TOPK_DEPLOYED, M_PER_CLASS=P.M_PER_CLASS,
                           PCA_DIM=P.PCA_DIM, PAIR_FIT_CAP=P.PAIR_FIT_CAP,
                           MLP_EPOCHS=P.MLP_EPOCHS, MLP_SEED=P.MLP_SEED,
                           INNER_FOLDS=INNER_FOLDS, INNER_SEED=INNER_SEED,
                           N_PERM=N_PERM, PERM_SEED=PERM_SEED, DEG_KILL=DEG_KILL,
                           CLASSBAL_TOL=CLASSBAL_TOL,
                           MIN_CHANGED_FOR_RATE=MIN_CHANGED_FOR_RATE,
                           PRIMARY_FAMILY=PRIMARY_FAMILY,
                           LAM_POW=[None if not np.isfinite(x) else x for x in LAM_POW],
                           LAM_EXP=list(LAM_EXP), LAM_LIN=list(LAM_LIN),
                           F94_K_GRID=list(F94_K_GRID)),
            "test_contact": "NONE -- only the train split is loaded by this script"}

    if a.stage == "selftest":
        OUT = {"meta": meta, "selftest": selftest(a.ckpt, log)}
        atomic_json(a.out, OUT)
        log(f"SELFTEST COMPLETE -> {a.out}")
        logf.close()
        return

    assert a.dataset, "--dataset required"
    cfg = P.DATASETS[a.dataset]
    ids, img, txt, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
    X = P.build_space(img, txt, SPACE)
    meta.update({"dataset": a.dataset, "ds": cfg["ds"], "model": cfg["model"],
                 "n_train_items": int(len(lab)),
                 "pos_rate": round(float(lab.mean()), 4)})
    ck = os.path.join(a.ckpt, a.dataset)
    log(f"[{a.dataset}] n={len(lab)} pos-rate {lab.mean():.4f} dim={X.shape[1]} ck={ck}")

    if a.stage == "anchor":
        # TIER-2 PARITY ANCHOR.  Calls the FROZEN F95 module's own run_space UNMODIFIED
        # (sha asserted above) on the PRIMARY fused cell, in THIS session, so the
        # trained-arm quantities can be gated against a same-session reference.  It adds
        # no arm, changes no constant, and produces no VSW quantity.
        t0 = time.time()
        R = P.run_space(X, lab, SPACE, log)
        log(f"[{a.dataset}] ANCHOR acc_deployed {R['pooled']['acc_deployed']} "
            f"acc_cos_shape {R['pooled']['acc_cos_shape']} "
            f"acc_mlp_max {R['pooled']['acc_mlp_max']} ({time.time() - t0:.1f}s)")
        atomic_json(a.out, {"meta": meta, "space": SPACE, "run_space": R})
        log(f"[{a.dataset}] ANCHOR COMPLETE -> {a.out}")
        logf.close()
        return

    if a.stage == "main":
        A = emit_arena(X, lab, ck, log)
        # Non-positive cosines in the deployed top-20 are REPORTED, not fatal.  They arise
        # only from degenerate ZERO-NORM keys (mechnov_pairverify.l2n leaves an all-zero
        # row as the zero vector), whose every inner product is exactly 0.0; the deployed
        # rule then votes 0/210 = 0 and predicts 1, and this harness replays that
        # bit-exactly (PARITY-lambda0).  The pow/exp/lin multipliers are functions of the
        # VERIFIER score, not of the cosine, so they are unaffected; only the DEG-D cosine
        # twin sees them, and there they clip to P_FLOOR, i.e. the smallest possible score
        # gets the smallest weight, which is the monotone-correct treatment.
        nc = A["nbr_cos"]
        cosdiag = {"n_nonpositive_cos": int((nc <= 0).sum()), "n_cos_cells": int(nc.size),
                   "min_cos": round(float(nc.min()), 6),
                   "n_items_affected": int(((nc <= 0).any(1)).sum()),
                   "n_zero_norm_keys": int((np.linalg.norm(X, axis=1) == 0).sum())}
        log(f"  [{a.dataset}] cosine diagnostic {cosdiag}")
        pg = parity_lambda0(A, log, tag=f"[{a.dataset}] ")
        f95 = parity_f95(a.dataset, A, log)
        OUT = {"meta": meta, "per_fold": A["per_fold"], "parity_lambda0": pg,
               "parity_f95": f95, "cos_diagnostic": cosdiag, "result": run_arms(A)}
        atomic_json(a.out, OUT)
        log(f"[{a.dataset}] MAIN COMPLETE -> {a.out}")
        logf.close()
        return

    draws = list(range(N_PERM))
    A = emit_arena(X, lab, ck, log, perm_draws=draws, tag=f"[{a.dataset}] ")
    rows = []
    for d in draws:
        ef = os.path.join(ck, f"eval_d{d}.json")
        if os.path.exists(ef):
            rows.append(json.load(open(ef)))
            continue
        Pn = np.clip(load_perm_table(ck, A, d), P_FLOOR, 1.0)
        Rn = run_arms(A, Pm=Pn, perm_mode=True)
        r = {"draw": int(d),
             "arms": {k: {kk: Rn["arms"][k][kk] for kk in
                          ("dacc", "fixed", "broke", "exchange_rate", "changed",
                           "posrate", "foldsigns")}
                      for k in Rn["arms"]},
             "curve": {fam: [{"lam": x["lam"], "lam_is_inf": x["lam_is_inf"],
                              "dacc": x["dacc"], "fixed": x["fixed"], "broke": x["broke"],
                              "changed": x["changed"], "exchange_rate": x["exchange_rate"]}
                             for x in Rn["curve"][fam]] for fam in FAMILIES}}
        atomic_json(ef, r)
        rows.append(r)
        log(f"    {a.dataset} eval draw {d}: "
            f"{r['arms']['VSW_' + PRIMARY_FAMILY]['dacc']:+.4f}")
    atomic_json(a.out, {"meta": meta, "draws": rows})
    log(f"[{a.dataset}] PERM COMPLETE ({len(rows)} draws) -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
