#!/usr/bin/env python
"""
aggnet_pregate.py -- FROZEN implementation of the $0 CPU pregate for membank
candidate C3, the LEARNED AGGREGATION PROFILE NETWORK.

RECORD: refine-logs/AGGNET_PREGATE_RECORD.md
DESIGN SOURCE (binding): refine-logs/LITSWEEP6_MEMBANK.md section 3 (a)-(f), whose
frozen bars are quoted verbatim in the record's section 2 before any number in this
file was computed. Context (read in full first): refine-logs/VGA_PREGATE_RECORD.md
(the +0.0269 F47-gate benchmark; verifier features measured DEAD -> none appear here)
and refine-logs/RESTRANS_PREGATE_RECORD.md (the threshold-shift degeneracy this record
must rule out in weighting form).

THE IDEA UNDER TEST
    Deployed:  v = SUM_i (2*lab_i - 1) * cos_i * w_i / SUM_i w_i,  w = [20..1]
    C3:        v = SUM_i s_i * cos_i * g_i / SUM_i g_i,  g = g_theta(profile) >= 0

    Retrieval, key space, k=20, candidate set, threshold and the LABEL FIELD are all
    untouched. Only the WEIGHTING changes, and it changes per query. The plain s_i
    summand is used (NOT the r_i residual composition of LITSWEEP6 3(d)): C1
    residual transport is measured dead and closed (RESTRANS record section 7).

ARENA (F95 harness verbatim)
    Banked RAW encoder key spaces (seed-independent), TRAIN SPLIT ONLY, K=5
    StratifiedKFold(shuffle=True, random_state=0), item-disjoint. PRIMARY = fused.
    The trained head is NOT the arena (LOO train acc 0.998, F47 -> memorisation).

TEST-SPLIT CONTACT: NONE. Only <cache>/train_<model>.pt and, for the SECONDARY
+dlen arm only, data/gt/<DS>/train.jsonl. dev_seen / test_seen are never loaded.

COST: CPU only, <= 8 threads. Zero GPU, zero SLURM, zero Modal.
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
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechfix_ops as M          # noqa: E402  F89-frozen, 15/15 floor parity
import mechnov_pairverify as PV  # noqa: E402  F95-frozen harness (folds, loaders, spaces)

# ------------------------------------------------------- FROZEN OPERATOR CONTRACT
MECHFIX_OPS_SHA = "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d"
PAIRVERIFY_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"

# --------------------------------------------------------------- FROZEN CONSTANTS
K_FOLDS = 5                  # == PV.K_FOLDS
FOLD_SEED = 0                # == PV.FOLD_SEED
TOPK = 20                    # deployed budget, unchanged
PATHOLOGY_RANK = 5           # ERRPAT/F95 pathology population definition
SPACES = ("fused", "text", "img")
PRIMARY_SPACE = "fused"

# g_theta
HIDDEN = 16                  # 60*16+16 + 16*20+20 = 1316 params (LITSWEEP6: ~1-3k)
SOFTPLUS_EPS = 1e-12         # strict positivity of the denominator
NET_LR = 1e-2
LAM_DEFAULT = 1e-4           # only a signature default; every real fit selects lam
NET_EPOCHS = 300             # fixed; NO early stopping on any held-out quantity
NET_SEED = 0                 # PRIMARY (F95 MLP_SEED=0 precedent)
NET_SEEDS_STABILITY = (1, 2)
LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # shrinkage toward the deployed rule
INNER_FOLDS = 5              # inner CV inside the fitting pool selects lam
INNER_SEED = 17              # == VGA's inner-CV seed
METAK_INIT_LOGIT = 8.0       # initial mixture mass 0.9997 on k=20

# fixed monotone profile family (bar 2) and the F94 grid (DEG-B)
F94_K_GRID = (1, 2, 3, 5, 7, 10, 15, 20)
EXP_GAMMAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
POW_ALPHAS = (0.25, 0.5, 1.0, 2.0)

MONO_TOL = 0.01              # bar 3: rise > 1% of the profile max => non-monotone
DEG_KILL = 0.95              # DEG-A / DEG-B agreement threshold

N_PERM = 100                 # mandatory label-shuffled null; 100 (not VGA's 200)
                             # because each draw re-runs the FULL nested pipeline = 31 net fits
PERM_SEED = 12345

LOGIT_C = 1.0                # DEG-C readout, sklearn default L2
LOGIT_MAXITER = 1000

DATASETS = {
    "hatemm": dict(ds="HateMM", model="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
                   cache_dir=os.path.join(REPO, "data/CLIP_Embedding/HateMM"),
                   gt=os.path.join(REPO, "data/gt/HateMM/train.jsonl"), vol="words"),
    "zh": dict(ds="MHC_zh", model="Qwen2.5-VL-7B-Instruct-LoRA_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC_zh"),
               gt=os.path.join(REPO, "data/gt/MHC_zh/train.jsonl"), vol="chars"),
    "en": dict(ds="MHC", model="Qwen2.5-VL-7B-Instruct_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC"),
               gt=os.path.join(REPO, "data/gt/MHC/train.jsonl"), vol="words"),
}


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
    """F89-T3 frozen transcript-volume definition (mechfix_run.volume_scalar)."""
    txt = {}
    for line in open(gt_path):
        r = json.loads(line)
        txt[r["id"]] = r["text"]
    miss = [i for i in ids if i not in txt]
    assert not miss, ("gt text missing", miss[:5])
    return np.asarray([len(txt[i].split()) if mode == "words" else len(txt[i])
                       for i in ids], dtype="float64")


# ------------------------------------------------------------- the C3 vote engine
def vote_with_weights(S, C, G):
    """v = SUM_i s_i cos_i g_i / SUM_i g_i ; predict 1 iff v >= 0.

    With G = tile([20..1]) this is bit-for-bit mechfix_ops.deployed_vote (asserted
    every fold by the IMPL gate): the elementwise product order and the float64
    dtype are identical, and SUM(G_row) == w.sum() == 210.0 exactly.
    """
    num = (S * C * G).sum(1)
    den = G.sum(1)
    v = num / den
    return v, (v >= 0).astype(int)


def build_profile(sim, nlab, dlen=None):
    """[cos_1..20 ; s_1..20 ; purity-prefix_1..20 ; (optional) dlen_1..20].

    No verifier feature appears anywhere (VGA measured them dead). The rank-1
    margin, the vote, purity and the label ratio are all deterministic functions of
    these blocks and are therefore inside the function class already.
    """
    s = 2.0 * nlab - 1.0
    pfx = np.cumsum(nlab, axis=1) / np.arange(1, TOPK + 1, dtype="float64")[None, :]
    blocks = [sim, s, pfx]
    if dlen is not None:
        blocks.append(dlen)
    return np.concatenate(blocks, axis=1)


# ------------------------------------------------------------------ the networks
def _softplus_inv(x):
    return float(np.log(np.expm1(x)))


class ProfNet(torch.nn.Module):
    """profile -> 20 non-negative weights. DEPLOYED-ANCHORED INIT: the output layer
    starts at zero weight with bias softplus^-1(w_i / w_1), so g_theta at
    initialisation IS the deployed profile [20..1] and the arm starts bit-identical
    to the floor. It can only move away if the fitting-fold data pushes it."""

    def __init__(self, d_in):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, HIDDEN).double()
        self.l2 = torch.nn.Linear(HIDDEN, TOPK).double()
        w = M._rank_weights(TOPK)
        with torch.no_grad():
            self.l2.weight.zero_()
            self.l2.bias.copy_(torch.tensor(
                [_softplus_inv(v / w[0]) for v in w], dtype=torch.float64))

    def forward(self, p):
        return torch.nn.functional.softplus(self.l2(torch.tanh(self.l1(p)))) + SOFTPLUS_EPS


class MetaKNet(torch.nn.Module):
    """The Meta-k form: a per-query SOFTMAX MIXTURE over the eight F94 k-profiles.
    A convex mixture of monotone profiles is monotone, so this arm is C3 restricted
    to F94's family. Init puts 0.9997 of the mixture on k=20 (the deployed rule)."""

    def __init__(self, d_in, W):
        super().__init__()
        self.l1 = torch.nn.Linear(d_in, HIDDEN).double()
        self.l2 = torch.nn.Linear(HIDDEN, W.shape[0]).double()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float64))
        with torch.no_grad():
            self.l2.weight.zero_()
            b = torch.zeros(W.shape[0], dtype=torch.float64)
            b[list(F94_K_GRID).index(TOPK)] = METAK_INIT_LOGIT
            self.l2.bias.copy_(b)

    def forward(self, p):
        pi = torch.softmax(self.l2(torch.tanh(self.l1(p))), dim=1)
        return pi @ self.W + SOFTPLUS_EPS


def fit_net(kind, P_fit, S_fit, C_fit, y_fit, seed, W=None, lam=LAM_DEFAULT):
    """Full-batch Adam on BCEWithLogits(v, y) with the logit taken as v itself --
    the deployed convention (sigmoid(v) >= 0.5 <=> v >= 0). v is invariant to a
    positive rescaling of g, so the net cannot manufacture confidence by inflating
    magnitudes; only the RELATIVE weighting is learnable.

    SHRINKAGE TOWARD THE DEPLOYED RULE (explicit penalty, not Adam weight decay):

        loss = BCE(v, y) + lam * ( ||l2.weight||^2 + ||l2.bias - b_deployed||^2 )

    Both penalised terms are at their deployed-anchored values at initialisation, so
    `lam -> inf` returns EXACTLY the deployed profile [20..1] and `lam -> 0` returns
    the free conditional network. The inner CV of fit_net_nested can therefore always
    fall back to the floor, which is what makes an honest null reachable. Decaying the
    output bias toward ZERO instead (the naive weight_decay) would pull the profile
    toward UNIFORM -- a different, already-measured member of the fixed family, and
    not a fallback to anything deployed.
    """
    torch.manual_seed(seed)
    net = ProfNet(P_fit.shape[1]) if kind == "net" else MetaKNet(P_fit.shape[1], W)
    b0 = net.l2.bias.detach().clone()
    opt = torch.optim.Adam(net.parameters(), lr=NET_LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    P, S, C = torch.from_numpy(P_fit), torch.from_numpy(S_fit), torch.from_numpy(C_fit)
    Y = torch.from_numpy(y_fit.astype("float64"))
    net.train()
    for _ in range(NET_EPOCHS):
        opt.zero_grad()
        g = net(P)
        v = (S * C * g).sum(1) / g.sum(1)
        pen = (net.l2.weight ** 2).sum() + ((net.l2.bias - b0) ** 2).sum()
        (lossf(v, Y) + lam * pen).backward()
        opt.step()
    net.eval()
    return net


def fit_net_nested(kind, P_fit, S_fit, C_fit, y_fit, seed, W=None):
    """Select the shrinkage lam by an INNER StratifiedKFold inside the fitting pool,
    then refit on the whole pool at the chosen lam. The held-out fold is never seen,
    and no held-out quantity chooses lam (the VGA nesting precedent). Ties break
    toward the LARGEST lam, i.e. toward the deployed rule -- the conservative side."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    scores = {}
    for lam in LAM_GRID:
        pred = np.full(len(y_fit), -1, dtype=int)
        for tr, va in inner.split(np.zeros((len(y_fit), 1)), y_fit):
            net = fit_net(kind, P_fit[tr], S_fit[tr], C_fit[tr], y_fit[tr], seed, W, lam)
            _, p = vote_with_weights(S_fit[va], C_fit[va], apply_net(net, P_fit[va]))
            pred[va] = p
        scores[lam] = acc(y_fit, pred)
    best = max(LAM_GRID, key=lambda l: (scores[l], l))
    return fit_net(kind, P_fit, S_fit, C_fit, y_fit, seed, W, best), best, scores


def apply_net(net, P):
    with torch.no_grad():
        return net(torch.from_numpy(P)).numpy()


# --------------------------------------------------- fixed monotone profile family
def fixed_profiles():
    """27 declared non-increasing profiles (record 1.6). Order matters: ties in the
    fitting-fold selection break toward the earlier member, and 'dep' is first."""
    out = []
    r = np.arange(1, TOPK + 1, dtype="float64")
    out.append(("dep", M._rank_weights(TOPK)))
    for k in F94_K_GRID:
        w = np.zeros(TOPK); w[:k] = np.arange(k, 0, -1, dtype="float64")
        out.append((f"k{k}", w))
    for k in F94_K_GRID:
        w = np.zeros(TOPK); w[:k] = 1.0
        out.append((f"unif{k}", w))
    for g in EXP_GAMMAS:
        out.append((f"exp{g}", g ** (r - 1.0)))
    for a in POW_ALPHAS:
        out.append((f"pow{a}", r ** (-a)))
    for nm, w in out:                      # declared family invariant
        assert (np.diff(w) <= 1e-12).all(), nm
    return out


def best_threshold(v_fit, y_fit):
    """DEG-A: the global decision threshold tau maximising fitting-fold accuracy.
    Candidates are the midpoints between consecutive distinct fit votes, plus the
    deployed tau=0 and both open ends -- i.e. the exact optimum over all thresholds."""
    u = np.unique(v_fit)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0], [0.0]])
    best, bt = -1.0, 0.0
    for t in cand:
        a = acc(y_fit, (v_fit >= t).astype(int))
        if a > best:
            best, bt = a, float(t)
    return bt, best


def frac_nonmonotone(G):
    """bar 3: a learned profile is NON-MONOTONE if any adjacent rise exceeds
    MONO_TOL of that profile's own maximum."""
    rise = np.diff(G, axis=1).max(1)
    return rise > MONO_TOL * G.max(1)


# ---------------------------------------------------------------- one fold's data
def fold_tensors(X, lab, fit_idx, ho_idx, lv):
    """Everything the arms need, computed ONCE per (space, fold)."""
    Xb, yb = X[fit_idx], lab[fit_idx]
    # held-out: the identical call that produces the floor
    dv, dp, dI, dS = M.deployed_vote(Xb, yb, X[ho_idx], topk=TOPK)
    # fitting items: leave-one-out inside the fitting bank (never sees itself)
    fv, fp, fI, fS = M.deployed_vote(Xb, yb, Xb, topk=TOPK, exclude_self=True)

    nlab_ho, nlab_fit = yb[dI].astype("float64"), yb[fI].astype("float64")
    dl_ho = dl_fit = None
    if lv is not None:
        dl_ho = np.abs(lv[ho_idx][:, None] - lv[fit_idx][dI])
        dl_fit = np.abs(lv[fit_idx][:, None] - lv[fit_idx][fI])
    return dict(fit_idx=fit_idx, ho_idx=ho_idx, yb=yb,
                dv=dv, dp=dp, dI=dI, dS=dS, fv=fv, fI=fI, fS=fS,
                S_ho=2.0 * nlab_ho - 1.0, C_ho=dS, nlab_ho=nlab_ho,
                S_fit=2.0 * nlab_fit - 1.0, C_fit=fS, nlab_fit=nlab_fit,
                dl_ho=dl_ho, dl_fit=dl_fit)


# --------------------------------------------------------------------- one dataset
def run_dataset(key, log, do_perm=True, spaces=SPACES):
    cfg = DATASETS[key]
    ids, img, txt, lab = PV.load_cache(cfg["cache_dir"], "train", cfg["model"])
    n = len(ids)
    vol = load_volume(cfg["gt"], ids, cfg["vol"])
    lv = np.log1p(vol)
    log(f"[{key}] n={n} pos-rate={lab.mean():.4f} vol({cfg['vol']}) "
        f"min/med/max {vol.min():.0f}/{np.median(vol):.0f}/{vol.max():.0f}")

    anchors = json.load(open(os.path.join(
        REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")))
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(skf.split(np.zeros((n, 1)), lab))
    FAM = fixed_profiles()
    Wk = np.stack([w / w.sum() for k in F94_K_GRID
                   for nm, w in FAM if nm == f"k{k}"])          # (8, 20), row-normalised

    out = {"n_items": n, "pos_rate": round(float(lab.mean()), 4),
           "vol_mode": cfg["vol"], "spaces": {}, "perm": {}}

    for space in spaces:
        X = PV.build_space(img, txt, space)
        arms = ["C3_net", "C3_net_dlen", "C3_metak", "FIXBEST_mono", "FIXBEST_oracle",
                "THRESH_best", "DIRECT_logit"] + \
               [f"C3_net_s{s}" for s in NET_SEEDS_STABILITY] + \
               [f"FIXK_{k}" for k in F94_K_GRID]
        coll = {a: np.full(n, -1, dtype=int) for a in arms}
        coll["dep"] = np.full(n, -1, dtype=int)
        sc_rank = np.full(n, -1, dtype=int)
        mixed = np.zeros(n, dtype=bool)
        nonmono = np.zeros(n, dtype=bool)
        Gsave = np.zeros((n, TOPK), dtype="float64")
        per_fold, fixsel, lamsel = [], [], {}

        for f, (fit_idx, ho_idx) in enumerate(folds):
            t0 = time.time()
            fit_idx, ho_idx = np.asarray(fit_idx), np.asarray(ho_idx)
            T = fold_tensors(X, lab, fit_idx, ho_idx, lv)
            yb, dv, dp = T["yb"], T["dv"], T["dp"]
            coll["dep"][ho_idx] = dp
            mixed[ho_idx] = (T["nlab_ho"].min(1) != T["nlab_ho"].max(1))

            # ------------- IMPL GATE: the C3 engine at g=[20..1] IS the deployed vote
            Gdep = np.tile(M._rank_weights(TOPK), (len(ho_idx), 1))
            v0, p0 = vote_with_weights(T["S_ho"], T["C_ho"], Gdep)
            assert np.array_equal(p0, dp) and np.array_equal(v0, dv), \
                f"IMPL GATE FAIL {key}/{space}/fold{f}: C3 engine != deployed_vote at g=[20..1]"

            # ------------- profiles (standardised on the FITTING rows only)
            P_fit_raw = build_profile(T["C_fit"], T["nlab_fit"])
            P_ho_raw = build_profile(T["C_ho"], T["nlab_ho"])
            mu, sd = P_fit_raw.mean(0), P_fit_raw.std(0)
            sd[sd == 0] = 1.0
            P_fit, P_ho = (P_fit_raw - mu) / sd, (P_ho_raw - mu) / sd

            P_fit_d_raw = build_profile(T["C_fit"], T["nlab_fit"], T["dl_fit"])
            P_ho_d_raw = build_profile(T["C_ho"], T["nlab_ho"], T["dl_ho"])
            mud, sdd = P_fit_d_raw.mean(0), P_fit_d_raw.std(0)
            sdd[sdd == 0] = 1.0
            P_fit_d, P_ho_d = (P_fit_d_raw - mud) / sdd, (P_ho_d_raw - mud) / sdd

            # ------------- PRIMARY arm + stability seeds
            for tag, seed, Pf, Ph in [("C3_net", NET_SEED, P_fit, P_ho)] + \
                    [(f"C3_net_s{s}", s, P_fit, P_ho) for s in NET_SEEDS_STABILITY] + \
                    [("C3_net_dlen", NET_SEED, P_fit_d, P_ho_d)]:
                net, lam, lsc = fit_net_nested("net", Pf, T["S_fit"], T["C_fit"],
                                              yb, seed)
                lamsel.setdefault(tag, []).append(
                    {"fold": f, "lam": lam,
                     "inner_acc": {str(k): round(v, 4) for k, v in lsc.items()}})
                G = apply_net(net, Ph)
                _, p = vote_with_weights(T["S_ho"], T["C_ho"], G)
                coll[tag][ho_idx] = p
                if tag == "C3_net":
                    Gsave[ho_idx] = G
                    nonmono[ho_idx] = frac_nonmonotone(G)

            # ------------- Meta-k (monotone-restricted twin)
            net, lam, lsc = fit_net_nested("metak", P_fit, T["S_fit"], T["C_fit"],
                                          yb, NET_SEED, W=Wk)
            lamsel.setdefault("C3_metak", []).append(
                {"fold": f, "lam": lam,
                 "inner_acc": {str(k): round(v, 4) for k, v in lsc.items()}})
            _, p = vote_with_weights(T["S_ho"], T["C_ho"], apply_net(net, P_ho))
            coll["C3_metak"][ho_idx] = p

            # ------------- bar 2: best fixed monotone profile, selected on FIT folds
            fit_accs, ho_accs = {}, {}
            for nm, w in FAM:
                _, pf = vote_with_weights(T["S_fit"], T["C_fit"],
                                          np.tile(w, (len(fit_idx), 1)))
                fit_accs[nm] = acc(yb, pf)
                _, ph = vote_with_weights(T["S_ho"], T["C_ho"],
                                          np.tile(w, (len(ho_idx), 1)))
                ho_accs[nm] = acc(lab[ho_idx], ph)
                if nm.startswith("k") and nm[1:].isdigit():
                    coll[f"FIXK_{int(nm[1:])}"][ho_idx] = ph
            bn = max(FAM, key=lambda t: (fit_accs[t[0]], -[x[0] for x in FAM].index(t[0])))[0]
            on = max(FAM, key=lambda t: (ho_accs[t[0]], -[x[0] for x in FAM].index(t[0])))[0]
            wsel = dict(FAM)[bn]
            _, p = vote_with_weights(T["S_ho"], T["C_ho"], np.tile(wsel, (len(ho_idx), 1)))
            coll["FIXBEST_mono"][ho_idx] = p
            _, p = vote_with_weights(T["S_ho"], T["C_ho"],
                                     np.tile(dict(FAM)[on], (len(ho_idx), 1)))
            coll["FIXBEST_oracle"][ho_idx] = p
            fixsel.append({"fold": f, "fit_selected": bn, "fit_acc": round(fit_accs[bn], 4),
                           "oracle_selected": on, "oracle_ho_acc": round(ho_accs[on], 4)})

            # ------------- DEG-A: best global threshold on the fit folds
            tau, tacc = best_threshold(T["fv"], yb)
            coll["THRESH_best"][ho_idx] = (dv >= tau).astype(int)

            # ------------- DEG-C: unconstrained logistic readout of the SAME profile
            clf = LogisticRegression(penalty="l2", C=LOGIT_C, solver="lbfgs",
                                     max_iter=LOGIT_MAXITER, n_jobs=1).fit(P_fit, yb)
            coll["DIRECT_logit"][ho_idx] = clf.predict(P_ho)

            # ------------- pathology rank (F95 definition, full-space cosine)
            Sfull = X[ho_idx] @ X[fit_idx].T
            order = np.argsort(-Sfull, axis=1, kind="stable")
            bl = yb[order]
            for r_, q_ in enumerate(ho_idx):
                hit = np.flatnonzero(bl[r_] == lab[q_])
                sc_rank[q_] = int(hit[0]) + 1 if len(hit) else 10 ** 6

            rec = {"fold": f, "n_fit": int(len(fit_idx)), "n_ho": int(len(ho_idx)),
                   "acc_deployed": round(acc(lab[ho_idx], dp), 4),
                   "tau": round(tau, 6), "tau_fit_acc": round(tacc, 4),
                   "secs": round(time.time() - t0, 1)}
            for a in arms:
                rec[f"acc_{a}"] = round(acc(lab[ho_idx], coll[a][ho_idx]), 4)
                rec[f"d_{a}"] = round(rec[f"acc_{a}"] - rec["acc_deployed"], 4)
            per_fold.append(rec)
            log(f"    [{key}/{space}] fold {f} dep {rec['acc_deployed']:.4f}  " +
                "  ".join(f"{a}:{rec['d_' + a]:+.4f}" for a in
                          ("C3_net", "C3_metak", "FIXBEST_mono", "THRESH_best",
                           "DIRECT_logit")) + f"  ({rec['secs']}s)")

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
               "got_n_pathology": int(((coll["dep"] != lab) & (sc_rank <= PATHOLOGY_RANK)
                                       & (sc_rank > 0)).sum())}
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
        patho = dep_wrong & (sc_rank <= PATHOLOGY_RANK) & (sc_rank > 0)
        near = (sc_rank <= PATHOLOGY_RANK) & (sc_rank > 0)
        pooled = {"acc_deployed": par["got_pooled_acc"],
                  "mF1_deployed": par["got_pooled_mF1"],
                  "posrate_bank": round(float(lab.mean()), 4),
                  "posrate_deployed": round(float(coll["dep"].mean()), 4),
                  "n_deployed_wrong": int(dep_wrong.sum()),
                  "n_pathology": int(patho.sum()),
                  "n_class_mixed_top20": int(mixed.sum()),
                  "frac_class_mixed_top20": round(float(mixed.mean()), 4),
                  "frac_nonmonotone_C3net": round(float(nonmono.mean()), 4),
                  "n_nonmonotone_C3net": int(nonmono.sum())}
        for a in arms:
            p = coll[a]
            assert (p >= 0).all(), a
            fixed = dep_wrong & (p == lab)
            broke = (~dep_wrong) & (p != lab)
            fs = [r[f"d_{a}"] for r in per_fold]
            e = {"acc": round(acc(lab, p), 4), "mF1": round(M.macro_f1(lab, p), 4)}
            e["d_acc"] = round(e["acc"] - pooled["acc_deployed"], 4)
            e["d_mF1"] = round(e["mF1"] - pooled["mF1_deployed"], 4)
            e["posrate"] = round(float(p.mean()), 4)
            e["fold_deltas"] = fs
            e["fold_signs"] = "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in fs)
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
                                               / float(patho.sum()), 4) if patho.sum() else None)
            e["changed_rank1to5"] = int(((p != coll["dep"]) & near).sum())
            e["changed_rank6plus"] = int(((p != coll["dep"]) & ~near).sum())
            # degeneracy agreements (pooled over held-out items)
            e["agree_C3net"] = round(float((p == coll["C3_net"]).mean()), 4)
            e["agree_deployed"] = round(float((p == coll["dep"]).mean()), 4)
            pooled[a] = e

        c3 = coll["C3_net"]
        pooled["DEG"] = {
            "A_agree_threshold_shift": round(float((c3 == coll["THRESH_best"]).mean()), 4),
            "B_agree_fixk_max": round(max(float((c3 == coll[f"FIXK_{k}"]).mean())
                                          for k in F94_K_GRID), 4),
            "B_agree_fixk": {str(k): round(float((c3 == coll[f"FIXK_{k}"]).mean()), 4)
                             for k in F94_K_GRID},
            "B_argmax_k": int(max(F94_K_GRID,
                                  key=lambda k: float((c3 == coll[f"FIXK_{k}"]).mean()))),
            "C_agree_direct_logit": round(float((c3 == coll["DIRECT_logit"]).mean()), 4),
            "agree_deployed": round(float((c3 == coll["dep"]).mean()), 4),
            "agree_fixbest_mono": round(float((c3 == coll["FIXBEST_mono"]).mean()), 4)}
        # bar 3: is the delta concentrated on the non-monotone subpopulation?
        pooled["nonmono_read"] = {
            "n_nonmono": int(nonmono.sum()), "n_mono": int((~nonmono).sum()),
            "d_acc_nonmono": (round(acc(lab[nonmono], c3[nonmono])
                                    - acc(lab[nonmono], coll["dep"][nonmono]), 4)
                              if nonmono.sum() else None),
            "d_acc_mono": (round(acc(lab[~nonmono], c3[~nonmono])
                                 - acc(lab[~nonmono], coll["dep"][~nonmono]), 4)
                           if (~nonmono).sum() else None),
            "n_changed_nonmono": int(((c3 != coll["dep"]) & nonmono).sum()),
            "n_changed_mono": int(((c3 != coll["dep"]) & ~nonmono).sum()),
            "mean_rise_over_max": round(float((np.diff(Gsave, axis=1).max(1)
                                               / Gsave.max(1)).mean()), 4),
            "mean_G_profile": [round(float(x), 4) for x in Gsave.mean(0)]}
        pooled["fixed_profile_selection"] = fixsel
        pooled["lambda_selection"] = lamsel
        out["spaces"][space] = {"per_fold": per_fold, "parity": par, "pooled": pooled}

    # ------------------------------------------------ permutation null (PRIMARY only)
    if do_perm:
        X = PV.build_space(img, txt, PRIMARY_SPACE)
        obs = out["spaces"][PRIMARY_SPACE]["pooled"]["C3_net"]["d_acc"]
        rng = np.random.RandomState(PERM_SEED)
        pre = []
        for f, (fit_idx, ho_idx) in enumerate(folds):
            T = fold_tensors(X, lab, np.asarray(fit_idx), np.asarray(ho_idx), None)
            Praw = build_profile(T["C_fit"], T["nlab_fit"])
            Hraw = build_profile(T["C_ho"], T["nlab_ho"])
            mu, sd = Praw.mean(0), Praw.std(0)
            sd[sd == 0] = 1.0
            pre.append((T, (Praw - mu) / sd, (Hraw - mu) / sd, np.asarray(ho_idx)))
        t0 = time.time()
        nulls = []
        for b in range(N_PERM):
            pnull = np.full(n, -1, dtype=int)
            for (T, Pf, Ph, ho_idx) in pre:
                ysh = T["yb"][rng.permutation(len(T["yb"]))]
                net, _, _ = fit_net_nested("net", Pf, T["S_fit"], T["C_fit"],
                                           ysh, NET_SEED)
                _, p = vote_with_weights(T["S_ho"], T["C_ho"], apply_net(net, Ph))
                pnull[ho_idx] = p
            nulls.append(round(acc(lab, pnull)
                               - out["spaces"][PRIMARY_SPACE]["pooled"]["acc_deployed"], 6))
            if (b + 1) % 25 == 0:
                log(f"  [{key}] perm {b + 1}/{N_PERM} "
                    f"({time.time() - t0:.0f}s, mean {np.mean(nulls):+.4f})")
        nl = np.asarray(nulls)
        out["perm"] = {"arm": "C3_net", "space": PRIMARY_SPACE, "n_perm": N_PERM,
                       "observed": obs, "null_mean": round(float(nl.mean()), 4),
                       "null_sd": round(float(nl.std()), 4),
                       "null_q95": round(float(np.quantile(nl, 0.95)), 4),
                       "null_max": round(float(nl.max()), 4),
                       "p": round(float((1 + int((nl >= obs).sum())) / (N_PERM + 1)), 4),
                       "nulls": nulls}
        log(f"  [{key}] PERM obs {obs:+.4f} null {nl.mean():+.4f}+-{nl.std():.4f} "
            f"q95 {np.quantile(nl, 0.95):+.4f} p={out['perm']['p']:.4f}")
    return out


# ------------------------------------------------------------------- self-test
def selftest(log):
    """Machinery validity on SYNTHETIC DATA ONLY, run BEFORE the freeze, at the real
    problem's scale (n=700, 560 fitted / 140 held out). Three arms:

      A_conditional     the right weighting IS a conditional function of the profile
                        (two populations needing OPPOSITE monotone treatments, keyed
                        by the rank-1 margin -- LITSWEEP6 3(c)'s own example).
                        The harness must recover it.
      B_priorfallback   the neighbourhood says nothing about y. Documents the reach of
                        the function class: because g >= 0 and the neighbourhoods are
                        class-mixed, C3 can emit ANY sign per query, so it can always
                        fall back to the marginal class prior. This is not a null -- it
                        is the DEG-C concern made concrete, and it is why bar 2 and the
                        permutation null, not the deployed floor, are the real controls.
      C_deployedoptimal the deployed rule is already optimal and the residual errors are
                        unpredictable from the profile. The harness must return ~0:
                        this is the honest-null arm, and it prices the overfitting cost
                        of fitting 1316 parameters on 560 examples.
    """
    res = {}
    for arm in ("A_conditional", "B_priorfallback", "C_deployedoptimal"):
        rng = np.random.RandomState(7)
        n, nb = 700, 560
        y = (rng.rand(n) < 0.4).astype(int)
        S, C = np.zeros((n, TOPK)), np.zeros((n, TOPK))
        big = rng.rand(n) < 0.5      # the CONDITIONAL variable: rank-1 margin
        for i in range(n):
            if arm == "C_deployedoptimal":
                # ~14/20 neighbours carry the majority label m; y = m 84% of the time and
                # the 16% flips are NOT a function of anything in the profile
                m = int(rng.rand() < 0.5)
                y[i] = m if rng.rand() < 0.84 else 1 - m
                lb = np.full(TOPK, m, dtype=float)
                lb[rng.choice(TOPK, size=6, replace=False)] = 1 - m
                S[i] = 2 * lb - 1
                C[i] = -np.sort(-(0.94 + 0.05 * rng.rand(TOPK)))
                continue
            yy = y[i] if arm == "A_conditional" else int(rng.rand() < 0.4)
            if big[i]:
                # large rank-1 margin: the lone rank-1 neighbour is the right analogue,
                # the other 19 are wrong-class -- the ERRPAT configuration
                s = np.full(TOPK, -1.0 if yy == 1 else 1.0)
                s[0] = 1.0 if yy == 1 else -1.0
                c = np.full(TOPK, 0.940); c[0] = 0.990
            else:
                # small rank-1 margin: rank 1 is the odd one out and the majority is right
                s = np.full(TOPK, 1.0 if yy == 1 else -1.0)
                s[0] = -1.0 if yy == 1 else 1.0
                c = np.full(TOPK, 0.940); c[0] = 0.945
            S[i], C[i] = s, c
        nlab = (S + 1) / 2
        P = build_profile(C, nlab)
        mu, sd = P[:nb].mean(0), P[:nb].std(0); sd[sd == 0] = 1.0
        Pn = (P - mu) / sd
        net, lam, lsc = fit_net_nested("net", Pn[:nb], S[:nb], C[:nb], y[:nb], NET_SEED)
        G = apply_net(net, Pn[nb:])
        _, p = vote_with_weights(S[nb:], C[nb:], G)
        _, pd = vote_with_weights(S[nb:], C[nb:], np.tile(M._rank_weights(TOPK),
                                                          (n - nb, 1)))
        res[arm] = {"acc_deployed": round(acc(y[nb:], pd), 4),
                    "acc_C3": round(acc(y[nb:], p), 4),
                    "d_acc": round(acc(y[nb:], p) - acc(y[nb:], pd), 4),
                    "majority_class_rate": round(float(max(y[nb:].mean(),
                                                           1 - y[nb:].mean())), 4),
                    "frac_nonmonotone": round(float(frac_nonmonotone(G).mean()), 4),
                    "lam_selected": lam,
                    "inner_acc": {str(k): round(v, 4) for k, v in lsc.items()}}
        log(f"  SELFTEST {arm}: {json.dumps(res[arm])}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hatemm,zh,en")
    ap.add_argument("--spaces", default=",".join(SPACES))
    ap.add_argument("--out", default=os.path.join(
        REPO, "scripts/analysis/aggnet_pregate_OUT.json"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-perm", action="store_true")
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

    if a.selftest:
        log("SELFTEST (synthetic data only, pre-freeze machinery validity)")
        selftest(log)
        logf.close()
        return

    me = os.path.abspath(__file__)
    ops_sha = sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py"))
    pv_sha = sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py"))
    assert ops_sha == MECHFIX_OPS_SHA, ("mechfix_ops.py is not the F89-frozen file", ops_sha)
    assert pv_sha == PAIRVERIFY_SHA, ("mechnov_pairverify.py is not the F95-frozen file", pv_sha)

    OUT = {"meta": {
        "script": me, "script_sha256": sha256_of(me),
        "mechfix_ops_sha256": ops_sha, "mechnov_pairverify_sha256": pv_sha,
        "cpu_only": True, "gpu_jobs": 0, "slurm_jobs": 0, "modal_jobs": 0,
        "test_contact": "NONE -- only train_*.pt and data/gt/*/train.jsonl are opened",
        "frozen": dict(K_FOLDS=K_FOLDS, FOLD_SEED=FOLD_SEED, TOPK=TOPK,
                       PATHOLOGY_RANK=PATHOLOGY_RANK, SPACES=list(SPACES),
                       PRIMARY_SPACE=PRIMARY_SPACE, HIDDEN=HIDDEN,
                       SOFTPLUS_EPS=SOFTPLUS_EPS, NET_LR=NET_LR,
                       NET_EPOCHS=NET_EPOCHS, NET_SEED=NET_SEED,
                       NET_SEEDS_STABILITY=list(NET_SEEDS_STABILITY),
                       METAK_INIT_LOGIT=METAK_INIT_LOGIT,
                       F94_K_GRID=list(F94_K_GRID), EXP_GAMMAS=list(EXP_GAMMAS),
                       POW_ALPHAS=list(POW_ALPHAS), MONO_TOL=MONO_TOL, LAM_GRID=list(LAM_GRID),
                       INNER_FOLDS=INNER_FOLDS, INNER_SEED=INNER_SEED,
                       DEG_KILL=DEG_KILL, N_PERM=N_PERM, PERM_SEED=PERM_SEED,
                       LOGIT_C=LOGIT_C, LOGIT_MAXITER=LOGIT_MAXITER),
    }, "datasets": {}}

    for k in a.datasets.split(","):
        OUT["datasets"][k] = run_dataset(k, log, do_perm=not a.no_perm,
                                         spaces=tuple(a.spaces.split(",")))
        json.dump(OUT, open(a.out, "w"), indent=1)
    json.dump(OUT, open(a.out, "w"), indent=1)
    log(f"DONE -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
