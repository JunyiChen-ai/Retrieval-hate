#!/usr/bin/env python
"""c02_a0_arena_v9.py -- C02 A0 Stage-0 density-orbit reachability oracle.

Record: refine-logs/C02_A0_V9_RECORD.md.  Config: configs/c02/c02_a0_v9.json.
Registry authority: TARGET_STATE.json::iteration_8_stage0_bounded_extraction_amendment.

THE QUESTION
    Each video carries a discrete orbit of controlled evidence-density views of its
    own text channel.  Give retrieval the QUOTIENT similarity induced by that orbit,

        s_Q(i, j) = max_{a in A_i, b in A_j} cos(z_i^a, z_j^b)

    -- the orbit metric of the quotient space -- and ask whether it clears the
    Stage-0 bar of +0.050 accuracy AND +0.050 macro-F1 over the paired native floor
    on BOTH HateMM and MHC-ZH.

    s_Q is OPTIMISTIC in the ordinary sense: it may use the best view of every item on
    BOTH sides of every comparison, which no deployable system may do, and it is not a
    router.  It is NOT a proven supremum over all orbit-contracting representations --
    it is ONE PARTICULAR orbit-invariant similarity, the canonical max-matching quotient
    pseudo-metric.  A KILL is therefore a GATE verdict under the registry's frozen
    Stage-0 rule, not a proof that no orbit-contracting representation could ever help;
    a PASS authorises Stage-1 design plus a fresh review only.  (The v1/v2 wording
    "OPTIMISTIC UPPER BOUND ... a failure is decisive" is RETRACTED.)

ARENA -- fold-head / deployed-head, because a raw arena may not promote (F113)
    StratifiedKFold(5, shuffle=True, random_state=0) over the train split, asserted
    item-for-item against the banked scripts/analysis/vsw_ckpt/<ds>/f<fold>.npz inside
    the mint.  Bank = the fitting pool's head keys from a head trained on that same
    fitting pool; queries = the held-out fifth, never seen by that head in any role.
    Bank and query index sets are disjoint, so a query's own orbit can never be
    retrieved -- FULL SELF-ORBIT EXCLUSION, asserted per fold.  Pooled over all train
    items, 3 head seeds; the 3-seed mean is the primary read.  The raw fused key space
    is computed as well and is SECONDARY: it may corroborate a KILL and may never
    promote.

EVERY ARM IS A SUB-ORBIT OF ONE EXTRACTION.  No arm needs its own GPU spend.

TEST CONTACT: NONE.  Only train_* mint npz files and train_* caches are opened; a path
guard refuses anything test-like before any file handle is created.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
sys.path.insert(0, os.path.join(REPO, "src/utils"))
os.chdir(REPO)

import faiss  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

if not __debug__:
    raise SystemExit("REFUSING TO RUN: python -O strips the assert-based guards")

_ORIG_TORCH_LOAD = torch.load


def _guarded_torch_load(f, *a, **kw):
    sp = str(f).lower()
    for tok in ("test_seen", "/test", "test.jsonl"):
        if tok in sp:
            raise RuntimeError("TEST-SPLIT GUARD: refusing to open {}".format(f))
    return _ORIG_TORCH_LOAD(f, *a, **kw)


torch.load = _guarded_torch_load

import mechfix_ops as M  # noqa: E402
import mechnov_pairverify as P  # noqa: E402
import c02_density_views as V  # noqa: E402

FROZEN = {
    "scripts/analysis/mechfix_ops.py":
        "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d",
    "scripts/analysis/mechnov_pairverify.py":
        "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "scripts/analysis/headspace_mint.py":
        "cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612",
}

# ------------------------------------------------------------------ FROZEN CONSTANTS
TOPK = 20
DATASETS = ("hatemm", "zh")
SEEDS = (0, 1, 2)
BAR_ACC = 0.050
BAR_MF1 = 0.050
BAR_NETFIX_RATE = 0.030          # net (fixed - broken) / n, the +0.030 final-bar clause
VIEW_SUPPORT_MIN = 0.60
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260730
NOISE_SEED = 20260730
SHUFFLE_SEED = 20260730
ALPHA = 0.05
ARENA2_MARGIN = 0.02
ARENA2_CEILING = 0.98
EXT_PARITY_MEDIAN_COS_MIN = 0.99
TINY_NORM = 1e-12
KRR_RIDGE = 1.0                  # frozen: RBF gamma = 1/d on L2-normalised keys

ARM_NAMES = ("NATIVE", "FULL", "REPEAT_ONLY", "LOCALIZED_REPEAT_ONLY",
             "RANDOM_WINDOW_REPEAT", "MIN_WINDOW_REPEAT", "MAX_WINDOW_REPEAT",
             "SHUFFLE", "NOISE")

HALT = {
    "SCHEMA": "HALT_C02_A0_SCHEMA",
    "PARITY": "HALT_C02_A0_NATIVE_PARITY",
    "FOLD": "HALT_C02_A0_FOLD_PARITY",
    "ARENA": "HALT_C02_A0_ARENA_DEGENERATE",
    "EXTRACT": "HALT_C02_A0_EXTRACTION_PARITY",
    "ZERO": "HALT_C02_A0_ZERO_CONTRACT",
    "SUPPORT": "HALT_C02_A0_VIEW_SUPPORT",
    "NONFINITE": "HALT_C02_A0_NONFINITE",
    "TEST": "HALT_C02_A0_TEST_PATH",
}


class Halt(Exception):
    pass


def halt(label, msg):
    raise Halt("{}: {}".format(label, msg))


def guard_path(path):
    low = str(path).lower()
    for tok in ("test_seen", "/test", "test.jsonl"):
        if tok in low:
            halt(HALT["TEST"], "refusing path {}".format(path))
    return path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


def l2n(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def _norm32(X):
    """float32 C-contiguous L2-normalised copy -- the deployed pre-index step.

    Numerically identical to mechfix_ops._norm32, but ALWAYS copies.  The frozen
    version can alias its input when handed a float32 C-contiguous array, and
    faiss.normalize_L2 works in place, so an aliasing call would silently normalise a
    caller's stored keys and make a second normalisation of the same buffer differ at
    float32 ulp level.  Every array this arena hands to faiss is therefore private.
    """
    Y = np.array(X, dtype="float32", order="C", copy=True)
    faiss.normalize_L2(Y)
    return Y


# ------------------------------------------------------------------ the orbit oracle
def orbit_vote(bank_views, bank_lab, query_views, topk=TOPK):
    """The quotient similarity s_Q, then the DEPLOYED top-20 rank-weighted vote.

    bank_views / query_views: lists (the orbit) of [n, d] arrays.

    EXACTNESS OF THE k = topk SEARCH PER VIEW PAIR.  Let tau be the topk-th largest
    per-item maximum m_j = max_{a,b} s(q_a, B_b[j]).  Any row (a, b, j) with
    s >= tau forces m_j >= tau, so j is one of the topk items; hence each (a, b) pair
    contributes AT MOST topk rows at or above tau and its own top-topk list already
    contains every one of them.  Searching k = topk per pair is therefore exact for
    the top-topk items and their maxima WHENEVER AT MOST topk ITEMS ATTAIN tau: with
    topk or more EXACT float32 ties AT tau inside a single pair, a boundary item could
    be dropped.  The only exactly-tied rows reachable on this data are duplicate keys,
    i.e. the structural all-zero rows (1 on HateMM train, 0 on MHC-ZH), where every
    similarity is 0, the vote is identically 0 and the prediction is invariant.
    Crucially, for a singleton orbit this is
    the LITERAL deployed k = 20 faiss call, which is what makes PARITY-NAT bit-exact.
    WHY k = topk RATHER THAN AN EXHAUSTIVE SEARCH.  Not for bit-exactness: an exhaustive
    k = n_bank search IS bit-equal to the deployed k = 20 call (re-measured on synthetic
    arrays with private, singly-normalised operands: similarities, neighbour ids and votes
    all identical, max |delta sim| = 0.0).  An earlier draft of this record claimed the
    opposite and attributed a measured 1.4901161193847656e-07 discrepancy to the search
    width; that discrepancy was ENTIRELY the _norm32 ALIASING defect described below --
    the operands had been normalised twice -- and the claim is RETRACTED.  k = topk is
    used because it is the smaller and more obviously exact object: it is provably
    sufficient for the top-topk (above); it returns an (nq x topk) result and selects
    with a topk-sized heap instead of materialising all n_bank similarities per view pair
    (the SCAN is O(n_bank) either way -- a flat inner-product index computes every inner
    product regardless of k, which bounds only the heap and the result width, and an
    earlier wording that claimed an O(topk) search cost is corrected here); and for a
    singleton orbit it is LITERALLY the deployed call rather than merely equal to it.

    Ties in s_Q resolve to the LOWER bank index.
    """
    nb = bank_views[0].shape[0]
    nq = query_views[0].shape[0]
    if nb < topk:
        halt(HALT["SCHEMA"], "bank of {} rows is smaller than topk {}".format(nb, topk))
    q32 = [_norm32(x) for x in query_views]
    b32 = [_norm32(x) for x in bank_views]
    best = np.full((nq, nb), -np.inf, dtype="float64")
    rows = np.arange(nq)[:, None]
    for bv in b32:
        ix = faiss.IndexFlatIP(bv.shape[1])
        ix.add(bv)
        for qv in q32:
            D, I = ix.search(qv, topk)
            if I.shape[1] != topk or np.any(I < 0):
                halt(HALT["SCHEMA"], "faiss search did not return topk valid ids")
            D64 = D.astype("float64")
            best[rows, I] = np.maximum(best[rows, I], D64)

    idx = np.tile(np.arange(nb), (nq, 1))
    order = np.lexsort((idx, -best), axis=1)[:, :topk]
    sim = np.take_along_axis(best, order, axis=1)
    if not np.all(np.isfinite(sim)):
        halt(HALT["NONFINITE"], "top-{} contains a non-finite similarity".format(topk))
    lab = np.asarray(bank_lab)[order].astype("float64")
    w = M._rank_weights(topk)
    votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()
    return votes, (votes >= 0).astype(int), order, sim


# -------------------------------------------------------------------- per-item views
def per_item_view_matrix(keys, choice):
    """[n, d] whose row i is taken from view `choice[i]`."""
    n, d = keys["NAT"].shape
    out = np.empty((n, d), dtype=keys["NAT"].dtype)
    for i in range(n):
        out[i] = keys[choice[i]][i]
    return out


def shuffle_groups(fit, ho, degenerate_mask):
    """Donor groups = partition x degeneracy class.

    The partition split (fitting pool vs held-out fifth) is the LEAKAGE boundary and is
    absolute.  The degeneracy split keeps an item whose own orbit is the identity from
    receiving a real displacement it would never have under FULL.  A class group of
    exactly one member is DROPPED, not merged: see the comment below for why that is the
    conservative choice.  The dropped count is returned and reported.
    """
    out, dropped = [], 0
    for part in (np.asarray(fit), np.asarray(ho)):
        nd = part[~degenerate_mask[part]]
        dg = part[degenerate_mask[part]]
        # A class group of exactly one member cannot be deranged.  DROP it rather than
        # merge it into the other class.  Dropping leaves that item carrying its OWN
        # displacement in SHUFFLE, and for the case that matters -- a lone DEGENERATE
        # item, whose displacement is ZERO by construction -- that makes SHUFFLE EXACTLY
        # MATCHED to FULL for that item, so it can contribute nothing to either side of
        # `FULL > SHUFFLE`.  Merging would instead have handed that degenerate item a
        # real displacement it never has under FULL, which makes the conjunct EASIER.
        # (For a lone NON-degenerate item the drop leaves a single unshuffled real
        # displacement in the control, i.e. a trace of the treatment, which can only make
        # the conjunct harder.)  Dropped items are excluded from the fixed-point check
        # and counted.
        for g in (nd, dg):
            if g.size >= 2:
                out.append(g)
            elif g.size == 1:
                dropped += 1
    return out, dropped


def derangement_within(groups, n, seed, fold):
    """Derange WITHIN each supplied index group, never across groups.

    The partition part of the grouping is load-bearing.  A GLOBAL derangement would hand
    a held-out query a donor that is IN THE BANK; combined with the displacement-donation
    rule below that is not a near-identity leak, but keeping donor and recipient in the
    same partition also keeps them in the same statistical regime, so it is retained.
    """
    perm = np.arange(n)
    for gi, g in enumerate(groups):
        sub = np.asarray(g)
        if sub.size < 2:
            continue
        rng = np.random.default_rng([seed, fold, gi])
        # Sattolo's algorithm: a uniformly random CYCLIC permutation, hence a derangement
        # BY CONSTRUCTION for every size >= 2, in O(m) with no rejection and no repair
        # loop.  The previous pairwise-swap repair could oscillate forever on a size-2
        # identity draw and exit with fixed points -- and v3's degeneracy-class grouping
        # is exactly what made size-2 groups reachable.
        m = int(sub.size)
        p = np.arange(m)
        for i in range(m - 1, 0, -1):
            j = int(rng.integers(0, i))          # strictly j < i => single cycle
            p[i], p[j] = p[j], p[i]
        assert np.sum(p == np.arange(m)) == 0, "within-group derangement failed"
        perm[sub] = sub[p]
    covered = np.concatenate([np.asarray(g) for g in groups]) if groups else np.array([], int)
    if np.sum(perm[covered] == covered) != 0:
        # Unreachable by construction: shuffle_groups emits only groups of size >= 2 and
        # Sattolo is a derangement for every such group.  It must nevertheless fail CLOSED
        # through Halt so the run still publishes a decision artifact instead of dying on
        # a bare AssertionError.
        halt(HALT["ARENA"], "derangement left a fixed point: a donor group violated "
                            "shuffle_groups' size >= 2 rule")
    return perm


def load_p3_windows(ds_name):
    """id -> list[4] int P3 evidence-density scores.  TRAIN split only."""
    path = os.path.join(REPO, "data/MLLM_scores", ds_name, "train_segscoreK4_qwen.jsonl")
    guard_path(path)
    out = {}
    if not os.path.exists(path):
        return out, path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sc = o.get("scores")
            if isinstance(sc, list) and len(sc) == V.K_WINDOWS:
                out[str(o["id"])] = [int(x) for x in sc]
    return out, path


def build_arms(keys, choices, perm):
    """orbit dict: arm name -> list of [n, d] key matrices.

    `perm` is the FOLD-LOCAL within-partition derangement supplied by the caller.
    """
    K_rand = per_item_view_matrix(keys, choices["rand"])
    K_min = per_item_view_matrix(keys, choices["min"])
    K_max = per_item_view_matrix(keys, choices["max"])

    # SHUFFLE donates the DISPLACEMENT, not the absolute view vector:
    #     z_i^v  :=  NAT_i + ( view_v(pi(i)) - NAT_pi(i) )
    # An earlier draft donated pi(i)'s absolute view keys.  That made bank row j a
    # near-duplicate of bank row pi(j), so every true neighbour was mirrored onto an
    # unrelated-label row and SHUFFLE degraded under the design's OWN null -- the
    # `FULL > SHUFFLE` conjunct was then satisfiable at H0 and proved nothing.
    # Displacement donation removes that: no component of pi(i)'s POSITION enters, so no
    # spurious cross-item near-duplicate is created, while the correspondence between a
    # density displacement and the video it came from is destroyed.
    #
    # SCOPE OF THE RESULTING CONJUNCT, stated precisely.  FULL and SHUFFLE are
    # exchangeable under the EXCHANGEABILITY NULL: displacements drawn i.i.d. across
    # items and independent of the item they attach to.  They are NOT exchangeable under
    # every null in which the orbit is uninformative -- e.g. a purely RADIAL displacement
    # d_i = eps * NAT_i leaves FULL identical to NATIVE after L2 normalisation while
    # SHUFFLE still perturbs.  `FULL > SHUFFLE` is therefore NECESSARY, NOT SUFFICIENT.
    # It cannot manufacture a PASS: under that radial null FULL == NATIVE exactly, so
    # delta_acc = 0 and the binding +0.050 bar is unreachable.  The bar against the
    # paired native floor is what carries the verdict; SHUFFLE and NOISE only exclude
    # ways of clearing it that do not need the correct within-video orbit.
    nat_sh = keys["NAT"].astype("float64")
    shuffled = [keys["NAT"]] + [
        (nat_sh + (keys[v].astype("float64")[perm] - nat_sh[perm])).astype(
            keys["NAT"].dtype) for v in V.NON_NATIVE_VIEWS]

    rngn = np.random.default_rng(NOISE_SEED)
    noised = [keys["NAT"]]
    nat64 = keys["NAT"].astype("float64")
    for v in V.NON_NATIVE_VIEWS:
        disp = keys[v].astype("float64") - nat64
        nrm = np.linalg.norm(disp, axis=1, keepdims=True)
        g = rngn.standard_normal(disp.shape)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
        noised.append((nat64 + g * nrm).astype(keys["NAT"].dtype))

    # DECLARED ASSUMPTION.  SHUFFLE and NOISE are built by vector arithmetic in head-key
    # space, whereas FULL's non-native views are genuine head outputs.  If the head is
    # materially nonlinear at the measured orbit radius, the two nulls sit slightly off
    # the head's image manifold and are handicapped for a reason unrelated to the null --
    # in the direction that makes BOTH conjuncts EASIER, so it can weaken a PASS but
    # cannot manufacture a KILL.  orbit_radius_median_oof and its per-view breakdown are
    # reported precisely so this can be bounded post hoc; it is not gated.
    return {
        "NATIVE": [keys["NAT"]],
        "FULL": [keys[v] for v in V.VIEW_NAMES],
        "REPEAT_ONLY": [keys["NAT"], keys["RFULL"]],
        "LOCALIZED_REPEAT_ONLY": [keys["NAT"]] + [keys["RW%d" % k]
                                                  for k in range(1, V.K_WINDOWS + 1)],
        "RANDOM_WINDOW_REPEAT": [keys["NAT"], K_rand],
        "MIN_WINDOW_REPEAT": [keys["NAT"], K_min],
        "MAX_WINDOW_REPEAT": [keys["NAT"], K_max],
        "SHUFFLE": shuffled,
        "NOISE": noised,
    }


def make_choices(ids, p3):
    rand_choice, min_choice, max_choice, n_missing = [], [], [], 0
    for vid in ids:
        rand_choice.append("RW{}".format(V.random_window(vid)))
        sc = p3.get(str(vid))
        if sc is None:
            n_missing += 1
            min_choice.append("NAT")
            max_choice.append("NAT")
        else:
            min_choice.append("RW{}".format(V.argmin_window(sc)))
            max_choice.append("RW{}".format(V.argmax_window(sc)))
    return ({"rand": rand_choice, "min": min_choice, "max": max_choice},
            {"n_p3_missing": int(n_missing),
             "random_window_hist": {k: rand_choice.count(k)
                                    for k in sorted(set(rand_choice))},
             "min_window_hist": {k: min_choice.count(k) for k in sorted(set(min_choice))},
             "max_window_hist": {k: max_choice.count(k) for k in sorted(set(max_choice))}})


# ------------------------------------------------------------------------- statistics
def acc_of(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


def paired_bootstrap(y, preds_a, preds_b, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """Paired ITEM bootstrap of (arm - floor), on the SAME estimand as the bar.

    preds_* are dicts seed -> per-item prediction vectors.  Both the point estimate and
    every replicate are the 3-SEED MEAN of the pooled metric, which is exactly the
    quantity the +0.050 bar is stated on -- an earlier draft bootstrapped the macro-F1
    of the 3-seed MAJORITY prediction, a different estimand from the bar's mean of
    per-seed macro-F1.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    ss = sorted(preds_a)

    def stat(idx):
        ya = y[idx]
        da = float(np.mean([acc_of(ya, preds_a[k][idx]) - acc_of(ya, preds_b[k][idx])
                            for k in ss]))
        dm = float(np.mean([M.macro_f1(ya, preds_a[k][idx]) - M.macro_f1(ya, preds_b[k][idx])
                            for k in ss]))
        return da, dm

    full = np.arange(n)
    d_acc, d_mf1 = stat(full)
    da = np.empty(b)
    dm = np.empty(b)
    for t in range(b):
        da[t], dm[t] = stat(rng.integers(0, n, n))
    return {"delta_acc": d_acc, "delta_mf1": d_mf1,
            "estimand": "3-seed mean of the pooled metric, identical to the bar",
            "acc_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "mf1_ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
            "acc_p_one_sided": float((np.sum(da <= 0) + 1) / (b + 1)),
            "mf1_p_one_sided": float((np.sum(dm <= 0) + 1) / (b + 1)),
            "b": b, "seed": seed}


def holm(pvals, alpha=ALPHA):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for r, (k, p) in enumerate(items):
        thr = alpha / (m - r)
        rej = still and (p <= thr)
        if not rej:
            still = False
        out[k] = {"p": float(p), "threshold": float(thr), "reject_null": bool(rej)}
    return out


def krr_length_probe(K_oof, lengths, fold_of, ridge=KRR_RIDGE):
    """Strict-OOF RBF kernel-ridge prediction of log1p(native text length).

    Frozen: gamma = 1/d, ridge = 1, the same 5 folds.  Row i uses the head of item i's
    OWN fold, so no row is predicted by a head that trained on it.  This is the declared
    length-predictability INSTRUMENT baseline; the contraction comparison it feeds
    belongs to Stage-1, not to A0.

    ONE DECLARED REPAIR to the C02_EXPERIMENT_PLAN wording.  The plan said
    "gamma = 1/d, ridge = 1" on the representation.  gamma = 1/d is the sklearn
    convention for PER-DIMENSION STANDARDISED features, where squared distances are
    about 2d and gamma * dist^2 is about 2.  On L2-normalised 1024-d keys squared
    distances lie in [0, 4], so gamma * dist^2 < 0.004, the kernel is numerically
    constant and the probe is uninformative BY CONSTRUCTION (measured: R^2 ~ 0.009 on
    a synthetic planted signal).  Features are therefore z-scored on the FITTING fold
    only -- the assumption under which gamma = 1/d is meaningful -- and gamma and ridge
    are otherwise untouched.  No parameter is tuned and nothing is selected on the
    held-out fold.
    """
    X = np.asarray(K_oof, dtype="float64")
    y = np.log1p(np.asarray(lengths, dtype="float64"))
    gamma = 1.0 / X.shape[1]
    pred = np.zeros_like(y)
    for f in range(P.K_FOLDS):
        ho = np.flatnonzero(fold_of == f)
        fit = np.flatnonzero(fold_of != f)
        mu_x = X[fit].mean(0)
        sd_x = X[fit].std(0)
        sd_x[sd_x == 0] = 1.0
        Xf, Xh = (X[fit] - mu_x) / sd_x, (X[ho] - mu_x) / sd_x
        sf = np.sum(Xf ** 2, 1)
        sq_ff = sf[:, None] + sf[None, :] - 2 * Xf @ Xf.T
        sq_hf = np.sum(Xh ** 2, 1)[:, None] + sf[None, :] - 2 * Xh @ Xf.T
        Kff = np.exp(-gamma * np.maximum(sq_ff, 0.0))
        Khf = np.exp(-gamma * np.maximum(sq_hf, 0.0))
        mu = y[fit].mean()
        alpha = np.linalg.solve(Kff + ridge * np.eye(len(fit)), y[fit] - mu)
        pred[ho] = Khf @ alpha + mu
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"r2_oof": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "gamma": float(gamma), "ridge": float(ridge), "n": int(len(y))}


def spearman(a, b):
    """Tie-corrected Spearman (scipy, average ranks).  Text lengths tie heavily on short
    transcripts, so plain argsort-of-argsort ranks would be wrong here."""
    r = spearmanr(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")).statistic
    return float(r) if np.isfinite(r) else 0.0


def parity_native(arms, lab, fit, ho, preds, nbrs, sims, tag):
    """PARITY-NAT: the {NAT} orbit must reproduce the FROZEN deployed vote.

    Predictions and the sorted top-20 similarity vector must be bit-equal ON EVERY ROW,
    tied or not.  Only neighbour IDs are exempted, and only on rows whose top-20
    similarities are not all distinct, because tie ORDER is a faiss heap detail.

    STATED PRECISELY: tie order is NOT vote-invariant in general -- two tied neighbours
    with different labels at adjacent ranks move the vote by
    2*s*(w_r - w_{r+1})/sum(w).  The exemption is safe not because the vote cannot move
    but because predictions AND sorted similarities are bit-checked on EVERY row, so a
    tie-induced vote flip HALTs rather than passing silently.  It is genuinely
    vote-invariant in the one case actually reachable here, the structural all-zero
    query, where every similarity is 0.  The count of exempted rows is reported.
    """
    _, pr_ref, I_ref, sim_ref = M.deployed_vote(arms["NATIVE"][0][fit], lab[fit],
                                                arms["NATIVE"][0][ho])
    if not np.array_equal(pr_ref, preds["NATIVE"][ho]):
        halt(HALT["PARITY"], "{}: NAT predictions != deployed_vote".format(tag))
    if not np.array_equal(sim_ref, sims["NATIVE"][ho]):
        halt(HALT["PARITY"], "{}: NAT top-20 similarities != deployed_vote".format(tag))
    tie = np.array([len(np.unique(sim_ref[i])) < TOPK for i in range(sim_ref.shape[0])])
    ok = ~tie
    if ok.any() and not np.array_equal(fit[I_ref][ok], nbrs["NATIVE"][ho][ok]):
        halt(HALT["PARITY"], "{}: NAT neighbour ids != deployed_vote".format(tag))
    return int(tie.sum())



# ------------------------------------------------------------------- in-job self-test
def oracle_self_test():
    """Fail-closed self-test of the oracle on SYNTHETIC arrays.

    No project cache, model, label, video or test path is touched.  It runs before any
    real data is opened so a numerical-contract break costs seconds, not a queue slot.
    """
    rng = np.random.default_rng(0)
    d, nb, nq = 64, 120, 40
    bank = rng.standard_normal((nb, d)).astype("float32")
    qry = rng.standard_normal((nq, d)).astype("float32")
    lab = (rng.random(nb) < 0.4).astype(int)
    cases = []

    # 1. PARITY: a singleton orbit IS the deployed vote, bit-exactly
    v1, p1, o1, s1 = orbit_vote([bank.copy()], lab, [qry.copy()])
    v2, p2, i2, s2 = M.deployed_vote(bank.copy(), lab, qry.copy())
    assert np.array_equal(p1, p2), "singleton orbit predictions != deployed_vote"
    assert np.array_equal(s1, s2), "singleton orbit similarities != deployed_vote"
    assert np.array_equal(o1, i2), "singleton orbit neighbour ids != deployed_vote"
    assert np.max(np.abs(v1 - v2)) == 0.0, "singleton orbit votes != deployed_vote"
    cases.append("parity_singleton_bit_exact")

    # 2. a structural all-zero query: every similarity ties at 0, the vote is 0, and the
    #    prediction is invariant to tie order -- the justification for the PARITY-NAT
    #    neighbour-id tie exemption, verified rather than assumed
    q0 = qry.copy()
    q0[3] = 0.0
    v3, p3_, o3, s3 = orbit_vote([bank.copy()], lab, [q0.copy()])
    v4, p4, i4, s4 = M.deployed_vote(bank.copy(), lab, q0.copy())
    assert np.all(s3[3] == 0.0) and v3[3] == 0.0, "zero query did not tie at zero"
    assert np.array_equal(p3_, p4) and np.array_equal(s3, s4), "zero-query parity broke"
    cases.append("zero_query_tie_invariant")

    # 3. the k = topk per view pair search is exact for the top-20 item set
    V1 = bank
    V2 = bank + 0.05 * rng.standard_normal((nb, d)).astype("float32")
    Q1 = qry
    Q2 = qry + 0.05 * rng.standard_normal((nq, d)).astype("float32")
    _, _, o5, s5 = orbit_vote([V1.copy(), V2.copy()], lab, [Q1.copy(), Q2.copy()])

    def _l2(X):
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        return X / nn

    brute = np.max(np.stack([_l2(np.float32(q)) @ _l2(np.float32(b)).T
                             for q in (Q1, Q2) for b in (V1, V2)]), axis=0)
    top = np.sort(brute, axis=1)[:, ::-1][:, :TOPK]
    assert np.allclose(np.sort(s5, axis=1)[:, ::-1], top, atol=1e-5, rtol=0), \
        "s_Q values disagree with a brute-force max over view pairs"
    bt = np.argsort(-brute, axis=1)[:, :TOPK]
    assert all(set(bt[i].tolist()) == set(o5[i].tolist()) for i in range(nq)), \
        "k=topk-per-pair search missed a top-20 item"
    cases.append("multiview_topk_exact")

    # 4. the derangement never maps across the bank/query boundary
    g0 = np.arange(0, 70)
    g1 = np.arange(70, 120)
    perm = derangement_within([g0, g1], 120, SHUFFLE_SEED, 0)
    assert np.sum(perm == np.arange(120)) == 0, "derangement left a fixed point"
    assert set(perm[g0].tolist()) == set(g0.tolist()), "donor crossed into the other group"
    assert set(perm[g1].tolist()) == set(g1.tolist()), "donor crossed into the other group"
    cases.append("within_partition_derangement")

    # 5. degeneracy-matched grouping keeps the partition boundary absolute
    dm = np.zeros(120, bool)
    dm[np.arange(0, 120, 7)] = True
    grp, ndrop = shuffle_groups(g0, g1, dm)
    assert ndrop == 0 and sum(g.size for g in grp) == 120
    for g in grp:
        assert set(g.tolist()) <= set(g0.tolist()) or set(g.tolist()) <= set(g1.tolist()), \
            "a donor group straddles the bank/query boundary"
    dm1 = np.zeros(120, bool)
    dm1[3] = True                       # a singleton degenerate class must be DROPPED
    grp1, ndrop1 = shuffle_groups(g0, g1, dm1)
    assert ndrop1 == 1, "a singleton class group must be dropped, not merged"
    assert sum(g.size for g in grp1) == 119, "the dropped item must leave the donor pool"
    assert all(3 not in g.tolist() for g in grp1), "the dropped item must not be regrouped"
    cases.append("degeneracy_matched_groups")

    # 6. SHUFFLE donates DISPLACEMENT, never position -- exercised THROUGH build_arms,
    #    so the test fails if the arm construction itself regresses.  (An earlier draft
    #    re-typed the formula locally and asserted an algebraic identity of its own line,
    #    which would have passed unchanged had SHUFFLE still donated absolute position.)
    m = 10
    tk = {"NAT": rng.standard_normal((m, 8)).astype("float32")}
    for vname in V.NON_NATIVE_VIEWS:
        tk[vname] = (tk["NAT"] + rng.standard_normal((m, 8)).astype("float32"))
    tch = {"rand": ["RW1"] * m, "min": ["RW2"] * m, "max": ["RW3"] * m}
    tperm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    tarms = build_arms(tk, tch, tperm)
    assert np.array_equal(tarms["SHUFFLE"][0], tk["NAT"]), "SHUFFLE moved the native view"
    for vi, vname in enumerate(V.NON_NATIVE_VIEWS, start=1):
        got = tarms["SHUFFLE"][vi].astype("float64")
        want = (tk["NAT"].astype("float64")
                + tk[vname].astype("float64")[tperm]
                - tk["NAT"].astype("float64")[tperm])
        assert np.allclose(got, want, atol=1e-5), \
            "SHUFFLE {} is not the donor DISPLACEMENT applied to the own native key".format(vname)
        assert not np.allclose(got, tk[vname][tperm], atol=1e-5), \
            "SHUFFLE {} leaked the donor's ABSOLUTE position".format(vname)
    assert len(tarms["FULL"]) == len(V.VIEW_NAMES) and len(tarms["NATIVE"]) == 1
    assert np.array_equal(tarms["NOISE"][0], tk["NAT"])
    for vi in range(1, len(V.VIEW_NAMES)):
        dn = np.linalg.norm(tarms["NOISE"][vi].astype("float64") - tk["NAT"], axis=1)
        df = np.linalg.norm(tarms["FULL"][vi].astype("float64") - tk["NAT"], axis=1)
        assert np.allclose(dn, df, atol=1e-4), "NOISE is not norm-matched to FULL"
    cases.append("shuffle_donates_displacement_via_build_arms")
    return cases


def _final_diagnostics(n, lab, lengths, nat_path, view_paths, man_path):
    return {"n_train": int(n), "pos_rate": float(lab.mean()),
            "text_length_chars": {"median": float(np.median(lengths)),
                                  "p90": float(np.percentile(lengths, 90)),
                                  "max": float(np.max(lengths))},
            "lambda_selection": "NOT APPLICABLE AT A0 -- the A0 oracle has no "
                                "lambda_orbit. lambda_orbit first exists at Stage-1, "
                                "where it is selected inside outer-train folds only.",
            "native_cache_sha256": sha256_of(nat_path),
            "view_cache_sha256": {v: sha256_of(view_paths[v]) for v in V.VIEW_NAMES},
            "extract_manifest_sha256": sha256_of(man_path)}


# ------------------------------------------------------------------------ per dataset
def run_dataset(ds, mintdir):
    cfg = P.DATASETS[ds]
    cache_dir, model_name, ds_name = cfg["cache_dir"], cfg["model"], cfg["ds"]
    res = {"dataset": ds, "ds": ds_name, "encoder_model": model_name,
           "gates": {}, "seeds": {}, "diagnostics": {}}

    # ---------------------------------------------------------------- raw caches
    nat_path = guard_path(os.path.join(cache_dir, "train_{}.pt".format(model_name)))
    nat = torch.load(nat_path, map_location="cpu", weights_only=False)
    ids = nat["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    ids = [str(x) for x in ids]
    n = len(ids)
    lab = np.asarray(nat["labels"]).astype(int)
    img_raw = nat["img_feats"].float().numpy().astype("float64")
    txt_banked = nat["text_feats"].float().numpy().astype("float64")

    view_raw, view_paths = {}, {}
    for v in V.VIEW_NAMES:
        p = guard_path(os.path.join(cache_dir,
                                    "train_{}-c02den-{}.pt".format(model_name, v)))
        d = torch.load(p, map_location="cpu", weights_only=False)
        if d.get("c02_view") != v:
            halt(HALT["SCHEMA"], "view tag mismatch in {}".format(p))
        if "img_feats" in d:
            halt(HALT["SCHEMA"], "view file must not carry img_feats: {}".format(p))
        vids = d["ids"]
        if isinstance(vids, list) and len(vids) == 1 and isinstance(vids[0], list):
            vids = vids[0]
        if [str(x) for x in vids] != ids:
            halt(HALT["SCHEMA"], "ID/order mismatch vs native bank in {}".format(p))
        if not np.array_equal(np.asarray(d["labels"]).astype(int), lab):
            halt(HALT["SCHEMA"], "label mismatch vs native bank in {}".format(p))
        view_raw[v] = d["text_feats"].float().numpy().astype("float64")
        view_paths[v] = p

    for name, X in ([("img_banked", img_raw), ("text_banked", txt_banked)]
                    + [("view_" + v, view_raw[v]) for v in V.VIEW_NAMES]):
        if not np.all(np.isfinite(X)):
            halt(HALT["NONFINITE"], "non-finite values in {}".format(name))

    # ---------------------------------------- ZERO CONTRACT (C01 criteria 1-4 applied)
    for nm, X in (("banked_img", img_raw), ("banked_text", txt_banked)):
        nrm0 = np.linalg.norm(X, axis=1)
        tiny0 = np.flatnonzero((nrm0 > 0.0) & (nrm0 <= TINY_NORM))
        if tiny0.size:
            halt(HALT["ZERO"], "{} has non-structural tiny rows {}".format(
                nm, tiny0.tolist()))
    zero_banked = np.flatnonzero(np.linalg.norm(txt_banked, axis=1) == 0.0)
    zrep = {"banked_text_zero_rows": zero_banked.tolist(),
            "banked_text_zero_ids": [ids[i] for i in zero_banked],
            "banked_text_zero_labels": lab[zero_banked].tolist(),
            "banked_img_zero_rows":
                np.flatnonzero(np.linalg.norm(img_raw, axis=1) == 0.0).tolist(),
            "per_view_zero_rows": {}}
    for v in V.VIEW_NAMES:
        nrm = np.linalg.norm(view_raw[v], axis=1)
        zv = np.flatnonzero(nrm == 0.0)
        zrep["per_view_zero_rows"][v] = zv.tolist()
        if not np.array_equal(zv, zero_banked):
            halt(HALT["ZERO"], "view {} zero mask {} != banked {}".format(
                v, zv.tolist(), zero_banked.tolist()))
        tiny = np.flatnonzero((nrm > 0.0) & (nrm <= TINY_NORM))
        if tiny.size:
            halt(HALT["ZERO"], "view {} has non-structural tiny rows {}".format(
                v, tiny.tolist()))
    zrep.update({
        "criterion_1_documented_structural_null": {
            "kind": "DOCUMENTARY_CITATION_NOT_COMPUTED",
            "evidence": "video-decode-failure zero-guard; "
                        "refine-logs/C01_ZERO_CONTRACT_PROBE.md and "
                        "refine-logs/PROVENANCE_AUDIT_2026-07-28.md:187-193"},
        "criterion_2_exact_zero_mask_match_across_all_arms": {
            "kind": "COMPUTED_AND_ASSERTED", "held": True},
        "criterion_3_no_nonstructural_tiny_rows": {
            "kind": "COMPUTED_AND_ASSERTED", "held": True, "tiny_norm": TINY_NORM},
        "criterion_4_same_baseline_consumed_the_row": {
            "kind": "DOCUMENTARY_CITATION_NOT_COMPUTED",
            "evidence": "the banked native cache used by the paired floor carries the "
                        "same zero rows, which criterion 2 verifies numerically"},
        "treatment": "retained in bank and queries, identically in every arm; a "
                     "sensitivity read excluding them is reported separately"})
    res["gates"]["ZERO_CONTRACT"] = zrep

    # -------------------------------------------------- GATE-EXT (extraction parity)
    nz = np.setdiff1d(np.arange(n), zero_banked)
    cos_nat = np.sum(l2n(view_raw["NAT"][nz]) * l2n(txt_banked[nz]), axis=1)
    ext = {"n_compared": int(len(nz)),
           "median_cos": float(np.median(cos_nat)),
           "min_cos": float(np.min(cos_nat)),
           "mean_cos": float(np.mean(cos_nat)),
           "max_abs_diff": float(np.max(np.abs(view_raw["NAT"][nz] - txt_banked[nz]))),
           "threshold_median_cos": EXT_PARITY_MEDIAN_COS_MIN}
    ext["pass"] = bool(ext["median_cos"] >= EXT_PARITY_MEDIAN_COS_MIN)
    res["gates"]["GATE_EXT_reextracted_NAT_vs_banked"] = ext
    if not ext["pass"]:
        halt(HALT["EXTRACT"], "re-extracted NAT median cos {:.6f} < {}".format(
            ext["median_cos"], EXT_PARITY_MEDIAN_COS_MIN))

    # -------------------------------------------------------- extraction manifest
    man_path = guard_path(os.path.join(
        REPO, "artifacts/c02_edq/v1/extract/C02-DEN-v1/manifest_{}.json".format(ds_name)))
    with open(man_path) as f:
        man = json.load(f)
    # The extractor recorded each view file's sha256 at write time.  Compare it with the
    # file on disk NOW, so a stale or swapped view cache with a matching id/label vector
    # cannot slip through the id-set check.
    written = man["splits"]["train"]["written"]
    for v in V.VIEW_NAMES:
        rec = written.get(v, {})
        if os.path.abspath(rec.get("path", "")) != os.path.abspath(view_paths[v]):
            halt(HALT["SCHEMA"], "manifest path for view {} is {}, opened {}".format(
                v, rec.get("path"), view_paths[v]))
        got = sha256_of(view_paths[v])
        if got != rec.get("sha256"):
            halt(HALT["SCHEMA"], "view {} sha256 {} != manifest {}".format(
                v, got, rec.get("sha256")))
    tr_man = man["splits"]["train"]
    per_item = {m["id"]: m for m in tr_man["per_item"]}
    if sorted(per_item) != sorted(ids):
        halt(HALT["SCHEMA"], "manifest id set != cache id set")
    lengths = np.array([per_item[i]["len_native"] for i in ids], dtype="float64")
    # A video-decode-failure (zero-guard) row is an identity orbit in EVERY space -- all
    # six views share one zero text vector and therefore one head key -- even though the
    # manifest's text-derived identity_views does not say so.  Without this it would sit
    # in the NON-degenerate donor class and receive a real donated displacement under
    # SHUFFLE while FULL leaves it untouched, which is exactly the asymmetry the
    # degeneracy-matched grouping exists to prevent, in the direction that makes
    # FULL > SHUFFLE easier.
    degen_text = np.array(
        [len(per_item[i]["identity_views"]) == len(V.NON_NATIVE_VIEWS) for i in ids])
    degen_zero = np.zeros(n, dtype=bool)
    degen_zero[zero_banked] = True
    degen_mask = degen_text | degen_zero
    shuffle_singletons_dropped = 0
    n_ident = int(degen_mask.sum())
    support = 1.0 - n_ident / float(n)
    res["gates"]["VIEW_SUPPORT"] = {
        "view_support": round(support, 6), "threshold": VIEW_SUPPORT_MIN,
        "n_full_identity_orbit": int(n_ident),
        "n_degenerate_items": int(tr_man["n_degenerate_items"]),
        "zero_guard_videos": int(tr_man["zero_guard_videos"]),
        "degenerate_causes": {c: sum(1 for i in ids
                                     if per_item[i]["degenerate"] == c)
                              for c in (V.DEGEN_EMPTY_TEXT, V.DEGEN_LENGTH_GUARD)},
        "pass": bool(support >= VIEW_SUPPORT_MIN)}
    if support < VIEW_SUPPORT_MIN:
        halt(HALT["SUPPORT"], "view support {:.4f} < {}".format(support,
                                                               VIEW_SUPPORT_MIN))

    p3, p3_path = load_p3_windows(ds_name)
    choices, cmeta = make_choices(ids, p3)
    res["diagnostics"]["p3_scores_path"] = p3_path
    res["diagnostics"]["view_selection"] = cmeta

    # ------------------------------------------------------------------ head arenas
    per_seed_pred = {a: {} for a in ARM_NAMES}
    fold_of_ref = None
    tie_rows_total = 0
    parity_cells = 0

    for seed in SEEDS:
        keys_by_fold, fit_by_fold, fold_of_seed = {}, {}, None
        for f in range(P.K_FOLDS):
            z = np.load(guard_path(os.path.join(
                mintdir, "mint_{}_s{}_f{}.npz".format(ds, seed, f))), allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            if not all(meta["fold_parity_vs_banked_vsw_ckpt"]):
                halt(HALT["FOLD"], "fold parity false in mint s{} f{}".format(seed, f))
            if fold_of_seed is None:
                fold_of_seed = z["fold_of"]
                if not np.array_equal(np.asarray(z["lab"]).astype(int), lab):
                    halt(HALT["SCHEMA"], "mint labels != native cache labels")
            fit_by_fold[f] = np.asarray(z["fit_idx"])
            keys_by_fold[f] = {v: z["K_" + v] for v in V.VIEW_NAMES}
        if fold_of_ref is None:
            fold_of_ref = fold_of_seed
        elif not np.array_equal(fold_of_ref, fold_of_seed):
            halt(HALT["FOLD"], "fold assignment differs across mints")

        preds = {a: np.full(n, -1, dtype=int) for a in ARM_NAMES}
        sims = {a: np.zeros((n, TOPK)) for a in ARM_NAMES}
        nbrs = {a: np.zeros((n, TOPK), dtype=int) for a in ARM_NAMES}
        K_oof = np.zeros_like(keys_by_fold[0]["NAT"])
        rad_oof = {v: np.zeros(n) for v in V.NON_NATIVE_VIEWS}

        for f in range(P.K_FOLDS):
            ho = np.flatnonzero(fold_of_ref == f)
            fit = fit_by_fold[f]
            if np.intersect1d(ho, fit).size:
                halt(HALT["ARENA"], "self-orbit exclusion violated in fold {}".format(f))
            kf = keys_by_fold[f]
            grp, ndropped = shuffle_groups(fit, ho, degen_mask)
            shuffle_singletons_dropped += ndropped
            perm = derangement_within(grp, n, SHUFFLE_SEED, f)
            arms = build_arms(kf, choices, perm)
            for a in ARM_NAMES:
                _, pr, order, sim = orbit_vote([X[fit] for X in arms[a]], lab[fit],
                                               [X[ho] for X in arms[a]])
                preds[a][ho] = pr
                sims[a][ho] = sim
                nbrs[a][ho] = fit[order]
            tie_rows_total += parity_native(arms, lab, fit, ho, preds, nbrs, sims,
                                            "head s{} fold {}".format(seed, f))
            parity_cells += 1
            K_oof[ho] = kf["NAT"][ho]
            natn = l2n(kf["NAT"].astype("float64"))
            for v in V.NON_NATIVE_VIEWS:
                rad_oof[v][ho] = 1.0 - np.sum(
                    l2n(kf[v].astype("float64"))[ho] * natn[ho], axis=1)

        if np.any(preds["NATIVE"] < 0):
            halt(HALT["SCHEMA"], "unfilled predictions in seed {}".format(seed))

        maj = float(max((lab == 0).mean(), (lab == 1).mean()))
        nat_acc = acc_of(lab, preds["NATIVE"])
        a2 = {"pooled_native_acc": nat_acc, "majority_rate": maj,
              "lower": maj + ARENA2_MARGIN, "upper": ARENA2_CEILING,
              "pass": bool(maj + ARENA2_MARGIN <= nat_acc <= ARENA2_CEILING)}
        res["gates"].setdefault("ARENA2", {})["seed{}".format(seed)] = a2
        if not a2["pass"]:
            halt(HALT["ARENA"], "seed {} pooled native acc {:.4f} outside [{:.4f}, {}]"
                 .format(seed, nat_acc, maj + ARENA2_MARGIN, ARENA2_CEILING))

        block = {"arms": {}}
        nat_mf1 = M.macro_f1(lab, preds["NATIVE"])
        for a in ARM_NAMES:
            fixed = int(np.sum((preds[a] == lab) & (preds["NATIVE"] != lab)))
            broken = int(np.sum((preds[a] != lab) & (preds["NATIVE"] == lab)))
            block["arms"][a] = {
                "acc": acc_of(lab, preds[a]),
                "macro_f1": M.macro_f1(lab, preds[a]),
                "delta_acc_vs_native": acc_of(lab, preds[a]) - nat_acc,
                "delta_mf1_vs_native": M.macro_f1(lab, preds[a]) - nat_mf1,
                "fixed": fixed, "broken": broken, "net_fix": fixed - broken,
                "net_fix_rate_IDENTICAL_TO_delta_acc": (fixed - broken) / float(n),
                "changed": int(np.sum(preds[a] != preds["NATIVE"])),
                "precision_on_changed": (
                    fixed / float(fixed + broken) if (fixed + broken) else None),
                "mean_top20_overlap_with_native": float(np.mean(
                    [len(np.intersect1d(nbrs[a][i], nbrs["NATIVE"][i]))
                     for i in range(n)])),
                "retrieval_length_spearman": spearman(
                    lengths, np.median(lengths[nbrs[a]], axis=1))}
            per_seed_pred[a][seed] = preds[a]
        block["orbit_radius_median_oof"] = float(np.median(
            np.stack([rad_oof[v] for v in V.NON_NATIVE_VIEWS])))
        block["orbit_radius_median_per_view_oof"] = {
            v: float(np.median(rad_oof[v])) for v in V.NON_NATIVE_VIEWS}
        block["krr_length_probe_oof_native_head_keys"] = krr_length_probe(
            K_oof, lengths, fold_of_ref)
        res["seeds"]["seed{}".format(seed)] = block

    res["diagnostics"]["shuffle_control"] = {
        "rule": "displacement donation: z_i^v := NAT_i + (view_v(pi(i)) - NAT_pi(i))",
        "grouping": "partition (fitting pool vs held-out fifth) x degeneracy class; the "
                    "partition boundary is never crossed. A class group of exactly one "
                    "member is DROPPED, so that item keeps its OWN displacement in "
                    "SHUFFLE. For the case that matters -- a lone DEGENERATE item, whose "
                    "displacement is ZERO by construction -- SHUFFLE is then EXACTLY "
                    "MATCHED to FULL for that item and it contributes to neither side of "
                    "the conjunct; merging would instead have handed it a real "
                    "displacement and made FULL > SHUFFLE EASIER. A zero-guard "
                    "(video-decode-failure) row counts as degenerate, because its orbit "
                    "is the identity in every space.",
        "n_singleton_class_groups_dropped_head_arena": int(shuffle_singletons_dropped),
        "H0_behaviour": "FULL and SHUFFLE are exchangeable under the EXCHANGEABILITY "
                        "null (displacements i.i.d. across items and independent of the "
                        "item). They are NOT exchangeable under every uninformative-orbit "
                        "null -- a purely radial displacement leaves FULL identical to "
                        "NATIVE while SHUFFLE still perturbs. FULL > SHUFFLE is therefore "
                        "NECESSARY, NOT SUFFICIENT, and it cannot manufacture a PASS: "
                        "under that radial null FULL == NATIVE so delta_acc = 0 and the "
                        "binding +0.050 bar is unreachable."}
    if parity_cells != len(SEEDS) * P.K_FOLDS:
        halt(HALT["PARITY"], "expected {} parity cells, checked {}".format(
            len(SEEDS) * P.K_FOLDS, parity_cells))
    res["gates"]["PARITY_NAT"] = {
        "rule": "orbit oracle on the {NAT} orbit must reproduce mechfix_ops.deployed_vote",
        "cells_checked_seed_x_fold": int(parity_cells),
        "cells_expected": int(len(SEEDS) * P.K_FOLDS),
        "predictions_and_sorted_similarities": (
            "BIT-EQUAL on every checked cell; a mismatch HALTs and the cell count is "
            "asserted equal to seeds x folds, so this line is reachable only when all "
            "{} cells passed".format(parity_cells)),
        "neighbour_id_rows_exempted_for_exact_float32_ties": tie_rows_total,
        "pass": True}

    # --------------------------------------------------------- 3-seed primary read
    nat_accs = [acc_of(lab, per_seed_pred["NATIVE"][s]) for s in SEEDS]
    nat_mf1s = [M.macro_f1(lab, per_seed_pred["NATIVE"][s]) for s in SEEDS]
    summary = {}
    for a in ARM_NAMES:
        accs = [acc_of(lab, per_seed_pred[a][s]) for s in SEEDS]
        mf1s = [M.macro_f1(lab, per_seed_pred[a][s]) for s in SEEDS]
        nets = [res["seeds"]["seed{}".format(s)]["arms"][a]["net_fix"] for s in SEEDS]
        summary[a] = {
            "acc_3seed_mean": float(np.mean(accs)),
            "mf1_3seed_mean": float(np.mean(mf1s)),
            "delta_acc_3seed_mean": float(np.mean(accs) - np.mean(nat_accs)),
            "delta_mf1_3seed_mean": float(np.mean(mf1s) - np.mean(nat_mf1s)),
            "per_seed_delta_acc": [float(x - y) for x, y in zip(accs, nat_accs)],
            "per_seed_delta_mf1": [float(x - y) for x, y in zip(mf1s, nat_mf1s)],
            "net_fix_3seed_mean": float(np.mean(nets)),
            "net_fix_rate_3seed_mean": float(np.mean(nets) / n),
            "net_fix_rate_note": "fixed - broken = n * delta_acc EXACTLY, so this is the "
                                 "accuracy delta re-expressed in items, not an "
                                 "independent quantity",
            "precision_on_changed_3seed_mean": float(np.mean(
                [res["seeds"]["seed{}".format(s2)]["arms"][a]["precision_on_changed"]
                 for s2 in SEEDS
                 if res["seeds"]["seed{}".format(s2)]["arms"][a]["precision_on_changed"]
                 is not None])) if any(
                     res["seeds"]["seed{}".format(s2)]["arms"][a]["precision_on_changed"]
                     is not None for s2 in SEEDS) else None}
    res["summary_3seed"] = summary
    res["bootstrap_FULL_vs_NATIVE"] = paired_bootstrap(
        lab, per_seed_pred["FULL"], per_seed_pred["NATIVE"])

    if zero_banked.size:
        keep = np.setdiff1d(np.arange(n), zero_banked)
        res["sensitivity_excluding_structural_nulls"] = {
            "n_excluded": int(zero_banked.size),
            "delta_acc_3seed_mean": float(np.mean(
                [acc_of(lab[keep], per_seed_pred["FULL"][s][keep])
                 - acc_of(lab[keep], per_seed_pred["NATIVE"][s][keep]) for s in SEEDS])),
            "delta_mf1_3seed_mean": float(np.mean(
                [M.macro_f1(lab[keep], per_seed_pred["FULL"][s][keep])
                 - M.macro_f1(lab[keep], per_seed_pred["NATIVE"][s][keep])
                 for s in SEEDS]))}
    else:
        res["sensitivity_excluding_structural_nulls"] = {"n_excluded": 0}

    # ------------------------------------------------------- SECONDARY raw-key arena
    raw_keys = {v: l2n(np.concatenate([l2n(img_raw), l2n(view_raw[v])],
                                      axis=1)).astype("float32")
                for v in V.VIEW_NAMES}
    # The raw arena is SECONDARY (F113: it may corroborate a KILL, never promote), so it
    # is not allowed to destroy a completed PRIMARY measurement: any Halt it raises is
    # caught and recorded, and the primary result stands.
    try:
        raw_preds = {a: np.full(n, -1, dtype=int) for a in ARM_NAMES}
        for f in range(P.K_FOLDS):
            ho = np.flatnonzero(fold_of_ref == f)
            fit = np.flatnonzero(fold_of_ref != f)
            raw_arms = build_arms(raw_keys, choices, derangement_within(
                shuffle_groups(fit, ho, degen_mask)[0], n, SHUFFLE_SEED, f))
            for a in ARM_NAMES:
                _, pr, _, _ = orbit_vote([X[fit] for X in raw_arms[a]], lab[fit],
                                         [X[ho] for X in raw_arms[a]])
                raw_preds[a][ho] = pr
    except Halt as e:
        res["secondary_raw_arena"] = {
            "status": "SECONDARY_ARENA_HALTED_PRIMARY_UNAFFECTED", "halt": str(e)}
        res["diagnostics"].update(_final_diagnostics(
            n, lab, lengths, nat_path, view_paths, man_path))
        return res
    rna, rnm = acc_of(lab, raw_preds["NATIVE"]), M.macro_f1(lab, raw_preds["NATIVE"])
    res["secondary_raw_arena"] = {
        "note": "raw fused key space l2n(concat(l2n(img), l2n(text_view))). SECONDARY "
                "under F113: may corroborate a KILL, may never promote a lead.",
        "native_acc": rna, "native_macro_f1": rnm,
        "arms": {a: {"acc": acc_of(lab, raw_preds[a]),
                     "macro_f1": M.macro_f1(lab, raw_preds[a]),
                     "delta_acc_vs_native": acc_of(lab, raw_preds[a]) - rna,
                     "delta_mf1_vs_native": M.macro_f1(lab, raw_preds[a]) - rnm}
                 for a in ARM_NAMES}}

    res["diagnostics"].update(_final_diagnostics(
        n, lab, lengths, nat_path, view_paths, man_path))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mintdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    for rel, want in FROZEN.items():
        got = sha256_of(os.path.join(REPO, rel))
        assert got == want, "FROZEN MODULE CHANGED: {} -> {}".format(rel, got)

    st = oracle_self_test()
    print("[c02arena] oracle self-test PASS: {}".format(", ".join(st)))

    with open(a.config) as f:
        cfg = json.load(f)

    os.makedirs(a.outdir, exist_ok=True)
    rp = os.path.join(a.outdir, "C02_A0_OUT.json")
    dp = os.path.join(a.outdir, "C02_A0_DECISION.json")
    assert not os.path.exists(rp), "NO-CLOBBER: {}".format(rp)
    assert not os.path.exists(dp), "NO-CLOBBER: {}".format(dp)

    t0 = time.time()
    out = {"schema_version": "c02_a0_result_v9",
           "run_id": cfg["run_id"],
           "config_path": a.config,
           "config_sha256": sha256_of(a.config),
           "arena_script_sha256": sha256_of(os.path.abspath(__file__)),
           "mint_script_sha256": sha256_of(
               os.path.join(REPO, "scripts/analysis/c02_a0_mint.py")),
           "view_module_sha256": sha256_of(
               os.path.join(REPO, "src/utils/c02_density_views.py")),
           "frozen_modules": FROZEN,
           "oracle_self_test_cases": st,
           "bars": {"acc": BAR_ACC, "macro_f1": BAR_MF1,
                    "net_fix_rate": BAR_NETFIX_RATE,
                    "view_support_min": VIEW_SUPPORT_MIN,
                    "alpha": ALPHA, "bootstrap_B": BOOTSTRAP_B},
           "datasets": {}}

    halted = None
    try:
        for ds in DATASETS:
            out["datasets"][ds] = run_dataset(ds, a.mintdir)
    except Halt as e:
        halted = str(e)

    out["seconds"] = round(time.time() - t0, 1)
    if halted:
        out["halt"] = halted
    tmp = rp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=jsonable)
    os.replace(tmp, rp)
    print("[c02arena] result -> {}".format(rp))

    dec = {"schema_version": "c02_a0_decision_v9", "run_id": cfg["run_id"],
           "bars": out["bars"]}
    if halted:
        dec["verdict"] = "HALT_FAIL_CLOSED_NO_DECISION"
        dec["halt"] = halted
        dec["target_met"] = False
        dec["result_exists"] = False
    else:
        per, pv = {}, {}
        for ds in DATASETS:
            s = out["datasets"][ds]["summary_3seed"]
            bs = out["datasets"][ds]["bootstrap_FULL_vs_NATIVE"]
            per[ds] = {
                "delta_acc": s["FULL"]["delta_acc_3seed_mean"],
                "delta_mf1": s["FULL"]["delta_mf1_3seed_mean"],
                "net_fix_rate_implied_by_acc_bar":
                    s["FULL"]["net_fix_rate_3seed_mean"],
                "precision_on_changed": s["FULL"]["precision_on_changed_3seed_mean"],
                "beats_shuffle_acc": bool(s["FULL"]["delta_acc_3seed_mean"]
                                          > s["SHUFFLE"]["delta_acc_3seed_mean"]),
                "beats_shuffle_mf1": bool(s["FULL"]["delta_mf1_3seed_mean"]
                                          > s["SHUFFLE"]["delta_mf1_3seed_mean"]),
                "beats_noise_acc": bool(s["FULL"]["delta_acc_3seed_mean"]
                                        > s["NOISE"]["delta_acc_3seed_mean"]),
                "beats_noise_mf1": bool(s["FULL"]["delta_mf1_3seed_mean"]
                                        > s["NOISE"]["delta_mf1_3seed_mean"]),
                "acc_ci_lower": bs["acc_ci95"][0], "mf1_ci_lower": bs["mf1_ci95"][0],
                "acc_p": bs["acc_p_one_sided"], "mf1_p": bs["mf1_p_one_sided"]}
            pv["{}_acc".format(ds)] = per[ds]["acc_p"]
            pv["{}_mf1".format(ds)] = per[ds]["mf1_p"]
        hol = holm(pv)
        dec["holm_family"] = hol
        dec["per_dataset"] = per
        # NOTE: net_fix_rate is retained as a defensive check and reported, but it is
        # ALGEBRAICALLY IDENTICAL to delta_acc (fixed - broken = n * delta_acc), so with
        # BAR_ACC = 0.050 > BAR_NETFIX_RATE = 0.030 it can never bind. The registry's
        # net-fix clause is discharged BY the accuracy bar, and this is stated rather
        # than dressed up as an independent gate.
        ok = all(per[ds]["delta_acc"] >= BAR_ACC and per[ds]["delta_mf1"] >= BAR_MF1
                 and per[ds]["net_fix_rate_implied_by_acc_bar"] >= BAR_NETFIX_RATE
                 and per[ds]["beats_shuffle_acc"] and per[ds]["beats_shuffle_mf1"]
                 and per[ds]["beats_noise_acc"] and per[ds]["beats_noise_mf1"]
                 and per[ds]["acc_ci_lower"] > 0 and per[ds]["mf1_ci_lower"] > 0
                 and hol["{}_acc".format(ds)]["reject_null"]
                 and hol["{}_mf1".format(ds)]["reject_null"]
                 for ds in DATASETS)
        dec["verdict"] = ("PASS_C02_DENSITY_ORBIT_REACHABLE" if ok
                          else "KILL_C02_DENSITY_ORBIT_UNREACHABLE")
        dec["target_met"] = False
        dec["result_exists"] = True
        dec["interpretation_boundary"] = (
            "A0 is the registry's Stage-0 REACHABILITY gate, instantiated as an "
            "optimistic max-matching oracle over the extracted density orbit in the "
            "deployed head key space of train-split fold heads. A PASS authorises "
            "Stage-1 design plus a fresh review only: it is not a training gain, not a "
            "development result and not a test result. A KILL is a GATE verdict under "
            "the registry's frozen Stage-0 rule -- s_Q is one particular orbit-invariant "
            "similarity, NOT a proven supremum over all orbit-contracting "
            "representations, so a KILL closes C02 under the rule rather than proving "
            "that no such representation could ever help.")
    tmp = dp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dec, f, indent=1, sort_keys=True, default=jsonable)
    os.replace(tmp, dp)
    print("[c02arena] decision -> {} : {}".format(dp, dec["verdict"]))
    if halted:
        # A fail-closed HALT must not be reported to SLURM as COMPLETED.  Both artifacts
        # are already written, so the exit code is the only remaining signal.
        raise SystemExit(3)


if __name__ == "__main__":
    main()
