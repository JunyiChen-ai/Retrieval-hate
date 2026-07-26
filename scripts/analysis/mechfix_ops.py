#!/usr/bin/env python
"""
mechfix_ops.py -- FROZEN eval-time decision-operator implementations for the
MECHFIX $0 pregate (refine-logs/MECHFIX_PREGATE_2026-07-27.md).

This file is the operator contract. Its sha256 is recorded in the record's §1 at
freeze time. Nothing here is tuned: every constant below is a literature default
or a copy of the deployed configuration, fixed before any test number was read.

DEPLOYED REFERENCE OPERATOR (src/utils/metrics.py:262-301 + src/model/evaluate_rac.py:405-465)
    keys       : the trained head's fused embedding, mlp[:-2](normalize(img_proj) * normalize(text_proj))
    retrieval  : faiss.IndexFlatIP over float32 L2-normalised keys, memory = own train split,
                 largest_retrieval = topk = 20, similarity_threshold = -1 (no truncation)
    vote       : v = sum_i (2*lab_i - 1) * cos_i * w_i / sum_i w_i,  w = [20, 19, ..., 1]
    decision   : predict 1 iff sigmoid(v) >= 0.5  <=>  v >= 0

ALL FIVE TREATMENT ARMS operate on the SAME deployed key space, at EVAL TIME ONLY,
globally and symmetrically over every item. No arm reads a test label, no arm makes
a per-item channel/branch choice, no arm trains anything.

ENGINE UNIFORMITY. Every arm gets its cosines from the same faiss.IndexFlatIP search
over float32 L2-normalised keys that the deployed path uses, so no arm-to-arm delta
can be an artefact of a different similarity engine. Arms whose ranking is not plain
cosine (T1 per-class quota, T2a adjusted similarity) obtain the full faiss similarity
row (k = n_bank) and then rank within it.
"""
import numpy as np

import faiss

TOPK = 20                 # deployed
T1_K_PER_CLASS = 10       # T1: 10 per class => 20 neighbours total, same budget as deployed
T2A_HUB_K = 10            # CSLS default neighbourhood for the hubness term (Lample et al. 2018)


# --------------------------------------------------------------------- helpers
def _norm32(X):
    """float32 C-contiguous copy, L2-normalised in place -- exactly the deployed
    pre-index step (evaluate_rac.py: faiss.normalize_L2 on .astype('float32'))."""
    Y = np.ascontiguousarray(np.asarray(X, dtype="float32"))
    faiss.normalize_L2(Y)
    return Y


def _flat_ip(bank32, query32, k):
    """faiss exact inner-product search; both inputs already L2-normalised."""
    ix = faiss.IndexFlatIP(bank32.shape[1])
    ix.add(bank32)
    return ix.search(query32, k)          # D (sims), I (bank indices)


def _rank_weights(k):
    return np.arange(1, k + 1)[::-1].astype("float64")


def macro_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    fs = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


# ------------------------------------------------------------- DEPLOYED (floor)
def deployed_vote(bank_keys, bank_lab, query_keys, topk=TOPK, exclude_self=False):
    """Bit-faithful replay of the deployed top-20 rank-weighted signed-cosine vote.

    exclude_self=True is used ONLY for train-side leave-one-out sanity checks
    (§3 of the record); it is never used on a test read.
    """
    b = _norm32(bank_keys); q = _norm32(query_keys)
    k = topk + (1 if exclude_self else 0)
    D, I = _flat_ip(b, q, k)
    if exclude_self:
        keep_I = np.empty((I.shape[0], topk), dtype=I.dtype)
        keep_D = np.empty((I.shape[0], topk), dtype=D.dtype)
        for i in range(I.shape[0]):
            m = I[i] != i
            keep_I[i] = I[i][m][:topk]
            keep_D[i] = D[i][m][:topk]
        I, D = keep_I, keep_D
    lab = np.asarray(bank_lab)[I].astype("float64")
    sim = D.astype("float64")
    w = _rank_weights(topk)
    votes = ((lab * 2 - 1) * sim * w).sum(1) / w.sum()
    return votes, (votes >= 0).astype(int), I, sim


# ------------------------------------------------------- T1 class-balanced vote
def t1_class_balanced(bank_keys, bank_lab, query_keys, k_per_class=T1_K_PER_CLASS):
    """Retrieve the top-k_per_class bank items of EACH class separately; score each
    class by its own rank-weighted cosine sum with w = [k..1] normalised; predict
    hate iff score_hate >= score_nonhate.

    Removes the neighbourhood's local class prior by construction: the neighbour
    count per class is fixed at k_per_class/k_per_class for every item, so the bank's
    length-conditional class base rate can no longer enter the decision.
    """
    b = _norm32(bank_keys); q = _norm32(query_keys)
    lab = np.asarray(bank_lab).astype(int)
    w = _rank_weights(k_per_class)
    scores = {}
    for c in (0, 1):
        idx = np.flatnonzero(lab == c)
        assert len(idx) >= k_per_class, f"class {c} has only {len(idx)} bank rows"
        D, _ = _flat_ip(np.ascontiguousarray(b[idx]), q, k_per_class)
        scores[c] = (D.astype("float64") * w).sum(1) / w.sum()
    margin = scores[1] - scores[0]
    return margin, (margin >= 0).astype(int), scores[1], scores[0]


# ------------------------------------------------------ T2a CSLS hubness correction
def bank_hubness(bank_keys, hub_k=T2A_HUB_K):
    """r(x) = mean cosine of bank item x to its hub_k nearest OTHER bank items.
    Bank-side only, train-only, precomputed once -- no query and no label involved."""
    b = _norm32(bank_keys)
    D, I = _flat_ip(b, b, hub_k + 1)
    r = np.empty(b.shape[0], dtype="float64")
    for i in range(b.shape[0]):
        m = I[i] != i
        r[i] = D[i][m][:hub_k].astype("float64").mean()
    return r


def t2a_csls(bank_keys, bank_lab, query_keys, r, topk=TOPK):
    """adjusted sim(q,x) = 2*cos(q,x) - r(x); take the top-topk under the adjusted
    similarity, then the DEPLOYED signed rank-weighted vote using the adjusted sims."""
    b = _norm32(bank_keys); q = _norm32(query_keys)
    n = b.shape[0]
    D, I = _flat_ip(b, q, n)                       # full faiss similarity rows
    lab = np.asarray(bank_lab).astype("float64")
    w = _rank_weights(topk)
    votes = np.empty(q.shape[0], dtype="float64")
    nb_idx = np.empty((q.shape[0], topk), dtype="int64")
    nb_sim = np.empty((q.shape[0], topk), dtype="float64")
    for i in range(q.shape[0]):
        adj = 2.0 * D[i].astype("float64") - r[I[i]]
        order = np.argsort(-adj, kind="stable")[:topk]
        sel = I[i][order]
        s = adj[order]
        nb_idx[i] = sel
        nb_sim[i] = s
        votes[i] = ((lab[sel] * 2 - 1) * s * w).sum() / w.sum()
    return votes, (votes >= 0).astype(int), nb_idx, nb_sim


# ------------------------------------------------------------- T2b whitened keys
def fit_whitener(train_keys):
    """Mean + inverse square root of the Ledoit-Wolf shrinkage covariance of the
    TRAIN bank keys. sklearn's LedoitWolf picks its own shrinkage coefficient in
    closed form from the data -- there is no hyper-parameter to tune."""
    from sklearn.covariance import LedoitWolf
    X = np.asarray(train_keys, dtype="float64")
    mu = X.mean(0)
    lw = LedoitWolf(assume_centered=True).fit(X - mu)
    S = lw.covariance_
    ev, V = np.linalg.eigh(S)
    ev = np.clip(ev, 1e-12, None)
    W = (V / np.sqrt(ev)) @ V.T                    # S^{-1/2}, symmetric
    return mu, W, float(lw.shrinkage_), ev


def apply_whitener(X, mu, W):
    """x -> W (x - mu), then L2-renormalise (the renorm is done by the vote engine's
    normalize_L2; we return the un-normalised transform and let _norm32 do it, which
    is the same thing)."""
    return ((np.asarray(X, dtype="float64") - mu) @ W.T).astype("float32")


# ------------------------------------------------- T3 nuisance-direction removal
def fit_length_direction(train_keys, length_scalar):
    """v = least-squares regression direction of log(1 + transcript_volume) on the
    TRAIN bank keys (train only, 1 dimension, no labels anywhere). d > n for our
    banks, so lstsq returns the minimum-norm solution. Returns the unit direction."""
    X = np.asarray(train_keys, dtype="float64")
    y = np.asarray(length_scalar, dtype="float64")
    Xc = X - X.mean(0)
    yc = y - y.mean()
    v, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    nv = np.linalg.norm(v)
    assert nv > 0, "degenerate length direction"
    return v / nv


def remove_direction(X, vhat):
    """x -> x - (x . vhat) vhat, then L2-renormalise (renorm via the vote engine)."""
    A = np.asarray(X, dtype="float64")
    return (A - np.outer(A @ vhat, vhat)).astype("float32")
