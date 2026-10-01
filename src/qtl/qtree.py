"""Query tree (README sections 2.1, 2.2, 2.4): the binary partition tree of a video's 1-fps timeline, the
zero-inflated OR-tree model over it, exact inference in the log domain, and the expected-information-gain policy.

Model. G in {0, 1} (video harmful), P(G=1) = g. Given G = 0 every second is 0; given G = 1 the seconds are
independent Bernoulli(pi_t) conditioned on at least one being 1. A node's state z_n = OR of its seconds. The VLM
answer o_n (5 categories x 4 levels) depends only on s_n in {0: G=0, 1: G=1 and z_n=0, 2: G=1 and z_n=1}; the five
categories are independent given s_n: P(o_nk = l | s, len) = softmax_l(theta[k, s, l] + omega[k, s, l] * u_n),
u_n = standardised log length of the node.

Inference. The OR factor links a node to its two children only, so sum-product on the tree is exact. Upward pass
(normalised): for each node u_n = P(z_n = 1 | answers in its subtree) under the independent prior, and the log
normaliser lz_n; U_n(1) = exp(lz_n) u_n, U_n(0) = exp(lz_n) (1 - u_n). log P(o | G=1) = lz_root + log u_root -
log P(root = 1) (the "at least one" condition). Downward pass: outside messages with the root fixed to 1 give the
node and second marginals under G = 1; P(G = 1 | o) mixes the two hypotheses.
"""
from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

F_FRAMES = 4                       # frames per question; queryable nodes are >= F_FRAMES seconds long
N_CAT, N_LEV, N_STATE = 5, 4, 3
CATEGORIES = ("hate", "harassment", "sexual", "violence", "self-harm")
NEG = -1e30


# ------------------------------------------------------------------ tree structure
@functools.lru_cache(maxsize=4096)
def tree(T):
    """Binary partition tree of [0, T) down to single seconds, breadth-first. Node [a, b) splits at
    a + (b - a) // 2. Returns a dict of int64 arrays (length 2T - 1): a, b, depth, left, right (-1 for leaves),
    parent (-1 for the root), and `index` {(a, b): node id}."""
    T = int(T)
    assert T >= 1
    a, b, depth, parent = [0], [T], [0], [-1]
    left, right = [], []
    i = 0
    while i < len(a):
        if b[i] - a[i] >= 2:
            m = a[i] + (b[i] - a[i]) // 2
            left.append(len(a))
            a.append(a[i]); b.append(m); depth.append(depth[i] + 1); parent.append(i)
            right.append(len(a))
            a.append(m); b.append(b[i]); depth.append(depth[i] + 1); parent.append(i)
        else:
            left.append(-1)
            right.append(-1)
        i += 1
    out = {k: np.asarray(v, dtype=np.int64) for k, v in
           (("a", a), ("b", b), ("depth", depth), ("left", left), ("right", right), ("parent", parent))}
    out["index"] = {(int(x), int(y)): n for n, (x, y) in enumerate(zip(a, b))}
    out["queryable"] = (out["b"] - out["a"]) >= F_FRAMES
    return out



def level_nodes(T, L):
    """Diagnostic arm (README section 12): the nodes of ONE depth of the tree of [0, T), i.e. equal-length
    non-overlapping windows. The depth is the deepest one whose nodes are all at least L seconds long (node lengths
    at depth d are floor or ceil of T / 2^d, so they lie in [L, 2L + 1)); the root when T < 2L."""
    T = int(T)
    d = 0
    while T // (2 ** (d + 1)) >= L:
        d += 1
    return np.where(tree(T)["depth"] == d)[0]

def leaf_levels(tr):
    """Per depth d (deepest first): node ids ordered leaves first, then internal nodes; and each node's
    position inside its level."""
    D = int(tr["depth"].max())
    levels = []
    pos = np.empty(len(tr["a"]), dtype=np.int64)
    for d in range(D + 1):
        nd = np.where(tr["depth"] == d)[0]
        is_leaf = tr["left"][nd] < 0
        nd = np.concatenate([nd[is_leaf], nd[~is_leaf]])
        pos[nd] = np.arange(len(nd))
        levels.append((nd, int(is_leaf.sum())))
    return levels, pos


# ------------------------------------------------------------------ answer model
class AnswerModel(nn.Module):
    """log P(o_n | s_n, u_n) for the three states; theta, omega of shape (N_CAT, N_STATE, N_LEV)."""

    def __init__(self, theta0, len_mu, len_sd, length_term=True, n_state=N_STATE, categories=tuple(range(N_CAT)),
                 omega0=None):
        super().__init__()
        self.cats = list(categories)
        self.theta = nn.Parameter(torch.as_tensor(theta0, dtype=torch.float32).clone())
        om = torch.zeros_like(self.theta) if omega0 is None else torch.as_tensor(omega0, dtype=torch.float32).clone()
        self.omega = nn.Parameter(om) if length_term else None
        self.len_mu, self.len_sd = float(len_mu), float(len_sd)
        self.n_state = n_state

    def u(self, length):
        return (torch.log(torch.as_tensor(length, dtype=torch.float32)) - self.len_mu) / self.len_sd

    def level_logp(self, length):
        """(N, N_CAT, N_STATE, N_LEV) log P(o_k = l | s, length) for node lengths (N,)."""
        u = self.u(length).to(self.theta.device)
        logits = self.theta[None].expand(u.shape[0], -1, -1, -1)
        if self.omega is not None:
            logits = logits + self.omega[None] * u[:, None, None, None]
        return F.log_softmax(logits, dim=-1)

    def loglik(self, answers, length):
        """(N, 3) log P(o_n | s) for integer answers (N, N_CAT) in 0..3."""
        lp = self.level_logp(length)                                     # N, C, S, L
        idx = answers.long().to(lp.device)[:, :, None, None].expand(-1, -1, lp.shape[2], 1)
        per = torch.gather(lp, 3, idx).squeeze(-1)                       # N, C, S
        ll = per[:, self.cats].sum(1)                                    # N, S (arm hate_only: category 0 only)
        if self.n_state == 2:                                            # arm two_state: s=1 shares s=0's table
            ll = torch.stack([ll[:, 0], ll[:, 0], ll[:, 1]], dim=1)
        return ll


def init_theta(train_answers, n_state=N_STATE):
    """Data-driven start: state 0 (and 1) = level frequencies of all answers of negative training videos;
    state 2 = frequencies of the root answers of positive training videos (their root is harmful by definition).
    Laplace smoothing. train_answers: list of (label, [(a, b, T, answer vector)]) per video."""
    neg = np.ones((N_CAT, N_LEV))
    root = np.ones((N_CAT, N_LEV))
    for label, nodes in train_answers:
        for a, b, T, o in nodes:
            if o is None:
                continue
            if label == 0:
                neg[np.arange(N_CAT), o] += 1
            elif a == 0 and b == T:
                root[np.arange(N_CAT), o] += 1
    ln = np.log(neg / neg.sum(1, keepdims=True))
    lr = np.log(root / root.sum(1, keepdims=True))
    if n_state == 2:
        return np.stack([ln, lr], axis=1)
    return np.stack([ln, ln, lr], axis=1)                                # C, S, L


def fit_weighted(O, U, W, length_term=True):
    """Multinomial logistic per category of the level on the standardised log length, maximum weighted likelihood
    (weights W per answer) with one Laplace pseudo-count per level at u = 0. O (n, N_CAT) int, U (n,), W (n,).
    Returns theta, omega (N_CAT, N_LEV)."""
    O = torch.as_tensor(O, dtype=torch.long)
    U = torch.as_tensor(U, dtype=torch.float64)
    W = torch.as_tensor(W, dtype=torch.float64)
    th = torch.zeros(N_CAT, N_LEV, dtype=torch.float64, requires_grad=True)
    om = torch.zeros(N_CAT, N_LEV, dtype=torch.float64, requires_grad=bool(length_term))
    opt = torch.optim.LBFGS([th, om] if length_term else [th], max_iter=500, tolerance_grad=1e-10,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lp = F.log_softmax(th[None] + om[None] * U[:, None, None], dim=-1)
        nll = -(W[:, None] * lp.gather(2, O[:, :, None]).squeeze(-1)).sum() - F.log_softmax(th, dim=-1).sum()
        nll.backward()
        return nll

    opt.step(closure)
    return th.detach().numpy(), om.detach().numpy()


def fit_anchored(train_answers, len_mu, len_sd, length_term=True):
    """Two-state answer model fitted only on the training answers whose state is known from the video label
    (README section 8): every node of a negative video is state 0 (no harmful second); the root of a positive
    video is state 1 (it contains the harmful seconds). Per category and state, a multinomial logistic regression
    of the level on the standardised log length (theta + omega * u), maximum likelihood with one Laplace
    pseudo-count per level at u = 0. The answer model is then fixed: no latent variable enters its estimate.
    Returns theta, omega (N_CAT, 2, N_LEV) and the number of answers per state."""
    rows = {0: ([], []), 1: ([], [])}
    for label, nodes in train_answers:
        for a, b, T, o in nodes:
            if o is None:
                continue
            if label == 0:
                s = 0
            elif a == 0 and b == T:
                s = 1
            else:
                continue
            rows[s][0].append(o)
            rows[s][1].append((np.log(b - a) - len_mu) / len_sd)
    theta, omega = np.zeros((N_CAT, 2, N_LEV)), np.zeros((N_CAT, 2, N_LEV))
    for s in (0, 1):
        O = torch.as_tensor(np.array(rows[s][0]), dtype=torch.long)
        U = torch.as_tensor(np.array(rows[s][1]), dtype=torch.float64)
        th = torch.zeros(N_CAT, N_LEV, dtype=torch.float64, requires_grad=True)
        om = torch.zeros(N_CAT, N_LEV, dtype=torch.float64, requires_grad=bool(length_term))
        opt = torch.optim.LBFGS([th, om] if length_term else [th], max_iter=500, tolerance_grad=1e-10,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lp = F.log_softmax(th[None] + om[None] * U[:, None, None], dim=-1)
            nll = -lp.gather(2, O[:, :, None]).sum() - F.log_softmax(th, dim=-1).sum()
            nll.backward()
            return nll

        opt.step(closure)
        theta[:, s], omega[:, s] = th.detach().numpy(), om.detach().numpy()
    return theta, omega, {s: len(rows[s][0]) for s in (0, 1)}


# ------------------------------------------------------------------ batched upward pass (training, autograd)
class TreeBatch:
    """Level structure for a batch of videos (lengths Ts) and the observed nodes (obs[i] = (node ids, answers
    (n, N_CAT), lengths (n,)) or empty)."""

    def __init__(self, Ts, obs, Tmax):
        self.B = len(Ts)
        trees = [tree(int(T)) for T in Ts]
        offs = np.cumsum([0] + [len(t["a"]) for t in trees])
        self.N = int(offs[-1])
        depth = np.concatenate([t["depth"] for t in trees])
        left = np.concatenate([np.where(t["left"] >= 0, t["left"] + o, -1) for t, o in zip(trees, offs[:-1])])
        right = np.concatenate([np.where(t["right"] >= 0, t["right"] + o, -1) for t, o in zip(trees, offs[:-1])])
        a = np.concatenate([t["a"] for t in trees])
        vid = np.concatenate([np.full(len(t["a"]), i) for i, t in enumerate(trees)])
        self.roots = offs[:-1].astype(np.int64)
        D = int(depth.max())
        pos = np.empty(self.N, dtype=np.int64)
        self.levels = []
        for d in range(D + 1):
            nd = np.where(depth == d)[0]
            is_leaf = left[nd] < 0
            nd = np.concatenate([nd[is_leaf], nd[~is_leaf]])
            pos[nd] = np.arange(len(nd))
            n_leaf = int(is_leaf.sum())
            leaves = nd[:n_leaf]
            internal = nd[n_leaf:]
            self.levels.append({
                "leaf_row": torch.as_tensor(vid[leaves] * Tmax + a[leaves]),
                "leaves": torch.as_tensor(leaves),
                "internal": torch.as_tensor(internal),
                "n_leaf": n_leaf,
            })
        for d in range(D):
            it = self.levels[d]["internal"].numpy()
            self.levels[d]["L"] = torch.as_tensor(pos[left[it]])
            self.levels[d]["R"] = torch.as_tensor(pos[right[it]])
        self.pos_root = torch.as_tensor(pos[self.roots])
        ids, ans, lens, owner = [], [], [], []
        for i, (o, off) in enumerate(zip(obs, offs[:-1])):
            if o is None or len(o[0]) == 0:
                continue
            ids.append(np.asarray(o[0]) + off)
            ans.append(np.asarray(o[1]))
            lens.append(np.asarray(o[2]))
            owner.append(np.full(len(o[0]), i))
        self.obs_ids = torch.as_tensor(np.concatenate(ids)) if ids else torch.zeros(0, dtype=torch.long)
        self.obs_ans = torch.as_tensor(np.concatenate(ans)) if ans else torch.zeros(0, N_CAT, dtype=torch.long)
        self.obs_len = torch.as_tensor(np.concatenate(lens)) if lens else torch.zeros(0)
        self.obs_owner = torch.as_tensor(np.concatenate(owner)) if owner else torch.zeros(0, dtype=torch.long)
        self.n_obs = torch.bincount(self.obs_owner, minlength=self.B)

    def log_evidence(self, s, seq_mask, answer_model):
        """s (B, Tmax) per-second logits of pi, seq_mask (B, Tmax) bool. Returns (logP(o | G=1), logP(o | G=0))
        per video, both (B,)."""
        dev = s.device
        flat = s.reshape(-1)
        ll = answer_model.loglik(self.obs_ans, self.obs_len)             # n_obs, 3
        A1 = torch.zeros(self.N, 2, device=dev, dtype=s.dtype)
        if len(self.obs_ids):
            A1 = A1.index_put((self.obs_ids.to(dev),), ll[:, 1:].to(s.dtype))
        lp_g0 = torch.zeros(self.B, device=dev, dtype=s.dtype)
        if len(self.obs_ids):
            lp_g0 = lp_g0.index_add(0, self.obs_owner.to(dev), ll[:, 0].to(s.dtype))
        prev = None
        for d in range(len(self.levels) - 1, -1, -1):
            lev = self.levels[d]
            parts_lu, parts_lv, parts_lz = [], [], []
            if lev["n_leaf"]:
                sl = flat[lev["leaf_row"].to(dev)]
                a = A1[lev["leaves"].to(dev)]
                lP0 = a[:, 0] + F.logsigmoid(-sl)
                lP1 = a[:, 1] + F.logsigmoid(sl)
                lZ = torch.logaddexp(lP0, lP1)
                parts_lu.append(lP1 - lZ)
                parts_lv.append(lP0 - lZ)
                parts_lz.append(lZ)
            if len(lev["internal"]):
                L, R = lev["L"].to(dev), lev["R"].to(dev)
                luL, lvL, lzL = prev[0][L], prev[1][L], prev[2][L]
                luR, lvR, lzR = prev[0][R], prev[1][R], prev[2][R]
                a = A1[lev["internal"].to(dev)]
                lP0 = a[:, 0] + lvL + lvR
                lP1 = a[:, 1] + torch.logaddexp(luL, luR + lvL)
                lZ = torch.logaddexp(lP0, lP1)
                parts_lu.append(lP1 - lZ)
                parts_lv.append(lP0 - lZ)
                parts_lz.append(lZ + lzL + lzR)
            prev = (torch.cat(parts_lu), torch.cat(parts_lv), torch.cat(parts_lz))
        pr = self.pos_root.to(dev)
        lu_root, lz_root = prev[0][pr], prev[2][pr]
        # log P(root = 1) under the independent prior = log(1 - prod(1 - pi_t))
        sum_lv = (F.logsigmoid(-s) * seq_mask.to(s.dtype)).sum(1)
        log_p_any = torch.log(-torch.expm1(sum_lv.clamp(max=-1e-12)))
        return lz_root + lu_root - log_p_any, lp_g0


# ------------------------------------------------------------------ single-video exact inference (numpy)
class VideoTree:
    """Numpy float64 inference for one video (evaluation and the question policy)."""

    def __init__(self, T):
        self.T = int(T)
        self.tr = tree(self.T)
        self.levels, self.pos = leaf_levels(self.tr)
        self.N = len(self.tr["a"])
        self.length = (self.tr["b"] - self.tr["a"]).astype(np.float64)
        self.q_ids = np.where(self.tr["queryable"])[0]

    def infer(self, s, g_logit, logA):
        """s (T,) prior logits of pi; g_logit scalar; logA (N, 3) answer log-likelihoods (0 for unasked nodes).
        Returns p_G (posterior P(G=1)), node marginal m (N,) = P(z_n = 1 | G=1, o) and second posterior
        p (T,) = P(y_t = 1 | o) = p_G * P(y_t = 1 | G=1, o)."""
        tr, N = self.tr, self.N
        lu = np.zeros(N); lv = np.zeros(N); lz = np.zeros(N)
        leaf = tr["left"] < 0
        sec = tr["a"][leaf]
        # leaves (a leaf can carry an answer only if F_FRAMES == 1; kept exact in general)
        lP0 = logA[leaf, 1] - np.logaddexp(0.0, s[sec])
        lP1 = logA[leaf, 2] - np.logaddexp(0.0, -s[sec])
        lZ = np.logaddexp(lP0, lP1)
        lu[leaf], lv[leaf], lz[leaf] = lP1 - lZ, lP0 - lZ, lZ
        for d in range(len(self.levels) - 1, -1, -1):
            nd, n_leaf = self.levels[d]
            it = nd[n_leaf:]
            if len(it) == 0:
                continue
            L, R = tr["left"][it], tr["right"][it]
            lP0 = logA[it, 1] + lv[L] + lv[R]
            lP1 = logA[it, 2] + np.logaddexp(lu[L], lu[R] + lv[L])
            lZ = np.logaddexp(lP0, lP1)
            lu[it] = lP1 - lZ
            lv[it] = lP0 - lZ
            lz[it] = lZ + lz[L] + lz[R]
        sum_lv = float(np.sum(-np.logaddexp(0.0, s)))
        log_p_any = np.log(-np.expm1(min(sum_lv, -1e-12)))
        lp1 = lz[0] + lu[0] - log_p_any
        lp0 = float(logA[:, 0].sum())
        p_G = 1.0 / (1.0 + np.exp(-(g_logit + lp1 - lp0)))
        # downward: outside messages under G=1 with the root fixed to 1
        lo0 = np.full(N, NEG); lo1 = np.zeros(N)
        for d in range(len(self.levels) - 1):
            nd, n_leaf = self.levels[d]
            it = nd[n_leaf:]
            if len(it) == 0:
                continue
            L, R = tr["left"][it], tr["right"][it]
            a0, a1 = logA[it, 1], logA[it, 2]
            for c, sib in ((L, R), (R, L)):
                c0 = np.logaddexp(lo0[it] + a0 + lv[sib], lo1[it] + a1 + lu[sib])
                c1 = lo1[it] + a1
                nz = np.logaddexp(c0, c1)
                lo0[c] = c0 - nz
                lo1[c] = c1 - nz
        m = 1.0 / (1.0 + np.exp(-((lo1 + lu) - (lo0 + lv))))
        p = np.empty(self.T)
        p[sec] = m[leaf]
        return p_G, m, p_G * p


def outcome_table(level_logp):
    """level_logp (n, C, S, L) -> (n, S, L**C) log P(o | s) over all answer vectors (row-major over categories)."""
    n, C, S, L = level_logp.shape
    if n == 0:                                   # video shorter than F_FRAMES seconds: nothing to ask
        return np.zeros((0, S, L ** C))
    out = np.zeros((n, S, 1))
    for k in range(C):
        out = (out[:, :, :, None] + level_logp[:, k, :, None, :]).reshape(n, S, -1)
    return out


def answer_index(o):
    """Answer vector (C,) -> index into outcome_table's last axis."""
    idx = 0
    for x in o:
        idx = idx * N_LEV + int(x)
    return idx


def eig(log_po_s, w):
    """Expected information gain I(o_n; (G, y)) for candidate nodes: log_po_s (n, S, O) log P(o | s), w (n, S)
    state probabilities. Returns (n,) in bits."""
    po_s = np.exp(log_po_s)
    po = np.einsum("nso,ns->no", po_s, w)
    h = -np.sum(po * np.log(np.clip(po, 1e-300, None)), axis=1)
    h_s = -np.sum(po_s * log_po_s, axis=2)                               # n, S
    return (h - np.sum(w * h_s, axis=1)) / np.log(2.0)
