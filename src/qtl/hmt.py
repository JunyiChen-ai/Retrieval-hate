"""Hidden Markov tree answer model (experiments/20261109_tree_within/README.md section 3).

The VLM's answers to nested questions repeat each other far more than the node states explain (residual correlation
parent-child .66-.81, two levels apart about its square, disjoint nodes of the same video .05-.47;
diag_answer_redundancy.py), while the answer does not depend on how much of a node is hateful
(diag_answer_fraction.py). Model: every node n carries a latent answer bias b_n on a support grid; given the node
state s_n (anchored two states as in revision 6) and b_n, an ordered logit on the answer level (the levels are
quantile bins of P(Yes)):
    P(o_n <= l | s_n, b_n, len_n) = sigmoid(kappa_{s_n, l} - omega u(len_n) - b_n),  l = 0 .. L - 2,
with increasing thresholds kappa_s and one length slope omega shared by both states. (Version 1 tilted a multinomial
table, softmax_l(theta_{s, l} + omega_l u + b c_l); it cannot make a mid-range level likely for a given bias, so the
fitted model reproduced only part of the answer correlations: README section 6.)
The biases form a Markov tree (Crouse, Nowak and Baraniuk 1998): b_root ~ pi; b_child = b_parent with probability
rho, otherwise a fresh draw from pi. rho = 0: independent answers; rho = 1: one offset per video (answer_offset.py).

Fitted once before training, label-only like qtree.fit_anchored_shared: kappa_0, omega, pi (nonparametric mixing
distribution on a fixed grid, Laird 1978; points below PRUNE dropped) and rho by maximum likelihood on the negative
training videos (every node in state 0, the biases summed out exactly over each video's tree of queryable nodes);
kappa_1 on the roots of the positive training videos with b_root ~ pi summed out, omega shared.

Inference is exact: the chain-tree upward pass of ctree._up with one more index per node, its bias (a child's table
is carried to the parent's bias by the transition in O(K): rho times the same bias plus (1 - rho) times the
pi-average), the answer factor depending on (b_n, z_n); the G = 0 branch (all nodes state 0) is the same pass over
the biases alone. Marginals by automatic differentiation as in ctree.marginals.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import ctree, qtree

GRID = np.arange(-12.0, 12.0 + 1e-9, 0.5)       # bias grid (logit units of the ordered logit)
PRUNE = 1e-4                                     # drop grid points whose fitted weight is below this
RHO_CLAMP = 1e-9                                 # keeps log(1 - rho) and log(rho) finite


def ordinal_logp(kappa, eta):
    """log P(o = l) of the ordered logit, torch: kappa (..., L - 1) increasing thresholds, eta (...) location.
    Returns (..., L)."""
    cum = torch.sigmoid(kappa - eta[..., None])
    zero = torch.zeros_like(cum[..., :1])
    p = torch.cat([cum, 1.0 - cum[..., -1:]], dim=-1) - torch.cat([zero, cum], dim=-1)
    return torch.log(p.clamp(min=1e-12))


def ordinal_logp_np(kappa, eta):
    cum = 1.0 / (1.0 + np.exp(-(kappa - eta[..., None])))
    zero = np.zeros_like(cum[..., :1])
    p = np.concatenate([cum, 1.0 - cum[..., -1:]], axis=-1) - np.concatenate([zero, cum], axis=-1)
    return np.log(np.clip(p, 1e-12, None))


def _thresholds(k0, dl):
    """Increasing thresholds from the first one and softplus increments: (C, L - 1)."""
    return torch.cat([k0[:, None], k0[:, None] + torch.cumsum(F.softplus(dl), dim=1)], dim=1)


def _init_thresholds(O, L):
    """Empirical cumulative logits of the levels (C, L - 1) -> (k0, dl) for _thresholds."""
    C = O.shape[1]
    k0, dl = np.zeros(C), np.zeros((C, L - 2))
    for k in range(C):
        f = np.bincount(O[:, k], minlength=L) + 1.0
        cum = np.clip(np.cumsum(f)[:-1] / f.sum(), 1e-4, 1 - 1e-4)
        kap = np.log(cum) - np.log1p(-cum)
        k0[k] = kap[0]
        inc = np.maximum(np.diff(kap), 1e-3)
        dl[k] = np.log(np.expm1(inc))                     # inverse softplus
    return torch.as_tensor(k0), torch.as_tensor(dl)


def _to_parent(X, logpi, lr, l1r):
    """X (n, K, ...) log tables indexed by a child's bias -> the same indexed by its parent's bias:
    log(rho exp X[b] + (1 - rho) sum_j pi_j exp X[j])."""
    shape = (1, -1) + (1,) * (X.dim() - 2)
    mix = torch.logsumexp(logpi.view(shape) + X, dim=1, keepdim=True)
    return torch.logaddexp(lr + X, l1r + mix)


def bias_up(par, depth, A0, logpi, lr, l1r, roots):
    """Upward pass over the biases alone. par (n,) parent index or -1, depth (n,), A0 (n, K) log answer factors
    (0 for nodes without an answer), roots (B,) root indices. Returns (B,) log sum over all bias configurations of
    prod_n exp(A0[n, b_n]) under the Markov tree."""
    beta = A0
    for d in range(int(depth.max()), 0, -1):
        idx = torch.nonzero(depth == d).squeeze(1)
        beta = beta.index_add(0, par[idx], _to_parent(beta[idx], logpi, lr, l1r))
    return torch.logsumexp(logpi[None] + beta[roots], dim=1)


def _compact(videos):
    """Queryable-node trees of several videos: videos = [(T, {(a, b): answer vector or None})]. Returns par, depth,
    roots and the observed rows (node index, answer vector, length)."""
    par, depth, roots, rows = [], [], [], []
    off = 0
    for T, ans in videos:
        tr = qtree.tree(T)
        q = np.where(tr["queryable"])[0]
        if len(q) == 0:
            continue
        loc = {int(n): i for i, n in enumerate(q)}
        par += [off + loc[int(tr["parent"][n])] if tr["parent"][n] >= 0 else -1 for n in q]
        depth += [int(tr["depth"][n]) for n in q]
        roots.append(off + loc[0])
        for n in q:
            o = ans.get((int(tr["a"][n]), int(tr["b"][n])))
            if o is not None:
                rows.append((off + loc[int(n)], np.asarray(o), float(tr["b"][n] - tr["a"][n])))
        off += len(q)
    return (np.asarray(par, dtype=np.int64), np.asarray(depth, dtype=np.int64), np.asarray(roots, dtype=np.int64),
            rows, off)


def _lbfgs(params, closure_fn):
    opt = torch.optim.LBFGS(params, max_iter=1000, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = closure_fn()
        loss.backward()
        return loss
    opt.step(closure)


def fit_hmt(train_answers, len_mu, len_sd, grid=GRID):
    """train_answers: [(label, T, {(a, b): answer vector (N_CAT,) or None}), ...]. Returns an HMTAnswerModel and fit
    details."""
    C, L = qtree.N_CAT, qtree.N_LEV
    bgrid = torch.as_tensor(np.asarray(grid, dtype=np.float64))
    par, depth, roots, rows, n = _compact([(T, ans) for label, T, ans in train_answers if label == 0])
    par_t, depth_t, roots_t = map(torch.as_tensor, (par, depth, roots))
    par_t = torch.where(par_t < 0, torch.zeros_like(par_t), par_t)          # roots are never scattered (depth 0)
    node = torch.as_tensor(np.array([r[0] for r in rows]))
    O_np = np.stack([r[1] for r in rows]).astype(np.int64)
    O = torch.as_tensor(O_np)                                                 # m, C
    U = torch.as_tensor((np.log([r[2] for r in rows]) - len_mu) / len_sd)    # m
    k0, dl = _init_thresholds(O_np, L)
    k0, dl = k0.clone().requires_grad_(True), dl.clone().requires_grad_(True)
    om = torch.zeros(C, dtype=torch.float64, requires_grad=True)
    zeta = torch.zeros(len(bgrid), dtype=torch.float64, requires_grad=True)
    rl = torch.zeros((), dtype=torch.float64, requires_grad=True)          # rho = sigmoid(rl)
    Kg = len(bgrid)
    lev = torch.arange(L)

    def answer_ll(kap, omg):
        eta = omg[None, None, :] * U[:, None, None] + bgrid[None, :, None]       # m, K, C
        lp = ordinal_logp(kap[None, None], eta)                                 # m, K, C, L
        idx = O[:, None, :, None].expand(-1, Kg, -1, 1)
        return lp.gather(3, idx).squeeze(-1).sum(-1)                           # m, K

    def prior(kap):                     # one pseudo-count per level at the centre (eta = 0), as the anchored fit
        return ordinal_logp(kap, torch.zeros(C, dtype=torch.float64)).sum()

    def nll0():
        kap = _thresholds(k0, dl)
        A0 = torch.zeros(n, Kg, dtype=torch.float64).index_add(0, node, answer_ll(kap, om))
        rho = torch.sigmoid(rl).clamp(RHO_CLAMP, 1 - RHO_CLAMP)
        ll = bias_up(par_t, depth_t, A0, F.log_softmax(zeta, 0), torch.log(rho), torch.log1p(-rho), roots_t)
        return -ll.sum() - prior(kap)
    _lbfgs([k0, dl, om, zeta, rl], nll0)
    with torch.no_grad():
        nll_fit = float(nll0())
        kappa0 = _thresholds(k0, dl).numpy()
    w = torch.softmax(zeta, 0).detach().numpy()
    keep = w >= PRUNE
    support = np.asarray(grid, dtype=np.float64)[keep]
    w = w[keep] / w[keep].sum()
    mean = float(np.sum(w * support))                    # centre: a shift of all biases equals a shift of kappa
    support = support - mean
    kappa0 = kappa0 - mean
    omega = om.detach().numpy()
    rho = float(torch.sigmoid(rl).clamp(RHO_CLAMP, 1 - RHO_CLAMP))
    # state 1 (positive roots): one answer per video, its bias integrated out under pi
    rt = [(ans[(0, T)], T) for label, T, ans in train_answers if label == 1 and ans.get((0, T)) is not None
          and T >= qtree.F_FRAMES]
    Rt_np = np.stack([o for o, T in rt]).astype(np.int64)
    Rt = torch.as_tensor(Rt_np)                                                # R, C
    Ur = torch.as_tensor((np.log([float(T) for o, T in rt]) - len_mu) / len_sd)
    bs, lws = torch.as_tensor(support), torch.as_tensor(np.log(w))
    om_t = torch.as_tensor(omega)
    k1, d1 = _init_thresholds(Rt_np, L)
    k1 = (k1 - mean).clone().requires_grad_(True)
    d1 = d1.clone().requires_grad_(True)

    def nll1():
        kap = _thresholds(k1, d1)
        eta = om_t[None, None, :] * Ur[:, None, None] + bs[None, :, None]       # R, K, C
        lp = ordinal_logp(kap[None, None], eta)                                 # R, K, C, L
        per = lp.gather(3, Rt[:, None, :, None].expand(-1, len(bs), -1, 1)).squeeze(-1).sum(-1)   # R, K
        return -torch.logsumexp(per + lws[None], dim=1).sum() - prior(kap)
    _lbfgs([k1, d1], nll1)
    with torch.no_grad():
        kappa1 = _thresholds(k1, d1).numpy()
    kappa = np.stack([kappa0, kappa1], axis=1)                               # C, 2, L - 1
    info = {"n_negative_videos": int(len(roots)), "n_negative_answers": int(len(rows)), "n_roots": len(rt),
            "rho": rho, "support": support.tolist(), "weights": w.tolist(), "omega": omega.tolist(),
            "bias_sd": float(np.sqrt(np.sum(w * support ** 2))), "nll_negative": nll_fit,
            "kappa": kappa.tolist()}
    return HMTAnswerModel(kappa, omega, support, np.log(w), rho, len_mu, len_sd), info


class HMTAnswerModel:
    """Anchored two-state ordered-logit answer model with the per-node bias and its Markov tree; states 0 and 1 share
    the state-0 thresholds (anchored two states, as revision 6)."""

    def __init__(self, kappa, omega, support, logw, rho, len_mu, len_sd, categories=(0,)):
        self.kappa = np.asarray(kappa, dtype=np.float64)                  # C, 2, L - 1
        self.omega = np.asarray(omega, dtype=np.float64)                  # C
        self.support = np.asarray(support, dtype=np.float64)              # K
        self.logw = np.asarray(logw, dtype=np.float64)                    # K
        self.rho = float(rho)
        self.len_mu, self.len_sd = float(len_mu), float(len_sd)
        self.cats = list(categories)
        self.K = len(self.support)
        self.L = self.kappa.shape[-1] + 1

    def to_dict(self):
        return {"kappa": self.kappa.tolist(), "omega": self.omega.tolist(), "support": self.support.tolist(),
                "logw": self.logw.tolist(), "rho": self.rho, "len_mu": self.len_mu, "len_sd": self.len_sd,
                "categories": self.cats}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kappa"], d["omega"], d["support"], d["logw"], d["rho"], d["len_mu"], d["len_sd"],
                   d["categories"])

    def params(self, device, dtype=torch.float64):
        """(log pi (K,), log rho, log(1 - rho)) as tensors."""
        rho = min(max(self.rho, RHO_CLAMP), 1 - RHO_CLAMP)
        return (torch.as_tensor(self.logw, dtype=dtype, device=device),
                torch.tensor(np.log(rho), dtype=dtype, device=device),
                torch.tensor(np.log1p(-rho), dtype=dtype, device=device))

    def level_logp(self, length):
        """(n, K, C, 3, L) log P(o_c = l | s, b_k, length)."""
        length = np.asarray(length, dtype=np.float64)
        u = (np.log(length) - self.len_mu) / self.len_sd
        eta = self.omega[None, None, :] * u[:, None, None] + self.support[None, :, None]       # n, K, C
        kap = self.kappa[:, [0, 0, 1], :]                                                       # C, 3, L - 1
        return ordinal_logp_np(kap[None, None], eta[:, :, :, None])                            # n, K, C, 3, L

    def outcome_tables(self, length):
        """(n, K, 3, O) log P(o | s, b_k) over all answer vectors of the used categories."""
        lp = self.level_logp(length)[:, :, self.cats]                    # n, K, C', 3, L
        n = lp.shape[0]
        if n == 0:
            return np.zeros((0, self.K, 3, self.L ** len(self.cats)))
        return np.stack([qtree.outcome_table(lp[:, k]) for k in range(self.K)], axis=1)

    def loglik(self, answers, length):
        """(n, K, 3) torch float64 log P(o_n | s, b_k) for integer answers (n, N_CAT)."""
        lp = self.level_logp(length)                                     # n, K, C, 3, L
        o = np.asarray(answers, dtype=np.int64)
        out = np.zeros(lp.shape[:2] + (3,))
        for k in self.cats:
            out += np.take_along_axis(lp[:, :, k], o[:, k][:, None, None, None].repeat(self.K, 1).repeat(3, 2),
                                      axis=3)[..., 0]
        return torch.as_tensor(out)


def forest_parents(forest):
    """(par (N,), depth (N,)) in the forest's global node ids; par = 0 for roots (never scattered: depth 0)."""
    par, depth = [], []
    for T, off in zip(forest.Ts, forest.offs):
        tr = qtree.tree(T)
        par.append(np.where(tr["parent"] >= 0, tr["parent"] + off, 0))
        depth.append(tr["depth"])
    return torch.as_tensor(np.concatenate(par)), torch.as_tensor(np.concatenate(depth))


def _up_hmt(forest, s, A3K, chain, params, eta=None, phi=None, eta0=None):
    """ctree._up with the bias index. A3K (N, K, 3) log P(o_n | s, b_k) (0 rows for unasked nodes); params from
    HMTAnswerModel.params; eta (N, K, 2) optional, added to the G = 1 answer factor (for joint marginals of bias and
    node state); phi (N,) node potentials; eta0 (N, K) optional, added to the G = 0 factor. Returns log V1 (B,) (paths
    with a harmful second, answers under states 1 / 2, biases summed out), the all-zero path weight, and lp0 (B,) =
    log P(answers | G = 0)."""
    dev, dt = s.device, s.dtype
    logpi, lr, l1r = (x.to(dev).to(dt) for x in params)
    K = logpi.shape[0]
    flat = s.reshape(-1)
    logA = chain.logA_for(forest).to(dev).to(dt)
    per_video = logA.dim() == 3
    nv = forest.node_video.to(dev)
    a1 = A3K[:, :, 1:]
    if eta is not None:
        a1 = a1 + eta
    if phi is not None:
        phi = phi.to(a1.dtype)
        a1 = a1 + torch.stack([torch.zeros_like(phi), phi], dim=1)[:, None, :]
    a0 = A3K[:, :, 0]
    if eta0 is not None:
        a0 = a0 + eta0
    prev, prev0 = None, None
    for d in range(len(forest.levels) - 1, -1, -1):
        lev = forest.levels[d]
        nodes = lev["nodes"].to(dev)
        n_leaf = lev["n_leaf"]
        tabs, lzs, b0s, lz0s = [], [], [], []
        if n_leaf:
            sl = flat[lev["leaf_row"].to(dev)]
            t = torch.full((n_leaf, 2, 2, 2), -ctree.BIG, device=dev, dtype=dt)
            ar = torch.arange(n_leaf, device=dev)
            z = torch.zeros(n_leaf, dtype=torch.long, device=dev)
            t = t.index_put((ar, z, z, z), torch.zeros_like(sl))
            t = t.index_put((ar, z + 1, z + 1, z + 1), sl)
            tabs.append(t[:, None].expand(-1, K, -1, -1, -1))
            lzs.append(torch.zeros(n_leaf, device=dev, dtype=dt))
            b0s.append(torch.zeros(n_leaf, K, device=dev, dtype=dt))
            lz0s.append(torch.zeros(n_leaf, device=dev, dtype=dt))
        if len(nodes) > n_leaf:
            L, R = lev["L"].to(dev), lev["R"].to(dev)
            ML = _to_parent(prev[0][L], logpi, lr, l1r)
            MR = _to_parent(prev[0][R], logpi, lr, l1r)
            n = ML.shape[0]
            la = logA[nv[nodes[n_leaf:]]].repeat_interleave(K, dim=0) if per_video else logA
            mg = ctree._merge(ML.reshape(n * K, 2, 2, 2), MR.reshape(n * K, 2, 2, 2), la).reshape(n, K, 2, 2, 2)
            tabs.append(mg)
            lzs.append(prev[1][L] + prev[1][R])
            b0s.append(_to_parent(prev0[0][L], logpi, lr, l1r) + _to_parent(prev0[0][R], logpi, lr, l1r))
            lz0s.append(prev0[1][L] + prev0[1][R])
        tab = torch.cat(tabs) + a1[nodes][:, :, None, None, :]          # answer factor on (bias, node's "any")
        c = torch.logsumexp(tab.reshape(len(nodes), -1), dim=1)
        prev = (tab - c[:, None, None, None, None], torch.cat(lzs) + c)
        b0 = torch.cat(b0s) + a0[nodes]
        c0 = torch.logsumexp(b0, dim=1)
        prev0 = (b0 - c0[:, None], torch.cat(lz0s) + c0)
    pr = forest.pos_root.to(dev)
    root = torch.logsumexp(logpi.view(1, K, 1, 1, 1) + prev[0][pr], dim=1)   # B, f, l, a
    a00 = logA[:, 0, 0] if per_video else logA[0, 0]
    Tm1 = forest.Tm1.to(dev).to(dt)
    if chain.closed:
        enter = logA[:, 0, :] if per_video else logA[0, :][None].expand(forest.B, -1)
        leave = logA[:, :, 0] if per_video else logA[:, 0][None].expand(forest.B, -1)
        v1 = prev[1][pr] + torch.logsumexp(enter[:, :, None] + root[:, :, :, 1] + leave[:, None, :], dim=(1, 2))
        w0 = (Tm1 + 2) * a00
    else:
        logpi_c = chain.logpi().to(dev).to(dt)
        v1 = prev[1][pr] + torch.logsumexp(logpi_c[None, :, None] + root[:, :, :, 1], dim=(1, 2))
        w0 = logpi_c[0] + Tm1 * a00
    lp0 = prev0[1][pr] + torch.logsumexp(logpi[None] + prev0[0][pr], dim=1)
    return v1, w0, lp0


def up_hmt(forest, s, g, A3K, chain, params, phi=None):
    """ctree.up (zero-inflated chain) with the HMT answers: (logW1, logW0) per video."""
    assert getattr(chain, "zero_inflated", False), "the HMT answer model is implemented for the zero-inflated chain"
    v1, _, lp0 = _up_hmt(forest, s, A3K, chain, params, None, phi)
    v1_prior, _, _ = ctree._up(forest, s, torch.zeros(forest.N, 3, device=s.device, dtype=s.dtype), chain, None, phi)
    return g + v1 - v1_prior, lp0


def log_evidence_hmt(forest, s, g, A3K, chain, params, phi=None):
    """ctree.log_evidence with the HMT answers: log P(Y = 1 | x), log P(Y = 0 | x) (no answers), log P(o | Y = 1, x),
    log P(o | Y = 0, x)."""
    zero = torch.zeros(forest.N, 3, device=s.device, dtype=s.dtype)
    w1n, w0n = ctree.up(forest, s, g, zero, chain, phi=phi)
    w1a, w0a = up_hmt(forest, s, g, A3K, chain, params, phi=phi)
    logZ = torch.logaddexp(w1n, w0n)
    return w1n - logZ, w0n - logZ, w1a - w1n, w0a - w0n


def marginals_hmt(forest, s, g, A3K, chain, params, phi=None):
    """Posterior under the HMT answers: p_G (B,), J1 (N, K, 2) = P(b_n = k, z_n = a | G = 1, o), Q0 (N, K) =
    P(b_n = k | G = 0, o), second posteriors p (B, Tmax) = P(G = 1 | o) P(y_t = 1 | G = 1, o). No gradient."""
    K = A3K.shape[1]
    with torch.enable_grad():
        s = s.detach().double().requires_grad_(True)
        eta = torch.zeros(forest.N, K, 2, dtype=torch.float64, device=s.device, requires_grad=True)
        eta0 = torch.zeros(forest.N, K, dtype=torch.float64, device=s.device, requires_grad=True)
        g = g.detach().double()
        phi = None if phi is None else phi.detach().double()
        chain_d = ctree._DoubleChain(chain)
        v1, _, lp0 = _up_hmt(forest, s, A3K.double(), chain_d, params, eta, phi, eta0)
        gs, ge, ge0 = torch.autograd.grad(v1.sum() + lp0.sum(), (s, eta, eta0))
    with torch.no_grad():
        v1_prior, _, _ = ctree._up(forest, s.detach(), torch.zeros(forest.N, 3, dtype=torch.float64,
                                                                    device=s.device), chain_d, None, phi)
        pG = torch.sigmoid(g + v1.detach() - v1_prior - lp0.detach())
    return pG, ge.detach(), ge0.detach(), (pG[:, None] * gs).detach()
