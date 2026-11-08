"""Hidden Markov tree answer model (experiments/20261109_tree_within/README.md section 3).

The VLM's answers to nested questions repeat each other far more than the node states explain (residual correlation
parent-child .66-.81, two levels apart about its square, disjoint nodes of the same video .05-.47;
diag_answer_redundancy.py), while the answer does not depend on how much of a node is hateful
(diag_answer_fraction.py). Model: every node n carries a latent answer bias b_n on a support grid; given the node
state s_n (anchored two states as in revision 6) and b_n,
    P(o_n = l | s_n, b_n, len_n) = softmax_l(theta_{s_n, l} + omega_l u(len_n) + b_n c_l),  c_l = l / (L - 1) - 1/2.
The biases form a Markov tree (Crouse, Nowak and Baraniuk 1998): b_root ~ pi; b_child = b_parent with probability
rho, otherwise a fresh draw from pi. rho = 0: independent answers; rho = 1: one offset per video (answer_offset.py).

Fitted once before training, label-only like qtree.fit_anchored_shared: theta_0, omega, pi (nonparametric mixing
distribution on a fixed grid, Laird 1978; points below PRUNE dropped) and rho by maximum likelihood on the negative
training videos (every node in state 0, the biases summed out exactly over each video's tree of queryable nodes);
theta_1 on the roots of the positive training videos with b_root ~ pi summed out, omega shared.

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

GRID = np.arange(-12.0, 12.0 + 1e-9, 0.5)       # bias grid (tilt units over the full level range), as answer_offset
PRUNE = 1e-4                                     # drop grid points whose fitted weight is below this
RHO_CLAMP = 1e-9                                 # keeps log(1 - rho) and log(rho) finite


def level_scores(L):
    return np.arange(L, dtype=np.float64) / max(L - 1, 1) - 0.5


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
    c = torch.as_tensor(level_scores(L))
    bgrid = torch.as_tensor(np.asarray(grid, dtype=np.float64))
    par, depth, roots, rows, n = _compact([(T, ans) for label, T, ans in train_answers if label == 0])
    par_t, depth_t, roots_t = map(torch.as_tensor, (par, depth, roots))
    par_t = torch.where(par_t < 0, torch.zeros_like(par_t), par_t)          # roots are never scattered (depth 0)
    node = torch.as_tensor(np.array([r[0] for r in rows]))
    O = torch.as_tensor(np.stack([r[1] for r in rows]), dtype=torch.long)    # m, C
    U = torch.as_tensor((np.log([r[2] for r in rows]) - len_mu) / len_sd)    # m
    freq = np.ones((C, L))
    for k in range(C):
        freq[k] += np.bincount(O[:, k].numpy(), minlength=L)
    th0 = torch.as_tensor(np.log(freq / freq.sum(1, keepdims=True))).clone().requires_grad_(True)
    om = torch.zeros(C, L, dtype=torch.float64, requires_grad=True)
    zeta = torch.zeros(len(bgrid), dtype=torch.float64, requires_grad=True)
    rl = torch.zeros((), dtype=torch.float64, requires_grad=True)          # rho = sigmoid(rl)
    Kg = len(bgrid)

    def answer_ll(th, omg):
        x = th[None, None] + omg[None, None] * U[:, None, None, None] + bgrid[None, :, None, None] * c
        lp = F.log_softmax(x, dim=-1)                                          # m, K, C, L
        idx = O[:, None, :, None].expand(-1, Kg, -1, 1)
        return lp.gather(3, idx).squeeze(-1).sum(-1)                           # m, K

    def nll0():
        A0 = torch.zeros(n, Kg, dtype=torch.float64).index_add(0, node, answer_ll(th0, om))
        rho = torch.sigmoid(rl).clamp(RHO_CLAMP, 1 - RHO_CLAMP)
        ll = bias_up(par_t, depth_t, A0, F.log_softmax(zeta, 0), torch.log(rho), torch.log1p(-rho), roots_t)
        return -ll.sum() - F.log_softmax(th0, dim=-1).sum()
    _lbfgs([th0, om, zeta, rl], nll0)
    with torch.no_grad():
        nll_fit = float(nll0())
    w = torch.softmax(zeta, 0).detach().numpy()
    keep = w >= PRUNE
    support = np.asarray(grid, dtype=np.float64)[keep]
    w = w[keep] / w[keep].sum()
    mean = float(np.sum(w * support))                    # centre: a shift of all biases equals a shift of theta
    support = support - mean
    theta0 = th0.detach().numpy() + mean * level_scores(L)[None, :]
    omega = om.detach().numpy()
    rho = float(torch.sigmoid(rl).clamp(RHO_CLAMP, 1 - RHO_CLAMP))
    # state 1 (positive roots): one answer per video, its bias integrated out under pi
    rt = [(ans[(0, T)], T) for label, T, ans in train_answers if label == 1 and ans.get((0, T)) is not None
          and T >= qtree.F_FRAMES]
    Rt = torch.as_tensor(np.stack([o for o, T in rt]), dtype=torch.long)       # R, C
    Ur = torch.as_tensor((np.log([float(T) for o, T in rt]) - len_mu) / len_sd)
    bs, lws = torch.as_tensor(support), torch.as_tensor(np.log(w))
    om_t = torch.as_tensor(omega)
    th1 = torch.zeros(C, L, dtype=torch.float64, requires_grad=True)

    def nll1():
        x = th1[None, None] + om_t[None, None] * Ur[:, None, None, None] + bs[None, :, None, None] * c
        lp = F.log_softmax(x, dim=-1)                                          # R, K, C, L
        per = lp.gather(3, Rt[:, None, :, None].expand(-1, len(bs), -1, 1)).squeeze(-1).sum(-1)   # R, K
        return -torch.logsumexp(per + lws[None], dim=1).sum() - F.log_softmax(th1, dim=-1).sum()
    _lbfgs([th1], nll1)
    theta = np.stack([theta0, th1.detach().numpy()], axis=1)                 # C, 2, L
    info = {"n_negative_videos": int(len(roots)), "n_negative_answers": int(len(rows)), "n_roots": len(rt),
            "rho": rho, "support": support.tolist(), "weights": w.tolist(),
            "bias_sd": float(np.sqrt(np.sum(w * support ** 2))), "nll_negative": nll_fit}
    return HMTAnswerModel(theta, omega, support, np.log(w), rho, len_mu, len_sd), info


class HMTAnswerModel:
    """Anchored two-state answer table with the per-node bias and its Markov tree; states 0 and 1 share the state-0
    table (anchored two states, as revision 6)."""

    def __init__(self, theta, omega, support, logw, rho, len_mu, len_sd, categories=(0,)):
        self.theta = np.asarray(theta, dtype=np.float64)                  # C, 2, L
        self.omega = np.asarray(omega, dtype=np.float64)                  # C, L
        self.support = np.asarray(support, dtype=np.float64)              # K
        self.logw = np.asarray(logw, dtype=np.float64)                    # K
        self.rho = float(rho)
        self.len_mu, self.len_sd = float(len_mu), float(len_sd)
        self.cats = list(categories)
        self.K = len(self.support)

    def to_dict(self):
        return {"theta": self.theta.tolist(), "omega": self.omega.tolist(), "support": self.support.tolist(),
                "logw": self.logw.tolist(), "rho": self.rho, "len_mu": self.len_mu, "len_sd": self.len_sd,
                "categories": self.cats}

    @classmethod
    def from_dict(cls, d):
        return cls(d["theta"], d["omega"], d["support"], d["logw"], d["rho"], d["len_mu"], d["len_sd"],
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
        c = level_scores(self.theta.shape[-1])
        x = (self.theta[None, None][:, :, :, [0, 0, 1], :] + self.omega[None, None, :, None, :] * u[:, None, None,
             None, None] + self.support[None, :, None, None, None] * c[None, None, None, None, :])
        return x - np.logaddexp.reduce(x, axis=-1, keepdims=True)

    def outcome_tables(self, length):
        """(n, K, 3, O) log P(o | s, b_k) over all answer vectors of the used categories."""
        lp = self.level_logp(length)[:, :, self.cats]                    # n, K, C', 3, L
        n = lp.shape[0]
        if n == 0:
            return np.zeros((0, self.K, 3, self.theta.shape[-1] ** len(self.cats)))
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
