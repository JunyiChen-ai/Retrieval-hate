"""Chain prior on the query tree (README section 7, revision 1).

Revision 0 (qtree.py) gave every second an independent prior; with answers and labels that only say "at least one
hateful second in this interval", the model learned sparse hate (mean score .09 in HateMM positive videos whose
GT is 61% hateful) and the per-second head never saw a negative video. Revision 1 replaces the prior by a
conditional random field over the seconds:
    P(y | x) proportional to  exp(g * any(y) + sum_t s_t y_t) * pi0(y_1) * prod_t A(y_{t-1}, y_t)
with per-second logits s_t and video logit g from the network, and a learned 2-state transition A and initial
distribution pi0. Hate comes in contiguous runs; "no hate at all" is the all-zero path. The video label is
Y = any(y); VLM answers depend on s_n in {0: Y=0, 1: Y=1 and z_n=0, 2: Y=1 and z_n=1} as in revision 0.

Exact inference: each tree node carries a table over (first second's state, last second's state, any hate in the
node) = 8 log-weights; two children merge through one transition A(last_L, first_R); OR combines the "any" flags.
Upward pass: O(64 T). Marginals by automatic differentiation: E[y_t] = d logW / d s_t, E[z_n] = d logW / d eta_n
(eta_n added to the node's z = 1 factor).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qtree

BIG = 1e4


class Chain(nn.Module):
    """Learned 2-state transition (rows: from state) and initial distribution.
    closed (revision 2, README section 9.1): the chain starts from and ends in a virtual non-harmful second, i.e.
    P(y) proportional to A(0, y_1) prod_t A(y_{t-1}, y_t) A(y_T, 0); every harmful run then pays one entry and one
    exit transition wherever it lies (with free ends, pi0 and no exit factor, a run touching the start or the end of
    the video pays one transition fewer, so the prior piles harmful mass on the first and last seconds). The initial
    distribution is not used when closed."""

    def __init__(self, closed=False, zero_inflated=False, normalized=False):
        super().__init__()
        self.closed = bool(closed)
        self.zero_inflated = bool(zero_inflated)
        self.normalized = bool(normalized)
        self.trans = nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))   # start: stay with p ~ .98
        self.init = nn.Parameter(torch.zeros(2))

    def logA(self):
        return F.log_softmax(self.trans, dim=1)

    def logpi(self):
        return F.log_softmax(self.init, dim=0)

    def logA_for(self, forest):
        return self.logA()


class HazardChain(nn.Module):
    """Diagnostic (README section 8): no learned transition. Symmetric per-video switch probability 1/T, i.e. a
    priori one expected change of state over the video whatever its length (the constant-hazard change-point prior
    with expected run length T), and a uniform initial state. The per-second logits s_t and the video logit g carry
    everything the data says."""

    def __init__(self, closed=False):
        super().__init__()
        self.closed = bool(closed)

    def logA_for(self, forest):
        h = torch.as_tensor([1.0 / max(T, 2) for T in forest.Ts], dtype=torch.float64)
        stay, move = torch.log1p(-h), torch.log(h)
        return torch.stack([torch.stack([stay, move], -1), torch.stack([move, stay], -1)], 1)

    def logpi(self):
        return torch.log(torch.full((2,), 0.5, dtype=torch.float64))


class Forest:
    """Batched tree structure for videos of lengths Ts (padded logits of width Tmax). Node ids are global
    (video offsets); `offs[i]` is video i's first node id (its root)."""

    def __init__(self, Ts, Tmax):
        self.Ts = [int(T) for T in Ts]
        self.B = len(Ts)
        trees = [qtree.tree(T) for T in self.Ts]
        offs = np.cumsum([0] + [len(t["a"]) for t in trees])
        self.offs = offs[:-1].astype(np.int64)
        self.N = int(offs[-1])
        depth = np.concatenate([t["depth"] for t in trees])
        left = np.concatenate([np.where(t["left"] >= 0, t["left"] + o, -1) for t, o in zip(trees, self.offs)])
        right = np.concatenate([np.where(t["right"] >= 0, t["right"] + o, -1) for t, o in zip(trees, self.offs)])
        a = np.concatenate([t["a"] for t in trees])
        vid = np.concatenate([np.full(len(t["a"]), i) for i, t in enumerate(trees)])
        self.node_video = torch.as_tensor(vid)
        D = int(depth.max())
        pos = np.empty(self.N, dtype=np.int64)
        self.levels = []
        for d in range(D + 1):
            nd = np.where(depth == d)[0]
            is_leaf = left[nd] < 0
            nd = np.concatenate([nd[is_leaf], nd[~is_leaf]])
            pos[nd] = np.arange(len(nd))
            n_leaf = int(is_leaf.sum())
            self.levels.append({"nodes": torch.as_tensor(nd), "n_leaf": n_leaf,
                                "leaf_row": torch.as_tensor(vid[nd[:n_leaf]] * Tmax + a[nd[:n_leaf]])})
        for d in range(D):
            it = self.levels[d]["nodes"].numpy()[self.levels[d]["n_leaf"]:]
            self.levels[d]["L"] = torch.as_tensor(pos[left[it]])
            self.levels[d]["R"] = torch.as_tensor(pos[right[it]])
        self.pos_root = torch.as_tensor(pos[self.offs])
        self.Tm1 = torch.as_tensor([T - 1 for T in self.Ts], dtype=torch.float32)


def _merge(L, R, logA):
    """L, R: (n, f, l, a) log tables of two adjacent intervals -> (n, f, l, a) of their union. logA (2, 2), or
    (n, 2, 2) with one transition per merged pair."""
    la = logA[None] if logA.dim() == 2 else logA
    X = L[:, :, :, :, None, None, None] + la[:, None, :, None, :, None, None] + R[:, None, None, None, :, :, :]
    X = torch.logsumexp(X, dim=(2, 4))                                   # n, f, aL, l, aR
    a0 = X[:, :, 0, :, 0]
    a1 = torch.logsumexp(torch.stack([X[:, :, 0, :, 1], X[:, :, 1, :, 0], X[:, :, 1, :, 1]]), dim=0)
    return torch.stack([a0, a1], dim=-1)


def up(forest, s, g, A3, chain, eta=None):
    """Upward pass. s (B, Tmax) per-second logits, g (B,) video logits, A3 (N, 3) answer log-likelihoods per state
    (0 rows for unasked nodes), eta (N,) optional (added to the z = 1 factor, for node marginals).
    Returns (logW1, logW0), each (B,): log-weights (up to a common per-video constant) of Y = 1 with the answers
    under the Y = 1 answer states, and of Y = 0 with the answers under state 0.
    Coupled chain (revisions 1-2): P(y) proportional to exp(g any(y) + sum_t s_t y_t) x chain; Y = any(y).
    Normalized chain (chain.normalized, README section 10): as coupled, with the chain's prior mass of Y = 1 and of
    Y = 0 each rescaled to 1/2 for every video length (no length bias of the video decision).
    Zero-inflated chain (chain.zero_inflated, README section 10): P(G = 1 | x) = sigmoid(g) whatever the length;
    given G = 1 the seconds follow the chain with unary s_t conditioned on at least one harmful second; given G = 0
    all seconds are 0. Then logW1 = g + log V1(answers) - log V1(no answers), logW0 = answers under state 0."""
    v1, w0_chain, lp0 = _up(forest, s, A3, chain, eta)
    if getattr(chain, "zero_inflated", False):
        v1_prior, _, _ = _up(forest, s, torch.zeros_like(A3), chain, None)
        return g + v1 - v1_prior, lp0
    if getattr(chain, "normalized", False):
        # README section 10: the chain's own prior mass of "some harmful second" and of "none" are both rescaled to
        # 1/2 for every length: P(y) prop. to exp(g any(y) + s.y) chain(y) / chain(any(y)), where chain(1) is the
        # chain's total weight of paths with a harmful second and chain(0) its all-zero weight (both at s = 0).
        # The per-second logits still move the video decision; the length of the video no longer does.
        v1_flat, w0_flat, _ = _up(forest, torch.zeros_like(s), torch.zeros_like(A3), chain, None)
        return g + v1 - v1_flat, w0_chain - w0_flat + lp0
    return g + v1, w0_chain + lp0


def _up(forest, s, A3, chain, eta=None):
    """Returns log V1 (B,): log-weight of the paths with at least one harmful second, answers under states 1/2,
    without the video factor g; the log-weight of the all-zero path under the chain (without answers); and the
    answers' log-likelihood under state 0, lp0 (B,)."""
    dev = s.device
    flat = s.reshape(-1)
    logA, logpi = chain.logA_for(forest).to(dev).to(s.dtype), chain.logpi().to(dev).to(s.dtype)
    per_video = logA.dim() == 3
    nv = forest.node_video.to(dev)
    a1 = A3[:, 1:]
    if eta is not None:
        a1 = a1 + torch.stack([torch.zeros_like(eta), eta], dim=1)
    prev = None
    for d in range(len(forest.levels) - 1, -1, -1):
        lev = forest.levels[d]
        nodes = lev["nodes"].to(dev)
        n_leaf = lev["n_leaf"]
        tabs, lzs = [], []
        if n_leaf:
            sl = flat[lev["leaf_row"].to(dev)]
            t = torch.full((n_leaf, 2, 2, 2), -BIG, device=dev, dtype=s.dtype)
            t = t.index_put((torch.arange(n_leaf, device=dev), torch.zeros(n_leaf, dtype=torch.long, device=dev),
                             torch.zeros(n_leaf, dtype=torch.long, device=dev),
                             torch.zeros(n_leaf, dtype=torch.long, device=dev)), torch.zeros_like(sl))
            t = t.index_put((torch.arange(n_leaf, device=dev), torch.ones(n_leaf, dtype=torch.long, device=dev),
                             torch.ones(n_leaf, dtype=torch.long, device=dev),
                             torch.ones(n_leaf, dtype=torch.long, device=dev)), sl)
            tabs.append(t)
            lzs.append(torch.zeros(n_leaf, device=dev, dtype=s.dtype))
        if len(nodes) > n_leaf:
            L, R = lev["L"].to(dev), lev["R"].to(dev)
            tabs.append(_merge(prev[0][L], prev[0][R], logA[nv[nodes[n_leaf:]]] if per_video else logA))
            lzs.append(prev[1][L] + prev[1][R])
        tab = torch.cat(tabs)
        lz = torch.cat(lzs)
        tab = tab + a1[nodes][:, None, None, :]                          # answer factor on the node's "any"
        c = torch.logsumexp(tab.reshape(len(nodes), -1), dim=1)
        prev = (tab - c[:, None, None, None], lz + c)
    pr = forest.pos_root.to(dev)
    root = prev[0][pr]                                                   # B, f, l, a
    lp0 = torch.zeros(forest.B, device=dev, dtype=s.dtype).index_add(0, forest.node_video.to(dev), A3[:, 0])
    a00 = logA[:, 0, 0] if per_video else logA[0, 0]
    Tm1 = forest.Tm1.to(dev).to(s.dtype)
    if chain.closed:                                                     # enter from and exit to a virtual 0
        enter = logA[:, 0, :] if per_video else logA[0, :][None].expand(forest.B, -1)
        leave = logA[:, :, 0] if per_video else logA[:, 0][None].expand(forest.B, -1)
        v1 = prev[1][pr] + torch.logsumexp(enter[:, :, None] + root[:, :, :, 1] + leave[:, None, :], dim=(1, 2))
        w0 = (Tm1 + 2) * a00
    else:
        v1 = prev[1][pr] + torch.logsumexp(logpi[None, :, None] + root[:, :, :, 1], dim=(1, 2))
        w0 = logpi[0] + Tm1 * a00
    return v1, w0, lp0


def log_evidence(forest, s, g, A3, chain):
    """Training quantities per video: log P(Y = 1 | x), log P(Y = 0 | x) (no answers) and the answer
    log-likelihoods log P(o | Y = 1, x), log P(o | Y = 0, x)."""
    zero = torch.zeros_like(A3)
    w1n, w0n = up(forest, s, g, zero, chain)
    w1a, w0a = up(forest, s, g, A3, chain)
    logZ = torch.logaddexp(w1n, w0n)
    return w1n - logZ, w0n - logZ, w1a - w1n, w0a - w0n


def marginals(forest, s, g, A3, chain):
    """Posterior under the answers: p_G (B,), node marginals m (N,) = P(z_n = 1 | Y = 1, o), second posteriors
    p (B, Tmax) = P(Y = 1 | o) P(y_t = 1 | Y = 1, o). No gradient to the caller."""
    with torch.enable_grad():
        s = s.detach().double().requires_grad_(True)
        eta = torch.zeros(forest.N, dtype=torch.float64, device=s.device, requires_grad=True)
        g = g.detach().double()
        chain_d = _DoubleChain(chain)
        w1, w0 = up(forest, s, g, A3.double(), chain_d)
        v1, _, _ = _up(forest, s, A3.double(), chain_d, eta)
        gs, ge = torch.autograd.grad(v1.sum(), (s, eta))
    pG = torch.sigmoid(w1 - w0).detach()
    return pG, ge.detach(), (pG[:, None] * gs).detach()


class _DoubleChain:
    def __init__(self, chain):
        self._chain = chain
        self.closed = chain.closed
        self.zero_inflated = getattr(chain, "zero_inflated", False)
        self.normalized = getattr(chain, "normalized", False)

    def logA_for(self, forest):
        return self._chain.logA_for(forest).detach().double()

    def logpi(self):
        return self._chain.logpi().detach().double()
