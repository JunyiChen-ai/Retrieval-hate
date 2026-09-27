"""Perception layer on the query tree (revision-5 candidate, README section 16.4).

Revision 4 (ctree) lets the VLM answer o_n depend on the node state (G, z_n = any harmful second in n) and treats
the answers of different nodes as independent given those states. Nested nodes show the VLM overlapping content
(the child's transcript is part of the parent's), so a child's answer repeats evidence the parent's answer already
gave, and the model counts it twice.

Here a second layer v_t in {0, 1} says whether the VLM perceives harm at second t:
    P(v_t = 1 | y_t = 1) = h,  P(v_t = 1 | y_t = 0, G = 1) = f,  P(v_t = 1 | G = 0) = f0,
independent over seconds given y; the answer depends only on w_n = OR of v_t over the node: P(o_n | w_n), the same
tables for G = 0 and G = 1. Answers of nested nodes are then dependent through the shared seconds; a node's answer
rate rises with its length (more seconds that can be misperceived); harm the VLM sees in the clean part of a
harmful video (f > f0) is a per-second rate, not a node state.

Exact inference: each node carries a table over (first second's y, last second's y, any y in the node, any v in the
node) = 16 log-weights; two children merge through the chain transition A(last_L, first_R); OR combines the two
"any" flags; the node potential phi_n (revision 4) sits on "any y", the answer factor on "any v". Given G = 0 all
y = 0 and a 2-entry table (any v) suffices. Marginals by automatic differentiation: P(y_t = 1 | G = 1, o) =
d log V1 / d s_t; P(w_n = 1 | G, o) = d log V / d eta_n with eta_n on the node's "any v".
"""
from __future__ import annotations

import torch

BIG = 1e4


def _or(X, d):
    """OR-combine two adjacent flag axes d, d + 1 of X into one axis at position d."""
    a = X.select(d, 0).select(d, 0)
    b = torch.logsumexp(torch.stack([X.select(d, 0).select(d, 1), X.select(d, 1).select(d, 0),
                                     X.select(d, 1).select(d, 1)]), dim=0)
    return torch.stack([a, b], dim=d)


def _merge1(L, R, logA):
    """L, R (n, f, l, ay, av) -> union (n, f, l, ay, av); logA (2, 2)."""
    X = (L[:, :, :, :, :, None, None, None, None] + logA[None, None, :, None, None, :, None, None, None]
         + R[:, None, None, None, None, :, :, :, :])                   # n, f, lL, ayL, avL, fR, lR, ayR, avR
    X = torch.logsumexp(X, dim=(2, 5))                                  # n, f, ayL, avL, lR, ayR, avR
    X = X.permute(0, 1, 4, 2, 5, 3, 6)                                  # n, f, l, ayL, ayR, avL, avR
    X = _or(X, 3)                                                       # n, f, l, ay, avL, avR
    return _or(X, 4)                                                    # n, f, l, ay, av


def up1(forest, s, A2, logA, h, f, phi=None, eta=None):
    """log V1 (B,): log-weight of the paths with at least one harmful second under the closed chain, with the
    answers A2 (N, 2) = log P(o_n | w_n) (0 rows for unasked nodes) and unary s (B, Tmax); phi (N,) node potentials
    on "any y"; eta (N,) on "any v" (for node marginals). h, f: scalars (tensors) in (0, 1)."""
    dev, dt = s.device, s.dtype
    flat = s.reshape(-1)
    lh, l1h, lf, l1f = torch.log(h), torch.log1p(-h), torch.log(f), torch.log1p(-f)
    fac = A2
    if eta is not None:
        fac = fac + torch.stack([torch.zeros_like(eta), eta], dim=1)
    prev = None
    for d in range(len(forest.levels) - 1, -1, -1):
        lev = forest.levels[d]
        nodes = lev["nodes"].to(dev)
        n_leaf = lev["n_leaf"]
        tabs, lzs = [], []
        if n_leaf:
            sl = flat[lev["leaf_row"].to(dev)]
            t = torch.full((n_leaf, 2, 2, 2, 2), -BIG, device=dev, dtype=dt)
            t[:, 0, 0, 0, 0] = l1f
            t[:, 0, 0, 0, 1] = lf
            t[:, 1, 1, 1, 0] = sl + l1h
            t[:, 1, 1, 1, 1] = sl + lh
            tabs.append(t)
            lzs.append(torch.zeros(n_leaf, device=dev, dtype=dt))
        if len(nodes) > n_leaf:
            L, R = lev["L"].to(dev), lev["R"].to(dev)
            tabs.append(_merge1(prev[0][L], prev[0][R], logA))
            lzs.append(prev[1][L] + prev[1][R])
        tab = torch.cat(tabs)
        lz = torch.cat(lzs)
        tab = tab + fac[nodes][:, None, None, None, :]
        if phi is not None:
            tab = tab + torch.stack([torch.zeros_like(phi[nodes]), phi[nodes]], dim=1)[:, None, None, :, None]
        c = torch.logsumexp(tab.reshape(len(nodes), -1), dim=1)
        prev = (tab - c[:, None, None, None, None], lz + c)
    pr = forest.pos_root.to(dev)
    root = prev[0][pr]                                                  # B, f, l, ay, av
    enter = logA[0, :][None].expand(forest.B, -1)
    leave = logA[:, 0][None].expand(forest.B, -1)
    return prev[1][pr] + torch.logsumexp(enter[:, :, None, None] + root[:, :, :, 1, :] + leave[:, None, :, None],
                                         dim=(1, 2, 3))


def up0(forest, A2, f0, eta=None):
    """log V0 (B,): log P(answers | G = 0) (all seconds clean, v_t ~ Bernoulli(f0))."""
    dev, dt = A2.device, A2.dtype
    fac = A2
    if eta is not None:
        fac = fac + torch.stack([torch.zeros_like(eta), eta], dim=1)
    leaf = torch.stack([torch.log1p(-f0), torch.log(f0)])
    prev = None
    for d in range(len(forest.levels) - 1, -1, -1):
        lev = forest.levels[d]
        nodes = lev["nodes"].to(dev)
        n_leaf = lev["n_leaf"]
        tabs, lzs = [], []
        if n_leaf:
            tabs.append(leaf[None].expand(n_leaf, -1).to(dt))
            lzs.append(torch.zeros(n_leaf, device=dev, dtype=dt))
        if len(nodes) > n_leaf:
            L, R = lev["L"].to(dev), lev["R"].to(dev)
            X = prev[0][L][:, :, None] + prev[0][R][:, None, :]
            tabs.append(torch.stack([X[:, 0, 0], torch.logsumexp(torch.stack([X[:, 0, 1], X[:, 1, 0], X[:, 1, 1]]),
                                                                  0)], dim=1))
            lzs.append(prev[1][L] + prev[1][R])
        tab = torch.cat(tabs) + fac[nodes]
        lz = torch.cat(lzs)
        c = torch.logsumexp(tab, dim=1)
        prev = (tab - c[:, None], lz + c)
    pr = forest.pos_root.to(dev)
    return prev[1][pr] + torch.logsumexp(prev[0][pr], dim=1)


def log_answers(forest, s, g, A2, logA, h, f, f0, phi=None, v1_prior=None):
    """(logW1, logW0): log P(G = 1, answers | x) and log P(G = 0, answers | x) up to the common 1 / (1 + e^g)."""
    if v1_prior is None:
        v1_prior = up1(forest, s, torch.zeros_like(A2), logA, h, f, phi)
    return g + up1(forest, s, A2, logA, h, f, phi) - v1_prior, up0(forest, A2, f0)


def posterior(forest, s, g, A2, logA, h, f, f0, phi=None, v1_prior=None):
    """p_G (B,), P(w_n = 1 | G = 1, o) (N,), P(w_n = 1 | G = 0, o) (N,), scores (B, Tmax) = P(G = 1 | o)
    P(y_t = 1 | G = 1, o). No gradient to the caller."""
    with torch.enable_grad():
        s = s.detach().double().requires_grad_(True)
        eta1 = torch.zeros(forest.N, dtype=torch.float64, device=s.device, requires_grad=True)
        eta0 = torch.zeros(forest.N, dtype=torch.float64, device=s.device, requires_grad=True)
        A2 = A2.double()
        phi = None if phi is None else phi.detach().double()
        v1 = up1(forest, s, A2, logA, h, f, phi, eta1)
        v0 = up0(forest, A2, f0, eta0)
        gs, m1 = torch.autograd.grad(v1.sum(), (s, eta1))
        (m0,) = torch.autograd.grad(v0.sum(), (eta0,))
    with torch.no_grad():
        if v1_prior is None:
            v1_prior = up1(forest, s.detach(), torch.zeros_like(A2), logA, h, f, phi)
        pG = torch.sigmoid(g.double() + v1.detach() - v1_prior - v0.detach())
    return pG, m1.detach(), m0.detach(), (pG[:, None] * gs).detach()
