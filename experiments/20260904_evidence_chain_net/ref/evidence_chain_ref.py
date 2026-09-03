"""NumPy reference implementation of the evidence-chain output head (spec from the
main session, 2026-09-04).  Used only as an oracle for the torch implementation;
no training, no data access.

Model (per video, T segments, J coarse blocks, block_of_t non-decreasing):
  augmented states  0 = (s0,h0)   1 = (s1,h1)   2 = (s0,h1)
  h is "hate seen so far in the current block"; it resets to 0 at the first
  segment of every block (t = 0 or block_of_t[t] != block_of_t[t-1]).
  s transition   t = 0:   p0 = (1-d, d)
                 t > 0:   A(d) = [[1 - a*d, a*d], [a*(1-d), 1 - a*(1-d)]]
  h_new = 1  iff  s = 1 or h_prev = 1 (h_prev := 0 at a block start);
  transitions violating the h constraint have probability 0.
  emission log-potential (zero for s=0,h=0):
     e_t(state) = [s=1]*(u_t + gamma_f_t*phi_f_t)
                + [block_end_t]*[h=1]*gamma_c[b_t]*phi_c[b_t]
Outputs: log Z (all paths), log Z_0 (all-zero path), P(y=1) = 1 - exp(log Z_0 - log Z),
posterior p_t = P(s_t = 1 | everything) = marginal of state 1.
"""
from __future__ import annotations

import numpy as np

STATES = ((0, 0), (1, 1), (0, 1))   # (s, h)
NEG_INF = -np.inf


def block_end_from(block_of_t):
    b = np.asarray(block_of_t, int)
    T = len(b)
    return np.array([t == T - 1 or b[t + 1] != b[t] for t in range(T)])


def new_block_from(block_of_t):
    b = np.asarray(block_of_t, int)
    T = len(b)
    return np.array([t == 0 or b[t] != b[t - 1] for t in range(T)])


def transition_matrix(d, a):
    return np.array([[1.0 - a * d, a * d], [a * (1.0 - d), 1.0 - a * (1.0 - d)]])


def _logsumexp(x, axis=None):
    x = np.asarray(x, float)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def _log(p):
    with np.errstate(divide="ignore"):
        return np.where(p > 0, np.log(np.maximum(p, 1e-300)), NEG_INF)


def build(u, phi_f, phi_c, gamma_f, gamma_c, d, a, block_of_t):
    """Return (log_init[3], log_trans[T,3,3] (rows t>=1 used), log_emit[T,3])."""
    u = np.asarray(u, float); phi_f = np.asarray(phi_f, float)
    gamma_f = np.asarray(gamma_f, float)
    phi_c = np.asarray(phi_c, float); gamma_c = np.asarray(gamma_c, float)
    b = np.asarray(block_of_t, int)
    T = len(u)
    assert len(phi_f) == T and len(gamma_f) == T and len(b) == T
    assert np.all(np.diff(b) >= 0) and b[0] == 0 and b[-1] < len(phi_c)
    end = block_end_from(b)
    new = new_block_from(b)
    p0 = np.array([1.0 - d, d])
    A = transition_matrix(d, a)
    # initial (t = 0): h_prev = 0, so state 2 impossible
    log_init = np.full(3, NEG_INF)
    for j, (s, h) in enumerate(STATES):
        h_new = 1 if s == 1 else 0
        if h == h_new:
            log_init[j] = _log(p0[s])
    log_trans = np.full((T, 3, 3), NEG_INF)
    for t in range(1, T):
        for i, (s_prev, h_prev) in enumerate(STATES):
            hp = 0 if new[t] else h_prev
            for j, (s, h) in enumerate(STATES):
                h_new = 1 if (s == 1 or hp == 1) else 0
                if h == h_new:
                    log_trans[t, i, j] = _log(A[s_prev, s])
    log_emit = np.zeros((T, 3))
    for t in range(T):
        for j, (s, h) in enumerate(STATES):
            v = 0.0
            if s == 1:
                v += u[t] + gamma_f[t] * phi_f[t]
            if end[t] and h == 1:
                v += gamma_c[b[t]] * phi_c[b[t]]
            log_emit[t, j] = v
    return log_init, log_trans, log_emit


def forward_backward(u, phi_f, phi_c, gamma_f, gamma_c, d, a, block_of_t):
    """Return dict(log_Z, log_Z0, p_y1, post[T]) via log-space forward-backward."""
    log_init, log_trans, log_emit = build(u, phi_f, phi_c, gamma_f, gamma_c, d, a, block_of_t)
    T = log_emit.shape[0]
    alpha = np.full((T, 3), NEG_INF)
    alpha[0] = log_init + log_emit[0]
    for t in range(1, T):
        # alpha[t, j] = logsumexp_i(alpha[t-1, i] + log_trans[t, i, j]) + log_emit[t, j]
        alpha[t] = _logsumexp(alpha[t - 1][:, None] + log_trans[t], axis=0) + log_emit[t]
    log_Z = _logsumexp(alpha[-1])
    beta = np.zeros((T, 3))
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(log_trans[t + 1] + (log_emit[t + 1] + beta[t + 1])[None, :], axis=1)
    post_states = np.exp(alpha + beta - log_Z)          # (T, 3)
    post = post_states[:, 1]                            # P(s_t = 1)
    # all-zero path: state 0 throughout; emissions there are 0 by construction
    log_Z0 = log_init[0] + sum(log_trans[t, 0, 0] for t in range(1, T))
    p_y1 = 1.0 - np.exp(log_Z0 - log_Z)
    return {"log_Z": float(log_Z), "log_Z0": float(log_Z0), "p_y1": float(p_y1),
            "post": post, "post_states": post_states}


def brute_force(u, phi_f, phi_c, gamma_f, gamma_c, d, a, block_of_t):
    """Enumerate all s in {0,1}^T under the same generative model."""
    log_init, log_trans, log_emit = build(u, phi_f, phi_c, gamma_f, gamma_c, d, a, block_of_t)
    b = np.asarray(block_of_t, int)
    T = len(b)
    new = new_block_from(b)
    weights = []
    post_num = np.zeros(T)
    for code in range(2 ** T):
        s = [(code >> t) & 1 for t in range(T)]
        # h from s and block structure
        h, states = [], []
        for t in range(T):
            hp = 0 if new[t] else h[t - 1]
            h.append(1 if (s[t] == 1 or hp == 1) else 0)
            states.append(STATES.index((s[t], h[t])))
        lw = log_init[states[0]] + log_emit[0, states[0]]
        for t in range(1, T):
            lw += log_trans[t, states[t - 1], states[t]] + log_emit[t, states[t]]
        w = np.exp(lw) if np.isfinite(lw) else 0.0
        weights.append((tuple(s), w))
        for t in range(T):
            if s[t] == 1:
                post_num[t] += w
    Z = sum(w for _, w in weights)
    Z0 = [w for s, w in weights if not any(s)][0]
    return {"log_Z": float(np.log(Z)), "log_Z0": float(np.log(Z0)),
            "p_y1": float(1.0 - Z0 / Z), "post": post_num / Z}
