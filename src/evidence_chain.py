"""Differentiable evidence chain (batched, log-space, PyTorch).

Latent hate state s_t in {0,1} per snippet t, augmented with the block-OR
variable h ("hate seen so far in the current block"), exactly as in
src/verdict_hmm.py:  states 0=(s0,h0), 1=(s1,h1), 2=(s0,h1); h resets at the
first snippet of every block.

Transitions (t-1 -> t) for the s part, conditioned on a per-video density d:
    t = 0:   p0(d) = (1-d, d)
    t > 0:   A(d) = [[1 - a d,      a d    ],
                     [a (1-d),  1 - a (1-d)]]      stationary distribution (1-d, d)
with a fixed switching rate a.  The h part is deterministic (h = s OR h_prev,
h_prev := 0 at a block start).

Emission log-potentials, normalised to zero for (s0,h0):
    e_t(s1,h1) = u_t + gf_t * phi_f_t + [end_t] gc_{j(t)} * phi_c_{j(t)}
    e_t(s0,h1) =                          [end_t] gc_{j(t)} * phi_c_{j(t)}
where u_t is the network's content evidence (log-likelihood ratio for s=1),
phi_f_t the fine-verdict log-likelihood ratio (already divided by the number
of snippets in its window), phi_c_j the coarse-verdict log-likelihood ratio
emitted at the last snippet of block j, and gf / gc in [0,1] the reliability
gates.

Outputs: log Z (all paths), log Z0 (the all-zero path), P(y=1) = 1 - Z0/Z,
and the posterior P(s_t = 1 | everything).  Padded steps (mask False) get an
identity transition and zero emission, so they do not change Z.
"""

from __future__ import annotations

import math

import numpy as np
import torch

NEG = -1e4   # "log 0" that keeps gradients finite


def block_layout(block_of_t: torch.Tensor):
    """block_of_t [B,T] long (non-decreasing per row) -> (new_block [B,T], block_end [B,T]) bool."""
    prev = torch.cat([block_of_t[:, :1] - 1, block_of_t[:, :-1]], dim=1)
    nxt = torch.cat([block_of_t[:, 1:], block_of_t[:, -1:] + 1], dim=1)
    return block_of_t != prev, block_of_t != nxt


def log_transitions(d: torch.Tensor, a: float, new_block: torch.Tensor, mask: torch.Tensor):
    """d [B] in (0,1); returns log T [B,T,3,3] (row = previous state, col = next).

    Step 0 uses p0(d) (from a virtual all-zero predecessor); padded steps use
    the identity.
    """
    B, T = new_block.shape
    eps = 1e-6
    d = d.clamp(eps, 1 - eps)
    A00 = torch.log(1 - a * d)
    A01 = torch.log(a * d)
    A10 = torch.log(a * (1 - d))
    A11 = torch.log(1 - a * (1 - d))
    neg = torch.full_like(A00, NEG)
    # continuing step (h_prev kept)
    cont = torch.stack([
        torch.stack([A00, A01, neg], -1),       # from (s0,h0)
        torch.stack([neg, A11, A10], -1),       # from (s1,h1)
        torch.stack([neg, A01, A00], -1),       # from (s0,h1)
    ], -2)                                      # [B,3,3]
    # block start (h_prev := 0): only h = s allowed
    start = torch.stack([
        torch.stack([A00, A01, neg], -1),
        torch.stack([A10, A11, neg], -1),
        torch.stack([A00, A01, neg], -1),
    ], -2)
    # t = 0: p0 row regardless of the previous state
    p0 = torch.stack([torch.log(1 - d), torch.log(d), neg], -1)      # [B,3]
    init = p0[:, None, :].expand(B, 3, 3)
    ident = torch.full((3, 3), NEG, device=d.device, dtype=d.dtype)
    ident.fill_diagonal_(0.0)
    logT = torch.where(new_block[:, :, None, None], start[:, None], cont[:, None])   # [B,T,3,3]
    logT = torch.where((torch.arange(T, device=d.device) == 0)[None, :, None, None],
                       init[:, None], logT)
    logT = torch.where(mask[:, :, None, None], logT, ident[None, None])
    return logT


def emissions(u, phi_f, gf, phi_c_rows, gc_rows, block_end, mask):
    """All [B,T]; phi_c_rows / gc_rows already gathered per snippet (value of its block).
    Returns log e [B,T,3] with the (s0,h0) column at 0."""
    coarse = torch.where(block_end, gc_rows * phi_c_rows, torch.zeros_like(u))
    e1 = u + gf * phi_f + coarse
    e2 = coarse
    e0 = torch.zeros_like(u)
    log_e = torch.stack([e0, e1, e2], -1)
    return torch.where(mask[:, :, None], log_e, torch.zeros_like(log_e))


def forward_backward(log_e, logT):
    """log_e [B,T,3], logT [B,T,3,3] -> (log_Z [B], posterior [B,T,3])."""
    B, T, S = log_e.shape
    alphas = []
    # step 0: predecessor is the virtual all-zero state (row 0 of logT[:,0])
    la = logT[:, 0, 0, :] + log_e[:, 0]
    alphas.append(la)
    for t in range(1, T):
        la = torch.logsumexp(la[:, :, None] + logT[:, t], dim=1) + log_e[:, t]
        alphas.append(la)
    log_alpha = torch.stack(alphas, 1)
    log_Z = torch.logsumexp(log_alpha[:, -1], dim=-1)
    betas = [torch.zeros(B, S, device=log_e.device, dtype=log_e.dtype)]
    lb = betas[0]
    for t in range(T - 2, -1, -1):
        lb = torch.logsumexp(logT[:, t + 1] + (log_e[:, t + 1] + lb)[:, None, :], dim=2)
        betas.append(lb)
    log_beta = torch.stack(betas[::-1], 1)
    post = torch.softmax(log_alpha + log_beta, dim=-1)
    return log_Z, post


def log_Z0(logT, mask):
    """log-likelihood of the all-(s0,h0) path (emission there is 0 by normalisation)."""
    z = logT[:, :, 0, 0]
    return torch.where(mask, z, torch.zeros_like(z)).sum(1)


def chain(u, phi_f, gf, phi_c_rows, gc_rows, d, a, block_of_t, mask):
    """Convenience wrapper. Returns dict(log_Z, log_Z0, p_video, post_s1 [B,T])."""
    new_block, block_end = block_layout(block_of_t)
    logT = log_transitions(d, a, new_block, mask)
    log_e = emissions(u, phi_f, gf, phi_c_rows, gc_rows, block_end, mask)
    lz, post = forward_backward(log_e, logT)
    lz0 = log_Z0(logT, mask)
    p_video = 1.0 - torch.exp((lz0 - lz).clamp(max=0.0))
    return {"log_Z": lz, "log_Z0": lz0, "p_video": p_video,
            "post_s1": post[..., 1], "post": post}


# ----------------------------------------------------------- self-check
def _brute_force(u, phi_f, gf, phi_c, gc, d, a, block_of_t):
    """Enumerate all s sequences (numpy, one video). Returns (log_Z, log_Z0, post_s1)."""
    T = len(u)
    J = int(block_of_t.max()) + 1
    ends = [t for t in range(T) if t == T - 1 or block_of_t[t + 1] != block_of_t[t]]
    logps, s1 = [], np.zeros(T)
    total = 0.0
    for bits in range(2 ** T):
        s = [(bits >> t) & 1 for t in range(T)]
        lp = math.log(d if s[0] else 1 - d)
        for t in range(1, T):
            A = [[1 - a * d, a * d], [a * (1 - d), 1 - a * (1 - d)]]
            lp += math.log(A[s[t - 1]][s[t]])
        h = {}
        cur = 0
        for t in range(T):
            if t == 0 or block_of_t[t] != block_of_t[t - 1]:
                cur = 0
            cur = 1 if (s[t] or cur) else 0
            h[t] = cur
            if s[t]:
                lp += u[t] + gf[t] * phi_f[t]
            if t in ends and h[t]:
                lp += gc[block_of_t[t]] * phi_c[block_of_t[t]]
        logps.append((bits, lp))
    lps = np.array([lp for _, lp in logps])
    m = lps.max()
    Z = np.exp(lps - m).sum()
    log_Z = m + math.log(Z)
    log_Z0 = logps[0][1]
    for (bits, lp) in logps:
        w = math.exp(lp - log_Z)
        for t in range(T):
            if (bits >> t) & 1:
                s1[t] += w
    return log_Z, log_Z0, s1


def self_check(n_cases=20, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        T = int(rng.integers(2, 8))
        J = int(rng.integers(1, 3))
        block_of_t = np.sort(rng.integers(0, J, size=T))
        block_of_t = block_of_t - block_of_t.min()
        u, phi_f = rng.normal(size=T), rng.normal(size=T)
        gf = rng.uniform(size=T)
        phi_c = rng.normal(size=J + 1)
        gc = rng.uniform(size=J + 1)
        d, a = float(rng.uniform(.05, .95)), float(rng.uniform(.05, .95))
        bl, bz0, bs1 = _brute_force(u, phi_f, gf, phi_c, gc, d, a, block_of_t)
        tt = lambda x: torch.tensor(np.asarray(x, np.float64))[None]   # noqa: E731
        bo = torch.tensor(block_of_t)[None]
        out = chain(tt(u), tt(phi_f), tt(gf), tt(phi_c[block_of_t]), tt(gc[block_of_t]),
                    torch.tensor([d], dtype=torch.float64), a, bo,
                    torch.ones(1, T, dtype=torch.bool))
        assert abs(out["log_Z"].item() - bl) < 1e-8, (out["log_Z"].item(), bl)
        assert abs(out["log_Z0"].item() - bz0) < 1e-8, (out["log_Z0"].item(), bz0)
        assert np.abs(out["post_s1"][0].numpy() - bs1).max() < 1e-8
    # padding invariance: appending padded steps must not change Z or posteriors
    T = 5
    u = torch.randn(1, T, dtype=torch.float64)
    bo = torch.tensor([[0, 0, 1, 1, 1]])
    base = chain(u, u * 0.3, torch.ones_like(u) * .7, u * 0, torch.ones_like(u), torch.tensor([.3], dtype=torch.float64), .2, bo,
                 torch.ones(1, T, dtype=torch.bool))
    pad = torch.cat([u, torch.zeros(1, 3, dtype=torch.float64)], 1)
    bop = torch.cat([bo, torch.tensor([[1, 1, 1]])], 1)
    mk = torch.tensor([[True] * T + [False] * 3])
    out = chain(pad, pad * 0.3, torch.ones_like(pad) * .7, pad * 0, torch.ones_like(pad), torch.tensor([.3], dtype=torch.float64), .2, bop, mk)
    assert abs(out["log_Z"].item() - base["log_Z"].item()) < 1e-8
    assert torch.allclose(out["post_s1"][0, :T], base["post_s1"][0])
    return True


if __name__ == "__main__":
    print("evidence_chain self-check:", self_check())
