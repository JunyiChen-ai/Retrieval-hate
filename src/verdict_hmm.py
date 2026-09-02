"""Hierarchical evidence HMM over frozen-VLM segment verdicts.

Generative model (per video, K fine windows, J coarse blocks, block(t) maps a
fine window to its coarse block):
  s_t in {0,1}      latent hate state of fine window t, Markov chain (A, p0)
  b_fine_t in {0,1} verdict of the fine window: P(b=1|s=1)=q_f, P(b=1|s=0)=r_f
  h_j = OR_{t in j} s_t
  b_coarse_j        verdict of the coarse block: P(b=1|h=1)=q_c, P(b=1|h=0)=r_c
Exact inference on the augmented state (s_t, h_t) where h_t = "hate seen so far
in the current block"; states are 0=(s0,h0), 1=(s1,h1), 2=(s0,h1).  The coarse
verdict is emitted at the last fine window of its block.

Parameters are estimated from TRAIN video labels only: negative videos have
s = 0 everywhere (they fix r_f, r_c together with the s=0 mass of positives);
positive videos contribute through EM (Baum-Welch with the augmented chain).
No frame labels are used anywhere.

`posterior_log_odds` returns, per fine window, log P(s_t=1|b) - log P(s_t=0|b)
with optional evidence tempering exponents (w_fine, w_coarse in [0,1]).
"""

from __future__ import annotations

import json

import numpy as np

STATES = ((0, 0), (1, 1), (0, 1))   # (s, h)


def _block_map(k, j):
    return np.clip((np.arange(k) * j) // k, 0, j - 1)


class HierEvidenceHMM:
    def __init__(self, k=30, j=4):
        self.k = int(k)
        self.j = int(j)
        self.block = _block_map(self.k, self.j)
        self.end = np.array([t == self.k - 1 or self.block[t + 1] != self.block[t]
                             for t in range(self.k)])
        self.A = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.p0 = np.array([0.6, 0.4])
        self.q_f, self.r_f, self.q_c, self.r_c = 0.8, 0.1, 0.9, 0.2

    # ------------------------------------------------------------------ model
    def params(self):
        return {"k": self.k, "j": self.j, "A": self.A.tolist(),
                "p0": self.p0.tolist(), "q_fine": self.q_f, "r_fine": self.r_f,
                "q_coarse": self.q_c, "r_coarse": self.r_c}

    @classmethod
    def from_params(cls, d):
        m = cls(d["k"], d["j"])
        m.A = np.asarray(d["A"], float)
        m.p0 = np.asarray(d["p0"], float)
        m.q_f, m.r_f = float(d["q_fine"]), float(d["r_fine"])
        m.q_c, m.r_c = float(d["q_coarse"]), float(d["r_coarse"])
        return m

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.params(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_params(json.load(fh))

    def _build(self, b_fine, b_coarse, w_fine=1.0, w_coarse=1.0,
               A=None, p0=None, independent=False, flat_coarse=False):
        """Per-step transition matrices Tm[t] (3x3) and emissions e[t] (3,).

        independent: no temporal coupling (A replaced by the stationary
        row p0 at every step).  flat_coarse: the coarse verdict is emitted at
        every fine window of its block (no OR structure), tempered by 1/|block|.
        """
        A = self.A if A is None else A
        p0 = self.p0 if p0 is None else p0
        k = self.k
        Tm = np.zeros((k, 3, 3))
        e = np.ones((k, 3))
        for t in range(k):
            newblock = (t == 0) or (self.block[t] != self.block[t - 1])
            for i, (s_prev, h_prev) in enumerate(STATES):
                for jj, (s, h) in enumerate(STATES):
                    hp = 0 if newblock else h_prev
                    h_new = 1 if (s == 1 or hp == 1) else 0
                    if h != h_new:
                        continue
                    if t == 0 or independent:
                        Tm[t, i, jj] = p0[s]
                    else:
                        Tm[t, i, jj] = A[s_prev, s]
            for jj, (s, h) in enumerate(STATES):
                pf = self.q_f if s else self.r_f
                pe = pf if b_fine[t] else 1.0 - pf
                e[t, jj] = pe ** w_fine
                bc = b_coarse[self.block[t]]
                if flat_coarse:
                    pc = self.q_c if s else self.r_c
                    pe_c = pc if bc else 1.0 - pc
                    n_in = float(np.sum(self.block == self.block[t]))
                    e[t, jj] *= pe_c ** (w_coarse / n_in)
                elif self.end[t]:
                    pc = self.q_c if h else self.r_c
                    pe_c = pc if bc else 1.0 - pc
                    e[t, jj] *= pe_c ** w_coarse
        return Tm, e

    @staticmethod
    def _fb(Tm, e):
        k = Tm.shape[0]
        al = np.zeros((k, 3))
        be = np.zeros((k, 3))
        al[0] = Tm[0, 0] * e[0]
        al[0] /= al[0].sum()
        for t in range(1, k):
            al[t] = (al[t - 1] @ Tm[t]) * e[t]
            al[t] /= al[t].sum()
        be[-1] = 1.0
        for t in range(k - 2, -1, -1):
            be[t] = Tm[t + 1] @ (e[t + 1] * be[t + 1])
            be[t] /= be[t].sum()
        g = al * be
        g /= g.sum(1, keepdims=True)
        return g, al, be

    # -------------------------------------------------------------------- fit
    def fit(self, pos_videos, neg_videos, n_iter=40):
        """pos/neg_videos: lists of (b_fine (k,), b_coarse (j,)) binary arrays."""
        k, j = self.k, self.j
        neg_f = sum(int(bf.sum()) for bf, _ in neg_videos)
        neg_c = sum(int(bc.sum()) for _, bc in neg_videos)
        if neg_videos:
            self.r_f = (neg_f + 1e-3) / (k * len(neg_videos))
            self.r_c = (neg_c + 1e-3) / (j * len(neg_videos))
        for _ in range(n_iter):
            xi = np.zeros((2, 2)) + 1e-3
            g0 = np.zeros(2) + 1e-3
            ns = np.zeros(2) + 1e-3
            cf = np.zeros(2) + 1e-3
            nh = np.zeros(2) + 1e-3
            cc = np.zeros(2) + 1e-3
            for bf, bc in pos_videos:
                Tm, e = self._build(bf, bc)
                g, al, be = self._fb(Tm, e)
                ps = np.stack([g[:, 0] + g[:, 2], g[:, 1]], 1)
                for t in range(1, k):
                    x = al[t - 1][:, None] * Tm[t] * (e[t] * be[t])[None, :]
                    x /= x.sum()
                    xi[0, 0] += x[0, 0] + x[0, 2] + x[2, 0] + x[2, 2]
                    xi[0, 1] += x[0, 1] + x[2, 1]
                    xi[1, 0] += x[1, 0] + x[1, 2]
                    xi[1, 1] += x[1, 1]
                g0 += ps[0]
                ns += ps.sum(0)
                cf += np.array([(ps[:, 0] * bf).sum(), (ps[:, 1] * bf).sum()])
                for t in np.where(self.end)[0]:
                    ph = np.array([g[t, 0], g[t, 1] + g[t, 2]])
                    nh += ph
                    cc += ph * bc[self.block[t]]
            self.A = xi / xi.sum(1, keepdims=True)
            self.p0 = g0 / g0.sum()
            self.q_f = float(cf[1] / ns[1])
            self.q_c = float(cc[1] / nh[1])
            self.r_f = float((cf[0] + neg_f) / (ns[0] + k * len(neg_videos)))
            self.r_c = float((cc[0] + neg_c) / (nh[0] + j * len(neg_videos)))
        return self

    # -------------------------------------------------------------- inference
    def posterior(self, b_fine, b_coarse, w_fine=1.0, w_coarse=1.0, **kw):
        """(fine-window P(s_t=1), coarse-block P(h_j=1))."""
        Tm, e = self._build(b_fine, b_coarse, w_fine, w_coarse, **kw)
        g, _, _ = self._fb(Tm, e)
        p_s = g[:, 1]
        p_h = np.array([g[t, 1] + g[t, 2] for t in np.where(self.end)[0]])
        return p_s, p_h

    def posterior_log_odds(self, b_fine, b_coarse, w_fine=1.0, w_coarse=1.0,
                           eps=1e-6, **kw):
        p_s, _ = self.posterior(b_fine, b_coarse, w_fine, w_coarse, **kw)
        return np.log(p_s + eps) - np.log(1.0 - p_s + eps)


def binarize(levels, threshold=2):
    return (np.asarray(levels, float) >= threshold).astype(int)


def rows_from_windows(values, n_rows, k):
    """Value of the fine window containing each row's midpoint (n_rows rows)."""
    idx = np.clip(((np.arange(n_rows) + 0.5) * k / max(n_rows, 1)).astype(int),
                  0, k - 1)
    return np.asarray(values)[idx]
