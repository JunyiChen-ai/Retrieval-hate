"""Hierarchical evidence HMM with a context-conditioned fine-verdict emission.

Same generative model as src/verdict_hmm.HierEvidenceHMM (latent Markov hate
state s_t over K fine windows, block OR h_j, coarse verdicts emitted at block
ends) except that the fine verdict is emitted conditionally on observed
context:

    P(b_fine_t = 1 | s_t = s, c_t) = p[s][c_t],
    c_t = (b_coarse_{block(t)}, b_fine_{t-1})     (b_fine_0 := 0 before t = 1)

The context variables are observations (the coarse verdict of the window's own
block and the previous fine verdict), so the joint
P(b_coarse, b_fine | s, h) = prod_j P(b_coarse_j | h_j) prod_t P(b_fine_t | s_t, c_t)
is a valid directed factorisation and forward-backward stays exact.

`context` selects which observations condition the emission:
  "global"    none  (1 context; identical to HierEvidenceHMM)
  "b4"        coarse verdict of the block only (2 contexts)
  "bprev"     previous fine verdict only (2 contexts)
  "cond"      both (4 contexts)

Parameters from TRAIN video labels only: the s = 0 rates p[0][c] use the
negative videos (s = h = 0 throughout) plus the s = 0 posterior mass of the
positives; the s = 1 rates p[1][c], the coarse rates and (A, p0) come from EM
on the positives (M-step = posterior-weighted per-context rates). Idea
discovery 2026-09-05, cluster C7 ("coarse-first conditional label model").
"""

from __future__ import annotations

import numpy as np

from verdict_hmm import HierEvidenceHMM, STATES

CONTEXTS = ("global", "b4", "bprev", "cond")


class CondEvidenceHMM(HierEvidenceHMM):
    def __init__(self, k=30, j=4, context="cond"):
        super().__init__(k, j)
        assert context in CONTEXTS, context
        self.context = context
        self.n_ctx = {"global": 1, "b4": 2, "bprev": 2, "cond": 4}[context]
        self.p_fine = np.array([[self.r_f] * self.n_ctx, [self.q_f] * self.n_ctx])  # (2 states, n_ctx)

    # ----------------------------------------------------------------- context
    def ctx(self, b_fine, b_coarse):
        """(k,) context index of every fine window."""
        b4 = np.asarray(b_coarse)[self.block].astype(int)
        bprev = np.concatenate([[0], np.asarray(b_fine)[:-1]]).astype(int)
        if self.context == "global":
            return np.zeros(self.k, int)
        if self.context == "b4":
            return b4
        if self.context == "bprev":
            return bprev
        return 2 * b4 + bprev

    def params(self):
        d = super().params()
        d.update({"context": self.context, "p_fine": self.p_fine.tolist()})
        return d

    @classmethod
    def from_params(cls, d):
        m = cls(d["k"], d["j"], d.get("context", "cond"))
        m.A = np.asarray(d["A"], float)
        m.p0 = np.asarray(d["p0"], float)
        m.q_c, m.r_c = float(d["q_coarse"]), float(d["r_coarse"])
        m.p_fine = np.asarray(d["p_fine"], float)
        m.q_f, m.r_f = float(m.p_fine[1].mean()), float(m.p_fine[0].mean())
        return m

    # ------------------------------------------------------------------- build
    def _build(self, b_fine, b_coarse, w_fine=1.0, w_coarse=1.0,
               A=None, p0=None, independent=False, flat_coarse=False):
        A = self.A if A is None else A
        p0 = self.p0 if p0 is None else p0
        k = self.k
        c = self.ctx(b_fine, b_coarse)
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
                    Tm[t, i, jj] = p0[s] if (t == 0 or independent) else A[s_prev, s]
            for jj, (s, h) in enumerate(STATES):
                pf = self.p_fine[s, c[t]]
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

    # --------------------------------------------------------------------- fit
    def fit(self, pos_videos, neg_videos, n_iter=40, monotone=True):
        k, j, nc = self.k, self.j, self.n_ctx
        neg_cf = np.zeros(nc)
        neg_n = np.zeros(nc)
        neg_c = 0
        for bf, bc in neg_videos:
            c = self.ctx(bf, bc)
            np.add.at(neg_cf, c, np.asarray(bf, float))
            np.add.at(neg_n, c, 1.0)
            neg_c += int(np.sum(bc))
        if neg_videos:
            self.p_fine[0] = (neg_cf + 1e-3) / np.maximum(neg_n, 1e-9)
            self.r_c = (neg_c + 1e-3) / (j * len(neg_videos))
        for _ in range(n_iter):
            xi = np.zeros((2, 2)) + 1e-3
            g0 = np.zeros(2) + 1e-3
            ns = np.zeros((2, nc)) + 1e-3
            cf = np.zeros((2, nc)) + 1e-3
            nh = np.zeros(2) + 1e-3
            cc = np.zeros(2) + 1e-3
            for bf, bc in pos_videos:
                c = self.ctx(bf, bc)
                Tm, e = self._build(bf, bc)
                g, al, be = self._fb(Tm, e)
                ps = np.stack([g[:, 0] + g[:, 2], g[:, 1]], 1)   # (k, 2) P(s_t = 0/1)
                for t in range(1, k):
                    x = al[t - 1][:, None] * Tm[t] * (e[t] * be[t])[None, :]
                    x /= x.sum()
                    xi[0, 0] += x[0, 0] + x[0, 2] + x[2, 0] + x[2, 2]
                    xi[0, 1] += x[0, 1] + x[2, 1]
                    xi[1, 0] += x[1, 0] + x[1, 2]
                    xi[1, 1] += x[1, 1]
                g0 += ps[0]
                bfa = np.asarray(bf, float)
                for s in (0, 1):
                    np.add.at(ns[s], c, ps[:, s])
                    np.add.at(cf[s], c, ps[:, s] * bfa)
                for t in np.where(self.end)[0]:
                    ph = np.array([g[t, 0], g[t, 1] + g[t, 2]])
                    nh += ph
                    cc += ph * bc[self.block[t]]
            self.A = xi / xi.sum(1, keepdims=True)
            self.p0 = g0 / g0.sum()
            self.p_fine[1] = cf[1] / ns[1]
            self.p_fine[0] = (cf[0] + neg_cf) / (ns[0] + neg_n)
            if monotone:
                # identifiability: the hate state must not lower the fine-verdict
                # rate in any context (the unconstrained EM label-switches on
                # HateClipSeg, gate run 1)
                self.p_fine[1] = np.maximum(self.p_fine[1], self.p_fine[0] + 0.05)
            self.q_c = float(cc[1] / nh[1])
            self.r_c = float((cc[0] + neg_c) / (nh[0] + j * len(neg_videos)))
        self.q_f, self.r_f = float(self.p_fine[1].mean()), float(self.p_fine[0].mean())
        return self
