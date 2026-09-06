"""Interval evidence HMM over frozen-VLM verdicts (candidate 3 revision 3, 2026-09-07).

Replaces the index hierarchy of src/verdict_hmm.py (fine window t -> coarse
block floor(4t/30)) by the intervals each VLM call actually looked at.

Per video of duration D seconds, each verdict v has a time interval
[a_v, b_v) (a fraction of D: fine window k/30, coarse block j/4). All interval
boundaries of all families are merged into one sorted grid of G segments
(32 for 30/4). Generative model:
  s_g in {0,1}         latent hate state of segment g, constant on the segment;
                       between consecutive segments a two-state continuous-time
                       chain with rates lam01 (0->1), lam10 (1->0) run for the
                       length of the previous segment, so the transition is
                       P = expm(Q * dt) in closed form
  b_v ~ Bernoulli(q_f if OR_{g in [a_v,b_v)} s_g else r_f)   (fine family)
  b_v ~ Bernoulli(q_c if OR_{g in [a_v,b_v)} s_g else r_c)   (coarse family)
A missing verdict (b_v = -1) emits nothing; every other factor is unchanged,
which is what the adaptive-query replay needs.

Each family is a partition of [0, 1), so exact inference runs on the augmented
state (s, h_fine, h_coarse) = 8 states, h_f = "hate seen so far in the current
interval of family f"; the OR factor of an interval is emitted at its last
segment. Parameters (lam01, lam10, p0, q_f, r_f, q_c, r_c) are estimated from
TRAIN video labels only: negative videos fix r_f, r_c; positive videos go
through EM (rates by L-BFGS on the expected complete-data log-likelihood).

Options (each a separate ablation arm of the candidate):
  positive_constraint  during EM, condition every positive training video on
                       "at least one segment has s = 1" (the all-zero path is
                       removed from the posterior). Inference never uses the
                       video label, so this changes the fitted parameters only.
  video_effect         per-video shift delta_v ~ N(0, sigma^2) on the logit of
                       both false-alarm rates, shared by all verdicts of the
                       video, integrated by 5-point Gauss-Hermite quadrature;
                       sigma estimated in EM (r_f, r_c updated by the same
                       counts as without the effect: approximate M-step).

No frame labels are used anywhere.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

MISSING = -1
N_STATES = 8            # (s, h_fine, h_coarse)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(5)   # for N(0, sigma^2): delta = sqrt(2) sigma x, weight w / sqrt(pi)
_GH_W = _GH_W / np.sqrt(np.pi)


def _state(s, hf, hc):
    return s * 4 + hf * 2 + hc


S_OF = np.array([st // 4 for st in range(N_STATES)])
HF_OF = np.array([(st // 2) % 2 for st in range(N_STATES)])
HC_OF = np.array([st % 2 for st in range(N_STATES)])


def make_grid(k, j):
    """Segments of the merged 1/k and 1/j boundaries (fractions of the video).

    Returns dict with start/end (G,), fine_of/coarse_of (G,) interval index of
    each segment, fine_end/coarse_end (G,) bool = last segment of its interval,
    fine_new/coarse_new (G,) bool = first segment of an interval."""
    b = np.unique(np.concatenate([np.arange(k + 1) / k, np.arange(j + 1) / j]))
    start, end = b[:-1], b[1:]
    mid = (start + end) / 2.0
    fine_of = np.clip(np.floor(mid * k).astype(int), 0, k - 1)
    coarse_of = np.clip(np.floor(mid * j).astype(int), 0, j - 1)
    G = len(start)
    fine_end = np.array([g == G - 1 or fine_of[g + 1] != fine_of[g] for g in range(G)])
    coarse_end = np.array([g == G - 1 or coarse_of[g + 1] != coarse_of[g] for g in range(G)])
    fine_new = np.array([g == 0 or fine_of[g - 1] != fine_of[g] for g in range(G)])
    coarse_new = np.array([g == 0 or coarse_of[g - 1] != coarse_of[g] for g in range(G)])
    return dict(start=start, end=end, fine_of=fine_of, coarse_of=coarse_of,
                fine_end=fine_end, coarse_end=coarse_end, fine_new=fine_new,
                coarse_new=coarse_new, G=G)


def rows_from_segments(values, grid, row_bounds, duration):
    """Value of the segment containing each row's midpoint (rows in seconds)."""
    rb = np.asarray(row_bounds, dtype=np.float64)
    frac = ((rb[:, 0] + rb[:, 1]) / 2.0) / max(float(duration), 1e-6)
    idx = np.searchsorted(grid["start"], frac, side="right") - 1
    return np.asarray(values)[np.clip(idx, 0, grid["G"] - 1)]


def _ctmc(lam01, lam10, dt):
    """Two-state transition matrices for gaps dt (n,) -> (n, 2, 2)."""
    tot = lam01 + lam10
    dt = np.asarray(dt, dtype=np.float64)
    decay = np.exp(-tot * dt)
    p01 = lam01 / tot * (1.0 - decay)
    p10 = lam10 / tot * (1.0 - decay)
    T = np.empty((len(dt), 2, 2))
    T[:, 0, 0], T[:, 0, 1] = 1.0 - p01, p01
    T[:, 1, 0], T[:, 1, 1] = p10, 1.0 - p10
    return T


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class IntervalEvidenceHMM:
    def __init__(self, k=30, j=4, positive_constraint=False, video_effect=False,
                 normalized_time=False):
        self.k, self.j = int(k), int(j)
        self.grid = make_grid(self.k, self.j)
        # normalized_time: transitions run over fractions of the video instead
        # of seconds (state persistence scales with the video's own length)
        self.normalized_time = bool(normalized_time)
        self.lam01, self.lam10 = (2.0, 2.0) if normalized_time else (0.02, 0.02)
        self.p0 = np.array([0.6, 0.4])
        self.q_f, self.r_f, self.q_c, self.r_c = 0.8, 0.1, 0.9, 0.2
        self.positive_constraint = bool(positive_constraint)
        self.video_effect = bool(video_effect)
        self.sigma = 0.5 if video_effect else 0.0
        self._masks = self._build_masks()

    # ------------------------------------------------------------------ model
    def params(self):
        return {"model": "interval", "k": self.k, "j": self.j,
                "lam01": self.lam01, "lam10": self.lam10, "p0": self.p0.tolist(),
                "q_fine": self.q_f, "r_fine": self.r_f, "q_coarse": self.q_c,
                "r_coarse": self.r_c, "positive_constraint": self.positive_constraint,
                "video_effect": self.video_effect, "sigma": self.sigma,
                "normalized_time": self.normalized_time}

    @classmethod
    def from_params(cls, d):
        m = cls(d["k"], d["j"], d.get("positive_constraint", False),
                d.get("video_effect", False), d.get("normalized_time", False))
        m.lam01, m.lam10 = float(d["lam01"]), float(d["lam10"])
        m.p0 = np.asarray(d["p0"], float)
        m.q_f, m.r_f = float(d["q_fine"]), float(d["r_fine"])
        m.q_c, m.r_c = float(d["q_coarse"]), float(d["r_coarse"])
        m.sigma = float(d.get("sigma", 0.0))
        return m

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.params(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_params(json.load(fh))

    def _build_masks(self):
        """Deterministic h-update structure: allowed[g, i, j] for the augmented
        transition from state i at segment g-1 to state j at segment g."""
        gr = self.grid
        G = gr["G"]
        allowed = np.zeros((G, N_STATES, N_STATES), dtype=bool)
        for g in range(G):
            for i in range(N_STATES):
                for jj in range(N_STATES):
                    s_new = S_OF[jj]
                    hf_prev = 0 if gr["fine_new"][g] else HF_OF[i]
                    hc_prev = 0 if gr["coarse_new"][g] else HC_OF[i]
                    if HF_OF[jj] != (1 if (s_new or hf_prev) else 0):
                        continue
                    if HC_OF[jj] != (1 if (s_new or hc_prev) else 0):
                        continue
                    allowed[g, i, jj] = True
        init = np.zeros(N_STATES, dtype=bool)
        for jj in range(N_STATES):
            init[jj] = HF_OF[jj] == S_OF[jj] and HC_OF[jj] == S_OF[jj]
        return allowed, init

    def _segment_dt(self, duration):
        scale = 1.0 if self.normalized_time else float(duration)
        return (self.grid["end"] - self.grid["start"]) * scale

    def _transitions(self, duration, lam01=None, lam10=None, p0=None):
        """Tm[g] (8x8) for g >= 1 (Tm[0] holds the initial distribution in row 0)."""
        lam01 = self.lam01 if lam01 is None else lam01
        lam10 = self.lam10 if lam10 is None else lam10
        p0 = self.p0 if p0 is None else p0
        allowed, init = self._masks
        G = self.grid["G"]
        dt = self._segment_dt(duration)[:-1]                 # gap g-1 -> g uses len(g-1)
        P = _ctmc(lam01, lam10, dt)                          # (G-1, 2, 2)
        Tm = np.zeros((G, N_STATES, N_STATES))
        Tm[0, 0, :] = np.where(init, p0[S_OF], 0.0)
        for g in range(1, G):
            Tm[g] = allowed[g] * P[g - 1][S_OF[:, None], S_OF[None, :]]
        return Tm

    def _emissions(self, b_fine, b_coarse, delta=0.0, w_fine=1.0, w_coarse=1.0):
        """e[g, state] = product of the OR-factors emitted at segment g."""
        gr = self.grid
        G = gr["G"]
        e = np.ones((G, N_STATES))
        r_f = _sigmoid(_logit(self.r_f) + delta) if delta else self.r_f
        r_c = _sigmoid(_logit(self.r_c) + delta) if delta else self.r_c
        b_fine = np.asarray(b_fine)
        b_coarse = np.asarray(b_coarse)
        for g in range(G):
            if gr["fine_end"][g] and w_fine > 0:
                b = int(b_fine[gr["fine_of"][g]])
                if b != MISSING:
                    p1 = np.where(HF_OF == 1, self.q_f, r_f)
                    e[g] *= (p1 if b else 1.0 - p1) ** w_fine
            if gr["coarse_end"][g] and w_coarse > 0:
                b = int(b_coarse[gr["coarse_of"][g]])
                if b != MISSING:
                    p1 = np.where(HC_OF == 1, self.q_c, r_c)
                    e[g] *= (p1 if b else 1.0 - p1) ** w_coarse
        return e

    @staticmethod
    def _fb(Tm, e):
        """Scaled forward-backward. Returns gamma (G,8), xi (G,8,8) with xi[0]
        unused, log Z."""
        G = Tm.shape[0]
        al = np.zeros((G, N_STATES))
        be = np.zeros((G, N_STATES))
        c = np.zeros(G)
        al[0] = Tm[0, 0] * e[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for g in range(1, G):
            al[g] = (al[g - 1] @ Tm[g]) * e[g]
            c[g] = al[g].sum()
            al[g] /= c[g]
        be[-1] = 1.0
        for g in range(G - 2, -1, -1):
            be[g] = Tm[g + 1] @ (e[g + 1] * be[g + 1])
            be[g] /= c[g + 1]
        gamma = al * be
        gamma /= gamma.sum(1, keepdims=True)
        xi = np.zeros((G, N_STATES, N_STATES))
        for g in range(1, G):
            x = al[g - 1][:, None] * Tm[g] * (e[g] * be[g])[None, :]
            xi[g] = x / max(x.sum(), 1e-300)
        return gamma, xi, float(np.log(c).sum())

    def _log_all_zero(self, Tm, e):
        """log-likelihood of the single path s = 0 everywhere (state 0)."""
        G = Tm.shape[0]
        lz = np.log(Tm[0, 0, 0]) + np.log(e[0, 0])
        for g in range(1, G):
            lz += np.log(Tm[g, 0, 0]) + np.log(e[g, 0])
        return lz

    def _nodes(self):
        if self.video_effect and self.sigma > 0:
            return np.sqrt(2.0) * self.sigma * _GH_X, _GH_W
        return np.zeros(1), np.ones(1)

    def _posterior_video(self, b_fine, b_coarse, duration, w_fine=1.0, w_coarse=1.0,
                         constrain=False, Tm=None):
        """Mixture over quadrature nodes of (gamma, xi); also node weights."""
        Tm = self._transitions(duration) if Tm is None else Tm
        deltas, wts = self._nodes()
        gammas, xis, logZs, logZ0s = [], [], [], []
        for d in deltas:
            e = self._emissions(b_fine, b_coarse, d, w_fine, w_coarse)
            g, x, lz = self._fb(Tm, e)
            gammas.append(g)
            xis.append(x)
            logZs.append(lz)
            logZ0s.append(self._log_all_zero(Tm, e) if constrain else -np.inf)
        logZs = np.asarray(logZs)
        logZ0s = np.asarray(logZ0s)
        if constrain:
            # remove the all-zero path from every node's posterior
            for n in range(len(deltas)):
                w0 = np.exp(logZ0s[n] - logZs[n])
                w0 = min(w0, 1.0 - 1e-9)
                gammas[n][:, 0] = (gammas[n][:, 0] - w0) / (1.0 - w0)
                gammas[n][:, 1:] /= (1.0 - w0)
                xis[n][1:, 0, 0] = (xis[n][1:, 0, 0] - w0) / (1.0 - w0)
                m = np.ones((N_STATES, N_STATES), bool)
                m[0, 0] = False
                xis[n][1:, m] /= (1.0 - w0)
            logmarg = logZs + np.log1p(-np.exp(np.minimum(logZ0s - logZs, -1e-12)))
        else:
            logmarg = logZs
        lw = np.log(wts) + logmarg
        lw -= lw.max()
        post_w = np.exp(lw) / np.exp(lw).sum()
        gamma = sum(p * g for p, g in zip(post_w, gammas))
        xi = sum(p * x for p, x in zip(post_w, xis))
        gamma = np.clip(gamma, 0.0, 1.0)
        return gamma, xi, post_w, deltas

    # -------------------------------------------------------------------- fit
    def fit(self, pos_videos, neg_videos, n_iter=30):
        """pos/neg_videos: lists of (b_fine (k,), b_coarse (j,), duration_seconds)."""
        gr = self.grid
        k, j = self.k, self.j
        obs_f = lambda bf: int(np.sum(np.asarray(bf) != MISSING))     # noqa: E731
        obs_c = lambda bc: int(np.sum(np.asarray(bc) != MISSING))     # noqa: E731
        neg_f = sum(int(np.sum(np.asarray(bf) == 1)) for bf, _, _ in neg_videos)
        neg_c = sum(int(np.sum(np.asarray(bc) == 1)) for _, bc, _ in neg_videos)
        neg_nf = sum(obs_f(bf) for bf, _, _ in neg_videos)
        neg_nc = sum(obs_c(bc) for _, bc, _ in neg_videos)
        if neg_videos:
            self.r_f = (neg_f + 1e-3) / max(neg_nf, 1)
            self.r_c = (neg_c + 1e-3) / max(neg_nc, 1)
        for _ in range(n_iter):
            g0 = np.zeros(2) + 1e-3
            nf = np.zeros(2) + 1e-3      # observed fine verdicts by P(h_fine)
            cf = np.zeros(2) + 1e-3      # ... that were 1
            nh = np.zeros(2) + 1e-3      # observed coarse verdicts by P(h_coarse)
            cc = np.zeros(2) + 1e-3
            trans = []               # (dt, xi_s (2x2)) per gap, for the rate M-step
            e_delta2 = 0.0
            for bf, bc, dur in pos_videos:
                Tm = self._transitions(dur)
                gamma, xi, pw, deltas = self._posterior_video(
                    bf, bc, dur, constrain=self.positive_constraint, Tm=Tm)
                e_delta2 += float(np.sum(pw * deltas ** 2))
                ps = np.stack([gamma[:, S_OF == 0].sum(1), gamma[:, S_OF == 1].sum(1)], 1)
                g0 += ps[0]
                dt = self._segment_dt(dur)[:-1]
                for g in range(1, gr["G"]):
                    x2 = np.zeros((2, 2))
                    for a in range(2):
                        for b in range(2):
                            x2[a, b] = xi[g][S_OF == a][:, S_OF == b].sum()
                    trans.append((dt[g - 1], x2))
                bf = np.asarray(bf)
                bc = np.asarray(bc)
                for g in np.where(gr["fine_end"])[0]:
                    b = int(bf[gr["fine_of"][g]])
                    if b == MISSING:
                        continue
                    ph = np.array([gamma[g, HF_OF == 0].sum(), gamma[g, HF_OF == 1].sum()])
                    nf += ph
                    cf += ph * b
                for g in np.where(gr["coarse_end"])[0]:
                    b = int(bc[gr["coarse_of"][g]])
                    if b == MISSING:
                        continue
                    ph = np.array([gamma[g, HC_OF == 0].sum(), gamma[g, HC_OF == 1].sum()])
                    nh += ph
                    cc += ph * b
            # M-step: emissions (negative videos: h = 0 everywhere)
            self.q_f = float(cf[1] / nf[1])
            self.q_c = float(cc[1] / nh[1])
            self.r_f = float((cf[0] + neg_f) / (nf[0] + neg_nf))
            self.r_c = float((cc[0] + neg_c) / (nh[0] + neg_nc))
            self.p0 = g0 / g0.sum()
            if self.video_effect:
                self.sigma = float(np.sqrt(max(e_delta2 / max(len(pos_videos), 1), 1e-6)))
            # M-step: rates by L-BFGS on the expected complete-data log-likelihood
            if trans:
                dts = np.array([t for t, _ in trans])
                X = np.stack([x for _, x in trans])           # (n, 2, 2)

                def nll(theta):
                    l01, l10 = np.exp(theta)
                    P = np.clip(_ctmc(l01, l10, dts), 1e-12, 1.0)
                    return -float(np.sum(X * np.log(P)))

                res = minimize(nll, np.log([self.lam01, self.lam10]), method="L-BFGS-B",
                               bounds=[(np.log(1e-5), np.log(500.0))] * 2)
                self.lam01, self.lam10 = [float(v) for v in np.exp(res.x)]
        return self

    # -------------------------------------------------------------- inference
    def posterior(self, b_fine, b_coarse, duration, w_fine=1.0, w_coarse=1.0):
        """(segment P(s_g=1) (G,), coarse-interval P(h_j=1) (J,)). Never uses a label."""
        gamma, _, _, _ = self._posterior_video(b_fine, b_coarse, duration, w_fine, w_coarse)
        p_s = gamma[:, S_OF == 1].sum(1)
        ends = np.where(self.grid["coarse_end"])[0]
        p_h = np.array([gamma[g, HC_OF == 1].sum() for g in ends])
        return p_s, p_h

    def posterior_log_odds(self, b_fine, b_coarse, duration, eps=1e-6, **kw):
        p_s, _ = self.posterior(b_fine, b_coarse, duration, **kw)
        return np.log(p_s + eps) - np.log(1.0 - p_s + eps)

    def coarse_of_rows(self, row_bounds, duration):
        return rows_from_segments(self.grid["coarse_of"], self.grid, row_bounds, duration)
