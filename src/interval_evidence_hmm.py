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
                       Diverged in revision 3 (README section 4.1); kept for
                       the record only.
  regimes = R > 1      (2026-09-10, module-1 iteration 2) video-level
                       reliability mixture: a latent z_v in {0..R-1} per video
                       with its own emission parameters (q_f, r_f, q_c, r_c)^z
                       and mixing weights pi_z; rates, p0 shared. Verdicts of a
                       video are conditionally independent given (s, z_v), so
                       the video-level correlation of VLM errors (a video is
                       rated all-hate or all-benign as a whole) is modelled
                       instead of assumed away. Exact inference: one
                       forward-backward per regime, mixed by the regime's
                       marginal likelihood times pi_z. EM: regime
                       responsibilities per video (positive AND negative
                       videos), closed-form weighted-count M-step for the
                       emissions and pi, unchanged L-BFGS step for the rates.
                       R = 1 is numerically identical to the model above.

  text = True          (2026-09-10, module-1 iteration 2 "text evidence") a
                       third and fourth observation family that is FREE and
                       always available: the per-second hate probability of a
                       frozen text classifier over the ASR chunk (family "asr")
                       and over the OCR text of the fine window (family "ocr")
                       covering the second (data/text_hate/<corpus>/<id>.npz).
                       Each second's probability is binned into 10 levels
                       (10 levels, edges TEXT_BINS) and emitted at its segment conditioned
                       on s_g alone (categorical tables t_fam[s, bin], 2 x 10
                       per family), weighted so that one ASR chunk / one OCR
                       window counts once in total (weights w_t = 1 / seconds
                       covered), tempered by text_weight (protocol constant
                       1). Missing text (NaN) emits nothing. M-step: weighted
                       counts by P(s_g), negatives count under s = 0 (like r_f).
                       VLM verdicts remain the only PAID evidence; the
                       acquisition policy asks where the free evidence leaves
                       the output uncertain.

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


TEXT_BINS = np.array([0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99])   # 10 levels of the text classifier's hate probability
N_TEXT_BINS = len(TEXT_BINS) + 1
TEXT_FAMILIES = ("asr", "ocr")


def text_counts(grid, p, w, bins=TEXT_BINS):
    """Weighted bin counts (G, N_TEXT_BINS) of per-second probabilities p (T,)
    (NaN = no text) with per-second weights w (T,); second t belongs to the
    segment containing its midpoint fraction (t + 0.5) / T."""
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    T = len(p)
    C = np.zeros((grid["G"], N_TEXT_BINS))
    if T == 0:
        return C
    frac = (np.arange(T) + 0.5) / T
    seg = np.clip(np.searchsorted(grid["start"], frac, side="right") - 1, 0, grid["G"] - 1)
    ok = np.isfinite(p) & (w > 0)
    b = np.searchsorted(bins, np.clip(p[ok], 0.0, 1.0), side="right")
    np.add.at(C, (seg[ok], b), w[ok])
    return C


def text_observation(grid, arrays):
    """dict family -> (G, N_TEXT_BINS) counts from the text_hate npz arrays
    (keys p_asr / w_asr / p_ocr / w_ocr), or None when no text at all."""
    out = {}
    for fam in TEXT_FAMILIES:
        p, w = arrays.get("p_" + fam), arrays.get("w_" + fam)
        if p is None or w is None:
            continue
        C = text_counts(grid, p, w)
        if C.sum() > 0:
            out[fam] = C
    return out or None


class IntervalEvidenceHMM:
    def __init__(self, k=30, j=4, positive_constraint=False, video_effect=False,
                 normalized_time=False, regimes=1, text=False, text_weight=1.0):
        self.k, self.j = int(k), int(j)
        self.grid = make_grid(self.k, self.j)
        # free text evidence families (module-1 iteration 2): categorical tables t[s, bin]
        self.text = bool(text)
        self.text_weight = float(text_weight)
        self.text_emit = {fam: np.full((2, N_TEXT_BINS), 1.0 / N_TEXT_BINS) for fam in TEXT_FAMILIES}
        # normalized_time: transitions run over fractions of the video instead
        # of seconds (state persistence scales with the video's own length)
        self.normalized_time = bool(normalized_time)
        self.lam01, self.lam10 = (2.0, 2.0) if normalized_time else (0.02, 0.02)
        self.p0 = np.array([0.6, 0.4])
        self.R = int(regimes)
        assert self.R >= 1
        # per-regime emission parameters (regime 0 = nominal; R = 1 is the plain model)
        if self.R == 1:
            init = [(0.8, 0.1, 0.9, 0.2)]
        else:
            init = [(0.8, 0.1, 0.9, 0.2),          # nominal
                    (0.9, 0.6, 0.95, 0.7),         # over-firing: high false-alarm rates
                    (0.3, 0.05, 0.4, 0.1)]         # under-firing: low hit rates
            init = (init + [(0.8, 0.1, 0.9, 0.2)] * self.R)[:self.R]
        self.q_f_z = np.array([x[0] for x in init])
        self.r_f_z = np.array([x[1] for x in init])
        self.q_c_z = np.array([x[2] for x in init])
        self.r_c_z = np.array([x[3] for x in init])
        self.pi = np.ones(self.R) / self.R if self.R == 1 else np.array(([0.6] + [0.4 / (self.R - 1)] * (self.R - 1)))
        self.positive_constraint = bool(positive_constraint)
        self.video_effect = bool(video_effect)
        self.sigma = 0.5 if video_effect else 0.0
        self.fit_history = []
        self._masks = self._build_masks()

    # scalar views of regime 0 (the plain model's parameters; setters require R = 1)
    def _get(self, name):
        return float(getattr(self, name + "_z")[0])

    def _set(self, name, value):
        assert self.R == 1, "set %s_z for a regime mixture" % name
        getattr(self, name + "_z")[0] = float(value)

    q_f = property(lambda self: self._get("q_f"), lambda self, v: self._set("q_f", v))
    r_f = property(lambda self: self._get("r_f"), lambda self, v: self._set("r_f", v))
    q_c = property(lambda self: self._get("q_c"), lambda self, v: self._set("q_c", v))
    r_c = property(lambda self: self._get("r_c"), lambda self, v: self._set("r_c", v))

    # ------------------------------------------------------------------ model
    def params(self):
        d = {"model": "interval", "k": self.k, "j": self.j,
             "lam01": self.lam01, "lam10": self.lam10, "p0": self.p0.tolist(),
             "q_fine": self.q_f, "r_fine": self.r_f, "q_coarse": self.q_c,
             "r_coarse": self.r_c, "positive_constraint": self.positive_constraint,
             "video_effect": self.video_effect, "sigma": self.sigma,
             "normalized_time": self.normalized_time, "regimes": self.R,
             "text": self.text, "text_weight": self.text_weight}
        if self.text:
            d["text_emit"] = {fam: self.text_emit[fam].tolist() for fam in TEXT_FAMILIES}
            d["text_bins"] = TEXT_BINS.tolist()
        if self.R > 1:
            d.update({"pi": self.pi.tolist(), "q_fine_z": self.q_f_z.tolist(),
                      "r_fine_z": self.r_f_z.tolist(), "q_coarse_z": self.q_c_z.tolist(),
                      "r_coarse_z": self.r_c_z.tolist()})
        if self.fit_history:
            d["fit_loglik"] = [float(x) for x in self.fit_history]
        return d

    @classmethod
    def from_params(cls, d):
        m = cls(d["k"], d["j"], d.get("positive_constraint", False),
                d.get("video_effect", False), d.get("normalized_time", False),
                int(d.get("regimes", 1)), bool(d.get("text", False)), float(d.get("text_weight", 1.0)))
        if m.text:
            m.text_emit = {fam: np.asarray(d["text_emit"][fam], float) for fam in TEXT_FAMILIES}
        m.lam01, m.lam10 = float(d["lam01"]), float(d["lam10"])
        m.p0 = np.asarray(d["p0"], float)
        if m.R > 1:
            m.pi = np.asarray(d["pi"], float)
            m.q_f_z = np.asarray(d["q_fine_z"], float)
            m.r_f_z = np.asarray(d["r_fine_z"], float)
            m.q_c_z = np.asarray(d["q_coarse_z"], float)
            m.r_c_z = np.asarray(d["r_coarse_z"], float)
        else:
            m.q_f, m.r_f = float(d["q_fine"]), float(d["r_fine"])
            m.q_c, m.r_c = float(d["q_coarse"]), float(d["r_coarse"])
        m.sigma = float(d.get("sigma", 0.0))
        m.fit_history = [float(x) for x in d.get("fit_loglik", [])]
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

    def _text_loglik(self, xt):
        """(G, 2) tempered log-likelihood of the text observations under s = 0 / 1
        (zeros when the model has no text or the video no text)."""
        ll = np.zeros((self.grid["G"], 2))
        if not self.text or not xt:
            return ll
        for fam, C in xt.items():
            ll += C @ np.log(np.clip(self.text_emit[fam], 1e-6, 1.0)).T
        return self.text_weight * ll

    def _emissions(self, b_fine, b_coarse, delta=0.0, w_fine=1.0, w_coarse=1.0, z=0, xt=None):
        """e[g, state] = product of the OR-factors emitted at segment g under regime z
        (times the text factors of the segment, conditioned on s only)."""
        gr = self.grid
        G = gr["G"]
        e = np.ones((G, N_STATES))
        q_f, q_c = self.q_f_z[z], self.q_c_z[z]
        r_f = _sigmoid(_logit(self.r_f_z[z]) + delta) if delta else self.r_f_z[z]
        r_c = _sigmoid(_logit(self.r_c_z[z]) + delta) if delta else self.r_c_z[z]
        b_fine = np.asarray(b_fine)
        b_coarse = np.asarray(b_coarse)
        for g in range(G):
            if gr["fine_end"][g] and w_fine > 0:
                b = int(b_fine[gr["fine_of"][g]])
                if b != MISSING:
                    p1 = np.where(HF_OF == 1, q_f, r_f)
                    e[g] *= (p1 if b else 1.0 - p1) ** w_fine
            if gr["coarse_end"][g] and w_coarse > 0:
                b = int(b_coarse[gr["coarse_of"][g]])
                if b != MISSING:
                    p1 = np.where(HC_OF == 1, q_c, r_c)
                    e[g] *= (p1 if b else 1.0 - p1) ** w_coarse
        if self.text and xt:
            ll = self._text_loglik(xt)                       # (G, 2)
            ll = ll - ll.max(1, keepdims=True)               # per-segment scale is irrelevant to the posterior
            e *= np.exp(ll[:, S_OF])                         # (_posterior_video adds the shift back into logmarg)
        return e

    def _text_shift(self, xt):
        """Sum over segments of the per-segment maximum text log-likelihood that
        _emissions divides out (a per-video constant)."""
        if not (self.text and xt):
            return 0.0
        return float(self._text_loglik(xt).max(1).sum())

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
                         constrain=False, Tm=None, xt=None):
        """Mixture over (quadrature node, regime) components of (gamma, xi).

        Returns dict: gamma (G,8) and xi mixed over all components; post_w and
        deltas per component (for the video-effect sigma step); rho (R,) regime
        responsibilities; gamma_z (R,G,8) node-mixed posterior per regime;
        logmarg = log marginal likelihood of the video (conditional on "at
        least one s = 1" when constrain), including the pi_z / node weights."""
        Tm = self._transitions(duration) if Tm is None else Tm
        deltas, wts = self._nodes()
        shift = self._text_shift(xt)
        gammas, xis, logw, comp_delta, comp_z = [], [], [], [], []
        for z in range(self.R):
            for d, wt in zip(deltas, wts):
                e = self._emissions(b_fine, b_coarse, d, w_fine, w_coarse, z=z, xt=xt)
                g, x, lz = self._fb(Tm, e)
                if constrain:
                    lz0 = self._log_all_zero(Tm, e)
                    w0 = min(np.exp(lz0 - lz), 1.0 - 1e-9)
                    g[:, 0] = (g[:, 0] - w0) / (1.0 - w0)
                    g[:, 1:] /= (1.0 - w0)
                    x[1:, 0, 0] = (x[1:, 0, 0] - w0) / (1.0 - w0)
                    m = np.ones((N_STATES, N_STATES), bool)
                    m[0, 0] = False
                    x[1:, m] /= (1.0 - w0)
                    lz = lz + np.log1p(-np.exp(min(lz0 - lz, -1e-12)))
                lz = lz + shift
                gammas.append(g)
                xis.append(x)
                logw.append(np.log(wt) + np.log(self.pi[z]) + lz)
                comp_delta.append(d)
                comp_z.append(z)
        logw = np.asarray(logw)
        lmax = logw.max()
        logmarg = lmax + np.log(np.exp(logw - lmax).sum())
        post_w = np.exp(logw - logmarg)
        gamma = np.clip(sum(p * g for p, g in zip(post_w, gammas)), 0.0, 1.0)
        xi = sum(p * x for p, x in zip(post_w, xis))
        comp_z = np.asarray(comp_z)
        rho = np.array([post_w[comp_z == z].sum() for z in range(self.R)])
        gamma_z = np.zeros((self.R, gamma.shape[0], N_STATES))
        for z in range(self.R):
            idx = np.where(comp_z == z)[0]
            wz = post_w[idx] / max(rho[z], 1e-300)
            gamma_z[z] = np.clip(sum(wz[i] * gammas[j] for i, j in enumerate(idx)), 0.0, 1.0)
        return {"gamma": gamma, "xi": xi, "post_w": post_w, "deltas": np.asarray(comp_delta),
                "rho": rho, "gamma_z": gamma_z, "logmarg": float(logmarg)}

    # -------------------------------------------------------------------- fit
    def _neg_loglik_z(self, bf, bc, xt=None):
        """Per-regime log-likelihood of a negative video (h = 0 everywhere, only
        observed verdicts; plus the text factors under s = 0): (R,)."""
        text_ll = float(self._text_loglik(xt)[:, 0].sum()) if (self.text and xt) else 0.0
        bf = np.asarray(bf)
        bc = np.asarray(bc)
        of = bf[bf != MISSING]
        oc = bc[bc != MISSING]
        n1f, n0f = int((of == 1).sum()), int((of == 0).sum())
        n1c, n0c = int((oc == 1).sum()), int((oc == 0).sum())
        return (n1f * np.log(self.r_f_z) + n0f * np.log1p(-self.r_f_z)
                + n1c * np.log(self.r_c_z) + n0c * np.log1p(-self.r_c_z) + text_ll)

    def fit(self, pos_videos, neg_videos, n_iter=30):
        """pos/neg_videos: lists of (b_fine (k,), b_coarse (j,), duration_seconds[, text
        observation dict from text_observation() or None])."""
        gr = self.grid
        R = self.R
        pos_videos = [tuple(v) + (None,) * (4 - len(v)) for v in pos_videos]
        neg_videos = [tuple(v) + (None,) * (4 - len(v)) for v in neg_videos]
        any_text = self.text and any(xt for *_, xt in pos_videos + neg_videos)
        obs_f = lambda bf: int(np.sum(np.asarray(bf) != MISSING))     # noqa: E731
        obs_c = lambda bc: int(np.sum(np.asarray(bc) != MISSING))     # noqa: E731
        neg_stats = [(int(np.sum(np.asarray(bf) == 1)), obs_f(bf),
                      int(np.sum(np.asarray(bc) == 1)), obs_c(bc)) for bf, bc, _, _ in neg_videos]
        neg_f = sum(s[0] for s in neg_stats)
        neg_nf = sum(s[1] for s in neg_stats)
        neg_c = sum(s[2] for s in neg_stats)
        neg_nc = sum(s[3] for s in neg_stats)
        if neg_videos and neg_nf > 0:          # a family with no observed verdict keeps its init (rule-6 must-fix)
            self.r_f_z[0] = (neg_f + 1e-3) / neg_nf
        if neg_videos and neg_nc > 0:
            self.r_c_z[0] = (neg_c + 1e-3) / neg_nc
        self.fit_history = []
        for _ in range(n_iter):
            g0 = np.zeros(2) + 1e-3
            nf = np.zeros((R, 2)) + 1e-3      # observed fine verdicts by P(h_fine), per regime
            cf = np.zeros((R, 2)) + 1e-3      # ... that were 1
            nh = np.zeros((R, 2)) + 1e-3      # observed coarse verdicts by P(h_coarse)
            cc = np.zeros((R, 2)) + 1e-3
            trans = []               # (dt, xi_s (2x2)) per gap, for the rate M-step
            e_delta2 = 0.0
            rho_sum = np.zeros(R)
            loglik = 0.0
            tc = {fam: np.zeros((2, N_TEXT_BINS)) + 1e-3 for fam in TEXT_FAMILIES}   # text counts by P(s)
            for bf, bc, dur, xt in pos_videos:
                Tm = self._transitions(dur)
                post = self._posterior_video(bf, bc, dur, constrain=self.positive_constraint, Tm=Tm, xt=xt)
                gamma, xi = post["gamma"], post["xi"]
                loglik += post["logmarg"]
                rho_sum += post["rho"]
                e_delta2 += float(np.sum(post["post_w"] * post["deltas"] ** 2))
                ps = np.stack([gamma[:, S_OF == 0].sum(1), gamma[:, S_OF == 1].sum(1)], 1)
                g0 += ps[0]
                if self.text and xt:
                    for fam, C in xt.items():
                        tc[fam] += ps.T @ C                                   # (2, G) @ (G, bins)
                dt = self._segment_dt(dur)[:-1]
                for g in range(1, gr["G"]):
                    x2 = np.zeros((2, 2))
                    for a in range(2):
                        for b in range(2):
                            x2[a, b] = xi[g][S_OF == a][:, S_OF == b].sum()
                    trans.append((dt[g - 1], x2))
                bf = np.asarray(bf)
                bc = np.asarray(bc)
                for z in range(R):
                    gz = post["gamma_z"][z]
                    rz = post["rho"][z]
                    for g in np.where(gr["fine_end"])[0]:
                        b = int(bf[gr["fine_of"][g]])
                        if b == MISSING:
                            continue
                        ph = np.array([gz[g, HF_OF == 0].sum(), gz[g, HF_OF == 1].sum()])
                        nf[z] += rz * ph
                        cf[z] += rz * ph * b
                    for g in np.where(gr["coarse_end"])[0]:
                        b = int(bc[gr["coarse_of"][g]])
                        if b == MISSING:
                            continue
                        ph = np.array([gz[g, HC_OF == 0].sum(), gz[g, HC_OF == 1].sum()])
                        nh[z] += rz * ph
                        cc[z] += rz * ph * b
            # negative videos: h = 0 everywhere; regime responsibilities in closed form
            neg_rho = np.zeros((len(neg_videos), R))
            for i, (bf, bc, _, xt) in enumerate(neg_videos):
                if self.text and xt:
                    for fam, C in xt.items():
                        tc[fam][0] += C.sum(0)
                lw = np.log(self.pi) + self._neg_loglik_z(bf, bc, xt)
                m = lw.max()
                loglik += m + np.log(np.exp(lw - m).sum())
                neg_rho[i] = np.exp(lw - m) / np.exp(lw - m).sum()
            ns = np.asarray(neg_stats, dtype=float).reshape(-1, 4)
            negf_z = neg_rho.T @ ns[:, 0] if len(neg_videos) else np.zeros(R)
            negnf_z = neg_rho.T @ ns[:, 1] if len(neg_videos) else np.zeros(R)
            negc_z = neg_rho.T @ ns[:, 2] if len(neg_videos) else np.zeros(R)
            negnc_z = neg_rho.T @ ns[:, 3] if len(neg_videos) else np.zeros(R)
            self.fit_history.append(float(loglik))
            # M-step: emissions per regime (a family with no observed verdict at
            # all keeps its parameters: nothing to estimate them from)
            if any_text:
                for fam in TEXT_FAMILIES:
                    self.text_emit[fam] = tc[fam] / tc[fam].sum(1, keepdims=True)
            if any(obs_f(bf) for bf, _, _, _ in pos_videos) or neg_nf > 0:
                self.q_f_z = cf[:, 1] / nf[:, 1]
                self.r_f_z = (cf[:, 0] + negf_z) / (nf[:, 0] + negnf_z)
            if any(obs_c(bc) for _, bc, _, _ in pos_videos) or neg_nc > 0:
                self.q_c_z = cc[:, 1] / nh[:, 1]
                self.r_c_z = (cc[:, 0] + negc_z) / (nh[:, 0] + negnc_z)
            if R > 1:                                # mixture: keep emissions off exact 0 / 1 (0 * log 0 in the E-step)
                for name in ("q_f_z", "r_f_z", "q_c_z", "r_c_z"):
                    setattr(self, name, np.clip(getattr(self, name), 1e-4, 1.0 - 1e-4))
            self.p0 = g0 / g0.sum()
            if R > 1:
                tot = rho_sum + neg_rho.sum(0)
                self.pi = tot / tot.sum()
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
    def infer(self, b_fine, b_coarse, duration=1.0, w_fine=1.0, w_coarse=1.0, Tm=None, xt=None):
        """Label-free inference for one video. Returns dict: gamma (G,8) regime-
        mixed posterior, gamma_z (R,G,8), rho (R,), p_s (G,), p_hf (k,),
        p_hc (j,), pred_fine (k,) = p(b_w = 1 | observed verdicts) under the
        mixture (meaningful for unobserved w). MISSING verdicts emit nothing."""
        post = self._posterior_video(b_fine, b_coarse, duration, w_fine, w_coarse, Tm=Tm, xt=xt)
        p_s, p_hf, p_hc = self.summarize_gamma(post["gamma"])
        fe = np.where(self.grid["fine_end"])[0]
        pred = np.zeros(self.k)
        for z in range(self.R):
            phz = np.array([post["gamma_z"][z][g, HF_OF == 1].sum() for g in fe])
            pred += post["rho"][z] * (self.q_f_z[z] * phz + self.r_f_z[z] * (1.0 - phz))
        return {"gamma": post["gamma"], "gamma_z": post["gamma_z"], "rho": post["rho"],
                "p_s": p_s, "p_hf": p_hf, "p_hc": p_hc, "pred_fine": pred,
                "logmarg": post["logmarg"]}

    def any_hate_logodds(self, b_fine, b_coarse, duration, w_fine=1.0, w_coarse=1.0, xt=None):
        """logit P(at least one segment has s = 1 | observed verdicts) for one video
        (R = 1): the video-level evidence term of the decomposed log-odds (module-1
        iteration 3). Uses the all-zero path probability of the same chain."""
        assert self.R == 1, "any_hate_logodds is the R = 1 formula"
        Tm = self._transitions(duration)
        e = self._emissions(b_fine, b_coarse, 0.0, w_fine, w_coarse, z=0, xt=xt)
        _, _, lz = self._fb(Tm, e)
        lz0 = self._log_all_zero(Tm, e)
        p0 = float(np.clip(np.exp(lz0 - lz), 1e-6, 1.0 - 1e-6))
        return float(np.log1p(-p0) - np.log(p0))

    def posterior(self, b_fine, b_coarse, duration, w_fine=1.0, w_coarse=1.0, xt=None):
        """(segment P(s_g=1) (G,), coarse-interval P(h_j=1) (J,)). Never uses a label."""
        gamma = self._posterior_video(b_fine, b_coarse, duration, w_fine, w_coarse, xt=xt)["gamma"]
        p_s = gamma[:, S_OF == 1].sum(1)
        ends = np.where(self.grid["coarse_end"])[0]
        p_h = np.array([gamma[g, HC_OF == 1].sum() for g in ends])
        return p_s, p_h

    def posterior_gamma(self, b_fine, b_coarse, duration, w_fine=1.0, w_coarse=1.0, xt=None):
        """Full augmented-state posterior gamma (G, 8), regime-mixed; MISSING
        verdicts emit nothing. Used by the adaptive-query module (0 labels)."""
        return self._posterior_video(b_fine, b_coarse, duration, w_fine, w_coarse, xt=xt)["gamma"]

    def summarize_gamma(self, gamma):
        """(P(s_g=1) (G,), P(h_fine_w=1) (k,), P(h_coarse_j=1) (j,)) from gamma."""
        p_s = gamma[:, S_OF == 1].sum(1)
        fe = np.where(self.grid["fine_end"])[0]
        ce = np.where(self.grid["coarse_end"])[0]
        p_hf = np.array([gamma[g, HF_OF == 1].sum() for g in fe])
        p_hc = np.array([gamma[g, HC_OF == 1].sum() for g in ce])
        return p_s, p_hf, p_hc

    def predictive_fine(self, gamma):
        """p(b_w = 1 | observed verdicts) for every fine window w (k,) from a
        plain (R = 1) posterior gamma: q_f * P(h_w=1) + r_f * (1 - P(h_w=1)).
        Regime mixtures need the per-regime posteriors: use infer()["pred_fine"]."""
        assert self.R == 1, "predictive_fine(gamma) is the R = 1 formula; use infer() for a mixture"
        _, p_hf, _ = self.summarize_gamma(gamma)
        return self.q_f * p_hf + self.r_f * (1.0 - p_hf)

    def posterior_log_odds(self, b_fine, b_coarse, duration, eps=1e-6, **kw):
        p_s, _ = self.posterior(b_fine, b_coarse, duration, **kw)
        return np.log(p_s + eps) - np.log(1.0 - p_s + eps)

    def coarse_of_rows(self, row_bounds, duration):
        return rows_from_segments(self.grid["coarse_of"], self.grid, row_bounds, duration)
