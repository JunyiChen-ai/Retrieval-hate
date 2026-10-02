"""Existing stopping statistics promoted without numerical changes from the r5 checks."""
import numpy as np
from scipy.stats import spearmanr

def instability(scores):
    """Per call k (1..n): 1 - Spearman(scores[k], scores[k-1]); index 0 (before any call) is +inf."""
    out = [np.inf]
    for k in range(1, len(scores)):
        a, b = np.asarray(scores[k]), np.asarray(scores[k - 1])
        if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
            out.append(0.0 if np.allclose(a, b) else 1.0)
        else:
            out.append(float(1.0 - spearmanr(a, b).correlation))
    return out


def _hb(p):
    """Binary entropy in bits of a probability array."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def rule_values(run, T):
    """Per rule, the value read before call k+1 (k = 0..n-1) so that policy.stop_calls applies unchanged.
    Next-question rules (README 17.4 baseline): eig, voi, voi_norm = VOI / T (variant (b)); last-change rules: stab1,
    stab2; <rule>_f<k> = a floor of k calls (variant (c), kept for reference only: the user's 2026-09-29 ruling forbids
    floors); the prior-stratified thresholds (variant (a)) are handled in main.
    State-confidence rules (README 17.4 "no floor", the rule family of the stopping literature: the posterior after
    k calls, not the value of the next question; Wald's SPRT, Naghshvar & Javidi 2013, FrameExit / AdaFrame gates,
    VideoAgent's sufficiency score, EcoFrame's output-entropy gate):
      hG     binary entropy of the video posterior P(G = 1 | answers)
      hT     mean over seconds of the binary entropy of the per-second posterior p_t
      hmax   max(hG, hT): ask while either the video-level or the per-second map is uncertain
      vsum   sum over seconds of p_t (1 - p_t): the expected squared error of the per-second map, weighted as the
             pooled metrics weight seconds (a long uncertain video counts more)
      vmean  the same per second (length-free)"""
    n = len(run["eig"])
    inst = instability(run["scores"])                 # length n + 1: inst[k] = change made by call k
    vals = {"eig": list(run["eig"]), "voi": list(run["voi"]),
            "voi_norm": [v / float(T) for v in run["voi"]],
            "stab1": [inst[k] for k in range(n)],
            "stab2": [max(inst[k], inst[k - 1] if k >= 1 else np.inf) for k in range(n)]}
    for r in ("eig", "voi", "stab1"):
        for f in FLOORS:
            vals["%s_f%d" % (r, f)] = [np.inf if k < f else vals[r][k] for k in range(n)]
    hG = _hb(run["p_G"][:n])
    ps = [np.asarray(run["scores"][k], dtype=np.float64) for k in range(n)]
    hT = np.array([float(np.mean(_hb(p))) for p in ps]) if n else np.zeros(0)
    vals["hG"] = [float(x) for x in hG]
    vals["hT"] = [float(x) for x in hT]
    vals["hmax"] = [float(max(a, b)) for a, b in zip(hG, hT)]
    vals["vsum"] = [float(np.sum(p * (1.0 - p))) for p in ps]
    vals["vmean"] = [float(np.mean(p * (1.0 - p))) for p in ps]
    # last-change rules on the log-odds scale (README 17.4 "no floor", second family: the change the last answer made
    # to the scores on the scale the pooled ranking sees; +inf before the first call, so the first call is always made)
    lp = [_logit(run["scores"][k]) for k in range(n + 1)]
    lg = _logit(run["p_G"][:n + 1])
    d = [np.inf] + [float(np.mean(np.abs(lp[k] - lp[k - 1]))) for k in range(1, n + 1)]
    dg = [np.inf] + [float(abs(lg[k] - lg[k - 1])) for k in range(1, n + 1)]
    vals["dlogit"] = d[:n]                                                    # mean over seconds, last call
    vals["dlogit2"] = [max(d[k], d[k - 1] if k >= 1 else np.inf) for k in range(n)]   # larger of the last two
    vals["dlogit3"] = [float(np.mean(d[max(1, k - 2):k + 1])) if k >= 1 else np.inf for k in range(n)]  # mean of last 3
    ds = [np.inf] + [float(np.sum(np.abs(lp[k] - lp[k - 1]))) for k in range(1, n + 1)]
    vals["dlsum"] = ds[:n]                                                    # sum over seconds (pool-weighted)
    vals["dlsum2"] = [max(ds[k], ds[k - 1] if k >= 1 else np.inf) for k in range(n)]
    vals["dG"] = dg[:n]                                                       # video-level log-odds change
    vals["dG2"] = [max(dg[k], dg[k - 1] if k >= 1 else np.inf) for k in range(n)]
    # remaining-information rules (README 17.4, the nonmyopic direction): the EIG summed over ALL candidate questions
    # still askable (first-order approximation of the information left in the video; a slowly moving video with many
    # small-EIG questions keeps a large sum, a converged one does not), the sum of the three largest, and the sum per
    # candidate (recorded by cpolicy.run_batch; absent in older dumps)
    if "eig_sum" in run and len(run["eig_sum"]) >= n:
        vals["eigsum"] = [float(x) for x in run["eig_sum"][:n]]
        vals["eigtop3"] = [float(x) for x in run["eig_top3"][:n]]
        vals["eigmean"] = [float(x) / max(1, m) for x, m in zip(run["eig_sum"][:n], run["n_cand"][:n])]
    # expected remaining movement (README 17.4, nonmyopic; cpolicy.run_batch record_erm; absent in older dumps)
    if "erm" in run and len(run["erm"]) >= n:
        vals["erm"] = [float(x) for x in run["erm"][:n]]
        vals["ermsum"] = [float(x) for x in run["ermsum"][:n]]
    # one-sided undecidedness (README 17.4 third round): the band applies only to videos on the negative side of the
    # decision boundary (P(G) < .5); a likely-positive video is left to the change rule (its calls go to localisation)
    pg = np.asarray(run["p_G"][:n], dtype=np.float64)
    vals["hGn"] = [float(h) if g < 0.5 else 0.0 for h, g in zip(hG, pg)]
    for u, name in ((0.5, "n50"), (0.2, "n20")):
        for base in ("dlogit", "dlogit2", "dlsum"):
            vals["%s_%s" % (base, name)] = [np.inf if (hG[k] >= u and pg[k] < 0.5) else vals[base][k] for k in range(n)]
    # composites: an undecided video (H(P(G)) >= u bits; u = .5: P(G) in [.11, .89]; u = .2: [.03, .97], the decision
    # band of a sequential test) keeps asking whatever the last change was; a decided one stops by the change rule
    for u, name in ((0.5, "u50"), (0.2, "u20")):
        for base in ("dlogit", "dlogit2", "dlogit3"):
            vals["%s_%s" % (base, name)] = [np.inf if hG[k] >= u else vals[base][k] for k in range(n)]
    return vals


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


QMIX = {"qmix": ("dlogit", "stab1"), "qmix2": ("dlogit2", "stab2"), "qmixG": ("dlogit", "stab1", "hG"),
        "qmix2G": ("dlogit2", "stab2", "hG"),
        # second round (README 17.4): the pool-weighted change (dlsum) in place of the per-second mean, the video-level
        # undecidedness alone as the state component, and the per-second entropy (hT) as a localisation state
        "qmixD": ("dlogit", "hG"), "qmixD2": ("dlogit2", "hG"), "qmixS": ("dlsum", "hG"), "qmixS2": ("dlsum2", "hG"),
        "qmixSG": ("dlsum", "stab1", "hG"), "qmixS2G": ("dlsum2", "stab2", "hG"),
        "qmixT": ("dlogit", "stab1", "hG", "hT"), "qmixST": ("dlsum", "stab1", "hG", "hT"), "qmixDT": ("dlogit", "hG", "hT"),
        "qmixS_T": ("dlsum", "hT"),
        # third round: the remaining-information sum as the state component
        "qmixE": ("dlsum", "eigsum"), "qmixEG": ("dlsum", "eigsum", "hG"), "qmixE3": ("dlsum", "eigtop3"),
        "qmixDE": ("dlogit", "eigsum"), "qmixSE": ("dlsum", "stab1", "eigsum"),
        # one-sided undecidedness as the state component
        "qmixN": ("dlogit", "stab1", "hGn"), "qmix2N": ("dlogit2", "stab2", "hGn"), "qmixSN": ("dlsum", "stab1", "hGn"),
        "qmixDN": ("dlogit2", "hGn"), "qmixSGn": ("dlsum", "hGn"),
        # expected remaining movement as the state component
        "qmixRS": ("dlsum", "ermsum"), "qmixRN": ("ermsum", "hGn"), "qmixRSN": ("dlsum", "ermsum", "hGn"),
        "qmixRm": ("dlogit", "erm"), "qmixRmN": ("erm", "hGn"),
        # "qavg": the MEAN of the component quantiles instead of the largest (a softer combination: a video keeps
        # asking when its components are jointly high, not when any one of them is)
        "qavgRS": ("dlsum", "ermsum"), "qavgRm": ("dlogit", "erm"), "qavgRSN": ("dlsum", "ermsum", "hGn"),
        "qavgRmN": ("dlogit", "erm", "hGn"), "qavgR2N": ("dlogit2", "ermsum", "hGn"), "qavgSN": ("dlsum", "hGn"),
        "qavg2N": ("dlogit2", "hGn"), "qavgR2": ("dlogit2", "ermsum")}


def add_quantile_rules(vals):
    """Constant-free combinations (README 17.4): each component value is replaced by its rank among all finite
    validation (video, call) values of that component (its validation quantile, no labels), and the rule is the
    largest quantile: ask while any component (cross-video log-odds change, within-video rank change, video-level
    undecidedness) is still larger than a fraction c of what validation runs show. One threshold, no floor."""
    have = set(next(iter(next(iter(vals.values())).values())).keys())
    for name, comps in QMIX.items():
        if any(c not in have for c in comps):
            continue
        ref = {}
        for c in comps:
            x = np.concatenate([np.asarray(vals["val"][v][c], dtype=np.float64) for v in vals["val"]])
            ref[c] = np.sort(x[np.isfinite(x)])
        avg = name.startswith("qavg")
        for sp in vals:
            for v in vals[sp]:
                n = len(vals[sp][v]["eig"])
                q = np.zeros(n)
                for c in comps:
                    x = np.asarray(vals[sp][v][c], dtype=np.float64)
                    qc = np.where(np.isfinite(x), np.searchsorted(ref[c], x, side="right") / max(1, len(ref[c])), np.inf)
                    q = q + qc / len(comps) if avg else np.maximum(q, qc)
                vals[sp][v][name] = [float(t) for t in q]


RELAX = (("qmixSG", "qmix2G", "qmixS2G", "qmixD2", "qmixS", "dlsum", "dlogit2_n20"), (0.05, 0.10))


def add_relaxed_rules(vals):
    """EcoFrame-style relaxing threshold (README 17.4 item 10): the stop threshold is relaxed with the number of calls
    made, tau_k = tau_1 + (k - 1) * delta in EcoFrame; in the "ask while value >= c" form used here the value of call
    k is multiplied by (1 - delta)^k, so a video that keeps moving a little stops after some calls without a hard
    maximum. delta is a hand-set schedule constant (two values reported); inf values stay inf."""
    for base, delta in ((b, d) for b in RELAX[0] for d in RELAX[1]):
        name = "%s_rt%02d" % (base, int(round(delta * 100)))
        for sp in vals:
            for v in vals[sp]:
                if base not in vals[sp][v]:
                    continue
                vals[sp][v][name] = [float(x) * (1.0 - delta) ** k if np.isfinite(x) else float(x)
                                     for k, x in enumerate(vals[sp][v][base])]


FLOORS = (2, 4, 6)                                     # floor sensitivity (README 17.4 variant (c))
STATE = ("hG", "hT", "hmax", "vsum", "vmean")          # README 17.4 "no floor": state-confidence rules
CHANGE = (("dlogit", "dlogit2", "dlogit3", "dlsum", "dlsum2", "dG", "dG2", "eigsum", "eigtop3", "eigmean", "erm", "ermsum")
          + tuple("%s_%s" % (b, u) for u in ("u50", "u20") for b in ("dlogit", "dlogit2", "dlogit3"))
          + tuple("%s_%s" % (b, u) for u in ("n50", "n20") for b in ("dlogit", "dlogit2", "dlsum")))   # log-odds change rules
RULES = (("eig", "voi", "voi_norm", "stab1", "stab2") + tuple("%s_f%d" % (r, f) for r in ("eig", "voi", "stab1") for f in FLOORS)
         + STATE + CHANGE + tuple(QMIX)
         + tuple("%s_rt%02d" % (b, int(round(d * 100))) for b in RELAX[0] for d in RELAX[1]))
STRAT = ("eig", "voi", "stab1")                        # variant (a): thresholds per prior half
