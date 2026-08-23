"""Registered metric surface (appendix section 12).

Only the quantities the appendix names are implemented here, with the exact
semantics it fixes: sklearn binary macro-F1 with labels [0,1] and
``zero_division=0``, score-based AUROC that is null when a resample lacks a
class, log-loss clipped to [1e-7, 1-1e-7], and the AUPAC trapezoid rule with
equal-cost collapse.  No real data is read by this module.
"""
from __future__ import annotations

import numpy as np

from .protocol import macro_f1_binary

LOGLOSS_CLIP = 1e-7


def macro_f1(y, pred) -> float:
    """sklearn semantics with fixed labels [0,1]; stays defined on a one-class resample."""
    return macro_f1_binary(y, pred)


def auroc(y, score):
    """Score-based AUROC; ``None`` when the sample contains a single class."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def log_loss(y, prob) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), LOGLOSS_CLIP, 1.0 - LOGLOSS_CLIP)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def aupac(cost_fractions, f1_values) -> float:
    """Trapezoid integral of macro-F1 over realized mean incremental-cost fraction.

    Points are sorted by cost; equal-cost points collapse to the lower F1; the
    integral is divided by the observed 0--1 cost span.
    """
    c = np.asarray(cost_fractions, dtype=float)
    f = np.asarray(f1_values, dtype=float)
    if c.shape != f.shape or c.size == 0:
        raise RuntimeError("HALT_AUPAC_SHAPE")
    order = np.lexsort((f, c))
    c, f = c[order], f[order]
    keep_c, keep_f = [], []
    for x, v in zip(c.tolist(), f.tolist()):
        if keep_c and x == keep_c[-1]:
            keep_f[-1] = min(keep_f[-1], v)  # equal cost keeps the lower F1
            continue
        keep_c.append(x)
        keep_f.append(v)
    if len(keep_c) < 2:
        raise RuntimeError("HALT_AUPAC_SPAN")
    span = keep_c[-1] - keep_c[0]
    if span <= 0:
        raise RuntimeError("HALT_AUPAC_SPAN")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(np.asarray(keep_f), np.asarray(keep_c)) / span)


def complete_run_metric(fold_rows, threshold_by_arm, arms):
    """Step 1--2 of the primary point estimate: concatenate five outer folds, then score.

    ``fold_rows`` is an iterable of per-fold dicts with ``y``, ``scores`` and
    ``costs``.  Folds are concatenated inside one (split_seed, refit_seed) run;
    fold metrics are never averaged.
    """
    rows = list(fold_rows)
    if len(rows) != 5:
        raise RuntimeError("HALT_COMPLETE_RUN_FOLD_COUNT")
    y = np.concatenate([np.asarray(r["y"]) for r in rows])
    out = {}
    for arm in arms:
        s = np.concatenate([np.asarray(r["scores"][arm], float) for r in rows])
        c = np.concatenate([np.asarray(r["costs"][arm], float) for r in rows])
        t = threshold_by_arm[arm]
        out[arm] = {"f1": macro_f1(y, s >= t), "cost": float(np.mean(c))}
    return out


def mean_of_complete_runs(values) -> float:
    """Step 3: arithmetic mean of the nine complete-run values."""
    a = np.asarray(values, dtype=float)
    if a.shape != (9,):
        raise RuntimeError("HALT_EXPECTED_NINE_RUNS")
    return float(a.mean())


def upper_envelope(points):
    """Empirical upper envelope: best F1 among all points with no greater cost."""
    ordered = sorted(((float(c), float(f), name) for name, f, c in points))
    best = float("-inf")
    out = []
    for c, f, name in ordered:
        if f > best:
            best = f
        out.append({"cost": c, "arm_id": name, "envelope_f1": best})
    return out


def envelope_at(points, cost):
    """Envelope value at a candidate cost; no interpolation is used in a gate."""
    legal = [(f, c, n) for n, f, c in ((n, float(f), float(c)) for n, f, c in points)
             if c <= float(cost)]
    if not legal:
        raise RuntimeError("HALT_NO_ADMISSIBLE_BASELINE")
    return max(legal, key=lambda x: (x[0], -x[1], x[2]))
