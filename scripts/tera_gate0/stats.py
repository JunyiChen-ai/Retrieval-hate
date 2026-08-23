#!/usr/bin/env python
"""TERA Gate-0 — paired video bootstrap and interval helpers (appendix sec 7.7, 8.2).

10,000 paired resamples of videos, seed 20260809, stratified by video label; the
index matrix is drawn once over the SORTED outer-OOF video-id list and reused
identically by every arm and every paired delta -- that is what makes the deltas
paired.  Macro-F1 is recomputed inside every resample at the frozen threshold.
"""
from __future__ import annotations

import math

import numpy as np

from .common import BOOTSTRAP_SEED, macro_f1_counts


def stratified_indices(labels, n_resamples, seed=BOOTSTRAP_SEED):
    """int32[n_resamples, N]; within each label stratum draw n_label with replacement."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == c) for c in (0, 1)]
    out = np.empty((n_resamples, labels.size), dtype=np.int32)
    for b in range(n_resamples):
        parts = []
        for pos in strata:
            if pos.size:
                parts.append(rng.choice(pos, size=pos.size, replace=True))
        out[b] = np.concatenate(parts).astype(np.int32)
    return out


def macro_f1_bootstrap(y, pred, idx):
    """Macro-F1 of one arm inside each resample."""
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    yy = y[idx]
    pp_arr = pred[idx]
    n_pos = (yy == 1).sum(axis=1).astype(np.float64)
    n_neg = (yy == 0).sum(axis=1).astype(np.float64)
    tp = ((yy == 1) & (pp_arr == 1)).sum(axis=1).astype(np.float64)
    pp = (pp_arr == 1).sum(axis=1).astype(np.float64)
    return macro_f1_counts(tp, pp, n_pos, n_neg)


def paired_delta_ci(y, pred_a, pred_b, idx, alpha=0.05):
    """Percentile CI of macro_f1(A) - macro_f1(B) over the shared index matrix."""
    da = macro_f1_bootstrap(y, pred_a, idx) - macro_f1_bootstrap(y, pred_b, idx)
    lo = float(np.percentile(da, 100 * alpha / 2.0))
    hi = float(np.percentile(da, 100 * (1 - alpha / 2.0)))
    return {"ci_lower": lo, "ci_upper": hi, "excludes_zero": bool(lo > 0 or hi < 0),
            "n_resamples": int(idx.shape[0])}


def mean_ci(values, idx, alpha=0.05):
    """Percentile CI of a mean over a bootstrap index matrix (temporal metric)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"ci_lower": None, "ci_upper": None, "n_resamples": int(idx.shape[0])}
    means = values[idx].mean(axis=1)
    return {"ci_lower": float(np.percentile(means, 100 * alpha / 2.0)),
            "ci_upper": float(np.percentile(means, 100 * (1 - alpha / 2.0))),
            "n_resamples": int(idx.shape[0])}


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return {"lower": None, "upper": None}
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return {"lower": max(0.0, centre - half), "upper": min(1.0, centre + half)}
