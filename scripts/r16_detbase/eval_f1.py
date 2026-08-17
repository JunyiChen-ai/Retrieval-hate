#!/usr/bin/env python
"""R16-DETBASE: the HateClipSeg paper's proposal-level F1@tIoU for the offensive class.

Paper (2508.01712, sec 4.2 "Evaluation Metrics"): "For temporal video localization, we use
temporal Intersection over Union (tIoU) between predicted and ground truth segments,
reporting performance at tIoU thresholds of 0.3, 0.5, and 0.7.  Since only offensive
segments are predicted, macro metrics are not applicable; we report Accuracy, Precision,
Recall, and F1 for the offensive class."  Their Table 4 satisfies
F1 = 2PR/(P+R) exactly (e.g. V @ 0.5: 2*0.7514*0.4052/(0.7514+0.4052) = 0.5265), so
P = matched predictions / all predictions and R = matched GT / all GT, one-to-one matching.

This module reuses the project's own matcher semantics (`scripts/r14_loc/recon_decode.py`,
`match_f1`) so the detector number and the per-window score-curve number are computed by the
same rule.  The one difference, declared in the freeze: predictions are matched in *score*
order (highest first), because a detector emits scored proposals while the score-curve
decoder emits unscored intervals.  With unscored intervals the two are identical.
"""
from __future__ import annotations

import numpy as np


def iou_1d(p, q):
    inter = max(0.0, min(p[1], q[1]) - max(p[0], q[0]))
    uni = max(p[1], q[1]) - min(p[0], q[0])
    return inter / uni if uni > 0 else 0.0


def match_prf(preds, golds, tiou):
    """preds: {vid: [(s, e, score)]}, golds: {vid: [(s, e)]}.  Greedy, score-ordered."""
    tp = npd = ng = 0
    for v in golds:
        P = sorted(preds.get(v, []), key=lambda x: -x[2])
        G = golds[v]
        npd += len(P)
        ng += len(G)
        used = set()
        for p in P:
            best, bi = -1.0, -1
            for i, q in enumerate(G):
                if i in used:
                    continue
                o = iou_1d(p, q)
                if o > best:
                    best, bi = o, i
            if best >= tiou:
                tp += 1
                used.add(bi)
    P_ = tp / max(npd, 1)
    R_ = tp / max(ng, 1)
    F_ = 2 * P_ * R_ / max(P_ + R_, 1e-12)
    return dict(P=100 * P_, R=100 * R_, F1=100 * F_, tp=tp, n_pred=npd, n_gold=ng)


def prf_at(preds, golds, thr, tious=(0.3, 0.5, 0.7)):
    cut = {v: [p for p in ps if p[2] >= thr] for v, ps in preds.items()}
    return {t: match_prf(cut, golds, t) for t in tious}


def sweep_threshold(preds, golds, grid, tiou=0.5):
    """Return (best_thr, best_F1, table)."""
    tab = []
    for thr in grid:
        m = match_prf({v: [p for p in ps if p[2] >= thr] for v, ps in preds.items()},
                      golds, tiou)
        tab.append((float(thr), m["F1"], m["P"], m["R"]))
    best = max(tab, key=lambda r: r[1])
    return best[0], best[1], tab


def average_precision(preds, golds, tiou):
    """Standard TAL AP for a single class (used only as a threshold-free descriptive
    cross-check; the frozen endpoint is F1@tIoU)."""
    flat = []
    for v, ps in preds.items():
        for s, e, sc in ps:
            flat.append((sc, v, s, e))
    flat.sort(key=lambda x: -x[0])
    ng = sum(len(g) for g in golds.values())
    if ng == 0 or not flat:
        return 0.0
    used = {v: set() for v in golds}
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for i, (sc, v, s, e) in enumerate(flat):
        G = golds.get(v, [])
        best, bi = -1.0, -1
        for j, q in enumerate(G):
            if j in used[v]:
                continue
            o = iou_1d((s, e), q)
            if o > best:
                best, bi = o, j
        if best >= tiou:
            tp[i] = 1
            used[v].add(bi)
        else:
            fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / ng
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    # interpolated AP (all-point)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(100 * np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
