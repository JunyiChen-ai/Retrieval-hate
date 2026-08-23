#!/usr/bin/env python
"""TERA Gate-0 — temporal evaluation (prereg sec 5.2 item 5, sec 8.2; appendix sec 8).

Second `t` is positive iff `t + 0.5` lies in a gold span and is scored by window
`w(t) = min(29, floor((t+0.5)*30/D))` -- the repository's existing midpoint rule
(`scripts/analysis/eval_localization_ours.py:184-186`).
"""
from __future__ import annotations

import numpy as np

from .common import K_WINDOWS, sha256_obj


def second_rows(duration, spans, k=K_WINDOWS):
    """Return (window_index[], label[]) for every evaluated second of one video."""
    if duration is None or duration <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n = int(np.floor(float(duration)))
    win, lab = [], []
    for t in range(n):
        mid = t + 0.5
        win.append(min(k - 1, int(mid * k / float(duration))))
        lab.append(1 if any(s <= mid < e for s, e in spans) else 0)
    return np.asarray(win, dtype=np.int64), np.asarray(lab, dtype=np.int64)


def window_overlaps_span(k, duration, spans, n_windows=K_WINDOWS):
    """Strict positive-duration overlap of window k with any gold span."""
    lo = k * duration / n_windows
    hi = (k + 1) * duration / n_windows
    for a, b in spans:
        if min(hi, b) - max(lo, a) > 0.0:
            return True
    return False


def gold_overlap_windows(duration, spans, n_windows=K_WINDOWS):
    if duration is None or duration <= 0:
        return []
    return [k for k in range(n_windows) if window_overlaps_span(k, duration, spans, n_windows)]


def eligible_videos(ids, label_of, durations, spans):
    """Frozen once, shared across arms (appendix sec 8.1)."""
    out = []
    for vid in sorted(ids):
        if int(label_of(vid)) != 1:
            continue
        sp = spans.get(vid, [])
        if not sp:
            continue
        dur = durations.get(vid)
        if dur is None or dur <= 0:
            continue
        _, lab = second_rows(dur, sp)
        if lab.size == 0 or lab.min() == lab.max():
            continue
        out.append(vid)
    return out


def _auroc(labels, scores):
    """Rank-based AUROC with average ranks for ties."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def within_video_auroc(vid, seg_scores, durations, spans):
    win, lab = second_rows(durations[vid], spans.get(vid, []))
    if win.size == 0:
        return None
    scores = np.asarray(seg_scores, dtype=np.float64)[win]
    return _auroc(lab, scores)


def temporal_metrics(arm, eligible, seg_scores_by_video, durations, spans,
                     n_windows=K_WINDOWS):
    """Primary temporal metric plus the registered secondary/diagnostic ones."""
    per_video = {}
    pooled_scores, pooled_labels = [], []
    recall = {1: 0, 2: 0, 4: 0}
    separations, stdevs = [], []
    for vid in eligible:
        seg = np.asarray(seg_scores_by_video[vid], dtype=np.float64)
        auc = within_video_auroc(vid, seg, durations, spans)
        if auc is not None:
            per_video[vid] = auc
        win, lab = second_rows(durations[vid], spans.get(vid, []))
        pooled_scores.append(seg[win])
        pooled_labels.append(lab)
        gold_windows = set(gold_overlap_windows(durations[vid], spans.get(vid, []),
                                                n_windows))
        order = sorted(range(n_windows), key=lambda k: (-seg[k], k))
        for n in (1, 2, 4):
            if gold_windows & set(order[:n]):
                recall[n] += 1
        hit = np.array([k in gold_windows for k in range(n_windows)])
        if hit.any() and (~hit).any():
            separations.append(float(seg[hit].mean() - seg[~hit].mean()))
        stdevs.append(float(seg.std(ddof=0)))

    values = np.array(list(per_video.values()), dtype=np.float64)
    pooled_auroc = pooled_ap = None
    if pooled_scores:
        ps = np.concatenate(pooled_scores)
        pl = np.concatenate(pooled_labels)
        pooled_auroc = _auroc(pl, ps)
        if 0 < pl.sum() < pl.size:
            from sklearn.metrics import average_precision_score
            pooled_ap = float(average_precision_score(pl, ps))
    n_elig = max(1, len(eligible))
    return {
        "arm": arm,
        "n_eligible": len(eligible),
        "mean_within_video_auroc": float(values.mean()) if values.size else None,
        "n_scored": int(values.size),
        "per_video_auroc": {k: float(v) for k, v in sorted(per_video.items())},
        "gold_span_recall": {str(n): recall[n] / n_elig for n in (1, 2, 4)},
        "selected_vs_unselected_separation": (float(np.mean(separations))
                                              if separations else None),
        "mean_within_video_score_std": float(np.mean(stdevs)) if stdevs else None,
        "pooled_second_level_auroc": pooled_auroc,
        "pooled_second_level_ap": pooled_ap,
    }


def eligible_manifest(eligible):
    return {"n": len(eligible), "video_ids": list(eligible),
            "sha256": sha256_obj(sorted(eligible))}
