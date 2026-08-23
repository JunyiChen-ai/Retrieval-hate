#!/usr/bin/env python
"""TERA Gate-0 — O1 and O2 deterministic oracle rules (appendix sec 5).

Both oracles reuse the A1 fold-trained linear head as the fixed fold-trained
segment scorer and pool its SEGMENT LOGITS.  Every row they touch is marked
`oracle_or_eval_only: true`; O2 additionally carries `label_leaking: true`.
Neither may select a deployable arm.
"""
from __future__ import annotations

import math

from .common import K_WINDOWS, TeraHalt


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def o1_video_logit(seg_logits, spans, duration, k=K_WINDOWS):
    """Gold-span pooling with the registered A1 mean-pooling fallback (sec 5.1).

    The routine may inspect span PRESENCE but never branches on the video label.
    """
    windows = []
    if duration is not None and duration > 0:
        for idx in range(k):
            lo = idx * duration / k
            hi = (idx + 1) * duration / k
            for a, b in spans:
                if min(hi, b) - max(lo, a) > 0.0:
                    windows.append(idx)
                    break
    if not windows:
        sel, fallback = list(range(k)), True
    else:
        sel, fallback = windows, False
    logit = sum(seg_logits[i] for i in sel) / float(len(sel))
    return logit, sel, fallback


def o2_video_logit(seg_logits, y_true, k=K_WINDOWS):
    """True-label-aware best candidate subset; the optimum is a singleton (sec 5.2)."""
    if int(y_true) == 1:
        k_star = max(range(k), key=lambda i: (seg_logits[i], -i))
    else:
        k_star = min(range(k), key=lambda i: (seg_logits[i], i))
    return seg_logits[k_star], [k_star]


def assert_oracle_ordering(records):
    """Fixture F3 / run-time guarantee: O2 >= O1 for y=1 and <= for y=0."""
    for rec in records:
        if int(rec["gold_label"]) == 1 and rec["o2_logit"] < rec["o1_logit"] - 1e-9:
            raise TeraHalt("HALT_ORACLE_ORDERING", "%s: O2 < O1 with y=1" % rec["video_id"])
        if int(rec["gold_label"]) == 0 and rec["o2_logit"] > rec["o1_logit"] + 1e-9:
            raise TeraHalt("HALT_ORACLE_ORDERING", "%s: O2 > O1 with y=0" % rec["video_id"])
    return True
