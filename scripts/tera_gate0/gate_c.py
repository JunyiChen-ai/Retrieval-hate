#!/usr/bin/env python
"""TERA Gate-0 — Gate-C sampling, tercile weighting and coverage (appendix sec 11).

Sampling and the frozen population weights follow sec 11.3 exactly: terciles are
cut WITHIN the false-negative population, the deficit redistribution is
deterministic, controls are drawn after the FN draw from the same generator and
never enter the FN coverage denominator, and the coverage bootstrap reapplies the
FROZEN weights instead of recomputing them.
"""
from __future__ import annotations

import math

import numpy as np

from .common import BOOTSTRAP_SEED, GATE_C_SEED, TeraHalt

TAXONOMY = ("short_localized", "multi_segment_complementary", "cross_modal",
            "quotation_or_counterstance", "external_knowledge", "global_evidence",
            "annotation_ambiguity_or_noise", "representation_failure_other")
UNION_SET = ("short_localized", "multi_segment_complementary", "cross_modal")
FN_TOTAL = 120
FN_PER_TERCILE = 40
CONTROL_TOTAL = 30
CONTROL_PER_TERCILE = 10


def tercile_assign(scores):
    """Half-open deterministic terciles cut within the given population."""
    scores = np.asarray(scores, dtype=np.float64)
    q33, q67 = np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    out = np.where(scores < q33, 0, np.where(scores < q67, 1, 2))
    return out.astype(np.int64), float(q33), float(q67)


def redistribute(targets, sizes):
    """Deterministic deficit redistribution (appendix sec 11.3)."""
    targets = dict(targets)
    for t in (0, 1, 2):
        if sizes[t] < targets[t]:
            deficit = targets[t] - sizes[t]
            targets[t] = sizes[t]
            for u in (0, 1, 2):
                if u == t:
                    continue
                take = min(deficit, sizes[u] - targets[u])
                if take > 0:
                    targets[u] += take
                    deficit -= take
                if deficit == 0:
                    break
    return targets


def _draw_population(rng, ids, scores, per_tercile, total):
    terc, q33, q67 = tercile_assign(scores)
    by_t = {t: sorted([ids[i] for i in range(len(ids)) if terc[i] == t]) for t in (0, 1, 2)}
    sizes = {t: len(by_t[t]) for t in (0, 1, 2)}
    if len(ids) <= total:
        drawn = {t: list(by_t[t]) for t in (0, 1, 2)}
    else:
        targets = redistribute({t: per_tercile for t in (0, 1, 2)}, sizes)
        drawn = {}
        for t in (0, 1, 2):
            picks = rng.choice(np.array(by_t[t], dtype=object), size=targets[t],
                               replace=False) if targets[t] else []
            drawn[t] = sorted([str(x) for x in picks])
    tercile_of = {}
    for t in (0, 1, 2):
        for vid in by_t[t]:
            tercile_of[vid] = t
    return drawn, sizes, tercile_of, {"q33": q33, "q67": q67}


def select_audit_sample(rows, seed=GATE_C_SEED):
    """rows: [{video_id, score, gold_label, prediction}] from the A0 OOF run."""
    rng = np.random.default_rng(seed)
    fn = sorted([r for r in rows if int(r["gold_label"]) == 1 and int(r["prediction"]) == 0],
                key=lambda r: r["video_id"])
    tp = sorted([r for r in rows if int(r["gold_label"]) == 1 and int(r["prediction"]) == 1],
                key=lambda r: r["video_id"])
    fp = sorted([r for r in rows if int(r["gold_label"]) == 0 and int(r["prediction"]) == 1],
                key=lambda r: r["video_id"])
    if not fn:
        raise TeraHalt("HALT_GATE_C_NO_FN", "no false negatives in the A0 OOF run")

    fn_ids = [r["video_id"] for r in fn]
    fn_scores = [float(r["score"]) for r in fn]
    drawn, sizes, tercile_of, cuts = _draw_population(rng, fn_ids, fn_scores,
                                                      FN_PER_TERCILE, FN_TOTAL)
    audit_fn, weights, sampled = [], {}, {}
    for t in (0, 1, 2):
        sampled[t] = len(drawn[t])
        audit_fn.extend(drawn[t])
    for t in (0, 1, 2):
        for vid in drawn[t]:
            weights[vid] = sizes[t] / float(sampled[t]) if sampled[t] else 0.0

    controls = {}
    for name, pop in (("true_positives", tp), ("false_positives", fp)):
        if not pop:
            controls[name] = []
            continue
        ids = [r["video_id"] for r in pop]
        scores = [float(r["score"]) for r in pop]
        drawn_c, _, _, _ = _draw_population(rng, ids, scores, CONTROL_PER_TERCILE,
                                            CONTROL_TOTAL)
        controls[name] = sorted(sum((drawn_c[t] for t in (0, 1, 2)), []))

    return {
        "audit_fn": audit_fn,
        "weights": weights,
        "tercile_of": {v: tercile_of[v] for v in audit_fn},
        "population_sizes": {str(t): sizes[t] for t in (0, 1, 2)},
        "sampled_sizes": {str(t): sampled[t] for t in (0, 1, 2)},
        "tercile_cuts": cuts,
        "controls": controls,
        "n_fn_population": len(fn),
        "seed": seed,
        "audited_all": len(fn) <= FN_TOTAL,
    }


def mechanisms_of(record):
    """Primary-or-secondary presence rule (prereg sec 4.3)."""
    mech = {record["primary_cause"]}
    mech.update(record.get("secondary_causes", []))
    return mech


def resolve_audit_rows(audit_rows):
    """One row per audited video: the adjudicated row if present, else the first
    row in file order (appendix sec 6.7 resolution; deviation D-3)."""
    by_video = {}
    for row in audit_rows:
        by_video.setdefault(row["video_id"], []).append(row)
    resolved = {vid: ([r for r in rws if r.get("adjudicated")] or rws[:1])[0]
                for vid, rws in by_video.items()}
    return by_video, resolved


def weighted_coverage(audit_fn, mech_by_video, weights, mech_set):
    num = sum(weights[v] for v in audit_fn
              if mech_by_video[v] & set(mech_set))
    den = sum(weights[v] for v in audit_fn)
    return (num / den) if den else 0.0


def unweighted_coverage(audit_fn, mech_by_video, mech_set):
    if not audit_fn:
        return 0.0
    return sum(1 for v in audit_fn if mech_by_video[v] & set(mech_set)) / len(audit_fn)


def coverage_bootstrap(audit_fn, tercile_of, mech_by_video, weights, mech_set,
                       n_resamples=10000, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    by_t = {t: [v for v in audit_fn if tercile_of[v] == t] for t in (0, 1, 2)}
    vals = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        num = den = 0.0
        for t in (0, 1, 2):
            pool = by_t[t]
            if not pool:
                continue
            pick = rng.integers(0, len(pool), size=len(pool))
            for i in pick:
                vid = pool[int(i)]
                den += weights[vid]
                if mech_by_video[vid] & set(mech_set):
                    num += weights[vid]
        vals[b] = (num / den) if den else 0.0
    return {"ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
            "n_resamples": int(n_resamples)}


def cohen_kappa(pairs):
    from sklearn.metrics import cohen_kappa_score
    if not pairs:
        return None, None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    return float(agree), float(cohen_kappa_score(a, b))


def gate_c_decision(coverage_union, ci_lower, coverage_msc, coverage_noise, kappa):
    checks = {
        "union_ge_0.30": coverage_union >= 0.30,
        "union_ci_lower_ge_0.20": ci_lower is not None and ci_lower >= 0.20,
        "msc_ge_0.15": coverage_msc >= 0.15,
        "noise_le_0.25": coverage_noise <= 0.25,
        "kappa_ge_0.60": kappa is not None and kappa >= 0.60,
    }
    return {"checks": checks, "pass": all(checks.values())}


def msc_subset(audit_rows):
    """Frozen msc subset (sec 6.7): EVERY audited video of any category whose
    resolved cause carries multi_segment_complementary as primary or secondary.
    Resolution is adjudicated-else-first, so a double-coded video on which the
    coders agreed (two rows, no adjudication row) is included (deviation D-3)."""
    _, resolved = resolve_audit_rows(audit_rows)
    return sorted(vid for vid, rec in resolved.items()
                  if "multi_segment_complementary" in mechanisms_of(rec))


def rescue_metrics(msc_ids, labels, pred_b0, pred_b2):
    """Rescue rate and the count-based FP side condition (appendix sec 6.7)."""
    pos = [v for v in msc_ids if int(labels[v]) == 1]
    neg = [v for v in msc_ids if int(labels[v]) == 0]
    den = [v for v in pos if int(pred_b0[v]) == 0]
    num = [v for v in den if int(pred_b2[v]) == 1]
    fp_b0 = sum(1 for v in neg if int(pred_b0[v]) == 1)
    fp_b2 = sum(1 for v in neg if int(pred_b2[v]) == 1)
    if den:
        rate = len(num) / len(den)
        rescue_ok = rate >= 0.20
        rescue_state = "evaluated"
    else:
        rate = None
        rescue_ok = False                 # a positive requirement; vacuous => NOT satisfied
        rescue_state = "not_evaluable"
    if neg:
        fp_ok = fp_b2 <= fp_b0 + max(1, math.ceil(0.10 * fp_b0))
        fp_state = "evaluated"
    else:
        fp_ok = True                      # do-no-harm guard; vacuous => satisfied
        fp_state = "not_evaluable"
    return {"rescue_numerator": len(num), "rescue_denominator": len(den),
            "rescue_rate": rate, "rescue_state": rescue_state, "rescue_pass": rescue_ok,
            "fp_b0": fp_b0, "fp_b2": fp_b2, "fp_state": fp_state, "fp_pass": fp_ok}
