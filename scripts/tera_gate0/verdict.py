#!/usr/bin/env python
"""TERA Gate-0 — frozen decision rules (prereg sec 4.3, 5.2, 6, 9).

No threshold here may be relaxed, substituted, or promoted on a near miss.
"""
from __future__ import annotations

NOT_EVALUATED = "not_evaluated"


def gate_a_decision(macro, delta_ci, temporal_mean, confirmation):
    """prereg sec 5.2.  `macro` carries A0/A1/D/O1/O2 macro-F1 on outer OOF."""
    base = max(macro["A0"], macro["A1"])
    d_delta = macro["D"] - base
    checks = {
        "1_o1_minus_base_ge_0.050": (macro["O1"] - base) >= 0.050,
        "2_o2_minus_base_ge_0.050": (macro["O2"] - base) >= 0.050,
        "3_d_minus_base_ge_0.020": d_delta >= 0.020,
        "4_paired_ci_excludes_zero": bool(delta_ci.get("excludes_zero", False)),
        "5_temporal_auroc": (temporal_mean is not None and temporal_mean >= 0.60
                             and temporal_mean >= 0.53),
        "6_confirmations_positive": confirmation.get("all_positive", NOT_EVALUATED),
    }
    values = {
        "base_max_a0_a1": base,
        "o1_delta": macro["O1"] - base,
        "o2_delta": macro["O2"] - base,
        "d_delta": d_delta,
        "d_delta_ci": delta_ci,
        "temporal_mean_within_video_auroc": temporal_mean,
        "temporal_thresholds": {"absolute": 0.60, "vs_a0_broadcast": 0.53},
        "confirmation": confirmation,
    }
    oracles_pass = checks["1_o1_minus_base_ge_0.050"] and checks["2_o2_minus_base_ge_0.050"]
    all_pass = all(v is True for v in checks.values())
    if not oracles_pass:
        verdict = "NO-GO-A-NO-HEADROOM"
    elif not all_pass:
        verdict = "NO-GO-A-SELECTOR"
    else:
        verdict = "GATE-A-PASS"
    return {"checks": checks, "values": values, "pass": all_pass, "verdict": verdict}


def gate_b_decision(macro, delta_ci, rescue, confirmation):
    """prereg sec 6."""
    base = max(macro["B0"], macro["B1"], macro["B3"])
    b2_delta = macro["B2"] - base
    checks = {
        "1_b2_minus_base_ge_0.020": b2_delta >= 0.020,
        "2_paired_ci_excludes_zero": bool(delta_ci.get("excludes_zero", False)),
        "3_b2_minus_b4_ge_0.015": (macro["B2"] - macro["B4"]) >= 0.015,
        "4_b2_minus_b5_ge_0.015": (macro["B2"] - macro["B5"]) >= 0.015,
        "5_msc_rescue": (rescue.get("rescue_pass") and rescue.get("fp_pass"))
                        if rescue.get("state") != NOT_EVALUATED else NOT_EVALUATED,
        "6_confirmations_positive": confirmation.get("all_positive", NOT_EVALUATED),
    }
    values = {
        "base_max_b0_b1_b3": base,
        "b2_delta": b2_delta,
        "b2_delta_ci": delta_ci,
        "b2_minus_b4": macro["B2"] - macro["B4"],
        "b2_minus_b5": macro["B2"] - macro["B5"],
        "rescue": rescue,
        "confirmation": confirmation,
    }
    all_pass = all(v is True for v in checks.values())
    return {"checks": checks, "values": values, "pass": all_pass,
            "verdict": "GATE-B-PASS" if all_pass else "NO-GO-B"}


def overall_verdict(gate_c, gate_a, gate_b, forced_stage_b=False):
    """prereg sec 9 stopping rule.

    `forced_stage_b` is the synthetic-fixture path only: Gate-B never runs after a
    failed Gate-A in a real run, so when a fixture forces it the reported verdict
    is the Gate-B outcome (the Gate-A verdict is still recorded in verdict.json
    under `gate_a`).
    """
    if gate_c is not None and gate_c.get("pass") is False:
        return "NO-GO-C"
    if gate_a is None:
        return "INCOMPLETE-A-NOT-RUN"
    if not gate_a["pass"]:
        if forced_stage_b and gate_b is not None and not gate_b["pass"]:
            return gate_b["verdict"]
        return gate_a["verdict"]
    if gate_b is None:
        return "INCOMPLETE-B-NOT-RUN"
    if not gate_b["pass"]:
        return "NO-GO-B"
    if gate_c is None:
        return "INCOMPLETE-C-NOT-RUN"
    return "GO-TERA"
