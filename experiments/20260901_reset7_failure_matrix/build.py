#!/usr/bin/env python
"""Build RESET7's cross-candidate matrix from existing formal artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/20260901_reset7_cross_candidate_failure_matrix/main/matrix.json"
SOTA = {
    "hatemm": {"pooled_ap": .5938315566, "pooled_roc": .8161837922,
               "within_roc": .6315317180},
    "hateclipseg": {"pooled_ap": .6193710950, "pooled_roc": .6050224699,
                    "within_roc": .5619078936},
}
GOAL_GAPS = {
    "hatemm": {"pooled_ap": .100831, "pooled_roc": .077925,
               "within_roc": .003076},
    "hateclipseg": {"pooled_ap": .066350, "pooled_roc": .060950,
                    "within_roc": .038207},
}


def delta(a, b):
    return {m: float(a[m] - b[m]) for m in SOTA["hatemm"]}


def main():
    specs = [
        {
            "name": "lexically_anchored_dcc",
            "path": "runs/20260901_lexically_anchored_dcc/formal_seed234/summary.json",
            "core": "aligned", "control": "shifted", "anchor": "anchor",
            "signal": "train/inference-available lexical anchors define cross-video region memory",
            "observed_headroom": "MultiHateLoc test errors showed localization/fusion headroom; aligned lexical timing was the proposed correction observation",
            "focused_test_evidence": "HMM positive-occupancy quartiles all lost within ROC and score variation contracted; HCS quartile directions were mixed",
        },
        {
            "name": "policy_cluster_transport",
            "path": "runs/20260901_policy_cluster_transport/formal_seed234/summary.json",
            "core": "policy", "control": "binary", "anchor": "anchor",
            "signal": "train/inference-available policy-state clusters constrain temporal transport",
            "observed_headroom": "semantic policy states were computable, but no pre-admission dual-corpus evidence showed they corrected the measured metric gaps",
            "focused_test_evidence": "HMM policy-minus-binary within was negative in every occupancy group; HCS improved only the lowest group and lost 0.070092 in the highest; harmful mass was 0.8741",
        },
        {
            "name": "active_speaker_bound_utterance_mil",
            "path": "runs/20260901_active_speaker_bound_utterance_mil/formal_seed234/summary.json",
            "core": "core", "control": "permuted", "anchor": "anchor",
            "signal": "frozen TalkNet active-speaker identity and face embedding are available at train/inference",
            "observed_headroom": "speaker/source ambiguity was semantically plausible, but eligible multi-face seconds cover only 2.33% HMM and 4.78% HCS test seconds",
            "focused_test_evidence": "eligible-video core-minus-permuted within ROC is -0.001277 HMM and +0.000249 HCS, with improvement counts no larger than worsening counts",
        },
    ]
    rows = []
    for spec in specs:
        summary = json.loads((ROOT / spec["path"]).read_text())
        row = {k: v for k, v in spec.items() if k not in ("core", "control", "anchor")}
        row["arms"] = {"core": spec["core"], "matched_control": spec["control"],
                       "anchor": spec["anchor"]}
        row["corpora"] = {}
        for corpus in ("hatemm", "hateclipseg"):
            d = summary["corpora"][corpus]
            core = d["arms"][spec["core"]]
            control = d["arms"][spec["control"]]
            anchor = d["arms"][spec["anchor"]]
            row["corpora"][corpus] = {
                "core_test": {m: float(core[m]) for m in SOTA[corpus]},
                "matched_control_test": {m: float(control[m]) for m in SOTA[corpus]},
                "core_minus_matched_control": delta(core, control),
                "core_minus_anchor": delta(core, anchor),
                "remaining_to_fixed_sota": {
                    m: float(SOTA[corpus][m] - core[m]) for m in SOTA[corpus]
                },
                "all_sota": False,
            }
        rows.append(row)
    payload = {
        "date": "2026-09-01",
        "purpose": "RESET7 cross-candidate failure matrix",
        "fixed_sota": SOTA,
        "pre_candidate_goal_gap_budget": GOAL_GAPS,
        "goal_gap_source": "research-wiki/RESET6_GOAL_GAP_AUDIT.md",
        "candidates": rows,
        "common_failure_group": (
            "All three retained the POWA raw scorer as the main decision path and added "
            "a semantic auxiliary constraint/adapter. Correct-vs-matched-control evidence "
            "did not provide same-direction, load-bearing gains on all three metrics in both corpora."
        ),
        "admission_consequence": (
            "Do not reopen these families or default to POWA-plus-auxiliary. A later candidate "
            "must cite an observed dual-corpus correction signal and a numeric gain budget "
            "commensurate with the pooled and within gaps before novelty review."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
