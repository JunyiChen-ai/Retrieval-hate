# Review Summary

**Problem:** Meaningful, novel train-only MLLM integration that causes substantial final full-video kNN improvement under video-label-only supervision.
**Initial approach:** CTE-RGCL whole-modality weak relations supervising a full-bank retrieval tangent.
**Date:** 2026-07-10
**Rounds:** 3 / 5
**Final score:** 9.20 / 10
**Final verdict:** READY
**Continuous reviewer agent:** `/root/cte_method_refine/cte_reviewer`

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Round-by-Round Resolution Log

| Round | Main reviewer concern | What changed | Solved? | Remaining risk |
|---:|---|---|---|---|
| 1 | Withholding/prototype mismatch; unidentified gold orientation; stale-bank and control ambiguity | Recast as withholding-informed weak relation; joint support; separate class/two-radius transfer; exact epoch bank, A0/A1 and controls | Partial | Tangent identity could change after transfer validation |
| 2 | Dynamic medoid/radius reselection; small protocol ambiguities | Froze modality anchor IDs and radii before calls; direction-drift STOP; exact probe/ESS/noise/gradient/statistics rules | Yes | Empirical gates may fail, especially reverse-cell support |
| 3 | Final handoff wording only | Explicit modality anchors, reliability-weighted cells, shared video-ID bootstrap, microbenchmark handoff | Yes | No specification blocker |

## Overall Evolution

- Teacher withholding is no longer claimed to be identical to the student tangent; transfer is a class-conditional empirical hypothesis.
- Zero/blank/black-frame neutralization is prohibited; only typed teacher withholding and train-supported local prototype paths remain.
- One shared query/key encoder and an exact epoch-refreshed full-video bank directly connect the loss to final kNN.
- A0 is explicitly an empirical cost screen, never a theoretical upper bound; A1 is capped at 128 train videos per dataset.
- REMOVE, assignment-free multiview, label-only, heuristic, random, whole-record SHUFFLE and NOISE isolate MLLM-specific assignment value.

## Final Status

- **Anchor:** preserved.
- **Focus:** tight; one parameter-free loss, zero new trainable components.
- **Modernity:** appropriately frontier-aware privileged ordinal weak supervision.
- **Supervision:** video-label-only; no segment/timestamp/span/localization gold or pseudo-endpoint.
- **Remaining weakness:** READY is specification readiness only. The route can still fast-fail at A0/A1/A2, and no final performance gain has yet been shown.
