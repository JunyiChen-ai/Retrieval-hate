# EDCM-RGCL Refinement Report

**Problem:** Meaningful, novel MLLM integration that substantially improves final hateful-video RGCL accuracy and macro-F1.  
**Initial approach:** MLLM counterfactual `V/S/O` coalition judgments control the final-memory listwise gradient.  
**Date:** 2026-07-10  
**Rounds:** 3 / 5  
**Final score:** 9.11 / 10  
**Final verdict:** READY for experiment handoff, not target completion.

## Problem Anchor

The canonical immutable anchor is `refine-logs/edcm/PROBLEM_ANCHOR.md` (SHA-256 `0a964b6732c03d6adb2037b1883140a2df7610fc41e56d82c0cece4d283aebda`). It is copied verbatim into every full round proposal and the final proposal.

The decisive supervision statement is: **video-level binary label is the only gold; no segment-level gold exists or may be assumed.** Fixed frames are whole-video inputs. OCR is deterministic input extraction. Every coalition rank/signature/confidence is a weak/privileged train-only pseudo-signal. Validation/test load none of them.

## Output Files

- Clean final proposal: `refine-logs/edcm/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/edcm/REVIEW_SUMMARY.md`
- Initial proposal: `refine-logs/edcm/round-0-initial-proposal.md`
- Full refinements: `round-1-refinement.md`, `round-2-refinement.md`
- Score history: `refine-logs/edcm/score-history.md`
- Grounding evidence: `refine-logs/edcm/GROUNDING_EVIDENCE.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.0 | 6.5 | 6.0 | 7.0 | 7.0 | 8.0 | 6.0 | 6.93 | REVISE |
| 2 | 9.5 | 8.0 | 8.0 | 8.5 | 7.5 | 9.0 | 7.5 | 8.28 | REVISE |
| 3 | 9.8 | 9.2 | 8.8 | 9.0 | 8.8 | 9.6 | 8.6 | 9.11 | READY |

## Method Evolution Highlights

1. Rejected teacher-selected coalition memory keys because they recreate P3/P11 view weighting and full-query/clean-key mismatch; kept ordinary full-video keys at train and test.
2. Converted a vague “dense signature” claim into an all-OOF teacher-specific gradient gate with standard TV, `R=||Delta g||/||g_uniform||` and equal-step `DeltaD` on reachable errors.
3. Added exactly one teacher-semantic-free strength-matched low-level proxy so MLLM semantics must beat generic availability/content conditioning at equal induced gradient magnitude.
4. Froze a finite pre-MLLM A0 reachability/cost screen, fully fold-local OOF artifacts and a no-segment-gold schema.

## Pushback / Drift Log

| Round | Potential expansion | Decision |
|---:|---|---|
| 0 | selected teacher keys / sufficient-view student branch | rejected: mismatch and module growth |
| 1 | optional absolute verdict diagnostic | removed: not load-bearing |
| 1–3 | routers, learned signature encoder, second teacher, localization, SSR stacking | rejected as drift/bloat |
| 2 | more than one low-level proxy | rejected: one strength-matched proxy is sufficient |

## Remaining Weaknesses

- No A0/A1/A2/A3 result exists yet; the final +3/+3 goal is completely unproven.
- A0 may stop before MLLM if top-64/two-swap video-level reachability is too small.
- A1 may stop if reliable signatures are within-list homogeneous, gradient-inert, or no better directed than Label-only.
- A2 may show the strength-matched proxy explains the gain or that the kNN readout does not improve.

## Raw Reviewer Response Archive

Every review file contains the **full verbatim raw response** in a `<details>` block and records the same reviewer agent ID `/root/edcm_pivot_refine/edcm_reviewer`:

| Round | File | SHA-256 |
|---:|---|---|
| 1 | `refine-logs/edcm/round-1-review.md` | `6f13f9d628efd73fe97aa028c2d8f01d1a3ba78e70a3ed592f6b60d04f6e776e` |
| 2 | `refine-logs/edcm/round-2-review.md` | `a54ceb974d69be38a47eecf69f5fb9a0d4f37df6c9d31aebf9d9e2feb4e66b44` |
| 3 | `refine-logs/edcm/round-3-review.md` | `b80b18f48b1d9b6be0223875c83918b4307f2626d36e7cff93a9203d7f86b114` |

The raw text is not duplicated here so these hashed originals remain the single authoritative transcript.

## Final Proposal Snapshot

- Frozen 7B MLLM compares seven deterministic `V/S/O` coalitions for train videos only.
- Six-dimensional necessity/synergy signature sets a list-normalized NCA measure over ordinary full-video memory keys.
- No new trainable component and no validation/test teacher artifact.
- A0 precedes every MLLM call; A1 requires broad nonzero and better-directed teacher gradients; A2 requires causal kNN repair; A3 alone can prove the global target.

## Next Step

Run `/experiment-plan` on `refine-logs/edcm/FINAL_PROPOSAL.md`, then implement/execute **A0 only**. Do not invoke the MLLM, create teacher caches or unlock A1 until A0's frozen two-dataset video-level reachability/cost screen passes.
