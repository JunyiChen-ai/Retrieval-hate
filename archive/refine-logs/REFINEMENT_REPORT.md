# Refinement Report

**Problem:** MLLM × RGCL substantial final-performance integration.  
**Initial approach:** Gold-signed MLLM-typed hard-pair graph for final kNN memory geometry.  
**Date:** 2026-07-10  
**Rounds:** 4 / 5  
**Final score:** 9.03 / 10  
**Final verdict:** READY for experiment handoff

## Problem Anchor

Only video-level binary labels are gold; no segment-level gold may be assumed. MLLM stance/target/proposition/mechanism outputs are weak train-only pseudo-signals with reliability, missing fallback, remove/shuffle/noise controls and no test-time annotation role. Actual completion still requires ≥+3 accuracy and macro-F1 on MHC-EN/ZH, seeds 0/1/2, against the moving strongest baseline with corrected statistics.

## Output Files

- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Score history: `refine-logs/score-history.md`
- Full initial/revised proposals: `round-0-initial-proposal.md`, `round-1-refinement.md`, `round-2-refinement.md`, `round-3-refinement.md`
- Full verbatim reviewer responses: `round-1-review.md`, `round-2-review.md`, `round-3-review.md`, `round-4-review.md`

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 6 | 8 | 6 | 6 | 6 | 6.6 | REVISE |
| 2 | 8 | 7 | 7 | 9 | 7 | 6 | 6 | 7.4 | REVISE |
| 3 | 10 | 8 | 7 | 9 | 7 | 8 | 7 | 8.1 | REVISE |
| 4 | 10 | 9.5 | 8 | 9.5 | 8.5 | 9.5 | 8 | 9.03 | READY |

## Final Proposal Snapshot

- Frozen Qwen2.5-VL-7B produces repeated, label-blind pair relations on train videos only.
- Agreement-derived reliability routes missing/uncertain records to exact non-MLLM behavior.
- MI+ and/or SC− survive only as one common family set across EN/ZH after strict OOF gates.
- Video labels sign edges; one parameter-free hinge changes the shared final kNN geometry.
- Inference is unchanged; exact remove, label-only, canonical shuffle and noise controls identify causality.

## Method Evolution Highlights

1. Removed cross-seed/validation drift and made every retrieval event directionally and geometrically well-defined.
2. Simplified relation types and confidence handling without adding learned modules.
3. Replaced informal controls with canonical dependence-preserving nulls and dual-metric oracle headroom.

## Pushback / Drift Log

| Round | Potential drift/complexity | Response | Outcome |
|---|---|---|---|
| 1 | Cross-seed union supervision | Activate paired-seed arcs only | Corrected |
| 2 | Validation relation endpoints | Strict train-only OOF universe | Corrected |
| 3 | Possible new architecture to strengthen novelty | Rejected; evidence must carry the narrow claim | Accepted by reviewer |

## Remaining Weaknesses

READY means the method definition is ready, not that the target is achieved. B0/B1 must still prove a common relation family, Wilson precision, conditional information, ≥+0.05 oracle headroom for both metrics, and exact shuffle feasibility. Any failure stops SSR without scale/prompt/architecture rescue.

## Raw Reviewer Responses

The complete verbatim responses, reviewer agent id (`/root/ssr_method_refine/ssr_reviewer`), parsed score and verdict are preserved in the four `round-N-review.md` files listed above; no response was summarized in place of the raw record.

## Next Steps

Proceed to experiment planning/implementation for B0/B1 only. Freeze event/statistic, hashes and folds before inspecting results; launch B2/B3 only after every upstream gate passes.
