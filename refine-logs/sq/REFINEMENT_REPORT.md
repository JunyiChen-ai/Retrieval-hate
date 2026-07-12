# SQ-RGCL Refinement Report

**Date:** 2026-07-11  
**Canonical reviewer:** `/root/sq_reviewer_replacement`  
**Reviewer continuity:** Rounds 1–4 used the same replacement reviewer. The initially spawned `/root/sq_method_refine/sq_reviewer` interrupted without output and was discarded.  
**Rounds:** 4 / 5  
**Score path:** `6.88 REVISE → 7.90 REVISE → 8.46 REVISE → 9.12 READY`  
**Experiment activity:** no code change, no SLURM job, no new teacher call.

## Outputs

- Anchor: `refine-logs/sq/PROBLEM_ANCHOR.md`
- Grounding: `refine-logs/sq/GROUNDING_EVIDENCE.md`
- Clean final proposal: `refine-logs/sq/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/sq/REVIEW_SUMMARY.md`
- Raw reviews: `round-1-review.md` through `round-4-review.md`
- Full anchored revisions: `round-1-refinement.md` through `round-3-refinement.md`
- Scores/state: `score-history.md`, `REFINE_STATE.json`

## Method Snapshot

- Six-way whole-video presentation posterior only; stance/harm/evidence/label-related archive fields are forbidden nuisance inputs.
- Same-label/cross-presentation full-bank positives are ranked above different-label/same-presentation memories only when those negatives harm the current exact top-20 vote.
- Shared encoder and bank co-move; the final embedding is read by unchanged ordinary kNN. No teacher artifact at validation/test.
- P0 and learned strict-OOF SQ-0 run before any new call. SQ-0 must pass +.05 accuracy/+.05 macro-F1 on both datasets.
- Actual MLLM SQ-1 is representative, graph-closed, power-valid, common-edge, class-specific, and fail-closed under the call cap.

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.5 | 6.4 | 5.8 | 7.3 | 7.2 | 6.1 | 5.6 | 6.88 | REVISE |
| 2 | 9.8 | 7.8 | 7.1 | 8.2 | 7.4 | 8.0 | 6.7 | 7.90 | REVISE |
| 3 | 9.8 | 8.6 | 8.0 | 8.3 | 8.1 | 8.2 | 7.5 | 8.46 | REVISE |
| 4 | 10.0 | 9.3 | 8.7 | 9.0 | 8.9 | 9.2 | 8.3 | 9.12 | READY |

## Pushback / Drift Log

- No reviewer recommendation required drift.
- `q→y<=0.70` was removed as a hard gate because presentation imbalance can legitimately correlate with labels; it remains a contamination diagnostic.
- Rank>20 exposure was deleted because the evaluator assigns zero vote weight there. Full-bank positives preserve geometry reach beyond old fixed universes.
- CTE C0 STOP is explicitly retained as a numerical/cost-policy outcome, not a performance upper bound.

## Remaining Weaknesses

- Bibliographic novelty is deliberately narrow and requires a final independent novelty check before submission.
- Existing archive summary provenance may fail, which must stop zero-new-call SQ-0 rather than invite repair.
- The 128-video graph closure may be underpowered/too large and therefore stop SQ-1 before calls.
- No result yet supports any accuracy or macro-F1 claim.

## Raw Reviewer Responses

Full verbatim responses, including every score and blocker, are preserved in `round-1-review.md`, `round-2-review.md`, `round-3-review.md`, and `round-4-review.md`.

## Next Step

Create an independent experiment plan and implementation audit for P0/SQ-0 only. Freeze the power formula, finite-population correction, bootstrap count, evaluator rank/cosine/tie parity, audit examples, archive provenance check, and microbenchmark. **Do not make a new teacher call before SQ-0 GO.**
