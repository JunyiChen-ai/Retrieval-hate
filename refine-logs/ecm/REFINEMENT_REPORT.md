# ECM-RGCL Refinement Report

**Problem:** meaningful/novel train-only MLLM integration with final two-dataset, three-seed ordinary-kNN `+0.030/+0.030`.  
**Initial approach:** all-train strict-OOF MLLM residual-mode posterior controlling projected/minimax final-embedding optimization.  
**Date:** 2026-07-11  
**Rounds:** 1 / 5  
**Final score:** 4.98 / 10  
**Final verdict:** **RETHINK — ABANDONED**

## Outputs

- Canonical terminal disposition: `refine-logs/ecm/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/ecm/REVIEW_SUMMARY.md`
- Complete raw reviewer response: `refine-logs/ecm/round-1-review.md`
- Initial proposal: `refine-logs/ecm/round-0-initial-proposal.md`
- Non-canonical future-distinct sketch: `refine-logs/ecm/round-1-refinement.md`
- Grounding/prior-art: `refine-logs/ecm/GROUNDING_EVIDENCE.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 6.5 | 4.5 | 4.0 | 6.0 | 5.0 | 5.5 | 4.0 | 4.98 | RETHINK |

## Method Evolution and Decision

1. The initial proposal correctly enforced all-train strict OOF, no direct gold/error fields, train-only weak modes, test-clean inference and no segment gold.
2. The reviewer algebraically showed that the central QP is still dynamic sample reweighting plus generic gradient surgery and does not constrain the actual AdamW step.
3. A full-bank proximal-target response was drafted, but the research decision correctly treats it as a different future hypothesis rather than score-chasing under the ECM name.
4. ECM therefore stops before implementation or teacher spend. `round-1-refinement.md` is archival/non-canonical and explicitly not executable.

## Pushback / Drift Log

| Reviewer said | Author response | Outcome |
|---|---|---|
| supervision/no-segment/test path are sound | accepted and preserved | PASS |
| current core is GroupDRO/JTT/gradient-surgery reducible | accepted; no cosmetic defense | ECM abandoned |
| raw-gradient constraints are invalid after AdamW | accepted | frozen QP prohibited |
| teacher may infer correctness propensity | accepted; future work must use matched ERROR-PROPENSITY | “correctness firewall” claim removed |
| full-bank proximal target could be distinct | retained only as a future fresh-hypothesis boundary | not an ECM revision |

## Remaining Weaknesses / Next Scientific Requirement

No current hypothesis has proven the target. A next Gate-0 candidate must operate on realized final-bank geometry or actual optimizer updates, be non-reducible to scalar sample weighting, prove semantic identity beyond an MLLM-derived scalar error propensity, and pass a zero-call two-dataset capacity screen before teacher spend. It must continue to use only video binary gold and no segment/timestamp/span annotation.

## Execution Audit

- ECM code changes: 0
- ECM SLURM jobs: 0
- ECM teacher/MLLM/OCR calls: 0
- ECM validation/test reads: 0
- Segment gold exists/used: `false/false`
- Final objective met: `false`
