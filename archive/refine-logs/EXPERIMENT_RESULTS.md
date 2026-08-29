# Initial Experiment Results

**Date**: 2026-07-10  
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`  
**Authorized scope**: B0/B1 only

## M0: Integrity, OOF geometry, and mining

- Frozen config/folds: jobs 12686–12687, passed.
- Static no-leak/no-repair/no-segment-gold sanity: job 12688, passed.
- Ten strict train-only OOF heads: jobs 12691–12700, all completed.
- Exact pair/event mining: jobs 12701–12702, completed with 1,200 canonical pairs per dataset.
- MLLM smoke/full extraction: not run because the later necessary oracle gate was already proven impossible under an all-candidates optimistic bound.

## M1: Necessary oracle upper bound

| Dataset | Family | Baseline acc | Baseline mF1 | Upper oracle acc | Upper oracle mF1 | Δacc | ΔmF1 | Touched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MHC | MI | 0.7687 | 0.6934 | 0.7723 | 0.6982 | 0.0036 | 0.0048 | 2 |
| MHC | SC | 0.7687 | 0.6934 | 0.7814 | 0.7110 | 0.0128 | 0.0176 | 7 |
| MHC_zh | MI | 0.7599 | 0.7194 | 0.7651 | 0.7259 | 0.0052 | 0.0065 | 3 |
| MHC_zh | SC | 0.7599 | 0.7194 | 0.7858 | 0.7501 | 0.0259 | 0.0307 | 15 |

All four cells fail the frozen `>=+0.050` accuracy and macro-F1 headroom gate even when every selected candidate is assumed accepted. Therefore no common family can pass B1.

## Summary

- `B1_DECISION=STOP`, verified by job 12705.
- Relation extraction, reliability/audit, conditional permutations, exact shuffle, B2, and B3 were correctly skipped as moot or locked.
- The final project target is not met. SSR-MemRGCL is a documented negative route, not a paper claim of performance improvement.
- Ready for auto-review-loop: NO. The next action is a new hypothesis whose pre-MLLM correctable-event universe passes the dual-metric ceiling before semantic extraction.
