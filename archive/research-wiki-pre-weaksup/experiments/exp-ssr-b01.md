# SSR-MemRGCL B0/B1 execution log

## Frozen supervision contract

- The only gold supervision is the parent video's binary label.
- No segment-level gold annotation exists or is assumed.
- Existing subclips may inherit the parent video label only; that inherited value is explicitly checked and is not described as segment gold.
- MLLM stance/target/proposition/mechanism fields are weak, privileged, train-only pseudo-signals. Validation/test videos are not relation endpoints.

## 2026-07-10 — implementation review and B0 static sanity

- Independent read-only reviews: `/root/ssr_bridge_executor/ssr_b0_code_review` and `/root/ssr_bridge_executor/ssr_fix_review`.
- Main fixes before execution: strict whole-response JSON (no brace/code-fence/value repair), dev/test ID-only disjointness without reading labels, per-call atomic shards with provenance/input-hash validation, canonical missing/no-edge records, frozen train/subclip hashes, fold-local membership assertions, and repository vote cross-check.
- Config bootstrap job `12686`: `COMPLETED`, canonical config digest `b32e57e8361392516c0e5087ea8c7bb5c85a6544ba384c5ea4bb3e9a4d64718e`.
- Frozen fold/config job `12687`: `COMPLETED`; MHC and MHC_zh split-ID disjointness and train-only cache hashes frozen.
- Static sanity job `12688`: `COMPLETED / GO`.
  - Config frozen: yes.
  - Strict parse positive and no-repair negative tests: pass.
  - BA per-video field canonicalization: pass.
  - Synthetic four-call record and post-record video-label signing: pass.
  - Forbidden MLLM payload keys: zero.
  - Segment/span/localization schema fields: zero.
- Logs: `slurm/logs/ssr_b01_cpu_{12686,12687,12688}.out`.
- This is implementation integrity only; B0 full OOF/relation extraction and every B1 empirical gate remain unproven. B2/B3 remain locked.

## 2026-07-10 — strict OOF mining and terminal B1 upper-bound STOP

### Jobs and artifacts

- OOF jobs `12691–12700`: all `COMPLETED/0:0`. MHC uses fixed epoch index 25 and MHC_zh index 28; each fold trains on the other four train folds only. Query folds are projected once after training and never select checkpoints, thresholds, prompts, or hyperparameters.
- Mining jobs `12701` (MHC) and `12702` (MHC_zh): `COMPLETED/0:0`.
  - MHC: 1,200 canonical pairs, 1,219 directed arcs; MI 634 arcs / 6 positive events; SC 585 / 18.
  - MHC_zh: 1,200 canonical pairs, 1,222 arcs; MI 619 / 6; SC 603 / 29.
- Formal all-candidates oracle upper bound: job `12704`, `COMPLETED/0:0`, `artifacts/ssr/v1/b1/preflight_oracle_upper_bound.json`.
- Machine verification: job `12705`, `COMPLETED/0:0`, `artifacts/ssr/v1/B1_DECISION.json`, decision `STOP`.

### Dual-metric oracle upper bound

This calculation is strictly more optimistic than the preregistered MLLM oracle: it assumes **every selected pre-MLLM candidate arc** is an accepted reliable relation. A real MLLM accepted set can only be a subset, so its event-touched correction set and both metric gains cannot exceed these values.

| Dataset | Family | OOF baseline acc / mF1 | Optimistic oracle acc / mF1 | Δacc / ΔmF1 | Unique touched errors | Required touched for +0.05 acc | Gate |
|---|---|---|---|---|---:|---:|---|
| MHC | MI | 0.7687 / 0.6934 | 0.7723 / 0.6982 | +0.0036 / +0.0048 | 2 | 28 | FAIL |
| MHC | SC | 0.7687 / 0.6934 | 0.7814 / 0.7110 | +0.0128 / +0.0176 | 7 | 28 | FAIL |
| MHC_zh | MI | 0.7599 / 0.7194 | 0.7651 / 0.7259 | +0.0052 / +0.0065 | 3 | 29 | FAIL |
| MHC_zh | SC | 0.7599 / 0.7194 | 0.7858 / 0.7501 | +0.0259 / +0.0307 | 15 | 29 | FAIL |

The common-family upper-bound set is empty. In particular, the strongest cell (MHC_zh/SC) still misses both thresholds by large margins. Relation reliability, audit precision, conditional significance, or a better MLLM parser cannot create new exact event-positive arcs outside the frozen candidate universe.

### Stop action and saved resources

- `SSR-B0-SMOKE-Q25VL7B-v1` and both full 7B extraction jobs were not launched.
- Human 2+1 bilingual audit, four 10,000-permutation conditional jobs, two exact-shuffle solves, B2, and B3 were not launched.
- Avoided the plan's remaining approximately 8–20 A100 GPU-hours for OOF relation extraction (plus smoke), 12–20 person-hours of audit, and substantial permutation/solver CPU time. B2/B3 compute was also preserved.
- This is not target success. It is a verified negative result for SSR-MemRGCL's frozen candidate/event geometry; the global +3 acc/+3 mF1 goal remains active and requires a different scientific hypothesis.

### SSR postmortem / anti-repeat rule

The failure is **coverage before semantics**. Exact one-neighbour SC removal and MI replacement events touch too few unique OOF errors, even when relation acceptance is assumed perfect. The MLLM cannot meaningfully improve final accuracy/macro-F1 through a selector whose entire correctable event universe has less than the preregistered headroom.

Do not retry SSR by changing prompts, teacher size, reliability thresholds, audits, relation predicates, loss weight, or extraction budget while retaining the same `C_SC/C_MI` candidate universe and single-neighbour `Y_SC/Y_MI` events. A successor must first prove, with label-only strict train OOF and before any MLLM calls, a common two-dataset dual-metric oracle ceiling of at least +0.05; it must enlarge the *causal correctable unit* rather than merely improve semantic assignment inside this failed universe. No successor may assume segment gold.
