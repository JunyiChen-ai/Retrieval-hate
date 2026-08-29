# SQ-RGCL S0 provenance / blind-presentation fast-fail

**Date:** 2026-07-11  
**Decision:** `SQ-S0-DECISION-v1 = STOP`  
**Authoritative artifact:** `artifacts/sq/v1/S0_DECISION.json`  
**Scope:** S0 only. S1 and S2--S4 were not authorized after STOP.

## What ran

| SLURM job | Run | Result |
|---:|---|---|
| 12722 | config canonical hash | completed; `8645c972...c2149` |
| 12723 | reviewed static sanity | PASS; Python compile and sbatch syntax checks |
| 12724 | formal config/data/code freeze | PASS |
| 12725 / 12726 | MHC-ZH / MHC archive provenance | `PASS_PROXY_ONLY` |
| 12727 / 12728 | MHC-ZH / MHC six-way local-CLIP proxy | PASS; coverage 1.0 |
| 12729 | blind whole-video QC sheet freeze | artifact prepared; human QC not ingested |
| 12730 | independent joint S0 decision | **STOP** |

Before execution, an independent code review found four CRITICAL and nine HIGH issues. The reachable formal-STOP path was repaired and re-reviewed at **0 CRITICAL / 0 HIGH**. In particular, self is removed before train-anchor ranking, ranks are reassigned, exact top-20 weights are always `20..1`, and full-rank order is never reused as top-20 vote arithmetic.

## Evidence and decision

- MHC archive: 549/549 unique frozen train IDs, SHA-256 `fa1796...fd6`.
- MHC-ZH archive: 579/579 unique frozen train IDs, SHA-256 `7c83e3...a27`.
- Runtime archive reader accessed only `id`, `split`, `parse_ok`, and `archive.neutral_summary`; forbidden-key access count was zero.
- Neither archive nor its original log embeds cryptographically linked original prompt, exact model revision, generator-code hash, and input-manifest hash. Current-source inspection or timestamp adjacency is not accepted as original provenance.
- The deterministic six-way proxy used local `openai/clip-vit-large-patch14-336` over `neutral_summary` only. It made no MLLM/teacher call and cannot be renamed as a promoted MLLM posterior.
- Two 64-video, label-blinded, whole-video presentation-QC sheets were frozen. They contain no dataset-label column and are QC only—not annotation or training supervision. No human judgments were invented.

The independent decision rehashed/recomputed the freeze, current implementation, both archive hashes, posterior numeric validity/ID uniqueness, posterior output hashes, and zero-call counters. For both datasets, `original_provenance_complete=false`, `blind_presentation_audit_completed=false`, and `blind_presentation_audit_pass=false`. Therefore:

```text
q_signal_status = PROXY_ONLY_CHEAP_FORMAT
S0 = STOP
lambda_Q = null
S1_unlocked = false
S2_unlocked = false
S2--S4 = locked
```

## Scientific boundary

This is a governance/provenance fast-fail, not a learned-performance result and not a theoretical upper bound. P0 conditional relevance, power, GPU microbenchmark, learned strict-OOF SQ-0, and all accuracy/macro-F1 comparisons were not run after the binding failure. It would be incorrect to claim that SQ improves or fails classification from this execution.

The only gold supervision remains the parent-video binary label. `segment_gold_exists=false` and `segment_gold_used=false`; no segment/timestamp/span/localization annotation was assumed.
