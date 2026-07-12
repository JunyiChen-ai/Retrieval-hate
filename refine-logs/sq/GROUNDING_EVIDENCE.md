# SQ-RGCL Iteration 4 — Grounding Evidence

## Authoritative local evidence read before proposing SQ

- `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, and the Iteration-3 section of `TARGET_REVIEW_RAW.md`.
- `research-wiki/TARGET_GATE0_ITER3_LITERATURE.md`, including its prior-art map and the earlier `SQ-RGCL` REVISE verdict.
- `refine-logs/EXPERIMENT_RESULTS.md` and `artifacts/ssr/v1/B1_DECISION.json`.
- `refine-logs/edcm/EXPERIMENT_RESULTS.md` and `artifacts/edcm/v1/A0_DECISION.json`.
- `refine-logs/cte/EXPERIMENT_TRACKER.md`, `refine-logs/cte/CTE_C0_C1_CODE_REVIEW.md`, and `artifacts/cte/v1/C0_DECISION.json`.
- `research-wiki/EXP_p2_neighbor_rerank.md` and `scripts/analysis/p2_out/p2_results_block.md`.
- `research-wiki/EXP_p4_schema_distill.md` and `scripts/analysis/p4_out/probe_gate.json`.
- Existing train archive schema in `data/Archive/{MHC,MHC_zh}/v2/train_Qwen2.5-VL-7B-Instruct_archive.jsonl`.

## Binding empirical lessons

1. **P2 comparability is not vote correctness.** The 7B archive judge dropped 82.9% of correct-vote versus 84.0% of wrong-vote neighbors on MHC (only +1.1-point selectivity) and 71.3% versus 68.1% on MHC-ZH (−3.2 points, anti-selective). Final mean deltas were −0.0016 accuracy on MHC and −0.0201 on MHC-ZH. SQ therefore must not use presentation similarity to keep/drop/rerank neighbors. It must prove conditional gradient/correction enrichment before training.
2. **P4 fields are decodable and label-informative but redundant.** Archive-field probes passed strongly, yet auxiliary prediction produced no registered win and harmed val-selected cells. SQ must not predict or concatenate fields; it must beat a same-posterior P4-style prediction control.
3. **SSR is sparse, not a universal impossibility result.** Its optimistic all-candidate bound touched only 2/7 MHC and 3/15 MHC-ZH errors depending on family, with every dual-metric +0.05 gate failing. SQ must use a dense full-bank action family rather than another sparse edge selector.
4. **EDCM is bounded by its frozen action space.** Top-64/two-swap reachability was 15 MHC and 22 MHC-ZH errors, with +0.0273/+0.0394 and +0.0380/+0.0444 accuracy/mF1. SQ moves the shared query/key representation and can admit old top-64 outsiders, but this only avoids that bound; it does not prove gain.
5. **CTE C0 did not test performance capacity.** CTE stopped because frozen FP32 scalar/vector parity exceeded the preregistered `2e-5` tolerance (`T` errors about `8.05e-5/1.01e-4`; cost errors about `2.22e-5/2.19e-5`). Support, memory, margin, and gradient checks passed; C1 was never run and teacher calls remained zero. This is a numerics/cost-policy STOP, not a scientific performance ceiling.
6. **Existing v2 archive schema has no nuisance posterior.** Each train record contains `target_groups`, `mechanism`, `modality_cues`, `explicitness`, and `neutral_summary`. Only `neutral_summary` may enter a zero-new-call presentation proxy. The loader must reject/ignore outer `label`, `raw_output`, target/mechanism/explicitness/modality fields for nuisance construction. A promoted teacher artifact may contain only the six-way posterior, confidence, parse/provenance metadata, and no free-form semantic fields.

## Prior-art pressure and defensible boundary

- Yang et al. (ICML 2023) already use language-discovered spurious attributes and multimodal decorrelation.
- CDAL already covers semantic/sensitive subspaces, orthogonality, and HSIC; CARE covers invariant/environment-specific concept directions; dependent-factor work warns against independent-factor assumptions.
- TextTeacher and EmbedDistill already cover train-time semantic supervision and retrieval-embedding geometry distillation.
- HVGuard, RAMF, DR-HM, ExPO-HM, and BPDMoE-Hate crowd reasoning features, rationale distillation, policy optimization, gating, and MoE.

Therefore SQ may claim only a narrow mechanism-level delta: **confidence-bearing, train-only whole-video presentation posteriors define two conjugate class-conditional full-bank relations—cross-environment/same-label contraction and same-environment/different-label repulsion—on the exact final RGCL memory embedding, with no nuisance predictor or teacher artifact at inference.** It must not claim causal deconfounding, generic disentanglement, first semantic KD, or first environment invariance.
