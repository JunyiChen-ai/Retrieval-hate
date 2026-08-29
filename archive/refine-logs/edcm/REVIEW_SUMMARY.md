# EDCM-RGCL Review Summary

**Problem:** Give an MLLM a meaningful, novel and removable role in hateful-video RGCL that can produce at least +3 accuracy and +3 macro-F1 on two datasets × seeds 0/1/2.  
**Initial approach:** MLLM judgments over deterministic visual/speech/on-screen-text coalitions control final-memory listwise gradients.  
**Date:** 2026-07-10 (Pacific/Auckland)  
**Reviewer:** `/root/edcm_pivot_refine/edcm_reviewer` (one continuous independent reviewer)  
**Rounds:** 3 / 5  
**Final score:** 9.11 / 10  
**Final verdict:** READY for staged experiment handoff; global performance target not yet met.

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Round-by-Round Resolution

| Round | Main concern | Simplification / mechanism correction | Result | Remaining risk |
|---:|---|---|---|---|
| 1 | Cache coverage did not prove within-list MLLM gradient density; kernel double-exponentiated; modality availability could explain the signature | Corrected `q=softmax(-d/tau)`; replaced global density with TV/`Delta g`/directional support; added one low-level proxy; removed absolute-score diagnostic | 6.93 REVISE | proxy strength and OOF formulas |
| 2 | Proxy range/gradient strength unmatched; fold-local gradient contract incomplete | Fold-bank robust signed proxy; one shared median-`R` strength match; exact `vE/vU/DeltaD`, 8+8, self-exclusion, bank/hash contract | 8.28 REVISE | bounded freeze details |
| 3 | Audit of strength matching, leakage, final-memory locus and no-segment-gold | Reviewer found both blockers closed; final proposal freezes finite proxy grid, edge cases, query-signature hashes and corruption-survival rule | **9.11 READY** | empirical A0–A3 outcomes only |

## Overall Evolution

- Sparse SSR-style event support was replaced with all-video listwise influence, and A0 now stops the route before any MLLM call if the video-level OOF neighborhood lacks target-sized reachability.
- “Dense” became measurable MLLM-specific training influence through TV, relative gradient ratio `R` and reachable-error directional advantage `DeltaD`, not a claim of dense annotation.
- One strength-matched teacher-semantic-free proxy separates semantic alignment from generic modality availability and perturbation magnitude.
- The method stayed at zero new trainable modules: one frozen teacher cache and one listwise auxiliary loss.
- No segment-level gold was introduced at any stage; all coalition outputs remain weak train-only pseudo-signals.

## Final Status

- **Anchor:** preserved.
- **Focus:** tight; one dominant retrieval-geometry contribution.
- **Modernity:** appropriate privileged interventional supervision; no forced module.
- **Strongest point:** the MLLM-dependent gradient is measured and falsified against Label-only, strength-matched proxy, shuffle and noise before final test.
- **Remaining weakness:** the route has no experimental evidence yet. `READY` authorizes only A0 experiment planning/implementation; target completion remains false.
