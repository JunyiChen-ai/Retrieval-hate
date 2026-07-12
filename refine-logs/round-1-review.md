# Round 1 External Review

- **Reviewer agent id:** `/root/ssr_method_refine/ssr_reviewer`
- **Review standard:** independent GPT-5.4 xhigh-level top-venue method review
- **Parsed overall score:** 6.6 / 10
- **Parsed verdict:** REVISE

<details>
<summary>Full raw reviewer response (verbatim)</summary>

# Round 1 Review

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8/10 |
| Method Specificity | 6/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 6/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 6/10 |

**Weighted overall: 6.6/10**  
**Verdict: REVISE**

## Dimension Critiques

### 1. Problem Fidelity — 8/10

SSR-MemRGCL attacks the correct failure locus: wrong-neighbour attraction in the train-memory geometry used by the final kNN classifier. It correctly restricts gold supervision to video-level binary labels. The MLLM relations are train-only privileged pseudo-signals, not gold annotations, and no MLLM is used at test time.

The main fidelity exception is the cross-seed candidate-universe construction described under Drift Warning.

### 2. Method Specificity — 6/10

**Weakness:** The pair relation is cached as an unordered, duplicate-canonicalized object, while the retrieval error and ranking loss are directional. After canonicalization, it is unclear which video is the erroneous query anchor, whose rank/margin defines difficulty, and how the same pair is treated if it is harmful in only one retrieval direction.

The MLLM fields also lack an operational confidence variable. Exact A/B-order agreement provides partial filtering, but the paraphrase check is only a pilot diagnostic, not a per-edge reliability rule over the final graph.

**Method-level fix:** Cache MLLM annotations canonically, but project them back onto explicitly directed arcs `(query i, neighbour j)`. Define rank, margin, error involvement, reference selection, and degree with respect to `i`. Assign every pseudo-relation an agreement-derived reliability value from fixed order/paraphrase queries; retain only preregistered reliability levels and map all failures to `missing → no edge`. Never describe these fields as gold or as segment annotations.

**Priority:** CRITICAL

### 3. Contribution Quality — 6/10

**Weakness:** The MLLM relation type currently affects geometry mainly by selecting eligible edges and balancing buckets. CS− and TC− use the same negative hinge, while MI+ is a standard positive triplet. Thus the actual contribution is closer to “MLLM semantic hard-edge selection for kNN geometry” than to a genuinely relation-conditioned metric. This remains close to semantic hard-pair mining prior art.

**Method-level fix:** State the narrower mechanism honestly: typed relations select directed constraints at the specific wrong-neighbour locus. Either define a distinct, preregistered tuple-construction rule for each type or merge types that induce the same optimization. Require each retained type to pass the existing B1 conditional-information test separately; delete non-informative types before the method is frozen rather than preserving ornamental ontology fields.

**Priority:** CRITICAL

### 4. Frontier Leverage — 8/10

The foundation model is used appropriately as a frozen, constrained train-only relation annotator. It neither replaces labels nor becomes a feature-concatenation or test-time judging channel. No additional modernization is needed.

### 5. Feasibility — 6/10

**Weakness:** The final pair-pool size, maximum degree per anchor, number of MLLM calls, and number of unique relation endpoints forwarded per optimizer step are not bounded. Full-pool paraphrase-based reliability checking would also increase the stated extraction budget. The 20–40 GPU-hour estimate is therefore not auditable.

**Method-level fix:** Predefine a deterministic per-anchor candidate cap, post-template edge cap, total pair budget, and duplicate-endpoint batching policy. Re-estimate extraction and training cost from those fixed counts, including all reliability queries. This needs no new component.

**Priority:** IMPORTANT

### 6. Validation Focus — 6/10

**Weakness:** The proposed shuffle may be non-diagnostic. Shuffling CS− versus TC− within negative-polarity strata changes almost nothing because both use the same loss. Reassigning endpoints without an exact null definition may instead change degree, difficulty, or coverage. There is also no explicit pseudo-relation noise-sensitivity control, now required by the no-segment-gold constraint.

The 20% baseline-error coverage gate is insufficient evidence that +3 accuracy is plausible. For MHC-ZH, 20% coverage of the 17.45% baseline errors has a theoretical ceiling of only 3.49 points; reaching +3 would require correcting roughly 86% of covered errors with zero collateral damage. The analogous MHC-EN requirement is about 71%.

**Method-level fix:** Define the shuffle by permuting complete MLLM pseudo-relation records across directed candidate arcs within polarity, similarity, rank/margin, and error strata, then rebuilding templates while matching degree, edge count, and missingness. Add a noise-matched control that corrupts or drops pseudo-fields at the audit-estimated error rate while preserving graph statistics. Extend the existing B1 audit to measure unique-query coverage and type-specific information; treat 20% error coverage only as a minimum validity floor, not evidence that +3 is plausible.

**Priority:** CRITICAL

### 7. Venue Readiness — 6/10

**Weakness:** The proposal is focused and timely, but the defensible novelty is still a narrow combination. Until directed construction, pseudo-signal reliability, and the causal shuffle are fixed, a reviewer can interpret the method as ordinary hard-pair mining with MLLM-generated tags.

**Method-level fix:** Resolve the above blockers without adding a relation adapter or auxiliary module. Frame the paper around one claim: reliable train-only MLLM relation typing selects causal directed constraints that repair the exact geometry consumed by the final kNN classifier.

**Priority:** IMPORTANT

## Simplification Opportunities

1. Remove `evidence_binding_a/b`: these pseudo-fields are not used by any edge template or loss.
2. Treat the “mechanism-identification protocol” as validation, not a supporting paper contribution.
3. Either merge CS− and TC− into one negative loss bucket or give them genuinely distinct tuple-construction semantics; separate names with an identical hinge add ornamental complexity.

## Modernization Opportunities

**NONE.** The frozen train-only MLLM role is already appropriate. Additional adapters, routers, generators, or test-time reasoning would weaken the proposal.

## Drift Warning

The union of candidate pairs mined from baseline seeds 0/1/2 exposes every SSR seed to supervision selected by the other baseline seeds. This is cross-seed ensemble mining and violates the no-cross-seed-ensembling and paired same-protocol constraint.

Annotating the union once for efficiency is acceptable only if each SSR seed activates exclusively the edges mined by its corresponding paired baseline seed. Alternatively, use one preregistered seed-independent train-only miner.

There is **no segment-level gold annotation assumption** in the current method. All stance, target, mechanism, proposition, and evidence-binding outputs must remain explicitly described as weak/privileged train-only pseudo-signals.

**Anchor status: DRIFTED at the protocol level until the cross-seed graph is corrected.**

## Status Summary

- **Dominant contribution status:** Focused, but not yet fully defensible; currently semantic hard-edge selection rather than clearly relation-conditioned geometry.
- **Complexity status:** Architecturally lean and not a module pile. The blocker is ambiguous artifact/control construction, not model complexity.

## Ranked Remaining Actions

1. **CRITICAL:** Remove cross-seed supervision and preserve directed query–neighbour arcs.
2. **CRITICAL:** Define agreement-derived pseudo-signal confidence, explicit missing/no-edge fallback, and full-pool reliability handling.
3. **CRITICAL:** Replace the potentially inert shuffle with a precise graph-matched semantic null and add noise sensitivity.
4. **CRITICAL:** Clarify whether relation types create distinct constraints or merely select hard edges; tighten the claim accordingly.
5. **IMPORTANT:** Bound graph size and recompute the extraction/training budget.
6. **IMPORTANT:** Make B1 coverage target-aware; 20% error coverage alone does not make +3/+3 scientifically plausible.

**Final verdict: REVISE.**

</details>
