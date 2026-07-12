# Round 3 External Review

- **Reviewer agent id:** `/root/ssr_method_refine/ssr_reviewer`
- **Parsed overall score:** 8.1 / 10
- **Parsed verdict:** REVISE

<details>
<summary>Full raw reviewer response (verbatim)</summary>

# Round 3 Review

The revision closes all five Round-2 issues at the policy level: B1 is train-only, headroom covers both target metrics, the relation-family set is shared across datasets, the shuffle cannot be relaxed post hoc, and pseudo-signal reliability/noise handling is operationally bounded. The proposal is now focused and close to execution-ready. Two remaining specification issues prevent READY.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10/10 |
| Method Specificity | 8/10 |
| Contribution Quality | 7/10 |
| Frontier Leverage | 9/10 |
| Feasibility | 7/10 |
| Validation Focus | 8/10 |
| Venue Readiness | 7/10 |

**Weighted overall: 8.1/10**  
**Verdict: REVISE**

## Dimension Assessment

### 1. Problem Fidelity — 10/10

The amended anchor is fully preserved:

- Video-level binary labels are the only gold supervision.
- No segment-level gold or dense annotation is assumed.
- Every MLLM field is explicitly a train-only weak/privileged pseudo-signal.
- Reliability, parse failure, `unclear`, and missingness have deterministic no-edge fallback.
- Validation and test videos receive no MLLM annotation.
- No pseudo-signal or MLLM is available at inference.
- Success remains final train-memory kNN accuracy and macro-F1 under the fixed +3/+3 protocol.

**Anchor status: PRESERVED.**

### 2. Method Specificity — 8/10

Most interfaces are now sufficiently concrete: inputs and decoding are hashed, graph direction and allocation are deterministic, reliability and loss equations are explicit, and resource ceilings are fixed.

One blocking ambiguity remains in B1. The audited pairs are mined by full-train seed-specific baselines, while the diagnostic outcomes and margins come from five-fold OOF models. For SC−, a neighbour selected by a full-train geometry may not be retrieved—and therefore cannot be a wrong-vote contributor—in the corresponding OOF geometry. The MI+ “missing helpful neighbour” outcome is likewise undefined unless its rank and vote role are recomputed in the OOF memory. This mixes two geometries in the conditional-information gate.

**Required fix:** Define a separate OOF diagnostic arc universe using each held-out training fold as query and the other four folds as memory. Compute ranks, margins, decisive wrong votes, and MI+ omissions entirely in that OOF geometry. Use the same frozen relation templates and miner policy; final training graphs can remain seed-specific full-train graphs.

### 3. Contribution Quality — 7/10

The contribution is now singular and honestly scoped. The method no longer claims a novel metric: novelty lies in assigning reliable semantic pseudo-relations to the correct directed hard pairs and internalizing those constraints into the final kNN geometry.

The residual weakness is intrinsic rather than architectural: both families use a familiar ranking hinge, so top-venue novelty depends on demonstrating that correct relation-to-pair assignment provides conditional information and beats an exact graph-matched null. The proposal cannot strengthen this by adding modules; it must win its existing causal tests.

**Dominant contribution status: SHARP AND FOCUSED, but empirically evidence-dependent.**

### 4. Frontier Leverage — 9/10

The frozen MLLM is used at the appropriate abstraction level: constrained cross-video semantic typing under privileged train-only access. It neither predicts labels nor becomes a feature, router, generator, or test-time judge.

**Frontier leverage status: APPROPRIATE.**

### 5. Feasibility — 7/10

Compute, graph size, annotation effort, endpoint count, and training workload are now bounded. The route fits the stated resources.

The main risk is the exact shuffle’s feasibility. Matching fine-grained strata, query out-degree, neighbour indegree, family counts, reliability, missingness, and a no-fixed-pair constraint may often yield no solution. Stopping when infeasible is scientifically correct, but it lowers execution probability. This is a risk rather than a reason to weaken the null.

### 6. Validation Focus — 8/10

The validation suite is claim-driven and contains the necessary controls: exact removal, label-only matched mining, semantic shuffle, calibrated corruption, topology diagnostics, and the frozen final statistical gate.

A second specification issue concerns the shuffle unit. The teacher produces one record per canonical unordered pair, while the integer program assigns records to directed arcs. If both directions of one pair are present, the proposal does not say whether the canonical record is duplicated and shuffled independently or must remain coupled across directions. Independent duplication changes the dependence structure relative to full SSR.

**Required fix:** Define the assignment unit as the canonical pair record and project each reassigned record jointly onto all eligible directed arc instances, or explicitly prove that directed copies are the operative independent units in both full and shuffle. The same dependence structure must hold in both arms.

The dual-metric oracle is now correct as a feasibility ceiling. It remains intentionally optimistic: `+0.05` headroom means +3 still requires realizing at least 60% of the oracle repair with no offsetting errors. B2 appropriately determines whether that potential is actionable.

### 7. Venue Readiness — 7/10

The method is timely, parsimonious, and falsifiable. It is not yet at READY because the novelty is narrow and the two causal gates still contain geometry/unit ambiguities. After those are corrected, further proposal-level module invention would be counterproductive; venue readiness should be decided by B1/B2 evidence.

## Simplification Opportunities

**NONE.** The remaining cross-fit and matched-control machinery is necessary for causal attribution. The learned method itself remains one loss over two edge templates with no new parameters.

## Modernization Opportunities

**NONE.** Additional foundation-model components would add complexity without repairing a current bottleneck.

## Drift Warning

**NONE.** The prior validation-split drift is corrected. OOF diagnostics use only training videos, and no segment-level or test-time pseudo-annotation is introduced.

## Complexity Status

**LEAN METHOD, HEAVY BUT NECESSARY VALIDATION.** There is no module pile. The only procedural complexity risk is the exact semantic-null construction.

## Ranked Remaining Actions

1. **CRITICAL:** Build B1 outcomes and candidate arcs in one internally consistent OOF retrieval geometry; do not mix full-train-selected neighbours with OOF vote outcomes.
2. **CRITICAL:** Specify whether the semantic-shuffle assignment unit is a canonical unordered record or a directed copy, preserving the same dependence structure as full SSR.
3. **IMPORTANT:** Freeze the minimum accepted audit count and whether the ≥0.80 precision gate applies to the point estimate or Wilson lower bound.
4. **IMPORTANT:** After these specification fixes, proceed to B0/B1 rather than adding method components. Conditional information, common-family survival, null feasibility, and dual-metric headroom are now the decisive unknowns.

**Final verdict: REVISE.**

</details>
