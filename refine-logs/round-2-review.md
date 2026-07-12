# Round 2 External Review

- **Reviewer agent id:** `/root/ssr_method_refine/ssr_reviewer`
- **Parsed overall score:** 7.4 / 10
- **Parsed verdict:** REVISE

<details>
<summary>Full raw reviewer response (verbatim)</summary>

# Round 2 Review

The revision resolves most Round-1 mechanism problems: retrieval constraints are now directed and seed-isolated, pseudo-signal uncertainty has an explicit no-edge fallback, the ontology has been reduced to two operative families, and the controls target semantics rather than added compute. It is substantially sharper, but two validation issues and one split ambiguity still block READY.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 8/10 |
| Method Specificity | 7/10 |
| Contribution Quality | 7/10 |
| Frontier Leverage | 9/10 |
| Feasibility | 7/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 6/10 |

**Weighted overall: 7.4/10**  
**Verdict: REVISE**

## Dimension Assessment

### 1. Problem Fidelity — 8/10

The revised mechanism preserves the substantive anchor:

- Video-level binary labels are the only gold supervision.
- No segment-level gold is assumed.
- All MLLM fields are explicitly weak, train-only privileged pseudo-signals.
- Low-reliability or missing fields produce no edge.
- The MLLM and relation artifacts are absent at inference.
- The final target remains unchanged kNN accuracy and macro-F1, not a native-head or localization result.

One protocol contradiction remains: B1 refers to “250 train/validation-only arcs,” while candidate construction and the amended anchor require train-only pseudo-signals and forbid validation graph nodes. If validation videos receive MLLM pseudo-relations, the amended anchor is violated.

### 2. Method Specificity — 7/10

The directed arc definition, query-relative difficulty, tuple equations, edge caps, reliability computation, seed masking, and endpoint budget are now implementable.

Remaining specification gaps are bounded but material:

- The global selection rule from up to six candidates per query to the 1,200-pair ceiling is unspecified.
- Family removal after B1 does not state whether MI+/SC− must pass on each claimed dataset or may be selected separately per dataset.
- Generation settings for the four MLLM calls and the exact human-audit adjudication rule are not frozen in the proposal.
- The constrained shuffle may have no feasible exact assignment in sparse strata; adjacent-bin merging does not guarantee feasibility.

These do not require a new component, only deterministic rules.

### 3. Contribution Quality — 7/10

The dominant contribution is now substantially sharper:

> Reliable MLLM pseudo-relations select directed semantic constraints on real hard neighbours; video labels sign them; a parameter-free ranking loss changes the final kNN geometry.

Merging CS−/TC− was correct. The proposal also now honestly claims semantic constraint selection rather than a new relation-specific metric.

Novelty remains narrow and compositional. Both MI+ and SC− ultimately instantiate a standard positive-versus-negative hinge; the novelty therefore lives entirely in which directed constraints the MLLM selects and whether that selection survives the matched null. This is defensible only if the conditional-information and shuffle evidence are decisive.

### 4. Frontier Leverage — 9/10

The MLLM role is appropriate and restrained. It provides privileged cross-video structure that binary labels cannot identify, without becoming a classifier, feature stream, generator, router, or test-time oracle. No modernization is needed.

### 5. Feasibility — 7/10

The graph and call budgets are now bounded, and the trainable model remains small. The route fits the stated resource envelope in principle.

The 28,800 pair-prompt ceiling makes the 20–60 GPU-hour estimate uncertain but auditable. The main feasibility risk is the exact graph-matched shuffle: preserving out-degree, indegree, family, reliability, missingness, polarity, and fine difficulty strata simultaneously may be infeasible for sparse relation families. A declared failure condition is needed; constraints must not be silently relaxed until a shuffle happens to exist.

### 6. Validation Focus — 6/10

**Weakness 1: split inconsistency.** B1 currently permits “train/validation-only arcs,” contradicting the train-only pseudo-signal contract.

**Fix:** Use a held-out audit subset of train-only directed arcs. If cross-fitting is needed, define train folds explicitly. No validation video should be sent to the MLLM or become a relation endpoint.

**Priority:** CRITICAL

**Weakness 2: the headroom gate covers accuracy only.**  
`H = error rate × touched-error fraction` is an accuracy upper bound. It does not establish macro-F1 headroom or class balance. Even with `H=0.05`, reaching +0.03 requires correcting 60% of all touched errors with zero collateral damage, so the gate remains permissive relative to the stated target.

**Fix:** Using only permitted video-level labels, calculate an oracle touched-query ceiling separately for accuracy and macro-F1: correct all and only baseline errors touched by reliable typed arcs and recompute both metrics. Require at least +0.05 oracle headroom in both metrics on each proposed dataset, and report the required realized fraction. This replaces the current scalar `H`; it is not an added experiment.

**Priority:** CRITICAL

**Weakness 3: family selection can become dataset-adaptive.** It is unclear whether MI+ may survive on MHC-EN but be removed on MHC-ZH.

**Fix:** Freeze one common family set before B2. A family should survive only under a preregistered cross-dataset rule—preferably passing reliability, precision, and conditional-information gates on each claimed dataset. If no common family survives, stop.

**Priority:** IMPORTANT

**Weakness 4: noise calibration is underdefined.** Estimating a categorical per-field confusion process from only 250 total arcs across two datasets and two families is likely unstable.

**Fix:** Calibrate corruption from template-level accepted-edge validity and missingness, with dataset/family stratification only where sample counts meet a preregistered minimum. Do not claim a detailed per-field confusion model that the audit cannot estimate.

**Priority:** IMPORTANT

### 7. Venue Readiness — 6/10

**Weakness:** The method is now elegant enough to test, but the novelty remains interface-specific and its causal evidence protocol still has the above ambiguities. A top-venue reviewer can still reduce it to semantically filtered triplet mining if the matched shuffle or common-family result is weak.

**Fix:** Resolve the validation blockers without adding modules. Keep the paper centered on one causal claim: correct MLLM relation-to-pair assignment repairs directed wrong-neighbour geometry beyond identical label, difficulty, degree, reliability, missingness, and compute.

**Priority:** IMPORTANT

## Simplification Opportunities

1. Use template-level accepted-edge error and missingness for `e_hat` instead of an underpowered per-field categorical confusion model.
2. Freeze one shared MI+/SC− family set across both target datasets rather than allowing dataset-specific method variants.

## Modernization Opportunities

**NONE.** Additional adapters, learned confidence models, routers, generators, or test-time reasoning would make the method worse.

## Drift Warning

**Exact potential drift:** B1’s “train/validation-only arcs” wording permits MLLM annotation of validation videos, conflicting with the amended train-only pseudo-signal constraint. Replace this with a held-out subset or cross-fit partition of training arcs only.

**Anchor status: DRIFTED at the B1 audit-protocol level until this wording and procedure are corrected; otherwise the method anchor is preserved.**

## Status Summary

- **Dominant contribution:** Sharper and singular; no longer a decorative three-relation stack.
- **Complexity:** Model-level mechanism is lean. The control suite is claim-essential, although the exact constrained shuffle is procedurally complex.
- **Frontier leverage:** Appropriate and well bounded.
- **Scientific plausibility of +3/+3:** Improved by the headroom gate but still modest; the current scalar gate does not yet establish macro-F1 headroom.

## Ranked Remaining Actions

1. **CRITICAL:** Make every B1 MLLM annotation and relation endpoint strictly train-only.
2. **CRITICAL:** Replace scalar accuracy headroom `H` with oracle touched-query ceilings for both accuracy and macro-F1.
3. **IMPORTANT:** Freeze one common surviving relation-family set across both claimed datasets.
4. **IMPORTANT:** Specify an exact shuffle solver and a preregistered “null infeasible → stop” rule.
5. **IMPORTANT:** Use estimable template-level noise calibration rather than sparse per-field confusion.
6. **MINOR:** Freeze global pair-budget allocation, MLLM generation settings, and audit adjudication details.

**Final verdict: REVISE.**

</details>
