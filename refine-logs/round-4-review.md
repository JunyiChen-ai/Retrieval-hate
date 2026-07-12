# Round 4 External Review

- **Reviewer agent id:** `/root/ssr_method_refine/ssr_reviewer`
- **Parsed overall score:** 9.03 / 10 (reported as 9.0)
- **Parsed verdict:** READY for experiment handoff

<details>
<summary>Full raw reviewer response (verbatim)</summary>

# Round 4 Review

The revised proposal closes the final two blocking specification issues. B1 now uses one internally consistent OOF retrieval geometry, and the canonical-pair dependence is preserved exactly between full and shuffled graphs. Remaining uncertainty concerns whether the proposed signal exists and is strong enough—not how the method should be defined.

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 10/10 |
| Method Specificity | 9.5/10 |
| Contribution Quality | 8/10 |
| Frontier Leverage | 9.5/10 |
| Feasibility | 8.5/10 |
| Validation Focus | 9.5/10 |
| Venue Readiness | 8/10 |

**Weighted overall: 9.03/10, rounded to 9.0/10**  
**Verdict: READY for experiment handoff**

This does not mean the performance or paper claim is proven. It means further proposal rewriting is no longer the right way to reduce uncertainty; B0/B1 evidence is now required.

## Dimension Assessment

### 1. Problem Fidelity — 10/10

The amended anchor is fully preserved:

- Video-level binary labels are the only gold supervision.
- No segment-level gold or dense semantic annotation is assumed.
- Every MLLM relation is explicitly a weak/privileged train-only pseudo-signal.
- Reliability, missingness, parsing failure, and `unclear` values have deterministic no-edge fallback.
- Validation and test videos never receive MLLM annotations.
- The MLLM and relation files are absent at inference.
- Success remains the unchanged train-memory kNN readout under the fixed two-dataset, three-seed, +3 accuracy/+3 macro-F1 protocol.

**Anchor status: PRESERVED.**

### 2. Method Specificity — 9.5/10

The proposal is implementation-ready. It now specifies:

- Separate OOF diagnostic and final full-train graph universes.
- Internally consistent ranks, margins, predictions, neighbour events, and references.
- Deterministic global pair allocation and direction masks.
- Fixed MLLM inputs, decoding, prompts, schema, reliability, and hashes.
- Exact tuple equations, reference selection, caps, workload, and fallback.
- One common cross-dataset family-selection rule.
- A canonical-record shuffle that preserves directional dependence.
- Explicit stop behavior when the causal null is infeasible.

One minor handoff detail remains: the exact binary definition of the MI “missing-helpful-neighbour event” should be frozen in the experiment specification—for example, whether it means the matched opposite-label item outranks the MI+ item, the MI+ item falls outside top-k, or a specified conjunction. This is not a conceptual blocker.

### 3. Contribution Quality — 8/10

The method now has one clean contribution:

> Correct assignment of reliable MLLM pseudo-relations to directed real hard pairs supplies constraints that video labels and generic hardness cannot, directly repairing the final kNN geometry.

The architecture is parsimonious, the division of responsibility is precise, and the proposal does not overclaim a new metric or general graph-learning paradigm.

The score remains below 9 because the novelty window is intrinsically narrow: the loss is familiar, and novelty depends on the retrieval-specific relation-to-pair interface plus decisive causal evidence. No proposal-level module can improve this cleanly; B1/B2 must demonstrate the claimed conditional information and matched-null effect.

**Dominant contribution status: SHARP, SINGLE, AND DEFENSIBLE IF THE FROZEN GATES PASS.**

### 4. Frontier Leverage — 9.5/10

The foundation model is used exactly where it is potentially informative: constrained cross-video stance–target–mechanism relation typing. It is not used as a classifier, feature stream, generator, router, or test-time judge.

**Frontier leverage status: APPROPRIATE AND NON-DECORATIVE.**

### 5. Feasibility — 8.5/10

Pair calls, OOF models, graph sizes, endpoint counts, human review, and GPU estimates are bounded and compatible with the stated resources.

Two empirical feasibility risks remain:

- A common relation family may not yield 80 accepted records with a Wilson lower bound of at least 0.80 on both datasets.
- The exact canonical shuffle integer program may be infeasible.

Both have correct stop rules. They are legitimate fast-fail outcomes, not missing method definitions.

### 6. Validation Focus — 9.5/10

The three validation blocks are minimal and sufficient for the two claims:

1. Strict train-only OOF reliability, conditional information, and dual-metric oracle headroom.
2. Seed-0 causal comparison against label-only mining, canonical semantic shuffle, removal, and calibrated noise.
3. Frozen two-dataset, three-seed evaluation against the moving binding baseline with the required statistics and mechanism controls.

The controls preserve labels, difficulty, directed/canonical dependence, graph statistics, missingness, reliability, and workload. This is unusually strong causal attribution for an MLLM-supervised representation method without expanding into benchmark bloat.

### 7. Venue Readiness — 8/10

If the method reaches the frozen +3/+3 gate and shows significant costs under removal and canonical semantic shuffle, it would form a focused and timely top-venue submission. Venue readiness is below 9 only because the novelty is narrow and the ex-ante probability of reaching the hard numerical target remains modest.

That uncertainty can now only be resolved experimentally.

## Simplification Opportunities

**NONE.** The learned method is already minimal: two possible edge families, one ranking term, zero new parameters, and no inference change. The procedural controls are necessary for the causal claim.

## Modernization Opportunities

**NONE.** Adapters, routers, generated data, learned confidence, or test-time reasoning would weaken attribution and create contribution sprawl.

## Drift Warning

**NONE.**

The OOF universe is strictly train-only; final seed graphs remain isolated; pseudo-relations are never gold, segment annotations, or test-time signals.

## Complexity Status

**LEAN METHOD; RIGOROUS BUT JUSTIFIED VALIDATION.**

The exact shuffle and cross-fit are procedural complexity required to test the claim, not additional method modules.

## Remaining Uncertainties

There are no blocking proposal-level issues. The remaining questions are empirical:

- Does either relation family meet the common two-dataset reliability and conditional-information gates?
- Is dual-metric oracle headroom at least +0.05?
- Is the exact semantic null feasible?
- Does seed-0 full SSR beat baseline, label-only, and shuffle by at least one point in both metrics?
- Can the frozen method realize +3/+3 across both datasets and all paired seeds?

Failure of any gate should stop the route as specified, not trigger architecture expansion or prompt/model scaling.

## Ranked Actions

1. **Proceed to experiment handoff:** Implement B0/B1 exactly as frozen.
2. Freeze the precise MI omission-event predicate and conditional-test statistic before inspecting B1 results.
3. Test the canonical integer-program feasibility on the frozen graph; stop if infeasible.
4. Apply the common-family intersection and Wilson/headroom gates without dataset-specific rescue.
5. Launch B2/B3 only after every preceding gate passes.

**Final verdict: READY for experiment handoff.**

</details>
