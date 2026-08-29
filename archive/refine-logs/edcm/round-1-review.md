# Round 1 Review

**Reviewer agent:** `/root/edcm_pivot_refine/edcm_reviewer`  
**Reviewer continuity:** this same agent is required for all later rounds.  
**Overall:** 6.93 / 10  
**Verdict:** REVISE  
**Anchor:** preserved; drift warning NONE  
**No-segment-gold audit:** PASS

## Parsed Review

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 6.5 |
| Contribution Quality | 6.0 |
| Frontier Leverage | 7.0 |
| Feasibility | 7.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 6.0 |
| **Overall** | **6.93** |

### Priority actions

1. **CRITICAL:** replace `softmax(exp(-d))` with one explicit normalized kernel and freeze its shared scale.
2. **CRITICAL:** measure teacher-specific influence inside actual lists: total-variation from uniform and `Delta g_i = grad L_EDCM - grad L_uniform`; bind promotion to dense all-video activity.
3. **CRITICAL:** call A0 a conservative frozen-geometry reachability/cost screen, not an upper bound on a learned method.
4. **CRITICAL:** add exactly one matched teacher-free modality availability/content proxy control.
5. **IMPORTANT:** freeze checkpoint/decoding/prompts/frames/OCR/serialization/canonicalization hashes.
6. **IMPORTANT:** use “interventional coalition weak pseudo-signal,” not causal ground truth; reserve causal language for controlled effect.
7. Remove the optional absolute-score diagnostic and replace global pattern-count/headroom with the direct teacher-active gradient gate.

## Full Raw Reviewer Response (verbatim)

<details>
<summary>Round 1 raw response</summary>

# Round 1 Senior ML Review — EDCM-RGCL

## Overall assessment

EDCM preserves the target problem and is materially better aligned with the updated SSR failure than the previous sparse-event routes. It is one focused mechanism: a train-only MLLM coalition signature modifies the listwise gradient of the same full-video embedding used by final kNN. It does not degenerate into concatenation, segment weighting, teacher-key replacement, or test-time MLLM inference.

However, the current proposal has one blocking scientific issue: **dense signature coverage is not yet dense teacher-specific gradient support**. A globally non-degenerate signature cache can still yield nearly uniform compatibility inside most mined lists, reducing EDCM to Label-only ListNCA. The post-extraction “correctable upper bound” does not test this. The proposed loss also unintentionally compresses its compatibility weights through `softmax(c)`.

**Verdict: REVISE**

## Scores

| Dimension | Score | Weighted contribution |
|---|---:|---:|
| 1. Problem Fidelity | 9.0 | 1.350 |
| 2. Method Specificity | 6.5 | 1.625 |
| 3. Contribution Quality | 6.0 | 1.500 |
| 4. Frontier Leverage | 7.0 | 1.050 |
| 5. Feasibility | 7.0 | 0.700 |
| 6. Validation Focus | 8.0 | 0.400 |
| 7. Venue Readiness | 6.0 | 0.300 |
| **OVERALL** | **6.93 / 10** | |

## 1. Problem Fidelity — 9.0/10

The exact anchor is preserved:

- The endpoint remains final test accuracy and macro-F1, not localization, explanation, native-head accuracy, or audit value.
- The method targets the unchanged full-video memory geometry used by final kNN.
- MLLM outputs are train-only privileged pseudo-signals and disappear completely at validation/test.
- The proposal retains paired seeds, moving comparator, statistical requirements, removability, shuffle/noise controls, and the +3/+3-on-two-datasets stopping rule.
- It explicitly incorporates SSR’s sparse-support failure instead of stacking another sparse relation mechanism.

No substantive problem drift is present.

## 2. Method Specificity — 6.5/10

**Weakness — CRITICAL:** The loss does not implement the stated compatibility normalization cleanly. With

\[
c_{ij}=\exp(-d_{ij}),
\qquad q_{ij}=\operatorname{softmax}(c_{ij}),
\]

the already exponentiated kernel is exponentiated again. Since \(c\in(0,1]\), this compresses rather than clearly controls the list distribution; even maximally different signatures can produce relatively weak weight ratios. More importantly, accepted/non-uniform signatures do not guarantee non-uniform weights within each actual positive and negative list.

**Concrete fix:** Use one explicit kernel normalization:

\[
d_{ij}=\operatorname{mean}|s_i-s_j|,\qquad
q^\pm_{ij}=
\frac{\exp(-d_{ij}/\tau_s)}
{\sum_{k\in\mathcal L_i^\pm}\exp(-d_{ik}/\tau_s)}.
\]

Freeze a single shared \(\tau_s\) before development; do not tune it per dataset. The parameter-free alternative is \(\tau_s=1\), equivalently \(q=c/\sum c\).

Add a teacher-specific activity definition against the matched Label-only loss:

\[
\Delta g_i =
\nabla_{z_i}L_{\text{EDCM}}-
\nabla_{z_i}L_{\text{uniform}}.
\]

A reliable video should count as EDCM-active only when its positive/negative list weights depart materially from uniform and \(\|\Delta g_i\|\) is non-negligible. Report the active fraction over **all** OOF videos, not merely accepted signatures or errors.

**Weakness — IMPORTANT:** Reproducibility details remain underspecified: exact frozen MLLM checkpoint, decoding parameters, prompt texts, frame resolution, coalition serialization, tie handling, OCR checkpoint/version, and canonicalization rules.

**Concrete fix:** Freeze these in a teacher protocol manifest before extraction, including hashes and deterministic failure/fallback behavior.

## 3. Contribution Quality — 6.0/10

**Weakness — CRITICAL:** Signature-weighted supervised NCA is, by itself, close to generic conditional metric learning. The proposal establishes that the source is an MLLM, but not yet that MLLM reasoning contributes information beyond labels, modality availability, or generic listwise regularization.

The current Label-only and shuffle controls are necessary but insufficient. A signature may mostly encode whether transcript/OCR exists or how much content each channel contains. Shuffling can destroy this alignment even if no MLLM reasoning was required.

**Concrete fix:** Add exactly one matched teacher-free control, not another module: construct a deterministic modality-availability/content proxy with the same six-dimensional format, coverage, list sizes, loss, and workload. It may use only channel presence and pre-registered low-level quantities such as transcript/OCR length and frame availability. Full EDCM must beat this control as well as Label-only and shuffle. This is the smallest control that supports “the MLLM is indispensable” rather than “a per-video modality signature helps.”

Also derive and report the exact teacher-dependent gradient difference from Label-only ListNCA. That isolates the claimed contribution without adding architecture.

**Weakness — IMPORTANT:** “Causal signal” is too strong for the semantic status of the coalition ranks. Deterministic omissions are interventions on the teacher input, but the resulting preservation scores are not causal ground truth.

**Concrete fix:** Call them **interventional coalition pseudo-signals**. Reserve “causal role” for the controlled performance effect established by remove/shuffle/noise experiments.

## 4. Frontier Leverage — 7.0/10

Using a frozen MLLM for relative same-video coalition comparison is more natural than using its weak absolute hate verdict. The LUPI-style path—teacher only during training, internalized into a compact retrieval encoder—is timely and defensible.

The remaining risk is that the MLLM merely estimates modality availability. Passing the proposed teacher-free matched control is therefore central to the frontier claim.

## 5. Feasibility — 7.0/10

The proposal is implementable within the existing RGCL bank/mining path and adds no trainable module. The deterministic fallback is clean.

Primary risks are teacher throughput, strict four-call full-signature acceptance, and insufficient within-list compatibility variance. These should be resolved through the proposed staged gates before multi-seed training. All execution must remain SLURM-only.

## 6. Validation Focus — 8.0/10

The staged A0–A3 design is disciplined and falsifiable. Remove-MLLM, Label-only, signature shuffle, calibrated corruption, kNN-specific evaluation, per-class behavior, and no head↔memory redistribution are appropriate.

The main correction is to replace global cache-level “density” evidence with actual teacher-specific list-gradient density.

## 7. Venue Readiness — 6.0/10

**Weakness — IMPORTANT:** The proposal currently has a plausible narrow novelty claim, not yet a top-venue-ready one. A reviewer can describe it as supervised NCA with MLLM-derived sample similarities. The causal terminology and weak mechanism-aligned preflight further expose it to rejection.

**Concrete fix:** Freeze the contribution as:

> Relative MLLM coalition judgments define a train-only conditional neighborhood measure whose teacher-specific gradient changes the final kNN embedding geometry.

Then establish three separations experimentally:

1. EDCM versus generic Label-only ListNCA.
2. EDCM versus a matched teacher-free modality proxy.
3. EDCM versus shuffled/corrupted MLLM signatures.

Do not broaden the paper with localization, rationale, teacher-key, router, or generated-view claims.

## Dense-Support Preflight Assessment

### A0 pre-MLLM gate

The two-swap top-64 gate is **not vacuous**: it restricts the label oracle to nearby keys and a bounded number of replacements, and it uses only permitted video-level training labels. It is a useful fixed-geometry cost screen after SSR’s extremely small correctable universe.

It is not, however, a valid upper bound on EDCM representation learning:

- EDCM changes embeddings, whereas A0 freezes them.
- Training can move a key currently outside rank 64 into the final neighborhood.
- A0 uses no coalition information and therefore cannot establish MLLM usefulness.
- Passing mainly demonstrates that enough current OOF errors have nearby same-label alternatives; it does not show that EDCM will select or create those alternatives.

Therefore A0 should be described as a **conservative frozen-geometry reachability screen**, not a method upper bound. Its mandatory stop is an operational cost decision, not a scientific impossibility result.

### Post-extraction density

The existing coverage, agreement, pattern-support, and reliable-list gates establish that pseudo-signals are widely available. They do not establish dense MLLM influence. Four globally common signature patterns can still be homogeneous inside nearly every mined list.

Add a binding teacher-active gate:

- Compute total-variation divergence of \(q_i^+\) and \(q_i^-\) from their uniform counterparts.
- Compute \(\Delta g_i\) relative to Label-only ListNCA.
- Require a pre-registered large fraction of **all OOF videos** to have non-trivial teacher-specific weights and gradients.
- Within the structurally correctable error universe, require the teacher-specific gradient—not merely signature availability—to improve a fixed differentiable neighborhood-margin proxy relative to Label-only.

This makes the route genuinely dense and falsifiable before full training.

## No-Segment-Gold Audit

**PASS.**

- The only gold used is the video-level binary training label.
- Fixed frames are deterministic input sampling, not segment annotations.
- OCR is teacher input extracted from frames, not gold.
- Coalition ranks, necessities, synergies, preservation and confidence are weak train-only pseudo-signals.
- No timestamps, spans, segment labels, localization targets, or segment correctness metrics enter training or gating.
- Validation/test load no coalition or teacher artifacts.

One wording fix is advisable: replace “raw annotation transcript” with “dataset-provided transcript used as a raw input modality; it contains no temporal or segment-level gold annotation.”

## Degeneration audit

- **Concatenation:** no.
- **Static segment weighting:** no.
- **Segment-weighted memory:** no.
- **Teacher-selected memory key:** no.
- **Native-head substitution:** no.
- **Sparse relation-edge repetition:** no at the query-coverage level.
- **Possible degeneration:** yes, to Label-only ListNCA if within-list signature weights are nearly uniform. The teacher-active gradient gate must catch this.

## Simplification Opportunities

1. Remove the optional absolute-score diagnostic from the core validation package; it does not establish the central claim.
2. Replace global signature-pattern count plus the current post-extraction optimistic bound with one direct teacher-active list-gradient gate.
3. Keep the single EDCM loss. Do not add routers, student coalition views, teacher keys, or SSR stacking.

## Modernization Opportunities

1. Aggregate the four teacher calls as a distribution over ordinal ranks rather than retaining only a modal rank, while still producing a deterministic fixed signature and reliability scalar. This preserves uncertainty without adding a trainable module.
2. Frame EDCM explicitly as privileged interventional supervision for retrieval geometry, not as causal annotation or generic MLLM reasoning distillation.

## Drift Warning

**NONE.** The current route preserves the anchor. Adding test-time MLLM use, segment supervision, teacher-selected keys, generated views, or a new router would constitute drift and is unnecessary.

## Anchor Preserved

**Yes.**

## Remaining Action Items

1. **CRITICAL:** Correct the compatibility normalization and freeze its shared scale.
2. **CRITICAL:** Replace cache-level density with a binding teacher-specific within-list gradient-density gate.
3. **CRITICAL:** Recast A0 as a frozen-geometry cost/reachability screen rather than an upper bound on learned geometry.
4. **CRITICAL:** Add one matched teacher-free modality/content proxy control.
5. **IMPORTANT:** Freeze the exact teacher/OCR/prompt/decoding/canonicalization manifest.
6. **IMPORTANT:** Replace semantic “causal signal” claims with “interventional weak pseudo-signal”; reserve causality for controlled method effects.
7. **IMPORTANT:** Preserve the strict no-segment-gold wording in every payload, cache schema, gate, and paper claim.
8. Run A0/A1 only after these revisions; do not promote to multi-seed training merely because signature coverage is high.

## Final verdict

**REVISE**

The route is focused, anchor-faithful, implementable, and does not repeat concatenation or segment weighting. It is not READY because the present gates do not yet prove dense MLLM-specific gradient support, the compatibility weighting is underspecified/miscalibrated, and MLLM indispensability is not separated from a simple modality proxy.

</details>
