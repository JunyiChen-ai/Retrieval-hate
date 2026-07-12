# CTE-RGCL Round 2 Review Checkpoint

<details><summary>Raw independent reviewer response</summary>

# CTE-RGCL Independent Senior Review — Round 2

## Executive Assessment

The revision resolves the two central Round 1 conceptual objections in an acceptable way:

- It no longer equates typed withholding with prototype shrink; the mechanism is correctly framed as a withholding-informed tangent whose transfer must be empirically falsified.
- It no longer claims that the relation logically identifies the gold-margin direction; class-conditional two-radius transfer is now a hard weak-label gate.

The bank, bounded-cost claim, support audit, pilot cap, controls, and video-label-only supervision are substantially better specified. The proposal remains focused: one relation cache, one auxiliary loss, one shared encoder, zero new trainable components, and unchanged full-video kNN inference.

One important implementation blocker remains: the tangent whose transfer is tested in A1 is not frozen. Recomputing the spherical medoid and potentially reselecting the adjacent radius pair at every bank refresh can silently change the intervention after the teacher-to-tangent transfer gate has passed. Several smaller protocol ambiguities should also be resolved before coding.

## Anchor and Drift Assessment

**Anchor: PRESERVED.**

The revision retains every immutable endpoint requirement:

- meaningful train-only MLLM integration;
- ordinary full-video train-memory kNN at validation/test;
- no teacher, relation, confidence, prototype view, or view artifact at inference;
- at least `+0.030` accuracy and macro-F1 on both MHC-EN and MHC-ZH;
- paired seeds 0/1/2;
- REMOVE, SHUFFLE, label-only, heuristic, random, multiview, and NOISE attribution;
- moving strongest non-MLLM comparator.

**Method drift: NONE.**

No teacher label, rationale, segment signal, teacher key, second encoder, adapter, router, score fusion, reranking, scaling, or SSR/EDCM operation has been introduced.

## Dominant Contribution and Simplicity

The dominant contribution is now sharp:

> Label-blind whole-modality withholding relations, after class-conditional transfer validation, supervise the supported local response of an epoch-refreshed full-bank true-class retrieval margin.

This is one coherent mechanism rather than a collection of modules. The support audit, A0, A1, and assignment controls are falsification infrastructure, not separate claimed contributions.

Simplicity is strong:

- zero new trainable components;
- one shared query/key encoder;
- one prototype path per modality;
- one interval loss;
- one relation schema;
- unchanged test inference.

No architectural expansion is needed.

## Remaining Blocking Mechanism Issue

### The validated tangent can change after A1

The proposal currently recomputes the spherical medoid at every bank refresh and says to choose the largest supported adjacent radius pair. This creates two forms of nonstationarity:

1. The medoid can switch to a different train video as projected geometry changes.
2. The selected pair can change among `(.05,.10)`, `(.10,.20)`, and `(.20,.30)`.

A1 validates teacher transfer for the A0 encoder’s medoid and two radii. If training later switches the medoid or radii, the clean method is optimizing a different tangent from the one that passed the transfer gate. Per-epoch joint support does not establish ordinal transfer for the new direction.

Required fix:

- Before teacher calls, freeze the adjacent pair `(a1,a2)` for each fold/dataset using the A0-selected checkpoint. Never dynamically switch it.
- Freeze the spherical-medoid **video ID** selected at that checkpoint.
- At refresh, re-encode that fixed medoid ID using the current shared encoder; do not select a new medoid ID.
- Continue recomputing the teacher-independent per-example support mask at the frozen radii.
- Log fixed-anchor direction drift. If support coverage falls below the frozen threshold or anchor-direction drift exceeds a preregistered threshold, stop; do not select a replacement prototype or radius.
- Apply exactly the same fixed anchor/radii and support rule in every control.

This preserves the single-component design and requires no new supervision or module.

## Remaining Specification Fixes

### 1. Make A0 probe cross-fitting unambiguous

“Fit on two folds, choose regularization on the second fold, and predict the third” admits multiple implementations.

Freeze the rotation as:

1. fit each regularization candidate on fold A;
2. select it on fold B;
3. refit that candidate on A∪B;
4. predict fold C;
5. rotate A/B/C.

No target for a video may come from a probe trained or selected using that video.

### 2. Define A1 effective weight

“Effective weight ≥10” needs one formula. Use a video-clustered effective sample size such as

\[
n_{\mathrm{eff}}=\frac{(\sum_v w_v)^2}{\sum_v w_v^2},
\]

where modalities belonging to one video are combined before computing the class-by-relation cell weight. Freeze whether this requirement applies per dataset × label × relation, as the surrounding text implies.

### 3. Resolve the NOISE-rate inconsistency

The pilot-update section requires one noise rate, while the control section specifies two rates.

A minimal consistent protocol is:

- A1 uses the higher of two preregistered rates as a falsification gate;
- A2/final uses both rates and requires monotone degradation;
- both rates are computed once from pilot call disagreement before any pilot update outcome is examined.

### 4. Identify the A1 label-only target source

Specify whether pilot videos use precomputed strict OOF A0 targets or new update-fold-only probe targets. The simplest leakage-safe choice is to cache the strict OOF A0 target for every eligible train video before pilot selection and reuse that cache unchanged.

### 5. Clarify gradient matching language

The relation-free arm is teacher-independent in its mask and assignment, but its global norm-matching scalar uses the clean arm. Describe it as **assignment-free and teacher-mask-free**, not completely teacher-independent. This global strength matching is acceptable because it deliberately gives the control the clean arm’s aggregate optimization magnitude without its per-video relations.

At A0, match multiview/random gradients to the label-only arm because no teacher clean arm exists.

### 6. Freeze the statistical p-value construction

The hierarchical resampling and percentile lower bounds are specified, but “10,000 replicates yield four p-values” is incomplete. Define a centered-null bootstrap or another exact frozen conversion from replicates to one-sided p-values.

Also distinguish:

- the deterministic `+0.030` gate against the maximum including the historical point;
- paired statistical inference against the same-seed comparator for which paired predictions exist.

Do not imply paired inference against a historical scalar if its sample-level predictions are unavailable.

None of these six fixes requires another module or experiment family.

## Full-Bank and Numerical Audit

**Pass, with the frozen-path correction above.**

The revised specification correctly states:

- every full train key is used;
- self-ID is excluded;
- the bank is exact over the epoch-start keys, not continuously current geometry;
- query and key share one encoder;
- keys are detached and refreshed;
- query stochastic semantics match bank construction;
- each video becomes a CTE query once per epoch;
- drift has a frozen threshold and a common refresh fallback;
- cost and reliability weight are bounded, but gradients are not claimed bounded;
- MAD and normalization floors, non-finite failure handling, clipping, and gradient-ratio logging are present.

The interval loss is now mathematically executable.

## A0 Audit

**Conceptually valid and correctly bounded in interpretation.**

A0 is explicitly an empirical screen of this exact action family, not a theoretical upper bound and not MLLM evidence. The nested video-label-only target is appropriate, provided the probe rotation is clarified.

The shared-dataset minimax hyperparameter rule is conservative and resists dataset-specific rescue. The interval triplet does not affect the continuous A0 target directly; therefore its A0 “selection” will effectively follow the frozen tie rule. It would be simpler to freeze the interval triplet directly before A0, but this is not a conceptual blocker.

## A1 Audit

**Budget: PASS.**

The absolute maximum remains:

\[
128\times2\times2\times2\times2=2048
\]

teacher calls. Only strict train videos enter the pilot.

**Orientation test: PASS in design.**

The revised test:

- separates `y=0` and `y=1`;
- uses two supported radii;
- requires ordered preserve/weaken/reverse effects;
- clusters permutation and bootstrap at video level;
- prevents pooled evidence from rescuing a failed class;
- does not request an absolute teacher verdict.

This is an appropriate empirical justification for the otherwise unidentified sign mapping.

The per-level effective-sample gate is stringent, especially for `reverse`, but that is a scientific risk rather than a specification flaw. Failure should correctly stop the route.

## Control Audit

The critical controls are now largely valid.

- **REMOVE:** exact endpoint comparator.
- **Multiview:** teacher-mask-free and assignment-free after the wording clarification.
- **Label-only:** same loss family.
- **Energy/random:** same relation space and update budget.
- **SHUFFLE:** the coarse pre-audited derangement is feasible and avoids post-outcome relaxation.
- **NOISE:** preserves distributions and record structure once the one-versus-two-rate wording is aligned.

The indivisible two-modality SHUFFLE is especially important because it tests video-specific teacher assignment rather than merely global relation frequencies.

## Supervision Audit

**PASS.**

The only gold supervision is the parent-video binary label. It is used for:

- true-class full-bank margins;
- A0 probe orientation;
- train-only stratification;
- class-conditional transfer testing;
- video-level endpoint evaluation.

There is no segment, timestamp, span, localization, stance, target, mechanism, or rationale gold.

The teacher receives whole-video frames and full-video ASR/OCR with timestamps, segment IDs, span fields, and localization metadata stripped. It returns only relation and confidence. Uniformly sampled frames are model input, not segment annotations.

No hidden segment-gold assumption remains.

## Frontier Leverage

The method uses the MLLM in a modern and defensible role: privileged structured weak supervision that is removed at inference. It does not compete with the endpoint through a second head or teacher representation. Its narrow distinction from semantic KD, retrieval-geometry KD, modality intervention, and counter-reasoning is credible if the assignment-specific removability results pass.

For the eventual paper, the mechanism comparison should explicitly contrast:

- teacher output type;
- whether gold is shown to the teacher;
- optimized geometric object;
- query/key sharing;
- test-time teacher dependency.

This is a positioning clarification, not a request for another benchmark.

## Simplification Opportunities

1. Freeze the interval triplet directly rather than placing it in an A0 grid where it does not affect the continuous A0 objective.
2. Freeze the medoid ID and radius pair once; remove all dynamic path reselection.
3. Use one definition of effective sample size throughout.
4. State one primary A1 noise gate and reserve the second rate for A2/final monotonicity.
5. Keep effective rank, alternative mutual information, and native-head results diagnostic only.

## Modernization Opportunities

- Implement the class-masked full-bank log-sum-exp as a vectorized stable operation.
- Record bank drift, anchor-direction drift, support coverage, and CTE/base gradient ratios in one compact training diagnostic.
- Present the teacher relation as privileged ordinal weak supervision, not causal counterfactual identification.
- Include a concise mechanism-level prior-work table in the paper; no additional experimental axis is necessary.

## Drift Warning

Do not repair a failed transfer or support gate by adding:

- teacher absolute labels or rationales;
- timestamps, spans, salient segments, or segment pseudo-labels;
- teacher embeddings or teacher-selected memory keys;
- score concatenation, score fusion, reranking, or test-time views;
- adapters, auxiliary heads, EMA encoders, routers, or MoE;
- model/data/epoch/ensemble scaling;
- SSR/EDCM operations.

All such changes would drift from the endpoint and are rejected.

## Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 10/10 |
| Method Specificity | 25% | 8/10 |
| Contribution Quality | 25% | 9/10 |
| Frontier Leverage | 15% | 9/10 |
| Feasibility | 10% | 8/10 |
| Validation Focus | 5% | 9/10 |
| Venue Readiness | 5% | 8/10 |

\[
\textbf{OVERALL}
=0.15(10)+0.25(8)+0.25(9)+0.15(9)+0.10(8)+0.05(9)+0.05(8)
=\boxed{8.75/10}.
\]

No dimension is below 7.

## Remaining Action Items

1. Freeze medoid ID and adjacent radius pair before teacher calls; prohibit dynamic path reselection.
2. Freeze the anchor-direction drift rule and apply support masks identically across arms.
3. Make the three-fold probe fit/select/refit rotation exact.
4. Define video-clustered effective sample size.
5. Reconcile the A1 and final NOISE-rate protocol.
6. Fix the A1 label-only target source and A0 gradient-matching reference.
7. Freeze the bootstrap-to-p-value construction and historical-versus-paired comparator wording.

Only item 1 is a blocking mechanism issue; items 2–7 are required handoff specifications.

## Verdict

**REVISE**

The core route is now coherent, focused, supervision-compliant, and close to implementation-ready. Freeze the validated tangent rather than allowing its prototype/radii to change after A1, then resolve the remaining deterministic protocol definitions. No architectural or supervision expansion is warranted.

</details>
