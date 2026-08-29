# Refinement Report

**Problem:** Meaningful and novel train-only MLLM integration with substantial final accuracy and macro-F1 gain.
**Initial approach:** CTE-RGCL whole-modality ordinal relations to full-bank tangent geometry.
**Date:** 2026-07-10
**Rounds:** 3 / 5
**Final score:** 9.20 / 10
**Final verdict:** READY
**Continuous reviewer agent:** `/root/cte_method_refine/cte_reviewer`

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Output Files

- Review summary: `refine-logs/cte/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/cte/FINAL_PROPOSAL.md`
- Score history: `refine-logs/cte/score-history.md`
- Grounding evidence: `refine-logs/cte/GROUNDING_EVIDENCE.md`

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9 | 6 | 6 | 7 | 6 | 8 | 5 | 6.65 | REVISE |
| 2 | 10 | 8 | 9 | 9 | 8 | 9 | 8 | 8.75 | REVISE |
| 3 | 10 | 9 | 9 | 9 | 9 | 10 | 9 | 9.20 | READY |

## Method Evolution Highlights

1. Replaced an unjustified withholding=tangent claim with a falsifiable withholding-informed transfer hypothesis.
2. Added class-specific two-radius orientation gates and froze the validated modality anchors/radii before teacher calls.
3. Made the full-bank loss, A0/A1 gates, controls and endpoint inference executable without adding a module.

## Pushback / Drift Log

| Round | Potential repair | Author response | Outcome |
|---:|---|---|---|
| 1–3 | Absolute teacher class/rationale to resolve orientation | Rejected; empirical class-conditional transfer under relation-only schema | Accepted; no drift |
| 1–3 | Segment/span evidence or weighting | Rejected because no such gold exists and endpoint is full-video classification | Accepted; supervision audit PASS |
| 2 | Dynamic prototype/radius adaptation | Rejected; fixed anchor identity and STOP on drift | Reviewer blocker closed |
| 1–3 | Extra encoder/adapter/router | Rejected as unnecessary complexity | READY with zero trainable additions |

## Remaining Weaknesses

No method-specification blocker remains. Empirical success is unproven: A0 may fail capacity/headroom, A1 may fail relation coverage/class transfer, A2 may fail control separation, and final +3/+3 may fail. READY must not be reported as target completion.

## Raw Reviewer Responses

### round-1-review.md

# CTE-RGCL Round 1 Review Checkpoint

<details><summary>Raw independent reviewer response</summary>

# CTE-RGCL Independent Senior Review — Round 1

## Executive Assessment

CTE-RGCL is unusually faithful to the actual endpoint: a train-only, label-blind MLLM signal changes the shared representation, while validation/test retain ordinary full-video train-memory kNN. It has one focused contribution, no new trainable module, no teacher key, no test-time artifact, and no segment-level supervision assumption.

However, the current formulation has two blocking semantic issues and several specification/control defects:

1. The teacher judges complete typed withholding, while the student optimizes a small tangent toward a marginal modality prototype. Neither marginal support nor teacher agreement establishes that these interventions have the same ordinal effect.
2. `weaken/reverse → negative gold-class margin tangent` is not identified by the allowed teacher record. The relation is relative to the MLLM’s latent full-video interpretation, whose class orientation is never observed and may disagree with the video label.
3. The interval cost is bounded, but the proposal incorrectly infers that its gradient is bounded.
4. Some CTE-1 statistics and controls are not yet executable as written, especially the teacher-dependent “relation-free” mask and severely over-stratified SHUFFLE.

These are repairable without changing the route or adding modules, but they must be repaired before implementation.

## Blocking Mechanism Review

### 1. Teacher–student intervention mismatch

The teacher receives a complete semantic intervention: one entire modality is explicitly withheld. The student does something materially different:

\[
p_i^m\longrightarrow \operatorname{norm}((1-a)p_i^m+a c_m),\quad a\le 0.3.
\]

This is not a local approximation to modality absence without an additional monotonicity assumption. Moving toward a global prototype may inject average modality content, alter cross-modal congruence, or move in a direction unrelated to withholding. The current 5-NN audit checks only marginal modality support. It does not show that the joint pair `(perturbed modality, unchanged modality)` or the fused embedding remains supported.

CTE-1 is the right place to test this assumption, but “conditional information” and a held-out gradient effect are not yet a direct test that teacher ordinals transfer to the selected tangent.

Required fix:

- Name the mechanism precisely as a **withholding-informed prototype tangent**, not the tangent of withholding itself.
- Freeze one prototype estimator; remove “spherical medoid/normalized robust mean” ambiguity.
- Audit support per example in the joint projected-modality space and fused space, not only the perturbed modality marginal. Unsupported examples must be inactive.
- In CTE-1, preregister an explicit class-conditional ordinal-transfer test at every supported `a`: the gold-oriented tangent target must exhibit the frozen order `preserve ≈ 0 > weaken > reverse`, with a positive held-out monotonic association and positive lower confidence bound separately for `y=0` and `y=1`.
- Require the sign/order to be stable across at least two adjacent supported `a` values. If not, stop. Do not repair it with dataset-specific prompts or path choices.

This remains video-label-only and adds no model component.

### 2. Unidentified orientation of `weaken` and `reverse`

The allowed teacher output does not reveal the MLLM’s latent full-video class interpretation. Suppose the MLLM incorrectly interprets a benign video as hateful and reports that visual withholding “weakens” that interpretation. CTE nevertheless forces the perturbation to reduce the gold benign-class margin. The relation alone does not justify that sign.

Prompt agreement and confidence do not solve this; four calls can consistently share the same incorrect latent orientation. Nor should the proposal imply that the teacher relation is causal ground truth.

Required fix:

- Define the prompt semantics exactly:

  - `preserve`: the neutralized whole-video condition supports the same latent moderation interpretation with comparable support;
  - `weaken`: it retains that interpretation but reduces its support;
  - `reverse`: the dominant latent moderation interpretation changes;
  - `unclear`: none can be asserted reliably.

  The emitted schema remains relation plus confidence only.
- State explicitly that orientation to the gold-class margin is an empirical weak-label hypothesis, not logically identified from the teacher record.
- Make the class-conditional CTE-1 ordinal-transfer test above a hard gate.
- Report relation coverage, order, and transfer separately for both video labels. A pooled association cannot pass.
- Do not filter individual records using teacher-predicted classes, rationales, segments, or timestamps; those would violate the immutable interface.

This issue is the central reason the current method is not yet venue-ready.

### 3. Full-bank margin and epoch-bank semantics

The margin is a valid differentiable all-bank surrogate:

\[
M_i(z)=\tau\operatorname{LSE}_{y_j=y_i,j\ne i}(s/\tau)
-\tau\operatorname{LSE}_{y_j\ne y_i}(s/\tau).
\]

It genuinely reaches beyond the old top-64. However, “exact current full-bank” is overstated. After the first optimizer update of an epoch, the detached keys are stale. The computation remains exact over every key in the epoch-start bank, but not over the current encoder geometry.

Required specification:

- Call it the **exact epoch-refreshed full bank**.
- Fix `s(z,k)=z^\top k`, `tau`, stable log-sum-exp implementation, self-ID exclusion, and minimum same-class/other-class bank counts.
- State whether query forwards use train or eval mode and ensure the bank/query discrepancy is not caused by dropout or batch-statistics.
- Ensure every training video is used as a CTE query each epoch so shared parameters receive query-side supervision across the whole bank.
- Log within-epoch encoder/bank drift and preregister a stop threshold. Keep epoch refresh and one shared encoder; do not introduce EMA or teacher keys.
- Rebuild both prototypes and the full bank after checkpoint loading and at the same frozen refresh boundary.

The shared query/key design is otherwise sound.

### 4. “Bounded loss” is not “bounded gradient”

The interval cost is indeed bounded. For \(T\in[-1,1]\) and \(I\subseteq[-1,1]\), squared distance divided by four lies in `[0,1]`. But bounded function values do not imply bounded gradients. The factor

\[
1/(a\,s_t+\epsilon)
\]

can be arbitrarily large when the MAD is small, and normalization can amplify gradients near a small pre-normalization norm.

Required fix:

- Claim only bounded **cost/influence weight**, not bounded gradient.
- Define \(\hat s_t=\max(s_t,s_{\min})\), with `s_min` frozen before teacher results.
- Specify normalization epsilons and log the minimum pre-normalization norm.
- Retain global gradient clipping and report CTE/base gradient-norm ratios.
- Give actual frozen values or a leakage-free selection rule for `tau`, `d0`, `dw`, `dr`, `lambda_CTE`, `s_min`, and `a`.
- Define interval distance explicitly as `max(lower−T, 0, T−upper)`.

### 5. CTE-0 is a valid empirical screen, not an upper bound

This part is conceptually correct. CTE-0 tests whether this exact local response family and training implementation can move enough full-bank geometry under a bounded supervised target. It does not bound what other MLLM integrations could achieve.

The remaining defect is underspecification:

- Define precisely which probe supplies each modality target, its feature input, its true-class margin, target clipping, target interval width, and inner-cross-fitting.
- Freeze the nested selection rule for `lambda` and all tangent thresholds. No outer-fold prediction may choose them.
- Compare against a paired strongest non-MLLM implementation with identical optimizer steps, checkpoint rule, and bank refresh—not an ambiguously “frozen-geometry” weaker baseline.
- If CTE-0 passes, its label-only result must enter the moving non-MLLM comparator exactly as stated.

### 6. CTE-1 budget is compliant but its statistics need an executable definition

The proposal respects the maximum of 128 strict train videos per dataset. The teacher sees no labels or strata, and the maximum 2,048 calls is correctly derived.

The current phrases “held-out conditional information,” “stratified permutation lower bound,” and “fixed norm-matched CTE gradient step” are too ambiguous for a gate.

Required fix:

- Freeze the ≤128 video IDs before any call, using only train-video label, OOF margin, and OOF error strata.
- Define one primary transfer statistic, estimator, fold construction, residualization variables, permutation unit, number of permutations, bootstrap unit, and confidence level.
- With only 128 videos, do not form a Cartesian permutation stratum from all listed covariates. Use cross-fitted residualization or a frozen low-dimensional stratification.
- Define the pilot update exactly: parameter subset, learning rate, number of steps, examples used for the update, bank rebuild timing, and held-out metric.
- Teacher extraction beyond the pilot remains locked unless both datasets pass.

### 7. Control construction contains two important confounds

The control family is broadly appropriate, but two controls are invalid or infeasible as written.

**Relation-free multiview:** If “active items” or weights come from teacher consensus/confidence, the control is not relation-free. Teacher missingness itself can carry video-specific information.

Fix: use every support-valid view with a fixed uniform weight, then globally scale the auxiliary update to match the clean CTE gradient norm. Alternatively use a frozen teacher-independent mask. It must not reuse teacher activity, confidence, or per-video assignment.

**SHUFFLE:** Exact derangement within fold × label × margin decile × energy decile × missingness will likely create singleton cells, especially at pilot scale. Declaring the causal claim dead after discovering this is avoidable protocol design failure.

Fix: audit cell sizes before teacher extraction and freeze a feasible shuffle such as fold × video label × coarse baseline-margin bin, deranging the indivisible two-modality record. Adjust energy/difficulty through the preregistered conditional statistic rather than exact Cartesian matching. No post-outcome relaxation is allowed.

Also make the label-only control an exact same-loss comparator, rather than only a conceptually related probe result. REMOVE, random, heuristic, and relation-free controls must use the same optimizer-step and checkpoint budget. Keep NOISE confidence and coverage fixed.

## Supervision Audit

**Video-label-only requirement: PASS, subject to implementation wording fixes.**

Gold information currently used is limited to:

- the parent-video binary label in the true-class bank margin;
- train-only probe construction;
- train-video pilot stratification and controls;
- video-level endpoint evaluation.

No segment, timestamp, span, localization, stance, target, mechanism, or rationale gold is assumed. Uniformly sampled frames and full-video ASR/OCR are inputs, not segment annotations.

Implementation must additionally ensure:

- ASR/OCR timestamps, segment IDs, span scores, and localization metadata are stripped from the teacher bundle;
- teacher artifacts are keyed only by train-video ID and modality condition;
- no per-segment relation, confidence, weighting, target, loss, or endpoint is introduced;
- no validation/test record is sent to the teacher.

Any suggestion to add timestamp labels, salient-segment pseudo-labels, span supervision, or segment-weighted retrieval is **DRIFT and rejected**.

## Statistical Endpoint Audit

The final success condition is properly demanding, but the statistical test needs to be made exact:

- Clarify that “all three paired-seed deltas positive” means all 12 dataset × metric × seed deltas.
- State that the `+0.030` effect is measured against the maximum of the historical strongest point and paired same-seed strongest non-MLLM mean.
- Define the hierarchical bootstrap unit: paired method predictions, resampling videos within each seed and seeds at the outer level.
- Predefine how macro-F1 is recomputed in every bootstrap replicate.
- Define the four Holm-adjusted tests and their null hypotheses.
- Attribution must include positive, uncertainty-qualified FULL-minus-REMOVE and FULL-minus-SHUFFLE effects on both metrics and both datasets. Merely observing a lower point estimate after shuffle is insufficient.
- Test data must remain locked until seed-0 dev gates pass.

## Simplification Opportunities

1. Choose one prototype estimator and one joint-support audit; remove alternative wording.
2. Use one primary CTE-1 ordinal-transfer statistic. Treat conditional-information variants and effective rank as diagnostics.
3. Collapse control implementation into the same CTE loss with only the relation assignment changed: clean, label-only, heuristic, random, shuffle, noise, and teacher-independent uniform preserve.
4. Use one frozen shared hyperparameter rule across both datasets. Dataset-specific rescue tuning would weaken the contribution.
5. Keep neighbour churn as a mechanism diagnostic rather than presenting it as a second contribution.
6. Do not add an adapter, auxiliary classifier, second encoder, teacher embedding, or additional benchmark to repair these issues.

## Modernization Opportunities

- Add a concise mechanism-comparison table against train-only semantic KD, retrieval-geometry KD, modality intervention/gradient-control work, and counter-reasoning fusion. Compare teacher output, optimized object, query/key treatment, and test-time dependency.
- Use vectorized, numerically stable class-masked all-bank log-sum-exp and report its measured memory/runtime cost.
- Report class-conditional teacher calibration and ordinal transfer without learning a calibration module.
- Treat CTE explicitly as privileged weak-relation supervision whose value is established by removability and assignment-sensitive controls, not as causal counterfactual identification.

## Drift Warning

The proposal presently remains on target. The following repairs would constitute unacceptable drift:

- teacher absolute hate labels, rationales, scores, targets, or mechanisms;
- timestamp/span/segment outputs or any segment-level gold/pseudo-gold endpoint;
- segment selection or weighting;
- teacher-selected keys, teacher embeddings, or MLLM features concatenated into the representation;
- score fusion, reranking, vetoes, routers, MoE, or test-time MLLM/views;
- model/data/epoch/ensemble scaling;
- reuse or retuning of SSR/EDCM actions.

The required fixes above need only clearer semantics, stricter gates, and corrected controls.

## Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 9/10 |
| Method Specificity | 25% | 6/10 |
| Contribution Quality | 25% | 6/10 |
| Frontier Leverage | 15% | 7/10 |
| Feasibility | 10% | 6/10 |
| Validation Focus | 5% | 8/10 |
| Venue Readiness | 5% | 5/10 |

\[
\textbf{OVERALL}
=0.15(9)+0.25(6)+0.25(6)+0.15(7)+0.10(6)+0.05(8)+0.05(5)
=\boxed{6.65/10}.
\]

### Required remediation for every score below 7

**Method Specificity — 6/10**

- Weakness: teacher-relation orientation, teacher/student intervention transfer, CTE-0 targets, CTE-1 statistics, bank staleness, and hyperparameter values remain insufficiently specified.
- Concrete fix: implement the exact semantic definitions, class-conditional ordinal-transfer gate, joint-support audit, fixed scale floor, explicit epoch-bank semantics, and fully preregistered estimators/hyperparameters described above.
- Priority: **CRITICAL**.

**Contribution Quality — 6/10**

- Weakness: the narrow contribution is promising, but its essential mapping from label-blind relation to signed gold-class tangent is not currently identified, and the withholding-to-prototype claim is stronger than the mechanism warrants.
- Concrete fix: recast the contribution as empirically validated withholding-informed ordinal tangent supervision; require transfer in both classes and across supported tangent radii before claiming teacher value.
- Priority: **CRITICAL**.

**Feasibility — 6/10**

- Weakness: the over-stratified shuffle may be impossible, the ≤128 pilot analysis is underdefined, and exact all-bank compute/runtime estimates are not grounded in an implementation audit.
- Concrete fix: freeze a feasible coarse derangement before teacher extraction, specify one adequately powered pilot statistic, and benchmark vectorized full-bank loss through SLURM before committing teacher calls.
- Priority: **IMPORTANT**.

**Venue Readiness — 5/10**

- Weakness: there is no empirical evidence yet, and the current specification would not support a defensible causal/removability claim even if headline accuracy improved.
- Concrete fix: resolve all critical mechanism issues, pass CTE-0 and the corrected ≤128-per-dataset CTE-1 gates, then demonstrate seed-0 control separation before unlocking final test evaluation.
- Priority: **CRITICAL**.

## Blocking Action Items

1. Resolve and formalize the relation-orientation problem without changing the allowed teacher schema.
2. Add joint-support and multi-radius ordinal-transfer tests for typed withholding versus prototype shrink.
3. Correct the bounded-gradient claim and freeze all numerical definitions.
4. Specify epoch-stale full-bank behavior precisely.
5. Make CTE-0 target construction and nested selection executable.
6. Make CTE-1’s primary statistic and held-out update exact while retaining the ≤128 cap.
7. Remove teacher-derived masks from the relation-free control.
8. Replace infeasible Cartesian SHUFFLE strata with a preregistered feasible derangement.
9. Freeze the final paired-seed bootstrap/Holm protocol.
10. Preserve the explicit no-segment-gold audit in every implementation document.

## Verdict

**REVISE**

The route should not be abandoned, expanded, or replaced. It is focused, endpoint-faithful, and potentially novel, but the present semantic mapping and intervention transfer are not yet sound enough to justify implementation or a top-venue claim.

</details>

### round-2-review.md

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

### round-3-review.md

# CTE-RGCL Round 3 Review Checkpoint

<details><summary>Raw independent reviewer response</summary>

# CTE-RGCL Independent Senior Review — Round 3

## Executive Assessment

All Round 2 blocking issues are substantively closed. The proposal is now implementation-ready as a focused, falsifiable train-only MLLM integration:

- the validated tangent has a frozen identity;
- teacher-to-gold orientation is explicitly an empirical hypothesis with class-specific gates;
- bank and numerical semantics are executable;
- A0 remains correctly bounded as a cost/capacity screen;
- A1 respects the absolute 128-video-per-dataset cap;
- assignment controls isolate teacher-specific information;
- endpoint statistics distinguish the historical effect threshold from paired inference;
- test inference remains ordinary full-video kNN.

No remaining issue requires a new module, broader experiment, additional supervision, or route change.

## Anchor Audit

**PRESERVED.**

The binding goal remains unchanged:

- meaningful and novel MLLM integration into the learned shared representation;
- train-only teacher use;
- ordinary full-video train-memory kNN at validation/test;
- `+0.030` accuracy and macro-F1 on both MHC-EN and MHC-ZH;
- paired seeds 0/1/2;
- positive per-seed effects;
- uncertainty and Holm correction;
- REMOVE/SHUFFLE/multiview/label-only/heuristic/random/NOISE attribution;
- comparison against the moving strongest non-MLLM bar.

The method does not substitute a native head, teacher key, reranked endpoint, or weaker localization result for this goal.

## Drift Audit

**NO DRIFT.**

The revision adds only frozen identities, formulas, stop rules, and leakage controls. It does not add:

- teacher absolute labels, rationales, scores, or hidden semantic fields;
- segment, timestamp, span, localization, stance, target, or mechanism supervision;
- teacher embeddings or teacher-selected keys;
- adapters, auxiliary heads, second/EMA encoders, routers, or MoE;
- score fusion, reranking, or test-time neutral views;
- scaling or SSR/EDCM reuse.

## Dominant Contribution and Simplicity

The contribution is now both narrow and technically meaningful:

> Confidence-bearing, label-blind whole-modality withholding relations are empirically transferred to a fixed-anchor supported tangent of the epoch-refreshed full-bank true-class margin, changing shared query/key geometry while disappearing entirely at inference.

It remains one contribution:

- one shared encoder;
- one parameter-free interval loss;
- one frozen anchor per modality;
- one frozen radius pair;
- one train-only relation cache;
- zero new trainable components;
- unchanged kNN inference.

A0, A1, support audits, and controls are falsification machinery rather than competing method components.

## Fixed Tangent Identity Audit

**PASS.**

The Round 2 blocker is closed:

- anchor identity is selected at the A0 checkpoint and hashed before teacher calls;
- the adjacent `(a1,a2)` pair is frozen;
- refresh only re-encodes the same anchor ID;
- support masks are recomputed teacher-independently;
- anchor or radius replacement is prohibited;
- all arms share the same identity, radii, and support rules;
- support and direction drift can only trigger STOP.

The median and lower-tail direction-cosine thresholds prevent a nominally fixed anchor from evolving into an unvalidated tangent.

Implementation should make the modality indexing explicit as `anchor_id^V` and `anchor_id^L` if separate modality medoids are intended. This is a notation clarification, not a mechanism blocker.

## Full-Bank and Loss Audit

**PASS.**

The proposal correctly specifies:

- one shared query/key encoder;
- every epoch-start full-video train key;
- detached keys;
- self-ID exclusion;
- same-class and opposite-class availability;
- eval-mode stochastic semantics for bank and tangent queries;
- complete query coverage once per epoch;
- exact all-bank log-sum-exp margin;
- explicit acknowledgement of within-epoch staleness;
- frozen bank-drift thresholds and common refresh fallback;
- fixed intervals;
- bounded cost and weight, without a false bounded-gradient claim;
- scale/norm floors, clipping, non-finite stopping, and gradient-ratio logging.

The expression `max(MAD,.sMin)` should be implemented as `max(MAD,sMin)`; this is an evident typographical correction.

## A0 Audit

**PASS.**

A0 is correctly described as an empirical screen of this exact tangent/loss/bank action family. It is explicitly:

- not a theoretical upper bound;
- not MLLM evidence;
- not a claim about other representation-learning routes.

The fit-A/select-B/refit-A∪B/predict-C rotation prevents target leakage. Targets are strict OOF, cached before pilot selection, and reused unchanged in A1. Hyperparameters are selected without outer, teacher, dev, or test outcomes. A successful label-only method correctly raises the moving comparator.

## A1 Cap and Orientation Audit

**PASS.**

The absolute cap remains at most 128 strict train videos per dataset and 2,048 total teacher calls. No broader teacher extraction is allowed unless both datasets pass.

The relation-orientation problem is handled appropriately as empirical weak supervision:

- `y=0` and `y=1` are tested separately;
- both frozen radii must pass;
- all three active ordinal levels require sufficient video-clustered effective sample size;
- preserve must remain near zero;
- preserve/weaken/reverse means must be ordered;
- class-specific ordinal slopes require positive lower bounds;
- pooled evidence cannot rescue a failed class;
- the MLLM never emits or exposes an absolute class.

This is a strong and valid gate under the immutable teacher schema.

The wording should state explicitly that the preserve and ordinal-mean conditions use reliability-weighted cell means, consistent with the regression. That is an implementation detail, not a blocker.

## Control Audit

**PASS.**

The controls now isolate the intended factors:

- **REMOVE** isolates the complete auxiliary mechanism.
- **Multiview** is assignment-free and teacher-mask-free while conservatively matching aggregate optimization strength.
- **Label-only** uses the same loss family and strict OOF targets.
- **Energy** tests a cheap intervention heuristic.
- **Random** retains distributions and optimization strength without assignment information.
- **SHUFFLE** preserves whole two-modality records and removes video-specific assignment within pre-audited feasible cells.
- **NOISE** preserves coverage and confidence while testing monotone degradation at frozen rates.

All controls share anchor identity, radii, support masks, encoder, bank refresh, steps, and checkpoint budget. This is sufficient to separate assignment-specific teacher value from generic multiview regularization, labels, missingness, relation histograms, and extra optimization.

## Statistical Audit

**PASS, with one implementation interpretation to freeze.**

The proposal properly separates:

1. the deterministic `+0.030` effect gate against the maximum of the historical scalar and paired non-MLLM mean; and
2. paired inference against same-seed methods with sample-level predictions.

The centered-null bootstrap p-value, four Holm-adjusted dataset-by-metric tests, percentile lower bounds, per-seed sign requirements, and FULL-minus-REMOVE/SHUFFLE uncertainty are all specified.

Because the same test videos occur across seeds, each bootstrap replicate should draw one shared paired video-ID sample per dataset and apply it to every resampled seed, rather than drawing unrelated video samples independently inside each seed. This preserves same-video dependence across seeds and is the natural interpretation of “paired videos.” Freeze this implementation in the experiment handoff.

## Supervision Audit

**PASS: VIDEO-LABEL-ONLY.**

The only gold is the parent-video binary label. It is used for:

- full-bank key labels and true-class margin orientation;
- strict OOF A0 probe targets;
- train-only strata;
- class-conditional A1 analysis;
- video-level endpoint metrics.

There is no segment, timestamp, span, localization, stance, target, mechanism, or rationale gold anywhere.

The teacher input is whole-video evidence with timestamps, segment IDs, spans, and localization metadata stripped. Its only output is relation plus confidence. Uniformly sampled frames and full-video ASR/OCR are input evidence, not segment annotations.

No hidden segment-gold assumption remains.

## Frontier Leverage

The proposal uses the MLLM as privileged structured weak supervision rather than an inference-time classifier, free-text feature generator, or memory-key provider. The optimized object is the exact epoch-refreshed full-bank retrieval margin that directly underlies the final endpoint.

The defensible novelty remains narrow but strong:

- label-blind ordinal whole-modality relation;
- explicit empirical class-orientation gate;
- fixed supported prototype tangent;
- shared full-bank query/key geometry;
- complete teacher removal at inference.

If the final removability and shuffle/noise results pass, this is a credible top-venue mechanism-level contribution rather than a generic application of semantic KD.

## Feasibility Audit

**PASS.**

The route has bounded staged cost:

- A0 precedes all teacher expense;
- A1 is hard-capped;
- A2 and final runs unlock only after falsification gates;
- all-bank computation is vectorizable at the dataset scale;
- no new backbone or inference component is introduced;
- SLURM and environment constraints are acknowledged.

The stringent reverse-cell effective-sample requirement may cause A1 to stop, but this is an intentional scientific gate, not an implementation defect.

## Simplification Opportunities

1. Use explicit `anchor_id^V` and `anchor_id^L` notation.
2. Define preserve and ordinal cell tests once as reliability-weighted means.
3. Record support coverage, anchor-direction drift, bank drift, and gradient ratio in one diagnostic artifact.
4. Keep all alternative dependence measures and native-head results diagnostic only.
5. Do not add any component if A0 or A1 fails.

## Modernization Opportunities

- Use a stable vectorized class-masked log-sum-exp implementation.
- Hash anchor IDs, radii, support-rule parameters, teacher prompts, relation cache, and control permutations in one manifest.
- Present CTE as privileged ordinal weak supervision with empirical removability, not causal counterfactual identification.
- Use a concise prior-work mechanism table rather than adding another benchmark.

## Remaining Blockers

**None.**

The following are final handoff clarifications, not blockers:

1. Make separate modality anchor IDs explicit.
2. Correct the `sMin` typography.
3. State that ordinal cell summaries are reliability-weighted.
4. Use one shared paired video-ID resample across seeds.
5. Run the promised SLURM microbenchmark before estimating final wall-clock cost.

## Drift Warning

If a frozen gate fails, do not rescue the route with teacher labels/rationales, segments, alternate anchors, adaptive radii, prompt specialization, teacher keys, extra heads, reranking, scaling, or SSR/EDCM operations. Such changes would invalidate the reviewed contribution.

## Scores

| Dimension | Weight | Score |
|---|---:|---:|
| Problem Fidelity | 15% | 10/10 |
| Method Specificity | 25% | 9/10 |
| Contribution Quality | 25% | 9/10 |
| Frontier Leverage | 15% | 9/10 |
| Feasibility | 10% | 9/10 |
| Validation Focus | 5% | 10/10 |
| Venue Readiness | 5% | 9/10 |

\[
\textbf{OVERALL}
=0.15(10)+0.25(9)+0.25(9)+0.15(9)+0.10(9)+0.05(10)+0.05(9)
=\boxed{9.20/10}.
\]

## Verdict

**READY**

The proposal is ready for implementation under its frozen staged gates. READY denotes method/specification readiness, not evidence that the final `+0.030/+0.030` endpoint has already been achieved. The research goal remains active until A0, A1, A2, and the final two-dataset three-seed full-video kNN requirements actually pass.

</details>

## Next Steps

1. Hand the frozen proposal to experiment planning/implementation.
2. First run a vectorized full-bank SLURM microbenchmark; then implement and independently audit A0 only.
3. Do not call the teacher unless A0 passes both datasets; cap A1 at 128 strict train videos per dataset.
4. Keep the global target active until A0, A1, A2 and final two-dataset three-seed +3/+3 gates are proven.
