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
