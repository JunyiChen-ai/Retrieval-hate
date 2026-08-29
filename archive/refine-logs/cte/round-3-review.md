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
