# Round 3 Review

**Reviewer agent:** `/root/edcm_pivot_refine/edcm_reviewer`  
**Overall:** 9.11 / 10  
**Verdict:** READY  
**Anchor:** preserved; drift warning NONE  
**No-segment-gold audit:** PASS  
**Meaning:** ready for staged experiment handoff only; the +3/+3 target remains unproven.

## Parsed Review

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.8 |
| Method Specificity | 9.2 |
| Contribution Quality | 8.8 |
| Frontier Leverage | 9.0 |
| Feasibility | 8.8 |
| Validation Focus | 9.6 |
| Venue Readiness | 8.6 |
| **Overall** | **9.11** |

### Final non-blocking freezes

1. Use a finite predeclared `tau_proxy` log grid, tolerance and tie rule.
2. Use “teacher-semantic-free strength-matched proxy” consistently.
3. Hash held-out query-signature rows in every fold manifest.
4. Freeze decoded-frame/SSIM/zero-IQR proxy edge cases.
5. Operationalize calibrated-corruption survival before A2.

## Full Raw Reviewer Response (verbatim)

<details>
<summary>Round 3 raw response</summary>

# Round 3 Senior ML Review — EDCM-RGCL

## Overall assessment

The bounded Round 3 changes close the two substantive Round 2 blockers.

The proxy is now a fair teacher-**semantic**-free control: it shares coverage, reliability, candidate lists, loss, workload and approximately matched gradient strength, while using neither per-video MLLM alignment nor any downstream performance metric. The OOF contract now excludes held-out/self IDs, full-data embeddings and fold-external scaling statistics. The teacher-active test is mathematically specified, non-circular and implementable using only video-level labels.

EDCM remains one focused contribution:

> Privileged MLLM coalition semantics define a broadly active conditional-neighborhood gradient that is internalized into the exact full-video geometry consumed by final kNN.

There is no method bloat, no segment supervision, and no unresolved mechanism or novelty flaw that should prevent staged execution.

**Verdict: READY**

This means the proposal is ready for experiment handoff. It does not mean the immutable +3/+3 research target has been achieved; A0–A3 remain binding.

## Scores

| Dimension | Score | Weighted contribution |
|---|---:|---:|
| 1. Problem Fidelity | 9.8 | 1.470 |
| 2. Method Specificity | 9.2 | 2.300 |
| 3. Contribution Quality | 8.8 | 2.200 |
| 4. Frontier Leverage | 9.0 | 1.350 |
| 5. Feasibility | 8.8 | 0.880 |
| 6. Validation Focus | 9.6 | 0.480 |
| 7. Venue Readiness | 8.6 | 0.430 |
| **OVERALL** | **9.11 / 10** | |

No dimension is below 7.

## 1. Problem Fidelity — 9.8/10

The immutable anchor is preserved almost exactly:

- final accuracy and macro-F1 remain the endpoint;
- the target remains +0.030/+0.030 on two datasets with seeds 0/1/2;
- the MLLM is removable and train-only;
- the actual final kNN readout must improve;
- splits, preprocessing, labels, epochs, checkpointing, retrieval and voting remain fixed;
- no test-time MLLM, segment objective, teacher key, generated content or native-head substitution is introduced.

The method directly responds to SSR’s sparse-support failure through all-video list gradients rather than renaming another sparse relation mechanism.

## 2. Method Specificity — 9.2/10

The loss, gradients, bank behavior and OOF calculations are sufficiently concrete for implementation.

The teacher-active test now distinguishes:

- non-uniform list weights through TV;
- non-negligible teacher-specific influence through \(R\);
- useful direction relative to Label-only through \(\Delta D\).

Using the same frozen 8+8 list, temperature and normalized query-step makes the directional comparison well-defined. The exact comparator bank object and refresh schedule close the train/final-geometry contract.

One minor implementation note remains: because median \(R\) need not be globally monotonic in `tau_proxy`, use a predeclared finite log grid as the authoritative search. Apply bisection only inside a bracket whose monotonicity has been verified. Freeze grid bounds, resolution, match tolerance and tie rule before extraction results are inspected.

## 3. Contribution Quality — 8.8/10

The contribution is now sharply identifiable against:

- exact RGCL;
- generic Label-only ListNCA;
- equally strong low-level modality/content conditioning;
- semantically misassigned MLLM signatures;
- calibrated teacher corruption.

This is enough to distinguish “MLLM semantics matter” from “any listwise loss,” “any modality metadata,” or “any non-uniform gradient” matters.

The remaining novelty risk is empirical rather than conceptual: if the proxy matches full EDCM, shuffle retains the gain, or final kNN does not improve, the contribution fails by design. That is appropriate falsifiability, not a proposal weakness.

## 4. Frontier Leverage — 9.0/10

The MLLM is used naturally and economically:

- relative multimodal judgment rather than weak absolute classification;
- deterministic modality interventions rather than generated counterfactual content;
- privileged train-only supervision rather than test-time reasoning;
- no gain sought through teacher scaling.

Retaining ordinal distributions and reliability is modern enough. No learned uncertainty model, rationale encoder, router or second teacher is warranted.

## 5. Feasibility — 8.8/10

The implementation path is credible:

- zero new trainable modules;
- existing detached bank and refresh schedule;
- query-only gradients against frozen keys;
- explicit self exclusion;
- deterministic missing-signal fallback;
- fold-local OOF artifacts;
- staged SLURM cost control.

The remaining risks—teacher acceptance coverage, gradient inertness, OCR/frame failures and throughput—are precisely what A0/A1 and the extraction smoke are intended to test.

## 6. Validation Focus — 9.6/10

The proposal is strongly claim-driven:

- A0 tests whether paying for the teacher is reasonable.
- A1 tests whether MLLM influence is broad, non-negligible and better directed than Label-only.
- The proxy separates semantic content from perturbation strength.
- A2 tests actual development kNN repair and removability.
- A3 alone evaluates the immutable final endpoint.

The proposal correctly prevents a favorable cache statistic, native-head gain or isolated seed result from being mistaken for success.

## 7. Venue Readiness — 8.6/10

The narrow retrieval-specific claim is credible for a top venue if the registered experiments pass. It can be stated without inflated first-ever claims:

> Relative label-blind MLLM coalition judgments define a privileged conditional-neighborhood measure whose teacher-specific gradient is internalized into the ordinary full-video kNN geometry and then discarded.

Final venue strength still depends on the substantial two-dataset result. The method proposal itself is ready to test that claim.

## Strength-matched proxy audit

### Fairness

**Pass.**

The proxy now matches the relevant experimental factors:

- exact accepted query mask;
- exact \(\rho_i\);
- exact 8+8 lists;
- exact loss and auxiliary coefficient;
- identical training workload;
- approximately matched median teacher-specific gradient ratio \(R\).

The low-level proxy includes actual visual variation rather than nearly constant frame availability alone.

### Leakage

**Pass.**

Scaling is fitted only on each fold’s bank. Temperature calibration uses pooled OOF training artifacts and no validation/test result. No accuracy, macro-F1, \(\Delta D\), development result or per-video MLLM alignment is optimized.

Video-level labels enter \(R\) through the supervised training lists and loss, which is permitted. They are not exposed to the MLLM or used to select a performance-maximizing proxy.

### Interpretation

The control is not fully teacher-independent because it reuses the MLLM acceptance mask, \(\rho_i\), and aggregate target strength. That is intentional and scientifically useful: it is a **teacher-semantic-free strength-matched control**, designed to isolate sample-specific semantic alignment.

Use this wording consistently instead of simply “teacher-free baseline.”

Matching pooled median \(R\) need not make every per-dataset distribution identical. Reporting distance, TV, \(R\) and activity distributions per dataset is the correct solution; dataset-specific temperature tuning would be less fair.

## Teacher-active dense-support audit

**Pass.**

The gate is mathematically coherent:

\[
\Delta g_i
=
\nabla L_{\mathrm{EDCM}}
-
\nabla L_{\mathrm{uniform}}
\]

isolates the teacher-dependent gradient on identical geometry and lists. TV prevents a nominally different but practically uniform kernel from passing, while \(R\) prevents negligible differences from being called dense.

The normalized directional test

\[
\Delta D_i
=
\nabla\mu_i^\top v_i^E
-
\nabla\mu_i^\top v_i^U
\]

asks whether EDCM’s unit descent direction improves the supervised neighborhood margin more than Label-only. It does not award EDCM for raw gradient scale.

The gate is non-circular because:

- signatures and prompts are frozen before results;
- the teacher never receives the video label;
- thresholds are route-wide and pre-registered;
- OOF labels are used only as permitted training supervision;
- no validation/test metric selects the method.

It is still a first-order mechanism preflight, not proof of final kNN gain. A2 correctly supplies that proof.

## OOF contract audit

**Pass.**

The revision closes the earlier leakage risks:

- fold model trained only on \(T\setminus F\);
- held-out queries and bank encoded by the same fold model;
- bank IDs disjoint from query IDs;
- self-neighbors impossible;
- candidate signatures only from the fold bank;
- query/bank/list/scaling artifacts frozen before gradients;
- no full-data embedding or statistic substituted;
- hashes bind checkpoints, IDs, embeddings, signatures and lists.

The held-out query signature should also appear explicitly in each fold hash manifest, since it is required to form \(q_i^\pm\). This is a minor bookkeeping point.

## Does \(L_{\mathrm{EDCM}}\) directly shape final kNN geometry?

**Yes.**

The auxiliary loss acts on the ordinary full-video embedding and the comparator’s actual detached full-video bank. The shared encoder internalizes the update; later bank refreshes and final memory construction use that encoder. No signature is concatenated, no key is replaced, and no score is injected into final retrieval.

Reusing the comparator bank object, refresh/reindex schedule and final vote is decisive. This is direct representation-geometry training rather than native-head absorption or post-hoc memory weighting.

## A0 audit

**Pass.**

A0 is non-vacuous because reachability is constrained to the frozen top-64 locality and at most two replacements. It uses only video-level OOF labels.

Its interpretation is now correct: it is a conservative cost screen, not an upper bound on learned embeddings and not evidence for MLLM usefulness.

## No-Segment-Gold Audit

**PASS.**

- Binary video label is the only gold.
- Uniform frames are whole-video input samples, not annotated segments.
- Transcript is a raw input modality without temporal labels.
- OCR is deterministic input extraction.
- MLLM outputs are whole-video coalition preservation distributions.
- No timestamp, temporal span, segment class, localization target or segment metric exists.
- TV, \(R\), \(Delta D\), neighborhood labels and kNN metrics are all video-level.
- Validation/test load no teacher-side artifact.

“Dense” refers only to broad training influence. It must never be described as dense human annotation.

## Degeneration audit

- Feature/rationale concatenation: **no**
- Static segment weighting: **no**
- Segment-weighted memory: **no**
- Teacher-selected key: **no**
- Test-time MLLM or reranking: **no**
- Sparse relation-edge repetition: **no**
- Generic Label-only ListNCA: **binding control**
- Low-level modality/content effect: **strength-matched control**
- Native-head redistribution: **binding failure condition**

## Simplicity / overbuilding assessment

The method is not overbuilt. It still contains:

- one frozen teacher cache;
- one parameter-free signature;
- one listwise auxiliary loss;
- zero trainable modules;
- no inference addition.

The numerous controls are justified by the strength of the MLLM-specific causal and removability claim. They should remain controls, not be presented as separate contributions.

## Simplification Opportunities

1. Make the finite frozen `tau_proxy` log grid authoritative; avoid an unnecessarily elaborate optimizer for one nuisance control parameter.
2. Consolidate all fold hashes, query signatures, scaling statistics, lists and gradient diagnostics into one A1 manifest.
3. Add no further proxy, module, teacher or loss.

## Modernization Opportunities

**NONE required.** The distributional teacher cache, privileged-supervision framing and teacher-specific gradient diagnostics are already appropriate. Further modernization would more likely add bloat than scientific value.

## Drift Warning

**NONE.**

Test-time MLLM use, segment objectives, learned signature encoders, teacher-key memories, generated counterfactuals or SSR stacking would be drift. None is present.

## Anchor Preserved

**Yes.**

## Remaining Action Items

1. **MINOR:** Freeze the exact finite `tau_proxy` search grid, pooled match tolerance and tie rule; use bisection only within verified monotone brackets.
2. **MINOR:** Use “teacher-semantic-free strength-matched proxy” consistently.
3. **MINOR:** Include held-out query-signature rows in the fold artifact hashes.
4. **MINOR:** Freeze proxy edge cases for fewer than two decoded frames, undefined SSIM and zero IQR.
5. **IMPORTANT documentation item before A2:** Operationalize “survive calibrated corruption” from the immutable anchor—for example, empirical-rate corruption should retain a positive paired advantage over remove-MLLM while degrading from clean EDCM, with 2× corruption weaker still. Freeze the exact rule rather than interpreting it after results.
6. Execute A0 → A1 → A2 → A3 without relaxing gates or tuning after failures.

## Final verdict

**READY**

The proposal now meets the READY criteria: weighted score above 9, exact anchor preservation, one focused dominant contribution, no method bloat, a defensible MLLM-specific novelty claim, and no blocking mechanism, supervision, leakage or feasibility issue.

</details>
