# Round 2 Review

**Reviewer agent:** `/root/edcm_pivot_refine/edcm_reviewer`  
**Overall:** 8.28 / 10  
**Verdict:** REVISE  
**Anchor:** preserved; drift warning NONE  
**No-segment-gold audit:** PASS

## Parsed Review

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.5 |
| Method Specificity | 8.0 |
| Contribution Quality | 8.0 |
| Frontier Leverage | 8.5 |
| Feasibility | 7.5 |
| Validation Focus | 9.0 |
| Venue Readiness | 7.5 |
| **Overall** | **8.28** |

### Priority actions

1. **CRITICAL:** strength-match the single teacher-free proxy's induced TV/gradient scale to EDCM without a learned module.
2. Freeze the normalized directional derivative equations and exact candidate set.
3. Specify/hash fold-local OOF model, query/bank, self-exclusion and no-full-data-embedding invariants.
4. Match core bank refresh/self-exclusion to the comparator.
5. Define the TV convention exactly.

## Full Raw Reviewer Response (verbatim)

<details>
<summary>Round 2 raw response</summary>

# Round 2 Senior ML Review — EDCM-RGCL

## Overall assessment

The revision is substantially stronger. It resolves the Round 1 loss-definition error, distinguishes cache coverage from actual teacher-specific gradient density, limits A0 to its legitimate role, adds a necessary non-MLLM control, and tightens both supervision language and reproducibility.

The contribution is now sharper:

> An MLLM-derived, train-only coalition measure changes the query gradient over the same full-video memory geometry used by the unchanged final kNN.

This is a focused mechanism, not concatenation, segment weighting, teacher-key replacement, or a parallel classifier. The validation package is somewhat elaborate, but the method itself remains simple: one cache, one loss, zero new trainable modules.

Two issues still block READY:

1. The teacher-free proxy is matched in coverage and workload but not yet in induced weight/gradient strength. Its `[0,1]` distance distribution can differ materially from EDCM’s `[-1,1]` signature distribution.
2. The OOF gradient gate needs a fully explicit fold-local implementation and directional-derivative formula to rule out leakage and implementation ambiguity.

**Verdict: REVISE**

## Scores

| Dimension | Score | Weighted contribution |
|---|---:|---:|
| 1. Problem Fidelity | 9.5 | 1.425 |
| 2. Method Specificity | 8.0 | 2.000 |
| 3. Contribution Quality | 8.0 | 2.000 |
| 4. Frontier Leverage | 8.5 | 1.275 |
| 5. Feasibility | 7.5 | 0.750 |
| 6. Validation Focus | 9.0 | 0.450 |
| 7. Venue Readiness | 7.5 | 0.375 |
| **OVERALL** | **8.28 / 10** | |

No dimension is below 7. The remaining revisions are readiness blockers rather than reasons to abandon the route.

## 1. Problem Fidelity — 9.5/10

The immutable anchor remains intact:

- Final accuracy and macro-F1 remain the only success endpoint.
- The target remains +0.030/+0.030 on two datasets and three paired seeds.
- The MLLM acts only during training.
- Final evaluation uses unchanged full-video embeddings, ordinary train memory, FAISS retrieval and the original vote.
- There is no localization endpoint, native-head substitution, test-time reranking, generated content, or protocol relaxation.
- SSR’s sparse-support failure is addressed through all-video listwise influence rather than another sparse edge mechanism.

The only terminology tension is that the immutable anchor still says “causal signal,” while the revised method correctly calls the outputs interventional weak pseudo-signals. This is acceptable if the paper reserves causality for the controlled effect of removing or corrupting the MLLM signal.

## 2. Method Specificity — 8.0/10

The corrected kernel is coherent:

\[
q_{ij}^{\pm}
=
\frac{\exp(-d_{ij}/\tau_s)}
{\sum_{k\in\mathcal L_i^\pm}\exp(-d_{ik}/\tau_s)},
\qquad \tau_s=1.
\]

This implements the intended conditional list measure without double exponentiation. The detached bank and full-video query make the route directly implementable in the existing RGCL training path.

The teacher-active definitions are also mathematically meaningful:

- TV detects departure from Label-only uniform weights.
- \(\Delta g_i\) measures the exact teacher-dependent query-gradient component.
- \(R_i\) prevents calling an arbitrarily tiny gradient difference active.
- The equal-step directional comparison checks whether the teacher gradient points toward a better supervised neighborhood margin, rather than merely being nonzero.

### Remaining specification issue — IMPORTANT

The directional test should be frozen explicitly as:

\[
v_i^{E}=
-\frac{\nabla_{z_i}L_{\mathrm{EDCM}}}
{\|\nabla_{z_i}L_{\mathrm{EDCM}}\|+\epsilon},
\qquad
v_i^{U}=
-\frac{\nabla_{z_i}L_{\mathrm{uniform}}}
{\|\nabla_{z_i}L_{\mathrm{uniform}}\|+\epsilon},
\]

\[
D_i^E=\nabla_{z_i}\mu_i^\top v_i^E,\qquad
D_i^U=\nabla_{z_i}\mu_i^\top v_i^U,
\qquad
\Delta D_i=D_i^E-D_i^U.
\]

The gate should bootstrap \(\Delta D_i\). Define whether `logsumexp` uses the same 8+8 lists and temperature as the training loss; it should.

Also freeze these OOF invariants:

- Each fold’s query and candidate embeddings come from the same fold model.
- Candidate keys come only from that fold’s training partition.
- The held-out query is never present in its bank.
- Candidate lists and signatures are frozen before gradient calculation.
- No full-data-trained embedding is substituted for an OOF embedding.

Without this clarification, “OOF geometry” remains vulnerable to implementation leakage.

## 3. Contribution Quality — 8.0/10

The contribution is much sharper than in Round 1. The proposal now isolates four alternatives:

- ordinary RGCL;
- generic Label-only ListNCA;
- low-level modality/content conditioning;
- arbitrarily assigned or corrupted MLLM signatures.

That is the correct causal decomposition for an MLLM-specific contribution.

### Remaining proxy issue — CRITICAL for READY

The proxy is not yet fully matched. EDCM signatures can span `[-1,1]`, while the proxy is scaled to `[0,1]`; their covariance, pairwise-distance distribution, TV and gradient norms can differ. EDCM could beat the proxy because it induces stronger list perturbations, not because it contains better semantic information.

The visual proxy is also nearly constant when all videos have four valid frames, while several product dimensions can duplicate transcript/OCR availability. This makes it a legitimate low-level baseline but potentially a weak one.

### Minimal fix

Retain exactly one proxy, but strength-match it without adding a module:

1. Reuse the exact EDCM accepted mask and \(\rho_i\).
2. Transform proxy dimensions using train-only robust signed scaling to the same numeric range as the EDCM signature.
3. Freeze one proxy-only temperature before development so its OOF median TV—or preferably median \(R_i\)—matches full EDCM.
4. Report both methods’ TV, \(R\), pairwise-distance and active-fraction distributions.

This calibration may use aggregate EDCM statistics but no per-video MLLM alignment; call it a **teacher-semantic-free strength-matched proxy**. It then tests semantic alignment rather than perturbation magnitude.

No additional proxy or learned baseline is needed.

## 4. Frontier Leverage — 8.5/10

The frontier primitive is appropriate. The proposal uses the MLLM where it is plausibly strongest—relative multimodal interpretation under controlled omissions—and discards its weaker absolute class verdict.

The mechanism also has a clean privileged-learning interpretation: expensive semantic judgments supervise training geometry but are absent during deployment. Using a fixed 7B teacher is preferable to seeking gains through teacher scaling, which would conflict with the anchor.

No additional MLLM module, rationale generator, router, or test-time reasoner is justified.

## 5. Feasibility — 7.5/10

The method is feasible in the existing architecture:

- zero new trainable components;
- detached keys and differentiable full-video queries;
- existing same/opposite-label mining;
- deterministic missing-signal fallback;
- no validation/test teacher path.

The main implementation risks are:

- correct fold-local OOF banks;
- excluding self-neighbors;
- computing exact per-query gradient differences efficiently;
- repeated multimodal serialization of seven coalitions;
- strict all-seven-coalition acceptance reducing coverage.

These are engineering risks, not conceptual blockers. The proposed smoke and staged stopping rule are appropriate.

## 6. Validation Focus — 9.0/10

The validation design is now unusually falsifiable:

- A0 can stop an economically unjustified route before MLLM cost.
- A1 can reject a reliable but gradient-inert teacher.
- A2 distinguishes generic listwise gain, modality metadata, MLLM alignment and noise sensitivity.
- A3 retains the complete final endpoint.
- kNN improvement is required separately from native-head behavior.

The gates must remain pre-registered. They should not be relaxed because a teacher produces low TV, low coverage, or an unfavorable proxy comparison.

## 7. Venue Readiness — 7.5/10

The novelty claim is now defensible but still conditional on results. A reviewer can still summarize EDCM as privileged signature-weighted NCA, but the teacher-active gradient analysis and strength-matched proxy would make the MLLM-specific retrieval contribution much harder to dismiss.

Venue readiness requires:

- proxy strength matching;
- leakage-free fold-local gate implementation;
- empirical evidence that full EDCM beats every matched control;
- final +3/+3 results in the unchanged kNN readout.

A proposal alone cannot establish the final novelty-strength claim, but the method is now sufficiently focused to justify staged execution after the two specification fixes.

## Teacher-active gate audit

### Mathematical coherence

**Mostly yes.**

TV, \(\Delta g\), \(R\), and the equal-step neighborhood-margin comparison measure distinct properties:

- non-uniform teacher weights;
- nonzero teacher-specific gradient;
- meaningful gradient magnitude;
- useful gradient direction relative to Label-only.

The equal normalization avoids awarding EDCM merely for having a larger raw gradient norm.

### Non-circularity

**Yes, subject to strict pre-registration.**

The gate uses video-level training labels to define supervised positive/negative neighborhoods and margin. That is permitted training supervision, not an illicit oracle. The MLLM does not receive those labels. The gate asks whether a frozen label-blind pseudo-signal produces a better supervised training direction.

It becomes circular only if prompts, thresholds, signature formula, temperature, or proxy are changed after examining the gate outcomes. Their hashes therefore need to precede extraction.

### Implementability

**Yes, after the OOF formula/fold clarifications above.**

The calculation requires no segment labels. It is a gradient diagnostic on full-video embeddings and video-level class membership.

### What it proves

Passing proves broad, mechanism-specific first-order support on the frozen OOF geometry. It does not prove eventual kNN improvement or +3/+3; A2 and A3 correctly remain binding.

## A0 assessment

A0 is now correctly characterized.

It is:

- a constrained fixed-geometry label-oracle reachability diagnostic;
- non-vacuous because it limits replacements to top-64 candidates and at most two swaps;
- permitted because it uses only OOF training videos and video-level labels;
- useful as a teacher-cost screen after SSR.

It is not:

- an upper bound on learned EDCM;
- evidence of MLLM usefulness;
- proof that the teacher can select the reachable corrections.

The revised wording handles these distinctions appropriately.

## Does \(L_{\mathrm{EDCM}}\) directly shape final kNN geometry?

**Yes, with one implementation condition.**

The loss differentiates the full-video query embedding against full-video keys from the same training memory construction used by retrieval. Because every train video is subsequently encoded by the shared student, query-side updates alter both future queries and rebuilt memory keys. No pseudo-signal is written into final memory.

The implementation must:

- rebuild/refresh the detached bank according to the exact comparator schedule;
- exclude the query itself;
- preserve identical final memory construction and voting.

Under those conditions, this is direct geometry training, not head-only absorption or post-hoc reweighting.

## No-Segment-Gold Audit

**PASS.**

- The only gold is the binary video label.
- Four uniform frames are deterministic whole-video inputs, not annotated segments.
- Dataset transcripts are raw input modalities and contain no temporal gold.
- OCR is deterministic input extraction, not annotation.
- Teacher outputs are whole-video coalition distributions.
- The cache schema contains no timestamp, span, segment label, localization score, or segment correctness target.
- All gates use full-video embeddings and video-level labels.
- Validation/test receive no MLLM, OCR, coalition or confidence artifact.

The text should continue to avoid phrases such as “dense annotation.” The signal is dense in training coverage/influence, not densely annotated ground truth.

## Degeneration audit

- Concatenation: **no**
- Static segment weighting: **no**
- Segment-weighted memory: **no**
- Selected teacher keys: **no**
- Test-time reranking: **no**
- Sparse SSR-style edge mechanism: **no**
- Generic Label-only ListNCA: **explicitly falsified by the teacher-active and A2 controls**
- Low-level modality metadata: **addressed, but proxy strength matching remains necessary**

## Simplicity / overbuilding assessment

The method is simple and appropriately scoped. The number of diagnostics is high, but they are gates and controls rather than modules. Given the unusually strict causal/removability target and eleven failed routes, that validation burden is justified.

Do not add:

- further teachers;
- learned signature encoders;
- student coalition branches;
- localization supervision;
- teacher-key memories;
- extra contrastive objectives.

## Simplification Opportunities

1. Define the directional derivative once in closed form and reuse it throughout the proposal, implementation and logs.
2. Treat TV as an interpretable support statistic and \(R/\Delta D\) as the binding mechanism statistics; avoid creating further overlapping density metrics.
3. Keep only the single strength-matched low-level proxy.

## Modernization Opportunities

1. The retained four-call ordinal distributions are already the right minimal uncertainty representation; no learned uncertainty model is needed.
2. Present the method explicitly as privileged interventional supervision for retrieval geometry, with teacher influence measured through gradient differences.
3. Use paired prediction-level uncertainty for final kNN comparisons while retaining the immutable hierarchical/bootstrap and Holm protocol.

## Drift Warning

**NONE.**

The revision remains inside the exact problem. Adding test-time semantics, segment targets, learned routers, teacher-key replacement, generated evidence, or a second method would be drift.

## Anchor Preserved

**Yes.**

## Remaining Action Items

1. **CRITICAL:** Strength-match the teacher-free proxy’s induced TV/gradient scale to EDCM while preserving one proxy and no learned module.
2. **IMPORTANT:** Freeze the exact normalized directional-derivative equations and candidate set.
3. **IMPORTANT:** Specify and hash the fully fold-local OOF embedding/bank/self-exclusion procedure.
4. **IMPORTANT:** Confirm the core training bank refresh and self-neighbor exclusion match the comparator exactly.
5. **MINOR:** Define the TV convention explicitly to avoid an accidental extra factor of \(1/2\).
6. Run A0 only after these specifications are frozen; proceed to teacher extraction only if its registered cost screen passes.
7. Treat any A1/A2 gate failure as route failure rather than tuning permission.

## Final verdict

**REVISE**

EDCM is now a coherent, focused and falsifiable route with appropriate frontier leverage. It remains below READY because the proxy is not yet strength-matched and the OOF directional-gradient gate still has implementation ambiguities that matter for causal interpretation. These are bounded revisions; the core method does not need rethinking.

</details>
