# Round 1 Review

<details open>
<summary>Full raw independent reviewer response</summary>

# Round 1 Independent Senior ML Review

## Executive Verdict

**Verdict: RETHINK**

**Overall score: 4.98 / 10**

The proposal preserves several hard constraints unusually well: only video-level binary gold is used; no segment/timestamp/span/localization gold is assumed; the MLLM is train-only; validation/test use the unchanged full-video embedding and ordinary train-memory kNN; SSR/EDCM/CTE are interpreted within their actual bounds; SQ is correctly not called a performance failure.

However, the core method currently violates its own immutable bottleneck. Mathematically, it is a soft sample-weighted collection of group risks followed by generic gradient projection. Worse, its constraints are imposed on raw gradients but the executed optimizer is AdamW, so the claimed common-descent constraints do not constrain the actual parameter update. These are core-mechanism failures, not presentation gaps.

The proposal is not READY and cannot be repaired by adding implementation detail around the current QP. The optimizer mechanism must be re-derived.

---

## Anchor Audit

### Preserved

- **Only gold supervision:** parent-video binary label.
- **No segment-gold assumption:** PASS. Uniform frames, transcript/OCR and whole-video modes are inputs or weak pseudo-signals, never segment annotations.
- **No localization substitution:** PASS.
- **Train-only privileged MLLM:** PASS at the stated data-flow level.
- **No validation/test teacher, mode or extra head:** PASS.
- **Final endpoint:** unchanged ordinary full-video train-memory kNN.
- **SSR/EDCM/CTE interpretation:** correct and bounded.
- **SQ status:** correctly described as formal plan-only/unrun, not a performance failure.

### Drift / blocking inconsistency

1. The anchor forbids the route from becoming sample reweighting or generic gradient surgery, but the proposed update is exactly reducible to both.
2. The method optimizes worst semantic-mode risks, while the binding target is substantial average accuracy and macro-F1. “Base preserving” on a raw gradient does not ensure preservation after AdamW.
3. The frozen Gate-0 ordering says ECM execution follows SQ termination. SQ is still `PLAN_ONLY_NOT_RUN`. The proposal may be refined as a reserve, but it must explicitly state that **no ECM-0 code, job or teacher call is unlocked before SQ reaches a documented terminal decision**.

**Drift Warning: PRESENT.** Supervision and endpoint fidelity are preserved, but the core optimization mechanism drifts into two explicitly excluded families.

---

## Central Blocking Issues

### 1. The method is dynamically sample-weighted gradient surgery

For mode \(m\),

\[
R_m=\sum_i w_{im}\ell_i,\qquad
w_{im}\propto r_iq_{im}
\]

within each class, so

\[
g_m=\nabla R_m=\sum_i w_{im}\nabla\ell_i.
\]

The QP solution has the KKT form

\[
d^\star=c_0g_0+\sum_m c_mg_m,
\]

and therefore

\[
d^\star=\sum_i \widetilde w_i\nabla\ell_i
\]

plus the ordinary RGCL components. It is a dynamically reweighted example gradient. The fact that weights arise through dual coefficients instead of a single scalar loss does not make it non-reweighting.

The projection is also a standard multi-objective/common-descent operation. PCGrad/CAGrad controls are necessary, but beating them empirically would show that the semantic grouping is useful; it would not make the optimizer primitive new. Pseudo-group inference followed by downstream robust/invariant optimization is already a mature family, including recent work on more precise inferred groups such as [GIC](https://proceedings.mlr.press/v235/han24g.html).

**Required fix — CRITICAL:** replace the parameter-gradient QP with a mechanism that changes the target retrieval geometry rather than coefficients on existing sample gradients. A defensible candidate is a full-bank semantic proximal target program:

1. Optimize stopped-gradient target embeddings \(Z^\star\), not parameter gradients.
2. Minimize displacement from the current full bank.
3. Constrain ordinary exact-vote performance and semantic-mode vote-margin changes directly in embedding space.
4. Fit the shared encoder toward \(Z^\star\) with a fixed-capacity target-fitting loss.
5. Demonstrate numerically that the resulting update cannot be reproduced by scalar reweighting of the ordinary RGCL per-example losses.

This would remain train-only, parameter-count matched, full-bank, capable of reaching outside old top-64 neighborhoods, and directly tied to the final embedding. If such a construction is not feasible, ECM should be abandoned rather than calling the current QP non-reweighting.

### 2. The constraints do not constrain the executed AdamW step

The proposal enforces

\[
g_m^\top d\ge0,
\]

but `src/run_rac.py` passes the edited gradients through AdamW momentum, adaptive diagonal preconditioning and weight decay. The executed direction is not \(d\). In general,

\[
g_m^\top d\ge0
\]

does not imply descent under the actual AdamW update. EMA mode gradients create an additional gap between the constrained direction and current mode risk.

Thus “executable constraint optimizer,” “base-preserving,” and “common-descent” are currently false as operational claims.

**Required fix — CRITICAL:** if a projection route is retained, formulate it in actual update space:

- compute the provisional AdamW step \(u_0\) from explicitly frozen optimizer state;
- solve for \(u^\star\) near \(u_0\);
- constrain current base and semantic gradients against \(u^\star\);
- specify whether moments are updated from \(g_0\), the combined gradient, or another frozen rule;
- apply \(u^\star\) directly;
- verify scalar/vector parity and realized post-step loss changes.

Do not claim descent from an EMA direction alone; report it as a stabilized proxy and separately measure realized current-risk change.

### 3. “Correctness-firewalled” is overstated

The teacher does not receive a literal gold label or correctness bit, which is good. But it receives:

- raw whole-video evidence;
- the model’s predicted decision;
- confidence;
- modality-specific decisions;
- entropy and neighborhood dispersion.

An MLLM can form its own implicit hate judgment from the video and compare it with the supplied prediction, reconstructing an error propensity. Block 2 then explicitly rewards \(q/r\) for predicting OOF error. This can produce a semantic-looking JTT signal without direct correctness fields.

The existing within-correctness MODE-SHUFFLE is strong but insufficient: it preserves binary correctness, not necessarily continuous reconstructed error propensity or teacher self-verdict confidence.

**Required fix — CRITICAL:**

- Rename the property to **no-direct-outcome-field**, not correctness-firewalled.
- Add a same-input **ERROR-PROPENSITY** control: cross-fit a scalar error-propensity score from the complete teacher posterior/trace using train-only labels, then give it the same downstream capacity.
- Add a semantic shuffle within fine bins of label, prediction, margin/confidence and cross-fitted error propensity.
- Require FULL to beat that scalar control, JTT and margin bins.
- Change the teacher-value gate from “q improves error AUC” alone to “semantic mode identity improves mode-specific correction/gradient evidence beyond a matched scalar error propensity.”
- Report how much q predicts label, correctness and true-class margin separately. None alone establishes semantic mechanism value.

### 4. The alleged exact-vote loss is underspecified and numerically risky

The current repository endpoint is top-20 arithmetic similarity voting. Within a frozen ranking, the proposed expression simplifies to a signed similarity-vote margin, which is directionally relevant. However:

- Self-neighbor exclusion is unspecified. A train query evaluated against the same full train bank would retrieve itself with similarity near 1 and the largest rank weight.
- The margin is not normalized by the arithmetic weight sum and has no temperature. With weights summing to 210, softplus can saturate, concentrating almost all gradient on current errors and making the method even closer to JTT.
- It is unclear whether the bank is computed in `model.eval()` as the endpoint is, or with training dropout.
- It is unclear whether memory keys are in the same autograd graph. Epoch-wise refresh does not make a detached key co-move in the current gradient.
- Rank is detached, so this is a local frozen-neighborhood surrogate, not an exact differentiable kNN objective.
- The handling of rank changes, ties, duplicate IDs and outer-fold memory exclusion is absent.

**Required fix — CRITICAL:**

- Explicitly exclude the query ID from the bank.
- Use the evaluator-normalized signed arithmetic-vote margin and a preregistered temperature.
- Define eval-mode full-bank construction and whether both query and key roles receive gradients.
- If keys are detached, call the method an alternating query-side surrogate and symmetrize roles; do not claim simultaneous query/key gradient movement.
- Specify stable top-k/tie behavior and exact parity tests against `compute_metrics_retrieval`.
- Measure gradient mass on correct versus incorrect examples; a near-error-only gradient invalidates the claimed distinction from JTT.

### 5. The ontology mixes observable conditions with unsupported causal diagnoses

`modality_conflict` is observable from the trace. `presentation_context_inversion` and `target_binding` are plausible semantic hypotheses. But `surface_shortcut` claims what internal feature dominated the OOF model without attribution evidence, and `evidence_dilution` partially overlaps presentation/context ambiguity and confidence.

A six-way posterior can be dense while semantically arbitrary. Self-consistency, support and error predictiveness do not establish that the categories are distinct failure mechanisms.

**Required fix — IMPORTANT:**

- Reframe outputs as **teacher-hypothesized whole-video trace discrepancy modes**, not verified failure causes.
- Give every actionable mode an operational trace signature.
- Require pairwise distinguishability beyond label, prediction, margin, modality decisions and embedding cluster.
- Permit unsupported modes to be preregisteredly pruned based only on support/reliability—not outcome—rather than forcing every category to receive mass.
- Merge categories whose conditional gradient or correction signatures are indistinguishable.
- Keep `undiagnosed` as exact REMOVE fallback.
- Do not add spans, target annotations or mechanism gold; none exists.

### 6. ECM-0 and hyperparameter selection are not statistically closed

“One global beta is selected by the zero-call screen,” while the same OOF endpoints determine whether ECM-0 passes. Unless selection is nested, this tunes beta on the screen used to claim capacity.

The frozen Gate-0 evidence also required loss-only difficulty bins and embedding-cluster controls. The detailed binding list currently emphasizes margin-bin/random modes but does not clearly include standalone `LOSS-BIN-PROJECT` and `EMBEDDING-CLUSTER-PROJECT`.

**Required fix — IMPORTANT:**

- Pre-register beta independently of outcome or choose it inside inner-inner folds, then evaluate it once on untouched outer folds.
- Add explicit loss-bin and embedding-cluster projection controls.
- Freeze one control-tuning budget shared by FULL and every robust optimizer.
- Do not let ECM-0 proxy performance select ontology, temperature, beta, EMA decay and mode batching simultaneously.
- Keep ECM-0 explicitly zero-call and label-only; it is a cost/capacity screen, not evidence for the MLLM.

### 7. Execution ordering must respect SQ’s current state

The proposal correctly says SQ is not a performance failure, but does not bind ECM execution to that fact.

**Required fix — CRITICAL for process fidelity:**

Add an explicit lock:

> ECM remains a reserve specification. No ECM implementation, SLURM job, teacher call, cache generation or performance claim is authorized until SQ reaches a verified terminal decision under its frozen S0/S1 protocol. SQ’s current `PLAN_ONLY_NOT_RUN` status provides no evidence for or against its performance.

---

## Seven-Dimension Scores

### 1. Problem Fidelity — 6.5 / 10

**Weakness:** Supervision, endpoint and no-segment-gold constraints are preserved, but the core reduces to two method families explicitly forbidden by the anchor. The proposal also lacks a binding SQ-before-ECM execution lock.

**Fix:** replace the gradient-QP core with a non-reweighting final-geometry mechanism and add the SQ terminal-state lock.

**Priority:** CRITICAL.

### 2. Method Specificity — 4.5 / 10

**Weakness:** AdamW update semantics, full-bank autograd, self exclusion, vote-margin normalization/temperature, EMA schedule, macro-batch construction, QP state, fallback certification and beta selection are not implementation-complete.

**Fix:** specify the actual executed update mathematically and provide an exact evaluator/optimizer parity protocol. If moving to proximal target geometry, define the full program, target fitting and refresh schedule end to end.

**Priority:** CRITICAL.

### 3. Contribution Quality — 4.0 / 10

**Weakness:** The dominant contribution is presently “MLLM pseudo-groups + weighted group objectives + common-descent projection + retrieval endpoint.” That is a task/interface combination, not yet a new mechanism. The supporting “firewall” is protocol hygiene and is not actually a full correctness firewall.

Recent work already combines semantic pretrained-model guidance with gradient alignment and representation disentanglement, further narrowing generic semantic-guidance claims; see [Superclass-Guided Representation Disentanglement](https://proceedings.mlr.press/v328/liu26a.html).

**Fix:** make the paper’s single contribution a novel final-bank geometry operator whose behavior cannot be reduced to sample weights, GroupDRO or PCGrad/CAGrad. Move OOF/firewall handling to methodology integrity, not a second contribution.

**Priority:** CRITICAL.

### 4. Frontier Leverage — 6.0 / 10

**Weakness:** The MLLM is used in a sensible train-only privileged role, but operationally it is still a semantic group annotator. Its model-specific reasoning is not yet connected to an observable mechanism beyond inferred error propensity.

**Fix:** make the MLLM output control a semantic geometry intervention or target constraint that a scalar difficulty/error score cannot reproduce. Do not add a larger model, more prompts, free text, rationale heads or test-time reasoning.

**Priority:** IMPORTANT.

### 5. Feasibility — 5.0 / 10

**Weakness:** Cache generation is feasible, but five separate full-bank gradients, an ambiguous autograd bank, AdamW-state mismatch and many OOF control arms make the stated compute estimate weak. Pilot power with 128 videos and high-dimensional conditional controls is also unproven.

**Fix:** first microbenchmark one exact full-bank update with all active modes; freeze peak memory, wall time, backward count and realized constraint rate. Perform an explicit power calculation before teacher calls. Preserve SLURM-only execution.

**Priority:** IMPORTANT.

### 6. Validation Focus — 5.5 / 10

**Weakness:** The three-stage structure is good, and margin-bin, mode-shuffle, GroupDRO, JTT, EIIL, PCGrad and CAGrad are appropriately recognized. But the validation currently rewards error reconstructability, omits explicit loss-bin/embedding-cluster projection arms, allows circular beta selection and does not isolate semantic modes from a matched MLLM-derived scalar error score.

**Fix:** add the scalar error-propensity control, loss-bin and embedding-cluster controls; nest hyperparameter selection; make semantic correction beyond matched error propensity the teacher-value gate.

**Priority:** CRITICAL.

### 7. Venue Readiness — 4.0 / 10

**Weakness:** A reviewer can currently summarize the contribution as “use an MLLM to define soft groups and run a constrained multi-task optimizer on a kNN loss.” The optimizer guarantee is also invalid under AdamW. Even successful results would face a serious pseudo-novelty objection.

**Fix:** re-derive the core as a final-bank semantic geometry method, make the MLLM causally necessary beyond error scoring, and prove actual-update rather than raw-gradient constraints.

**Priority:** CRITICAL.

### Weighted Overall

\[
0.15(6.5)+0.25(4.5)+0.25(4.0)+0.15(6.0)+0.10(5.0)+0.05(5.5)+0.05(4.0)
=\mathbf{4.975}.
\]

**Overall: 4.98 / 10**

---

## Simplification Opportunities

1. Remove “correctness firewall” as a supporting contribution. Keep the no-direct-outcome protocol as necessary integrity hygiene.
2. Do not force five actionable modes. Start with the smallest preregistered set of operationally distinguishable modes plus `undiagnosed`; merge or prune unsupported categories outcome-blindly.
3. Stage controls rather than running the entire menu at every gate:

   - ECM-0: non-MLLM capacity and optimizer-family controls.
   - ECM-1: semantic modes versus scalar error propensity, margin/loss/cluster groups and shuffle.
   - Seed-0/final: only the strongest binding controls plus REMOVE/SHUFFLE.

---

## Modernization Opportunities

1. Use the MLLM as a **train-only semantic geometry critic**, not merely a pseudo-group labeler: its posterior should alter target bank geometry in a way that cannot be expressed as weights on existing RGCL losses.
2. If retaining constrained optimization, operate on the actual adaptive optimizer step or on full-bank target embeddings. Raw-gradient projection followed by AdamW is not an executable guarantee.
3. Do not add model scaling, more teacher calls, rationales, MoE, routers or test-time reasoning. The problem is mechanism identity, not insufficient foundation-model capacity.

---

## No-Segment-Gold Audit

**PASS.**

The proposal does not assume segment, timestamp, span, stance, target, mechanism, rationale or localization gold. `evidence_dilution` is stated as a whole-video weak pseudo-mode, and no segment weighting or segment loss is introduced. Preserve this exactly in every revision.

---

## Conditions for a Future READY Verdict

A revised proposal cannot be READY merely by clarifying the present QP. It must:

1. Replace or fundamentally re-derive the core so it is not reducible to dynamic sample reweighting plus generic gradient surgery.
2. Constrain the actual optimizer/embedding update, not a raw gradient later transformed by AdamW.
3. Prove semantic mode value beyond a matched scalar error-propensity teacher.
4. Operationalize the ontology without causal overclaim or nonexistent mechanism gold.
5. fully specify exact-vote parity, self exclusion, bank autograd and numerical scaling.
6. Close beta/temperature/EMA selection without OOF screen reuse.
7. Add the explicit SQ-terminal execution lock.
8. Preserve the current no-segment-gold and test-time-clean inference contract.

Until those are addressed, the correct verdict is **RETHINK**, not REVISE or READY.

</details>
