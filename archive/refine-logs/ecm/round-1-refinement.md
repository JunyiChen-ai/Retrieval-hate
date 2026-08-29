# Round 1 Refinement

> **ARCHIVAL / NOT CANONICAL / DO NOT EXECUTE.** This post-review proximal-bank sketch is intentionally preserved as the Phase-3 author response, but it was not sent for a second score. The parent research decision accepted Round 1's `RETHINK`: the frozen ECM identity is abandoned rather than renamed. Any future full-bank proximal-target route must enter as a distinct new hypothesis with a fresh novelty review; this file authorizes no code, job or teacher call.

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causally removable part of hateful-video RGCL and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on MHC-EN and MHC-ZH, paired seeds `0/1/2`, under the complete statistical and mechanism-attribution protocol.
- **Must-solve bottleneck:** The MLLM must diagnose dense whole-video failure mechanisms from a strict-OOF, label-blind prediction trace for every training video, and those modes must directly alter the optimizer of the shared final RGCL embedding. The route must reach errors outside SSR's sparse edge universe and EDCM's frozen top-64/two-swap universe, without becoming sample reweighting, a router, a renamed GroupDRO/JTT/EIIL method, or generic gradient surgery.
- **Non-goals:** No localization endpoint; no segment weighting or segment loss; no rationale/schema/summary concatenation; no teacher-selected key, score fusion, reranking, veto, auxiliary/native-head claim, test-time MLLM/mode/teacher, MoE/router, model/data/epoch/ensemble scaling, or rescue of frozen SSR/EDCM/CTE routes. SQ-RGCL is still at formal S0/S1 plan status and is not declared a performance failure here.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment, timestamp, span, stance, target, mechanism, rationale, or localization gold. MLLM modes are confidence-bearing train-only weak/privileged pseudo-signals. The teacher sees neither gold label nor any correctness/error/loss/true-class-margin indicator. Every train video is processed under the same strict-OOF rule; validation/test have no teacher, trace, mode, or extra head. All eventual computation must use SLURM in `HateVideo`, with at most 2 GPU / 16 CPU / 128 GB and no `--time`.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains `>=+0.030` accuracy and `>=+0.030` macro-F1 on both datasets; all three paired deltas are positive; hierarchical paired-bootstrap lower bounds exceed zero and four dataset-by-metric tests survive Holm correction. FULL significantly beats REMOVE and within-train MODE-SHUFFLE, and must beat margin-bin, standard GroupDRO, JTT, EIIL and generic gradient-surgery controls under matched capacity. Only final ordinary full-video kNN counts.

## Anchor Check

- **Original bottleneck:** obtain dense, semantically meaningful, train-only MLLM influence on the shared final embedding and substantial ordinary-kNN gains without nonexistent segment gold.
- **Why the revision still addresses it:** the MLLM remains a weak all-train strict-OOF critic, but its modes now constrain a stopped-gradient full-bank target geometry rather than weight sample gradients. The same shared encoder is fitted toward that geometry; inference remains unchanged.
- **Drift rejected:** no teacher at test, no segment endpoint, no teacher key, no new head and no scaling. SQ is still unrun.
- **Execution lock added:** ECM is a reserve specification. **No ECM implementation, SLURM job, teacher call, cache generation or performance claim is authorized until SQ reaches a verified terminal decision under its frozen S0/S1 protocol.** SQ's current `PLAN_ONLY_NOT_RUN` status is no evidence for or against performance.

## Simplicity Check

- **Dominant contribution:** one semantic proximal full-bank target operator controlled by MLLM whole-video discrepancy modes.
- **Removed:** parameter-gradient QP, EMA gradient claims, raw-gradient common-descent claim and “correctness firewall” as a contribution.
- **Kept frozen:** RGCL model, AdamW, total steps, bank endpoint, teacher scale and test path.
- **Why smallest adequate:** the only new training object is a stopped-gradient target bank `Z*`; there is no mode head, router, per-mode adapter or test artifact.

## Changes Made

1. **Replaced weighted mode-gradient projection.** Reviewer proved it was reducible to sample reweighting plus generic surgery and was invalid after AdamW. The revision solves a mode-constrained proximal program in full-bank embedding coordinates and treats AdamW only as a fixed-capacity target fitter; it verifies realized geometry after fitting rather than claiming a raw-gradient guarantee.
2. **Weakened the data-flow claim honestly.** “Correctness firewall” is now “no-direct-outcome-field.” A matched cross-fitted ERROR-PROPENSITY target arm and a fine semantic shuffle preserve reconstructed error difficulty.
3. **Specified repository geometry.** Self IDs are excluded; eval-mode normalized full banks, stable ties, exact top-20 arithmetic similarity vote, alternating query/key symmetry and scalar/vector/evaluator parity are binding.
4. **Operationalized modes and closed selection.** Modes are teacher-hypothesized trace discrepancies, not causal truth. Support pruning is outcome-blind. Hyperparameters are selected only inside inner-inner folds under a shared control budget.
5. **Added missing closest controls.** LOSS-BIN, EMBEDDING-CLUSTER, PG-DRO and same-mode PCGrad/CAGrad are explicit.

## Revised Proposal

# Research Proposal: ECM-RGCL — Semantic Proximal Bank Targeting from OOF Discrepancy Modes

### Technical gap and route choice

SSR/EDCM could not touch enough errors inside fixed old-neighbour actions; CTE stopped on a numerical policy gate; SQ remains formally unexecuted. Existing semantic fields/scores/heads were redundant or endpoint-displacing. The open mechanism is to let a train-only MLLM criticize **how an OOF model may be wrong**, without ever being told whether it is wrong, and convert the semantic structure into a full-bank target that is not a scalar weighting of existing RGCL losses.

GroupDRO/PG-DRO, JTT, EIIL, DISC and PCGrad/MGDA/CAGrad own group robust learning, error upweighting, inferred environments, concept mitigation and gradient manipulation. Therefore ECM claims none of those primitives. Its single thesis is:

> If label-hidden MLLM discrepancy identities distinguish multiple ways in which final retrieval geometry needs to move beyond scalar error propensity, a minimum-displacement worst-mode target bank can guide the shared encoder to a better ordinary-kNN geometry without a teacher or mode at inference.

The rejected Route A was soft mode risks plus a parameter-gradient QP. The chosen Route B optimizes explicit target embeddings under exact vote-margin constraints, then fits the same encoder with a vector target field.

### System and complexity

```text
strict five-fold train OOF model
  -> every train video: whole-video evidence + label-free OOF trace
  -> frozen MLLM: weak six-way discrepancy posterior q_i, reliability r_i
  -> freeze train-ID cache; only now join parent-video binary y_i

each registered target refresh:
  eval-mode full train bank Z -> exact self-excluded top-20 vote margins
  -> semantic proximal program -> stopped-gradient target bank Z*
  -> fixed share of existing AdamW steps fits shared encoder to Z*
  -> rebuild actual bank and verify realized train-only constraints

val/test: full video -> shared encoder -> ordinary train-memory kNN only
```

New trainable components and inference components: zero. The target program is an offline differentiable operator on the current train bank. All arms have the same model parameters, total AdamW steps, data, initialization, optimizer state policy, epochs, bank refresh count and checkpoint selection.

### No-direct-outcome-field strict-OOF trace

Use the frozen SSR five folds. `f^{-k}` never trained on fold `k`; for every held-out train video it emits a trace containing deterministic uniform whole-video frames, full-video transcript/OCR with temporal metadata stripped, full/visual-only/language-only predicted decisions and quantized confidence bands, prediction entropy, and unlabeled neighbour-similarity dispersion.

Forbidden teacher inputs are video gold, correctness, error flag/rank, supervised loss, true-class margin, neighbour labels/IDs, fold/seed, selection indicator and any validation/test record. Every train video—correct or incorrect—uses the identical payload, so cache presence carries no error bit. The teacher can still infer an error propensity from raw evidence; therefore the proposal makes no stronger “correctness firewall” claim.

### Teacher-hypothesized whole-video discrepancy modes

The frozen ontology is:

1. `presentation_context_inversion`: surface cues and whole-video pragmatic presentation may support opposing readings;
2. `target_binding`: the predicted reading is vulnerable to speaker/target/affected-entity attribution;
3. `modality_conflict`: visual and language channels support incompatible readings;
4. `surface_shortcut`: the trace is consistent with an isolated cue rather than integrated context;
5. `evidence_dilution`: relevant support is weak relative to whole-video irrelevant material, without locating it;
6. `undiagnosed`: none/ambiguous.

These are weak hypotheses, not verified causes or annotations. The schema contains only six probabilities, confidence in `{0,.25,.5,.75,1}`, parse status and hashes—no verdict, rationale, target identity, mechanism text, segment, timestamp or span. Two prompts × two input orders are averaged. `r_i=min(confidence)*(1-normalized mean pairwise JS)`; parse failure, argmax agreement `<.75`, JS `>.10` or `r_i<.5` maps to `undiagnosed` and exact REMOVE fallback.

Outcome-blind promotion retains an actionable mode only if both datasets have reliable total effective mass `>=30`, class-specific effective mass and Kish ESS `>=8`, and conditional pairwise distinguishability from every other retained mode using trace-only features after controlling predicted decision/confidence/modality decisions/embedding cluster. At least three common actionable modes and `>=90%` reliable-or-undiagnosed coverage are required. Unsupported modes are pruned before reading error outcomes; outcome-based ontology merging is forbidden.

### Exact full-bank vote geometry

Build `Z={z_i}` in `model.eval()` from every outer-train full video and L2-normalize. For any train query, remove all bank entries with the same canonical video ID. Stable ranking is cosine descending then canonical ID ascending; duplicate IDs, exact ties, negative cosine and vote exactly zero have frozen test fixtures.

For the current registered endpoint, let `j_r(i)` be the first 20 remaining keys, `w_r=21-r`, and signed label `c_i=2y_i-1`. The normalized exact similarity-vote margin is

`M_i(Z) = c_i * [sum_{r=1}^{20} w_r s(z_i,z_jr) c_jr] / 210`.

The helper must reproduce repository IDs/ranks/cosines/signed votes/predictions/accuracy/macro-F1 on every actual OOF query. If the frozen comparator configuration uses a different threshold/use-sim flag, the helper inherits it exactly rather than silently changing the endpoint.

### Semantic proximal target program

At a target refresh, optimize a stopped-gradient displacement `D={d_i}` and `Z*(D)=row_norm(Z+D)`. Top-20 is recomputed at each of exactly three registered proximal iterations; within each iteration its stable ranking is held fixed for a convex first-order subproblem. Both query and key coordinates are variables, so target geometry moves symmetrically.

For mode `m` and class `c`, define normalized weak membership over that class `a_im = r_i q_im / sum_{h:y_h=c} r_h q_hm`. Define the class-balanced realized target improvement

`Delta_m(D)=.5 sum_c sum_{i:y_i=c} a_im [M_i(Z*(D))-M_i(Z)]`.

Solve the epigraph program

`min_{D,t} .5||D||_F^2 - lambda*t`

subject to:

- `Delta_m(D) >= t` for every retained actionable mode;
- class-global mean margin change `>=0` separately for `y=0` and `y=1`;
- overall mean margin change `>=0`;
- `||d_i||_2 <= epsilon` and class-centroid displacement `<=epsilon_centroid`;
- no row may cross more than a registered maximum number of exact top-20 rank boundaries per proximal iteration.

This is a mode-level minimax curriculum in **final-bank coordinate space**, not a coefficient update on per-example RGCL losses. `q` specifies aggregate geometry constraints; it never scales the RGCL loss, chooses a minibatch or routes inference. The target operator is considered valid only if a nonnegative least-squares fit of `vec(D*)` by scalar-reweighted ordinary per-example RGCL embedding gradients leaves relative residual `>=.25` on both datasets. Otherwise the route is empirically reducible to sample reweighting and stops.

### Shared-encoder target fitting and actual-update semantics

`Z*` is detached. Within the unchanged total step budget, a registered fraction of AdamW minibatches replaces ordinary RGCL updates with

`L_fit = mean_i ||row_norm(z_i(theta))-Z_i*||_2^2`

using uniform video sampling; the remaining steps use ordinary RGCL. REMOVE spends the same replaced steps on additional ordinary RGCL; every control receives the same target-refresh and fit-step budgets. The optimizer, moments, weight decay, clipping and scheduler remain unchanged. ECM does **not** claim that AdamW executes the proximal displacement exactly.

After each target-fitting block, rebuild the actual eval-mode bank. Log target versus realized `Delta_m`, constraint satisfaction, displacement cosine, rank turnover and both class/global margins. If either class/global actual margin decreases or fewer than 80% of retained modes realize nonnegative change, restore the pre-block model and optimizer state and execute the corresponding REMOVE steps. This deterministic train-only fallback is frozen before validation outcomes. Final claims rely only on realized ordinary-kNN results.

### Error-propensity and prior-art controls

- `REMOVE`: strongest unmodified RGCL, same steps.
- `ERROR-PROPENSITY`: on inner-train only, cross-fit a scalar `e_i=P(OOF error | complete teacher posterior + trace)`; bin it into five equal-mass cells and run the identical proximal target/fit capacity.
- `MODE-SHUFFLE`: derange complete q/r records within outer-train `label × predicted-decision × confidence-bin × correctness × error-propensity-bin` cells; frozen adjacent-bin merging, no fixed point.
- `MARGIN-BIN`, `LOSS-BIN`, `EMBEDDING-CLUSTER`, `RANDOM-MODE`: identical target operator with five matched cheap groups.
- `GROUPDRO-MODE`, `PGDRO-MODE`, `JTT`, `EIIL+GROUPDRO`: standard robust/error/environment families, matched steps and tuning budget.
- `PCGRAD-MODE`, `CAGRAD-MODE`: generic gradient surgery on the identical semantic mode risks.
- `SCALAR-REWEIGHT-FIT`: best nonnegative scalar fit to `D*`, used both as a reducibility audit and learned control.

Teacher-value is not established by error AUC. Report `q->label`, `q->correctness`, `q->true-class-margin` and `q->error-propensity`. The binding semantic test is that FULL produces a per-mode realized margin/correction vector and final OOF improvement not matched by ERROR-PROPENSITY, MODE-SHUFFLE or SCALAR-REWEIGHT-FIT.

### Nested selection and three validation blocks

All `epsilon`, `epsilon_centroid`, `lambda`, target-fit fraction and proximal-iteration choices come from a small preregistered grid selected inside inner-inner train folds. The untouched outer fold is evaluated once. FULL and controls share the same number of candidates and selection metric; ontology and teacher thresholds never use outcomes.

#### ECM-0: zero-new-call capacity, parity and cost

Only after SQ has a verified terminal result, use existing OOF traces and deterministic trace clusters as proxy modes; make zero teacher calls. Verify evaluator parity, scalar/vector target-program parity, self exclusion, rank stability, symmetric query/key gradients, NNLS non-reducibility, solver completeness, realized-fallback rate, memory and wall time. Five-fold OOF proxy-ECM must improve actual concatenated accuracy and macro-F1 by `>=.050` on both datasets, every fold sign positive, not lose to the strongest margin/loss/cluster/GroupDRO/JTT/EIIL control, and have realized fallback `<=5%`. This is a cost/capacity policy gate, not an upper bound or MLLM result.

#### ECM-1: teacher legality, density and semantic necessity

After ECM-0 GO, freeze a label-hidden 128-video/dataset governance pilot before calls; power and mode coverage determine GO/STOP without changing ontology. Pilot GO permits the identical four-call template for every remaining training video. No validation/test call is ever made.

On full strict-OOF caches, require no-direct-outcome-field provenance, coverage/support/distinguishability and parse gates. FULL must beat ERROR-PROPENSITY, MODE-SHUFFLE, MARGIN/LOSS/CLUSTER, PGDRO, PCGrad, CAGrad and SCALAR-REWEIGHT-FIT in realized class-balanced worst-mode margin change with anchor-bootstrap adjusted lower bounds above zero on both datasets. The scalar control receives the same full teacher outputs and capacity. Failure means modes are difficulty/groups under new names.

#### ECM-2: seed-0 then final ordinary-kNN proof

Seed 0 on both datasets must improve validation ordinary-kNN accuracy and macro-F1 `>=.010` over every binding control, reduce worst retained-mode vote deficit, correct errors outside SSR/EDCM unions, and degrade monotonically when q is mixed with its train marginal at `{.25,.50,.75,1}`. No mode/teacher/epoch expansion is allowed after failure.

Final MHC-EN/ZH × paired seeds 0/1/2 uses FULL, REMOVE, MODE-SHUFFLE and the strongest registered prior-art control. Each metric must exceed `max(historical strongest, paired same-seed REMOVE mean)+.030`; 3/3 deltas are positive; 10,000 hierarchical paired bootstraps resample seeds then shared video IDs; lower bounds exceed zero and four dataset×metric tests pass Holm FWER `.05`. FULL-minus-REMOVE and FULL-minus-SHUFFLE use the same paired inference. Only final ordinary full-video kNN can close the project target.

### Failure handling, novelty and compute

- Fewer than three dense common semantic modes, mode collapse to label/difficulty/presentation, NNLS residual `<.25`, proxy capacity failure, solver/fitting fallback excess, or control parity -> STOP.
- Missing/uncertain records -> undiagnosed/REMOVE fallback.
- No segment/timestamp/span/localization field or gold is introduced. `evidence_dilution` remains whole-video and cannot mark a segment.
- SQ remains the active formal S0 route until its own frozen protocol ends; ECM refinement alone authorizes nothing.

The narrow novelty is **MLLM discrepancy-mode constrained minimum-displacement targeting of the exact full-bank retrieval geometry**, followed by fixed-capacity fitting of the same inference embedding. It is not first DRO, group discovery, semantic debiasing, gradient surgery or geometry distillation. Its necessity is falsified if a scalar error propensity, scalar reweighting or standard robust optimizer matches it.

Estimated cost after unlock: ECM-0 30--80 A100-hours; governance pilot <=1,024 calls; full cache <=4,512 calls; seed-0 30--70 GPU-hours; final 80--180 GPU-hours. A SLURM vectorized target-program/fit microbenchmark must replace estimates before execution. No code/job/call is performed in refinement.
