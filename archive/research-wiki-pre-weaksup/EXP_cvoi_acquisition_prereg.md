# EXP: CVoI — costed, set-conditioned evidence acquisition for hateful video detection

> **Status: DESIGN ONLY / PRE-REGISTRATION DRAFT (2026-08-08).** No CVoI
> candidate metric has been computed. This document does not authorize an experiment.
> Before execution, every item in the pre-execution completeness gate (§13) must be
> resolved, independently reviewed, and hash-frozen. Results must later be appended under
> `## RESULTS`; an endpoint, arm, split, cost rule, or decision-rule change is a registered
> deviation rather than a silent edit.
> **KILLED 2026-08-09** — see EXP_cvoi_acquisition_KILL_2026-08-09.md

## 0. Purpose

This study asks whether a hateful-video classifier can learn **which additional evidence is
worth acquiring**, at temporal-window level, before paying the acquisition cost. Its proposed
mechanism is counterfactual value of information (CVoI), not OCR presence, evidence-type
relevance, temporal salience, or access to an already-computed OCR cache.

For video `i`, an already acquired action set `S`, and a legal next action `a`, the target is

```text
u_i(a | S) = loss(y_i, p_i(S)) - loss(y_i, p_i(S union {a})).
```

The two predictions must come from the same cross-fitted classifier and differ only in the
available acquired evidence. Positive utility means that executing `a` reduced held-out
classification loss in that state. The proposed policy estimates expected utility per cost
from information available **before** executing `a`, and acquires actions sequentially under a
hard measured-cost budget.

This prospective experiment supersedes the earlier proposal to route using a predicted
`required_modalities` type. The type is retained only as a comparator. Relevance is not utility.

## 1. Frozen audit facts and unresolved assets

### 1.1 OCR asset

The current K=30 midpoint cache contains 30 OCR observations per video for 744 HateMM-train
and 107 HateMM-val videos. Test is absent and remains sealed. The cache records OCR output and
run-level timing, but it does **not** record a valid per-action measured cost. Therefore it may
support integrity/QC and later outcome construction, but cannot by itself establish an
accuracy-cost Pareto claim.

### 1.2 Dense-visual asset

The current dense visual cache uses K=30 windows and M=120 frames: four frames are encoded and
mean-pooled inside each window. The mean is irreversible. It does not preserve the four
per-frame embeddings, exact per-frame timestamps, or a measured cost for each candidate action.
It is consequently not yet an admissible costed acquisition action.

Before any candidate metric, dense visual extraction must emit, per frame: video ID, window ID,
timestamp, decode status, raw/processed resolution, feature vector or its immutable reference,
and measured action cost. The four-frame mean may then be derived reproducibly. It must not be
reverse-engineered from the existing aggregate.

### 1.3 The 22/28 lineage distinction

Two Gate-C subsets have different definitions and must never be interchanged:

- `strict_ocr_no_speech`: `on_screen_text` is required and the intersection of
  `required_modalities` with `{speech, transcript}` is empty. The audited count is 22.
  The frozen sorted-ID serialization (one ID per line, trailing newline) has SHA256
  `058aac5fa3bc8360429fe99331fdbfbb4dc3c025de14740827c86aaeddf1f317`.
- `ocr_no_speech_flag`: `on_screen_text` is required and a broader `no speech` condition is
  true, operationalized exactly as `speech not in required_modalities`; `transcript` may still
  be present. The audited count is 28. Its identically serialized sorted-ID list has SHA256
  `9d268f469bb0e8731d2681b9ceaab5fa3db6db14cc03ba4773f608d690cafe74`.

Before either subset is read by an analysis, a steward must materialize the exact video-ID lists,
definition/version, source-row provenance and SHA256. Both are descriptive diagnostic strata.
Neither may select a model, budget, threshold, arm, headline, or verdict.

## 2. Data roles and sealing

### 2.1 HateMM

- **Train (744):** sole development population; all fitting and nested OOF decisions occur here.
- **Val (107):** its raw videos may be decoded before freeze solely to build and cost-audit the
  action cache, without loading labels or producing predictions. Its labels, historical
  predictions and any performance-bearing per-example artifact are first read exactly once for
  confirmation after the train-OOF method, primary budget, thresholds, claims and verdict code are
  frozen. Val cannot select anything.
- **Test:** sealed throughout this study. A later full-method preregistration may authorize one
  final read after method freeze; this study does not.

Only video-level hate labels are ordinary gold supervision. Gold temporal spans, Gate-C labels,
Claude outputs and audit error categories are prohibited from every deployable training and
selection path.

### 2.2 External confirmation

A full cross-dataset/general claim requires a separately frozen dataset with both acquisition
actions and a compatible endpoint. HateClipSeg may be used only if its label-domain and survivor
subset caveats are carried forward. MHC-EN/ZH require language-appropriate OCR/text encoders and
cannot inherit an English 77-token representation silently. Until an external arm is completely
specified, a pass is HateMM-specific.

### 2.3 Contact ledger

Every run records opened paths, split role and a test-contact counter. Loading any official test
label, prediction, span, OCR record or per-example historical test artifact during design,
implementation, debugging or selection invalidates the prospective test claim and halts.

## 3. Groups, folds, seeds and refits

### 3.1 Frozen grouping

The resampling unit is a frozen connected component of duplicate, repost, template and creator
links. All videos, windows, frames, actions, seeds and refits belonging to one component stay in
one fold. The group-construction inputs, algorithm, connected components, class composition and
SHA256 are frozen before metrics. A video-only fallback is prohibited unless an audit proves that
no usable grouping source exists and a deviation is registered before metrics.

### 3.2 Grouped nested OOF

Primary development uses 5 group-stratified outer folds for each of three predeclared split seeds.
Inside every outer-training partition, use 4 group-stratified inner folds. If group/class
constraints make the requested fold count impossible, apply the pre-frozen deterministic fallback
`5 -> 4 -> 3` outer or `4 -> 3 -> 2` inner and record the reason; never redraw until a favourable
partition appears.

Within each outer fold:

1. fit all preprocessing, normalization, fusion/classification models and memories on legal
   outer-train data only;
2. construct CVoI targets only through inner-OOF predictions, so no group supplies both a target
   and the model that generated it;
3. choose arm-local hyperparameters, epoch and threshold using pooled inner-OOF predictions only;
4. refit the selected procedure three times on full outer-train with predeclared refit seeds;
5. emit one score per outer-query video per refit and never adapt to that query's label.

Fold metrics are not averaged. Within each split-seed/refit run, concatenate all outer-query
predictions and compute one metric. The headline point estimate is the mean of the nine complete
OOF-run metrics. Repeated rows for one video are not independent observations.

### 3.3 Threshold rule

Each arm-by-budget operating point receives its own threshold derived from its pooled inner-OOF
scores by maximum binary macro-F1. Ties are resolved by distance to 0.5, then the smaller numeric
threshold. The selected threshold is carried unchanged to the outer query. Neither 0.5 switching
nor retrospective threshold sharing is permitted.

## 4. Acquisition actions and legal information

The initial action universe is:

- `OCR(k)`: acquire OCR for temporal window `k`, K=30;
- `DENSE(k)`: acquire the registered four-frame dense visual observation for window `k`.

Each action has a typed output, success/missing indicator and measured cost. A joint policy may
select either type. OCR and dense visual are scientifically distinct actions; calls are not treated
as exchangeable merely because both concern one window.

Before acquisition, a policy may use only frozen cheap whole-video/segment features, transcript,
title, timestamps, previously acquired outputs, remaining budget, and cost estimates that do not
execute the candidate action. OCR text, OCR confidence, bounding boxes, text count or OCR-derived
embeddings are illegal pre-OCR features. Dense per-frame features are illegal before `DENSE(k)`.
If a proxy requires the rationed extractor, its full cost is charged and the action is considered
acquired.

Failed actions consume their actual measured cost and return a frozen missing token. No method may
retry for free or impute from the full cache.

## 5. Counterfactual targets and the proposed policy

### 5.1 Actual counterfactual utility

For every legal inner-held-out video, classifier state, and sampled legal action, compute
cross-entropy/log-loss before and after executing that action. Both predictions use one frozen
inner-fold model; only acquired evidence changes. Video labels are allowed to score utility on the
inner-held-out item, but that item/group was absent from model fitting.

The state classifier is trained on inner-training videos with the same frozen mask/state generator
used to construct policy states; it is not a full-information classifier evaluated by merely
zeroing features at inference. Its parameters, including missing tokens and fusion layers, are
fixed before either member of a before/after pair is evaluated. For a target-producing
inner-held-out video, neither its label nor any of its action outcomes may enter classifier fitting,
normalization, state-generator fitting or checkpoint selection.

The primary utility is **set-conditioned**. The training-state distribution contains the empty set,
sets generated by every fixed baseline policy, and on-policy prefixes generated without the
held-out label. State sampling probabilities and maximum trajectory length must be frozen in the
implementation appendix. Singleton utility `u(a | empty)` is a separate additive-control target.

Macro-F1 is not decomposable per item and is not used as the regression target. It is the binding
population endpoint after executing a policy. The protocol reports realized held-out log-loss
reduction as direct utility and macro-F1 change as decision utility.

### 5.2 Policy

The proposed CVoI model estimates `E[u(a | S) | cheap features, acquired state]`. At each step it
selects the feasible action with the largest registered utility/cost acquisition score, updates
`S`, and stops when no action fits or the model selects the null action. Training may use a
differentiable relaxation, but inference is discrete and budget-feasible.

If registered action costs are effectively uniform, the method is described as sequential hard
top-k, not knapsack. A knapsack claim requires heterogeneous measured costs and superiority to both
hard top-k and greedy utility/cost at matched measured cost.

## 6. Cost contract

### 6.1 Measurement

Before model metrics, freeze OCR/dense engine and weights, environment/container, GPU model,
decoder, resolution, batch size, warm-up, repetition count and synchronization method. For every
action record wall time, GPU time where available, frames processed, failures and retries. Also
measure policy, feature encoding, retrieval and optimizer/DP overhead. Raw timing records are
immutable artifacts.

The binding x-axis is online measured incremental latency per video on the frozen hardware. Report
OCR/dense calls, frames, GPU-seconds and energy if instrumentation is valid. Offline cache-build
cost is reported separately and may be amortized only under a predeclared deployment horizon; it
cannot substitute for online cost.

### 6.2 Budget grid and compliance

Freeze budgets as fractions of the measured always-acquire incremental cost:

```text
B = {0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00}.
```

`b=0.10` is the primary budget unless the pre-execution cost audit shows that no non-null action can
fit; that event is a design failure requiring a pre-metric amendment, not permission to inspect
performance. Policies use estimated costs to decide, but are plotted and judged using actual cost.
When an estimate would overshoot the remaining budget, the action is skipped and the policy stops;
cost overruns are reported, not relabelled into a more favourable budget.

Dense midpoint/four-frame and any future sampling density are different action definitions. Each
requires its own always/random/uniform baselines. Sampling density, aggregation, OCR deduplication,
reading order, long-text chunking and empty-modality handling are frozen globally, not selected per
arm.

For a joint OCR+dense policy, the denominator is the cost of acquiring all 60 registered actions;
for each action-specific G1 analysis, it is the cost of acquiring all 30 actions of that type. The
appendix must freeze one deterministic conversion from these normalized budgets to per-video
latency budgets, including how video-specific estimated always-acquire cost is obtained without
executing a candidate action. A mean corpus budget may not be retrospectively substituted for a
per-video budget, or vice versa.

## 7. Mandatory arms

All learned arms use the same base representations, classifier capacity envelope, outer/inner
folds, optimizer budget and action outcomes. Parameter counts and realized costs are reported.

| ID | arm | purpose |
|---|---|---|
| B0 | no acquisition | zero-cost floor |
| B1-O | acquire all 30 OCR actions | OCR full-information ceiling |
| B1-D | acquire all 30 dense actions | dense full-information ceiling |
| B1-J | acquire all 60 legal actions | joint full-information ceiling |
| B2 | random feasible actions, averaged over frozen draws | random-budget control |
| B3 | uniformly spaced feasible actions | coverage control |
| B4 | cheap visual/text-likelihood or salience top-k | cheap heuristic selector |
| B5 | classifier uncertainty/margin acquisition | uncertainty router |
| B6 | per-video all-or-none router | matched-cost coarse routing |
| B7 | MultiHateLoc-style hard temporal selector | in-domain selection control |
| B8 | evidence-type/relevance gate | relevance-versus-utility control |
| B9 | singleton/additive counterfactual-utility policy | non-set-conditioned CVoI control |
| B10 | set-conditioned sequential CVoI | proposed method |
| B11 | greedy predicted singleton benefit/cost | optimization control |
| B12 | cost-aware learned top-k/knapsack | admissible only with heterogeneous cost |
| O1 | label-aware sequential marginal-utility/cost oracle | evaluation-only oracle ceiling |
| O2 | exhaustive feasible subset on a pre-frozen reduced universe | evaluation-only subset ceiling |

B8 may use Claude/MLLM or gold evidence-type annotations only as train-fold privileged
supervision, never as outer-query inputs and never to select the headline. It must also include a
cheap non-MLLM relevance implementation. O1/O2 may use the query label only inside their explicitly
oracle evaluation routine; they never set a deployable threshold, arm or budget.

B11 uses the same legal pre-action inputs as deployable arms, predicts only the singleton target
`u(a | empty)`, freezes that prediction for the whole trajectory, and greedily orders feasible
actions by predicted singleton utility divided by estimated cost. It never reads realized benefit
or the query label. O1 instead recomputes the **realized**, label-aware marginal log-loss reduction
after every acquired action and greedily chooses its ratio to actual measured cost; it is the
binding sparse oracle. O2 enumerates all feasible subsets only after the appendix freezes a reduced
action universe small enough for exact enumeration, and is diagnostic rather than a gate input.
B12's solver, relaxation and inference rounding must be fixed in the appendix; if measured costs
fail the heterogeneity criterion in §5, B12 is marked inapplicable rather than silently becoming
another top-k arm.

The registered factorial additionally crosses acquisition on/off with retrieval on/off, and uses a
strong all-OCR fusion comparator. This prevents an OCR, fusion, extra-neighbour or retrieval gain
from being attributed to the acquisition policy. A width/parameter-matched policy control is
mandatory.

Every deployable selector B2--B12 is instantiated and frozen separately for OCR-only, dense-only
and joint action universes where applicable. These variants are trained in advance or produced in
mechanically sealed later-stage outputs; G1 only determines which already-specified universe may
bind G2--G5. It does not authorize redesign, hyperparameter changes or retraining a different
policy after seeing G1.

## 8. Endpoints and matched-cost Pareto inference

### 8.1 Primary endpoint

The primary endpoint is binary macro-F1 at `b=0.10`. The binding delta is

```text
Delta* = macroF1(B10) - max macroF1(admissible non-CVoI baselines)
```

where admissible comparators are B2--B8 and B11--B12 whose **actual mean measured cost is no greater
than B10's**. B0 and the applicable B1-O/B1-D/B1-J ceiling remain anchors. The strongest
comparator is reselected inside every bootstrap
replicate; freezing whichever baseline looks weakest after the run is prohibited.

The complete macro-F1-versus-cost curve and normalized area under the performance-acquisition-cost
curve over the frozen budget grid are co-primary efficiency summaries, but cannot rescue failure
of `Delta*`.

### 8.2 Secondary and mechanism endpoints

- balanced accuracy, accuracy, positive-class F1, AUROC, class recalls and predicted-positive rate;
- actual latency, calls, frames, GPU-seconds/energy and budget violations;
- realized held-out log-loss reduction from executing the acquired set;
- positive-utility capture, ranking regret and calibration against O1/O2;
- `u(a | S) - u(a | empty)` and B10-minus-B9 at matched cost, as the registered evidence for
  set-conditioning;
- per-action-type and acquisition-order results.

Evidence-type AUROC, temporal localization, OCR-presence prediction and the 22/28 audit strata are
diagnostics only. No diagnostic substitutes for global held-out classification and cost.

### 8.3 Pareto definition

At each candidate point, form the empirical upper envelope of every admissible baseline with no
greater measured cost. Recompute that envelope within each resample. A point Pareto-dominates only
if its paired macro-F1 delta has a 95% lower bound above zero and its cost is no higher under the
registered cost uncertainty analysis. Equivalently, a cost-saving claim at matched performance
requires the lower bound on saving to exceed zero. Point estimates alone are called
`Pareto-shaped`, never Pareto improvements.

## 9. Uncertainty

Use 10,000 paired hierarchical bootstrap replicates with a frozen seed. First resample frozen
groups, keeping every video/window/action together. Within a replicate, retain the complete paired
arm structure and average across all three split seeds and three refit seeds. Apply each stored
inner-OOF-selected threshold unchanged to its stored outer scores, then recompute macro-F1, actual
cost, comparator envelope and every delta. No threshold is re-estimated from resampled outer labels.
This group bootstrap is the primary sampling interval.

Training uncertainty is not fabricated by treating nine predictions of one video as independent.
Report split-seed and refit variance components and the nine complete-run deltas. As a sensitivity
analysis, a second hierarchical interval may resample complete split-seed/refit runs in addition to
groups. It must be labelled accordingly and cannot replace the primary interval after seeing which
is favourable.

All arms share bootstrap indices. Report percentile two-sided 95% CIs; binding one-sided lower
bounds use the lower endpoint specified below. Fold standard deviation is not a CI.

## 10. Ordered stage gates

Later gates run only after earlier gates pass. A measurement failure halts without counting as a
scientific negative.

### G0 — provenance, leakage, action and cost validity

Require: frozen group/fold hashes; zero test contact; exact 744/107 OCR ID coverage or explicit
missing records; frozen 22/28 ID lists and hashes; per-frame dense timestamps/features; per-action
costs; action replay; missing/failure policy; synthetic leakage, budget and Pareto-envelope fixtures;
and an independently reviewed freeze manifest. Failure: `HALT-MEASUREMENT`.

### G1 — action headroom

For each action type separately, compare its matched always-acquire arm with B0 on train outer OOF.
An action is live only if macro-F1 improves by at least **+0.015** and its paired 95% lower bound is
above zero. The joint action universe remains live only if at least one type passes; a type that
fails cannot bind a later gate. Any already-produced downstream outputs for that type remain sealed;
no new learned utility policy is fitted for it after G1. If all fail:
`NO-GO-ACTION-HEADROOM`.

### G2 — sparse ceiling

At the primary cost, binding oracle O1 must recover at least **80%** of the corresponding
always-acquire gain
over B0 and beat the matched-cost non-oracle envelope by at least **+0.020** macro-F1. Otherwise:
`NO-GO-SPARSE-HEADROOM`. This licenses policy learning, not an oracle performance claim.
O2 is a diagnostic exactness check on its pre-frozen reduced universe and cannot pass or fail G2.

### G3 — learnable CVoI

B10 must satisfy all on train outer OOF at the primary budget:

1. `Delta* >= +0.010` macro-F1;
2. paired 95% lower bound of `Delta* > 0`;
3. realized log-loss reduction versus B0 is positive with paired 95% lower bound above zero;
4. B10 exceeds B9 by at least +0.005 macro-F1 with a positive paired delta point estimate;
5. the delta is positive in at least 7 of 9 complete split-seed/refit runs.

Failure: `NO-GO-CVOI`. Relevance/type AUROC cannot overturn it.

### G4 — efficiency/Pareto

B10 must recover at least **80%** of the applicable B1-O/B1-D/B1-J always-acquire macro-F1 gain at
no more than **20%** of that ceiling's measured incremental cost, and Pareto-dominate the baseline
envelope at the primary budget and at least one adjacent non-zero frozen budget. Failure:
`CVOI-PREDICTIVE-BUT-NOT-EFFICIENT`; no compute saving or cost-effective claim.

### G5 — untouched confirmation

After G1--G4 are frozen, HateMM-val labels and performance-bearing artifacts are read once. Require
positive B10 delta over the frozen matched-cost comparator and at least 70% recovery of the
train-OOF applicable always-acquire gain. A second dataset, if fully preregistered, must also show a
positive matched-cost delta for a cross-dataset claim. Confirmation never tunes the method.
Failure: `TRAIN-OOF-ONLY` or `HATEMM-SPECIFIC`.

## 11. Overall verdict and claims

- G0 failure: `HALT-MEASUREMENT`.
- G1 failure: OCR/dense acquisition has no registered incremental value under this representation.
- G1 pass but G2 failure: information exists only at too much cost; sparse acquisition is unsupported.
- G2 pass but G3 failure: a sparse oracle exists, but CVoI is not learnable from legal pre-action inputs.
- G3 pass but G4 failure: CVoI improves a fixed-budget prediction, but efficiency/Pareto is unsupported.
- G1--G5 pass: `GO-CVOI`, authorizing a separate sealed-test full-method evaluation.

A full pass supports only:

> On the registered data, actions, hardware and representation, a cross-fitted set-conditioned
> CVoI policy selects temporal evidence actions that improve hateful-video classification over
> the strongest measured-cost comparator and recover most of the always-acquire gain at lower
> measured cost.

It does not support: causal necessity of OCR; creator evasion or intent; SOTA without a same-protocol
sealed test; human-quality or novel evidence-type annotation; multilingual/general deployment;
knapsack under uniform costs; compute saving when full features were actually computed before
masking; or benefit from retrieval without the registered factorial.

Negative results are bounded to the registered actions, costs, representation, learners, datasets
and budgets. They do not prove that all active acquisition is ineffective.

## 12. Gold, Claude and oracle firewall

Gold spans, Gate-C annotations, Claude labels, `required_modalities`, the 22/28 membership and
oracle actions are held by evaluation-only loaders with explicit provenance flags. Deployable
training asserts that these objects were never imported. They may support frozen descriptive
tables or B8's explicitly privileged train-fold ablation only.

No gold/Claude-defined subset may choose the headline method, primary action type, budget, cost
proxy, threshold, stopping rule or claim. A global failure remains a failure even if one such
subset improves. Any post-hoc subgroup is labelled exploratory and requires an independent cohort.

## 13. Pre-execution completeness gate

This preregistration is not executable until an implementation appendix and freeze manifest pin:

1. exact OCR and dense asset paths, schemas, ID counts and SHA256 values;
2. per-frame dense re-extraction code and timestamps; per-action OCR/dense cost instrumentation;
3. duplicate/repost/template/creator inputs, grouping algorithm, components and fold assignments;
4. three split seeds, three refit seeds, deterministic fold fallback and class/group counts;
5. base classifier, feature/fusion contracts, action representations and missing-value tokens;
6. legal pre-action features and explicit forbidden-feature assertions;
7. state/prefix sampling distribution and leakage-free CVoI-target construction;
8. every arm's architecture, parameter count, optimizer, epoch grid and selection rule;
9. primary cost hardware, timing protocol, budget feasibility and overrun handling;
10. bootstrap implementation, baseline-envelope recomputation and all gate/verdict code;
11. frozen 22- and 28-video ID lists, definitions and hashes;
12. synthetic fixtures proving group isolation, inner-OOF targets, threshold isolation, label-free
    policy inference, exact budget accounting, oracle isolation and comparator reselection;
13. test-path denial and contact ledger;
14. one clean end-to-end synthetic rehearsal producing no real candidate metric;
15. independent review and one payload hash covering prereg, appendix, code, configs and fixtures.

Only after all fifteen pass may the formal campaign begin. Each ordered gate may have one
predeclared submission, but no gate may be rerun after its metric is observed and no later-stage
submission may occur unless the preceding gate passes. If all arms are executed in one submission,
later-gate outputs remain mechanically sealed unless their predecessor passes. GPU work follows the
repository SLURM rules; no job sets `--time`. Any result observed before the freeze is pilot evidence
and cannot enter the registered verdict.

## 14. Deviations

Before any affected metric, a discovered implementation defect may be fixed only by a timestamped
deviation that states the defect, directional consequences, exact patch and new payload hash. After
an affected metric is observed, the result remains archived and labelled invalid for the registered
claim; a silent rerun is prohibited. Cosmetic corrections that cannot change execution or verdict
are logged but do not reopen scientific choices.

## RESULTS

Not run. No candidate metric has been computed under this protocol.
