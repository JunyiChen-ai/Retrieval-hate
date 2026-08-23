# EXP: TERA Gate-0 — temporal-evidence headroom and compositionality

> **Status: PRE-REGISTERED DESIGN ONLY (2026-08-07).** No implementation, asset
> restoration, feature extraction, metric computation, or experiment is authorized by
> this document. The decision rules below are frozen before any TERA candidate metric is
> computed. Results, if the study is authorized and run later, must be appended under
> `## RESULTS` without rewriting the registered design. A correction that changes an arm,
> endpoint, threshold, split, or decision rule is a deviation and follows §12.

## 0. Purpose and relation to prior negative evidence

TERA (Temporal Evidence Retrieval and Aggregation) asks whether hateful-video detection
benefits from retrieving and composing **temporally localized evidence**, rather than
retrieving one whole-video embedding. Gate-0 is a falsification screen, not the final
TERA method evaluation.

The screen does **not** repeat the prior MHC segment-mode experiment. That experiment used
K=4 windows with parent-video pseudo-labels and found a language-dependent result
(MHC-EN about +0.015 macro-F1; MHC-ZH about -0.066). Both datasets lack gold temporal
spans, and the repository records the inherited segment-supervision channel as the lesion.
Consequently:

- MHC-EN/ZH are not primary Gate-0 development datasets;
- no positive result may be obtained by treating every segment of a positive MHC video as
  a positive segment;
- HateMM, which has official `hate_snippet` spans, is the primary diagnostic dataset;
- HateClipSeg is a secondary gold-segment confirmation dataset, with its label-domain and
  surviving-subset caveats kept explicit.

The sequence is binding: **C (error opportunity) → A (temporal headroom) → B
(multi-segment complementarity)**. Failure at a stage stops all later stages.

## 1. Research questions and hypotheses

### RQ-C — Is there a large enough correctable error population?

**H-C:** At least a material fraction of held-out development false negatives are caused
by short, cross-modal, or temporally distributed evidence that a whole-video representation
can plausibly dilute. The hypothesis is about error coverage, not performance.

### RQ-A — Does temporally localized evidence contain usable classification headroom?

**H-A1 (availability):** A non-deployable oracle restricted to localized gold evidence
substantially outperforms the matched whole-video/global-pooling baseline.

**H-A2 (recoverability):** A deployable selector trained without segment/span labels
recovers a non-trivial, statistically supported part of that oracle headroom.

### RQ-B — Is more than one temporal evidence unit necessary?

**H-B:** An interactive, ordered two-segment model outperforms both the strongest
single-segment model and a non-interactive top-two control, and loses its advantage when
the second segment or its temporal relation is destroyed.

Gate-0 does **not** test the stronger claim that one selected segment is a protected target
and the other is an attack/stance segment. Neither HateMM nor HateClipSeg supplies those
segment-role annotations. That terminology is prohibited unless a separately
pre-registered, reliability-checked role annotation study is completed.

## 2. Data roles, frozen boundaries, and test sealing

### 2.1 HateMM (primary)

- Use the canonical HateMM video-level split records (or the IDs/labels embedded in the
  canonical feature caches) and the official parsed `hate_snippet` spans in
  `data/gt/HateMM/hate_spans.json`. The current checkout does **not** contain
  `data/gt/HateMM/{train,val,test}.jsonl`; an implementation must therefore hash and
  record the actual canonical split source it uses and exact-match its IDs and labels to
  the feature cache. It must not silently assume those three local paths exist.
- Gate-C and all nested-OOF development use **train only**. The existing validation split
  may be used once to confirm the stage verdict after all choices are frozen; it may not
  select an arm, threshold, epoch, granularity, or decision rule.
- The official HateMM test set is sealed throughout Gate-0. It is reserved for a later
  full-method study and is not consumed by this screen.

### 2.2 HateClipSeg (secondary confirmation only)

- Surviving corpus: the frozen 395-video (90.8%) subset documented in
  `DATASET_hateclipseg.md`; exact IDs must be hashed in the run manifest.
- Use the already-declared `data/gt/HateClipSeg/p11_split.json`. Train is development,
  val is a one-time secondary confirmation, and test remains sealed.
- The binding binary endpoint is specifically **has at least one segment labelled
  hateful** versus no segment labelled hateful (180/395 positives on the frozen surviving
  corpus, recomputed from `gold_segments.json` at registration). The P11 split was
  stratified by `has_toxic_second`, not by this endpoint; the run must report the binding
  endpoint counts in each split and halt the confirmation as underpowered if either class
  has fewer than 10 videos in validation. `any-toxic` (which also includes insulting,
  sexual, violence, and harm) is a separate descriptive endpoint and cannot replace a
  failed binding confirmation.
- Absolute results are not comparable to the original 435-video corpus because attrition
  is non-random. Only matched method comparisons on the identical surviving IDs are valid.

### 2.3 Historical-contact disclosure and prospective seal

Both datasets have appeared in earlier repository studies, and aggregate test metrics and
some historical predictions may exist. This is not a claim that the datasets are
historically untouched. The prospective TERA rule is stricter: after this registration,
TERA design, implementation, debugging, arm selection, and thresholding must not load any
official test labels, predictions, spans, or per-example artifacts. A test-contact counter
and a list of opened paths are mandatory provenance fields. Historical headline numbers
must not serve as Gate-0 comparators.

## 3. Temporal units and supervision boundary

The registered temporal grid is **K=30 uniform timestamp windows per video**. It matches
the existing HateClipSeg localization granularity and is materially finer than the failed
K=4 segment mode. Boundaries are `[kD/K,(k+1)D/K)` using the probed media duration `D`;
the last interval is closed at `D`. Feature sampling must preserve these timestamp
boundaries. K=4 may be reported only as a historical/coarse descriptive control and cannot
promote the route.

All deployable arms obey the **weak/no-span** boundary:

- allowed supervision: video-level binary label, raw input modalities, and train-fold
  videos;
- forbidden supervision: gold spans, gold segment labels, test/held-out labels, target
  roles, or an MLLM/teacher output that was prompted with gold spans or labels;
- all segments from one video remain in the same fold;
- if a retrieval memory is used, the held-out video's segments and all derivatives are
  absent from that memory.

Gold spans may be read only by Gate-C annotation/evaluation and the explicitly named
non-deployable oracle/evaluation routines. They must not affect feature extraction,
training, checkpoint selection, thresholds, or retrieved supports.

## 4. Gate-C — blinded error-coverage audit

### 4.1 Prediction source and sampling

Generate one fully OOF whole-video baseline prediction for every HateMM-train video using
the outer folds in §7. Gate-C samples from these predictions; it does not use val/test.
Audit all false negatives if there are at most 120. Otherwise sample 120 false negatives,
stratified equally by baseline positive-class score tercile. Add 30 true positives and 30
false positives, sampled by score tercile, as controls. Sampling seed is `20260807`.

When false negatives are subsampled equally across score terciles, coverage estimates and
their confidence intervals are population estimates over **all** OOF false negatives:
each audited item is weighted by its tercile's full-population count divided by its sampled
count. The video bootstrap resamples within tercile and reapplies these frozen weights.
Unweighted audit-sample proportions are reported only as diagnostics. Controls never enter
the false-negative coverage denominator.

Annotators see the video, transcript if it is an ordinary model input, and official span
overlay, but not the model score, correctness category, retrieval output, or TERA output.
At least 20% of audited videos are independently double-coded; disagreements are retained
and then adjudicated. Report raw agreement and Cohen's kappa.

### 4.2 Frozen taxonomy

Each video receives one primary cause and zero or more secondary causes:

1. `short_localized`: one temporally short evidence interval is sufficient;
2. `multi_segment_complementary`: at least two non-contiguous intervals are jointly
   necessary and no recorded single interval is sufficient;
3. `cross_modal`: evidence requires more than one modality;
4. `quotation_or_counterstance`;
5. `external_knowledge`;
6. `global_evidence`: evidence is distributed/global rather than localized;
7. `annotation_ambiguity_or_noise`;
8. `representation_failure_other`.

Also record minimal sufficient intervals, required modalities, whether one interval is
sufficient, span/video duration ratio, and confidence (`high/medium/low`). Annotation
instructions and the blank form must be hashed before labels are entered.

### 4.3 C decision

C passes only if all are true on false negatives (a listed mechanism counts in the union
when it is either the primary cause or a recorded secondary cause; the
`multi_segment_complementary` criterion likewise uses primary-or-secondary presence):

- union of `short_localized`, `multi_segment_complementary`, and `cross_modal` is at least
  30%, with a video-bootstrap 95% CI lower bound at least 20%;
- `multi_segment_complementary` alone is at least 15%;
- `annotation_ambiguity_or_noise` is at most 25%;
- double-coded primary-cause Cohen's kappa is at least 0.60.

Failure means **NO-GO-C**: do not run A or B, and do not claim temporal evidence is a
large enough performance lever. A reliability failure (`kappa < 0.60`) is a measurement
failure, not evidence against the hypothesis; it still stops the registered route.

## 5. Gate-A — availability and recoverability

### 5.1 Matched arms

All arms use the same frozen segment encoder/features, outer folds, training examples,
optimizer budget, and video-level labels. Only aggregation/selection changes.

| ID | arm | deployable | role |
|---|---|---:|---|
| A0 | whole-video/global mean representation + linear classification head | yes | primary baseline |
| A1 | mean of the K=30 segment representations + same head | yes | segmentation/no-selection control |
| A2 | max/top-k MIL pooling; `k ∈ {1,2,4}` selected in inner OOF | yes | simple selector |
| A3 | learned attention pooling with video-level BCE only | yes | primary recoverability arm |
| A4 | log-sum-exp pooling; temperature in `{0.1,0.3,1.0}` selected in inner OOF | yes | smooth-selection control |
| O1 | gold-span pooling of the **fixed fold-trained segment scores** | no | localization-availability oracle |
| O2 | true-label-aware best candidate subset from fixed fold-trained scores | no | candidate-set ceiling/sanity only |

O1 applies one deterministic rule to every video: pool windows having positive-duration
overlap with an annotated hate span; if no such window exists, fall back to registered A1
mean pooling. The implementation may inspect span presence but may not branch directly on
the video label. Because span availability is itself gold information and is correlated
with the label, O1 remains explicitly non-deployable and is reported only as an upper
bound. O2 may choose max for a positive and min for a negative and is therefore explicitly
label-leaking; it can diagnose a missing candidate but can never be presented as model
performance. Neither oracle may select a deployable arm.

No gold-span-trained arm is permitted in Gate-A. Such an arm would answer a different,
fully supervised localization question and would contaminate the intended weak/no-span
claim.

### 5.2 A decision

Let `D` be the strongest deployable temporal arm among A2–A4, selected solely by pooled
inner-OOF macro-F1. A passes only if all are true on HateMM-train outer OOF:

1. `O1 − max(A0,A1) ≥ +0.050` macro-F1;
2. `O2 − max(A0,A1) ≥ +0.050` macro-F1 (candidate availability sanity);
3. `D − max(A0,A1) ≥ +0.020` macro-F1;
4. the paired video-bootstrap 95% CI for item 3 excludes zero;
5. D has **mean within-video second-level AUROC** at least 0.60 over hateful videos
   containing both positive and negative evaluated seconds, and exceeds the matched A0
   video-score broadcast by at least 0.03. The eligible video set is frozen once from gold
   spans and shared across arms; a video-broadcast score is defined as AUROC 0.5 within
   each eligible video. A pooled hate-video-only AUROC is not an admissible substitute
   because it retains between-video separability;
6. on the one-time HateMM-val confirmation, item 3 remains positive; and on
   HateClipSeg-val the matched `has-hateful-segment` delta is positive. These confirmation
   checks do not require significance because the validation sets are small.

If O1/O2 fail, verdict is **NO-GO-A-NO-HEADROOM**. If the oracles pass but D fails, verdict
is **NO-GO-A-SELECTOR**: localized evidence exists, but the registered weak selector cannot
recover it. Neither result authorizes B.

## 6. Gate-B — multi-segment complementarity

B runs only after a full A pass. It freezes D as the segment-scoring/selection basis and
does not reopen A hyperparameters.

| ID | arm | purpose |
|---|---|---|
| B0 | strongest single selected segment | single-evidence baseline |
| B1 | non-interactive mean of the top two segment representations | extra-evidence/no-interaction control |
| B2 | ordered top-two representations plus an interaction term and relative-time encoding | primary pair model |
| B3 | width-matched single-segment model with parameter count within 5% of B2 | capacity control |
| B4 | B2 with pair order randomly permuted within video | temporal-order lesion |
| B5 | B2 with the second segment replaced from a different legal-support video whose **D-predicted** video label matches the query's D-predicted label | second-evidence lesion without query-label leakage |

Top-two selection is learned using video labels only. A minimum separation of two K=30
windows is required; adjacent-window duplicates are ineligible. B4/B5 use fixed seed
`20260807`, and replacements never cross into the held-out outer fold. B5 matching uses
only D's frozen predictions, never the query's true label; if the matching support stratum
is empty, draw from all legal-support videos and record the fallback.

B passes only if all are true on HateMM-train outer OOF:

- `B2 − max(B0,B1,B3) ≥ +0.020` macro-F1 and its paired bootstrap 95% CI excludes zero;
- B2 exceeds each of B4 and B5 by at least +0.015 macro-F1;
- on the frozen Gate-C `multi_segment_complementary` subset, B2 rescues at least 20% of
  B0 false negatives without increasing false positives on that subset by more than 10%;
- the B2 delta remains positive on HateMM-val and HateClipSeg-val.

Failure is **NO-GO-B**. A B pass licenses development of a temporal-composition method, but
not a target–attack factorization claim.

## 7. Nested OOF and model-selection protocol

Primary development uses **5 outer video-stratified folds**, shuffled with seed `20260807`.
Within every outer-training partition, use **4 inner video-stratified folds**, seed
`20260808`. If either class has fewer members than the fold count, stop with a measurement
failure rather than silently changing folds.

For each outer fold:

1. fit preprocessing, normalizers, retrieval memory, and model on outer-train only;
2. select arm-local hyperparameters, epoch, and decision threshold from pooled inner-OOF
   predictions only;
3. refit once on full outer-train using the selected fixed epoch/budget;
4. emit scores once for outer-query videos;
5. assert every video appears in exactly one outer-query fold and no video/segment/derived
   identifier overlaps outer train and query.

The primary metric is computed by concatenating all outer-query predictions, not by
averaging fold metrics. The same folds are shared across arms. Encoder fine-tuning is out
of scope for Gate-0; if later introduced, its training and checkpoint selection must also
be nested and requires a new preregistration.

## 8. Metrics and statistics

### 8.1 Primary detection metrics

- binary macro-F1 (primary);
- balanced accuracy, accuracy, positive-class F1, and AUROC (secondary);
- predicted-positive rate and confusion matrix (diagnostic).

Thresholds are selected only from inner-OOF macro-F1, with ties resolved by threshold
closest to 0.5 and then the smaller numeric threshold. No 0.5/test-threshold switching is
allowed after results are seen.

### 8.2 Temporal metrics

- HateMM mean within-video second-level AUROC on hateful videos containing both classes
  (primary temporal metric; also report the number of eligible videos and a video-level
  bootstrap CI);
- pooled full second-level AP/AUROC (secondary; known to include video-level separation);
- gold-span recall@1, @2, and @4 selected windows;
- selected-vs-unselected score separation and within-video score standard deviation.

Seconds follow the existing midpoint rule: second `t` is positive when `t+0.5` falls in a
gold span. Per-video AP is not averaged because videos without both classes make it
undefined.

### 8.3 Uncertainty

Use 10,000 paired bootstrap resamples of **videos**, seed `20260809`, stratified by video
label. All segments/seconds from a sampled video travel together. Report percentile 95%
CIs for absolute metrics and paired deltas. Macro-F1 is recomputed inside each bootstrap;
fold standard deviation is not a confidence interval. Report exact numerator/denominator
for coverage and rescue rates and Wilson intervals where appropriate. No multiple-testing
adjustment is used because one ordered decision path and one promoted arm are binding;
all non-binding arm comparisons are descriptive.

## 9. Overall verdict and stopping rule

- C fails → stop, `NO-GO-C`.
- C passes but A fails → stop with the applicable A verdict.
- C and A pass but B fails → `TEMPORAL-SELECTION-SUPPORTED; COMPOSITION-NOT-SUPPORTED`.
- C, A, and B all pass → `GO-TERA`: authorize a separate full-method preregistration.

There is no partial-credit promotion and no substitution of accuracy, positive-class F1,
full-frame AP, an oracle, or a secondary dataset for a failed binding criterion.

## 10. Claim boundaries

A full Gate-0 pass supports only:

> On the registered datasets and frozen representation, weakly supervised temporal
> selection recovers useful localized evidence, and interaction between two separated
> evidence units improves video-level hate detection under the registered controls.

It does not support claims of:

- state-of-the-art performance;
- generality across languages, platforms, or all hateful-video datasets;
- target–attack, speaker–target–stance, causal, or counterfactual identification;
- accurate temporal localization outside the registered span protocols;
- benefit from retrieval specifically (Gate-0 tests evidence selection/composition; a
  retrieval contribution requires its own matched ablation);
- a deployable benefit from O1/O2 or any use of gold spans.

A negative result falsifies only the registered granularity, frozen representation,
weak/no-span learners, data, and thresholds. It does not prove that all temporal methods
are ineffective.

## 11. Artifact and provenance contract

### 11.1 Pre-execution completeness gate

This document freezes the scientific arms and decision rules but deliberately does not yet
specify enough implementation detail to define A2--A4 or B2 uniquely (including exact
pooling equations, attention/pair architecture, dimensionality, initialization seeds,
optimizer, learning-rate/regularization grids, epoch cap, early-stopping rule, and the
construction of the within-5% B3 capacity match). **No candidate metric may be computed**
until a versioned implementation appendix and `frozen_config.json` fill every one of these
fields, give deterministic synthetic fixtures for every arm and lesion, and are hashed
before execution. That appendix may instantiate the already listed arms but may not add or
remove an arm, change an endpoint/threshold/split, or inspect candidate results. Any such
scientific change is a material deviation under §12.

The completeness gate must additionally define whether max/top-k/LSE operate on scalar
segment logits or vector representations and use the same definition in training and
evaluation. For B4, "order permuted" must be a genuine lesion (swap the selected pair's
order/relative-time encoding while retaining both segment contents), not an arbitrary
permutation that sometimes leaves order unchanged. For B5, train and held-out-query
predictions must each use replacements drawn only from their legally available support
partition; no held-out label may be used to choose a replacement at inference.

If run, use an immutable `artifacts/tera_gate0/<run_id>/` namespace containing:

```text
frozen_config.json
split_manifest.json
feature_manifest.json
annotation_protocol.json
gate_c_audit.jsonl
folds/fold_<k>/{train_ids,query_ids,selected_hparams}.json
folds/fold_<k>/{segment_scores,selected_evidence,video_predictions}.jsonl
oof_predictions.jsonl
oracle_predictions.jsonl
metrics.json
bootstrap_indices.npz
verdict.json
manifest.json
```

`manifest.json` must include git commit and dirty-state, host/execution environment,
Python/conda environment, GPU model if used, command line, random seeds, SLURM/Modal/local
job identifier, start/end timestamps, input/output SHA256, split-ID hashes, exact surviving
ID hash, encoder/checkpoint hash, duration and boundary rules, decode/zero-vector failures,
test-contact count and opened test paths, and overlap assertions. `frozen_config.json` must
hash itself via a canonical payload hash and outputs are non-overwriting.

Each prediction row records at minimum video ID, dataset, outer fold, gold video label,
score, prediction, threshold and its inner-OOF source, selected segment IDs and second
intervals, arm, seed, and gold overlap fields (the latter populated only by evaluation).
Every artifact containing gold-span-derived fields is marked `oracle_or_eval_only: true`.

## 12. Deviations and corrections

- Typographical/documentary corrections that cannot change execution or verdict are
  appended to a dated errata subsection and do not require rerun.
- Any change to data IDs, split, temporal grid, feature family, arm, gold boundary,
  hyperparameter grid, metric, CI, success threshold, or stopping rule is a **material
  deviation**. Stop before computing the affected candidate metric, append the reason and
  expected directional effect, freeze a new version/hash, and restart the affected stage.
- If a material defect is discovered after unblinding, the original registered verdict is
  retained and labelled invalid; the corrected run is explicitly post-deviation and cannot
  be represented as the original preregistered confirmation.
- Missing/corrupt assets, decode failures above 1% of videos, train/query overlap, gold-span
  access in a deployable path, or any prospective test contact cause immediate HALT. They
  are implementation/provenance failures, not performance negatives.
- Clean registered failures are accepted without rule changes. No threshold relaxation,
  arm substitution, dataset substitution, or post-hoc "near pass" promotion is allowed.

## 13. Resource plan (not authorization to execute)

1. **C:** CPU/manual audit using OOF predictions; no raw-video upload to cloud.
2. **A asset audit:** read-only inventory first. Reuse a valid K=30 timestamp-aligned cache
   only if its IDs, boundaries, encoder, and hashes satisfy §11. Missing assets are not
   silently regenerated.
3. **A/B heads:** frozen-feature heads are expected to be CPU or short single-GPU jobs.
   Probing/triage should use Modal only when it needs no raw video, subject to the current
   `CLAUDE.md` cloud rules. Any formal cloud comparison reruns candidate and paired baseline
   on the same GPU model and image; cloud and local results are never mixed in one table.
4. Tasks reading raw videos or extracting features remain local. Before any local GPU work,
   check the actual GPU count and queue: a one-GPU machine may run directly; otherwise use
   SLURM, `conda activate HateVideo`, `sbatch scripts/slurm/<name>.sbatch`, and no `--time`.
5. Respect the user limit of 16 CPU / 128 GB / 2 GPU. Initial `PENDING (JobHeldUser)` is
   allowed to release automatically and is never manually forced.
6. Formal GPU runs, if reached, require one frozen submission per registered stage. No
   speculative full TERA training is authorized by Gate-0.

---

## REGISTERED DEVIATIONS / ERRATA

None at registration.

### Close-out back-fill (2026-08-07)

Everything above the `---` separator preceding the `REGISTERED DEVIATIONS / ERRATA` heading
is the registration text and is **unchanged, byte for byte**, from the version whose sha256
`f6c1ce6c652bcedd18451d4ee3a490ca2c72c603489e89c6a161855537ed6e98` is embedded in both frozen
payloads (`7ba80eaf…`, `f2caade9…`) and was re-verified at the start of Run 1 and Run 2. This
subsection and the `RESULTS` subsection below are appended at **campaign close-out**, after the
terminal verdict, exactly as `refine-logs/TERA_GATE0_DEVIATION_D1_2026-08-07.md` §3 anticipated
("the documentary back-fill into the prereg's `REGISTERED DEVIATIONS / ERRATA` subsection is
therefore deferred to campaign close-out"). Appending changes this file's digest away from
`f6c1ce6c…`; no registered execution remains to be launched under either frozen payload, and any
future re-execution would require a fresh freeze under §12 in any case.

#### D-1 — stage A is necessarily executed twice

- Record: `refine-logs/TERA_GATE0_DEVIATION_D1_2026-08-07.md`
  (sha256 `0eb9c2c7344a426bc6a6a8a791762e9e244fa85e543c3c1333dfd974c7826255`).
- Registered 2026-08-07, **before** any Gate-A/B/C/temporal number existed
  (`artifacts/tera_gate0/` then held only `_fixtures/`). Authority: main-conversation
  adjudication.
- Cause: the hash-frozen harness couples the gates through in-process state, not on-disk
  artefacts. Stage C raises `HALT_STAGE_ORDER` unless stage A ran in the same process; the
  confirmation protocol likewise reads live stage-A state; stage B needs `msc_ids`, produced
  only by stage C under `--gate-c-audit`, i.e. only after human/model annotation exists. There
  is therefore no schedule in which stage A runs exactly once.
- Consequence: **Run 1** (`--stages A,C --confirmation none`) is the prediction-source run that
  emits the Gate-C sampling frame and blank annotation package; **Run 2**
  (`--stages A,C,B --gate-c-audit … --confirmation all`) is the registered decision run.
- Expected directional effect on any registered endpoint: **none** — an execution-count
  artefact with a determinism proof (fixture F13; same frozen config, harness bytes, seed
  register and input caches, so Run 2's stage-A output is bit-determined before Run 1 launches).
- **Isolation clause (D-1 §2), binding here.** Run 1's `verdict.json` is void *ab initio*; the
  A1/A2/A3/A4/O1/O2/arm-`D`/temporal/bootstrap sections of Run 1 may not be read, quoted or
  acted on (the A0 confusion matrix is exempt — it is the Gate-C sampling frame). §2 clause 4:
  **if Gate-C returns NO-GO, Run 1's `metrics.json` and `verdict.json` remain sealed.** Gate-C
  returned NO-GO; they are sealed permanently. The same seal is applied to Run 2's Gate-A and
  Gate-B sections — see the sealing declaration in `RESULTS` below.

#### D-2 — the Gate-C annotators are Claude Opus 5 agents, not humans

- Record: `refine-logs/TERA_GATE0_DEVIATION_D2_2026-08-07.md`
  (sha256 `0c21e04d15c921937351bebdcd993fc28b5914243d0511d7351e68b28968bd25`).
- Registered 2026-08-07, **before** any Gate-C label was produced (`gate_c_audit.jsonl` did not
  exist; no coverage, kappa or msc quantity had been computed). Authority: user adjudication,
  including two recorded revisions (DUA frame-exposure exemption widened from single-item to
  general; second coder changed from a Qwen model to a second Claude instance). Registration
  basis: §12, registered before the affected stage begins with reason and expected direction.
- What is executed: every §4 label is produced by a Claude Opus 5 agent —
  `claude-opus-5-c1` (primary coder, all 133 audited items), `claude-opus-5-c2` (second coder,
  the 27 double-coded items registered in `gate_c_sample.json["double_coded"]`),
  `claude-opus-5-adj` (adjudicator, only double-coded items whose two `primary_cause` values
  disagree — 5 items). Each item is labelled by a separate instance with no shared context; c2
  never sees c1's output or its existence. The coordinating process makes no labelling
  judgement and computes no reliability, coverage or decision quantity.
- Registered expected directional effects, stated before any number existed: **kappa biased
  upward** (two draws from one model, not two annotators); **union coverage biased downward**
  (non-speech audio evidence cannot be perceived), which makes Gate-C *harder* to pass and so
  makes a pass conservative w.r.t. the 30% / CI-20% / msc-15% thresholds. No threshold,
  taxonomy entry, sampling weight, seed, decision rule or harness byte is changed by D-2.
- The observed kappa **passed** its 0.60 bar and the observed union **passed** its 30%/20%
  bars; the criterion that failed is `msc >= 0.15`, for which D-2 registers no directional
  claim.

#### D-3 — `msc_subset` dropped agreeing double-coded videos (frozen-byte defect)

- Record: `refine-logs/TERA_GATE0_DEVIATION_D3_2026-08-07.md`
  (sha256 `ae252f569e7dc0b6d7a9179b5f948e20d222db4a287bf4bfd14cc29ccb008033`);
  re-freeze record `refine-logs/TERA_GATE0_REFREEZE_2026-08-07.md`
  (sha256 `842df40ebd08bc63edc7cdbc5dc82ecbff473038d2a3aa2dfd8e8b53816d044f`).
- Registered 2026-08-07, after Gate-C annotation was assembled and **before** Run 2 was
  submitted. At registration `msc_subset.json` did not exist in any run directory and no msc
  subset, rescue rate, FP side-condition count, Gate-B decision or Gate-C coverage/kappa
  quantity had ever been computed. Found by code reading, not by inspecting a result. This is
  the §12 "stop before computing the affected candidate metric" path, **not** a
  post-unblinding correction; no verdict is invalidated.
- Defect (frozen bytes, `run_gate0.py:817-819`): the row filter feeding `gc.msc_subset`
  admitted a row only if it was an adjudication row or its video had exactly one row. A
  double-coded video whose two coders **agreed** has two rows and no adjudication row, so
  neither row passed and the video was dropped from the msc subset regardless of its cause —
  narrowing a registered denominator by ~17% of audited videos on a coding-process artefact
  orthogonal to the scientific quantity.
- Fix (predeclared before any edit): a shared `resolve_audit_rows` helper in `gate_c.py`
  implementing adjudicated-else-first resolution, with `msc_subset` taking raw rows and
  resolving them itself, so the coverage path and the msc path cannot diverge. Candidate pool
  grows from 111 to the full 133 audited videos.
- Registered directional effect: **Gate-B only** — rescue rate can move either way, the FP
  do-no-harm guard can move either way, and no neutrality is claimed; the point is that the
  post-fix evaluation is the registered one. **Gate-A, Gate-C, temporal metrics and the
  confirmation protocol are unaffected and the Gate-C coverage/kappa path is bit-identical.**
  The Gate-C verdict below is therefore invariant to D-3.
- Freeze consequence: harness and appendix edits changed `payload_sha256`, hence the `run_id`
  prefix, `7ba80eaf…` → **`f2caade97712f8421232dee0a9c6b02545e3ac9ce95357e82e664802316a81e0`**
  (appendix v3 → v4). Release evidence per the project's proportional-ceremony rule was author
  self-test: fixture battery v2 **16/16 PASS** (`fix-20260807T083546Z`, 75 assertions), with
  three new F11 assertions shown to **FAIL on the v3 bytes** and pass on v4, i.e. they test the
  defect rather than merely the code.

#### Post-run errata (discovered after the verdict; no effect on it)

These two are documentary. Both were found after Run 2 completed and after the verdict was
read, so neither can have influenced any registered decision; both are recorded because §12
requires defects in registered inputs to be written down, and because a future re-execution
would have to handle them.

1. **The confirmation set was consumed by a run that stopped at C.** `run_gate0.py`'s
   orchestration executes `run_confirmation()` under `if self.args.confirmation != "none":`
   unconditionally — it is not gated on the Gate-C outcome, and §9's "C fails → stop" is
   enforced only in the reported verdict, not in the harness control flow. Run 2 was launched
   with `--confirmation all`, so the confirmation passes were spent:
   `manifest.json` records `confirmation_unlock_utc = 2026-08-07T09:25:22Z` and
   `confirmation_passes = {"hateclipseg_val": 1, "hatemm_val": 1}`, and
   `confirmation_predictions.jsonl` / `confirmation_summary.json` exist in the Run 2 directory.
   This does **not** touch test data (`test_contact_count = 0`, `opened_test_paths = []`); the
   consumed budget is the §7.10 val-side confirmation allowance. Because the run stopped at C,
   the consumed confirmation supports no registered claim, and its outputs fall under the
   Gate-A/B seal declared below. Any future re-execution of this design must either re-register
   the confirmation budget or launch the decision run with `--confirmation none` until Gate-C
   has passed.
2. **Stage B is structurally unreachable in a non-fixture run.** The stage-B branch reads
   `if not self.gate_a["pass"] and not self.args.fixture_mode: note("stage B skipped …")`,
   and the `else` branch — the only place `self.forced_stage_b` is assigned — is entered only
   when `self.gate_a["pass"]` is true, so `forced_stage_b` can never be true outside fixture
   mode; it is dead code. Consequently `"B"` in `--stages` runs stage B only if Gate-A passed.
   In Run 2 `stages_run = ["A","C","B"]` while `gate_b = null` and `forced_stage_b = false`:
   stage B was requested and skipped. **No effect on this verdict** — §9 stops at a Gate-C
   failure and forbids B from carrying any decision, so a Gate-B number would have been
   inadmissible regardless. If this design is ever re-run, this is a **D-4-class material
   deviation**: it must be registered, fixed and re-frozen before a run in which Gate-B could
   bind.

## RESULTS

**Superseded.** The line below is the registration-time state, retained verbatim for the
record; the campaign that followed is reported underneath it.

> Not run. No code was implemented, no asset was restored, and no experiment was executed as
> part of this preregistration.

### Registered decision run

| item | value |
|---|---|
| run_id (Run 2, the registered decision run) | `tera-gate0-20260807T090111Z-f2caade9` |
| artefact directory | `artifacts/tera_gate0/tera-gate0-20260807T090111Z-f2caade9/` |
| frozen payload | `f2caade97712f8421232dee0a9c6b02545e3ac9ce95357e82e664802316a81e0` (appendix v4, post-D-3) |
| command line | `run_gate0.py --config research-wiki/tera_gate0_frozen_config.json --stages A,C,B --gate-c-audit artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/gate_c_audit.jsonl --confirmation all` |
| audit input | `gate_c_audit.jsonl` from the Run 1 directory (sha256 `491a2fbaa1bc15c41e960f39c2b54e8ccd4ecdf5694c6be01ffa418a88fc071d`, 165 rows = 133 c1 + 27 c2 + 5 adjudicated) |
| start / end (UTC) | `2026-08-07T09:01:11Z` / `2026-08-07T09:26:09Z`, wall clock 1497.888 s |
| device | CPU (`torch_num_threads = 8`, deterministic algorithms on), `gpu_used = false` |
| status | `COMPLETE`, `halt = null` |
| Run 1 (prediction source, D-1) | `tera-gate0-20260807T000625Z-7ba80eaf`, `2026-08-07T00:06:25Z` → `00:30:04Z` |
| `verdict.json` sha256 | `7c97e16c664c281433d0a6a92b8a9543737155605e0834c56c5b01c4287339eb` |
| `metrics.json` sha256 | `ccc5d51d474c174c46fdcef6f2cc4b22833cf7457675a66506901aac47027bf5` |
| integrity | `test_contact_count = 0`, `opened_test_paths = []`, `failure_rate = 0.001344` (HALT bar 0.01), all four overlap assertions true |

### Gate-C — every registered criterion, with its decision

Sample (`metrics.json → gate_c_sampling`): the OOF false-negative population on HateMM-train is
**73** videos, which is at or below the §4.1 cap of 120, so `audited_all = true` — every false
negative was audited and the tercile weights are all 1, hence weighted and unweighted coverage
coincide (`union_coverage = unweighted_union = 0.8356164383561644`). Tercile population sizes
24 / 24 / 25 equal the sampled sizes; tercile cuts `q33 = 0.20950712263584137`,
`q67 = 0.3440778851509094`; controls 30 true positives + 30 false positives (never in the
false-negative denominator); 27 of the 133 audited videos double-coded (20.3%, above the §4.1
20% floor); sampling seed `20260807`; video bootstrap `n_resamples = 10000`, seed `20260809`.

| §4.3 criterion | bar | observed | exact count | decision |
|---|---|---|---|---|
| union{`short_localized`, `multi_segment_complementary`, `cross_modal`}, primary-or-secondary | `>= 0.30` | **0.8356164383561644** | 61 / 73 | **PASS** |
| union video-bootstrap 95% CI lower bound | `>= 0.20` | **0.7534246575342466** | 55 / 73 (upper 0.9178082191780822 = 67 / 73) | **PASS** |
| `multi_segment_complementary` alone, primary-or-secondary | `>= 0.15` | **0.0821917808219178** | **6 / 73** | **FAIL** |
| `annotation_ambiguity_or_noise` | `<= 0.25` | **0.1643835616438356** | 12 / 73 | **PASS** |
| double-coded primary-cause Cohen's kappa | `>= 0.60` | **0.7326732673267327** | raw agreement 0.8148148148148148 = 22 / 27 | **PASS** |

`gate_c.pass = false`. The single binding failure is the `multi_segment_complementary >= 0.15`
criterion: **6 of 73** audited false negatives (8.22%) carry multi-segment complementarity as
primary or secondary cause, against a required 15% — the observed value is roughly *half* the
bar, and 11 of the 73 would have been needed to reach it.

### Verdict

**`NO-GO-C`** (`verdict.json → verdict`), under §9 bullet 1: *"C fails → stop, `NO-GO-C`."*
Reinforced by §4.3: *"Failure means **NO-GO-C**: do not run A or B, and do not claim temporal
evidence is a large enough performance lever."* This is a clean registered failure of the kind
§12 requires to be accepted without rule changes: no threshold was relaxed, no arm or dataset
substituted, and no "near pass" promoted. Note that the failure is **not** the §4.3 reliability
escape hatch — kappa passed at 0.733, so this is evidence against the registered hypothesis and
not a measurement failure.

### Gate-A / Gate-B — sealed, not read

Under §9 the decision path stops at C, so Gate-A and Gate-B carry no registered decision in this
campaign, and under the D-1 §2 isolation clause (clause 4: *"if Gate-C returns NO-GO, Run 1's
`metrics.json` and `verdict.json` remain sealed"*) the seal is permanent. It is applied here to
**both** runs:

- No Gate-A quantity — A0/A1/A2/A3/A4 or O1/O2 macro-F1 and deltas, the arm-`D` identity and its
  selection statistics, the paired bootstrap CI, the temporal within-video AUROC, or the
  confirmation deltas — and no Gate-B quantity is read, quoted, transcribed or acted on in this
  document, in `TARGET_FINDINGS.md`, or in
  `refine-logs/TERA_GATE0_CAMPAIGN_RECORD_2026-08-07.md`. Every number appearing in this
  `RESULTS` section comes from `gate_c`, `gate_c_sampling`, the manifest, or the sampling frame.
- Run 2's `gate_b` is `null` on disk (stage B was skipped — see post-run erratum 2), so there is
  no Gate-B number to seal in the first place.
- The files themselves are retained unmodified for audit completeness and are hashed above.
  Confirming their existence and hashing them is permitted; opening their Gate-A/B sections is
  not.

### Claim boundary actually earned

§10 authorises a claim only on a **full** Gate-0 pass, so **nothing in §10 is claimed**. The one
descriptive fact that may be stated, because it is a Gate-C measurement and not a performance
result: on this frozen representation and baseline, **83.6% (61/73)** of whole-video false
negatives are attributable to short-localized, multi-segment-complementary or cross-modal
evidence — a large addressable error population — but the compositional component specifically
is **8.2% (6/73)**. Any route that would exploit the single-segment part of that population
requires its own preregistration; it is not authorised by this document.
