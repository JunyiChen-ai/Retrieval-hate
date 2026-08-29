# Research Proposal: CTE-RGCL — Counterfactual Tangent Evidence for Full-Bank Retrieval Geometry

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL as a train-only privileged teacher, and do not stop until the unchanged ordinary full-video train-memory kNN endpoint improves by at least `+0.030` absolute in both accuracy and macro-F1 on at least two datasets and paired seeds `0/1/2`, with the full statistical and mechanism-attribution gates.
- **Must-solve bottleneck:** SSR and EDCM proved that sparse relation edges and bounded edits inside the frozen old neighbourhood cannot touch enough errors. The successor must use label-blind MLLM information to change the shared full-video representation and the whole train-memory geometry, while proving that the information is not reducible to video labels, generic modality dropout, intervention artifacts, shuffled relations, or extra optimization.
- **Non-goals:** No localization, segment classification, segment weighting, teacher-selected/replaced memory key, rationale/schema/score/summary concatenation, score fusion, test-time MLLM, reranking, veto, router/MoE, model/data/epoch/ensemble scaling, SSR or EDCM reuse/retuning, native-head-only gain, or protocol relaxation. A zero-teacher screen is a bounded empirical cost/capacity screen, never a theoretical upper bound or evidence of MLLM success.
- **Constraints:** The only gold supervision that exists is the parent video's binary label. There is no segment gold, timestamp gold, span gold, localization gold, stance gold, target gold, mechanism gold, or rationale gold. The MLLM never sees the gold label and may output only confidence-bearing weak relations `preserve`, `weaken`, `reverse`, or `unclear` between a train video's `full` condition and deterministic whole-modality `visual-neutralized` or `language-neutralized` conditions. Validation/test receive only full videos; no teacher record, neutralized view, confidence, relation, or other view artifact exists in their inference path.
- **Success condition:** Relative to `max(historical strongest non-MLLM point, paired same-seed strongest non-MLLM mean)`, FULL gains at least `+0.030` accuracy and `+0.030` macro-F1 on both MHC-EN and MHC-ZH; all three paired-seed deltas are positive; hierarchical paired-bootstrap 95% lower bounds exceed zero and the four dataset-by-metric tests survive Holm correction. FULL must also beat REMOVE, within-fold relation SHUFFLE, relation-free multiview, label-only/heuristic/random-order controls, and calibrated relation NOISE in actual final kNN, with no teacher or neutralized input at test.

## Technical Gap

The current endpoint is a train-memory kNN over the embedding returned by the shared RGCL fusion head. Previous MLLM routes either supplied sparse edge/edit events, information already recoverable from the video label, a segment score that the fusion head absorbed, a teacher-selected key that mismatched the full query, or a competing head. SSR and EDCM add a sharper diagnosis: editing a fixed old neighbourhood cannot reach enough errors. The missing mechanism is a dense train-only semantic signal that changes the response of the **shared representation itself**, so both full-video queries and all full-video keys can move and new neighbours may enter from outside the old top-64.

Naive modality dropout is not enough. It says every missing modality should have the same effect and can exploit degenerate zero/blank artifacts. Generic multiview consistency can even erase a genuinely necessary modality. Free-text KD, score fusion, teacher keys, and segment weighting either repeat prior work or violate the anchor. CTE instead asks the MLLM for only one item that the binary label cannot specify: for this particular full video, does removing one whole modality preserve, weaken, or reverse the full bundle's moderation-relevant semantic interpretation? The gold label is never shown to the MLLM; it is used later only to orient the supervised true-class retrieval margin.

## Route Choice

- **Route A — CTE (selected):** label-blind ordinal whole-modality relations supervise a bounded local response of the exact full-bank true-class margin. It adds no trainable module and directly targets final kNN geometry.
- **Route B — semantic quotient (reserve only):** MLLM nuisance environments could reshape the whole representation, but overlap with language-guided spurious-correlation and invariant-learning work is stronger, and deletion of label signal is hard to rule out.

CTE has the cleaner single contribution, the most direct endpoint connection, and the smallest architecture change. The two routes are not combined.

## Method Thesis

- **One-sentence thesis:** A frozen label-blind MLLM's confidence-bearing ordinal relation between a train video's full and whole-modality-neutralized evidence can supervise the bounded tangent response of the exact full-bank true-class retrieval margin, moving the shared query/key geometry while leaving ordinary full-video kNN inference unchanged.
- **Why this is the smallest adequate intervention:** CTE adds one parameter-free loss on the existing shared embedding and one train-only pseudo-relation cache; it does not add a backbone, branch, adapter, key type, head, router, or inference component.
- **Why this route is timely:** It uses a foundation model as privileged counterfactual-relation teacher rather than as an always-on classifier or text-feature generator, while targeting the retrieval geometry that actually makes the final decision.

## Contribution Focus

- **Dominant contribution:** ordinal whole-modality counterfactual relation supervision of a full-bank retrieval-margin tangent under a shared query/key encoder.
- **Optional supporting contribution:** none; the OOF screens and controls are validation, not a second method.
- **Explicit non-contributions:** counterfactual generation in general, modality intervention in general, semantic/retrieval KD in general, a new neutralization model, a new classifier, localization, or test-time reasoning.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** the cached full-video visual and language features, current normalized modality projections, multiplicative `align` fusion, fusion MLP, base RGCL loss, checkpoint selection, train-memory FAISS construction, top-k/vote rule, splits, epochs, and full-video test protocol.
- **New trainable components:** zero. CTE is a parameter-free auxiliary loss over the existing shared encoder.
- **New train-only artifacts:** two weak relations plus reliability per train video; label-blind modality prototypes recomputed from the current inner-train bank.
- **Intentionally excluded:** zero/blank modality inputs, segment views, second/EMA encoder, teacher keys, teacher embeddings, relation embedding, auxiliary head, score channel, router, test artifact, and SSR/EDCM operations.

### System Overview

```text
TRAIN ONLY
full-video evidence x_i
  ├─ MLLM: full vs visual-withheld  ─> (r_i,V, confidence)
  └─ MLLM: full vs language-withheld ─> (r_i,L, confidence)
                         no gold/prediction/margin/ID shown

cached full visual/language features
  -> same projected modality encoder
  -> full z_i and two local, non-zero prototype-shrink views z_i,V^a / z_i,L^a
  -> exact current full-video bank B_t from the same encoder
  -> true-class full-bank soft retrieval margin M_i
  -> base RGCL + CTE interval loss on bounded tangent response

VALIDATION / TEST
full video -> selected shared encoder -> ordinary full train bank -> unchanged kNN vote
no MLLM, relation, confidence, neutral view, or CTE branch
```

### Whole-Modality Intervention Without Zero/Blank OOD Artifacts

The teacher and student use two linked but deliberately non-degenerate interfaces.

**Teacher interface.** The full evidence bundle contains the project's deterministic uniform full-video frames and automatic full-video ASR/OCR. For `visual-neutralized`, the language bundle is unchanged and the visual field is replaced by the typed control marker `VISUAL CHANNEL WITHHELD BY DESIGN`; for `language-neutralized`, the frames are unchanged and the language field is replaced by `LANGUAGE CHANNEL WITHHELD BY DESIGN`. No black image, empty string, zero feature, generated content, segment selection, timestamp, or span is supplied. The marker is an explicit intervention operator, not fake content. Two prompt templates and both presentation orders are used.

**Student interface.** Let the current normalized projected visual and language features be

\[
p_i^V=\operatorname{norm}(W_V e_i^V),\qquad
p_i^L=\operatorname{norm}(W_L e_i^L).
\]

At each bank refresh, compute detached label-blind inner-train spherical medoids/normalized robust means `c_V,c_L`. For modality `m`, define only a small local path

\[
\tilde p_i^m(a)=\operatorname{norm}((1-a)p_i^m+a c_m),\qquad a\in(0,1),
\]

and keep the other modality unchanged. The same fusion and MLP encode this path. `a` is the largest value in the preregistered set `{0.05,0.10,0.20,0.30}` for which at least 95% of inner-train perturbed modality points remain inside the 95th percentile of the leave-one-out inner-train 5-NN radius. Selection is label-blind and fold-local. If no value passes, CTE stops. Thus the student never evaluates a zero/blank endpoint; it estimates the local direction toward a train-supported neutral prototype. The teacher supplies only the ordinal direction expected under complete withholding. CTE explicitly assumes that this ordinal direction is locally valid; CTE-1 tests that assumption rather than declaring the neutralized input causal ground truth.

### MLLM Weak-Relation Record

For each train video and modality, the strict JSON schema is:

```json
{"relation":"preserve|weaken|reverse|unclear","confidence":0.0}
```

`confidence` is restricted to `{0,.25,.5,.75,1}` and means confidence in the **relative relation**, never confidence in a hate label. The prompt forbids absolute hateful/benign verdicts, rationales, scores, timestamps, spans, localization, and segment fields. It receives neither `y_i`, baseline predictions/margins/errors, neighbours, dataset row IDs, nor any validation/test record.

Four calls per modality (two prompts by two presentation orders) are canonicalized. The modal relation is retained only when at least three calls agree. Reliability is

\[
\rho_i^m = (\text{agreement fraction})\times(\text{median reported confidence}).
\]

Any parse failure, tie, `unclear`, agreement below 0.75, or `rho<0.5` deterministically becomes `unclear,rho=0`, which is exactly the non-MLLM path for that modality/video. Raw calls, canonical records, prompt hashes, model hash, and failures are immutable train-ID-only artifacts.

### Exact Full-Bank True-Class Margin

The query and key encoder is the **same** `f_theta`; there is no EMA or teacher key encoder. At the start of every epoch, and after loading a checkpoint, encode every full inner-train video in eval mode with current `theta` and detach the keys:

\[
B_t=\{(k_j,y_j,id_j): k_j=\operatorname{norm}(f_\theta(x_j^{full}))\}_{j\in train}.
\]

For query `z`, exclude `id_i` and define the exact full-bank smooth true-class margin

\[
M_i(z;B_t)=\tau\log\sum_{j\ne i,y_j=y_i}e^{s(z,k_j)/\tau}
-\tau\log\sum_{j,y_j\ne y_i}e^{s(z,k_j)/\tau}.
\]

This uses every full-video key, not an old top-k candidate set. Keys are stop-gradient within an epoch for stability, but all keys move together at the next refresh because the same encoder parameters have changed. Full and neutral queries also share `f_theta`. This is the minimal buildable match to the repository's existing detached epoch bank while preventing query/key encoder drift.

For modality `m`, compute

\[
T_i^m=\tanh\left(\frac{M_i(z_i^{m,a};B_t)-M_i(z_i^{full};B_t)}{a\,s_t+\epsilon}\right)\in[-1,1],
\]

where `s_t` is the detached inner-train MAD scale of full-bank margins at the refresh. `T` is a finite-difference local tangent response: negative values mean neutralizing that modality reduces the margin of the video's gold full-video class. The MLLM never sees that class.

### Bounded Continuous Ordinal Tangent Cost

Freeze intervals before any teacher result:

- `preserve`: `I_p=[-d0,d0]`;
- `weaken`: `I_w=[-dr,-dw]`;
- `reverse`: `I_r=[-1,-dr]`, with `0<d0<dw<dr<1`;
- `unclear`: inactive.

The distance-to-interval cost is

\[
c(T,I)=\frac{\operatorname{dist}(T,I)^2}{4}\in[0,1],\qquad
L_{CTE}=\frac{\sum_{i,m}\rho_i^m c(T_i^m,I_{r_i^m})}{\sum_{i,m}\rho_i^m+\epsilon}.
\]

`reverse` is only the strongest ordinal negative-tangent level; CTE does not claim that a small local perturbation crosses the class boundary. Total training is `L_base_RGCL + lambda_CTE L_CTE`, with one preregistered `lambda_CTE` selected inside A0 and then frozen. Because `T`, targets, confidence, and per-example cost are bounded, a high-confidence pseudo-relation cannot create an unbounded gradient. Global gradient clipping remains the baseline value.

### Why This Is the Main Novelty

The MLLM does not create a feature or key and does not choose a neighbour. Its only non-redundant output is a per-video ordinal statement about how a whole-modality intervention changes the full bundle's semantic conclusion. That statement is converted, using only the available video label, into a constraint on the response of the exact full-bank true-class retrieval margin. This is narrower than semantic KD, modality dropout, CGO-style generic gradient control, or counter-reasoning fusion and is attached directly to the final kNN geometry.

### Training Plan and Bank Refresh

1. Freeze splits, comparator recipe, cached full-video features, OOF folds, prompt/schema, neutral support rule, interval thresholds, loss weight candidates, seeds, and control construction hashes.
2. Run CTE-0 with no MLLM calls. Each outer fold builds prototypes, label-only proxy records, and all tuning using inner-train only. The outer query is encoded once as a full video; its label is used only after prediction for the endpoint.
3. If CTE-0 passes on both datasets, run CTE-1 on at most 128 strict train videos per dataset. The teacher sees only the evidence bundles. No validation/test record enters the cache.
4. If CTE-1 passes, extract teacher records for the remaining train videos, freeze them once, and run CTE-2 seed 0 on both datasets.
5. Only after CTE-2 passes, run paired seeds 0/1/2. Each epoch refreshes the full-video bank with the current shared encoder; final validation/test rebuilds the ordinary full train bank from the selected checkpoint.

### CTE-0: Zero-Teacher Bounded Continuous Tangent Cost Screen

CTE-0 is an empirical **learned cost/capacity screen**, not an oracle upper bound and not evidence for the MLLM. In each strict outer fold, an inner-cross-fitted visual-only/language-only probe trained solely on other inner-train video labels supplies a continuous target `a_i^m in [-1,1]`: the clipped change in the probe's true-class margin from full to the same prototype-shrink path. The CTE interval loss is replaced by bounded squared error to a small interval around `a_i^m`. This tests whether the proposed local tangent interface can move the shared full-bank geometry under realistic bounded training.

All of the following must hold independently on MHC-EN and MHC-ZH:

- the neutral support audit passes and at least 80% of inner-train videos have non-degenerate tangent support;
- actual aggregated outer-fold full-video kNN accuracy and macro-F1 each improve by at least `+0.050` over the paired frozen-geometry REMOVE run;
- at least 28 MHC-EN and 29 MHC-ZH baseline-wrong OOF videos become correct, and net correction is positive in both classes;
- top-20 full-video neighbour churn exceeds a target-histogram/gradient-strength-matched random continuous-target control by at least 0.10 Jaccard distance, with paired bootstrap lower bound above zero;
- the label-only target beats relation-free multiview and random-target controls in both metrics.

If A0 passes, this label-only method becomes a stronger non-MLLM moving comparator. If it fails, CTE stops for cost reasons; the failure must not be described as a theoretical impossibility for all MLLM-guided representation learning.

### CTE-1: At-Most-128-Per-Dataset Teacher-Value Pilot

Select at most 128 strict train videos per dataset using frozen strata of video label, OOF baseline margin decile, and OOF correct/error status; the MLLM never sees the strata. Use two prompts and both presentation orders for both modalities. The pilot passes only if both datasets satisfy:

- strict parse completeness at least 95%, non-`unclear` consensus coverage at least 80%, exact/modal four-call agreement at least 0.75, Fleiss kappa at least 0.60, and no single active relation occupies more than 85%;
- teacher relations add held-out conditional information about the cross-fitted beneficial tangent target after controlling video label, baseline margin, modality energy, neutral-path norm, and difficulty; the stratified permutation lower bound is positive on both datasets;
- in fourfold pilot cross-fitting, a fixed norm-matched CTE gradient step learned on three folds improves the fourth fold's full-video true-class margin and wrong-neighbour rate more than label-only order, modality-energy heuristic, relation-free multiview, relation shuffle, and strength-matched random order, with paired bootstrap lower bound above zero;
- teacher relation removal, shuffling, or one calibrated corruption level reduces the effect; directional coverage/effective rank alone cannot pass the gate.

No dataset-specific prompt, threshold, relation interval, or fallback is permitted after inspecting outcomes. A failure stops CTE before full teacher extraction.

### CTE-2 and Final Endpoint

At seed 0, both datasets' actual dev full-video kNN accuracy and macro-F1 must each beat every critical control by at least `+0.010`: REMOVE, relation-free multiview, label-only continuous/order proxy, modality-energy heuristic, strength-matched random orders, and within-fold relation SHUFFLE. Relation corruption at frozen rates must show monotone degradation. Only then is test and three-seed expansion unlocked.

The final claim uses the original full-video train bank, query, FAISS metric, top-k, vote and checkpoint protocol. It must meet the immutable +0.030/+0.030, two-dataset, 3/3-seed, bootstrap/Holm, REMOVE/SHUFFLE/removability gates. Native-head results are diagnostic only.

### Control Construction

- **REMOVE:** exact strongest non-MLLM recipe, paired seed and schedule, no teacher relation.
- **Relation-free multiview:** same two neutral paths and forward/gradient budget, but every active item has the same `preserve` interval and the teacher assignment is absent. This isolates generic multiview/neutralization regularization.
- **Label-only proxy:** the CTE-0 cross-fitted continuous target; if stronger, it is a moving baseline.
- **Modality-energy heuristic:** relation derived only from fold-local full/neutral projected-energy change, confidence-distribution matched.
- **Strength-matched random:** random relation pairs are sampled to match modality, active coverage, relation histogram, confidence histogram, baseline-margin decile, label, and aggregate CTE gradient norm.
- **Relation SHUFFLE:** the indivisible `(rV,rhoV,rL,rhoL)` record is deranged within dataset/fold, video label, baseline-margin decile, modality-energy decile and missingness stratum. No original assignment remains. If exact matching is infeasible, the causal claim stops; strata are not relaxed post hoc.
- **NOISE:** active records undergo frozen distribution-preserving relation swaps at two rates calibrated from pilot disagreement; confidence and coverage remain fixed. Clean must be best and degradation monotone.

### Failure Modes and Diagnostics

- **Neutral path is off-support:** fail the 5-NN radius audit or show extreme tangent norms. Mitigation: reduce only within the frozen `a` set; if none passes, stop. Never substitute zero/blank views.
- **Teacher keys on absence markers:** relation collapse, high order sensitivity, or no conditional information. Mitigation: strict two-prompt/two-order pilot; failure becomes `unclear` or route STOP, not prompt scaling.
- **Teacher relation conflicts with gold full-video semantics:** clean relation fails to beat label-only, shuffle, and random controls. Mitigation: stop; do not ask for absolute verdicts or filter using gold.
- **Stale query/key geometry:** high bank-age drift. Mitigation: one shared encoder, epoch refresh, self-ID exclusion, and logged current-parameter/bank hashes; no second encoder.
- **Improvement is generic multiview or extra optimization:** relation-free multiview, strength-matched random, REMOVE and same-step controls explain the gain. Mitigation: CTE claim fails.
- **Only native head moves:** final kNN gate fails. Mitigation: stop.
- **Teacher coverage/missingness differs by label:** report label-conditional coverage, confidence and fallback; match them in shuffle/random controls; no missingness imputation from gold.

### Novelty and Elegance Argument

TextTeacher already establishes inference-free semantic teachers; EmbedDistill/geometric KD establish retrieval geometry distillation; CGO establishes modality intervention/gradient control in harmful video; RAMF establishes counter-reasoning fusion. CTE does not claim any of those broad firsts. Its narrow mechanism-level delta is the composition of (i) label-blind ordinal whole-modality relations, (ii) a bounded local response of the **true-class exact full-bank** retrieval margin, (iii) a single shared query/key encoder whose complete full-video bank refreshes during training, and (iv) complete teacher/view removal at inference. Removing any one of these collapses CTE into a known or previously failed route.

The method remains focused because there is no new trainable component. Neutral prototypes are a safety interface for a local perturbation, and the many controls are falsification tools, not modules.

## Claim-Driven Validation Sketch

### Claim 1: The CTE action family can move enough final retrieval geometry without segment supervision

- **Minimal experiment:** CTE-0 strict nested train OOF on MHC-EN and MHC-ZH.
- **Baselines/ablations:** REMOVE, relation-free multiview, strength-matched random continuous targets.
- **Metric:** actual OOF full-video kNN accuracy/macro-F1, corrected errors, neighbour churn, neutral support.
- **Expected evidence:** both metrics +0.050, dense correction/churn, and no zero/blank OOD path. Passing is capacity evidence only.

### Claim 2: MLLM ordinal relations add assignment-specific information beyond label-only and cheap intervention controls

- **Minimal experiment:** CTE-1, at most 128 strict train videos per dataset, four-call relation audit plus cross-fitted norm-matched tangent update.
- **Baselines/ablations:** label-only proxy, energy heuristic, relation-free multiview, relation shuffle, random order, corruption.
- **Metric:** reliability/coverage, conditional information, held-out margin and wrong-neighbour changes.
- **Expected evidence:** stable non-collapsed relations and significant assignment-specific improvement on both datasets.

### Claim 3: CTE causes substantial final classification improvement

- **Minimal experiment:** CTE-2 seed-0 dev gate, then MHC-EN/MHC-ZH paired seeds 0/1/2 final kNN.
- **Baselines/ablations:** REMOVE, multiview, label-only, heuristic, random, SHUFFLE, two NOISE levels.
- **Metric:** final accuracy/macro-F1, paired seed signs, hierarchical paired-bootstrap/Holm, removal/shuffle cost.
- **Expected evidence:** seed-0 +0.010 over every critical control, then final +0.030/+0.030 over the moving non-MLLM bar on both datasets with causal ablations.

## Experiment Handoff Inputs

- **Must-prove claims:** action-family reach, MLLM conditional value, and substantial causal final-kNN gain.
- **Must-run ablations:** REMOVE, relation-free multiview, label-only, heuristic, random, SHUFFLE and NOISE.
- **Critical datasets/metrics:** MHC-EN and MHC-ZH; full-video kNN accuracy and macro-F1.
- **Highest-risk assumptions:** local prototype-shrink tangent represents the teacher's full-modality withholding order; relation coverage is non-collapsed; label-only CTE does not consume all available gain; final +3 points remain reachable after the moving comparator updates.

## Compute & Timeline Estimate

- **CTE-0:** ten strict outer-fold candidate runs plus paired controls; same head-level GPU/CPU profile as existing OOF RGCL, submitted only through SLURM. Approximate 20–40 GPU-hours total pending an executor audit.
- **CTE-1:** at most 128 videos per dataset, two modalities, two prompts, two orders: at most 2,048 teacher calls; one fixed teacher, no scale sweep. Pilot updates are small SLURM jobs.
- **CTE-2/final:** only after gates; two datasets by paired seeds and required controls, within the project maximum of two GPUs and 16 CPUs. No extra epochs versus the paired comparator.
- **Annotation cost:** zero new gold annotation. Teacher records are weak pseudo-relations, not annotations.
- **Timeline:** A0 implementation/audit 2–4 days; A0 execution 1–2 days; A1 extraction/audit 1–2 days; seed-0 1–2 days; final only if all gates pass.
