# Round 2 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Anchor Check

- **Original bottleneck:** Video-label RGCL lacks pair-specific semantic structure at the wrong-neighbour locus.
- **Why the revised method still addresses it:** MLLM relation-to-pair assignment is the sole new information; video labels alone sign constraints and final inference remains unchanged kNN.
- **Reviewer suggestions rejected as drift:** No validation/test pseudo-relations, segment gold, learned confidence, new adapter, extra data, or test-time MLLM was introduced.
- **Drift corrected:** B1 now uses only five-fold cross-fitted **training** videos. Validation videos are used only by the pre-existing checkpoint-selection protocol and never become relation endpoints.

## Simplicity Check

- **Dominant contribution:** Reliable MLLM pseudo-relations select directed constraints that repair final kNN geometry.
- **Components removed/merged:** Noise calibration is template-level rather than per-field; one common relation-family set is frozen across datasets.
- **Unnecessary additions rejected:** No new model or loss family; exact shuffle feasibility is a gate, not a relaxation/tuning loop.
- **Smallest adequate route:** Two edge templates, one hinge, no new trainable component and no inference change.

## Changes Made

1. **Strict train-only B1:** Five fixed stratified folds produce out-of-fold train predictions/embeddings. All audits, human review, conditional models and headroom calculations use train videos only.
2. **Dual-metric oracle headroom:** Replace scalar `H` with touched-query oracle improvements in both accuracy and macro-F1; each must be ≥+0.05 on each target dataset.
3. **Common family freeze:** MI+ or SC− survives only if it passes every gate on both MHC-EN and MHC-ZH; the common intersection is frozen before B2, or the route stops.
4. **Exact null feasibility:** A preregistered integer assignment defines the shuffle. If no exact matched null exists, validation is impossible and SSR stops; no post-hoc bin relaxation.
5. **Estimable noise:** Use accepted-edge invalidity and missingness, pooled only under a fixed minimum-cell rule; no sparse categorical confusion model.
6. **Operational details frozen:** Deterministic global pair allocation, MLLM decoding/input limits, bilingual audit/adjudication, and hashes are specified.

## Revised Proposal

# Research Proposal: SSR-MemRGCL — Reliable Gold-Signed Semantic Constraints for Retrieval Memory

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Technical Gap and Route Choice

Current `classifier_hateClipper` produces shared `z`; the final evaluator normalizes train/query `z`, indexes train embeddings in FAISS, and votes train video labels. Binary labels cannot identify whether a similar pair shares mechanism across domains or differs in stance/target despite surface overlap. Generic MLLM comparability (P2), per-video fields (P4), and LMM heads (P9/P9b) already failed at this locus.

Gate 0 preferred SSR over counterfactual memory-key intervention and a rule router. SSR alone is the minimal route that lets foundation-model semantics alter the exact final memory geometry without a new feature stream, model component, or test-time path.

## Method Thesis and Contribution

- **Thesis:** Reliable train-only MLLM stance–target–mechanism pseudo-relations select directed semantic constraints unavailable from video labels/generic hardness; video labels sign them, and a parameter-free ranking loss internalizes them into final kNN geometry.
- **One contribution:** A gold-signed directed semantic-constraint graph on real RGCL hard neighbours.
- **Validation only:** Matched-null and conditional-information tests establish that correct relation-to-pair assignment—not compute or generic mining—causes the gain.
- **Non-claims:** General relation learning, hard-pair learning, stance modeling, or LLM graph supervision firsts.

## Proposed Method

### Complexity Budget

- Reuse exact per-dataset strongest RGCL, frozen features, shared embedding, segment term, optimizer/epochs/checkpoint rule, FAISS/top-k/vote.
- Add no trainable parameters. Add only directed arc, pseudo-relation and ranking-tuple files.
- Exclude adapters, routers, generated examples, feature/score fusion and test-time MLLM.

### 1. Paired-seed directed arc universe

For baseline seed `s`, extract train-only normalized embeddings. For each query `i`, exclude self and retain up to the top 3 same-label and top 3 opposite-label neighbours. All metadata are directional:

```text
a=(s, query=i, neighbour=j, y_i, y_j,
   cosine_ij, rank_i(j), kNN_margin_i, OOF/error_stratum_i).
```

The global 1,200 canonical-pair budget per seed/dataset is allocated deterministically:

1. queries are ordered by `(OOF-error first, H(seed,dataset,query_id))`;
2. pass 1 adds each error query’s highest-ranked opposite-label arc;
3. passes 2–7 round-robin the next opposite/same candidate by rank for every query;
4. stop at 1,200 unique unordered pairs; retain both directions if both were selected, with one shared MLLM record.

Seed `s` activates only arcs selected by its paired baseline seed. Cross-seed cache reuse only deduplicates identical MLLM calls; runtime assertions reject cross-seed active arcs. Validation/test IDs are illegal graph nodes.

### 2. Weak pseudo-relation and reliability

Frozen `Qwen2.5-VL-7B-Instruct` receives, per video, four uniformly sampled frames plus timestamped ASR/OCR truncated by a deterministic balanced head/tail rule to 2,048 Unicode characters. It never receives labels, predictions, rank, margin or intended sign. Decoding is fixed: `do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=256`; model, processor, prompt, schema and input-builder hashes are recorded.

Four calls cover two orderings × two frozen prompt wordings. JSON fields are:

```text
target_relation: same|different|unclear
proposition_relation: same|opposed|unrelated|unclear
stance_a/b: endorse|quote|report|condemn|satire|unclear
stance_relation: same|opposed|unclear
mechanism_a/b: slur|dehumanization|threat|exclusion|violence_praise|
               stereotype|harassment|none|unclear
mechanism_relation: same|different|unclear
topic_surface_relation: same|different|unclear
```

There is no hate verdict, label, rationale, segment output, score or self-confidence. After order canonicalization, field reliability is modal agreement across four calls and edge reliability `rho` is the minimum over required fields. `rho∈{0.75,1}` is accepted and used as a fixed loss weight. `rho<0.75`, `unclear`, missing/invalid JSON or failed calls deterministically produce no edge, i.e. exact non-MLLM behavior for that candidate. No pseudo-signal is generated/loaded at validation or test inference.

### 3. Common typed families, video-label sign

- **MI+ (typed positive):** `y_i=y_j`, pseudo mechanism relation is same, and pseudo topic surface or target relation is different.
- **SC− (typed negative):** `y_i!=y_j`, pseudo topic surface is same, and pseudo stance is opposed or target/proposition is different/opposed/unrelated.

Only video-level labels set `+/-`. MLLM fields remain weak pseudo-signals. Stance versus target/proposition confounds are reported SC subtypes, not loss modules. A family joins the final method only if it passes reliability, human precision, conditional-information, coverage, and oracle-headroom gates **on both MHC-EN and MHC-ZH**. Freeze the cross-dataset intersection before B2; if empty, stop.

### 4. Directed ranking loss

For accepted `i→j`, choose a same-seed/query-relative reference in the same cosine decile with closest rank: opposite-label `n_e` for MI+, same-label `p_e` for SC−. Seed+ID hash breaks ties.

```text
Delta_e = cos(z_i,z_j)-cos(z_i,z_n_e)       [MI+]
Delta_e = cos(z_i,z_p_e)-cos(z_i,z_j)       [SC-]
L_r = sum rho_e ReLU(0.1-Delta_e) / sum rho_e
L_SSR = mean over surviving nonempty families
L = L_RGCL + 0.2 L_SSR
```

The margin reuses RGCL’s 0.1. `lambda=0.2` is frozen, not swept. Cap at 2 accepted arcs/family/query and 1,600 arcs/seed/dataset. Per existing optimizer step, sample 8 tuples/surviving family, deduplicate and encode at most 48 endpoints once. No new optimizer steps, epochs, data or parameters; all control arms execute identical sampler work.

### 5. Exact controls

- **Remove:** Exact strongest RGCL, `lambda=0`.
- **Label-only RankRGCL:** Identical tuple counts/sign/query-degree/difficulty/workload, diagnostic endpoints selected without pseudo-relations.
- **Semantic shuffle:** Assign all complete pseudo-relation records, including missing records, to different directed candidate arcs using a preregistered binary integer program. Assignment is allowed only within exact seed×dataset×gold-polarity×cosine-decile×rank-bin×margin/error stratum. Constraints exactly match full-graph query out-degree, neighbour indegree, family counts, reliability histogram and missing/fallback count; no record may remain on its original unordered pair. Rebuild templates and references, then verify invariants/hash. If the integer program has no feasible solution, the matched-null requirement fails and SSR **stops**; strata are never relaxed post hoc.
- **Noise:** Blind audit estimates template-level invalid-accepted-edge rate `e_hat` and full-pool fallback rate `m_hat`. Estimate per dataset×family only with ≥40 audited accepted arcs; otherwise use the preregistered pooled-family estimate across datasets. At `e_hat`, replace that fraction of accepted semantic endpoints with difficulty/polarity-matched label-only endpoints; at `m_hat`, route a fixed hashed fraction to no-edge while retaining no-op sampler slots for workload. Run `e_hat` at all final seeds and `2e_hat` at seed 0. This tests weak-signal fragility without pretending to know a per-field confusion matrix.

## Training and Inference

### B0 — baseline and train-only cross-fit

For each dataset, make five fixed stratified folds inside the training split. Seed-0 baseline models trained on four folds produce out-of-fold train embeddings/predictions for B1 diagnostics only. Separately, paired full-train baselines seeds 0/1/2 mine their final candidate arcs. Validation remains the existing checkpoint-selection split; it is never sent to the MLLM.

### B1 — strict train-only audit

Run MLLM extraction on candidate **training** pairs only. Blind audit up to 80 accepted arcs per dataset×family (maximum 320) plus 40 fallback records/dataset. Two bilingual annotators independently see raw video/ASR/OCR and the proposed relation but not video labels, baseline outputs or loss role; disagreements go to a third annotator. Report agreement and Wilson intervals; accepted-edge precision must be ≥0.80 in every dataset×family cell.

Using OOF train predictions:

- SC− outcome: whether the opposite-label neighbour is a decisive wrong-vote contributor;
- MI+ outcome: whether the same-label neighbour is a missing helpful neighbour relative to a matched opposite-label neighbour.

For each family/dataset, held-out conditional models control video-label relation, cosine, rank, query margin and error stratum; typed relations must significantly improve likelihood/AUC over labels and generic comparability.

For dual-metric headroom, take OOF baseline predictions and create a touched-query oracle that corrects **all and only** baseline mistakes whose queries have a reliable family edge; recompute accuracy and macro-F1 using only video labels. Require oracle improvements ≥+0.05 for **both** metrics on **each** target dataset and report required realized fractions `0.03/oracle_gain`. This is only a feasibility gate, not evidence of performance.

### B2 — seed-0 dev fast-fail

From identical initialization/schedule, run baseline/remove, label-only, full, exact shuffle, and calibrated-noise controls. Full must exceed baseline, label-only and shuffle by ≥+0.010 dev accuracy and macro-F1; improve kNN topology; and avoid head-memory redistribution. Validation samples receive no pseudo-relations.

### B3 — frozen final test

Freeze common families, graph builder, teacher/prompts, reliability, loss, controls, hashes, seeds and statistics. Run MHC-EN/ZH paired seeds 0/1/2. Test inference is unchanged `retrieve_evaluate_RAC_`: normalized train-memory cosine kNN and the comparator’s exact vote. MLLM/pseudo-relation artifacts are not loaded.

## Failure Diagnostics

- **Weak signal:** Report schema validity, reliability histogram, accepted family/arc/unique-query coverage, and fallback rate; low-confidence/missing always no-edge. Failure of any B1 gate stops the family; empty common family set stops SSR.
- **Insufficient target headroom:** Either OOF oracle acc or macro-F1 gain <+0.05 on either dataset stops SSR.
- **Pseudo semantics unnecessary:** Full fails label-only or exact shuffle; novelty/mechanism is falsified.
- **Null infeasible:** Exact assignment has no solution; causal claim cannot be tested and SSR stops without relaxed matching.
- **Noise fragile:** Audit-rate corruption erases the effect or yields uninterpretable non-monotonic behavior; record honestly, never scale/prompt-rescue.
- **Geometry conflict/collapse:** Log `grad L_RGCL`–`grad L_SSR` cosine on shared `z`, active hinge, embedding variance, class recall, label purity and wrong-neighbour rate. Persistent conflict/no topology gain stops.
- **Redistribution:** Head improves but kNN does not = fail.
- **Leakage/drift:** Assert train-only relation IDs, paired-seed masks, identical workloads/hashes and unchanged validation/retrieval rules.
- **Target miss:** Any <+3 metric, dataset/seed/statistical/control failure remains `not_working`.

## Novelty and Elegance

RCML already covers general relation-conditioned representation; HateSieve generated hate triplets; RGCL/RA-HMD cover retrieval hard pairs; CCLRec covers LLM graph contrastive selection. The defensible, single claim is:

> In hateful-video RGCL, reliable train-only MLLM stance–target–mechanism pseudo-relations assign directed constraints to real hard-neighbour pairs; video-level labels alone sign them; a parameter-free ranking loss repairs the same geometry used by final train-memory kNN.

Matched controls fix labels, difficulty, degrees, reliability, missingness and compute, so only correct pseudo-relation-to-pair assignment differs. The method is not a module stack and has no test-time semantic channel.

## Claim-Driven Validation

1. **Information and headroom:** Strict train-only cross-fit/audit proves each common family is reliable, conditionally informative and has ≥+0.05 oracle headroom in both accuracy and macro-F1 on both datasets.
2. **Causal geometry:** Seed-0 baseline/label-only/full/exact-shuffle/noise comparison proves relation assignment, not generic hard mining, improves kNN topology and both dev metrics by ≥+1 point.
3. **Actual stop condition:** Two datasets×three seeds prove ≥+3 accuracy and macro-F1 over the moving baseline, all positive deltas, corrected CIs, significant remove/shuffle costs, and confidence/fallback/noise reports.

## Compute and Handoff

- **Ceiling:** 1,200 canonical pairs×3 seeds×2 datasets; ≤28,800 deterministic pair prompts before cache dedup; ≤1,600 accepted arcs/seed/dataset; ≤48 relation endpoints/step.
- **Compute:** Teacher 20–60 GPU-hours based on B1-measured throughput; RGCL/control 20–35 GPU-hours; five-fold diagnostic cross-fit 5–10 GPU-hours. All SLURM-only.
- **Human audit:** Up to 400 records, approximately 10–16 person-hours.
- **Timeline:** B0/B1 3–5 days; B2 1–2 days; B3 3–5 days only if gates pass.
- **Freeze before implementation handoff:** Pair allocator, fold IDs, teacher/model/input/prompt/schema hashes, family rule, reliability, loss, exact-null program, noise generator, metric/statistics code and all seeds.
