# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why two superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Anchor Check

- **Original bottleneck:** Binary-label RGCL cannot identify cross-video stance–target–mechanism confounds that produce wrong-memory ordering.
- **Why the revised method still addresses it:** Reliable MLLM pseudo-relations select directed constraints at that exact ordering locus; video labels alone sign every constraint; the unchanged shared embedding is still the only test-time state.
- **Reviewer suggestions rejected as drift:** No relation adapter, router, generator, scale sweep, test-time MLLM, segment gold, or combined route was added.
- **Drift correction:** Each SSR seed now activates only arcs mined by its own paired baseline seed. A shared annotation cache is storage deduplication only; it cannot expose a seed to another seed’s arcs.

## Simplicity Check

- **Dominant contribution after revision:** Reliable train-only MLLM relation typing selects directed gold-signed constraints that repair final kNN geometry.
- **Components removed or merged:** Removed unused evidence-binding fields; merged CS− and TC− into one semantic-confound negative family; downgraded the mechanism-identification protocol from “supporting contribution” to validation.
- **Reviewer suggestions rejected as unnecessary complexity:** No learned relation metric or auxiliary confidence model; reliability is fixed repeat-agreement and missing means no edge.
- **Why this is the smallest adequate route:** One cached pseudo-relation record, two deterministic edge templates, one existing-margin ranking loss, no new trainable parameters or inference path.

## Changes Made

### 1. Directed, seed-isolated graph

- **Reviewer said:** Unordered pairs conflict with directional retrieval error, and cross-seed union supervision drifts from paired protocol.
- **Action:** Store one canonical MLLM record per unordered pair but project it onto explicit `(query i → neighbour j)` arcs. All rank/margin/error/degree/reference fields are query-relative. Seed `s` activates only arcs mined by baseline seed `s`.
- **Impact:** Removes drift and makes every loss tuple implementable.

### 2. Reliability and missing fallback under the hard supervision contract

- **Reviewer said:** Exact order agreement was not a full-pool confidence mechanism.
- **Action:** Run four fixed label-blind queries (two orderings × two preregistered prompt wordings). Per-field reliability is modal agreement; edge reliability is the minimum required-field agreement. `rho∈{0.75,1}` is accepted and used as a fixed loss weight; `rho<0.75`, missing, invalid, or `unclear` routes to `no edge`.
- **Impact:** MLLM outputs remain explicitly weak pseudo-signals; uncertainty never becomes fabricated gold.

### 3. Relation families and loss claim simplified

- **Reviewer said:** CS−/TC− had identical hinges and looked ornamental.
- **Action:** Retain only `MI+` and `SC−`. They have distinct tuple semantics (typed positive vs typed negative). Stance-confound and target/proposition-confound are audit subtypes of SC−, not separate loss modules.
- **Impact:** The claim is honestly semantic directed-constraint selection, not a relation-specific metric.

### 4. Diagnostic shuffle and noise control

- **Reviewer said:** Prior shuffle might preserve the operative signal and no noise sensitivity existed.
- **Action:** Use constrained assignment of complete pseudo-relation records to different directed arcs within seed/dataset/polarity/similarity/rank/margin/error strata, preserving degree, edge-family counts, reliability and missingness. Add audit-calibrated corruption/drop controls at `eta=e_hat` and `2e_hat`.
- **Impact:** Full-vs-shuffle identifies pair semantics; the noise curve tests weak-signal fragility.

### 5. Bounded graph and target-aware coverage

- **Reviewer said:** Costs were unbounded and 20% error coverage does not support +3.
- **Action:** Cap at 6 outgoing candidates/query, 1,200 canonical pairs/seed/dataset before dedup, 2 accepted edges/family/query and 1,600 accepted arcs/seed/dataset. Use 16 tuples/step and ≤48 deduplicated endpoints. B1 now requires dev-error unique-query coverage sufficient for a ≥0.05 correction ceiling, not merely 20%.
- **Impact:** Auditable cost and a meaningful performance headroom gate.

## Revised Proposal

# Research Proposal: SSR-MemRGCL — Reliable Gold-Signed Semantic Constraints for Retrieval Memory

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why two superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Technical Gap

`classifier_hateClipper` produces shared embedding `z`; `retrieve_evaluate_RAC_` normalizes all train/test embeddings, indexes train `z` in `IndexFlatIP`, and votes train video labels. Binary supervision treats semantically different hard pairs uniformly. P2 shows generic comparability is not vote correctness; P4 shows single-video fields are label-redundant; P9/P9b shows an LMM head competes with memory. The missing object is a **directed, reliable cross-video pseudo-relation that identifies a particular wrong-memory constraint while video labels retain sole authority over attraction/repulsion**.

Gate 0 compared SSR with intervention-based EDCM/CCGC and a rule router. SSR is selected alone because it most directly changes the final memory geometry with fewer moving parts and less prior-art/geometry mismatch risk.

## Method Thesis and Contribution

- **Thesis:** Reliable train-only MLLM stance–target–mechanism pseudo-relations identify directed semantic constraints that video labels and generic hardness cannot; gold-signing and a shared-space ranking loss internalize them into final kNN geometry.
- **Dominant contribution:** Gold-signed directed semantic-constraint memory graph on real baseline hard pairs.
- **Validation, not a second contribution:** Conditional-information and matched-null protocol proving the MLLM relation is causal.
- **Explicit non-claims:** First relation-conditioned learning, hard-pair learning, stance modeling, or LLM graph supervision.

## Proposed Method

### Complexity Budget

- **Reuse:** Exact strongest RGCL recipe, frozen input features, shared MLP embedding, dataset-specific segment term, optimizer/epochs/checkpoint rule, FAISS/top-k/vote.
- **New trainable components:** None.
- **New artifacts:** Directed candidate arcs; cached weak pseudo-relation records; deterministic signed ranking tuples.
- **Excluded:** Relation adapters, extra feature streams, routers, generated data, score fusion, test-time MLLM.

### 1. Same-seed directed candidate arcs

For paired seed `s`, recover/run its exact baseline and mine train-only normalized embeddings. For each query `i`, exclude self and select at most 3 same-label plus 3 opposite-label neighbours, prioritizing top rank, wrong-vote involvement, then seeded ID hash. Store directed arc

```text
a=(seed=s, query=i, neighbour=j, y_i, y_j,
   cosine_ij, rank_i(j), query_margin_i, error_stratum_i).
```

Cap annotation selection at 1,200 canonical unordered pairs per seed/dataset. Records may be physically deduplicated across seeds, but SSR seed `s` can activate only arcs from paired baseline seed `s`; cross-seed arcs are masked and asserted absent. Validation/test IDs are forbidden.

### 2. Label-blind weak pseudo-relation and reliability

Frozen `Qwen2.5-VL-7B-Instruct` sees the existing fixed frames and timestamped ASR/OCR for video A/B, never labels, predictions, rank, margin, or intended edge sign. Four fixed calls cover `(A,B)` and `(B,A)` under two preregistered prompt wordings. Output has no hate verdict, score, rationale, segment label, or self-confidence:

```json
{
  "target_relation": "same|different|unclear",
  "proposition_relation": "same|opposed|unrelated|unclear",
  "stance_a": "endorse|quote|report|condemn|satire|unclear",
  "stance_b": "endorse|quote|report|condemn|satire|unclear",
  "stance_relation": "same|opposed|unclear",
  "mechanism_a": "slur|dehumanization|threat|exclusion|violence_praise|stereotype|harassment|none|unclear",
  "mechanism_b": "slur|dehumanization|threat|exclusion|violence_praise|stereotype|harassment|none|unclear",
  "mechanism_relation": "same|different|unclear",
  "topic_surface_relation": "same|different|unclear"
}
```

Reverse outputs are canonicalized. For required field `f`, `rho_f` is modal agreement among four calls; edge reliability `rho_e=min_f rho_f`. Accept `rho_e∈{0.75,1.0}` and weight its loss by `rho_e`. Any required `unclear`, `rho_e<0.75`, missing key, schema failure, or call failure deterministically yields **no edge**, so that sample follows exact non-MLLM RGCL. These are weak/privileged pseudo-signals only; no pseudo-signal is generated or loaded at test.

### 3. Two preregistered typed constraint families

Gold video labels alone set sign.

- **MI+ — mechanism-invariant positive:** `y_i=y_j`, mechanism relation same, and topic surface or target different. The MLLM pseudo-relation selects `j` as a positive that should survive surface/domain change.
- **SC− — semantic-confound negative:** `y_i!=y_j`, topic surface same, and either stance opposed or target/proposition different/opposed/unrelated. The MLLM pseudo-relation selects `j` as a negative whose surface similarity is misleading. `stance-confound` and `target/proposition-confound` remain reporting subtypes, not separate losses.

Any relation not matching these templates is missing/no-edge. Each family must independently pass B1 reliability, precision, coverage, and conditional-information gates; otherwise that family is removed **before** method freeze. If neither survives, SSR stops.

### 4. Directed tuple construction and loss

All difficulty is query-relative. For accepted arc `i→j`:

- MI+: select opposite-label reference `n_e` from query `i`’s same-seed universe, same cosine decile, closest rank;
- SC−: select same-label reference `p_e` under the same matching rule.

Ties use the seed+ID hash. Normalize current shared embeddings and define

```text
Delta_e = cos(z_i,z_j) - cos(z_i,z_n_e)       for MI+
Delta_e = cos(z_i,z_p_e) - cos(z_i,z_j)       for SC-

L_r   = sum_e rho_e ReLU(0.1-Delta_e) / sum_e rho_e
L_SSR = mean of nonempty {L_MI, L_SC}
L     = L_RGCL + 0.2 L_SSR.
```

`0.1` reuses the existing triplet margin; `lambda_SSR=0.2` is preregistered, not swept by dataset or test. Cap at 2 accepted outgoing edges/family/query and 1,600 accepted arcs/seed/dataset. Each existing optimizer step samples 8 MI+ and 8 SC− tuples, deduplicates at most 48 endpoints, encodes them once with the same model, and adds the loss. Optimizer steps, epochs, data and parameters do not change; label-only, shuffle and noise arms use identical work.

### 5. Exact nulls and weak-signal noise sensitivity

- **Remove-MLLM:** Exact strongest RGCL, `lambda_SSR=0`, same seed/protocol.
- **Label-only RankRGCL:** Same tuple counts, sign, query degree, similarity/rank/margin/error strata and workload, but diagnostic endpoints are sampled without pseudo-relations.
- **Pseudo-relation shuffle:** Within seed×dataset×gold-polarity×similarity-decile×rank-bin×margin/error stratum, use constrained bipartite assignment to move each complete pseudo-relation record to a different directed arc. Rebuild templates while exactly matching accepted query out-degree, neighbour indegree, family counts, reliability histogram and missing rate; deterministically merge adjacent bins only if no derangement exists. Hash and verify all invariants.
- **Noise controls:** Estimate per-field confusion and missingness `e_hat` from the blind B1 audit. With fixed seeds, corrupt/drop complete pseudo-records at `eta=e_hat` (all final seeds) and `2e_hat` (seed-0 sensitivity), rebuild graphs under identical caps, and report performance/topology degradation. Noise never uses test labels.

### Training and Inference

1. **B0:** Run/recover exact baselines; freeze/hashes same-seed directed arc universes.
2. **B1:** Audit 250 train/validation-only arcs, stratified by dataset/polarity/difficulty/error; freeze teacher/prompt/ontology/reliability rule/templates if gates pass.
3. **B2:** Seed-0 dev fast-fail: baseline, label-only, full, shuffle, and audit-noise arms from identical init/schedule.
4. **B3:** If B2 passes, paired seeds 0/1/2 on MHC-EN/ZH under the frozen protocol.
5. **Inference:** Unchanged `retrieve_evaluate_RAC_` only: train-memory normalized cosine kNN and the exact comparator vote. No relation artifact or MLLM is loaded.

### Failure Modes and Diagnostics

- **Pseudo-signal unreliable/missing:** Report schema validity, `rho` histogram, family/arc/query coverage, and no-edge fallback rate. Gate fails if agreement or blind precision <0.80; missing/low confidence always reduces to baseline.
- **No conditional information/P2-P4 redux:** For SC−, model decisive wrong-neighbour involvement; for MI+, model correct-neighbour omission. Control video-label relation, cosine, rank, query margin, dataset and error stratum. Each family must add significant held-out likelihood/AUC over generic comparability and label-only features.
- **Insufficient +3 headroom:** On dev, compute `H=baseline error rate × fraction of unique baseline-error queries touched by reliable typed arcs`. Require `H≥0.05` on each proposed dataset and report the required correction fraction `0.03/H`; also require ≥15% hard-pair coverage. This is a gate, not a success claim.
- **MLLM semantics unnecessary:** Full fails to beat label-only or shuffle; mechanism is falsified even if baseline is beaten.
- **Noise fragility:** Full collapses under audit-calibrated `e_hat`, or degradation is non-monotone/uninterpretable. Do not relabel pseudo-signals as annotations or scale the teacher.
- **RGCL conflict/collapse:** Log gradient cosine `grad L_RGCL` vs `grad L_SSR` on shared `z`, active-hinge rates, embedding variance, per-class recall, neighbour label purity and wrong-neighbour rate. Persistent conflict/no topology gain kills the route.
- **Head-memory redistribution:** Native head up but kNN flat/down is failure.
- **Leakage/protocol drift:** Assert no validation/test graph node, no cross-seed active arc, identical graph-control workloads/hashes and unchanged checkpoint/retrieval rules.
- **Target miss:** <+3 in either metric, one dataset/seed only, failed statistics, or insignificant removal/shuffle cost is `not_working`.

### Novelty and Elegance

Closest work already covers generic relation-conditioned representation (RCML), generated hate triplets (HateSieve), RGCL/RA-HMD hard pairs, and LLM graph contrastive selection (CCLRec). The narrow claim is:

> For hateful-video RGCL, reliable train-only MLLM stance–target–mechanism pseudo-relations select directed constraints on real hard neighbours; video-level labels alone sign them; a parameter-free ranking loss repairs the same shared geometry consumed by the final kNN memory classifier.

Unlike prior local routes, the MLLM neither predicts correctness nor becomes a feature/head. Unlike generic hard mining, matched nulls hold label, difficulty, degree, missingness and workload fixed while destroying only pair-relation semantics.

## Claim-Driven Validation Sketch

### Block 1 — Does the weak pseudo-relation add conditional information and enough headroom?

- **Claim:** Reliable MI+/SC− relations identify omitted helpful neighbours or decisive wrong neighbours beyond labels/similarity/generic comparability.
- **Test:** B1 250-arc audit; family-specific conditional models and blind audit.
- **Decisive metrics:** agreement/precision ≥0.80; reliable coverage; incremental held-out likelihood/AUC; `H≥0.05` per dataset; confidence/missing distribution.

### Block 2 — Does relation semantics cause a kNN geometry gain?

- **Claim:** Full SSR improves the final memory geometry, not head capacity or generic hard-pair compute.
- **Test:** B2 baseline/remove, label-only, full, graph-matched shuffle, `e_hat` noise and seed-0 `2e_hat`.
- **Decisive metrics:** Full ≥+0.010 dev accuracy and macro-F1 over baseline, label-only and shuffle; lower wrong-neighbour rate/higher purity; interpretable noise curve; no head↔memory redistribution.

### Block 3 — Does the frozen method meet the actual stop condition?

- **Claim:** The causal MLLM interface yields substantial replicated final improvement.
- **Test:** MHC-EN/ZH, paired seeds 0/1/2, exact same-protocol baseline/full/remove/shuffle/noise, with label-only necessity control.
- **Decisive metrics:** Both accuracy and macro-F1 ≥+0.030 over the moving binding bar on both datasets; 3/3 positive; corrected hierarchical-bootstrap lower bounds >0; significant removal/shuffle cost; coverage/confidence/fallback/corruption reports.

## Experiment Handoff and Compute

- **Must freeze:** Pair miner/caps, teacher, four queries, schema, reliability threshold, templates, references, loss, controls, hashes, seeds and statistics before final test.
- **Pair/call ceiling:** ≤1,200 canonical pairs×3 seeds×2 datasets = 7,200 before cross-seed cache dedup; 4 calls/pair = ≤28,800 pair prompts.
- **Training ceiling:** 16 tuples/step, ≤48 unique relation endpoints, no new optimizer steps; five primary arms×3 seeds×2 datasets only after gates.
- **Estimated compute:** Teacher 20–60 GPU-hours, depending on audit-measured throughput; RGCL/control training 20–35 GPU-hours; all SLURM-only. Audit approximately 6–10 person-hours.
- **Timeline:** B0/B1 2–4 days; B2 1–2 days; frozen B3 3–5 days if gates pass.
- **Highest risks:** MI+/SC− coverage and reliability across EN/ZH; conditional information vanishes after controls; pseudo-signal noise; +3 correction headroom; loss conflict with dataset-specific segment RGCL.
