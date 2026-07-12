# Round 3 Refinement

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Anchor Check

- The method still targets directed wrong-neighbour geometry and the final kNN readout.
- Video-level labels remain the only gold signal; MLLM relations are weak train-only pseudo-signals with no-edge fallback and no inference path.
- B1 now mines and evaluates arcs wholly inside one OOF geometry; full-train graphs are not reused for B1 outcomes.
- No reviewer request caused drift, module expansion, or segment/test pseudo-annotation.

## Simplicity Check

- **Dominant contribution:** Correct MLLM pseudo-relation-to-pair assignment selects directed signed constraints for final memory geometry.
- **Model:** Still one existing embedding plus one ranking term; zero new parameters.
- **Controls:** Procedurally exact because causal attribution requires them, not additional method contributions.
- **Further modules:** Explicitly rejected; evidence, not architecture, is the next unknown.

## Changes Made

1. **One OOF geometry for B1:** Each held-out train fold is query set; the other four folds are training and memory. Candidate arcs, ranks, margins, votes, SC wrong-neighbour outcomes and MI omissions are all computed in that fold’s OOF geometry.
2. **Canonical-pair shuffle unit:** One pseudo-relation record remains coupled across its one/two directed projections in full and shuffle. The integer program reassigns canonical records to canonical pairs with the same direction mask and projects them jointly.
3. **Audit rule frozen:** Each dataset×candidate-family needs at least 80 accepted canonical records; adjudicated precision must have a 95% Wilson lower bound ≥0.80. Otherwise the family fails that dataset.
4. **Budget corrected:** OOF diagnostic pairs are included in the call ceiling; no hidden extraction cost.

## Revised Proposal

# Research Proposal: SSR-MemRGCL — Reliable Gold-Signed Semantic Constraints for Retrieval Memory

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Previous MLLM routes supplied signals orthogonal to vote correctness, duplicated label supervision, were absorbed by the fusion head, or shifted accuracy between a native head and memory without improving final kNN.
- **Non-goals:** Localization-only, audit-only, guard-rail-only, or native-head-only success; test-time MLLM annotation/judging/reranking/fusion; rationale/schema concat; generated counterfactuals; model-size/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**; no segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization, segment score, or other semantic output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is absent as test-time annotation. Low-confidence, missing, or parse-failed pseudo-signals deterministically fall back to the non-MLLM path. Use the exact strongest per-dataset RGCL recipe; edge polarity comes only from video labels; no test-derived decisions; fixed splits, preprocessing, labels, epochs, checkpoint rule, retrieval and seeds; all later compute is SLURM-only in `HateVideo` within 2 GPUs/16 CPUs/128 GB and without `--time`.
- **Success condition:** On at least two datasets, full SSR beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must beat remove-MLLM and a degree/polarity/difficulty/missingness-matched pseudo-relation shuffle, with paired 95% CIs excluding zero in both metrics; label-only and calibrated noise controls must show the gain requires reliable MLLM relations. Report pseudo-signal coverage/confidence/fallback/noise sensitivity. The gain must occur in the unchanged kNN readout and survive novelty review.

## Technical Gap, Thesis, and Focus

The current shared embedding `z` is both the RGCL training object and the vector indexed by the final normalized FAISS train-memory kNN. Binary labels cannot tell whether a hard pair is a cross-domain mechanism invariant or a same-topic stance/target confound. P2 comparability, P4 per-video fields, and P9/P9b LMM heads failed because they did not provide this pair-specific geometry signal.

**Thesis:** A frozen MLLM can supply reliable train-only stance–target–mechanism pseudo-relations that select the correct directed constraints; video labels alone sign them, and one parameter-free ranking loss internalizes them into the final kNN geometry.

This is the sole contribution. The method does not claim a new metric, encoder, relation-learning primitive, graph supervisor, or test-time reasoner. Gate 0 selected it over EDCM/CCGC and a router because it is the smallest route attached directly to the memory readout.

## Method

### Complexity budget

- Reuse the exact strongest MHC-EN/MHC-ZH RGCL recipes, frozen input features, shared MLP, segment term, optimizer/epochs/checkpoint rule, FAISS/top-k/vote.
- Add no trainable parameters; add only train directed-arc, pseudo-relation, and ranking-tuple files.
- Exclude adapters, feature streams, routers, generated data, score fusion and test-time MLLM.

### 1. Directed graph universes

#### B1 OOF diagnostic universe

Partition each dataset’s **training split** into five fixed stratified folds. For fold `f`, train the exact baseline on the other four folds for the already-frozen recipe/epoch budget; queries are videos in `f`, and the memory contains only videos in the other four folds. For every query, exclude self by construction and retrieve up to the top 3 same-label and top 3 opposite-label memory items. Compute within this same OOF geometry:

```text
rank_i(j), cosine_i(j), kNN_margin_i, baseline OOF prediction,
SC decisive wrong-neighbour contribution, MI missing-helpful-neighbour event.
```

No full-train neighbour or margin is mixed into B1. Pool the five held-out-query folds into one OOF diagnostic arc universe. Every endpoint is a training video; validation/test videos never enter.

#### Final paired-seed universe

For each full-train paired baseline seed `s`, mine the same top-3 same/top-3 opposite directed candidates from that seed’s train-memory geometry. Seed `s` activates only its own arcs. Annotation cache deduplication across seeds does not expose arcs across seed masks; assertions enforce this.

#### Deterministic budget allocation

For either universe, order queries by `(baseline-error first, H(seed_or_fold,dataset,id))`; first include each error query’s top opposite-label candidate, then round-robin the next opposite/same candidate by rank. Stop at 1,200 unique canonical unordered pairs per universe; retain a direction mask `{low→high, high→low}` showing which directed instances were selected. One canonical pair is prompted once even if both directions exist.

### 2. MLLM weak pseudo-relation

Frozen `Qwen2.5-VL-7B-Instruct` receives four uniform frames plus timestamped ASR/OCR per video, deterministically head/tail-truncated to 2,048 Unicode characters. It never sees video labels, predictions, ranks, margins, folds, seed, or intended sign. Decoding is fixed (`do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=256`) and model/processor/input/prompt/schema hashes are recorded.

Four calls use two pair orders × two frozen wordings. Schema:

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

No hate verdict, label, score, rationale, localization, segment output, or self-confidence exists. After canonicalizing order, per-field reliability is modal agreement over four calls; record reliability `rho` is the minimum required-field agreement. Accept `rho∈{0.75,1}` and use it as fixed loss weight. Any lower reliability, `unclear`, missing/invalid JSON, or call failure is a canonical missing record and yields no edge/baseline fallback in every directed projection. Pseudo-relations are train-only weak signals and are never generated or loaded for validation/test inference.

### 3. Video-label-signed common families

- **MI+ typed positive:** video labels equal; pseudo mechanism same; pseudo topic surface or target different.
- **SC− typed negative:** video labels differ; pseudo topic surface same; pseudo stance opposed or target/proposition different/opposed/unrelated.

Only video-level labels set polarity. A family survives only if it passes every B1 reliability, precision, conditional-information, coverage and dual-headroom gate on **both** MHC-EN and MHC-ZH. Freeze the common intersection before B2; empty intersection stops SSR. SC stance/target subtypes are reporting fields, not separate loss modules.

### 4. Directed ranking loss

For accepted `i→j`, choose a query-relative reference from the same geometry, cosine decile and nearest rank: opposite-label `n_e` for MI+, same-label `p_e` for SC−. Seed/fold+ID hash breaks ties.

```text
Delta_e = cos(z_i,z_j)-cos(z_i,z_n_e)    [MI+]
Delta_e = cos(z_i,z_p_e)-cos(z_i,z_j)    [SC-]
L_r = sum rho_e ReLU(0.1-Delta_e) / sum rho_e
L_SSR = mean over surviving families
L_total = L_RGCL + 0.2 L_SSR
```

Margin 0.1 reuses RGCL’s value; `lambda=0.2` is frozen, never swept. Cap two accepted arcs/family/query and 1,600 arcs/seed/dataset. Each existing optimizer step samples eight tuples/surviving family, deduplicates and encodes at most 48 endpoints once. No extra optimizer step, epoch, datum, or parameter. All graph controls execute identical work.

### 5. Controls with the same canonical dependence

- **Remove:** exact strongest RGCL, `lambda=0`.
- **Label-only:** identical sign, tuple count, direction mask, query degree, difficulty and workload; endpoint selection ignores pseudo-relations.
- **Semantic shuffle:** The indivisible assignment unit is one canonical unordered-pair record `(pseudo fields, rho/missing, original pair)` plus its direction mask. A preregistered integer program assigns each complete record to a **different canonical target pair** and jointly projects it onto all target directed instances; it is never duplicated or independently shuffled by direction. Source and target must share seed/dataset, video-label polarity, direction-mask pattern, per-direction cosine decile/rank bin/margin-error stratum. Global constraints exactly reproduce full-graph query out-degree, neighbour indegree, family counts, reliability histogram and missing count after template rebuilding. Original pair assignment is forbidden. If no exact solution exists, the required null is infeasible and SSR stops; no bin/constraint is relaxed.
- **Noise:** Corrupt at canonical-record level, so both directions remain coupled. Blind audit yields template-level invalid-accepted-record rate `e_hat`; the full extraction pool yields fallback rate `m_hat`. Dataset×family `e_hat` requires ≥80 audited accepted records; otherwise that family already fails. At `e_hat`, replace accepted records with polarity/difficulty/direction-mask-matched label-only endpoints; at `m_hat`, map a fixed hash-selected fraction to canonical missing/no-edge while retaining no-op sampler workload. Run `e_hat` on all final seeds and `2e_hat` on seed 0.

## Training, Gates, and Inference

### B1 train-only information gate

Extract pseudo-relations for OOF diagnostic canonical pairs. For every dataset×candidate-family, require at least **80 accepted canonical records**. Two bilingual annotators independently judge the proposed pair relation from raw video/ASR/OCR while blinded to labels, baseline outputs and loss role; a third adjudicates disagreements. The family passes human validity only if the adjudicated precision’s **95% Wilson lower bound is ≥0.80**.

Within each OOF fold geometry:

- SC outcome is decisive wrong-vote contribution by that OOF neighbour;
- MI outcome is omission of that OOF same-label neighbour relative to its matched opposite-label reference.

Held-out conditional models control video-label relation, OOF cosine/rank/margin and error stratum. Each family/dataset must add significant likelihood/AUC over label-only variables and generic comparability.

For headroom, use OOF predictions and correct all and only OOF baseline mistakes whose queries are touched by a reliable family edge. Recompute accuracy and macro-F1 from video-level labels. Each family/dataset must provide ≥+0.05 oracle improvement in both metrics; report required realized fraction `0.03/oracle_gain`. These are feasibility ceilings, not results.

### B2 seed-0 dev fast-fail

Using the paired full-train seed-0 graph, train exact baseline/remove, label-only, full, exact shuffle and audit-noise arms from identical initialization/schedule. Full must exceed baseline, label-only and shuffle by ≥+0.010 dev accuracy and macro-F1, repair topology, and avoid head↔memory redistribution. Validation videos receive no MLLM pseudo-relations.

### B3 frozen final protocol

Freeze common family set, graph/teacher/prompt/reliability/loss/control hashes, seeds and statistics. Run MHC-EN/ZH seeds 0/1/2. Test inference remains unchanged normalized train-memory FAISS kNN and exact comparator vote; no MLLM/relation file is loaded.

## Failure Diagnostics

- Report schema validity, `rho` histogram, accepted canonical/directed/family/unique-query coverage and fallback rate. Weak/missing always no-edge.
- Stop a family on any dataset if accepted count, Wilson precision, conditional information, or either metric’s +0.05 oracle headroom fails; empty common family stops SSR.
- Stop if exact shuffle is infeasible or if full fails label-only/shuffle: causal MLLM assignment is untestable/unnecessary.
- Report `e_hat/2e_hat` corruption sensitivity; fragility is not rescued by scale/prompt tuning.
- Log `grad L_RGCL`–`grad L_SSR` cosine on `z`, active hinge, embedding variance, per-class recall, neighbour purity and wrong-neighbour rate; conflict/collapse/no topology repair stops.
- Native head up but kNN flat/down is failure.
- Assert train-only relation IDs, paired-seed masks, canonical direction coupling, graph/control hashes, identical workload and unchanged validation/retrieval protocol.
- Any <+3 metric, dataset/seed/statistical/remove/shuffle gate remains `not_working`.

## Novelty and Elegance

RCML covers general relation-conditioned representation; HateSieve generates hate triplets; RGCL/RA-HMD cover retrieval hard pairs; CCLRec uses LLM graph contrastive selection. The defensible claim is deliberately specific:

> In hateful-video RGCL, reliable train-only MLLM stance–target–mechanism pseudo-relations assign directed constraints to real hard-neighbour pairs; video-level labels alone sign them; a parameter-free ranking loss repairs the exact shared geometry used by final train-memory kNN.

Correct relation-to-pair assignment is isolated from labels, difficulty, directed/canonical dependence, degrees, reliability, missingness and compute. There is one mechanism, no module stack, and no test-time semantic channel.

## Three Claim-Driven Validation Blocks

1. **Conditional information/headroom:** Internally consistent OOF train arcs show a common family is reliable, conditionally informative and has ≥+0.05 oracle acc/macro-F1 headroom on both datasets.
2. **Causal geometry:** Seed-0 baseline/label-only/full/canonical-shuffle/noise shows full gains ≥+1 point in both dev metrics and repairs kNN topology.
3. **Stop condition:** Two datasets×three seeds show ≥+3 acc and macro-F1 over moving same-protocol baselines, 3/3 positive, corrected CIs, significant remove/shuffle costs, and complete reliability/fallback/noise reporting.

## Compute and Handoff

- **Pair ceiling:** B1 OOF ≤1,200 canonical pairs/dataset plus final ≤1,200×3 seeds×2 datasets: ≤9,600 pairs before cache dedup, ≤38,400 deterministic prompts.
- **Training ceiling:** ≤1,600 arcs/seed/dataset, ≤48 relation endpoints/step, unchanged optimizer steps.
- **Estimate:** Teacher 25–75 GPU-hours after measured audit throughput; OOF diagnostics 5–10; RGCL/controls 20–35. All SLURM-only.
- **Human audit:** up to 320 accepted records plus fallback analysis, roughly 12–20 person-hours.
- **Next step after Gate 1:** Implement B0/B1 exactly; do not add components. B1 evidence, common-family survival, exact-null feasibility and dual-metric headroom determine continuation.
