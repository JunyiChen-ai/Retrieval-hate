# Research Proposal: SSR-MemRGCL — Reliable Gold-Signed Semantic Constraints for Retrieval Memory

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Binary-label RGCL and generic embedding hardness cannot identify why superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—so semantically wrong neighbours distort final memory voting. Previous MLLM signals were orthogonal to vote correctness, label-redundant, absorbed by fusion, or redistributed accuracy between head and memory.
- **Non-goals:** Localization/audit/guard-rail/native-head-only success; test-time MLLM annotation, judging, reranking or fusion; rationale/schema concat; generated counterfactuals; scale/data/epoch/ensemble/protocol engineering; SSR+EDCM/router stacking.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM stance, target, proposition, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold/dense annotation/oracle, and is unavailable as test-time annotation. Low-confidence, missing or invalid pseudo-signals deterministically fall back to non-MLLM RGCL. Edge polarity comes only from video labels. Splits, preprocessing, labels, epochs, checkpoint rule, retrieval, seeds and strongest per-dataset comparator remain fixed. All compute is SLURM-only in `HateVideo`, no `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets, full SSR exceeds `max(historical strongest point, paired same-seed non-MLLM mean)` by ≥+0.030 in both accuracy and macro-F1; all 3/3 seed deltas are positive; Holm-corrected 95% hierarchical paired-bootstrap lower bounds exceed zero for all four dataset×metric claims. Full must significantly beat remove-MLLM and a degree/polarity/difficulty/missingness-matched relation shuffle in both metrics. Label-only and calibrated-noise controls must show that reliable MLLM relations cause the gain. Report pseudo-signal coverage, reliability, fallback and corruption sensitivity. The gain must occur in unchanged final kNN.

## Technical Gap and Method Thesis

`classifier_hateClipper` produces shared embedding `z`; `retrieve_evaluate_RAC_` normalizes train/query embeddings, indexes train `z` in FAISS, and votes train video labels. Video labels say whether a pair should ultimately agree, but not whether it is a mechanism-invariant positive or a surface-similar stance/target confound.

**Thesis:** A frozen MLLM can provide reliable train-only stance–target–mechanism pseudo-relations that select the correct directed hard-pair constraints; video labels alone sign those constraints, and a parameter-free ranking loss internalizes them into the exact geometry used by final kNN.

Gate 0 selected SSR over intervention-based memory keys and rule routing because it is the smallest route attached directly to the final readout. The single contribution is the reliable gold-signed directed semantic-constraint graph. Conditional-information and matched-null procedures are validation, not parallel contributions.

## Proposed Method

### Complexity Budget

- Reuse the exact strongest MHC-EN/MHC-ZH RGCL recipes, frozen input features, shared MLP, dataset-specific segment term, optimizer, epochs, checkpoint rule, FAISS/top-k/vote.
- Add no trainable parameters; add only train directed-arc, pseudo-relation and ranking-tuple artifacts.
- Exclude relation adapters, feature streams, routers, generated data, score fusion and test-time MLLM.

### System Graph

```text
train-only exact RGCL geometry
  -> directed hard-pair universe
  -> frozen label-blind MLLM relation records + repeat-agreement reliability
  -> deterministic no-edge fallback / MI+ or SC- template
  -> video-label sign + difficulty-matched directed tuple
  -> L_RGCL + 0.2 L_SSR on the existing shared embedding z
  -> unchanged train-memory cosine kNN at validation/test
```

### 1. Two Directed Graph Universes

#### B1 OOF diagnostic universe

Split each dataset’s training videos into five fixed stratified folds. For fold `f`, train the exact baseline on the other four folds for the frozen recipe/epoch budget. Held-out fold `f` is query; the other folds are memory. Retrieve at most the top 3 same-label and top 3 opposite-label neighbours per query. Candidate arcs, cosine, rank, kNN margin, prediction, neighbour event and reference are computed entirely in this one OOF geometry. No validation/test video is an endpoint.

#### Final paired-seed universe

For each full-train baseline seed `s`, mine the same directed candidates from its train-memory geometry. SSR seed `s` activates only arcs from paired baseline seed `s`; assertions forbid cross-seed active arcs. Cache reuse only deduplicates identical MLLM calls.

#### Deterministic pair budget

Order queries by `(baseline-error first, H(seed_or_fold,dataset,id))`. First include each error query’s highest-ranked opposite-label candidate, then round-robin remaining opposite/same candidates by rank. Stop at 1,200 unique canonical unordered pairs per graph universe. Retain the direction mask `{low→high, high→low}`; prompt each canonical pair once.

### 2. Weak Pseudo-Relation Interface and Reliability

Frozen `Qwen2.5-VL-7B-Instruct` receives four uniform frames plus timestamped ASR/OCR per train video, deterministically head/tail-truncated to 2,048 Unicode characters. It never sees labels, predictions, ranks, margins, folds, seed or loss role. Decoding is fixed: `do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=256`. Record model, processor, input-builder, prompt and schema hashes.

Four calls use two pair orders × two frozen prompt wordings. JSON schema:

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

There is no hate verdict, label, score, rationale, localization, segment output or self-confidence. After order canonicalization, per-field reliability is modal agreement across four calls; record reliability `rho` is the minimum required-field agreement. Accept `rho∈{0.75,1}` and use it as a fixed loss weight. Lower reliability, any required `unclear`, missing/invalid JSON or call failure becomes one canonical missing record and produces no edge in every direction. No pseudo-relation is generated or loaded for validation/test inference.

### 3. Common Gold-Signed Constraint Families

- **MI+ (typed positive):** video labels equal; pseudo mechanism relation same; pseudo topic surface or target relation different.
- **SC− (typed negative):** video labels differ; pseudo topic surface same; pseudo stance opposed or target/proposition different/opposed/unrelated.

Only video-level labels set polarity. A family survives only if it passes all B1 reliability, human precision, conditional-information, coverage and dual-metric headroom gates on **both** MHC-EN and MHC-ZH. Freeze the common cross-dataset intersection before B2; an empty intersection stops SSR. SC stance/target distinctions are reporting subtypes, not separate loss modules.

### 4. Frozen B1 Events and Conditional Test

Let `N_k(i)` be the exact comparator’s OOF top-k list, with identical similarity weighting and vote rule.

- **SC decisive wrong-neighbour event** `Y_SC(i,j)=1` iff: baseline OOF prediction for `i` is wrong; opposite-label `j∈N_k(i)`; remove `j`, promote rank `k+1`, and recompute the exact vote; the prediction becomes `y_i`.
- **MI missing-helpful-neighbour event** `Y_MI(i,j,n)=1` iff: baseline OOF prediction is wrong; same-label MI candidate `j∉N_k(i)`; its matched opposite-label reference `n∈N_k(i)`; replacing `n` by `j` under the exact vote makes the prediction `y_i`.

For each dataset×candidate-family, fit five-fold query-clustered logistic models on the corresponding same- or opposite-label OOF candidate arcs. The reduced model uses cosine, normalized rank, query margin, query video label, fold and error stratum; the full model adds the reliable MI/SC indicator. Freeze statistics:

```text
DeltaNLL = NLL_reduced - NLL_full
DeltaAUC = AUC_full - AUC_reduced
```

Use 10,000 canonical-record permutations within polarity×cosine-decile×rank-bin×margin/error strata, preserving direction coupling. Pass only if `DeltaNLL>0`, `DeltaAUC>0`, and the permutation p-value is Holm-corrected `<0.05` across all dataset×candidate-family tests.

### 5. Directed Ranking Loss

For accepted directed arc `i→j`, choose a reference from the same geometry, cosine decile and nearest rank: opposite-label `n_e` for MI+, same-label `p_e` for SC−. Fold/seed+ID hash breaks ties.

```text
Delta_e = cos(z_i,z_j)-cos(z_i,z_n_e)    [MI+]
Delta_e = cos(z_i,z_p_e)-cos(z_i,z_j)    [SC-]
L_r = sum rho_e ReLU(0.1-Delta_e) / sum rho_e
L_SSR = mean over surviving families
L_total = L_RGCL + 0.2 L_SSR
```

Margin 0.1 reuses RGCL’s setting; `lambda=0.2` is frozen, never dataset/test-swept. Cap two accepted arcs/family/query and 1,600 arcs/seed/dataset. Each existing optimizer step samples eight tuples/surviving family, deduplicates and encodes at most 48 endpoints once. No additional step, epoch, datum or parameter. Controls execute identical workload.

### 6. Exact Controls

- **Remove-MLLM:** Exact strongest RGCL, `lambda=0`.
- **Label-only RankRGCL:** Same signs, tuple count, direction masks, query degree, difficulty and workload; endpoints chosen without pseudo-relations.
- **Canonical semantic shuffle:** The indivisible assignment unit is one canonical unordered-pair pseudo-relation record plus its direction mask. A preregistered binary integer program moves every record to a different canonical target pair and jointly projects it to all target directed instances. Source/target must match seed/dataset, video-label polarity, direction-mask pattern and per-direction cosine-decile/rank-bin/margin-error strata. Exact constraints reproduce full-graph query out-degree, neighbour indegree, family counts, reliability histogram and missing count. Original-pair assignment is forbidden. If no exact solution exists, the causal null is infeasible and SSR stops; no constraint/bin is relaxed.
- **Calibrated noise:** Corrupt at canonical-record level so directions remain coupled. Human audit estimates template-level invalid-accepted-record rate `e_hat`; the extraction pool supplies fallback rate `m_hat`. At `e_hat`, replace records with polarity/difficulty/direction-matched label-only endpoints; at `m_hat`, map a fixed hash-selected fraction to canonical missing/no-edge while retaining no-op sampler slots. Run `e_hat` on all final seeds and `2e_hat` at seed 0.

## Training and Gate Sequence

### B0/B1 — strict train-only fast-fail

Extract OOF pseudo-relations. Every dataset×family needs at least 80 accepted canonical records. Two bilingual annotators independently judge raw video/ASR/OCR relation validity while blinded to labels, baseline outputs and loss role; a third adjudicates. Adjudicated precision’s **95% Wilson lower bound must be ≥0.80**.

Apply the frozen conditional test above. For dual-metric headroom, correct all and only OOF baseline mistakes touched by reliable family edges; recompute accuracy and macro-F1 from video-level labels. Each family/dataset needs oracle improvement ≥+0.05 in both metrics; report required realized fraction `0.03/oracle_gain`. These are feasibility ceilings, not method results.

### B2 — seed-0 dev mechanism gate

Using paired full-train seed-0 graphs, train baseline/remove, label-only, full, exact shuffle and calibrated-noise arms from identical initialization/schedule. Full must beat baseline, label-only and shuffle by ≥+0.010 dev accuracy and macro-F1, repair kNN topology and avoid head↔memory redistribution. Validation videos receive no pseudo-relations.

### B3 — frozen final test

Freeze common family, graph/teacher/prompt/reliability/loss/control hashes, seeds and statistics. Run MHC-EN/ZH seeds 0/1/2. Test inference is unchanged normalized train-memory FAISS kNN and exact comparator vote; MLLM/relation artifacts are absent.

## Failure Modes and Diagnostics

- Report schema validity, `rho` histogram, accepted canonical/directed/family/unique-query coverage and no-edge fallback rate.
- Stop a family on either dataset if accepted count, Wilson precision, conditional test, or either metric’s +0.05 oracle headroom fails; empty common family stops SSR.
- Stop if exact shuffle is infeasible or full fails label-only/shuffle; MLLM assignment is then untestable/unnecessary.
- Report `e_hat/2e_hat` sensitivity. Weak-signal fragility is not rescued by larger teacher, prompt tuning or extra epochs.
- Log `grad L_RGCL`–`grad L_SSR` cosine on `z`, active hinge, embedding variance, per-class recall, neighbour purity and wrong-neighbour rate. Persistent conflict, collapse or no topology repair stops.
- Native head up but kNN flat/down is failure.
- Assert train-only relation IDs, paired-seed masks, canonical direction coupling, graph/control hashes, identical workload and unchanged validation/retrieval protocol.
- Any <+3 metric, dataset/seed/statistical/remove/shuffle failure remains `not_working`.

## Novelty and Elegance

RCML covers general relation-conditioned representation; HateSieve generates hate triplets; RGCL/RA-HMD cover retrieval hard pairs; CCLRec uses LLM graph contrastive selection. The defensible claim is deliberately narrow:

> In hateful-video RGCL, reliable train-only MLLM stance–target–mechanism pseudo-relations assign directed constraints to real hard-neighbour pairs; video-level labels alone sign them; a parameter-free ranking loss repairs the exact shared geometry used by final train-memory kNN.

Correct relation-to-pair assignment is isolated from labels, difficulty, canonical/directional dependence, degrees, reliability, missingness and compute. There is one mechanism, no module stack and no test-time semantic channel.

## Three Claim-Driven Validation Blocks

1. **Conditional information/headroom:** Internally consistent OOF train arcs establish a common family that is reliable, conditionally informative and has ≥+0.05 oracle accuracy/macro-F1 headroom on both datasets.
2. **Causal geometry:** Seed-0 baseline/label-only/full/canonical-shuffle/noise establishes ≥+1 point in both dev metrics plus kNN topology repair.
3. **Actual stop condition:** Two datasets×three seeds establish ≥+3 accuracy and macro-F1 over moving same-protocol baselines, 3/3 positive deltas, corrected CIs, significant remove/shuffle costs, and complete reliability/fallback/noise reporting.

## Compute and Experiment Handoff

- **Ceiling:** B1 OOF ≤1,200 canonical pairs/dataset plus final ≤1,200×3 seeds×2 datasets: ≤9,600 pairs before cache dedup and ≤38,400 deterministic prompts. Final graph ≤1,600 arcs/seed/dataset and ≤48 relation endpoints/step.
- **Estimate:** Teacher 25–75 GPU-hours after audit throughput measurement; OOF diagnostics 5–10; RGCL/controls 20–35. All SLURM-only. Human audit up to 320 accepted records, roughly 12–20 person-hours.
- **Freeze before implementation:** Fold IDs; pair allocator; MI/SC event code; teacher/model/input/prompt/schema hashes; family/reliability/loss rules; conditional statistic/permutations; canonical-null program; noise generator; metric/statistics code; seeds.
- **Next action:** Implement B0/B1 exactly. Failure of any frozen gate stops SSR; no architecture, prompt or model scaling rescue is allowed.
