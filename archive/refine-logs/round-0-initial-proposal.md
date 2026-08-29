# Research Proposal: SSR-MemRGCL — Gold-Signed Semantic Relations for Retrieval-Memory Geometry

## Problem Anchor

- **Bottom-line problem:** Integrate an MLLM meaningfully and novelly into RGCL/RA-HMD hateful-video detection such that, under the frozen same-protocol evaluation, the final train-memory kNN classifier improves **accuracy and macro-F1 each by at least +0.030 absolute** over the strongest non-MLLM RGCL comparator on at least two datasets, initially MHC-EN and MHC-ZH, with paired seeds 0/1/2.
- **Must-solve bottleneck:** Existing RGCL geometry knows only binary video labels and generic embedding hardness. It cannot distinguish why two superficially similar videos should attract or repel—especially endorsement versus quotation/condemnation/satire, target/proposition mismatch, and shared harm mechanism—and therefore admits wrong-neighbour attraction. Every previous MLLM route either supplied a signal orthogonal to vote correctness, duplicated label supervision, was absorbed by the fusion head, or moved accuracy between a native head and memory without improving the final kNN readout.
- **Non-goals:** Localization-only improvement; MLLM audit/guard-rail-only value; a better native MLP head without a better kNN readout; test-time MLLM judging, reranking, veto, score fusion, or arbitration; embedding/rationale/schema concatenation; generated counterfactual samples; model-size scaling; more data, epochs, ensembling, or protocol changes as the primary source of gain; combining SSR with CCGC/EDCM or a router into a multi-module system.
- **Constraints:** Use the existing RGCL fused embedding and exact strongest non-MLLM configuration per dataset; MLLM access is frozen and train-only; edge polarity comes only from gold labels; the MLLM may supply only typed stance–target–mechanism relations; no test labels or test-derived ontology/prompt/edge decisions; all later GPU work must run through SLURM in `HateVideo`, with no `--time`; at most 2 GPUs / 16 CPUs / 128 GB; same splits, preprocessing, label space, checkpoint selection, retrieval rule, top-k, epochs, and seeds as the comparator.
- **Success condition:** On at least two datasets, the frozen full method beats `max(historical strongest point, paired same-seed non-MLLM mean)` by at least +0.030 in both accuracy and macro-F1; all 3/3 paired seed deltas are positive; mean±std and Holm-corrected 95% hierarchical paired-bootstrap intervals have lower bounds above zero for all four dataset×metric claims. The full method must also beat both remove-MLLM and a degree/polarity/difficulty-preserving shuffle of MLLM relation information, with same-direction paired effects and 95% CIs excluding zero for both primary metrics. The gain must occur in the unchanged train-memory kNN readout, and the narrow novelty claim must survive a final literature/reviewer check.

## Technical Gap

The current implementation maps frozen image/text features through `classifier_hateClipper` to a shared embedding `z`, trains it with label-based in-batch and retrieval-guided losses, then puts **all train embeddings** into a normalized `faiss.IndexFlatIP`; test prediction is the unchanged top-k train-label vote. This exposes the exact failure locus: if a quotation and an endorsement, or two videos sharing topic words but not target/proposition, occupy the wrong local order, the final memory classifier fails even when the native head is reasonable.

The project’s P2/P2b results show why generic “comparable?” judgments are insufficient: topical comparability is almost independent of whether a neighbour contributes the correct vote. P4 shows why per-video target/mechanism fields are insufficient: fields can be decodable yet redundant with binary supervision. P9/P9b shows why an LMM head is insufficient: head and memory compete rather than synergize. Therefore the missing information is neither another semantic feature nor another classifier; it is **a gold-signed, cross-video relation specifying which real hard pairs expose a semantic violation of the final memory ordering**.

### Route decision

- **Route A, minimal/elegant:** SSR-MemRGCL. Mine real baseline hard pairs, let a frozen MLLM type only their stance–target–mechanism relation, use gold labels to sign them, and add one parameter-free signed ranking loss on the existing shared embedding.
- **Route B, frontier-native alternative:** EDCM/CCGC-style MLLM counterfactual intervention signatures select sufficient evidence views and memory keys. It is more intervention-heavy, risks full-query/clean-key mismatch, and is pre-weakened by P3/P11 plus same-domain CGO prior art.
- **Choice:** Route A. It attaches directly to the final kNN geometry, uses a foundation model where binary labels are information-theoretically insufficient (cross-pair semantic typing), introduces no test-time path or new trainable module, and has the strongest Gate-0 novelty/performance expected value. The routes are not combined.

## Method Thesis

- **One-sentence thesis:** Gold labels should decide whether a real hard-pair edge attracts or repels, while a frozen train-only MLLM should decide *which stance–target–mechanism relation makes that edge semantically diagnostic*; optimizing those signed typed edges in the shared RGCL embedding directly repairs the geometry consumed by the final kNN memory classifier.
- **Why this is the smallest adequate intervention:** It adds one offline typed-edge file, one graph edge sampler, and one loss term; it reuses the current fused embedding, optimizer, training schedule, and inference code, with zero new trainable parameters.
- **Why this route is timely in the foundation-model era:** A strong MLLM is used as a constrained relation extractor over multimodal evidence—not as a noisy absolute hate judge or extra feature—and its privileged training-only structure is internalized into a lightweight retrieval classifier.

## Contribution Focus

- **Dominant contribution:** A gold-signed semantic relation graph over real RGCL hard neighbours whose typed edges directly train the final retrieval-memory geometry.
- **Optional supporting contribution:** A mechanism-identification protocol that tests whether typed relations add conditional information and uses remove/shuffle/difficulty-matched controls to separate MLLM semantics from ordinary hard-pair mining.
- **Explicit non-contributions:** General relation-conditioned multimodal learning; the first LLM graph supervisor; a new MLLM, encoder, fusion architecture, generated hard-negative method, or test-time reasoning system.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone:** Existing per-dataset strongest non-MLLM RGCL recipe, frozen CLIP/video/text features, `classifier_hateClipper`, original `L_RGCL` (including the dataset’s fixed segment term), optimizer, epoch budget, warmup/validation selection, FAISS cosine memory, top-k and vote rule.
- **New trainable components:** **None.** SSR changes the supervision graph of the existing embedding only.
- **New non-trainable artifacts:** One train-only candidate-pair table, one frozen-MLLM relation JSONL, and one deterministic signed-edge/triplet table per dataset.
- **Tempting additions intentionally not used:** Natural-language rationale embeddings, relation encoders/adapters, cross-attention, MoE/router, counterfactual generator, MLLM score fusion, test-time relation prediction, multiple MLLM sizes, or SSR+EDCM stacking.

### System Overview

```text
exact RGCL baseline (train only; same seed family)
    -> normalized train embeddings + baseline top-K/error strata
    -> fixed real-pair universe (IDs, labels, cosine/rank/margin strata)
    -> frozen Qwen2.5-VL-7B, no labels, pair order A/B and B/A
    -> stable typed relation only (stance, target/proposition, mechanism, binding)
    -> deterministic template + gold labels
    -> sparse signed graph / difficulty-matched ranking tuples

ordinary RGCL training from the same seeded initialization
    L_total = L_RGCL + lambda_SSR * L_SSR
    -> shared full-video embedding z
    -> unchanged train-memory FAISS index
    -> unchanged test-query kNN label vote (no MLLM at inference)
```

### Core Mechanism

#### 1. Fixed pair universe

For each dataset, obtain normalized train embeddings from the exact non-MLLM baseline without looking at test errors. Exclude self matches. For each train video `i`, retain a fixed, duplicate-canonicalized universe from:

1. its top-20 nearest same-label and opposite-label neighbours;
2. the opposite-label neighbours contributing to an out-of-fold or leave-one-out wrong train-memory vote;
3. a matched set of same-label hard positives from the same cosine deciles.

The final three-seed universe is the union mined by baseline seeds 0/1/2 before any SSR run; the relation extraction is performed once on that union so the MLLM information does not change by candidate seed. Candidate IDs, labels, baseline cosine, rank, query margin, error stratum, and seed provenance are frozen. Validation and test examples never become graph nodes.

#### 2. Label-blind MLLM relation interface

Use one fixed frozen teacher, `Qwen2.5-VL-7B-Instruct`, with the project’s existing uniformly sampled frames plus timestamped ASR/OCR evidence for each video. The teacher sees the two evidence bundles but **never their gold labels, baseline prediction, rank, or loss role**. It must emit schema-valid JSON only:

```json
{
  "pair_id": "...",
  "target_relation": "same|different|unclear",
  "proposition_relation": "same|opposed|unrelated|unclear",
  "stance_a": "endorse|quote|report|condemn|satire|unclear",
  "stance_b": "endorse|quote|report|condemn|satire|unclear",
  "stance_relation": "same|opposed|unclear",
  "mechanism_a": "slur|dehumanization|threat|exclusion|violence_praise|stereotype|harassment|none|unclear",
  "mechanism_b": "slur|dehumanization|threat|exclusion|violence_praise|stereotype|harassment|none|unclear",
  "mechanism_relation": "same|different|unclear",
  "topic_surface_relation": "same|different|unclear",
  "evidence_binding_a": "visual|speech|ocr|cross_modal|unclear",
  "evidence_binding_b": "visual|speech|ocr|cross_modal|unclear"
}
```

There is deliberately no `hate`, `label`, `correct`, probability, free-form rationale, or confidence field. Run the canonical prompt on `(A,B)` and `(B,A)` and canonicalize the reversal. Retain only fields that agree exactly after reversal; any relation needed by a template that is `unclear`, invalid, or order-inconsistent produces no edge. The pilot additionally uses one paraphrased prompt to measure stability; after Gate B1 the prompt and ontology freeze.

#### 3. Gold-signed edge templates

The MLLM decides only relation type; gold labels `y_i,y_j` decide polarity `s_ij = +1` iff `y_i=y_j`, otherwise `-1`. Only three preregistered templates survive:

- **MI+ (mechanism-invariant positive):** same mechanism, different topic surface or target, and `y_i=y_j`. This asks the geometry to preserve label-relevant mechanism across surface/domain changes.
- **CS− (counter-stance negative):** same target and same proposition, opposed stance, and `y_i!=y_j`. This prevents quotation/condemnation/satire from collapsing onto endorsement.
- **TC− (topic-confound negative):** same topic surface but different target or opposed/unrelated proposition, and `y_i!=y_j`. This prevents lexical/topic similarity from dominating the hate boundary.

No edge is created by an MLLM verdict. If a typed relation and gold sign do not match one of these templates, the pair is simply unused; templates are never edited using validation/test errors.

#### 4. Difficulty-matched signed ranking tuples

Each retained typed edge `e=(i,j,r,s)` is paired with one non-MLLM reference endpoint from the frozen candidate universe:

- for an MI+ edge, choose an opposite-label hard reference `n(e)` in the same baseline cosine decile and nearest available rank;
- for a CS− or TC− edge, choose a same-label hard reference `p(e)` in the same cosine decile and nearest available rank.

Ties are resolved by a seeded ID hash. Thus MLLM information selects the diagnostic endpoint, while gold labels and baseline difficulty construct the comparison. It cannot gain merely by receiving easier positives or harder negatives.

For normalized current embeddings `z`, define

```text
Delta_e = cos(z_i,z_j) - cos(z_i,z_n(e))           if r = MI+
Delta_e = cos(z_i,z_p(e)) - cos(z_i,z_j)           if r in {CS-, TC-}

L_r = mean_{e in E_r} ReLU(m - Delta_e)
L_SSR = (L_MI + L_CS + L_TC) / |nonempty relation types|
L_total = L_RGCL + lambda_SSR * L_SSR.
```

Use the existing triplet margin `m=0.1`; preregister `lambda_SSR=0.2` and do not sweep it by dataset, MLLM size, or test performance. Relation types are averaged separately so a common type cannot erase a rare but diagnostic one. A relation edge minibatch of 8 tuples/type (24 maximum) is drawn at each existing optimizer step; all endpoints are ordinary train videos encoded by the same `f_theta`. This adds no optimizer steps or data and gives gradients to both the typed and reference endpoints. Full, label-only, and shuffled graph arms use the identical sampler workload.

This relative loss is preferred over a relation adapter: it has no latent relation-specific metric that disappears at inference. It directly orders the **same shared cosine geometry** indexed by the final kNN classifier.

### Modern Primitive Usage

- **Primitive:** A frozen multimodal foundation model used for constrained pair-relation extraction.
- **Exact role:** Train-only privileged graph annotator. It supplies equivalence/opposition types over target, proposition, stance, mechanism, and evidence binding; it never supplies labels, scores, embeddings, generated examples, or test predictions.
- **Why it is more natural than an old-school alternative:** Binary labels and cosine ranks cannot identify “same proposition, opposed stance” or “same mechanism, different surface.” Manually annotating every cross-video hard pair is expensive, while an MLLM can parse frames, speech, OCR, and cross-modal stance under a fixed ontology. Gold signing and consistency filtering bound its weaker absolute classification reliability.

### Integration into the Downstream Pipeline

1. Add the signed relation artifact to the train dataset by video ID; do not alter stored image/text features.
2. Keep the current forward path `output, z = model(image_feats, text_feats, return_embed=True)`.
3. Compute the untouched whole-video/segment `L_RGCL` exactly as the comparator.
4. Draw relation tuples, forward their endpoints through the same model, compute `L_SSR`, and add it with fixed `lambda_SSR`.
5. Select checkpoints using the same validation criterion and warmup floor as the exact baseline.
6. At validation/test, call the current `retrieve_evaluate_RAC_`: normalize train/query `z`, build `IndexFlatIP`, retrieve the same top-k, and apply the same vote. The relation files and MLLM are not loaded.

### Training Plan

#### Stage B0 — pair-universe freeze

- Run/recover the exact strongest non-MLLM baseline on train/validation only for seeds 0/1/2.
- Export train embeddings and kNN logs; form the fixed candidate universe and pre-register its hash.
- Prefer MHC-EN and MHC-ZH because they expose complementary speech-heavy and visual/OCR-heavy failures. HateMM is confirmation, not a substitute for two-dataset success.

#### Stage B1 — information-value audit before training

- Randomly choose 250 universe pairs, stratified by dataset, label polarity, cosine decile, query margin, and baseline error involvement.
- Run `(A,B)`, `(B,A)`, and the audit paraphrase. Measure schema validity, order/paraphrase agreement, per-type coverage, eligible edge coverage, baseline-error coverage, and a human-blind precision audit.
- Define a decisive wrong-neighbour event for opposite-label candidates as: the neighbour is in the frozen top-k and its removal either corrects the vote or improves the gold-signed kNN margin by at least the pre-registered upper quartile. Fit a train/validation-only conditional model with gold-label relation, cosine, rank, query margin, dataset, and error stratum as controls; test the incremental likelihood/AUC of the typed relation indicators against generic comparability.
- Proceed only if agreement and blind precision are each at least 0.80, typed edges cover at least 15% of hard pairs and 20% of baseline errors, and typed relations provide statistically positive incremental information. Otherwise SSR is P2/P4 in disguise and stops without scale/prompt rescue.

#### Stage B2 — seed-0 mechanism fast-fail

Train from the same seeded initialization and exact epoch budget:

1. exact non-MLLM RGCL (`lambda_SSR=0`);
2. label-only RankRGCL using the same number, polarity, degree, relation-bucket count, and difficulty of randomly matched hard pairs;
3. SSR-MemRGCL full;
4. SSR-shuffle: within dataset × gold polarity × cosine decile × query-margin/error stratum, reassign relation types/endpoints with constrained degree preservation, keeping edge counts and sampler workload fixed.

Full must beat arms 1, 2, and 4 by at least +0.010 dev accuracy and macro-F1, and the gain must appear in kNN readout plus improved wrong-neighbour topology. Otherwise do not launch the final campaign.

#### Stage B3 — frozen final protocol

- Freeze teacher, prompt, ontology, graph builder, hash, loss, margin, lambda, sampler, seeds, datasets, and checkpoint rule before final test.
- Run paired seeds 0/1/2 for exact baseline, full SSR, remove-MLLM, and SSR-shuffle on MHC-EN and MHC-ZH; include label-only RankRGCL as the necessity control.
- Report both primary metrics, per-seed deltas, mean±std, hierarchical paired-bootstrap intervals over seeds and test examples, and Holm correction over four dataset×metric claims.

### Failure Modes and Diagnostics

- **Relation extraction is unstable:** Detect order/paraphrase agreement below 0.80 or high `unclear` rate. Mitigation: reject the route; do not increase model size or keep prompt-tuning against validation.
- **Typed relations duplicate labels/P4:** Detect no conditional likelihood/AUC gain after label, cosine, rank, margin, dataset, and error controls, or label-only RankRGCL matching full. Mitigation: stop SSR and record redundancy.
- **P2 redux / relation not tied to vote errors:** Detect no enrichment among decisive wrong-neighbour events or generic comparability matching typed relations. Mitigation: stop before training.
- **Coverage too sparse for +3 points:** Detect <15% hard-pair or <20% baseline-error coverage, low graph degree, or one relation type dominating. Mitigation: stop; do not relax ontology post hoc.
- **Graph loss fights RGCL:** Log cosine between `grad(L_RGCL)` and `grad(L_SSR)` on the shared embedding, relation-wise hinge-active rate, and validation kNN purity. Persistent negative gradient cosine plus no topology gain kills the method; no loss-weight sweep rescue.
- **Representation collapse / over-separation:** Track embedding variance, same-label and cross-label cosine histograms, per-class recall, relation-wise margin satisfaction, and train-to-validation neighbour purity. Stop if one class or one language stratum degrades >1 point in the fast-fail.
- **Gain comes from hard-pair compute, not MLLM:** Full fails to beat label-only or shuffle. This falsifies the MLLM mechanism even if full beats the plain baseline.
- **Head–memory redistribution recurs:** Native head improves while kNN is flat/down. This is failure by definition; inference remains kNN.
- **Protocol or artifact leakage:** Pair graph contains a validation/test ID, prompt sees labels/ranks, graph hashes differ across arms, or checkpoint rules diverge. Any occurrence invalidates the run.
- **Target not reached:** A positive but <+0.030 gain, a single dataset/seed win, or only macro-F1 improvement is explicitly `not_working`; it cannot be relabeled as success.

### Novelty and Elegance Argument

RGCL and RA-HMD already cover retrieval-guided hard pairs and LMM contrastive adaptation. RCML covers general relation-conditioned multimodal representation; HateSieve covers generated semantic triplets for hateful memes; CCLRec covers LLM reasoning plus graph-guided contrastive positives; target/stance are known multimodal tasks. Therefore the defensible claim is deliberately narrow:

> In hateful-video RGCL, gold labels fix edge polarity, a frozen MLLM supplies only stable stance–target–mechanism relation types on real hard-neighbour pairs, and a parameter-free signed ranking loss directly repairs the shared geometry consumed by the final train-memory kNN classifier.

The novelty does not rest on any one familiar ingredient. It rests on the causal locus and division of responsibility: the MLLM cannot vote, label, concatenate, generate, or rerank; it selects semantically diagnostic real edges that binary labels cannot identify, while gold labels prevent it from deciding correctness. The method remains one graph-supervision mechanism rather than a pipeline of independent contributions.

## Claim-Driven Validation Sketch

### Claim 1: Typed MLLM relations contain conditional information about RGCL memory errors that labels and generic hard-pair similarity do not

- **Minimal experiment:** Stage B1 audit on 250 train/validation-only pairs across MHC-EN/ZH.
- **Baselines / ablations:** Generic comparability, gold-label relation + cosine/rank/margin controls, shuffled typed relation, human-blind edge audit.
- **Metric:** Order/paraphrase agreement, precision, hard-pair/error coverage, incremental conditional likelihood/AUC for decisive wrong-neighbour events.
- **Expected evidence:** Agreement/precision ≥0.80, coverage ≥15% hard pairs and ≥20% errors, and statistically positive incremental information over generic comparability.

### Claim 2: Gold-signed typed edges causally improve the final kNN memory geometry rather than adding capacity or shifting accuracy to the head

- **Minimal experiment:** Seed-0 fast-fail followed, only if passed, by paired seeds 0/1/2 on MHC-EN and MHC-ZH.
- **Baselines / ablations:** Exact strongest non-MLLM RGCL/remove-MLLM, label-only difficulty-matched RankRGCL, degree/polarity/difficulty-preserving SSR-shuffle; unchanged native-head results reported as diagnostic.
- **Metric:** Final kNN accuracy and macro-F1, per-seed paired deltas and corrected hierarchical-bootstrap CIs; wrong-neighbour rate, neighbour label purity, and typed margin satisfaction.
- **Expected evidence:** Fast-fail full ≥+1 point over all controls in both dev metrics; final full ≥+3 points over the binding same-protocol baseline in both metrics on both datasets, 3/3 positive seeds, corrected CI lower bounds >0, and a significant removal/shuffle cost aligned with repaired topology.

## Experiment Handoff Inputs

- **Must-prove claims:** Conditional relation information exists; its signed graph—not extra compute, difficulty, labels, or head capacity—repairs final kNN geometry and reaches the frozen +3/+3 target.
- **Must-run ablations:** Remove-MLLM/exact baseline; constrained relation shuffle; label-only difficulty-matched RankRGCL. No additional module menu is needed.
- **Critical datasets / metrics:** MHC-EN and MHC-ZH; accuracy and macro-F1; seeds 0/1/2; kNN topology diagnostics; native head only as a redistribution check.
- **Highest-risk assumptions:** Stable stance extraction across EN/ZH; enough eligible hard-pair/error coverage; typed relation remains informative after conditioning; relation loss cooperates with dataset-specific segment-RGCL; +3 points is achievable without relation adapters.

## Compute & Timeline Estimate

- **Estimated GPU-hours:** Audit extraction about 6–12 GPU-hours; full graph extraction about 20–40 GPU-hours for both datasets; frozen-feature RGCL/controls about 15–30 GPU-hours, all through SLURM. Exact cost is gated by retained pair count and measured before B2.
- **Data / annotation cost:** Existing train videos, frames, ASR/OCR only; 250-pair audit with a stratified 20% second human check, roughly 6–10 person-hours; no new training examples.
- **Timeline:** 2–3 days for B0/B1 and audit decision; 1–2 days for seed-0 B2; 3–5 days for the frozen two-dataset, three-seed campaign if and only if gates pass.
