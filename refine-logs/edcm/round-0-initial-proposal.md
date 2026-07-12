# Research Proposal: EDCM-RGCL — Dense Counterfactual Coalition Control of Retrieval Memory Geometry

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Technical Gap

The project has repeatedly shown that plausible MLLM semantics do not automatically move the final retrieval boundary. Absolute verdicts are weaker than the memory vote; schema fields are redundant with the video label; segment scores are absorbed by the align-fusion head; and sparse relation edges do not touch enough errors. SSR makes the last point quantitative: before relation extraction, its most optimistic candidate universe could not possibly create the required dual-metric headroom.

The missing mechanism is neither another semantic input nor another sparse hard-pair rule. It is a **dense per-video counterfactual signal that changes how the whole memory list contributes gradient**. For every training video, deterministic omissions of visual (`V`), speech transcript (`S`) and on-screen/title text (`O`) expose whether the full interpretation depends on one modality or a coalition. A frozen MLLM is used only as a relative teacher across these seven evidence coalitions. It never emits or sees the binary gold label. The resulting signature changes a list-normalized NCA target over current full-video memory keys. Thus the MLLM controls the exact representation geometry read by final kNN, but neither its scores nor its outputs are present at validation/test.

### Why naive alternatives are insufficient

- Choosing an MLLM-weighted segment or memory key repeats P3/P11 and creates full-query/clean-key mismatch.
- Concatenating the signature or rationale repeats P4/P8 and lets the classifier absorb it.
- Weighting isolated hard pairs repeats the sparsity failure and invites direct CGO overlap.
- Training student unimodal coalition views requires changing the multiplicative `align` architecture; that is not the smallest intervention.

### Route comparison

- **Route A — selected minimal route:** keep full-video student queries and memory keys unchanged; use one parameter-free coalition-signature listwise NCA term over the current train memory.
- **Route B — frontier-native teacher-key route:** select a sufficient coalition key per memory video and retrieve against it. This makes the MLLM even more visible, but P11 and code inspection predict a train-key/test-query mismatch and force extra consistency machinery.
- **Decision:** Route A. The frozen MLLM remains a counterfactual teacher, while the only new learning object is the listwise relevance distribution of the final memory.

## Method Thesis

- **One-sentence thesis:** A label-blind MLLM's reliable ranking of deterministic `V/S/O` coalitions can provide a dense per-training-video necessity/synergy signature that changes the **entire RGCL memory-list gradient**, causing the unchanged full-video embedding to organize by label-relevant modality interactions and substantially improve the final kNN classifier.
- **Why this is the smallest adequate intervention:** it reuses the exact encoder, align-fusion head, train feature bank, neighbour miner, optimizer, epochs and final kNN; it adds one offline pseudo-signal and one parameter-free list-normalized loss, with no new trainable component.
- **Why timely:** MLLM intervention comparison is used as privileged causal supervision, not reasoning text or a final judge, and it is internalized into a non-generative retrieval classifier.

## Contribution Focus

- **Dominant contribution:** Evidence-Directed Coalition Memory NCA (`L_EDCM`), in which a reliable train-only MLLM coalition signature sets listwise relevance weights over the exact full-video train memory that final kNN will use.
- **Optional supporting contribution:** none. The structural/dense preflight and controls validate the contribution; they are not separate modules or claims.
- **Explicit non-contributions:** no new backbone, fusion architecture, OCR model, segment localizer, counterfactual generator, test-time reasoner, metric, dataset or annotation.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** strongest per-dataset CLIP-RGCL recipe; fixed frame sampler; frozen CLIP towers; current projection/fusion head; full-video feature bank; neighbour retrieval; binary classification labels; training schedule; final train-memory kNN.
- **New trainable components:** **zero**.
- **New non-trainable artifacts:** one strict train-only coalition-signature cache and one listwise loss function.
- **Intentionally excluded:** segment selection, memory-key replacement, rationale embeddings, adapters, routers, relation graphs, extra heads, MLLM score fusion, prompt/model scaling and SSR stacking.

### System Overview

```text
TRAIN video i only
  existing frames + raw Transcript + Title/fixed OCR-from-same-frames
                       |
          deterministic 7 coalitions C={V,S,O,VS,VO,SO,VSO}
                       |
      frozen label-blind MLLM, 2 prompts x 2 coalition orders
                       |
  reliable relative preservation ranks -> s_i=[3 necessities,3 synergies], rho_i
                       |
full-video RGCL embedding z_i ---- current full-train memory list {z_j,y_j}
                       |                         |
                       +--- signature compatibility c_ij ---+
                                      |
                  one list-normalized memory-NCA loss L_EDCM
                                      |
                unchanged full-video embedding geometry

VAL/TEST video q -> unchanged full-video encoder -> unchanged full-train
memory kNN/vote. No coalition packet, MLLM output or signature is loaded.
```

### A0: Mandatory Pre-MLLM Dense Correctable-Universe Gate

This gate runs **before any MLLM call** and uses only five-fold OOF train embeddings, the video-level binary labels and the exact comparator vote. For every OOF query, search the top-64 fold-local full-video keys; use the comparator's exact final `k`. A query is structurally correctable if the baseline prediction is wrong and replacing at most two opposite-label keys currently inside top-`k` with same-label keys from ranks `k+1:64` flips the exact vote. No segment, semantic or MLLM information is involved.

The route stops before teacher extraction unless both MHC-EN and MHC-ZH satisfy all of:

1. at least 80% of **all** OOF training videos have at least four same-label and four opposite-label candidates within top 64;
2. the structurally correctable error set contains at least `ceil(0.05*N)` unique videos;
3. optimistically correcting all and only those videos yields at least `+0.050` OOF accuracy **and** `+0.050` OOF macro-F1;
4. hashes prove train-only fold disjointness and the candidate universe, predictions, vote rule and metrics are internally consistent.

This is only a geometry/headroom bound, not a method result. It explicitly prevents a repeat of SSR, whose selected candidate arcs could not touch enough errors even under an all-accepted oracle.

### Counterfactual Coalition Teacher

For each train video, construct seven deterministic packets from the same underlying example:

- `V`: four fixed uniformly sampled frames, no speech/OCR text;
- `S`: raw annotation transcript only;
- `O`: raw title plus text returned by one frozen deterministic OCR engine from those same frames;
- `VS`, `VO`, `SO`, `VSO`: exact unions; absent channels are explicitly marked absent.

The OCR output is ordinary teacher input extracted from existing frames, not an annotation and not a gold target. Empty transcript/OCR remains empty; it is never filled by generation. The MLLM receives no dataset label, model prediction, split name, neighbour, margin or training role.

Each call compares all seven packets and returns strict JSON only:

```text
coalition: V|S|O|VS|VO|SO|VSO
preservation: 0|1|2|3
confidence: 0|1|2|3
```

`preservation` asks whether the coalition retains the full video's label-relevant interpretation of target, proposition and stance/context; it does **not** ask for a hateful/non-hateful verdict. `0` means reversed/contradictory context, `1` insufficient/ambiguous, `2` mostly preserved and `3` preserved. The schema contains no rationale, segment, timestamp, target label, stance label, mechanism label or hate score.

Use two frozen prompt wordings and both forward/reverse coalition order: four deterministic calls. After canonicalization, each coalition needs modal agreement at least `3/4` and modal confidence at least 2. A video-level reliability `rho_i` is the minimum agreement over all seven coalitions. Any invalid/missing/unclear/low-confidence coalition makes the entire video signature missing; its `L_EDCM(i)=0`, exactly falling back to ordinary RGCL for that query.

Let the accepted modal ranks be `r_i(C) in {0,1,2,3}`. Derive one six-dimensional signature, with no learned encoder:

```text
n_V = r(VSO)-r(SO)     h_VS = r(VS)-max(r(V),r(S))
n_S = r(VSO)-r(VO)     h_VO = r(VO)-max(r(V),r(O))
n_O = r(VSO)-r(VS)     h_SO = r(SO)-max(r(S),r(O))
s_i = [n_V,n_S,n_O,h_VS,h_VO,h_SO] / 3
```

Necessity and synergy may be negative; this preserves teacher-identified interference rather than clipping it into salience. It is a weak relative pseudo-signal, never causal ground truth.

### Dense-Support and Reliability Gate After Extraction

Teacher extraction is promoted to training only if both datasets independently satisfy:

1. accepted full-signature coverage at least 85%, with a 95% Wilson lower bound at least 0.80;
2. median four-call Kendall rank agreement at least 0.75, plus full reporting of per-coalition agreement/confidence/fallback;
3. non-degeneracy: at least four signature patterns cover at least 5% each and no single pattern covers more than 70%;
4. at least 80% of **all OOF videos**, not merely errors, produce an active list containing four reliable same-label and four reliable opposite-label keys;
5. in the pre-registered structural correctable universe, requiring a reliable non-uniform query signature and reliable candidate signatures still leaves an optimistic video-level upper bound of at least +0.050 accuracy and +0.050 macro-F1 on each dataset.

No segment correctness, span overlap or hidden human localization is used in any gate. Failure stops the route; it is not rescued by a larger teacher, relaxed reliability, more frames, more epochs or a narrower cherry-picked stratum.

### Core Mechanism: Evidence-Directed Coalition Memory NCA

For every reliable full-video training query embedding `z_i`, retrieve from the current detached full-train bank the top `K+=8` same-label and `K-=8` opposite-label keys. This is a list for every reliable video, not a sparse error-edge graph. Let `a_ij=cos(z_i,z_j)` and define fixed signature compatibility

```text
c_ij = exp(-mean(abs(s_i-s_j))) in (0,1].
```

For positive and negative list members separately, normalize the compatibility weights:

```text
q+_ij = softmax(c_ij) over y_j=y_i
q-_ij = softmax(c_ij) over y_j!=y_i
A+_i  = sum_j q+_ij exp(a_ij / tau)
A-_i  = sum_j q-_ij exp((a_ij + m) / tau)
L_EDCM(i) = -log( A+_i / (A+_i + A-_i) )
```

`tau` and margin `m` reuse the exact comparator/RGCL values; no dataset-specific sweep is allowed. A signature-compatible same-label key receives the strongest attraction, while a signature-compatible opposite-label key is the hardest confound and receives the strongest repulsion. Because numerator and denominator are log-sum-exp aggregates over the full candidate list, this is a single memory-NCA/ListNet-style gradient, not independent scalar pair losses or a post-hoc memory reweight.

```text
L_total = L_exact_RGCL + lambda_EDCM * mean_i rho_i L_EDCM(i),
lambda_EDCM = 0.2 (frozen once, before dev/test).
```

The memory keys are the same full-video embeddings used at final inference. No teacher-selected key, signature or MLLM value is written into test-time memory. Removing the teacher term yields the exact current baseline; unreliable samples also take this path.

### Exact Controls

- **Remove-MLLM:** `lambda_EDCM=0`; exact strongest RGCL, bit-for-bit within deterministic tolerance.
- **Label-only ListNCA:** identical list sizes, loss, workload and `lambda`; set `q+` and `q-` uniform. This asks whether the MLLM signature adds information beyond binary labels and generic listwise training.
- **Signature shuffle:** an exact derangement of complete `s_i,rho_i` records within dataset × video-label × confidence-bin × modality-availability pattern, preserving coverage, candidate degrees and workload; no record remains on its source video.
- **Calibrated noise:** at the observed four-call disagreement rate, ordinally corrupt complete coalition-rank records while preserving availability/confidence strata; run the empirical rate on all seeds and twice that rate at seed 0.
- **Absolute-score diagnostic:** optional reporting only, not a competing method: absolute single-view verdict confidence must not substitute for relative coalition reliability.

### Training and Inference

1. A0: run the pre-MLLM OOF correctable-universe and density gate; stop if it fails.
2. A1: extract strict train-only coalition signatures; run reliability, non-degeneracy and post-extraction headroom gates.
3. A2: seed-0 development mechanism gate with paired baseline, label-only, full, shuffle and noise arms from identical initialization and schedule. Full must beat baseline, label-only and shuffle by at least +0.010 dev accuracy and macro-F1; kNN topology must improve without native-head redistribution.
4. A3: only after A2, freeze all hashes and run MHC-EN/ZH seeds 0/1/2 under the original final protocol.

At validation/test, the data loader, full-video encoder, train-memory construction, FAISS search and vote are unchanged. No coalition packet, OCR teacher artifact, MLLM output, confidence or signature is loaded.

### Failure Modes and Diagnostics

- **Structural headroom sparse:** A0 stops before any MLLM cost.
- **Teacher nearly constant or unreliable:** coverage/rank-pattern gates stop; report missingness by label and modality availability.
- **Label leakage/redundancy:** teacher input audit forbids labels; Label-only ListNCA matching/full comparison is binding.
- **Generic gradient weighting rather than MLLM mechanism:** shuffle and calibrated-noise must erase the gain; report gradient cosine between `L_RGCL` and `L_EDCM`, active-query fraction and per-query gradient norm.
- **P3-style absorption:** require improvement in the actual kNN readout and wrong-neighbour rate, not only the native output layer.
- **Class fragmentation:** report same-label neighbour purity, signature-cluster connectivity, per-class recall and embedding variance. Collapse or improved head with flat/down kNN is failure.
- **Pseudo-signal missingness:** entire-query `L_EDCM=0` fallback, no imputation. Report coverage/confidence/fallback and control metrics.
- **No segment gold:** assert the signature schema contains no segment/timestamp/span field; every written document and payload calls it weak train-only pseudo-signal.

## Modern Primitive Usage

- **Primitive:** a frozen MLLM used as a counterfactual coalition critic/teacher.
- **Exact role:** compare deterministic omissions of the same train video and emit only relative preservation ranks; these define a listwise gradient controller for a small discriminative retrieval model.
- **Why natural:** the MLLM's useful competence in this project is relative multimodal interpretation, while its absolute final verdict is weak and unstable. The interface exploits the former and discards the latter.

## Integration into RGCL

EDCM attaches inside `compute_loss` after the existing full-video embeddings and train feature bank are available. It reuses current nearest same/opposite-label mining and returns one scalar. It does not change `classifier_hateClipper`, the frozen encoders, the checkpoint criterion or `retrieve_evaluate_RAC_`. Final memory contains ordinary full-video embeddings only.

## Novelty and Elegance Argument

CGO controls harmful-video gradients using student perturbation reliability and convergence; general modality-interference work uses intervention consistency; RAMF/IARE feed or train reasoning; RGCL/RA-HMD provide retrieval hard pairs. EDCM's narrow claim is different:

> A frozen MLLM's relative same-video modality-coalition signature defines the list-normalized attraction/repulsion measure of the final RGCL train memory, densely for every reliable training video, and is then discarded.

The paper must not claim first causal intervention, first gradient control or first MLLM hateful-video reasoning. Its novelty is the **retrieval-specific composition**: relative coalition teacher → dense signature → exact final-memory listwise gradient → unchanged kNN readout. There is one mechanism and zero new trainable modules.

## Claim-Driven Validation Sketch

### Claim 1: The route has dense, reliable and target-sized train-only causal support

- **Minimal experiment:** A0 structural OOF correctable-universe before MLLM, followed by A1 four-call signature reliability/density and post-extraction headroom on MHC-EN/ZH.
- **Baselines/ablations:** SSR upper-bound record as negative reference; uniform/label-only signature; complete-record shuffle for non-degeneracy.
- **Metric:** active all-video coverage, correctable unique error count, dual-metric optimistic gain, Wilson coverage lower bound, Kendall agreement, signature entropy/pattern support.
- **Decisive evidence:** both datasets pass every pre-registered gate without any segment annotation.

### Claim 2: MLLM coalition information causally changes final memory geometry

- **Minimal experiment:** paired seed-0 baseline/remove, Label-only ListNCA, full EDCM, signature-shuffle and calibrated-noise under identical workload.
- **Metric:** dev accuracy/macro-F1, final kNN neighbour purity/wrong-neighbour rate, gradient alignment, per-class recall; head vs kNN difference.
- **Decisive evidence:** full beats each baseline/label-only/shuffle by at least +0.010 in both dev metrics; noise degrades monotonically; the kNN readout itself improves.

### Claim 3: The method meets the actual target

- **Minimal experiment:** MHC-EN and MHC-ZH, paired seeds 0/1/2, exact moving baselines and frozen final test.
- **Metric:** accuracy and macro-F1; mean±std; hierarchical paired bootstrap; Holm correction; full-minus-remove/shuffle/noise costs.
- **Decisive evidence:** all target conditions in the immutable Problem Anchor pass jointly. Anything less remains `not_working`.

## Experiment Handoff Inputs

- **Must-prove claims:** dense support before teacher cost; reliable/non-degenerate relative signatures; conditional advantage over label-only listwise training; final kNN +3/+3 on two datasets and three seeds.
- **Must-run ablations:** remove-MLLM, Label-only ListNCA, complete-signature shuffle, empirical noise, 2×noise seed 0.
- **Critical datasets/metrics:** MHC-EN and MHC-ZH; final accuracy and macro-F1.
- **Highest-risk assumptions:** the structural top-64 universe has sufficient two-swap dual-metric headroom; reliable coalition patterns are diverse; modality-pattern compatibility is conditionally useful rather than another label-redundant weighting.

## Compute & Timeline Estimate

- **Estimated compute:** A0 OOF preflight reuses existing OOF embeddings/predictions and needs no MLLM; A1 approximately 4 deterministic calls per training video (about 4,500 calls for EN+ZH), estimated 10–30 GPU-hours after a frozen throughput smoke; A2/A3 roughly 30–60 GPU-hours depending on baseline cache reuse. Every compute task is submitted through SLURM.
- **Data/annotation cost:** no new gold annotation and no segment annotation. Optional blinded video-level relation audit is diagnostic only and is never loaded by training.
- **Timeline:** A0 0.5 day; strict input/schema smoke and A1 extraction 1–2 days; A2 seed-0 gate 1 day; A3 paired-seed final 2–3 days if unlocked.
