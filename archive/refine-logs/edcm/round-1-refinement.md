# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Anchor Check

- **Original bottleneck:** sparse or redundant MLLM information has not produced a broad, removable change in final kNN geometry.
- **Preservation:** the method remains train-only, full-video, final-memory and +3/+3 classification focused. The only gold remains the video binary label; fixed frames and all coalition outputs are weak inputs/pseudo-signals, not segments or annotations.
- **Reviewer suggestions rejected as drift:** none. We accept the teacher-free control and gradient-density gate because they test the same contribution. We do not add test-time MLLM, teacher keys, student coalition branches, routers or localization.

## Simplicity Check

- **Dominant contribution:** relative MLLM coalition judgments define a train-only conditional neighborhood measure whose teacher-specific gradient changes the final kNN embedding geometry.
- **Components removed/merged:** remove the absolute-score diagnostic; replace global signature-pattern and post-extraction “upper-bound” gates with one direct teacher-active list-gradient gate.
- **Added control, not module:** one deterministic availability/content proxy uses the exact EDCM loss and workload to test MLLM indispensability; it has no trainable component.
- **Smallest adequate route:** one frozen teacher cache, one zero-parameter listwise loss, zero new trainable modules, no inference artifact.

## Changes Made

### 1. Corrected the list kernel

- **Reviewer said:** `softmax(exp(-d))` double-compresses compatibility and leaves the scale unclear.
- **Action:** use `q=softmax(-d/tau_s)` once with shared `tau_s=1.0`, frozen across datasets and before development.
- **Impact:** a clear list-normalized teacher measure with no dataset sweep.

### 2. Replaced availability density with actual MLLM-gradient density

- **Reviewer said:** accepted signatures can be homogeneous inside actual lists and degenerate to Label-only ListNCA.
- **Action:** define total variation from uniform and the exact query-gradient difference `Delta g`; require both for a query to be teacher-active. Bind promotion to all-OOF active coverage and a fixed correctable-error directional-margin test against Label-only.
- **Impact:** “dense” now means broad nonzero MLLM-specific training influence, not merely a full cache.

### 3. Added the minimal indispensability control

- **Reviewer said:** the signature might encode only channel presence/length.
- **Action:** add one six-dimensional teacher-free availability/content proxy, masked to identical coverage and run through identical lists/loss/workload.
- **Impact:** any gain explainable by low-level modality availability cannot support the MLLM claim.

### 4. Tightened terminology and reproducibility

- **Reviewer said:** coalition ranks are not causal ground truth and the protocol is underspecified.
- **Action:** call them interventional coalition weak pseudo-signals; freeze Qwen2.5-VL-7B, decoding, prompts, frames, OCR checkpoints, serialization and canonicalization in a pre-extraction manifest.
- **Impact:** no semantic overclaim and an implementation-ready teacher interface.

## Revised Proposal

# Research Proposal: EDCM-RGCL — Dense Interventional Coalition Control of Retrieval Memory Geometry

## Problem Anchor

- **Bottom-line problem:** Make an MLLM a meaningful, novel, causal and removable part of hateful-video RGCL, and do not stop until one frozen method improves **final test accuracy and macro-F1 by at least +0.030 absolute each** over the moving strongest same-protocol non-MLLM RGCL comparator on **at least two datasets**, using paired seeds 0/1/2.
- **Must-solve bottleneck:** Prior MLLM routes supplied sparse neighbour events, absolute verdicts, static segment salience, extra embeddings, auxiliary semantic fields or a competing native head. They were sparse, redundant with the video label, absorbed by the fusion head, or merely redistributed accuracy between head and memory. SSR now adds decisive evidence: even its optimistic all-candidate OOF oracle touched only 2/7 EN and 3/15 ZH unique MI/SC error queries and could not reach its dual-metric gate. The successor must therefore provide a **reliable, dense, per-training-video causal signal** that directly changes the listwise gradient of the same full-video embedding geometry used by the final kNN memory.
- **Non-goals:** Localization-only, explanation-only, audit/guard-rail-only or native-head-only success; test-time MLLM annotation, judging, score fusion, reranking or veto; simple MLLM score/embedding/rationale concatenation; static segment weighting or segment-weighted memory; generated counterfactual content; a second parallel method stacked with SSR; gains primarily from a larger model, more data, more epochs/steps, ensembling, altered preprocessing, altered checkpoint selection, changed retrieval/voting, changed labels or any protocol relaxation.
- **Constraints:** The **only gold supervision is the video-level binary label**. No segment-level gold annotation exists or may be assumed. Every MLLM modality, coalition, necessity, sufficiency, preservation, stance, target, mechanism, rationale, localization or segment output is a confidence-bearing **weak/privileged train-only pseudo-signal**, never gold, dense annotation or oracle evidence. Validation/test receive no such annotation or pseudo-signal; low-confidence, missing or invalid train pseudo-signals deterministically reduce to the exact non-MLLM path. Use the exact strongest per-dataset RGCL comparator, fixed splits/preprocessing/labels/epochs/checkpoint rule/retrieval/seeds. Required controls are remove-MLLM, within-split signature shuffle and calibrated noise/corruption, with coverage/confidence/fallback reporting. All later compute is SLURM-only in `HateVideo`, without `--time`, within 2 GPUs/16 CPUs/128 GB.
- **Success condition:** On at least two datasets and paired seeds 0/1/2, both final accuracy and macro-F1 gain at least +0.030 over `max(historical strongest point, paired baseline mean)`; every seed delta is positive; mean±std and hierarchical paired-bootstrap uncertainty are reported; the four dataset×metric primary tests pass Holm-corrected familywise α=0.05 with 95% lower bounds above zero. The full method must beat remove-MLLM and shuffled-MLLM controls with same-direction paired effects and 95% CIs excluding zero in both metrics, survive calibrated corruption, improve the **kNN readout itself** without head↔memory redistribution, and retain a defensible retrieval-specific novelty claim. Any missing metric/dataset/seed/statistical/mechanism/supervision/protocol item is `not_working`, not success.

## Technical Gap

Existing MLLM signals have been sparse, redundant or absorbed. SSR's optimistic candidate universe proves that sparse decisive-neighbour events cannot reach the target. EDCM instead asks one narrow question for every train video: which deterministic combination of visual (`V`), dataset-provided speech transcript (`S`) and on-screen/title text (`O`) preserves the full video's label-relevant interpretation? The MLLM never sees or predicts the binary label. Its relative answers become a six-dimensional **interventional coalition weak pseudo-signal**. That signal changes the normalized gradient over the exact full-video train-memory list; it is absent at validation/test.

This is not segment salience, a generated view, a signature feature, a selected memory key or a pairwise semantic edge. The student always receives and stores ordinary full-video embeddings. The MLLM affects only one listwise training measure.

## Method Thesis and Contribution

- **Thesis:** Relative MLLM coalition judgments define a train-only conditional neighborhood measure whose teacher-specific gradient changes the final kNN embedding geometry.
- **Dominant contribution:** Evidence-Directed Coalition Memory NCA (`L_EDCM`).
- **Complexity:** zero new trainable components; one strict pseudo-signal cache and one scalar loss.
- **MLLM role:** frozen counterfactual coalition critic/teacher, not classifier, feature encoder, localizer or test-time reasoner.

## System Graph

```text
train video i: fixed frames + dataset transcript + title/fixed OCR
       -> deterministic {V,S,O,VS,VO,SO,VSO} packets
       -> frozen label-blind MLLM, 2 prompts x 2 orders
       -> ordinal distributions -> reliable s_i, rho_i

full-video z_i + current full-video memory {z_j,y_j}
       -> q_i^+,q_i^- from signature distances
       -> one list-normalized L_EDCM + exact L_RGCL
       -> full-video geometry used by unchanged final kNN

validation/test: full-video encoder -> ordinary train-memory kNN only
                 (no MLLM, packet, OCR artifact, signature or confidence loaded)
```

## Complexity Budget

- **Frozen/reused:** exact strongest dataset-specific RGCL recipe, CLIP towers, frame sampler, projection/align-fusion head, feature bank, miner, optimizer, epochs, checkpoint selection, final FAISS/vote and seeds.
- **New trainable pieces:** none.
- **New non-trainable pieces:** a hashed teacher protocol/cache and `L_EDCM`.
- **Excluded:** teacher-selected keys, student coalition views, relation graphs, adapters, routers, extra heads, rationales, segment losses, test-time pseudo-signals, SSR stacking.

## A0: Pre-MLLM Frozen-Geometry Reachability/Cost Screen

Before **any** MLLM call, use only five-fold train OOF full-video embeddings, the video-level binary labels and exact comparator vote. Search each query's top-64 fold-local keys. A wrong query is structurally reachable if replacing at most two opposite-label keys currently in top-`k` with same-label keys from ranks `k+1:64` flips the exact vote.

Stop before teacher extraction unless, on both MHC-EN and MHC-ZH:

1. at least 80% of all OOF train videos have at least four same-label and four opposite-label candidates in top 64;
2. at least `ceil(0.05*N)` unique errors are structurally reachable;
3. correcting all and only reachable errors yields at least +0.050 OOF accuracy and +0.050 OOF macro-F1;
4. fold disjointness, candidate lists, predictions, vote and metric hashes verify.

A0 is a conservative fixed-geometry reachability and cost screen. It is **not** an upper bound on learned EDCM, because learning can move keys. Passing does not establish MLLM usefulness; failing means this frozen candidate locality is too narrow to justify teacher cost under the registered route.

## Frozen Teacher Protocol

Before extraction, write a manifest with SHA-256 hashes for every item below:

- MLLM: local snapshot of `Qwen/Qwen2.5-VL-7B-Instruct`; bf16; `do_sample=false`, `temperature=0`, `top_p=1`, `max_new_tokens=384`;
- visual packet: four uniform frames at the processor's frozen 336-pixel setting, ordered by timestamp;
- `S`: dataset-provided transcript used as a raw input modality; it contains no temporal or segment-level gold annotation;
- `O`: dataset title plus OCR from the same four frames using the local PP-OCRv4 Chinese detector/recognizer and `ch_ppocr_mobile_v2.0` angle classifier; exact local checkpoint directory hashes and engine version are frozen; empty stays empty;
- coalition serialization: fixed channel tags, Unicode NFC, whitespace collapse, each text channel head/tail truncated to 1,024 Unicode code points before union;
- two prompt files and two coalition orders: `[V,S,O,VS,VO,SO,VSO]` and its exact reverse;
- strict JSON schema, parser, canonicalizer, rank distribution/tie rules and deterministic full-video fallback.

Both prompt wordings ask only: “Compared with the complete `VSO` evidence, how faithfully does each packet preserve the interpretation needed to distinguish targeted harmful endorsement from quotation, condemnation, reportage, satire or unrelated/offensive context? Do not output a hateful/non-hateful label.” Prompt 2 is a frozen semantic paraphrase, not a post-result repair. Exact texts live in the manifest before a smoke.

Each of four deterministic calls outputs for all seven coalitions:

```text
coalition: V|S|O|VS|VO|SO|VSO
preservation: 0|1|2|3
confidence: 0|1|2|3
```

There is no label, rationale, target field, stance field, mechanism field, timestamp, span or segment score. For coalition `C`, retain the four-call ordinal distribution `pi_i^C(r)` and expected rank `rbar_i(C)`. Accept only if modal rank agreement is at least 3/4 and modal confidence at least 2 for all seven coalitions. `rho_i` is the minimum modal agreement. Invalid/missing/low-confidence in any coalition makes the whole signature missing and sets `L_EDCM(i)=0`.

Derive, without learned parameters:

```text
n_V = rbar(VSO)-rbar(SO)     h_VS = rbar(VS)-max(rbar(V),rbar(S))
n_S = rbar(VSO)-rbar(VO)     h_VO = rbar(VO)-max(rbar(V),rbar(O))
n_O = rbar(VSO)-rbar(VS)     h_SO = rbar(SO)-max(rbar(S),rbar(O))
s_i = [n_V,n_S,n_O,h_VS,h_VO,h_SO] / 3
```

Negative values are retained. This is an interventional weak pseudo-signal, never ground-truth causality.

## Core Mechanism: EDCM Listwise Memory NCA

For every reliable full-video query `z_i`, mine from the current detached full-train bank the top `K+=8` same-label and `K-=8` opposite-label full-video keys. Let

```text
d_ij = mean(abs(s_i-s_j))
q+_ij = exp(-d_ij/tau_s) / sum_{k in L_i+} exp(-d_ik/tau_s)
q-_ij = exp(-d_ij/tau_s) / sum_{k in L_i-} exp(-d_ik/tau_s)
tau_s = 1.0, shared and frozen
a_ij = cos(z_i,z_j)
A+_i = sum_{j in L_i+} q+_ij exp(a_ij/tau_rgcl)
A-_i = sum_{j in L_i-} q-_ij exp((a_ij+m_rgcl)/tau_rgcl)
L_EDCM(i) = -log(A+_i/(A+_i+A-_i))
L_total = L_exact_RGCL + 0.2 * mean_i rho_i L_EDCM(i)
```

`tau_rgcl`, `m_rgcl` and `lambda=0.2` are frozen once from the exact comparator/one route-wide registration, never tuned by dataset or test. Compatible same-label keys are strongest positives; compatible opposite-label keys are hardest confounds. The log-sum-exp numerator/denominator forms one normalized memory-list gradient rather than independent edge losses. Final memory keys remain ordinary full-video embeddings.

## Binding Teacher-Active Dense-Support Gate

Run on the frozen OOF baseline geometry after extraction. Label-only ListNCA is the exact same equation with uniform `u+=1/K+`, `u-=1/K-`. Define

```text
TV_i = 0.5 * [ TV(q_i+,u_i+) + TV(q_i-,u_i-) ]
Delta g_i = grad_{z_i} L_EDCM(i) - grad_{z_i} L_uniform(i)
R_i = ||Delta g_i||_2 / (||grad_{z_i} L_uniform(i)||_2 + 1e-12)
teacher_active(i) := TV_i >= 0.10 and R_i >= 0.10
```

Promotion requires independently on both datasets:

1. complete reliable signature coverage at least 85%, Wilson 95% lower bound at least 0.80;
2. at least 80% of all OOF videos have reliable 8+8 lists;
3. at least 70% of **all OOF videos** are teacher-active; report TV/R distributions rather than only the pass count;
4. among A0 structurally reachable errors, at least 60% are teacher-active;
5. define the differentiable gold neighbourhood margin `mu_i=logsumexp(a_same/tau_rgcl)-logsumexp(a_opp/tau_rgcl)`. At equal unit query-step norm, compare directional derivatives along `-grad L_EDCM` and `-grad L_uniform`. The mean teacher-minus-uniform derivative on reachable errors must be positive with a query-bootstrap 95% lower bound above zero, and at least 60% of reachable errors must have positive individual difference.

The thresholds are route-wide and frozen before extraction. This gate directly tests broad MLLM-specific gradient influence; cache coverage alone cannot pass it. It uses only video-level labels and full-video OOF geometry. No segment gold, span, localization or human dense annotation exists or is used.

## Exact Controls

- **Remove-MLLM:** `lambda=0`, exact RGCL.
- **Label-only ListNCA:** uniform positive/negative weights, identical list/loss/workload.
- **Teacher-free modality/content proxy:** six deterministic low-level values `[frame availability, log1p transcript length, log1p title+OCR length, V*S, V*O, S*O]`, train-only robust-scaled to `[0,1]`; mask it with the exact accepted-signature mask so coverage, lists, loss, lambda and compute equal full EDCM. It contains no semantic model output.
- **Signature shuffle:** derange complete `(s_i,rho_i)` records within dataset × video label × confidence bin × raw modality-availability pattern; preserve coverage/degrees/workload, forbid fixed points.
- **Calibrated noise:** corrupt complete ordinal-distribution records at the observed four-call disagreement rate within availability/confidence strata; empirical rate all seeds and twice-rate at seed 0.

Full EDCM must beat all of remove, Label-only, teacher-free proxy and shuffle. The proxy is one control, not another contribution or module.

## Training and Inference Sequence

1. **A0:** mandatory pre-MLLM frozen-geometry reachability/cost screen.
2. **A1:** frozen teacher manifest/smoke, train-only extraction, reliability and binding teacher-active gate.
3. **A2:** seed-0 paired baseline/remove, Label-only, proxy, full, shuffle and noise. Full must beat baseline, Label-only, proxy and shuffle by at least +0.010 dev accuracy and macro-F1, improve kNN topology and avoid head↔memory redistribution.
4. **A3:** only after A2, freeze hashes and run MHC-EN/ZH seeds 0/1/2 under the immutable final protocol.

Validation/test inference is exactly the existing full-video encoder, ordinary full-train memory, FAISS and vote. No teacher-side OCR, coalition packet, rank, confidence or signature is generated or loaded.

## Failure Modes and Diagnostics

- A0 sparse reachability -> stop before teacher cost.
- High cache coverage but uniform in-list weights -> teacher-active gate stops.
- Availability proxy matches full -> MLLM is unnecessary; stop.
- Shuffle/noise does not remove gain -> no causal attribution; stop.
- Native head rises while kNN is flat/down -> P3/P9 redistribution; stop.
- Low confidence/missing -> entire-query exact RGCL fallback; report label/modality missingness.
- Report gradient cosine with base RGCL, TV/R, active fraction, neighbour purity, wrong-neighbour rate, embedding variance, class recall and head-vs-kNN delta.
- Assert every cache and payload has no segment/timestamp/span field. Fixed frames are inputs, not annotated segments.

## Novelty and Elegance

CGO controls harmful-video gradients from student perturbation/convergence; modality-interference work uses causal consistency; RAMF/IARE use reasoning text/training; RGCL uses retrieval hard pairs. The defensible claim is narrow:

> Relative label-blind MLLM judgments over deterministic same-video modality coalitions define a privileged conditional neighbourhood measure; its demonstrably non-uniform teacher-specific gradient is internalized into the exact full-video geometry read by final kNN and then discarded.

The matched Label-only, teacher-free proxy, shuffle and noise controls separate MLLM reasoning from video labels, generic listwise training, modality availability and arbitrary per-video signatures. There is one contribution, no new trainable module and no test-time semantic path.

## Claim-Driven Validation

### Claim 1: teacher influence is dense, reliable and non-redundant

- **Experiment:** A0/A1 on EN/ZH.
- **Metrics:** frozen reachability, signature coverage/agreement, all-OOF teacher-active fraction, TV/R, reachable-error directional-margin advantage.
- **Controls:** Label-only and matched modality/content proxy.
- **Evidence:** every frozen gate passes on both datasets without segment annotation.

### Claim 2: the MLLM-specific list gradient repairs final memory geometry

- **Experiment:** A2 paired seed 0.
- **Metrics:** dev acc/macro-F1, kNN purity/wrong-neighbour rate, gradient diagnostics, head-vs-kNN.
- **Controls:** remove, Label-only, proxy, shuffle, calibrated noise.
- **Evidence:** full is at least +0.010 in both metrics over every binding clean control; corruption degrades monotonically; kNN itself improves.

### Claim 3: EDCM satisfies the immutable endpoint

- **Experiment:** A3, MHC-EN/ZH × seeds 0/1/2.
- **Metrics/evidence:** exact +3/+3, 3/3 positive seed deltas, mean±std, hierarchical paired bootstrap, Holm correction, significant remove/shuffle costs and complete confidence/fallback/noise reporting.

## Compute and Handoff

- A0 reuses OOF full-video artifacts and precedes any MLLM work.
- A1 is approximately four deterministic 7B calls per train video (~4,500 calls for EN+ZH), estimated 10–30 GPU-hours after smoke.
- A2/A3 estimated 30–60 GPU-hours with baseline/cache reuse.
- No new gold or segment annotation. Optional human interpretation audit is aggregate diagnostic only and never enters training.
- All compute is SLURM-only in `HateVideo`, no `--time`, within the fixed resource ceiling.

## Experiment Handoff Inputs

- **Must prove:** A0 broad reachability; A1 broad teacher-active gradients and proxy separation; A2 causal kNN repair; A3 actual target.
- **Must run:** remove, Label-only, teacher-free proxy, signature shuffle and calibrated noise.
- **Highest risks:** the top-64 two-swap universe is too small; the teacher is reliable but within-list homogeneous; the low-level proxy explains the effect; teacher gradient does not improve reachable-error margin beyond Label-only.
