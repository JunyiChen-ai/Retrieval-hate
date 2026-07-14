# ROUND-3 NOVELTY-MECHANISM CANDIDATES (2026-07-14)

**Scout:** round-3 novelty-mechanism ideation scout. ZERO GPU (design + literature only).
**Mandate:** find a mechanism that is (a) NOVEL within hateful-video detection, (b) meaningfully
integrates Qwen2.5-VL-7B (local; + CLIP), (c) can plausibly clear **≥+0.03 acc AND ≥+0.03 macro-F1,
3/3 seeds, both protocols** on ≥1 of HateMM (744 tr) / MHC-EN (549) / MHC-ZH (579), non-isomorphic to
the 22 dead routes + banned list, and compatible with every veto.
**Method:** read the graveyard (`state/directions_tried.json`, `REFLECTION_mllm_integration_failures.md`,
`EXHAUSTION_AUDIT_2026-07-14.md`, `LITERATURE_mllm_integration_2026-07-13.md`) → derive the one
structural axis every dead route shares → cross-domain literature sweep (2020-2026 top venues) with
every load-bearing citation fetched and verified (§Provenance). No fabricated cites.

---

## 0. THE STRUCTURAL HOLE THAT GENERATES ALL FIVE CANDIDATES

Re-read the 22 dead routes through one lens — **what object does the retrieval operate on?** Every
single route, positive and negative, retrieves/votes over **one pooled vector per video** (Qwen
last-layer mean-pooled 3584-d, or CLIP pooled), and then either (a) swaps which encoder produces that
pooled vector (encoder-swap, B1, B2, P9, C1-QLoRA — the only class that ever cleared +3, HateMM only),
or (b) bolts a low-bandwidth decision-side signal onto the pooled-vector vote (P1-P5, P10, P11, TARC,
archive-repair — all dead by **D1**). SAV probed whether other heads/layers add label *information*
beyond the pooled vector and found a null (U-1 ≈ last-layer-pooled).

**Nobody ever changed the retrieval OBJECT itself.** Qwen2.5-VL is a *video-language* model: a video
is natively a **set/sequence of per-frame (and per-patch) language-aligned token embeddings**, and the
pipeline throws all of that away by mean-pooling before the very first retrieval operation. The three
components the diagnosis frame names — representation, retrieval metric, memory — have each only ever
been touched at the **pooled-vector** granularity. That is the hole. It is a **representation-geometry**
hole, which is exactly the class **D2** says is the only one that ever clears +3, and it is **not** a
decision-side signal add, so **D1** does not bite it.

Three project-internal facts say this hole is where the signal is:
- **P6 (POSITIVE):** the MLLM *segment* localizer beats the pooled memory (wv-AUC 0.5435 vs 0.5140,
  paired p=0.007). Hate is **locally** concentrated in videos; the project already proved segment
  structure carries label signal that the pooled vector dilutes.
- **encoder-swap (POSITIVE):** the only +3 lever, and it is representation-level.
- **SAV's own null is about information CONTENT, not MATCHING GEOMETRY** — see §C1 non-isomorphism.

The cross-domain literature says the same thing independently and loudly: in **few-shot video
classification** (the closest analog to a ~600-1000-video hateful-video kNN task), **set-to-set /
temporal-alignment matching over per-frame features robustly and substantially beats pooling** — OTAM
(CVPR 2020, +significant on Kinetics/SSv2 by aligning per-frame distances instead of pooling), DeepEMD
(CVPR 2020, EMD optimal-matching over dense local regions, +significant on 5 few-shot benchmarks), CMOT
/ TSAM (optimal-transport set-matching for few-shot action). In **retrieval**, ColBERT (SIGIR 2020),
ColPali (ICLR 2025, late interaction over *VLM patch* embeddings) and Video-ColBERT (CVPR 2025,
frame-level late interaction over video tokens) all show that **preserving token/patch/frame vectors +
a set-matching operator (MaxSim/EMD/OT) beats single-pooled-vector retrieval**. **None of this exists in
hateful-video or hateful-meme detection** — RGCL (ACL 2024), RA-HMD (EMNLP 2025), and MoRE (WWW 2025,
the in-domain retrieval-augmented SOTA) all retrieve over pooled single vectors.

Candidates C1-C3 each occupy this hole at a different injection point (retrieval metric / memory
structure / training objective). C4-C5 are the two remaining literature-suggested cells, included for
completeness with honest low priors.

---

## C1 — [LEAD] Set-to-set retrieval over Qwen-VL per-frame tokens ("hate is local, so match locally")

**Mechanism (1 para).** Stop mean-pooling. Run one frozen Qwen2.5-VL-7B forward and keep the
**per-frame** hidden-state set `{f_1..f_T}` (T=8-16; optionally per-patch for a larger set) for every
video. The alignment head (triplet+BCE) projects each frame vector. Retrieval distance between a query
video and a memory video becomes a **set-matching** score — MaxSim / TopK-Sim (ColBERT/ColPali family)
or an optimal-transport / temporal-alignment distance (DeepEMD/OTAM family) — instead of cosine between
two pooled vectors. Top-20 kNN vote is unchanged; only the object retrieved and the metric change. The
mechanism says: two hateful videos that share a hateful *segment* but differ globally will now match on
that segment, whereas pooled cosine averages the match away.

**Injection point + bandwidth class.** Retrieval representation **and** retrieval metric
(representation-geometry level). Bandwidth = **T× the current** (a set of 8-16 frame vectors vs one
pooled vector) — a strictly *higher*-bandwidth representation, the opposite of the low-bandwidth
decision-side adds that D1 kills. Encoder unchanged; no MLLM scalar/score; no added channel.

**Non-isomorphism vs the routes it most resembles.**
- **vs P3 (segment hate-density pooling, dead):** P3 kept ONE pooled vector and re-weighted segments by
  an **MLLM hate-density SCORE** (decision-side, ~tens of bits, score-driven). C1 keeps the **full frame
  set** and changes the **distance metric**; no score, no weighting — matching is data-driven
  (MaxSim/OT). Different injection point, different bandwidth class, no MLLM score. Non-isomorphic.
- **vs SAV (attention-head mining, 18th, U-1 null):** SAV asked "do other heads/layers carry label
  *information* beyond last-layer-pooled?" → null. C1 makes **no** extra-information claim; it claims the
  **matching geometry** is better when you align frame-to-frame rather than pool-then-match. This is the
  DeepEMD/OTAM thesis: **pooling discards ALIGNMENT, not bits** — even if every frame's information is
  contained in the pooled average, the pooled cosine cannot *align* the shared hateful frame across two
  videos. SAV's information-content null does not touch this; C1's own probe (set-kNN AUC vs pooled-kNN
  AUC) resolves it empirically. Non-isomorphic on mechanism, with a built-in kill switch.
- **vs encoder-swap (positive):** swapped WHICH encoder; kept pooled retrieval. C1 keeps the encoder,
  changes pooled→set. Different injection point; **composable** with the Qwen encoder-swap.
- **vs P6 (localizer, positive):** P6 *scores* segments for a localization read-out. C1 *matches*
  frame-sets for classification retrieval — a different operation. P6 supplies the premise (hate is
  local); C1 is the first route to convert that premise into the **main-table retrieval** accuracy.
- **vs MoRE (WWW 2025, in-domain closest prior):** MoRE retrieves whole instances with a pooled joint
  video retriever, then feeds them to a mixture-of-experts (retrieval → classifier-conditioning). C1
  changes the retrieval **distance** itself and votes; no MoE, no instance-as-context. Different object.

**Novelty within hateful-video.** No hateful-video or hateful-meme method uses token/frame-set or
late-interaction retrieval — RGCL/RA-HMD/MoRE are all pooled-single-vector. The raw *mechanism* (set-
matching for few-shot video) is established (OTAM, DeepEMD, CMOT), so the honest novelty is **domain +
representation transfer**: first set-to-set retrieval in hateful-video, first over **MLLM video-language
tokens** rather than raw CNN features, inside a retrieval-contrastive kNN-vote head. This is the same
*class* of "transfer novelty" the user already entertains for encoder-swap and LoRA; whether it clears
the novelty clause is the same pending user ruling as B3 — but on the **performance** clause it has the
best prior of anything left.

**Why it could clear +3 where 22 didn't (D1/D2/D3 engaged).** **D2:** representation-geometry — the only
class that ever cleared +3 — and this is the *richest untried* member of it. **D1:** adds no decision-
side signal, so the redundancy law does not apply; the "is the extra structure redundant?" question is
answered empirically, not assumed. **D3:** set-matching gains in few-shot video are typically **3-8 pts**
(OTAM/CMOT), i.e. above the ±1-2pt floor, *and* the probe measures the **paired** Δ (set-kNN vs pooled-
kNN on identical features/seeds), which cancels seed noise and makes a 3pt effect measurable. Realistic
band: strongest expected on **HateMM/MHC-EN** where the encoder-swap already showed Qwen represents these
well but MHC-EN failed to convert (locality/dilution is the SAV-hypothesized failure mode — set-matching
attacks it directly). Honest risk: on 149-300 test videos even a real +3 needs the paired protocol to
show, and OT/temporal metrics add hyperparameters (T, K, alignment).

**Veto compliance.** Single-dataset own-train-split memory ✓; no OCR ✓; no gold annotations ✓ (frames
are unlabeled tokens); no cross-seed ensemble ✓ (within-model kNN); no external API ✓; no MLLM-score-as-
signal ✓; no pool expansion ✓ (same memory, richer keys); local Qwen-7B only ✓.

**G0-cond probe (cheap; oracle kill-switch built in).**
1. One frozen Qwen-7B forward over all videos, dumping per-frame hidden states (≈1-2 GPU-h; P9 confirmed
   8-frame Qwen-7B runs on one A100; infra exists). CLIP per-frame set as a paired control.
2. **Zero-training first pass:** on the *frozen* projected features, compute set-matching kNN (MaxSim,
   then TopK/OT) vs pooled-cosine kNN. Report retrieval **AUC** and **oracle-threshold** + **honest
   (dev-calibrated)** acc/F1, per dataset, both protocols, **paired** (set vs pooled, same features/seeds).
3. **Kill switch (mandated oracle arm):** if paired oracle Δ(set − pooled) AUC→acc projection < +0.03 on
   *every* dataset, pooling was not discarding convertible alignment structure ⇒ **dead, zero further GPU.**
4. Only if the oracle arm clears on ≥1 dataset: train the head with the set-matching distance and run the
   formal 3-seed both-protocol ceremony on that dataset.
Calibration guard (per the C3-probe erratum): include a label-oracle arm that must reach ~100% Fano
headroom, else the probe machine is void.

**GPU cost.** ~1-2 GPU-h extraction + CPU/near-zero set-kNN for the whole oracle screen; full head-retrain
only if the screen clears. Hours, not days.

**Prior: FAIR.** *Falsifiable:* if per-frame set-matching kNN AUC does not beat pooled-vector kNN AUC by
a paired margin projecting to +3 acc on at least one dataset's oracle arm, C1 is dead — pooling was
lossless for retrieval here.

---

## C2 — Asymmetric / multi-view MLLM memory bank (the retrieval MEMORY as integration point)

**Mechanism (1 para).** Keep the query encoded as usual, but store each **memory** (own-train-split)
exemplar as a **multi-view set** of Qwen-derived keys — its per-frame/per-segment token vectors (or a
small learned set of "view" prototypes per exemplar) — so one training video contributes several keys to
the bank. Retrieval is **asymmetric**: a learned query-side attention-pool (trained by the triplet+BCE
head) selects which memory *views* to match, i.e. the query learns to align to the memory's discriminative
frames. The kNN vote aggregates over matched views. This is C1's cousin but the injection point is the
**memory structure + asymmetric query aggregation**, not the symmetric distance metric.

**Injection point + bandwidth class.** The memory bank representation + an asymmetric (learned query /
frozen multi-view memory) retrieval — representation-level. Bandwidth: multi-view (high). No decision-side
scalar.

**Non-isomorphism.** No dead route touched the memory at representation level (archive-repair only
*deleted* pooled entries; P2 *reranked* pooled neighbors with an MLLM). Asymmetric dual encoders and
multi-vector memory (ColBERT/MUVERA lineage) are untried in hateful-video. Distinct from C1: C1 is a
symmetric set distance; C2 makes the **memory** multi-view and lets the **query learn** the aggregation
(training-objective coupling on the query side only). Non-isomorphic to encoder-swap (memory structure,
not encoder identity).

**Novelty within hateful-video.** Multi-vector / asymmetric memory is standard in text/vision retrieval
(ColBERT, MUVERA) but **absent in hateful-video**, where the memory is always a pooled bank (RGCL/RA-HMD/
MoRE). Novel as "MLLM-structured multi-view memory for hateful-video kNN."

**Why it could clear +3 (D-laws).** **D2:** representation/memory-geometry level. **D1:** no signal add.
The extra lever over C1 is the **learned asymmetric query aggregation** — if the discriminative frames
differ query-side vs memory-side, symmetric matching (C1) underperforms and C2's learned selection wins.
**D3:** the learned aggregation adds parameters → higher overfit risk on 549-744 samples than C1's
parameter-free distance; keep the query pool tiny (single attention head) and monitor under no-selection.

**Veto compliance.** Same as C1 (all ✓). Multi-view keys are from own-train-split videos only.

**G0-cond probe.** Reuse C1's per-frame extraction. Probe cheaply as: does a **fixed** (untrained)
multi-view memory + query-max aggregation already beat pooled kNN (isolates the memory-structure gain
from the learned-aggregation gain)? Then add the tiny learned query pool and measure the increment,
oracle + honest, paired, both protocols. Kill if fixed-multi-view ≤ pooled AND learned pool adds < noise.

**GPU cost.** Shares C1's extraction (~0 marginal); learned query pool = a small head retrain (~1 GPU-h).

**Prior: MODEST.** *Falsifiable:* if a learned asymmetric query aggregation over a multi-view Qwen memory
does not beat both pooled kNN and C1's symmetric set distance by a paired +3 on any dataset, the memory-
structure lever adds nothing beyond the symmetric metric.

---

## C3 — MLLM-embedding-geometry hard-negative mining for the contrastive head (training-objective level)

**Mechanism (1 para).** Leave inference pooled/standard; change **how the triplet+BCE head is trained**.
Mine hard negatives using **Qwen's embedding geometry** (not scores, not generated text): for each
anchor, the hardest negatives are the *opposite-label* training videos that Qwen places **nearest** in
its representation space (semantically confusable benign↔hateful pairs). Training on Qwen-mined hard
negatives sharpens the CLIP-head (or Qwen-head) decision boundary where it matters. Optionally add a
relational term: preserve Qwen's pairwise *ordering* of neighbors (a light relational distillation of
geometry, not logits).

**Injection point + bandwidth class.** Training-objective (negative sampling / relational term) —
representation-shaping. Signal = Qwen **pairwise distances** (embedding structure), which the banned-list
explicitly permits (it bans MLLM *scores/labels as training signal*, not embedding geometry).

**Non-isomorphism.** **vs P5 (counterfactual twins, dead):** P5 *synthesized* contrastive pairs; C3
*selects real* training pairs by Qwen distance — no synthesis. **vs MLLM-scores-as-training-signal (ban):**
no score/label used, only geometry (the audit §3(a) explicitly keeps representation-geometry distillation
alive as an open cell). **vs LLM-generated hard negatives (SyNeg etc.):** those *generate text* negatives;
C3 mines *existing* videos by embedding distance. Non-isomorphic to all.

**Novelty within hateful-video.** Hard-negative mining is standard in metric learning; **MLLM-geometry-
guided** hard-negative mining for a hateful-video retrieval-contrastive head is novel in-domain. Honest:
the weakest-novelty of the three (a training-recipe change).

**Why it could clear +3 (D-laws).** **D2:** shapes the learned representation geometry (not decoration).
**D1:** training-time, no inference-side signal add. **D3 is the real risk:** on 549-744 samples, hard
mining can **overfit** rather than help, and the effect may be <3pt (below floor). This is why it ranks
below C1/C2.

**Veto compliance.** All ✓ (embedding geometry, own-train-split, no scores).

**G0-cond probe.** Zero-GPU-ish: compute the Qwen-hard-negative set on cached features; retrain the tiny
head with vs without Qwen-mined negatives; paired Δ, oracle + honest, both protocols. Kill if paired
Δ < +0.03 on all datasets or if no-selection protocol shows overfit (train↑ test↓).

**GPU cost.** Head retrains only (~1-2 GPU-h total). Hours.

**Prior: MODEST-LOW.** *Falsifiable:* if training the head on Qwen-geometry hard negatives does not beat
random/uniform negatives by a paired +3 on any dataset under the no-selection protocol, mined hardness is
redundant with what triplet+BCE already learns.

---

## C4 — Retrieval-augmented Qwen inference / memory-conditioned MLLM classification (flip the direction)

**Mechanism.** Retrieve top-k labeled neighbors from the own-train memory and feed them into Qwen's
context as in-context demonstrations; Qwen predicts hateful/benign conditioned on the neighbors (RAG-ICL,
memory→MLLM instead of MLLM→memory).

**Honest disposition — I kill this myself (LOW).** Injection point = Qwen's **final decision** (one
logit/label), bandwidth = bits → **D1 bites directly**. It is **near-isomorphic to P2** (memory→MLLM
comparability judgment, dead) and **P10 A-fuse** (MLLM logit fused with vote, localization-only). The
in-domain evidence is against it: LMM-as-reasoner classifiers (TANDEM 0.78 on HateMM) lag the supervised-
fusion frontier, and the reflection's RecSys analog (LLM-as-ranker redundant given strong collaborative
features) predicts the same. Included only because the mandate asked me to check the flip explicitly:
checked, closed. **Do not spend GPU** unless C1-C3 all die and the user reopens decision-side routes.

*Non-kill caveat:* the one un-isomorphic sub-variant is conditioning Qwen on retrieved neighbors to
produce a *representation* (not a label) for the head — but that collapses into C1/C2 (representation-
level) and should be pursued there, not as MLLM-as-classifier.

---

## C5 — 7B relational contrastive distillation (CRD) into the head

**Mechanism.** Distill Qwen-7B's *relational* embedding geometry (pairwise structure) into the small head
via a CRD-style loss (representation-level, no logits/scores). Open per the exhaustion audit §3(a) (C4 in
the prior literature doc was deferred only for lack of a 72B teacher; a 7B-teacher relational variant is
uncovered by any epitaph or ban).

**Honest disposition — LOW.** The audit's objection stands: a 7B-teacher student cannot exceed *using the
7B encoder directly*, which already fails to convert on EN/ZH (B1). CRD's value is normally a big→small
teacher gap; here teacher=student-scale. Keep as a last-resort representation-level cell if C1-C3 die.
Cost: distillation training (~1-2 GPU-h). *Falsifiable:* if a CRD-distilled head does not beat the direct
frozen-Qwen head on any dataset, same-scale relational distillation adds nothing.

---

## RANKING (novelty × prior × goal-relevance / cost)

| # | candidate | injection point | novelty (in-domain) | prior | cost | goal-relevance |
|---|---|---|---|---|---|---|
| **1** | **C1 set-to-set retrieval over Qwen frame tokens** | retrieval metric + representation | MODERATE (first in hateful-video; transfer from few-shot-video) | **FAIR** | ~1-2 GPU-h | HIGH (D2 class; attacks MHC-EN dilution + HateMM locality) |
| 2 | C2 asymmetric multi-view MLLM memory | memory structure + query aggregation | MODERATE-HIGH (memory untouched in-domain) | MODEST | shares C1 + ~1 GPU-h | HIGH |
| 3 | C3 Qwen-geometry hard-negative mining | training objective | MODERATE (recipe change) | MODEST-LOW | ~1-2 GPU-h | MODERATE |
| 4 | C4 memory-conditioned Qwen classifier | decision side | LOW | LOW (self-killed: D1 + P2/P10 iso) | — | LOW |
| 5 | C5 7B relational CRD into head | training objective | LOW | LOW | ~1-2 GPU-h | MODERATE |

Adjacent non-novelty cell for the record: the exhaustion audit's **ROC→acc threshold conversion of
frozen-Qwen ZH** (`EXHAUSTION_AUDIT_2026-07-14.md §1`) targets the same binding gap but is a *calibration*
lever, not novel MLLM integration — it belongs to the audit, not this doc, and is a cleaner *performance*
bet than C3-C5 if the user's novelty bar can be met elsewhere.

---

## RECOMMENDED FIRST MOVE

**Run C1's G0-cond probe.** It is the cheapest (one frozen forward + CPU set-kNN oracle screen, ~1-2
GPU-h), has the highest prior (representation-geometry = D2's only winning class; +3-8pt precedent in
few-shot video; P6 supplies the in-project locality premise), sits on the binding gap (HateMM/MHC-EN),
and — decisively for efficiency — the **per-frame extraction it requires is also the exact prerequisite
for C2**, so one extraction de-risks the top two candidates at once. The oracle arm is a hard, mandated
kill switch: if pooling was lossless for retrieval, C1 (and by extension the whole "don't-pool" family,
C1-C2) dies in one cheap measurement with zero head training. That is the correct next experiment: it
either opens the first genuinely new representation-geometry lever since the encoder swap, or it closes
the last untouched component of the pipeline with a single decisive probe.

---

## PROVENANCE — verified citations (each fetched or confirmed via primary source this session)

- **DeepEMD** — Zhang, Cai, Lin, Shen. "DeepEMD: Few-Shot Image Classification With Differentiable Earth
  Mover's Distance and Structured Classifiers." **CVPR 2020**, arXiv **2003.06777**. EMD optimal-matching
  over dense local regions; +significant on 5 few-shot benchmarks. [verified: CVPR OpenAccess + arXiv]
- **OTAM** — Cao, Ji, Cao, Chang, Niebles. "Few-Shot Video Classification via Temporal Alignment."
  **CVPR 2020**, arXiv **1906.11415**. Per-frame temporal-alignment distance (no pooling); +significant on
  Kinetics/SSv2. [verified: arXiv abstract fetched]
- **ColBERT** — Khattab, Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late
  Interaction over BERT." **SIGIR 2020**. Token-level MaxSim late interaction. [verified: multiple sources]
- **ColPali** — Faysse et al. "ColPali: Efficient Document Retrieval with Vision Language Models."
  **ICLR 2025**, arXiv **2407.01449**. Late interaction (MaxSim) over VLM patch embeddings; beats pooled
  pipelines on ViDoRe. [verified: ICLR poster page + arXiv]
- **Video-ColBERT** — Reddy, Martin, Yang, Yates, Sanders, Murray, Kriz, de Melo, Van Durme, Chellappa.
  "Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval." **CVPR 2025**, arXiv
  **2503.19009**. Frame-level token-wise late interaction (MeanMaxSim), text-to-video retrieval; NOT
  applied to hateful content or video-to-video classification. [verified: arXiv abstract fetched]
- **CMOT / TSAM** — optimal-transport (Sinkhorn) set-matching for few-shot action recognition; TSAM =
  arXiv 2408.12475 (2024). [verified: search snippet; cite only as "OT set-matching precedent"]
- **MUVERA** — "Multi-Vector Retrieval via Fixed Dimensional Encodings." arXiv **2405.19504**. Makes
  multi-vector late-interaction retrieval cheap (single-vector MIPS). [verified: search snippet — efficiency
  infra for C1/C2 if needed]
- **ColMate (TopKSim)** — arXiv **2511.00903**. Top-K sim more robust than MaxSim for patch noise (K=5).
  [verified: search snippet — a robustness variant for C1's metric]
- **MoRE** — "Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video
  Hate Detection." **The Web Conference (WWW) 2025**, dl.acm 10.1145/3696410.3714560. Pooled joint-video
  retriever → mixture-of-experts. In-domain closest prior; pooled retrieval. [verified: ACM/OpenReview]
- **RGCL** — Mei et al. "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning."
  **ACL 2024**, arXiv **2311.08110**. The pipeline's own method; pooled-vector retrieval. [verified]
- **RA-HMD** — Mei et al. "Robust Adaptation of Large Multimodal Models for Retrieval-Augmented Hateful
  Meme Detection." **EMNLP 2025 Oral**, arXiv **2502.13061**. Pooled retrieval. [verified]
- **SyNeg / LLM hard-negative mining** — LLM-guided hard-negative synthesis exists (SyNeg; GPT-4 clinical
  hard negatives), but *generates* negatives; C3 *mines existing* videos by embedding distance —
  non-isomorphic. [verified: search snippet]

**Provenance of internal facts:** 22-route dead list + bans + D1/D2/D3 —
`autoresearch/goal_mllm_plus3/state/directions_tried.json`, `REFLECTION_mllm_integration_failures.md`.
Pooled-only cache (needs one new forward for per-frame states) + P9 confirms 8-frame Qwen-7B on one A100
— `LITERATURE_mllm_integration_2026-07-13.md:100`. P6 localizer positive; encoder-swap HateMM-only
positive — `directions_tried.json` positives_bank. Open-cell / exhaustion context —
`EXHAUSTION_AUDIT_2026-07-14.md`.
