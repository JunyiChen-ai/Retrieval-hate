# Round-6 brainstorm bundle — hateful video detection, method direction

You are the idea jury for round 6 of a research project that has generated **65 candidates across
five rounds and killed every one**, plus nine more kills in the last two weeks. Your job is to
generate and then adversarially score new mechanism candidates. Read the constraint map first. A
candidate that is isomorphic to anything in the constraint map is worth zero and must be said so.

---

## 1. The task and the numbers

Binary hateful/harmful vs normal classification of short videos. Four fixed datasets, no additions.

| dataset | train | val | test | test pos | lang |
|---|---|---|---|---|---|
| HateMM | 744 | 107 | 215 | 86 | EN |
| MultiHateClip-EN (MHC) | 549 | 80 | 161 | 49 | EN |
| MultiHateClip-ZH (MHC_zh) | 579 | 78 | 149 | 45 | ZH |
| ImpliHateVid | 1283 | 325 | 401 | 200 | EN |

**Contrast line to beat** (best single-encoder bare head, test macro-F1):
HateMM **0.8774** · MHC-EN **0.7331** · MHC-ZH **0.7821** · ImpliHateVid **0.9118**.

A three-encoder ensemble of pairwise-trained heads (0.8732 / 0.7776 / 0.8183 / 0.9276) exists and is
classified as a **trick, not a method** — it may be reported but cannot be the contribution.
Ensembling and the pairwise objective are both already banked and neither is available as novelty.

One flipped test item is worth ~0.5-0.6 macro-F1 points on MHC-EN/ZH. Seed std is 0.004.

**Published frontier** (independent sweep, 2026-08-17): HateMM — MM-HSD 0.874 (statistical tie with
our line, same architectural class: frozen encoders + cross-modal attention + PaddleOCR),
SAGE 0.8710, RAMF 0.851, MoRE 0.8235, MARS 0.758, LELA 0.7043. MHC-ZH — HVGuard 0.822 is **above**
our 0.7821 and is the one published cell that clearly beats us. ImpliHateVid — TCL 0.8773.
No new hateful-video method paper has appeared in ~2 months; the niche has 14 arXiv papers ever.

## 2. The substrate

Frozen encoders → cached per-video embeddings → ~5M-param MLP head (element-wise "align" fusion) →
BCE and/or retrieval-guided contrastive/pairwise loss → optional FAISS kNN at inference. Only the
head trains.

Cache format: `{ids, img_feats [N,D], text_feats [N,D], labels [N]}`, D=1024 (CLIP ViT-L/14-336) or
3584 (Qwen2.5-VL-7B-Instruct and its LoRA variants). Also on disk: Qwen2.5-VL-32B caches
(HateMM, MHC), Molmo2-8B (HateMM), ~99 pooling/readout variants.

**Measured head training cost: 11 seconds per run** (a 12-run grid takes 127 s wall). Head-level
pilots are effectively free. Reject a candidate for a weak premise, never for pilot cost.

Hardware: one idle RTX 5090 (32 GB). QLoRA fits at ~15 GiB. Raw video exists locally for HateMM /
MHC / MHC_zh but **ImpliHateVid raw video is gone** — every encoder-level or LoRA candidate is at
most 3 of 4 datasets. API budget for the whole round: **¥60 total, ≤¥10 per pilot**, DashScope.

## 3. Where the headroom is (measured, not assumed)

Purchasable headroom to a panel-resample ceiling: MHC-EN +15.0, MHC-ZH +12.0, HateMM +12.7,
ImpliHateVid +7.2 macro-F1. **Annotation noise is not the binding constraint.**

All 108 test errors of the round-4 comparator, coded item by item before counting:

| bucket | n | share | oracle-fix value | status |
|---|---|---|---|---|
| **S** stance / use-vs-mention (the video shows hateful material the uploader is quoting, reporting or condemning) | 49 | 45.4 % | **+6.46 mean macro-F1** | open only via paid human supervision |
| **O** decisive evidence burned into on-screen text | 5 | 4.6 % | — | HateMM-only |
| **M** transcript empty or music-only | 5 | 4.6 % | — | tiny |
| **A** annotators split / label conflicts with the material | 9 | 8.3 % | — | not purchasable |
| **D** duplicate or degenerate item | 3 | 2.8 % | — | not purchasable |
| **X** ordinary ranking error | 37 | 34.3 % | — | **SEALED** |

X is sealed: 5 hypotheses tested, it is a diffuse deterministic residue (28/37 wrong under all 3
seeds), its nearest-neighbour purity is no worse than that of non-X errors, and repairing it needs
per-item member selection, which is banned. Purchasable mass after removing X:
+0.0635 / +0.1199 / +0.1031 / +0.0300.

---

## 4. CONSTRAINT MAP — everything already dead

### 4.1 Three framing killers
1. **SAGE** (ACL 2026, HateMM 0.8710/0.8628) — the HateMM accuracy race is closed; a pure accuracy
   claim is not publishable alone.
2. **HCG-MPB** (ICMR 2026) — argues in its motivation that instance-based retrieval is a flawed
   design. Every RGCL-family hateful-video paper must now rebut it.
3. **`2607.23304` + `2602.05152`** — under squared loss, a linear head and fixed features, explicit
   parameter adaptation and implicit routing are the same kernel ridge regression, and query
   expansion ≡ key expansion. So "our retrieval module is a form of test-time adaptation" and "we
   improved the query/key construction" are both formally absorbed and cannot be claims.

### 4.2 Retrieval structure — CLOSED
Segment-level retrieval keys; multi-segment complementarity; single-segment selection;
visual-purity segment selection; type-hard-partitioned memory; streaming/continual memory;
cross-lingual EN-rescues-ZH (EN→ZH transfer measured −0.138 macro-F1); CVoI acquisition;
OCR−ASR residual keys; near-duplicate/label-conflict memory (5 conflicting pairs vs bar 10, and
vs permutation-expected 24.1 — near duplicates are label *concordant*); late-interaction segment
retrieval (−0.043). Human-Agreement Retrieval permanently closed on all three legs (vote-feature
arm −0.0174 EN / −0.0105 ZH against a *trained* baseline; contrastive leg −0.00506 with a
shuffled-vote placebo capturing 87 % of the gain). For the residual error set, kNN repair is
foreclosed: top-20 gold purity 0.255-0.517 for errors, and non-X errors sit closer to train than X
errors in 9/9 encoder cells.

### 4.3 Temporal / segment / pooling — AXIS CLOSED at four levels
Order kernels (soft-DTW Δacc +0.0059 = exactly the shuffle-null 95th percentile); retrieval over
frame groups (+0.0035 vs bar +0.05); causal-prefix conditional information **exactly 0.0000**;
segment granularity +0.0012/+0.0032 with 91-98 % of the oracle inside banned per-item selection;
frame count 8→16 = −0.0077. And the dilution premise is FALSE: HateMM hate-span coverage **median
0.8289**, 74 % a single contiguous span, only 22 % below 0.5 coverage. Sub-video units are not a
different object (unit↔own-video pooled cosine 0.95).

### 4.4 Audio / prosody — 0 for 4, FAMILY CLOSED
eGeMAPS, LAUD, CLAP, and prosody-as-operator (Δ −0.0436 / −0.0392 vs bar +0.010, 0/3 seeds both
arms). The failure mode is **redundancy, not weakness**: label-permuted prosody adds *more* to a
text head (+0.0294/+0.0448) than real prosody (+0.0031/+0.0122).

### 4.5 OCR / on-screen text — one dataset at best
Provenance typing −0.0020 (0/3 seeds, label-permuted null = 90 % of the effect); mean fusion +0.0094
vs bar +0.015; the same vector through the learned fusion MLP **−0.0246** with a sign flip;
proposition-mass firewall rho −0.0345 vs bar 0.24, negative 5/5 seeds. Measured: OCR is
complementary on HateMM and **redundant on MultiHateClip** — 95 % EN / 99 % ZH of MHC videos carry
≥20 chars of screen text and **not one MHC error is decidable from screen text the transcript
lacks**. Occupied externally by MM-HSD (0.874 on HateMM with OCR).

### 4.6 Annotation disagreement / votes — CLOSED
MHC has 2.2 annotators. The flagship multi-annotator architecture ranks last on a 6-annotator
corpus. `Counter Narrative` votes: 139 videos, never a majority or even a tie, and only **1 of 15**
S-bucket false positives carries one (bar 25 %); error rate on CN-voted videos is *lower*.

### 4.7 MLLM in front of the head — all five access points measured negative (2026-08-13/14)
| access point | measured |
|---|---|
| description as a new 768-d third input stream | **−0.0371**, 0/3 seeds |
| description merged into transcript before the encoder | **−0.0105**, 0/3 seeds |
| MLLM as pre-classifier stance labeller | see 4.8 |
| MLLM as uncertainty-gated decision arbiter | **−0.0135**, 0/3 seeds; the judge is less accurate than the head in **21 of 24** in-band cells and better in **0** |
| MLLM as counterfactual training-data augmenter | **−0.0507**, 0/3 seeds, **0.0444 worse than a random-negative control** |

Scope note: the code review found the counterfactual construction defective (25 of 284 rewrites
still matched the slur list, 24 were byte-identical), so that row kills *this construction*, not the
idea of counterfactual augmentation for video.

### 4.8 Zero-supervision stance extraction — six routes, all negative
Direct 5-way prompt 0.257 → masked 0.371 → symmetric two-alternative **0.469 (chance 0.50,
p = 0.86)**. Perception questionnaire gate-0: content described accurately 15/18, but a
direction-bearing fact present in only **2/18 = 0.111** (bar 0.30). Synthetic attribution-pair
supervision: 0.98-1.00 synthetic dev, **AUC 0.441/0.467 on real ASR**, sign inverted in 6/6 cells —
and the mechanism is decisive: **only 10 of 99 real transcripts contain any attribution marker at
all**, so the cue is not present in speech. Likelihood read-out: all four arms emit a **constant
OPPOSE**; base vs instruct margin correlation **r = 0.980**; the template moves the margin 3.8× more
than the video. The alignment/RLHF explanation is refuted.

Standing counter-fact: matched policy violation/exemption clauses embed at cosine **0.920** (CLIP),
and clause directions **lose to dimension-matched random directions by −0.046 mean ROC on 3 of 4
datasets**. Stance is not linearly present in the final frozen features.

### 4.9 Training-level families
F1 rationale-then-verdict SFT — occupied (IARE, LEAF, ExPO-HM), and naive explain-then-detect
*loses* (Direct-SFT 75.0 > CoT-SFT 74.5 > GRPO 74.5 on Qwen2.5-VL-7B).
F2 generative-MLLM-as-classifier — adjacent, nulled in-house, and small-n is a loss.
F4 votes-as-target — dropped. F5a/b OCR integration — occupied.
**F3 stance-as-supervision is the only OPEN family and its supervision does not exist.**
F5c text-bearing frame selection — open, but relevance sampling still misses >90 % of harmful
content (`2508.10974`).

### 4.10 Standing negatives a candidate must not re-derive
- **Cross-encoder complementarity is additive.** A monotone non-additive lattice over per-encoder
  OOF logits gives ΔROC mean −0.0000, bootstrap LCB95 −0.00253, with a validated positive control.
- **Decision-rule / calibration is capped** at +1.2 to +4.6 points by a *test-label oracle
  threshold*.
- **No measurable train/test covariate shift on any of the four datasets** (domain-classifier AUC
  0.42-0.56, MMD p 0.17-0.96). Every distribution-alignment / shift-correction premise is dead.
- **Random projections are a strong baseline** (val ROC up to 0.88). Any "K interpretable
  directions" claim must beat random directions averaged over draws.
- **A large oracle ceiling is not evidence for a candidate** — it is the precondition every failed
  candidate already met. AGGNET carried the largest oracle ever measured here (+0.149/+0.152/+0.219)
  and delivered +0.013/−0.007/+0.000. Gate on demonstrated *conversion*, in net items.
- **Law III / F47: per-item selection is banned** — no operator may choose, per test item, which
  member / segment / encoder to believe.
- **Within-hard-label permutation nulls are invalid**: they manufacture the conditionally
  independent ideal case and would force KILL for every possible input.

### 4.11 Externally occupied slots found in this round's literature sweep
| framing | occupant |
|---|---|
| representation-level MLLM→small-head distillation for hateful content | **Just KIDDIN'**, `2411.12174`, Findings ACL 2025 — LLaVA-NeXT teacher, **Hate-CLIPper student on frozen CLIP**, plain L2 in representation space, +10.6 % F1 / **+0.5 % AUC** (i.e. almost all of it is threshold movement) |
| gated/projected fusion head on frozen CLIP for hate | **GatedCLIP**, `2602.20818` (weak occupant: AUROC 0.66) |
| MoE head for hateful video | MoRE, WWW 2025 |
| frozen encoders + cross-modal attention + OCR on HateMM | MM-HSD, `2508.20546` |
| shared trunk + per-dataset heads as an architecture | UniDet, `2102.13086` CVPR 2022 |
| pool datasets + retrieve neighbours + train on union, cross-lingual hate | `2505.14272` (text only) |
| soft labels from annotator distributions in hate | Fornaciari et al., NAACL 2021 |
| loss correction for noisy hate labels | `2311.00619` |
| threshold tuning as a contribution | routine in abusive-language shared tasks |
| GMM-EM transduction on frozen VLM embeddings with a class-prior term | TransCLIP `2406.01837`, UNEM `2412.16739`, StatA `2501.03729` |
| gradient surgery for multi-task gains | `2209.11379`, `2604.08939` — both find no gain over tuned scalarization |
| post-hoc calibration for accuracy/F1 | `2601.19944` — max expected accuracy gain **+0.008 %** |
| prototype/centroid memory head | textbook ProtoNet; this project already ranked it LOW |

### 4.12 In-house findings that pre-empt the families the 2026-08-17 literature sweep called "open"

The F-registry (`autoresearch/goal_mllm_plus3/state/findings.jsonl`, F1-F123) contains direct hits on
three of the four families §5 below calls open. **Read this before scoring anything in §5.**

- **F70 — the readout axis was already killed at $0, and it included an intermediate hidden layer.**
  "hidden layer **L24** / one-word prompts / last-token span all inside the permutation null; one-word
  regresses HateMM." So "use an intermediate layer instead of the final one" has been tried once, at
  a single layer, and landed inside the null. Multi-layer *fusion* is not literally the same
  experiment, but the premise is damaged and a candidate must say why.
- **F92 — no further pooling-span variants under bidirectional attention without weight adaptation.**
  Stream collapse is monotone in the pooled span; under bidirectional attention every text token
  attends all ~720 vision tokens, so excluding vision *positions* does not exclude vision
  *information*.
- **F63 — multi-hop label propagation / graph diffusion over the frozen kNN graph is KILLED on all
  three datasets and is monotone-negative in the diffusion coefficient** (HateMM −0.0187, ZH
  −0.0385; α=0.9 catastrophic at −0.19/−0.22). This is the classic transductive operator on this
  exact feature space. A GMM-EM transductive assignment is not identical to label propagation, but a
  candidate must state the difference in mechanism, not in name.
- **F75 — head-loss swaps are 0/8 FORMAL and 7/8 arm-dead** (NCA at τ=0.1/0.2, SupCon, mixup-BCE).
  Recorded as "the first measured negative for *trained reshaping unlocks the oracle headroom*" —
  Law I holds against a trained operator. τ/α retunes are named as a banned tactic.
- **F79 — noise-robust head training was already priced** by a measured proxy: boundary-dominated
  error is a **13-17 % upper bound** and the fixable part is single-digit. Standing rule: **a new
  noise-robust proposal must beat that arithmetic before it is worth a run.**
- **F73 — SAM (ρ=0.05) + modality dropout: all four cells killed** in <0.15 GPU-h; the flat-minima
  premise is refuted at this head scale. The family is marked "one bite consumed, do not re-tune
  knobs".
- **F20 — the model-scale axis is CLOSED and the 32B encoder is *worse* than the 7B** (regresses on
  HateMM, below CLIP on EN/ZH, fails 32B-vs-7B on all three). Any "distil from the bigger teacher"
  candidate has to explain how a teacher that loses to the student supplies anything.
- **F91 / Molmo2-8B — the best image stream ever measured on HateMM (+0.0558 on the stream) still
  *lowers* the deployed number** (−0.0217/−0.0249). "The binding constraint is NOT visual
  representation quality. If the encoder axis reopens it must be on the **text** side."
- **F34 — per-encoder threshold calibration is closed as a conversion lever** (label-oracle best cut
  is worth only +0.0022 acc / +0.0213 macro-F1).
- **F98 / F105 / F112 — the whole conditional-aggregation and top-20 re-ordering family is closed**,
  including learned re-weighting, soft mixtures, attention or gating over the neighbour list, and
  prototype / centroid / class-conditional subspace residual comparison at any rank.
- **F95 — a trained pair verifier really is 2.5-3.5× more relational than the deployed cosine
  (CONTROL-1 passes 18/18 by 4.3-8.8×) and it converts to nothing end-to-end (CONTROL-2 fails
  0/36).** This is the project's sharpest single datum: better relational signal, zero conversion.
- **Standing campaign constraint, not yet lifted: "training data = single-dataset train split only"
  and "no cross-dataset mixing".** Any joint-multi-dataset candidate is blocked on a user ruling
  before it can be run, and must be flagged as such rather than quietly assumed.
- **Structural Law I, now arithmetic:** across nine certified instances a signal is demonstrably
  richer than what the pipeline has and no legal operator converts any of it. The F66 decomposition
  makes it exact — HateMM oracle headroom +0.0776 = +0.0012 legal-symmetric + **+0.0764
  banned-selection**; MHC-EN +0.0700 = +0.0064 + **+0.0636**. **91-98 % of convertible headroom is
  formally disjoint from every legal operator.** A candidate whose story is "we give the model a
  better signal" is, by default, the tenth instance.
- **The operator pincer:** any operator is either a fixed symmetric aggregation (banned family A) or
  a per-item choice, which is selection (banned by Law III / F47). **There is no third kind.** A
  candidate must say which of the two it is, or exhibit the third kind explicitly.
- **Screening currency:** `net = changed × (2·precision − 1)`. The binding screen is NET ITEMS
  against 22.3 / 17.4 / 16.5 (HateMM / MHC-ZH / MHC-EN) for +0.030, in the **train arena** at
  n = 744/579/549. The "exchange rate is bounded near 1.2" heuristic is explicitly refuted.
- **F118 K-REACH — the one large, real, unclaimed pool:** flipping the OOF-stable high-confidence
  inversions is worth **+0.0981 acc / +0.1021 macro-F1** (HateMM) and **+0.0864 / +0.1010** (MHC-ZH),
  against a +0.050 bar. Verbatim: *"the pool is roughly twice big enough to pay the price. What is
  absent is any label-free way to find it."*

---

## 5. What the literature says is still open, with honest gain estimates

1. **Noise-robust training objective on a frozen-feature head.** Open in hateful video —
   `2508.04900` (ACM MMWS 2025) documents that **58.64 % of HateMM "hate" videos contain
   non-hateful material** and quantifies the resulting decision-boundary shift, but proposes no
   method. `2605.22591` benchmarks 8 noise-robust methods across 150 conditions **specifically in
   the frozen-encoder + light-head regime** and finds: ELR wins most often (49/150); the small-loss
   assumption **breaks** under frozen features (clean/noisy loss distributions overlap 53-61 %);
   Co-Teaching collapses to 35.1 % balanced accuracy with **zero recall on three minority classes**.
   ⇒ ELR-style regularization or a robust loss; never sample-selection methods.
   Anchors: GCE `1805.07836`, SCE `1908.06112`, APL `2006.13554`, ELR `2007.00151`, JAL
   `2507.17692` (ICCV 2025). **Estimate +1 to +3.**
2. **Intermediate / multi-layer frozen features** instead of the last-layer pooled vector.
   Supports: `2605.10494` (ICASSP 2026), `2601.09322`, HiProbe-VAD `2507.17394` (ACM MM 2025, picks
   the most informative intermediate layer of a frozen MLLM for video anomaly detection).
   Unoccupied in hate. Costs one encoder re-extraction pass. **Estimate +0 to +2.**
   Note the in-house hook: the likelihood probe explicitly left open "a trained probe on
   intermediate hidden states", having measured only the output distribution.
3. **Transductive inference over the unlabelled test pool** (test *inputs* are unsealed by user
   ruling; test labels stay sealed). Unoccupied in hateful content, training-free, nearly free to
   pilot. Headwinds: TransCLIP's own limitations section shows gains decaying with shot count
   (16-shot cells at +0.8, +0.9, **−1.0**); `2204.11181` shows transductive methods fall **below**
   inductive once the query class distribution is imbalanced (ours is 25-40 % positive); **no paper
   in this line reports macro-F1**; and framing killer 3 absorbs the retrieval≡TTA claim.
   **Estimate +0 to +1.**
4. **Joint multi-dataset training.** No hateful-video paper trains jointly across datasets — every
   one trains a separate model per dataset. Label spaces differ (HateMM binary; MHC
   hateful/offensive/normal; ImpliHateVid implicit/explicit/non). UniDet's own claim is that a
   unified detector merely *matches* dataset-specific models in-domain, so **≈0 on HateMM, +1 to +3
   plausible on the small/low-resource sets**. Architecture novelty is occupied by UniDet; only a
   cross-lingual/low-resource-transfer framing survives.
5. Attentive probing heads. Blocked as-is: attention needs token sequences and the project caches
   pooled per-video vectors. On CLIP/SigLIP backbones the reported gain over linear probing is only
   +1.1 to +2.0 pp anyway.

Still-open items the project's own documents record: F3 stance-as-supervision (blocked on ~750
human judgements); F5c text-bearing frame selection; B-SRTD named-intervention response-tensor
distillation (never killed by a mechanism failure, an occupant or a null — blocked on a bounded
balanced data build); EAPD accountability-path distillation (blocked on a 330-video annotation
build); the `Metadata` title/description channel, which is the largest human-annotated contributing
modality in ZH (294) and second in EN (197) and which **no model in this project consumes**,
recorded as an observation because "adding a modality is not a mechanism".

---

## 6. Your task

**Part A — generate 10-14 candidate mechanisms.** Each must be:
- a *mechanism* (something that changes how the model computes or is trained), not a slice
  analysis, not a metric, not an ablation table, not "add a modality";
- pilotable on this machine within 2 GPU-hours, preferably on cached features in minutes;
- not isomorphic to anything in §4;
- accompanied by a **falsifiable premise that can be checked cheaply BEFORE the pilot**, stated as
  a concrete measurement with a number.

For each candidate give: name, one-sentence mechanism, the premise and its cheap pre-check, the
pilot design (arms, controls, seeds, datasets), the expected gain with reasoning, the closest prior
work you know of, and the single most likely reason it dies.

**Part B — score every candidate 0-10** on: premise survivability against §4, novelty against §4.11
and the general literature, expected gain size, and pilot cost. Be hostile. This project's failure
mode is candidates that sound good and die on a premise that was already measured false. Explicitly
name, for each candidate, which §4 entry is closest to it and why it is or is not the same thing.

**Part C — rank, and state honestly how many (if any) are worth a pilot.** If the answer is zero,
say zero and say what that implies. Do not manufacture a survivor.
