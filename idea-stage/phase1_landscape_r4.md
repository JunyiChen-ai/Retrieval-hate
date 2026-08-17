# Phase 1 — Landscape update, round 4 (2026-08-10)

> **Scope:** incremental only. Four axes that changed since round 3. Does **not** re-survey
> hateful-video / retrieval / memory occupancy — see `research-wiki/NOVELTY_RECON_2026-08-09.md`
> and `idea-stage/phase1_landscape_update.md`, both still in force.
> **Cost:** literature only, zero GPU, zero test contact. 33 web queries (WebSearch + arXiv Atom API + WebFetch).
>
> **Verification convention:** **[A]** = I fetched the arXiv abs/API entry or paper page myself.
> **[B]** = DOI / venue seen in a search result but not independently fetched.
> **[C]** = read only from a search snippet — treat as `[UNVERIFIED]`.
> Every ID below was seen in a result page. **No ID in this file was reconstructed from memory.**

---

## (A) Transductive / TTA — what is still unoccupied, and the counter-literature

### A.1 The set-structure branch is **not** open. It has a canonical owner and a live sub-field.

| work | id / venue | mechanism | what it occupies |
|---|---|---|---|
| **TransCLIP — Boosting VLMs with Transduction** (Zanella, Gérin, Ben Ayed) | arXiv **2406.01837**, **NeurIPS 2024 Spotlight** (proceedings v37 pp. 62223–62256) **[A]** | GMM/regularized MLE over the **whole unlabelled test set**, KL penalty anchoring to text-encoder knowledge, block Majorize-Minimize with decoupled sample assignments; plug-and-play on any zero-/few-shot VLM | **This is the canonical "use the test set as a set" method.** It is the desk-reject citation for any "we exploit the joint structure of the unlabelled test pool" claim. |
| **StatA — Realistic Test-Time Adaptation of VLMs** (Zanella et al.) | arXiv **2501.03729**, **CVPR 2025** **[A]** | statistical-anchor regularizer preserving text-encoder knowledge; evaluated with a *variable number of effective classes per batch* and non-i.i.d. online batches | Occupies "transduction that does not assume all classes present"; **and is the strongest critique in the family** (see A.3). |
| **ZLaP — Label Propagation for Zero-shot Classification with VLMs** | arXiv **2404.04072**, **CVPR 2024** **[A]** | geodesic label propagation on a graph over text + image features of the unlabelled pool | Occupies **graph/spectral transduction** over the test pool. |
| **ECALP — Efficient and Context-Aware Label Propagation** | arXiv **2412.18303** **[A]** | dynamic graph over prompts + few-shot examples + test samples; no extra unlabelled support set | Occupies the training-free graph-transduction variant. |
| **Semantic Anchor Transport (SAT)** | arXiv **2411.17002** **[C]** | **batch-wise label assignment via optimal transport** for VLM TTA | Occupies **Sinkhorn/OT assignment over the test batch**. |
| **vMF Mixture Model with Dynamic Shrinkage for Realistic Test-Time Transduction** | arXiv **2607.15851** (2026-07-17) **[A]** | vMF mixture + class-level shrinkage for **imbalanced test batches**; explicitly targets label proportions | The 2026 frontier of "transduction without the uniform-prior assumption". |
| **Language-Aware Information Maximization for Transductive Few-Shot CLIP** | arXiv **2509.00305** **[C]** | transductive InfoMax with language regularizer | adjacent |
| **Ramen — Robust TTA of VLMs with Active Sample Selection** | arXiv **2604.21728** **[C]** | active selection over the test stream | adjacent |
| **Label-shift branch:** BBSE (arXiv **1802.03916**, ICML 2018) **[A]**; *Label Distribution Shift-Aware Prediction Refinement for TTA* (arXiv **2411.15204**) **[C]**; *Bayesian Online Label Shift with Dynamic Dirichlet Priors* (arXiv **2511.18615**) **[C]**; *Open Set Label Shift with Test-Time OOD Reference* (arXiv **2505.05868**) **[C]** | — | estimate p(y) on the unlabelled test pool and re-weight | **Occupies "estimate the test label prior and correct".** This is where the project's only surviving W4 positive result (threshold recalibration) already lives. |
| **Transductive conformal:** *Fundamental bounds on efficiency-confidence trade-off for transductive CP* (arXiv **2509.04631**) **[C]**; *Stable Localized CP via Transduction* (arXiv **2605.01452**) **[C]**; *CP benchmarking for transductive node classification* (arXiv **2409.18332**) **[C]** | — | joint prediction set over the **whole test batch**; class-wise thresholds | Occupies **conformal-over-the-test-pool**. |

**Verdict on A(i):** the four sub-branches named in the brief — prior/label-proportion estimation,
OT/Sinkhorn assignment, graph/spectral transduction, conformal over the pool — **are all occupied,
each by at least one top-venue paper.** "We use the test set as a set" is not an available claim.

### A.2 Application to moderation / hate / video — still essentially empty (one occupant)

Two targeted queries returned **only SCANNER** (arXiv **2602.00132**, **AAAI 2026** **[A]**) —
source-free TTA for hate video via centroid-guided alignment + sample-level adaptation +
intra-cluster diversity regularization, +4.69 macro-F1 averaged over six cross-dataset pairs.
**SCANNER is centroid alignment, not test-pool prior estimation, not OT assignment, not
graph transduction, and not conformal.** No transductive-as-a-set method has been applied to
hateful video or to content moderation. ⚠️ This is an inference from search silence over two
queries, weaker than a systematic sweep.

### A.3 Counter-literature (the important part)

1. **On Pitfalls of Test-Time Adaptation** — arXiv **2306.03536**, **ICML 2023** **[A]**.
   Online **batch dependency** makes hyper-parameter/model selection a min-max equilibrium problem;
   BN-Adapt / TENT / TTT fail to improve on CIFAR-10.1 and OfficeHome; **no method wins everywhere
   under fair evaluation**; episodic adaptation with oracle model selection gives "steady but limited"
   gains only.
2. **The Illusion of Progress?** — arXiv **2506.24000**, **NeurIPS 2025 D&B** (inherited, unchanged):
   gains are small relative to the earliest work and **come at the cost of trustworthiness**.
3. **StatA / Realistic TTA (CVPR 2025)** — the sharpest one for *our* setting: current transductive
   and TTA methods for VLMs **"systematically compromise the model's initial zero-shot robustness"**
   and gain only **under advantageous assumptions about the test distribution** (all classes present,
   near-uniform priors). TransCLIP itself is documented to **degrade as the number of effective
   classes drops, even at large batch size**, because it needs several same-class samples per batch **[C]**.
4. **Evaluation of TTA under computational time constraints** — arXiv **2304.04795** **[C]**:
   the more expensive the TTA method, the less data it adapts on, and the worse it does.
5. **Tempora — time-contingent utility of online TTA** — arXiv **2602.06136** **[C]**.
6. Survey: **Adapting Vision-Language Models Without Labels** — arXiv **2508.05547** **[C]**.
7. Classical, general: transductive graph SSL (GFHF/LGC/GTAM) **fails under class imbalance**;
   balancing helps *only if the prior is known, and is harmful otherwise* **[C]**.
   Also *Transductive Model Selection under Prior Probability Shift*, CEUR **Vol-4132** **[C]**.

**Net for this project:** unlocking test-input use does **not** open a mechanism slot by itself.
It opens exactly one *structurally* under-served regime — see §E.1.

---

## (B) Implicit vs explicit hate

### B.1 ImpliHateVid source paper — verified facts

**ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit
Hate Speech Detection in Videos.** Rehman, Bhatnagar, Kabde, Bansal, Kumar (IIT Indore + CBIT).
arXiv **2508.06570** (v1 2025-08-07) **[A]**; **ACL 2025 Main, Long** — `aclanthology.org/2025.acl-long.842` **[A]**.

- **2,009 videos** = 1,000 non-hate / **509 implicit** / **500 explicit**; ~86.5 h.
- **Splits: train 1,283 / val 325 / test 401** — consistent with the project's 92 EX + 108 IM + 201 NH = 401.
- **Binary (their headline): Acc 87.53 / F1 87.73 / P 87.96 / R 87.52.**
  Project's bare head reports **0.9118** on this dataset — above the source paper, but protocols
  must be checked before any same-table comparison.
- **They DO report an implicit/explicit breakdown, and it is not what one would guess.**
  3-class macro-F1 **69.18**: non-hate **84.48**, **implicit 66.05**, **explicit 57.02**.
  ⇒ **In their own numbers the explicit class is the *worst*, not the best.** The confusion is
  EX↔IM, not hate↔non-hate. Any "our accuracy is carried by the explicit subset" narrative must
  be reconciled with this published table.
- HateMM binary reported as **97.58 / 97.58** — incomparable protocol (already flagged in the
  project's number-discipline list; do not put in a table with 0.8774).
- Encoders: **ImageBind** (1024-d, image/text/audio) + NRCLex emotion + VADER sentiment +
  OFA captions → BERT; stage 1 modality-specific contrastive, stage 2 cross-encoder contrastive.

### B.2 Mechanisms published specifically for implicit hate

**Text domain — crowded, and it is the *mechanism* home of this axis:**
- **DuPL — Rethinking Implicit Hate Speech Detection: Focusing on Latent Hate Components via
  Dual-Process Argumentation**, **WWW 2026**, `10.1145/3774904.3792159` **[B]**. Mines *Latent Hate
  Components*, then multi-agent structured debate over them. IHC/SBIC/ToxiGen: +8.36 / +4.93 / +3.78 acc.
- **HatePrototypes** — arXiv **2511.06391** (v2 2026-03) **[A]**. Class-level prototype vectors from
  safety-tuned LMs; **50 examples/class**; prototypes are **interchangeable between implicit and
  explicit hate benchmarks**. ⚠️ This is the strongest rebuttal to any "implicit needs its own
  representation" claim.
- **FiADD** (focal inferential infusion + density discrimination) — arXiv **2309.11896** **[C]**.
- **Label-aware hard-negative sampling + momentum contrastive for implicit hate** — arXiv **2406.07886** **[C]**.
  ⚠️ Directly occupies "contrastive positives/negatives chosen for implicitness".
- **Aligning Implied Statements** — arXiv **2606.18852** (2026-06-17) **[A]**, triplet framework with
  context-bounded semi-hard negatives.
- Latent Hatred / ImplicitHate (ElSherief et al., EMNLP 2021) and ToxiGen remain the benchmarks **[C]**.

**Video / multimodal domain — thin, and already partially claimed:**
- **ImpliHateVid / TCL** (ACL 2025) — the dataset owner, and the only *contrastive* mechanism.
- **DeHate: A Holistic Hateful Video Dataset for Explicit and Implicit Hate Detection**,
  **ACM MM 2025**, `10.1145/3746027.3758272` **[B]**. ⚠️ **New to this project's map on this axis** —
  the title itself claims the explicit/implicit split for video. Round 3 had DeHate filed only under
  "segment-level contributing-modality annotation".
- **TIHD / QGC-Net** (ICMR 2026, `10.1145/3805622.3810673`) **[B]** — implicit hate in videos via
  cross-modal incongruity; evaluated on ImpliHateVid + HateMM. **Occupies "contradiction ⇒ implicit hate".**
- **IARE / Decoding Multimodal Cues** — arXiv **2606.11953** (2026-06-10) **[A]**, SIGIR 2026;
  Ex-HateMM + Ex-ImpliHateVid, multimodal CoT + DPO. Explicitly framed around *implicit meaning*.
- **TANDEM** — arXiv **2601.11178** v3 **[A]**; validated on ImpliHateVid, claims robustness in
  "multi-class and implicit hate settings".
- **Deciphering Implicit Hate: Evaluating Automated Detection Algorithms for Multimodal Hate** —
  arXiv **2106.05903** **[C]** (older, meme-domain).

### B.3 Is "accuracy is carried by the explicit subset" already a published finding?

- **As a qualitative claim: yes, in the meme/VLM literature** — models do well when hate is explicit
  in at least one modality and fail on sarcasm/implicature/world-knowledge cases **[C]**.
- **As a quantitative per-subset table on video: the ImpliHateVid paper already publishes one**,
  and it points the *other* way in 3-class (EX 57.02 < IM 66.05). Nobody appears to publish a
  **binary-protocol** EX/IM/NH recall decomposition on ImpliHateVid. That specific table is
  probably still unpublished — but it is an **analysis**, not a mechanism.
- **Bangla external-validation crisis paper** (arXiv **2607.11597**, 2026-07-13) **[A]** independently
  reports generalization failure concentrated on implicit expressions.

### B.4 What a reviewer would count as a *mechanism* on this axis

Not "we measured the EX/IM gap". It has to be a component whose *form* is derived from the
implicit/explicit distinction and that a reviewer cannot get by relabelling: e.g. a decision rule
whose functional form changes with an estimated implicitness coordinate; a training objective in
which EX items act as *supervision for* IM items (or are provably excluded from carrying the
gradient); or a representation with a proven EX→IM transfer property. Note **HatePrototypes already
demonstrates prototype interchangeability across the EX/IM boundary**, which pre-empts the naive
"separate representations" version, and **DuPL** owns "decompose into latent hate components".

---

## (C) "Frozen features + bare BCE head beats the elaborate pipeline"

### C.1 The observation is already published — in this domain, on HateMM

**Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content.**
Koushik, Kanojia, Treharne. arXiv **2502.07138**, **MM4SG @ WebConf 2025 Companion** **[A]**.
Abstract, verbatim: *"while **simple embedding fusion achieves state-of-the-art performance on video
content (HateMM dataset)** with a 9.9% points F1-score improvement, it struggles with complex
image-text relationships in memes."*

⇒ **"Simple beats complex on HateMM" is published, at workshop tier.** A 99-cell ablation confirming
it is a stronger version of an existing workshop result. It is **not** a methods-paper contribution,
and publishing it as-is is blocked by the project's method-paper-only constraint anyway.
Adjacent out-of-domain precedent: *Stronger Baselines for RAG with Long-Context LMs* (arXiv
**2506.03989**, **EMNLP 2025 Main**) **[C]**, where the trivial DOS-RAG matches/beats ReadAgent and RAPTOR.

### C.2 What actually beats a frozen-feature linear probe at 500–2000 examples

Everything verified below is **image classification with text-anchored class names and many classes**.
That is the load-bearing caveat.

| mechanism | id / venue | what it beats and by how much |
|---|---|---|
| **LP++** — linear classifier weights as *learnable functions of the text embedding*, class-wise image/text blending multipliers, block-MM with implicit step sizes | arXiv **2404.02285**, **CVPR 2024** **[A]** | beats CoOp-style prompt learning; **orders of magnitude faster**; black-box; **no validation-set hyper-parameter search**. The single most relevant "how to build a better head on frozen features". |
| **CLAP** — validation-free few-shot adaptation: well-initialized zero-shot linear probe + class-adaptive constraints | **CVPR 2024** (github `jusiro/CLAP`) **[B]** | removes the validation set, which is exactly the failure mode of small-n tuning. |
| **GDA / A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation** | arXiv **2402.04087** **[C]** | closed-form Gaussian discriminant analysis on frozen features beats trained adapters. |
| **Tip-Adapter** (key-value cache over few-shot features, multimodal cache) | arXiv **2111.03930** / ECCV 2022 **[C]** | **+38.5 / +29.1 pts over linear-probe CLIP at 1-/2-shot**; the advantage collapses as shots grow. |
| **Frozen Feature Augmentation** | arXiv **2403.10519**, **CVPR 2024** **[A]** | augmenting in *frozen feature space* beats linear probe at 5–25 shots, "at worst the same". |
| **LDA on frozen features** | arXiv **2604.03928** **[C]** | 1936 method; improves accuracy while cutting dimension 61–95%. |
| **Attentive probe** (cross-attention pooling head) | generic; see e.g. DINOv3 protocol **[C]** | stable improvement over linear probe **because of extra parameters** — reviewers read this as capacity, not mechanism. |
| **Calibration-objective choice for frozen-feature linear probing** | ICML 2026 benchmark listing `icml.cc/virtual/2026/68328` **[C]** | training-time calibration objectives change linear-probe behaviour on frozen foundation features. |

**Two structural observations.** (a) Every entry above **exploits a text-encoder class anchor** —
they are not generic "better heads", they are *image-language* heads. A binary policy label
("hateful") has a much weaker text anchor than "goldfinch", so the transfer is not free.
(b) Everything that reliably beats a linear probe at this scale is either **non-parametric/closed-form**
(GDA, Tip-Adapter, LDA) or **removes hyper-parameter search** (LP++, CLAP). Nothing on the list is
an elaborate learned pipeline. This is consistent with, not contradicted by, the project's 99-cell result.

---

## (D) 2026-06 → 2026-08 new work, and the SOTA table

**Sweep:** arXiv Atom API, `all:"hateful"` and `abs:"hate video" OR "hateful video" OR "harmful video"`,
sorted by submission date descending, plus two targeted WebSearches.

**Finding: no new HateMM / MultiHateClip / ImpliHateVid / HateClipSeg SOTA number appeared in
2026-06 → 2026-08.** The window is dominated by benchmarks, audits, safety/jailbreak work and
text-side hate papers. New entries worth recording:

| id / venue | date | what |
|---|---|---|
| **2608.05210** *Innocent Panels, Hateful Stories* **[A]** | 2026-08-05 | T2I multi-turn visual narratives vs detection defenses |
| **2608.05259** IMMENSE (inductive multi-perspective user classification) **[A]** | 2026-08-05 | user-level, not content-level |
| **2607.15442** *Beyond a Joke: Multi-Angle Reasoning* **[A]** | 2026-07-16 | VLM, twelve structured perspectives, soft-gated attention |
| **2607.11597** *Beyond Benchmarks: Bangla Hate Speech Detection Crisis* **[A]** | 2026-07-13 | external validation; failures concentrate on **implicit** expressions |
| **2607.21151** V-DEAL **[A]** | 2026-07-23 | video-LLM safety de-calibration; 81%+ recognition, 48.33% attack success |
| **2606.18852** *Aligning Implied Statements* **[A]** | 2026-06-17 | implicit-hate triplets, context-bounded semi-hard negatives |
| **2606.09700** *What Eyes See, LLMs Miss* **[A]** | 2026-06-08 | typographic human-perceptible adversarial attacks |
| **2605.31563** *Disagreeing Rationales* **[A]** | 2026-05-29 | label + rationale representation protocol (feeds the disagreement axis) |
| **2603.21298** *More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech* **[C]** | 2026-03 | not previously on the map |
| **DeHate** ACM MM 2025 `10.1145/3746027.3758272` **[B]** | 2025 | **explicit AND implicit** hateful video dataset — reclassify on our map |
| **LEAF** `aclanthology.org/2026.findings-acl.604` **[B]** | 2026-07 | venue confirmed: ACL 2026 **Findings** |

### D.1 Updated comparator table (protocol-tagged — do not merge rows)

| system | HateMM | MHC-EN | MHC-ZH | ImpliHateVid | protocol note |
|---|---|---|---|---|---|
| **this project, bare BCE head on frozen features** | **0.8774** | **0.7331** | **0.7821** | **0.9118** | official splits, single number as given to me |
| **SAGE** (ACL 2026 Main, `2026.acl-long.817`) **[B]**, inherited | 0.8628 M-F1 / 0.8710 acc | 0.7962 / 0.8375 | 0.7484 / 0.7901 | — | current published frontier on HateMM |
| **MM-HSD** (ACM MM 2025, `10.1145/3746027.3754558`; arXiv 2508.20546) **[B]** | 0.874 M-F1 self-reported | — | — | — | **5-fold CV, not the official split**; SAGE's reproduction gives 0.8054 |
| **ImpliHateVid / TCL** (ACL 2025) **[A]** | 97.58 F1 | — | — | **87.73 F1 / 87.53 acc** | HateMM number uses a non-comparable protocol |
| **simple embedding fusion** (2502.07138, MM4SG@WWW25) **[A]** | "SOTA on HateMM, +9.9 F1 pts" | — | — | — | workshop; exact number not in the abstract |
| MultiHateClip original (ACM MM 2024) **[B]**, inherited | — | 0.79 binary | 0.78 binary | — | binary; the EN>ZH gap exists only in 3-class |

**Reading:** the project's 0.8774 on HateMM is above SAGE's 0.8628 and above MM-HSD's self-reported
0.874, and 0.9118 on ImpliHateVid is above the dataset paper's 0.8773. **On the accuracy axis the
project is at or past the published frontier on two of four benchmarks — with a bare head.**
That is a validity finding about the benchmarks, not a methods contribution (§C.1).

---

## (E) Structural gaps a mechanism could occupy — one-liners

**E.1** Every transductive-as-a-set method (TransCLIP, StatA, ZLaP, SAT, vMF-shrinkage) is built for
**many-class, text-anchored, roughly-balanced** image label sets; **nobody owns transduction for a
binary, prior-shifted, few-hundred-item moderation test pool whose "class anchors" are policy
definitions rather than object names** — and the imbalance literature says balancing helps only if
the prior is known and is harmful otherwise, which makes the prior the mechanism, not a nuisance.

**E.2** The project's own W4 result says drift failure is **calibration, not separability**, and the
label-shift branch (BBSE/EM/Dirichlet) is the exact matching mechanism — **but no one has ever
estimated the test-pool label prior transductively for hateful video** (SCANNER does centroid
alignment; it does not estimate p(y)).

**E.3** Implicit vs explicit is currently treated as a **dataset partition**, never as a **latent
coordinate the decision rule is a function of** — and the published 3-class table (EX 57.02 < IM 66.05
< NH 84.48) shows the live confusion is EX↔IM, so a mechanism whose *form* changes along that
coordinate is not the same paper as any of DuPL / HatePrototypes / TIHD.

**E.4** Every published implicit-hate *mechanism* is text-only (DuPL, HatePrototypes, FiADD,
label-aware hard negatives, implied-statement triplets); the video-side is datasets + reasoning
pipelines — **nobody has a mechanism that uses the fact that in video the surface cue and the
required inference sit in *different modalities***.

**E.5** Everything that reliably beats a frozen-feature linear probe at n≈10²–10³ is
**closed-form or hyper-parameter-free** (GDA, LDA, Tip-Adapter cache, LP++, CLAP) and **borrows a
text class anchor** — the unoccupied version is a closed-form/validation-free head for a **binary
policy label with heterogeneous multimodal frozen features**, where the text anchor is a policy
sentence rather than a class name.

**E.6** The counter-literature has converged on one demand — **"prove your adaptation cannot destroy
the un-adapted model"** (StatA's anchoring, Illusion-of-Progress's trustworthiness finding, Pitfalls'
model-selection result) — and **no work states that guarantee in the binary/moderation regime**,
where a false-positive-side regression is a deployment failure rather than an accuracy delta.

---

## (F) Coverage and blind spots

**Searched:** WebSearch (~24 queries), arXiv Atom API over `export.arxiv.org` (4 queries: implicit-hate
video, transductive×test-time, OT×TTA, hate/hateful/harmful video, all sorted by submission date
descending), direct WebFetch of `arxiv.org/abs/2502.07138` and `arxiv.org/html/2508.06570v1`.

**Not searched / cannot claim:**
1. **Non-English literature** (Chinese journals / CCF-Chinese venues) — not queried, same as prior rounds.
2. **Paywalled full texts.** HCG-MPB, TIHD, MATCH, DeHate, DuPL are ACM DL / IEEE — I have DOIs and
   abstract-level descriptions only. **DeHate's actual EX/IM protocol and numbers are unknown**, and
   it is the most likely direct competitor on axis B. Getting that PDF is the highest-value follow-up.
3. **Venue accept-lists not yet indexed:** ACM MM 2026 (November), EMNLP 2026, NeurIPS 2026. I cannot
   claim there is no competitor there. My "no new SOTA in 2026-06→08" statement covers **arXiv +
   indexed DOIs only**.
4. **arXiv abstract-field coverage.** My video sweep keys on the strings "hate"/"hateful"/"harmful"
   + "video" in the abstract. A paper phrased purely as "toxic short-form content" would be missed.
5. **No leaderboard cross-check exists** — papers-with-code is gone; every number in §D.1 is
   self-reported by its own paper under its own protocol.
6. **Snippet-level items are marked [C] and were not fetched**: SAT `2411.17002`, `2411.15204`,
   `2511.18615`, `2505.05868`, `2509.04631`, `2605.01452`, `2409.18332`, `2304.04795`, `2602.06136`,
   `2508.05547`, `2509.00305`, `2604.21728`, `2402.04087`, `2111.03930`, `2604.03928`, `2309.11896`,
   `2406.07886`, `2106.05903`, `2603.21298`, `2506.03989`, and the ICML-2026 calibration-probe listing.
   **Titles and IDs came from result pages, but I did not open them — verify before citing in a paper.**
7. **Inherited, not re-verified this round:** SAGE, HCG-MPB, TIHD, MM-HSD, MultiHateClip, SCANNER
   numbers all carry their round-3 verification level.
8. I did **not** re-run any general hateful-video / retrieval / memory survey, per the mandate.
