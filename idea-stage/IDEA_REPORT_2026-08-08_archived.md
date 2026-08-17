# Research Idea Report — hateful video detection, post-Gate-0

## STATUS: ARCHIVED (2026-08-08, user decision)

**This report is closed; no follow-on experiment will be run.** All four phases of the
idea-discovery pipeline completed — landscape and gap map (§1), 25 raw candidates → 14 after dedup
→ 3 piloted (§2–§4), deep novelty check (§7), and a Phase-4 external hostile review (§9). The
Phase-4 reviewer (Codex `gpt-5.6-sol`, reasoning effort **ultra**) scored the residual-led headline
proposal **2/10 — strong reject**, on three independent grounds: the mechanism
`r = CLIP(OCR) − CLIP(ASR)` is a **fixed rank-constrained linear metric** over `[OCR ‖ ASR]` that a
learned projection reproduces exactly, so it is packaging rather than a mechanism (§9.2a); the
census → residual → classification-flip → marginal-gain-after-naive-fusion chain is **open at every
link**, including a stop-work 22-vs-28 data-lineage failure and an R1 complementarity gate that a
random key would pass (§9.2b); and the **performance ceiling is 0.862 macro-F1 under *perfect*
repair of the entire target stratum, against MM-HSD's published 0.874** — impossible perfect repair
does not close the gap (§9.2c). Only the acquisition reframe (counterfactual-OCR-value acquisition
gate, §9.5) scored higher, **5/10 — ACM MM / ACL Findings tier**, and it is the single surviving
shape. **The §9.6 falsification factorial the reviewer prescribed was never executed**: the user
stopped the line before submission, so no arm A0–A9 number exists anywhere. What this round leaves
behind as reusable assets: the **OCR cache** — PaddleOCR PP-OCRv6, K=30 midpoint windows, **1246
videos** (851 HateMM + 395 HateClipSeg), ~19 min GPU wall, under `data/OCR/` with per-file sha256 in
`data/OCR/SHA256SUMS.json` and builder `scripts/ocr_cache/extract_ocr_windows.py` — and the **I5
redundancy-gate data** (`data/OCR/HateMM/i5_redundancy_gate.json`,
`scripts/ocr_cache/i5_redundancy_gate.py`), whose R1/R2/R3 numbers stand as measurements even though
§9.2b showed the gate's decision rules are not informative.

**If this line is restarted, resume from these three pointers.**

1. **§9.6 — the frozen falsification suite.** Paste-ready pre-registration: arms A0–A9 (no-OCR,
   OCR-as-feature, naive fusion, OCR-only / ASR-only / residual / learned-metric / MLP / shuffled
   second indices, neighbour-budget control), estimand `Δ*` against the strongest non-residual
   comparator, six conjunctive GO conditions, no AMBIGUOUS branch. Nothing in it was run.
2. **§9.4 — the results-to-claims matrix.** Frozen before any factorial number existed; it is the
   adjudication table for whatever §9.6 returns, and must be used as written rather than re-derived
   after seeing results.
3. **22-vs-28 lineage repair is a prerequisite to both** (§9.7 item 1, §9.10). Reconcile the census
   `OCR-required AND speech-not-required` count at video-ID level — §2 Idea 3 says 22/73, the gate
   script found 28 — before any R2 subgroup number, and hence any §9.6 stratum analysis, is quoted
   again.

---

**Direction**: a new method for hateful video detection, motivated by the TERA Gate-0 error-population
evidence and the Phase-1 gaps G-A / G-B / G-C / G-D / G-F. Novel mechanism + strong performance;
large departures from the existing RGCL-style pipeline are allowed.
**Generated**: 2026-08-08
**Ideas evaluated**: 25 raw candidates (5 Claude lens shards + 1 cross-model brainstorm)
→ 14 after mechanical dedup → 14 through the objective feasibility gate
→ 3 piloted → **3 recommended**, 4 killed outright, 4 demoted.

**Artifacts**
- generation bundle `idea-stage/codex_brainstorm_bundle.md`
- triage bundle `idea-stage/codex_triage_bundle.md`
- frozen pilot pre-spec `idea-stage/PILOT_FREEZE.md`
- pilot code `idea-stage/pilots.py`, results `idea-stage/pilot_results.json`
- reviewer traces `.aris/traces/idea-creator/2026-08-08_run01/`
- **deep novelty check (Phase 3)** `.aris/traces/novelty-check/2026-08-08_run01/` — dossier +
  Codex `gpt-5.6-sol` xhigh verdict, thread `019fdc7a-62ea-7140-a634-82e2555f2ccd`. Results folded
  into §2 (per-idea "Novelty" blocks) and §8.
- **external hostile review (Phase 4)** `.aris/traces/research-review/2026-08-08_run02/` — brief +
  Codex `gpt-5.6-sol` **ultra** verdict, thread `019fdc94-f8ca-7360-a75f-f837489219b7`.
  **Verdict: 2/10, kill the residual-led paper.** Folded into **§9**, which supersedes §2 Idea 3,
  §5 and §6 wherever they conflict.
- reviewer backend: **Codex MCP, `gpt-5.6-sol`, reasoning effort xhigh** (thread
  `019fdc59-19ba-7e31-9e52-9226097669fa`). Codex's shell sandbox was broken
  (`bwrap: loopback: Failed RTM_NEWADDR`), so both bundles were pasted inline rather than read from
  disk; content was identical. Novelty verification ran as a separate multi-source search pass.

---

## 0. Headline

The Gate-0 campaign left a signpost — "83.6% of our false negatives are short-and-localized or
cross-modal" — that reads like a mandate for single-segment selection. **Three things measured in
this session say that reading is wrong, and point somewhere else instead.**

1. **Localization is not what separates our successes from our failures.** In the Gate-C census,
   `short_localized` is the primary cause for 50.7% of false negatives but **60.0%** of the
   true-positive controls (Fisher OR 0.69, p=0.51), and the hateful span is not shorter in the
   failures (median span/duration 0.145 vs 0.100, Mann-Whitney p=0.75). The 83.6% union is a
   property of the data, not a property of failure.
2. **Modality coverage is what separates them.** `on_screen_text required` holds for 53.4% of false
   negatives vs 33.3% of true positives (Fisher OR **2.29**, p=0.083); 30.1% of false negatives
   need on-screen text *and* have no usable speech, vs 16.7% of true positives. Our pipeline has no
   OCR modality at all and its segment features are visual-only. (Underpowered — 30 TP controls —
   but it is the only enriched property in the census.)
3. **HateMM's official hateful spans are not localized.** Measured over the 298 hateful train
   videos: gold spans cover **mean 0.717 / median 0.829** of the video, 34.6% of hateful videos are
   annotated ≥90% hateful, and only 11.7% are ≤25%. The same blinded coders' *minimal sufficient*
   evidence intervals cover a median of **0.131**. On the 99 videos where both exist, official
   coverage is 2.0× the minimal-evidence coverage. **Trimming a HateMM video to its gold span
   removes about 18% of it.**

Fact 3 is the one that moves the project. It means (a) top-1 segment selection on HateMM has a
chance hit rate of **0.76**, so uncorrected localization metrics in this field are close to
vacuous; and (b) the published claim that trimming to gold spans buys **+19.34 / +30.45 macro-F1**
(Revealing Temporal Label Noise, arXiv:2508.04900) cannot be a dilution effect on this data — we
measure the oracle-alignment term at **+0.48 points**.

---

## 1. Landscape summary

The field has three occupied positions and one thin one. **MM-HSD** (ACM MM 2025, arXiv:2508.20546)
holds the performance frontier on HateMM at macro-F1 0.874 by adding on-screen (OCR) text as a
fourth modality and using it as the *query* of a cross-modal attention block — no retrieval, no
contrastive learning, no temporal structure, English only. **MoRE** (WWW 2025) holds the retrieval
position at 0.8235: it is the only retrieval-augmented hateful-video method, and it retrieves
*whole videos* with a frozen weighted-cosine retriever into per-modality experts, trained entirely
with BCE — its "contrastive" component is attention in costume. **MultiHateGNN** (BMVC 2025, 0.771
F1) holds soft attention-weighted segment aggregation, which is why any new mechanism in the
temporal direction must be hard/discrete to be publishable. Inference-time MLLM detection (HVGuard
and 7+ others) is saturated.

The novelty check revised one Phase-1 gap materially. **G-A is no longer empty**: MultiHateLoc
(arXiv:2512.10408) already performs genuinely discrete, non-attention selection in hateful video —
modality-aware top-K MIL that hard-selects ~a third of the frames per modality. Three things keep
G-A open in a narrower form: its K is a third of the video rather than a single segment, its
deliverable is weakly-supervised localization (mAP 0.645 / AUC 0.799 on HateMM) rather than
video-level classification, and it is positive-only. Any G-A submission must now ablate against
MultiHateLoc's top-K MIL and justify top-1 over top-K.

**The Phase-3 deep check (2026-08-08) revised the landscape again, in three places.**

- **G-C is no longer thin — it is occupied in-domain.** *LEAF* (ACL Findings 2026,
  `2026.findings-acl.604`, Lang et al., "Towards Lightweight Explainable Hateful Video Detection via
  Self-Grounding CoT Guided Stage-Wise Distillation") already does LMM-teacher → smaller-multimodal-
  model-student distillation for hateful **video**, with no LMM at inference, evaluated on HateMM and
  MultiHateClip. It distils *explanations*. The generic claim "we are the first to distil an MLLM into
  an inference-time-MLLM-free hateful-video model" is therefore dead and must not be written.
- **The evidence-type taxonomy is not new either.** *DeHate* (ACM MM 2025,
  `10.1145/3746027.3758272`, 6,689 TikTok/BitChute videos) ships human **"contributing modality"**
  labels at segment level alongside hate localization and target groups; *Ex-HateMM / Ex-ImpliHateVid*
  (arXiv:2606.11953, IARE) ship fine-grained annotations of multimodal harmful elements. Our MLLM
  labelling pass produces a label type that already exists as public human annotation. It buys us
  HateMM coverage and zero annotation cost, not conceptual priority.
- **Correction to the Phase-1 reading of Tzelepi & Mezaris (CVPRW/MULA 2025, arXiv:2504.09914).**
  Verified against the paper: the LMM (MiniGPT-4) generates descriptions and elicited emotions at
  **both training and inference** — the authors say so explicitly when discussing cost. It is *not*
  a train-time-only distillation and should not be cited as one.

Two further in-domain entrants matter for positioning. *LELA* (arXiv:2602.09637) is the first
training-free LLM framework for hate video localization; it decomposes each video into **five**
modalities including OCR and prompts an LLM per frame — the maximal-cost end of the design space our
budget argument attacks. *SCANNER* (arXiv:2602.00132) is test-time *adaptation* for shifting hate
video — adjacent vocabulary ("test-time"), orthogonal mechanism (no acquisition).

G-B remains n=1 (MoRE) and whole-video-keyed. G-C survives only in the narrow form
"training-only evidence supervision → a test-time policy over *what to compute*". G-D is diagnosed
(arXiv:2508.04900; Majority Vote Silences Minority Values, arXiv:2606.28772) with no method
anywhere. G-F is empty for the part that matters: joint classification-and-grounding metrics exist
in video QA (NExT-GQA's Acc@GQA, CVPR 2024; EG-VQA's EG-F1), but **chance correction against the
video's own span coverage appears nowhere in video localization, WS-TAL, VAD, or pointing-game
evaluation** — and, per fact 3 above, it is exactly what HateMM needs.

---

## 2. Recommended ideas (ranked, post-pilot)

### Idea 1 — Pay-for-Evidence: a distilled evidence-type gate driving budgeted discrete modality acquisition
*(fusion of candidates I8 and I7; gaps G-C × G-A; pilot **POSITIVE**)*

- **Method (what we actually do).**
  1. Train a small head on frozen CLIP features to predict, per video, **which evidence type the
     decision needs** — on-screen text / visual symbol / speech. The supervision is a *type*, not a
     verdict: at training time only, an MLLM (Claude, under the existing frame-reading exemption)
     reads frames and labels the evidence type per segment. No MLLM at inference.
  2. Use that predicted type as the input to a **discrete acquisition policy**: a differentiable
     knapsack / smoothed-DP top-k operator (arXiv:2601.21775) picks, under a real cost budget, at
     most a handful of (segment, modality) pairs on which we actually *pay* for the expensive
     modality — running OCR on 3 of 30 segments instead of all 30, or on none at all when the gate
     says the video is speech-carried.
  3. Classify from the acquired sparse evidence plus the always-cheap whole-video features.
- **Hypothesis.** The failure population is modality-structured, not time-structured, so a gate that
  predicts *which* modality is needed — before paying for it — recovers most of OCR's benefit at a
  fraction of OCR's cost, and repairs the 30.1% of false negatives that carry on-screen text with
  no usable speech.
- **Pilot result: POSITIVE.** The reviewer's stated kill-risk for this family was "the policy must
  decide where OCR is needed *before* seeing OCR, which may be impossible from pooled visual/ASR
  features." Pilot P3(a) tested exactly that and it is possible: a logistic probe on frozen
  [whole-video visual ‖ whole-video text ‖ max-pooled segment visual] predicts
  `on_screen_text required` at **OOF AUROC 0.842**, bootstrap 95% lower bound **0.773**, against the
  frozen GO bar of 0.68/0.55. Post-hoc, this is not an error-status artifact of the stratified audit
  sample: AUROC is 0.768 within the false-negative stratum, 0.880 within true positives, 0.898
  within false positives.
- **What the pilot did NOT support.** The originally proposed use of the type — hard-partitioning
  the retrieval memory so only same-type entries can be retrieved — is **NO-GO**: typed retrieval
  improved neighbour label purity by ≥0.05 in **0 of 5** folds (per-fold deltas +0.018, +0.011,
  +0.010, and two smaller). The type is learnable and useful; routing the *memory* by it is not.
  The recommendation is therefore the acquisition-policy form, not the index-partition form.
- **Minimum next experiment.** Build the OCR cache (PaddleOCR/EasyOCR over ~89k HateMM-train frames,
  1–3 GPU-h — currently absent from the project), then one frozen-fold OOF readout comparing
  zero-OCR / 3-segment-OCR / all-30-OCR. GO if all-30 OCR gives ≥ +0.015 macro-F1 **and** the
  3-segment policy recovers ≥90% of that gain with ≤10% of the OCR calls.
- **Novelty (Phase-3 deep check, multi-source + Codex `gpt-5.6-sol` xhigh): PARTIAL — 7/10,
  PROCEED WITH CAUTION.** Axis-by-axis:
  - **(a) evidence-acquisition budgeting in hateful/harmful video — SURVIVES.** No in-domain method
    treats modality extraction as a costed action. MM-HSD runs OCR on everything; LELA runs five
    modalities plus an LLM on every frame; MultiHateLoc *weights* modality features it has already
    computed. The nearest thing in content moderation is Filter-And-Refine (arXiv:2507.17204), a
    per-video router gating an expensive MLLM — a model cascade, not modality acquisition.
  - **(b) MLLM annotation distilled into a test-time-MLLM-free gate — DEAD AS WORDED, survives
    narrowly.** LEAF owns "LMM → small model, no LMM at inference, hateful video". DeHate owns
    "contributing-modality labels". What survives is only: *training-only evidence supervision
    supervising a policy over what to compute*. Required re-wording — say **"MLLM-supervised
    acquisition policy"**, never "distillation", and never "first evidence-type taxonomy".
  - **(c) cost-aware modality routing at segment granularity — SURVIVES in-domain**, and survives
    cross-domain on granularity: every cost-aware router we found decides **per query** (ModaRoute,
    VOILA, SAFE-Cascade, post-hoc escalation) or **per sample** (DyMo), never per temporal segment of
    an untrimmed video with no query available.

  | # | Nearest neighbour | Where | Differentiator |
  |---|---|---|---|
  | 1 | **Post-hoc Selective Modality Escalation** (arXiv:2607.05438, 2026-07) | Multimodal RAG, MultiModalQA | Same economics — verifier localizes the missing modality, calibrated value-of-escalation router pays for VLM evidence only there. But: per *question*, decision made *after* a draft answer, no segment structure, no knapsack. Its headline finding ("modality relevance ≠ modality utility") is a direct threat to our type-prediction gate and must be answered with a counterfactual-utility ablation. |
  | 2 | **MultiHateLoc** (arXiv:2512.10408, 2025-12) | Hateful video | Modality-aware hard top-K MIL over segments — the only in-domain discrete modality-and-segment selector. But it selects among *already-extracted* features, targets weakly-supervised localization, is positive-only, and has no cost model. Mandatory ablation baseline. |
  | 3 | **LEAF** (ACL Findings 2026) | Hateful video | Occupies "LMM teacher → MLLM-free student" in our exact domain and datasets. Distils explanations for explainability; nothing is acquired, nothing is budgeted. Must be cited in the first paragraph of related work. |
  | 4 | **DeHate** (ACM MM 2025, `10.1145/3746027.3758272`) | Hateful video dataset | Already supplies human segment-level *contributing modality* labels. Kills the "novel evidence-type supervision" claim; becomes an external-validity asset (cross-dataset check of our MLLM labels). |
  | 5 | **ModaRoute** (arXiv:2507.13374, ICCVW 2025) | Multimodal video retrieval | GPT-4.1 routes each query across ASR/OCR/visual indices, 1.78/3.0 modalities, −41% compute. LLM *at inference*, query-driven, indices pre-built offline, retrieval not classification. Also independently reports that 34% of clips carry scene text absent from ASR — cite it as external support for our census. |

  Runners-up to cite but not table: VOILA (arXiv:2602.03007, per-query *fidelity* tiers), DyMo
  (arXiv:2601.22853, inference-time modality selection but for *imputed* missing modalities, no
  cost), SAFE-Cascade (arXiv:2606.19646), CLaMR (arXiv:2506.06144, indexes on-screen text + speech
  with a modality-aware loss), Q-Gate (arXiv:2604.17422) and Q-Frame (arXiv:2506.22139) — the
  frame-selection line spends *resolution/tokens*, never a different modality, and is training-free
  and query-conditioned. VisionSelector (arXiv:2510.16598) is the differentiable-top-K-under-budget
  precedent inside one modality. Differentiable knapsack (arXiv:2601.21775) still has no
  vision-language application. Classic AFA (survey arXiv:2502.11067; CAMA arXiv:2505.16791;
  BRiG-AFA arXiv:2608.02305; AFABench arXiv:2508.14734) is entirely tabular/medical — **no
  content-moderation or video application exists**, so it is inspiration to cite, not a collision.
- **The hostile review to pre-empt** (Codex's predicted sentence): *"The method combines LEAF-style
  LMM-to-student supervision, DeHate contributing-modality labels, MultiHateLoc-style segment
  selection, and an off-the-shelf differentiable knapsack; the only new element is using these known
  components to decide where to run OCR."* Codex judges this criticism **credible**. The composition
  is genuinely unreported and system-level novel, but it degrades to an engineering paper unless:
  (i) the acquisition policy is trained/calibrated against **marginal classification-loss reduction
  per unit cost**, not against evidence-type relevance alone; (ii) segment OCR costs actually vary
  (otherwise it is plain top-k and the word "knapsack" must be dropped); (iii) retrieval interacts
  with acquisition rather than being appended.
- **Feasibility**: one RTX 5090; ~5M trainable head; the only real cost is the one-off OCR cache.
- **Risk**: MEDIUM. **Contribution type**: method + empirical.
- **Reviewer's likely objection.** "Predicting *that* on-screen text matters is not the same as
  reading it; and a cheap heuristic OCR-presence detector may be as good as an MLLM-distilled type."
  Both must be ablated head-to-head — heuristic typing is the baseline that decides whether the
  distillation earns its place.
- **Mandatory ablations for the novelty defence** (Phase-3; these go into the pre-registration, not
  the rebuttal):
  1. Budget sweep as a **macro-F1 × measured-cost Pareto curve** over 0…30 purchased segments, with
     real cost units (OCR calls, wall-clock, GPU-s), against: zero-OCR, all-30 OCR (= MM-HSD's
     operating point), random-3, uniform-3, cheap text-likelihood/salience-3, per-**video** gate at
     matched cost, and oracle-3.
  2. **Relevance vs. utility** (the arXiv:2607.05438 attack): gate trained on evidence *type* vs.
     gate trained on **counterfactual OCR benefit** (per-video Δloss with OCR minus without).
     Report acquisition recall and realized loss reduction, not just type AUROC.
  3. **Knapsack earns its name or loses it**: if per-segment OCR cost is uniform, benchmark against
     hard top-k and rename. To keep the DP operator, demonstrate heterogeneous costs (frame count,
     resolution, text density) and a win over greedy/top-k.
  4. **Label-source ablation**: our MLLM evidence-type labels vs. DeHate human contributing-modality
     labels vs. a blinded human-coded HateMM subset; report agreement.
  5. Factorial ablation of {evidence supervision} × {routing} × {purchased OCR} × {kNN retrieval}.
  6. Stratified results on the four census cells: OCR-required/no-speech, OCR-required/with-speech,
     speech-carried, OCR-not-required.
  7. At least one dataset beyond HateMM (MultiHateClip or DeHate) or the cost claim reads as
     HateMM-specific.
- **Why we should do this.** It targets the only property that is statistically enriched in our
  failures, it inherits MM-HSD's demonstrated 0.874 ceiling as evidence that the modality is worth
  money, and it produces the "mechanism + error-population repair + efficiency" claim shape the
  project wants rather than a bare SOTA shout.

### Idea 2 — CCEF: chance-corrected evidence faithfulness, and the HateMM span-coverage result
*(candidate I12, carrying the corpse of I1; gap G-F; pilot **evidence delivered**)*

- **Method (what we actually do).**
  1. Report, for any method that emits a per-segment score, a **chance-corrected** top-1 hit rate:
     κ-style normalization of the gold-span hit rate against *that video's own* span/duration
     coverage, so a video whose spans cover 83% of its runtime cannot make a random selector look
     grounded.
  2. Report a **faithfulness gap**: macro-F1 minus evidence-faithful macro-F1, where a true positive
     counts only if its argmax segment has tIoU ≥ 0.5 with a gold span.
  3. Publish the accompanying dataset result that makes the instrument necessary, plus the
     controlled trim decomposition (full view / length-matched random window / gold-span window)
     that separates generic shortening from oracle alignment.
- **Hypothesis.** Localization numbers in hateful video are currently uninterpretable because the
  chance rate is enormous and unreported, and the published trimming gain is a label-alignment
  artifact rather than temporal headroom.
- **Pilot result: the supporting numbers are in hand.**
  - HateMM official gold-span coverage over 298 hateful train videos: **mean 0.717, median 0.829**,
    q25 0.570, q75 0.933; 34.6% of hateful videos are annotated ≥90% hateful. ⇒ the chance top-1
    hit rate is **0.762**.
  - The same blinded coders' minimal sufficient evidence intervals cover **median 0.131**; paired on
    99 videos, official coverage is **2.0×** the minimal-evidence coverage.
  - Pilot P1 (frozen decomposition): macro-F1 full **0.8196**, length-matched random window
    **0.8155**, gold-span window **0.8203**. Generic-trim term **−0.41 pt**; oracle-alignment term
    **+0.48 pt**, bootstrap 95% CI **[−0.79, +1.76]**. Under the frozen rule this is **AMBIGUOUS**
    (the upper bound 1.76 misses the ≤+0.5 GO-AS-NEGATIVE bar), but the point estimate is null and
    the mechanism is now explained: trimming to a gold span removes ~18% of the video, so there is
    almost nothing to concentrate.
  - Contrast: arXiv:2508.04900 reports **+19.34 / +30.45 macro-F1** from exactly this trimming
    operation, with no random-window control and no decomposition. That gap is the paper.
- **Novelty (Phase-3 lightweight check): PARTIAL — 5/10, PROCEED WITH CAUTION as a *secondary*
  contribution.** Part (a), per-video chance correction against the video's own attainable hit
  probability, was not found in video localization, WS-TAL, VAD, or pointing-game evaluation, and
  survives. Part (b), the faithfulness gap, is anticipated by NExT-GQA's Acc@GQA (CVPR 2024) and
  EG-VQA's EG-F1 — cite as prior art, claim only the hate-video adaptation. Codex additionally rates
  the gap itself as "a derived difference between two metrics, not a strong independent contribution".
  Nearest neighbours: Acc@GQA; EG-F1; arXiv:2508.04900 (the trimming claim CCEF exposes);
  HateClipSeg (arXiv:2508.01712, segment-level annotations, conventional localization eval);
  MultiHateLoc (arXiv:2512.10408, reports mAP/AUC with no coverage correction).
- **Statistical repair required before writing (Codex, and it is right).** Averaging per-video
  κᵢ = (hᵢ − pᵢ)/(1 − pᵢ) blows up as pᵢ → 1, is undefined at pᵢ = 1, and is not Cohen's κ from a
  single Bernoulli draw. Report the aggregate form Σᵢ(hᵢ − pᵢ) / Σᵢ(1 − pᵢ) — achieved excess hits
  over maximum possible excess — alongside it, and stop calling the per-video form "κ" unqualified.
  Also: pᵢ must be derived from the **actual discrete candidate segments and hit rule**, not from raw
  duration coverage (duration coverage is wrong once segments have finite width, overlap, or a tIoU
  threshold). Verify pᵢ by Monte-Carlo random selection. Define precisely how an ungrounded positive
  moves through the whole confusion matrix, and report plain macro-F1, grounded macro-F1, and the gap
  as three numbers — never the gap alone.
- **Expected hostile review**: *"The grounded-F1 component is Acc@GQA/EG-F1 transplanted to hate
  videos, while the proposed κ score is a straightforward correction for unusually long gold spans
  and is not sufficient as a standalone metric contribution."* ⇒ ship CCEF as Idea 1's evaluation
  protocol, led by the 0.829-median-coverage dataset result, not as a standalone metric paper.
- **Feasibility**: zero GPU, minutes of CPU. **Risk**: LOW. **Contribution**: diagnostic + evaluation.
- **Reviewer's likely objection.** "A metric paper without a method." Mitigation: ship it as the
  companion instrument to Idea 1 or as a short analysis paper, and lead with the dataset result
  (0.829 median coverage) rather than the metric definition — the result is what makes the metric
  necessary.
- **Why we should do this.** It costs an afternoon, it is the cheapest citable contribution
  available to us, it applies to any method emitting a per-segment scalar (including MultiHateGNN's
  attention weights and MoRE's retrieval scores), and it gives our own selection work an honest
  yardstick now that the Gate-A oracle numbers are permanently sealed.

### Idea 3 — Unsaid-text retrieval: an OCR-minus-ASR residual as the retrieval key
*(candidate I5; gaps G-A × G-B; **not piloted** — needs the OCR cache)*

- **Method (what we actually do).**
  1. Build segment-level OCR text and segment-level ASR text (the ASR already exists at K=30 in
     `data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl`), encode both with the frozen CLIP text
     tower.
  2. Hard-select the segment with the largest learned **OCR-minus-ASR semantic residual** — the
     on-screen claim that was never spoken — and use *that residual vector*, not the whole video,
     as the retrieval key into the kNN memory.
  3. Classify from the retrieved neighbourhood plus the whole-video features.
- **Hypothesis.** Whole-video keys are dominated by the ~85% benign remainder and by topic, so an
  OCR-carried video retrieves speech-carried rants; the residual isolates precisely the 22/73
  false-negative population that needs on-screen text and has no speech.
- **Gate before building it.** The cheap OCR-free prior question — *is retrieval already doing OCR's
  job?* — was designed but not run within the 3-pilot budget: build a fold-internal kNN memory over
  whole-video features, then measure (i) neighbour label purity conditioned on
  `on_screen_text required`, and (ii) the fraction of the 22 OCR-required-no-speech census false
  negatives that a kNN vote flips. GO-REDUNDANT if recovery ≥0.30 with a purity gap ≥0.10 (retrieval
  substitutes for OCR — the redundancy result is itself the paper); GO-COMPLEMENTARY if recovery
  ≥0.30 with purity gap ≤0.02 (then the OCR cache is worth building); NO-GO below 0.15.
- **Novelty (Phase-3 deep check): CONFIRMED-NOVEL, narrowly — 5/10, PROCEED WITH CAUTION.** No work
  in any searched domain uses an aligned OCR−ASR embedding difference as a retrieval key. The
  construction "OCR minus ASR residual" returned **zero** hits across arXiv full-text, WebSearch and
  OpenAlex; the closest recorded observation is ModaRoute's finding that 34% of clips carry scene text
  absent from ASR — an existence proof for the population, not a method. Write it as "to our
  knowledge" and cite the search: embedding differences and cross-modal discrepancy features are
  generic enough that a categorical first-ever claim is unsafe.

  | # | Nearest neighbour | Where | Differentiator |
  |---|---|---|---|
  | 1 | **MoRE** (WWW 2025, `10.1145/3696410.3714560`) | Hateful video | The only retrieval-augmented hateful-video method — but whole-video / per-modality keys, frozen weighted-cosine retriever, BCE. Never asks what the key *should* be. |
  | 2 | **CLaMR** (arXiv:2506.06144) | Multimodal content retrieval | Jointly indexes frames + transcribed speech + **on-screen text** + metadata with a modality-aware loss for dynamic modality selection. Keeps OCR and ASR as **separate streams**; never forms a cross-modal residual; everything pre-extracted. Strongest "already done" citation. |
  | 3 | **ModaRoute** (arXiv:2507.13374) | Video retrieval | Routes queries across OCR/ASR/visual indices; no residual, needs a query, LLM at inference. Supplies the 34%-scene-text-not-in-ASR statistic. |
  | 4 | **MM-HSD** (arXiv:2508.20546) | Hateful video | Uses OCR heavily and as the CMA *query*, i.e. the closest thing to "privilege the on-screen text channel" — but always-on, classification-side, no retrieval. |
  | 5 | **MultiHateLoc** (arXiv:2512.10408) | Hateful video | Selects informative temporal regions per modality, but forms no residual representation and retrieves nothing. |
- **Expected hostile review**: *"OCR minus ASR is merely a fixed linear projection of concatenated
  OCR and ASR embeddings, so the proposed retriever is a handcrafted metric over modalities already
  used by CLaMR and MoRE rather than a new retrieval mechanism."* This is the central risk, and it is
  formally correct: r = o − a = [I, −I]·[o; a], so a learned metric over the concatenation can exactly
  reproduce Euclidean residual retrieval. The residual is a **constrained inductive bias, not extra
  information**.
- **Two frozen kill criteria to add to the pre-registration.**
  1. **Reparameterization kill**: a parameter-matched learned linear/Mahalanobis metric over
     concatenated [OCR ‖ ASR] must be run as the decisive control. If it matches or beats the
     residual, the mechanism claim collapses and only the empirical finding survives.
  2. **Semantic-subtraction kill**: CLIP embedding subtraction is not semantic subtraction. Large
     residuals may be OCR errors, paraphrase, timing misalignment, benign background text, or an
     empty-ASR embedding pointing in an arbitrary direction. Mandatory: manual audit of the top
     high-residual segments with a per-cause breakdown, plus the noise-floor control already planned.
     If the principled form o − E[o|a] is needed to make it work, that is a **different, learned**
     mechanism and must be presented as such.
- **Further mandatory ablations**: residual vs. OCR-only vs. ASR-only vs. concatenation vs.
  equal-capacity MLP projection; subtraction before vs. after L2-normalization; cosine vs. Euclidean;
  hard argmax vs. soft pooling vs. top-m vs. oracle gold-segment; gains isolated to the
  OCR-required/no-speech stratum with **no regression** on speech-carried hate; a controlled-overlay
  synthetic test (add on-screen text without changing speech) as the clean causal probe; and
  near-duplicate/repost/template deduplication between the retrieval memory and the eval split.
- **Feasibility**: gate is CPU-minutes; the method needs the same OCR cache as Idea 1, so the two
  share their only real cost. **Risk**: MEDIUM. **Contribution**: method + empirical.
- **Reviewer's likely objection.** Once OCR exists, the residual may be dominated by OCR and ASR
  *errors* rather than genuinely unsaid meaning. A noise-floor control (residual computed on
  videos where OCR and ASR agree) is mandatory.
- **Why we should do this.** It is the only remaining candidate that sits inside the project's own
  retrieval identity (G-B, our validated cross-dataset kNN memory) while targeting the modality
  property the census actually enriched, and it shares its cost with Idea 1.

---

## 3. Pilot experiment results

All three ran on CPU in under one minute total, well inside the 8 GPU-h budget (**~0.02 GPU-h
consumed**). HateMM-train only, frozen seed-20260807 5-fold OOF, zero test contact. Decision rules
were frozen in `idea-stage/PILOT_FREEZE.md` before any pilot number existed.

| Pilot | Idea | Cost | Frozen bar | Observed | Verdict |
|---|---|---|---|---|---|
| P1 trim-gain decomposition | I1 | ~25 s CPU | oracle-alignment ≥+1.5 pt with CI LB>0 → GO; CI UB ≤+0.5 → GO-AS-NEGATIVE | full 0.8196 / rand-window 0.8155 / gold-window 0.8203; generic −0.41 pt, oracle **+0.48 pt** CI [−0.79,+1.76] | **AMBIGUOUS** (point estimate null) |
| P2 segment-keyed retrieval purity | I4 | ~15 s CPU | hit ≥2× chance and ≥0.35 with LB>chance; **and** ≥+1.5 pt macro-F1 | hit **0.544** vs chance **0.762** (ratio 0.71, LB 0.487); macro-F1 0.8231 → 0.8172, **−0.59 pt** | **NO-GO** (both metrics) |
| P3 evidence-type routability | I8 | ~10 s CPU | (a) AUROC ≥0.68 with LB>0.55; (b) typed purity ≥+0.05 in ≥4/5 folds | (a) AUROC **0.842**, LB **0.773** → GO; (b) **0/5** folds → NO-GO | **(a) GO, (b) NO-GO** |

> **[勘误 2026-08-09]** P2 行的 0.544 低于 chance 系 argmax 并列破序 artifact(`pilots.py:175` 的
> `np.argmax` 在 K=20 邻居的离散统计量上按最低下标破并列,51.3% 仇恨视频被送到 k=0;随机破并列后
> 同一选择器 hit **0.768** ≈ chance **0.762**);NO-GO 维持(ratio 1.008 < 1.3×),−0.59 pt 为 null
> (CI [−2.16,+0.99])。详见 `idea-stage/P2_FORENSIC_MEMO.md`。

Post-hoc (exploratory, computed after the frozen verdicts, revising nothing):

- P2 re-scored against the coders' *minimal sufficient intervals* instead of the coarse official
  spans: hit 0.444 vs chance 0.361 vs a random-selector control 0.354 — ratio 1.23, bootstrap CI
  [0.343, 0.545] straddling chance. The NO-GO stands on the sharper target too.

  > **[勘误 2026-08-09]** 这里的 0.444 同样带 argmax 并列破序 artifact;去掉破并列后为 hit **0.410**
  > vs chance 0.361,lift +0.050 CI [−0.009,+0.109](随机选择器对照 0.363)。结论方向不变:
  > 相对更锐利的 minimal-interval 目标也只是"与 chance 不可区分",NO-GO 维持。详见
  > `idea-stage/P2_FORENSIC_MEMO.md`。
- P3(a) probe AUROC by stratum: FN 0.768 / TP 0.880 / FP 0.898 — the probe is not predicting
  error status.
- HateMM span-coverage statistics as reported in §0 and §2.

**What the pilots changed.** The cross-model jury's top-ranked idea (I4, segment-keyed retrieval
purity) was killed by its own MVE — the purity-selected segment lands *below* chance on gold spans
and costs 0.59 macro-F1 points. The jury's second pick (I8) survived in half: the evidence-type
signal is real and strong, its proposed use as a memory partition is dead. This is exactly the
re-ranking the pilot phase exists to produce.

> **[勘误 2026-08-09]** "the purity-selected segment lands *below* chance on gold spans and costs
> 0.59 macro-F1 points" 两处均需修正:0.544 低于 chance 系 argmax 并列破序 artifact(随机破并列后
> 0.768 ≈ chance 0.762,即 *at* chance,不存在"反相关于证据"的现象);−0.59 pt 为 null
> (CI [−2.16,+0.99]),不是实证伤害。I4 的 NO-GO 维持(ratio 1.008 < 1.3× 杀线),且被 forensic
> 以机制理由独立加强:`p_j` 的 within-video AUROC 仅 0.511 [0.488,0.533]。详见
> `idea-stage/P2_FORENSIC_MEMO.md`。

---

## 4. Eliminated and demoted ideas

| # | Idea | Status | Reason |
|---|---|---|---|
| I4 | Segment-keyed retrieval-purity closed loop | **KILLED by pilot** | P2: selection hit 0.544 vs chance 0.762 (ratio 0.71); classification −0.59 pt. Was the jury's #1 on paper. |
| I8(b) | Typed hard partition of the kNN memory | **KILLED by pilot** | 0/5 folds reached +0.05 purity. The type survives as an acquisition gate (Idea 1); the partition does not. |
| I1 | Trim-gain decomposition as a standalone method | **DEMOTED, folded into Idea 2** | P1 AMBIGUOUS with a null point estimate; the regularizer half is a known crop-consistency loss (Schirrmeister et al., tied-sample loss). Its diagnostic half is now the backbone of Idea 2. |
| I6 | Silence route + cross-tower text imputation | **KILLED (jury)** | A ridge map img→text adds no information the image embedding does not already contain; any gain is reparameterization. Prior art: SMIL, ActionMAE. |
| I14 | Knapsack encoder routing under a FLOP budget | **KILLED (jury)** | Outside all five gaps, disconnected from the demonstrated error population, and heavily occupied by MSDNet / BlockDrop / SkipNet / early-exit. |
| I13 | HateMM annotation-noise ceiling (fresh census) | **KILLED as a standalone** | A train-only audit cannot establish a test-set ceiling, and 120 videos give class-specific CIs too wide for the leaderboard claim. Retain a smaller audit as supporting evidence for G-D. |
| I11 | Noise-evicted kNN memory | **DEMOTED to an ablation** | Wilson's Edited Nearest Neighbour and robust-kNN cover the mechanism; evicting classifier-disputed entries is actively dangerous in minority-value hate detection (arXiv:2606.28772). |
| I2 | Within-video normality deviation | **DEMOTED** | Cheap and informative but a negative diagnostic only; as a *method* it is standard video-anomaly detection. |
| I3 | Two-sided hard evidence margin | **DEMOTED (4th, still live)** | Mechanically the most distinctive OCR-free idea, but weak-supervision identifiability is unresolved, and MARS (arXiv:2601.15115) / RAMF (arXiv:2512.02743) already occupy the two-sided *framing* in this domain as prompt-level VLM reasoning. P1/P2's nulls also weaken every temporal branch. |
| I9 | Contested-label routing / selective risk | **DEMOTED** | Operationally sensible and G-D is method-empty, but SelectiveNet / Deep Abstaining Classifier / confident learning cover the mechanism, and the route will likely capture *hard* rather than *mislabelled* items — improving the clean-subset score by rejecting difficulty. |
| I10 | Selection-margin certified abstention | **DEMOTED** | top1−top2 is not a certificate: multiple legitimate evidence intervals produce a low margin and one spurious spike produces a high one, so it may measure evidence multiplicity rather than label noise. Must not be called "certified" without a proof. |
| I5 | Unsaid-text OCR−ASR residual retrieval | **RECOMMENDED 3rd** | Not piloted; gated on a CPU-minutes redundancy test. Phase-3: CONFIRMED-NOVEL narrowly (5/10) — carries a reparameterization kill criterion. |
| I7 | Budgeted discrete modality acquisition | **MERGED into Idea 1** | Its key risk was de-risked by pilot P3(a). |
| I12 | CCEF | **RECOMMENDED 2nd** | Part (a) genuinely unoccupied; part (b) anticipated by NExT-GQA/EG-VQA. Phase-3: PARTIAL (5/10), per-video κ needs statistical repair. |

> **[勘误 2026-08-09]** I4 行的杀因表述("selection hit 0.544 vs chance 0.762,ratio 0.71;
> classification −0.59 pt")需修正:0.544 低于 chance 系 argmax 并列破序 artifact(随机破并列后
> 0.768 ≈ chance 0.762),−0.59 pt 为 null(CI [−2.16,+0.99])。**KILLED by pilot 的判决维持**
> (ratio 1.008 < 1.3× 杀线),杀因应读作"选择器只到 chance,且邻域纯度统计量无 within-video
> 定位信号(AUROC 0.511)",而非"低于 chance / 实证有害"。详见 `idea-stage/P2_FORENSIC_MEMO.md`。

No candidate was eliminated on the executor's own taste — the objective feasibility gate dropped
nothing (all 14 fit one RTX 5090 with cached features), and every elimination above traces to the
cross-model jury or to a frozen pilot threshold.

---

## 5. Suggested execution order

1. **Build the OCR cache** (1–3 GPU-h, one-off). It is the shared prerequisite of Ideas 1 and 3 and
   the single largest unknown in the plan. Before spending it, run the Idea-3 gate (CPU-minutes) —
   if retrieval turns out to be redundant with OCR, the cache buys less than expected and Idea 3
   becomes a redundancy paper rather than a method paper.
2. **Idea 2 (CCEF + span-coverage)** in parallel — it is already ~70% measured, costs no GPU, and
   is the instrument every later selection claim will be scored on. Writing it first also forces us
   to fix the evaluation protocol before we have a horse in the race.
3. **Idea 1 (Pay-for-Evidence)** as the headline method, once OCR exists. Pre-register before the
   first real run; the head-to-head ablation against a cheap heuristic OCR-presence detector must be
   in the pre-registration, not added later.
4. **Idea 3** as the second method paper or as Idea 1's retrieval arm, depending on how its gate
   resolves.
5. Keep **I3** (two-sided margin) as the fallback if OCR extraction proves unexpectedly poor on
   BitChute-quality video.

## 6. Next steps

- [ ] Run the Idea-3 redundancy gate (CPU-minutes, frozen thresholds already written).
- [ ] Extract the HateMM OCR cache; record cost and per-frame yield.
- [ ] Write the CCEF definition + the span-coverage result; verify it reproduces on HateClipSeg
      (which ships segment-level annotations) before claiming it generalizes beyond HateMM.
- [ ] Pre-register Idea 1 (single submission, decision rules frozen, zero test contact) and queue it.
- [ ] Fix the Gate-0 harness D-4 defect before any reuse of that infrastructure.
- [ ] The confirmation sets (HateMM-val, HateClipSeg-val) have already been consumed once by Gate-0
      Run 2 — budget for that when designing Idea 1's confirmation stage.
- [ ] **Phase-3 follow-ups**: obtain LEAF (ACL Findings 2026) and DeHate (ACM MM 2025) and read them
      before writing any related-work paragraph; check whether DeHate's contributing-modality labels
      can be used directly as an external validation set for our evidence-type gate; add the
      counterfactual-utility gate variant and the learned-concatenation-metric control to the two
      pre-registrations.

## 7. Deep novelty check (Phase 3, 2026-08-08) — verdicts and citation obligations

**Method.** Multi-source pass: 13 WebSearch formulations, 16 arXiv Atom-API queries (14 keyword +
2 `id_list` verification batches), 2 OpenAlex queries, targeted WebFetch of arXiv abs/html and ACL
Anthology pages; Semantic Scholar was rate-limited (HTTP 429) and unused. Every arXiv ID quoted in
this report was resolved through the arXiv API and its title/date confirmed; LEAF, DeHate and MoRE
were confirmed through their publisher pages. Cross-model verification: Codex `gpt-5.6-sol` at
`model_reasoning_effort: xhigh`, trace in `.aris/traces/novelty-check/2026-08-08_run01/`.
Concurrency window scanned: 2025-08 → 2026-08.

| Idea | Verdict | Score | Recommendation | Which axis collided |
|---|---|---|---|---|
| **1 Pay-for-Evidence** | **PARTIAL** | 7/10 | PROCEED WITH CAUTION | axis (b) — "MLLM distilled into an inference-time-MLLM-free hateful-video model" is occupied **in-domain** by LEAF (ACL Findings 2026); the evidence-type taxonomy is occupied by DeHate (ACM MM 2025). Axes (a) and (c) survive. |
| **2 CCEF** | **PARTIAL** | 5/10 | PROCEED WITH CAUTION, as Idea 1's companion protocol | part (b), the grounded/faithful score, is anticipated cross-domain by NExT-GQA Acc@GQA and EG-VQA EG-F1. Part (a), per-video chance correction, survives. |
| **3 Unsaid-text residual** | **CONFIRMED-NOVEL** (narrowly) | 5/10 | PROCEED WITH CAUTION, with a frozen reparameterization kill | no collision found in any domain. The risk is not prior art, it is that r = o − a is a fixed linear projection of [o ‖ a]. |

**Nothing was KILLED.** No in-domain paper implements budgeted modality *acquisition*, and no paper
in any domain implements the OCR−ASR residual retrieval key.

### Papers we must cite and explicitly differentiate from

Ordered by how much damage they do if a reviewer finds them first and we did not.

| Paper | Handle | The sentence we owe the reader |
|---|---|---|
| **LEAF**, ACL Findings 2026 `2026.findings-acl.604` | Lang et al., SG-CoT stage-wise distillation, HateMM + MHClip | "LEAF already distils an LMM into an inference-time-LMM-free hateful-video model; it distils explanations for interpretability, whereas our training-only supervision targets a decision about **what to compute**." |
| **DeHate**, ACM MM 2025 `10.1145/3746027.3758272` | 6,689 videos, human segment-level *contributing modality* labels | "Modality-attribution labels for hateful video already exist; we do not claim the annotation, we claim its use as an acquisition signal — and we validate our MLLM labels against DeHate's human ones." |
| **Post-hoc Selective Modality Escalation**, arXiv:2607.05438 | cost-aware modality escalation in multimodal RAG | "Relevance ≠ utility, as that work shows for QA; we therefore ablate a relevance-trained gate against a counterfactual-utility-trained gate at segment granularity in untrimmed video, where no query exists to condition on." |
| **MultiHateLoc**, arXiv:2512.10408 | modality-aware top-K MIL over hate-video segments | "MultiHateLoc hard-selects among features it has already extracted; we decide where to spend extraction cost. It is our primary in-domain selection baseline." |
| **MM-HSD**, arXiv:2508.20546 | HateMM SOTA 0.874, always-on OCR as CMA query | "MM-HSD establishes that on-screen text is worth having; it pays for it on every frame. It is our all-30-segment operating point." |
| **ModaRoute**, arXiv:2507.13374 (ICCVW 2025) | GPT-4.1 routes queries across ASR/OCR/visual indices, −41% compute | "Per-query modality routing with an inference-time LLM over pre-built indices; ours is per-segment, LLM-free at inference, and pays at extraction time. Also our external evidence that 34% of clips carry scene text absent from ASR." |
| **CLaMR**, arXiv:2506.06144 | late-interaction retrieval over frames/speech/on-screen-text/metadata | "CLaMR keeps OCR and speech as separate indexed streams with a modality-aware loss; we form their residual and retrieve with it." (Primary threat to Idea 3.) |
| **MoRE**, WWW 2025 `10.1145/3696410.3714560` | the only retrieval-augmented hateful-video method | "MoRE retrieves with whole-video keys; the question of what the key should be is what we study." |
| **LELA**, arXiv:2602.09637 | training-free, 5 modalities incl. OCR, LLM prompted per frame | "The maximal-cost end of the design space: every modality on every frame plus an LLM. We report our Pareto curve against it." |
| **VOILA** arXiv:2602.03007 · **SAFE-Cascade** arXiv:2606.19646 · **DyMo** arXiv:2601.22853 · **Filter-And-Refine** arXiv:2507.17204 | cost-aware fidelity / routing / modality selection / moderation cascade | One related-work sentence each: all decide **per query, per sample, or per video**, never per temporal segment, and none price a modality *extractor*. |
| **Q-Frame** arXiv:2506.22139 · **Q-Gate** arXiv:2604.17422 · **VisionSelector** arXiv:2510.16598 | frame/token selection under budget | "The frame-selection line spends resolution or token budget within one modality and is query-conditioned; we spend the cost of invoking a different modality, with no query." |
| **Differentiable Knapsack/Top-k**, arXiv:2601.21775 | the operator | "We give this operator its first vision-language application — and we drop the word *knapsack* if our per-segment costs turn out uniform." |
| **AFA line**: survey arXiv:2502.11067, CAMA arXiv:2505.16791, BRiG-AFA arXiv:2608.02305, AFABench arXiv:2508.14734 | classic active feature acquisition | "The AFA literature is tabular and clinical; to our knowledge it has never been applied to content moderation or to video. We inherit its cost-vs-value framing and its evaluation discipline." |
| **NExT-GQA** Acc@GQA (CVPR 2024) · **EG-VQA** EG-F1 | grounded-answer metrics | "The joint correct-and-grounded requirement is theirs; our contribution is per-video chance correction, which they do not perform." |
| **arXiv:2508.04900** (temporal label noise) · **HateClipSeg** arXiv:2508.01712 · **SCANNER** arXiv:2602.00132 · **IARE** arXiv:2606.11953 · **Tzelepi & Mezaris** arXiv:2504.09914 | context | Note in particular: Tzelepi & Mezaris runs its LMM at **inference** as well as training — do not cite it as train-time-only distillation. |

### The three sentences we are no longer allowed to write

1. "We are the first to distil an MLLM into an MLLM-free hateful-video detector." (LEAF.)
2. "We introduce evidence-type / contributing-modality supervision for hateful video." (DeHate, IARE.)
3. "OCR−ASR residual retrieval has never appeared anywhere." (Write "to our knowledge", and cite the
   search — embedding-difference features are generic.)

## 8. Integrity notes

- **Zero test contact.** No pilot opened `dev_seen` or `test`. All pilots read only the HateMM-train
  caches, the frozen seed-20260807 fold files, the Gate-C audit rows, and `hate_spans.json`.
- **Decision rules frozen first.** `idea-stage/PILOT_FREEZE.md` was written after the cross-model
  jury returned its threshold corrections and before `idea-stage/pilots.py` was executed.
- **Sealed-number disclosure.** While retrieving the A0 head configuration, `folds/fold_0/
  selected_hparams.json` was opened, exposing fold-0 inner-OOF hyperparameter-selection macro-F1
  values for arms A0–A4 — numbers adjacent to the permanently sealed Gate-A family. They are not
  used, quoted, or relied upon anywhere in this report or in any pilot; the pilots define their own
  views and their own head (`sklearn` logistic regression, deliberately independent of the Gate-0
  arm hyperparameters) and measure their own numbers. The pilot designs were frozen before that
  file was opened.
- **Post-hoc analyses are labelled as such** and revise no frozen verdict.
- **Reviewer independence.** Generation used five Claude lens shards plus one Codex seed; the
  accept/reject verdict and the ranking came from Codex `gpt-5.6-sol` (xhigh), not from the
  executor. Novelty verification was a separate multi-source search pass.
- **Phase-3 novelty check.** Ran after the pilots and after the ranking, on a separate Codex thread
  (`019fdc7a-62ea-7140-a634-82e2555f2ccd`) that saw the idea descriptions and the candidate-prior-art
  list but no pilot verdicts other than those quoted as motivating facts. Codex's sandbox again could
  not read scratchpad paths, so the dossier was pasted inline; the pasted text is byte-identical to
  `.aris/traces/novelty-check/2026-08-08_run01/01_prompt_dossier.md`. No paper is cited in §7 that was
  not resolved through the arXiv API or its publisher page — `verify_papers.py` is not installed in
  this project, so Policy D1's degraded-output fallback applied and verification was done by direct
  fetch instead. No search touched any test split; the check is literature-only.

---

## 9. Phase-4 external hostile review (2026-08-08) — verdict, claims matrix, pre-registration

**Reviewer**: Codex `gpt-5.6-sol`, `model_reasoning_effort: ultra` (deep-audit tier), NeurIPS/ICML
area-chair posture, two rounds. Thread `019fdc94-f8ca-7360-a75f-f837489219b7`. Full transcript:
`.aris/traces/research-review/2026-08-08_run02/` (brief `RESEARCH_REVIEW_REQUEST.md`,
responses `002-*.response.md`, `003-*.response.md`).

**Fallback disclosure**: Codex's sandbox again could not read local paths
(`bwrap: loopback: Failed RTM_NEWADDR`, third occurrence in this project). Round 1 returned a
refusal-to-review rather than a fabricated verdict (traced as `001-*`, status `error`); the brief and
four appendices (PILOT_FREEZE, pilot_results.json, the gate's metric code, §2 Idea 3) were then
pasted inline. **The reviewer therefore verified nothing against the filesystem** — it audited the
numbers as pasted. Its arithmetic re-derivations (Wilson intervals, McNemar, the binomial tie
analysis, the FP-count reconstruction) are its own and were not checked by the executor.

### 9.0 The proposal reviewed

"Evasion-aware retrieval-guided hateful video detection": (1) add the OCR channel to the existing
frozen-encoder + light-head + RGCL pipeline; (2) mechanism = **unsaid-text residual**
`r = CLIP(OCR) − CLIP(ASR)` as a **complementary** second retrieval key beside the fused primary key,
with kNN vote fusion and joint hard-negative mining on both keys; subline = selective escalation /
abstention; demoted-to-ablation = budgeted segment-level OCR acquisition.

### 9.1 Verdict

| | |
|---|---|
| **Score (proposal as submitted)** | **2/10 — strong reject** |
| **Score (acquisition reframe, §9.5)** | **5/10 — ACM MM / ACL Findings tier, not NeurIPS/ICML** |
| **Mock review of the *best realistic* residual outcome** | **5/10 weak reject, confidence 4/5** |
| **Recommendation** | **Kill the residual-led campaign.** Fund only the cheap pre-registered falsification suite (§9.6) because head fits cost ~52 s. |

### 9.2 Claim triad — attack results

**(a) Mechanism novelty → ENGINEERING + PACKAGING, not a mechanism.**
`r = o − a = [I, −I][o; a]`, so residual distance is a *fixed, rank-constrained Mahalanobis metric*
over `[OCR ‖ ASR]`; a learned linear projection reproduces it exactly and a parameter-matched learned
metric can only be more expressive. The surrounding parts are all occupied: adding OCR (MM-HSD),
OCR/ASR as separate indexed streams (CLaMR), video retrieval (MoRE), multi-index vote fusion + hard
negatives (standard). Residual definition problems: "per video (or per segment)" tests two different
hypotheses and is not a specification; at 2161 mean OCR chars under a 77-token cap, whole-video
subtraction "largely compares two unrelated truncated prefixes"; empty ASR, OCR errors, banners,
repeated subtitles and timing misalignment can dominate; pre- vs post-normalization subtraction gives
different geometries and near-equal vectors give unstable near-zero residuals.
**"Evasion-aware" is unsupported and must be dropped** — unspoken on-screen text does not establish
creator intent; subtitles, watermarks, handles and stylistic overlays are observationally equivalent.
Honest term: **"OCR–ASR discordance" / "novel on-screen text"**.

**(b) Error-population repair → the chain is OPEN AT EVERY LINK.**
Direct answers: *does anything measure the residual?* **No.** *Does anything measure a classification
flip?* **No.** *Does anything measure marginal gain after naive OCR fusion?* **No.** Specific breaks:

1. Census is hypothesis-generating: OR 2.29 ≈ 95% CI **[0.94, 5.57]**, p = 0.083; "the only enriched
   property found" implies outcome selection ⇒ winner's-curse-prone.
2. The residual's actual target (`OCR required AND speech not required`) was never tested: 22/73 vs
   5/30 has OR ≈ 2.16, CI ≈ **[0.73, 6.37]**, no reported test.
3. **"Speech not required" ≠ "absent from ASR."** The audit label does not operationalize OCR–ASR
   semantic novelty in either direction.
4. **22 vs 28 is a stop-work data-lineage failure.** 30.1% × 73 = 22; the gate script found 28
   (= 38.4%). Not rounding. Until video-IDs and provenance reconcile, **every R2 subgroup number is
   untrustworthy.**
5. R1 never computes `o − a` at all — it compares `CLIP(ASR)` vs `CLIP(OCR)`.
6. **R1 measures diversity, not complementarity.** ov@10 = 0.048 is 0.48 shared neighbours vs 0.168
   expected at random — i.e. the two keys overlap **2.86× more than random** while still returning
   mostly different lists. **A random or corrupted key passes the frozen ≤0.25 "COMPLEMENTARY" rule.**
   The executor's own concern #1 is confirmed: the gate is unfalsifiable by construction.
7. **R2 is an oracle diagnostic, not recovery.** All target queries are known positives; `recovery`
   asks whether their *gold-labelled* neighbours vote positive, with no false-positive cost — an
   always-positive vote scores 1.0. Decisively: these 28 are baseline **false negatives despite
   15/28 already getting a positive transcript-neighbour vote**, which *directly demonstrates* that a
   favourable vote is not a classifier repair.
8. The vote rule is positively biased: at k = 20, `>= 0.5` calls a 10–10 tie hateful. At prevalence
   0.4005 a random neighbourhood yields ≥10 positives **24.6%** of the time (strict majority 12.8%),
   so the frozen recovery GO bar of 0.30 sits barely above the tie-biased random baseline.
9. The OCR-vs-transcript difference is **two videos** (15/28 → 17/28). Wilson ≈ [.36,.71] vs
   [.42,.76]; under the pairing *most* favourable to OCR, exact McNemar gives **p = 0.50**. Without
   the paired table, OCR-only wins could be anywhere from 2 to 13, and no implementable fusion gets
   the oracle union for free.
10. The purity analysis is class-confounded (mixes FN/TP/FP strata with different label
    compositions); 0.5198 is random-pair label *agreement*, not a classification baseline — always
    retrieving negatives could score ≈0.60, **above Key_O's 0.569**.
11. Equal 77-token caps are **not** "like-for-like": OCR and ASR differ in length, ordering and
    repetition, so the keys may encode the first banner/subtitle. The 26.6% of videos failing the
    OCR population filter are unevaluated.
12. ModaRoute's 34% establishes scene-text prevalence, not hateful relevance or error repair.

**(c) Performance realism → the premise is rejected and the ceiling does not reach SOTA.**
The transcript key does *not* "recover 53.6%" at classifier level; it gives a positive neighbourhood
vote to 15/28 videos that **remain false negatives**. Planning estimate for the residual beyond a
strong OCR fusion: **central +0.2 to +0.3 macro-F1 points; plausible range negative to +0.5;
negligible chance of closing the SOTA gap.** Arithmetic (assuming the 73 audited FNs are the full FN
set, ≈54–55 FPs at macro-F1 ≈0.82): 2 perfect FN fixes ≈ **+0.30 pt**; perfectly fixing all 22 target
FNs ≈ **+3.3 pt → 0.853**; all 28 ≈ **+4.2 pt → 0.862**; reaching 0.874 needs ≈**36–37** net
FN-equivalent corrections with zero new FPs. **Even impossible perfect target repair does not close
the 5.4-point gap.** Separately, the gap itself may be invalid — our 0.82 is train-OOF, MM-HSD's
0.874 is a published number under a different protocol; **a same-split reproduction is required
before the gap is quoted anywhere.** Any large gain is far more plausibly attributable to adding OCR,
better long-text handling, or fusion changes than to the residual.

### 9.3 The cheapest-alternative ruling (naive OCR fusion) — the decisive question

**Ruling: if naive OCR fusion captures all or nearly all of the gain, the residual mechanism is dead
on arrival.** What remains would be "OCR improves an inherited hateful-video pipeline" — useful
engineering, already occupied by MM-HSD/CLaMR, insufficient for a top venue.
**R1's overlap@10 = 0.048 does not license the complementarity inference**, because a random key
scores the same. Useful complementarity requires *incremental classifier performance conditional on
the fused key*, plus superiority to a random-diversity control.

**The one narrow regime where the residual could still win** (pre-register exactly this): OCR and ASR
temporally aligned and accurately transcribed; OCR contains both speech-duplicating subtitles *and*
an additional consequential hateful proposition; that proposition is semantically absent from ASR;
it is not a watermark/banner/template/OCR-error/empty-ASR artifact; and fused retrieval is
demonstrably diluted by shared topic or visual content. Probe it with controlled overlays holding
video+speech fixed: (1) exact spoken subtitles; (2) same subtitles + an unspoken hateful claim;
(3) matched-length benign unspoken text; (4) OCR-corruption/watermark controls. **The residual must
move retrieval and predictions selectively for condition 2, not for any mismatched text.**

### 9.4 Results-to-claims matrix (frozen before the factorial runs)

| Factorial outcome | Allowed claim | Forbidden claim | Next action |
|---|---|---|---|
| Naive OCR fusion captures **≥80%** of total gain | OCR improves this frozen retrieval pipeline; residual is at most a minor diagnostic | residual is the main mechanism; residual explains the gain; "evasion-aware repair" | Kill residual headline; continue only as acquisition or an OCR systems paper |
| Learned `[OCR‖ASR]` metric **matches/beats** residual | residual is a simple constrained metric / low-parameter prior | novel retrieval mechanism; semantic subtraction has special status | Enforce the frozen reparameterization kill; use the learned metric |
| Residual beats **every** matched non-residual arm by **≥1.0 pt**, paired 95% LB > 0 | a fixed OCR–ASR discrepancy bias gives incremental predictive value **in this setting** | semantic subtraction; creator evasion; generality without audits + replication | Proceed to overlay audit, grouped-dedup eval, MHC replication; headline stays conditional |
| **All** OCR arms < +0.5 pt over no-OCR | this OCR representation has no material value under this backbone | OCR repairs the error population; acquisition has accuracy headroom | Kill residual **and** acquisition under this representation; write only as a negative result |
| Gains land **outside** the frozen OCR-required/no-speech stratum | generic OCR / ensemble gains occur in another population | the proposed error population was repaired; the causal chain was validated | Keep as exploratory; new hypothesis, independent cohort; **do not rewrite the current story** |
| **Random/shuffled** second key matches residual | extra neighbour budget / ensemble diversity helps | residual semantics cause the gain | Kill residual; attribute to ensembling |
| OCR-only index beats naive fusion, residual does not beat OCR-only | modality-separated OCR retrieval is useful | subtraction is necessary | Pursue only as an OCR-retrieval systems result vs CLaMR |

The ≥80% rule is scored only when the residual is best:
`ρ = [F1(naive OCR) − F1(no OCR)] / [F1(residual) − F1(no OCR)]`; if the denominator < 0.5 pt, the
flat-result kill applies instead.

### 9.5 The reframe that scores higher — 5/10

Dropping the residual-as-mechanism claim and re-centring on a **counterfactual-OCR-value acquisition
gate** (predict `Δloss_j`, select which segments to pay OCR on under budget, deliver an
accuracy-vs-cost Pareto against always-OCR / never-OCR / random-k / uniform-k / salience-k /
per-video gate at matched cost / oracle-k / ModaRoute-like router) scores **5/10 — a solid ACM MM or
ACL Findings paper if the Pareto result is strong and externally replicates; not yet NeurIPS/ICML.**
Two mandatory design corrections:
- the target must be **set-conditioned**, `Δ_j(S) = L(f(S)) − L(f(S ∪ {j}))`, because independently
  estimated `Δ_j(∅)` values are non-additive when segments repeat the same text;
- **OCR–ASR discordance cannot be an inference-time gate feature** if computing it requires running
  the OCR being rationed — it is admissible only as training supervision/diagnostic, or with its
  cheaper proxy's cost explicitly charged.

**Most likely failure mode**: the learned *segment* ranking fails to beat uniform-k or a per-video
gate at matched real cost, because pre-OCR features poorly predict noisy, redundant segment-level OCR
value. P3 established coarse *video-level* detectability; it says nothing about segment-level
value-of-information, and P2 warns that segment selection is hard here. For top-venue viability the
policy would have to generalize beyond this pipeline/task, optimize **measured latency/energy** rather
than segment count, and hold up on both HateMM and MHC.

> **[勘误 2026-08-09]** 评审引用的 "P2 warns that segment selection is hard here" 仍然成立,但其
> 依据要换:0.544 低于 chance 系 argmax 并列破序 artifact(随机破并列后 0.768 ≈ chance 0.762),
> −0.59 pt 为 null(CI [−2.16,+0.99]),都不是"段级选择有害"的证据。真正的警告是 forensic 测到的
> 机制性空档 —— 冻结 CLIP 视觉段键的邻域纯度对 within-video 证据位置无信号(AUROC 0.511
> [0.488,0.533]),而它对视频级标签有强信号(0.782)。NO-GO 维持(ratio 1.008 < 1.3×)。详见
> `idea-stage/P2_FORENSIC_MEMO.md`。

### 9.6 Pre-registration for the authorized falsification suite (paste-ready)

**Data**: HateMM-train only, no val/test contact. **Splits**: one newly frozen,
**duplicate/creator/template-grouped**, label-stratified 5-fold outer split, seed `20260807`.
**Seeds**: 10, `20260808`–`20260817`. **Model selection**: 4 inner folds inside each outer-train fold;
thresholds, projections and stopping use **inner-OOF only** (the pilots' train-prediction threshold
tuning is a defect to fix). **Retrieval**: k = 20 per index; dual-index arms 20+20; matched
single-index control top-40; exclude the query parent *and its duplicate/template group*; hard
negatives matched at 40.

**Arms** — A0 no-OCR baseline · A1 OCR as classifier feature only · A2 naive OCR fusion, one fused
index · A3 A2 + OCR-only 2nd index · A4 A2 + ASR-only 2nd index · **A5 A2 + residual 2nd index** ·
A6 A2 + parameter-matched linear/Mahalanobis `[OCR‖ASR]` index · A7 A2 + parameter-matched MLP
projection · A8 A2 + seed-frozen **shuffled** residual index · A9 A2 with fused top-40 (neighbour-budget
control). Every second key is used identically for voting and mining; the 2×2 mining/inference
decomposition is permitted **only after A5 passes**.

**Primary estimand**: `Δ* = mean_F1(A5) − max over B of mean_F1(b)`, `B = {A0,A1,A2,A3,A4,A6,A7,A8,A9}`,
means over 10 seeds of OOF macro-F1, in absolute points.

**Inference**: 10,000 paired two-level bootstrap replicates resampling **duplicate/creator groups**
within outer folds and seeds, preserving arm pairing; **recompute the strongest non-residual
comparator inside every replicate**; two-sided 95% percentile intervals; identical bootstrap indices
for global / target / complement analyses.

**GO to the full campaign only if ALL hold** — (1) `Δ* ≥ +1.00` pt; (2) paired 95% LB of `Δ*` > 0;
(3) target-stratum recall vs the globally strongest non-residual comparator improves **≥ +10 pp**
with paired 95% LB > 0; (4) **outside** the target stratum, macro-F1 difference ≥ **−0.50** pt with
one-sided 95% LB > −0.50; (5) naive OCR captures **< 80%** of A5's improvement over A0; (6) neither
A6 nor A8 is equivalent to A5 within **±0.50** pt (equivalence = paired 90% CI wholly inside
[−0.50, +0.50]).
**KILL otherwise — there is no AMBIGUOUS continuation category.** Kill immediately if the 95% upper
bound of A5 vs A0 ≤ +0.50 pt, or learned concatenation (A6) matches/beats A5, or the random key (A8)
matches A5.

**Blinding**: before the OCR-required/no-speech stratum membership is unblinded, freeze the cohort
reconciliation procedure **and its hash**, duplicate groups and folds, all arms and representation
choices, seeds / k / vote-and-mining budgets / missing-text handling, hyperparameters and threshold
procedure, estimands, bootstrap code, margins and GO/KILL rules, and all exclusion + failure
handling. A **separate steward** reconciles 22 vs 28 and seals the membership hash; analysts unblind
only after all global OOF predictions, code hashes and the strongest comparator are immutable.

### 9.7 Minimum viable improvements, ordered by acceptance lift per GPU-hour

1. **Data-integrity and provenance repair — 0 GPU-h, prerequisite not improvement.** Reconcile 22 vs
   28 at video-ID level; pin the exact prediction checkpoint and annotation version; report duplicate
   IDs, class/error-stratified counts, tie counts, and the **paired** OCR/ASR vote table; group folds
   by repost/template/creator.
2. **Build a semantically valid residual and audit it — ~0 GPU-h + human time.** Temporally align the
   K=30 OCR/ASR windows, dedupe repeated OCR, **chunk beyond 77 tokens**, mask empty modalities,
   freeze normalization/pooling. Then blind-code high/mid/low-residual segments for: genuine unspoken
   hate · benign novel text · subtitles · OCR/ASR error · banner/watermark · timing mismatch.
   **Kill if the high-residual tail is not substantially enriched for genuine unsaid evidence.**
3. **Run the matched factorial of §9.6** — the highest-value computational experiment; do it before
   any joint mining or full campaign.
4. **Fix the statistics** — inner-OOF thresholds, repeated group-aware CV, paired
   permutation/bootstrap that **includes refitting uncertainty**, fold-specific nulls, per-class
   retrieval metrics, one-neighbour-per-parent controls.
5. **Controlled-overlay causal probe** (§9.3). If the residual responds equally to benign mismatch,
   OCR corruption or empty ASR, **drop "evasion" and the mechanism claim.**
6. **Strong OCR baselines + train/inference isolation** — an MM-HSD-style OCR-query baseline and
   modality-separated late interaction; 2×2 ablation of residual-index on/off × residual-mining
   on/off, or extra mining/ensemble capacity gets misattributed.
7. **Only after survival**: sealed HateMM and MHC EN/ZH, preserving the single test submission; MHC
   needs a frozen multilingual-text control (a 77-token English-oriented CLIP tower is a weak basis
   for Chinese OCR/ASR).
8. **Elevate the contribution** — even a successful raw subtraction is a multimedia-tier result. Top
   tier likely needs a principled token-aligned semantic-novelty objective and/or a released
   benchmark annotating spoken-vs-written novelty, artifact causes, and counterfactual overlays.

### 9.8 The subline — CUT

**Verdict: cut selective escalation / abstention from this paper.** Classifier–kNN disagreement and
boundary abstention are standard; including them adds a second evaluation burden (risk–coverage,
AURC, calibration, review-cost model, coverage-matched SelectiveNet/DAC baselines) without
strengthening the core claim. At most a small operational appendix *after* the core mechanism works.
**It does not automatically deserve a separate paper either** — that would require a real human-review
cost model or a new selective-risk method.

### 9.9 Free-fire findings the executor must absorb

- **Post-gate narrative drift.** The gated idea was a hard-selected *segment* residual as the *sole*
  key. P2 undermined segment selection; the proposal then became fused-key + parallel residual index
  + vote fusion + joint mining. That is a **materially new hypothesis**, not an adaptation of the one
  the gate authorized.
- **Asymmetric reading of our own evidence.** P1 is AMBIGUOUS, not proof of null headroom (its CI
  still permits +1.76 pt). P2's official-span GO rule was **arithmetically impossible** on this data
  (2 × 0.762 = 1.524 > 1). P3's routing failed in 5/5 folds. R2 said GO-**REDUNDANT**. Yet R1 —
  the one gate a random key would pass — is what gets cited to justify a complementary architecture.
  **Frozen rules prevent p-hacking; they do not make badly calibrated rules informative.**

  > **[勘误 2026-08-09]** 本段两处 P2 引用需注:(i)"P2 undermined segment selection" 的原始依据
  > 0.544 低于 chance 系 argmax 并列破序 artifact(随机破并列后 0.768 ≈ chance 0.762),−0.59 pt
  > 为 null(CI [−2.16,+0.99]);段级选择被削弱的正确依据是纯度统计量的 within-video AUROC 0.511。
  > (ii)"P2's official-span GO rule was arithmetically impossible" 的指摘不受影响,仍然成立
  > (2 × 0.762 = 1.524 > 1)。NO-GO 维持(ratio 1.008 < 1.3×)。详见
  > `idea-stage/P2_FORENSIC_MEMO.md`。
- Further pilot defects: threshold tuning on training predictions rather than inner-OOF; a single
  random-window draw in P1; fixed-prediction bootstraps that omit fitting uncertainty; a 2816-d P3
  probe on 133 selectively sampled videos.
- **Direction**: kill the residual-led paper now; the acquisition reframe is the better prospective
  mainline but **has not earned a full campaign either** — gate it on an accuracy–latency/energy
  Pareto against always-OCR, never-OCR, random acquisition, uncertainty routing and a ModaRoute-like
  router, on HateMM **and** MHC; otherwise kill that too.
- **Do not abandon the direction merely to chase MHC numbers.** Use MHC EN/ZH as external stress
  tests. A leaderboard gain without a defensible mechanism does not solve the contribution problem.

### 9.10 Consequent edits owed to the rest of this report

- §2 Idea 3's "22/73" and the gate's `n_census_fn_ocr_required_no_speech = 28` must be reconciled at
  video-ID level before either number is used again. **Both are currently suspect.**
- §2 Idea 3's gate summary must record that **R1's COMPLEMENTARY band is passed by a random key** and
  that **R2's `recovery` is a gold-label neighbourhood vote, not a classification flip**.
- The phrase "evasion-aware" is retired project-wide pending the overlay probe; use "OCR–ASR
  discordance" / "novel on-screen text".
- §5's execution order is superseded: the OCR cache is already built, so the order is
  (1) §9.7 item 1 data-integrity repair → (2) §9.7 item 2 residual construction + blind audit →
  (3) the §9.6 factorial → GO/KILL.
- The MM-HSD 0.874 comparison must not be quoted until a same-split reproduction exists.
