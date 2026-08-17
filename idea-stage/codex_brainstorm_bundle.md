# Context brief — hateful video detection, idea generation (ARIS Phase 2)

Project: `/home/jehc223/Retrieval-hate`. Adapting RGCL / RA-HMD (retrieval-guided contrastive,
originally hateful-meme) to **hateful VIDEO detection**. We need a NEW method with a novel
mechanism AND strong performance. Large changes to the existing method are allowed.

## 1. Assets we already own (this bounds what a cheap idea can be)

- **Frozen encoders**: CLIP ViT-L/14-336 (visual 1024-d, text 768-d) and Qwen2.5-VL-7B/32B
  embedding caches, all precomputed to `.pt` files. Encoders are FROZEN; only a ~5M-param
  MLP head trains (HateClipper-style element-wise fusion). Head training is seconds-to-minutes
  on CPU/GPU.
- **HateMM train split**: 744 videos (298 hateful / 446 not).
  - whole-video cache `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt`
    → `img_feats [744,1024]`, `text_feats [744,768]` (transcript+title), `labels [744]`.
  - **K=30 segment cache** `train_subclipK30_...pt` → `subclip_img_feats [22320,1024]`
    (744×30 uniform temporal segments, 4 frames each), `subclip_parent [22320]`.
    **Segment features are VISUAL-ONLY.** There is no segment-level text vector yet.
  - **Segment ASR exists as raw text**: `data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl`
    (word-level timestamps, K=30 aligned; also K=4 and K=60). Encoding it with the CLIP text
    tower is a few GPU-minutes → segment-level text features are cheap to create.
  - HateMM val (`dev_seen`, 107 videos) has the same K=30 caches. **Test is untouched and stays
    untouched.**
  - **Gold hateful spans** `data/gt/HateMM/hate_spans.json` (HateMM has timestamp annotations)
    — usable for mechanism validation, train split only.
- FAISS (CPU) retrieval infrastructure; a validated cross-dataset updatable kNN memory.
- Frozen 5-fold split of HateMM-train, seed 20260807, at
  `artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf/folds/fold_{0..4}/`
  with `train_ids.json` / `query_ids.json`, and A0 OOF predictions in `oof_predictions.jsonl`.
- Hardware: **one RTX 5090 (32 GB)**, 16 CPU, no SLURM. Head-level training is ~seconds.
- Claude API may read raw video frames (DUA exemption granted) → usable for **training-time**
  weak labelling / distillation, not for inference-time cost.

## 2. Gate-0 evidence (TERA campaign, closed 2026-08-07, verdict NO-GO-C)

Population: the **73 OOF false negatives** of the A0 baseline (whole-video mean + linear head)
on HateMM-train, census-audited (no sampling) by blinded Claude coders, κ=0.733.
Controls: 30 true positives + 30 false positives.

Registered verdict: `multi_segment_complementary` = **6/73 = 8.2%** vs a 15% bar → **multi-segment
compositional evidence is FALSIFIED as a lever.** Route closed.

Signposts from the same audit:
- `short_localized` 37/73, `cross_modal` 7/73, union{short, multi, cross} = 61/73 = **83.6%**
  (bootstrap 95% CI lower bound 0.753).
- `single_interval_sufficient = True` for **64/73 = 87.7%** of false negatives.
- `annotation_ambiguity_or_noise` 12/73 among FN; **15/30 of the FP controls are annotation
  noise** (i.e. half of the "false positives" are arguably correct predictions).
- Gate-A / oracle single-segment headroom numbers are **permanently sealed and unreadable**.
  Any "selection headroom" claim must be measured fresh by a new pilot.

## 3. NEW forensic measurements (computed today, from the Gate-C audit; FN vs TP contrast)

These were not in the Gate-0 verdict and they matter a great deal:

| property | FN (n=73) | TP (n=30) | Fisher OR | p |
|---|---|---|---|---|
| `short_localized` primary cause | 0.507 | 0.600 | 0.69 | 0.51 |
| span/duration ratio (median) | 0.145 | 0.100 | — | MWU p=0.75 (n.s.) |
| `single_interval_sufficient` | 0.877 | 0.967 | 0.25 | 0.27 |
| **`on_screen_text` required** | **0.534** | **0.333** | **2.29** | **0.083** |
| `visual` required | 0.589 | 0.433 | 1.87 | 0.19 |
| `speech`/`transcript` required | 0.685 | 0.833 | 0.43 | 0.15 |
| needs on-screen text AND no speech/transcript | **22/73 = 0.301** | 5/30 = 0.167 | — | — |

**Interpretation.** Temporal localization is NOT enriched in the failures — the videos the model
already gets right are just as short-and-localized (60% vs 51%), and their hateful spans are not
longer (median 0.100 vs 0.145). So "the video-level average dilutes the one hateful segment" does
**not** discriminate success from failure. What IS enriched in the failures is the **modality**:
failures need on-screen text and pixels; successes are carried by the transcript. Our pipeline has
**no OCR modality at all**, and its segment features are visual-only.
This is directionally strong but underpowered (30 TP controls); treat it as a hypothesis with a
free measurement, not as a settled fact.

## 4. Literature state (Phase 1 survey) — the red lines

**Occupied / do not re-do:**
- **MultiHateGNN (BMVC 2025)** — soft attention-weighted segment aggregation over a graph;
  HateMM 0.771 F1. **Any idea using soft attention weights over segments is a near-duplicate.**
  A new idea must be *hard/discrete/differentiable-discrete* or otherwise mechanically distinct.
- **MLLM-at-inference-time detection is saturated** (HVGuard, 7+ papers). Do not propose
  "prompt an MLLM to judge the video."
- **MM-HSD (ACM MM 2025)** — tetra-modal (transcript + audio + frames + **OCR on-screen text**),
  cross-modal attention with OCR as query. **HateMM macro-F1 0.874 = the fusion SOTA.**
  It has no retrieval, no contrastive, no temporal localization, English-only.
- **MoRE (WWW 2025)** — the only retrieval-augmented hateful-video method; retrieves **whole
  videos** with a frozen weighted-cosine retriever into per-modality experts; all-BCE, no
  contrastive; HateMM macro-F1 0.8235. Retrieval is n=1 in this field and is whole-video only.
- Our own validated headline novelty is **cross-dataset updatable kNN memory** (vs MoRE).

**Open gaps identified in Phase 1:**
- **G-A** hard / differentiable single-segment selection → video-level classification: EMPTY in
  hateful video.
- **G-B** selection-driven retrieval / segment-keyed RGCL: retrieval line is n=1 (MoRE) and
  retrieves whole videos; nobody keys retrieval on a selected segment.
- **G-C** distilling MLLM frame-level evidence into a lightweight selection head that is
  **MLLM-free at inference**: EMPTY.
- **G-D** annotation-noise-robust training / abstention for hateful video: diagnosed
  (Revealing Temporal Label Noise 2508.04900; Majority Vote Silences Minority Values 2606.28772)
  but **no method**.
- **G-F** a joint metric coupling selection quality × classification quality: free evaluation
  contribution, nobody has one.

**Mechanism reservoir (papers already ingested, to be *adapted* not copied):**
- Differentiable knapsack / top-k via smoothed dynamic programming (arXiv 2601.21775) — gives
  exact discrete top-k selection with usable gradients; entropy is the unique
  permutation-equivariant regularizer.
- HFS (2512.11534) — set-level differentiable frame selection (relevance/coverage/redundancy,
  Gumbel-Softmax) + student-teacher distillation from an MLLM teacher so pseudo-labels are not
  static.
- SESAD (2607.10298) — structured evidence selection for weakly-supervised video anomaly
  detection; dual-prototype geometric decision instead of a score head.
- Select Less Reason More / EARL (2510.15440) — evidence-purity RL reward for frame selection.
- Temporal Label Noise in hateful video (2508.04900) — trimming to gold spans changes decision
  boundaries and confidence.
- Adaptive View Retrieval (2607.19061) — retrieve-then-calibrate over a *view bank* for hidden
  hateful illusions, frozen CLIP, 93.2% balanced acc.
- ExPO-HM (2510.08630), ARCADE/H-VLI (2603.21298), M³ (2603.21686), MoRE, MM-HSD as above.

## 5. Hard constraints on any proposed idea

1. **Four red lines:** zero test-set contact; decision rules frozen before results are seen;
   blindness (no candidate metric computed during design/implementation); one single submission
   for the real run. Pilots use **HateMM-train only**, 5-fold OOF with the frozen seed-20260807
   folds, or a train-internal nested split. `val`/`test` are not touched.
2. **No cross-dataset co-training** (single-dataset training rule). A cross-dataset kNN *memory*
   consulted at inference time is allowed — that is an inference-time mechanism.
3. Claude API may read video frames — allowed for **training-time** supervision only.
4. Gate-A / oracle sealed numbers may not be used; measure selection headroom yourself.
5. Performance targets that matter: MultiHateClip EN/ZH accuracy ≥ 0.85 is still OPEN (current
   0.783–0.832); HateMM fusion SOTA is MM-HSD 0.874 macro-F1, MoRE 0.8235.
   Claims should be framed as "mechanism + specific error-population repair + efficiency",
   not a bare SOTA shout.
6. Pilot budget: at most 3 pilots, ≤2 GPU-h each, ≤8 GPU-h total. Most realistic pilots here are
   CPU/GPU-minutes because only a ~5M head trains.

## 6. Banned near-duplicates (our own failed ideas — do not re-propose)

- Multi-granularity / segment-level temporal retrieval with AUTO sub-clip FAISS + MIL drifting
  hard negatives — tested NEGATIVE (language sign-flips, no gain).
- K=4 pseudo-label segment scoring — dead.
- Soft attention over segments — occupied by MultiHateGNN.
- MLLM structured archive as a retrieval key / third stream — tested, no accuracy gain (its value
  was auditability).
- LoRA-SFT of the Qwen2.5-VL encoder — mixed, regresses EN.
- Multi-segment complementary/compositional temporal evidence aggregation — FALSIFIED by Gate-0
  (6/73).

---

# YOUR TASK (read everything above first)

You are a senior ML researcher brainstorming research ideas for a top-venue submission
(CVPR / ACL / EMNLP / WWW / ACM MM class). The research direction is:

> A **new method for hateful video detection**, motivated by the Gate-0 error-population evidence
> in §2–§3 above and aimed at the open gaps G-A / G-B / G-C / G-D / G-F in §4. The method may
> depart substantially from the existing RGCL-style pipeline. It must have a **novel mechanism**
> and target **strong performance**, not just an analysis.

Generate **8–12 concrete research ideas**. For each:

1. One-sentence summary.
2. Core hypothesis — what you expect to find and why, grounded in the §2/§3 numbers.
3. Minimum viable experiment — the cheapest thing that would give a positive OR negative signal,
   named against the **actual cache files** in §1, with an explicit go/no-go threshold.
4. Expected contribution type: empirical finding / new method / theoretical result / diagnostic.
5. Risk: LOW / MEDIUM / HIGH.
6. Estimated effort: days / weeks / months.
7. Which gap(s) from §4 it fills, and the single strongest reason a reviewer would say it is
   novel against MultiHateGNN / MM-HSD / MoRE.

Prioritise ideas that:
- are testable on ONE RTX 5090 with frozen features and a ~5M-parameter trainable head,
- produce a clear positive **or** negative result (both are publishable here),
- are NOT "apply X to Y" unless the application reveals something genuinely surprising,
- have a mechanism that is **hard-distinct from soft attention over segments** — that space is
  occupied by MultiHateGNN (BMVC'25) and a soft-weighting idea is dead on arrival,
- do not require inference-time MLLM calls (that space is saturated),
- do not repeat any banned near-duplicate in §6.

Be honest about the §3 finding: our own census says temporal localisation does **not** separate
our successes from our failures, while modality coverage (on-screen text / pixels) does. An idea
that assumes "single-segment selection will obviously help" is arguing against our own data and
must say how it survives that, or must instead be an idea that turns this finding into the
contribution.

At least two ideas must be **composite** (two or more gaps fused into one mechanism where the
fusion is the novelty). At least one idea per gap G-A, G-B, G-C, G-D.

Output format: numbered list, one block per idea with the seven fields above. Nothing else.
