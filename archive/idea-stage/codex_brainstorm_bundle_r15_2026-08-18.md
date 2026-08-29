# R15 hostile-review bundle — hateful video temporal localization, round 13

Purpose: you are the hostile external reviewer. Score a candidate slate, and answer one structural
question: **is any legal mechanism family left in this sub-direction, or is the goal unreachable
under the project's constraints?**

---

## 0. Standing constraints (hard, not negotiable)

1. **Method paper only.** The contribution must be a mechanism that raises a number. Benchmark
   papers, audit papers, metric papers, measurement papers, dataset papers are all permanently
   banned by user rule. An analysis section may support a method claim; it may not be the claim.
2. **No manual annotation, no new dataset construction.** Using an already-public annotated corpus
   is allowed (HateClipSeg is downloaded, split frozen).
3. **Incremental but real gains are acceptable** — there is no "must gain 5 points" bar. A +1 that
   replicates is worth having. But it must be a gain over a fair control, not over a strawman.
4. **Four hard experimental red lines:** zero test-set label contact during design; decision rules
   frozen before any candidate metric exists; blindness (the designer may not see candidate numbers
   before freezing); a single submission for the confirmatory run.
5. Hardware: one RTX 5090, no SLURM. CPU-level probes (≤ ~1 h) get one review round. This round's
   external-API budget is ¥15 and ¥0 has been spent.

---

## 1. The arena

**HateClipSeg** (ACM MM 2025, arXiv 2508.01712), local 395-video surviving subset (90.8% of 435,
non-random attrition, so **no SOTA claim is available** — method-vs-method on the identical frozen
subset only). Frozen split `p11_split.json` 237 train / 39 val / 119 test.

Corpus geometry, measured locally: median duration 239.1 s, median 27 gold segments per video,
median segment length 8.12 s, toxic coverage median 0.544, single contiguous toxic block 22.0%,
3.5 toxic blocks/video, 10 572 gold segments total.

Three tasks exist. Two are closed for us:

| task | status |
|---|---|
| online per-timestamp macro-F1 | **closed, round 11.** A zero-temporal-resolution oracle broadcast predictor scores **79.42** there, above every published number. The metric pays for getting the video right. |
| trimmed-segment classification | not entered; it is a classification task on gold-trimmed inputs, i.e. it presupposes the localization we cannot do |
| **proposal-level localization, F1@tIoU** | **the live arena.** Oracle broadcast scores only **10.9** (train) / 7.1 (val) at tIoU 0.5, so the task is genuinely non-degenerate. Published dataset-paper baseline: **ActionFormer 52.65** F1@tIoU0.5 (visual only; V+T+A 50.92). |

---

## 2. What is measured, and it is a lot

All numbers are this project's own, on the train split (5-fold video-grouped CV) or val; the
119-video test split has never been opened for this axis.

### 2.1 The substrate and its ceiling

Features are all **frozen**, mean-pooled inside each window of a uniform K=30 grid (~8.0 s/window
against a median gold segment of 8.12 s):
visual = CLIP-L/14-336 image; speech text = CLIP text tower over the Whisper window transcript;
on-screen text = CLIP text tower over EasyOCR window text; audio =
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` masked-mean pooled; 3586-d concat.
Head = linear projection + 2-layer MLP, plain BCE.

Primary read-out: **video-macro within-video AUC (wv-AUC)** — a broadcast predictor scores exactly
0.500 by construction, so a video-level classifier cannot inflate it.

| reading | wv-AUC |
|---|---|
| per-window head, 5-fold CV in train, no epoch selection (**the honest number**) | **0.5878** (seed sd 0.0015) |
| same head, val split with val-based epoch selection (an upper reading) | 0.671 |
| same head, test split, R11 protocol | 0.6349 |
| **Qwen2.5-VL-72B** per-window MLLM scoring | 0.5755 |
| frozen hate-tuned RoBERTa instead of CLIP text tower | 0.5842 |
| project-wide ceiling across 13 earlier routes | 0.576 |
| restricted to the 84.2% of windows that are label-pure | 0.6142 |
| single channels: audio 0.623 · visual 0.587 · CLIP-text(ASR) 0.583 · CLIP-text(OCR) 0.572 | |

**Within-video Spearman between the model score and the gold per-window offensive fraction: 0.137.**

### 2.2 Where the gap lives

2×2 oracle substitution on F1@tIoU0.5, real pipeline at 23.8:

| video-level term | within-video residual | F1@tIoU0.5 |
|---|---|---|
| model | model | 23.8 |
| **gold** | model | 28.1 (+4.3) |
| model | **gold** | **86.5 (+62.7)** |
| gold | gold | 87.0 |

So the task is a **within-video discrimination** problem and essentially nothing else. Also: 207 of
237 train videos contain a toxic segment, so the video-level sub-task is nearly vacuous here.

### 2.3 Grid / representation ceilings (oracle labels → merged runs → F1@tIoU 0.3/0.5/0.7)

gold segments 100/100/100 · uniform 1 s 99.9/99.5/99.1 · uniform 2 s **98.2**/98.2/95.7 ·
uniform 8 s (ours) 93.9/**87.6**/68.1 · Whisper-chunk grid 58.5/50.2/42.1.

**Gold boundaries are not free**: Whisper-large-v3 chunk boundaries recover only **32% recall /
27% precision** of gold boundaries at 1 s tolerance.

### 2.4 The modality asymmetry (the one strong positive on this axis)

Within-video circular shift of each video's 30 window vectors, then retrain (matched heads, 12 seeds,
video-clustered CI):

- **audio loses 3.30 macro-F1, CI [+0.71, +5.89] excluding zero** → the audio channel carries genuine
  *moment-level* information.
- **CLIP visual loses 0.28, CI [−3.66, +2.97]** → the visual channel's entire contribution is
  *video-level identity*; shuffling the visual timeline within a video is free.

The published modality inversion (F1@tIoU visual 52.65 ≫ audio 25.40; online macro-F1 audio 60.84 >
visual 57.52) **did not reproduce under matched heads** (both required directions null), i.e. it is
consistent with an architecture/context-window confound.

---

## 3. The kill list — do not propose anything on it

| # | closed | evidence |
|---|---|---|
| K1 | **score→interval decoding as a contribution** | occupied in sound event detection since 2019 (`1906.06909`: 22.9→32.0 event-F1 from post-processing alone), formalised as SEBB `2406.04212`, ported to video anomaly detection 2026-04 (`2604.09327`), and its per-instance *unsupervised* variant is **nSEBB `2505.11889`** (per-recording contrast + run-length → filter length + merge threshold). Priced here: naive 21.6 → tuned 23.8 → per-video-normalised 25.5. And our best decode config (23.8) is **below** ActionFormer's published 52.65, which consumes no score curve. |
| K2 | **within-video contrastive / ranking objectives** | bracketed by two nulls: score-derived intra-video negatives (UniVTG-style) +0.31 macro-F1 CI containing zero; **gold-certified** within-video pairwise ranking **−0.0052 wv-AUC, CI [−0.0094, −0.0011], significantly negative**. Vacuity argument supplied in advance and confirmed: if the head can represent a per-video intercept, the BCE logit already ranks within a video as the Bayes-optimal residual does. |
| K3 | **temporal architecture families** | three independent families null vs a per-window independent head on the same features: GTEA-line causal MS-TCN (+0.27 / −0.23), LSTR-family causal Transformer (+1.84, CI [−0.08, +3.96]), dense action detection MS-TCT/PAT-shaped (−0.33). Run-to-run GPU nondeterminism alone is ±0.5 macro-F1, i.e. larger than any claimed effect. |
| K4 | **per-timestamp macro-F1 as the target metric**, and HateMM / MultiHateClip as localization arenas | broadcast oracle 79.42 dominates the first; HateMM gold coverage median 0.806 / MHC-ZH median **1.000 with 69.6% at exactly 1.0** — "the span is the video". Any localization number must carry a broadcast control. |
| K5 | **"use a bigger scoring model"** | 72B MLLM per-window = 0.5755 wv-AUC, **below** a two-layer head on frozen features (0.5878) under a stricter protocol. |
| K6 | **"the text substrate was wrong"** | frozen hate-tuned RoBERTa on per-window ASR/OCR: −0.0044 wv-AUC (CI contains zero) and it *raises* the between-video variance share 0.432 → 0.493. |
| K7 | **window impurity as the explanation** | 84.2% of train windows are label-pure; restricting the read-out to pure windows is worth only +0.025 wv-AUC. |
| K8 | **coverage-budget / count-constrained decoding** | budget-constrained decode is −9.4 macro-F1 at achievable budget accuracy (best per-video coverage predictor reaches r ≈ 0.5 on test; the decode needs r ≳ 0.7 to break even). No stratum helps. |
| K9 | **gold spans as training supervision for video-level accuracy** | crop augmentation killed by equivalence on 3/3 datasets (1-sided upper bound +0.0024 vs δ=+0.015); span-privileged distillation killed by equivalence (upper bound +0.0101); trimming to the *gold* span does not beat not trimming (−0.0041). |
| K10 | **generic boundary detection → classify** | saturated (GEBD line, BSN/BMN/TAG); and editing/scene boundaries need not coincide with toxicity changes. |
| K11 | **within-video affine score normalization** | a positive affine map preserves within-video ranks exactly. |
| K12 | **measurement / benchmark / degeneracy-aware-evaluation contributions** | banned by user rule, permanently, even though the field-map sweep rates it the single highest-value available contribution. |

Occupancy already established for the neighbourhood: **SafeLens** (AAAI-26, same lab as HateClipSeg)
= Whisper + EasyOCR + Qwen2.5-VL → LoRA Llama-3-8B scoring segments **independently**, identical
modality set and corpus; **StreamSense** (WWW 2026) = streaming per-timestamp detector with VLM
escalation + deferral (and its deferral primitive is this project's already-dead uncertainty-gated
deferral); **MultiHateLoc** (WWW 2026) = weakly-supervised MIL frame localization, whose own ablation
selects 33% of the timeline on a corpus with median coverage 0.806 and whose selection mechanism is
worth only 0.033 mAP over pooling everything; **ActionFormer** as the dataset-paper baseline;
**nSEBB** for unsupervised per-instance decode.

---

## 4. The candidate slate to score

Score each 0-3 (0 = kill now, 3 = pilot immediately). State the **load-bearing** reason in one or two
sentences: the occupant, the vacuity argument, or the measurement that already prices it. Candidates
D1-D3 are families a previous reviewer named but that were never tested; D4-D12 are new.

**D1 — TRANSD, toxic-state transition discrimination.** Do not predict "how toxic is window i";
predict "does the toxicity state change between i−1 and i", from `(x_i, x_{i−1}, x_i − x_{i−1})` plus
audio delta features. Intervals are then the regions between predicted change points, labelled by
pooling. Motivation: absolute within-video level is stuck at 0.588 across three encoder swaps, but
the boundary target is a different function of the same features, and the audio shift control says
audio's moment-level information is exactly the kind that a difference operator exposes. Known
obstacle: on the 8 s grid **73.4% of train windows already contain a gold boundary**, so the target is
near-saturated at that resolution and would need a finer grid; and the same vacuity argument as K2
may apply (if the absolute head were good, |Δscore| already gives boundaries).

**D2 — TGATT, target/attack state factorization.** Two states on the timeline: a slowly-varying
"who/what is being targeted" state and a fast "an attack act is happening now" state, with a learned
interaction; score = interaction, not a single per-window logit. HateClipSeg ships per-segment
category labels, which are free within-video supervision for the target axis. Motivation: hate is
constitutively target × act; the persistent half is re-estimated from scratch at every window today.
Known obstacle: if the target state degenerates to a per-video constant this is RUBi / Learned-Mixin
and is covered by K2's vacuity argument.

**D3 — STRIDESPAN, decouple output stride from evidence span.** Predict every 2 s but build each
prediction from overlapping 2 / 8 / 24 s contexts. Motivation: §2.3's representation ceiling rises
87.6 → 98.2, and the pure-window post-hoc says a finer output grid is worth about +0.025 wv-AUC.
Known obstacle: we sit at 23.8, nowhere near the 87.6 the current grid already permits, so the grid
is not the binding constraint; and multi-scale context is symmetric and universal.

**D4 — SEGPOOL, homogeneous-region pooling as the prediction unit.** Run a label-free
temporally-constrained segmentation / clustering of each video's own multimodal feature sequence into
K̂ homogeneous regions, then score the *region* by pooling its windows, instead of scoring windows
independently. Motivation: the within-video Spearman with gold window toxicity is 0.137, i.e. window
scores are extremely noisy; averaging over the ~4 windows of a true segment should cut that noise by
about 2×, and the project has separately observed that segment-level purity beats whole-video
pooling. Cheap decisive falsifier available with zero new code: pool the *existing* per-window scores
inside **oracle gold segments** and re-read wv-AUC — that is the ceiling of the whole family, and if
it is small the family dies. Known obstacle: unsupervised temporal action segmentation *is*
temporally-constrained clustering (TW-FINCH, CTE, ABD), so this is a first application, not a
mechanism, unless the clustering is doing something hate-specific.

**D5 — CAPACITY, is 0.588 an information ceiling or an estimation gap?** Not a method; a decisive
diagnostic that routes the entire round. Fit the same head to the *training* windows and read
training-set wv-AUC. If the head cannot fit within-video ordering in-sample either, the frozen
features do not carry moment-level toxicity and **no mechanism operating on them can work** — the
honest conclusion is that the goal is unreachable without new features. If it fits in-sample and
generalises to 0.588, the bottleneck is sample efficiency and an entirely different family opens
(regularization, cross-corpus segment supervision, data-efficient heads).

**D6 — XSEG, cross-corpus segment-level supervision.** HateMM (744 train videos with gold spans),
MultiHateClip EN/ZH and ImpliHateVid are all on disk with span annotations that this project has
**never used as segment-level training supervision** (declared validation-only in three prior
pre-registrations; K9 killed only their use as *video-level* crop augmentation, a different
consumer). Pre-train the per-window head on the union, fine-tune on HateClipSeg. The framing that
would make it a mechanism rather than engineering: **high-coverage corpora (HateMM 0.806, MHC-ZH
1.000) teach what hate looks like; only low-coverage corpora teach where it starts and stops** — so
the two must enter the objective differently, not be concatenated. Known obstacle: cross-dataset
transfer is a standard engineering move, and HateMM/MHC "spans" cover ~the whole video.

**D7 — SPKUNIT, speaker-turn conditioned units.** Run speaker diarization (pyannote, fully automatic,
no manual annotation) on the raw audio; make the prediction unit the speaker turn and add speaker
identity as a within-video conditioning state. Motivation: hate is a stance act by a speaker and
persists over that speaker's turn; 4-minute BitChute/YouTube videos contain interviews, clips and
inserts. Known obstacle: M3 says Whisper chunk boundaries recover only 32% of gold boundaries;
speaker turns are far sparser still, and monologue videos degenerate to one unit.

**D8 — RELGATE, reliability-conditioned multimodal interaction.** Gate the cross-modal interaction on
per-window ASR confidence, OCR occupancy and audio SNR, all of which are already on disk, so a window
with no speech and no on-screen text is not scored as if it had both. Known obstacle: partly occupied
by StreamSense's deferral/escalation, which is also this project's dead uncertainty-gated deferral.

**D9 — CATAUX, category timeline as free auxiliary supervision.** Train the per-window head with the
6-way multi-hot category target alongside the binary one. Known obstacle: R11 v2's `B2_DENSE` already
used a per-window multi-hot head over 5 categories and was null on the per-timestamp metric; only the
wv-AUC / proposal read-out is untested.

**D10 — OFFSET, evidence–label misalignment.** Diagnostic-then-method: measure wv-AUC of the existing
per-window scores against gold labels shifted by ±1, ±2 windows. If a non-zero shift is better, the
grid's evidence-to-label alignment is misspecified and an asymmetric receptive field is the fix.
Costs minutes on existing arrays.

**D11 — MEMSEG, retrieval over a labelled segment memory.** This project's lineage is
retrieval-augmented hateful-meme detection; the train split supplies 10 572 labelled gold segments.
Score a query window by retrieval against that segment memory instead of by a parametric head. Known
obstacle: the project's zero-training cross-dataset kNN on HateClipSeg already measured wv-AUC 0.5259
and frame-AP 0.5447 against a 0.5252 broadcast control, i.e. the naive version is measured and weak.

**D12 — UNITLATTICE, annotation-free adaptive unit grid.** Union of Whisper sentence boundaries,
EasyOCR text-change points and PySceneDetect shot cuts as an adaptive unit grid, replacing the
uniform 8 s grid. Known obstacle: SafeLens already segments by "transcript/scene-change heuristic";
Vid2Seq uses ASR sentence boundaries as pseudo event boundaries; DuVOG derives chapters from OCR'd
subtitles. Likely occupied outright.

---

## 5. What I need from you

1. **Score all twelve**, with the load-bearing reason. Name the occupant where one exists, by
   identifier if you can, and say if you are unsure.
2. **The structural question, answered directly.** Given §2 and §3 — three encoder swaps within
   0.01 wv-AUC of each other, three temporal architecture families null, the objective family
   bracketed by two nulls one of which is significantly negative, decode occupied and priced at 2-4
   points, and our own pipeline sitting at 23.8 F1@tIoU0.5 against a published off-the-shelf
   ActionFormer at 52.65 — **is there a legal mechanism family left that could plausibly move
   within-video discrimination on this substrate?** If your answer is no, say so plainly: the round
   will then be written up as "the temporal localization goal is unreachable under the current
   constraints" and escalated to the user for a scope ruling. Do not soften this to be helpful.
3. **Name any family the slate misses**, including cross-domain transplants (change-point detection,
   speaker diarization, topic segmentation, sound event detection, video anomaly detection,
   time-series regime switching, sequential change detection / CUSUM, survival analysis). For each,
   say whether it is occupied and by what.
4. **The gap to ActionFormer.** 23.8 vs 52.65 with the same corpus and roughly the same features.
   Is any candidate in §4 even addressing the reason for that gap? If the answer is "no, the gap is
   that the pipeline is a thresholded per-window score curve and ActionFormer is a trained detector
   with boundary regression on a feature pyramid", then say whether *reproducing a competent detector
   baseline first* is a precondition for any method claim here — and whether that is a legitimate
   round or an admission that the direction is not ready.
5. **Prescribe at most two sub-hour discriminating experiments** that would separate the surviving
   candidates, in enough detail to pre-register: arms, primary endpoint, control, and the smallest
   worthwhile gain given a seed sd of 0.0015-0.0049 on wv-AUC and a ±0.5 macro-F1 GPU-nondeterminism
   floor. Prefer experiments that can *falsify* a family cheaply over experiments that could confirm
   one.

Be hostile. A "2" that wastes a week is worse than a "0" that is wrong.
