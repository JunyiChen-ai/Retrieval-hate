# Round 11 brainstorm bundle — temporal span / localization sub-direction

Reviewer: gpt-5.6-sol, xhigh reasoning, hostile. Date 2026-08-18.

## 0. What you are being asked

This project has run 10 idea-discovery rounds, 114 candidates, and banked exactly one entry.
Round 11 is scoped to a single sub-direction: **temporal span / localization in hateful video**.
A dedicated 828-line landscape survey was completed today
(`research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md`). I want two things from you:

1. A hostile score (0-5) and a kill/keep verdict on each of the 12 candidates in §3.
2. **The main question: is there a legal family in this sub-direction that I have missed?**
   Not a variant of one of my 12 — a family. If the answer is "no, the sub-direction is closed
   for this project under its constraints", say that plainly.

Do not be diplomatic. Prior rounds' most useful reviewer output was the kill list.

## 1. Hard constraints (violating any of these is an automatic kill)

- **Method paper aimed at classification-accuracy gains only.** No benchmark, dataset, audit,
  metric, evaluation-protocol or measurement paper. (User rule, absolute.)
- **No manual annotation.** No point-level supervision, no new labels, no crowdsourcing.
- **No new dataset construction.**
- Four red lines: zero test-label tuning; decision rule frozen before results; blindness during
  design; the frozen run is submitted once.
- Compute: one local RTX 5090. API budget for this whole round ≤ ¥15.
- Incremental but *real* gains are acceptable (user rule). "One step must gain 5 points" is NOT
  the bar.
- Datasets currently in scope: HateMM, MultiHateClip-EN, MultiHateClip-ZH, ImpliHateVid.
  **HateClipSeg (395 videos already downloaded, split frozen and unconsumed) is pending a user
  ruling** — candidates depending on it are tagged `[needs-ruling]`, not auto-excluded.
  DeHate raw video is behind an application form → `[gated]`.

## 2. The facts any candidate must survive

### 2.1 The benchmarks are temporally degenerate (measured today, on disk)

| dataset | span coverage median | single contiguous block | degenerate-oracle frame AP |
|---|---|---|---|
| HateMM (hateful) | **0.806** | **72.8%** | **0.675** |
| MHC-EN | 0.937 | 95.8% | 0.786 |
| MHC-ZH | **1.000** | **98.2%** | 0.853 |
| HateClipSeg (any toxic) | 0.544 | 22.0% | 0.530 |

A predictor with a perfect video-level classifier and **zero temporal resolution** scores frame-AP
0.675 on HateMM — *above* the published weakly-supervised SOTA (MultiHateLoc, WWW 2026, 0.645).
On MHC it roughly doubles the published number. On HateMM the project's own video-probability
**broadcast control** reaches frame AP 0.5776 vs the actual segment model's 0.5892 (+0.012).
Conclusion: on the three in-scope span-annotated corpora, "temporal localization" is video-level
classification wearing a timeline. HateClipSeg is the only corpus where it is a real task.

### 2.2 The field is five methods wide, and each slot's status

Occupied: weakly-supervised MIL frame scoring (MultiHateLoc, WWW 2026, code never released);
training-free LLM per-frame scoring (LELA); MLLM emitting timestamps + RL (TANDEM — **the trained
model, HateMM Avg IoU 0.43, loses to zero-shot Qwen3-Omni-30B at 0.53**); supervised TAL port as a
dataset baseline (HateClipSeg/ActionFormer); streaming per-timestamp classification (StreamSense,
WWW 2026, HateClipSeg online Macro-F1 72.06).

Empty: proposal-based detection as a method contribution; DETR-style moment queries;
query-conditioned grounding; audio-first localization; dense-captioning framing; **localize → trim →
re-classify with predicted boundaries**; a shared protocol; any DeHate localization baseline;
degeneracy-aware evaluation; an OCR channel in weakly-supervised hate localization; purpose-built
temporal-grounding LLMs applied to hate; annotation-free synthetic span supervision.

Two governing facts from the adjacent literature:
- **OpenTAD controlled study (2502.20361):** holding the head fixed and swapping the feature
  backbone moves THUMOS avg mAP 49.8 → 72.4 (+22.6); five years of neck/head architecture spans
  67.9 → 68.4. Detection-head novelty is a rounding error.
- **Frame-AUC and mAP@tIoU are different measurements.** VadCLIP: UCF-Crime frame AUC 88.02 with
  mAP@[0.1:0.5] = 6.68. Both hate-domain frame-level papers report the metric that hides this.

### 2.3 The unreported empirical fact that motivated the round

The **official** HateMM hate span covers a median 0.806 of the video. The **minimal interval
actually sufficient to justify the label**, coded by this project's own Gate-C audit over 133
videos, has median coverage **0.100**, and is a **single interval in 111/133 = 83.5%** of cases.
The released spans are roughly 8× more generous than the evidence needs to be. Nobody in the field
reports this.

### 2.4 This project's own kills on this exact axis (do not re-propose any of these)

Six independent levels, all measured, all negative:

| level | mechanism | result |
|---|---|---|
| order / sequence | soft-DTW + signed transition kernel over frame sets, with within-video order-shuffle null | Δacc +0.0059 = the null's 95th percentile exactly |
| retrieval object | set-to-set / **late-interaction MeanMaxSim / Chamfer** over frame-group tokens | HateMM +0.0035 vs a +0.05 bar; MHC −0.0397; "pooling is effectively lossless on these representations" |
| causal-prefix conditional info | flat frame-group tensor + arc, supervised, label-oracle calibrated | **exactly +0.0000** HateMM / −0.0029 MHC |
| segment granularity | independently re-encoded per-segment features → uniform non-selecting kNN vote-mean | +0.0012 / +0.0032; **oracle headroom decomposes into symmetric (legal, ≈0) + selection (banned, 91-98%)** |
| frame count | 8 → 16 frames | −0.0077 |
| within-video signal | frozen-CLIP segment neighbour purity, within-video AUROC | **0.511** (chance) |

Plus: MLLM segment hate-density **weighted pooling** (P3) — no method role on any of EN/ZH/HateMM.
MLLM weak supervision for a trained segment head (P11) — MLLM teacher wv-AUC 0.5913 vs plain
video-label top-k MIL 0.5526-0.5580, matched Δ +0.0359, CI [−0.0009, +0.0730], n.s.
TERA Gate-0 — multi-segment complementarity 6/73 = 8.2% vs a 15% bar, NO-GO.
Multi-granularity segment retrieval — sign-flips by language, killed.
Segment-keyed retrieval-purity loop — killed, ratio 1.008 vs a 1.3× bar.

**Structural laws this project operates under:**
- **Law III / F47: per-item selection is banned.** No operator may choose, per test item, which
  segment / member / encoder to believe. This is what makes the 91-98% selection-locked temporal
  headroom formally unreachable.
- **Don't-pool ban (F37/S2S):** set-matching over sub-video units is dead on two encoders.
- **Label-free linear feature transforms are provably inert** — if the head's first op is dense
  linear, `x → Ax+c` with invertible A is an exact reparameterisation.
- **A large oracle ceiling is not evidence.** The largest oracle this project ever measured
  (+0.149/+0.152/+0.219) delivered +0.013/−0.007/+0.000. Gate on demonstrated conversion.
- Sub-video unit ↔ its own video's pooled vector: **cosine 0.95**. Trimming barely moves the key.

### 2.5 Adjacent axes already closed (a candidate must not route through these)

OCR as an input channel: mean fusion +0.0094 (bar +0.015); the *same vector* through the learned
fusion MLP **−0.0246, 3/3 seeds**; provenance typing −0.0020 with a label-permuted null at 90% of
the real effect. Audio/prosody: 0 for 4, failure mode is redundancy not weakness (label-permuted
prosody helps a text head *more* than real prosody). MLLM in front of the head: five access points,
all negative. Zero-supervision stance extraction: six routes, all negative. Decision-rule /
calibration mechanisms: capped at +0.25 to +1.2 points by a train+val-fitted global-threshold oracle.

### 2.6 Assets on disk (relevant to cost)

`data/CLIP_Embedding/HateMM/train_subclipK30_*.pt` — 744 train videos × 30 uniform contiguous
sub-clips × 1024-d CLIP ViT-L/14-336 image features. `train_subclipK4_*` on all four splits.
`data/ASR/HateMM/train_asrK{4,30,60}_whisper-large-v3.jsonl` — per-window ASR text.
`data/gt/HateMM/hate_spans.json` — released gold spans (430 videos, 784 segments).
MultiHateClip `Duration` column = released gold spans (under-advertised; nobody uses them).
`data/OCR/` — OCR cache for HateMM (1246 videos) + HateClipSeg + MHC test.
`data/gt/HateClipSeg/gold_segments.json` — 10,604 segments / 395 videos, 6-class multi-hot,
frozen unconsumed split at `data/gt/HateClipSeg/p11_split.json`.
Per-window MLLM hate scores: `data/MLLM_scores/HateMM/train_segscoreK30_p10-*` (72B, best segment
scorer this project has; HateMM wv-AUC 0.5913, HateClipSeg 0.5755).

**One structural asset gap:** the released spans exist for 100% of hateful videos and 0% of
non-hateful videos, so *any* span-derived quantity is a perfect label leak on the training set
unless the transformation is applied to both classes in a distribution-matched way.

## 3. The twelve candidates

Group A = inside the four in-scope datasets. Group B = `[needs-ruling]` on HateClipSeg.

### A1 — GSA, Gold-Span Crop Augmentation
Train-time only. For each hateful TRAIN video, add a second training item whose key is pooled over
only the sub-clips inside the released hate span, same label. Negatives get a coverage-matched
random crop so the transformation is label-blind in distribution. Test-time pipeline unchanged.
- *Why it is not covered by §2.4:* every prior temporal attempt used **noisy** supervision (MIL
  pseudo-labels, MLLM scores). The released gold spans have never been used as training supervision
  — in this project they were declared validation-only, and in the field they are only ever a
  localization target.
- *Ban escape:* not a test-time operator (Law III not engaged), not a change of the pooling object
  at inference (F37 not engaged), not a per-item selector.
- *Prior against:* sub-clip↔video cosine 0.95, HateMM coverage median 0.806, MHC-ZH 1.000 (literal
  no-op on one of three datasets).
- Cost: $0 CPU probe, then ~1 GPU-h.

### A2 — CRHN, Complement-Region Negatives
The complement of the hate span inside a hateful video is, given that the official span is ~8×
over-broad, a **high-precision** negative region — and it is matched to the positive on speaker,
channel, style, production and topic. Add those complement crops as extra negative training items.
- *Novelty claim:* the coverage degeneracy is normally read as a defect; this reads it as a
  certificate. Unreported.
- *Prior against:* asks a small head to separate cosine-0.95 pairs carrying opposite labels;
  within-video AUROC on frozen CLIP segments is 0.511.

### A3 — PBT, Predicted-Boundary Trim → Re-classify
The one slot the landscape calls empty and method-shaped: Yang et al. (2508.04900, MUWS@MM 2025)
show HateMM video classification going 79.30 → **98.64** macro-F1 and MHC-EN 64.37 → **97.31** when
**ground-truth** boundaries trim the video. Nobody has shown predicted boundaries recover any of it.
- *Prior against:* (a) it is per-item selection → **Law III**; (b) on HateMM the trim is a no-op on
  a third of videos (coverage ≥ 0.90); (c) the oracle gain is partly a relabelling artifact of the
  trimmed protocol (the trimmed task is a different, easier task); (d) §2.4's rule that a large
  oracle is not evidence.

### A4 — CDN, Coverage-Dependent Noise reweighting
Treat the annotated span as a bag of unknown purity, estimate purity from coverage, reweight the
per-video loss.
- *Occupants:* Yang et al. 2508.04900 is literally "temporal label noise"; StreamSense's
  IoU-weighted CE is the segment-level version.
- *Prior against:* this project measured that its confidently-wrong items are 1.8-4.0% and on both
  MHC splits **100% of them are positives called normal** — hard positives, not symmetric noise.

### A5 — SYNSPAN, annotation-free synthetic span supervision
Splice clip-level-labelled hateful and benign material into synthetic videos with known boundaries
(the Speech-Emotion-Diarization concatenation trick), train a segment scorer on free strong labels.
- *Prior against:* the downstream consumers are all closed — trimming is A3 (Law III), MIL is P11
  (killed), weighted pooling is P3 (killed). Also splice artifacts are trivially learnable.

### A6 — OCRSEG, OCR-burst boundaries as annotation-free semantic units
On-screen-text appearance/change gives free semantic boundaries. LELA's own ablation makes OCR the
largest single modality gain in hate localization (+2.6 ROC-AUC, larger than image, music, or video
context) and MultiHateLoc has no OCR channel at all. This project's Gate-C found on-screen text is
the only significantly enriched modality gap in its failures (30.1%, OR 2.29).
- *Prior against:* §2.5 — every OCR wiring tried here is negative, and the boundaries only matter
  if some segment operator survives, which §2.4 says none does.

### A7 — PACE, absolute-time / fps injection
Verified un-varied: the extraction passes no fps, so Qwen2.5-VL is pacing-blind — a 10-second clip
and a 3-minute clip are temporally identical up to group ordering. Injecting real duration/fps
engages mRoPE absolute-time encoding.
- *Prior against:* the finer frame-**group** tensor the model does encode was measured at exactly
  +0.0000 conditional information; a coarser pacing scalar is a decision-side scalar. Needs raw-video
  re-extraction.

### A8 — TGQ, target-group-conditioned scoring as a read-out
HateMM, MHC and DeHate all ship target-group labels. Enumerate candidate groups, score the best
matching moment per group, take the max as the video score. This is the only slot where the hate
domain has structure that generic moment retrieval lacks, and it is empty.
- *Prior against:* zero-supervision stance/target extraction is 6/6 dead here; MLLM in front of the
  head is 5/5 dead; and max-over-hypotheses is arguably per-item selection.

### A9 — SPANAUX, coverage as an auxiliary regression target
Multi-task: the shared head also regresses each video's span coverage fraction, a free label from
the released annotation.
- *Prior against:* the target is defined only on positives → leak; auxiliary-task gains on frozen
  features with a ~5M-param head are within the objective-level family that R12-ANCHOR priced at
  −0.0003/−0.0002 with error-set Jaccard 0.84-0.96.

### B1 — HCS-XFER, cross-dataset boundary transfer `[needs-ruling]`
Train a segment scorer on HateClipSeg's real multi-block segment labels; apply its predicted
boundaries to HateMM/MHC to trim or weight. Nobody in this field trains and tests across corpora.
- *Prior against:* the downstream is still Law-III-blocked, and this project's zero-training
  cross-dataset kNN on HateClipSeg reaches frame AP 0.5447 against a 0.5252 broadcast control.

### B2 — HCS-DIRECT, enter HateClipSeg's two accuracy tasks `[needs-ruling]`
HateClipSeg defines three tasks and **two of them are classification-accuracy tasks**, not
localization metrics: trimmed segment classification (published 69.48 macro-F1) and online
per-timestamp classification (published 72.06 macro-F1, StreamSense). The substrate is genuinely
different from everything in §2.4: 4-minute videos, ~27 segments/video, coverage 0.544, 22%
single-block, 6-class multi-hot. **Every one of the six closure levels in §2.4 was measured on
30-second videos where the span is the video.**
- *Honest limitation:* our copy is the 90.8% surviving subset (platform attrition is non-random),
  so absolute numbers are **not** comparable to the paper's; only method-vs-method on the identical
  subset is valid. That means no SOTA claim, only a mechanism claim with in-house baselines.
- *Status:* this is a substrate proposal, not yet a mechanism.

### B3 — MODASYM, modality-factored boundary vs label `[needs-ruling]`
HateClipSeg's own tables contain an unexplained inversion: for **localization**, visual 52.65 >
text 34.60 > audio 25.40 F1@tIoU 0.5, and late fusion of V+T+A is *worse* than visual alone at every
threshold; for **online per-timestamp classification**, the same paper's LSTR baseline has
audio-only 60.84 **beating** visual-only 57.52. Pixels carry the boundary, prosody carries the
label. Nobody has explained or exploited this. Mechanism: factor the model so the boundary head
consumes visual and the label head consumes audio+text, instead of one jointly-fused representation.
- *Prior against:* this project's audio axis is 0-for-4 with redundancy (not weakness) as the
  measured failure mode — but that was measured at the *whole-video* level, where prosody is
  averaged over 30 seconds, not at the moment level where HateClipSeg's number lives.

### B4 — Beat StreamSense on the online task `[needs-ruling]`
Occupied, freshly (WWW 2026), by a method with selective VLM escalation and deferral. Deferral and
escalation are also this project's dead "uncertainty-gated deferral" (−0.0135, 0/3 seeds).

### C1 — Degeneracy-aware evaluation / one protocol (recorded, not a candidate)
The highest-value contribution available in this sub-direction — report coverage, single-block
fraction and a broadcast control; unify the five incompatible protocols — is **closed by the
method-paper-only rule**. Recorded so the reviewer knows it was not overlooked.

## 4. The probe I intend to run before anything else

$0, CPU, HateMM train + MHC-EN train, LOO within train, zero test contact. Uses
`train_subclipK30` (visual) + `train_asrK30` (text) + `hate_spans.json`.

Arms, all producing one pooled key per video so the same evaluation applies:
- **P0** pool all K sub-clips (reproduces the standard whole-video key).
- **P1 GOLD** pool only sub-clips inside the released span (positives), with negatives given a
  coverage-matched random crop.
- **P2 RAND** identical coverage and block count to P1, random position — **the control that
  isolates span *location* from crop *duration***.
- **P3 COMP** pool only sub-clips outside the span.
- **P4 PRED** pool the top-m windows by the banked 72B segment score (the realistic, non-oracle arm).

Primary quantity: **Δ(P1 − P2)** with a paired bootstrap CI. If the CI contains zero, the released
spans carry no information beyond crop duration, and A1/A2/A3/A5/B1 all die at once.
Secondary descriptive: the distribution of cos(pooled_all, pooled_span).

## 5. Questions for you

1. Score each candidate 0-5 and give a kill/keep verdict with the load-bearing reason.
2. **Is there a legal family in this sub-direction I have missed?** Legal = accuracy-metric method
   contribution, no manual annotation, not isomorphic to §2.4/§2.5, not per-item test selection.
3. Is the §4 probe the right first spend, and is Δ(P1−P2) the right primary? What would you change?
4. Is B2 (entering HateClipSeg's accuracy tasks) worth asking the user to rule on, given the 90.8%
   subset limitation forecloses a SOTA claim? Argue both sides.
5. If your honest answer is "this sub-direction is closed for this project", say so and state the
   single cheapest measurement that would make that conclusion defensible in writing.
