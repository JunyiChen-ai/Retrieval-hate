# R17 hostile-review bundle (2026-08-18)

Round 14 of idea discovery. Sub-direction: **hateful video temporal localization**, now on a
competent detector base. Reviewer: gpt-5.6-sol xhigh, conversation only, no tool use.

---

## 1. The new base (round 16, `idea-stage/R16_DETBASE_RESULT.md`)

Official ActionFormer (vendored `61ea7eb`) + 4 FPS CLIP-L/336 features on HateClipSeg
(395 videos of 435 usable, frozen split 237 train / 39 val / 119 test, 4-minute videos,
contiguous segment tiling, mean segment 8.88 s).

Ground-truth convention `rawseg` (every offensive segment its own instance, 12.39/video,
median 8.4 s) — established as the dataset paper's convention by three independent arguments.

**Test F1@tIoU 0.5, 3 seeds, mean ± sd:**

| system | F1@0.3 | **F1@0.5** | F1@0.7 |
|---|---|---|---|
| per-window score-curve test bed (the project's old substrate) | 20.55 ± 2.23 | 10.68 ± 0.54 | 3.36 ± 0.61 |
| ActionFormer, visual only | 50.28 ± 1.82 | **38.22 ± 1.63** | 19.95 ± 0.09 |
| **ActionFormer, V ⊕ wav2vec2-emotion ⊕ BERT-over-ASR, early fusion** | 54.72 ± 0.69 | **42.02 ± 0.73** | 21.11 ± 0.86 |
| *published paper, visual, 435 videos, 80/20 split* | *59.38* | *52.65* | *30.99* |

Retraining cost: **4 minutes per seed**. The residual 42.02 vs 52.65 is corpus subset (90.8%),
training share (60% vs 80% — measured, +16% training data buys nothing) and encoder version.

Established by that round:

1. The detector's advantage over the old test bed is **candidate coverage**: proposal-pool
   recall at tIoU 0.5 is **90.7%** for ActionFormer vs **25.6%** for the K=30 window-grid decoder.
2. **Boundary regression contributes ~0 at block level and −5.6 at segment level** when snapped
   to the 30-window grid. The advantage is *which* spans are proposed, not sub-window precision.
3. **Early fusion works on this base: +3.8 F1@0.5.** The paper's late fusion *lost* 1.7.
4. Warning carried forward: `rawseg` boundaries are **Whisper sentence boundaries** inside
   otherwise homogeneous offensive stretches, so part of what a detector is rewarded for is
   predicting speech pauses. Any claim resting on `rawseg` needs an audio-onset control.

## 2. New recon this round (val split only, 39 videos, 531 gold segments, 3 seeds, descriptive)

The R16 report priced the ranking bottleneck with a fixed-small-k table, which conflates
ranking quality with a recall cap. Re-priced at the system's **actual operating point**:

VAT detector, rawseg, **val**, pool = 200 proposals/video, model keeps 22.0/video at its
val-selected threshold:

| read-out | F1@0.5 |
|---|---|
| model at its own threshold | **48.76 ± 0.65** (P 39.54 / R 63.72) |
| **oracle re-ranking, same per-video budget** (choose the best 22 of 200 within each video) | **63.87 ± 0.59** |
| oracle re-ranking, budget = true gold count per video | 77.97 ± 1.07 |
| oracle binary verifier (keep exactly the matching proposals) | 96.83 ± 0.29 (P 100 / R 93.85) |
| pool recall @0.5 | 93.85 |

So **+15.1 F1 is available from purely within-video re-selection at a fixed per-video count.**
This is not a recall-cap artifact.

Decomposition of the 857 kept proposals (mean over seeds):

| kept proposal class | n | share |
|---|---|---|
| matched, tIoU ≥ 0.5 | 338 | 39.5% |
| partial, 0 < tIoU < 0.5 (right region, wrong extent) | 219 | 25.5% |
| **zero overlap with any gold** | 300 | **35.0%** |

| gold class | n | share of 531 |
|---|---|---|
| matched | 338 | 63.7% |
| **missed although a ≥0.5 proposal was in the pool** | 160 | **30.1%** |
| missed, no ≥0.5 proposal in pool | 33 | 6.2% |

Mean length: matched proposals 9.8 s, zero-overlap false alarms 9.9 s (indistinguishable).

**The single most awkward number for this round:**

| within-video Spearman with oracle tIoU, over the full 200-proposal pool | ρ |
|---|---|
| **the detector's own classification score** | **0.350** |
| **the proposal's duration alone** | **0.423** |

A duration prior orders the pool better than the trained classification score does.

## 3. Constraint map — what is already dead in this project (do not re-derive)

**Hard rules.** Method paper only (never benchmark / audit / metric / evaluation-protocol
papers). No manual annotation. No new datasets. Four red lines: zero test-label tuning;
decision rule frozen before results are seen; blindness during design/implementation; the
frozen run is submitted exactly once. Incremental but real and stackable gains are acceptable —
"one step must gain 5 points" is explicitly *not* the kill line. Round API budget ¥15 (~$2);
local RTX 5090, single GPU; a detector retrain is 4 min/seed.

**Standing closures on the localization axis** (all measured on the OLD per-window score-curve
substrate, which round 16 showed was capped at 25.6% pool recall — so these are *not*
transferable as evidence about the mechanisms, but they are evidence about what this project
has already spent):

- **Round 11**: three temporal architecture families null vs a per-window independent head;
  circular-shift control — shuffling audio within a video costs 3.30 macro-F1 (CI excludes 0),
  shuffling CLIP visual costs 0.28 (CI contains 0).
- **Round 12 (R14-WVD)**: 2×2×2 factorial. Within-video ranking objective **−0.0052 wv-AUC,
  CI excluding zero** (significantly harmful; the vacuity argument was supplied in advance —
  with a per-video intercept, BCE logits already rank Bayes-optimally within a video).
  Hate-tuned RoBERTa text encoder vs CLIP text tower: −0.0044, null. Video-relative features
  (leave-one-out centroid residual + within-video ranks): −0.0031, null.
- **Round 13 (R15-NT)**: seven-arm channel-composition panel. Plain concatenation of all four
  channels wins outright (0.5901). Audio-only 0.5507, visual-only 0.5535, text-only 0.5288.
  Within-video centering does exactly what it was designed to do — between-video share of score
  variance 0.451 → 0.150 → 0.095 — and wv-AUC falls monotonically with it. Deleting CLIP visual
  costs −0.0214 (CI excludes 0) even though shuffling it within a video is free.
- **R15-FS**: no label offset beats zero alignment (monotone degradation in |shift|). Oracle
  region pooling — using the labels themselves to define pooling regions — buys +0.006 against a
  +0.015 bar, so label-free temporal clustering as a prediction unit is closed by its own ceiling.
  A plain width-3 running mean beats both oracle partitions (+0.0135): the score error is
  broad-band, not region-structured.
- **Segment-memory retrieval (D11/MEMSEG)**: the project's own zero-training kNN over labelled
  train segments measured within-video AUC **0.5259** against a **0.5252** broadcast control.
- **MLLM per-window scoring**: a 72B MLLM reading frames + ASR scores within-video AUC 0.5755,
  *below* a two-layer head on frozen features (0.588) under a stricter protocol.
- **Decode axis (K1)**: score→interval decoding as a contribution is occupied since 2019 in SED
  (`1906.06909`), SEBB `2406.04212`, nSEBB `2505.11889`, video anomaly `2604.09327`; and the
  project's own measurement prices real-score decode leverage at 2–4 F1 points.
- **Count-constrained decode (COUNTDEC)**: no occupant found, but an off-by-one count destroys
  the tIoU match; scored 0.
- **Speaker-turn units, topic segmentation, adaptive unit grids from ASR/OCR/shot boundaries**:
  scored 0; semantic/editing units match toxicity boundaries at 32% recall / 27% precision, and
  the unit-grid slot is occupied by SafeLens, Vid2Seq `2205.14315`, DuVOG `2208.11307`.
- **Head architecture changes are downweighted by external evidence**: OpenTAD reports the TAL
  detection-head axis moved ~0.5 mAP in five years while the feature axis moved 22.6.
- Generic occupants a reviewer will name instantly: RUBi `1906.10169` / Learned-Mixin
  `1909.03683` for any "score = bias branch + residual"; GMU `1702.01992` for gated fusion;
  RevIN `2105.15078` / cepstral mean normalization for any per-instance feature centering;
  UMIL `2303.12369` (CVPR 2023) for "video-level context bias corrupts snippet-level prediction".

**Domain occupants.** HateClipSeg `2508.01712` (ACM MM 2025, the dataset + ActionFormer/LSTR
baselines); StreamSense `2601.22738` (WWW 2026, streaming per-timestamp + selective VLM
escalation + deferral — note this project's own uncertainty-gated deferral measured −0.0135,
0/3 seeds); SafeLens (AAAI-26, same lab as HateClipSeg — Whisper + EasyOCR + Qwen2.5-VL into a
LoRA Llama3-8B, per-segment scoring with **no temporal model**); MultiHateLoc `2512.10408`
(WWW 2026, weakly-supervised, learned per-modality importance, no OCR channel); LELA (training-
free LLM per-frame scoring, its own ablation makes OCR the largest single modality gain,
+2.6 ROC-AUC); `2508.04900` (MMUW'25) which concludes *"hate speech does not possess uniquely
distinguishable acoustic signatures when isolated by temporal annotations."*

**One measured domain fact that is not yet used by any model here**: a Gate-C annotation
re-analysis found 30.1% of this project's localization misses have evidence in **on-screen text
only** (odds ratio 2.29 vs other modality gaps) — on_screen_text is the only significantly
enriched modality gap in the failure set. The R16 detector has **no OCR channel** (it uses CLIP
visual + wav2vec2-emotion + BERT over ASR).

## 4. The candidate slate

| # | name | mechanism |
|---|---|---|
| E1 | **XPOOL** extent-conditioned span verifier | second stage: pool the dense 4 FPS V/A/T features over each proposal's *own* predicted extent (plus a context ring), score with a small MLP, rank by `cls × verifier`. Motivating measurement: §2's ρ 0.350 vs 0.423 — the anchor-free point classifier never sees the extent it regresses, so a duration prior out-ranks it |
| E2 | **IOUHEAD** quality-aware ranking | auxiliary head predicting each point's tIoU with its matched gold; rank by `cls × predicted IoU`. Expected to be the standard occupant (BMN proposal-evaluation module, IoU-Net, GFL, VarifocalNet); proposed as the *mandatory baseline* E1 must beat, not as a contribution |
| E3 | **OCRDENSE** dense on-screen-text channel in the detector | 4 FPS OCR → text encoder as a fourth early-fusion stream. Backed by §3's 30.1% / OR 2.29 measurement, by LELA's +2.6 ablation, and by R16's +3.8 early-fusion result. Weakness: "adding a modality is not a mechanism" |
| E4 | **MODROLE** modality-asymmetric role assignment | inside one detector, feed the *boundary-regression* branch and the *classification* branch different modality subsets (e.g. audio owns boundaries, visual+text own the label). A prior occupancy sweep recorded this as the one slot with no structural twin found (Decouple-SSAD `1904.07442` splits localization from classification within one modality; AVVP `2007.10558` predicts modality-specific *events*, not modality-specific *roles*). Extra pull: §1 item 4 says `rawseg` boundaries are literally speech boundaries, so the assignment is predicted, not arbitrary |
| E5 | **SPANRET** retrieval over labelled train spans | kNN in extent-pooled feature space against the 10 572 labelled train segments; re-rank by retrieval margin. Prior: MEMSEG measured 0.5259 vs a 0.5252 broadcast control at window level |
| E6 | **MLLMVERIFY** MLLM as span verifier | send the top-k proposals' frames + ASR to an MLLM as a *comparative* ranking question over few candidates, rather than as per-window scoring. Argument for why this differs from the 0.5755 null: the task shape is relative ordering within a video over actual span content, not absolute per-window scoring. Constraint: ¥15 total budget |
| E7 | **SETSEL** set-level, overlap- and count-aware selection | choose the kept set jointly instead of by independent thresholding. Expected K1/COUNTDEC collision |
| E8 | **VIDPRIOR** video-level score conditioning the span score | expected self-kill: §2's headroom is entirely *within-video*, and a per-video multiplier cannot change within-video order |
| E9 | **HARDNEG** hard-negative mining from the detector's own zero-overlap false alarms | 35% of kept proposals have zero overlap with any gold; train the second stage on exactly those. Generic OHEM |
| E10 | **CONSIST** cross-modal agreement as span confidence | three per-modality scoring branches in one detector; rank by inter-branch agreement rather than by the fused score |
| E11 | **CTXNEG** complement-region negatives | at HateClipSeg's 0.54 coverage the normal stretches inside hateful videos are genuine negatives, so the high-coverage invalidity argument does not apply here; mine them as within-video negatives for span classification. Recorded as "empty in hate, occupied in general TAL" (CoLA `2103.16392`, CPL, CNM, UniVTG `2307.16715`) |
| E12 | **DURPRIOR** explicit duration prior in the ranking | trivial; must appear as a *control arm*, never as a candidate |

## 5. What is being asked of the reviewer

1. Score every candidate 0–3 (0 = kill now, 3 = pilot immediately) with the load-bearing reason.
2. Name any family the slate missed, and whether it is occupied.
3. Say plainly whether the +15.1 within-video re-ranking headroom is a real method opening or a
   measurement artifact, and in particular whether the ρ(duration) > ρ(score) fact kills E1/E2 by
   showing the gap is geometric rather than semantic.
4. Prescribe a **single discriminating experiment** that fits: one RTX 5090, ≤ 2 h, detector
   retrain 4 min/seed, contrast line 42.02 ± 0.73 (3 seeds) on test with the val-selected
   threshold, or an out-of-fold protocol inside the 237 train videos if that is sounder.
5. If no candidate is a paper candidate, say so directly.
