# R14 — candidate slate, hostile scoring, and the occupancy sweep (2026-08-18)

Round 12 of idea discovery. Sub-direction: **hateful video temporal localization at the proposal
level** (mAP / F1@tIoU), the one part of the axis round 11 left open.

Reviewer: **gpt-5.6-sol at xhigh reasoning**, conversation only, no tool use, given the full
constraint map, all round-11 kills and all seven new measurements. Occupancy sweep: a separate
zero-GPU literature agent (arXiv API, WebSearch/WebFetch; Semantic Scholar and OpenAlex **not**
reached — recorded as a coverage gap).

---

## 1. Why the round is not a repeat of round 11

Round 11 closed *per-timestamp macro-F1* on HateClipSeg: a zero-temporal-resolution broadcast
predictor scores 79.42 there, above every published number, and three temporal architecture families
are null against a per-window independent head. The proposal metric is a different object:

| | online per-timestamp macro-F1 | proposal F1@tIoU 0.5 |
|---|---|---|
| oracle whole-video broadcast | **79.42** (above all published) | **10.9** (train) / 7.1 (val) |

So the proposal task is **not** degenerate, and a null on macro-F1 does not close it.

## 2. The 12 candidates and the hostile scores

Scores are the external reviewer's (0 = kill now, 3 = pilot immediately); the load-bearing reason
is its wording, condensed.

| # | candidate | mechanism | score | verdict |
|---|---|---|---|---|
| C1 | **WVCOND** | train the per-window head with a within-video conditional / pairwise-ranking objective instead of plain per-window BCE, so supervision is spent on within-video comparisons rather than cross-video prevalence | **3** | pilot now, but conditional logit is classical and the pairwise form is RankNet; not novelty on its own |
| C2 | **AUDLEAD** | factor score = f(video-level, visual) + g(within-video, audio+text), assignment derived from the round-11 circular-shift diagnostic | 2 | plausible inductive bias, but a hard visual→global assignment throws away visual's real 0.587 within-video AUC; reviewers name RUBi / Learned-Mixin |
| C3 | **TXTENC** | replace the CLIP text tower with a real sentence / hate-tuned text encoder on per-window ASR and OCR | **3** | mandatory substrate repair, **not a contribution**; SafeLens and LELA already use strong language models |
| C4 | **RESGRID** | 8 s → 2 s output windows (representation ceiling 87.6 → 98.2 at tIoU 0.5) | 2 | necessary for high-tIoU work, but not the current bottleneck; do not confuse output stride with evidence-window duration |
| C5 | **DECODECOND** | predict decode parameters per video from label-free score-curve statistics | **0** | occupied near-exactly by **nSEBB `2505.11889`** (per-recording contrast + run-length → filter length and merge threshold, unsupervised); and M5 caps the whole axis at 2-4 points |
| C6 | **SEBBV** | port Sound Event Bounding Boxes to video: multi-threshold candidates, extent decoupled from confidence | 1 | the central idea *is* **SEBB `2406.04212`**; at best an adaptation |
| C7 | **PROSBND** | prosodic change-point detection proposes boundaries, other modalities classify | 1 | audio-derived boundaries are a known prior — **DASH `2603.15685`**, **AutoAD II `2310.06838`**, Vid2Seq; and a fine uniform grid already avoids the recall bottleneck |
| C8 | **COUNTDEC** | predict the number of toxic blocks, decode exactly that many intervals | **0** | no occupant found (3C-Net `1908.08216` uses count as a *loss*, not a decode constraint) but an off-by-one count destroys the tIoU match; the failed coverage-prior family under a new scalar |
| C9 | **REFPROP** | resolve target-group referents across segments before classifying each segment | 2 | legal and mechanistic, but brittle on ASR/slurs/implicit targets; and it is a score-quality mechanism, i.e. a different round |
| C10 | **XVIDCAL** | within-video robust z of the score before decoding | 1 | a positive affine map **preserves within-video ranks exactly**, so it can only move threshold crossings; a trick |
| C11 | **METRICSWITCH** | re-evaluate round 11's three killed temporal families on the proposal metric | 2 | mandatory baseline rerun, not a candidate method |
| C12 | **GEBD2S** | class-agnostic generic boundary detection then classify | **0** | saturated in general video (GEBD line, BSN/BMN/TAG); editing boundaries need not coincide with toxicity changes |

## 3. Occupancy sweep — the decode axis is closed before we enter it

| slot | status | firmest occupant | rating |
|---|---|---|---|
| score→interval decoding as a contribution | occupied in SED since 2019, in video anomaly detection since 2026-04 | `1906.06909` (22.9→32.0 event-F1 from post-processing alone) · SEBB `2406.04212` (.644→.686 PSDS1, model untouched) · `2604.09327` | **d** |
| per-instance decode parameters from label-free score statistics | near-exact match | **nSEBB `2505.11889`** | **d** |
| per-video score normalization before decoding | universal implementation detail; the field encoded it as a *metric* choice (macro- vs micro-AUC) | VAD convention; `2008.12328` | c |
| per-modality temporal bandwidth / role from a shift-permutation diagnostic | multi-scale is symmetric everywhere; the **diagnostic itself has 0 arXiv hits** | AVE-CLIP `2210.05060` (body unread — residual risk) | **b** |
| audio change-point → boundaries → classify | present in token compression and audio description, absent in harmful-content localization | DASH `2603.15685`, AutoAD II `2310.06838` | b/c |
| generic boundary proposal → classify | saturated | GEBD line, `2101.10511` | d |
| count-constrained decode | no occupant found on five queries | 3C-Net `1908.08216` (count-as-loss only) | a (but a trick) |
| cross-segment discourse / coreference | occupied text-side, empty for video segments | CoSyn `2303.03387`, CAD | b |
| metric-mismatch used to motivate a method | occupied in VAD as of 2026-04 | `2604.09327` (AUC-ROC 0.617 → event precision 3.9% at tIoU 0.5) | d |

**Consequence.** Four of the twelve candidates (C5, C6, C10, and the framing behind C11) are
post-processing tuning with a 2019-2026 citation attached. The reviewer's rejection of a decode
paper is short and correct: our best measured decode configuration (38.3 synthetic, 23.8 real) is
*below* the number the dataset paper already publishes (ActionFormer 52.65), so the "decode leverage"
quantifies how badly a per-window head was configured, not a new capability.

## 4. The families the reviewer added that were not on the slate

1. **Video-set-conditioned reference head** — classify each window *through its relation to a
   label-free summary of its own video* (leave-one-out centroid residual, within-video ranks,
   low-rank interactions), rather than normalizing a scalar afterwards. This can change the
   within-video ordering; a scalar z-normalization provably cannot. **This became factor C of the
   pilot.**
2. **Decouple output stride from evidence span** — predict every 2 s but build each prediction from
   overlapping 2 / 8 / 24 s contexts. Distinct from C4, which conflates the two.
3. **Toxic-state transition discrimination** — a head on `(x_i, x_{i-1}, x_i − x_{i-1})` predicting
   *changes in toxicity*, not generic acoustic or visual events.
4. **Target–attack state factorization** — a slowly-updated target/group state and a current-window
   attack state with a learned interaction, of which coreference (C9) is only one updater.
5. **Reliability-conditioned multimodal interaction** — gate cross-modal interactions on ASR
   confidence, OCR occupancy, audio SNR. Partly occupied by StreamSense.

## 5. What went to pilot

The reviewer's prescribed sub-hour discriminating experiment, adopted verbatim as the pilot and
extended by one factor: a **2×2×2 factorial of objective (C1) × text substrate (C3) × video-relative
representation (new family 1)**, 5-fold video-grouped CV inside the 237 training videos, primary
endpoint video-macro within-video AUC, secondary the proposal F1@tIoU under a decoder frozen in
advance. Pre-registered in `idea-stage/R14_WVD_FREEZE.md` (commit `0f20505`), which was committed
before any pilot code existed.
