# R17-OCRV — pre-registration (frozen 2026-08-18)

Round 14 of idea discovery. Sub-direction: **hateful video temporal localization on the R16
ActionFormer base**. This file is committed **before `scripts/r17_ocrv/` exists**. No number
below has been computed. The only measurements taken before this freeze are the two descriptive
recon panels reproduced in §1.3, both on the 39-video val split, both ceilings/geometry rather
than candidate metrics.

Reviewer: gpt-5.6-sol at xhigh reasoning, conversation only, no tool use. Bundle:
`idea-stage/codex_brainstorm_bundle_r17_2026-08-18.md`. Slate and scores:
`idea-stage/R17_CANDIDATES.md`.

Round cost so far: **¥0** of the ¥15 round budget (¥0 of ¥60 cumulative). No paid API call is
made by either pilot.

---

## 1. What is being tested and why

### 1.1 The measured bottleneck

R16 established that on this base the detector's proposal pool recalls 90.7% of gold segments at
tIoU 0.5 while the system scores 42.02, i.e. the binding constraint is **which proposals get
kept**, not which are generated and not where their boundaries fall.

### 1.2 The candidate

The reviewer scored twelve candidates. E1 (extent-conditioned span verifier), E2 (IoU-quality
head) and E12 (duration control) scored 3; nothing scored 3 as a *paper*. Its verdict on the
composition that this round actually tests, quoted:

> "Comparing VAT against VAT+OCR-verifier is unfair: only one system can see OCR. The decisive
> comparison is compute- and encoder-matched. … Run the VAT versus VATO detector comparison
> first. It addresses the cheapest and most damaging alternative explanation. A positive
> reranker result is not publishable until direct OCR fusion has been tested."

So the round runs the reviewer's prescribed order: **P1 answers the modality question inside the
detector, P2 answers the ranking question on P1's own out-of-fold proposals.** Both are
pre-registered here; P2 consumes P1's dumps and costs no additional detector training.

The domain premise for OCR, measured and already on record: **30.1% of this project's
localization misses have decisive evidence in on-screen text only** (odds ratio 2.29; the only
significantly enriched modality gap in the failure set), and the R16 detector has no OCR stream.
External agreement: LELA's own ablation makes OCR its largest single modality gain (+2.6 ROC-AUC);
`TEMPORAL_SPAN_LANDSCAPE §6.2` item 10 records the slot as empty in hate localization.

### 1.3 The two recon panels taken before this freeze (descriptive, val split, 3 seeds)

Reproduced so that nothing in this freeze can be re-derived after the fact.

**Ranking headroom at the actual operating point** — VAT detector, rawseg, val (39 videos,
531 gold), pool 200/video, model keeps 22.0/video:

| read-out | F1@0.5 |
|---|---|
| model at its own threshold | 48.76 ± 0.65 (P 39.54 / R 63.72) |
| oracle re-ranking, same per-video budget | 63.87 ± 0.59 |
| oracle re-ranking, budget = gold count | 77.97 ± 1.07 |
| oracle binary verifier | 96.83 ± 0.29 |
| pool recall @0.5 | 93.85 |

**Error composition** of the 857 kept proposals: matched 338 (39.5%), partial 0<tIoU<0.5 219
(25.5%), **zero overlap 300 (35.0%)**. Of 531 gold: matched 338, **missed although a ≥0.5
proposal was in the pool 160 (30.1%)**, missed and not in pool 33. Mean length of matched
proposals 9.8 s vs zero-overlap false alarms 9.9 s. Within-video Spearman with oracle tIoU over
the 200-proposal pool: **model score 0.350, proposal duration alone 0.423**.

---

## 2. Data, features, protocol

**Corpus / split.** HateClipSeg, 395 videos, frozen `data/gt/HateClipSeg/p11_split.json`
(237 train / 39 val / 119 test), unchanged. **Ground-truth convention: `rawseg`** throughout
(every offensive segment its own instance), matching R16's reproduction arm.

**The 119 test videos are not opened by either pilot.** Both pilots run entirely inside the
237 train videos, with the 39-video val split used only for epoch and threshold selection
exactly as R16 used it. Every generated annotation JSON must contain zero test ids; the runner
asserts this and prints the assertion.

**Feature channels.**

| tag | content | dim |
|---|---|---|
| `V` | 4 FPS CLIP-L/14-336 pooler output, `dense4fps_clipL336` | 1024 |
| `A` | wav2vec2-large-robust-emotion hidden states pooled over `[t−4s, t]`, `dense4fps_w2vemo` | 1024 |
| `T` | BERT-base CLS over ASR chunks overlapping `[t−2s, t]`, `dense4fps_bertbase` | 768 |
| `O` | **new**: BERT-base CLS over the PaddleOCR text of the K=30 window containing `t` | 768 |

The `O` channel is built from the **existing** OCR cache
`data/OCR/HateClipSeg/ocr_windows_K30.jsonl` (PaddleOCR v6, 11 850 windows, one frame sampled at
each window midpoint, 70.3% of windows non-empty, mean 128.7 characters when non-empty, 35 of
395 videos empty throughout). No new OCR extraction is run. **Declared limitation:** the OCR
temporal resolution is one reading per `duration/30` window, ≈ 7.6 s, against a median gold
segment of 8.4 s. The channel can therefore change *classification* but cannot sharpen
boundaries. A null on P1 is consequently **not** evidence that OCR is uninformative at finer
resolution, and the result text must say so.

Texts are lower-cased and concatenated in reading order with confidence ≥ 0.5, encoded once per
unique string with the same frozen `bert-base-uncased` and the same CLS read-out as the `T`
channel, and broadcast to every 4 FPS index inside their window. Empty windows are the zero
vector, matching the `T` channel's convention for empty ASR.

**Arms of P1** (three materialised feature directories, identical in every other respect):

| arm | input | dim |
|---|---|---|
| `VAT` | V ⊕ A ⊕ T — **the contrast line** | 2816 |
| `VATO` | V ⊕ A ⊕ T ⊕ O | 3584 |
| `VATO_SHUF` | V ⊕ A ⊕ T ⊕ shuffle(O) — the O rows of each video randomly permuted **within that video** (permutation drawn once per video from a fixed seed, identical across arms' seeds and folds) | 3584 |

`VATO_SHUF` holds width, per-video OCR content and marginal statistics fixed and destroys only
the timing. It is the round-11 circular-shift control in its width-matched form.

**Cross-fitting.** The 237 train videos are partitioned into 3 folds by sorted video id modulo 3
(deterministic, no seed). For each (arm, seed, fold): train on the other two folds (158 videos),
select epoch and score threshold on the 39-video val split by F1@tIoU0.5 exactly as
`run_af.py` does, then predict on the held-out fold. The union of the three held-out folds is
the 237-video out-of-fold prediction set for that (arm, seed).

**Seeds.** 6200 / 6201 / 6202 for detector training (avoiding every consumed range). Bootstrap
seed 6299. Reranker seeds 6210 / 6211 / 6212. Within-video OCR shuffle seed 6280.

**Everything else is R16's configuration byte for byte**: `configs/hateclipseg_clip.yaml`,
30 epochs, the same optimiser, EMA, NMS and `max_seg_num: 200`. No hyper-parameter is tuned in
this round, in any arm.

---

## 3. P1 — dense OCR as a fourth early-fusion channel

**Primary endpoint.** F1@tIoU 0.5 computed by `scripts/r16_detbase/eval_f1.py:match_prf` over
the pooled 237 out-of-fold videos, per (arm, seed).

**Primary contrast.** `Δ1 = F1(VATO) − F1(VAT)`, averaged over the 3 seeds.

**Uncertainty.** Video-clustered paired bootstrap over the 237 out-of-fold videos, 10 000
resamples, seed 6299, recomputing the corpus-level F1 inside each resample and pooling seeds.

**δ = +1.5 F1 points.** Justification, fixed in advance: R16's whole audio+text early-fusion
step was worth +3.8, and its 3-seed sd on test was 0.73; a single additional channel at 7.6 s
resolution is priced at under half the two-channel step, and +1.5 is above two seed sd.

**Decision rule (frozen).**

- **PASS** iff `Δ1 ≥ +1.5` **and** the 95% bootstrap lower bound on `Δ1` is `> 0`.
- **KILL** otherwise.

**Secondary, reported whatever P1 decides.**

- `Δ2 = F1(VATO) − F1(VATO_SHUF)`, same bootstrap. If `Δ1` passes but `Δ2`'s CI contains zero,
  the written conclusion is that **the OCR gain is not moment-level** — it is video-level
  identity information entering through a wider input — and it may not be reported as a
  localization mechanism.
- Proposal-pool recall @tIoU 0.5 per arm, on the same out-of-fold set. Pre-committed reading:
  recall rises ⇒ OCR improves proposal generation; recall flat but F1 rises ⇒ OCR improves
  point classification / ranking; neither ⇒ dense OCR is inert at this resolution.
- F1 at tIoU 0.3 and 0.7, descriptive.

**What a KILL means, pre-committed.** A KILL on P1 does **not** close the OCR evidence line. Per
the reviewer's outcome table it makes the extent-conditioned hypothesis *more* interesting —
sparse 7.6 s-resolution OCR may be diluted inside a dense 4 FPS fusion and still be usable when
conditioned on a proposed extent — and P2 is the pre-registered test of exactly that. What a KILL
does close is the claim that a fourth dense channel is worth anything on this base.

---

## 4. P2 — extent-conditioned proposal re-ranking

Runs on P1's dumped out-of-fold proposals. **No additional detector training.** Pool = the full
200 proposals per video emitted by the `VAT` arm, before thresholding.

**Nested cross-fitting.** For held-out fold f, the re-ranker is fitted on the out-of-fold
proposals of the other two folds and applied to fold f. Every arm sees exactly the same
partition, the same proposals and the same capacity budget (one hidden layer, 256 units, 40
epochs, Adam 1e-3, identical in every arm; only the input vector differs). Target: binary,
1 iff the proposal is matched at tIoU ≥ 0.5 by the greedy score-ordered matcher against that
video's gold, computed on training folds only.

**Arms.**

| arm | ranking signal |
|---|---|
| `R0` | the detector's own proposal score (the baseline) |
| `R1` | proposal duration alone, through the same head (the reviewer's mandatory geometry control) |
| `R2` | learned geometry: `[detector score, duration, centre/video duration, start, end, count of pool proposals overlapping it at tIoU ≥ 0.5]` |
| `R3` | `R2` ⊕ mean-pooled V/A/T features over `[s, e]` ⊕ the same over the two 50%-length context rings on either side |
| `R4` | `R3` ⊕ mean-pooled `O` over `[s, e]` |
| `R5` | `R3` with the pooled content block permuted **within video, within duration decile** (the reviewer's content-permutation control) |

**Endpoint.** F1@tIoU 0.5 keeping exactly the **top 22 proposals per video** by the arm's score,
pooled over the 237 out-of-fold videos and averaged over the 3 detector seeds × 3 reranker seeds.
The fixed per-video budget isolates ranking from count selection, and 22 is the mean count the
R16 threshold already produces (recon §1.3) — it is not selected in this round.

**Decision rules (frozen).**

- **G1 (primary).** `R3 − max(R0, R2) ≥ +2.0` F1 with a video-clustered bootstrap 95% lower
  bound `> 0`. This is the reviewer's number, adopted verbatim.
- **G2.** `R4 − R3 ≥ +1.0` with 95% LCB `> 0` ⇒ extent-conditioned OCR adds beyond V/A/T at the
  verifier stage.
- **G3 (mechanism check, applies only if G1 passes).** At least **75%** of `R3 − R0` must
  disappear under `R5`, i.e. `R5 − R0 ≤ 0.25 × (R3 − R0)`. If it does not, the gain is
  attributable to the geometry and pooling arrangement rather than to span content, and G1 is
  overridden to a fail.
- **KILL** if G1 fails: the extent-conditioned span-verifier family is closed on this base and
  the round reports no surviving candidate.

---

## 5. Blindness, single submission, and what is not claimable

- No metric of any arm of P1 or P2 has been computed. The only pre-freeze measurements are the
  two ceilings/geometry panels in §1.3, neither of which is an arm of either pilot.
- Each pilot is **one submission**. A crash is re-run only after the crash log is kept and the
  deviation is recorded; no design element may change on a re-run.
- The 119 test videos are opened by neither pilot. If both gates pass, the frozen system would be
  retrained on all 237 and evaluated on test exactly once, in a separate round with its own
  freeze; this round does not do that and may not report a test number.
- **K1 stands.** Nothing in P2 changes the decode; the per-video budget is held fixed at 22 in
  every arm precisely so that no arm can win by decoding differently.
- **The `rawseg` boundary caveat.** R16 recorded that `rawseg` boundaries are Whisper sentence
  boundaries. Both pilots are *paired* contrasts on the identical ground truth, so the artifact
  cancels; no arm can gain from it that the others cannot. This freeze therefore does not run an
  audio-onset control, and correspondingly **no absolute localization-quality claim may be made
  from this round** — only the paired differences.
- **Novelty is not settled by these pilots.** The reviewer's standing judgment is that even a
  positive P1 is "we forgot OCR, then added OCR" and not a method contribution; a positive P2
  with G3 satisfied is the only outcome that opens a mechanism claim, and that claim would still
  need the concentration analysis (gain concentrated on OCR-only misses) before it is written up.

## 6. Sentence a double null forces, committed in advance

> If P1 kills and P2 kills, then on the detector base as well as on the per-window base there is
> no live mechanism family for hateful-video temporal localization under this project's
> constraints, the OCR evidence line is exhausted at the resolution the existing cache supports,
> and the sub-direction is handed back to the user as a scope question — not quietly
> re-attempted with a fourth substrate.
