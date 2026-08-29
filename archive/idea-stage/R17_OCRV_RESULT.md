# R17-OCRV — result

**Date** 2026-08-18 · **Freeze** `idea-stage/R17_OCRV_FREEZE.md`, commit **`1e268c6`**, committed
before `scripts/r17_ocrv/` existed · **Cost** ¥0 (no cloud, no paid API, no annotation), local
RTX 5090 · **Wall** OCR feature build 7 s, P1 27 detector runs **5 866 s**, P1 analysis 40 s,
P2 **198 s** · **Seeds** detector 6200/6201/6202, re-ranker 6210/6211/6212, bootstrap 6299,
OCR shuffle 6280 · **Test contact: zero** — every number below is out-of-fold inside the 237
train videos; the 119-video test split was never loaded (asserted per fold, printed in `run.log`).

## VERDICT: **both pre-registered gates fail. P1 KILL, P2 KILL.**

| pilot | frozen contrast | result | bar | verdict |
|---|---|---|---|---|
| **P1** | `D1 = VATO − VAT` | **−0.16 [−1.34, +0.97]** | ≥ +1.50 and LCB > 0 | **KILL** |
| **P2** | `G1 = R3 − max(R0, R2)` | **−7.38 [−8.60, −6.40]** | ≥ +2.00 and LCB > 0 | **KILL** |
| P2 | `G2 = R4 − R3` | −0.13 [−0.49, +0.22] | ≥ +1.00 | fail |
| P2 | `G3` | not evaluated (G1 failed) | — | — |

Under §6 of the freeze, committed before any number existed, a double null forces this sentence:

> On the detector base as well as on the per-window base there is no live mechanism family for
> hateful-video temporal localization under this project's constraints, the OCR evidence line is
> exhausted at the resolution the existing cache supports, and the sub-direction is handed back to
> the user as a scope question — not quietly re-attempted with a fourth substrate.

---

## 1. P1 — dense on-screen text as a fourth early-fusion channel

Three arms, identical in every respect except the input feature file. 3-fold cross-fitting inside
the 237 train videos (158 train / 39 val for epoch+threshold selection / 79 held out), 3 seeds,
27 ActionFormer runs. Endpoint: corpus-level F1@tIoU 0.5 over the pooled 237 out-of-fold videos.

| arm | input | dim | **F1@0.5** | F1@0.3 | F1@0.7 | P | R | n_pred |
|---|---|---|---|---|---|---|---|---|
| **`VAT`** (contrast) | V ⊕ A ⊕ T | 2816 | **35.60 ± 0.15** | 46.96 ± 0.73 | 16.84 ± 0.73 | 28.21 | 48.56 | 4 721 |
| **`VATO`** | V ⊕ A ⊕ T ⊕ OCR | 3584 | **35.44 ± 0.13** | 47.68 ± 0.82 | 15.60 ± 0.46 | 28.67 | 46.44 | 4 428 |
| **`VATO_SHUF`** | V ⊕ A ⊕ T ⊕ shuffle(OCR) | 3584 | **36.44 ± 1.13** | 47.71 ± 0.74 | 16.88 ± 0.74 | 27.96 | 52.32 | 5 112 |

Per-seed: `VAT` 35.59 / 35.78 / 35.42; `VATO` 35.40 / 35.62 / 35.30; `VATO_SHUF` 34.84 / 37.13 /
37.35.

**Contrasts** — video-clustered paired bootstrap over the 237 out-of-fold videos, 10 000
resamples, seed 6299, corpus F1 recomputed inside each resample, seeds pooled:

| contrast | Δ | 95% CI | reading |
|---|---|---|---|
| `D1` **VATO − VAT** | **−0.16** | [−1.34, +0.97] | **KILL**; the CI excludes the +1.5 bar and contains zero |
| `D2` VATO − VATO_SHUF | −1.00 | [−2.03, +0.01] | correctly timed OCR is, if anything, *worse* than temporally shuffled OCR |
| `D3` VATO_SHUF − VAT | +0.84 | [−0.15, +1.88] | descriptive; the width-only variant is the best of the three and still does not clear the bar |

**Proposal-pool recall @tIoU 0.5** on the full 200-proposal pool, the pre-committed diagnostic:

| arm | pool recall |
|---|---|
| `VAT` | 90.79 ± 0.22 |
| `VATO` | 90.68 ± 0.64 |
| `VATO_SHUF` | 91.45 ± 0.55 |

The freeze pre-committed the reading: recall rises ⇒ OCR improves proposal generation; recall flat
and F1 rises ⇒ OCR improves point classification; **neither ⇒ dense OCR is inert at this
resolution**. Recall is flat to within 0.11 and F1 does not rise. The third branch fires.

**The `D2` row is the informative one and it is not a null.** `VATO_SHUF` destroys the temporal
alignment of the OCR rows within each video while preserving the width, the video's own OCR
content and every marginal statistic. It scores **1.00 F1 above** the correctly aligned channel,
with a CI whose upper end just touches zero. Whatever the OCR stream contributes on this substrate,
**it is not moment-level**, and forcing the detector to attend to it at the right moments costs
more than it buys. This is the same shape as round 13's finding for the visual channel, arrived at
independently on a completely different architecture.

**Two honest limitations, both pre-declared.**

1. **Resolution.** The channel is built from the existing K=30 cache: one PaddleOCR reading at the
   midpoint of each `duration/30` ≈ 7.6 s window, 70.3% of windows non-empty, 35 of 395 videos
   empty throughout. That is coarse relative to a 4 FPS visual grid, though comparable to the 8.4 s
   median gold segment. **This result does not show that OCR is uninformative at finer temporal
   resolution.** It shows that the resolution the project already owns buys nothing.
2. **Paired contrast only.** The `rawseg` boundaries are Whisper sentence boundaries; all three
   arms share that artifact exactly, so it cancels in `D1`/`D2`/`D3` — and correspondingly no
   absolute localization-quality claim is made from this round.

## 2. P2 — the extent-conditioned span verifier

On the `VAT` arm's own out-of-fold 200-proposal pools. Nested cross-fitting, identical head
(one 256-unit hidden layer, 40 epochs, Adam 1e-3) and identical partition in every arm; only the
input vector differs. Endpoint: F1@tIoU 0.5 keeping exactly the top 22 proposals per video, pooled
over 237 out-of-fold videos, averaged over 3 detector seeds × 3 re-ranker seeds (9 cells).

| arm | ranking signal | input dim | **F1@0.5** |
|---|---|---|---|
| `R0` | the detector's own proposal score | — | **34.20 ± 0.41** |
| `R1` | proposal duration alone | 1 | **24.28 ± 0.76** |
| `R2` | learned geometry (score, duration, centre, start, end, overlap count) | 6 | **34.21 ± 0.26** |
| `R3` | `R2` ⊕ extent-pooled V/A/T ⊕ two context rings | 8 454 | **26.83 ± 0.29** |
| `R4` | `R3` ⊕ extent-pooled OCR | 9 222 | **26.71 ± 0.43** |
| `R5` | `R3` with the content block permuted within video, within duration decile | 8 454 | **28.27 ± 0.44** |

| gate | contrast | Δ | 95% CI | bar | verdict |
|---|---|---|---|---|---|
| **G1** | `R3 − max(R0, R2)` (= `R2`) | **−7.38** | [−8.60, −6.40] | +2.00 | **FAIL** |
| **G2** | `R4 − R3` | −0.13 | [−0.49, +0.22] | +1.00 | **FAIL** |
| G3 | — | not evaluated | | | G1 failed |

**Four readings, in order of how much they close.**

1. **The duration prior does not transfer to the operating point, and the reviewer predicted this
   in advance.** The recon that motivated the round measured within-video Spearman with oracle
   tIoU of **0.350 for the detector score against 0.423 for duration alone** over the whole
   200-proposal pool. Ranking by duration at the actual top-22 budget scores **24.28** against the
   score's **34.20** — nearly ten points worse. The reviewer's objection was exactly right:
   *"Spearman over all 200 proposals is dominated by easy pool-tail geometry and does not measure
   top-22 one-to-one matching."* **The ρ inversion is a real property of the pool tail and a
   worthless guide to the head of the ranking.** Recorded so that no future round re-motivates a
   candidate from it.

2. **A learned head on geometry plus the score reproduces the score and nothing more.** `R2`
   34.21 against `R0` 34.20. Six geometric features, cross-fitted, with the detector's score
   available as an input, add **+0.01 F1**. There is no free geometric structure left in the
   proposal set.

3. **Adding extent-pooled span content makes the ranking dramatically worse, and it is worse than
   its own permutation control.** `R3` 26.83 against `R0` 34.20 is −7.4; and `R5`, which is `R3`
   with the content block permuted within video and within duration decile — **identical
   dimensionality, identical capacity, identical optimisation** — scores **28.27**, i.e. **1.44
   points above the real content**. Real span content is not merely uninformative here; under a
   matched-capacity comparison it generalises *worse than noise*, which is the signature of the
   head fitting video-specific content that does not transfer across folds.

4. **OCR at the verifier stage adds nothing either**: `G2` = −0.13 [−0.49, +0.22]. The
   stage-placement hypothesis the reviewer named as the only route to a method claim — *"sparse
   OCR is diluted or temporally misassigned in dense fusion but works when conditioned on a
   proposed extent"* — is tested and does not hold.

**The limitation that must be stated, because it is the one a reviewer would raise.** `R3`/`R4`/
`R5` feed an 8 454- to 9 222-dimensional vector into a 256-unit hidden layer fitted on ~31 600
proposals from 158 videos. That capacity/dimensionality regime was fixed in the freeze and is
plainly part of why `R3` collapses. **The experiment therefore cannot separate "there is no usable
span-content signal" from "this head cannot use it".** What it *can* separate — because `R5`
matches `R3` on every axis except whether the content is the proposal's own — is that within this
regime the real content carries no transferable advantage over permuted content. A stronger claim
than that is not available from these runs, and none is made.

## 3. What this round establishes

1. **A fourth dense evidence channel does not move this detector.** The single best-evidenced
   modality gap in the project's own failure analysis (30.1% of localization misses with on-screen
   text as the only evidence, OR 2.29) is worth **−0.16 ± 1.1 F1** when supplied to the detector at
   7.6 s resolution, and its temporally shuffled control does slightly better.
2. **Neither of the two places a span scorer can be improved responds.** Inside the detector (P1)
   and as a second stage on its own proposals (P2), the answer is the same.
3. **The +15.1 oracle re-ranking headroom measured on val is not reachable by any arm run here** —
   and three of the six P2 arms are *below* the detector's own score, one of them by 7.4 points.
   An oracle ceiling remains what §6.10 of `RESEARCH_BRIEF.md` already said it is: the precondition
   every failed candidate met, not evidence for any of them.
4. **The occupancy sweep independently removed the novelty of the P2 family before it ran.**
   BREM `2204.11695` (ACM MM 2022) publishes this round's motivating diagnostic — swap a detector's
   classification score for the true tIoU and watch mAP jump — and builds Boundary-Evaluate and
   Region-Evaluate modules on it for +3.6 avg mAP. The diagnostic is four years old in generic TAL.
5. **The OpenAlex coverage gap carried since round 11 is now closed, and the answer is that
   OpenAlex is unusable here.** After the daily budget reset, `filter=cites:W6967194700` returns
   **0 citers** for HateClipSeg, against Semantic Scholar's 4. There is no second citation graph
   for this corpus; future absence claims must say so rather than promise an OpenAlex check.

## 4. Deviations

- **D1 — a plumbing smoke run.** Before the frozen run, `run_oof.py` was executed once with
  `--arms VATO --seeds 9998 --epochs 1` on a scratch output directory, to verify the fold plumbing
  end to end. It produced out-of-fold F1@0.5 = 16.88 from a 6-epoch untrained model on a single
  arm. **It touched no test video** (the fold JSONs mark all 119 as `unused`), it produced no
  contrast, and no threshold, epoch, arm, seed or decision rule was derived from it. Reported here
  rather than buried, following the project's rule that the cheapest check before submitting is the
  submission itself.
- **D2 — OCR resolution, declared in the freeze.** The `O` channel is built from the existing K=30
  cache (≈7.6 s per reading, one sampled frame per window), not from new dense extraction. §1
  limitation 1 states what this does and does not license.
- **D3 — input standardisation in P2, not specified by the freeze.** Every P2 arm z-scores its
  input using training-fold statistics before the head. This is arm-symmetric and was applied
  identically everywhere; the freeze fixed the architecture and optimiser but was silent on
  scaling, and an 8 454-dimensional unscaled input would not train at all.
- **D4 — the P2 target is computed over the full 200-proposal pool**, as the freeze states
  ("matched at tIoU ≥ 0.5 by the greedy score-ordered matcher"), which makes the label mildly
  dependent on the detector's own ordering: where two proposals both overlap one gold at ≥ 0.5,
  only the higher-scored one is positive. This is the literal reading of the frozen sentence and is
  identical across arms.
- **D5 — no crash, no re-run.** Each pilot was a single submission. `logging/runs/r17_p1/run.log`
  and `logging/runs/r17_p2/run.log` are the only run logs; there is no `run_attempt1_crash.log`.

## 5. Reproduction

| artifact | path |
|---|---|
| freeze (pre-code, commit `1e268c6`) | `idea-stage/R17_OCRV_FREEZE.md` |
| candidate slate, hostile scores, occupancy sweep | `idea-stage/R17_CANDIDATES.md` |
| review bundle | `idea-stage/codex_brainstorm_bundle_r17_2026-08-18.md` |
| OCR + fused feature builder | `scripts/r17_ocrv/build_ocr_feats.py` → `dense4fps_{ocrbert,vato,vato_shuf}/` |
| cross-fitting annotation JSONs | `scripts/r17_ocrv/make_fold_json.py` → `third_party/actionformer/data/hateclipseg/hateclipseg_rawseg_fold{0,1,2}.json` |
| P1 runner | `scripts/r17_ocrv/run_oof.py` → `idea-stage/r17_ocrv/out/{res_p1.json, preds_oof_*.json, pool_*.json}` |
| P1 frozen analysis | `scripts/r17_ocrv/analyze_p1.py` → `analysis_p1.json` |
| P2 panel | `scripts/r17_ocrv/run_p2.py` → `analysis_p2.json` |
| logs | `logging/runs/r17_{feats,p1,p2}/` |

`run_oof.py` asserts per fold that train / val / out-of-fold id sets are pairwise disjoint and that
their union contains no test id, and prints the split guard at the top of `run.log`.
