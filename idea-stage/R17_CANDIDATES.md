# R17 — candidate slate and hostile scoring (2026-08-18)

Round 14 of idea discovery. Sub-direction: **hateful video temporal localization**, on the
detector base built in round 16 (`idea-stage/R16_DETBASE_RESULT.md`), not on the per-window
score curve rounds 11-13 exhausted.

Reviewer: **gpt-5.6-sol at xhigh reasoning**, conversation only, no tool use, given the R16
result, the new recon in §1, the full constraint map and all thirteen standing closures. Bundle:
`idea-stage/codex_brainstorm_bundle_r17_2026-08-18.md`.

---

## 1. The recon that reframed the round (val split, descriptive, pre-freeze)

R16 §5.4 priced the ranking bottleneck with a fixed-small-k table, which conflates ranking
quality with a recall cap: at rawseg k=5 the model scores 17.5 and the oracle 42.6, but the
unconstrained system already scores 38.22, so the headline "20+ points" is not readable off that
table. Re-priced at the system's **own operating point** — VAT detector, rawseg, val, pool 200
proposals/video, model keeps 22.0/video at its val-selected threshold:

| read-out | F1@0.5 |
|---|---|
| model at its own threshold | **48.76 ± 0.65** (P 39.54 / R 63.72) |
| **oracle re-ranking, same per-video budget** | **63.87 ± 0.59** |
| oracle re-ranking, budget = gold count | 77.97 ± 1.07 |
| oracle binary verifier | 96.83 ± 0.29 |
| pool recall@0.5 | 93.85 |

**+15.1 F1 is available from within-video re-selection alone, at a fixed per-video count.** That
is the honest number; it is not a recall-cap artifact, and it is not 20+.

Error composition of the 857 kept proposals: matched 338 (39.5%), partial 0<tIoU<0.5 219 (25.5%),
**zero overlap 300 (35.0%)**. Of 531 gold: matched 338 (63.7%), **missed although a ≥0.5 proposal
was in the pool 160 (30.1%)**, missed and absent from the pool 33 (6.2%). Matched proposals mean
9.8 s, zero-overlap false alarms mean 9.9 s.

And the fact that shaped the whole review — within-video Spearman with oracle tIoU over the
200-proposal pool:

| ranking signal | ρ |
|---|---|
| the detector's own classification score | **0.350** |
| the proposal's duration alone | **0.423** |

A duration prior orders the pool better than the trained score does.

## 2. The twelve candidates and the hostile scores

| # | candidate | mechanism | score | load-bearing reason |
|---|---|---|---|---|
| E1 | **XPOOL** extent-conditioned span verifier | second stage pools dense V/A/T over each proposal's own extent plus a context ring, ranks by `cls × verifier` | **3** | run it as the main diagnostic, but "the premise is overstated: ActionFormer's point feature has a large temporal receptive field; proposal pooling gives explicit extent alignment, not previously unseen content. Two-stage RoI/context proposal verification is heavily occupied and is not a paper contribution by itself" |
| E2 | **IOUHEAD** quality-aware ranking | auxiliary tIoU-prediction head, rank by `cls × predicted IoU` | **3** | mandatory, and baseline-only: BMN's proposal-evaluation module, IoU-Net, GFL, VarifocalNet own it |
| E3 | **OCRDENSE** dense on-screen-text channel in the detector | 4th early-fusion stream | 2 | "best evidence-backed ingredient: it targets a documented missing-evidence population. Still just *add the omitted modality*, already telegraphed by LELA and neighbouring work" |
| E4 | **MODROLE** modality-asymmetric role assignment | boundary branch and classification branch see different modality subsets | 2 | "the only remotely fresh structural hypothesis. Unfortunately *audio owns boundaries* may simply exploit Whisper-generated speech-pause boundaries, while R16 already says fine boundary regression has little leverage" |
| E5 | **SPANRET** retrieval over labelled train spans | kNN against 10 572 labelled segments in extent-pooled space | **0** | the project already measured chance-level segment retrieval (0.5259 vs a 0.5252 broadcast control); pooling differently is not a reason to rerun a weak, non-novel family |
| E6 | **MLLMVERIFY** MLLM as comparative span verifier | rank the top-k proposals by an MLLM judgement | **0** | the 72B per-window result lost to a two-layer head; ¥15 cannot support reliable comparative evaluation; prompt-based reranking gives neither reproducibility nor novelty |
| E7 | **SETSEL** set-level selection | choose the kept set jointly | 1 | may suppress duplicates but cannot identify the 300 zero-overlap false alarms without a better quality signal; count prediction is fragile; DETR/TadTR-style set selection is occupied |
| E8 | **VIDPRIOR** video-level score conditioning | per-video multiplier on span scores | **0** | mathematically dead — a video-constant multiplier cannot alter within-video order, and §1 shows the headroom is entirely within-video |
| E9 | **HARDNEG** hard negatives from the detector's own zero-overlap false alarms | OHEM on 35% of kept outputs | 2 | sensible training hygiene aimed at a real population; ordinary OHEM and cannot carry a paper |
| E10 | **CONSIST** cross-modal agreement as span confidence | rank by inter-branch agreement | **0** | "agreement is the wrong inductive bias when valid hateful evidence is frequently modality-specific; it will confidently suppress OCR-only or speech-only positives" |
| E11 | **CTXNEG** complement-region negatives | mine normal stretches inside hateful videos as within-video negatives | 1 | ActionFormer already trains outside-GT locations as background; unless this yields a demonstrably different proposal-level objective it is renamed negative mining, occupied by CoLA / CPL / CNM / UniVTG |
| E12 | **DURPRIOR** duration prior | rank by duration | **3** | mandatory control, zero contribution — any E1/E2 gain reported without a learned duration/scale control is uninterpretable |

**Three 3s, none of them a paper.** Verbatim: *"No candidate, as currently stated, is a top-venue
method-paper candidate. E1/E2 are standard proposal-quality machinery; E3 is modality completion;
E9/E11 are training tactics; the rest are weak."*

## 3. Families the slate missed

- **Proposal completeness / boundary-consistency scoring** from start-end evidence and
  inside-versus-outside contrast — "thoroughly occupied by BSN/BMN/DBG and later two-stage TAL".
- **One-to-one set prediction** (TadTR/DETR-style TAL) and **listwise proposal ranking** (generic
  ranking losses). Neither is a clean opening.

## 4. Is the +15.1 real, and does ρ(duration) > ρ(score) kill E1/E2?

The reviewer's answer, condensed and adopted:

- The headroom is a **real conditional oracle**, not a recall-cap artifact: pool and per-video
  output count are held fixed and only selection changes. It is **not** evidence that 15.1 is
  learnable — the oracle uses gold tIoU, perfect duplicate awareness, and a 39-video split that
  has now been inspected repeatedly. Treat it as an upper bound.
- The duration result does **not** kill E1/E2, for three stated reasons: Spearman over all 200
  proposals is dominated by easy pool-tail geometry and does not measure top-22 one-to-one
  matching; at the operating point matched and zero-overlap proposals have essentially identical
  mean length (9.8 vs 9.9 s), so duration cannot explain those false alarms; duration may explain
  extent quality but not length-matched, zero-overlap semantic errors.
- **The discriminator**, adopted as P2's arm set: compare the verifier against a capacity-matched
  geometry model, then permute span content among same-video, same-duration-bin proposals. If
  geometry matches the verifier, the opening was geometric. If content still separates matched
  from zero-overlap proposals after duration matching, it is semantic.

## 5. The reviewer's ordering ruling, and what went to pilot

Asked whether to spend the 2-hour budget on the reranker panel or on a detector-level modality
test first, the reviewer chose the detector test, unprompted and with the reason stated:

> "Comparing VAT against VAT+OCR-verifier is unfair: only one system can see OCR. The decisive
> comparison is compute- and encoder-matched. … Run the VAT versus VATO detector comparison
> first. It addresses the cheapest and most damaging alternative explanation. A positive reranker
> result is not publishable until direct OCR fusion has been tested; every reviewer will ask that
> control immediately."

and pre-committed the reading of each outcome: pool recall rises ⇒ OCR improves proposal
generation; recall flat and F1 rises ⇒ OCR improves point classification; neither ⇒ dense OCR is
inert **and extent-conditioned OCR becomes more interesting, because temporal dilution is then a
supported hypothesis**.

Both pilots are therefore pre-registered together in `idea-stage/R17_OCRV_FREEZE.md`, committed
before `scripts/r17_ocrv/` existed:

- **P1** — three arms (`VAT`, `VATO`, `VATO_SHUF`), 3-fold cross-fitting inside the 237 train
  videos, 3 seeds, 27 detector runs, primary `Δ1 = VATO − VAT` on out-of-fold F1@tIoU0.5,
  δ = +1.5, video-clustered paired bootstrap.
- **P2** — six re-ranking arms on P1's own out-of-fold proposals at a fixed 22-proposal budget,
  including the duration control and the within-video duration-matched content permutation, with
  the reviewer's +2.0 gate and the 75%-permutation-collapse requirement.

## 6. The reviewer's judgment on the paper question, recorded in full

On whether a decomposition-plus-system paper is available:

> "Proposal coverage is largely solved; selection fails because the detector lacks particular
> evidence channels, with duration-controlled decomposition identifying OCR-only errors. That
> could be a useful analysis or application-system paper. It is not a method paper. … The
> submission applies an established detector, adds a known modality, and provides a
> dataset-specific error decomposition. The analysis is useful, but methodological novelty is
> limited. State-of-the-art performance is an outcome, not a contribution."

and on what would qualify:

> "To qualify, the paper needs a stage-placement mechanism — e.g. proposal-conditioned
> sparse-evidence verification that demonstrably beats direct fusion — not merely a detector plus
> an explanation of its errors."

That sentence is the round's actual target, and it is exactly the P1-kills-P2-passes cell of the
frozen design.
