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

## 5b. Occupancy sweep (zero-GPU agent, run in parallel with the pilots)

Seven questions, arXiv API + Semantic Scholar + WebSearch. **Reachability, recorded first:**
OpenAlex is now **credit-metered and returned `Insufficient budget … you only have $0 remaining`
on every call**, so the OpenAlex citer cross-check on HateClipSeg asked for since round 11 is
**still not done** — and the SafeLens precedent proves Semantic Scholar alone misses citers.
Semantic Scholar's `/paper/search` returned 429 for most of the session; the citations and
references endpoints worked. WebSearch hit its 200-call session cap. CVF openaccess 403s.
arXiv's Atom API works only over **https with a User-Agent**; plain `http://` returns a 301 that
silently yields an empty body — which ate three queries before it was noticed, and is a plausible
cause of some earlier rounds' "zero hits".

| # | question | rating | firmest occupant |
|---|---|---|---|
| Q1 | proposal re-ranking / proposal-relation modelling in TAL | **d** | P-GCN `1909.03252` (ICCV 2019) and its T-PAMI generalisation `2112.00302`; TCANet `2103.13141` (CVPR 2021, proposal *refinement*); BSN++ `2009.07641`; ContextLoc `2107.12960`; and **GAP `2211.14924` (CVPR 2023)** — a model-agnostic post-hoc module on a *frozen* off-the-shelf detector, worth **+0.2 to +0.7 avg mAP**. No 2025-26 paper's contribution is proposal re-ranking |
| Q2 | retrieval / kNN / memory-based span scoring | **b** | VideoPatchCore `2409.16225`, CKNN `2408.03014`, RSKP `2203.02925`, AUMN `2104.14135`, SlowFastVAD `2504.10320` (detector → route ambiguous segments → RAG-VLM → fuse). MemAE `1904.02639` / MNAD `2003.13228` are *reconstruction* memories, a clean distinction. Scoring spans by kNN against a **labelled bipolar span memory** is empty everywhere checked — but it is a port of RGCL's instance-level kNN vote |
| Q3 | MLLM/VLM as verifier over external proposals | **c** | **OSGNet + MLLM Reranking `2605.20818`** — Ego4D EM Challenge 2026, *first place* in NLQ and GoalStep, verbatim "obtain a set of candidate segments from existing localization model … then employ MLLM to select the segment that best matches", motivated by preserving candidate recall. Also TFVTG `2408.16219` (ECCV 2024), F2G `2605.21973` (ICML 2026), TimeProVe `2606.20561`, FreeZAD `2501.13795` |
| Q4 | multimodal fusion **inside** a TAL detector | **c**, **b** for one sliver | "Hear Me Out" `2106.14118` owns the audio-helps-vs-hurts question (audio-only THUMOS 4.73 mAP; proposal-level late fusion **collapses** 56.16 → 39.37); MRAV-FF `2310.03456` owns the gated audio-per-FPN-scale contribution; DEL `2506.23196`, UniAV `2404.03179`, UnAV-100 `2303.12930`; ActionVLM `2601.21078` (Jan 2026) owns "estimate the language advantage and reweight". **ASR/subtitle text fused inside a TAL head was not found** — that is the b sliver |
| Q5 | detector-based harmful-content temporal localization | **a** | **Nothing.** `(abs:"hateful" OR abs:"harmful" OR abs:"toxic") AND abs:"temporal localization"` returns 2 entries, one irrelevant. The incumbents are weakly-supervised MIL (MultiHateLoc `2512.10408`, WWW 2026, on HateMM + MultiHateClip, **not** HateClipSeg) and training-free LLM (LELA `2602.09637`, which already uses OCR). Semantic Scholar returns the same 4 citers of `2508.01712` as round 11, none new; MultiHateLoc, TANDEM `2601.11178`, LELA and SafeLens `2605.17610` do not cite HateClipSeg at all |
| Q6 | video-level prior conditioning a span score | **c** | Ships as plumbing since 2019: BMN `1907.09702` multiplies UntrimmedNet's video-level class score by the proposal confidence; BaS-Net `1911.09963` gates classes on video-level probability. RUBi / Learned-Mixin *delete* the prior at inference where we would inject it — a framing difference over identical arithmetic. **Open item: Tan et al., WACVW 2024, "Overlooked Video Classification in Weakly Supervised VAD" (XD-Violence 78.84 → 82.10) is the closest named occupant and could not be fetched (CVF 403).** |
| Q7 | boundary-quality / IoU-aware confidence | **d** | **BREM `2204.11695` (ACM MM 2022) runs this round's headline diagnostic and builds its method on it** — replace the classification score with the true tIoU, watch mAP jump, conclude the classification score cannot represent localization quality, then add Boundary-Evaluate and Region-Evaluate modules (+3.6 avg mAP). Plus ReAct `2207.07097` ("segment quality prediction"), TadTR `2106.10271` (actionness regression), ALQA `2407.07673` (ECCV 2024), Centre Stage `2311.16446`, CLTDR `2412.09202`; and IoU-Net `1807.11590` / GFL `2006.04388` / GFLv2 `2011.12885` / VarifocalNet `2008.13367` settled it in object detection in 2018-2021 |

**Three consequences, and they are not small.**

1. **Q7 kills E1/E2 as method novelty outright.** The round's motivating measurement — an oracle
   ranking of a detector's own proposals beats its classification score — is BREM's opening
   paragraph, and BREM's fix is E2. ActionFormer genuinely has no quality head, so bolting one on
   will probably work; that is engineering, not a contribution. P2 therefore runs as a
   *diagnostic* with its frozen gates intact, and no positive P2 result may be written up as the
   discovery of the ranking gap.
2. **Q1 kills the post-hoc-re-ranker-on-a-frozen-detector framing**, and prices it: GAP is the
   precedent and it bought +0.5 mAP.
3. **Q3 kills the MLLM-verifier route** the slate had already scored 0 on cost grounds — a
   CVPR 2026 challenge was won with that exact topology and that exact motivation.

**What survives is Q5 and the Q4 sliver**, which is what P1 tests: a detector-based localizer
for harmful content is unoccupied, and ASR/OCR text fused *inside* the detector head was not
found anywhere. Neither is a mechanism; both are the slot P1 occupies if it passes.

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
