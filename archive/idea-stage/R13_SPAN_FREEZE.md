# R13-SPAN — kill probe freeze (Round 11, temporal span sub-direction)

**Frozen 2026-08-18, before any arm metric was computed.** Author: round-11 executor.
Cost class: CPU/$0 probe (one short GPU feature-extraction pass for CLIP text encoding, no training).
Ceremony per user ruling 2026-08-05: ≤1 h experiments get at most one review round.

---

## 1. Question

Do the **released gold hate spans** carry information that improves video-level hate
classification, *beyond the amount of video the crop removes*?

This is the ceiling question for the whole "localize → trim → re-classify" family
(`research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` §7.4) and for the two train-time
span-supervision candidates (gold-span crop augmentation, complement-region negatives).
If the **oracle** span location buys nothing over a duration-matched random crop, no
*predicted* boundary can, and the family closes at zero GPU cost.

## 2. Data, and the test-set red line

- Arena: **HateMM train split only** (744 videos, 298 positive), evaluated by 5-fold
  out-of-fold prediction **inside train**. Secondary arena: MHC-EN train (549) and MHC-ZH
  train (579), restricted to videos whose `Duration` field is non-empty.
- **No test or dev/val file is opened.** `data/gt/HateMM/hate_spans.json` contains all 1083
  videos including test; the loader must filter to `data/gt/HateMM/train.jsonl` ids and
  assert that no test/val id survives.
- Gold spans are used as *inputs to a transformation*, never as evaluation labels.

## 3. Arms

Every arm produces exactly one key vector per video, so a single evaluation applies to all.
Two channels are built independently and also concatenated:

- **visual** = mean of the `subclipK30` CLIP image features whose window overlaps the kept
  interval (`data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt`,
  window bounds from `src/utils/generate_segment_asr_HF.window_time_bounds(duration, M=120, K=30)`).
- **text** = CLIP text embedding of the transcript restricted to the kept interval, built from
  the word-level `chunks` in `data/ASR/HateMM/train_asrK30_whisper-large-v3.jsonl` (a word is kept
  iff its chunk midpoint falls in the kept interval), encoded with the project's existing
  `encode_text` chunk-mean pipeline.

| arm | kept interval, positives | kept interval, negatives |
|---|---|---|
| **P0 FULL** | whole video | whole video |
| **P1 GOLD** | the released hate span (union of intervals) | one random interval, coverage drawn from the positives' coverage distribution |
| **P2 RAND** | one random interval whose length equals **that same video's own gold coverage**, random position | identical draw to P1 (same seed) |
| **P3 COMP** | the complement of the gold span | the complement of the P1 negative interval |
| **P4 PRED** | top-m windows by the banked K=4 MLLM hate score, m chosen so coverage matches that video's gold coverage | same rule, coverage drawn as in P1 |

P2 is the control that isolates span **location** from crop **duration**. P1 and P2 differ *only*
in where the positive's crop sits; negatives are byte-identical between P1 and P2.

Randomisation: 20 independent draws, **seeds 2000-2019** (outside every reserved range:
0-119, 400-429, 500-529, 600-629, 700-729, 1300-1524, 41000-41029). Every arm's reported
metric is the mean over the 20 draws; P0/P1's positive side is deterministic.

## 4. Evaluation

- 5-fold stratified out-of-fold logistic regression (scikit-learn, `C=1.0`, `max_iter=2000`,
  L2, fold seed **2020**), on L2-normalised keys.
- **Primary metric: OOF ROC-AUC.** Threshold-free, stable at n=744.
- Secondary: OOF macro-F1 at 0.5, and leave-one-out kNN(k=20) vote macro-F1 (the project's
  standard $0 probe read-out).
- Paired bootstrap over videos, 10 000 resamples, seed 2021, on the paired OOF score vectors;
  report the 95% CI of each Δ.

## 5. Decision rule — frozen

Primary judgement is on **HateMM, concatenated visual⊕text channel**.

- **Δ₁ = P1 − P2** (gold span vs duration-matched random crop). This is the load-bearing test.
- **Δ₂ = P1 − P0** (gold-span trim vs no trim).

**KILL the span-supervision family** iff the 95% CI of **Δ₁ contains zero** on HateMM.
In that case: gold-span crop augmentation (A1), complement-region negatives (A2),
predicted-boundary trimming (A3), synthetic span supervision (A5) and cross-dataset boundary
transfer (B1) are all closed at once, because each of them is strictly weaker than the oracle
this arm measures.

**PROCEED to a GPU pilot** iff Δ₁ > 0 with the 95% CI excluding zero on HateMM **and** the
point estimate is positive on at least one MHC split. The pilot would be gold-span crop
augmentation through `src/run_rac.py`, pre-registered separately.

**Ambiguous** (CI excludes zero on HateMM but sign disagrees across MHC splits) → record as
one-dataset-only, no escalation, consistent with this project's ≥2-dataset standing bar.

Δ₂, P3 and P4 are **descriptive only** and cannot change the verdict. They are recorded to
explain *why* the verdict came out as it did:
- Δ₂ prices how much information trimming destroys.
- P3 (complement) prices whether the non-span region is separable from the span region.
- P4 prices the realistic, non-oracle version against the same control.

Also recorded, no decision weight: the distribution of `cos(key_P0, key_P1)` per video, which
the project's prior measurements predict sits near 0.95.

## 6. Blindness

No arm metric has been computed at the time of this freeze. The implementation subagent is
instructed to write results to `idea-stage/r13_span/r13_span.json` and to print them once,
after all arms are complete. No arm may be inspected and then re-specified.

## 7. What this probe cannot decide

It says nothing about **HateClipSeg**, whose span statistics (coverage 0.544, 22% single-block,
median 3 toxic blocks) are qualitatively different from HateMM's (0.806 / 72.8% / 1). A kill
here is a kill for the four in-scope datasets, not for HateClipSeg, which is separately pending
a user ruling.
