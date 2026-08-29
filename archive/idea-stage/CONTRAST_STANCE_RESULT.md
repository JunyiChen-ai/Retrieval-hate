# CONTRAST_STANCE_RESULT — does a symmetric pinned-comment comparison rescue the stance judgement?

**Verdict: FAIL, and it is the worst of the three mechanisms.** Replacing the 5-way stance
classification with a symmetric two-alternative forced choice — *"which of these two sentences
would the publisher pin under the video?"* — gives S-bucket accuracy **0.469** against a
pre-registered bar of **0.70**. Its own chance baseline is **0.50**, so the mechanism does not
beat a coin (exact binomial **p = 0.860**). On the same 32 rows, binarising the two earlier
rounds' answers gives **0.500** = 16/32 (round 1) and **0.563** = 18/32 (masked). **Forcing the choice made the
measurement worse, not better.**

The reason is unambiguous and is visible in one line: over the 98 scored items the model answered
**ENDORSE 90 times and OPPOSE 8 times**. The mechanism was designed to remove a response bias by
making the options symmetric; instead it removed the model's only escape hatch and routed
essentially everything to `endorses`.

Run date 2026-08-13 (Pacific/Auckland). Rules frozen in
`idea-stage/MASK_STANCE_PILOT_FREEZE.md` **Appendix B** before any API call of this pilot,
including before the smoke pass. Same 99-item sample (seed 20260811), same gold, same model
(`qwen3-vl-plus`), same 8-frame / 512 px / full-transcript input spec as
`STANCE_PILOT_RESULT.md` and `MASK_STANCE_PILOT_RESULT.md`. Raw outputs in
`idea-stage/contrast_stance/`. **Nothing downstream was built; the pilot stops at these numbers.**

---

## 1. Headline

| criterion | value | n | bar | pass |
|---|---|---|---|---|
| **M1** binary stance accuracy on S, view A, variant 1, smoke items excluded | **0.469** (15/32) | 32 | ≥ 0.70 | **no** |
| **M2** `CTRL_HATE` answered OPPOSE | **0.000** (0/18) | 18 | ≤ 0.15 | yes |
| **verdict (M1 ∧ M2)** | | | | **FAIL** |

M1 exact binomial against two-way chance 0.50: **p = 0.860**. The result is statistically
indistinguishable from flipping a coin, and its point estimate is *below* 0.50.

M2 passes for exactly the reason it passed in both previous rounds: **the model almost never
assigns a distancing stance to anything.** It is a pass by degeneracy, not by discrimination.

### 1.1 The three mechanisms, same rows, both口径

The task asked for a like-for-like comparison, and the 口径 difference is real: this round is a
**two-way** task (chance 0.50) while the two earlier rounds were **five-way** (chance 0.20). Both
readings were frozen in advance (Appendix B.6 and B.10 R2) and both are given.

| method | task | chance | acc on the same 32 S rows | margin over its own chance |
|---|---|---|---|---|
| round 1 — direct classification | 5-way | 0.20 | 0.281 (9/32) | +0.081 |
| round 2 — masked classification | 5-way | 0.20 | 0.375 (12/32) | +0.175 |
| round 1 — binarised | 2-way | 0.50 | 0.500 (16/32) | 0.000 |
| round 2 — binarised | 2-way | 0.50 | 0.563 (18/32) | +0.063 |
| **round 3 — pinned-comment contrast** | **2-way** | **0.50** | **0.469 (15/32)** | **−0.031** |

Read the way the task framed it (0.257 / 0.371 / this round), the raw number 0.469 looks like the
best of the three. **That reading is wrong and the freeze anticipated it.** Against its own chance
baseline the contrast mechanism is the only one of the three that is *below* chance, and against
the two earlier rounds converted to the identical two-way decision it is last. Paired on those 32
rows it goes **6 wins / 7 losses** against round 1 binarised and **3 wins / 6 losses** against the
masked round.

For continuity, the same table with the three mandatory smoke items added back (35 rows):
contrast **0.457**, round 1 binarised 0.486, masked binarised 0.543.

---

## 2. The decisive number: the mention cell was destroyed, not rescued

| cell | gold | round 1 (5-way) | masked (5-way) | **contrast (2-way)** |
|---|---|---|---|---|
| **`S_FP`** — counter-speech / quotation / reportage | OPPOSE | 1/21 = 0.048 | 2/21 = 0.095 | **1/18 = 0.056** (2/21 = 0.095 incl. smoke items) |
| `S_FN` | ENDORSE | 8/14 = 0.571 | 11/14 = 0.786 | **14/14 = 1.000** |

`S_FN` is **perfect** and `S_FP` is **near-zero**. That is not a stance judgement; it is a
constant. A model that answered ENDORSE on every single item would score 14/14 on `S_FN`, 0/18 on
`S_FP`, and 0.438 overall — which is within one item of what was actually measured.

**Across all 98 scored items, only 8 were ever called OPPOSE, and only 2 of them are `S_FP`
items** — `MHC::KDcCiUU8q5E` (excluded from the primary metric as a mandatory smoke check) and
`MHC_zh::BV1Qk4y1g7PM`, the one item carrying the primary metric's entire `S_FP` credit.

## 3. Where the collapse comes from: the forced choice deleted the escape hatch

Round 2 established that asserting hate is present converts `no_hate_content` into `endorses`, and
that this is bookkeeping rather than reasoning. **A two-alternative forced choice does the same
thing to every item at once, by construction** — there is no third option to fall into.

Thirteen of the 32 primary rows had answered `no_hate_content` or a non-endorsing class in round 1
and were routed to ENDORSE here. Seven of those thirteen are `S_FP` items, i.e. the routing moves
them from wrong-but-uncommitted to wrong-and-committed:

| item | cell | round 1 | masked | contrast |
|---|---|---|---|---|
| `non_hate_video_528`, `DxcRdnzBZoo`, `N68vmAE5s_g`, `OMSByZ-o3Ww`, `j_foVftOOs4` | `S_FP` | `no_hate_content` | `no_hate_content` | **ENDORSE** |
| `BV1Km4y1u7ri`, `BV1vK41177zi` | `S_FP` | `no_hate_content` | `endorses` | **ENDORSE** |
| `hate_video_365` | `S_FN` | `condemns` | `endorses` | **ENDORSE** |
| `_qldaPBgkk0`, `BV1Vy4y1p7x2`, `BV1qZ4y1T71a`, `EEC98aHSgIY`, `BV12G4y1S7mN` | `S_FN` | `no_hate_content` | mixed | **ENDORSE** |

The damage side confirms it. The non-hate controls, which are `no_hate_content` items by
construction and are excluded from the primary metric by the frozen gold:

| | round 1 | masked | **contrast** |
|---|---|---|---|
| `CTRL_NONHATE` falsely answered ENDORSE (view A) | 1/18 = 0.056 | 5/18 = 0.278 | **14/18 = 0.778** |
| `CTRL_HATE` answered ENDORSE | 17/18 | 16/18 | **18/18** |

**The forced choice is a threshold set to maximum aggressiveness.** It calls 90 of 98 videos
endorsements of hate. The detector already has a threshold; this buys nothing it cannot buy by
moving it.

## 4. The diagnostics that rule out the cheap explanations

The freeze pre-registered three checks whose job was to catch the mechanism failing for an
uninteresting reason. All three come back clean, which makes the failure a real property of the
model rather than an artefact of the instrument.

**Position bias — ruled out.** Slot A was chosen on **0.515** of calls (183/355, view A). The
endorsing template was randomised into slot A on half of all (item, pair) combinations by a fixed
hash, and the direction is recovered at scoring time, so a slot-preferring model would score at
chance. The ENDORSE preference tracks the *direction*, not the slot.

**Template artefact — ruled out.** All five pairs behave alike:

| pair | acc on S | endorse rate over all items |
|---|---|---|
| 1 | 0.469 | 0.845 |
| 2 | 0.531 | 0.944 |
| 3 | 0.438 | 0.901 |
| 4 | 0.469 | 0.803 |
| 5 | 0.469 | 0.845 |

No pair reaches 0.55; no pair falls below 0.80 endorse rate. The result is not one bad sentence.

**Lexical overlap — ruled out.** Per-pair, the vote agrees with the sign of the transcript-overlap
difference on **0.487** of 158 scored pairs, i.e. exactly chance. At item level the
**overlap-aligned share is 0.333** — the winning side has the *higher* word overlap with the
transcript in only a third of items, which is *less* than chance. Mean overlap difference per
group is within ±0.022 of zero everywhere. **The model is not word-matching; it is applying a
prior.** (Aligned items score 0.667 and non-aligned 0.444, but on n = 9 and n = 18 that is not a
finding.)

**Confidence.** The 5-vote ensemble is not hedging: **21 of 32** primary items are unanimous 5-0,
and those score **0.429**. The 7 items at 4-1 score 0.714. The model is confidently wrong, and the
few items where it wavers are the ones it gets right.

## 5. Stratification by voice form — the predicted main battleground did not appear

Using the hand-coded gold voice of `idea-stage/voice_field_analysis.py::GOLD_VOICE` (coded blind
for 46 of 49 S items; `OWN` = the hate-associated surface is the author's own → **作者有话**,
`NOT_OWN` = archive / broadcast / named third party / embedded clip → **作者无话**):

| stratum | meaning | n | **contrast** | round 1 binarised |
|---|---|---|---|---|
| `OWN` | 作者有话 | 13 | **0.385** | 0.692 |
| `NOT_OWN` | 作者无话 | 10 | **0.400** | 0.400 |
| `UNDET` | undeterminable | 9 | 0.667 | 0.333 |

The freeze predicted a priori that the `OWN` subset would be the mechanism's main battleground.
**It is not.** `OWN` and `NOT_OWN` are indistinguishable from each other (0.385 vs 0.400), and
`OWN` is where the contrast mechanism loses most heavily to the binarised round 1 (0.385 vs
0.692). The pinned-comment framing was adopted precisely because many uploaders say nothing in the
transcript; the framing does work as intended — it does not fall apart on the silent-author
items — but it delivers no gain on the talking-author items either. Whatever governs the answer is
not sensitive to whether the author speaks.

## 6. Variant 2 (target named) vs variant 1, paired

On the 22 target-bearing frame-bearing S rows, with the group name slotted in from the previous
pilot's extraction step:

| | acc on S | `S_FP` | `S_FN` | `CTRL_NONHATE` → ENDORSE |
|---|---|---|---|---|
| variant 1 (generic) | 0.500 | 1/12 = 0.083 | 10/10 = 1.000 | 6/6 = 1.000 |
| variant 2 (target named) | 0.409 | 1/12 = 0.083 | 8/10 = 0.800 | 3/6 = 0.500 |

Naming the attacked group does **not** improve the S bucket — it is 0.091 worse overall, and the
`S_FP` cell is identical at 1/12. What it does is soften the blanket ENDORSE: variant 2 halves the
false endorsement of non-hate controls (1.000 → 0.500) and is the only arm to put any real OPPOSE
mass on the board. It buys that by giving up two `S_FN` items. This is the same trade the whole
direction keeps re-discovering — a threshold move, in both directions, with no gain in
discrimination.

The one place variant 2 is clearly better is the item the user named: on `MHC::KDcCiUU8q5E`, the
Trump-misogyny counter-speech video, variant 2 answers OPPOSE 4/5 while variant 1 splits 3/5. That
single item is why the smoke revision was made, and it remains the only counter-speech item in the
whole pilot that any arm handles convincingly.

## 7. The three mandatory qualitative checks

All three are `S_FP` items, all three were run in the smoke pass, and all three are therefore
**excluded from the primary metric** by freeze B.8. Their formal-batch answers:

| item | what it is | contrast v1 | v2 | round 1 | masked |
|---|---|---|---|---|---|
| `MHC::KDcCiUU8q5E` | commentator denouncing Trump's misogyny | **OPPOSE** (3-2) ✓ | OPPOSE ✓ | `no_hate_content` | `condemns` |
| `HateMM::non_hate_video_32` | Lennon/Ono song *"Woman Is the N— of the World"* | ENDORSE (5-0) ✗ | ENDORSE ✗ | `endorses` | `endorses` |
| `HateMM::non_hate_video_16` | 1956 segregationist newsreel | ENDORSE (5-0) ✗ | — | `endorses` | `endorses` |

The two archival items are answered ENDORSE unanimously by every arm of every round. A 1956
newsreel and a 1972 protest song are both re-uploads of third-party material with no authorial
commentary in the transcript, and the model has never once distinguished *re-publishing* from
*asserting*. Including these three would move the primary metric from 0.469 to 0.457 — the
exclusion does not flatter the result.

## 8. Views and per-dataset

| view | n_S | contrast | round 1 binarised | masked binarised |
|---|---|---|---|---|
| **A — frame-bearing (the verdict)** | 32 | **0.469** (15/32) | 0.500 (16/32) | 0.563 (18/32) |
| A incl. the 3 smoke items | 35 | 0.457 | 0.486 | 0.543 |
| B — all 99 as frozen | 45 | 0.422 | 0.444 | 0.489 |
| C — text-only ImpliHateVid | 13 | 0.308 | 0.308 | 0.308 |

| dataset | n_S | contrast |
|---|---|---|
| HateMM | 6 | 0.167 |
| MHC (EN) | 15 | 0.400 |
| MHC_zh | 11 | 0.727 |
| ImpliHateVid (no frames) | 13 | 0.308 |

MHC_zh is the only split above chance, and it is the split with the fewest archival / third-party
re-uploads. HateMM stays worst for the reason both previous rounds recorded: its `S_FP` items are
newsreel and song performances whose entire content is the hateful surface.

## 9. Losses and cost

**One item produced no prediction: `MHC_zh::BV1m8411z7mV`.** All 10 of its requests (5 pairs ×
2 variants) were rejected by DashScope with `InternalError.Algo.DataInspectionFailed: Input image
data may contain inappropriate content`. This is the **third consecutive round** in which this
same item is lost on this same surface. View A is therefore 71 items rather than 72, and every
denominator above accounts for it. No other request failed: **805 of 805 delivered votes parsed
into a valid A/B** (100 %), with no ties and no salvage parsing needed.

**Cost — measured tokens:**

| run | endpoint | calls | input tok | output tok |
|---|---|---|---|---|
| smoke C1.0 (11 items) | realtime | 65 | 103,040 | 130 |
| smoke C1.1 after the revision (11 items) | realtime | 65 | 103,275 | 130 |
| **eval, 5 pairs × 2 variants** | **Batch** | **805** | **1,113,479** | **2,010** |
| **total** | | **935** | **1,319,794** | **2,270** |

At the list price assumed throughout this series (¥0.002/1K in, ¥0.008/1K out, Batch discount
ignored as an upper bound): **≈ ¥2.66 ≈ USD 0.37**, against a ¥5 cap. With the 50 % Batch discount
applied to the eval run it is ≈ ¥1.54.

## 10. Instrument quality — one revision, logged, and why it mattered

Freeze B.8 permitted a single template revision on the smoke pass; it was used and is logged as
**B.10 R1**. The original variant-1 endorsing templates were **referentially open**: "I stand by
every word said here" and "Say it louder for the ones in the back" are things the author of *any*
video would pin under it, including a counter-speech author endorsing their own criticism. The
evidence was `ImpliHateVid::NH_836` — a sermon containing no identity attack at all — choosing the
endorsing option 4/5, and `MHC::KDcCiUU8q5E` returning ENDORSE on exactly the three referentially
open pairs while the target-named variant 2 returned OPPOSE. Every option was rewritten to name
its referent ("the people the video's harsh words are about"). This is worth recording because it
means **the reported failure is not the failure of a badly-worded instrument** — the obvious
wording defect was found and fixed before the paid run, and the mechanism still failed.

## 11. What this establishes

Freeze B.8 pre-registered the reading of a FAIL verbatim, and it applies:

> **"零样本 MLLM 判立场"全路径关闭.** Direct classification (0.257 five-way), content masking
> (0.371 five-way) and symmetric two-way comparison (0.469 against a 0.50 chance baseline) have
> all now been measured on the same 99 items with the same model, and **no prompt-level
> intervention reaches the bar.** No fourth prompt-level mechanism is attempted.

Three corollaries, at the strength the evidence supports:

1. **The asymmetric-options diagnosis was wrong — or rather, it was not the binding constraint.**
   The hypothesis was that `endorses` wins because it is the safety-salient option in an
   unbalanced 5-way menu. The menu was made perfectly symmetric, position-randomised, and stripped
   of every moderation label, and the model answered ENDORSE on 90 of 98 videos anyway. The
   response bias is not a property of the label set. It survives the deletion of the label set.

2. **Every intervention in this series has turned out to be a threshold move.** Masking bought
   `S_FN` (+3) and cost `CTRL_NONHATE` (−4). The forced choice buys `S_FN` (14/14, perfect) and
   costs `CTRL_NONHATE` (14/18 false endorsements). Naming the target moves it back the other way.
   None of the three changes the model's ability to tell use from mention, which is the only thing
   that would have been worth having — `S_FP` sits at 0.048, 0.095, 0.056 across the three rounds,
   i.e. flat at zero within noise.

3. **The residual explanation is the topic, not the wording, the framing or the option set.**
   Round 2 removed the hateful wording from the transcript and the bias survived. This round
   removed the asymmetric option set and the bias survived. What is left is the video's *subject*:
   the model appears to answer "is this video about an identity group being attacked?" and report
   that as the author's stance. The pilots cannot separate that from a frame-borne trigger — the
   8 frames were never masked in any round — and do not claim to.

**`STANCE_PILOT_RESULT.md`'s KILL stands, and is now final for the zero-shot prompt-level route.**
The routes those documents listed as untouched — stance as *metadata* rather than content
inference, and fine-tuning a stance typer on labelled data — remain untouched by this measurement
too, and each would still need its own gate.

## 12. 与现有工作的区分

This is a negative capability measurement, so the relevant question is not novelty but whether the
finding is already known. It is adjacent to, and consistent with, two published results, and
extends both:

- **Gligorić et al., NAACL 2024 (`arXiv 2404.01651`), *NLP Systems That Can't Tell Use from
  Mention Censor Counterspeech, but Teaching the Distinction Helps*.** That paper shows the
  use-vs-mention failure in **text** classifiers and reports that **prompting** — embedding the
  use–mention definition plus CoT and few-shot exemplars — partially repairs it. Our three rounds
  are the **video** analogue and they qualify that repair claim in a specific way: V1.3 already
  carries an explicit use-vs-mention class definition and a calibration instruction, and across
  three structurally different prompt-level interventions the `S_FP` cell never moved off zero.
  Whatever prompting buys on text, it does not transfer to 8-frame + full-transcript video on this
  corpus with this model.
- **`arXiv 2510.20154` (EMNLP 2025), *Are Stereotypes Leading LLMs' Zero-Shot Stance Detection?***
  reports that zero-shot LLM stance is target-group-stereotype-biased. Our corollary 3 is an
  independent, video-side observation of the same shape: the answer tracks the topic — that the
  video concerns an identity group under attack — rather than the author's relation to it. Our
  variant 2 tested this directly by naming the group, and naming it changed the operating point
  without improving discrimination.

The one thing here that is not in either paper is the **decomposition of the apparent gains**.
Rounds 2 and 3 both produce headline improvements (0.257 → 0.371 → an apparent 0.469) that
dissolve under the right control: round 2's gain is bookkeeping from withdrawing
`no_hate_content`, and round 3's is an artefact of comparing a two-way number to five-way numbers.
The methodological point — that a stance-repair intervention must be scored against its own chance
baseline and against its damage on non-hate controls, or it will look like it worked — is the
transferable content of this series.

## 13. Reproducibility index

| artefact | path |
|---|---|
| frozen rules, mechanism, template bank, revision log | `idea-stage/MASK_STANCE_PILOT_FREEZE.md` **Appendix B** |
| prompt frame C1.0 + template bank C1.1 / C1.0 | `idea-stage/contrast_stance/contrast_prompts.py` |
| runner (build / smoke / batch submit / poll / fetch / merge) | `idea-stage/contrast_stance/run_contrast.py` |
| background driver | `idea-stage/contrast_stance/drive_contrast.sh` |
| frozen scorer (views, strata, overlap, position bias, baselines) | `idea-stage/contrast_stance/score_contrast.py` |
| reporting tables | `idea-stage/contrast_stance/report_tables.py` → `report_c1.txt` |
| smoke passes (C1.0, then C1.1) | `smoke_s1.jsonl`, `smoke_s2.jsonl` |
| per-pair request metadata (slot→direction map) | `reqmeta_c1_p{0..4}.jsonl` |
| per-pair raw / parsed / errors | `batch_out_c1_p*.jsonl`, `pred_c1_p*.jsonl`, `batch_err_c1_p*.jsonl` |
| batch request payloads (5 × 27 MB of base64 frames) | **deleted after the run**; regenerate byte-identically with `run_contrast.py submit --tag c1 --pair N --dry` |
| merged votes / scores | `pred_c1.jsonl` (805 votes), `score_c1.json` |
| run logs | `logging/runs/contrast_stance/{run.log, run.pid, smoke.log, smoke2.log, score.log}` |
| sample (re-used unchanged, seed 20260811) | `idea-stage/stance_pilot/sample.json` |
| hand-coded voice form | `idea-stage/voice_field_analysis.py::GOLD_VOICE` |

Data boundary as authorised: video frames were sent to the user's own DashScope account (user
ruling 2026-08-11); test-set **inputs** were used, test **labels** only as the anchor of a
disclosed capability measurement (user ruling 2026-08-09); no label tuned anything and no detector
hyper-parameter was selected here. The API key was read from `~/.dashscope_api_key` at runtime and
appears in no file in this repository.
