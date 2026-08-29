# VOICE_FIELD_ANALYSIS — does `primary_voice` alone separate stance-class errors from correct calls?

**Verdict: BURY.** The `primary_voice` field is *reasonably accurate* (0.83 binary,
n = 23 hand-checked) and *almost completely non-discriminative*: the enrichment of
"not the uploader's own voice" in stance-bucket errors versus correctly-classified controls is
**OR 1.70, p = 0.43** (round 1, primary view) and **OR 1.00, p = 1.00** (fallback ②), against a
pre-registered bar of OR ≥ 3 and p < 0.05. The one-directional suppression rule it would license
loses **≈ 50 correct hate videos to buy back ≈ 11 false positives** in population projection.

Date 2026-08-12. **Pure offline re-analysis: zero API calls, zero GPU.** Nothing here re-scores
the stance pilot; `STANCE_PILOT_RESULT.md`'s KILL stands unchanged. This document answers the one
question that document explicitly left open (§8 "not killed", item 2): *the `primary_voice` field
on its own was never measured.* It has now been measured. It does not survive either.

---

## 1. Scope, inputs, and what was frozen

**Question.** `STANCE_PILOT_RESULT.md` §3(a) observed that the model "recovers *who speaks* and then
ignores it" — it assigned a non-uploader voice to 10 of 21 `S_FP` items and still answered
`endorses` on 7 of them. That raised a weaker, cheaper hypothesis: forget the stance conversion,
use the **voice** field directly as a feature. If videos whose hateful surface comes from an
archive / a broadcast / a quoted third party are over-represented among the detector's
stance-class errors, a typed voice feature could be worth something even though stance is dead.

**Inputs** (all already on disk, all already paid for):
`idea-stage/stance_pilot/pred_strong.jsonl` (round 1, 98 rows) ·
`pred_fb2.jsonl` (fallback ②, 99 rows) · `sample.json` (99 eval items, seed 20260811) ·
`idea-stage/r5_buckets.json` · `r5_error_dump.json` · `r5_phase_a.json` · `data/gt/*/test.jsonl`.

**Stratification** inherited unchanged from `STANCE_PILOT_FREEZE.md` §6.1 (deviation D1):
**view A = the 72 frame-bearing items (HateMM + MHC + MHC_zh) is the primary read**; view B (all
99) and view C (the 27 transcript-only ImpliHateVid items) are appendix only. Round 1 has 71 rows
in view A and fallback ② has 72 (one has a stage-A moderation error, deviation D6).

**Judgement criteria were written into the head of `idea-stage/voice_field_analysis.py` (F0–F8)
before any cell count in this document was computed.** Summarised:

| id | rule |
|---|---|
| F1 | Primary binarisation **V-strict**: `OWN = {uploader}`; `NOT_OWN = {on_screen_speaker, quoted_third_party, archival_source, caption_overlay}`; `none` **excluded** from the 2×2, because the prompt defines `none` as "there is no such material (Q1 false)" — it is a restatement of `hate_surface_present`, and including it would smuggle the hate/non-hate axis back into a test meant to isolate voice. Sensitivity variants **V-loose** (`OWN = {uploader, on_screen_speaker}`) and **V-incl** (`none` folded into `NOT_OWN`) are reported and may not be promoted to primary after the fact. |
| F2 | Primary contrast = all S errors (`S_FP` + `S_FN`) vs all controls (`CTRL_HATE` + `CTRL_NONHATE`). Sub-contrasts reported: `S_FP` vs `CTRL_NONHATE`, `S_FP` vs `CTRL_HATE`, `S_FN` vs `CTRL_HATE`. |
| F3 | Odds ratio + two-sided Fisher exact; Haldane–Anscombe +0.5 applied iff a cell is 0. |
| **F4** | **SIGNAL iff OR ≥ 3.0 **and** p < 0.05 on the primary contrast, in the primary view, under V-strict, in at least one round. Otherwise BURY.** A sensitivity variant or appendix view clearing the bar alone does **not** overturn BURY. |
| F5 | Flip rule `R_voice`: `voice ∈ NOT_OWN` ⇒ push towards non-hate, one-directional. Gains = `S_FP` rescued; damage = `CTRL_HATE` destroyed; `S_FN` is neutral (already wrong); `CTRL_NONHATE` untouched by a one-directional non-hate push. |
| F6 | Two-round agreement declared in advance as a **prompt-robustness** measure (both runs are temperature 0; fallback ② asks the same question in a decomposed prompt), reported not gated. |
| F7 | Gold voice coded by hand from the `r5_error_dump` transcript + OCR **only**, written down before the model's output for that item was read; undeterminable items dropped from the denominator with n stated. |

---

## 2. Voice distribution tables (part 1)

### 2.1 View A — frame-bearing 72, round 1 (the primary table)

| group | uploader | on_screen_speaker | quoted_third_party | archival_source | caption_overlay | none | n |
|---|---|---|---|---|---|---|---|
| `S_FP` (detector wrong, label 0) | 9 | 7 | 0 | 1 | 2 | 2 | 21 |
| `S_FN` (detector wrong, label 1) | 7 | 3 | 0 | 0 | 3 | 1 | 14 |
| `CTRL_HATE` (detector right, label 1) | 10 | 5 | 0 | 0 | 2 | 1 | 18 |
| `CTRL_NONHATE` (detector right, label 0) | 7 | 1 | 0 | 1 | 1 | 8 | 18 |

### 2.2 View A — frame-bearing 72, fallback ② (decomposed prompt)

| group | uploader | on_screen_speaker | quoted_third_party | archival_source | caption_overlay | none | null | n |
|---|---|---|---|---|---|---|---|---|
| `S_FP` | 4 | 6 | 0 | 0 | 2 | 9 | 1 | 22 |
| `S_FN` | 2 | 5 | 0 | 0 | 0 | 7 | 0 | 14 |
| `CTRL_HATE` | 5 | 8 | 0 | 0 | 4 | 1 | 0 | 18 |
| `CTRL_NONHATE` | 1 | 0 | 1 | 0 | 0 | 16 | 0 | 18 |

**Read these two tables together and the first finding is already visible: the categories the
hypothesis is actually about — `quoted_third_party` and `archival_source`, i.e. 引用 and 档案 —
are emitted 2 times in 71 items (round 1) and 1 time in 72 (fallback ②).** All of the `NOT_OWN`
mass sits on `on_screen_speaker` and `caption_overlay`, which are *camera and modality* facts
("someone is filmed", "text is burned in"), not *provenance* facts. See §7.1.

The only visible group difference is `none` in `CTRL_NONHATE` (8/18 round 1, 16/18 fallback ②) —
and that is precisely the mechanical `hate_surface_present = false` restatement F1 excluded in
advance. It carries no voice information.

### 2.3 Appendix — views B and C, round 1

| | B (all 99) `S_FP` | `S_FN` | `CTRL_HATE` | `CTRL_NONHATE` | | C (27 text-only) `S_FP` | `S_FN` | `CTRL_HATE` | `CTRL_NONHATE` |
|---|---|---|---|---|---|---|---|---|---|
| uploader | 16 | 10 | 16 | 13 | | 7 | 3 | 6 | 6 |
| on_screen_speaker | 7 | 4 | 5 | 1 | | 0 | 1 | 0 | 0 |
| quoted_third_party | 0 | 0 | 0 | 0 | | 0 | 0 | 0 | 0 |
| archival_source | 1 | 0 | 0 | 1 | | 0 | 0 | 0 | 0 |
| caption_overlay | 2 | 3 | 2 | 1 | | 0 | 0 | 0 | 0 |
| none | 3 | 2 | 2 | 9 | | 1 | 1 | 1 | 1 |
| **n** | **29** | **19** | **25** | **25** | | **8** | **5** | **7** | **7** |

In the text-only view the field degenerates almost entirely to `uploader` (22/27 round 1) — with
no frames the model has essentially nothing to type provenance from, which is consistent and
uninteresting.

---

## 3. Separability (part 2) — odds ratios and Fisher exact

### 3.1 The pre-registered primary contrast, view A, V-strict

| round | S errors `NOT_OWN` | controls `NOT_OWN` | OR | p (Fisher, 2-sided) | F4 bar |
|---|---|---|---|---|---|
| round 1 | **16/32 = 0.500** | **10/27 = 0.370** | **1.70** | **0.431** | fail |
| fallback ② | **13/19 = 0.684** | **13/19 = 0.684** | **1.00** | **1.000** | fail |

### 3.2 All contrasts × all binarisations, view A (nothing clears the bar)

| contrast | binarisation | round 1 OR / p | fallback ② OR / p |
|---|---|---|---|
| **S all vs CTRL all** (primary) | **V-strict** | **1.70 / 0.431** | **1.00 / 1.000** |
| S all vs CTRL all | V-loose | 1.33 / 0.741 | 0.33 / 0.405 |
| S all vs CTRL all | V-incl | 1.06 / 1.000 | 0.97 / 1.000 |
| `S_FP` vs `CTRL_NONHATE` | V-strict | 2.59 / 0.434 | 2.00 / 1.000 |
| `S_FP` vs `CTRL_NONHATE` | V-loose | 0.75 / 1.000 | 0.20 / 0.396 |
| `S_FP` vs `CTRL_HATE` (the trade-off pair) | V-strict | 1.59 / 0.525 | 0.83 / 1.000 |
| `S_FP` vs `CTRL_HATE` | V-incl | 1.67 / 0.527 | 1.64 / 0.706 |
| `S_FN` vs `CTRL_HATE` | V-strict | 1.22 / 1.000 | 1.04 / 1.000 |

The best number in the whole family is **OR 2.59, p = 0.43** (`S_FP` vs `CTRL_NONHATE`, round 1) —
below the effect bar and nowhere near the significance bar. Under the `V-loose` binarisation
(treating a filmed speaker as the uploader) the sign of the primary effect **reverses** in both
rounds. An effect whose sign depends on how one of six categories is bucketed is not a channel.

Appendix views: view B round 1 primary = OR 1.90 / p = 0.241; view B fallback ② = OR 0.78 /
p = 0.785. View C round 1 primary = OR 3.57 / p = 0.478 — the single OR ≥ 3 anywhere in the
document, resting on **1 event** (1/11 vs 0/12) with a Haldane correction, in the appendix
population, at p = 0.48. Per F4 it explicitly does not overturn BURY, and it should not be quoted.

### 3.3 Is this a power failure? Partly, and the honest statement is bounded

At the primary view's n (32 error / 27 control, control rate 0.370), the smallest enriching count
that reaches p < 0.05 is **21/32 = 0.656, which corresponds to OR 3.25.** So the design **is**
powered to detect exactly the pre-registered effect: an OR of 3 sits at the edge of detectability
and would have been caught. The observed count is 16/32.

What this analysis therefore rules out: **OR ≥ 3.** What it does **not** rule out: a weak
association around OR ≈ 1.5–2.0, which this n cannot resolve. §4 disposes of that residual
separately — a weak association is not merely undetectable here, it is *insufficient* by
arithmetic even if real.

---

## 4. Net flip projection (part 3) — rule `R_voice`

Rule: `primary_voice ∈ NOT_OWN` ⇒ push the item towards non-hate. One-directional, so
`CTRL_NONHATE` cannot be damaged and `S_FN` cannot be made worse than it already is.

### 4.1 View A (primary), round 1

| | count |
|---|---|
| **gains** — `S_FP` errors rescued (rule fires on a false positive) | **10** of 21 |
| neutral — `S_FN` errors the rule fires on (already wrong, stays wrong) | 6 of 14 |
| **damage** — `CTRL_HATE` correctly-detected hate videos destroyed | **7** of 18 |
| sample-level net | **+3** |

The sample-level net is positive **only because the sample deliberately over-samples errors**
(36 S errors against 36 controls, where the real test splits hold 81 errors against 444 correct
items). Projecting each rate onto its true population — and noting that the sample already
contains essentially every S_FP error in the corpus (7/7 HateMM, 10/10 MHC, 4–5/5 MHC_zh, 8/8
ImpliHateVid), so the gain term is close to a complete enumeration rather than an extrapolation:

| dataset | fire rate on `S_FP` | S_FP in population | projected gain | fire rate on `CTRL_HATE` | correct hate in population | projected damage |
|---|---|---|---|---|---|---|
| HateMM | 0.714 | 7 | 5.00 | 0.500 | 71 | 35.50 |
| MHC | 0.200 | 10 | 2.00 | 0.500 | 39 | 19.50 |
| MHC_zh | 0.750 | 5 | 3.75 | 0.167 | 32 | 5.33 |
| **total (view A)** | | **22** | **+10.75** | | **142** | **−60.33** |

**Population net = −49.6.** The rule pays roughly **1 rescued false positive for every 5.6
destroyed true detections.**

### 4.2 All views, both rounds

| view | round | sample gains / damage / net | population gain / damage / **net** |
|---|---|---|---|
| **A (primary)** | round 1 | 10 / 7 / +3 | +10.75 / 60.33 / **−49.6** |
| **A (primary)** | fallback ② | 8 / 12 / −4 | +8.00 / 98.96 / **−91.0** |
| B (all 99) | round 1 | 10 / 7 / +3 | +10.75 / 60.33 / **−49.6** |
| B (all 99) | fallback ② | 10 / 16 / −6 | +10.00 / 208.67 / **−198.7** |
| C (text-only) | round 1 | 0 / 0 / 0 | 0 / 0 / **0** |
| C (text-only) | fallback ② | 2 / 4 / −2 | +2.00 / 109.71 / **−107.7** |

Every population projection is ≤ 0, in both rounds, in every view.

**Why this is decisive independently of §3's power limit.** The arithmetic that kills the rule is
not the *difference* in `NOT_OWN` rate between errors and controls; it is the **absolute level of
`NOT_OWN` among correctly-detected hate videos: the rule fires on 7 of 18 `CTRL_HATE` items
(39 %) in round 1 and 12 of 18 (67 %) in fallback ②.** Four in ten videos that the detector *correctly* calls hateful also carry a non-uploader voice —
because reposted clips, filmed speakers and burned-in captions are the ordinary grammar of
short-form video, hateful or not. Any suppression rule keyed on that variable hits them. Even at
a true OR of 2 the base rate would still make the trade negative, because the correct-hate pool
(142 items in view A, 334 across all four splits) dwarfs the S_FP pool (22 / 30).

---

## 5. Two-round stability of the voice field itself (part 4a, F6)

Both runs are temperature 0 with the same seed; the difference between them is the prompt
(single-call V1.3 vs the decomposed fallback ②). This is therefore prompt robustness, not
sampling noise.

| view | n compared | 6-way agreement | κ (6-way) | binary agreement (OWN/NOT_OWN/NONE) | κ (binary) |
|---|---|---|---|---|---|
| **A — frame-bearing** | 71 | **0.493** | **0.344** | **0.507** | **0.305** |
| B — all 99 | 98 | 0.510 | 0.348 | 0.520 | 0.324 |
| C — text-only | 27 | 0.556 | 0.299 | 0.556 | 0.299 |

**The field disagrees with itself on half the items across a prompt rewrite, κ ≈ 0.34** — "fair"
agreement on the Landis–Koch scale, and well below what any usable feature should show under a
paraphrase of its own question. Top confusions in view A:

| round 1 → fallback ② | n |
|---|---|
| `uploader` → `uploader` | 11 |
| `none` → `none` | 11 |
| `on_screen_speaker` → `on_screen_speaker` | 11 |
| **`uploader` → `none`** | **11** |
| **`uploader` → `on_screen_speaker`** | **7** |
| `on_screen_speaker` → `none` | 5 |
| `caption_overlay` → `none` | 5 |
| `uploader` → `caption_overlay` | 3 |
| `archival_source` → `on_screen_speaker` | 1 |

The `→ none` mass (22 of 71) is the fallback's stricter `hate_surface_present` gate propagating
into the voice field, i.e. the two fields are not independent. The `uploader → on_screen_speaker`
mass (7) is the substantive instability: the model has no stable position on whether the person
in front of the camera is the person who posted the video — which is the exact distinction the
whole hypothesis rests on.

---

## 6. Is the voice field even correct? (part 4b)

**Protocol (F7).** I coded a gold voice form for all 49 sampled S-bucket items from the
`r5_error_dump` transcript + OCR text **only**, and wrote each code into
`voice_field_analysis.py::GOLD_VOICE` **before** reading the model's `primary_voice` for that item.
Adjudication is at the binary OWN / NOT_OWN level; the 6-way call is usually undecidable from
text. Items with no determining cue are coded `UNDET` and dropped.

- 49 items coded → **21 OWN · 16 NOT_OWN · 12 UNDET**.
- **Declared contamination: 3 of the 49** (`hate_video_365`, `non_hate_video_121`,
  `non_hate_video_149`) had their model output visible in this session before I coded them, from
  the first few lines of the prediction files. They are not blind. Removing all three does not
  change the picture (one is a hit, one a miss, one a hit).
- Items where the model answered `none` are not scored (no voice call was made): 2 of 25 in
  round 1 view A, 10 of 25 in fallback ②.

| run | view | scored n | correct | **accuracy (V-strict)** | accuracy (V-loose) |
|---|---|---|---|---|---|
| **round 1** | **A — frame-bearing** | **23** | 19 | **0.826** | 0.696 |
| round 1 | B — all 99 | 32 | 27 | 0.844 | 0.719 |
| round 1 | C — text-only | 9 | 8 | 0.889 | 0.778 |
| **fallback ②** | **A — frame-bearing** | **15** | 12 | **0.800** | 0.400 |
| fallback ② | B — all 99 | 22 | 17 | 0.773 | 0.500 |
| fallback ② | C — text-only | 7 | 5 | 0.714 | 0.714 |

**≈ 0.83 on n = 23** is a real but small-n estimate, and it is the single most important number in
this document, because it changes the *kind* of negative result this is:

> The voice field is **not broken**. It is roughly 80 % right about who is speaking, and it is
> still useless for separating the detector's stance errors from its correct calls. The failure is
> not measurement error in the feature — it is that the underlying variable is **not associated
> with the partition we need**.

Round-1 errors (all four, view A): `non_hate_video_121` gold OWN → `on_screen_speaker`;
`DxcRdnzBZoo` gold OWN → `on_screen_speaker`; `BV1Km4y1u7ri` gold OWN → `on_screen_speaker`;
`pofgIFZpR7c` gold NOT_OWN → `uploader`. Three of four are the same error: a filmed person is
called `on_screen_speaker` when they *are* the uploader. Hence the large V-strict/V-loose gap.

---

## 7. Why it fails — two mechanisms, both structural

### 7.1 The model answers a camera question, not a provenance question

Of the **13 view-A S-bucket items I hand-coded as genuinely NOT_OWN** (archival newsreel,
broadcast clip, named third party's monologue, reposted drama excerpt), the model emitted a
*provenance* category (`archival_source` or `quoted_third_party`) for exactly **1** in round 1 and
**0** in fallback ②. What it emitted instead:

| round 1 prediction on the 13 gold-NOT_OWN items | n |
|---|---|
| `on_screen_speaker` | 6 |
| `caption_overlay` | 3 |
| `archival_source` | **1** |
| `uploader` | 1 |
| `none` | 1 |
| (moderation-rejected) | 1 |

Concrete cases:

- `non_hate_video_16` — a 1956 segregationist newsreel whose OCR literally carries a
  **GlobalImageWorks stock-footage licensing timecode watermark** (`GIW.CUSTOM TCR 01:04:29.10`).
  Round 1: `on_screen_speaker` / `endorses`. Fallback ②: `on_screen_speaker` / **`reports`** —
  with the evidence string "*The video is archival footage with neutral timecode and watermark
  overlays*". It **describes** the archival provenance in prose and still does not put it in the
  typed field.
- `non_hate_video_149` — the one item ever typed `archival_source` (round 1). Fallback ② moved it
  to `on_screen_speaker`.
- `non_hate_video_400` — an Ellen Show broadcast clip (OCR: `ellentube`, `Watch Ellen Weekdays`) →
  `on_screen_speaker` in both rounds.
- `dK43yHIUMKA` — a Dave Allen stand-up routine, the performer dead since 2005, named in the video
  title → `on_screen_speaker` in both rounds.

So the axis the hypothesis wanted (第一人称 vs 档案/引用/转播) is **never actually populated**. What
the field populates instead is "is a human visible on camera / is text burned in", and that
variable has no reason to correlate with detector error and, empirically, does not.

### 7.2 The base rate forbids it even if it were populated

§4's arithmetic: 41–71 % of *correctly detected* hate videos also carry a non-uploader voice. Hate
videos are overwhelmingly compilations, reposts, filmed rants and captioned clips — the same
surface grammar as the counter-speech and news items in the S bucket. There is no version of a
voice-keyed suppression rule that does not walk straight through the true positives.

### 7.3 Exploratory, declared post-hoc, not part of the verdict

The obvious rescue is "use voice only where the model already said `endorses`". Within the view-A
items where stance = `endorses`, `S_FP` vs `CTRL_HATE` on `NOT_OWN`: round 1 **7/12 vs 7/17,
OR 2.00, p = 0.462**; fallback ② **6/10 vs 10/15, OR 0.75, p = 1.00**. This was computed after the
fact and is labelled as such; it does not clear the F4 bar either, and the sign flips between
rounds. The rescue does not exist.

---

## 8. Verdict and death certificate

**BURY.** Cause of death, in the order the evidence lands:

1. **No enrichment.** Primary contrast OR 1.70 / p = 0.43 (round 1) and OR 1.00 / p = 1.00
   (fallback ②), against a pre-registered OR ≥ 3 and p < 0.05. The design was powered for the
   pre-registered effect (OR 3.25 would have reached p = 0.038 at this n); it is not there.
2. **The sign is not even stable across a defensible re-bucketing** — under V-loose the primary OR
   goes to 1.33 and 0.33.
3. **Negative net flip in every population projection**, −49.6 (round 1) and −91.0 (fallback ②) in
   the primary view; ~1 false positive rescued per 5.6 true detections destroyed. This is driven
   by the *base rate* of non-uploader voice among correct hate detections (39 % / 67 %), which no
   improvement in the feature's accuracy can change.
4. **The field is unstable across a prompt paraphrase** — κ = 0.34, 6-way agreement 0.49.
5. **And the discriminative categories are never emitted**: `archival_source` +
   `quoted_third_party` appear 2 times in 71 items, while hand-coding says 13 of the view-A S items
   genuinely are archival / quoted / broadcast.

**What is specifically closed by this.** `STANCE_PILOT_RESULT.md` §8 listed "the `primary_voice`
field on its own" as not-refuted. It is now refuted, for the same instrument class: **a frozen,
prompted VL model emitting a typed voice field over frames + transcript.** No further probe of
that instrument is warranted, and no head should be trained on this feature.

**What this does not close** (kept narrow so the kill is not over-read):

1. **Provenance obtained from something other than content inference** — reverse image/audio
   search, duplicate detection against a clip corpus, upload metadata, channel priors. §7.1 shows
   the failure is that the model will not *type* provenance it demonstrably *perceives*; a
   non-inferential provenance signal is a different object and is untouched here. Note that
   `research-wiki/ideas/ocr-provenance-typing.md` and `duplicate-conflict-memory.md` already sit in
   the idea pool and are unaffected.
2. **A fine-tuned voice typer.** Only a frozen prompted model was measured.
3. Nothing here bears on the detector, the retrieval stack, or any frozen result.

**Minimum next step: none in this direction.** This was a free re-analysis of data already paid
for; it consumed no API budget and no GPU. The correct move is to spend nothing further on
voice/stance typing from a frozen VL front-end and to return to the round-4/round-5 candidate
queue.

---

## 9. Reproducibility

| artefact | path |
|---|---|
| frozen criteria (F0–F8), all computation, hand-coded gold table | `idea-stage/voice_field_analysis.py` |
| full numeric output (all views, both rounds, all binarisations) | `idea-stage/voice_field_analysis.json` |
| this report | `idea-stage/VOICE_FIELD_ANALYSIS.md` |
| round-1 / fallback ② predictions (re-read, not re-generated) | `idea-stage/stance_pilot/pred_strong.jsonl`, `pred_fb2.jsonl` |
| sample definition (seed 20260811) | `idea-stage/stance_pilot/sample.json` |
| bucket gold + per-error text used for hand coding | `idea-stage/r5_buckets.json`, `idea-stage/r5_error_dump.json` |
| parent pilot: rules and verdict | `idea-stage/STANCE_PILOT_FREEZE.md`, `idea-stage/STANCE_PILOT_RESULT.md` |

Run with `python3 idea-stage/voice_field_analysis.py` (CPU, < 2 s, no network).
Zero API calls and zero GPU were used in producing this document. No test label was used to select
or tune anything; test labels appear only as the group definition of an already-frozen sample.
