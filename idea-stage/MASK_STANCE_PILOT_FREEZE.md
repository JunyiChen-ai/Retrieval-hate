# MASK_STANCE_PILOT_FREEZE — does masking the hateful content rescue the stance judgement?

**Frozen: 2026-08-12 (Pacific/Auckland), before any evaluation-sample API call was made.**
Author: subagent (Opus 5). Scope: one cheap capability measurement. No GPU, no training,
no downstream model, no test *labels* used for tuning. The run stops at an accuracy number.

---

## 0. The question, and why it is worth one more measurement

`idea-stage/STANCE_PILOT_RESULT.md` killed the "ask a frozen VL model for the stance of a
hateful video" instrument at **P1 = 0.257** (round 1, direct ask) and **0.167** (fallback ②,
decomposed) against a 0.70 bar, on the frame-bearing subset (view A). The observed mechanism
was a collapse of the 5-way stance field onto `{endorses, no_hate_content}`: across all 71
round-1 items the model emitted `quotes_mentions` **0 times** and `reports` **0 times**, and on
the `S_FP` cell (gold = non-endorsing) it answered `endorses` 12/21 — including items where it
had *itself* just labelled the voice `archival_source`.

**The hypothesis under test here:** the collapse is driven by the *presence of the hateful
material itself* in the model's context. The hateful surface acts as a trigger that dominates the
stance head, so the framing cues (attribution, criticism, news register) never get used. If the
material is removed from the transcript and replaced by a neutral placeholder that merely
*asserts* that attacking material exists, the trigger is absent while the framing cues survive,
and the model should be able to type the stance from framing alone.

This is a mechanism test, not a product. **It runs to an accuracy number and stops.** Nothing
downstream is built regardless of the outcome.

---

## 1. Authorisations and data boundary

- **User ruling 2026-08-11:** the user's own DashScope (Alibaba Qwen) API may be used, and
  **video frames may be sent to it**. Unchanged and re-used here.
- **User ruling 2026-08-09 (test protocol):** test-set *inputs* may be used freely; test *labels*
  may not be used to tune and then be re-reported as held-out. This pilot sends **test inputs**
  and uses test labels only as the anchor of a **disclosed capability measurement**. Nothing here
  selects a model, threshold or hyper-parameter for the detector.
- The API key is read from `~/.dashscope_api_key` at runtime and is **never** written into any
  script, log, output file or report.
- Budget cap for the whole pilot: **< USD 5 equivalent**.

---

## 2. Sample — IDENTICAL to the previous pilot, byte for byte

`idea-stage/stance_pilot/sample.json` (seed 20260811) is re-used unchanged. No re-selection, no
re-sampling. This is the entire point: the same items, the same model, the same gold, so the
difference in P1 is attributable to the masking mechanism and nothing else.

| group | definition | n |
|---|---|---|
| `S_FP` | primary-S bucket errors, false positives (label = non-hate, comparator said hate) | 30 |
| `S_FN` | primary-S bucket errors, false negatives (label = hate, comparator said non-hate) | 19 |
| `CTRL_HATE` | correctly predicted hate items (label = 1) | 25 |
| `CTRL_NONHATE` | correctly predicted non-hate items (label = 0) | 25 |

**Smoke sample: the same 8 items**, all disjoint from the 99 eval items. Prompt iteration happens
on the smoke items only; smoke items never enter any reported metric.

**Model: `qwen3-vl-plus`** — frozen to match the previous round exactly. No model shopping. The
Batch API does not accept the pinned snapshot (`model_not_found`, previous pilot deviation D2), so
the moving alias is used again and the run date is recorded in place of a pin.

---

## 3. The two-step pipeline (frozen)

### Step 1 — extraction (one call per item, transcript only, NO frames)

The model is asked to return the **verbatim spans of the transcript** that constitute material
attacking / demeaning / dehumanising a person or group **because of a group identity**, using the
identity definition of V1.3 verbatim (so the "what counts as hate surface" question is held fixed
against the previous round), plus, for each span, a **neutral descriptor of the targeted group**.

**Critical anti-circularity rule.** Step 1 is *forbidden* to reason about who owns the material or
why it is shown. It extracts attacking material **regardless of speaker, attribution, purpose or
framing**. If step 1 were allowed to extract only "quoted / displayed / third-party" material it
would have to solve the stance problem in order to define its own output, and the pilot would
measure nothing. What is masked is therefore *all* identity-attacking surface; what survives is
the author's own framing language.

Step 1 gets **no frames**: it is a text-span task over the transcript, and withholding frames
keeps it cheap and keeps the frames' contribution confined to step 2.

Output record per item:

```
{"spans": [{"text": "<verbatim substring of the transcript>",
            "target": "<neutral group descriptor, e.g. 'Black people'>"}, ...],
 "any_hate_surface": true|false}
```

### Step 1.5 — masking (programmatic, no model)

Each returned span is located in the transcript by: (1) exact match; (2) whitespace-normalised
match; (3) `difflib` best sliding-window match with ratio ≥ 0.80. A span that cannot be located at
ratio ≥ 0.80 is **dropped and counted** (the unmatched-span rate is reported). Located spans are
replaced, longest-first and non-overlapping, by a placeholder:

- transcript is CJK-dominant (≥ 15 % CJK characters) → `[一段针对<目标群体>的攻击性言论]`
- otherwise → `[a passage of attacking speech targeting <group>]`

`<group>` is the model's neutral descriptor. Adjacent placeholders with the same target that end
up separated by ≤ 3 characters are merged into one. The masked transcript, the span table, the
match ratios and the masked-character fraction are all cached.

### Step 2 — masked stance question (one call per item, 8 frames + masked transcript)

**8 evenly spaced frames, max side 512 px, JPEG q80 — identical to the previous round.**
Transcript = the masked transcript, **full, no windowing, no truncation**. No OCR (held out, same
as the previous round 1).

The prompt is **V1.3 verbatim** (`idea-stage/stance_pilot/prompts.py::V1`) — same five stance
classes, same class definitions, same DEFAULT RULE, same CALIBRATION line — with exactly one
inserted paragraph, the MASKING NOTE, which:

1. explains that bracketed placeholders stand for attacking material removed by a previous pass;
2. states that when ≥ 1 placeholder is present, the presence of identity-attacking material is
   **established** — `hate_surface_present` is true and `no_hate_content` is not available;
3. instructs that the question is only where **this video's own authorial voice** stands.

Point 2 is deliberate and is declared here as a design choice, not discovered afterwards. Without
it, an item whose entire transcript is hateful material would be masked down to a placeholder and
the model would answer `no_hate_content` — a mechanically induced wrong answer that measures the
masking implementation, not the stance judgement. Items with **zero** placeholders keep V1.3's
free Q1 exactly as in round 1.

Decoding: `temperature=0.0`, `seed=20260811`, `max_tokens=400`, both steps.

### 3.1 What this design is, in terms of the previous round's two arms

|  | hateful surface in context | Q1 (`hate_surface_present`) | measured P1 (view A) |
|---|---|---|---|
| round 1 (direct) | present | asked freely | **0.257** |
| fallback ② (decomposed) | present | given by stage A | **0.167** |
| **this pilot (masked)** | **removed** | given by the placeholders | *to be measured* |

The task-specified headline comparison is against **0.257**. The mechanistically cleaner
comparison — the one that isolates *masking* rather than *masking + Q1 pre-answering* — is against
**fallback ②'s 0.167**, because both pre-answer Q1. Both deltas are reported; neither is chosen
after seeing the numbers.

---

## 4. Gold standard — UNCHANGED, frozen collapse from the previous pilot §5

| group | gold stance class (collapsed) | anchored also by |
|---|---|---|
| `S_FP` | **non-endorsing** = {`quotes_mentions`, `condemns`, `reports`} | dataset label = non-hate |
| `S_FN` | **`endorses`** | dataset label = hate |
| `CTRL_HATE` | **`endorses`** (assumed) | dataset label = hate |
| `CTRL_NONHATE` | **`no_hate_content` or non-endorsing** (assumed) | dataset label = non-hate |

Source of the S coding: `idea-stage/r5_buckets.json` (round-5 agent, transcript + HateMM OCR,
coded before any count was taken). Known weakness carried forward unchanged: the bucket coding was
done from transcripts only, and `S_FN` is the more contestable cell, so `S_FP` is reported
separately as the cleaner sub-measurement.

---

## 5. Judgement rule — FROZEN, computed after the run, never adjusted

Stance field `s(x)`; non-endorsing set `N = {quotes_mentions, condemns, reports}`.

**P1 — stance accuracy on the S bucket, view A (frame-bearing).**
`acc_S = ( #{x ∈ S_FP : s(x) ∈ N} + #{x ∈ S_FN : s(x) = endorses} ) / |S|`
**PASS iff `acc_S ≥ 0.70`.**

**P2 — directional damage on the hate controls, view A.**
`false_distancing = #{x ∈ CTRL_HATE : s(x) ∈ {quotes_mentions, condemns}} / |CTRL_HATE|`
**PASS iff `false_distancing ≤ 0.15`.**

**Overall verdict: PASS requires P1 ∧ P2.** (The previous pilot's P3 net-flip projection is
computed and reported for continuity but, per the task specification for this pilot, **does not
enter the verdict**.)

### 5.1 Stratification (frozen, same as previous pilot deviation D1)

| view | population | status |
|---|---|---|
| **A — primary** | frame-bearing: **72** items = 36 S errors (HateMM 8 + MHC 16 + MHC_zh 12) + 36 controls | **the verdict** |
| **B — as-frozen** | all **99** items incl. the 27 transcript-only ImpliHateVid items | reference only |
| **C — text-only** | the **27** ImpliHateVid items alone | descriptive only, never in the verdict |

### 5.2 Additional strata, frozen now, purely descriptive

Reported but **not** part of the verdict:

- P1 on S items **with ≥ 1 masked span** vs **0 masked spans**. The mechanism cannot act on an
  item whose transcript contained no extractable attacking material, so its reach is bounded by
  this fraction; the "≥ 1 span" cell is the mechanism's own best case.
- The full 5-class stance histogram per group, and specifically whether `quotes_mentions` and
  `reports` — emitted **0 times** in round 1 — are emitted at all.
- Per-dataset P1.
- Extraction quality: unmatched-span rate, masked-character fraction, and a **10-item random
  audit** of the formal batch (`numpy.default_rng(20260812)` over the sorted eval id list),
  hand-read against the transcript.

---

## 6. Pre-registered failure plan (at most ONE step, then stop)

1. **If the smoke inspection shows the extraction step is bad** — spans that are not attacking
   material, spans that miss obvious attacking material, spans that cannot be located in the
   transcript, or placeholders that destroy the framing text — the extraction prompt is revised
   **once** and re-smoked. Prompt iteration on the 8 smoke items is permitted and every revision
   is logged in Appendix A.
2. **If extraction is accurate but the stance answers still collapse**, the verdict is **FAIL**
   and the recorded conclusion is: *the model's stance bias does not depend on the verbatim
   hateful wording being present in the transcript*; the residual candidate sources are the frames
   and the topic/target itself. This is a substantive finding and is written up as such.

No second mechanism, no third arm, no model swap. The pilot ends at the number.

---

## 7. Process rules in force

- The judgement rule (§5) is frozen **before** any eval-sample call and is not adjusted afterwards.
- Prompt iteration during smoke is permitted; the 8 smoke items are disjoint from the 99 eval
  items; every revision is logged in Appendix A; the §5 thresholds are untouched.
- Single submission for the paid run; re-submission only on infrastructure failure, which is
  logged as a deviation.
- Both steps' intermediate products are cached under `idea-stage/mask_stance_pilot/` and every
  stage is idempotent/resumable.
- Logs: `logging/runs/mask_stance_pilot/run.{log,pid}`; the run is backgrounded and survives SSH
  disconnect.
- Honest reporting: if the numbers are bad they are reported as they are.

---

## 8. Known structural limits, declared in advance

1. **Items with no extractable hateful surface cannot move.** Round 1 already found that 13 of 35
   S-bucket items were called `hate_surface_present = false`, i.e. the model and the corpus
   disagree about whether hate is present at all. Those items will yield zero spans, mask to an
   unchanged transcript, and the mechanism has no purchase on them. This is reported, not hidden.
2. **Frames are not masked.** The 8 frames still carry whatever visual hateful content the video
   has. A negative result therefore cannot distinguish "bias is not caused by transcript wording"
   from "bias is caused by the frames", and the write-up will say so.
3. **Masking is lossy.** Some HateMM `S_FP` items (archival newsreel, song performances) are
   *entirely* hateful surface; masking them leaves a transcript consisting only of placeholders
   and the model is left with frames plus the placeholder text. That is the honest form of the
   mechanism, not a bug, but it means those items test "can frames + an assertion of hate
   determine stance", which is a harder task than the mechanism's motivating story.
4. **Two calls per item** means step 2's input is contaminated by step 1's errors. Extraction
   quality is therefore audited explicitly (§5.2) rather than assumed.

---

## 10. Execution deviations (logged when they happened)

| # | deviation | when | reason |
|---|---|---|---|
| **D1** | a **JSON-salvage path** was added to the masking stage, and the step-2 batch was rebuilt and re-submitted | 2026-08-12 21:40, **after step 1 was fetched but before any step-2 output existed** — the step-2 upload was killed mid-flight and no stance batch was ever created, so no masked-stance result was in existence when this was decided | 5 of the 98 extraction replies were unparseable because `qwen3-vl-plus` corrupted the `"text"` **key** of a span object into a bare commentary string (e.g. `{" irresponsibly transcribed as: \"…\"", "target": "…"}`). Those 5 items would have gone to step 2 **completely unmasked**, i.e. silently as baseline-condition items inside the masked arm — including `non_hate_video_16` (`S_FP`) and `BV1Kh411T7FJ` (`S_FN`), 2 of the 36 items in the primary metric. `salvage_spans()` recovers every well-formed `"text"` field and pairs it with the next `"target"` before the following `"text"`; unpaired spans keep the generic placeholder. It never invents a span. Effect: spans 230 → 275, items with ≥ 1 placeholder 64 → 69, unmatched still 0. **No judgement rule, prompt, sample or gold was touched.** |
| **D2** | 1 of 99 items (`ImpliHateVid::EX_329`) has no extraction and therefore no stance prediction | at step-1 run time | DashScope **output** moderation rejected the extraction reply (`InternalError.Algo.DataInspectionFailed: Output data may contain inappropriate content`). Note this is a *different* item and a *different* moderation surface from the previous pilot's `MHC_zh::BV1m8411z7mV` input-image rejection: here the model's own extracted spans tripped the filter. `EX_329` is an ImpliHateVid item, so it falls in view C only and the primary view-A denominator is unaffected. |

## Appendix A — prompt revision log

All iteration was on the **8 smoke items only**, all outside the 99-item eval sample, plus 5
hand-written synthetic transcripts belonging to no dataset. **No eval item was sent during prompt
development.** Nothing in §5 was changed at any point.

| version | change | reason |
|---|---|---|
| E1.0 / M1.0 | initial extraction prompt (`mask_prompts.py::EXTRACT`) + V1.3 verbatim + MASKING NOTE | frozen design above |
| **masking code E1.1** (no prompt text changed) | (a) the minimum span length for acceptance dropped from 3 normalised characters to 2 **for exact matches** (fuzzy matching still needs ≥ 5); (b) **every** exact occurrence of a span is masked, not only the first; (c) a residual-leak audit was added, checking whether any extracted span survives verbatim in the masked transcript | (a) the extractor returned the span `贱人` (2 characters) for the MHC-ZH smoke control `BV1Dm4y1J7Pj`, whose entire transcript is `贱人一个真服了`; the 3-character floor silently dropped it and the item went to step 2 unmasked — a CJK-specific implementation bug that would have voided the mechanism on short Chinese items. (b) a slur repeated N times but listed once by the extractor would have leaked N−1 times into step 2's context, which is precisely the trigger the pilot exists to remove. |

**Smoke inspection, 8 items, hand-read (this is the §6 item-1 gate).** After E1.1: 17/17 spans
located, **0 unmatched**, **0 residual leaks**, 5 of 8 items carry ≥ 1 placeholder. Judgement on
each: `hate_video_295` (anti-refugee song) — 6 spans, the anti-refugee/anti-Muslim lines, correct,
44 % of characters masked; `hate_video_54` — 8 spans, the racial and misogynist lyrics, correct,
46 % masked; `lzKJ_AWegCc` — the one slur in the title, correct; `BV1Dm4y1J7Pj` — `贱人`, correct;
`NH_180` — one ASR-garbled racial phrase, defensible over-extraction, 1.8 % masked; `NH_350`,
`sVA-q76vNBo`, `NH_836` — zero spans, correct (profanity aimed at the CIA / a coming-out sketch /
a sermon; no identity attack). **Framing text survives masking intact** in every case, which is
the property the mechanism depends on. The extraction step is accepted; the §6 item-1 retry is
NOT consumed.

**Label-space reachability under masking (synthetic, zero dataset contact).** Five hand-written
transcripts were pushed through the *complete* two-step pipeline (extract → mask → masked stance,
text-only) to confirm the MASKING NOTE does not mechanically force `endorses`:

| synthetic transcript | placeholders | emitted `stance` |
|---|---|---|
| creator denouncing an "immigrants are cockroaches" clip | 1 | **condemns** ✓ |
| neutral news item about a councillor's "vermin" remark | 1 | **reports** ✓ |
| creator replaying a misogynistic screenshot, taking no side | 1 | **quotes_mentions** ✓ |
| creator asserting the same misogynistic line in their own voice | 2 | **endorses** ✓ |
| sourdough baking tutorial | 0 | **no_hate_content** ✓ |

5/5. The label space is live under masking; any collapse observed on real videos is therefore not
an artefact of the MASKING NOTE.

Prompts frozen at **E1.0 (extraction text) / masking code E1.1 / M1.0 (stance)** before the eval
batch was built.

---

# Appendix B — CONTRAST STANCE PILOT (pinned-comment comparison), frozen 2026-08-13

**Frozen 2026-08-13 (Pacific/Auckland), before any API call of this pilot was made — including
before the smoke pass.** Author: subagent (Opus 5). Mechanism authorised by the user on
2026-08-13 and specified by the user turn-by-turn; this appendix records it verbatim as the
binding pre-registration. Results go to `idea-stage/CONTRAST_STANCE_RESULT.md`.

Scope: **one cheap capability measurement, run to an accuracy number and stopped.** No GPU, no
training, no downstream component, no detector hyper-parameter selected. Same data boundary and
authorisations as §1 above (DashScope, frames permitted, test *inputs* only, key from
`~/.dashscope_api_key`, never written to any file). Budget cap **≤ ¥5**.

## B.1 The question

Direct 5-way classification asks stance at **0.257** (`STANCE_PILOT_RESULT.md`, view A).
Masking the hateful surface gives **0.371** and that lift was shown to be an accounting artefact
(`MASK_STANCE_PILOT_RESULT.md`). The diagnosis carried forward: **the answer options are not
symmetric.** In a 5-way classification whose classes are `endorses` / `quotes_mentions` /
`condemns` / `reports` / `no_hate_content`, `endorses` is the safety-salient option a
safety-tuned model reaches for whenever attacking material is present, and the other four are
not equally available. The measurement is therefore confounded with a response bias.

**The mechanism under test converts the classification into a symmetric two-alternative forced
choice between two concrete first-person sentences**, neither of which is a moderation label.
Both options are the same kind of object (a sentence the uploader might write), of comparable
length and comparable emotional heat, so a safety prior has no preferred landing spot.

**The framing is a pinned comment, not transcript continuation.** Many uploaders say nothing at
all inside the transcript (the video is archive footage, a song, a broadcast clip, a drama
excerpt), so a "what would this speaker say next" continuation would measure the *filmed
speaker*, not the *publisher* — the wrong object. Asking what the **publisher would pin under the
video** names the right party explicitly and works identically whether or not the publisher
speaks on the audio track.

## B.2 Sample, model, inputs — unchanged from the two previous rounds

- Sample: `idea-stage/stance_pilot/sample.json` (seed 20260811), **byte-identical**, 99 eval items
  (`S_FP` 30 / `S_FN` 19 / `CTRL_HATE` 25 / `CTRL_NONHATE` 25) + the 8 disjoint smoke items.
- Model: **`qwen3-vl-plus`** (moving alias; the pinned snapshot is not accepted by the Batch API,
  previous pilot deviation D2). Run date recorded in place of a pin.
- Visual input: **8 evenly spaced frames, max side 512 px, JPEG q80** — identical spec to both
  previous rounds.
- Text input: the **full, unmasked, untruncated** transcript (`data/gt/<ds>/test.jsonl` `text`,
  HTML stripped). For `MHC` / `MHC_zh` the first sentence before the first ` . ` is the video
  title and is flagged as such in the prompt, exactly as in the previous rounds. **No OCR**
  (held out, same as both previous rounds).
- Decoding: `temperature=0.0`, `seed=20260813`, `max_tokens=8`.
- System prompt: `idea-stage/stance_pilot/prompts.py::SYSTEM`, verbatim.

## B.3 The prompt (frame C1.0)

```
VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese,
and may be noisy or empty):
<<<
{transcript}
>>>

Suppose the PERSON WHO MADE AND POSTED THIS VIDEO -- the uploader/publisher, NOT anyone who is
merely filmed, quoted, sampled or replayed inside it -- pins one comment under the video that
sums up their own view. Which of these two sentences is more like what they would write?

A: {opt_a}
B: {opt_b}

Answer with exactly one character, A or B. No explanation, no punctuation, nothing else.
```

Reply parsing: uppercase, strip; the first standalone `A` or `B` is the vote; anything else is an
unparseable vote and is dropped (counted and reported).

## B.4 Template bank — FROZEN, fixed for all videos, 5 pairs per language

Language of the templates for an item: **ZH iff CJK characters ≥ 15 % of the non-whitespace
characters of the transcript** (the same rule as the masking stage of the previous pilot); if the
transcript is shorter than 10 characters the dataset default is used (`MHC_zh` → ZH, else EN).

**Variant 1 (generic, run on all items).** `E` = endorsing direction, `O` = opposing direction.

| # | EN endorsing | EN opposing |
|---|---|---|
| 1 | Honestly, they deserve every word of this. | This kind of talk is disgusting and needs to be called out. |
| 2 | I stand by every word said here — it needed saying. | I posted this to push back on what is said here, not to spread it. |
| 3 | That is exactly what people like that are, and I am done pretending otherwise. | Nobody should be talked about like that, full stop. |
| 4 | If you agree, share it — more people need to hear this. | Share it so more people see what is being said here and reject it. |
| 5 | Say it louder for the ones in the back. | This is the kind of thing we should all be ashamed of. |

| # | ZH endorsing | ZH opposing |
|---|---|---|
| 1 | 所以说这种人就该被这么说,没冤枉他们。 | 这种言论真的令人作呕,必须抵制。 |
| 2 | 这里说的每一句我都认,该说的就得说。 | 我发这个是为了反驳这些话,不是为了传播它。 |
| 3 | 他们本来就是这样,我不想再装看不见了。 | 谁都不该被这样对待,就这么简单。 |
| 4 | 认同的就转发,让更多人明白这个道理。 | 转发出去,让更多人看清这种话有多离谱。 |
| 5 | 说得太对了,大声点说给后面的人听。 | 这种话我们都该觉得羞耻。 |

**Variant 2 (target-slotted, secondary, run only on items that have a target).** Identical pairs
with the attacked group named. `{t}` is filled from the **previous pilot's extraction step**,
`idea-stage/mask_stance_pilot/extract_m1.jsonl`: the most frequent non-empty `target` field over
that item's extracted spans (ties → first occurrence). Items with no such target run variant 1
only. 64 of the 98 extracted items carry a target.

| # | EN endorsing (variant 2) | EN opposing (variant 2) |
|---|---|---|
| 1 | Honestly, {t} deserve every word of this. | Talking about {t} like this is disgusting and needs to be called out. |
| 2 | I stand by every word said here about {t} — it needed saying. | I posted this to push back on what is said here about {t}, not to spread it. |
| 3 | That is exactly what {t} are like, and I am done pretending otherwise. | Nobody should talk about {t} like that, full stop. |
| 4 | If you agree about {t}, share it — more people need to hear this. | Share it so more people see what is being said about {t} and reject it. |
| 5 | Say it louder for the ones in the back — {t} needed to hear that. | The way {t} get talked about here is something we should all be ashamed of. |

| # | ZH endorsing (variant 2) | ZH opposing (variant 2) |
|---|---|---|
| 1 | 所以说{t}就该被这么说,没冤枉他们。 | 这样说{t}真的令人作呕,必须抵制。 |
| 2 | 这里关于{t}说的每一句我都认,该说的就得说。 | 我发这个是为了反驳这些关于{t}的话,不是为了传播它。 |
| 3 | {t}本来就是这样,我不想再装看不见了。 | 谁都不该这样说{t},就这么简单。 |
| 4 | 认同的就转发,让更多人明白{t}是什么样。 | 转发出去,让更多人看清这样说{t}有多离谱。 |
| 5 | 说得太对了,大声点说给{t}听。 | 这里对{t}的说法,我们都该觉得羞耻。 |

**Position randomisation.** For every (item, variant, pair) the endorsing template goes to slot A
iff `sha256("20260813|<dataset>|<id>|<variant>|<pair>").digest()[0] & 1 == 0`. Deterministic,
order-independent, reproducible, and independent of the content. **The mapping from slot to
direction is stored per request and applied at scoring time**, so a model that always answers "A"
scores at chance rather than at 1.0.

**Aggregation.** Per item and variant, the 5 votes are collapsed by simple majority into
`ENDORSE` / `OPPOSE`. If an odd vote fails to parse and the remaining valid votes tie, the item
is scored **WRONG** in the primary metric (never dropped); if **zero** votes parse the item leaves
the denominator and is counted and named.

## B.5 Gold — binary collapse of the frozen §4 gold, decided now

| group | binary gold | source |
|---|---|---|
| `S_FP` | **OPPOSE** (`condemns` ∪ `reports` ∪ `quotes_mentions` all collapse to OPPOSE) | dataset label = non-hate, plus the round-5 agent bucket code |
| `S_FN` | **ENDORSE** | dataset label = hate |
| `CTRL_HATE` | **ENDORSE** | dataset label = hate, detector correct |
| `CTRL_NONHATE` | **not applicable** — these are `no_hate_content` items; the question presupposes a stance that does not exist. **Reported separately, never in the primary metric.** | dataset label = non-hate, detector correct |

## B.6 Judgement rule — FROZEN, computed after the run, never adjusted

**M1 — primary.** Binary stance accuracy on the S bucket (`S_FP` ∪ `S_FN`), **view A = the
frame-bearing items** (HateMM + MHC + MHC_zh, 72 items of which 36 are S), **variant 1**,
**excluding any eval item that was used in the smoke pass**.
**PASS iff `acc_S ≥ 0.70`.**

*Declared口径 difference, stated before the run:* this is a **two-way** task, so its chance
baseline is **0.50**, whereas the 0.257 and 0.371 it is compared against are **five-way** tasks
with a chance baseline of 0.20. The bar 0.70 is held at the same numeric value as the two previous
rounds for continuity of the decision, and **both baselines are reported side by side in the
result document.** A number above 0.50 but below 0.70 is a FAIL, and is additionally reported as
"above two-way chance but below the bar".

**M2 — damage control.** `#{x ∈ CTRL_HATE : vote = OPPOSE} / |CTRL_HATE|`, view A, variant 1.
**PASS iff ≤ 0.15.**

**Verdict: PASS requires M1 ∧ M2.**

## B.7 Strata and diagnostics — frozen now, all descriptive, none may become the verdict

1. **Views.** A = 72 frame-bearing (primary) · B = all 99 · C = the 27 transcript-only
   ImpliHateVid items. B and C are appendix.
2. **Cells.** `S_FP` and `S_FN` accuracy reported separately; `S_FP` is the double-anchored cell
   and the one the mechanism exists for.
3. **Voice form (`作者有话` vs `作者无话`).** Stratify the S items by the **hand-coded** gold voice
   of `idea-stage/voice_field_analysis.py::GOLD_VOICE` (coded blind for 46 of 49 items, per that
   document's F7): `OWN` = the hate-associated surface is produced by the video's own author →
   **作者有话**; `NOT_OWN` = archive / broadcast / named third party / embedded clip →
   **作者无话**; `UNDET` items are reported as their own cell and excluded from the two-way
   comparison. The `OWN` subset is expected a priori to be the mechanism's main battleground and
   its accuracy is reported explicitly.
4. **Vote-pattern distribution.** For each of the 5 template pairs, its own accuracy and its own
   endorsing-rate; the distribution of vote splits (5-0, 4-1, 3-2) and accuracy by split.
5. **Position bias.** The raw rate at which slot `A` was chosen, over all calls, irrespective of
   which direction slot A carried. At 0.50 there is no position bias.
6. **Lexical-overlap check.** For each item and pair, tokenise EN as lowercased `[a-z']+` minus a
   fixed stopword list, and ZH as CJK character bigrams; define
   `ov(template) = |tok(template) ∩ tok(transcript)| / |tok(template)|`, and
   `d = ov(endorsing) − ov(opposing)`. Per item let `D = mean_p d_p`. An item is **overlap-aligned**
   if `sign(D) = sign(vote)` and `D ≠ 0`. Report the overlap-aligned share ("重合度驱动占比"),
   the per-pair agreement rate between `sign(d_p)` and the pair's vote, and the accuracy of the
   aligned vs non-aligned subsets. A high aligned share means the model is matching words rather
   than judging stance.
7. **Variant 1 vs variant 2** on the 64-item target-bearing subset, paired.
8. **Per dataset.**

## B.8 Failure plan — at most ONE iteration, then stop

Smoke = the 8 disjoint smoke items **plus the 3 eval items the user named as mandatory
qualitative checks** — `MHC::KDcCiUU8q5E` (the Trump-misogyny counter-speech item),
`HateMM::non_hate_video_32` (the Lennon/Ono song), `HateMM::non_hate_video_16` (the 1956
segregationist newsreel). Prompt-frame and template edits are permitted **once** on the basis of
the smoke pass and are logged in B.10. Because those 3 eval items are seen before the freeze is
consumed, **they are removed from the primary metric denominator** (B.6 M1) whatever happens;
their formal-batch answers are still run and are reported as a named secondary line together with
the all-S number that includes them.

**If the mechanism FAILS**, the recorded conclusion is, verbatim:
**"零样本 MLLM 判立场"全路径关闭** — direct classification (0.257), content masking (0.371) and
symmetric comparison have all been measured on the same 99 items with the same model, and no
prompt-level intervention reaches the bar. No fourth prompt-level mechanism is attempted.

## B.9 Process rules

- B.4–B.7 are frozen before the first API call of this pilot and are not adjusted afterwards.
- Single submission for the paid run; re-submission only on infrastructure failure, logged as a
  deviation in B.10.
- Every stage caches under `idea-stage/contrast_stance/` and is idempotent/resumable.
- Logs `logging/runs/contrast_stance/run.{log,pid}`; backgrounded, survives SSH disconnect.
- Honest reporting: if the numbers are bad they are reported as they are.

## B.10 Deviation and prompt-revision log (appended as they happen)

| # | what | when | why |
|---|---|---|---|
| **R1** | **Variant-1 template bank revised C1.0 → C1.1** (variant 2, the prompt frame C1.0, the language rule, the position randomisation, the gold and every judgement rule in B.5–B.7 are UNTOUCHED). Every option now names its referent — "the people the video's harsh words are about" — instead of pointing at an unnamed "this"/"here". | 2026-08-13, on the smoke pass, **before the eval batch was built**; this consumes the single revision permitted by B.8 | The C1.0 endorsing options were **referentially open** and are simply *true of a counter-speech author's own criticism*, so they did not implement the intended contrast. Evidence from the smoke: on `ImpliHateVid::NH_836` (a sermon, no identity attack anywhere) the endorsing option won 4/5 — "I stand by every word said here" and "Say it louder for the ones in the back" are things the author of *any* video would pin; same pattern on `MHC::sVA-q76vNBo` (a coming-out sketch, 4/5). On the mandatory check `MHC::KDcCiUU8q5E` (Trump-misogyny counter-speech) the generic variant returned ENDORSE on exactly the three referentially-open pairs (1 "I stand by every word said here", 2 "people like that", 3 "if you agree, share it") while the target-named variant 2 — which does name its referent — returned OPPOSE 4/5. This is an implementation defect in the template bank, not a measurement of the model. |
| **R2** | **Scoring addendum, no rule changed:** the two previous rounds are reported against this one in *both* of two forms, fixed here. (a) **as published** — their 5-way accuracy under the §5 rule (0.257 direct / 0.371 masked). (b) **binarised for like-for-like comparison with a forced choice** — a 5-way answer maps to ENDORSE iff it is `endorses`, and to OPPOSE for all four of `quotes_mentions`, `condemns`, `reports`, `no_hate_content`; `no_hate_content` is a non-endorsing answer and so counts as OPPOSE. Both are computed on **exactly the same rows** as this pilot's metric. M1/M2 and the 0.70 bar are untouched. | 2026-08-13, written before the eval batch was submitted | The headline compares a two-way forced choice against two five-way classifications; the comparison is only honest if the chance baselines (0.50 vs 0.20) and a binarised version of the old rounds are both on the table. Fixing the binarisation now prevents choosing it after seeing the numbers. |
| **R3** | Smoke gate **passed** after R1; the B.8 single revision is now **consumed**. Templates frozen at **C1.1 (variant 1) / C1.0 (variant 2) / frame C1.0**. | 2026-08-13, before the eval batch was built | 65/65 smoke votes parsed, position randomisation verified active (slot-A rate 0.385, and answers track the direction rather than the slot), `KDcCiUU8q5E` now 3/5 OPPOSE under variant 1 and 4/5 under variant 2, `NH_836` (no hate content) no longer forced to the endorsing side. The instrument works mechanically. No further iteration is permitted. |
| **D1** | 1 of 99 items (`MHC_zh::BV1m8411z7mV`) produced **no prediction at all**: all 10 of its requests (5 pairs × 2 variants) were rejected with `InternalError.Algo.DataInspectionFailed: Input image data may contain inappropriate content`. View A is 71 items rather than 72 and the S denominators account for it. | at eval-batch run time | Vendor **input-image** moderation. Third consecutive round losing this same item on this same surface (round 1 deviation D6, round 2 §9). No rule, prompt, sample or gold was touched; the loss is reported, not worked around. |

**Outcome, recorded 2026-08-13:** M1 = **0.469** (15/32, bar 0.70, exact binomial p = 0.860 against
the two-way chance baseline 0.50) · M2 = **0.000** (0/18, bar 0.15, a pass by degeneracy) ·
**verdict FAIL**. 805/805 votes parsed, no ties. The B.8 failure clause is therefore in force:
**"零样本 MLLM 判立场"全路径关闭**. Full write-up: `idea-stage/CONTRAST_STANCE_RESULT.md`.
