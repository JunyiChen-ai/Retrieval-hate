# PERCEPT_STANCE_FREEZE — perception questionnaire + hand-written rules

**Status: FROZEN, NOT EXECUTED.** The gate-0 premise check (§1) came back at **0.111 against a
bar of 0.30**, so per the pre-specified stop rule no API call was made and no money was spent.
The design below is recorded in full so that it is on the record unmodified, in case the gate is
overridden by the user. Everything in §2–§5 was written *after* gate 0 returned; see the
contamination disclosure in §6 before treating it as a blind design.

Date 2026-08-15 (Pacific/Auckland). Model that would have been used: `qwen3-vl-plus`, realtime
endpoint, temperature 0, 8 frames sampled uniformly by duration, 512 px, full transcript. Sample:
`idea-stage/stance_pilot/sample.json` (99 eval items, seed 20260811), gold and grouping unchanged
from the three previous rounds. Budget cap ¥5. API key read from `~/.dashscope_api_key` at
runtime, never written to any file.

---

## 1. Gate 0 (zero cost) — the premise check, and its result

**Premise under test.** The three failed rounds are read as "the bias lives in the *judgement*
step; the *perception / transcription* step is clean". The direct evidence for that reading was
that the round-1 `evidence` field often states the right facts while the `stance` field answers
wrong. Gate 0 measures how often that is actually true.

**Criterion (specified before any code was written).** For every item whose round-1 stance answer
is wrong in the binary reading, does the `evidence` sentence explicitly state a fact of the class
that would have pointed to the *correct* direction — for `gold = OPPOSE`, an
attribution / quotation / reportage / criticism fact; for `gold = ENDORSE`, the mirror own-voice
fact. Full coding rule, per-item codes and adjudication notes:
`idea-stage/percept_stance/step0_evidence_audit.py`; machine output `step0_audit.json`.

**Stop rule (specified before any code was written).** `S_FP` ratio **< 0.30 → premise does not
hold → stop, write the report, spend nothing.**

**Result: 2 / 18 = 0.111.** Stop rule fires. See `PERCEPT_STANCE_RESULT.md`.

---

## 2. Input construction (frozen)

- Transcript: full, from `data/gt/<ds>/test.jsonl` field `text`, HTML stripped, exactly as the
  three previous rounds loaded it (`stance_pilot/run_pilot.py::load_texts`). For `MHC` / `MHC_zh`
  the field is `"<title> . <transcript>"` and stays that way.
- Spans: reused verbatim from `idea-stage/mask_stance_pilot/extract_m1.jsonl` (**no
  re-extraction**). Each span is wrapped in `«…»` **in place**; nothing is deleted or replaced.
- Boundary trim, program side, applied before wrapping: if a span's leading or trailing tokens
  fall inside the attribution/evaluation lead-in word list below, those tokens are cut out of the
  span and returned to the surrounding context. Trim events are logged per item.
  Word list (frozen): `said, says, say, told, stated, states, claims, claimed, wrote, writes,
  quoted, quotes, according to, criticize(d/s), condemn(ed/s), slam(med/s), calls out, blasted,
  disgusting, appalling, horrific, shocking, outrageous, racist, sexist, homophobic, bigoted,
  hateful, 说, 称, 表示, 批评, 谴责, 抨击, 恶心, 令人作呕, 种族歧视, 性别歧视, 仇恨`.
- Items with no span (mostly controls): the questionnaire is still answered; every P-relative
  field returns `null`.
- Frames: 8, uniform by duration, max side 512 px, JPEG q80 — identical to rounds 1–3. The 27
  ImpliHateVid items have no frames and are run text-only; **their results are reported separately
  and the verdict is taken on the frame-bearing set only.**

## 3. The questionnaire (frozen prompt)

Hard constraint: the prompt may not contain `attack`, `hate`, `hateful`, `target`, `stance`,
`endorse`, `oppose`, `offensive`, `slur`, `bias`, or any Chinese equivalent, and may not ask for
any evaluation of P. P is referred to **only** by its `«»` position.

```
SYSTEM
You are a careful transcription assistant. You copy text exactly as it appears. You never
summarise, never paraphrase, never explain, never judge. If something is not present, you return
an empty list or null. Output JSON only.

USER
Below is the full transcript of a video. One or more passages are marked with « ». Call the marked
passages P. [If frames are attached: Eight frames sampled evenly from the video are attached.]

Answer these six questions about the material. Copy text verbatim; do not rewrite it.

Q1 attribution: outside P, copy every phrase that introduces P or that says who is speaking in P
   (for example "he said", "in this 1956 broadcast", a title-style introduction, a name followed
   by a colon). If there are none, return [].
Q2 reaction: outside P, copy every complete sentence that talks about P or about whoever is
   speaking in P. Copy whole sentences only; do not summarise.
Q3 quote_marks: is P inside quotation marks, or does P follow a colon that comes after an
   introducing phrase? Answer true or false.
Q4 pronouns: looking only inside P, list (a) every second-person pronoun (you, your, 你, 你们)
   and (b) every third-person group reference (they, them, those people, or a group name). Copy
   the words only.
Q5 screen_text: copy every piece of text visible in the frames — watermarks, channel logos,
   burned-in captions, dates. Copy only; if there are no frames or no text, return [].
Q6 setting: one short factual phrase describing what the frames show, e.g. "person facing camera
   in a room", "news studio with lower-third graphics", "black-and-white archival footage",
   "text cards". No adjectives about mood, tone or intent.

Return exactly:
{"Q1_attribution": [...], "Q2_reaction": [...], "Q3_quote_marks": true|false,
 "Q4_second_person": [...], "Q4_third_person_group": [...], "Q5_screen_text": [...],
 "Q6_setting": "..."}
```

Validation, program side: every string in `Q1, Q2, Q4a, Q4b` must be a substring of the
transcript (whitespace- and case-normalised); `Q5` strings must be a substring of the OCR cache
where one exists, otherwise they are accepted but flagged. Non-substring strings are marked
invalid and dropped, and the invalid rate is reported per field.

Smoke: at most the 8 `sample.json` smoke items. Prompt wording may be revised on the smoke pass
only; after that it is frozen and the smoke items never enter any metric.

## 4. The rules (frozen; direction may not change)

Sentiment component: **VADER** (`vaderSentiment`) for English, `compound ≤ −0.25` = negative.
**`vaderSentiment` is not installed in the `HateVideo` environment** (verified 2026-08-15) and
would have to be installed before a run. No Chinese sentiment component is available, so for
`MHC_zh` the rules degrade to "an attribution phrase or a reaction sentence exists ⇒ OPPOSE",
and every Chinese number must be reported as coming from that degraded path.

```
R1  Q1 non-empty:
      any Q2 sentence negative  -> OPPOSE
      Q2 empty or all neutral   -> OPPOSE            (quotation / reportage side)
R2  Q1 empty AND Q2 empty AND Q4_second_person non-empty       -> ENDORSE
R3  Q1 empty AND any Q2 sentence negative                      -> OPPOSE
R4  Q5 contains a stock-footage watermark or a news-channel logo, OR Q6 matches
    archival / news / broadcast  -> +1 weight toward OPPOSE (breaks ties, and promotes
    R5 to OPPOSE)
R5  otherwise (signals conflicting or all empty)               -> ENDORSE   (conservative
                                                                  default; counted as an
                                                                  abstention)
```
Reported alongside: abstention rate (R5 fires), per-rule fire count and per-rule accuracy.

## 5. Decision rule (frozen)

Primary metric = **binary accuracy on the 32-row main view** — frame-bearing `S_FP` + `S_FN`
items, the 3 smoke-contaminated items removed, `endorse` vs `distanced` — identical to the view
`CONTRAST_STANCE_RESULT.md` §1.1 reports.

| band | reading |
|---|---|
| ≥ 0.70 | success |
| 0.563 – 0.70 | AMBIGUOUS (0.563 = the masked round binarised on the same rows) |
| < 0.563 | FAIL |

Also reported, not gated: full-`S` and control views, field-level quality (substring pass rate,
`Q1` agreement with `voice_field_analysis.py::GOLD_VOICE`, boundary-trim fire count), strata by
`S_FP` / `S_FN` / frames / no frames / dataset, and 5 failure cases traced item by item.

## 6. Contamination disclosure

Adjudicating gate 0 required checking whether the round-1 `evidence` sentences were factually
true, which required reading the transcripts of all 18 wrong `S_FP` items. While doing so I also
formed an impression of how the §4 rules would route several of those items. **That impression was
not written down as a number and no candidate metric was computed**, but blindness on those items
is no longer intact for me. If the user overrides the gate, the honest handling is that §4 must be
re-derived by someone who has not read those transcripts, or the run must be reported as
non-blind. The §4 rule skeleton is the user's, transcribed without changing its direction; the
only additions are the sentiment threshold, the negation of R1's two branches into the same
binary label, and the word list in §2.
