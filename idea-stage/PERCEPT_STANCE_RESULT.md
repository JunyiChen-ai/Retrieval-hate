# PERCEPT_STANCE_RESULT — the perception questionnaire was not run: its premise fails at gate 0

**Verdict: STOP at gate 0. Cost ¥0.00 — no API call was made.**

The design assumed that the bias in the three failed stance rounds lives only in the *judgement*
step, and that the *perception / transcription* step is clean. The pre-specified zero-cost check
of that assumption returns **0.111 against a bar of 0.30**. The assumption does not hold on the
data that motivated it, so the questionnaire was not built and no money was spent.

Date 2026-08-15 (Pacific/Auckland). Pure offline re-analysis of files already on disk:
`idea-stage/stance_pilot/pred_strong.jsonl` (round 1, 98 rows), `sample.json` (99 items, seed
20260811), `idea-stage/voice_field_analysis.py::GOLD_VOICE`, `data/gt/<ds>/test.jsonl`
transcripts. Zero API calls, zero GPU. Frozen design, unexecuted, in
`idea-stage/PERCEPT_STANCE_FREEZE.md`. Audit code and per-item codes in
`idea-stage/percept_stance/step0_evidence_audit.py` → `step0_audit.json`.

---

## 1. Gate 0 — the number

Question: for every item whose round-1 stance answer is **wrong** in the binary reading, does the
`evidence` sentence explicitly state a fact of the class that would have pointed to the **correct**
direction? For `gold = OPPOSE` that means an attribution / quotation / reportage / criticism fact
(or a fact showing the author supports the group); for `gold = ENDORSE` it means the mirror
own-voice fact. Full coding rule at the head of the audit script.

| stratum | items | stance wrong | **evidence carries the right direction fact** | ratio of wrong | ratio of all |
|---|---|---|---|---|---|
| **`S_FP` (the gate)** | 29 | 18 | **2** | **0.111** | 0.069 |
| `S_FP`, smoke items removed | 27 | 16 | 2 | 0.125 | — |
| `S_FP`, frame-bearing only | 21 | 12 | 1 | 0.083 | — |
| `S_FP`, no frames | 8 | 6 | 1 | 0.167 | — |
| `S_FN` | 19 | 9 | 3 | 0.333 | 0.158 |
| `CTRL_HATE` | 25 | 2 | 0 | 0.000 | 0.000 |
| `CTRL_NONHATE` | 25 | 4 | 2 | 0.500 | 0.080 |

**Gate: 0.111 < 0.30 → premise does not hold → stop, spend nothing.** The gate is not close, and
it does not move under either lenient variant: folding in the sibling `primary_voice` field as an
additional source of the direction fact raises `S_FP` to **3 / 18 = 0.167**; the ratio over all
`S_FP` items rather than only the wrong ones is **0.069**. The only strata above the bar are
`S_FN` (0.333, n = 9) and `CTRL_NONHATE` (0.500, n = 4) — neither is the cell the mechanism was
designed to fix, and both are too small to carry a decision.

The two `S_FP` items that pass are `HateMM::non_hate_video_121` (the evidence names "Greg Anglin"
as the utterer, i.e. it does attribute the line to a third party, and then answers `endorses`
anyway) and `ImpliHateVid::NH_988` (the evidence writes "self-identifies as supporting gay rights
for Palestinians" and then answers `endorses`).

## 2. Why the premise fails: the evidence field is a verdict, not a perception

A second, non-gating column separates the two things that "the evidence is right" could mean.

| reading | `S_FP` wrong items |
|---|---|
| the evidence's description of the **content** is factually accurate | **15 / 18 = 0.833** |
| the evidence states a fact pointing to the **correct direction** | **2 / 18 = 0.111** |

The model paraphrases what is said accurately in 5 of 6 wrong cases. What it does not do is
produce the provenance facts. And when it does say something about provenance, it is usually a
verdict rather than an observation: **8 of the 18** wrong `S_FP` items carry an explicit clause of
the form *"presented without criticism, attribution, or distancing"* / *"with no critical
framing"* / *"without critique or context"*, and on **5 of those 8** that clause is demonstrably
false against the transcript or against the hand-coded provenance
(`non_hate_video_121` is a sting call, `non_hate_video_149` an archival repost,
`non_hate_video_400` an Ellen-show clip with an on-air introduction, `NH_988` contains
"I stand for gay rights for all Palestinians", `NH_875` is mocking the clip it quotes). On the
other 10 items the evidence says nothing about provenance at all. Either way the perception step
does not hand the direction fact to a downstream rule.

Three of the 18 also carry an outright wrong **speaker attribution** — the evidence names the
uploader as the utterer of material that is demonstrably third-party:

| item | what it actually is | what the evidence says |
|---|---|---|
| `ImpliHateVid::NH_875` | the line "women belong in the kit" is inside the embedded Starbucks parody the uploader is mocking ("check this clown out") | "The uploader states 'women belong in the kit' … presented without criticism or attribution" |
| `MHC::pofgIFZpR7c` | title byline "YOU NEED to stop MASTURBATING - Hamza Ahmed"; a clip of a third party's monologue | "The uploader equates frequent masturbation with self-castration" |
| `HateMM::non_hate_video_642` | a third-party Kiffness music video re-uploaded with the re-uploader's own site URLs overlaid | "The uploader performs lyrics repeatedly stating 'being white is a 21st century crime'" |

Two more attribute to the wrong party in the other direction: `MHC::YDEsYXYlB8o`'s transcript
attributes the claim to "a STEM professor … published their findings" and the evidence attributes
it to the uploader; `MHC::XlJCNPi5inM`'s title reads "Kevin Samuels explains …" and the evidence
says only "the speaker asserts".

This is the same object `VOICE_FIELD_ANALYSIS.md` measured from the other side and buried:
the provenance categories the whole idea rests on — `quoted_third_party` and `archival_source` —
were emitted **2 times in 71 frame-bearing items**, and the voice field disagrees with itself on
half the items across a paraphrase of its own question (κ ≈ 0.34, §5 there). Gate 0 adds the
piece that document left open: the failure is not only that the *typed* voice field is unstable,
it is that the model's own free-text account of what it saw supplies the right provenance fact on
2 of 18 wrong `S_FP` items and a demonstrably false one on 5.

## 3. Where this leaves the four rounds (same 32-row view)

No new number was produced, because nothing was run. The measured table is unchanged:

| round | mechanism | task | chance | acc on the same 32 `S` rows |
|---|---|---|---|---|
| 1 | direct 5-way classification | 5-way | 0.20 | 0.281 (0.500 binarised) |
| 2 | content masking | 5-way | 0.20 | 0.375 (0.563 binarised) |
| 3 | symmetric pinned-comment contrast | 2-way | 0.50 | 0.469 |
| **4** | **perception questionnaire + hand-written rules** | 2-way | 0.50 | **not run — gate 0** |

Round 4 was the first of the four whose premise was checkable for free before paying, and it is
the first that stopped before paying.

## 4. Five items, traced

The task asked for five failure cases. Since nothing was run, these are the five most informative
gate-0 items — what the perception step returned, and why a downstream rule set could not have
recovered from it.

1. **`HateMM::non_hate_video_16`** (1956 segregationist newsreel, `gold = OPPOSE`, answered
   `endorses`). Evidence: *"The on-screen speaker uses the racial slur 'nigger' repeatedly to
   demean Black musicians …"* — accurate content, zero provenance. The transcript contains no
   attribution phrase and no reaction sentence at all; it is pure first-person committee speech.
   `Q1`, `Q2` would both return empty, `Q4` has no second-person pronoun, so the frozen rules fall
   to R5 (conservative `ENDORSE`) unless the frame/OCR branch R4 rescues it from the
   GlobalImageWorks stock-footage timecode watermark. On this item the whole design reduces to one
   OCR string.
2. **`HateMM::non_hate_video_32`** (Lennon/Ono song, `gold = OPPOSE`, answered `endorses`).
   Evidence explicitly asserts *"with no critical framing, attribution, or distancing by the
   uploader"*. The transcript is lyrics with no attribution and no reaction; same R5 fall-through.
   Every arm of every previous round answered this item `endorses` unanimously.
3. **`ImpliHateVid::NH_875`** (embedded parody, `gold = OPPOSE`, answered `endorses`). The
   transcript *does* contain the reaction material the questionnaire wants — "all this guy did was
   make himself look like an idiot", "check this clown out" — and the model's perception step
   nevertheless reported the uploader as the utterer. This is the case that most directly
   contradicts the premise: the cue was present and copyable, and the free-text perception got it
   wrong anyway.
4. **`MHC::8zLoOqXvk64`** (`gold = OPPOSE`, answered `endorses`). Whole transcript is 38
   characters: the title `"Be aware of transgender on road"`. Perception is exactly right — the
   title is quoted verbatim — and there is nothing to perceive that points anywhere. All six
   questionnaire fields would return empty or null, and the rules fall to R5. A meaningful
   fraction of the `S_FP` cell is this shape.
5. **`ImpliHateVid::NH_650`** (scripted two-voice debate skit, `gold = OPPOSE`, answered
   `endorses`). The transcript is a dialogue full of reaction sentences ("That's called circular
   reasoning", "All you're doing is rejecting evidence …"). The evidence sentence collapses both
   voices into "the uploader" and reports no dialogue structure. `Q2` is exactly the field designed
   to catch this, and the free-text perception did not catch it.

Items 1, 2 and 4 fail for a different reason than 3 and 5: for them the questionnaire has nothing
to copy, so the design's outcome is decided entirely by the R5 default, not by perception. That is
a second, independent problem the gate surfaced — a conservative `ENDORSE` default on an
information-free item reproduces exactly the blanket-`ENDORSE` behaviour that killed round 3.

## 5. What this does and does not establish

- It establishes that **the specific premise this design was built on — "`evidence` often states
  the right facts while `stance` answers wrong" — is not true on the `S_FP` cell**, which is the
  only cell that matters. It is true on `S_FN` (0.333) and on the four wrong non-hate controls
  (0.500), but those are not where the mechanism was supposed to earn anything.
- It does **not** establish that a *dedicated copy-only* questionnaire would fail. The round-1
  `evidence` field was a free-form justification produced in the same call as the stance verdict;
  it was never asked to copy attribution phrases, and it had every incentive to justify the answer
  it had just given. A separate call that asks only "copy the phrases outside «» that say who is
  speaking" is a strictly easier task and was never measured. The gate is a check on the *stated
  premise*, not a measurement of the questionnaire.
- What it does add against that optimistic reading is §2: on 5 of 18 items the model volunteered a
  *false provenance claim* rather than staying silent, and on 3 of 18 it named the uploader as the
  utterer of demonstrably third-party material. A copy-only prompt removes the incentive to
  justify a verdict; it does not obviously remove that.
- The frozen design in `PERCEPT_STANCE_FREEZE.md` remains available and unmodified if the user
  chooses to override the gate. §6 there records that gate-0 adjudication required reading the 18
  `S_FP` transcripts, so blindness on those items is compromised for me and a re-run would have to
  be reported as non-blind or have its rules re-derived by someone else. Two operational
  prerequisites are also on record: `vaderSentiment` is **not** installed in the `HateVideo`
  environment, and there is no Chinese sentiment component, so all `MHC_zh` numbers would come
  from a degraded rule path.

## 6. 与 2404.01651 结构化提示线的区分

Gligorić et al. (NAACL 2024, `arXiv 2404.01651`) show the use-vs-mention failure in text
classifiers and repair it partially by **prompting**: they put the use–mention distinction, CoT
and few-shot exemplars into the prompt, so the model is still the thing that decides whether a
passage is used or mentioned. The design frozen here is the opposite arrangement — the model is
forbidden from seeing the words `attack`, `hate`, `stance` or `endorse` and is only allowed to copy
strings and point at frames, while the use-vs-mention decision is made outside the model by a
hand-written rule table plus an off-the-shelf sentiment scorer. That is the distinguishing claim,
and gate 0 is a check on its load-bearing assumption rather than on the arrangement itself: the
arrangement only pays if the copying step is more reliable than the deciding step, and on the
evidence already paid for it is not measurably so on the cell that matters.

## 7. Cost

| item | value |
|---|---|
| API calls | **0** |
| DashScope spend | **¥0.00** (cap ¥5) |
| GPU | none |
| inputs | files already on disk |

## 8. Reproducibility index

| artefact | path |
|---|---|
| gate-0 coding rule, per-item codes, scorer | `idea-stage/percept_stance/step0_evidence_audit.py` |
| gate-0 machine output | `idea-stage/percept_stance/step0_audit.json` |
| gate-0 console log | `logging/runs/percept_stance/step0.log` |
| frozen, unexecuted design (prompt, rules, decision line, contamination disclosure) | `idea-stage/PERCEPT_STANCE_FREEZE.md` |
| round-1 predictions audited | `idea-stage/stance_pilot/pred_strong.jsonl` |
| sample, gold, grouping (unchanged, seed 20260811) | `idea-stage/stance_pilot/sample.json` |
| hand-coded provenance gold | `idea-stage/voice_field_analysis.py::GOLD_VOICE` |
| the three previous rounds | `STANCE_PILOT_RESULT.md`, `MASK_STANCE_PILOT_RESULT.md`, `CONTRAST_STANCE_RESULT.md` |
| the voice-field re-analysis this extends | `VOICE_FIELD_ANALYSIS.md` |
