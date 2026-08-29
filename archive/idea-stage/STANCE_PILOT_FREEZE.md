# STANCE_PILOT_FREEZE — capability gate for the gated-stance-annotation direction

**Frozen: 2026-08-11 (Pacific/Auckland), before any evaluation-sample API call was made.**
Author: subagent (Opus 5). Scope: one cheap capability measurement, no GPU, no training,
no test *labels* used for tuning.

---

## 0. The question this pilot answers, and why it is a gate

`idea-stage/MLLM_FRONT_RECON.md` proposes putting a frozen MLLM in front of the small
multimodal model and consuming a **typed stance record** (§4.0). `idea-stage/IDEA_REPORT.md`
§9.2 prices the prize: the **stance bucket S is 45.4 % of all 108 test errors and worth a mean
+6.46 macro-F1 if oracle-fixed**.

`idea-stage/STANCE_LIT_RECON.md` §5(3) names the binding risk explicitly:
`2406.00020` finds that when in-group / stance status must be **inferred from content**, the
**maximum F1 across every model and prompting scheme is 0.24** (whereas handing it to the model
as metadata lifts HARMFUL_IN 0.36 → 0.53). If a frontier VL model cannot type the stance of a
video, every design in `MLLM_FRONT_RECON.md` §4 (D1 SCLO, D2 TEV, D3 SCR) is dead at the root,
and the ~$50–150 generation pass plus a GPU day would be spent on noise.

**So this pilot measures only one thing: can a cloud VL model put the right stance type on a
video?** It trains nothing, touches no head, and produces no accuracy claim about the detector.

---

## 1. Authorisations and data boundary

- **User ruling 2026-08-11:** the user's own DashScope (Alibaba Qwen) API may be used for this
  pilot, and **video frames may be sent to it**. This is the same class of exemption as the
  2026-08-07 ruling for the Claude API; it does **not** relax the Modal raw-video block.
- **User ruling 2026-08-09 (test protocol):** test-set *inputs* may be used freely; test *labels*
  may not be used to tune and then re-reported as held-out. This pilot sends **test inputs** to
  the API and uses test labels only as the anchor of a **disclosed capability measurement**.
  Nothing here selects a model, a threshold or a hyper-parameter for the detector.
- The API key is read from `~/.dashscope_api_key` at runtime and is **never** written into any
  script, log, output file or report.

---

## 2. Sample (frozen, `idea-stage/stance_pilot/select_sample.py`, seed 20260811)

**Eval sample: 99 items**, disjoint from the smoke sample.

| group | definition | n |
|---|---|---|
| `S_FP` | primary-S bucket errors, false positives (label = non-hate, comparator said hate) | 30 |
| `S_FN` | primary-S bucket errors, false negatives (label = hate, comparator said non-hate) | 19 |
| `CTRL_HATE` | correctly predicted hate items (label = 1, not in `err_ids`) | 25 |
| `CTRL_NONHATE` | correctly predicted non-hate items (label = 0, not in `err_ids`) | 25 |

Per dataset — S errors 8 / 16 / 12 / 13 (HateMM / MHC-EN / MHC-ZH / ImpliHateVid); controls
allocated in the same proportion, 8 / 16 / 12 / 14, half hate and half non-hate within each.
Controls drawn with `numpy.default_rng(20260811)` over the lexicographically sorted id list.

**Smoke sample: 5 items, drawn from OUTSIDE the eval sample** so that prompt iteration on them
cannot contaminate the frozen metrics: 2 secondary-S errors (`hate_video_295`, `lzKJ_AWegCc`),
1 X-bucket error (`NH_350`), 2 controls from the leftover correct pool.

---

## 3. Inputs per item (round 1)

- **8 evenly spaced frames**, resized to max side **512 px**, JPEG q80 (~146 image tokens/frame
  measured). ImpliHateVid has **no local raw video** and therefore **no frames** — those 27 items
  are transcript-only, and are reported as a separate row throughout.
- **The FULL transcript, verbatim, no windowing and no truncation**
  (*user correction, 2026-08-11*: text is nearly free relative to image tokens, so round 1 gets
  the whole transcript; the earlier draft's "more transcript windows" fallback is deleted). The
  longest item in the sample is 11,971 characters ≈ 3 K tokens, far inside context. HTML tags in
  the MHC-ZH titles are stripped. If any item ever exceeded context it would be truncated at the
  tail and flagged in the output record — this did not occur.
- **No OCR text in round 1.** OCR is held back as fallback ① so the ladder is a real increment.

## 4. Task form

**Structured typed question schema, JSON only, no free-form chain of thought** — this is forced by
`STANCE_LIT_RECON.md` §3.7: MemeScouts (`2604.24179`) gains +8/+40/+46 macro-F1 from a constrained
question schema over direct prediction, while unconstrained CoT **loses** 10/22/13.

Emitted record (`idea-stage/stance_pilot/prompts.py`):

```
{"hate_surface_present": bool,
 "primary_voice": uploader|on_screen_speaker|quoted_third_party|archival_source|caption_overlay|none,
 "stance":        endorses|quotes_mentions|condemns|reports|no_hate_content,
 "target":        "<short noun phrase>"|null,
 "evidence":      "<=25 words"}
```

Three semantically equivalent prompt variants V1/V2/V3 exist; **only V1 is run**. V2/V3 are
stand-by and are used only if the fallback ladder in §7 is invoked.

Decoding: `temperature=0.0`, `seed=20260811`, `max_tokens=400`.

---

## 5. Gold standard (collapse rule, frozen)

The per-item human coding available is the **bucket** code in `idea-stage/r5_buckets.json`
(round-5 agent, transcript + HateMM OCR, coded before any count was taken). The S bucket is
defined there as *"hate-associated surface content whose stance is reportage, quotation,
counter-speech, satire, archival or education (FP side); or hate carried by endorsement framing
over neutral surface (FN side)"*. The collapse to the 5-way stance space is therefore:

| group | gold stance class (collapsed) | anchored also by |
|---|---|---|
| `S_FP` | **non-endorsing** = {`quotes_mentions`, `condemns`, `reports`} | dataset label = non-hate |
| `S_FN` | **`endorses`** | dataset label = hate |
| `CTRL_HATE` | **`endorses`** (assumed) | dataset label = hate |
| `CTRL_NONHATE` | **`no_hate_content` or non-endorsing** (assumed) | dataset label = non-hate |

The S groups have a **double anchor** (agent bucket code + the dataset's own label); the control
groups have only the dataset label plus an assumption, so their metric is stated as a
**directional error rate**, not as an accuracy. `depicts_without_comment` from the
`MLLM_FRONT_RECON.md` §4.0 draft schema is **merged into `quotes_mentions`** here to keep the
5-way space the user specified.

Known gold weakness, declared in advance: the bucket coding was done **from transcripts only, no
frames**, so an item may be miscoded; and the `S_FN` gold is the most contestable cell (§9.2 lists
`S_FN` members that read as news reports yet carry a hate label). `S_FP` (n = 30) is therefore the
cleanest sub-measurement and is reported separately.

---

## 6. Judgement rule — FROZEN, computed after the run, never adjusted

Let `s(x)` be the model's `stance` field. Non-endorsing set `N = {quotes_mentions, condemns, reports}`.

**P1 — stance accuracy on the S bucket.**
`acc_S = ( #{x ∈ S_FP : s(x) ∈ N} + #{x ∈ S_FN : s(x) = endorses} ) / |S|`
**PASS iff `acc_S ≥ 0.70`.**

**P2 — directional damage on the hate controls.**
`false_distancing = #{x ∈ CTRL_HATE : s(x) ∈ {quotes_mentions, condemns}} / |CTRL_HATE|`
**PASS iff `false_distancing ≤ 0.15`.**

**P3 — net flip projection > 0.** Freeze the naive stance-conditional decision rule:
`s ∈ N` or `s = no_hate_content` ⇒ push the item towards **non-hate**; `s = endorses` ⇒ push
towards **hate**. Then

- `gains` = number of the 49 S errors whose push direction corrects the error
  (`S_FP` pushed non-hate, `S_FN` pushed hate) — a complete enumeration, no extrapolation;
- `loss_rate_hate` = fraction of `CTRL_HATE` pushed non-hate, `loss_rate_nonhate` = fraction of
  `CTRL_NONHATE` pushed hate;
- the four test sets hold **926 items, 108 errors, 818 correct**. Projected population damage
  `= loss_rate_hate · N_correct_hate + loss_rate_nonhate · N_correct_nonhate`, using the true
  per-dataset correct-item class counts.
- **`net = gains − projected_damage`. PASS iff `net > 0`.**

P3 is deliberately the *hard-flip upper bound on damage*: a real head would learn a soft weight
and only move borderline items, so P3 failing while P1/P2 pass is reported as "the channel is
informative but must be consumed as a weighted feature, not a rule". P3 failing **together with**
P1 is an unambiguous kill.

**Overall verdict: PASS requires P1 ∧ P2 ∧ P3.** Any other combination is FAIL and triggers §7.

### 6.1 Deviation D1 — the judged population is the **frame-bearing subset (n = 72)**, not all 99

**Ruled by the user, 2026-08-12. Recorded BEFORE any batch result was seen** — at the time of
writing the 99-item input file was still uploading to DashScope, no output file existed, and the
only model outputs in existence were the 8 smoke items and the 4 synthetic probes.

**Reason (user's):** ImpliHateVid has no local raw video, so its 27 items were being sent to a
*multimodal* capability gate as transcript-only. Silently mixing a degraded-input arm into the
primary sample defeats the purpose of the measurement. Standing rule from now on: in a multimodal
pilot, items missing an input modality are excluded or reported separately — never silently
downgraded into the main sample.

**Consequence for scoring** (the three thresholds themselves are unchanged):

| view | population | status |
|---|---|---|
| **A — primary** | frame-bearing subset: **72 items** = 36 S errors (HateMM 8 + MHC-EN 16 + MHC-ZH 12) + 36 controls (18 hate / 18 non-hate) | **the verdict**; §6 P1/P2/P3 applied here |
| **B — as-frozen** | all **99** items, including the 27 transcript-only ImpliHateVid items | reference only, reported in full |
| **C — text-only** | the **27** ImpliHateVid items alone (13 S errors + 14 controls) | reported as a descriptive answer to "is transcript alone enough to type stance?"; does **not** enter the verdict |

The P3 population projection in view A is restricted to the three frame-bearing test splits
(HateMM 215 + MHC-EN 161 + MHC-ZH 149 = 525 items, 81 errors, 444 correct); view B keeps the
full four-split projection.

---

## 7. Pre-registered fallback ladder (at most ONE step, then KILL)

1. **More evidence:** re-run the same eval sample with **OCR text added** and **16 frames**
   instead of 8 (*user correction 2026-08-11: frames and OCR are the increment; transcript is
   already complete in round 1*).
2. **Task decomposition:** two-stage — first ask only `hate_surface_present`, then, conditioned on
   it, ask the stance question alone.

Only one of the two may be taken. If the chosen step still FAILs the §6 rule, the verdict is
**KILL** and the direction is closed with a written death certificate.

---

## 8. Model selection and cost accounting

Two tiers are run on the **same 99 items**:
- **strong tier** — the strongest VL model the account exposes;
- **cheap tier** — a flash-class VL model, to measure the cost/quality curve.

The strong tier runs first. **If the strong tier FAILs and the one permitted fallback step also
FAILs, the cheap tier is not run** — the verdict is KILL and the money is saved.

The paid runs go through the **DashScope Batch API** (OpenAI-compatible `/v1/batches`, ~50 %
discount, 24 h window) — *user instruction 2026-08-11*. The smoke run uses the realtime endpoint.
Token counts are recorded per item from the API `usage` field and extrapolated to (a) the four
datasets in full (4,671 videos) and (b) a 25 %-gated regime.

Budget cap for the whole pilot: **< USD 5 equivalent**.

---

## 9. Process rules in force

- Judgement rule (§6) frozen **before** any eval-sample call. It is not adjusted afterwards.
- **Prompt iteration during smoke is permitted** (*user ruling 2026-08-11*, overriding an earlier
  restriction), including iterating on whether the smoke items' stance was called correctly. The
  compensating controls are: (i) the smoke items are **disjoint** from the 99 eval items, so no
  eval item is ever seen during prompt development; (ii) every prompt revision is logged in
  Appendix A with its reason; (iii) the §6 thresholds are untouched.
- Single submission for the paid run per tier; re-submission only on infrastructure failure, and
  such an event is logged.
- Honest reporting: if the numbers are bad they are reported as they are.

---

## 10. Execution deviations (all logged when they happened)

| # | deviation | when | reason |
|---|---|---|---|
| D1 | primary judged population = frame-bearing 72, not all 99 | 2026-08-12, before any batch output existed | user ruling; see §6.1 |
| D2 | Batch API run with the moving alias `qwen3-vl-plus`, not the pinned snapshot `qwen3-vl-plus-2025-12-19` | at submission | the pinned snapshot is rejected by the Batch API (`model_not_found`); probed and confirmed for `qvq-max`, `qwen-vl-max-latest`, `qwen3-vl-flash-2026-01-22` too. Alias vs pinned agreed 8/8 on the smoke items; run date recorded instead of a pin |
| D3 | first batch submission failed and was re-uploaded | before any output | attempt 1 used the pinned snapshot (D2); also the SDK's 600 s default timeout silently retried a 17 MB upload on a ~30 KB/s link — client timeout raised to 3600 s. Infrastructure only, no input/prompt/rule change |
| D4 | fallback ② run on the **realtime** endpoint with 8 parallel workers instead of the Batch API | after round 1 failed | each batch submission costs ~25 min of upload for 17 MB of base64 frames and the fallback needs two passes; at full realtime price the whole fallback is ≈ ¥0.55, so the 50 % batch discount was not worth ~1 h of wall clock |
| D5 | fallback ②'s stage-B prompt drops V1.3's "no distancing ⇒ endorses" DEFAULT RULE | at fallback design, before running it | that line is a Q1→Q3 coupling; under decomposition the Q1 answer is already given, so the rule would be the only remaining instruction and would mechanically force the very collapse the fallback exists to test |
| D6 | 1 of 99 items (`MHC_zh::BV1m8411z7mV`) has no prediction in round 1 and a stage-A error in the fallback | at run time | DashScope input moderation rejected the frames (`DataInspectionFailed`). Denominators are 71 (round 1) / 72 (fallback) in view A |

## Appendix A — prompt revision log

All iteration below happened on the **8 smoke items only** (all outside the eval sample) plus 4
hand-written synthetic transcripts that belong to no dataset. No eval item was ever sent during
prompt development.

| version | change | reason |
|---|---|---|
| V1.0 | initial 5-question typed schema | frozen design |
| V1.1 | removed "lyric or song performance" from the `quotes_mentions` definition; added to `endorses`: hateful lyrics performed or used as the video's own soundtrack, and hateful material posted as entertainment without critical framing; added a DEFAULT RULE (no distancing ⇒ `endorses`) | V1.0 called `hate_video_54` (a HateMM hate video whose slurs are in the soundtrack) `quotes_mentions`. Performing hateful lyrics in your own video is use, not mention — a definitional error in V1.0, not a model error |
| V1.2 | tightened Q1 to **identity-based** attack (explicit protected-attribute list, explicit FALSE list for profanity / personal insults / political criticism / sexual content / violence); softened the DEFAULT RULE to require the material be the video's own content; added a CALIBRATION line stating that over-use of `endorses` is equally an error and that Q1 false forces Q3 `no_hate_content` | V1.1 was degenerate: it answered `endorses` on **7 of 8** smoke items, including a non-hate control and an X-bucket error whose transcript contains only profanity aimed at the CIA. Q1 was firing on any negativity and the DEFAULT RULE then forced `endorses` |
| V1.3 | added: gendered insults and gender-stereotype epithets aimed at a person do count as identity-based | V1.2 over-corrected and called the MHC-ZH hate control `BV1Dm4y1J7Pj` (title contains 贱人) `no_hate_content`. MultiHateClip's positive class is {Hateful, Offensive} and its ZH portion is largely gender-stereotype material, so a purely "protected group" reading of Q1 misses the corpus by construction |

**Label-space reachability check (synthetic, zero dataset contact).** Four hand-written
transcripts were run through V1.3 to confirm the model can reach every stance class and is not
collapsing onto one:

| synthetic transcript | emitted `hate_surface_present` / `stance` |
|---|---|
| creator denouncing an "immigrants are cockroaches" clip | true / **condemns** |
| neutral news item about a man charged for slurs | true / **reports** |
| creator replaying a misogynistic quote and taking no side | true / **quotes_mentions** |
| sourdough baking tutorial | false / **no_hate_content** |

V1.3 is the final prompt; it is what the paid batches use. **Nothing in §6 was changed at any
point during this iteration.**
