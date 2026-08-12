# DESC_CHANNEL — results

**Verdict: KILL.** The headline arm (measurement-gated targeted repair) loses **−0.0371** test
macro-F1 against the baseline, on 3/3 seeds. Every arm that adds a third stream loses, the
shuffled-description controls lose *less* than the real descriptions, and the pure-noise control
loses most. The descriptions themselves are good; the place we tried to spend them is not.

Protocol, arms, seeds, defect rule and decision rule were frozen in
`idea-stage/DESC_CHANNEL_FREEZE.md` (committed `a4dbd73`) **before** any description was
generated and before any candidate metric existed. Single submission: 21 runs in one background
process, no re-run, no tuning after seeing numbers.

- Freeze: `idea-stage/DESC_CHANNEL_FREEZE.md`
- Generation: `idea-stage/desc_channel/gen_desc.py`, prompt `prompts.py::V2`,
  log `idea-stage/desc_channel/PROMPT_LOG.md`, output `descriptions_hatemm.jsonl` (1066 rows)
- Encoding: `idea-stage/desc_channel/build_desc_feats.py` → `feats/{split}_{ARM}.pt`
- Runner: `idea-stage/desc_channel/run_arms.sh`; logs `logging/runs/desc_channel/`
  (`run.log` generation, `arms.log` training, `logs/<ARM>_s<SEED>.trainlog`)
- Analysis (frozen rule): `idea-stage/desc_channel/analyze_arms.py` → `results.json`
- Wall clock: generation 14 min 20 s, training 3 min 44 s for all 21 runs.

---

## 1. Step 1 — the descriptions

### 1.1 Coverage and failure accounting

| | n | share |
|---|---|---|
| videos with a parsed six-field description | **1034** | 96.999 % |
| refused by DashScope input moderation (`DataInspectionFailed`) | **29** | 2.7 % |
| response still truncated after the one permitted regeneration | **3** | 0.3 % |
| **no description at all** | **32** | 3.0 % |
| — of which in the DEFECT set (the ones the method needs) | **3** | 1.4 % of 218 |
| — of which in the DEFECT ∩ test set | **2** | 3.8 % of 52 |

So the moderation losses land almost entirely on videos the gate does not select:
**215 / 218 DEFECT videos and 50 / 52 DEFECT test videos got a description.** The 32 videos
without one receive the encoder's empty-string vector, exactly as frozen.

Description length: 239 / 1233 / 5565 characters (min / median / max over the six fields joined).

### 1.2 Forbidden-word rule

- Round 1: **4 / 972** parsed rows contained a banned term outside `on_screen_text`.
- The frozen single regeneration pass covered those 4 plus the 65 truncated rows (69 total).
- After regeneration, **6 rows** still violated; the 6 offending fields were blanked and counted
  (`hate_video_105`, `hate_video_19`, `hate_video_297`, `hate_video_378`, `hate_video_52`,
  `hate_video_59` — five `production_format`, one `scene`, one `actions`).
- Net: **6 blanked fields out of 6204** (0.10 %). The judgement ban held.

### 1.3 Human check of the smoke set (8 videos, all read in full)

Recorded in `idea-stage/desc_channel/PROMPT_LOG.md`. Both prompt versions parsed 8/8 with zero
violations. The perceptual quality is genuinely good and matches
`STANCE_PILOT_RESULT.md`'s finding that this model's *perception* is sound even though its
*judgement* is not: it typed a 1950s-style church recording as archival footage with the visual
evidence (4:3, motion blur, no modern production elements), typed a pixel-art shooter as
video-game capture, and — on `hate_video_100`, whose OCR cache is garbage (`"S \nS \nS …"`) —
read "THE COMPLETE | JOHNNY REBEL" off the frames rather than parroting the OCR input.

### 1.4 Two empty-transcript videos, verbatim

**`hate_video_57`** (upstream `Title` and `Transcript` both empty; label 1):

> **scene**: The video consists primarily of black screens and static white-background text
> cards; one frame shows a close-up indoor shot of a man against a plain wall with soft
> lighting. No outdoor or dynamic environments appear.
> **people**: One man appears in a single frame: light-skinned, short dark hair, wearing a
> light-colored collared shirt, resting his chin on his hand, looking toward the camera.
> **actions**: The sequence begins and ends with black screens. A still image of a man appears
> briefly. The rest of the frames display static blocks of text that change from frame to frame,
> with no motion, interaction, or object manipulation observed.
> **on_screen_text**: THIS VIDEO IS MADE WITH GGOPLAY … | UK: Muslim admits to plotting jihad
> massacres at Madame Tussauds, St Paul's Cathedral, & London's busy Oxford Street AUG 11, 2018
> … BY ROBERT SPENCER … | Yet another convert to Islam gets the idea that his new religion
> requires him to commit treason and mass murder. No authorities anywhere have ever shown any
> interest in this recurring phenomenon. | "Muslim convert admits London Oxford Street terror
> plot," BBC, August 10, 2018 … *(continues for ~1900 characters of transcribed text cards)*
> **production_format**: text-card/slideshow with one embedded still photograph; appears
> digitally produced with modern aspect ratio and clean typography, no film grain or vintage
> indicators.
> **audio_visible_cues**: no visible audio cues

**`non_hate_video_129`** (upstream text empty; label 0):

> **scene**: Outdoor residential street during daytime, with trees, houses, and utility poles
> visible in the background; lighting is bright natural daylight with clear shadows …
> **people**: At least 10 individuals appear across the frames … no one appears to be addressing
> the camera directly.
> **actions**: Initially, people are walking or standing near a black pickup truck; then a group
> gathers around a silver sedan, with some individuals pulling open doors and leaning into the
> vehicle; physical contact occurs between multiple people, including pushing and grappling; one
> person lies on the ground near the sedan's front wheel in later frames …
> **on_screen_text**: none
> **production_format**: other (handheld dashcam or passenger-window recording …)
> **audio_visible_cues**: no visible audio cues

This is the strongest possible version of the premise: a video whose ASR channel is literally
empty, whose decisive evidence is burned into text cards, and whose text cards the channel
recovers essentially in full — with no judgement word anywhere outside the verbatim field.
**And it still does not help downstream.** That gap is the finding.

### 1.5 Cost (measured tokens; price band flagged)

| pass | endpoint | items | input tok | output tok |
|---|---|---|---|---|
| smoke V1 + V2 | realtime | 16 | 33,856 | 4,960 |
| main pass | realtime | 1066 | 2,225,107 | 360,704 |
| repair pass | realtime | 69 | 185,822 | 61,064 |
| **total** | | | **2,444,785** | **426,728** |

At the assumed DashScope list price (¥0.002 / 1 K in, ¥0.008 / 1 K out; realtime, no batch
discount): **≈ ¥8.30 ≈ US$1.17**, against the frozen ¥15 cap. The cancelled batch jobs
(§2, deviation D1) recorded 0 completed requests, so they are assumed to have cost nothing;
DashScope does not report billed cost and the price page could not be fetched, so substitute the
real unit price for an exact figure. The token counts are measured and exact.

---

## 2. Deviations from the freeze

**D1 — Batch API abandoned for the realtime endpoint.** The freeze specified the Batch API. All
1066 items were submitted in 6 shards at 07:11–07:12 and sat at
`in_progress, completed=0` for **2 h 04 m** (a comparable 99-item batch in the stance pilot took
~64 min end to end). At 09:15 all six batches were cancelled — each reported `completed=0`, so no
generated work was discarded — and the identical requests were re-issued through the realtime
endpoint with 12 workers, finishing in 14 min 20 s. **Model, prompt, frames, OCR input,
temperature, seed and schema are unchanged**; only the transport and the price tier differ
(realtime is 2× batch, which is why the measured cost is ¥8.30 rather than ~¥4).
Logged in `logging/runs/desc_channel/run.log` at the moment of the switch.

**D2 — regeneration pass run at `max_tokens=1600`.** 65 / 1066 first-pass responses hit the
frozen `max_tokens=700` mid-JSON and failed to parse. The freeze's single regeneration pass was
therefore run with `max_tokens=1600`; nothing else changed. 62 of the 65 recovered, 3 did not.
This is an output-length limit, not a change to the prompt, schema or model. Logged in
`run.log` at the moment of the switch.

**D3 — `--keep_epoch_ckpts True` added to every arm.** The pipeline's default checkpoint
retention keeps only the epoch selected by the *kNN* readout, but this experiment reads out the
*classifier head* (I1), whose selected epoch often differs. All 30 epochs were therefore retained
so the per-sample defect-subset readout could be computed offline. This flag changes only which
files are deleted after training; no metric is affected.

**No deviation on the four hard red lines.** Test rows never influenced training or epoch
selection; the decision rule was frozen before results existed; no candidate metric was computed
during design or implementation (validation was on synthetic random-feature caches only); the
grid was submitted once.

---

## 3. Main table — test (215 videos), classifier head, val-selected epoch

Selection rule, unchanged and imported verbatim from `scripts/rgcl_ablation_analyze.py`:
best epoch ≥ warmup(5) by (dev head acc, dev head roc); report test macro-F1 at that epoch.

| arm | what the third stream carries | test macro-F1 (mean ± std) | per-seed | test ROC | val macro-F1 |
|---|---|---|---|---|---|
| **A0** | *(no third stream — baseline)* | **0.8774 ± 0.0041** | 0.8817 0.8771 0.8735 | 0.9248 ± 0.0035 | 0.8468 ± 0.0013 |
| **T** | transcript embedding, all videos | 0.8448 ± 0.0064 | 0.8501 0.8376 0.8466 | 0.9017 ± 0.0013 | 0.8156 ± 0.0064 |
| **B** | description, all videos *(接法①, control)* | 0.8504 ± 0.0080 | 0.8483 0.8437 0.8592 | 0.9083 ± 0.0006 | 0.8571 ± 0.0005 |
| **G** | description if DEFECT else transcript *(接法②, headline)* | 0.8403 ± 0.0119 | 0.8391 0.8290 0.8528 | 0.9044 ± 0.0158 | 0.8136 ± 0.0098 |
| **Bmis** | shuffled description, all videos | 0.8624 ± 0.0117 | 0.8489 0.8684 0.8699 | 0.9129 ± 0.0014 | 0.8365 ± 0.0102 |
| **Gmis** | shuffled description if DEFECT else transcript | 0.8471 ± 0.0076 | 0.8385 0.8499 0.8528 | 0.8871 ± 0.0067 | 0.8084 ± 0.0025 |
| **N** | fixed random unit vectors | 0.8236 ± 0.0166 | 0.8044 0.8324 0.8339 | 0.8662 ± 0.0020 | 0.7994 ± 0.0078 |

**Baseline reproduction check.** A0 here is **0.8774 ± 0.0041**, identical to four decimal places
to the published `LORA/HateMM/L1/I1` cell of `RGCL_ABLATION_RESULT.md` §3 (0.8774 ± 0.0041),
which was run on the previous hardware. Same frame, same numbers; the comparison table is sound.

### Paired-by-seed deltas (test macro-F1)

| comparison | mean | per-seed |
|---|---|---|
| **G − A0** (primary) | **−0.0371** | −0.0426 −0.0481 −0.0207 |
| B − A0 | −0.0270 | −0.0334 −0.0334 −0.0143 |
| T − A0 | −0.0327 | −0.0316 −0.0395 −0.0269 |
| Gmis − A0 | −0.0304 | −0.0432 −0.0272 −0.0207 |
| Bmis − A0 | −0.0150 | −0.0328 −0.0087 −0.0036 |
| N − A0 | −0.0539 | −0.0773 −0.0447 −0.0396 |
| G − T | −0.0045 | −0.0110 −0.0086 +0.0062 |
| B − T | +0.0056 | −0.0018 +0.0061 +0.0126 |
| G − B | −0.0101 | −0.0092 −0.0147 −0.0064 |

---

## 4. Defect-subset readout (frozen §7 item 4, descriptive)

Test split: 52 DEFECT videos, of which 26 have a literally empty transcript; 163 clean.
Counts are correct predictions at the same selected epoch, mean over 3 seeds.

| arm | DEFECT (/52) | empty (/26) | clean (/163) | Δ DEFECT vs A0 | Δ clean vs A0 |
|---|---|---|---|---|---|
| A0 | 46.67 | 24.00 | 143.33 | — | — |
| T | 47.00 | 25.00 | 135.67 | +0.33 | −7.67 |
| B | 45.33 | 22.00 | 139.00 | −1.33 | −4.33 |
| **G** | **47.00** | **24.67** | **135.67** | **+0.33** | **−7.67** |
| Bmis | 45.67 | 24.00 | 141.00 | −1.00 | −2.33 |
| Gmis | 46.67 | 24.00 | 137.33 | +0.00 | −6.00 |
| **N (noise)** | **48.00** | 24.00 | 131.33 | **+1.33** | −12.00 |

The gated arm gains **+0.33 of 52** on the videos it was designed to repair — and the
**pure-noise arm gains +1.33 on the same subset**. Whatever moves the defect subset is not the
description content; it is the presence of an extra stream shifting the operating point, trading
roughly 8 clean-subset errors for at most 1 defect-subset fix. On the 26 literally-empty-
transcript videos the gated arm is +0.67 and the noise arm is +0.00, both inside seed noise
(the per-seed deltas are +1/+1/0 and 0/0/0 on a base of 24/26).

---

## 5. Frozen verdict

| clause | requirement | measured | pass? |
|---|---|---|---|
| 1 | `mean(G − A0) ≥ +0.005` | **−0.0371** | ✗ |
| 2 | positive on 3/3 seeds | 0/3 positive | ✗ |
| 3 | `mean(Gmis − A0) < 0.5 × mean(G − A0)` and `< +0.005` | −0.0304 | ✓ (vacuously) |
| 4 | `mean(N − A0) < +0.005` | −0.0539 | ✓ (vacuously) |

## → **KILL**

Clauses 3 and 4 pass only because there is no gain to explain away. Reported as frozen:

- **Clause 5 (gate value, non-gating):** `mean(B − A0) = −0.0270 ≥ mean(G − A0) = −0.0371`, so
  **the gate carries no incremental value over undifferentiated captioning** — in fact
  undifferentiated captioning is 1.0 point *less* damaging than the gated version. The
  distinction this experiment was built to test does not pay, in the direction opposite to the
  hypothesis.
- **Clause 6 (third-stream artefact, non-gating):** not triggered, because `mean(G − A0)` never
  reached +0.005. But the quantity it was watching is the whole story: `G − T = −0.0045` and
  `B − T = +0.0056`, i.e. **swapping the channel's contents between real transcripts, real
  descriptions, shuffled descriptions and pure noise moves test macro-F1 by ~1 point, while
  merely opening the channel costs 1.5–5.4 points.** The channel's *contents* are close to
  irrelevant; its *existence* is what does the damage.

---

## 6. What actually happened, mechanically

Three facts line up:

1. **Adding a 768-d third stream to this head costs 1.5–5.4 macro-F1 points regardless of what
   it carries.** Ordering by damage: Bmis −0.0150 < B −0.0270 < Gmis −0.0304 < T −0.0327 <
   G −0.0371 < N −0.0539. Shuffled descriptions beat real ones; real transcripts (the most
   obviously informative content available) are worse than shuffled descriptions. That ordering
   is not consistent with any information-carrying story; it is consistent with a capacity /
   optimisation story, and the spread (0.039) is not much larger than the seed spread within
   arms (up to 0.033 within Bmis).
2. **This reproduces `A0_OCR_E2E_RESULT.md` exactly.** That experiment routed a 768-d OCR mean
   vector through the *same* `--archive_mode stream` path and measured **−0.0246** val macro-F1
   with 3/3 negative seeds. The present experiment measures −0.0270 to −0.0371 on test for
   description vectors through the same path. Two independent 768-d channels, two independent
   negative results of the same size, on the same mechanism. The mechanism, not the channel, is
   the problem.
3. **The baseline is a strong, well-fitted head on 744 training videos.** `archive_proj`
   (`Linear(768,1024)`) plus the doubled fusion-MLP input adds **+1.84 M trainable parameters,
   +36.8 %**, to a 4.99 M-parameter model trained on 744 examples. The noise arm's −0.0539 is the
   clean measurement of what that costs when the channel carries nothing.

The honest summary: **the descriptions are good, the gate is well-targeted, and neither matters,
because the fusion path available to us destroys more than any 768-d channel can add.**

---

## 7. 与现有工作的区分

The user-added constraint (2026-08-13) requires this section, and the result requires it to be
written in the past tense: this is the distinction the experiment *would have* claimed, and the
measurement that says it does not pay. Evidence base for the comparisons:
`research-wiki/MLLM_USAGE_LANDSCAPE.md` (2026-07-02, mechanisms verified against paper text and
official code).

That file draws the red line explicitly: *"凡是我们的 MLLM 角色 1 … 不能表述成 'generate
caption/description then classify'(Pro-Cap 占)、不能是 'LLM rationale as feature/distillation'
(HVGuard/RAMF/Mr.Harm 占)"*. **Arm B of this experiment *is* that occupied design**, which is why
it was demoted to a control; the claim was placed on arm G.

| axis | Pro-Cap (ACM MM 2023) | HVGuard (EMNLP 2025) | RAMF (TMLR) | MARS / LELA | Filter-And-Refine (TikTok, 2507.17204) | this work, arm G |
|---|---|---|---|---|---|---|
| what the MLLM produces | probing-question answers → caption | 3-step CoT rationale | 3 rationale views (T_O/T_H/T_N) | judgement + rationale | Yes/No moderation token | six-field perceptual description; judgement words banned and machine-scanned |
| who gets the call | every meme | every video | every video | every video / every frame × modality | every video hits the router, 2.5 % reach the MLLM | the 20.5 % of videos whose ASR channel fails a frozen input test |
| what decides the call | nothing (always-on) | nothing | nothing | nothing | embedding similarity to a hand-picked high-risk seed bank | a label-free input-quality statistic (in-vocabulary token count / non-word rate) |
| why the gate exists | — | — | — | — | **compute cost** (cut 97.5 % of traffic) | **input repair** (the measured error taxonomy says these inputs are broken) |
| what the output replaces | nothing, caption is appended | nothing, rationale is a 4th modality | nothing, rationale is a 4th modality | nothing | nothing | the transcript embedding, for gated videos only |
| what motivates it | — | — | — | — | traffic volume | `IDEA_REPORT.md` §9.2 error taxonomy + 12.1 % empty test transcripts |

Three claimed differences, and what the numbers did to each:

1. **The gate is an input-quality test, not a cost router and not a confidence threshold.**
   Filter-And-Refine is the only prior gate in this literature and it routes on similarity to a
   seed bank to save GPU time; its survivors are the *high-risk* items, not the *badly
   transcribed* ones. Ours is computed from the transcript alone before any model runs.
   **Status after measurement: the distinction is real and it costs 1.0 point** — the gated arm
   is worse than the ungated one (clause 5).
2. **The MLLM output substitutes for a failed channel instead of being appended as a new one.**
   Pro-Cap / HVGuard / RAMF all append. **Status: the substitution is worth −0.0045 relative to
   a plain transcript channel (G − T), i.e. nothing.** On the 52 videos where the substitution
   actually fires it is worth +0.33 correct predictions, which a random-noise channel beats.
3. **The design is derived from a measured error taxonomy of this system**, not from "captions
   might help". **Status: the derivation was sound — the taxonomy correctly identified videos
   whose text channel is empty, and the MLLM correctly recovered what was in them (§1.4) — and
   the downstream still lost.** The failure is between a correct diagnosis and the only
   available injection point, not in the diagnosis.

What was never claimed, and still is not: that perceptual description is a new idea, that this
is a new prompting scheme, or that this is a new fusion architecture. The only novel element
was *where the description is spent*, and that element measured negative.

---

## 8. What this kills and what it does not

**Killed.**
- "MLLM perceptual description as a downstream feature channel for HateMM", in the concatenated
  form (arm B) and in the measurement-gated targeted-repair form (arm G), on the
  `LORA/HateMM/L1/I1` backbone, through the pipeline's third-stream fusion.
- The specific novelty story in §7. With arm B beating arm G, the gate cannot be defended as the
  contribution even if some later variant of the channel worked.

**Not killed** (stated so the kill is not over-read):
- **The premise itself.** §1.4 shows the channel recovers exactly the evidence the ASR channel
  is missing, on exactly the videos the error taxonomy flagged. The measurement here says that
  evidence cannot be delivered through a 768-d third stream into a 5 M-parameter head trained on
  744 videos — not that the evidence is useless.
- **Other injection points.** The description was never allowed to enter the *text* channel
  proper, because the LoRA-Qwen adapter that produced `text_feats` no longer exists on this
  workstation (`logging/lora/` is absent — FREEZE §4). Overwriting the text embedding of a
  defective video in its own encoder's space is a different experiment and remains untested.
  Likewise the kNN memory-key path (`--archive_mode knn`) and any retrieval-side use.
- **Other backbones.** One dataset, one encoder, one loss rung, one readout. The third-stream
  penalty may be specific to a head this small on a training set this small.
- **The 29 moderation refusals** are an operational finding worth carrying: a hateful-video
  pipeline built on this vendor loses ~2.7 % of items, concentrated on `hate_video_*`. They did
  not drive this result (only 2 of the 52 DEFECT test videos were affected).

**The cheapest next probe, if this direction is revisited**: measure the third-stream penalty
directly by sweeping the archive dimensionality (e.g. PCA the same description vectors to 32/64
dims) against the same baseline. If the penalty scales with added parameters rather than with
channel content, the fusion path is the thing to fix, and no amount of better description will
move it. That is a CPU-hour experiment and it is the only version of this direction that is not
already answered by the table in §3.
