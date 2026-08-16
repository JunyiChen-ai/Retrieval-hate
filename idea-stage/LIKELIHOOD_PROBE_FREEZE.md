# LIKELIHOOD_PROBE — freeze

**Frozen 2026-08-17 (Pacific/Auckland), before any eval item was scored by any arm.**
Nothing below may be edited after the first `lp_<ARM>_eval.jsonl` line exists. The only
things run before this freeze were (a) a 15-comparison pipeline smoke on **3 non-hate TRAIN
videos** printing raw log-probabilities and no accuracy, and (b) a scorer self-check that
recomputed the three **already-published** baseline numbers on the 32 primary rows.

## 1. Question

Three rounds of generative prompting failed on the same 99 items
(`STANCE_PILOT_RESULT.md` 0.257 five-way, `MASK_STANCE_PILOT_RESULT.md` 0.375 five-way,
`CONTRAST_STANCE_RESULT.md` 0.469 two-way against a 0.50 chance baseline). Every round read
the model by making it **emit a token**. This round changes the read-out only:

1. **Q1 — is a readable stance signal left in the representation?** Compare
   `log P(endorsing continuation | frames + transcript)` against
   `log P(opposing continuation | frames + transcript)`. The model produces **no token**:
   one teacher-forced forward pass per continuation, log-probabilities read at the
   continuation positions only.
2. **Q2 — is the collapse caused by instruction/preference tuning?** Run the same read-out
   on a **base (pre-tuning) VL checkpoint** and on its **instruction-tuned sibling**, and
   report the difference.

Both outcomes are reportable. A failure here makes the representation-level damage claim
direct rather than inferred (`PERCEPT_STANCE` gate-0); a success gives a zero-API-cost
stance signal.

## 2. Models — and why the base partner is Qwen2-VL, not Qwen2.5-VL

`Qwen/Qwen2.5-VL-7B` **does not exist as a public repository** (HF API returns 401 for
`Qwen/Qwen2.5-VL-7B` and `Qwen/Qwen2.5-VL-7B-Base`; only `-Instruct` is published). Qwen
released no base VL checkpoint for the 2.5 generation. The nearest available **matched
base/instruct pair from one generation and one pretraining run** is Qwen2-VL-7B, where both
`Qwen/Qwen2-VL-7B` (base) and `Qwen/Qwen2-VL-7B-Instruct` are public. The tuning contrast is
therefore measured **within the Qwen2-VL generation**, and the deployed Qwen2.5-VL-7B-Instruct
is reported alongside it. This substitution is declared here, before any run.

| arm | checkpoint | continuation format | role |
|---|---|---|---|
| **A1** | `Qwen/Qwen2.5-VL-7B-Instruct` | plain text | **primary read-out** (the project's deployed encoder) |
| **A2** | `Qwen/Qwen2.5-VL-7B-Instruct` | chat template, continuation in the assistant turn | format control for A1 |
| **B1** | `Qwen/Qwen2-VL-7B-Instruct` | plain text | instruct half of the tuning contrast |
| **C1** | `Qwen/Qwen2-VL-7B` (base) | plain text | base half of the tuning contrast |

All arms: bf16, `attn_implementation="sdpa"`, single RTX 5090, `torch.no_grad`, no sampling,
no generation, `use_cache=False`. Plain-text format is used for the contrast so that the base
model — which has no chat template — is never asked for a format it was not pretrained on;
A2 exists to show whether the chat wrapper changes A1's answer.

**No system prompt is used in any arm.** The stance rounds' `SYSTEM` string instructs the
model to emit JSON, which is meaningless for a likelihood read-out and would not apply to the
base model.

## 3. Inputs — identical to the three previous rounds

- **Sample**: `idea-stage/stance_pilot/sample.json` `eval`, 99 items, seed 20260811. Unchanged.
- **Frames**: `data/lora_frames/<ds>/<id>`, 8 evenly spaced, longest side ≤ 512 px, JPEG q80
  round trip — the same sampler and the same pixels the API rounds sent
  (`stance_pilot/run_pilot.py::frame_urls`), handed to the processor as PIL images with
  `max_pixels = 360*420` (the project's own encoder convention). 72 items have frames; the 27
  ImpliHateVid items have none and are run text-only and **reported separately**.
- **Transcript**: `data/gt/<ds>/test.jsonl` `text`, HTML-stripped, **untruncated**.
- **Templates**: `idea-stage/contrast_stance/contrast_prompts.py`, revision **C1.1**,
  unmodified — 5 pairs × {v1 generic, v2 target-named} × {en, zh}. Language chosen by the
  frozen rule `run_contrast.lang_of`; v2 targets from `run_contrast.targets_map`
  (available for 64 of 99 items).

### 3.1 Prompt

Context (identical for both continuations of a pair, so any context effect cancels):

```
<8 x |vision_start||image_pad||vision_end|>VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; ...):
<<<
{transcript}
>>>

The person who made and posted this video -- the uploader/publisher, NOT anyone who is merely
filmed, quoted, sampled or replayed inside it -- pinned one comment under it that sums up
their own view.

Pinned comment:
```

Continuation: `" " + <template sentence>`, tokenized standalone with
`add_special_tokens=False` and concatenated to the context ids, so the two options share a
byte-identical context and the token span being scored is unambiguous.

### 3.2 Score

For continuation `c` with tokens `t_1..t_k`:
`mean_lp(c) = (1/k) * sum_j log P(t_j | context, t_<j)`. Reported also as `sum_lp`.

**Primary read-out (frozen):** per (item, pair), vote `ENDORSE` iff
`mean_lp(endorsing) > mean_lp(opposing)`, else `OPPOSE`. Item call = **majority of the 5
pairs**; a tie counts as wrong. This is byte-for-byte the aggregation rule the contrast round
used, so the numbers are directly comparable.

**Secondary read-outs**, all pre-registered here, none of them able to override the primary:
`sum_lp` instead of `mean_lp`; the prior-corrected reading of §5; variant v2.

## 4. Gold, views and denominators

Gold is imported unchanged: `S_FP → OPPOSE`, `S_FN → ENDORSE`, `CTRL_HATE → ENDORSE`,
`CTRL_NONHATE → excluded` (`contrast_stance/score_contrast.py::GOLD`).

| view | rows | definition |
|---|---|---|
| **A32 (primary)** | **32** | frame-bearing S rows, minus the 3 items burned as qualitative smoke checks in the contrast round, minus `MHC_zh::BV1m8411z7mV` |
| A33 | 33 | same, with `BV1m8411z7mV` restored |
| C | 13 | ImpliHateVid S rows, no frames |

`MHC_zh::BV1m8411z7mV` is excluded from the primary denominator **because DashScope refused
it in all three previous rounds**, so all three published numbers (0.500 / 0.563 / 0.469) are
32-row numbers. A local model has no such refusal, so A33 is reported too and neither is
allowed to be presented without the other. The scorer's reproduction of the three published
baselines on A32 was verified before this freeze: **0.500 (16/32), 0.563 (18/32), 0.469
(15/32)**, and 0.308 (4/13) on view C — exact matches to `CONTRAST_STANCE_RESULT.md` §8.

## 5. Template-prior control

The template pair may have a prior independent of the video. Control set: **20 non-hate TRAIN
videos with frames**, 7 HateMM / 7 MHC / 6 MHC_zh, sampled with `random.Random(20260817)`
from the sorted candidate lists (`lp_common.ctrl_items`). Same 10 sentences, same context
construction, same scoring.

Reported: per (variant, pair, language) the mean endorsing-minus-opposing margin and the share
of control videos on which the endorsing side wins. A pair whose control endorse-win-rate is
0.00 or 1.00 is carrying a pure prior; **the strength of that prior is reported whatever the
main result is.**

Pre-registered **prior-corrected secondary reading**: subtract the control mean margin of the
matching (variant, pair, language) cell from every item's margin, then re-vote and re-collapse
by the same majority rule. This is a secondary number; it cannot move the verdict.

## 6. Decision rule — frozen

On **A32, arm A1, variant v1, `mean_lp`, raw (uncorrected)**:

| accuracy | verdict |
|---|---|
| ≥ **0.70** | **SIGNAL** — the representation carries a readable stance signal |
| **0.563 – 0.699** | **WEAK** |
| < **0.563** | **FAIL** — the likelihood read-out does not beat the best previous round |

Reported alongside, not gating: exact two-sided binomial against chance 0.50, the `S_FP` and
`S_FN` cells, per-dataset and per-voice strata, the `CTRL_HATE` / `CTRL_NONHATE` endorse
rates, and the three previous rounds on the same rows.

**Tuning contrast, separate and independent:** `acc(C1) − acc(B1)` on A32, same read-out.
`≥ +0.10` ⇒ the hypothesis *"instruction/preference tuning causes the collapse"* is
**supported**; `≤ −0.10` ⇒ contradicted; in between ⇒ **not supported, no effect measured**.
`acc(C1) − acc(A1)` is reported but is cross-generation and cannot carry the claim.

## 7. Reading of each outcome, written before the numbers exist

- **FAIL + no tuning effect** — the strongest reading available: the stance information is not
  recoverable from the representation by a likelihood comparison either, and it is not the
  tuning stage that removed it. `PERCEPT_STANCE`'s inference that the representation itself is
  compromised becomes a direct measurement. The whole zero-shot MLLM stance route closes,
  including the non-generative read-out.
- **FAIL + base clearly better** — the collapse is attributable to tuning. The signal exists
  in the pretrained weights; the route reopens only through a base checkpoint or a probe
  trained on labelled data, and that would need its own gate.
- **SIGNAL / WEAK** — the damage is in the *generation policy*, not the representation, and
  there is a zero-cost stance feature to take downstream. Any downstream use needs its own
  pre-registration; nothing is built inside this experiment.

## 8. Red lines

1. Zero test-**label** tuning: labels are used only as the anchor of a disclosed capability
   measurement (user ruling 2026-08-09). No threshold, no template and no hyper-parameter is
   selected using any of them.
2. This freeze is committed to git before the first eval forward pass.
3. The formal run is a **single background submission** per arm; arms are independent files
   and are resumable by item key, but no arm is re-run after its numbers are seen.
4. Blindness: no candidate accuracy was computed during design. The pipeline smoke used 3
   TRAIN videos and printed only raw log-probabilities.
5. Cost: zero API calls, zero paid tokens. Local GPU only.

## 9. Artefacts

| artefact | path |
|---|---|
| shared input construction | `idea-stage/likelihood_probe/lp_common.py` |
| runner | `idea-stage/likelihood_probe/run_likelihood.py` |
| frozen scorer | `idea-stage/likelihood_probe/score_likelihood.py` |
| per-arm raw log-probs | `idea-stage/likelihood_probe/lp_<ARM>_{eval,ctrl}.jsonl` |
| scores | `idea-stage/likelihood_probe/score_lp.json` |
| result | `idea-stage/LIKELIHOOD_PROBE_RESULT.md` |
| logs | `logging/runs/likelihood_probe/run.{log,pid}` |
