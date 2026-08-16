# LIKELIHOOD_PROBE — does a non-generative read-out recover the stance signal, and is the collapse caused by tuning?

**Verdict: FAIL on both questions, and the failure is sharper than the three generative
rounds.** Comparing `log P(endorsing continuation | frames + transcript)` against
`log P(opposing continuation | ...)` under local Qwen VL weights gives **0.5625 (18/32)** on
the primary 32 rows — and it gives that number by **calling every one of the 99 items
`OPPOSE`**. 0.5625 is exactly the share of `S_FP` rows in the denominator. The read-out is a
constant, so it is not a measurement of anything about the videos.

**The constant is identical in all four arms**, including the pre-instruction-tuning base
checkpoint. The frozen tuning contrast `acc(C1 base) − acc(B1 instruct)` is **0.0000** against
a ±0.10 bar: **the hypothesis that instruction/preference tuning caused the collapse is not
supported.**

Design, arms, prompt, scorer and both decision rules frozen in
`idea-stage/LIKELIHOOD_PROBE_FREEZE.md`, committed `5e27240`, before any eval forward pass.
Single background submission, 3260 comparisons, **0 errors, 0 API calls, 0 cost**, 80.5 min
wall clock on an idle RTX 5090.

---

## 1. Headline — the primary read-out is a constant

Primary: view A32, arm A1 (`Qwen2.5-VL-7B-Instruct`), variant v1, `mean_lp`, uncorrected;
per-pair vote then majority over 5 pairs — byte-for-byte the aggregation rule of the contrast
round.

| arm | checkpoint | format | A32 acc | calls made | `S_FP` | `S_FN` | verdict |
|---|---|---|---|---|---|---|---|
| **A1** *(primary)* | Qwen2.5-VL-7B-Instruct | plain | **0.5625** (18/32) | **OPPOSE ×32** | 18/18 = 1.000 | 0/14 = 0.000 | **FAIL** |
| A2 | Qwen2.5-VL-7B-Instruct | chat | 0.5625 (18/32) | OPPOSE ×32 | 1.000 | 0.000 | FAIL |
| B1 | Qwen2-VL-7B-Instruct | plain | 0.5625 (18/32) | OPPOSE ×32 | 1.000 | 0.000 | FAIL |
| **C1** | **Qwen2-VL-7B (base)** | plain | 0.5625 (18/32) | OPPOSE ×32 | 1.000 | 0.000 | FAIL |

Exact two-sided binomial against chance 0.50: **p = 0.597**. `S_FP = 1.000` and `S_FN = 0.000`
is the signature of a constant, not of discrimination — the same signature the contrast round
showed with the signs reversed (`S_FN = 1.000`, `S_FP = 0.056`).

All 99 items are called `OPPOSE`, so the non-hate and hate controls are called `OPPOSE` too:
`CTRL_HATE` endorse rate **0.056** (A1) / **0.000** (A2, B1, C1); `CTRL_NONHATE` endorse rate
0.056–0.111. Under `sum_lp` instead of `mean_lp` both control rates are exactly 0.000 and the
A32 number is unchanged at 0.5625.

### 1.1 Against the three previous rounds, same 32 rows, same binary decision

| round | mechanism | read-out | A32 acc | what it actually did |
|---|---|---|---|---|
| 1 | direct 5-way classification, binarised | generate | 0.500 (16/32) | — |
| 2 | masked classification, binarised | generate | **0.563 (18/32)** | — |
| 3 | pinned-comment forced choice | generate | 0.469 (15/32) | answered ENDORSE on 90 of 98 |
| **4 (this)** | **same templates, likelihood comparison** | **no token emitted** | **0.5625 (18/32)** | **answered OPPOSE on 99 of 99** |

Rounds 2 and 4 land on the identical number 18/32 for opposite reasons and neither is a
measurement. The scorer was verified before the freeze to reproduce all three published
baselines on these exact rows (0.500 / 0.563 / 0.469, and 0.308 on view C).

Secondary views, unchanged conclusion: **A33** (the 33rd row restored, the item DashScope
refused in all three previous rounds) 0.5758 (19/33), still all-`OPPOSE`; **view C**
(13 ImpliHateVid rows, text-only, no frames) 0.6154 (8/13), still all-`OPPOSE`. Variant v2
(target named) is worse everywhere: 0.4375 / 0.3438 / 0.4063 / 0.4063 on A32 for A1/A2/B1/C1.

---

## 2. Why it is constant: the template prior is ~4× larger than the video effect

The pre-registered control — the same 10 sentences scored against **20 non-hate TRAIN
videos** — shows that the winner of each pair is fixed by the sentences, not by the video.

**A1, variant v1, control set (20 videos):**

| pair · language | mean margin (endorse − oppose) | share of control videos where endorsing wins |
|---|---|---|
| p0 · en | −0.995 | **0.000** |
| p1 · en | −1.732 | **0.000** |
| p2 · en | +0.054 | 0.643 |
| p3 · en | +1.321 | **1.000** |
| p4 · en | −1.758 | **0.000** |
| p0 · zh | +0.447 | **1.000** |
| p1 · zh | −0.004 | 0.500 |
| p2 · zh | −0.768 | **0.000** |
| p3 · zh | −0.198 | **0.000** |
| p4 · zh | +0.389 | **1.000** |

**Seven of the ten cells are saturated at 0.000 or 1.000.** The same picture holds on the 99
eval videos: the per-cell share of items where the endorsing side wins is 0.000, 0.000, 0.453,
1.000, 0.000 (en) and 0.958, 0.083, 0.000, 0.000, 1.000 (zh). Only one cell (`p2·en`, 0.453)
is genuinely contested; every other pair is decided before the video is seen.

Variance decomposition over the 495 v1 (video × pair) margins of arm A1:

```
between-template-pair sd = 0.898        within-pair (i.e. video-driven) sd = 0.238
```

The template choice moves the margin about **3.8× more** than the video does — a ~14×
variance ratio. Three of the five English pairs sit more than 1.0 nat away from the decision
boundary, and the video moves them by 0.24 nat.

Aggregated over 5 pairs, 4 of the 5 English cells and 3 of the 5 Chinese cells point at
`OPPOSE`, so the majority vote is `OPPOSE` for every item. **The published 0.5625 is the
prevalence of `S_FP` in the denominator, nothing else.**

## 3. Removing the prior does not reveal a signal underneath

Pre-registered secondary reading: subtract the control mean margin of the matching
(variant, pair, language) cell, then re-vote. This deletes the constant and leaves only the
video-driven residual.

| arm | A32 `mean_lp` prior-corrected | `sum_lp` prior-corrected | view C `mean_lp` prior-corrected |
|---|---|---|---|
| A1 | 0.4688 (15/32) | 0.4062 (13/32) | 0.7692 (10/13) |
| A2 | 0.5625 (18/32) | 0.5000 (16/32) | 0.6154 (8/13) |
| B1 | 0.4688 (15/32) | 0.4375 (14/32) | 0.7692 (10/13) |
| C1 | 0.5938 (19/32) | 0.5938 (19/32) | 0.5385 (7/13) |

Every frame-bearing number is between 0.406 and 0.594, i.e. at or below the best previous
round; the best of them (C1, 0.594) has binomial p = 0.377 against chance. The 13-row view-C
value of 0.769 appears in A1 and B1 but not in C1 (0.538) or A2 (0.615); on n = 13 with
p = 0.092 and no consistency across arms, it is noise, and it is a secondary reading on a
secondary view either way.

**The residual is not a video-level quantity at all.** If the video moved the endorse-minus-
oppose margin in a coherent direction, an item's residual would agree across the five template
pairs. It does not:

| arm | mean pairwise correlation of the per-video residual across template pairs |
|---|---|
| A1 (Qwen2.5-VL-7B-Instruct) | **+0.032** (20 pair-pairs) |
| C1 (Qwen2-VL-7B base) | **+0.028** |

Zero. And the residual does not separate the gold cells, in either model:

| group | A1 mean residual | C1 mean residual |
|---|---|---|
| `S_FP` (gold OPPOSE) | −0.019 | −0.017 |
| `S_FN` (gold ENDORSE) | +0.002 | −0.016 |
| `CTRL_HATE` (gold ENDORSE) | −0.038 | +0.005 |
| `CTRL_NONHATE` | +0.059 | +0.027 |

The gold direction requires `S_FN > S_FP`; the observed gap is **+0.021** (A1) and **+0.001**
(C1) against a between-item sd of ~0.12. `CTRL_HATE − CTRL_NONHATE` is **−0.096**, i.e. the
**wrong sign** — hateful videos make the endorsing sentence *less* likely than non-hate videos
do. Whatever the 0.24-nat video effect is, it is not stance and it is not hatefulness.

## 4. The tuning contrast: not supported

Frozen rule: `acc(C1 base) − acc(B1 instruct)` on A32 under the primary read-out;
`≥ +0.10` supports the hypothesis, `≤ −0.10` contradicts it, in between is no measured effect.

| quantity | value | bar | reading |
|---|---|---|---|
| **acc(C1 base) − acc(B1 instruct)**, same generation, primary read-out | **0.0000** | ±0.10 | **not supported** |
| acc(C1 base) − acc(A1), cross-generation, reported only | 0.0000 | — | — |

Both are 0.0000 because both checkpoints produce the same constant. The stronger statement is
the correlation: on the 495 v1 margins,

| arm pair | Pearson r |
|---|---|
| **B1 instruct vs C1 base (same generation)** | **0.980** |
| A1 vs A2 (same weights, plain vs chat format) | 0.933 |
| A1 vs B1 | 0.885 |
| A1 vs C1 | 0.891 |

**Instruction tuning changes this quantity less than the prompt format does** (r = 0.980
across the tuning step vs r = 0.933 across plain-vs-chat on identical weights). The
preference over these two sentences is set in pretraining and survives tuning essentially
untouched.

Under the prior-corrected secondary reading the base model is +0.125 above its instruct
sibling on A32 (0.5938 vs 0.4688 — 19 vs 15 of 32, a 4-item difference at p = 0.377 vs
chance). That is **not** the frozen contrast, it is 4 items on n = 32, and it does not
reproduce on view C, where the base model is 0.231 *below* B1. It does not support the
hypothesis and is recorded only so the number is not hidden.

## 5. What this establishes

1. **The generative round's answer bias is not what stops the stance judgement.** Round 3
   answered ENDORSE on 92 % of items; this round answers OPPOSE on 100 % of them, with the
   same templates, same frames, same transcripts, same items. Removing generation removed the
   safety-shaped bias and replaced it with a lexical-probability bias pointing the other way.
   Both are constants; neither reads the video.
2. **The information is not in the representation in a form this read-out can reach.** The
   video moves the endorse-minus-oppose margin by ~0.24 nat, that movement is uncorrelated
   across template pairs (r ≈ +0.03), it does not separate `S_FN` from `S_FP` (+0.02 nat), and
   it separates the hate from the non-hate controls **in the wrong direction** (−0.10 nat).
   This is the direct measurement that `PERCEPT_STANCE`'s gate-0 result only implied.
3. **Alignment is not the culprit.** The base checkpoint, which never went through
   instruction or preference tuning, behaves identically (r = 0.980 with its instruct sibling,
   identical constant, identical 0.5625). The 2026-08-13 diagnosis that a safety prior collapses
   *"attack content is present"* into *"the author asserts it"* is not confirmed at the level of
   the weights — that diagnosis describes the generative decoding policy of the API model, not
   a property that tuning wrote into the representation.
4. **Zero-shot stance from an MLLM is closed for this corpus, generatively and
   non-generatively.** `STANCE_PILOT_RESULT.md`'s KILL now also covers the likelihood read-out
   and the base-checkpoint escape hatch. What remains untouched is unchanged: stance as
   *metadata* rather than content inference, and a probe/typer **trained on labelled data**.
   Each would need its own gate; nothing was built here.

A caveat worth stating plainly: this measures **one** read-out — a two-sentence likelihood
comparison at the output distribution. It does not rule out that a *trained* linear probe on
intermediate hidden states could separate the classes. It does rule out reading the stance for
free, and it rules out the base checkpoint being the place where the free signal was hiding.

## 6. Deviations

**D1 — the base partner is Qwen2-VL-7B, not Qwen2.5-VL-7B (declared in the freeze §2 before
running).** `Qwen/Qwen2.5-VL-7B` and `Qwen/Qwen2.5-VL-7B-Base` do not exist as public
repositories (HF API returns 401 for both; only `-Instruct` is published). Qwen shipped no
base VL checkpoint for the 2.5 generation. The tuning contrast is therefore measured on the
matched pair `Qwen/Qwen2-VL-7B` ↔ `Qwen/Qwen2-VL-7B-Instruct`, one generation and one
pretraining run, and the deployed Qwen2.5-VL-7B-Instruct is reported alongside as arm A1.
Both checkpoints were downloaded fresh (32 GB, ~9 min).

**D2 — memory-lean forward.** `lm_head` is replaced by a no-op and the final hidden state is
read through a forward hook, exactly as `text_merge/extract_text_feats.py::encode_lean` does;
the real `lm_head` is then applied by hand to the ~15–20 continuation positions. This exists
so the `seq × 151936` fp32 logits are never materialised for a 6000-token prompt. It changes
no number: the log-probabilities at the scored positions are computed from the same hidden
states with the same weights.

**D3 — no system prompt in any arm.** The stance rounds' `SYSTEM` string instructs the model
to emit a JSON object, which is meaningless for a likelihood read-out and has no counterpart
in the base model. Declared in freeze §2.

**D4 — 33 rows are available locally, 32 are scored as primary.** `MHC_zh::BV1m8411z7mV` was
refused by DashScope in all three previous rounds, so every published baseline is a 32-row
number; a local model has no such refusal. Both views are reported everywhere and neither is
presented alone. The extra row does not change the verdict (all-`OPPOSE` either way).

## 7. Cost, hardware, wall clock

**Zero API calls. Zero paid tokens.** Everything ran on the local RTX 5090.

| stage | comparisons | wall clock |
|---|---|---|
| model download (Qwen2-VL-7B + Qwen2-VL-7B-Instruct, 32 GB) | — | ~9 min |
| pipeline smoke, 3 TRAIN videos × 4 arms | 60 | ~2 min |
| A1 control + eval | 100 + 815 | 176 s + 1069 s |
| A2 control + eval | 100 + 815 | 176 s + 1069 s |
| B1 control + eval | 100 + 815 | 160 s + 976 s |
| C1 control + eval | 100 + 815 | 160 s + 976 s |
| **formal run total** | **3660** | **08:44:31 → 10:05:03 = 80.5 min** |

Throughput 33 comparisons/min = 1.8 s per comparison (2 teacher-forced forwards each, 7320
forwards in total). **The GPU was idle at launch and held by no other tenant for the whole
run** (80 MiB of Xorg at start and at finish); peak usage 26.3 GB of 32 GB, so no CPU offload
was needed and the `--cpu_offload_gib` path was never taken. **0 errors, 0 OOM, 0 items lost**
— all 815 eval comparisons parsed in every arm.

## 8. Reproducibility index

| artefact | path |
|---|---|
| freeze (design, arms, prompt, decision rules) | `idea-stage/LIKELIHOOD_PROBE_FREEZE.md` (commit `5e27240`) |
| input construction (frames, transcripts, templates, control set) | `idea-stage/likelihood_probe/lp_common.py` |
| runner | `idea-stage/likelihood_probe/run_likelihood.py` |
| frozen scorer | `idea-stage/likelihood_probe/score_likelihood.py` |
| background driver | `idea-stage/likelihood_probe/drive_lp.sh` |
| raw per-comparison log-probs | `idea-stage/likelihood_probe/lp_{A1,A2,B1,C1}_{eval,ctrl,smoke}.jsonl` |
| all scores, strata, priors | `idea-stage/likelihood_probe/score_lp.json` |
| run log / pid | `logging/runs/likelihood_probe/run.{log,pid}` |
| templates (unmodified, revision C1.1) | `idea-stage/contrast_stance/contrast_prompts.py` |
| sample (unmodified, seed 20260811) | `idea-stage/stance_pilot/sample.json` |

Data boundary: everything stayed on the workstation. Test-set **inputs** were used; test
labels served only as the anchor of a disclosed capability measurement (user ruling
2026-08-09) — no threshold, template or hyper-parameter was selected with them.
