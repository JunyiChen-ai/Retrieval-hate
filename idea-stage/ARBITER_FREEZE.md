# ARBITER — freeze

Frozen 2026-08-13, **before** any band-set API call and before any candidate metric exists.
Everything below (design, band grid, fusion rules, final prompt, decision rule) is fixed at
this commit. The only thing not yet run when this file is committed is `mllm_judge.py run`
and `fuse.py`.

**Question.** The downstream head is untouched. On the videos where the head's output
probability is *not* confident, call an MLLM once for a binary hateful / not-hateful
judgement and fuse it with the head. Does test macro-F1 go up?

Prior art inside this project says the MLLM's *judgement* fails when used everywhere
(`STANCE_PILOT_RESULT.md`, `MASK_STANCE_PILOT_RESULT.md`, `CONTRAST_STANCE_RESULT.md`) and
that its *output as a feature channel* fails (`DESC_CHANNEL_RESULT.md`, `A0_OCR_E2E_RESULT.md`).
Neither tested a binary judgement consumed **only inside the head's uncertainty band**. The
premise being tested is that the MLLM's errors are sufficiently independent of the head's
errors *on the subset where the head is unsure*; a previously measured whole-set agreement of
0.94 / 0.43 is the reason to doubt it.

---

## 1. Baseline (already run, reproduced before this freeze)

`LORA/HateMM/L1/I1`: frozen feature cache
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt`
+ the `run_rac.py` classifier head. Command line copied verbatim from
`idea-stage/desc_channel/run_arms.sh` arm A0; runner `idea-stage/arbiter/run_a0.sh`,
group `ARBITER_20260813`, seeds 0/1/2, logs `logging/runs/arbiter/logs/A0_s{seed}.trainlog`.

Epoch selection is the frozen I1 rule imported verbatim from
`scripts/rgcl_ablation_analyze.py::parse_run`: argmax over epochs ≥ warmup(5) of
(dev head acc, dev head roc). No test quantity enters selection.

Reproduction, measured:

| seed | selected epoch | val macro-F1 | test macro-F1 |
|---|---|---|---|
| 0 | 19 | 0.8466 | 0.8817 |
| 1 | 23 | 0.8456 | 0.8771 |
| 2 | 14 | 0.8482 | 0.8735 |
| **mean ± std** | | 0.8468 ± 0.0013 | **0.8774 ± 0.0041** |

Identical to four decimals to `DESC_CHANNEL_RESULT.md` §3 arm A0 and to
`RGCL_ABLATION_RESULT.md` §3. **Baseline reproduced.**

`idea-stage/arbiter/dump_probs.py` reloads each seed's selected-epoch checkpoint and writes
per-video `sigmoid(logit)` for val (107 videos) and test (215 videos) into
`idea-stage/arbiter/head_probs.json`. Every one of the 6 dumps was cross-checked against its
trainlog macro-F1 and matched to < 5e-4. These probabilities are the *only* head quantity the
rest of the experiment uses; the head is never retrained, re-tuned or re-selected.

## 2. Uncertainty band

Band(w) = { video : |p_head − 0.5| < w }, per seed, per split, using that seed's own
probability. Frozen grid: **w ∈ {0.1, 0.2, 0.3, 0.4}**.

Band sizes, measured from `head_probs.json` (label-free, no test label read):

| w | s0 val | s0 test | s1 val | s1 test | s2 val | s2 test | union of all six |
|---|---|---|---|---|---|---|---|
| 0.1 | 4 | 7 | 5 | 5 | 4 | 6 | 20 |
| 0.2 | 7 | 11 | 7 | 9 | 8 | 16 | 37 |
| 0.3 | 14 | 19 | 14 | 16 | 55 | 99 | 154 |
| 0.4 | 20 | 36 | 17 | 30 | 59 | 110 | **170** |

The **call set is the w = 0.4 union: 170 videos**, listed in
`idea-stage/arbiter/band_union.json`. One call per video, deduplicated across seeds, splits
and w. No train-split video is called except the 8 smoke videos in §3, which never enter any
metric.

## 3. MLLM call — frozen

- Model `qwen3-vl-plus`, DashScope OpenAI-compatible endpoint, **realtime** (not Batch).
- `temperature=0.0`, `seed=20260813`, `max_tokens=200`, `enable_thinking=false`.
- Inputs: the 8 cached evenly spaced frames `data/lora_frames/HateMM/<id>/frame_*.jpg`
  (all 8 used, resized to ≤ 512 px long side, JPEG q80) + the **complete** transcript from
  `data/gt/HateMM/{train,val,test}.jsonl`, untruncated. Empty transcript →
  `(no speech transcribed)`.
- Output contract: one JSON object `{"hateful": true|false, "confidence": 0..1}`.
- Implementation: `idea-stage/arbiter/mllm_judge.py`; `judgements.jsonl` is append-only and
  keyed by id, so no video is ever called twice.
- The API key is read from `~/.dashscope_api_key` at call time and is never written into any
  repo file, log or report.

**Smoke (done, 8 TRAIN videos, one round, no wording change afterwards).**
`hate_video_{2,11,200,400}`, `non_hate_video_{5,44,300,500}`. 8/8 parsed on the first try,
zero moderation refusals, 15–16 completion tokens each, no reasoning tokens. Observed
property carried into the freeze: the model's `confidence` is saturated (0.85–1.0 on all 8),
so fusion rule (b) will behave close to rule (a). The prompt below is frozen as-is; smoke
outputs enter nothing.

**System prompt (frozen).**

```
You are a content-moderation classifier for a hateful-video research benchmark. You always reply with a single JSON object and nothing else.
```

**User prompt (frozen).** Verbatim source of truth: `idea-stage/arbiter/mllm_judge.py::PROMPT`.

```
{n} evenly spaced frames of a video are attached above, in temporal order.

The complete speech transcript of the same video is:
<<<
{transcript}
>>>

Decide whether this video is HATEFUL.

HATEFUL means the video, taken as a whole, attacks, demeans, dehumanises, threatens, ridicules or incites hostility against a person or group because of a protected characteristic - race, ethnicity, national origin, religion, caste, immigration status, sex, gender, gender identity, sexual orientation or disability. It includes slurs used as attacks, dehumanising comparisons, hateful stereotyping, calls for exclusion or violence, and celebration or endorsement of such attacks or of hate groups.

NOT HATEFUL covers everything else, including: violence, crime, accidents, insults or profanity with no protected-characteristic target; political, religious or social criticism that does not attack people for a protected characteristic; news reporting, documentary or educational coverage of hate; counter-speech, satire or commentary that quotes hateful material in order to condemn it.

Judge the video as a whole, using both what you see in the frames and what is said in the transcript. If the transcript is empty or unhelpful, judge from the frames alone.

Reply with exactly this JSON object and nothing else:
{"hateful": true or false, "confidence": a number between 0 and 1}

"confidence" is how certain you are about the label you just gave, where 0.5 means a coin flip and 1.0 means completely certain. No markdown, no code fences, no explanation.
```

**Failure handling (frozen).** DashScope input-moderation refusal (`DataInspectionFailed`)
→ recorded as `moderation_refused`, no retry, the video keeps the head's original
probability. Network/transport error → up to 3 attempts, then recorded as an error and the
head's probability is kept. A response that arrives but fails to parse → **exactly one**
identical retry, logged; if it fails again the head's probability is kept.

## 4. Fusion rules — frozen

`p_head` is the head probability. For a video in the band with a usable judgement:

```
p_mllm = confidence          if hateful
       = 1 - confidence      if not hateful

(a) hard replace   p_final = 1.0 if hateful else 0.0
(b) average        p_final = (p_head + p_mllm) / 2
(c) agree-only     agree    -> p_final = clip(0.5 + 2*(p_head - 0.5), 0, 1)
                   disagree -> p_final = 0.5
```

where *agree* means `(p_mllm >= 0.5) == (p_head >= 0.5)`. Out of band, moderation-refused or
unparsed → `p_final = p_head`.

Decision: `pred = 1 iff p_final >= 0.5`. This `>= 0.5` convention is imported unchanged from
`idea-stage/desc_channel/analyze_arms.py`. Applied to rule (c) it means: when the MLLM agrees
the head's label is kept (the extrapolation cannot cross the threshold), and when the MLLM
disagrees the video is called hateful. That is a consequence of the rule as written plus the
project's existing tie convention, stated here so it is not re-interpreted after the fact.
Rule (c) is expected to be the weak arm; it is kept in the grid because val selection, not
judgement after the fact, decides which rule is used.

Implementation: `idea-stage/arbiter/fuse.py`, verified end to end on synthetic random head
probabilities, random labels and random judgements (including missing rows and
moderation-refused rows) before this freeze. No real quantity was computed during that check.

## 5. Selection — val only

For each seed independently: evaluate all 4 w × 3 rules = 12 combinations on **that seed's
val split**, pick the combination with the highest val macro-F1. Ties break to the smaller w,
then to rule order a < b < c. The selected combination is then applied to test **once**, and
test macro-F1 is read once. Test labels are read at that step and nowhere earlier.

## 6. Decision rule — frozen

Let Δ_seed = (test macro-F1 with fusion) − (test macro-F1 of the same seed's baseline head).

> **GO** iff `mean(Δ) >= +0.005` **and** all 3 seeds have `Δ > 0`. Otherwise **KILL**.

## 7. Reported but not part of the verdict

1. Band size at each w, per seed and split.
2. In-band head accuracy vs in-band MLLM accuracy, per w.
3. In-band error overlap: both-wrong / head-wrong-only / MLLM-wrong-only / both-right, and
   the Jaccard overlap of the two error sets.
4. Number of videos refused by DashScope input moderation.
5. Measured token counts and the resulting cost.
6. A section distinguishing this design from HVGuard / RAMF, including an honest statement of
   whether the MLLM is actually more accurate than the head inside the band.

## 8. Red lines

1. **Zero test-set tuning.** Nothing in §1–§5 reads a test label; the single test read is the
   last step of §5.
2. **Decision rule frozen before results** — §6, this commit.
3. **Blindness** — no candidate metric was computed while designing or implementing.
   `fuse.py` was validated on synthetic data only.
4. **Single submission** — `mllm_judge.py run` is submitted once over the 170-video band
   union. No top-ups, no re-runs, no second grid.
5. Budget cap **¥8**. The key never leaves `~/.dashscope_api_key`.
