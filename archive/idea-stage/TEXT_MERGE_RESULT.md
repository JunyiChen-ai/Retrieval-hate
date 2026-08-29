# TEXT_MERGE — results

**Verdict: KILL.** Merging the MLLM perceptual description into the transcript string on
the **text side** — no new stream, no new parameter, same encoder, same head — loses
**−0.0105** test macro-F1 against the baseline, on 3/3 seeds. Undifferentiated merging
into every video loses more (−0.0161), and mismatched descriptions lose about as much as
correct ones (−0.0122), so the content of the merged text is worth roughly 0.002 macro-F1,
i.e. nothing measurable at 3 seeds.

Protocol, arms, seeds, merge rule, defect list and decision rule were frozen in
`idea-stage/TEXT_MERGE_FREEZE.md` (committed `4874cf07`) **before** any feature was
extracted and before any candidate metric existed. The 12-run training grid was a single
background submission; no re-run, no tuning after any number was seen.

Zero API cost: this experiment called no paid API. It reuses
`idea-stage/desc_channel/descriptions_hatemm.jsonl`
(`sha256 755f9116…`, 1034/1066 rows with a parsed description), already generated and paid
for by the previous experiment.

- Freeze: `idea-stage/TEXT_MERGE_FREEZE.md`
- Arm text construction: `idea-stage/text_merge/textmerge.py`
- Encoding: `idea-stage/text_merge/extract_text_feats.py` →
  `data/CLIP_Embedding/HateMM/{split}_TEXTMERGE-{ARM}.pt`, meta `extract_meta.json`
- Runner: `idea-stage/text_merge/run_all.sh` + `run_arms.sh`;
  logs `logging/runs/text_merge/run.log`, `logs/<ARM>_s<SEED>.trainlog`
- Analysis (frozen rule): `idea-stage/text_merge/analyze_arms.py` → `results.json`
- Wall clock: 132.2 min feature extraction (2291 forwards over the 1057 videos of the
  completed run; 9 videos / 22 forwards were carried over from the aborted attempt, 2313
  forwards in total; the GPU was shared with another tenant throughout), 127 s for all 12
  head runs.

---

## 1. What was actually changed

For every video the encoder prompt is the deployed English literal

```
{TEXT_INSTRUCTION}
Title: (none)
Transcript: <THIS STRING IS THE ONLY THING THAT DIFFERS BETWEEN ARMS>
```

fed to `Qwen2.5-VL-7B-Instruct` together with the same 8 frames, pooled over the same
assistant-header span, L2-normalised to the same 3584 dimensions. The head
(`classifier_hateClipper`) is byte-identical across arms: **the four arms differ by zero
parameters.**

| arm | transcript slot | rows differing from A0 |
|---|---|---|
| **A0** | transcript verbatim (baseline) | 0 |
| **TMt** | description replaces an empty transcript / is appended after a garbled one, **for the 218 DEFECT videos only** (headline) | 215 |
| **TMall** | description appended for **every** video (undifferentiated control) | 1034 |
| **TMshuf** | a **mismatched** video's description, DEFECT videos only (mismatch control) | 215 |

Defect list (frozen, imported from the previous experiment, `sha256 ca1b6f89…`): 218 of
1066 videos, 74 with a literally empty transcript and 144 long-but-garbled; test split
52 DEFECT of which 26 empty.

---

## 2. Main table — test (215 videos), classifier head, val-selected epoch

Selection rule unchanged and imported verbatim from `scripts/rgcl_ablation_analyze.py`:
best epoch ≥ warmup(5) by (dev head acc, dev head roc); report test macro-F1 at that epoch.

| arm | test macro-F1 (mean ± std) | per-seed | test ROC | val macro-F1 | selected epochs |
|---|---|---|---|---|---|
| **A0** *(baseline)* | **0.8679 ± 0.0036** | 0.8638 0.8704 0.8694 | 0.9317 ± 0.0027 | 0.8693 | 29 29 24 |
| **TMt** *(headline)* | 0.8574 ± 0.0021 | 0.8557 0.8598 0.8567 | 0.9246 ± 0.0010 | 0.8726 | 29 29 27 |
| **TMall** | 0.8518 ± 0.0166 | 0.8580 0.8330 0.8643 | 0.9192 ± 0.0130 | 0.8762 | 18 28 14 |
| **TMshuf** | 0.8557 ± 0.0054 | 0.8562 0.8501 0.8608 | 0.9219 ± 0.0083 | 0.8749 | 17 25 19 |

**Baseline sanity check.** A0 is our own re-extraction of the base-Qwen cell on this
hardware: **0.8679 ± 0.0036** against the banked `QWEN/HateMM/L1/I1` value of
**0.8640 ± 0.0097** (`RGCL_ABLATION_RESULT.md` §3). Inside one standard deviation, so the
re-extraction reproduces the cell. It is **not** the LoRA cell's 0.8774 and is never
compared against it (freeze §2, deviation D0).

### Paired-by-seed deltas (test macro-F1)

| comparison | mean | per-seed |
|---|---|---|
| **TMt − A0** (primary) | **−0.0105** | −0.0081 −0.0106 −0.0127 |
| TMall − A0 | −0.0161 | −0.0058 −0.0374 −0.0051 |
| TMshuf − A0 | −0.0122 | −0.0076 −0.0203 −0.0086 |
| TMt − TMall | +0.0056 | −0.0023 +0.0268 −0.0076 |
| TMt − TMshuf | +0.0017 | −0.0005 +0.0097 −0.0041 |

Note the **val/test split of behaviour**: every merged arm *improves* validation macro-F1
(0.8693 → 0.8726 / 0.8762 / 0.8749) while *losing* on test. The extra text makes the 107
validation videos easier to fit and does not transfer.

---

## 3. Frozen verdict

| # | clause | requirement | measured | pass? |
|---|---|---|---|---|
| 1 | `mean(TMt − A0) ≥ +0.005` | +0.005 | **−0.0105** | ✗ |
| 2 | positive on 3/3 seeds | 3/3 | 0/3 | ✗ |
| 3 | `mean(TMshuf − A0) < 0.5 × mean(TMt − A0)` and `< +0.005` | — | −0.0122 | ✓ (vacuously) |

## → **KILL**

Clause 3 passes only because there is no gain to explain away.

**Clause 4 (reported, not gating).** `mean(TMall − A0) = −0.0161 < mean(TMt − A0) =
−0.0105`, so **the targeted gate is not worse than undifferentiated concatenation** — the
opposite of what `DESC_CHANNEL_RESULT.md` clause 5 found for the third-stream version,
where undifferentiated captioning beat the gate. The gate is worth +0.0056 mean here, but
the per-seed signs are −/+/− (one seed carries all of it), so this is a direction, not a
measured effect, and both arms are below baseline. It buys nothing that can be claimed.

---

## 4. Defect-subset readout (frozen §6 item 4, descriptive)

Test split: 52 DEFECT videos, of which 26 have a literally empty transcript; 163 clean.
Counts are correct predictions at the val-selected epoch, mean over 3 seeds. Every
recomputed per-sample test macro-F1 matched the trainlog's own value exactly (12/12).

| arm | DEFECT (/52) | empty (/26) | clean (/163) | macro-F1 on DEFECT | ROC on DEFECT |
|---|---|---|---|---|---|
| A0 | 48.33 | 25.33 | 139.33 | 0.8017 | 0.9660 |
| **TMt** | **48.00** (−0.33) | **26.00** (+0.67) | **137.33** (−2.00) | 0.7897 (−0.0120) | 0.9149 |
| TMall | **50.00** (+1.67) | 26.00 (+0.67) | 134.33 (−5.00) | 0.8857 (+0.0840) | 0.9688 |
| TMshuf | 46.33 (−2.00) | 23.67 (−1.67) | 138.67 (−0.67) | 0.7280 (−0.0737) | 0.8355 |

Three things this says:

1. **On the 26 literally-empty-transcript videos the repair does work, and it is the one
   place it works**: TMt and TMall both reach 26.00/26 against the baseline's 25.33, and
   TMshuf — the same videos given somebody else's description — drops to 23.67. That is
   the only readout in this experiment where correct descriptions separate cleanly from
   mismatched ones.
2. **The gain does not survive the 52-video defect set**, because the 26 *garbled*-
   transcript videos in it get worse under TMt. Appending a description after a garbled
   ASR string is not the same operation as replacing an empty one, and the frozen rule
   applied both under one gate.
3. **The cost lands on the clean videos**, which the intervention never touches: −2.00 of
   163 for TMt, −5.00 for TMall. Their text is unchanged and their features are unchanged;
   what changed is the decision boundary the head learns from a training set whose defect
   rows now look different. This, not the defect rows, is what makes the arm lose.

TMall is the sharpest version of the same trade: **+1.67 correct on 52 defect videos,
−5.00 on 163 clean ones.**

---

## 5. Truncation and encoding accounting

**Truncation rate: 0 %.** The production extractor passes no `max_length` to the
processor, so nothing is truncated at any length; the model context is 128 k tokens.
Full prompt lengths (vision + text tokens) over the 1065 decodable videos:

| arm | min | median | max | over 128 k context |
|---|---|---|---|---|
| A0 | 434 | 950 | 17075 | **0** |
| TMt | 629 | 1037 | 17075 | **0** |
| TMall | 629 | 1269 | 17940 | **0** |
| TMshuf | 629 | 1040 | 17075 | **0** |

Median prompt length rises from 950 tokens (A0) to 1037 (TMt) and 1269 (TMall). No arm
comes within 7× of the context limit.

Other encoding facts:

- **1 video** (`hate_video_95`, the truncated file that decord and PyAV both refuse) has
  no decodable frames and receives the production zero-vector guard **in every arm**, so
  it is identical across arms. It is the `min = 0.0` entry in the train-split drift
  statistic below.
- **Encoder fidelity.** Our re-extracted A0 text vectors against the banked
  `Qwen2.5-VL-7B-Instruct_HF` cache: cosine mean **0.99966** (test), **0.99966** (val),
  **0.99834** (train, dragged down only by the zero-vector video); per-split minimum
  0.9978 / 0.9985. The residual is GPU/CPU placement and hardware drift (deviations D1/D2).
- **2313 forwards** for the 1066 videos (2291 of them in the completed run, the rest
  carried over from the aborted attempt): frames are decoded once per video and identical
  prompts across arms are encoded once, so a DEFECT video costs 3 forwards and a clean
  video 2.

---

## 6. Deviations

**D0 — encoder cell is the base Qwen2.5-VL-7B, not the LoRA cell (declared in the freeze
before running).** The task named the `LORA/HateMM/L1/I1` cell (banked 0.8774 ± 0.0041).
Its text features were produced by merging the LoRA adapter `logging/lora/HateMM_curric`
into the base model, and **that adapter does not exist and cannot be recovered**:
`logging/lora/` is absent on this workstation, `find / -maxdepth 6 -name
adapter_config.json` returns nothing, and no B2 backup contains it
(`RGCL_video/adapters/` holds only the `lora_p9` family; `manual_backup_2026-08-06/RGCL/`
has `logging/{slurm,temporal_memory}` but no `logging/lora`). This is the same blocker
`DESC_CHANNEL_FREEZE.md` §4 recorded — it is why that experiment could not put the
description into the text channel in the first place. Substituted encoder: the frozen base
`Qwen/Qwen2.5-VL-7B-Instruct` extracted by `generate_VideoMLLM_embedding_HF.py`, the parent
script of the LoRA one, identical in prompt scaffolding, frame sampler, pooling span and
cache contract. All four arms including A0 were re-extracted in one process, so no
cross-hardware drift enters any paired comparison.

**D1 — the encoder ran split between GPU and CPU.** Another user held 20.7 GB of the
32 GB card for the whole session, leaving 11.8 GB against the model's 16.6 GB. The model
was loaded with `device_map="auto"`, `max_memory={0: "7GiB", cpu: "45GiB"}`. Same weights,
same dtype (bf16), same prompts; every arm of every video was encoded in this one process,
so the placement is common-mode across arms and cannot bias a paired comparison. Measured
cost: 7.4 s/video instead of an estimated ~2 s on an idle card.

**D2 — memory-lean encoder call, verified bitwise identical.** The production `_encode`
requests `output_hidden_states=True` and lets the model compute the fp32-upcast
`seq × 151936` logits, which is ~600 MB for a 1000-token prompt and OOM'd at video 10 of
the first attempt. `extract_text_feats.py::encode_lean` instead reads the same final
hidden state through a forward hook on the text model, with `output_hidden_states=False`
and `lm_head` replaced by a no-op — the pooled vector never depended on the logits. Checked
against the unmodified `generate_VideoMLLM_embedding_HF._encode` on 3 videos:
`max|diff| = 0.000e+00`, `torch.equal` **True** on all three. Same prompt, same processor
call, same span arithmetic, same L2 norm.

**D3 — `--keep_epoch_ckpts True` on every arm**, so the per-sample defect-subset readout
could be recomputed at the val-selected epoch (identical to `DESC_CHANNEL_RESULT.md` D3).
It changes only which files are deleted after training; no metric is affected.

**Extraction restarts.** The feature build was started three times: attempt 1 waited for
exclusive GPU memory and was abandoned after 12 minutes; attempt 2 OOM'd at video 10
(cause fixed by D2); attempt 3 completed all 1066 videos. The build is idempotent
(per-video cache files) and produces **no metric**, so this does not touch the
single-submission rule. **The 12-run training grid was launched exactly once.**

**No deviation on the four hard red lines.** Test rows never influenced training or epoch
selection; the decision rule was frozen and committed before any feature existed; no
candidate metric was computed during design or implementation (the pipeline was validated
on synthetic random-feature caches and on encoder-fidelity cosines only); the grid was
submitted once.

---

## 7. 与现有工作的区分

Evidence base: `research-wiki/MLLM_USAGE_LANDSCAPE.md` (mechanisms verified against paper
text and official code). That file's red line is explicit: our MLLM role must not be
expressible as *"generate caption/description then classify"* (Pro-Cap) or *"LLM rationale
as feature"* (HVGuard / RAMF / Mr.Harm).

**Arm TMall is exactly the occupied design.** It appends an MLLM-written description to
every item's text before a classifier consumes it — Pro-Cap's mechanism, with a perceptual
schema instead of probing questions. It is run here as a control, not as the claim.

| axis | Pro-Cap (ACM MM 2023) | HVGuard (EMNLP 2025) | RAMF (TMLR) | **TMall** (control) | **TMt** (headline) |
|---|---|---|---|---|---|
| what the MLLM produces | probing answers → caption | 3-step CoT rationale | 3 rationale views | six-field perceptual description, judgement words banned and machine-scanned | same |
| who gets the call | every meme | every video | every video | every video | the 20.5 % of videos whose ASR channel fails a frozen, label-free input test |
| where the output goes | appended to the text input | 4th modality feature | 4th modality feature | appended to the transcript | **replaces** an empty transcript / appends to a garbled one |
| what motivates it | — | — | — | — | the measured error taxonomy of this system |

The claimed distinction was **targeted repair vs undifferentiated concatenation**. What
the measurement says about it:

1. **The distinction is real and, for once, points the right way.** `TMt − TMall =
   +0.0056`: gating the description onto the 20.5 % of videos whose input is measurably
   broken is *better* than pasting it onto all of them. This reverses
   `DESC_CHANNEL_RESULT.md` clause 5, where the third-stream version of the same gate was
   1.0 point *worse* than undifferentiated captioning. Moving the injection point from a
   new stream into the text channel is what flipped it.
2. **And it is worth nothing, because both arms lose.** TMt is −0.0105 against a baseline
   that does no captioning at all, on 3/3 seeds. A distinction that only separates two
   losing designs cannot be a contribution. The per-seed signs of `TMt − TMall`
   (−0.0023 / +0.0268 / −0.0076) do not support the +0.0056 as an effect either.
3. **The description content is nearly irrelevant at the metric level.** `TMt − TMshuf =
   +0.0017` — giving a defective video *somebody else's* description costs 0.2 macro-F1
   points relative to giving it its own. The only place correct content clearly beats
   mismatched content is the 26 empty-transcript videos (26.00 vs 23.67 correct, §4), and
   that subset is too small to move the headline metric.

So: nothing here supports a claim against the Pro-Cap / HVGuard family. The honest
statement is that the differentiation *mechanism* survives the test and the *result* does
not.

---

## 8. What this kills and what it does not

**Killed.**
- "MLLM perceptual description merged into the transcript text before the encoder" for
  HateMM on the base-Qwen cell, in both the gated (TMt) and undifferentiated (TMall) forms.
- Together with `DESC_CHANNEL_RESULT.md`, **both available injection points for these
  descriptions are now measured and both are negative**: as a new 768-d third stream
  (−0.0371 on the LoRA cell) and as merged text into the encoder's own input (−0.0105 on
  the base-Qwen cell). The two numbers are on different encoder cells and must not be put
  in one table, but the direction is the same twice.

**Worth carrying forward.**
- **The text-side merge is roughly 3–4× less damaging than the third stream.** That is
  consistent with the previous experiment's diagnosis that the third-stream fusion path,
  not the description, was doing most of the harm — the harm shrinks a lot once the new
  stream is removed. It just does not shrink past zero.
- **The failure is now located on the clean videos, not the repaired ones.** TMall buys
  +1.67 of 52 defect videos and pays −5.00 of 163 clean ones; TMt buys +0.67 of 26 empty
  ones and pays −2.00 of 163 clean ones. The intervention does what it was designed to do
  on its target subset and loses elsewhere, on rows whose features it never touched. Any
  future version of this direction has to stop the training set from re-fitting its
  boundary — the defect rows are 20.5 % of training, which is enough to move it.
- **Replacing an empty transcript is a different operation from appending to a garbled
  one**, and this experiment's frozen gate mixed them. The empty-transcript subset is the
  only place the correct description beat the mismatched one. A gate that fires only on
  the 74 literally-empty videos is a different, untested, and much narrower experiment.

**Not shown.** One dataset, one encoder cell, one loss rung, one readout, 3 seeds. Nothing
here transfers to the LoRA cell without re-encoding, which is impossible until that adapter
is retrained.
