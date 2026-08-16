# SYNTH_PAIR_PROBE -- pre-registration / freeze

Written 2026-08-17, before any classifier was trained and before any evaluation number was
computed. Zero API cost. Local only (CPU embeddings + one short RTX 5090 fine-tune).

## 0. Question

Can a small text classifier trained **only on rule-synthesised quoting / self-utterance
pairs** transfer to **real ASR transcripts** and separate `OWN` (the uploader is the source
of the charged material) from `NOT_OWN` (the charged material is quoted / archival /
third-party)?

Motivation: zero-shot MLLM source-and-stance judgement is closed on every route tried
(`CONTRAST_STANCE_RESULT.md`, `PERCEPT_STANCE_RESULT.md`). This probe tests a route that
does not involve an MLLM at all. Method precedent: NAACL 2024 `2404.01651` builds use /
mention pairs from CONAN by rule.

Two risks this probe is designed to measure, both named in advance:

* **R1 wrong-sign risk.** `2404.01651` Table 7: counter-speech *containing quotation* is
  false-positived **more** often (28.6% vs 7.2%), i.e. the quote marker can push the wrong
  way.
* **R2 transfer risk.** Real ASR transcripts have no punctuation, no quotation marks, and
  attribution is frequently implicit. A classifier trained on written-form frames may key
  on surface features that do not survive.

## 1. Source material (NO test contact)

CONAN is **not present on this machine** (exhaustive filesystem search 2026-08-17 returned
nothing; the claim in `MM_STANCE_LIT_RECON.md` §7 that "we already hold the CONAN family"
is not backed by any file on disk). The pre-registered fallback is therefore used:

Source sentences are drawn from **`data/gt/<ds>/train.jsonl` rows with `label == 1`**, for
`ds in {HateMM, MHC, MHC_zh, ImpliHateVid}` (298 + 168 + 180 + 649 = 1295 hateful training
videos). **No `test.jsonl` and no `val.jsonl` row is ever read by the pair builder.**

Extraction rule (frozen):

1. `strip_html` (drop `<em class="keyword">` etc.), drop the `🎼` music marker.
2. Language of the row: CJK-character ratio `> 0.15` -> `zh`, else `en`.
3. Sentence split: `en` on `[.!?;\n]`; `zh` on `[。！？；!?\n]`.
4. Keep `en` sentences of 5..25 whitespace tokens; keep `zh` sentences of 8..40 characters.
5. Deduplicate on the normalised form (§3). Shuffle with seed 0.
6. Cap at 2600 source sentences (`en` and `zh` proportional to what survives), so that the
   pair count is `>= 2000` as required. If fewer than 2000 survive the probe HALTS and
   reports the shortfall rather than lowering the bar.

Each surviving sentence `s` yields exactly **one pair**: one `OWN` example and one
`NOT_OWN` example built from the same `s`. Within a pair the *only* difference is the
frame, so nothing about the topic or the slur can be the discriminative feature.

## 2. Frame templates (frozen, `{s}` = the source sentence)

### 2a. EN `OWN` (self-utterance, no attribution) -- 12

1. `so anyway {s} and that's just how it is`
2. `look guys {s} i'm telling you right now`
3. `here's what i think {s}`
4. `welcome back to the channel today {s}`
5. `{s} that's my honest opinion on this`
6. `let me be clear about something {s}`
7. `i've been saying this for years {s}`
8. `okay so listen {s} alright moving on`
9. `and honestly {s} i stand by that`
10. `{s} that's coming straight from me`
11. `i want to say something to you {s}`
12. `you know what {s} and i mean every word`

### 2b. EN `NOT_OWN` (attribution / criticism / news frame) -- 14

1. `he said {s} end of quote`
2. `she literally said {s} can you believe that`
3. `the guy in this clip says {s}`
4. `reports say that {s}`
5. `here's the clip where he says {s}`
6. `someone commented {s} which is disgusting`
7. `this is what they posted {s} and it's vile`
8. `the video shows him saying {s}`
9. `according to the article {s}`
10. `one user wrote {s} and got destroyed in the replies`
11. `let me read you what he wrote {s} unbelievable`
12. `in the footage the speaker states {s}`
13. `{s} that's a direct quote from his speech`
14. `he was recorded saying {s} and then he doubled down`

### 2c. ZH `OWN` -- 10

1. `那我今天就说一句 {s} 就这样`
2. `大家好 今天我想说 {s}`
3. `我跟你们讲 {s} 我是认真的`
4. `{s} 这就是我的看法`
5. `说实话 {s} 我就这么想的`
6. `我一直觉得 {s}`
7. `咱们继续说 {s} 好吧`
8. `{s} 我把话放在这里`
9. `先讲一件事 {s}`
10. `反正我的意思就是 {s}`

### 2d. ZH `NOT_OWN` -- 10

1. `他说 {s} 原话就是这样`
2. `她当时说 {s} 你敢信吗`
3. `视频里这个人说 {s}`
4. `有报道称 {s}`
5. `网友评论说 {s} 真的很恶心`
6. `他发的原文是 {s} 太离谱了`
7. `根据这篇文章 {s}`
8. `画面中的发言人表示 {s}`
9. `{s} 这是他演讲里的原话`
10. `我念一下他写的 {s} 简直无语`

Template choice per example: uniform, RNG seed 0.

## 3. ASR-form normalisation (applied identically to synthetic training text and to every
evaluation transcript)

1. NFKC normalise.
2. Delete every Unicode punctuation character (categories `P*`) and every symbol category
   `S*` (this removes all quotation marks, so the attribution signal is lexical only).
3. Lowercase.
4. Collapse all whitespace runs to a single space; strip.

## 4. Classifiers -- two pre-registered tiers, no hyper-parameter search

* **Tier A (CPU-safe).** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  (already in the local HF cache) sentence embeddings ->
  `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, random_state=0)`.
* **Tier B (GPU).** `distilbert-base-multilingual-cased` +
  `AutoModelForSequenceClassification` (2 labels), `max_length=64`, `lr=2e-5`,
  `batch_size=32`, `2 epochs`, AdamW defaults, seed 0, fp32.

Positive class = `NOT_OWN`. **No hyper-parameter is tuned on anything.** If the GPU is
occupied, Tier B is skipped and this is stated in the result.

* **Length control (baseline).** A logistic regression on a single feature -- chunk word
  count -- fitted on the same synthetic training split. Reported so that a Tier A/B
  transfer number can be compared against what a pure length artefact would give.

Synthetic split: 85% of the *source sentences* (so both members of a pair land on the same
side) to train, 15% to a synthetic dev set. Synthetic dev accuracy is a **sanity check
only** and is explicitly **not** evidence of transfer.

## 5. Evaluation -- read-only, no training, no tuning

Every evaluation set below is derived from `test.jsonl`. **Explicit statement of test-set
handling:** these transcripts are read *once*, at scoring time, by an already-frozen
classifier. No test row influences pair construction, template choice, training, model
selection, thresholding, or any hyper-parameter. The decision rule in §6 was fixed before a
single evaluation number existed. This is the read-only usage permitted by the 2026-08-09
test-set protocol.

### 5.1 Primary -- GOLD_VOICE

`GOLD_VOICE` in `idea-stage/voice_field_analysis.py` (hand-coded under criterion F7 of that
file, blind to model output). It contains 49 items, which are exactly the 49 S-bucket
errors of `idea-stage/stance_pilot/sample.json`: **21 `OWN`, 16 `NOT_OWN`, 12 `UNDET`**.
The 12 `UNDET` are dropped, leaving **n = 37**.

Item text = the `text` field of `data/gt/<ds>/test.jsonl`, `strip_html`ed and normalised
per §3.

Scoring aggregation (frozen). The transcript is far longer than a training example, so it
is chunked and the chunk scores aggregated:

* `en` items: non-overlapping 40-word chunks; a trailing chunk of `< 10` words is merged
  into the previous one.
* `zh` items: non-overlapping 60-character chunks, same trailing rule at `< 15` chars.
* **PRIMARY aggregation = mean of chunk `P(NOT_OWN)`.**
* Sensitivity aggregations, reported but **not** decision-bearing and **not** promotable
  after the fact: (i) max of chunk probabilities; (ii) a single pass over the first 256
  tokens of the whole transcript.

Metrics: **ROC-AUC** of the aggregated `P(NOT_OWN)` against gold (`NOT_OWN` = 1), plus
accuracy at the fixed threshold 0.5, plus the `OWN`/`NOT_OWN` class-wise recall.

### 5.2 Secondary -- S_FP / S_FN stratification

The same 37 items, split by their stance-pilot bucket (`S_FP` = 30 of the 49, `S_FN` = 19
of the 49; after dropping `UNDET` the per-stratum n is whatever remains and is reported).
Report AUC and accuracy per stratum. Descriptive; does not gate.

### 5.3 Wrong-sign risk (R1)

Two measurements.

**(a) Grounded, on the 37 gold items.** A frozen attribution-marker lexicon (below) is
applied to the normalised transcript. Report accuracy on the subset that is
`gold == OWN AND has_marker` -- these are exactly the trap cases: a quotation/attribution
cue is present in the text but the source really is the uploader. A low accuracy here is
direct evidence of the `2404.01651` wrong-sign effect.

**(b) Descriptive, on the 50 control items** (`CTRL_HATE` 25 + `CTRL_NONHATE` 25 from
`stance_pilot/sample.json`; these have no hand-coded voice gold). Report n, mean
`P(NOT_OWN)` and rate of `P(NOT_OWN) > 0.5`, stratified by `has_marker` x
`{CTRL_HATE, CTRL_NONHATE}`. A marker-driven shift of `>= 0.15` in mean `P(NOT_OWN)`
within `CTRL_NONHATE` is recorded as "marker-driven shift present". Descriptive only;
it neither passes nor fails the probe.

Frozen marker lexicon (matched as substrings on the normalised text):
`en`: `he said`, `she said`, `they said`, `he says`, `she says`, `they say`, `quote`,
`quoted`, `according to`, `reports say`, `reported that`, `wrote`, `posted`, `tweeted`,
`commented`, `speaking to`, `told reporters`, `in his words`, `in her words`, `end quote`.
`zh`: `他说`, `她说`, `他们说`, `据报道`, `原话`, `引用`, `网友说`, `评论说`, `报道称`,
`表示`, `写道`, `发文`.

## 6. Decision rule (frozen)

On the **primary** evaluation (§5.1, GOLD_VOICE, n = 37, mean-of-chunks aggregation):

* **AUC >= 0.70** in **either** pre-registered tier (A or B) -> **SIGNAL.** The route is
  worth combining with other channels; a follow-up design is warranted.
* **0.60 <= AUC < 0.70** in the better tier -> **WEAK.** Reported to the user for a call;
  the probe itself makes no promotion.
* **AUC < 0.60** in both tiers -> **TRANSFER FAILURE.** Direction closed.

Exactly two tiers are pre-registered, so "either tier" is a two-way comparison stated in
advance; the sensitivity aggregations of §5.1 may not be substituted for the primary one.
Synthetic dev accuracy has **no** bar attached -- if it is high that only shows the
templates are learnable, which is expected and is not evidence of anything.

## 7. Artefacts

* Builder / trainer / scorer: `idea-stage/synth_pair/run_probe.py`
* Raw numbers: `idea-stage/synth_pair_probe.json`
* Result write-up: `idea-stage/SYNTH_PAIR_PROBE_RESULT.md`
* Log / pid: `logging/runs/synth_pair/run.log`, `logging/runs/synth_pair/run.pid`
* Env: conda `HateVideo`. Single execution of the evaluation.
