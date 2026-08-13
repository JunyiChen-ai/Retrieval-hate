# CAD — counterfactual data augmentation on the HateMM train split (FREEZE)

Frozen 2026-08-13, **before** the frozen generation run, before any feature was encoded,
before any head was trained and before any candidate metric existed. Nothing below is
revisable after this commit; any departure is written up as a numbered deviation.

## 1. Question and the one thing that changes

Every previous attempt on this grid changed the *input* to a frozen head (extra streams,
merged text, decision-level fusion) and lost. This experiment changes the **training
distribution** instead: it adds rows to the train split. The head architecture, the
encoder, the hyperparameters, the splits and the evaluation code are byte-identical to the
run that produced the A0 baseline.

An augmented row is the **counterfactual** of a training hate video: the same video, with
its transcript minimally rewritten so the identity attack is gone, labelled **0**.

Feature grid: **base Qwen2.5-VL-7B** (no LoRA; the adapter is lost). A0 baseline on this
grid = HateMM test macro-F1 **0.8679 ± 0.0036** over 3 seeds
(`idea-stage/TEXT_MERGE_RESULT.md` §2). The cache
`data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_TEXTMERGE-A0.pt` produced by that
run is reused verbatim as arm A0 here.

**Known risk, stated in advance.** The augmented row keeps the original video's image
feature and only changes the transcript. If a particular video's hate is carried by the
picture rather than by speech, that row's gold label 0 is wrong. Repo evidence says the
label is mainly text-carried (TEXTCARRIED recheck, clean margin +0.0275), which is why the
design is worth running, but it is not evidence that *every* row is text-carried. The
result document must discuss this, not bury it.

## 2. Generation (frozen)

* **Target set**: every video in `data/gt/HateMM/train.jsonl` with `label == 1`.
  Measured: **298** of 744 train rows.
* **Gate G0 (pre-API eligibility)**: a transcript with **< 40 non-whitespace characters**
  contains nothing to minimally edit and is skipped before any API call.
  Measured: **14** skipped, **284 eligible**.
* **Model**: `qwen-plus` (text-only), DashScope OpenAI-compatible endpoint,
  `temperature=0.0`, `seed=20260813`, `max_tokens=4096`. Key read from
  `~/.dashscope_api_key`; it is never written to any file, log, report or commit.
* **Direction is one-way by construction.** The prompt only ever asks for the *removal* of
  identity attacks from an already existing transcript. No prompt in this experiment asks
  for hateful content to be produced.
* **Prompt**: `idea-stage/cad/prompts_cad.py::V2`, reproduced verbatim in §7. It was
  selected by a smoke pass over 8 train videos (`cadgen.py smoke`, one iteration V1→V2 for
  two defects: the model echoed the `<<< >>>` transcript markers, and it missed
  ASR-garbled and veiled attacks). The smoke rewrites are **discarded as data**: the frozen
  run regenerates every id from scratch into a separate file. `smoke_v{1,2}.jsonl` are kept
  only as evidence of the prompt iteration and are read by no downstream script.
* **Output**: `idea-stage/cad/rewrites_train_hate.jsonl`, append-only, keyed by video id,
  idempotent. Schema `{"rewritten": str, "n_edits": int}`.
* **Moderation**: input-side refusals from the provider are recorded as
  `parse="moderation_refused"`, the row is skipped, and the count is reported.
* **Budget**: hard ceiling ¥5. Projected ≈ ¥0.4 at list price for 284 rewrites.

## 3. Quality gates (frozen; applied before any training, label-free and metric-free)

Every gate reads only the original transcript and the rewritten transcript. No gate looks
at a model, a metric, the val split or the test split.

| gate | rule | action |
|---|---|---|
| **G0** | original transcript `< 40` non-whitespace chars | skip before the API call |
| **G1** | API refused / errored / response did not parse to the schema | drop row |
| **G2** | `len(rewritten)/len(original)` (whitespace-normalised) outside **[0.40, 1.20]** | drop row |
| **G3** | rewritten text still matches a term in the frozen identity-attack word list (`gates.py::ATTACK_TERMS`, case-insensitive, word-boundary) | drop row |
| **G4** | rewritten text is identical to the original after whitespace/case normalisation | drop row |
| **G5** | the source video's 8 frames fail to decode | drop row |

G4 exists because an unchanged transcript relabelled 0 is a *known-wrong* gold label, not
a mild one. G5 exists because a zero image vector would make the augmented row meaningless.
Every drop is counted per gate and reported.

## 4. Arms and features (frozen)

The encoder path is imported verbatim from `idea-stage/text_merge/extract_text_feats.py`
(`encode_lean`, verified bit-equal to the production `_encode` by `--verify_lean`). For an
augmented row the **same 8 frames of the same original video** are decoded and the only
thing that differs from the A0 row is the string in the `Transcript: ` slot.

This grid has two feature streams, `img_feats` and `text_feats`; there is no separate audio
tensor, so "copy the visual/audio features of the source video" is implemented as copying
`img_feats`.

| arm | train rows | dev_seen / test_seen |
|---|---|---|
| **A0** (baseline) | the 744 original rows | A0, unchanged |
| **CAD** (candidate) | 744 + N augmented: `img_feats` copied from the source hate video, `text_feats = encode(original frames, REWRITTEN transcript)`, label **0**, id `<vid>__cad` | identical A0 copy |
| **CTRLRAND** (control) | 744 + N control: `img_feats` copied from the **same** source hate video, `text_feats` = the A0 text feature of a randomly drawn **distinct non-hate train video**, label **0**, id `<vid>__ctrl` | identical A0 copy |

Donor draw for CTRLRAND: `numpy.random.default_rng(20260813).permutation` over the 446
non-hate train videos, first N taken, without replacement, so no donor is reused.
CTRLRAND asks whether any gain comes from the *minimal-pair structure* or merely from
adding N more negative rows with a hate video's picture attached.

`dev_seen` and `test_seen` are byte-identical copies of the A0 cache in all three arms.
**The test split is never touched by generation, gating, donor sampling or assembly.**

## 5. Training and decision rule (frozen)

* 3 arms × seeds {0,1,2} = **9 head runs, one single background submission**, no re-run,
  no tuning after any number is seen. Command line copied verbatim from
  `idea-stage/text_merge/run_arms.sh`; only `--model` and `--exp_comment` differ between arms.
* Epoch selection imported verbatim from `scripts/rgcl_ablation_analyze.py::parse_run`
  (I1 head rung): best epoch ≥ warmup(5) by (dev head acc, dev head roc); report test
  macro-F1 at that epoch. **Selection uses dev only.**
* Let `Δ_CAD = mean_seeds(CAD − A0)` and `Δ_CTRL = mean_seeds(CTRLRAND − A0)` on test
  macro-F1.

**GO** iff all three hold:
1. `Δ_CAD ≥ +0.005`
2. `CAD − A0 > 0` on **3 of 3** seeds
3. `Δ_CTRL < 0.5 · Δ_CAD`

Otherwise **KILL**. Additionally: if `Δ_CTRL ≥ Δ_CAD` and both are positive, the result is
reported as "the gain comes from adding negative rows, not from the counterfactual
structure".

## 6. Reported regardless of verdict

Drop counts per gate (moderation refusal / G2 length / G3 word list / G4 unchanged /
G5 decode); number of augmented rows N; three rewrite examples (original excerpt +
rewritten excerpt); hate-class precision and recall on test per arm, recomputed offline
from the selected checkpoint and cross-checked against the trainlog; API spend; and a
novelty section on whether counterfactual augmentation has been done in the hateful-video
domain.

## 7. Prompt, verbatim (frozen)

SYSTEM:

```
You are a careful text editor working on an authorised content-moderation research dataset.
Your job is to produce a MINIMALLY edited version of a video transcript in which content
that attacks people for their group identity has been removed or made neutral, while
everything else in the transcript is preserved exactly. You never add any new insult,
attack or slur. You always reply with a single JSON object and nothing else.
```

USER (`V2`, `{transcript}` is the verbatim gt transcript, stripped):

```
Below is the verbatim automatic-speech-recognition (ASR) transcript of a short video. The
ASR is noisy: words are often mis-spelled, run together or garbled, and slurs frequently
appear in a corrupted spelling.

Rewrite the transcript so that it no longer attacks, demeans or dehumanises any person or
group on the basis of a protected characteristic (race, ethnicity, national origin,
religion, gender, sexual orientation, disability, immigration status, caste, or similar).

Remove or neutralise ALL of the following, including when the ASR has garbled the wording:
- slurs and epithets aimed at a group, in any spelling;
- dehumanising comparisons (calling a group animals, vermin, disease, subhuman);
- calls for violence, harm, deportation or exclusion, and demands that a group "go back"
  somewhere or leave;
- group-level accusations and stereotypes ("they are lazy / criminal / parasites",
  conspiracy claims that a group secretly controls or harms others);
- celebration or approval of harm done to a group;
- praise of movements or figures whose point is the supremacy of one group over another.

Make the SMALLEST edit that achieves this:
- Change ONLY the spans listed above. Replace a slur with a plain neutral referent
  ("people", "them", "he") or delete the span.
- KEEP everything else exactly as it is: the topic, the speaker's other opinions, the
  ordering of sentences, the sentence structure, the register, the filler words, the emoji
  and the ASR noise (repeated words, missing spaces, mis-transcriptions, broken punctuation).
- Do NOT summarise. Do NOT clean up grammar or spelling elsewhere. Do NOT translate. Do NOT
  add commentary, apology, disclaimer or counter-speech. Do NOT mention that anything was
  edited.
- Do NOT introduce any new attack on any person or group.
- Keep the length close to the original: aim for 70-100% of the original length.
- If the ONLY content of a sentence is the attack, delete that sentence rather than
  replacing it with a comment.
- If the transcript is pure noise with no attack in it, return it unchanged with
  "n_edits": 0.

Return ONE JSON object with exactly these two keys and nothing else. Do NOT copy the <<<
and >>> markers into your answer:
{"rewritten": "<the full rewritten transcript, as a single string>", "n_edits": <integer:
how many distinct spans you deleted or changed>}

TRANSCRIPT:
<<<
{transcript}
>>>
```

## 8. Code, frozen at this commit

| file | role |
|---|---|
| `idea-stage/cad/prompts_cad.py` | prompt bank; `V2` is the frozen prompt |
| `idea-stage/cad/cadgen.py` | target set, G0, generation, idempotent append-only output |
| `idea-stage/cad/gates.py` | G1–G4 + the identity-attack word list; writes `accepted.json` |
| `idea-stage/cad/build_cad_feats.py` | encoding of the rewrites (G5) + assembly of the 3 arm caches |
| `idea-stage/cad/run_arms.sh` | the 9 head runs |
| `idea-stage/cad/analyze.py` | frozen readout + the frozen decision rule of §5 |
| `idea-stage/cad/run_all.sh` | single background driver |

Run log: `logging/runs/cad/run.log`, pid `logging/runs/cad/run.pid`.

## 9. Red lines

1. Zero test-label contact: test is a byte copy of A0 in every arm; epoch selection is
   dev-only; no test number is looked at before the 9 runs finish.
2. The decision rule of §5 is fixed by this commit, before any candidate number exists.
3. Blind design: no candidate metric was computed while designing or implementing.
4. Single submission for the 9 runs.
5. The DashScope key never enters the repo, a log, or a report.
