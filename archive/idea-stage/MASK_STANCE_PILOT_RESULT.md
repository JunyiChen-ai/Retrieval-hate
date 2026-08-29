# MASK_STANCE_PILOT_RESULT — does masking the hateful content rescue the stance judgement?

**Verdict: FAIL.** Masking the identity-attacking material out of the transcript moves S-bucket
stance accuracy from **0.257 to 0.371** against a pre-registered bar of **0.70**. The lift is real
but small, not statistically distinguishable from noise (McNemar exact **p = 0.219**, 5 wins vs
1 loss on 35 items), and — the decisive finding — **it is almost entirely an accounting artefact
of asserting that hate is present, not a recovery of the use-vs-mention distinction.** On the
`S_FP` cell, the counter-speech / quotation / reportage cell the whole mechanism was designed for,
accuracy went from **1/21 to 2/21**.

Run date 2026-08-12 (Pacific/Auckland). Rules frozen in `idea-stage/MASK_STANCE_PILOT_FREEZE.md`
before any evaluation-sample API call. Raw outputs in `idea-stage/mask_stance_pilot/`.
Same 99-item sample, same gold, same model (`qwen3-vl-plus`), same 8-frame / 512 px input spec as
`STANCE_PILOT_RESULT.md`. **Nothing downstream was built; the pilot stops at these numbers.**

---

## 1. Headline

| view | n_S | **masked** | round 1 (direct) | fallback ② (decomposed) | Δ vs round 1 | bar |
|---|---|---|---|---|---|---|
| **A — frame-bearing (the verdict)** | 35 | **0.371** | 0.257 | 0.171 | **+0.114** | ≥ 0.70 |
| B — all 99 as frozen | 48 | 0.354 | 0.250 | 0.167 | +0.104 | ≥ 0.70 |
| C — text-only ImpliHateVid | 13 | 0.308 | 0.231 | 0.154 | +0.077 | ≥ 0.70 |

Baselines are recomputed on **exactly the same rows** as the masked run, so the deltas are
row-for-row paired comparisons on identical items with an identical denominator (35 in view A,
the same 35 as round 1).

| criterion | value | bar | pass |
|---|---|---|---|
| **P1** stance accuracy on S, view A | **0.371** | ≥ 0.70 | **no** |
| **P2** false distancing on `CTRL_HATE`, view A | **0.056** (1/18) | ≤ 0.15 | yes |
| **verdict (P1 ∧ P2)** | | | **FAIL** |

P3 (the previous pilot's net-flip projection, reference only this round) moves the wrong way:
**+0.75 → −83.9**.

## 2. The decisive number: the cell the mechanism was built for did not move

| cell | gold | round 1 | masked | wins | losses | McNemar exact p |
|---|---|---|---|---|---|---|
| **`S_FP`** (non-endorsing: quotes / condemns / reports) | non-endorsing | **1/21 = 0.048** | **2/21 = 0.095** | 1 | 0 | **1.000** |
| `S_FN` | `endorses` | 8/14 = 0.571 | 11/14 = 0.786 | 4 | 1 | 0.375 |
| S overall | — | 9/35 = 0.257 | 13/35 = 0.371 | 5 | 1 | 0.219 |

`S_FP` is the double-anchored cell (agent bucket code **and** dataset label = non-hate) and it is
the only cell that requires the use-vs-mention judgement. **It gained exactly one item.**
Every other gain is in `S_FN`, whose gold answer is `endorses` — the class the model was already
over-producing.

## 3. Where the +0.114 actually comes from

The MASKING NOTE does two things at once: it removes the hateful wording, **and** it tells the
model that hate is established, which withdraws `no_hate_content` as an answer. The second effect
alone accounts for most of the gain.

Restricting to the 26 S items where masking could act at all (≥ 1 placeholder):

| | accuracy |
|---|---|
| round 1, as measured | 0.308 (8/26) |
| **round 1 counterfactual**: relabel every round-1 `no_hate_content` as `endorses` — the one thing the MASKING NOTE structurally guarantees | **0.423 (11/26)** |
| masked run, as measured | 0.462 (12/26) |

So of the +0.154 gain on this stratum, **+0.115 is bookkeeping** (the escape hatch was closed and
the DEFAULT RULE then routes the item to `endorses`, which happens to be the gold for `S_FN`) and
**+0.038 — one single item — is new stance reasoning.**

The six item-level flips confirm this directly:

| | item | cell | round 1 → masked |
|---|---|---|---|
| win | `hate_video_365` | S_FN | `condemns` → `endorses` |
| win | `_qldaPBgkk0` | S_FN | `no_hate_content` → `endorses` |
| win | `BV1qZ4y1T71a` | S_FN | `no_hate_content` → `endorses` |
| win | `BV1Vy4y1p7x2` | S_FN | `no_hate_content` → `endorses` |
| **win** | **`KDcCiUU8q5E`** | **S_FP** | **`no_hate_content` → `condemns`** |
| loss | `h_wKRDyoG_c` | S_FN | `endorses` → `quotes_mentions` |

Five of the six flips move *towards* `endorses`. **`KDcCiUU8q5E` is the only item in the entire
pilot where masking produced a correct non-endorsing call on a counter-speech video.**

## 4. The internal validity check the freeze doc pre-registered

Freeze §5.2 froze the stratification by whether masking could act. It behaves exactly as a real
effect should:

| stratum | n | round 1 | masked | Δ |
|---|---|---|---|---|
| ≥ 1 masked span | 26 | 0.308 | 0.462 | +0.154 |
| **0 masked spans** | 9 | **0.111** | **0.111** | **0.000** |

The 9 items where the extractor found nothing to mask are **bit-identical in accuracy** across the
two runs. The delta is therefore attributable to the masking intervention and to nothing else —
the pipeline is not leaking a confound. This also bounds the mechanism's reach: it can only touch
**26 of 35** S items (74 %), and it masks on average 32 % of their characters.

## 5. The output distribution is still collapsed

View A, masked (round-1 count in brackets):

| group | endorses | quotes_mentions | condemns | reports | no_hate_content |
|---|---|---|---|---|---|
| `S_FP` | 13 (12) | 0 (0) | **2** (1) | 0 (0) | 6 (8) |
| `S_FN` | 11 (8) | **1** (0) | 0 (1) | 0 (0) | 2 (5) |
| `CTRL_HATE` | 16 (17) | 0 (0) | 1 (0) | 0 (0) | 1 (1) |
| `CTRL_NONHATE` | 5 (1) | 0 (0) | 1 (1) | 0 (0) | 12 (16) |
| **total** | **45** (38) | **1** (0) | **4** (3) | **0** (0) | **21** (30) |

Over the full 97 scored items: `endorses` 63, `no_hate_content` 28, `condemns` 5,
**`quotes_mentions` 1, `reports` 0**.

**Answering the question the task asked directly: `reports` was still chosen zero times, and
`quotes_mentions` was chosen once — on an item where it was wrong.** The field remains a
two-class variable. Masking did not unlock the mention classes; it only shifted mass from
`no_hate_content` to `endorses`, which is a move *within* the degenerate binary, not out of it.

## 6. The damage side, which the headline hides

The same mechanism that buys the `S_FN` gains costs the non-hate controls symmetrically:

| | round 1 | masked |
|---|---|---|
| `CTRL_HATE` correctly `endorses` (view A) | 17/18 | 16/18 |
| **`CTRL_NONHATE` falsely `endorses`** (view A) | **1/18 = 0.056** | **5/18 = 0.278** |
| P2 false distancing on `CTRL_HATE` | 0.000 | 0.056 |
| P3 net flip projection | +0.75 | **−83.9** |

Four of the five newly-false `endorses` on non-hate controls are `no_hate_content → endorses`
flips caused by the MASKING NOTE asserting hate on an item where the extractor over-extracted:
`non_hate_video_356` (14 placeholders covering 0.7 % of the characters — slurs used as
terminology in a dog-training video), `xqilG4oMvvI` (a child's joke), `UwLhLuZ3C9U` (speech by
the subject's fans), `non_hate_video_279`. **P2 passes only because the model still almost never
assigns a distancing stance to anything — it is a pass by degeneracy, exactly as in round 1.**

## 7. Per-dataset

| dataset | n_S | masked | round 1 | fallback ② |
|---|---|---|---|---|
| HateMM | 8 | 0.125 | 0.000 | 0.125 |
| MHC (EN) | 16 | 0.312 | 0.250 | 0.125 |
| MHC_zh | 11 | 0.636 | 0.455 | 0.273 |
| ImpliHateVid (view C, no frames) | 13 | 0.308 | 0.231 | 0.154 |

Every split improves and every split remains far below 0.70. HateMM stays worst: its `S_FP` items
are archival newsreel and song performances whose transcript is *entirely* hateful surface, so
masking leaves a string of placeholders and nothing to reason from — the structural limit declared
in freeze §8.3.

## 8. Extraction-step quality (step 1)

**Aggregate over the 97 scored items:** 274 spans extracted, **0 unmatched** (every span was
located in its transcript and replaced), 68 items carry ≥ 1 placeholder, mean masked character
fraction 0.251, residual verbatim leaks 0.

**Smoke hand-check (8 items, freeze §6 gate).** 17/17 spans located, 0 leaks, 5/8 items masked;
extraction judged correct on 7 of 8 and defensibly over-extended on 1 (`NH_180`, one ASR-garbled
racial phrase). Framing text survived intact in every case. The §6 retry was not consumed.

**Formal-batch 10-item audit** (`numpy.default_rng(20260812)`, selection frozen before results,
dump in `audit10_m1.txt`), hand-read:

| verdict | n | items |
|---|---|---|
| correct | 6 | `hate_video_377`, `non_hate_video_642`, `BV12B4y1x7eK`, `BV1Lz4y1T7rn` (correctly zero spans), `BV1aP4y1E7PF`, `NH_887` |
| over-extraction, defensible | 2 | `non_hate_video_149` (masked 3 non-hateful verses of the song alongside the one real slur, **and missed** the slur "Darkkey" in the opening line); `BV1vM411p7jy` (target "粉丝"/fans is not a protected identity) |
| **clear failure** | 2 | `NH_892` — "the Chinese did not build this wall" extracted as an attack on Chinese people (a historical claim, not an attack); **`KDcCiUU8q5E`** — see below |

**`KDcCiUU8q5E` is the most informative single item in this pilot.** It is a commentator
denouncing Trump's misogyny — a textbook `S_FP` counter-speech video. The extractor masked the
video's **title** ("Trump & his audience's Misogyny & Cruelty are disgusting") and the
commentator's own critique ("the way he talked about it … the amount of cruelty and misogyny …
this is why so many victims do not come forward"), i.e. **it masked the counter-speech itself**,
because it cannot distinguish *words that attack a group* from *words that describe an attack on a
group*. That is precisely the use-vs-mention distinction the two-step design was meant to route
around. The item still flipped to `condemns` on the surviving framing, but the mechanism only
survived here by luck.

**Parse robustness.** 5 of 98 extraction replies (5.1 %) were unparseable because the model
corrupted the `"text"` **key** of a span object into a bare commentary string. These were
recovered by a salvage parser (deviation D1) — without it those 5 items, two of them in the
primary metric, would have entered the masked arm completely unmasked.

## 9. Losses and cost

**Two of 99 items produced no prediction, both to vendor moderation, on two different surfaces:**

| item | stage | error |
|---|---|---|
| `ImpliHateVid::EX_329` | step 1 (text only) | `DataInspectionFailed: **Output** data may contain inappropriate content` — the model's own extracted spans tripped the filter |
| `MHC_zh::BV1m8411z7mV` | step 2 (frames) | `DataInspectionFailed: **Input image** data may contain inappropriate content` — the same item round 1 lost |

This reinforces the round-1 operational finding, and sharpens it: **an extract-then-mask pipeline
has two moderation surfaces instead of one, and the second one rejects the model's own output.**

**Cost — measured tokens:**

| run | endpoint | items | input tok | output tok |
|---|---|---|---|---|
| smoke step 1 (extraction) | realtime | 8 | 8,986 | 498 |
| smoke step 2 (masked stance) | realtime | 8 | 17,968 | 622 |
| synthetic reachability probe (5 × 2 calls) | realtime | 5 | 9,172 | 558 |
| **eval step 1 (extraction)** | **Batch** | 98 | **92,265** | **10,202** |
| **eval step 2 (masked stance)** | **Batch** | 97 | **208,803** | **6,259** |
| **total** | | | **337,194** | **18,139** |

At the assumed list price used in `STANCE_PILOT_RESULT.md` §7 (¥0.002/1K in, ¥0.008/1K out, the
Batch discount ignored as an upper bound): **≈ ¥0.82 ≈ USD 0.12**, against a USD 5 cap. The
two-step design roughly doubles per-item cost relative to a single ask (2,404 vs 2,042 input
tokens per item on step 2 alone, plus a ~940-token extraction call), which matters only if the
mechanism worked. It did not.

## 10. What this establishes

Freeze §6 pre-registered the reading of this outcome. Extraction was accurate (0 unmatched spans,
6/10 clean on audit, framing preserved) and the stance judgement still collapsed, so branch 2
applies:

> **The model's stance bias does not depend on the verbatim hateful wording being present in the
> transcript.** Removing the trigger text and leaving only the author's framing raises `S_FP`
> accuracy from 0.048 to 0.095. Whatever is driving the collapse onto `endorses` survives the
> deletion of the words.

Three specific corollaries, stated at the strength the evidence supports:

1. **The trigger hypothesis is refuted for the transcript channel.** It is *not* refuted for the
   frames: the 8 frames were never masked and still carry the video's visual hateful content, and
   round 1 already showed that frames buy almost nothing on their own (0.257 with frames vs 0.231
   text-only). Between those two facts the remaining live explanation is that the bias attaches to
   the **topic / target** — the mere fact that the video is *about* an identity group being
   attacked — rather than to any particular carrier of it. This pilot cannot separate that from a
   frame-borne trigger and does not claim to.

2. **A two-step extract-then-judge pipeline inherits the problem it was meant to avoid.** Step 1
   cannot mark attacking material without implicitly deciding what counts as an attack, and on
   counter-speech it masks the counter-speech (`KDcCiUU8q5E`, `NH_892`). The use-vs-mention
   distinction is not a preprocessing step that can be factored out ahead of the stance question;
   it *is* the stance question.

3. **Asserting `hate_surface_present` to the model is a lever, but a symmetric one.** It reliably
   converts `no_hate_content` into `endorses`. That buys `S_FN` (+3) and costs `CTRL_NONHATE`
   (−4), which is why P3 moves from +0.75 to −83.9. It is not a repair channel; it is a threshold
   shift, and the detector already has a threshold.

**`STANCE_PILOT_RESULT.md`'s KILL stands.** This pilot was the one mechanism-level rescue attempt
the direction was worth; 0.371 against 0.70, with the gain concentrated in the wrong cell and
sourced from bookkeeping rather than reasoning, does not reopen it. The routes that document
listed as untouched — stance as *metadata* rather than content inference, and fine-tuning a stance
typer — remain untouched by this measurement too, and each would still need its own gate.

## 11. Reproducibility index

| artefact | path |
|---|---|
| frozen rules, design, deviations, prompt log | `idea-stage/MASK_STANCE_PILOT_FREEZE.md` |
| prompt bank (extraction E1.0, masked stance M1.0 = V1.3 + MASKING NOTE) | `idea-stage/mask_stance_pilot/mask_prompts.py` |
| two-step runner + programmatic masker + salvage parser | `idea-stage/mask_stance_pilot/run_mask.py` |
| background driver | `idea-stage/mask_stance_pilot/drive_mask.sh` |
| frozen scorer (3 views, strata, paired baselines) | `idea-stage/mask_stance_pilot/score_mask.py` |
| reporting tables | `idea-stage/mask_stance_pilot/report_tables.py` → `report_m1.txt` |
| 10-item audit (rng 20260812) | `idea-stage/mask_stance_pilot/audit10.py` → `audit10_m1.txt` |
| step 1 raw / parsed | `batch_out_ext_m1.jsonl`, `extract_m1.jsonl`, `batch_err_ext_m1.jsonl` |
| masked transcripts + per-span match report | `masked_m1.jsonl` |
| step 2 raw / parsed / scored | `batch_out_stn_m1.jsonl`, `pred_m1.jsonl`, `score_m1.json`, `batch_err_stn_m1.jsonl` |
| smoke passes | `extract_s_e1.jsonl`, `masked_s_e1.jsonl`, `pred_s_e1.jsonl` |
| run logs | `logging/runs/mask_stance_pilot/{run.log, run_step1.log, run.pid}` |
| sample (re-used unchanged, seed 20260811) | `idea-stage/stance_pilot/sample.json` |

Data boundary as authorised: video frames were sent to the user's own DashScope account (user
ruling 2026-08-11); test-set **inputs** were used, test **labels** only as the anchor of a
disclosed capability measurement (user ruling 2026-08-09); no label tuned anything and no detector
hyper-parameter was selected here. The API key was read from `~/.dashscope_api_key` at runtime and
appears in no file in this repository.
