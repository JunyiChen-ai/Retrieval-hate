# STANCE_PILOT_RESULT — can a cloud VL model type the stance of a hateful video?

**Verdict: KILL.** Both the frozen round-1 measurement and the one permitted fallback step fail
the pre-registered bar, and they fail by a very large margin in the same direction.

Run date 2026-08-12 (Pacific/Auckland). Rules frozen in `idea-stage/STANCE_PILOT_FREEZE.md`
before any evaluation-sample call. Raw outputs in `idea-stage/stance_pilot/`.

---

## 1. Headline

| | round 1 (single call, V1.3) | fallback ② (decomposed, 2 calls) | bar |
|---|---|---|---|
| **P1 stance accuracy on the S bucket** (view A, frame-bearing) | **0.257** | **0.167** | ≥ 0.70 |
| — on `S_FP` alone (the clean, double-anchored cell) | **0.048** (1/21) | **0.091** (2/22) | — |
| — on `S_FN` alone | 0.571 (8/14) | 0.286 (4/14) | — |
| **P2 false distancing on hate controls** | 0.000 | 0.056 | ≤ 0.15 |
| **P3 net flip projection** | +0.75 | −23.9 | > 0 |
| **verdict (P1 ∧ P2 ∧ P3)** | **FAIL** | **FAIL** | |

P2 passes only because the model almost never assigns a non-endorsing stance to *anything* —
it is a pass by degeneracy, not by discrimination. P3 passes in round 1 by +0.75 on a projection
whose damage term is dominated by one dataset; it flips to −23.9 under the fallback. The
decisive number is P1, and it is 0.257 against a 0.70 bar, i.e. **the pre-registered kill
condition is met twice over, and the ladder is exhausted.**

## 2. The three-view table (deviation D1, user ruling 2026-08-12)

| | **A — primary** (frame-bearing, HateMM + MHC-EN + MHC-ZH) | **B — as-frozen** (all 99, incl. 27 transcript-only) | **C — text-only** (ImpliHateVid, descriptive) |
|---|---|---|---|
| n items | 71 (r1) / 72 (fb2) | 98 (r1) / 99 (fb2) | 27 |
| n S-bucket errors | 35 / 36 | 48 / 49 | 13 |
| **P1 round 1** | **0.257** | 0.250 | 0.231 |
| **P1 fallback ②** | **0.167** | 0.163 | 0.154 |
| P2 round 1 / fb2 | 0.000 / 0.056 | 0.000 / 0.040 | 0.000 / 0.000 |
| P3 net round 1 / fb2 | +0.75 / −23.9 | −100.7 / −100.3 | −101.4 / −76.4 |
| verdict | **FAIL** (both) | FAIL (both) | FAIL (both) |

Per-dataset P1, round 1: HateMM **0.000** (0/8) · MHC-EN 0.250 · MHC-ZH 0.455 · ImpliHateVid 0.231.

**Column C answers the incidental question "is the transcript alone enough to type stance?"**
— no, but frames buy almost nothing either: 0.257 with 8 frames vs 0.231 without. The
multimodal channel is not the binding constraint; the stance judgement is.

## 3. What actually goes wrong — the mechanism, not just the score

The stance field collapses onto **{`endorses`, `no_hate_content`}**, i.e. it degenerates into a
re-statement of "is this video hateful", which is precisely the variable the detector already has.

Round-1 stance histogram, view A:

| group | endorses | quotes_mentions | condemns | reports | no_hate_content |
|---|---|---|---|---|---|
| `S_FP` (gold = non-endorsing) | **12** | 0 | 1 | 0 | 8 |
| `S_FN` (gold = endorses) | 8 | 0 | 1 | 0 | 5 |
| `CTRL_HATE` | 17 | 0 | 0 | 0 | 1 |
| `CTRL_NONHATE` | 1 | 0 | 1 | 0 | 16 |

Across all 71 items the model emitted `quotes_mentions` **0 times** and `reports` **0 times**.

Three specific findings:

**(a) It recovers *who speaks* and then ignores it.** On the round-1 `S_FP` items it assigned a
non-uploader `primary_voice` (`on_screen_speaker` / `archival_source` / `caption_overlay`) 10
times out of 21 — and in **7 of those 10** it still answered `endorses`. It correctly labels
1950s segregationist newsreel as `archival_source` and then says the video endorses it. The
use-vs-mention axis is *perceived* and not *used*.

**(b) It disagrees with the human coding about whether hate is even present.** 13 of the 35
S-bucket items got `hate_surface_present = false` — for a bucket whose definition is
"hate-associated surface with a non-endorsing stance". Examples: `KDcCiUU8q5E`
("criticizes misogyny … but does not contain identity-based slurs"), `N68vmAE5s_g`,
`j_foVftOOs4`. So part of the error is not stance at all; it is that the model's hate threshold
and the corpus's hate threshold are different objects.

**(c) It fails on exactly the items the detector fails on.** Collapsing the stance field into a
binary decision (`endorses` → hate, else non-hate) and comparing with the dataset label:

| group | round 1 | fallback ② |
|---|---|---|
| `CTRL_HATE` (detector already right) | 0.944 | 0.833 |
| `CTRL_NONHATE` (detector already right) | 0.944 | 0.944 |
| `S_FP` (detector wrong) | 0.429 | 0.524 |
| `S_FN` (detector wrong) | 0.571 | 0.286 |

The channel is ~94 % right where the detector is already right and ~43–57 % right where the
detector is wrong. **It is correlated with the detector's competence, not complementary to it.**
That is the specific property that makes it useless as a repair channel, and it is a stronger
kill than a low average score would be.

## 4. Attribution — is this the model or the prompt?

A fair concern, since the prompt was iterated on the smoke items (Appendix A of the freeze doc).
Two pieces of evidence say the collapse is not a prompt artefact:

1. **The V1.3 prompt demonstrably reaches every class.** On four hand-written synthetic
   transcripts it answered `condemns` / `reports` / `quotes_mentions` / `no_hate_content`
   correctly (freeze doc, Appendix A). The label space is live; real videos just never trigger it.
2. **Removing the coupling made it worse.** Fallback ② deletes V1.3's "no distancing ⇒ endorses"
   default rule and separates the hate-surface question from the stance question. P1 fell from
   0.257 to 0.167 and P3 went from +0.75 to −23.9. If the default rule had been the cause, this
   would have improved.

What remains genuinely uncertain, and is stated rather than hidden: the smoke items happened to
be endorses-gold, so V1.2→V1.3 tuning ran in the endorsing direction, and a prompt developed on
counter-speech examples might sit at a different operating point. That is a real limitation of
this pilot; it is *not* enough to rescue a 0.257-vs-0.70 gap.

## 5. Relation to the literature threat

`STANCE_LIT_RECON.md` §5(3) flagged `2406.00020`: when stance/in-group status must be **inferred
from content**, max F1 across all models and prompt schemes is **0.24**. This pilot measured
**0.257 / 0.167** on the same kind of inference, on video, with a current frontier VL model, on
the exact items where it would have to pay off. **The published threat reproduces almost exactly.**
The corollary in that paper — that stance *handed to the model as metadata* helps a lot — is
untouched by this result and remains the only live version of the idea (see §8).

## 6. Model selection, and what the Batch API actually accepts

Probed via `GET /v1/models` on the OpenAI-compatible endpoint. VL-capable families exposed to
this account: `qwen3-vl-plus`, `qwen3-vl-flash`, `qwen-vl-max`, `qwen-vl-ocr`, `qvq-max/plus`,
`qwen3-omni-*`, `qwen3.5-omni-*`. **There is no `qwen3-vl-max`** — `qwen3-vl-plus` is the
strongest current-generation VL model available here, and `qwen-vl-max` is the previous
generation.

**Batch API support is narrower than the model list** (probed with one-line batch files):

| model | Batch API |
|---|---|
| `qwen3-vl-plus` | ✅ accepted |
| `qwen-vl-max` | ✅ accepted |
| `qwen3-vl-flash` | ✅ accepted |
| `qwen3-vl-plus-2025-12-19` (pinned snapshot) | ❌ `model_not_found` |
| `qwen3-vl-flash-2026-01-22`, `qwen-vl-max-latest`, `qvq-max` | ❌ `model_not_found` |

So **the Batch API cannot be given a pinned snapshot**; only the moving alias. The smoke run used
the pinned `qwen3-vl-plus-2025-12-19` and the batch used the alias `qwen3-vl-plus`; they agreed on
8/8 smoke items, and the run date is recorded here in lieu of a snapshot pin.

Chosen: **strong tier = `qwen3-vl-plus`** (run 2026-08-12). The **cheap tier (`qwen3-vl-flash`)
was not run**, per the freeze §8 rule: the strong tier failed and so did its one permitted
fallback, so the money is saved. Nothing about a flash-tier model would reverse a 0.257.

**One operational finding worth keeping:** 1 of 99 items (`MHC_zh::BV1m8411z7mV`) was rejected by
DashScope input moderation — `InternalError.Algo.DataInspectionFailed: Input image data may
contain inappropriate content` — in both the batch and the realtime path. **A hateful-video
pipeline built on this vendor will lose a small fraction of exactly the items it most needs.**

## 7. Cost accounting (measured tokens; price band flagged)

Measured, this pilot:

| run | endpoint | items | input tok | output tok |
|---|---|---|---|---|
| smoke + prompt iteration + alias check (5 passes × 5–8 items) | realtime | 37 calls | 83,881 | 3,046 |
| synthetic reachability probe | realtime | 4 | ~1,500 | ~250 |
| **round 1 eval** | **Batch** | 98 | **200,124** | **7,227** |
| **fallback ② eval** (2 calls per hate-surface item) | realtime | 99 items / 171 calls | **245,703** | **6,904** |
| **total** | | | **≈ 531 K** | **≈ 17 K** |

Per-item input tokens: **2,294** frame-bearing (8 frames at max-side 512 ≈ 1,170–1,550 image
tokens + ~600–800 text) and **1,381** text-only. Output ≈ 74 tokens/item.

Extrapolation to a full front-end pass over the four datasets (4,671 videos, of which 2,662
frame-bearing per `MLLM_FRONT_RECON.md` §5):

| regime | input tok | output tok |
|---|---|---|
| full pass, all 4 datasets | **≈ 8.9 M** | ≈ 0.35 M |
| gated regime, 25 % of videos | ≈ 2.2 M | ≈ 0.09 M |

**Price caveat — read this before quoting a dollar figure.** Alibaba's pricing pages
(`help.aliyun.com/zh/model-studio/models-price`, `alibabacloud.com/.../models-pricing`, the model
list anchors) all returned 404 or price-free content to the fetch tool during this session, and
the API does not report billed cost. Using the DashScope list price **assumed** at ¥0.002 / 1 K
input and ¥0.008 / 1 K output for `qwen3-vl-plus`, with Batch at 50 %:

| | ¥ | ≈ USD @ 7.1 |
|---|---|---|
| **this pilot, everything** | **≈ ¥1.0** | **≈ $0.14** |
| hypothetical full pass (batch) | ≈ ¥10 | ≈ $1.5 |
| hypothetical full pass (realtime) | ≈ ¥21 | ≈ $2.9 |
| gated 25 % (realtime, per epoch of inference) | ≈ ¥5 | ≈ $0.7 |

Substitute the real unit price for a exact figure; the token counts above are measured and exact.
**Budget outcome: the pilot cost roughly $0.14 against a $5 cap.** The conclusion is not
cost-sensitive — generation was never the constraint, which is exactly what
`MLLM_FRONT_RECON.md` §5 predicted.

## 8. What this kills, and what it does not

**Killed.** Every design in `MLLM_FRONT_RECON.md` §4 that consumes a *model-inferred* stance type
as its conditioning variable — D1 (SCLO), D2 (TEV stance fields), D3 (SCR) — at least in the
"ask a frozen VL model for the stance of the video" form. The +6.46 macro-F1 oracle of
`IDEA_REPORT.md` §9.2 is real, but the instrument that was supposed to reach it resolves the
stance axis at ~0.17–0.26 accuracy and, worse, is only accurate on the items that did not need
fixing. There is no weighting of that channel that recovers the prize.

**Not killed by this measurement** (stated so the kill is not over-read):

1. **Stance as *metadata* rather than inference.** `2406.00020`'s positive result (0.36 → 0.53)
   used *given* stance. Anything that obtains stance from a source other than content inference —
   channel/uploader priors, comment threads, upload context, an annotation pass — is untouched.
2. **The `primary_voice` field on its own.** Finding (a) shows the model *does* identify the
   utterance source reasonably often, and it is the *conversion to stance* that fails. A cheap
   typed `voice` feature is a much weaker claim than a stance channel, but it was not measured
   here and is not refuted here.
3. **Fine-tuning a stance typer.** This pilot measures a frozen, prompted model only.
4. **The hate-surface disagreement (finding b)** is a threshold-alignment problem, not a stance
   problem, and it would afflict any prompted front-end equally.

**Recommended next move:** do not spend the ~$1.5 generation pass or the GPU day. If the stance
direction is to be pursued at all, it has to be through route 1 or 3 above, and each needs its own
gate before any head is trained.

## 9. Reproducibility index

| artefact | path |
|---|---|
| frozen rules, sample spec, prompt revision log | `idea-stage/STANCE_PILOT_FREEZE.md` |
| sample selection (seed 20260811) | `idea-stage/stance_pilot/select_sample.py` → `sample.json` |
| prompt bank (V1 = final V1.3, V2/V3 unused) | `idea-stage/stance_pilot/prompts.py` |
| request builder + batch driver | `idea-stage/stance_pilot/run_pilot.py`, `drive.sh` |
| fallback ② decomposition | `idea-stage/stance_pilot/run_fallback.py` |
| frozen scorer (3 views) | `idea-stage/stance_pilot/score.py` |
| round-1 raw / parsed / scored | `batch_out_strong.jsonl`, `pred_strong.jsonl`, `score_strong.json` |
| round-1 rejected item | `batch_err_strong.jsonl` |
| fallback raw / scored | `pred_fb2.jsonl`, `score_fb2.json` |
| smoke passes (V1.0 → V1.3, alias check) | `smoke_v1_strong.jsonl` … `smoke_alias_check.jsonl` |
| run logs | `logging/runs/stance_pilot/{run.log, submit_strong.log, fb2.log}` |

Data boundary as authorised: video frames were sent to the user's own DashScope account
(user ruling 2026-08-11); test-set **inputs** were used, test **labels** only as the anchor of a
disclosed capability measurement (user ruling 2026-08-09); no label was used to tune anything and
no detector hyper-parameter was selected here. The API key was read from `~/.dashscope_api_key`
at runtime and appears in no file in this repository.
