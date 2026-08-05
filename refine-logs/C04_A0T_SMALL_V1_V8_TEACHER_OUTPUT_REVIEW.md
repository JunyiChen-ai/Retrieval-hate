# C04-A0T-SMALL-v1 impl-v8 — Independent Teacher-Output Reliability Review

Date: 2026-08-01
Reviewer: fresh independent Opus reviewer; zero exposure to the implementation
reasoning and zero exposure to the code/resource and payload reviews.
Instrument: the **frozen** five-rate semantic-reliability taxonomy. The reviewer
was instructed that the taxonomy is the verdict instrument and that inventing
criteria, relaxing a threshold or substituting judgement for it is forbidden.
Subject: `artifacts/c04/a0t_small_v1_impl_v8/seal/` from GPU job `13857`
(COMPLETED, 0:0, 2668 s, 1xA100).

## VERDICT: `KILL_C04_TEACHER_SEMANTIC_RELIABILITY`

**Scope.** This kills the C04 teacher's *semantic reliability* under the frozen
v8 contract: Qwen2.5-VL-7B-Instruct, 8 frames + native transcript, two frozen
prompt forms A/B, greedy decoding, `max_new_tokens=256`, `max_pixels=151200`, on
the sealed 200+200 pre-label tranche. It does **not** claim anything about
accuracy, label correlation, downstream utility, or the C04 method (`FULL_Q4` and
its controls are untested). It does not claim a 7B VLM cannot ever produce
SPaSH-style roles — only that *this frozen prompt/model/decode/agreement
configuration* does not, by margins far outside any threshold quibble.
`labels_opened: false` holds; no label was read.

## Recomputed rate table (reviewer's own numbers vs the sealed manifest)

Frozen thresholds (`configs/c04/c04_a0t_small_v1_v8.json` -> `reliability`,
unchanged since the v6 freeze): usable >= 0.85, missing <= 0.10,
conflict <= 0.20, max_nonfallback_value_frequency <= 0.90, joint >= 0.60.

**HateMM** (n=200) — `prompt_parse_rate` 0.6525; `joint_all_four_usable` 0.2100
(**FAIL**, bar 0.60)

| slot | stable/single/conflict/missing | usable | missing | conflict | maxfreq | cell |
|---|---|---|---|---|---|---|
| S | 57/61/66/16 | 0.590 **F** | 0.080 P | 0.330 **F** | 0.5593 P | **FAIL** |
| P | 43/35/119/3 | 0.390 **F** | 0.015 P | 0.595 **F** | 0.0256 P | **FAIL** |
| T | 74/40/82/4 | 0.570 **F** | 0.020 P | 0.410 **F** | 0.3860 P | **FAIL** |
| H | 66/49/75/10 | 0.575 **F** | 0.050 P | 0.375 **F** | 0.3130 P | **FAIL** |

**MHC_zh** (n=200) — `prompt_parse_rate` 0.5425; `joint_all_four_usable` 0.2100
(**FAIL**)

| slot | stable/single/conflict/missing | usable | missing | conflict | maxfreq | cell |
|---|---|---|---|---|---|---|
| S | 32/56/89/23 | 0.440 **F** | 0.115 **F** | 0.445 **F** | 0.4886 P | **FAIL** |
| P | 29/52/95/24 | 0.405 **F** | 0.120 **F** | 0.475 **F** | 0.0123 P | **FAIL** |
| T | 46/47/98/9 | 0.465 **F** | 0.045 P | 0.490 **F** | 0.4624 P | **FAIL** |
| H | 44/45/100/11 | 0.445 **F** | 0.055 P | 0.500 **F** | 0.5056 P | **FAIL** |

**8/8 slot cells fail; 2/2 joint gates fail.** The nearest miss is HateMM-S usable
0.590 against 0.85 — **0.26 absolute short**. Nothing is borderline.

Slot mapping, per the frozen design (the review request stated it wrongly and the
reviewer corrected it): **S = source_relation, P = proposition,
T = presenter_stance, H = the ordered pair protected_target;harm_act.** The
recomputation follows the frozen mapping. This was an error in the request text,
not in the data or the code.

## Independent recomputation vs the sealed manifest: AGREES exactly, 0 discrepancies

The reviewer re-implemented the parse and the four-state slot logic from the
frozen design description rather than importing the project's module.

- 800/800 raw strings re-parsed: **0** disagreements across 7,200 compared fields.
- 1,600/1,600 slot states re-derived: **0** disagreements against `canonical_bank.jsonl`.
- All 20 rates and all 32 state counts reproduce bit-for-bit.
- P-slot cosines independently recomputed on CPU from the frozen snapshot's
  `model.embed_tokens.weight` (152064x3584 bf16) with the reviewer's own
  normalizer: 214 cosines, **max |delta| = 1.7e-05, 0 threshold flips**.
- Sealed file SHA-256s match `seal_manifest.sealed_output_hashes`.
- `build_slot_reliability` implements the `C04_REFINED_PROPOSAL_V2.md` four-state
  rule exactly, including the "any available valid form has confidence <3 ->
  conflict" clause.

## Mechanism: this is the teacher, not the harness

Three independent, additive failure modes, quoted verbatim from `raw_output`.

**(1) Verbatim schema echo / multi-value enum picks — the largest single defect.**
The teacher copies the prompt's pipe-alternation *as the value*:

> `{"source_relation":"current_presenter|quoted_or_embedded|performed_or_lyric|mixed|uncertain","proposition":"The current presenter is questioning the actions of the other participants.","presenter_stance":"reject_or_counter|report_or_describe|perform_without_clear_commitment|uncertain","protected_target":"race|ethnicity|religion|...","harm_act":...}` — `hate_video_334` form B

Attribution over 400 forms per dataset: HateMM 72 verbatim-echo + 62 pipe-subset
+ 11 invented values; MHC_zh 11 verbatim-echo + 40 pipe-subset + 0 invented.
Sub-cases like `"current_presenter|performed_or_lyric"` (15 HateMM / 22 MHC_zh)
are the model refusing to commit. This is a prompt-comprehension failure by the
teacher, unrecoverable without inventing a disambiguation rule.

**(2) The `confidence` map is simply omitted.** 33 HateMM + 52 MHC_zh forms drop
it entirely, invalidating all four slots at once:

> `{"source_relation":"performed_or_lyric","proposition":"The speaker describes a group of people who use various tactics to win arguments.","presenter_stance":"report_or_describe","protected_target":"uncertain","harm_act":"none"}` — `non_hate_video_450` form A

**(3) Genuine A/B semantic disagreement — the binding wall.** Among pairs where
*both* forms are fully valid, the two prompt forms disagree at 0.351/0.638/
0.391/0.374 (HateMM S/P/T/H) and 0.380/0.421/0.453/0.389 (MHC_zh). Example, both
forms fully valid, `hate_video_103`:

> A: `{"source_relation":"uncertain","proposition":"A train is moving along a track.","presenter_stance":"uncertain","protected_target":"race","harm_act":"attack","confidence":{"S":3,...}}`
> B: `{"source_relation":"quoted_or_embedded","proposition":"The train is moving along the tracks.","presenter_stance":"uncertain","protected_target":"uncertain","harm_act":"none","confidence":{"S":1,...}}`

Identical visual content, opposite harm reading, self-reported confidence 3 -> 1.
The P-cosine distribution is bimodal, not thresholded away: HateMM median 0.633,
23.9% at >= 0.999. Relaxing the 0.80 cosine bar to **0.00** — accepting *any* two
propositions as agreeing — still leaves P usable at 0.765 (HateMM) / 0.545
(MHC_zh), both below 0.85.

### Refuting "the harness mis-parsed a fine teacher" — three diagnostic-only counterfactuals

- **Lenient parser** (strip ```json fences, outermost-brace slice, truncation
  repair) recovers 106 of 800 forms and lifts `prompt_parse_rate` to 0.7100 /
  0.6750 — but **joint FALLS**: 0.21 -> 0.175 (HateMM) and 0.21 -> 0.135
  (MHC_zh), because recovered forms convert `single_valid` into `conflict`. Being
  more permissive makes it worse.
- **Drop the confidence >= 3 gate entirely**: best case usable 0.720-0.760, joint
  0.330/0.440. Still fails 0.85 and 0.60 everywhere.
- **Perfect-parser ceiling**: if every form parsed, `single_valid` is by
  construction 0 and usable == A/B agreement == 0.362-0.649. **Structurally below
  0.85 in all 8 cells.**

Truncation at `max_new_tokens=256` is a minor real contributor: 4 HateMM + 8
MHC_zh forms have unbalanced braces, on long hate-lyric transcripts or degenerate
repetition. It explains <2% of the loss.

## The MHC-ZH transcript-poverty confound: real, but NOT explanatory

Transcript lengths re-derived through the frozen label-skipping projection
(`label_value_materialized == 0`); all 400 reconstructed transcripts hash-match
`input.transcript_sha256`, 0 mismatches. MHC_zh p10 = 8, median 63.5, 24 items
< 10 scalars; HateMM p10 = 38, median 999, 5 items < 10.

A monotone length gradient exists (and exists on HateMM too, so it is not
ZH-specific):

| MHC_zh stratum | n | S usable | P usable | T usable | H usable | joint |
|---|---|---|---|---|---|---|
| < 10 scalars | 24 | 0.25 | 0.21 | 0.33 | 0.33 | 0.17 |
| 10 <= s < 64 | 76 | 0.36 | 0.32 | 0.37 | 0.39 | 0.18 |
| s >= 64 | 100 | 0.55 | 0.52 | 0.57 | 0.51 | 0.24 |

Exclusion counterfactuals: drop all 24 short items (n=176) -> best slot usable
0.483, joint 0.216. Keep only >= median (n=100) -> best 0.570, joint 0.240. Keep
only >= p75 = 171 scalars (n=51) -> best 0.667, joint 0.255. **No stratum, at any
cut, reaches 0.85 on any slot or 0.60 joint.** The confound moves the numbers by
~0.15-0.25 and the bar is 0.26-0.45 away. **The MHC-ZH kill survives deleting
every poor-transcript item**, so it is separable from the confound.

Adjacent finding: 78/314 parsable MHC-ZH propositions are English despite the
`chinese_proposition_max_64_unicode_scalars` contract, and 24 of the 39 MHC-ZH
proposition-bounds violations are English sentences overflowing a scalar cap
sized for Chinese. This inflates MHC-ZH `P.missing_rate` (0.120) somewhat, but P
usable is 0.405 — not load-bearing for the verdict.

`HateMM/hate_video_95` (frozen all-black pack, both forwards): the teacher
answered fluently from transcript alone. 2 of 800 forwards; no effect on any rate
at the third decimal.

## What this licenses

**Licensed:** recording `terminal_state = KILL_C04_TEACHER_SEMANTIC_RELIABILITY`
as a measured, independently reproduced result; closing the v8 teacher
configuration; citing the tranche as evidence that a 7B open VLM under a
strict-schema two-form self-consistency protocol yields 0.39-0.59 slot usability
and 0.21 four-way joint coverage on hateful-video train items. The seal is
scientifically clean — pre-label, hash-lineage intact, exactly reproducible.

**Not licensed:** opening labels
(`label_access_allowed_after_this_seal_only_if_reliability_passes: false` — the
gate did not pass); any accuracy, C1 or C2 claim; any statement about the SPaSH
tensor, its role maps, or `FULL_Q4` vs `CONCAT_ALL4_MLP` /
`RETAINED_INDEPENDENT4`; any re-derivation of a verdict from these same 800
forwards under a rewritten parser, relaxed threshold or retuned prompt
(`reliability.prompt_model_threshold_rewrite_after_failure: false` and
`no_retry_redraw_prompt_rewrite: true` forbid it, and the counterfactuals show it
would not help anyway).

**If C04 is to continue**, three changes each requiring a *new* pre-registered
tranche, not a re-scoring of this one: (a) constrained/grammar-forced decoding or
per-field single-token classification instead of free-form JSON — kills
mechanisms (1) and (2), worth roughly +0.13 to +0.18 usable; (b) a different or
larger teacher — mechanism (3), the 0.36-0.65 A/B agreement wall, is the only one
that matters and is untouched by any harness fix; (c) dropping the two-form
self-consistency requirement, which would eliminate the `conflict` state but also
eliminate the reliability instrument itself. Given (3) alone caps usable at 0.65
against a 0.85 bar, the reviewer would not fund (a) without first cheaply
measuring A/B agreement for a candidate replacement teacher.
