# R11-SEG — result (pilot v1)

**Date**: 2026-08-18 · **Freeze**: `idea-stage/R11_SEG_PILOT_FREEZE.md`, commit **`96635e1`**,
committed before the runner was executed · **Cost**: ¥0, local RTX 5090 only, no cloud, no API
· **Wall**: extraction 7 min, run 12 min, 0 test-label contact before the final scripted pass.

## VERDICT: **KILL**

The temporal action segmentation arm (`A4_CTCN`) does **not** beat the causal broadcast control on
HateClipSeg's online per-timestamp task. Under the frozen rule the secondary gate
`Δ_bcast = A4 − A1 ≤ 0` is a KILL condition, and it fired: **−0.828 macro-F1**, CI
[−3.805, +2.139]. The primary contrast against the per-window independent head is also null:
`Δ_main = A4 − A2 = +0.270`, CI [−1.371, +1.967], point estimate far below the pre-declared
smallest worthwhile gain δ = +1.0.

The B3 pre-check verdict is **NOT REPRODUCED** — the published modality inversion does not survive
matched heads — but its circular-shift control produced the round's most useful measurement and it
points the opposite way from B3's proposal (§4).

---

## 1. What was actually run

Task: HateClipSeg task (3), online per-timestamp binary offensive classification. Split
`p11_split.json` **237 / 39 / 119**, frozen and previously unconsumed. Grid: the canonical K=30
window grid (median ≈ 8.0 s vs median gold segment 8.12 s). Evaluation: per-timestamp at 0.25 s
stride, each timestamp taking its containing window's prediction and its containing gold segment's
label — the paper's metric shape. 12 seeds **2200-2211**, video-clustered paired bootstrap
(10 000 resamples, seed 2299).

Every arm's causality was asserted in-run: perturbing windows ≥ 20 changed the output at windows
< 20 by exactly `0.00e+00` for `A1, A2, A3, A4, A5, A6`; `A1b` leaked as declared (`1.46e+01`) and
is reported only as a non-causal reference. Split disjointness asserted in-run.

**Features.** All frozen encoders, no fine-tuning, no LoRA. Visual = CLIP-L/14-336 (already on
disk); speech text and on-screen text = CLIP text tower over the whisper window transcript and the
easyOCR window text (extracted this round, 1 min); audio = `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`,
the paper's own Wav2Vec-Emotion checkpoint, masked-mean pooled per window (extracted this round);
plus openSMILE eGeMAPSv02 88-d for the §4 robustness cell. `ALL` = 3586-d concat; `V` = visual only.
The project's Qwen2.5-VL + LoRA deployment pipeline was **not** used: the visual grid already
existed and the LoRA adapter was trained on a different dataset, which would have imported an
unquantified transfer confound into a first pilot.

---

## 2. Arm results — per-timestamp macro-F1 / accuracy on the 119-video test split

Seed-averaged probability, threshold fixed at 0.5, never tuned.

### Input `ALL` (visual + speech text + OCR text + audio, 3586-d)

| arm | ts macro-F1 | ts acc | per-seed mean ± sd | window macro-F1 | wv-AUC |
|---|---|---|---|---|---|
| A0 `CONST` (train majority) | 34.59 | 52.88 | — | — | — |
| A1 `BCAST-CAUSAL` (broadcast control) | **65.77** | 66.25 | 64.85 ± 1.66 | 65.16 | 0.5709 |
| A1b `BCAST-VIDEO` *(non-causal ref)* | 65.14 | 65.56 | 63.91 ± 1.22 | 64.85 | 0.5000 |
| A2 `PERWIN` (no temporal context) | 64.67 | 64.93 | 64.55 ± 1.63 | 64.61 | 0.6349 |
| A3 `MIL-TOPK` (video labels only, top-33%) | 54.15 | 55.00 | 53.70 ± 1.47 | 53.95 | 0.5208 |
| **A4 `CTCN` (the candidate)** | **64.94** | 65.25 | 64.42 ± 1.12 | 64.61 | 0.6375 |
| A5 `CTCN-NOSMOOTH` | 64.65 | 65.03 | 64.00 ± 1.25 | 64.60 | 0.6418 |
| A6 `CTRANS` (LSTR-family shape) | **66.51** | 66.73 | 65.48 ± 1.27 | 66.06 | 0.6406 |

### Input `V` (visual only, 1024-d)

| arm | ts macro-F1 | ts acc | per-seed mean ± sd | wv-AUC |
|---|---|---|---|---|
| A1 `BCAST-CAUSAL` | 60.61 | 61.20 | 60.64 ± 1.09 | 0.5450 |
| A1b `BCAST-VIDEO` *(non-causal)* | 64.93 | 65.85 | 62.14 ± 1.45 | 0.5000 |
| A2 `PERWIN` | 63.06 | 63.45 | 61.66 ± 1.15 | 0.6054 |
| A3 `MIL-TOPK` | 46.79 | 51.38 | 46.95 ± 1.49 | 0.5046 |
| A4 `CTCN` | 62.18 | 62.40 | 61.63 ± 1.54 | 0.6033 |
| A5 `CTCN-NOSMOOTH` | 61.72 | 61.89 | 61.53 ± 1.75 | 0.6096 |
| A6 `CTRANS` | 62.56 | 62.74 | 62.38 ± 1.21 | 0.5676 |

### Frozen contrasts, video-clustered paired bootstrap, 10 000 resamples

| contrast | `ALL` Δ [95% CI] | `V` Δ [95% CI] |
|---|---|---|
| **A4 − A2** (primary) | **+0.270 [−1.371, +1.967]** | −0.878 [−3.493, +1.781] |
| **A4 − A1** (secondary gate) | **−0.828 [−3.805, +2.139]** | +1.571 [−1.233, +4.507] |
| A4 − A6 | −1.571 [−3.560, +0.304] | −0.386 [−2.820, +1.857] |
| A4 − A5 (is smoothing load-bearing?) | +0.290 [−0.825, +1.408] | +0.454 [−0.638, +1.577] |
| A6 − A2 | +1.841 [−0.075, +3.962] | −0.492 [−3.698, +2.848] |
| A2 − A1 | −1.097 [−4.332, +2.166] | +2.449 [−0.972, +5.971] |
| A3 − A2 (price of the MIL family) | **−10.524 [−15.624, −5.663]** | **−16.269 [−22.316, −10.124]** |
| A1b − A1 | −0.630 [−4.846, +3.283] | +4.327 [+0.193, +8.782] |

### Verdict against the frozen rule

- `Δ_main = +0.270`, CI contains zero → not a GO.
- `Δ_bcast = −0.828 ≤ 0` → **KILL** (the rule's second KILL condition).
- The equivalence kill did not fire on `Δ_main` (upper bound +1.967 > δ = +1.0), so the honest
  statement is *the effect is null, not proven-tiny*, on 119 test videos. Two independent runs of
  identical code and identical seeds moved `Δ_main` from −0.248 to +0.270 on GPU nondeterminism
  alone (§6, D1) — which is itself the size of the entire claimed effect.
- Only one contrast in the whole table clears zero decisively, and it is negative: the top-k MIL
  family costs **−10.5 (ALL) / −16.3 (V)** macro-F1 against a plain per-window head. The landscape's
  narrowed structural claim survives on that one point.

**Comparability.** As frozen in §2 of the pre-registration, these numbers are **not** comparable to
HateClipSeg's LSTR 62.75 or StreamSense's 72.06: different corpus (the 90.8% surviving subset with
non-random attrition — YouTube 20.8% loss vs BitChute 6.9%, rarest strata hit hardest, per
`DATASET_hateclipseg.md §4`), different split (60/10/30 vs 80/20), different resolution (~8 s
windows vs 0.25 s stride), different encoders. Nothing here is a SOTA claim and no published number
was used as a gate.

---

## 3. Cause of death — three post-hoc measurements

These are **post-hoc descriptive** (`scripts/r11_seg/posthoc.py`, `out/posthoc.json`), not gates,
and were computed after the frozen verdict.

**(1) The online per-timestamp task is itself dominated by video-level separability.** A predictor
with a perfect video-level classifier and *zero* temporal resolution — mark every window of a video
with that video's majority label — scores **79.42 per-timestamp macro-F1 / 79.42 accuracy** on our
test split. A perfect per-window predictor scores 95.55 (the shortfall from 100 is the 8 s
quantisation). Every arm we ran sits at 54-67, i.e. **entirely below the no-temporal-resolution
ceiling.** The landscape's degenerate-oracle argument was computed for HateClipSeg on the
*localization* metric (frame-AP 0.530, "genuinely open") and does **not** carry over to this task's
macro-F1: on macro-F1 at a 0.48 base rate, HateClipSeg's online task is as gameable by video-level
information as HateMM's frame-AP is. **This is new and it is the round's most consequential
measurement** — §14.5's premise that the online task escapes the degeneracy is wrong.

**(2) There is real within-video signal, and the per-window head already has all of it.**
Within-video AUC — the read-out a video-level classifier cannot inflate, and `A1b BCAST-VIDEO`
scores exactly **0.5000**, which validates the read-out — is **0.6349** for the plain per-window
head `A2` and **0.6375** for the temporal model `A4`. The temporal operator buys **+0.0026**.
For context, `research-wiki` §5.4 records this project's previous ceiling on within-video signal at
**wv-AUC 0.576** after a 13-route campaign that included 72B MLLMs; 0.635 from a linear projection
plus a two-layer head on frozen features is the highest within-video number this project has
measured. **It came from the substrate and the four channels, not from any temporal mechanism.**

**(3) Macro-F1 rewards video-level correctness, which is why the broadcast control wins.** Fraction
of prediction variance that is between-video: `A1b` 100.0%, `A1` 90.4%, `A4` 72.0%, `A6` 62.3%,
`A5` 59.8%, `A2` 59.5%. `A1` has the *least* within-video resolution of any learned arm
(wv-AUC 0.571) and the *best* macro-F1 (65.77). The metric and the mechanism point in opposite
directions: the arms with more temporal resolution are penalised, because per-timestamp macro-F1 at
this base rate pays mostly for getting the video right.

Test-split composition, for the record: 119 videos, window base rate 0.4790, **15 all-normal, 7
all-offensive, 97 with genuine within-video variation.**

---

## 4. B3 pre-check — **NOT REPRODUCED**, and the control is the finding

Matched 2 × 2, identical `A4_CTCN` head, identical hyperparameters, identical grid, identical
causal context, both modalities at 1024-d through the identical projection, 12 seeds.

| | `LABEL` (window macro-F1) | `CHANGE` (AP, test base rate 16.20%) |
|---|---|---|
| `VIS` CLIP visual | 61.23 | 20.51 |
| `AUD` wav2vec2-emotion | 61.06 | 21.69 |
| `EGE` eGeMAPSv02 *(robustness)* | 56.93 | 19.39 |
| `VIS_shift` within-video circular shift | **61.50** | — |
| `AUD_shift` within-video circular shift | **57.76** | — |

| contrast | Δ [95% CI] |
|---|---|
| `LABEL`: AUD − VIS (needs > 0 to reproduce) | **−0.165 [−5.741, +5.452]** |
| `CHANGE`: VIS − AUD (needs > 0 to reproduce) | **−1.183 [−4.013, +1.938]** |
| `LABEL`: AUD − AUD_shift (control) | **+3.301 [+0.710, +5.888]** |
| `LABEL`: VIS − VIS_shift (control) | **−0.275 [−3.659, +2.967]** |
| `LABEL`: EGE − VIS | −4.296 [−10.133, +1.419] |
| `CHANGE`: EGE − VIS | −1.119 [−5.069, +2.239] |

**Verdict: NOT REPRODUCED.** Both required directions are null. Under matched heads, one grid and
one causal context, audio does not beat visual at labelling the moment (−0.17) and visual does not
beat audio at the boundary proxy (−1.18, if anything audio is better). The published inversion
(F1@tIoU visual 52.65 ≫ audio 25.40 from ActionFormer; online macro-F1 audio 60.84 > visual 57.52
from LSTR) is **consistent with being an architecture / context-window confound**, exactly as
§14.5's condition 2 suspected. `B3 MODASYM` should not be built on.

**The control is the result, and it inverts B3's assignment.** Circularly shifting each video's
30 window vectors, then retraining:

- **Audio loses 3.30 macro-F1, CI excluding zero** — the audio channel carries genuine
  *moment-level* information; where in the video a window sits matters.
- **Visual loses nothing (−0.28, CI [−3.66, +2.97])** — the CLIP visual channel's entire
  contribution to this task is **video-level identity**. Shuffling the visual timeline within a
  video is free.

B3 proposed "visual draws boundaries, prosody labels moments". The half that survives is the
prosody half; the visual half is refuted in the strongest available form — visual carries no
within-video temporal information here at all. This also explains §3(2): the wv-AUC 0.635 of the
multimodal `ALL` head is being carried by audio and text, not by pixels.

**Declared deviation, as frozen**: `CHANGE` is a label-change-point proxy on an ~8 s grid, used
because the raw gold-boundary target is near-saturated on this grid (73.4% of train windows contain
a gold boundary). A null on `CHANGE` does not refute the published proposal-level F1@tIoU number.

---

## 5. What this closes, and what it does not

**Closed.**
1. **"Import a TAS-family causal temporal model into HateClipSeg's online task and it will beat a
   per-window head"** — measured, null, on 12 seeds with a video-clustered CI. Also null for the
   LSTR-family causal Transformer against the same comparator (+1.84, CI [−0.08, +3.96]).
2. **The MS-TCN smoothing term is not load-bearing here** (+0.29 / +0.45, both CIs containing zero).
3. **`B3 MODASYM` as published** — not reproduced under matched heads.
4. **§14.5's premise that HateClipSeg's online task escapes temporal degeneracy** — false on
   macro-F1: the zero-temporal-resolution ceiling is 79.42, above everything anyone has published
   on the task.

**Not closed, and explicitly not tested by this pilot.**
- Whether a mechanism that *consumes* the coverage prior (rather than a generic temporal operator)
  adds anything. Nothing in A1-A6 is coverage-aware. This is the gap the novelty check
  (`idea-stage/R11_SEG_NOVELTY_CHECK.md` §8) names as the one missing mechanism, and it is the
  subject of the v2 freeze.
- The dense action detection family (MS-TCT / PAT / RefDense line, per-instant multi-label sigmoid).
  A4 is GTEA-line TAS with a binary head; the novelty check is right that the dense family is a
  closer architectural fit to a 6-way multi-hot toxicity timeline and it was not run here.
- The controlled objective test (plain per-instant BCE vs UniVTG-style score-derived intra-video
  negatives) that would make the narrowed Part A claim falsifiable rather than rhetorical.

---

## 6. Deviations

**D1 — the run was executed twice; the reported run is the second.** The first invocation was
killed by the agent harness's 2-minute tool timeout (SIGTERM to the process group) after it had
completed all 15 arm cells and 4 of 9 B3 cells; no `results.json` was written. The second
invocation is byte-identical code, identical freeze, identical seeds, fully detached, and is the
one reported. Cause is harness-level, not design-level: **no design element, threshold, arm, seed or
decision rule was changed between the two invocations**, and the freeze commit `96635e1` predates
both. The killed log is retained at `logging/runs/r11_seg/run_attempt1_killed.log`.

  *Blindness disclosure attached to D1*: the operator saw the first invocation's arm numbers before
  the second was launched. Because the code and freeze were unchanged, no decision followed from
  them. The two runs differ only by GPU nondeterminism, and that difference is material to record:
  `Δ_main` moved from **−0.248** (run 1) to **+0.270** (run 2), and `Δ_bcast` from −1.345 to −0.828.
  The KILL verdict is identical under both. That run-to-run drift is roughly the size of the
  candidate effect and is the strongest single argument that this contrast is null.

**D2 — implementation note, not a design change.** The freeze specifies "BCE with pos_weight = 1.0".
The implementation uses 2-class softmax cross-entropy, which is the same objective for a binary
target and is what MS-TCN's inter-stage protocol requires (each stage consumes the previous stage's
class posterior). No class weighting was applied anywhere.

**D3 — one video has no decodable audio track** (394/395 `audio_ok`). Its audio channel is the zero
vector, as designed for empty channels. No video was dropped.

**D4 — post-hoc analysis added.** `scripts/r11_seg/posthoc.py` (§3) was written and run after the
frozen verdict was determined. It is descriptive, changes no gate, and is labelled post-hoc
throughout.

---

## 7. Reproduction

| artifact | path |
|---|---|
| freeze (pre-run, commit `96635e1`) | `idea-stage/R11_SEG_PILOT_FREEZE.md` |
| grid + gold labels | `scripts/r11_seg/build_grid.py` → `idea-stage/r11_seg/out/grid_labels.npz` |
| CLIP text (ASR / OCR) | `scripts/r11_seg/extract_text_feats.py` → `out/text_feats.npz` |
| audio (wav2vec2-emotion + eGeMAPS) | `scripts/r11_seg/extract_audio_feats.py` → `out/audio_feats.npz` |
| arms, B3, bootstrap | `scripts/r11_seg/run_pilot.py` → `out/results.json`, `out/probs_*.npy`, `out/b3_*.npy` |
| post-hoc diagnostics | `scripts/r11_seg/posthoc.py` → `out/posthoc.json` |
| logs | `logging/runs/r11_seg/{run.log, run_attempt1_killed.log, text.log, audio.log}` |

Test discipline: the runner asserts train/val/test id disjointness and prints the result; val (39
videos) carried all epoch selection; the 0.5 threshold was never tuned; test labels were read in a
single scripted pass after training.

---
---

# R11-SEG — result (pilot v2)

**Freeze**: `idea-stage/R11_SEG_PILOT_FREEZE_V2.md`, commit **`4a45d35`**, committed before
`scripts/r11_seg/run_v2.py` was written or executed. **Cost**: ¥0, local, ~9 min.

v2 was written after `idea-stage/R11_SEG_NOVELTY_CHECK.md` (`6ad6b32`) rated the direction **(c)**
and gave a revision path to **(b)**. It tests the two things v1 could not: **(a)** the *work*
carrier — does any temporal-structure arm beat the per-window independent head, now including the
dense action detection family the novelty check named — and **(b)** the *novelty* carrier —
does a **coverage-budget constrained decode** driven by a video-level score add anything.

## VERDICT: **KILL** — neither gate passes

| gate | contrast | result | pass? |
|---|---|---|---|
| **(a) work** | `max(B2_DENSE, A4_CTCN) − A2_PERWIN`, ALL, ONLINE | **−0.233 [−1.911, +1.392]** | **no** |
| **(b) novelty** | `C1_COVBUD_ONLINE − C0_UNCONSTRAINED`, ALL, ONLINE | **−7.531 [−10.403, −4.577]** | **no** |
| (b) fallback branch | same, MULTISPAN subset (n=80) | −5.595 [−8.795, −2.370] | no |

## 1. Gate (a) — the dense family fails too

Per-timestamp macro-F1 / accuracy, 119 test videos, 12 seeds 2200-2211. `A1/A2/A4` reproduce v1
**exactly** (65.77 / 64.67); `A4` re-drew at 64.44 on GPU nondeterminism.

| arm | claim | `ALL` | `V` |
|---|---|---|---|
| `A1_BCAST_CAUSAL` learned broadcast control | control | **65.77** / 66.25 | 60.61 / 61.20 |
| `A2_PERWIN` no temporal context | comparator | 64.67 / 64.93 | **63.06** / 63.45 |
| `A4_CTCN` GTEA-line causal MS-TCN | (a) | 64.44 / 64.67 | 62.23 / 62.42 |
| **`B2_DENSE`** MS-TCT/PAT-shaped, causal, per-window multi-hot sigmoid over 5 categories | (a) | 64.34 / 64.58 | 62.01 / 62.05 |

| contrast | `ALL` | `V` |
|---|---|---|
| B2_DENSE − A2_PERWIN | **−0.326 [−2.904, +2.291]** | −1.050 [−4.873, +2.634] |
| A4_CTCN − A2_PERWIN | −0.233 [−1.911, +1.392] | −0.821 [−3.514, +1.927] |
| B2_DENSE − A1_BCAST_CAUSAL | −1.424 [−4.672, +1.953] | +1.399 [−2.395, +5.163] |

**Three architecture families have now been tried against the same per-window independent head on
this task — GTEA-line TAS (`A4`), LSTR-family causal attention (`A6`, v1), and dense action
detection (`B2`) — and all three are null.** The novelty check was right that the dense family is
the closer architectural fit to a multi-hot toxicity timeline; it does not change the answer.
Across four independent executions of identical code with identical seeds, `A4_CTCN` on `ALL` came
out at 64.42 / 64.94 / 64.58 / 64.44 — a ±0.5 spread from GPU nondeterminism alone, which is larger
than any effect claimed for the temporal operator.

## 2. Gate (b) — the coverage budget is actively harmful, and the reason is measurable

All decoders operate on the **identical** `A2_PERWIN` seed-averaged scores; the decoder is the only
thing that changes. `A4_CTCN` scores give the same pattern and are in `results_v2.json`.

| decoder | protocol | ts macro-F1 | Δ vs `C0` [95% CI] |
|---|---|---|---|
| `C0_UNCONSTRAINED` (threshold 0.5) | ONLINE | 64.67 | — |
| **`C1_COVBUD_ONLINE`** budget + causal forward filter | ONLINE | **57.14** | **−7.531 [−10.403, −4.577]** |
| `C1a_BUDGET_ONLY` | ONLINE | 55.28 | **−9.387 [−12.645, −6.093]** |
| `C1b_TRANS_ONLY` causal forward filter alone | ONLINE | 64.79 | +0.119 [−0.998, +1.194] |
| `C2_COVBUD_OFFLINE` Viterbi + global budget + duration bound | **OFFLINE** | 63.31 | −1.356 [−4.447, +1.659] |
| `C3_ORACLE_BUDGET` (gold coverage) | diagnostic | **73.64** | **+8.969 [+5.255, +13.030]** |

Stratified `C1 − C0`: MULTISPAN (n=80) −5.595 [−8.795, −2.370]; SINGLESPAN (n=24) −12.445;
LOW coverage (n=35) −4.480; MID (n=56) −1.135 [−4.400, +2.115]; HIGH (n=28) −8.685. **There is no
stratum where the budget helps.**

**Decomposition.** The transition prior — the causal HMM forward filter, i.e. the CAD-style
"transition confidences" half — is **neutral** (+0.119, CI containing zero; measured train
persistence 0.855 / 0.834). Every point of the loss comes from the **budget** half (−9.387).

**The cause, quantified.** The coverage-budget mechanism needs a per-video estimate of what
fraction of the timeline is offensive. The frozen estimator (ridge on prefix-mean features, fitted
on train) reaches Pearson **r = 0.999 on train and r = 0.344 on test** — it does not generalise.
Post-hoc (`scripts/r11_seg/posthoc_v2.py`, labelled post-hoc, no gate):

| budget source | test r vs gold coverage | ts macro-F1 |
|---|---|---|
| constant (train mean) | — | 53.11 |
| ridge on features (frozen choice) | 0.344 | 57.14 |
| the model's own causal running-mean probability | 0.509 | 61.43 |
| gold coverage + noise sd 0.30 | ≈0.76 | 69.12 |
| gold coverage + noise sd 0.15 | ≈0.92 | 70.97 |
| gold coverage (oracle) | 1.00 | 73.64 |

So the decode **would** beat `C0` — by 4 to 9 points — given a coverage predictor at r ≳ 0.7. The
best estimator available on this substrate reaches **r ≈ 0.5**, and at that accuracy the decode is
3 points *below* doing nothing. **The binding constraint is video-level coverage predictability,
not the decoder.** That is a concrete, falsifiable diagnosis and it names exactly what a future
attempt would have to fix first — but per the round-11 standing rule, a large oracle is not
evidence, and AGGNET already priced oracle-to-delivery conversion at roughly 10%, which here would
be +0.9, below δ = +1.0.

## 3. The controlled objective test — the narrowed structural claim is not confirmed here

`E1` = `A4_CTCN` with plain per-instant BCE. `E2` = the same plus a **UniVTG-style score-derived
intra-video negative** term (anchor = the windows the current model scores highest; negatives = the
lowest-scoring windows *of the same video*, chosen by relative score, not by gold; InfoNCE τ=0.07,
weight 0.1).

| | ts macro-F1 | Δ = E2 − E1 [95% CI] |
|---|---|---|
| `E1_OBJ_BCE` | 64.63 | — |
| `E2_OBJ_INTRA` | 64.95 | **+0.314 [−0.612, +1.268]** |
| by stratum | | LOW +0.300 · MID +0.090 · **HIGH +0.487 [−1.461, +2.658]** · MULTISPAN −0.258 · SINGLESPAN −0.134 |

The narrowed Part A claim ("objectives that manufacture negatives from relative within-video scores
become inconsistent as coverage → 1") predicts `E2 − E1 < 0` on the high-coverage stratum. It is
**not observed**: +0.487 with a CI containing zero on n = 28. Honest reading — HateClipSeg's mean
coverage is 0.45 and the HIGH stratum is coverage ≥ 0.75, not → 1, and 28 videos is underpowered.
The result does not refute the claim; it does mean **this project has no positive evidence for it**,
and the claim must not be asserted as though the experiment supported it.

## 4. What v2 adds to the closure

1. **Gate (a) is closed across three architecture families**, not one. "Temporal structure over
   per-segment scores" buys nothing measurable on HateClipSeg's online task.
2. **Gate (b) — the novelty carrier — is closed as specified.** Coverage-budget decoding is
   *harmful* at achievable budget accuracy, in every stratum, on both score sources. Its transition
   half is a no-op. Its oracle version is large, which under this project's own rules is a warning,
   not a promise.
3. **The one substantive open sub-question it leaves** is narrow and expensive: could a video-level
   *coverage* predictor be pushed from r ≈ 0.5 to r ≳ 0.7? Nothing in this project's record suggests
   that a video-level regression head does better than r ≈ 0.5 on a 237-video train set, and the
   payoff would still have to survive the ±0.5 run-to-run noise floor measured in §1.
4. **`C1b`'s neutrality is worth carrying forward as a fact**: HateClipSeg's window labels persist
   at 0.855 / 0.834, so an HMM transition prior is already implicit in the scores. This is the
   quantitative reason "add temporal smoothing" cannot help here, and it agrees with v1's `A4` vs
   `A5` result (the MS-TCN smoothing term was worth +0.29, CI containing zero).

## 5. v2 deviations

- **D5 — one crash-and-fix.** The first v2 invocation died with `KeyError: 'y_multi'` after the
  carried arms: `run_pilot.load_all()` builds the per-window multi-hot target but does not return
  it. Fix: `run_v2.main()` reads `y_multi` directly from the frozen `grid_labels.npz`. The fix
  touches data plumbing only — no arm, threshold, seed, metric or decision rule changed, and the
  carried arms reproduced bit-identically after the restart (`A1` 65.77, `A2` 64.67 in both). Log
  retained at `logging/runs/r11_seg/run_v2_attempt1_crash.log`.
- **D6 — post-hoc budget diagnostic** (§2 lower table, `scripts/r11_seg/posthoc_v2.py`) was written
  and run after the frozen v2 verdict. It uses gold coverage as an oracle input and is labelled a
  diagnostic throughout; it changes no gate.
- **D7 — `C3_ORACLE_BUDGET` reads test labels** to build the gold-coverage budget. It was declared
  in the freeze as a labelled oracle diagnostic, is not a gate, and no arm, threshold or
  hyperparameter was selected using it.

## 6. v2 reproduction

| artifact | path |
|---|---|
| freeze (pre-code, commit `4a45d35`) | `idea-stage/R11_SEG_PILOT_FREEZE_V2.md` |
| runner | `scripts/r11_seg/run_v2.py` → `idea-stage/r11_seg/out/results_v2.json`, `v2_probs_*.npy` |
| post-hoc budget diagnostic | `scripts/r11_seg/posthoc_v2.py` → `out/posthoc_v2.json` |
| logs | `logging/runs/r11_seg/{run_v2.log, run_v2_attempt1_crash.log}` |

Causality was asserted in-run for every arm including the two new modules (`B2_DENSE` and the
projection-exposing CTCN used by `E1`/`E2`): `max|Δ_past| = 0.00e+00` for all. The Dense arm's
multi-scale branch initially leaked future through pool-and-upsample and was rewritten as causal
moving averages *before* the freeze was executed; the leak was caught by the synthetic smoke test,
never by a real-data metric.
