# R15-NT — pre-registration (frozen before any pilot code exists)

Round 13 of idea discovery. Sub-direction: **hateful video temporal localization, proposal level**,
the same arena as round 12. Date 2026-08-18. Arena: HateClipSeg, local 395-video subset, frozen
split `data/gt/HateClipSeg/p11_split.json` (237 / 39 / 119).

This document is committed **before** `scripts/r15_nt/run_nt.py` exists. No R15 pilot code has been
written. No candidate metric has been computed.

---

## 1. Why this pilot exists

Round 12 closed the objective axis, the text-substrate axis and the label-free video-relative
representation axis (`idea-stage/R14_WVD_RESULT.md`). The round-13 hostile review
(gpt-5.6-sol, xhigh, conversation only; bundle
`idea-stage/codex_brainstorm_bundle_r15_2026-08-18.md`) scored twelve candidates, gave **no candidate
a 3**, and named exactly one empirically live mechanism family, on the strength of an internal
contradiction in this project's own numbers:

| reading | wv-AUC | protocol |
|---|---|---|
| four-channel concat, 5-fold CV in train, no epoch selection | **0.5878** | `R14_WVD_RESULT.md` §1, cell `A0_B0_C0` |
| **audio channel alone** | **0.623** | `R14_WVD_FREEZE.md` §1, **M7 — val split, n=39** |
| all four channels | 0.671 | same M7 source |

If audio alone really beats the four-channel fusion, then the fusion is **diluting** the channel that
the round-11 circular-shift control identified as the only one carrying moment-level information
(audio −3.30 macro-F1 when shuffled, CI excluding zero; CLIP visual −0.28, CI containing zero), and
the live family is *temporal-informativeness-aware fusion / within-video nuisance suppression*.

**But the two numbers are not protocol-matched.** `R14_WVD_RESULT.md` §4 records that every
reconnaissance number, M5-M7 included, was produced with **val-based epoch selection** and is an
upper reading; the 0.671 in M7 is the same run that becomes 0.588 under the pilot's no-selection
5-fold protocol. The reviewer stated the consequence explicitly:

> If the 0.623 result was not produced under exactly the same five-fold/no-selection protocol as
> 0.5878, then it is not admissible evidence. Under that alternative, the honest answer becomes:
> no legal family is left.

**This pilot resolves that, and it is written so that either outcome closes something.** It is a
falsification probe, not a candidate confirmation. The head, the grid, the optimiser, the features
and the read-out are all unchanged from round 12; only the channel composition changes.

---

## 2. Design — part 1, the matched channel-composition panel (R15-NT)

**Arena.** HateClipSeg **train split only** (237 videos). 5-fold **video-grouped** cross-validation
inside train. Every reported number is out-of-fold. The 39-video val split and the 119-video test
split are **not opened by the runner**; an id-disjointness assertion and a path guard are printed at
the top of `run.log`.

**Grid.** The canonical frozen K=30 uniform window grid from round 11
(`idea-stage/r11_seg/out/grid_labels.npz`), ~8.0 s/window. Unchanged, deliberately: M2 puts this
grid's representation ceiling at F1@tIoU0.5 = 87.6 against a current 23.8, and changing it would
confound the only factor under test.

**Fixed across every arm.** Head = round 11's `PerWin` (Linear→GELU→Dropout(0.1)→256, then
256→256→GELU→2). AdamW lr 1e-3, weight decay 1e-2, full-batch, **exactly 40 epochs, no early
stopping, no per-fold model selection, no scheduler** — identical optimisation budget in every arm,
no selection channel anywhere. Loss = 2-class softmax cross-entropy on the per-window binary
offensive label `y_win`. Feature z-scoring fitted on the training folds only. The input width
differs between arms and nothing else does.

**Channel blocks** (all already on disk, all frozen encoders):

| block | source | width |
|---|---|---|
| `V` visual | CLIP-L/14-336 `subclipK30`, 4 frames/window mean-pooled | 1024 |
| `T` speech text | CLIP-L/14-336 **text tower** over the window's Whisper transcript | 768 |
| `O` on-screen text | CLIP-L/14-336 text tower over the window's PaddleOCR text | 768 |
| `A` audio | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, masked-mean per window | 1024 |
| `M` masks | ASR-nonempty and OCR-nonempty indicators | 2 |

**Within-video centering, defined now.** For a channel block `x` of a video with K = 30 windows, the
**leave-one-out** within-video centered version is

`cent(x)_k = x_k − mean_{j ≠ k} x_j`

computed **per video, per channel block, label-free**, using only that video's own windows. For the
masked blocks `T` and `O`, the mean is taken over non-empty windows only and empty windows are left
at the zero vector; the `M` mask channels are never centered. Leave-one-out is used so a window does
not subtract a shrunken copy of itself.

This is **feature-level residualization, not score normalization**: K11 (a positive affine map of
the score preserves within-video ranks exactly) does not apply. It is also **not** round 12's factor
C, which *appended* a leave-one-out residual and a within-video cosine rank to the absolute features
and was null (−0.0031). The distinguishing claim here is **removal** — making the video-identity
component inaccessible to the head — which is an inductive-bias claim, not an information claim,
because the centered features carry strictly less information than round 12's factor-C input.
That asymmetry is stated up front and is the honest prior against this pilot.

### The seven arms

| arm | input | width | role |
|---|---|---|---|
| `ALL` | V ⊕ T ⊕ O ⊕ A ⊕ M | 3586 | **protocol anchor** — must reproduce round 12's `A0_B0_C0` (0.5878) up to the seed/fold-seed change |
| `AUD` | A | 1024 | **the reviewer's real bar.** Tests M7's 0.623 under the matched protocol |
| `VIS` | V | 1024 | secondary, descriptive |
| `TXT` | T ⊕ O ⊕ M | 1538 | secondary, descriptive |
| `ALLCENT` | cent(V) ⊕ cent(T) ⊕ cent(O) ⊕ cent(A) ⊕ M | 3586 | fallback mechanism arm |
| **`AUDCENT`** | A ⊕ cent(V) ⊕ cent(T) ⊕ cent(O) ⊕ M | 3586 | **the primary candidate** — the reviewer's arm D: raw audio plus within-video residuals of everything else |
| `AUDVIS0` | A ⊕ T ⊕ O ⊕ M | 2562 | fallback mechanism arm — the cheapest possible nuisance removal: delete the channel the shift control found temporally uninformative |

7 arms × 5 folds × 5 seeds = **175 head fits**. At the round-12 measured cost of ~1.4 s per fit this
is under 5 minutes of GPU wall on the local 5090. No feature extraction is required: every array
listed above already exists.

**Seeds.** Model seeds `4300, 4301, 4302, 4303, 4304`; fold-assignment seed `4310`; bootstrap seed
`4399`. All outside the consolidated banned ranges (0-119, 400-429, 500-529, 600-629, 700-729,
1300-1524, 2000-2021, 2200-2211, 4200-4204, 4210, 4299, 41000-41029).

### Endpoint and inference

**Primary endpoint: video-macro within-video AUC (wv-AUC)** — per video, the Mann-Whitney statistic
of the window score against `y_win`, videos with a constant label excluded, unweighted mean over the
qualifying videos, then averaged over the 5 seeds. A broadcast predictor scores exactly 0.500 by
construction, so no video-level classifier can inflate it. Population: the **193** out-of-fold train
videos with within-video label variation — the same population round 12 used.

**Inference: video-clustered paired bootstrap, 10 000 resamples, seed 4399**, resampling videos with
replacement and recomputing the paired difference of seed-averaged per-video AUCs.

**Secondary, reported but not a gate:** proposal F1@tIoU 0.3/0.5/0.7 under round 12's frozen decoder
(no smoothing; merge intervals separated by ≤ 5 s; drop intervals shorter than 12 s; per-fold
prevalence-matched threshold fitted on that fold's training folds), and the between-video share of
score variance.

### Smallest worthwhile gain

**δ = +0.010 wv-AUC.** Justification, fixed now: measured per-cell seed sd in round 12 is
0.0015-0.0049, so δ is 2-6 seed sd; it is the value the external reviewer prescribed; and it is half
of round 12's δ = +0.020, which is defensible here because this pilot asks only *whether a family is
alive*, not whether a mechanism closes a 63-point gap. Any downstream proposal-level claim would
additionally have to clear **+1.0 absolute F1@tIoU0.5** against a paired same-seed control, since the
measured GPU-nondeterminism floor on macro-F1 is ±0.5.

---

## 3. Decision rules — frozen

### Gate G0 — is the reviewer's premise admissible at all?

`Δ_G0 = AUD − ALL`.

- If the 95% CI for `Δ_G0` has an **upper bound below +0.010**, the premise is **refuted under matched
  protocol**: audio alone does not beat the fusion, M7's 0.623 is confirmed as a val + epoch-selection
  artifact, and the negative-transfer motivation for the whole family collapses. This is recorded as
  the round's structural finding **whatever the mechanism arms do**.
- G0 is diagnostic. It does not by itself kill the mechanism arms, because a mechanism could work
  through a route other than audio dominance. It does determine what the round is allowed to claim
  the motivation was.
- **Pre-declared: `AUD > ALL` alone is a baseline correction, not a method, and licenses no paper.**

### Primary rule P1 — the candidate mechanism

Both contrasts must pass:

1. `Δ_P1a = AUDCENT − ALL ≥ +0.010` with its 95% video-clustered CI excluding zero, **and**
2. `Δ_P1b = AUDCENT − AUD ≥ +0.010` with its 95% video-clustered CI excluding zero.

If both pass → **GO**: the temporal-informativeness / nuisance-suppression family survives round 13
and proceeds to an independent confirmatory pre-registration at a fresh seed block.

If either fails → `AUDCENT` is dead.

### Fallback rule P2 — the two other mechanism arms

If P1 fails, a **conditional GO** is available to `ALLCENT` or `AUDVIS0`, and only under a
Bonferroni-corrected bar for the two extra looks: the arm must beat **both** `ALL` and `AUD` by
≥ +0.010 with **both** CIs excluding zero at the **97.5%** level. A conditional GO is not a result;
it mandates an independent confirmatory pre-registration at a fresh seed block before any claim is
made anywhere.

### KILL rule — and what it closes

If **no** arm in {`AUDCENT`, `ALLCENT`, `AUDVIS0`} clears its bar above, the
**temporal-informativeness / within-video nuisance-suppression family is KILLED**. Combined with the
twelve closures already on the round-13 kill list (decode, within-video contrastive/ranking
objectives, temporal architectures, per-timestamp macro-F1 and the HateMM/MHC arenas, bigger scoring
models, the text substrate, window impurity, coverage-budget decoding, gold spans as supervision,
generic boundary detection, affine score normalization, and measurement contributions), the round's
written conclusion is:

> **No legal mechanism family remains for hateful-video temporal localization on this substrate
> under the project's current constraints**, and the goal is escalated to the user as a scope
> question — not quietly re-attempted.

This sentence is committed **now**, before any number exists, so that a null cannot be reinterpreted
after the fact.

---

## 4. Design — part 2, the fixed-score falsification panel (R15-FS)

Computed on the **out-of-fold, seed-averaged per-window scores dumped by the `ALL` arm**. No
fitting, no model selection, no new training. CPU seconds. Two candidates from the slate are
falsified or kept here, and both gates are one-sided ceilings.

**FS-A — evidence/label offset (slate candidate D10).** For each shift s ∈ {−2, −1, +1, +2} windows,
re-read wv-AUC with the score sequence shifted by s against unchanged labels, windows falling off
either end dropped from that video's read-out. **D10 survives only if** some s gives Δ ≥ +0.010
against s = 0, with a Bonferroni-adjusted CI (4 tests, 98.75% level) excluding zero, **and** the same
sign wins in at least 4 of the 5 folds. Otherwise the evidence-to-label alignment is not
misspecified and D10 is dead.

**FS-B — region-pooling ceiling (slate candidate D4).** Two oracle read-outs, both explicitly
labelled oracle diagnostics because they consume gold structure:

- **FS-B1** partition each video at gold segment endpoints (each window assigned to the gold segment
  containing its midpoint), average the score inside each part, broadcast back, re-read wv-AUC.
- **FS-B2** partition each video into maximal runs of same-`y_win` windows, average, broadcast back,
  re-read wv-AUC. This is the **more generous** ceiling — it uses the label itself to define the
  region — and it is the gate.

**D4 survives only if FS-B2 gives Δ ≥ +0.015 wv-AUC.** If even a label-defined region partition
cannot buy that, no label-free clustering approximating it can, and the SEGPOOL family is dead
without a line of clustering code being written.

**FS-C — label-free control, descriptive, no gate.** A centred running mean of width 3 windows over
the score, re-read wv-AUC. Its purpose is to say whether any of an FS-B gain is reachable without
boundaries.

**Deliberate scope limit.** FS-A and FS-B operate on the score curve, which is K1 territory. They are
run here only as **falsifiers** — a null closes a slate candidate at zero cost. A positive FS result
would **not** be claimable as a decode contribution; it would only redirect the search, and that
restriction is frozen here.

---

## 5. What this pilot cannot do, stated in advance

1. **It cannot produce a method paper.** The reviewer scored no candidate a 3. Every outcome of this
   pilot is either a closure or a redirect to a confirmatory pre-registration.
2. **It cannot close the ActionFormer gap.** Our pipeline sits at F1@tIoU0.5 = 23.8 against the
   dataset paper's published ActionFormer 52.65. The reviewer's verdict — that a competent detector
   baseline (feature pyramid, point assignment, boundary-distance regression) is a **precondition**
   for any method claim in this arena, and that reproducing it is a baseline-readiness round rather
   than a method round — is recorded as an open item for the user, not resolved here.
3. **No SOTA claim is available on this corpus** in any case: our copy is the 90.8% surviving subset
   with non-random attrition. Method-vs-method on the identical frozen subset only, with the
   selection-bias statement attached.

## 6. Test discipline

Train split only. The runner asserts train/val/test id disjointness and that no val or test id
enters any tensor it fits or scores, and prints both assertions at the top of `run.log`. A path guard
raises on any attempt to open a file whose name contains `test.jsonl`. `gold_segments.json` is read
for train ids only, and only for the FS-B oracle partitions, which are labelled diagnostics.
`p11_split.json` is read for its `train` list; the `val` and `test` lists are read for the
disjointness assertion only.

## 7. Execution discipline

Single submission, background `setsid nohup`, log + PID at `logging/runs/r15_nt/{run.log, run.pid}`.
If the run crashes, the crash log is retained and the deviation is recorded; no arm, threshold, seed,
endpoint or decision rule may change between invocations.
