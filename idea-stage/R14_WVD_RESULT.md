# R14-WVD — result

**Date** 2026-08-18 · **Freeze** `idea-stage/R14_WVD_FREEZE.md`, commit **`0f20505`**, committed
before `scripts/r14_loc/run_wvd.py` existed · **Cost** ¥0 (no cloud, no API), local RTX 5090 ·
**Wall** text extraction 1 min, pilot ~6 min, post-hoc 2 min · **Test-label contact: zero** —
the pilot runs entirely inside the 237-video train split.

## VERDICT: **KILL** — the frozen KILL rule fires on all three factors

| factor | contrast | Δ video-macro wv-AUC [95% CI] | δ = +0.020 | verdict |
|---|---|---|---|---|
| **A objective** | within-video ranking + BCE − BCE | **−0.0052 [−0.0094, −0.0011]** | no | **significantly negative** |
| **B text substrate** | hate-tuned RoBERTa − CLIP text tower | **−0.0044 [−0.0226, +0.0138]** | no | null |
| **C representation** | absolute + video-relative − absolute | **−0.0031 [−0.0155, +0.0088]** | no | null |

n = 193 out-of-fold train videos with within-video label variation, video-clustered paired
bootstrap, 10 000 resamples, seed 4299, 5 seeds × 5 folds per cell.

## 1. All eight cells

Out-of-fold, 5-fold video-grouped CV inside train, 5 seeds, no early stopping and no per-fold model
selection in any cell.

| cell | objective | text | repr | wv-AUC (sd) | F1@tIoU .3/.5/.7 | window macro-F1 | between-video var share |
|---|---|---|---|---|---|---|---|
| A0_B0_C0 | BCE | CLIP | abs | **0.5878** (0.0015) | 28.3 / 16.2 / 7.0 | 58.8 | 0.432 |
| A1_B0_C0 | +WVR | CLIP | abs | 0.5797 (0.0029) | 28.1 / 16.1 / 6.0 | 58.5 | 0.419 |
| A0_B0_C1 | BCE | CLIP | +rel | 0.5799 (0.0028) | 30.1 / 16.5 / 6.7 | 59.7 | 0.383 |
| A1_B0_C1 | +WVR | CLIP | +rel | 0.5807 (0.0021) | 28.6 / 15.0 / 6.1 | 59.3 | 0.372 |
| A0_B1_C0 | BCE | hate-RoBERTa | abs | 0.5842 (0.0049) | 29.2 / 14.0 / 7.0 | 59.3 | 0.493 |
| A1_B1_C0 | +WVR | hate-RoBERTa | abs | 0.5739 (0.0039) | 29.5 / 14.0 / 6.2 | 59.3 | 0.460 |
| A0_B1_C1 | BCE | hate-RoBERTa | +rel | 0.5780 (0.0039) | 29.9 / 16.0 / 6.9 | 59.7 | 0.446 |
| A1_B1_C1 | +WVR | hate-RoBERTa | +rel | 0.5746 (0.0028) | 29.8 / 16.1 / 7.0 | 59.9 | 0.441 |

**The whole table spans 0.574 to 0.588 wv-AUC and 14.0 to 16.5 F1@tIoU 0.5.** Seed sd is 0.0015 to
0.0049, so the spread is real and small, not noise-hidden. No cell is meaningfully better than the
plainest one.

## 2. What each factor was supposed to do, and what it did

**A — the within-video ranking objective is not merely null, it is slightly harmful.** The
motivation was that 43-49% of the head's score variance is between-video while the proposal metric
only rewards within-video ordering. The external reviewer supplied the vacuity argument in advance:
if the head can represent a per-video intercept, then within a fixed video the BCE logit ranks
exactly as the Bayes-optimal residual does, so a conditional objective adds no information and can
only cost sample efficiency. **The measurement agrees with the vacuity argument**: −0.0052 with a CI
excluding zero. Certified gold within-video pairs do not rescue what round 11's score-derived
intra-video negatives failed to buy (that arm was +0.31 macro-F1, CI containing zero) — the two
results now bracket the family from both ends.

**B — a hate-tuned text encoder does not beat CLIP's text tower here.** This was the factor with the
largest expected raw gain and the strongest prior: the ASR and OCR channels were going through a
77-token, caption-trained text tower, which is an indefensible substrate for hate semantics. Under
matched conditions it makes no difference (−0.0044, CI containing zero) and it *raises* the
between-video variance share (0.432 → 0.493), i.e. the stronger language model buys video-level
separability, not moment-level discrimination. This closes the cheapest remaining "the substrate was
just wrong" explanation.

**C — video-relative features do not change the ordering either.** Leave-one-out centroid residuals
plus within-video cosine ranks, per channel, entirely label-free: −0.0031, CI containing zero. This
was the reviewer's own highest-priority missing family (a video-set-conditioned reference head, as
opposed to a scalar per-video normalization which provably cannot change within-video ranks). It is
tested and it does not deliver on this substrate.

## 3. Post-hoc diagnostics (descriptive, no gate; `scripts/r14_loc/posthoc.py`, run after the verdict)

- **Window impurity is not the explanation.** 84.2% of train windows are pure (gold offensive
  fraction ≤ 0.02 or ≥ 0.98). Restricting the read-out to pure windows moves the baseline cell from
  **0.5893 → 0.6142** wv-AUC. A finer grid is therefore worth roughly +0.025 wv-AUC, not the tens of
  points the proposal metric needs.
- **The model barely tracks continuous window toxicity**: within-video Spearman correlation between
  the score and the gold per-window offensive fraction is **0.137**.
- **The video-level task is nearly vacuous on this corpus and this split**: 207 of 237 train videos
  contain a toxic segment, and the same head's mean-pooled video-level AUC is 0.5765. This is why
  the earlier oracle substitution assigned only +4.3 F1 points to the video-level term.

## 4. One methodological correction this round produced

The reconnaissance run (`scripts/r14_loc/recon_decode.py`) reported wv-AUC **0.671** on the 39-video
val split. That number used **val-based epoch selection** — it picked the epoch maximising val
wv-AUC and then reported val wv-AUC. Under the pilot's protocol (fixed 40 epochs, no selection, 5-fold
CV inside train) the same features and the same head give **0.588**. The recon numbers in
`R14_WVD_FREEZE.md` §1 (M5, M6, M7) are therefore *upper* readings of the score quality and should
not be quoted as clean estimates; their **relative** structure — the 2×2 oracle substitution, the
decode sweep, the single-channel ordering — is unaffected, because every arm in those comparisons
shared the identical selection procedure. Recorded rather than buried, and fixed on sight.

## 5. What this closes

1. **Within-video discrimination on this project's frozen-feature substrate does not move.** Three
   independent levers — the training objective, the text encoder, and a label-free video-relative
   representation — are all null or negative under a pre-registered rule, on top of round 11's three
   null temporal architecture families. The proposal-level localization direction is closed for this
   substrate, exactly as the frozen KILL rule specified.
2. **The "the text substrate was inadequate" hypothesis is dead**, and it was the cheapest and most
   plausible remaining explanation for the ceiling.
3. **The decode axis was closed before it was entered**, by the occupancy sweep (SED since 2019,
   video anomaly detection since 2026-04, nSEBB `2505.11889` for the per-instance variant) and by
   the reconnaissance measurement that the real-score decode leverage is 2-4 F1 points, not the
   4-9× the synthetic study suggested.
4. **A prior project number is contextualised**: the 72B MLLM per-window scorer recorded at wv-AUC
   0.5755 (`TEMPORAL_SPAN_LANDSCAPE §5.4`) is *below* the 0.588 of a two-layer head on frozen
   features under a stricter protocol. Scaling the scorer is not the missing ingredient either.

## 6. Deviations

- **D1 — post-hoc diagnostic added.** `scripts/r14_loc/posthoc.py` (§3) was written and run after
  the frozen verdict was determined. It is descriptive, changes no gate, and is labelled post-hoc.
- **D2 — implementer blindness.** The reconnaissance scripts (§4) were written and run *before* the
  freeze and their val numbers were seen by the operator. They fixed no threshold, no arm and no
  decision rule; the freeze's δ, KILL rule and endpoints were written from the oracle-substitution
  structure, not from any candidate's score. The pilot itself was a single submission with no
  crash and no re-run.
- **D3 — decoder threshold applied per fold.** The freeze specifies a prevalence-matched threshold
  computed on "the training folds of that fold"; the runner stores a per-fold threshold and applies
  it to that fold's held-out videos, which is the literal reading. No global threshold was ever fitted.

## 7. Reproduction

| artifact | path |
|---|---|
| freeze (pre-code, commit `0f20505`) | `idea-stage/R14_WVD_FREEZE.md` |
| candidate slate + hostile scores + occupancy sweep | `idea-stage/R14_CANDIDATES.md` |
| reconnaissance (descriptive) | `scripts/r14_loc/recon_decode.py`, `scripts/r14_loc/recon_headroom.py` |
| hate-tuned text features | `scripts/r14_loc/extract_hate_text.py` → `idea-stage/r14_loc/out/hate_text_feats.npz` |
| pilot runner | `scripts/r14_loc/run_wvd.py` → `idea-stage/r14_loc/out/{results.json, per_video_auc.npz}` |
| post-hoc | `scripts/r14_loc/posthoc.py` |
| logs | `logging/runs/r14_wvd/{run.log, text.log, posthoc.log}`, `logging/runs/r14_recon/*` |

The runner asserts train/val/test id disjointness and that no val or test id enters any tensor it
fits or scores, and prints both assertions at the top of `run.log`.
