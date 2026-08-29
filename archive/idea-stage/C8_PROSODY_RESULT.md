# C8 — Prosody-as-operator binding: **KILL** (audio-operator family closed)

Round-4 pilot #1 (`IDEA_REPORT.md` §7.8 queue position 1). Run 2026-08-09, single submission,
CPU only, **874 s**. Rules frozen in `idea-stage/PILOT_FREEZE_2026-08-09.md` §C8 **before any
implementation line was written**; transcribed unedited below. Raw
`idea-stage/c8_prosody.json`, code `idea-stage/c8_prosody_operator.py`, log
`logging/runs/c8_prosody/run.log`.

---

## Verdict

**KILL.** Both pre-registered prosody representations fail all three frozen conditions. The
interaction term does not merely fail to help inside the text-boundary band — it **hurts more
inside the band than outside it**, which is the opposite sign of the effect §7.4 predicted.

Frozen rule, transcribed unedited from `PILOT_FREEZE_2026-08-09.md` §C8.7:

```
(a)  mean over 3 seeds of  Δ_int  ≥  +0.010          (AUC, inside the band)
(b)  all 3 seeds have      Δ_int  >  0               (3/3 same sign)
(c)  mean Δ_int  >  P95 of the 30 placebo Δ_int^perm (placebo does not reproduce it)

GO     iff  arm P PASSES  OR  arm C PASSES
KILL   otherwise
```

`Δ_int = AUC_band(M2) − AUC_band(M1)`, where M2 = text ⊕ prosody ⊕ bilinear(text×prosody) and
M1 = text ⊕ prosody. M2 ⊃ M1, so the increment isolates the interaction block.

| gate quantity | **arm P — eGeMAPSv02 (88-d)** | **arm C — CLAP `proj` (1024-d)** | required | met |
|---|---|---|---|---|
| Δ_int, seed 20260901 | **−0.0532** | **−0.0399** | | |
| Δ_int, seed 20260902 | **−0.0416** | **−0.0439** | | |
| Δ_int, seed 20260903 | **−0.0360** | **−0.0338** | | |
| **(a) mean Δ_int** | **−0.0436** | **−0.0392** | ≥ +0.010 | **no** |
| **(b) seeds with Δ_int > 0** | **0 / 3** | **0 / 3** | 3 / 3 | **no** |
| **(c) placebo P95 (n=30)** | −0.0110 (mean −0.0577) | −0.0115 (mean −0.0518) | mean > P95 | **no** |
| arm PASS | **false** | **false** | | |

`arm P PASS = false`, `arm C PASS = false` → **KILL**. There is no AMBIGUOUS branch. Per §C8.7
this **closes the audio-operator family** on this project's data: prosody-as-operator,
FiLM / gating / bilinear audio conditioning, and any successor whose mechanism is "audio modulates
text". No re-run, no re-tuning, no re-specification of this hypothesis on this dataset.

The VOID clause did not fire: every band held 211 items with ≥ 72 of the minority class
(minimum 72, bar 20).

---

## What was actually run

**Data.** HateMM **train split only**, 744 rows. The **39** whitespace-only-transcript rows were
excluded per the freeze → analysable **N = 705**, label base rate **0.4184** (up from 0.4005 on all
744, because the excluded rows are 92.3 % non-hate — the audit's §2d fact, visible again here).
`dev_seen`/val was not
used at all; `test.jsonl` and `test_seen_*` were never opened. `pilot_a`'s path guard was armed —
**4 paths touched**, listed in the JSON, none containing `test`:

```
data/gt/HateMM/train.jsonl
data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt
data/audio/HateMM/egemaps_v02_trainval.pt
data/audio/HateMM/clap_larger_clap_general_trainval.pt
```

**Features — named, per the §7.4 asset correction, with no silent substitution.**

| role | tensor | dim | file |
|---|---|---|---|
| text | CLIP ViT-L/14-336 `text_feats` | 768 | `train_openai_clip-vit-large-patch14-336_HF.pt` |
| prosody, arm **P** | openSMILE eGeMAPSv02 Functionals | 88 | `egemaps_v02_trainval.pt` |
| prosody, arm **C** | CLAP `laion/larger_clap_general` `proj` | 1024 | `clap_larger_clap_general_trainval.pt` |

**CLAP is cached for HateMM only** (§7.4/§7.7-5); this pilot is HateMM-only, so no substitution was
required. The eGeMAPS cache SHA256 `0bcb11cd5a55…70ff8944` matches the one recorded in
`refine-logs/APX_GATE_RECORD.md`, so arm P is measured on literally the same bytes the 2026-07-16
marginal kill used. Whisper-large-v3 encoder features were **pre-registered as excluded** because the Whisper
encoder carries lexical content and a text×Whisper interaction would be partly text×text.

**Arms.** `sklearn` logistic regression, `C=1.0`, 5-fold stratified OOF, seeds 20260901/2/3, all
preprocessing fitted on the training fold only.
M0 = `PCA_64(z(text))` · M1 = M0 ⊕ `PCA_16(z(prosody))` · M2 = M1 ⊕ `z(outer(t_8, p_8))` (64 terms).

**Band.** Middle 30 % by rank of M0's OOF probability — computed from the **text-only** model, so it
never sees prosody and is identical across arms and placebo replicates (fully paired). n = 211.

**Placebo.** Prosody rows permuted **within label strata** (§7.4's own wording, reduced to label
strata because HateMM is single-language), whole pipeline re-run, 3 seeds × 10 permutations = **30**.

**Smokes before submission.** Synthetic positive control with a planted text×prosody interaction:
Δ_int = **+0.3203** (32× the bar) — the pipeline detects an interaction when one exists.
Label-permuted negative control on the real features: Δ_int = **−0.0013** — it does not manufacture
one when none exists.

---

## The finding, stated at the right strength

### 1. The marginal replicates. The conditional is worse than the marginal.

| quantity | arm P (eGeMAPS) | arm C (CLAP) |
|---|---|---|
| **Δ_marg** = AUC_full(M1) − AUC_full(M0) — the Phase-1 estimand | **+0.0031** | **+0.0122** |
| **Δ_int** = AUC_full(M2) − AUC_full(M1), whole population | **−0.0352** | **−0.0194** |
| **Δ_int** inside the boundary band (the gate) | **−0.0436** | **−0.0392** |
| **Δ_int** outside the band (the outer 70 %) | **−0.0284** | **−0.0104** |

The marginal arm reproduces the project's existing audio record — eGeMAPS ≈ 0, CLAP ≈ +0.01 — which
is the independent check that the machinery is wired correctly (`APX_GATE_RECORD.md` best-k
−0.0038; `CLAP_GATE_RECORD.md` `proj`/deployed best-k −0.0009). **The conditional term is not a
rescue: it is a further loss.**

### 2. §7.4's dilution prediction comes out inverted

§7.4 predicted the interaction's effect would be **concentrated** in the boundary band and diluted
to invisibility in the global average. Measured, the interaction is **more damaging inside the band
than outside it** in both arms (P: −0.0436 vs −0.0284; C: −0.0392 vs −0.0104). The boundary band is
where the text signal is weakest, so it is where 64 extra parameters cost the most and return the
least. The estimand argument was a real argument and it was testable; the data answered it the other
way.

### 3. The most favourable reading available, reported so nothing is hidden

The real pairing is **less harmful than the permuted pairing** on the mean: −0.0436 vs placebo mean
−0.0577 (arm P), −0.0392 vs −0.0518 (arm C). The observed value sits at the **63rd / 67th
percentile** of its own placebo distribution. So there is a directional hint that the true
text–prosody pairing is worth something relative to a label-matched random pairing — roughly
**+0.014 AUC** of "less damage".

This changes no verdict, and the freeze pinned P95 rather than the mean for exactly this reason: a
30-replicate placebo distribution with SD 0.034 and max +0.0179 puts a 63rd-percentile observation
squarely inside noise, and the absolute quantity is still **negative and 5× below the bar in the
wrong direction**. Condition (a) alone fails by 0.054 AUC.

### 4. A by-product: prosody's label information on HateMM is *redundant with the transcript*

Not a gated quantity; a descriptive comparison read off the placebo runs, recorded because it is the
mechanically informative part.

| Δ_marg = AUC_full(M1) − AUC_full(M0) | observed | placebo mean (within-label permuted prosody) |
|---|---|---|
| arm P (eGeMAPS) | **+0.0031** | **+0.0294** |
| arm C (CLAP) | **+0.0122** | **+0.0448** |

Within-label permutation preserves `P(prosody | label)` exactly and destroys only the coupling
between prosody and text. A **randomised** prosody vector therefore adds **~4–10× more** to a
text head than the real one does. The natural reading: real prosody's label-relevant content is
largely *already in the transcript*, so concatenating it buys little; the permuted vector carries
the same label-marginal information but is now conditionally independent of text, so it is
non-redundant and the head can use it. **Audio on HateMM is redundant, not orthogonal.** That is a
stronger and more specific statement than "the audio prior is weak", and it is the reason both the
marginal and the conditional framings fail on the same data.

Hedge, stated: this is a between-condition comparison of two OOF AUCs over 3 seeds, not a frozen
endpoint with its own null, and it is single-dataset. It is a mechanism hypothesis worth one
sentence in any future audio discussion, not a result.

---

## Non-gating sensitivities (all reported, none can move the verdict)

| sensitivity | arm P | arm C |
|---|---|---|
| band = middle 20 % (mean Δ_int) | −0.0371 | −0.0364 |
| band = middle 40 % (mean Δ_int) | −0.0452 | −0.0482 |
| macro-F1 on the band instead of AUC (mean Δ) | −0.0216 | +0.0005 |
| **all 744 rows, empty transcripts kept in** (mean Δ_int) | **+0.0084**, 2/3 seeds > 0 | −0.0189, 1/3 seeds > 0 |

**The last row is the one worth reading.** The freeze excluded the 39 whitespace-transcript rows in
advance, on the argument that their CLIP text vector is a single constant point that is 92.3 %
non-hate against a 40.1 % base rate, so a text×prosody interaction on them degenerates into a pure
prosody main effect and "keeping them could only manufacture a GO". Keeping them moves arm P from
−0.0436 to **+0.0084 with 2/3 seeds positive** — most of the way to the bar, from a population of
39 rows with a constant text vector. The pre-registered exclusion is what stopped that artifact from
being reported as a signal. It still would not have passed (the bar is +0.010 with 3/3 seeds), so no
verdict was at stake, but the mechanism the freeze named is visible in the numbers.

---

## Discipline

- **Zero test-set contact.** Guard armed; 4 touched paths recorded in the JSON with input SHA256s;
  the guard was independently verified to HALT on `data/gt/HateMM/test.jsonl` and on
  `test_seen_..._HF.pt` before launch.
- **Rules frozen before implementation** and transcribed unedited above. No threshold, dimension,
  metric or arm was changed at any point.
- **Blind design.** No candidate endpoint was computed on real data during design or implementation;
  validation used a synthetic positive control and a label-permuted negative control only.
  (The synthetic control's generator was revised once — from isotropic noise to a low-rank factor
  model — because the first version measured PCA truncation on isotropic features rather than the
  pipeline's sensitivity. That is a smoke-test fix, made before the official run, touching no rule
  and no real-data quantity.)
- **One submission.** Background, `logging/runs/c8_prosody/run.{log,pid}`, 874 s, no re-run.

## What this pilot cannot establish

It cannot show that some *other* operator parameterisation (low-rank FiLM at the encoder, per-segment
conditioning, a learned gate trained end-to-end) would also fail — it tests a bilinear gate on frozen
pooled features, which is the cheapest member of the family and the one §7.4 named. It cannot
generalise beyond HateMM: MHC / MHC_zh have no CLAP cache (§7.4), and their Whisper features are
pre-registered as unfit for this test. It supports no effect-size claim — 705 rows, 211 in the band,
seeds are CV-partition replicates, not resampling of the population. And it produces no test-set
number of any kind.

What it does establish is that on the one dataset where this project has both a prosodic and a
general-audio cache, **the conditional estimand §7.4 argued Phase 1 had missed is not merely absent —
it is negative where the argument predicted it would be largest**, and the audio channel's failure
mode is redundancy with the transcript rather than weakness.
