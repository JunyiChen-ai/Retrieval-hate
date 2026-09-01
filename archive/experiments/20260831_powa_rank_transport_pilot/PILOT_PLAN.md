# Pilot plan: frozen-score temporal assignment from POWA

Frozen: 2026-08-31, after the independent novelty check and validation-only
failure diagnosis, before implementation, candidate training, or candidate test
inference.  Starting-point authority:
`runs/20260831_powa_starting_point/summary.json`.  Diagnostic authority:
`runs/20260831_powa_rank_transport_pilot/{hatemm,hateclipseg}_val_rank_diagnosis.json`.

## Fixed research question

Can a content-conditioned temporal rank head learn the missing within-video
ordering while a hard assignment layer preserves the complete per-video 1 fps
empirical score distribution of a frozen, same-corpus POWA model?

The only human annotation used for training is the same corpus's video label.
Negative-bag frames are called *label-certified benign under the negative-bag
assumption*, not ground-truth span labels.  No train set, span annotation, or
model is transferred between HateMM, MHC-EN, MHC-ZH, and HateClipSeg.  POWA's
existing train-only machine pseudo-targets remain part of the frozen starting
checkpoint and must be disclosed; this is not a claim of annotation-free
training.

## One core change

Let `a_v(t)` be the dense 1 fps score from the fixed corpus- and seed-specific
POWA checkpoint.  Let a trainable temporal head produce order logits `q_v(t)`
from frozen POWA multimodal representations.  At inference:

```
z_v = stable_sort(a_v)
P_v = stable_hard_argsort_assignment(q_v)
s_v = P_v^T z_v
```

Thus `s_v` is exactly a permutation of `a_v` on the final 1 fps grid.  The
method creates, scales, calibrates, or averages no score values.  Every
permutation-invariant per-video statistic (histogram, empirical quantiles,
mean, max, and top-k values) is identical to POWA.  The learned component only
changes score-to-time coupling.

This invariant does **not** guarantee that pooled frame AP or ROC is unchanged:
labels remain attached to timestamps and may receive different values.  Only
hard inference is claimed to preserve the multiset; a soft sorting relaxation
would not justify that claim.  The pilot therefore uses pairwise training of
`q` and a hard, non-differentiable assignment readout, not soft-OT leakage or a
straight-through estimator.

## Rank head and initialization

- The complete seed-matched POWA model is loaded and frozen.  Its existing
  output is the only source of score values.
- The rank head consumes detached POWA audio/visual contextual features,
  detached typed primitive logits and policy score, plus a learned projection
  of the aligned sentence feature.  A small masked temporal convolution and a
  scalar residual produce `q = logit(a) + r` on the training snippet grid.
- The final residual projection is exactly zero-initialized.  Before training,
  the assignment is therefore the POWA identity order (up to audited ties).
- The head is standard implementation machinery, not a novelty claim.  No
  second independently trained detector, ensemble, score blend, or post-hoc
  validation calibration is allowed.

## Position-sensitive train-only supervision

For each positive training bag, insert one continuous window drawn from a
negative training bag of the **same corpus**.  Use the existing frozen lengths:
12--36 snippet rows with a three-row boundary buffer.  The recipient is
unchanged and remains positive.  The donor interior is the only interval whose
benign status follows from a human video label.

The candidate loss is deliberately not a symmetric POWA MIL loss, because any
permutation-invariant bag readout is constant under assignment and cannot
identify `P_v`.

1. `L_order`: a softplus margin ranking loss requires the donor-interior top-k
   order logit to be below the top-k order logit of mapped recipient frames.
   Recipient top-k candidates are frozen from the original POWA order before
   insertion, so the learned head cannot choose an easy comparison set.
2. `L_stability`: Smooth-L1 consistency between centered original-recipient
   logits and their mapped composite logits, excluding the seam buffer.  It
   prevents the head from using a global composite shift as the ranking
   solution.
3. `L = L_order + 0.5 L_stability`.  Margin `1.0`, POWA top-k divisor `16`,
   AdamW `lr=2e-4`, weight decay `1e-4`, five epochs, batch 24, seed 234.  These
   constants are frozen from the POWA/previous insertion setup, not selected
   from candidate test results.

The intervention sampler must be deterministic in `(corpus, seed, epoch,
recipient, crop)` and log donor id/crop/window, insertion point, donor interior,
mapped recipient indices, and seam-excluded indices.  It may read only the
corpus's training split.

Pre-run feasibility amendment (2026-08-31, before any formal run): all arms use
the same named random draws, with the arm excluded from the draw key.  Donor
duration is first drawn in 12--36 rows subject to a label-independent recipient
length bound, then the donor is selected from same-corpus train videos long
enough for that draw.  Recipients shorter than the three-row boundary buffer
are identically ineligible in every arm.  If an eligible short recipient has
no seam-free position left for `L_stability`, it still contributes `L_order`,
`L_stability` is zero for that item, and the exception is counted and written
to the manifest.  The shifted-mask interval is a same-length valid contiguous
interval disjoint from the donor interior; it may include excluded donor
boundary rows because detecting that shortcut is the purpose of this control.
Negative-donor and shifted-mask composites must be byte-identical before their
supervision masks, and positive-donor shares donor duration, crop and insertion
draws wherever donor identity permits.

## Stage V: validation-only pilot and checkpoint rule

Run HateMM and HateClipSeg independently.  Epoch 0 (identity assignment) is
recorded as the exact POWA control.  A trained epoch is *feasible* only if its
validation pooled AP and ROC are each no more than `0.002` below epoch 0.  Among
feasible trained epochs, select maximum within-video macro ROC, then pooled AP,
then the earlier epoch.  If there is no feasible trained epoch, kill without
candidate test inference.

The core advances beyond validation only if **both** corpora satisfy:

- pooled AP and ROC feasibility above;
- within-video macro ROC improves by at least `0.020` over epoch 0;
- at least 55% of evaluable positive videos improve their per-video ROC;
- rank logits have at least 95% unique values on average, and reverse/random
  tie breaking changes within ROC by less than `0.002`;
- the sorted candidate and sorted POWA 1 fps score arrays agree exactly in
  float64 comparison for every video.

HateClipSeg additionally requires the `positive-frame fraction > 0.6`
validation stratum to improve by at least `0.015` and end above `0.50`.  This is
the preregistered check against another low-positive-fraction-only result.

## Required validation controls

Before candidate test inference, run these with identical data, initialization,
optimizer budget, and selection rule:

1. POWA identity assignment.
2. Uniform random within-video permutation and deterministic position-only
   orders (chronological, reverse chronological, edge-first, center-first).
3. Raw `q` ordering.  Apart from ties, its within-video ROC must equal the hard
   transported output's within ROC.
4. Direct additive output `sigmoid(logit(a)+r)` using the same trained head,
   showing whether the multiset constraint, rather than an extra head, avoids
   score-mass collapse.
5. No-insertion identity control.
6. Positive-donor control: the same operation but the donor comes from another
   positive training bag.
7. Shifted-mask/seam control: use the same negative insertion, but move the
   supervised interval off the donor interior while preserving its length and
   valid support.

The negative-donor core must beat the no-insertion, positive-donor, and
shifted-mask causal controls in within-video ROC on both corpora by at least
`0.010`; fixed position controls must not account for more than half the core
gain.  The raw-`q` and direct-additive readouts are expected to have the same
within ordering as transport; the additive readout must instead show worse
pooled feasibility or cross-video ordering for the constraint attribution to
hold.  Report original/new
Spearman and Kendall order correlation, Cross-AUC diagnostics, edge versus
interior results, positive-fraction strata, and improvement median/ratio.  A
control failure kills the mechanism claim even if aggregate performance rises.

## Stage P: one-shot test gate

Only after Stage V and the independent implementation/evaluation review pass,
run the selected seed-234 checkpoint once on test for HateMM and HateClipSeg.
The fixed evaluator is
`scripts/reproduction_baselines/eval_baseline_scores.py`; no evaluation logic
may be copied into this experiment.

The inference entry point defaults to validation.  Test mode must fail closed
unless the current `stage_v_summary.json` is PASS, its recorded independent
review is PASS, current plan/review/source/completion hashes match, and the
requested checkpoint is exactly the seed-234 negative-donor core selected for
that corpus.  Before test export, inference must recompute the complete Stage-V
summary from the current six run directories and require exact equality with
the stored summary; all six source snapshots must be identical and match the
current bound source.  Test output has one canonical path inside the selected
checkpoint, and an atomic permanent claim blocks alternate paths, concurrent
exports, and any second export (a crash after claiming requires manual integrity
audit rather than an automatic retry).  Every gated validation branch,
including tie and fixed-position
controls, is persisted and passed through the shared evaluator; the supervisor
cross-checks evaluator hashes and values against the training record before it
can issue Stage-V PASS.

The method advances only if all six strict test inequalities hold:

| corpus | pooled AP | pooled ROC | within-video ROC |
|---|---:|---:|---:|
| HateMM | `> .5938316` | `> .8161838` | `> .6315317` |
| HateClipSeg | `> .6193711` | `> .6050225` | `> .5619079` |

Any failed inequality kills this implementation without loss-weight,
temperature, epoch, tie-break, window, or checkpoint retuning from test.

## Later stages, only after Stage P

Run MHC-EN and MHC-ZH with the same frozen design and corpus-only train data.
Then run seeds 2025/234/3407 on all four corpora, the required controls,
paired-video bootstrap/statistical tests, and an independent deep novelty and
integrity audit.  Final success requires the three-seed mean to strictly exceed
all 12 fixed SOTA gates; a metric-specific fallback branch, ensemble,
calibration, corpus mixing, or test-selected configuration is prohibited.

## Falsification and claim boundary

The mechanism is falsified if it cannot learn non-positional ordering on both
pilot corpora, if HCS remains near chance, if pooled feasibility fails, if a
control reproduces the gain, or if the output ceases to be an exact per-video
permutation of POWA values.  The only potential novelty claim is a learned,
content-conditioned temporal assignment under a frozen per-video empirical
score marginal.  Sorting, optimal transport, benign insertion, and score
multiset reassignment individually are prior art and will not be claimed.
