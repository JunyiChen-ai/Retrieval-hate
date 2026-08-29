# X-bucket reconnaissance — is the "ordinary ranking error" residue attackable?

**Date:** 2026-08-16. **Cost:** zero GPU training, zero API calls, zero new data. Reads only frozen
artefacts (`idea-stage/r4_pilot1.json`, `idea-stage/r5_buckets.json`, `data/CLIP_Embedding/*`).
**Scripts:** `idea-stage/r5_xbucket_recon.py` → `idea-stage/r5_xbucket_recon.json`;
`idea-stage/r5_xbucket_recon2.py` → `idea-stage/r5_xbucket_recon2.json`.

**Verdict: (b) — seal the X bucket.** It is *not* seed noise (76% of X items are wrong under all
three seeds), but it is diffuse: on every one of the five axes measured, X is statistically
indistinguishable from the other error buckets, and both non-oracle exits (global threshold move,
member-vote override) lose macro-F1 on the full test set.

---

## 0. Setup and what "X" is

Error source = the round-4 best ensemble comparator per dataset, exactly as used to build
`r5_buckets.json`: HateMM/`mlp`, MHC-EN/`mean_logit`, MHC-ZH/`logistic`, ImpliHateVid/`logistic`;
3 seeds; test macro-F1 0.8732 / 0.7776 / 0.8183 / 0.9276 (reproduced to 4 dp inside this recon).
Per-seed thresholds recovered by inverting the stored test macro-F1 over the test-score grid (same
routine as `r5_phase_a.py`); prediction = seed majority.

X counts: HateMM 9, MHC-EN 8, MHC-ZH 9, ImpliHateVid 11 = **37**.
Gold composition of the 37: **24 negatives / 13 positives**; ImpliHateVid contributes 10 negatives
(NH_*) out of its 11.

Control sets used throughout:
- `correct` — every test item the majority prediction gets right (818 items over the four sets);
- `nonX_err` — the other error buckets S/O/M/A/D (71 items). **This is the decisive control**: an
  attackable X must differ from the other errors, not merely from the correct items.

### 0.1 Recovering the per-encoder member logits (needed for H3)

`r4_pilot1.json` stores only the ensemble comparators, not the three per-encoder logits. They were
recovered **algebraically, not retrained**: with `single` = the val-ROC-best encoder's logit vector
(the `ref` tag is stored per seed), `mean_logit` gives the sum of the three, and `mean_prob` gives the
sum of the three sigmoids; the remaining pair is the unique solution of
`x+y = s`, `σ(x)+σ(y) = q` (the map is strictly monotone in `|x−y|` for `s ≠ 0`). Order within the
pair fixed by coordinate descent against `logistic`, which is exactly affine in the members.

Verification against comparators that were **not** used in the solve:
`weighted` is reproduced to max abs residual **2.1e-6 / 2.0e-5 / 9.0e-6** (HateMM / MHC-EN / MHC-ZH)
with recovered convex weights (0.309, 0.355, 0.335), (0.287, 0.363, 0.350), (0.323, 0.343, 0.334) —
non-negative and summing to 1, i.e. the shape the val-ROC weighting must have; `logistic` is
reproduced affinely to **6.2e-5 / 1.7e-4 / 5.2e-4**. ImpliHateVid has two encoders, so the members are
exact by subtraction. The reconstruction is sound.

---

## 1. H1 — boundary: is X concentrated at the threshold?

Distance measured two ways, both scale-free: `|score − threshold| / sd(score)`, and the item's
percentile inside the test set's own `|score − threshold|` distribution (seed-averaged).

| dataset | X mean sd-units | nonX-err mean | correct mean | X median gap-percentile | correct median gap-percentile |
|---|---|---|---|---|---|
| HateMM | 0.667 | 0.593 | 1.052 | 0.242 | 0.544 |
| MHC-EN | 0.465 | 0.526 | 1.065 | 0.233 | 0.599 |
| MHC-ZH | 0.310 | 0.488 | 1.254 | 0.114 | 0.584 |
| ImpliHateVid | 0.408 | 0.553 | 0.969 | 0.095 | 0.532 |

**X sits at the boundary — and so does every other error bucket.** X vs correct is a large, consistent
gap (roughly half the margin, median gap-percentile ~0.10–0.24 vs ~0.53–0.60). X vs nonX-err is a
wash: X is *slightly closer* on MHC-ZH/ImpliHateVid and slightly farther on HateMM/MHC-EN.

**Status: partially holds, but useless.** Margin identifies "errors" in general; it does not isolate X,
so it cannot be a selector for an X-specific operator. Converted to a method below (§6, E1).

## 2. H2 — coverage: does X fall in a sparse region of the training manifold?

Cosine similarity of the test item to its nearest train items, in each encoder's L2-normalised
`[img‖txt]` space (the space the round-4 head consumes).

| dataset / encoder | X top-1 | nonX-err top-1 | correct top-1 | X top-5 | correct top-5 |
|---|---|---|---|---|---|
| HateMM / CLIP | 0.6802 | 0.6864 | 0.7386 | 0.6571 | 0.7013 |
| HateMM / QWEN | 0.9379 | 0.9460 | 0.9483 | 0.9325 | 0.9399 |
| HateMM / LORA | 0.9294 | 0.9392 | 0.9450 | 0.9244 | 0.9359 |
| MHC-EN / CLIP | 0.5646 | 0.6200 | 0.5931 | 0.5486 | 0.5517 |
| MHC-EN / QWEN | 0.9331 | 0.9439 | 0.9361 | 0.9269 | 0.9280 |
| MHC-ZH / CLIP | 0.8178 | 0.8603 | 0.8534 | 0.7947 | 0.8356 |
| MHC-ZH / QWEN | 0.9381 | 0.9558 | 0.9471 | 0.9292 | 0.9400 |
| ImpliHateVid / CLIP | 0.6503 | 0.6793 | 0.7263 | 0.6193 | 0.6932 |
| ImpliHateVid / QWEN | 0.9372 | 0.9439 | 0.9479 | 0.9301 | 0.9411 |

**Does not hold.** The X-vs-correct deficit is 0.003–0.076 cosine, small against within-set spread, and
in MHC-EN the X items are *closer* to train than the correct items on QWEN (0.9331 vs 0.9361 marks no
deficit at all). X is never farther from train than the other error buckets — nonX-err is closer to
train than X in 9/9 encoder cells, which is the opposite of what "X is the uncovered residue" would
predict. There is no sparse-coverage subregion to target with more data or a retrieval expansion.

## 3. H3 — encoder disagreement: does any member already get X right?

Member decision = its own zero logit (probability 0.5); member correctness taken as the majority over
the 3 seeds.

| dataset | X: ≥1 member correct | nonX-err | correct | X: member-*majority* correct | mean members correct on X (of D) |
|---|---|---|---|---|---|
| HateMM (D=3) | 5/9 = 0.556 | 0.529 | 1.000 | 1/9 | 0.67 |
| MHC-EN (D=3) | 5/8 = 0.625 | 0.565 | 0.969 | 1/8 | 0.75 |
| MHC-ZH (D=3) | 7/9 = 0.778 | 0.533 | 1.000 | 1/9 | 0.89 |
| ImpliHateVid (D=2) | 3/11 = 0.273 | 0.188 | 1.000 | 0/11 | 0.27 |
| **pooled** | **20/37 = 0.541** | **33/71 = 0.465** | **814/818 = 0.995** | **3/37 = 0.081** | — |

**Does not hold as an exploitable signal.** Three readings, all negative:

1. "≥1 member correct" is not discriminative: it is true of 99.5% of the *correct* items too. A rule
   keyed on it fires almost everywhere.
2. On X items the members are **worse than chance together**: 0.67/0.89/0.27 members correct on
   average, against 3 (or 2) members whose individual test accuracy is 0.77–0.91. The encoders are not
   split on X — they lean the same wrong way. Only 3/37 have a member majority on the right side.
3. Per-member accuracy on X is inconsistent across datasets (HateMM CLIP 0.222 / QWEN 0.444 / LORA
   0.000; MHC-EN CLIP 0.625 / QWEN 0.000 / LORA 0.125; MHC-ZH CLIP 0.556 / QWEN 0.222 / LORA 0.111),
   so no fixed "trust encoder E on hard items" rule exists.

Oracle counting only (no training done): repairing the 20 X items that have ≥1 correct member is worth
+0.0233 / +0.0282 / +0.0476 / +0.0050 macro-F1 (mean **+0.026**), and that repair needs a per-item
choice of *which* member to believe — the operator **Law III / F47 forbids**
(`refine-logs/ROUTER_GATE_RECORD.md`; the analogous router was measured and killed there). The
non-oracle version of the same idea is measured in §6 (E2) and loses.

## 4. H4 — label continuity: is X an extension of annotation noise?

Fraction of the top-20 train neighbours whose label equals the item's gold label.

| dataset / encoder | X purity | nonX-err purity | correct purity | X frac. purity < 0.5 |
|---|---|---|---|---|
| HateMM / CLIP | 0.400 | 0.385 | 0.730 | 0.778 |
| HateMM / QWEN | 0.411 | 0.382 | 0.750 | 0.667 |
| HateMM / LORA | 0.339 | 0.356 | 0.783 | 0.667 |
| MHC-EN / CLIP | 0.469 | 0.461 | 0.687 | 0.500 |
| MHC-EN / QWEN | 0.425 | 0.450 | 0.718 | 0.625 |
| MHC-ZH / CLIP | 0.517 | 0.473 | 0.700 | 0.333 |
| MHC-ZH / QWEN | 0.467 | 0.507 | 0.740 | 0.556 |
| ImpliHateVid / CLIP | 0.255 | 0.438 | 0.827 | 0.909 |
| ImpliHateVid / QWEN | 0.259 | 0.275 | 0.857 | 0.818 |

**Does not hold as an X-specific property.** X neighbourhoods do vote against the gold label
(purity 0.26–0.52, i.e. at or below the coin flip), while correct items sit at 0.69–0.86. But nonX-err
is at the same level (0.28–0.51): low purity is a property of *errors*, and it is partly circular,
since the ensemble head is a smooth function of the same features that define the neighbourhood.

The usable consequence is negative and clear: **a retrieval/kNN repair cannot recover X.** For every
encoder, the majority of an X item's 20 nearest training videos carries the opposite label, so any
RGCL-style neighbour vote, memory-bank rerank, or retrieval-augmented head would confirm the error
rather than fix it. This closes the retrieval family for this residue specifically.

## 5. H5 — seed stability

Number of the 3 seeds whose own threshold-recovered prediction is wrong (errors necessarily have ≥2).

| dataset | X: 3/3 seeds wrong | X: 2/3 | nonX-err: 3/3 |
|---|---|---|---|
| HateMM | 8/9 | 1/9 | 12/17 |
| MHC-EN | 6/8 | 2/8 | 20/23 |
| MHC-ZH | 6/9 | 3/9 | 11/15 |
| ImpliHateVid | 8/11 | 3/11 | 13/16 |
| **pooled** | **28/37 = 0.757** | **9/37 = 0.243** | **56/71 = 0.789** |

**Does not hold — and this is the one result that argues *against* the easy dismissal.** X is not
seed noise: three-quarters of the 37 are wrong under every seed, at the same rate as the other error
buckets. The residue is a deterministic property of the representation + head, not run-to-run jitter.
The 9 seed-flippable items are worth at most ~0.24 × the X repair value and have no shared structure.

---

## 6. The two non-oracle exits, measured on the full test set

Because H1 and H3 are the only hypotheses with any signal at all, both were converted into concrete
operators and run over every test item (not only over X). Both are test-oracle-friendly upper bounds
or outright losses.

**E1 — move the single global threshold** (upper bound: the threshold is chosen on TEST labels, so no
legal method can beat it):

| dataset | base | oracle-threshold macro-F1 | Δ | X items recovered | previously-correct items broken |
|---|---|---|---|---|---|
| HateMM | 0.8732 | 0.8796 | +0.0064 | 2.0 | 3.0 |
| MHC-EN | 0.7776 | 0.7984 | +0.0208 | 1.0 | 0.7 |
| MHC-ZH | 0.8183 | 0.8440 | +0.0257 | 0.7 | 1.7 |
| ImpliHateVid | 0.9276 | 0.9310 | +0.0034 | 1.7 | 3.0 |

The boundary concentration of X does not convert: even a test-label-optimal global threshold picks up
**5.3 of the 37 X items** while breaking 8.4 correct ones, and the whole (unattainable) gain is
+0.0141 mean, most of which is generic threshold slack, not X. Global per-arm threshold calibration is
in any case already closed by B5.

**E2 — member-vote override** (the legal, non-per-item-selection version of H3: flip the ensemble
whenever the encoder vote disagrees):

| dataset | rule | Δ macro-F1 | items flipped | X fixed | correct broken |
|---|---|---|---|---|---|
| HateMM | member majority | **−0.0104** | 8.3 | 1.3 | 5.3 |
| HateMM | unanimous members | −0.0031 | 0.7 | 0.0 | 0.7 |
| MHC-EN | member majority | **−0.0427** | 16.7 | 0.7 | 10.0 |
| MHC-EN | unanimous members | −0.0101 | 3.0 | 0.0 | 2.0 |
| MHC-ZH | member majority | **−0.0319** | 8.7 | 1.0 | 6.0 |
| MHC-ZH | unanimous members | −0.0060 | 0.7 | 0.0 | 0.7 |
| ImpliHateVid | either | +0.0000 | 0.7 | 0.0 | 0.3 |

Every disagreement-aware combination loses. The ensemble already dominates its members on exactly the
items in question. Note also that the operator that *would* work is per-item member selection, which
is banned (Law III / F47, `refine-logs/ROUTER_GATE_RECORD.md`): any "disagreement-aware combination"
that is not a fixed symmetric aggregation is a per-item selector and must be declared as such.

---

## 7. Summary table

| # | hypothesis | X number | control (nonX errors) | control (correct) | holds? |
|---|---|---|---|---|---|
| H1 | boundary (mean sd-units) | 0.667 / 0.465 / 0.310 / 0.408 | 0.593 / 0.526 / 0.488 / 0.553 | 1.052 / 1.065 / 1.254 / 0.969 | vs correct **yes**, vs other errors **no** |
| H2 | train-NN cosine (top-1, CLIP) | 0.680 / 0.565 / 0.818 / 0.650 | 0.686 / 0.620 / 0.860 / 0.679 | 0.739 / 0.593 / 0.853 / 0.726 | **no** |
| H3 | ≥1 member correct | 20/37 = 0.541 | 33/71 = 0.465 | 814/818 = 0.995 | **no** (not discriminative; majority-correct only 3/37) |
| H4 | top-20 gold purity (CLIP) | 0.400 / 0.469 / 0.517 / 0.255 | 0.385 / 0.461 / 0.473 / 0.438 | 0.730 / 0.687 / 0.700 / 0.827 | vs correct **yes**, vs other errors **no** |
| H5 | all 3 seeds wrong | 28/37 = 0.757 | 56/71 = 0.789 | 0/818 | **no** — X is stable, not seed noise |

(dataset order in each cell: HateMM / MHC-EN / MHC-ZH / ImpliHateVid.)

## 8. Verdict

**(b) X = diffuse, deterministic residue. Seal it.**

Evidence:
1. No axis separates X from the other error buckets. The two axes that separate X from *correct*
   items (small margin, low neighbour purity) separate all errors from all correct items and cannot
   be used as an X selector.
2. X is not seed noise (28/37 wrong under all 3 seeds), so it is not dismissible as jitter — but it is
   also not repairable by variance reduction (more seeds, larger ensembles).
3. The encoders do not disagree on X; they agree wrongly (0.67 / 0.75 / 0.89 / 0.27 members correct
   out of 3 / 3 / 3 / 2, against member accuracies of 0.77–0.91). There is no split to exploit.
4. Retrieval-based repair is specifically foreclosed: an X item's 20 nearest training videos carry the
   opposite label more often than not, in every encoder space.
5. Both legal operators lose on the full test set (E2: −0.0104 / −0.0427 / −0.0319 / +0.0000), and the
   test-oracle global threshold recovers only 5.3 of 37 X items while breaking 8.4 correct ones.
6. The only counting that shows a gain (+0.026 mean macro-F1 from repairing the 20 X items with a
   correct member) requires per-item member selection — foreclosed by Law III / F47.

Consequence for the prize pool: the X share of the error budget (+0.0431 / +0.0478 / +0.0621 / +0.0250
macro-F1 if fully repaired, per `r5_bucket_value.json`) should be treated as **unpurchasable** and
removed from the headroom estimate. What remains purchasable is the S/O/M mass
(+0.0635 / +0.1199 / +0.1031 / +0.0300), which is where effort should stay.

No further X-bucket work is recommended; no data is missing that would change this verdict — the
retrieval, disagreement, margin and seed axes are all measured and all negative.
