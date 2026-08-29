# R10-COMBO result — best combination of the banked span × layer blocks

Frozen design: `idea-stage/R10_COMBO_FREEZE.md`, committed at **`33580d4`** before any arm
metric existed. Single submission `idea-stage/r10_combo/run_all.sh`.

**Cost: ¥0.00 (no API). 450 head-training runs (300 MHC-ZH + 150 HateMM), 0 failures,
~49 min wall on the local RTX 5090. No new extraction — every vector came from the `-tp`
and `-ro_L{28,24}` caches R10 already banked. Zero test-label tuning: every epoch rule, arm
definition, PCA basis and threshold used train or dev only.**

---

## Headline

1. **Frozen verdict: none of the six candidates stands.** Not one beats the better of
   `L24⊕L28` and `CAT` by the required +0.005 on either dataset; every one of them is
   *worse* than `CAT` on both. The pre-committed fallback therefore applies: **the token
   axis and the layer axis are substitutes, and the default is the cheapest implementation
   — `CAT` (img L28 3584 + text `[A0_28‖TXT_28]` 7168, layer 28 only).**
2. **`CAT` replicated on a fresh seed range, on both datasets**: +0.0100 MHC-ZH (was
   +0.0076 on seeds 500–529) and +0.0087 HateMM (was +0.0101). Its dev-side contrast is
   positive on both, so it is not an epoch-selection artefact.
3. **The layer axis did not replicate.** `L24⊕L28 − A0` is **−0.0043 on HateMM** (5/15
   seeds) and only +0.0061 on MHC-ZH with a CI touching zero — and on MHC-ZH its **dev**
   contrast is **−0.0084 with the CI excluding zero** while its test contrast is positive.
   That is exactly the REAUDIT_NCA signature. `CAT` beats `L24⊕L28` head to head by +0.0129
   on HateMM. **The layer trick is the weaker of the two axes, not the stronger.**
4. **Redundancy is real but partial.** At the decision level, `CAT` and `L24⊕L28` share
   17.3 of ~23 test errors on MHC-ZH (Jaccard 0.605, 7.0× the independence null) and 22.2 of
   ~25 on HateMM (0.744, 11.3×). Of the A0 errors each axis fixes, **only about a third are
   the same items** (fix-set Jaccard 0.325 MHC-ZH / 0.341 HateMM, ≈2× independence). So the
   two axes are not reading literally the same items — but no combination in this grid
   captures the union.
5. **One unplanned, non-licensed observation.** The PCA-family *control* `PC0`
   (PCA-512 of the deployed readout alone, basis fit on train) is **+0.0252 over A0 and
   +0.0152 over CAT on MHC-ZH** (26/30 seeds, CI excluding zero) — and **−0.0140 vs A0 /
   −0.0226 vs CAT on HateMM** (0/15 seeds). It does not replicate. It is reported here as an
   observation with a named follow-up (§5), not as a result.

---

## 1. The table

P1 = epoch `argmax_{e≥5}` dev macro-F1, test macro-F1 @0.5. MHC-ZH 30 seeds 600–629,
HateMM 15 seeds 600–614, both fresh ranges. Controls in **bold**.

| arm | img | text | text dim | MHC-ZH P1 | HateMM P1 |
|---|---|---|---|---|---|
| **A0** (deployed) | `i28` | `a28` | 3584 | 0.8061 ± 0.0100 | 0.8693 ± 0.0101 |
| **LL** (layer axis) | `[i28‖i24]` | `[a28‖a24]` | 7168 | 0.8122 ± 0.0199 | 0.8650 ± 0.0116 |
| **CAT** (token axis) | `i28` | `[a28‖t28]` | 7168 | **0.8161 ± 0.0188** | **0.8779 ± 0.0066** |
| **PC0** (PCA control) | `i28` | `P512(a28)` | 512 | *0.8313 ± 0.0178* | 0.8553 ± 0.0088 |
| K1 low-rank fusion, both axes | `[i28‖i24]` | `P512([a28‖a24‖t28‖t24])` | 512 | 0.7942 ± 0.0307 | 0.8704 ± 0.0084 |
| K2 low-rank fusion, token axis | `i28` | `P512([a28‖t28])` | 512 | 0.8018 ± 0.0329 | 0.8558 ± 0.0088 |
| K3 layer axis on img + token axis on text | `[i28‖i24]` | `[a28‖t28]` | 7168 | 0.8153 ± 0.0210 | 0.8759 ± 0.0085 |
| K4 layer × span cross | `i28` | `[a28‖t24]` | 7168 | 0.8081 ± 0.0101 | 0.8730 ± 0.0029 |
| K5 additive, no extra width | `i28` | `n(a28+t28)` | 3584 | 0.8094 ± 0.0130 | 0.8745 ± 0.0068 |
| K6 all 14 blocks, compressed | `[i28‖i24]` | `P512(`7 spans × 2 layers`)` | 512 | 0.7659 ± 0.0224 | 0.8653 ± 0.0108 |

`REF` (the better of `LL`/`CAT`) is **`CAT` on both datasets**.

### 1.1 The judgement contrasts (frozen list, `idea-stage/reaudit/analyze_grid.py`)

Paired mean ± paired-bootstrap 95 % CI, B = 20000, seed 20260817.

| candidate | MHC-ZH vs LL | MHC-ZH vs CAT | HateMM vs LL | HateMM vs CAT | verdict |
|---|---|---|---|---|---|
| K1 | −0.0180 [−0.0319,−0.0036] | −0.0219 [−0.0349,−0.0082] | +0.0054 [−0.0019,+0.0138] | −0.0075 [−0.0129,−0.0019] | does not stand |
| K2 | −0.0104 [−0.0226,+0.0019] | −0.0143 [−0.0271,−0.0005] | −0.0093 [−0.0161,−0.0021] | −0.0222 [−0.0284,−0.0166] | does not stand |
| K3 | +0.0031 [−0.0075,+0.0138] | −0.0008 [−0.0112,+0.0095] | +0.0109 [+0.0042,+0.0177] | −0.0020 [−0.0064,+0.0021] | does not stand |
| K4 | −0.0041 [−0.0125,+0.0051] | −0.0080 [−0.0136,−0.0011] | +0.0080 [+0.0029,+0.0135] | −0.0050 [−0.0087,−0.0013] | does not stand |
| K5 | −0.0029 [−0.0104,+0.0057] | −0.0067 [−0.0150,+0.0023] | +0.0094 [+0.0041,+0.0152] | −0.0035 [−0.0074,+0.0008] | does not stand |
| K6 | −0.0464 [−0.0543,−0.0382] | −0.0503 [−0.0617,−0.0372] | +0.0003 [−0.0072,+0.0083] | −0.0126 [−0.0197,−0.0065] | does not stand |

PCA-family clause: K1/K2/K6 vs `PC0` = −0.0371 / −0.0295 / −0.0655 on MHC-ZH and
+0.0151 / +0.0004 / +0.0100 on HateMM. Only K1 and K6 clear it, and only on the dataset
where they fail the main clause anyway.

Mechanical application of the frozen rule: `idea-stage/r10_combo/verdict.py`. Every
candidate fails clause (2) — beating `CAT` — on both datasets, so the existential in §3.2
of the freeze is never satisfied and the harmlessness and demotion clauses are never
reached (K6 would additionally have been demoted on MHC-ZH: dev vs CAT −0.0101, CI
excluding zero).

`analyze_grid.py` prints its own aggregate "NOT REVIVED" over the 19 listed contrasts. As
in R10 leg 1 that aggregate is a conjunction over contrasts including deliberately-negative
ones and carries no meaning here; the per-contrast numbers above are the result.

### 1.2 Control contrasts (frozen list)

| contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| **CAT − A0** | **+0.0100 [+0.0019,+0.0169]** 26/30 | **+0.0087 [+0.0028,+0.0143]** 12/15 |
| LL − A0 | +0.0061 [−0.0004,+0.0122] 21/30 | **−0.0043** [−0.0110,+0.0020] 5/15 |
| CAT − LL | +0.0039 [−0.0069,+0.0143] 16/30 | **+0.0129 [+0.0063,+0.0194]** 13/15 |
| PC0 − A0 | **+0.0252 [+0.0180,+0.0321]** 26/30 | **−0.0140 [−0.0194,−0.0090]** 0/15 |

### 1.3 Descriptive contrasts (post-hoc, same runs, no judgement power)

`idea-stage/r10_combo/{zh,hm}_grid_descriptive.json`.

| contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| K3 − A0 | +0.0092 [−0.0000,+0.0165] 27/30 | +0.0066 [−0.0004,+0.0131] 11/15 |
| K5 − A0 | +0.0033 [−0.0028,+0.0091] 18/30 | +0.0052 [+0.0018,+0.0089] 9/15 |
| K4 − A0 | +0.0020 [−0.0028,+0.0072] | +0.0037 [−0.0014,+0.0091] |
| K1 − A0 | −0.0119 [−0.0238,+0.0004] | +0.0011 [−0.0053,+0.0084] |
| K2 − A0 | −0.0043 [−0.0168,+0.0087] | −0.0135 [−0.0205,−0.0066] |
| K6 − A0 | −0.0403 [−0.0490,−0.0316] | −0.0040 [−0.0119,+0.0036] |
| PC0 − CAT | +0.0152 [+0.0073,+0.0237] | −0.0226 [−0.0286,−0.0167] |

---

## 2. What the frozen rule returns, and what it means

**Nothing stands.** The pre-committed conclusion, quoted from freeze §3.2:

> the token-position axis and the layer axis are substitutes on this substrate; adopt the
> cheapest implementation as the default.

By the two-key ordering fixed in the freeze (fewest hidden-state layers, then smallest total
feature width), and by the "statistically indistinguishable from REF on both datasets"
escape clause:

- `CAT` = `REF` on both datasets, requires layer 28 only, 3584 + 7168 = 10752 features.
- `K3` is the only arm within 0.002 of `CAT` on MHC-ZH (−0.00077) but misses on HateMM by
  0.000028 (−0.002028 vs the −0.002 threshold), **and** it is strictly *more* expensive
  (two layers, 7168 + 7168 = 14336). Either way it cannot displace `CAT`.
- `K5` (additive, 3584 + 3584 = 7168, cheaper than `CAT`) is −0.0067 / −0.0035 vs `CAT`,
  outside the band.

**Default recorded: `CAT`.** Not `L24⊕L28` — and this pilot changes the standing of that
arm, see §2.1.

### 2.1 The layer axis is the weaker arm, and on MHC-ZH it is dev-negative

The REAUDIT_NCA check the freeze mandated (§3.2, "selection-rule-bound demotion") fires on a
**control**, not on a candidate:

| MHC-ZH | dev macro-F1 at the P1 epoch | test macro-F1 at the P1 epoch | dev−test gap | mean P1 epoch |
|---|---|---|---|---|
| A0 | 0.8385 | 0.8061 | 0.0324 | 20.4 |
| **LL** | **0.8301 (−0.0084 vs A0, CI [−0.0117,−0.0050])** | 0.8122 (+0.0061) | 0.0179 | 18.7 |
| **CAT** | **0.8449 (+0.0065 vs A0, CI [+0.0027,+0.0103])** | 0.8161 (+0.0100) | 0.0288 | 20.7 |

`L24⊕L28` **fits dev worse** than the deployed readout with the CI excluding zero, yet
scores higher on test. That is the pattern REAUDIT_NCA identified as selection-rule-bound:
a flatter dev curve buys back epoch-selection winner's curse (gap 0.0179 vs 0.0324) rather
than adding discriminative content. On HateMM, where the gaps are negative for both, the
layer axis simply loses: −0.0043 vs A0, −0.0129 vs CAT.

`CAT`'s gain is dev-positive on MHC-ZH (CI excluding zero) and dev-neutral on HateMM
(+0.0023, CI straddling zero). It is the only arm in the grid with a positive test contrast
against A0 on both datasets *and* no negative dev contrast.

### 2.2 Compression into the head does not recover the sum

R10 leg 2 lost −0.0097 when the token axis was appended to the layer axis, on a 14336-wide
text stream over 579 training rows. The natural hypothesis was **width**. It is wrong:

- `K1` compresses exactly those four blocks to 512 and is **worse still** (−0.0219 vs CAT,
  −0.0180 vs LL on MHC-ZH).
- `K6` compresses all 14 banked blocks to 512 and is the worst arm in the grid on MHC-ZH
  (0.7659, −0.0503 vs CAT).
- `K3` shows the same thing without any PCA: putting the layer axis on the **img** stream
  and the token axis on the **text** stream — the one combination R10 never tried — lands
  on top of `CAT` (−0.0008 / −0.0020), i.e. the img-side layer axis contributes nothing
  once the token axis is present.

So the failure to stack is not a width or conditioning problem. Adding blocks to the input
does not add usable signal, whatever shape they are packed into.

---

## 3. Diagnostic 1 — error-set overlap (no verdict power)

`idea-stage/r10_combo/diag_errors.py`. Belt: macro-F1 recomputed from the dumped per-item
head logits matched the trainlog **exactly (max abs diff 0.0)** on all 450 runs, so these are
the exact prediction sets behind the reported numbers.

Mean Jaccard of test error sets at each arm's own P1 epoch, seed-paired, against an
independence null (random subsets of the same observed sizes):

| pair | MHC-ZH J (null, ratio) | HateMM J (null, ratio) |
|---|---|---|
| A0 \| LL | 0.639 (0.090, 7.1×) | 0.799 (0.068, 11.7×) |
| A0 \| CAT | 0.677 (0.089, 7.6×) | 0.746 (0.065, 11.5×) |
| **LL \| CAT** | **0.605 (0.086, 7.0×)** | **0.744 (0.066, 11.3×)** |
| CAT \| K3 | 0.787 (0.085, 9.2×) | 0.833 (0.063, 13.2×) |
| CAT \| K5 | 0.702 (0.087, 8.1×) | 0.817 (0.064, 12.8×) |

Mean error counts: MHC-ZH A0 25.1 / LL 23.4 / CAT 23.1 of 149 test items;
HateMM A0 26.7 / LL 27.4 / CAT 24.9 of 215.

Decomposed against A0 (mean items per seed):

| | MHC-ZH | HateMM |
|---|---|---|
| A0 errors **fixed by CAT** | 5.9 | 4.7 |
| A0 errors **fixed by LL** | 6.5 | 2.9 |
| **fixed by both** | **3.0** | **1.9** |
| fixed by either | 9.3 | 5.7 |
| Jaccard of the two fix-sets | **0.325** | **0.341** |
| new errors introduced by CAT | 3.9 | 2.9 |
| new errors introduced by LL | 4.8 | 3.5 |

**The quantitative statement about "the same pool":** the two axes overlap about **twice as
much as independence would predict** (3.0 shared fixes observed vs 1.5 expected on MHC-ZH),
but two thirds of each axis's fixes are its own. They are *partially* redundant, not
identical. What kills the combination is the other column: each axis also **breaks** 3–5
items it did not break before, and the breakage does not cancel. Net effect per axis is
−2.0 (CAT) and −1.7 (LL) errors on MHC-ZH; a combination that captured the union of the
fixes without the breakage would be −9.3 errors ≈ +0.05 macro-F1. No arm in this grid gets
anywhere near that, so the headroom exists in principle and is not reachable by
concatenating or projecting these blocks at the head.

---

## 4. Diagnostic 2 — representation redundancy (no verdict power)

`idea-stage/r10_combo/diag_repr.py`, train split only, no labels. Blocks `a28`, `a24`,
`t28`, `t24`. Out-of-fold (5-fold) kernel-ridge R², because `n < d` makes an in-sample
linear map exact and in-sample partialling degenerate.

| | MHC-ZH | HateMM |
|---|---|---|
| R²(`a28` → `t28`) | +0.642 | +0.616 |
| R²(`a28` → `a24`) | +0.747 | +0.719 |
| R²(`a24` → `t28`) | +0.688 | +0.652 |
| R²(`t28` → `a24`) | +0.399 | +0.409 |
| CKA(`a28`,`a24`) | 0.888 | 0.885 |
| CKA(`a28`,`t28`) | 0.507 | 0.569 |
| CKA(`t28`,`t24`) | 0.945 | 0.950 |

**The transcript-span block is 62–64 % linearly predictable from the deployed
assistant-header block**, and the L24 block is 72–75 % predictable from it. Neither is a
copy (mean cosine `a28`·`t28` = 0.45, `a28`·`a24` = 0.24–0.25), but most of each is already
in the deployed readout.

Partialled — the quantity the substitution claim is about, on out-of-fold residuals:

| | MHC-ZH | HateMM |
|---|---|---|
| CKA( resid(`t28`\|`a28`), resid(`a24`\|`a28`) ) | 0.148 | 0.426 |
| row-permutation null (mean, p97.5) | 0.111, 0.122 | 0.090, 0.104 |
| Gaussian floor (frozen script) | 0.164 | 0.203 |
| residual energy fraction, `t28` given `a28` | 0.358 | 0.384 |
| residual energy fraction, `a24` given `a28` | 0.253 | 0.281 |

**Addition declared here, not in the freeze:** the frozen script's Gaussian floor does not
preserve the residuals' own spectra, so a row-permutation null was added post hoc
(`idea-stage/r10_combo/diag_repr_null.py`, 200 permutations). Against that proper null the
residuals are significantly but **weakly** aligned on MHC-ZH (0.148 vs 0.111) and
**substantially** aligned on HateMM (0.426 vs 0.090) — while raw block pairs sit at
0.48–0.95, so even HateMM's 0.43 is well below block-level similarity.

**Reading, stated as interpretation:** at the representation level the two axes' *new*
directions are far from identical — much less redundant than the label-level error overlap
suggests. The bottleneck is therefore not that the blocks encode the same thing; it is that
the head, on 579/744 training rows, cannot convert the extra directions into decisions. That
is consistent with §2.2 (compression does not help) and with §3 (each axis fixes and breaks
similar numbers of items).

---

## 5. `PC0` — an unplanned observation, explicitly not a result

`PC0` was declared in the freeze as a *control* for the PCA family: img `i28`, text =
PCA-512 of the deployed `a28`, basis fitted on train only. It is not a candidate and has no
route to STANDS under the frozen rule.

| | MHC-ZH | HateMM |
|---|---|---|
| PC0 − A0, P1 | **+0.0252 [+0.0180,+0.0321], 26/30** | **−0.0140 [−0.0194,−0.0090], 0/15** |
| PC0 − A0, P2 (final epoch) | +0.0439 [+0.0391,+0.0489], 30/30 | +0.0085 [+0.0028,+0.0144], 10/15 |
| PC0 − CAT, P1 | +0.0152 [+0.0073,+0.0237] | −0.0226 [−0.0286,−0.0167] |
| dev macro-F1 vs A0 | +0.0356 [+0.0321,+0.0393] | +0.0252 (per-arm: 0.8806 vs 0.8554) |
| dev−test gap | 0.0427 (A0: 0.0324) | 0.0253 (A0: −0.0138) |

It **improves dev on both datasets by ~+0.03** and improves test only on MHC-ZH; on HateMM
the same dev gain turns into a −0.014 test loss and the dev−test gap widens by 0.039. That
is an overfit signature, not a method. **It does not replicate and licenses nothing.**

Two things about it are worth one cheap, separately pre-registered follow-up:

1. The projection is **mean-centred**. Removing the corpus-mean direction of `a28` — an
   operation with no dimension reduction at all — may be doing the work. A one-arm test
   (`text = n(a28 − mean_train(a28))`, full 3584 width) separates centring from compression
   and costs ~6 minutes of head training.
2. The compression ratio was not swept. 512 was fixed by the train-split rank limit, not
   tuned.

Both belong in a new freeze. Neither is claimed here.

---

## 6. Answers to the three questions asked

**Q1 — what is the optimal combination?** Of the six enumerated, none. `CAT`
(A0 ⊕ transcript span, layer 28 only) remains the best configuration on both datasets and is
now also the cheapest one that is not dominated. Low-rank projection of the union (K1, K2,
K6) is actively harmful; layer × span crossing (K4) is neutral-to-harmful; putting the layer
axis on the img stream instead of the text stream (K3) is indistinguishable from `CAT` at
higher cost; additive fusion (K5) recovers roughly half of `CAT`'s gain at half the width.

**Q2 — where does the redundancy come from?** Not from the two axes reading identical
items: their fix-sets overlap at Jaccard 0.33, about twice independence. It comes from
(a) each axis's new content being largely predictable from the deployed readout in the first
place (out-of-fold R² 0.62–0.75), and (b) each axis breaking almost as many items as it
fixes, so the residual gains do not add. Error sets between `CAT` and `L24⊕L28` overlap at
Jaccard 0.605 (MHC-ZH) / 0.744 (HateMM), 7–11× the independence null and essentially as much
as either overlaps the deployed readout.

**Q3 — what goes on the ledger?** `CAT` as the default text readout, single layer.
`L24⊕L28` should be **demoted**: it does not replicate on HateMM (−0.0043) and its MHC-ZH
gain is dev-negative with the CI excluding zero.

---

## 7. Scope limits

- Same-machine, same-extraction-pass, head-level only. **No absolute number here is
  comparable to the project ledger** (those were extracted on A100); only the within-table
  contrasts are results, per R10 deviation D1.
- MHC-ZH and HateMM only. MHC-EN was not run; ImpliHateVid has no raw video left.
- Layers 28 and 24, spans `{A0, TXT, S1–S4, ALL}` only. One head (3-layer HateClipper-align
  MLP, triplet+BCE hybrid), one fusion mode, one hyperparameter set.
- +0.010 on MHC-ZH is ≈ 1.5 test items of 149; +0.009 on HateMM is ≈ 1.9 of 215. The
  replication across two datasets and two disjoint seed ranges is what carries the claim,
  not the magnitude.

## 8. Artefacts

| what | where |
|---|---|
| freeze (`33580d4`) | `idea-stage/R10_COMBO_FREEZE.md` |
| arm builder + per-arm sha256 | `idea-stage/r10_combo/build_combo.py`, `build_meta_{MHC_zh,HateMM}.json` |
| grid runner (fork, 1 added line) | `idea-stage/r10_combo/run_combo_grid.sh` |
| single submission | `idea-stage/r10_combo/run_all.sh` |
| judgement read-out | `idea-stage/r10_combo/{zh,hm}_grid.json` |
| mechanical verdict | `idea-stage/r10_combo/verdict.py` |
| dev/epoch panel | `idea-stage/r10_combo/{zh,hm}_devpanel.json` |
| error overlap | `idea-stage/r10_combo/{zh,hm}_errors.json` |
| representation redundancy | `idea-stage/r10_combo/{zh,hm}_repr.json`, `*_repr_null.json` |
| descriptive contrasts (post-hoc) | `idea-stage/r10_combo/{zh,hm}_grid_descriptive.json` |
| logs | `logging/runs/r10_combo/{run.log,run.pid,zh/,hm/}` |
| new caches | `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_R10CB-*.pt` |

Seeds 600–629 (MHC-ZH) and 600–614 (HateMM) are disjoint from every previously consumed
range (30–89, 100–129, 200–229, 300–329, 400–429, 500–529, 41000–41029).
