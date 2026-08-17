# R11-UNION result — the CAT ∪ LL union is not purchasable

Frozen design: `idea-stage/R11_UNION_FREEZE.md`, committed at **`b6847d4`** before any
arm metric on the R11 seed range existed. Deviation filed with the freeze:
`idea-stage/R11_UNION_DEVIATION_D1.md`. Single submission
`idea-stage/r11_union/run_all.sh`.

**Cost: ¥0.00 (no API). 630 head-training runs (420 MHC-ZH + 210 HateMM), 0 failures,
75 min wall on the local RTX 5090. No new extraction — three of the four caches are the
R10-COMBO caches re-verified by sha256, the fourth is built from the banked `-tp` blocks.
Zero test-label tuning: every epoch rule, arm definition, blend weight, reliability table
and λ used train or dev only.**

---

## Headline

1. **Frozen verdict: none of the five mechanisms stands.** Not one beats `CAT` by the
   required +0.005 on either dataset. Every one of them is **at or below `CAT` on both**.
   The pre-committed conclusion applies: *the union is not purchasable by these
   mechanisms; `CAT` alone remains the entry, and the ~+0.05 headroom R10-COMBO
   identified stays unclaimed.*
2. **The decisive number is the control, not the candidate.** Averaging two `CAT` heads
   that differ only in seed (`ECTL`) is **better** than averaging `CAT` with `LL`:
   `AVG − ECTL` = **−0.0075 [−0.0131,−0.0022]** on MHC-ZH and −0.0031 on HateMM. Whatever
   decision-level averaging buys is variance reduction; mixing in the layer axis costs
   more than duplicating the token-axis head.
3. **The dev signal cannot even choose a side.** `WAVG`'s dev-fitted weight lands on
   **w = 1.00 on MHC-ZH** (pure `CAT`) and **w = 0.00 on HateMM** (pure `LL`) — opposite
   corners of the same grid. `SEL` degenerates to `LL` on HateMM the same way. With 78 /
   107 dev items there is no stable basis for a blend, and this is the mechanism-level
   reason the decision-level family fails.
4. **Churn-anchored training is neutral at best and fails its own control.** The
   union-targeted `ANCL` (layer-axis teacher distilled into a token-axis head) is
   −0.0021 / −0.0008 vs `CAT` — indistinguishable, no gain. Neither anchor arm beats the
   hard-label anchor `LBL`: `ANCA − LBL` = −0.0082 [−0.0145,−0.0017] on MHC-ZH. Soft
   out-of-fold teacher knowledge is not worth more than simply weighting the labels more.
5. **`CAT` replicates a third time, on a third disjoint seed range**: **+0.0171** MHC-ZH
   (26/30) and **+0.0114** HateMM (13/15) over `A0`, both CIs excluding zero. **`LL` fails
   to replicate a second time** (+0.0046 with the CI touching zero / −0.0006), and
   `CAT − LL` is +0.0125 / +0.0120 with both CIs excluding zero.
6. **The `PC0` follow-up resolves negatively.** Mean-centring alone (`MC`) reproduces
   three quarters of `PC0`'s MHC-ZH gain (+0.0187 vs +0.0252) **and the same failure on
   HateMM** (−0.0044 vs −0.0140), with a dev contrast of −0.0165 (CI excluding zero) on
   MHC-ZH. It is the same selection-rule-bound artefact, not a method.

---

## 1. The table

P1 = epoch `argmax_{e≥5}` dev macro-F1, test macro-F1 @0.5. MHC-ZH 30 seeds 700–729,
HateMM 15 seeds 700–714 (`CATB` at 50700+), all fresh ranges. Controls in **bold**.

| arm | what it is | MHC-ZH P1 | HateMM P1 |
|---|---|---|---|
| **A0** | deployed readout | 0.8019 ± 0.0168 | 0.8669 ± 0.0112 |
| **LL** | layer axis | 0.8065 ± 0.0289 | 0.8663 ± 0.0115 |
| **CAT** | token axis — the reference | **0.8189 ± 0.0107** | **0.8783 ± 0.0058** |
| **CATB** | `CAT`, different seed | 0.8170 ± 0.0152 | 0.8755 ± 0.0097 |
| **ECTL** | ½·`CAT` + ½·`CATB` (ensembling control) | *0.8221 ± 0.0090* | 0.8776 ± 0.0068 |
| **LBL** | hard-label anchor (anchor control) | 0.8213 ± 0.0093 | 0.8746 ± 0.0081 |
| AVG | ½·`CAT` + ½·`LL` | 0.8146 ± 0.0136 | 0.8745 ± 0.0115 |
| WAVG | dev-fitted convex blend | 0.8189 ± 0.0107 | 0.8663 ± 0.0115 |
| SEL | dev-fitted reliability vote | 0.8131 ± 0.0151 | 0.8663 ± 0.0115 |
| ANCA | anchored to OOF deployed-readout teacher | 0.8130 ± 0.0142 | 0.8743 ± 0.0096 |
| ANCL | anchored to OOF layer-axis teacher | 0.8168 ± 0.0147 | 0.8775 ± 0.0078 |
| *MC* | mean-centred `a28`, full width (side record) | *0.8206 ± 0.0198* | 0.8625 ± 0.0055 |

### 1.1 The judgement contrasts (frozen list)

Paired mean ± paired-bootstrap 95 % CI, B = 20000, seed 20260817.

| contrast | MHC-ZH P1 | HateMM P1 | verdict |
|---|---|---|---|
| AVG − CAT | −0.0043 [−0.0103,+0.0013] 12/30 | −0.0039 [−0.0088,+0.0002] 6/15 | does not stand |
| WAVG − CAT | +0.0000 [+0.0000,+0.0000] 0/30 | −0.0120 [−0.0175,−0.0070] 2/15 | does not stand |
| SEL − CAT | −0.0059 [−0.0102,−0.0020] 4/30 | −0.0120 [−0.0175,−0.0071] 2/15 | does not stand |
| ANCA − CAT | −0.0059 [−0.0130,+0.0017] 10/30 | −0.0041 [−0.0098,+0.0014] 5/15 | does not stand |
| ANCL − CAT | −0.0021 [−0.0088,+0.0040] 10/30 | −0.0008 [−0.0043,+0.0026] 6/15 | does not stand |

Clause (a) — a +0.005 gain over `CAT` with the CI excluding zero — is never satisfied on
either dataset, so the existential in freeze §5 is never reached and the control clauses
are never load-bearing. They are reported anyway because they are the informative part:

| control contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| **AVG − ECTL** | **−0.0075 [−0.0131,−0.0022]** 9/30 | −0.0031 [−0.0097,+0.0028] 7/15 |
| WAVG − ECTL | −0.0031 [−0.0071,+0.0009] 8/30 | **−0.0112 [−0.0180,−0.0046]** 2/15 |
| **SEL − ECTL** | **−0.0090 [−0.0151,−0.0032]** 9/30 | **−0.0112 [−0.0179,−0.0046]** 2/15 |
| **ANCA − LBL** | **−0.0082 [−0.0145,−0.0017]** 8/30 | −0.0004 [−0.0068,+0.0061] 8/15 |
| ANCL − LBL | −0.0044 [−0.0102,+0.0011] 9/30 | +0.0029 [−0.0024,+0.0083] 9/15 |

Mechanical application of the frozen rule: `idea-stage/r11_union/verdict.py`,
`idea-stage/r11_union/verdict.json`. All five candidates: `STANDS=False`.

### 1.2 Reference contrasts

| contrast | MHC-ZH P1 | HateMM P1 |
|---|---|---|
| **CAT − A0** | **+0.0171 [+0.0109,+0.0232]** 26/30 | **+0.0114 [+0.0052,+0.0190]** 13/15 |
| LL − A0 | +0.0046 [−0.0078,+0.0165] 20/30 | −0.0006 [−0.0085,+0.0076] 8/15 |
| **CAT − LL** | **+0.0125 [+0.0018,+0.0246]** 19/30 | **+0.0120 [+0.0071,+0.0175]** 13/15 |
| ECTL − CAT | +0.0031 [−0.0008,+0.0071] 15/30 | −0.0008 [−0.0039,+0.0019] 6/15 |
| LBL − CAT | +0.0023 [−0.0018,+0.0066] 11/30 | −0.0037 [−0.0066,−0.0011] 2/15 |
| AVG − A0 | +0.0127 [+0.0048,+0.0213] 21/30 | +0.0076 [−0.0008,+0.0160] 11/15 |
| *MC − A0* | *+0.0187 [+0.0113,+0.0251]* 28/30 | *−0.0044 [−0.0095,+0.0015]* 3/15 |
| *MC − CAT* | *+0.0017 [−0.0059,+0.0085]* 17/30 | *−0.0158 [−0.0193,−0.0119]* 0/15 |

---

## 2. Why each mechanism fails, in its own terms

### 2.1 Decision-level averaging: the union has negative purchase value

`ECTL` is the arm that settles this. It averages two `CAT` heads whose only difference is
the seed, so it carries **zero** extra information — it is pure variance reduction at
matched parameter count. It scores 0.8221 / 0.8776, i.e. `ECTL − CAT` = +0.0031 / −0.0008.

`AVG`, which replaces the second `CAT` head with the genuinely different `LL` head, scores
**below** it on both datasets, and on MHC-ZH the gap has a CI excluding zero. Substituting
real extra information for a redundant copy makes the ensemble **worse**. Whatever the
layer axis contributes at the decision level is outweighed by the errors it drags in.

### 2.2 The dev fit points in opposite directions on the two datasets

| dev-fitted quantity (P1) | MHC-ZH | HateMM |
|---|---|---|
| `WAVG` weight on `CAT` | **1.00** (pure `CAT`) | **0.00** (pure `LL`) |
| consequence | `WAVG ≡ CAT`, contrast exactly 0.0000, 0/30 seeds differ | `WAVG ≡ LL`, −0.0120 |
| `SEL` behaviour under disagreement | mixed, −0.0059 | collapses to `LL`, −0.0120 |

This is the clearest diagnosis in the pilot. The blend weight is a single scalar fitted on
2340 (MHC-ZH) / 1605 (HateMM) pooled dev decisions, and it still lands on opposite corners
of the grid. On MHC-ZH the dev fit correctly refuses to use `LL` at all — which is the
honest outcome and costs nothing. On HateMM it does the opposite and pays −0.0120 for it.
A mechanism whose one free parameter cannot be estimated consistently across two datasets
cannot buy a union.

### 2.3 Churn-anchored training: the R9 diagnosis survives the move to training time

R9 ANCHOR-INT died because repair and breakage sat on the same knob. Moving the anchor
into the training objective does not separate them:

| | MHC-ZH | HateMM |
|---|---|---|
| dev-selected λ, `ANCA` / `ANCL` | 1.0 / 0.1 | 0.1 / 0.1 |
| `ANCA − CAT` | −0.0059 | −0.0041 |
| `ANCL − CAT` | −0.0021 | −0.0008 |
| `ANCA − LBL` | **−0.0082**, CI excl. zero | −0.0004 |
| `ANCL − LBL` | −0.0044 | +0.0029 |

The teachers are real and honest — 5-fold out-of-fold logistic probes with train macro-F1
0.766 / 0.864 (`A0` teacher) and 0.810 / 0.875 (`LL` teacher), mean |q − y| 0.25–0.31, and
under 3 % of items with `q` within 0.05 of their label. They are not a restatement of the
labels. And yet **anchoring to them is no better than anchoring to the labels themselves**.
On MHC-ZH the hard-label control `LBL` (+0.0023 vs `CAT`) is the best arm of the anchor
family. The soft-target term behaves as a regulariser whose useful content is the label
signal, not the teacher's knowledge.

`ANCL` — the union-targeted version, where layer-axis knowledge is distilled into a
token-axis-input head so a single model could in principle carry both — is the closest
thing to a null result in the grid: −0.0021 and −0.0008, both CIs straddling zero. It
neither helps nor hurts. The layer axis does not transfer through the decision surface.

### 2.4 REAUDIT_NCA check: the anchor family is dev-positive and test-negative

The frozen dev-side panel, P1, vs `CAT`:

| arm | MHC-ZH dev | MHC-ZH test | HateMM dev | HateMM test |
|---|---|---|---|---|
| ANCA | **+0.0139 [+0.0096,+0.0180]** | −0.0059 | −0.0002 | −0.0041 |
| ANCL | **+0.0065 [+0.0026,+0.0104]** | −0.0021 | −0.0012 | −0.0008 |
| AVG | −0.0230 [−0.0272,−0.0189] | −0.0043 | −0.0023 | −0.0039 |
| ECTL | −0.0053 [−0.0085,−0.0026] | +0.0031 | −0.0019 | −0.0008 |

On MHC-ZH both anchor arms **fit dev better than `CAT` with the CI excluding zero and
score worse on test**. That is an overfitting signature, and it means the dev-selected λ
is selecting the wrong λ: the criterion used to choose the hyper-parameter is the one the
mechanism corrupts. Any future revival of this family needs a selection signal that is not
dev macro-F1.

Conversely `ECTL` and `AVG` are dev-**negative** and test-neutral-to-positive — the
flat-dev-curve pattern R10-COMBO flagged on `LL`. Neither direction supports a claim.

### 2.5 P2 disagrees with P1 on MHC-ZH, and that is itself the finding

Under P2 (fixed last epoch, no epoch selection) several arms flip positive on MHC-ZH:
`AVG − CAT` = +0.0103 [+0.0037,+0.0190], `ANCA − CAT` = +0.0087, `ANCL − CAT` = +0.0077,
`ECTL − CAT` = +0.0062. On HateMM they do not (−0.0047 / +0.0014 / +0.0015 / +0.0016).

Reading, stated as interpretation: averaging and anchoring both mainly buy **stability
across epochs** — they stop the head degrading late in training. The P1 protocol already
buys that for free by selecting the epoch on dev, so the mechanisms have nothing left to
add. This does not change the verdict (P1 is the primary protocol and was frozen), but it
is the honest account of where their apparent value comes from.

---

## 3. Union accounting — what fraction of the union each mechanism ate

P1, test, per seed, from the dumped per-item logits. `union pool` = A0 errors that `CAT` or
`LL` gets right (the R10-COMBO quantity, reproduced here on the new seeds: 9.03 items of
149 on MHC-ZH, 5.80 of 215 on HateMM). `retained` = fraction of that pool the arm gets
right. `new` = A0-correct items the arm newly breaks. `net` = errors saved against A0.

| arm | MHC-ZH retained | new | net | HateMM retained | new | net |
|---|---|---|---|---|---|---|
| **CAT** | **0.650** | 4.07 | 1.67 | **0.822** | 2.40 | 2.33 |
| LL | 0.640 | 5.07 | 1.10 | 0.650 | 3.93 | 0.00 |
| AVG | 0.621 | 4.03 | 1.83 | 0.689 | 2.40 | 1.60 |
| WAVG | 0.650 | 4.07 | 1.67 | 0.650 | 3.93 | 0.00 |
| SEL | 0.597 | 4.57 | 0.83 | 0.650 | 3.93 | 0.00 |
| ECTL | 0.627 | **3.70** | **2.17** | 0.714 | 2.40 | 2.20 |
| ANCA | 0.489 | **3.67** | 1.40 | 0.626 | **2.27** | 1.53 |
| ANCL | 0.560 | 3.97 | 1.50 | 0.661 | **2.00** | 2.13 |
| LBL | 0.611 | 3.73 | 1.97 | 0.700 | 2.73 | 1.67 |
| MC | 0.541 | 2.83 | 3.17 | 0.280 | 3.47 | −0.73 |

**The answer to the question the pilot was set:** no mechanism ate more of the union than
`CAT` already eats on its own. `CAT` alone retains **65 %** of the union on MHC-ZH and
**82 %** on HateMM; every mechanism retains the same or less. The unbought headroom is not
in the retention column at all — it is in the **breakage** column, and every mechanism
trades breakage against retention roughly one for one:

- `ECTL` cuts new errors 4.07 → 3.70 and gives up retention 0.650 → 0.627. Net +0.50 items.
- `ANCA` cuts new errors 4.07 → 3.67 and gives up retention 0.650 → 0.489. Net −0.27 items.
- `SEL` gives up on both columns at once.

So R10-COMBO's "+0.05 macro-F1 of headroom exists in principle" survives as a statement
about an upper bound, and this pilot narrows where it lives: **the fixes are already
mostly captured; what is unpurchased is the ~4 items per seed that the token axis breaks,
and none of averaging, weighting, reliability-gating or output distillation reduces that
without giving back an equal number of fixes.**

---

## 4. `MC` — the R10-COMBO §5 follow-up, recorded separately

R10-COMBO's `PC0` (PCA-512 of `a28`) was +0.0252 on MHC-ZH and −0.0140 on HateMM, and its
§5 named one cheap question: is the mean-centring doing the work, with no compression at
all? Answer: **largely yes, and it inherits the same failure.**

| | MHC-ZH | HateMM |
|---|---|---|
| `PC0 − A0` (R10-COMBO) | +0.0252 | −0.0140 |
| **`MC − A0` (here)** | **+0.0187 [+0.0113,+0.0251]** 28/30 | **−0.0044 [−0.0095,+0.0015]** 3/15 |
| `MC − CAT` | +0.0017 [−0.0059,+0.0085] | −0.0158 [−0.0193,−0.0119] 0/15 |
| `MC` dev vs `A0` | **−0.0165 [−0.0195,−0.0135]** 0/30 | −0.0062 [−0.0095,−0.0029] |
| `MC` dev−test gap | −0.0022 (`A0`: +0.0330) | −0.0119 (`A0`: −0.0101) |

Removing the corpus-mean direction of the deployed block, with no dimension reduction at
all, recovers three quarters of `PC0`'s MHC-ZH gain. It also **fails on HateMM exactly as
`PC0` did**, and its dev contrast is strongly negative with the CI excluding zero on both
datasets while its MHC-ZH test contrast is positive. The dev−test gap collapses from
+0.0330 to −0.0022: a flatter dev curve buying back epoch-selection winner's curse. This
is the `LL` / REAUDIT_NCA signature, not a method. **`MC` licenses nothing and the `PC0`
line is closed.**

---

## 5. What this changes on the ledger

- **`CAT` stands, third replication.** +0.0076 / +0.0101 (R10-TOKPOS, seeds 500+),
  +0.0100 / +0.0087 (R10-COMBO, seeds 600+), **+0.0171 / +0.0114 (here, seeds 700+)**,
  every time with both CIs excluding zero on both datasets. It remains the single entry
  from this line of work.
- **`LL` (the layer axis) should be considered dead.** Two consecutive failures to
  replicate on HateMM (−0.0043, −0.0006), a MHC-ZH gain whose CI touches zero, a
  dev-negative contrast on MHC-ZH with the CI excluding zero, and `CAT − LL` positive with
  CIs excluding zero on both datasets in both rounds.
- **The union question is answered and should not be re-opened at the decision or
  output-distillation level.** Five mechanisms, two families, both with their own matched
  controls, all negative. Any future attempt needs a mechanism that changes *which items
  get broken*, not one that re-weights or re-combines the same two decision surfaces.
- **`PC0` / mean-centring is closed** (§4).

---

## 6. Deviation D1 — how it discharged

`idea-stage/R11_UNION_DEVIATION_D1.md` recorded that an analyzer smoke test on relabelled
R10-COMBO logs left three identity mappings, so union-accounting **error counts** for
`AVG`, `SEL` and `ECTL` on MHC-ZH seeds 600–628 were seen before the freeze. The memo
committed to treating the decision-level family on MHC-ZH as one degree less blind.

It discharges cleanly: **all three of those arms came out negative against `CAT`**, so the
slip cannot have manufactured the result. The one thing the peek showed — `ECTL` close
behind `AVG` — reappeared on the fresh seeds in stronger form (`ECTL` now **ahead** of
`AVG`, CI excluding zero), and the control clause that catches it was written before the
peek. No definition, bar or protocol was changed after it. The process fix (analyzer smoke
directories must use a derangement of arm names) stands for future pilots.

---

## 7. Scope limits

- Same-machine, same-extraction-pass, head-level only. **No absolute number here is
  comparable to the project ledger** (those were A100); only within-table contrasts are
  results.
- MHC-ZH and HateMM only. Layers 28 and 24, spans `A0`/`TXT` only. One head
  (3-layer HateClipper-align MLP, BCE-only rung), one hyper-parameter set, one fusion mode.
- The anchor teachers are linear out-of-fold probes, not the MLP head itself. A
  non-linear or self-distilled teacher is untested; given `LBL` matches or beats both
  teachers, that is not an obvious revival route.
- `SEL` uses 3 confidence buckets fitted on 2340 / 1605 pooled dev decisions. A richer
  stacker is untested and, given `WAVG`'s one scalar could not be estimated consistently,
  is not indicated.
- +0.005 is ≈ 0.7 test items of 149 (MHC-ZH) and ≈ 1.1 of 215 (HateMM).

## 8. Artefacts

| what | where |
|---|---|
| freeze (`b6847d4`) | `idea-stage/R11_UNION_FREEZE.md` |
| deviation D1 | `idea-stage/R11_UNION_DEVIATION_D1.md` |
| cache + teacher builder, per-arm sha256 | `idea-stage/r11_union/build_r11.py`, `build_meta_{MHC_zh,HateMM}.json` |
| frozen teachers | `idea-stage/r11_union/teacher_{MHC_zh,HateMM}_{A0,LL,LBL}.json` |
| anchor loss | `src/model/loss.py::compute_anchor_loss`; `--anchor_logits/--lambda_anchor` |
| grid runner (fork, anchor fields only) | `idea-stage/r11_union/run_union_grid.sh` |
| single submission | `idea-stage/r11_union/run_all.sh` |
| read-out | `idea-stage/r11_union/analyze_union.py` → `{zh,hm}_union.json` |
| mechanical verdict | `idea-stage/r11_union/verdict.py` → `verdict.json` |
| logs | `logging/runs/r11_union/{run.log,run.pid,zh/,hm/}` |
| new caches | `data/CLIP_Embedding/{MHC_zh,HateMM}/{train,dev_seen,test_seen}_R11UN-MC.pt` |

**Belts that passed.** (i) At λ = 0 the new code path is an exact no-op: arm `A0` at seed
600 reproduces the R10-COMBO trainlog metric lines *and* the dumped per-item logits byte
for byte. (ii) The nine reused R10-COMBO caches re-hash to the R10-COMBO manifest exactly.
(iii) macro-F1 recomputed from the dumped per-item logits matches the trainlog with
**max abs diff exactly 0.0** on all 630 runs, both datasets. (iv) Test id order is
identical across all 14 arms, so the per-item pairing is valid.

Seeds 700–729 / 700–714 and 50700–50729 / 50700–50714 are disjoint from every previously
consumed range (0–119, 30–89, 100–129, 200–229, 300–329, 400–429, 500–529, 600–629,
41000–41029).
