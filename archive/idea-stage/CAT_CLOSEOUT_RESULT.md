# CAT close-out — result

Frozen design: `idea-stage/CAT_CLOSEOUT_FREEZE.md`, committed at **`ae286c9`** before any code
for these legs was written and before any metric on seeds 1300–1319 / 1400–1429 / 1500–1524
existed. Deviation: `idea-stage/CAT_CLOSEOUT_DEVIATION_D1.md` (CV inner-split RNG seed outside
numpy's domain; filed before any cell was built or any head trained). Single submission
`idea-stage/cat_closeout/run_all.sh`.

**Cost: ¥0.00 (no API, no cloud, no DashScope). 63 minutes wall on the local RTX 5090.
Two extraction passes (806 + 790 videos, 2 forwards each, 0 decode failures, 0 zero-vector
guards, 37 min) and 250 head-training runs (60 + 90 + 50 + 50), 0 failures. Zero test-label
tuning: every epoch rule, arm definition, fold, span and threshold used train or dev only.**

---

## Headline

| leg | question | frozen verdict | `CAT − A0`, P1 |
|---|---|---|---|
| **A** | does the effect survive re-extracting the features from raw video? | **REPRODUCED** | **+0.0136 [+0.0074, +0.0200]**, 16/20 seeds |
| **B** | does it transport to MHC-EN? | **DOES NOT TRANSPORT AT THE FROZEN BAR** | **−0.0149 [−0.0235, −0.0059]**, 8/30 seeds |
| **C** | is it carried by the one fixed split? (MHC-ZH, 5×5 CV) | **NOT CV-SUPPORTED** | **+0.0020 [−0.0046, +0.0085]**, 15/25 cells |
| **C** | same, HateMM | **NOT CV-SUPPORTED** | **+0.0051 [−0.0014, +0.0116]**, 16/25 cells |
| **D** | which items does it fix? | descriptive, no verdict | no identifiable repaired set: 2/0 and 3/1 items |

1. **The extraction is bit-identical across passes** — max abs diff exactly 0.0 on every span,
   every split, every row, with the traversal order reversed. There is no extraction-time
   randomness in this pipeline, so the four banked replications were never at risk from it.
2. **On MHC-EN the effect reverses and the CI excludes zero on the wrong side.** `CAT` is
   **1.5 points worse** than the deployed read-out there. This is the first dataset on which the
   token-position concatenation has been measured and failed, and it narrows the coverage
   statement permanently.
3. **Neither dataset's fixed-split gain survives repeated cross-validation on train+dev.** The
   MHC-ZH delta falls from +0.010…+0.017 to **+0.0020** and the HateMM delta from +0.009…+0.011 to
   **+0.0051**; both CIs include zero, and the across-cell SD is **0.017** on both — three to eight
   times the effect. Sampling variation over the training population is large enough to absorb the
   whole thing.
4. **There is no set of items `CAT` reliably repairs.** 114/149 (MHC-ZH) and 195/215 (HateMM) test
   items are decided identically by the two read-outs; the frozen FIXED/BROKEN sets contain 2/0 and
   3/1 items.

---

## Leg A — end-to-end extraction reproducibility (§13.6 item 2)

Fresh two-forward pass over all 806 MHC-ZH videos from raw video, **item order reversed within
each split** and permuted back by id, with model / LoRA / frame sampler / prompts / dtype /
attention / spans / layer all frozen. 0 decode failures, 0 zero-vector guards, 19 min.
LoRA `logging/lora/MHC_zh` sha256 `35a510f4…dd8` — the hash pinned in `MNTP_S1_RECORD.md` §1.1.
Belt A1 (the `A0` span equals the frozen `RO._pool_span(span="response")` on the same forward)
**max abs diff exactly 0.0**; Belt A2 (row order restored) passes on all three splits.
Span statistics reproduce R10 exactly: median sequence 1137 tokens, `TXT` span 124–128.

### A.1 The extraction is deterministic — bit-identical, not merely close

Against the banked R10 `-tp` cache (same GPU, opposite traversal order):

| split | span | max abs diff | rows bit-identical |
|---|---|---|---|
| train / dev_seen / test_seen | `A0` | **0.000e+00** | **100 %** |
| train / dev_seen / test_seen | `TXT` | **0.000e+00** | **100 %** |
| train / dev_seen / test_seen | `ALL` | **0.000e+00** | **100 %** |

Every span at layer 28, every split, every row. **Reversing the traversal order changes nothing**;
this pipeline has no extraction-time variance at all. Per the branch pre-committed in freeze §2.4,
this establishes that the four banked `CAT` replications share a cache carrying **no hidden
extraction randomness** — the effect is not an artefact of one lucky extraction draw.

The image stream is the one thing that did move, and for a known reason: the banked `-ro_L28`
`img_feats` were extracted on an **A100**, and against the fresh 5090 pass they read max abs diff
0.11–0.15, **cos mean 0.9921–0.9928, min 0.926**. That is the platform drift R10 deviation D1/D2
already documented, not nondeterminism. It also means **Leg A is the first fully-5090 `A0`/`CAT`
table in the project**: R10/R11/R12 all carried the A100 `img_feats` over unchanged, so the image
half of the feature vector had never been re-derived on this machine until now.

Geometry reproduces R10: cos(`A0`,`TXT`) = **0.452** (R10: 0.452). cos(`A0`,img) = 0.303 and
cos(`TXT`,img) = 0.592 — R10 reported 0.307 / 0.605 against the A100 img stream.

### A.2 The grid — MHC-ZH, 3 arms × 20 seeds (1300–1319), all vectors from this one pass

60 head-training runs, **0 failures**, 378 s. Belt E1 (macro-F1 from dumped logits vs trainlog)
**4.96e-05**.

| arm | text_feats | dim | **P1 test macro-F1** | P2 |
|---|---|---|---|---|
| **A0** | `n(A0_28)` | 3584 | 0.8058 ± 0.0113 | 0.8094 |
| **CAT** | `[n(A0_28) ‖ n(TXT_28)]` | 7168 | **0.8195 ± 0.0076** | 0.8056 |
| **RAND** | `[n(A0_28) ‖ n(A0_28·R)]` | 7168 | 0.8077 ± 0.0083 | 0.8058 |

| contrast | P1 mean | P1 95 % CI | seeds > 0 | P2 mean |
|---|---|---|---|---|
| **CAT − A0** | **+0.0136** | **[+0.0074, +0.0200]** | 16/20 | **−0.0039** |
| **CAT − RAND** | **+0.0118** | **[+0.0070, +0.0171]** | 16/20 | −0.0003 |
| RAND − A0 | +0.0018 | [−0.0037, +0.0077] | 9/20 | −0.0036 |

**Frozen verdict (freeze §2.4): REPRODUCED.** `CAT − A0` clears the +0.005 bar with the CI
excluding zero, and the matched-width control gains nothing.

**Reported honestly, and it is the one blemish:** under **P2** (last epoch, no dev selection)
`CAT − A0` is **−0.0039**, i.e. the sign flips. The freeze deliberately did **not** put P2 in this
leg's rule — R10 leg 1 did, and there P2 agreed (+0.0031 / +0.0102) — so the verdict stands as
written, but the honest statement is that on this seed range the gain is present at the
dev-selected epoch and absent at the last epoch.

This is not new to Leg A, and the earlier rounds did not surface it. Recomputing P2 for every
banked MHC-ZH `CAT − A0` range with this analyzer gives **+0.0031** (R10 leg 1, seeds 500–529),
**+0.0056** (R10-COMBO, 600–629), **−0.0017** (R11-UNION, 700–729 — the R11 document reported only
P1 for this contrast) and **−0.0039** (Leg A, 1300–1319). P1 is positive with a CI excluding zero
in all four; P2 is positive in two and negative in two. **The `CAT` gain on MHC-ZH is an effect at
the dev-selected epoch, not a uniform shift of the whole training curve.** On HateMM P2 has agreed
in sign every time (+0.0102 R10 leg 3, **+0.0065** R10-COMBO recomputed here).

Raw: `idea-stage/cat_closeout/out/legA_MHC_zh.json`,
`idea-stage/cat_closeout/build_meta_CCA_MHC_zh.json`,
`idea-stage/cat_closeout/extract_meta_MHC_zh.json`.

---

## Leg B — MHC-EN transport check (§13.6 item 4)

### B.1 Encoder provenance

The MHC-EN task LoRA was restored from Backblaze B2
(`b2:junyi-data/RGCL_video/logs/lora/MHC` → `logging/lora/MHC`, top-level files only), sha256
**`084883d769650b69feef528cb9f9d0348f9d0a6f62b6bf0c0ae5a19a4a2489c3`**, dated 2026-07-02 —
the adapter `refine-logs/B4_FORENSIC_RECON.md` records as the EN LoRA.

**No prior sha256 record for the EN adapter exists in this repository**, unlike the ZH and HateMM
adapters. The substitute is Belt B1, and it passes comfortably: the re-extracted deployed spans
reproduce the banked A100 MHC-EN cache at cos mean **0.9950–0.9966** (min 0.926) on the text side
and **0.9912–0.9916** (min 0.961) on the image side, against a floor of 0.95 mean / 0.90 min. A
wrong adapter, prompt or span reads 0.3–0.6, so the encoder identity is established behaviourally.
Belts A1 and A2 pass on all three splits; 790 videos, 0 failures, 18 min.

Span statistics: median sequence 1173–1184 tokens, `TXT` span **147–163** (vs 124–128 on MHC-ZH),
geometry cos(`A0`,`TXT`) = 0.420, cos(`TXT`,img) = 0.658.

### B.2 The grid — MHC-EN, 3 arms × 30 seeds (1400–1429)

90 runs, **0 failures**. Belt E1 **4.99e-05**. Both streams from this one extraction pass.

| arm | dim | **P1 test macro-F1** | P2 |
|---|---|---|---|
| **A0** | 3584 | **0.7060 ± 0.0184** | 0.7226 |
| **CAT** | 7168 | 0.6911 ± 0.0158 | 0.7126 |
| **RAND** | 7168 | 0.7077 ± 0.0201 | 0.7199 |

| contrast | P1 mean | P1 95 % CI | seeds > 0 | P2 mean |
|---|---|---|---|---|
| **CAT − A0** | **−0.0149** | **[−0.0235, −0.0059]** | 8/30 | −0.0100 |
| **CAT − RAND** | **−0.0165** | **[−0.0250, −0.0080]** | 10/30 | −0.0073 |
| RAND − A0 | +0.0016 | [−0.0089, +0.0121] | 17/30 | −0.0027 |

**Frozen verdict (freeze §3.3): DOES NOT TRANSPORT AT THE FROZEN BAR.** And it is not a null —
the sign reverses with the CI excluding zero, under both protocols, and `CAT` also loses to its own
matched-width random control. On MHC-EN, concatenating the transcript-position read-out **costs**
1.5 points of macro-F1.

Two things this is not. It is not a width artefact: `RAND` (same 7168 width, random projection of
`A0`) is statistically indistinguishable from `A0`, so the damage is specific to the `TXT` block,
not to doubling the feature dimension. It is not a broken extraction: Belt B1, the span statistics
and the geometry all sit exactly where the other two datasets sit.

**The honest reading.** MHC-EN is the smallest and hardest of the three (549 train items, `A0`
scores 0.706 against 0.806 on MHC-ZH and 0.876 on HateMM). Doubling the text width to 7168 on 549
training rows with a block that is a *worse* standalone summary is a plausible way to lose, and it
is what happened. No post-hoc rescue was attempted and none is offered — the freeze forbade
re-running, extending seeds or switching protocol on a miss, and none was done.

*(A cross-dataset explanation was looked for and is not available. The Leg D within-dataset pattern
— long transcripts are where `CAT` loses — does not order the three datasets: median test
transcript length is 178 tokens on HateMM (gains), 102 on MHC-EN (loses), 73 on MHC-ZH (gains).
The transport failure is recorded as a fact, not explained.)*

Raw: `idea-stage/cat_closeout/out/legB_MHC_EN.json`,
`idea-stage/cat_closeout/build_meta_CCB_MHC.json`, `extract_meta_MHC.json`.

---

## Leg C — repeated stratified cross-validation on train+dev (§13.6 item 5)

Zero new extraction: the pooled population is the train ∪ dev_seen rows of the banked `R10CB-A0` /
`R10CB-CAT` caches — the exact caches behind the R10-COMBO / R11 / R12 numbers. 5 repeats × 5
stratified folds. Each cell trains on an inner-train split, selects the P1 epoch on an inner-dev
split sized to the deployed 12 % ratio, and scores the held-out fold once. **Belt C1: the official
`test_seen` id set is disjoint from every fold — 0 overlap, both datasets.** Belt E1 4.99e-05 /
5.00e-05. 100 runs, 0 failures.

| dataset | pool (train+dev) | inner-train / inner-dev / eval per cell |
|---|---|---|
| MHC-ZH | 657 (208 positive) | 463 / 62 / 132 |
| HateMM | 851 (341 positive) | 595 / 85 / 171 |

| dataset | A0 | CAT | **CAT − A0 (25 cells)** | 95 % CI | cells > 0 | SD across cells | P2 |
|---|---|---|---|---|---|---|---|
| MHC-ZH | 0.8607 ± 0.0269 | 0.8628 ± 0.0291 | **+0.0020** | **[−0.0046, +0.0085]** | 15/25 | 0.0170 | −0.0003 |
| HateMM | 0.8738 ± 0.0298 | 0.8789 ± 0.0263 | **+0.0051** | **[−0.0014, +0.0116]** | 16/25 | 0.0170 | +0.0005 |

Per-repeat means of `CAT − A0` (5 folds each), which is where the instability is visible:

| dataset | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 |
|---|---|---|---|---|---|
| MHC-ZH | +0.0059 | +0.0058 | +0.0025 | +0.0137 | **−0.0178** |
| HateMM | −0.0023 | +0.0128 | +0.0072 | +0.0096 | −0.0019 |

**Frozen verdict (freeze §4.3): NOT CV-SUPPORTED on both datasets.** Both CIs include zero.
HateMM's mean does clear +0.005 (reported without gate status, as frozen); MHC-ZH's does not.

**What this does and does not say.** As fixed in §4.3 before the numbers existed, the CV *level*
is not comparable to the fixed-split level — the folds are drawn from train+dev and are visibly
easier (0.861 / 0.874 against 0.806 / 0.876 on the official splits), the training sets are smaller
and the dev sets are smaller. Only sign, CI and dispersion are interpreted. Those say: the direction
is positive on both datasets in 15/25 and 16/25 cells, but **the across-cell SD is 0.0170 on both,
three to eight times the effect being measured**, and one MHC-ZH repeat out of five lands at
−0.0178. A single 5-fold repeat would have "found" anything between −0.018 and +0.014 on MHC-ZH
depending on which shuffle it drew.

So the mitigation §13.6 item 5 asked for was run and it **did not come back clean**. The fixed-split
result is not shown to be split-specific luck, and it is not shown not to be either — at this
sample size (657 / 851 items) the design cannot separate a +0.005…+0.015 effect from resampling
noise. And, as §6 of the freeze states and the disclosure below repeats, none of this addresses
adaptive reuse of the test split.

Raw: `idea-stage/cat_closeout/out/legC_MHC_zh.json`, `legC_HateMM.json`,
`idea-stage/cat_closeout/cv_meta_*.json`. Cell caches were deleted after analysis.

---

## Leg D — per-item read-out audit (descriptive, no verdict)

Source: the per-item head logits R10-COMBO already dumped, arms `A0` and `CAT`, MHC-ZH seeds
600–629 and HateMM seeds 600–614, each seed read at its own P1 epoch selected from its own dev
curve. **Belt D1** (macro-F1 recomputed from the dumped logits vs the trainlog value) passes at
**4.96e-05 / 4.97e-05**, i.e. exactly the logs' 4-decimal rounding.

### D.1 The frozen sets are nearly empty, and that is the first finding

The freeze fixed `FIXED` = misclassified by `A0` in ≥ 2/3 of seeds **and** by `CAT` in ≤ 1/3, and
`BROKEN` = the mirror. Applied:

| dataset | n test | **FIXED** | **BROKEN** |
|---|---|---|---|
| MHC-ZH | 149 | **2** | **0** |
| HateMM | 215 | **3** | **1** |

There is no identifiable set of items that `CAT` reliably repairs. The decision-level picture:

| dataset | items with identical error rate across arms | both arms always right | both arms always wrong | \|gap\| ≥ 0.05 | ≥ 0.2 | ≥ 0.5 | total \|gap\| mass |
|---|---|---|---|---|---|---|---|
| MHC-ZH | 114 / 149 | 104 | 9 | 28 | 12 | 2 | 6.73 items |
| HateMM | 195 / 215 | 178 | 17 | 20 | 11 | 4 | 6.07 items |

Roughly 80–90 % of the test split is decided identically by the two read-outs. The whole effect
lives in ~20–28 contested items per dataset, and the *net* is about 1.9 items (MHC-ZH) and 1.7
items (HateMM) — consistent with the +0.010 / +0.009 macro-F1 the fixed split reports.

### D.2 Mechanical read-out audit

`idea-stage/cat_closeout/out/token_composition.json`, plus the per-item span statistics recorded
by the Leg A extraction.

- The deployed `A0` read-out is the mean over exactly **three** positions whose tokens are
  `['<|im_start|>', 'assistant', '\n']` — ids `[151644, 77091, 198]`. One special token and two
  ordinary ones. **The frozen extractor applies no special-token exclusion**; the special token is
  one third of the read-out.
- The `TXT` span is `[v_end, hdr)` = everything between the last `<|video_pad|>` and the assistant
  header. Its **constant scaffolding is 49 tokens** (`TEXT_INSTRUCTION` + `\nTitle: (none)\nTranscript: `)
  plus ~3 format tokens. The variable part is the transcript: median **73** tokens (MHC-ZH test),
  **102** (MHC-EN test), **178** (HateMM test).
- So on MHC-ZH the transcript is roughly 60 % of the `TXT` span and on HateMM roughly 78 % at the
  median — but on the 20 % of HateMM test items with ≤ 10 transcript tokens, the `TXT` span is
  **almost entirely instruction scaffolding**.
- Every span is L2-normalised inside the frozen extractor, so per-item feature norms are 1.0 by
  construction (measured: 0.9999998–1.0000002 for both `A0` and `TXT` on all 149 MHC-ZH test
  items). Pre-normalisation magnitude is not stored and the head never sees it, so "feature norms"
  carry no per-item information on this read-out.
- Per-item verification on the Leg A extraction, all 149 MHC-ZH test items: the `A0` span is
  **exactly 3 positions for every item** (min = max = 3), the `TXT` span runs 59–304 positions
  (median 124), the full sequence 584–1330 tokens (median 1137), and the degenerate
  `v_end ≥ hdr` guard **never fires** (0 items). So the read-out geometry is uniform across the
  split — there is no subset where the mask silently collapsed.

### D.3 Where the gain comes from — post-hoc, descriptive

**Declared:** the lists and breakdowns in this subsection use **no threshold** and were added
**after** observing that the frozen sets in D.1 are near-empty. They are a description, not
evidence, and they had no gate power over anything.

Mean per-item error-rate improvement (`A0` error rate − `CAT` error rate over seeds), by
transcript-length quartile:

| dataset | Q1 shortest | Q2 | Q3 | Q4 longest | items ≤ 20 chars |
|---|---|---|---|---|---|
| MHC-ZH (quartiles 54 / 105 / 188 chars) | −0.006 (n=39) | **+0.050** (n=36) | +0.016 (n=37) | **−0.005** (n=37) | n=2, 0.000 |
| HateMM (quartiles 80 / 743 / 1572 chars) | **+0.028** (n=54) | +0.011 (n=54) | +0.014 (n=53) | **−0.020** (n=54) | **+0.043** (n=42) |

By true label:

| dataset | non-hate (label 0) | hate (label 1) |
|---|---|---|
| MHC-ZH | **+0.031** (n=104) | **−0.027** (n=45) |
| HateMM | +0.010 (n=129) | +0.006 (n=86) |

**The plain reading, in two sentences.** The gain is *not* "long transcripts benefit" — it is the
opposite: on both datasets the longest-transcript quartile is the one where `CAT` **loses**, and on
HateMM the largest gain is on the items with a near-empty transcript (+0.043 on the 42 items with
≤ 20 characters of speech). Where the transcript is short or absent, the `TXT` span is mostly
instruction tokens whose hidden states have nonetheless attended over the whole video, so
concatenating it acts as a **second read-out of the video at different positions**, not as a
transcript read-out — which is also why `TXT` alone is far worse than `A0` (R10: −0.019 / −0.105)
while still carrying information `A0` does not (cos = 0.45).

The item-level reading agrees. On HateMM the two largest repairs are `hate_video_10` (18
characters of transcript, 1078 characters of on-screen text; `A0` wrong in 14/15 seeds, `CAT`
wrong in 0/15) and `hate_video_1` (**empty** transcript, 2319 characters of on-screen text; 14/15 →
1/15). The four largest regressions are all long, speech-dense items (`hate_video_300`, 1963
characters; `hate_video_87`, 2323; `hate_video_427`, 1073; `hate_video_295`, 1098).

On MHC-ZH the same trade appears as a class asymmetry: `CAT` improves the non-hate class by
+0.031 and **costs** the hate class −0.027. Reading the top repairs, they are non-hate videos whose
transcripts are dense in hate-adjacent vocabulary but are reporting or commentary — a Balenciaga
child-imagery news item (`BV1a14y1n712`, "儿童色情"/"性虐待" in the transcript, `A0` wrong 15/30 →
`CAT` 1/30), a physiognomy explainer (`BV1694y1N7xU`, 23/30 → 10/30), an asexuality clip
(`BV1MU4y1D7Ks`, 22/30 → 1/30). The largest regressions are hate-labelled videos where the hate is
carried by the literal speech (`BV1Vy4y1p7x2` "日常欺负女同学" 16/30 → 30/30; `BV12G4y1S7mN`
18/30 → 30/30). So on this dataset the extra read-out mostly suppresses **keyword-triggered false
positives** and pays for it in recall.

Raw: `idea-stage/cat_closeout/out/legD_MHC_zh.json`, `legD_HateMM.json`.

---

## Selection-history disclosure (§13.6 item 6)

> Roughly 90 candidate configurations were evaluated against these official test splits before
> `CAT` was selected as the surviving entry. The paired-bootstrap intervals reported for `CAT`
> — here and in `R10_TOKPOS_RESULT.md`, `R10_COMBO_RESULT.md`, `R11_UNION_RESULT.md` and
> `R12_ANCHOR_RESULT.md` — are **conditional descriptive intervals, not post-selection-valid
> confirmatory intervals**; they do not account for the selection. Leg C (repeated stratified
> cross-validation on train+dev) is offered as mitigation of split-specific luck only — **it does
> not correct adaptive reuse of the test split**, because the candidate being cross-validated was
> itself chosen with knowledge of test-split outcomes. No uncontaminated confirmatory population
> exists for this result on this workstation.

`ImpliHateVid` cannot be measured on this axis at all: `data/video/ImpliHateVid/All` no longer
exists on this workstation (only `_id2b2path.tsv` remains), so no extraction is possible and no
substitute or estimate is offered.

---

## The final coverage statement

Everything that is now known about `CAT`, stated once, with nothing rounded in its favour.

**Where it helps.** On **MHC-ZH** and **HateMM**, on the official fixed test splits, under the P1
protocol, concatenating the transcript-content read-out with the deployed assistant-header
read-out beats the deployed read-out by **+0.008 to +0.017** macro-F1 (MHC-ZH: +0.0076 / +0.0100 /
+0.0171 / +0.0136 on four disjoint seed ranges; HateMM: +0.0101 / +0.0087 / +0.0114 on three), with
paired-bootstrap CIs excluding zero every time, and by more than it beats a matched-width random
projection of the same vector. That is 1–2 test items out of 149 and 2 out of 215.

**Where it does not help.** On **MHC-EN** the same frozen configuration, extracted the same way with
the dataset's own task adapter, is **−0.0149 [−0.0235, −0.0059]** — worse than the deployed
read-out and worse than the width control, under both protocols. On **ImpliHateVid** it cannot be
measured. So the effect covers **two of the three measurable datasets, and reverses on the third.**

**Under what conditions.** Only at the dev-selected epoch: on MHC-ZH the last-epoch contrast is
+0.0031 / +0.0056 / −0.0017 / −0.0039 across the four ranges — positive twice, negative twice.
Only in a fixed-split design: repeated stratified CV on train+dev returns +0.0020 [−0.0046,
+0.0085] and +0.0051 [−0.0014, +0.0116], both CIs containing zero, with an across-cell SD of 0.017.
Only as a substitute for, not an addition to, the layer axis (R10 leg 2: stacking costs −0.0097).
And only on features whose extraction is fully deterministic — which Leg A now establishes at the
bit level, so the effect is at least not an extraction-draw artefact.

**What it is made of.** ~80–90 % of test items are decided identically by the two read-outs; the
whole effect is ~20–28 contested items per dataset with a net of about 1.9 (MHC-ZH) and 1.7
(HateMM). Within a dataset the gain concentrates on short or empty transcripts and is negative on
the longest-transcript quartile, and on MHC-ZH it trades +0.031 on the non-hate class for −0.027
on the hate class.

**The defensible final sentence, updated from §13.6.** `CAT` improves the paired 5090 baseline on
two reused benchmark splits and beats a matched-width random control there, the effect is
reproducible from raw inputs at bit-level determinism, **and it does not transport to the third
dataset, is not separable from resampling noise under cross-validation, and holds only at the
dev-selected epoch.** It remains an exploratory, crowded feature-design result with no
uncontaminated confirmatory population. Under the method-paper-only rule the honest close-out is
unchanged and is now better supported: **no publishable method emerged.**

None of the four legs was capable of upgrading that conclusion, as freeze §8 fixed in advance; the
outcomes narrow it. §13.6's item list is now fully discharged: items 2, 4 and 5 are done and
returned REPRODUCED / DOES NOT TRANSPORT / NOT CV-SUPPORTED, item 3's per-item audit is done, item
6's disclosure is written above, and item 1 was already standing practice.

---

## Reproducibility index

| artifact | path |
|---|---|
| freeze (commit `ae286c9`, before any code) | `idea-stage/CAT_CLOSEOUT_FREEZE.md` |
| deviation D1 (CV RNG seed range) | `idea-stage/CAT_CLOSEOUT_DEVIATION_D1.md` |
| two-forward extractor (both streams) | `idea-stage/cat_closeout/extract_cc.py` |
| arm builder + belt B1 + determinism table | `idea-stage/cat_closeout/build_cc_arms.py` |
| CV cell builder + belt C1 | `idea-stage/cat_closeout/build_cv.py` |
| analyzer (exact macro-F1 from dumped logits) + belt E1 | `idea-stage/cat_closeout/analyze_cc.py` |
| frozen verdict | `idea-stage/cat_closeout/verdict.py`, `out/verdict.json` |
| Leg D audit + belt D1 | `idea-stage/cat_closeout/audit_items.py` |
| single submission | `idea-stage/cat_closeout/run_all.sh` |
| raw results | `idea-stage/cat_closeout/out/{legA_MHC_zh,legB_MHC_EN,legC_MHC_zh,legC_HateMM,legD_MHC_zh,legD_HateMM,verdict}.json` |
| build / extraction metadata | `idea-stage/cat_closeout/{build_meta_CCA_MHC_zh,build_meta_CCB_MHC,extract_meta_MHC_zh,extract_meta_MHC,cv_meta_MHC_zh,cv_meta_HateMM}.json` |
| token composition audit | `idea-stage/cat_closeout/out/{token_composition,header_tokens}.json` |
| log | `logging/runs/cat_closeout/run.log` |
| new caches | `data/CLIP_Embedding/{MHC_zh,MHC}/*-cc.pt`, `*_CCA-*.pt`, `*_CCB-*.pt` |

**Process notes.** (a) The freeze was committed at `ae286c9` before any leg's code was written; the
implementation commit `a382714` records the pre-run checks. (b) The only pre-run executions were:
the analyzer replayed on the **already-banked** R10-COMBO logs, where it reproduced the published
MHC-ZH table exactly (`A0` 0.8061 ± 0.0100, `CAT` 0.8161 ± 0.0188, `CAT − A0` +0.0100, 26/30) at
belt E1 = 5.0e-05; a 3-item extractor smoke on each dataset into a scratch directory; and the
LoRA restore. None computed a metric on any arm of this close-out. (c) Seeds 1300–1319, 1400–1429
and 1500–1524 are disjoint from every previously consumed range (0–89, 100–129, 300–329, 400–429,
500–529, 600–629, 700–729, 800–829, 900–929, 41000–41029, 50700–50729). (d) Each leg ran once;
no leg was re-run, no seed range was extended, and no protocol was switched after a miss — the
freeze forbade all three. (e) Every belt passed: A1 at exactly 0.0, A2 on all splits, B1 at
cos ≥ 0.991 mean on both datasets, C1 at 0 overlap, D1 and E1 at ≤ 5.0e-05. (f) 250 head-training
runs, 0 failures. (g) Total API spend: **¥0.00**.
