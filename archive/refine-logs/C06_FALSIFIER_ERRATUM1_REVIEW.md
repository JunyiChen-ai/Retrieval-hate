# C06 `$0` falsifier — ERRATUM 1, INDEPENDENT ADJUDICATION

*Object:* `refine-logs/C06_FALSIFIER_ERRATUM1_PROPOSAL.md`
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md`, sha256
`75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` — **re-computed by me, matches.**
*Reviewer:* fresh lineage; no part in the fifteen design rounds or the implementation.
*Compute spent:* `sha256sum`; file reads; one read-only query against the executed
`C01_A0_OUT.json`; CPU-only numpy re-derivations on synthetic data and one execution of C01's frozen
`holm_adjust` on synthetic p-vectors. **No battery arm was built, no mint read, no `deployed_vote`
call, no ro cache opened, no GPU, no job.**

---

## VERDICT

> ## **REVISE — 1 Critical, 1 High, 2 Important, 2 Minor**

**The direction is right and I endorse it.** The defect is real, option (i) is the correct repair,
and option (ii) is correctly rejected on arithmetic I reproduced. What fails review is not the
choice but **the warrant offered for it**: the proposal's self-designated *"decisive technical
fact"* — that the accuracy leg is unchanged — is **false as measured**, and §4.3 would freeze that
false sentence into the pre-registration. The repair is small and is specified below; it makes the
erratum *actually* additive rather than merely claimed to be.

---

## 1. What I verified and found sound

| # | claim | status |
|---|---|---|
| 1 | §5.4's statistic is an accuracy decomposition with no macro-F1 instance | **CONFIRMED** |
| 2 | S4 spans `holm_metrics = ["accuracy", "macro_f1"]` | **CONFIRMED** in `configs/c01/c01_a0_v2.json:134` |
| 3 | §5.5 counts both metrics: `(12+11) × 2 = 46` per `(dataset, lineage)`, `× 2 = 92` | **CONFIRMED**, quoted accurately |
| 4 | the account of how fifteen rounds missed it | **CONFIRMED**, spot-checked (§2 below) |
| 5 | the `paired_bootstrap` quotation and the four-step account | **CONFIRMED** verbatim, `:1738-1772` |
| 6 | `C01_A0_OUT.json` holds 90 bootstrap blocks, 30/30/30, and the three quoted mF1 rows | **CONFIRMED to the digit** |
| 7 | option (ii)'s family arithmetic and the §5.9 item 8 direction reversal | **CONFIRMED**, reproduced under C01's own `holm_adjust` |
| 8 | per-arm precompute is bit-identical to per-comparison | **CONFIRMED** (same operations, same order, shared draws) |
| 9 | design document and arena file unedited | **CONFIRMED** by sha (§7 below) |
| 10 | blindness statement | **CONFIRMED** — the stated compute is consistent with the artifact |

### The defect, confirmed from the frozen text alone

§5.4 fixes `Δ_b = mean_{i∈draw_b}[c̄_A(i)] − mean_{i∈draw_b}[c̄_c(i)]` with `c̄_X(i)` the seed-mean
0/1 correctness of item `i`. That is a mean of a per-item quantity. Macro-F1 is a function of four
confusion counts and admits no per-item decomposition, so the pre-registered statistic has **no
macro-F1 instance**, while S4 ranges over both metrics and §5.5 requires 46 macro-F1 rejections per
dataset. **The defect is real and it is a wrong-verdict path, not a documentation gap** — the
implementation makes that concrete at `scripts/analysis/c06_falsifier_arena.py:802-810`, where both
metric branches call `bootstrap_deltas` with **identical arguments**, so the 46 "macro-F1"
hypotheses are presently bit-copies of the accuracy hypotheses carrying an mF1 label.

### §2's account of the fifteen rounds — spot-checked

Round 4's **H-2** (`…REVIEW_R4.md:306`) is indeed what created §5.4: *"S4's statistic is not
pre-registered. The map from 2000 resamples to a Holm-testable p-value is nowhere in the
document."* It asked for a statistic; it did not ask whether the statistic had an instance per
metric.

Two Holm verifications, read directly:

* **Round 9, V11** (`…REVIEW_R9.md:70`): *"PASS, cell for cell, under C01's own `holm_adjust`.
  `m = 92`: 24/24, 23/24, 0/24 … Padding `1.0` instead of `0.5` … still gives 24/24."*
* **Round 14** (`…REVIEW_R14.md:374-386`): *"Executed through C01's own `holm_adjust`
  (`c01_policy_contrast_a0.py:1775-1784`), padding …"*, reproducing the same three rows.

Both fed **synthetic p-vectors** (`24 × 1/2001`, `23 × 1/2001 + 1 × 2/2001`, `24 × 2/2001`).
`holm_adjust` sorts and scales; it cannot know whether a p is computable. The proposal's diagnosis
is exactly right.

### `C01_A0_OUT.json`, queried by me

90 blocks carrying `{lower, upper, one_sided_raw_p, bootstrap_mean, observed_delta, n}`; by trailing
key **30 `accuracy` / 30 `macro_f1` / 30 `roc_auc`**; `n = 2000` in all 90. The three quoted rows,
re-read:

| path | lower | upper | one_sided_raw_p | observed_delta |
|---|---|---|---|---|
| `/datasets/HateMM/paired_bootstrap/primary_vs_controls/avg_score/macro_f1` | `−0.021075` | `0.020700` | `0.660170` | `0.000000` |
| `…/HateMM/…/primary_vs_controls/common/macro_f1` | `−0.029354` | `0.000000` | `1.000000` | `−0.009827` |
| `…/HateMM/…/primary_vs_controls/displacement/macro_f1` | `0.000000` | `0.027438` | `0.365317` | `0.009144` |

Exact. Note for the record — it bears on **C-1** and **H-1** below — that two of these three have a
bootstrap quantile of **exactly `0.000000`** and the first has an observed delta of exactly zero.
**Exact ties in this battery's macro-F1 are not hypothetical; they are in C01's own published
output.**

### Option (ii), reproduced

Executed through C01's frozen `holm_adjust`:

| witness p-values | `m = 92` | `m = 46` |
|---|---|---|
| 24 × `1/2001` | 24/24 | 24/24 |
| 23 × `1/2001` + 1 × `2/2001` | **23/24** | 24/24 |
| 24 × `2/2001` | **0/24** | 24/24 |

`92 × 2/2001 = 0.091954 > 0.05`; `46 × 2/2001 = 0.045977 ≤ 0.05`; `92 × 1/2001 = 0.045977`. The
floor flip is real, (ii-a) does invert §5.9 item 8's disclosed direction, and (ii-b) relocates the
defect rather than repairing it. **Option (ii) is correctly rejected.**

---

## 2. FINDINGS

### C-1 (Critical) — the accuracy leg **does** move, and §4.3 would freeze the claim that it does not

**The claim under review.** §2: *"the erratum leaves the accuracy leg bit-equivalent to the text
fifteen rounds reviewed."* §3 point 1: *"The accuracy leg is provably unchanged (`2.220e-16`). The
erratum cannot disturb any accuracy quantity fifteen rounds reviewed."* §4.3 instructs the landing
to *"Add one sentence recording that the accuracy leg is algebraically identical to the superseded
text (`2.220e-16`), **so no accuracy quantity moves**."*

**The means-commute algebra is correct.** I verified it symbolically and numerically: the two forms
are the same real number. **The inference from `2.220e-16` to "identical" is not.** Re-associating
`mean_i mean_s` into `mean_s mean_i` changes the floating-point summation order, and S4's two
decision predicates are a **strict inequality against zero** (`lower > 0`) and a **zero-count
threshold** (`p = 1/2001` demands *zero* draws with `Δ_b ≤ 0`). Those are the two predicates in the
entire design that a single ulp can decide.

**Measured, at the arena's exact shape** (`n = 743`, labels `297/446`, `B = 2000`, 3 seeds, C01's
frozen `statistics.seed = 20260728`), on near-identical arm pairs — the regime the rotation guard
arms actually occupy, since `GATE-ZEROOP` caps `guard_orthrot_0` vs `endpoint_concat` mismatches:

| measurement | result |
|---|---|
| `max|Δ_text − Δ_proposed|` | `5.5511e-16` |
| 400 arm pairs (1–9 flipped items): pairs whose **`one_sided_raw_p` differs** | **155 / 400 = 38.8 %** |
| 400 arm pairs: pairs whose **S4 `lower > 0` predicate itself flips** | **5 / 400 = 1.2 %** |
| 400 arm pairs: pairs where some form gives `lower` **exactly `0.0`** | 19 |
| 60 arm pairs (2–13 flipped items): draws with an exact accuracy tie | 168 |
| — of those, draws where the **adverse predicate differs** | **47** |
| — counted adverse by §5.4-as-frozen / by the proposed form | **138 (82.1 %) / 149 (88.7 %)** |

Three consequences:

1. **"No accuracy quantity moves" is false.** Both S4 conjuncts move on reachable data.
2. **The move is not even conservative.** On exactly-tied draws neither form dominates
   (82.1 % vs 88.7 % adverse), so the change cannot be excused under §4's conservatism rule the way
   §5.9 items 6, 8 and 9 are. §4's second condition — *"never let it excuse an arithmetic error"* —
   is engaged here, and the proposal invokes conservatism nowhere for this leg because it believes
   there is nothing to excuse.
3. **The false sentence would enter the frozen design.** §4.3 mandates it verbatim. An erratum
   whose warrant is *fidelity* cannot land a statement contradicted by the first measurement a
   reader would run.

**Resolution (binding, and it makes the erratum genuinely additive).** Do not re-write the accuracy
leg at all. Keep §5.4's *"Per-resample statistic"* bullet **verbatim as the computational form for
the accuracy leg** — the arena's `bootstrap_deltas` already implements it exactly — and **add** a
second bullet defining the macro-F1 leg by C01's recompute-per-resample form with the seed mean
inside. There is no fidelity cost: for accuracy the frozen expression and C01's form are the same
real number, so retaining the frozen expression departs from C01 by nothing, while re-writing it
departs from fifteen reviewed rounds by up to one ulp on the two predicates that matter. Then the
sentence §4.3 wants to add becomes **true** and can be stated as: *the accuracy leg's expression is
unchanged, therefore no accuracy quantity moves by even one ulp; for accuracy the superseded and
C01 forms are algebraically identical (means commute) but not bit-identical
(`max|Δ| = 5.55e-16` measured), which is why the frozen expression is retained rather than
restated.*

### H-1 (High) — the macro-F1 function is not pinned, and the two candidates differ on reachable resamples

This is the reservation the proposer asked to be checked. **They were right to flag it, their
recommendation is right, and their handling of it in §4 is not.**

**The two implementations.**

* `metric_bundle` (`c01_policy_contrast_a0.py:1474-1475`): `0.0 if 2tp+fp+fn == 0 else 2.0*tp/(2tp+fp+fn)`.
* `mechfix_ops.macro_f1` (`mechfix_ops.py:63-65`): `pr = tp/(tp+fp)`, `rc = tp/(tp+fn)`,
  `2*pr*rc/(pr+rc)`, each guarded to `0.0` on a zero denominator.

**Exhaustive measurement** over **all 68,915,480** confusion triples with `tp + fp + fn ≤ 743`:

* **27,456,838 triples (39.8413 %) are not bit-equal**; `max|bundle − mech| = 2.220e-16`.
* They agree **exactly on every degenerate case**: `tp = 0` (either class absent from the draw or
  never predicted) gives `0.0` from both, as does `tp = fp = fn = 0`. Neither returns `None`.

**The structural difference, which is what matters.** `metric_bundle` performs **one correctly
rounded division of exactly representable integers**, so its value is a deterministic function of
the *exact rational* F1. `mechfix_ops` performs four rounded operations, so its value depends on
the `FP/FN` split at a fixed exact value. Sweeping 39,591 `(TP, e = FP+FN)` cells at `n = 743`: the
exact value and `metric_bundle`'s float are invariant across the split in **all** of them;
`mechfix_ops`' float **varies in 27,837 (70.3 %)**, spread up to `3.331e-16`.

**The verdict-relevant consequence.** On a resample where two arms have *mathematically equal*
macro-F1:

| | behaviour |
|---|---|
| `metric_bundle` | `Δ_b` is **exactly `0.0`** → **adverse**, in **198,023 / 198,023 (100 %)** sampled tied pairs |
| `mechfix_ops` | `Δ_b` is `0.0` or `±1.11e-16` → adverse in **159,809 / 198,023 (80.7 %)** |

A concrete reachable instance at `n = 743`:

```
arm A = (TP=180, FP=0, FN=60, TN=503)      arm c = (TP=180, FP=1, FN=59, TN=503)
exact macro-F1 IDENTICAL, both = 6719/7462
metric_bundle : Delta = 0                 -> adverse (counted)
mechfix_ops   : Delta = +1.110e-16        -> NOT adverse (escapes the count)
```

The bias is **strictly one-directional**: `mechfix_ops`' adverse count is always `≤`
`metric_bundle`'s, never `>`. Since S4's floor requires **zero** adverse draws out of 2000, a
single tied draw decides a hypothesis, and this choice therefore decides S4 at its margin.

**My ruling on the question: `mechfix_ops.macro_f1` — the proposer's recommendation is correct.**
Three grounds, the first two theirs and the third theirs by implication but not computed:

1. It is the function that produces every other macro-F1 in this battery — S1's strict `>`, S3's
   `≥ 0.02`, `GATE-FLOOR`'s mF1 anchor (`c06_falsifier_arena.py:765`) and S5's null
   (`:858`). Using `metric_bundle` in S4 alone would give one conjunct a **different tie
   convention from its siblings**, which is a worse incoherence than sharing a helper with C01.
2. C01's `metric_value` calls `metric_bundle`, which also computes `roc_auc` — the **only** metric
   whose `metric_value` can return `None` (`c01_policy_contrast_a0.py:1449-1450`), and the sole
   object of C01's `die("bootstrap … produced class-degenerate resamples")` guard at `:1760-1761`.
   C06's `holm_metrics` excludes `roc_auc`, so that guard **has no object in C06** — the same
   disposition §5.4.1 records for `shuffle_fixed_point_bijection` — but wiring `metric_bundle`
   wholesale would reintroduce its failure mode for no benefit.
3. **Direction.** §4 defines conservative as *hardest to deliver the `$0` CLOSURE*, and §5.5 labels
   CLOSE-easier *anti-conservative*. `mechfix_ops` makes S4 **easier** → SURVIVE easier → CLOSE
   **harder** → **conservative**. `metric_bundle` leans the other way. Under §4 the recommendation
   is the correct lean.

**What fails review is the handling.** §4.1 discharges the choice as *"The macro-F1 used is named
explicitly at the call site"* — a **code comment, not a pre-registration** — and §4.3's §5.4 edit
does not require the statistic's text to name it. A pre-registered statistic whose value depends,
at the margin, on which of two same-named helpers is called is not pre-registered.

**Resolution (binding).** §5.4's new macro-F1 bullet must state, in the design text: (a) the
function by name and path — `mechfix_ops.macro_f1`, `mechfix_ops.py:56-66`, already sha-frozen at
§11 as `635c1312…`; (b) that the arena's per-item predictions are resampled directly, which is
identical to C01's resample-scores-then-threshold because `retrieval.prediction_cutoff = 0.0`
matches `deployed_vote`'s `votes >= 0` convention and item-wise thresholding commutes with
resampling; (c) the **degenerate-draw behaviour** — a class absent from a draw, or never predicted
in it, contributes per-class F1 `0.0`, both functions agree exactly there, neither returns `None`,
and at `n = 743` with 297 positives a class-degenerate draw is not reachable in practice; and (d)
the **tie convention** with its measured direction, per I-1.

### I-1 (Important) — §5.9 gets no disclosure item for a measured, one-directional lean

§4 binds this design to *disclose what its lean buys*, and §5.9 carries items 6, 8 and 9 for
exactly this. H-1 establishes a new one: relative to C01's own `metric_bundle`, the chosen
`mechfix_ops.macro_f1` can only **remove** draws from S4's adverse count, never add — measured at
19.3 % of mathematically-tied draws — which is S4-easier, SURVIVE-easier, **CLOSE-harder**, the
conservative direction. §4.3's table asserts *"§5.9 item 8 — No edit"*, which is true, and then
stops. **Resolution:** add a new §5.9 item recording the direction, the mechanism (float
association, not design), and the measured figures.

### I-2 (Important) — §4.2's compute arithmetic is derived from a unit that C-1 retires

§4.2 prices Phase 4 as `168 × 0.0380 s = 6.4 s` on the premise that **both** legs move to a per-arm
precompute. Under C-1 the accuracy leg stays where it is (per comparison, on the already-seed-meaned
`cell["correct"]` vectors) and only macro-F1 needs the precompute, so the row is
`168 × U_mF1 + 92 × U_acc`, not `168 × U_both`.

The **conclusion survives** — Phase 4 still gets cheaper, so §4.2's "the compute delta is negative"
stands and cannot be accused of buying anything. I measured my own units at `n = 743`, `B = 2000`,
10 reps: current-style `0.0030 s`, both-metrics-one-pass `0.0131 s` (the proposal reports `0.0025`
and `0.0380`; the spread is about what each timer enclosed, the lesson §8 already institutionalises).
Under either unit the row lands far below Phase 4's current `92 × U3 = 11.6 s`. **But the specific
literals in §4.2 and §4.3 — `2929.4`, `3661.8`, `48.8`, `61.0`, `85.6 %`, `3203.1`, `4024.2` — are
premature** and must be re-derived after C-1 fixes what the unit is.

For the record, I re-derived the *arithmetic* of those figures and it is internally consistent:
`2934.5 − 11.6 + 6.5 = 2929.4`; `× 1.25 = 3661.75 → 3661.8`; mint share
`2508.3 / 2929.4 = 85.63 % → 85.6 %`; Phase 3 share `273.7 / 2929.4 = 9.34 % → 9.3 %` unchanged;
the 2× and 5× miss rows shift by the same `5.1 s`. **The multiplication is right; the multiplicand
is not yet settled.** §4.3's §7.7 note is otherwise correct — round 14 did record `0.023–0.028 s`
against `U3`'s frozen `0.126 s` (`…REVIEW_R14.md:525`), and that measurement was of the superseded
two-accuracy-leg object.

### M-1 (Minor) — the three quoted `C01_A0_OUT.json` rows are quoted at an ambiguous path

§2's table elides the dataset: `…/primary_vs_controls/avg_score/macro_f1`. **Both** `HateMM` and
`MHC_zh` carry blocks at that exact suffix with different values (e.g. `MHC_zh`'s `avg_score` row is
`lower = −0.044137`, `p = 0.545227`). The quoted numbers are unambiguously HateMM's and are exact,
but a reader re-checking the claim can land on the wrong row. **Resolution:** print the dataset
segment.

### M-2 (Minor) — line-range citations overshoot

§2's header cites `c01_policy_contrast_a0.py:1742-1774` and §6 cites `1738-1774`. `paired_bootstrap`
ends at `1772`; `1773-1774` are blank and `1775` begins `holm_adjust`. The frozen design's own §5.4
cites `:1742-1772` exactly. Cosmetic, but this document's citations are load-bearing.

---

## 3. The §4 implementation delta — is it bounded as claimed?

**Partly.** "One function changes" is right in spirit and the blast radius is genuinely small, but
the enumeration is incomplete in two places already covered: §5.9 needs a new item (I-1) and the §8
literals are premature (I-2). Two further checks:

* **The per-arm precompute is exact, as claimed.** §5.4 shares draw indices across all comparators
  and both lineages within a dataset, so an arm's resampled metric vector is the same object in
  every comparison it appears in; computing it once per `(arm, seed, metric)` and differencing runs
  the same operations in the same order as computing it per comparison. **Bit-identical.
  Confirmed.** The count `14 arms × 3 seeds × 2 datasets × 2 lineages = 168` is a correct upper
  bound (only 13 distinct arms enter S4's comparator sets).
* **§13.1 needs no edit.** Item 16 requires the code lineage to verify *"the bootstrap statistic,
  the one-sided `p` and the Holm step-down match §5.4"* — a pointer, so it re-targets automatically.
  The proposal's omission of §13.1 is correct, not an oversight.

---

## 4. Process integrity

**Blindness — clean.** The proposal's stated compute is two frozen source reads, one query against
`C01_A0_OUT.json`, and one synthetic timing. `C01_A0_OUT.json` is C01's own executed dev-arena
output, already the evidence §1's table rests on; reading it computes nothing on a C06 arm. Nothing
in the artifact evidences a battery-arm accuracy or macro-F1 on a ro-derived arm, a mint read, an
arena build or a `deployed_vote` call. **My own compute obeyed the same boundary.**

**Nothing is edited.**

| file | sha256, re-computed by me | expected |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` | `75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` | ✓ the frozen design |
| `scripts/analysis/c06_falsifier_arena.py` | `3e423bc66d93d9da549f777c1941d53dbbde74e55da101c89c21d470e6a9eada` | ✓ implementation record row 2, `57730` bytes — **also matches on size** |
| `configs/c01/c01_a0_v2.json` | `f3997bdd…` | ✓ §11 |
| `scripts/analysis/c01_policy_contrast_a0.py` | `d2b9c2ff…` | ✓ §11 |
| `scripts/analysis/mechfix_ops.py` | `635c1312…` | ✓ §11 |

`configs/c06/c06_falsifier.json` still carries `projected_seconds: 2934.5` /
`projected_seconds_conservative: 3668.1`, and `c06_falsifier_arena.py:45` still carries
`PROJECTED_SECONDS = 2934.5`, so the §4.3 literal edits are indeed outstanding and unapplied.
`TARGET_STATE.json` untouched. **The proposal's §6 is accurate.**

---

## 5. Obligations on the landing

The erratum lands on satisfying **all six**:

1. **Do not re-write the accuracy leg.** §5.4's *"Per-resample statistic"* bullet is retained
   **verbatim** as the accuracy leg's computational form. The erratum **adds** a macro-F1 bullet;
   it replaces nothing. (C-1)
2. **State the accuracy relation truthfully.** The added sentence records that the frozen
   expression and C01's per-seed form are algebraically identical (means commute) but **not**
   bit-identical (`max|Δ| = 5.55e-16` measured; on near-identical arm pairs at `n = 743`,
   `B = 2000`, 38.8 % of pairs get a different `one_sided_raw_p` and 1.2 % flip the `lower > 0`
   predicate), and that the frozen expression is retained **for that reason**. The clause *"so no
   accuracy quantity moves"* may be kept only in the form *"the accuracy leg's expression is
   unchanged, therefore no accuracy quantity moves."* (C-1)
3. **Name the macro-F1 function in §5.4**: `mechfix_ops.macro_f1` (`mechfix_ops.py:56-66`, §11
   sha `635c1312…`), not merely at a call site. (H-1)
4. **State the degenerate-draw behaviour in §5.4**: per-class F1 is `0.0` when the class is absent
   from the draw or never predicted in it; both candidate functions agree exactly there; neither
   returns `None`; C01's `die("bootstrap … produced class-degenerate resamples")` guard has **no
   object** in C06 because `holm_metrics` excludes `roc_auc`. Also state that predictions are
   resampled directly, which is identical to C01's resample-then-threshold at
   `prediction_cutoff = 0.0`. (H-1)
5. **Add a §5.9 disclosure item** for the tie behaviour: relative to C01's `metric_bundle`,
   `mechfix_ops.macro_f1` can only remove draws from S4's adverse count, never add (19.3 % of
   mathematically-tied draws, measured), i.e. S4-easier / SURVIVE-easier / **CLOSE-harder** — the
   conservative direction under §4 — and the mechanism is float association, not design. (I-1)
6. **Re-derive §8 and §7.7 after obligation 1**, since the unit is `168 × U_mF1 + 92 × U_acc`, not
   `168 × U_both`. The `2929.4 / 3661.8 / 48.8 / 61.0 / 85.6 % / 3203.1 / 4024.2` literals, and the
   matching constants in `configs/c06/c06_falsifier.json` and `c06_falsifier_arena.py:29,45`, are
   re-computed from the settled unit. The direction (Phase 4 cheaper, delta negative) is confirmed
   and is not in question. (I-2)

Fix M-1 and M-2 in passing. **Options (i) and (ii) are not reopened: option (i) is adjudicated
correct and option (ii) correctly rejected.** Re-review scope is the six obligations only.
