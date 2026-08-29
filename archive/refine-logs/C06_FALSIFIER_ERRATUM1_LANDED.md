# C06 `$0` falsifier — **ERRATUM 1, LANDED**

*Proposal:* `refine-logs/C06_FALSIFIER_ERRATUM1_PROPOSAL.md`
*Adjudication:* `refine-logs/C06_FALSIFIER_ERRATUM1_REVIEW.md` — **REVISE — 1C / 1H / 2I / 2M**,
endorsing option (i)'s direction, refuting the proposal's *"decisive technical fact"*, and binding
the landing to **six obligations** (its §5).
*Landed:* 2026-08-05, by the implementation lineage, under all six.
*Design revision:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` — **v15 + erratum only**.
**`C06_FALSIFIER_PREREG_DRAFT_V15.md` remains on disk UNMODIFIED** as the GO'd record.

**No commit, no submission, no `TARGET_STATE.json` edit, no job, no mint, no GPU.**

---

## 1. What changed, in one paragraph

§5.4 pre-registered `Δ_b = mean_{i∈draw}[c̄_A(i)] − mean_{i∈draw}[c̄_c(i)]` — a **mean of a per-item
quantity**, i.e. an accuracy decomposition — as *the* bootstrap statistic, while S4 ranges over
`holm_metrics = [accuracy, macro_f1]` and §5.5 counts both into a 92-hypothesis family. The 46
macro-F1 hypotheses per dataset therefore had **no statistic**, and the implementation made that
concrete: both metric branches called `bootstrap_deltas` with identical arguments, so the macro-F1
hypotheses were bit-copies of the accuracy hypotheses carrying an mF1 label. **The erratum retains
the accuracy leg's expression verbatim and ADDS a macro-F1 bullet** defining C01's own
recompute-per-resample form via `mechfix_ops.macro_f1`. Sections touched: **§5.4** (added bullet +
added sentence), **§5.9** (added item 10), **§7.7** (`U3` retired, `U_acc`/`U_mF1` registered),
**§8** (Phase 4 re-priced, totals re-derived), **§9** (one clause on the denominator). **§5.2, §5.5,
§5.6, §5.8, §6, §13 untouched.**

---

## 2. Obligations checklist — **6 / 6**, each with evidence

### Obligation 1 — do not re-write the accuracy leg ✅

§5.4's *"Per-resample statistic"* bullet is **byte-identical** in V15E1 to v15; the erratum text
appears **after** it under the heading *"ERRATUM 1 — the macro-F1 leg (added; the bullets above are
unchanged)"*. In code, `bootstrap_deltas` keeps its three computational lines unchanged —
`a[draws].mean(axis=1) - c[draws].mean(axis=1)` — with only its docstring extended.

**Evidence:** `diff <(sed -n '/Per-resample statistic/,+3p' V15.md) <(same on V15E1.md)` is empty;
the arena's accuracy branch reads `d = bootstrap_deltas(cell["correct"][real], cell["correct"][comp], draws)`.

### Obligation 2 — state the accuracy relation truthfully ✅

V15E1 §5.4 carries the reviewer's prescribed form verbatim in substance, including every measured
figure: algebraically identical (means commute) but **not** bit-identical,
`max|Δ_frozen − Δ_C01form| = 5.55e-16`; **38.8 %** of near-identical arm pairs get a different
`one_sided_raw_p`; **1.2 %** flip the `lower > 0` predicate; **neither form dominates** on
exactly-tied draws (`82.1 %` vs `88.7 %` adverse). The permitted clause is used in exactly the
permitted form: *"The accuracy leg's expression is unchanged, therefore no accuracy quantity
moves."* The proposal's false sentence — *"no accuracy quantity moves"* as a consequence of
algebraic identity — appears **nowhere** in V15E1.

### Obligation 3 — name the macro-F1 function in §5.4 ✅

V15E1 §5.4: *"**The function is `mechfix_ops.macro_f1`** (`scripts/analysis/mechfix_ops.py:56-66`,
sha-frozen at §11 as `635c1312…`) — **named here rather than only at a call site**, because the two
same-named candidates differ: measured over all `68,915,480` confusion triples with
`tp + fp + fn ≤ 743`, … not bit-equal on `39.84 %` of them."* The coherence ground is stated: the
same function produces S1's strict `>`, S3's `≥ 0.02`, `GATE-FLOOR`'s mF1 anchor and S5's null.

### Obligation 4 — state the degenerate-draw behaviour in §5.4 ✅

V15E1 §5.4 states all four required facts: per-class F1 is `0.0` when the class is absent from the
draw or never predicted in it; **both candidate functions agree exactly there**; **neither returns
`None`**; C01's `die("bootstrap … produced class-degenerate resamples")` guard (`:1760-1761`) has
**no object in C06** because its sole trigger is `roc_auc`'s `None` and `holm_metrics` excludes
`roc_auc` — *"the same disposition §5.4.1 records for `shuffle_fixed_point_bijection`"*; and that
**predictions are resampled directly**, identical to C01's resample-then-threshold because
`prediction_cutoff = 0.0` matches `deployed_vote`'s `votes >= 0` and item-wise thresholding commutes
with resampling. It adds that at `n = 743` with 297 positives a class-degenerate draw is not
reachable in practice.

### Obligation 5 — §5.9 disclosure item for the tie behaviour ✅

V15E1 §5.9 gains **item 10**, with the measured figures and the direction: `metric_bundle` returns
`Δ_b` exactly `0.0` (**adverse**) in `100 %` of sampled tied pairs where `mechfix_ops` returns `0.0`
or `±1.11e-16` and escapes the count in **`19.3 %`**; the bias is **strictly one-directional**;
S4 easier → SURVIVE easier → **CLOSE harder** → **conservative under §4**; and the mechanism is
named as **float association, not design judgement**. It records that exact ties are not
hypothetical — two of the three quoted `C01_A0_OUT.json` macro-F1 rows carry a bootstrap quantile of
exactly `0.000000`.

### Obligation 6 — re-derive §8 and §7.7 from the settled unit ✅

The unit is `168 × U_mF1 + 92 × U_acc`, **not** `168 × U_both`. Measured at the arena's shape
(`n = 743`, `B = 2000`), **timing boundary stated**: the timed region is the vectorised computation
from the shared draws index matrix to the `(B,)` output vector, warm; the draws matrix is built once
and is **not** in the unit.

| unit | object | measured |
|---|---|---|
| `U_acc` | accuracy leg, §5.4's retained expression, **per comparison** | **`0.0049 s`** |
| `U_mF1` | macro-F1 leg, C01's recompute form via `mechfix_ops.macro_f1`, **per `(arm, seed)`** | **`0.0384 s`** |

`Phase 4 = 168 × 0.0384 + 92 × 0.0049 = 6.90 s`, **carried `7.0 s`** (above the measurement, the
design's convention). The adjudication independently measured the same two objects at `0.0030` and
`0.0131 s` — **below** the carried figures — so the carry bounds its sample as well as this one, and
that is recorded in §7.7's corroboration paragraph.

**Re-derived totals, and the printed column re-summed independently to confirm:**

| figure | v15 | V15E1 |
|---|---|---|
| Phase 4 | `11.6 s` | **`7.0 s`** |
| total | `2934.5 s` | **`2929.9 s`** |
| `× 1.25` | `3668.1 s` | **`3662.4 s`** |
| minutes | `48.9` / `61.1` | **`48.8` / `61.0`** |
| mint share | `85.5 %` | **`85.6 %`** |
| Phase 3 share | `9.3 %` | `9.3 %` |
| `2×` miss | `3208.2 s = 53.5 min` | **`3203.6 s = 53.4 min`** |
| `5×` miss | `4029.3 s = 67.1 min` | **`4024.7 s = 67.1 min`** |

An independent parse of V15E1's §8 printed product column returns **26 rows summing to `2929.9`**,
`× 1.25 = 3662.4`, mint `85.6 %`, Phase 3 `9.3 %`. `U3` is **retired** in §7.7 with its object
explained and round 14's `0.023–0.028 s` measurement marked as being of the superseded object.

**M-1 and M-2, fixed in passing:** the erratum record prints the dataset segment on every
`C01_A0_OUT.json` path (§4 below), and cites `paired_bootstrap` at its true extent `:1742-1772`
(the proposal's `:1738-1774` and `:1742-1774` both overshot; `1773-1774` are blank and `1775` begins
`holm_adjust`).

---

## 3. Code delta

**One function pair changed in one file, plus three literals.** No gate, no verdict path, no
population contract, no family size.

| file | change |
|---|---|
| `c06_falsifier_arena.py` | `bootstrap_deltas` — **computation unchanged**, docstring extended to record why the expression is retained. **`resampled_macro_f1` added** (C01's form, `mechfix_ops.macro_f1` replicated vectorised). `s4_family` — the macro-F1 branch now calls a per-`(arm, seed)` precompute (`mf1_vector`) and differences it; the accuracy branch is untouched. Literals: docstring `2934.5 → 2929.9` / `3668.1 → 3662.4`, `PROJECTED_SECONDS = 2934.5 → 2929.9`. |
| `configs/c06/c06_falsifier.json` | `design_document` → V15E1, `design_sha256` → V15E1's digest, new `erratum_1` block, `_erratum1_units`, `projected_seconds 2934.5 → 2929.9`, `projected_seconds_conservative 3668.1 → 3662.4`. |
| `c06_falsifier_mint.py`, `c06_falsifier_cpu.sbatch` | **unchanged** — hashes identical before and after. |

**`GATE-SHA` is unaffected.** `configs/c06/c06_falsifier.json` is **not** in its own digest table:
§11 lists the seven imported modules, the six read-for-definitions files and the eight input caches,
all of which are C01/C09 artifacts. The c06 config is new code and §11 declares the four new-code
artifacts **absent** from the tree at design time, so no §11 digest changes and the table needs no
regeneration. The re-run confirms **37/37** unchanged.

---

## 4. `C01_A0_OUT.json` rows, re-quoted with the dataset segment (M-1)

| path | lower | upper | one_sided_raw_p | observed_delta |
|---|---|---|---|---|
| `/datasets/**HateMM**/paired_bootstrap/primary_vs_controls/avg_score/macro_f1` | `−0.021075` | `0.020700` | `0.660170` | `0.000000` |
| `/datasets/**HateMM**/paired_bootstrap/primary_vs_controls/common/macro_f1` | `−0.029354` | `0.000000` | `1.000000` | `−0.009827` |
| `/datasets/**HateMM**/paired_bootstrap/primary_vs_controls/displacement/macro_f1` | `0.000000` | `0.027438` | `0.365317` | `0.009144` |

`MHC_zh` carries blocks at the same suffix with different values (its `avg_score` row is
`lower = −0.044137`, `p = 0.545227`), which is why the segment is now printed.

---

## 5. Before / after sha256 of every touched file

| file | before | after |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` | `75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` | **`75e3aa84…` (UNCHANGED — the GO'd record)** |
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | *(did not exist)* | `0b446b91675fd4ff8aea15f2648401d6ce589d089eadad34846f885b2ec9c2ab` (186549 B) |
| `scripts/analysis/c06_falsifier_arena.py` | `3e423bc66d93d9da549f777c1941d53dbbde74e55da101c89c21d470e6a9eada` (57730 B) | `6ba6a14e4120e683121f93d234f5794f7bab514dfe2f51a779c87246f484e7a8` (61574 B) |
| `configs/c06/c06_falsifier.json` | `a0ebe0dc29e3e820edc258bf96551fa5d68618f2a960ee010f9e49650da4bc56` (13039 B) | `3ebcc36c74b759d28612e0974227c08dea98f6ba72e09f36ca047f35d7f5087e` (14554 B) |
| `scripts/analysis/c06_falsifier_mint.py` | `1084b5be8c11ad60085115504e999b338db481801614452526084b87d1b3a1d0` | **unchanged** |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `76d061daf62c51dae584387160924cae482ca7ea20710423b443424b2a21b634` | **unchanged** |
| `refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md` | — | updated in place with an ERRATUM-1 section |

---

## 6. Dry-check re-run against the edited files

| check | result |
|---|---|
| `py_compile` (both `.py`), `bash -n` (sbatch), `json.load` (config), module exec (both) | **all OK** |
| `GATE-DET1` + `GATE-SHA` | **37/37**, exit `0` — unchanged by the erratum |
| `GATE-C01PARITY` HateMM / MHC-ZH | **`max|diff| = 0.0`** both, exit `0` |
| `GATE-ROWSUBSET` (HateMM) | **bit-exact bridge** |
| `GATE-RHORAW` | **13 arms at 4 dp**, both datasets |
| **NEW** — `resampled_macro_f1` vs a scalar `mechfix_ops.macro_f1` loop, 200 draws | **`max|diff| = 0.000e+00` — BIT-IDENTICAL** |
| **NEW** — `bootstrap_deltas` still returns the retained accuracy statistic | shape `(2000,)`, finite |
| `PROJECTED_SECONDS` as imported | **`2929.9`** |

The bit-identity check is the one that matters for the erratum: the vectorised replica used in the
arena performs the same operations in the same order as the frozen `mechfix_ops.macro_f1`, so the
statistic the design names and the statistic the code computes are the same function.

---

## 7. Blindness

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm at any point in landing
this erratum.** Verified by machine: a grep of the dry-check progress file for every decimal in the
closed `[0.6, 0.99]` returns **NONE**. All erratum measurements were on **synthetic** labels and
predictions at the arena's shape — no ro cache opened, no head-space arm built, `deployed_vote`
called zero times, no mint read. The `GATE-C01PARITY` / `GATE-ROWSUBSET` / `GATE-RHORAW` re-run
votes on nothing and prints `no arm accuracy computed`.

---

## 8. What remains

The erratum is landed; the implementation is **not frozen, not reviewed, not authorized, not
submitted**. Next: the **separate independent code/resource review lineage** over the four
executables, whose sole input is §13.1's 28 items and which should now also read this record. Its
sharpest targets remain the analytic tie-casualty bound in `vote_bounds_over_orderings` (§6.5) and
the newly added `resampled_macro_f1` / `mf1_vector` path.
