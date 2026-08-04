# C06 `$0` falsifier — **ERRATUM 1, PROPOSAL**

*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md`, sha256
`75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` (GO at round 15, 0C/0H/0I).
*Raised by:* the implementation lineage, on hitting the defect while wiring §5.4's S4 bootstrap.
*Status:* **PROPOSAL. Nothing is landed.** The design document and
`scripts/analysis/c06_falsifier_arena.py` are **unedited**. This document asks a fresh independent
reviewer to adjudicate; the erratum lands only on their GO, per the C02-v8 erratum precedent
(proposal → independent review → landed erratum).

*Compute spent producing this proposal:* reading two frozen sources, one query against the executed
`C01_A0_OUT.json`, and one synthetic timing at the arena's shape. **No battery-arm accuracy or
macro-F1 was computed on any ro-derived arm** — see §6.

---

## 1. The defect, stated precisely

**§5.4 pre-registers a statistic that has no macro-F1 instance, while S4 is scoped over two
metrics and §5.5's family arithmetic counts both.**

§5.4 ("The bootstrap statistic — pre-registered (round-4 H-2)") fixes the per-resample statistic as

> **Per-resample statistic:** `Δ_b = mean_{i ∈ draw_b}[ c̄_A(i) ] − mean_{i ∈ draw_b}[ c̄_c(i) ]`,
> where `c̄_X(i)` is the **mean over the three seeds** of item `i`'s 0/1 correctness under arm `X`

That is an **accuracy decomposition**: it is a mean of a per-item quantity, and accuracy is the only
metric in this battery that is such a mean. **Macro-F1 is not.** It is a function of four
confusion counts and does not decompose over items, so "the same statistic, on macro-F1" names
nothing. There is no object for the mF1 leg to compute.

Meanwhile:

* **S4** (§5.2) is scoped *"for every comparator in `C ∪ Θ`: bootstrap lower bound `> 0` **and**
  Holm rejects at `α = 0.05`, with the statistic pre-registered at §5.4"*, over C01's frozen
  `statistics.holm_metrics = ["accuracy", "macro_f1"]`.
* **§5.5** counts both metrics into the family and into the resolution floor:
  `common_displacement` 12 comparators + `displacement` 11 = 23; **`(12 + 11) × 2 metrics = 46`**
  per `(dataset, lineage)`; **`× 2 lineages = 92` per dataset**. The witness's floor is stated as
  *"22 or 24 comparators"* — `11 × 2` and `12 × 2` — i.e. **both metric legs of every comparator
  must reject at `p = 1/2001`**.

So the design requires 46 macro-F1 rejections per dataset from a statistic it never defined.

### How fifteen review rounds missed it — stated plainly for the record

Rounds 4 through 14 verified, repeatedly and by execution, **the family arithmetic and the Holm
mechanics**, and never **the existence of the mF1 statistic**:

* Round 4's H-2 is what created §5.4 in the first place — it found that v4 *"cited four C01
  constants but never said what `p` is"*. The repair supplied a statistic. **No round asked whether
  the supplied statistic had an instance for each metric the condition ranges over.**
* Rounds 9, 10, 11, 12, 13 and 14 each executed C01's own `holm_adjust` over the 92-family and
  reproduced §5.5's counterexample table cell for cell (`24/24`, `23/24`, `0/24` at `m = 92`).
  Every one of those runs fed `holm_adjust` **synthetic p-vectors** — `24 × 1/2001`,
  `23 × 1/2001 + 1 × 2/2001` — because that is what the table is about. `holm_adjust` sorts and
  scales p-values; it neither knows nor cares where a p came from. **The mechanics verify
  identically whether or not the mF1 p is computable.**
* Round 12's V9 and round 14's C.6 both re-derived `(12 + 11) × 2 × 2 = 92` and the
  `92 × 2/2001 > 0.05` floor. Both verified the **multiplication**. Neither asked what the second
  factor's operand was.
* The defect is invisible from the design document alone and becomes visible **the moment someone
  writes the loop**, which is what happened.

This is a clean instance of the campaign's own lesson that a separate code lineage is not optional:
seventeen clean design rounds on C09 preceded two wrong-verdict paths caught by the code lineage,
and fifteen clean rounds here preceded this.

---

## 2. The two options, code-anchored

### Option (i) — the mF1 leg recomputes macro-F1 on each resample

**What C01's frozen code actually does.** `c01_policy_contrast_a0.py:1742-1774`, verbatim in the
load-bearing part:

```python
def paired_bootstrap(candidate, control, gold, config, seed):
    count = int(config["statistics"]["n_bootstrap"])
    ...
    rng = np.random.default_rng(seed)
    deltas = {metric: [] for metric in config["statistics"]["metrics"]}
    for _ in range(count):
        sampled = rng.integers(0, len(gold), size=len(gold))
        sampled_gold = gold[sampled]
        for metric in deltas:
            cand = metric_value(metric, sampled_gold, candidate["scores"][sampled], cutoff)
            ctrl = metric_value(metric, sampled_gold, control["scores"][sampled], cutoff)
            if cand is not None and ctrl is not None:
                deltas[metric].append(cand - ctrl)
```

with `metric_value(metric, gold, scores, cutoff) = metric_bundle(gold, scores, cutoff)["metrics"][metric]`
(`:1738-1741`).

Read at the digit, C01's frozen battery:

1. resamples **item indices once per draw** (`sampled`), shared across every metric;
2. resamples the **gold labels alongside** (`sampled_gold = gold[sampled]`);
3. **recomputes each metric from scratch** on the resampled scores and resampled gold, for every
   metric in `statistics.metrics = ["accuracy", "macro_f1", "roc_auc"]`;
4. takes the **difference of recomputed metrics** as `Δ_b`, then the `5 %` quantile as `lower` and
   `(1 + #{Δ_b ≤ 0}) / (B + 1)` as `one_sided_raw_p`.

**Verified against the executed run, not inferred.** `C01_A0_OUT.json` contains **90 bootstrap
summary blocks — 30 accuracy, 30 macro_f1, 30 roc_auc** — each with `lower`, `upper`,
`one_sided_raw_p`, `bootstrap_mean` and `n = 2000`. Three of the thirty macro-F1 blocks:

| path | lower | upper | one_sided_raw_p | observed_delta | n |
|---|---|---|---|---|---|
| `…/primary_vs_controls/avg_score/macro_f1` | `−0.021075` | `0.020700` | `0.660170` | `0.000000` | 2000 |
| `…/primary_vs_controls/common/macro_f1` | `−0.029354` | `0.000000` | `1.000000` | `−0.009827` | 2000 |
| `…/primary_vs_controls/displacement/macro_f1` | `0.000000` | `0.027438` | `0.365317` | `0.009144` | 2000 |

**So macro-F1 bootstrap lower bounds exist in C01's executed output and were produced exactly this
way.** These are C01's own published dev-arena figures, already the evidence §1's table rests on;
reading them computes nothing new.

**The consequence for this erratum.** Option (i) is **not a new statistic**. It is *the* statistic
the unblock condition already names — `falsifier_spec`: *"re-run C01's real-displacement-versus-
matched-norm-orthogonal-rotation battery in the FOLD-HEAD ARENA"* — and **§5.4's prose is an
inaccurate paraphrase of it.** §5.4's own preamble says why the paraphrase was attempted: C01's
`paired_bootstrap` has *"no seed axis"* and is therefore *"non-reusable"* as-is. That observation
is correct and remains correct; the error was replacing the whole statistic when only the seed axis
needed adding.

**The corrected statement, which keeps everything §5.4 was right about:**

> For each draw `b`, resample item indices once (shared across all comparators and both lineages
> within a dataset, C01's frozen `statistics.seed = 20260728`). For each metric
> `m ∈ holm_metrics` and each arm `X`, compute `metric_m(gold[draw_b], pred_{X,s}[draw_b])` for
> each seed `s`, and take the **mean over the three seeds**. Then
> `Δ_b = mean_s metric_m(A, s, b) − mean_s metric_m(c, s, b)`.

The seed axis stays **inside** the statistic, exactly as §5.4 requires and for exactly the reason it
gives; the metric is recomputed per resample, exactly as C01 does.

**And the accuracy leg does not move — this is the decisive technical fact.** For accuracy the two
formulations are algebraically identical, because means commute:

```
mean_{i∈draw}[ c̄_A(i) ] = mean_{i∈draw}[ mean_s 1(pred_{A,s}(i) = y_i) ]
                        = mean_s [ mean_{i∈draw} 1(pred_{A,s}(i) = y_i) ]
                        = mean_s acc_s(A on draw)
```

**Measured on synthetic data at the arena's shape** (`n = 743`, `B = 2000`, 3 seeds):
`max|Δ_text − Δ_proposed| = 2.220e-16` over all 2000 draws — float rounding, i.e. **identical**.
So the erratum leaves the accuracy leg bit-equivalent to the text fifteen rounds reviewed, and is
**purely additive**: it gives macro-F1 a definition it did not have.

### Option (ii) — scope S4's bootstrap leg to accuracy only

**The family-size consequence, traced through §5.5's arithmetic exactly.**

§5.5 builds the family as `(12 + 11) comparisons × 2 metrics × 2 lineages = 92` per dataset. Removing
the mF1 leg gives two sub-options, and **both are worse:**

* **(ii-a) the mF1 hypotheses leave the family.** Then `(12 + 11) × 1 × 2 = ` **46** per dataset, and
  the witness's floor drops from *"22 or 24 comparators"* to **11 or 12**. Round 6 measured the
  consequence and §5.5 prints the table: at `m = 46` the witness rejects `24/24` even when a
  hypothesis sits one step off the floor at `2/2001`, where `m = 92` gives `23/24`. Concretely
  `46 × 2/2001 = 0.045977 ≤ 0.05` while `92 × 2/2001 = 0.091954 > 0.05`. **So S4 becomes strictly
  easier**, hence SURVIVE easier, hence **CLOSE harder**. Under §4 that direction is *conservative*
  and therefore not disqualifying — but it **reverses a disclosure the design already made**: §5.9
  item 8 discloses the 92-freeze as *easing* CLOSE, and round 6's H-1 forced that disclosure. Landing
  (ii-a) means re-deriving §5.5's counterexample table, re-writing §5.9 item 8's direction, and
  re-opening a question round 5 prescribed and round 6 adjudicated. **Round 5's lesson — that family
  size materially changes S4's attainability — is precisely why this cannot be done quietly.**
* **(ii-b) the mF1 hypotheses stay in the family at a non-bootstrap p.** Then the family stays 92,
  but 46 of its members carry a p from *some other test* that §5.4 does not define — which is the
  original defect, relocated rather than repaired. **Not viable.**

**Neither sub-option is free, and both touch text that survived adversarial review**, whereas option
(i) touches only the paragraph that was wrong.

---

## 3. Recommendation

**Adopt option (i). Re-write §5.4's statistic to C01's own; do not change the statistic the
falsifier runs.**

**The deciding argument.** This battery's entire authority is fidelity to C01 — the unblock
condition says *"re-run C01's battery"*, §11 sha-freezes C01's module and config, `GATE-C01PARITY`
asserts the arm algebra **bit-exactly** against C01's own `prepare_views` rather than against any
prose, and §13.1 item 23 instructs the code lineage *"test against `prepare_views`, do not
reimplement"*. **A battery that pins its arm algebra to C01 bit-exactly and then substitutes its own
bootstrap statistic is not re-running C01's battery.** C01's frozen `paired_bootstrap` recomputes
the metric per resample; that is the statistic; §5.4's paraphrase is the error.

Three supporting points, in descending weight:

1. **The accuracy leg is provably unchanged** (`2.220e-16`). The erratum cannot disturb any accuracy
   quantity fifteen rounds reviewed — it is additive on the mF1 side only.
2. **§5.5, §5.2, §5.9 and the resolution floor are all untouched.** The family stays 92, the witness
   floor stays 22/24, the counterexample table stays valid, and §5.9 item 8's disclosed direction
   stays correct. Option (ii) would move all four.
3. **The compute delta is negative** (§4). Option (i) makes the projection *smaller*, so it cannot
   be motivated by, or accused of, buying anything with compute.

I hold this with one reservation the reviewer should test rather than take from me: **I have not
verified that C01's `metric_bundle` macro-F1 and `mechfix_ops.macro_f1` agree**, and the battery
votes through `mechfix_ops`. If they differ (e.g. in zero-denominator handling), the erratum must
say which one the mF1 leg uses. My recommendation is `mechfix_ops.macro_f1`, because it is the
function that produces every other macro-F1 in this battery and consistency within C06 matters more
than sharing a helper with C01 — but that is a judgement, not a measurement, and it is the one place
in this proposal where I would most like to be checked.

---

## 4. Implementation delta for option (i)

### 4.1 Code

**One function changes in `scripts/analysis/c06_falsifier_arena.py`; nothing else.**

| what | where | bounded size |
|---|---|---|
| `bootstrap_deltas(correct_a, correct_c, draws)` — currently 6 lines, differences seed-mean per-item correctness | module level, replaced by an arm-level precompute `resampled_metrics(pred_by_seed, lab, draws)` returning a `(B,)` vector per `(arm, metric)` | **−6 / +~22 lines** |
| `Battery.s4_family` — the two metric branches currently call the same helper (the honest placeholder flagged in the implementation record) | one call site | **~8 lines changed** |

Total: **≈ 30 lines in one file.** No gate changes, no verdict-path change, no config change beyond
a comment. The macro-F1 used is named explicitly at the call site (see the reservation in §3).

The implementation **precomputes per arm, not per comparison**, which is exact rather than an
approximation: the draws are shared across comparators within a dataset (§5.4's *"the same draw
indices shared across all comparators"*), so an arm's resampled metric vector is identical in every
comparison it appears in. Computing it once per `(arm, seed, metric)` and differencing is therefore
**bit-identical** to computing it per comparison, at a fraction of the cost.

### 4.2 Compute, measured and multiplied honestly

Measured on synthetic data at the arena's shape (`n = 743`, `B = 2000`), 10 repetitions,
vectorised over all 2000 draws:

| quantity | measured |
|---|---|
| current statistic, per `(arm, seed)`: `c̄[draws].mean(1)` | **`0.0025 s`** |
| proposed statistic, per `(arm, seed)`, **both metrics in one pass** over the resampled confusion counts | **`0.0380 s`** (`15.3×`) |

**Two implementations, both multiplied through:**

* **Per-comparison (naive).** `92 comparison-cells × 2 arms × 3 seeds × 0.0380 = 21.0 s`, against
  Phase 4's current `92 × U3 (0.126 s) = 11.6 s`. **`+9.4 s`.** Note this exceeds `U3`, so `U3`'s
  frozen value would no longer bound the row — a second erratum. **Not recommended.**
* **Per-arm precompute (recommended, and exact).**
  `14 arms × 3 seeds × 2 datasets × 2 lineages = 168` vectorised evaluations
  `× 0.0380 s = ` **`6.4 s`**, plus the 92 comparison cells reduced to vector subtraction,
  quantile and p — the **sub-`0.1 s` class** §8 already carries at its upper bound. Row total
  **`6.5 s`**, in §8's own *measured unit × explicit count* form.

**Net effect on §8:** Phase 4 `11.6 → 6.5 s`, total **`2934.5 → 2929.4 s`**, `× 1.25 = ` **`3661.8 s`**.
Derived figures that move, all re-computed: `48.9 → 48.8 min`; `61.1 → 61.0 min`; mint share
`85.5 → 85.6 %`; Phase 3 share `9.3 %` unchanged; `2×` miss `3208.2 → 3203.1 s (53.4 min)`;
`5×` miss `4029.3 → 4024.2 s (67.1 min)`.

**The erratum makes the battery cheaper, not dearer.** Two consequences the reviewer should weigh:
`U3`'s object changes (it prices *"one comparison, both metrics"*, which is no longer the unit), so
§7.7's `U3` row needs re-scoping or retiring; and §9's heartbeat denominator is pinned to *"§8's
frozen projected"* value, so it tracks the move automatically — but the `2934.5` literal in
`c06_falsifier_arena.py` and `configs/c06/c06_falsifier.json` would need updating with the erratum.

### 4.3 Text edits required

| section | edit |
|---|---|
| **§5.4** | Replace the *"Per-resample statistic"* bullet with the corrected statement in §2 above. Keep the resample, lower-bound, one-sided-p, Holm and draw-sharing bullets **verbatim** — they are all correct and all match C01. Add one sentence recording that the accuracy leg is algebraically identical to the superseded text (`2.220e-16`), so no accuracy quantity moves. |
| **§7.7** | `U3`'s row: re-scope from *"one comparison, both metrics"* to the per-`(arm, seed)` metric-bundle unit, or retire `U3` and register the new unit. Round 14 recorded `U3` as conservative against `0.023–0.028 s`; that measurement was of the superseded object and should be marked as such. |
| **§8** | Phase 4 re-priced `11.6 → 6.5 s` as `168 × 0.0380 s` + the sub-`0.1 s` comparison class; total and all derived figures per §4.2. |
| **§5.5** | **No edit.** The family stays 92, the witness floor stays 22/24, the counterexample table stays valid. Stated explicitly so the reviewer checks it rather than assumes it. |
| **§5.2 (S4), §5.9 item 8, §5.6, §5.8** | **No edit.** S4's scope, the disclosed direction of the 92-freeze, the verdict combination and the CLOSE-attribution list are all untouched by a change to how a p is computed. *(§5.8 is named here because it was asked after; it lists what a CLOSE cannot be attributed to, and nothing in it references the bootstrap statistic.)* |
| **§14** | A new disposition block recording ERRATUM 1 at limb level, per the campaign's protocol. |
| **`configs/c06/c06_falsifier.json`, `c06_falsifier_arena.py`** | `projected_seconds` `2934.5 → 2929.4`, `projected_seconds_conservative` `3668.1 → 3661.8`, `PROJECTED_SECONDS` likewise. |

---

## 5. What this proposal does **not** ask for

No change to the arm algebra, any gate, the verdict rule, the population contract, the Holm family
size, the resolution floor, S5, or any disclosed direction. No new measurement on any battery arm.
No GPU. The erratum is one paragraph of §5.4, one §8 row, one §7.7 unit and ≈ 30 lines in one file.

---

## 6. Blindness statement

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm in producing this
proposal.** The work was: reading `c01_policy_contrast_a0.py:1738-1774`; one read-only query against
the executed `C01_A0_OUT.json`, whose macro-F1 bootstrap blocks are **C01's own published dev-arena
figures** and are already the evidence §1's table rests on; and one timing on **synthetic random
labels and predictions** at the arena's shape, which opens no ro cache, builds no head-space arm and
calls `deployed_vote` zero times. No mint was read, no head was minted, no vote was taken. The
design document and the arena file are unedited; no job was submitted; `TARGET_STATE.json` is
untouched.
