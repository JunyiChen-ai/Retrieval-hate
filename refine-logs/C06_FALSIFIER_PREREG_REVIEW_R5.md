# C06 `$0` falsifier — independent design review, **ROUND 5**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V5.md` (DRAFT v5, 2026-08-04)
**Reviewer posture:** fresh, independent of rounds 1–4 and of the designer. Read-only. No GPU,
SLURM, Modal, model load, head training, arena run, cache write, test-split access, job submission
or commit occurred. `TARGET_STATE.json`, all five drafts, all configs and all four prior reviews
were read and not modified. Nothing heavier than `sha256sum`, file reads, and numpy/torch-CPU
re-derivation on already-banked **train-split** caches and banked mint checkpoints was executed.
**No arm accuracy was computed at any point.**

---

# VERDICT

## **REVISE (3C / 3H / 6I)** — plus 5 Minor

Not `GO (0C/0H/0I)`.

**Ceremony floor: clean, and re-derived rather than read.** All **21** sha256 digests reproduce
character-for-character against disk. Both provenance chains are sha-gated in source. All **26**
`ρ_raw` values reproduce at 6 dp. The six `GATE-FLOOR` anchors match the banked JSONs on **both**
metrics and all 30 `fold_acc_deployed` entries are present. The trained-head `ρ` reference
reproduces to the digit on all 36 banked mints (**0/18 above `ρ*` on both**). Every §8 product
re-multiplies and the printed column sums **exactly** to `2927.5 s`. All population constants,
class counts, majorities, bands and tie caps re-derive exactly. The row-subset identity is
`0.000e+00` on all 13 arms. I rebuilt the 13-arm algebra from §3.4's prose alone and matched
`prepare_views` **bit-exactly** on both datasets — the fourth independent reproduction.
Test-split non-contact is sound by construction. Blindness is intact: I grepped every decimal in
`[0.6, 0.99]` across **v1–v5** and classified all 97 distinct values — every one is a `ρ`, a
`‖head_f(0,0)‖` magnitude, a cos/`‖Δ‖` geometry figure, a banked `GATE-FLOOR` anchor, a published
C01 dev-arena accuracy, a majority/band constant, or a unit-time arithmetic string. **No arm
accuracy from this battery appears anywhere in v1–v5.**

**Disposition audit: all 14 round-4 findings and all 4 Minors are genuinely adopted. 18 of 18
VERIFIED ADOPTED, 0 NOT ADOPTED, 0 PARTIAL.** This is the first round with no failed adoption, and
I verified each by executing C01's frozen code rather than by reading §14. The three Criticals in
particular landed: `GATE-ARMVIAB` is gone from every list, verdict path, gate table, §8 count and
heartbeat spec; S6 is a binding conjunct with a correct counterexample; and the fold axis is
carried at 60 cells with the guard arms counted.

**All three of my Criticals sit in the seam v5's own three repairs opened — the pattern every
round before has found.** C-1 and C-2 are both in **per-lineage gate scoping**, the structure v5
introduced to discharge round-4 H-3. C-3 is in **S7**, the object v5 created by discharging
round-4 H-1.

**C-1 is a wrong-verdict path: the drop is scoped on the lineage axis, but every per-lineage gate
is a `(dataset, lineage)` object.** All six per-lineage gates carry per-dataset constants — the
`GATE-ARENA` bands `[0.6203, 0.98]` vs `[0.7091, 0.98]`, `ρ*` `0.968176` vs `0.977223`,
`GATE-SELFTEST`'s `n_D` `743` vs `579`, `GATE-ZEROOP`'s cap `7` vs `5`. §5.6 says only *"a lineage
that fails one is … dropped"* and *"both lineages passed their per-lineage gates"*. It never says
whether a failure on **one** dataset drops the lineage on both. On the design's own most likely
instrument-failure path — Head-N, the out-of-domain transplant, missing `GATE-ARENA`'s lower bound
on HateMM but clearing it on MHC-ZH — the two readings give **HALT** and **CLOSE** on the same
event. A CLOSE is terminal: it retires C06 and forecloses the authorized `1.7–2.5 GPU-h`
permanently.

**C-2 is two decision rules in one subsection, on the path §5.7 names as live.** §5.6's combination
rule drops a lineage and continues; §5.6's own absence rule says *"An **absent** decision or gate
quantity HALTs on the same footing as a non-finite one."* A dropped lineage's S1–S7 quantities are
absent decision quantities. Under the absence rule the battery HALTs — which is exactly the defect
round-4 H-3 identified, reintroduced by the repair, so v5's claim that *"a clean Head-R SURVIVE is
no longer voided by the transplant lineage's failure"* is not delivered. The second limb is
§5.5's frozen Holm family: **92 per dataset spanning both lineages**, with no statement of what
happens to the dropped lineage's 46 hypotheses. I measured the difference — at 92 a comparator
needs `p = 1/2001` to reject below rank 42; at 46 even `p = 2/2001` rejects at **every** rank. The
family size on the drop path is undefined and materially changes S4's attainability.

**C-3: S7 is now a binding SURVIVE conjunct whose threshold, reference arm, space and seed axis
are all unregistered.** v5 pre-registers the quantile (`0.1`), its population (arena), the arm
scope (`common_displacement`) and the zero-fix convention — and stops. C01's
`displacement_audit` needs three more things the battery cannot inherit, because §11 does not
import it: the dominance threshold `max_small_displacement_fix_fraction = 0.5`, which appears
**nowhere in v5**; the reference arm, which C01 picks by *measured accuracy* over the five
`gain_controls` (`select_strongest_ordinary_control`, and C01's own runs selected `common` on
HateMM and `endpoint_concat` on MHC-ZH); and the statistic itself, `min(‖d_img‖, ‖d_text‖)` — a
per-row minimum over two modality norms that **does not exist in head space**, where the head emits
one fused block. §3.7 names the population and not the space. If S7 reads on the raw features,
§3.6's *"the raw leg … enters no decision rule"* becomes false.

**None of the three requires a GPU, an extraction, new data or a redesign.** C-1 is one sentence
naming the dataset axis. C-2 is one clause exempting a dropped lineage from the absence rule plus
one sentence fixing the family. C-3 is four constants and a space. **The falsifier can still
discharge its written condition at `$0`**, and at the corrected projection — `48.8` corroborating
/ `61.0` conservative CPU-minutes, which I re-multiplied from the design's own structure and which
is right.

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All 21 digests recomputed against disk and matched character-for-character: 7 imported modules, 6 read-for-definition files, 8 input caches. Spot values: `headspace_mint.py` `cefdf8dc…0916612`; `c01_policy_contrast_a0.py` `d2b9c2ff…8db1b855`; `c09guard.py` `aed50842…d745f062`; HateMM `ro_L24` `6a44cce4…0be045f`; MHC-ZH native `dev_seen` `4c07af75…7e4f5d3c`. Both provenance chains sha-gated **in source** (`_v4.py:52-55` → `_v3.py:48-51` → base), configs v4→v3→v2. |
| **V2** | **VERIFIED — C-1's premise is sound** | From `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4/C01_A0_OUT.json`, recomputed: raw `displacement` `0.8504672897` / `0.8846153846`; `common_displacement` `0.8598130841` / `0.8589743590`. Against the arena bars `0.6203` / `0.7091` they clear by `0.2302 / 0.2395` (HateMM) and `0.1755 / 0.1499` (MHC-ZH) — **range `0.1499–0.2395`**. §1's whole table is faithful: I re-derived every accuracy **and** every net from `n_dev × Δacc` (`+1/+2`, `+2/0`, `−2/−2`, `+3`, `+2`, `+3/+3`, `+1/+3`) and all eight rows match. `GATE-ARMVIAB`'s escape branch is indeed unreachable, and the retirement rests on measured fact. **One numeric exception, see M-1:** `gain_over_strongest_control` is `−0.009345794392523366` in the JSON, i.e. `−0.0093`, not the `−0.0094` §1 prints. |
| **V3** | **VERIFIED — exact** | `net = (2, 21, 22)`: mean `15.00 ≥ 0.02 × 743 = 14.86` ✓ (S3 satisfiable); `min = 2 < 3` ✓ (S6 fails). Spread `22 − 2 = 20` net items `= 20/743 = 0.026918 → 2.69` accuracy points ✓. MHC-ZH leg `0.02 × 579 = 11.58` ✓ against `minimum_net_fixes.MHC_zh = 2`. **S3 does not imply S6.** |
| **V4** | **VERIFIED — and the printed column sums to the total** | Head key matrices `2 × 3 × 5 × 2 = 60` ✓ (`headspace_arena.py:75` opens `mint_{ds}_s{seed}_f{fold}.npz` **inside** the fold loop; `:88` takes `X = P.l2n(zf["K_train"])` there). Phase 2b `60 × 0.1873 = 11.238 → 11.2` ✓; Phase 2D `(2 + 60) × 0.62 = 38.44 → 38.4` ✓; Phase 2z `120 × 0.00629 + 60 × (2/13) × 0.1873 = 0.755 + 1.729 = 2.484 → 2.5` ✓. **I re-multiplied all 22 rows independently from folds × seeds × lineages × datasets × arms × draws** and every product is right. Column sum = `2927.5` exactly ✓; `× 1.25 = 3659.375 → 3659.4` ✓; `48.8 / 61.0 min` ✓; mint share `2508.3/2927.5 = 85.68 → 85.7 %` ✓; Phase 3 share `9.349 → 9.3 %` ✓; sensitivities `3201.2` and `4022.3` ✓. The stated delta `2886.3 − 2.2 − 8.7 + 11.2 + 38.4 + 2.5` also lands on `2927.5` ✓. |
| **V5** | **PARTIAL — the arithmetic is right, the inference is not** | `1/2001 = 0.00049975 < α/92 = 0.00054348` ✓ clears at rank 1. `2/2001 = 0.00099950` first clears at 0-indexed rank 42 = **1-indexed rank 43** ✓ (I tabulated ranks 38–45). **But "42 of 92" is the requirement for all 92 to reject, and S4 does not require that.** S4 is scoped *"for every comparator in `C ∪ Θ`"* for the **witness** arm on the **witness** lineage — 24 hypotheses for `common_displacement`, 22 for `displacement`. Executing C01's `holm_adjust` over a 92-family with those 24 at `1/2001` and the other 68 at `0.5`: **24/24 reject**. Put one of the 24 at `2/2001`: 23/24. So the true floor is **22 of 92** (`displacement` disjunct) / 24 (`common_displacement`), not 42. → **H-1**. |
| **V6** | **VERIFIED — all four legs executed** | `‖head_f(0,0)‖` non-zero at torch seeds 0/1/2: `0.581950 / 0.597144 / 0.584977` (inside §7.4(a)'s `0.58–0.65` band for this emitter). `h_std[355] == h_ow[355]` **exactly** and not zero; **zero** exact-zero head rows on either dataset; row 355 the only such row. In head space, one-hot `{355}` **DIES** at the standard endpoint block, all-False at `n = 744` **DIES** at the displacement block ⇒ `common_displacement` unbuildable under either mask ✓. **The repair: all 13 head-space arms BUILD at `n = 743` with the explicit all-False array**, `float32`, dims **`{1024: 4 arms, 2048: 9 arms}`** — independently confirming §8 Phase 2's `240 / 540` split and §7.4(g). |
| **V7** | **VERIFIED — exact** | Raw arms at `n = 743` vs the `n = 744` one-hot build restricted to the 743 surviving rows: **`max\|diff\| = 0.000e+00` on all 13 arms, both datasets**. Algebra guards bit-identical (`8.940696716308594e-08` / `1.1920928955078125e-07` HateMM; `8.940696716308594e-08` twice on MHC-ZH). Every `ρ` unchanged. |
| **V8** | **VERIFIED — bit-exact, fourth independent reproduction** | I re-implemented §3.4's `fuse` / `paired` / `build_views` from the prose alone, calling the imported `l2_rows`, and compared against `prepare_views` with §3.7's mask forms: **`max\|diff\| = 0.000e+00`, 13 arms, both datasets, `float32` on both sides.** One note for §13: the prose does **not** determine the arm-name→formula map — I initially built `common_interaction` as `fuse` and had to correct it to `paired(common, ci)` against `prepare_views:1334-1338`. The map is pinned by the parity gate, not by the prose, which is fine for the two-block build and is why §13 must carry the one-construction claim (→ **I-6**). |
| **V9** | **VERIFIED — all 26, exact at 6 dp** | `ρ* = 0.968176` (HateMM, `endpoint_std`) / `0.977223` (MHC-ZH, `endpoint_std`); runner-ups `0.964446` / `0.969686`, both `common`. Every one of the 26 `ρ_raw` entries reproduces exactly as tabulated, including `common_interaction`'s cross-dataset asymmetry (`0.913840` vs `0.968188`) and `displacement` least-concentrated on both. Full-precision freeze removes the self-exemption by equality. `ρ` over 744 rows with the zero row left in: `0.966874` vs `0.968176`, max shift over 13 arms **`1.301e-03`** ✓. Trained-head reference over all 36 banked `mint_*.npz`: HateMM `0.447803 / 0.562434 / 0.632996`, MHC-ZH `0.340179 / 0.574247 / 0.667326`, **0/18 above `ρ*` on both** ✓. |
| **V10** | **VERIFIED — every constant** | HateMM full `n = 744`, `pos 298 / neg 446`, `446/744 = 0.599462 → 0.5995`. **Arena `n = 743`, `pos 297 / neg 446`, `446/743 = 0.600269 → 0.6003`, band `[0.620269 → 0.6203, 0.98]`, tie cap `⌊7.43⌋ = 7`.** MHC-ZH `n = 579`, `pos 180 / neg 399`, `399/579 = 0.689119 → 0.6891`, band `[0.709119 → 0.7091, 0.98]`, tie cap `5`. Exact-zero rows: HateMM `{355}` = `hate_video_95`, **label 1**, present in both modalities of both ro caches **and** the native cache, and the only such row; MHC-ZH none. `GATE-DOMAIN`'s two majorities `0.6003 / 0.5995` and `0.6891 / 0.6891` ✓. Small-displacement quantile population `743 / 579` ✓ **— but see C-3 on its space.** `GATE-IDPARITY` holds directly: both ro caches' `ids` order-identical and `labels` element-identical to the native bank on both datasets. |
| **V11** | **VERIFIED** | Six `GATE-FLOOR` anchors read from the banked `headspace_arena_*_OUT.json`: acc HateMM `0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946`; mF1 HateMM `0.8838 / 0.8811 / 0.8812`, ZH `0.8747 / 0.8710 / 0.8765` — identical to §6's anchors; all 30 `fold_acc_deployed` entries present. Holm family from `c01_a0_v2.json`: `gain_controls` = 5, `primary_vs_controls` = 6, `holm_metrics` = 2 ⇒ `(6+6) + (5+6) = 23`, `× 2 = 46` per `(dataset, lineage)`, `× 2 = 92` per dataset ✓, distinct from §8 Phase 4's `23 × 2 ds × 2 lin = 92` ✓ — two genuinely different products. |
| **V12** | **VERIFIED — by grep and by line count** | `headspace_fidelity.py` contains **no occurrence of `dev_seen`** at all; it reads `mint_{ds}_s{seed}_ffull.npz` (`:66`) and the banked floor trainlog (`with open(path)` at `:42`, patterns at `:30-33`), taking `lab_dev` out of the mint `.npz`. §12's `dev_path_opens` second term is **`0`** ✓. `headspace_mint.py:192-194` is exactly `if os.path.exists(a.out): / print(...) / return`, and `:199` is exactly `dv = load_split(cache_dir, "dev_seen", model_name)` — the return precedes both `:199` and the fold-parity block at `:203-216` (assertion at `:215-216`), so §12's `mints_executed` binding and §3.2's two-way `GATE-FOLD` discharge are both correct. `.npz` written at `:321-325`, after the assertion ✓. |

## Additional measurements v5 does not report

**(η) The head-space algebra residual, measured for the first time.** Round-4 I-4 noted that §6.5
quoted **raw**-key residuals for a **head-space** comparison, and v5's repair correctly binds the
criterion to the run-time head-space value — but keeps the raw-derived illustration. Measured, at
torch seeds 0/1/2 on both datasets with untrained heads (so weight-dependent, range only):
`θ = 0` residual **`7.451e-09`** (invariant across all six cells); `θ = 45` residual
**`1.863e-07` – `1.974e-07`**, i.e. **~1.6× the raw `1.192e-07`**. The `√d` similarity bound at
`d = 2048` is therefore **`8.43e-06` – `8.94e-06`**, not §6.5's `5.394e-06`. `GATE-ALGEBRA`'s
frozen `2e-6` key-level bar has roughly **10× headroom** against these — the first evidence that
C01's bar transfers to head space at all. → **M-4**.

**(θ) §3.3's near-orthogonality premise holds.** `median cos(native_img, ro_L24_img)` =
**`0.0234`** (HateMM) / **`0.0373`** (MHC-ZH), both caches unit-norm, exactly as claimed. (Text
streams sit at `0.2300 / 0.2495`; §3.3's claim is scoped to the image stream and is correct as
scoped.) Head-N is a genuine out-of-domain transplant, which is what makes C-1's HateMM/MHC-ZH
split a live path rather than a hypothetical.

**(ι) The two uncounted loops, measured.** Reading `meta` + `fold_of` from a banked mint `.npz`
costs **`0.5 ms`**, so `GATE-FOLD`'s new 66-file re-read is `0.033 s`; the arena process's own
ro-cache load is a 67th `U8` at `0.033 s`. Together `0.07 s` = `0.0024 %` of the total. → **I-3**,
which I deliberately do **not** rate Critical; see the reasoning there.

**(κ) The §6 gate table has nineteen rows, not eighteen.** → **M-5**.

---

# PART B — FINDINGS

## CRITICAL

### C-1. Per-lineage gate scoping is defined on the **lineage** axis, but every per-lineage gate is a `(dataset, lineage)` object. On the design's own most likely failure path the two readings publish **CLOSE** and **HALT** on the same event.
*Attaches to:* §5.6 both bullets and all three combination rules; §5.2's opening sentence; §5.3;
§5.7; §6's scope column; §10.2; §15.2.

§5.6 scopes the drop as: *"A lineage that fails one is marked `INSTRUMENT_FAILED` and is
**dropped**"*, and combines as *"**CLOSE** if **both** lineages passed their per-lineage gates and
neither clears."* Nothing anywhere states the **dataset** axis of either clause. But all six
per-lineage gates are per-`(dataset, lineage)` objects, and I verified that each carries a
different constant on each dataset:

| per-lineage gate | HateMM object | MHC-ZH object |
|---|---|---|
| `GATE-ARENA` | band `[0.6203, 0.98]` | band `[0.7091, 0.98]` |
| `GATE-ORBITDISP` | `ρ* = 0.968176` | `ρ* = 0.977223` |
| `GATE-SELFTEST` | `n_D = 743` | `n_D = 579` |
| `GATE-ZEROOP` | cap `7` | cap `5` |
| `GATE-NESTED`, `GATE-ALGEBRA` | 743-row folds | 579-row folds |

**The wrong-verdict path.** Head-N fails `GATE-ARENA`'s lower bound on **HateMM only** and clears
it on MHC-ZH; Head-R passes every per-lineage gate on both datasets and does **not** clear S1–S7.
This is not a contrived case: §5.7 already prices Head-N's `GATE-ARENA` risk at a `6.9 %` recovery
fraction and §6.4 explicitly refuses to invent a bar for the quantity; measurement (θ) confirms
Head-N is near-orthogonal to its features; and the two datasets' bars differ by `0.0888`.

* **Reading A — the lineage is dropped on both datasets.** Rule 2's *"both lineages passed"* fails,
  so rule 3 applies: **HALT**.
* **Reading B — the drop is per `(dataset, lineage)`, so Head-N still "passed" on MHC-ZH.** On each
  dataset there is at least one passing lineage and neither clears: **CLOSE**.

The document supports both. §5.6's prose (*"evaluated within a lineage"*) leans to A; §6's gates
are physically evaluated per dataset, which is B. A CLOSE is terminal — it retires C06's
first-order route and, by `falsifier_spec`, means *"the `1.7-2.5 GPU-h` of extraction is never
queued."* Publishing it where the conservative reading HALTs is precisely the wrong-verdict class
this lineage has spent four rounds removing, and it is **new in v5**: v4's global HALT had no such
seam because a gate failure anywhere voided everything.

Two smaller instances of the same missing axis: (a) §5.2 requires *"S1–S7 all hold on **BOTH**
datasets"* for one arm `A` on one lineage `L` — correct and conservative, but it silently assumes
`L` passed its gates on both datasets, which is exactly what is undefined; (b) `GATE-DOMAIN` and
`GATE-DEVFID` are Head-N-only reporting objects that §6.4 requires *"on the verdict face"*
unconditionally, while §5.6 now permits Head-N to be dropped (harmless on the CLOSE path, which
always retains Head-N, but §6.4's sentence is unsatisfiable on a Head-N-drop SURVIVE or HALT).

**Repair.** State the axis in §5.6, in the conservative direction: *"A lineage that fails a
per-lineage gate on **any** dataset is marked `INSTRUMENT_FAILED` and dropped on **both**; a
lineage 'passed its per-lineage gates' only if it passed every per-lineage gate on **both**
datasets."* Add the `(dataset, lineage)` cross to §6's scope column legend, and to §13's handoff
so a code lineage cannot re-open it. Then re-check §6.4's unconditional reporting sentence.

### C-2. §5.6 contains **two rules that conflict on the lineage-drop path**, and §5.5's frozen Holm family has no defined size on that path. Together they mean round-4 H-3's repair is not delivered.
*Attaches to:* §5.6's combination rule and its *"Finiteness, absence, and crashes"* bullets; §5.5;
§5.7; §8 Phase 4; §12; §14's H-3 row; §15.2.

**(a) The absence rule contradicts the combination rule.** §5.6 says both of these:

> **SURVIVE** if any lineage that **passed** its per-lineage gates clears S1–S7 on both datasets.

> An **absent** decision or gate quantity HALTs on the same footing as a non-finite one.

A dropped lineage's S1–S7 quantities are absent decision quantities. A context-free operator
following the second sentence HALTs on exactly the path the first sentence exists to rescue. This
is the round-4 C-2 pattern verbatim — a table and a footnote in the same subsection giving
different decision rules — recurring in the repair that replaced it. It matters because it is not
a corner: §5.7 names the Head-N drop as *"a recognised third outcome"* and *"a live path"*, so
the design expects to walk it.

Under the absence reading, v5's own claim in §5.6 — *"a clean Head-R SURVIVE is no longer voided by
the transplant lineage's failure"* — is **false**, and §14's H-3 row (*"a failing lineage is
dropped, not the battery"*) describes a behaviour the text does not produce. Under the Critical
definition in force, that is a claimed repair the artifact does not contain.

**(b) The Holm family has no defined size once a lineage is dropped.** §5.5 freezes **one family
per dataset spanning both lineages**, `92` hypotheses, and §8 Phase 4 prices `23 × 2 ds × 2 lin =
92` comparison cells — i.e. it budgets bootstrap comparisons for **both** lineages. If a lineage
is dropped, are its 46 hypotheses computed and kept in the family, or not computed at all? The
document does not say, and the answer changes S4 materially. Executing C01's `holm_adjust`:

* family `92`: a hypothesis at 0-indexed rank ≤ 41 rejects only at `p = 1/2001`;
* family `46`: **every** rank clears at `p = 2/2001`, and only 13 hypotheses need `p ≤ 2/2001`.

So the surviving lineage's S4 is meaningfully easier under a 46-family and harder under a 92-family
padded with a dead lineage's p-values. And if the dropped lineage's hypotheses are simply not
computed, limb (a) fires and the battery HALTs.

**Repair.** (i) Exempt a dropped lineage explicitly: *"the absence rule applies to the lineage(s)
that passed their per-lineage gates; quantities belonging to a dropped lineage are recorded as
`INSTRUMENT_FAILED`, are not required, and are excluded from every decision rule and from the
multiplicity family."* (ii) State the family rule: *"the Holm family is `23 × 2 metrics ×
(number of lineages that passed their per-lineage gates)` per dataset — `92` when both survive,
`46` when one is dropped"*, and re-price §8 Phase 4 as an upper bound at 92 (it already is). (iii)
Fix §5.6's summary sentence and §14's H-3 row to match.

### C-3. **S7 is a binding SURVIVE conjunct whose dominance threshold, reference arm, feature space, modality reduction and seed axis are all unregistered** — and one resolution of the space contradicts §3.6.
*Attaches to:* §5.2 S7 and its two-paragraph note; §3.6; §3.7's constant table; §8 Phase 7;
§13 item 15; §14's H-1 row; §15 (S7 is not among the six open issues).

Round-4 H-1 was right that `require_no_small_displacement_dominance` belongs in C01's `decision`
block and not in `required_halt_only_validity_guards` — I read that seven-entry list verbatim from
`c01_a0_v2.json` and it does not contain it. v5 adopted the promotion correctly. But promoting a
*reported* gate into a *binding conjunct* raises the specification bar, and v5 pre-registers only
four of the seven things `displacement_audit` needs. Executing the frozen source
(`c01_policy_contrast_a0.py:1965-2060`), the missing ones are:

1. **The dominance threshold.** `:1993-1996`: `dominated = full["fixed"] > 0 and fixed_fraction >
   float(transforms["max_small_displacement_fix_fraction"])`. That constant is **`0.5`**
   (`c01_a0_v2.json::transforms.max_small_displacement_fix_fraction`, also the script default at
   `:152`). I grepped v5 for `max_small_displacement`, for `0.5` and for "dominance": **the
   threshold appears nowhere in the document.** §5.2's S7 row names only
   `small_displacement_train_quantile = 0.1`, which defines the *small set*, not *dominance*. An
   un-preregistered threshold touching a decision.
2. **The reference arm.** `:1970-1972`: `strongest_control_name =
   select_strongest_ordinary_control(evaluations, config["decision"]["gain_controls"])` — and
   `:1940-1948` shows the selector is `max` keyed on `(accuracy, macro_f1, −frozen_order_index)`,
   i.e. a **measured-accuracy-dependent selection** over the five `gain_controls`. C01's own run
   selected `common` on HateMM and `endpoint_concat` on MHC-ZH
   (`C01_A0_OUT.json::decision…displacement_stability.reference`, read directly). v5 says nothing,
   and its §3.5 uses *"ordinary controls"* for a **different** two-element set (`endpoint_concat`,
   `common`), so even an inheriting reader gets an ambiguous answer. C01 also runs the selector a
   second time at `:2702-2724`, where `:2724` **dies** if the small-displacement gate's reference
   disagrees with it — a consistency requirement the battery inherits nothing of, and marks
   `endpoint_concat`'s role `diagnostic_only` in a second `concentration()` call v5 does not
   mention.
3. **The space and the reduction.** C01's threshold is
   `np.quantile(np.minimum(d_norm["train"]["img"], d_norm["train"]["text"]), 0.1)` — a per-row
   **minimum over two modality displacement norms**, computed on the raw features. §3.7 says
   *"`0.1` quantile of the **743** displacement norms"* and names the population but **not the
   space**. In head space the deployed head emits a single fused 1024-d vector
   (`classifier.py:140-146`, `--map_dim 1024 --proj_dim 1024 --fusion_mode align`, verified), so
   there is no modality axis and no minimum to take — the object simply does not exist in the form
   C01 defines it. If S7 instead reads on the raw features, then §3.6's *"[the raw leg] renders no
   verdict and **enters no decision rule** or multiplicity family. Its remaining job is `ρ_raw` …
   and the reported raw-vs-head `endpoint_std` comparison"* is false, and the raw leg becomes
   decisional.
4. **The seed axis.** `fix_break` consumes per-seed prediction vectors. S2 and S6 name `3/3`
   seeds; S1/S3 name the seed mean; S4 puts the seed axis inside the statistic. S7 names nothing
   — the same axis omission that produced round-4 C-2, in the conjunct v5 created this round.

There is a further structural point worth pre-registering: in C01 the threshold comes from the
**train** split and the small mask is applied to a **disjoint dev** split, so the small fraction on
the scored population is a free quantity. In C06 the arena **is** the train split, so a `0.1`
quantile of the arena norms makes the small set exactly the bottom decile of the scored population
by construction. That is defensible, but it is a different test from C01's and the record should
say so.

**Repair.** Add to §5.2's S7 row and to §3.7's table: the threshold `0.5`
(`transforms.max_small_displacement_fix_fraction`); the reference = C01's
`select_strongest_ordinary_control` over `decision.gain_controls`, evaluated **within the
`(dataset, lineage)` cell** and recorded on the verdict face; the space and reduction — either
*"the raw per-modality `min(‖d_img‖, ‖d_text‖)`, which makes the raw leg decisional for S7 alone
and amends §3.6 to say so"* or *"the head-space one-block `‖h_ow − h_std‖`, declared as a
deliberate departure from C01's statistic"*; and the seed axis. Extend §13 item 15 to all of them.

---

## HIGH

### H-1. §5.5's resolution floor — *"at least **42 of the 92** comparators must show zero adverse resamples"* — is wrong by ~2×. The true floor is **22** (or 24), and §15.4 asks round 5 to make a freeze decision on `B` on the strength of the wrong number.
*Attaches to:* §5.5's second paragraph; §5.2 S4; §14's H-2 row; §15.4.

The two arithmetic legs are correct and I reproduced both: `1/2001 = 0.00049975 < α/92 =
0.00054348`, and `2/2001` first clears at 1-indexed rank 43. The **inference** does not follow.
"42 of 92" is the condition for **all 92** hypotheses to reject. S4 requires rejection only for
*"every comparator in `C ∪ Θ`"* of the **witness** arm on the **witness** lineage: 12 comparators
× 2 metrics = **24** for `common_displacement`, 11 × 2 = **22** for `displacement`. Executing
`holm_adjust:1775-1784` over 92 with the witness's 24 at `1/2001` and the remaining 68 at `0.5`:
**24/24 reject** and S4 passes. Degrade one witness hypothesis to `2/2001`: 23/24, S4 fails. So the
floor is exactly *"every one of the witness's 22–24 comparators must show zero adverse resamples"*.

This is not pedantry. The direction is **anti-conservative in the record**: the design represents
its own statistical bar as roughly twice as demanding as it is, and §15.4 asks a reviewer to decide
whether `B` should rise on that basis. Round 1's binding condition — the conservative lean *"must
not be allowed to excuse an arithmetic error"* — cuts both ways. Secondarily, the sentence admits a
reading under which **all 92 must reject**, which is a *different decision rule* from S4's text;
that is the two-rules defect again.

**My ruling on §15.4: `B` should NOT rise.** `n_bootstrap = 2000` is C01's frozen constant;
the true floor (22–24 zero-adverse comparators for a single disjunct) is attainable for a genuinely
dominant arm; and raising `B` would depart from the frozen source, change §8 Phase 4's cost, and
buy nothing on the CLOSE path. What must change is the sentence.

**Repair.** Replace with: *"S4 is evaluated within the witness `(arm, lineage)`; its 22 or 24
hypotheses sit at ranks below 42 in the 92-family, so **each of them** must show zero adverse
resamples out of 2000 (`p = 1/2001`) for S4 to pass. This is a property of the frozen `B` and
family size, not of the data."* Then note limb (b) of **C-2**: with one lineage dropped the family
is 46 and the floor changes.

### H-2. **S5's statistic is not pre-registered** — the identical defect round 4 raised as H-2 for S4, fixed for S4 and not carried to S5. S5 is a binding conjunct and, unlike S4, applies to **both** real arms.
*Attaches to:* §5.2 S5; §5.4 (which covers only the bootstrap); §5.5's sentence placing S5 outside
the family; §8 Phase 3; §13 item 16; §11.

§5.2 S5 reads *"**both** real arms exceed the 95th percentile of their shuffled-pair null, **and**
the shuffle comparison Holm-rejects"* and cites two C01 booleans. §5.4 pre-registers the resample,
the per-resample delta, the lower bound, the one-sided `p` and the Holm step-down — **for the
bootstrap only**. For S5 the document never says:

* **what the compared quantity is.** §5.1 defines `acc(A)` as the seed mean; §8 Phase 3 counts
  `256 × 3 seeds × 2 ds × 2 lin`, i.e. a **per-seed** null of 256 draws. Is the p95 taken per seed
  (and then `3/3`?), or is a seed-mean statistic compared against a seed-mean null? The same
  seed-mean/per-seed distinction that produced round-4 C-2.
* **what the shuffle p-value is**, and **what family the Holm step-down runs over**. §5.5 argues
  correctly that S5's rejections belong outside the bootstrap family, but naming what they are
  outside of is not naming what they are inside of.
* **whether S5 is feasible at all at 256 draws.** With C01's `one_sided_raw_p` form the floor is
  `1/257 = 0.0038911`. I computed the rank-0 Holm condition: a shuffle family of `n` members needs
  `n × 0.0038911 ≤ 0.05`, i.e. **`n ≤ 12`**. A family of 4 (`2 arms × 2 metrics`) or 8 (`× 2
  lineages`) is feasible; **13 or more can never reject**, so S5 — and therefore SURVIVE —
  becomes unreachable and the battery can only CLOSE or HALT. That is a design that cannot deliver
  one of its two outcomes, decided by an unwritten family definition.
* **the disposition of C01's `shuffle_fixed_point_bijection` guard.** `n_id_hash_permutations = 256`
  is drawn by `id_hash_permutation(..., fixed_indices=...)`, and `zero_contract_v2.
  require_fixed_null_in_shuffle = True` pins the registered null as a permutation fixed point.
  In C06's arena that row is physically removed, so the guard **has no object** — exactly the
  situation v5 handles explicitly and well for `displacement_registered_null_exclusion` under
  `GATE-ROWSUBSET`, and silently for this one. (`registered_null_absent_from_all_top20` is in the
  same position: two of C01's seven `required_halt_only_validity_guards` are dropped without the
  "has no object here" sentence the third receives.)

**Repair.** Give S5 the §5.4 treatment: the null construction (`id_hash_permutation`, 256 draws,
`fixed_indices = ()` with one clause recording why), the compared statistic and its seed axis, the
p95 convention on 256 draws, the p-value form, and the shuffle Holm family with its member count
**and the `n ≤ 12` feasibility note**. State the disposition of all seven of C01's frozen
halt-only validity guards, not four. Add to §13 item 16.

### H-3. `GATE-ZEROOP`'s **cell granularity is undefined and its `1 %` cap is written in units the comparison does not have** — the round-4 I-6 defect, in the gate next door, surviving the round that fixed I-6.
*Attaches to:* §6 `GATE-ZEROOP`; §6.5's cap bullet; §5.9 item 5; §8 Phase 2z; §13 item 10; §15.6.

Round-4 I-6 found `ρ_head`'s cell granularity undefined and v5 fixed it well: *"per fold, all 60
cells, HALT if any fires"*, with §8 Phase 2D re-counted at 62 to match. `GATE-ZEROOP` compares
**predictions**, which are produced per fold cell — §8's own Phase 2z prices `2 arms × 60 cells =
120` votes, so the comparison is unambiguously per-fold. But §6.5's cap says:

> a mismatch on more than `⌊0.01 × n_D⌋` items (`7` HateMM, `5` MHC-ZH) HALTs regardless.

`n_D` is the **arena** size (743 / 579), while a single fold cell scores only its held-out fifth
(~149 / ~116 queries). So the same integer `7` is a **1 %** cap against a `(seed, lineage)`
aggregate, a **4.7 %** cap against a single fold cell, and a **0.31 %** cap against a
`3 seeds × 5 folds` lineage aggregate. The three readings differ by 15× in the HALT probability,
and §5.9 item 5's justification (*"a declared engineering choice … the cap is one-directional"*)
only makes sense at the `n_D` aggregate the number was written for.

The direction is safe — round 4's ruling that the cap can only convert REPORT → HALT is correct
and I re-verified it against §6.5's closed last bullet — so this cannot invert a verdict. It can
only cause the falsifier not to publish, which for a `$0` falsifier whose entire purpose is to
publish a CLOSE is a material outcome. It also interacts with per-lineage scoping: `GATE-ZEROOP`
is now scope **L**, so a granularity-induced false HALT drops a whole lineage.

**Repair.** State the aggregation explicitly, matching `GATE-ORBITDISP`'s new form: *"mismatches
are counted **per `(dataset, seed, lineage)`**, pooling the five folds' held-out items, so the
denominator is `n_D` and the cap is the `1 %` it is described as; the gate fails its lineage if any
`(dataset, seed)` cell exceeds it."* Add the aggregation to §13 item 10.

---

## IMPORTANT

### I-1. `GATE-SELFTEST`'s identity is written with the symbol §5.1 binds to the **seed mean**, while the gate is asserted **per seed**. Read literally it is false and would drop both lineages.
*Attaches to:* §6 `GATE-SELFTEST`; §5.1; §5.2 S6; §8 Phase 7.

§5.1: *"`acc(A,D,L)` is the **mean over the three seeds**; `net_s(A)` is the **per-seed integer**
net."* §6: *"`net(A) = n_D · (acc(A) − acc(endpoint_std))` exactly for **every one of the 14 arms**,
every seed, dataset and lineage."* The formula uses `acc(A)`, i.e. the seed mean, but is asserted
*every seed*; and it writes `net(A)`, not `net_s(A)`. Under the literal reading the identity is
`net_s = n_D × (seed-mean Δacc)`, which is false in general and would fail on every run — dropping
both lineages and HALTing. This is the same symbol/axis collision v5's own C-2 repair sharpened one
section earlier, left uncorrected in §6.

**Repair.** Write it as `net_s(A) = n_D · (acc_s(A) − acc_s(endpoint_std))` and define `acc_s` in
§5.1 alongside `acc`. Add to §13 item 16.

### I-2. `GATE-ORBITDISP` is scoped per-lineage, but one of its three legs is a **shared-data** check that belongs to no lineage.
*Attaches to:* §6 `GATE-ORBITDISP`; §5.6's classification; §6.1.

The gate asserts three things: the per-fold `ρ_head` comparison (genuinely per-lineage ✓), the
`ρ` row set (per-lineage ✓), and *"`ρ_raw` reproduces §6.1's frozen values at 4 dp"* — which is a
property of the raw leg, the ro caches and the frozen table, and is identical for both lineages. A
`ρ_raw` reproduction failure means the input caches or the builder have drifted; under the current
classification it is evaluated *within* a lineage and would drop that lineage while the other
proceeds on the same drifted data. The outcome happens to be safe only because the check would fail
in both lineages and rule 3 would HALT — but that is luck, not scoping.

**Repair.** Split the leg out: make the `ρ_raw` 4-dp reproduction a **global** clause (naturally,
alongside `GATE-C01PARITY`, which is the other shared-algebra global gate) and leave
`GATE-ORBITDISP` per-lineage for the `ρ_head` comparison only.

### I-3. §8's enumeration is still not literally exhaustive — two loops appear in no phase. I measured both; together they are `0.07 s`.
*Attaches to:* §8 Phases 1c and 7; §2 rule R1; §3.2; §13.

`rule_1_compute_projection` requires *"an enumerated count of **every unit** the run will actually
execute."* Two are absent:

* **`GATE-FOLD`'s banked-`.npz` re-read**, created by v5's adoption of round-4 I-1: §3.2 requires
  re-reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of` from **all 66** banked `.npz`.
  Measured on the banked C09 mints: **`0.5 ms` each, `0.033 s` for 66**.
* **The arena process's own ro-cache load.** Phase 1c prices *"ro cache loads, per process: `66`"*
  — the 66 mints. But the single arena process must also load the ro caches, for the raw leg
  (Phase 2R/2Ra), `GATE-C01PARITY`, `GATE-ROWSUBSET`, `GATE-ZEROMASK` and `GATE-IDPARITY`. That is
  a 67th `U8` at **`0.033 s`**.

Total `0.07 s` = **`0.0024 %`** of `2927.5 s`. **I am deliberately not rating this Critical**, even
though the severity table names *"any un-counted loop in §8"*: both are measured, both are
immaterial, and both fall inside Phase 7's already-declared *"sub-`0.1 s` class"* upper bound. The
honest repair is one clause, not two rows. I record the reasoning explicitly so a later round does
not read a silent omission as a missed criterion.

Two related clarity items in the same phase table: Phase 1b's decomposition `(30×3)+(6×4)+(30×2)`
carries **no legend**, yet round 4 used it as the proof that there are 60 fold mints and §8's own
header repeats that argument — the factors should be named (`30` Head-N fold mints × {native,
ro_std, ro_ow}; `6` Head-N full mints × 4; `30` Head-R fold mints × {ro_std, ro_ow}). And Phase 7
enumerates six gates by name but omits `GATE-FOLD`, `GATE-ZEROMASK` and the `mints_present` check.

**Repair.** Add `GATE-FOLD`'s 66-file re-read and the arena's ro load to Phase 7's enumeration (or
Phase 1c's count as `67`), with the measured `0.07 s` recorded; label Phase 1b's factors.

### I-4. `GATE-FLOOR` is the design's **only** anchor and the dry check never exercised it, although it ran seven real mints at full scale. Its failure is a **global** HALT.
*Attaches to:* §6 `GATE-FLOOR`; §7.2; §5.7; §2 rule R3 (F114); §8's risk paragraph.

`GATE-FLOOR` is global *"because it anchors the shared driver"* — I endorse that classification
(§15.2 below) — which means it is also the single point whose failure voids the whole battery. It
demands that a **new** driver reproduce **42 banked quantities** (6 pooled accuracies + 6 pooled
mF1 + 30 `fold_acc_deployed` entries) at **4 dp exactly**. §7.2 records seven real mints run at
full scale with fold parity passing, and §7.3 records that arm-building heads were untrained — but
nowhere does the dry check compare a re-minted Head-N fold head's accuracy against its own banked
floor. That comparison costs nothing beyond mints already paid for, discloses no arm accuracy
(the floors are already published in §6), and is the one measurement that would tell the design
whether its anchor holds.

R3/F114 requires the dry execution to exercise *"the first real operation of the payload path"* —
satisfied for the mint. But the anchor the entire two-lineage structure hangs on remains asserted,
never measured, and §5.7's *"recognised third outcome"* paragraph names only `GATE-ARENA`'s Head-N
lower bound as the instrument-failure path, not the larger one.

**Repair.** Either (i) discharge it: re-mint one Head-N fold cell through the intended driver path
and record whether `fold_acc_deployed` reproduces at 4 dp; or (ii) declare it: add `GATE-FLOOR`
reproduction to §5.7 as the dominant **global**-HALT risk, state that it is untested, and say what
happens if it fails (the battery HALTs and the falsifier must be re-preregistered). I cannot
discharge (i) myself — it requires head training, which this review is not authorized to run.

### I-5. §5.9 does not disclose that promoting `GATE-SMALLDISP` to S7 makes **CLOSE strictly easier** than v4 — a direction change under §4's binding disclosure condition.
*Attaches to:* §5.9; §4; §6.2's opening argument; §14's H-1 row.

Round-4 H-1's repair is correct and I endorse it: C01 places
`require_no_small_displacement_dominance` in `decision` and not among the seven
`required_halt_only_validity_guards`, so a SURVIVE condition is the right classification. But the
change has a direction. Under v4, *"S1 fails **and** the small-displacement gate fires"* produced
`INSTRUMENT_INCONCLUSIVE`; under v5 the same state produces **CLOSE**. v5 can therefore publish a
`$0` closure in a state where v4 could not. §4 fixes *"conservative"* as *hardest for the falsifier
to deliver the `$0` CLOSURE* and binds the design to *"disclose what the lean buys"*. §5.9 lists
five disclosures and this is not among them, while §5.9 item 1 discloses a much smaller relaxation
of the same kind.

**Repair.** Add a sixth §5.9 item: S7's reclassification is correct on C01's own placement **and**
runs in the CLOSE-easing direction relative to v4; the warrant is that a CLOSE is what S1's failure
warrants, not that the change is neutral.

### I-6. §13's handoff does not carry the one claim that transfers `GATE-C01PARITY`'s guarantee into head space.
*Attaches to:* §3.4; §13; §7.6.

§3.4's warrant is *"**One construction**, parameterised by an ordered list of blocks"*: two blocks
reproduce `prepare_views` bit-exactly, therefore the one-block instantiation inherits the arm→
formula map, the normalisation order, the Givens convention and the dtype. That inference holds
**only if the code really is one function with a block-list parameter**. If the head-space arms are
built by a second, hand-written function, `GATE-C01PARITY` certifies nothing about them and there
is no other anchor — the head-space arms have no independent parity target anywhere in the design.

I hit this directly: reconstructing the builder from §3.4's prose, I built `common_interaction` as
`fuse(ci)` and had to correct it to `paired(common, ci)` against `prepare_views:1334-1338`. The
prose does not fix the map; the parity gate does. So the transfer claim is load-bearing and
unpoliced. §13's eighteen items cover the mint driver, populations, masks, the tie diagnostic,
`GATE-POP`, the heartbeat, the fold axis, the guard arms, S7, the statistics, the constants and
`GATE-FOLD` — but not this.

**Repair.** Add item 19: *"the two-block and one-block builds are the **same function** called with
different block lists — no separate head-space implementation — and `GATE-C01PARITY` runs against
that function, not against a copy."*

---

## MINOR

* **M-1.** §1's `gain_over_strongest_control` `−0.0094` does not reproduce. `C01_A0_OUT.json`
  stores `-0.009345794392523366` (accuracy) and `-0.009827213924986422` (macro-F1); the accuracy
  figure rounds to **`−0.0093`**. The `−0.0094` is propagated verbatim from
  `TARGET_STATE.json::…gated[0].rotation_family_precision_R14` and `GATE0_REOPEN_2026-07-31.md:756`,
  so v5 quotes its source faithfully — but §1 presents the table as *"re-verified … with every
  accuracy recomputed from the stored confusion matrices"*, and this number was not. The MHC-ZH
  figure `−0.0256` is exact (`-0.02564102564102566`). Non-blocking: `pass: false` either way and
  the figure enters no decision. Worth an erratum line given this campaign's standing
  numeric-provenance discipline.
* **M-2.** §3.4 cites `classifier.py:116-120` for *"`classifier_hateClipper` emits a single fused
  1024-d vector"*. `:116-120` is the **two-stream** projection-and-normalise block — the opposite of
  a fused vector. The fusion is `:138-141` (`x = torch.mul(img_feats, text_feats)` under
  `fusion_mode == 'align'`) and the emit is `:146` (`embed = self.mlp[:-2](x)`). The **claim is
  true** — I verified `--map_dim 1024 --proj_dim 1024 --fusion_mode align` at
  `headspace_mint.py:130-131/143-144` and measured the emitted key at 1024-d — only the citation is
  wrong. Same class as round-4 M-1.
* **M-3.** §3.3 and §12 cite *"`:322` writes `lab_dev` into every `.npz`"*. The `np.savez` call
  spans `:322-324` and `lab_dev=dv[3].numpy().astype(int)` is on **`:323`**.
* **M-4.** §6.5's illustrative residual is stale. *"`ε = 1.192e-07` … `√d · ε` at `d = 2048` is
  `5.394e-06` — **45×** the threshold v4 applied"* — the arithmetic is exact
  (`√2048 = 45.2548`), but `1.192e-07` is the **raw**-key figure, and §6.5's own rule correctly
  binds the criterion to the **head-space** residual measured at run time. Measured (η): the
  head-space `θ = 45` residual is `1.863e-07`–`1.974e-07` at three seeds on both datasets, i.e.
  ~1.6× larger, giving a similarity window of `8.43e-06`–`8.94e-06`. Recommend recording the
  measured head-space range beside the raw illustration, and noting that `GATE-ALGEBRA`'s frozen
  `2e-6` key-level bar has ~10× headroom in head space (`θ = 0` residual `7.451e-09`).
* **M-5.** §6's header says *"**Eighteen gates**"*; the table has **nineteen rows** (11 `G` + 6 `L`
  + 2 `R`), and §5.6's own two lists name 11 and 6. v4's table had 21 rows and stated no count;
  removing `GATE-ARMVIAB` and `GATE-SMALLDISP` gives 19, not 18. (Round 4's *"I tested all twenty"*
  was likewise one short of v4's 21, and the miscount has now propagated into the round-5 review
  request. I answer for all nineteen below.) Also, §6.2 says the raw bars are cleared *"by
  `0.15`–`0.23`"* while the supersession header (line 20) says *"`0.18`–`0.23`"*; the measured range
  is `0.1499`–`0.2395`, so §6.2's form is right and the header's is not.

---

# PART C — REQUIRED RULINGS

## Deliverable 6 — is there any gate that can fire on a **warranted CLOSE**?

**NO — for the first time in this lineage, none.** I tested all **nineteen** rows of §6's table
(see M-5), not only the previously flagged ones.

| gate | scope | can it fire on a warranted CLOSE? |
|---|---|---|
| `GATE-DET1` | G | **No** — thread environment, science-independent |
| `GATE-SHA` | G | **No** — provenance; all 21 digests verified |
| `GATE-FOLD` | G | **No** — fold parity; resume-safe both ways (V12) |
| `GATE-FLOOR` | G | **No** — reproduces banked native floors; independent of every arm comparison. Its own failure risk is **I-4**, not this |
| `GATE-POP` | G | **No** — populations and constants; all re-derived exactly (V10) |
| `GATE-C01PARITY` | G | **No** — raw algebra; reproduces bit-exactly (V8) |
| `GATE-ROWSUBSET` | G | **No** — key-level row-subset identity; `0.000e+00` (V7) |
| `GATE-NULLREMOVED` | G | **No** — population predicate |
| `GATE-IDPARITY` | G | **No** — verified to hold directly |
| `GATE-ZEROMASK` | G | **No** — feature-space measurement; `{355}` / `{}` verified |
| `GATE-LEDGER` | G | **No** — bookkeeping; the process count is predictable on every legal path |
| `GATE-ORBITDISP` | L | **No** — measurement (V9) puts a trained head at `0.34`–`0.67` against `ρ* ≈ 0.97`, and the both-above branch is explicitly a warranted CLOSE. The per-fold form raises the test count to 780 but the margin is ~0.30, not marginal. Scope defect only (**I-2**) |
| `GATE-ARENA` | L | **No.** Lower bound is scoped to `endpoint_std` and fires only when the head space itself is dead — the retirement of `GATE-ARMVIAB` makes this literally true, and I confirmed no lower-bound instrument HALT reaches a real arm anywhere in §5, §6 or §12. Upper bound `≤ 0.98` catches a leak and cannot fire downward; against a native OOF floor of `0.888/0.893` a real arm above `0.98` is not a warranted CLOSE |
| `GATE-NESTED` | L | **No** — OOF construction |
| `GATE-SELFTEST` | L | **No** on the science; **I-1** is a notation defect that would fail it on *every* run, including a warranted SURVIVE — symmetric, not CLOSE-directed |
| `GATE-ZEROOP` | L | **No** on the science; disclosed one-directional false-HALT probability, whose magnitude is undefined (**H-3**) |
| `GATE-ALGEBRA` | L | **No** — key-level identity. Measurement (η) gives the head-space residual ~10× under the `2e-6` bar |
| `GATE-DOMAIN` | R | **No** — reports, no bar |
| `GATE-DEVFID` | R | **No** — reports |
| *(S7, no longer a gate)* | — | **No** — it now denies SURVIVE rather than HALTing, which is round-4 H-1's repair working. But it is under-specified (**C-3**) and its promotion eases CLOSE (**I-5**) |

**The recurring wrong-verdict class has moved.** For three of four rounds the answer was "yes, a
gate can fire on a warranted CLOSE." In v5 no gate can. The wrong-verdict risk has migrated to the
**verdict-combination layer** — C-1's undefined dataset axis and C-2's contradictory absence rule —
which is where §5.6's new structure put it.

## §15.1 — `GATE-ARMVIAB`'s retirement

**Right, and the retirement is total.** I verified the argument, the consequence and the sweep.

* **The redundancy argument holds.** `GATE-ARENA`'s lower bound HALTs whenever
  `acc_head(endpoint_std) < majority + 0.02`. A restricted `GATE-ARMVIAB` on `endpoint_std` would
  HALT on that condition **and** the raw counterpart clearing — a strict subset of `GATE-ARENA`'s
  firing set, with the additional (measured) property that the raw conjunct is always true. It could
  never fire when `GATE-ARENA` did not. Deleting it is correct, and stronger than round 4's
  prescription.
* **The premise is measured, not assumed.** V2 confirms the raw real arms clear their bars by
  `0.1499`–`0.2395`, and (θ) confirms the OOF-vs-dev direction argument.
* **The retirement is complete.** `GATE-ARMVIAB` survives in v5 only in §6.2's retirement argument
  and §14's disposition row — I grepped all 21 occurrences of every `GATE-` name. It is absent from
  the gate table, from §5.6's global and per-lineage lists, from every verdict path, from §8's
  phases, from §9's heartbeat granularity and from §13's handoff.
* **The forbidden one-sided HALT cannot be reproduced by `GATE-ARENA` as applied.** Its lower bound
  is scoped to `endpoint_std` and its upper bound cannot fire downward, so §6.3's sentence — *"real
  arms losing badly is a reportable scientific outcome … never an instrument HALT"* — is true for
  the first time. I checked every other §6 clause and §5.2/§5.6 for a re-entering lower bound on a
  real arm and found none.
* **Is the residual watch sufficient for a real arm that is *broken* rather than merely losing?**
  **Yes, and the discriminator swap is the right one.** A broken real arm shows up as a build
  failure (`l2_rows` `die()` → `INSTRUMENT_INCONCLUSIVE`), an algebra failure
  (`GATE-C01PARITY`/`GATE-ALGEBRA`/`GATE-ZEROOP`), a direction collapse (`GATE-ORBITDISP`, now all
  60 folds × 13 arms), a leak (`≤ 0.98`), or a self-test failure — none of which requires an
  accuracy floor on the arm. The one state that is *not* caught is a real arm that builds cleanly,
  is non-degenerate, and simply scores near the majority rate; and that state is the falsifier
  working, which §6.2 argues correctly and which is the whole reason `GATE-ARMVIAB` had to go.

## §15.2 — per-lineage gate scoping, gate by gate

**The global/per-lineage classification is right on the lineage axis, and incomplete on the dataset
axis (C-1). One leg is mis-scoped (I-2).**

* **The 11 global gates are all correctly global.** Each governs provenance, population, shared
  algebra or bookkeeping that is identical for both lineages: a failure in any of them means the
  inputs, the row sets, the raw algebra or the process ledger are wrong, and no lineage's verdict is
  trustworthy.
* **`GATE-FLOOR` global is the right reading**, and for a stronger reason than *"it anchors the
  shared driver"*: **Head-R has no independent anchor at all.** Its only warrant is that it goes
  through the same function as the lineage that does reproduce six banked floors (§13 items 3 and
  4). If `GATE-FLOOR` were per-lineage, a driver defect would drop Head-N and leave Head-R running
  on an unanchored path — the worst available outcome. Global is correct, and the design should say
  so in these terms rather than the weaker one.
* **The 6 per-lineage gates are all correctly per-lineage**, with one exception: `GATE-ORBITDISP`'s
  `ρ_raw` 4-dp reproduction leg is shared-data and belongs global (**I-2**).
* **The combination rule preserves the conservative lean, and a CLOSE cannot be rendered on one
  lineage** — rule 2 requires *both* lineages to have passed, so a CLOSE always rests on two clean
  negatives. **Provided C-1 is fixed.** Without the dataset axis, reading B of C-1 does exactly what
  rule 2 forbids: it renders a CLOSE where one lineage was clean on only one dataset.

## §15.3 — S6's binding form and the `7×` scale transfer

**Confirmed, and the `3/3` transfer is right. It is not too tight.** I verified the counterexample
exactly (V3) and endorse v5's corrected statement: S3 bounds `mean_s net_s ≥ 14.86 / 11.58`, says
nothing about `min_s net_s`, and the counterexample `(2, 21, 22)` sits inside ordinary seed noise
at `2.69` accuracy points of spread.

On the transfer question the request poses — *is `3/3` on an integer minimum frozen for a `7×`
smaller arena now too tight?* — **no**, for three reasons. (i) The integer got **easier**, not
harder: `+3` was `+2.80 %` accuracy at `n_dev = 107` and is `+0.40 %` at `n = 743`, so as a
*level* the bar has been relaxed by a factor of 7 relative to C01. (ii) All that remains binding is
across-seed dispersion, which is a property C01 never tested (single arena, no seed axis) and which
this campaign's own history says matters — the `3/3` requirement is the cheapest available guard
against a single-seed artifact, the exact failure class the memory index records as *"pillar-④ EN
deletion = single-seed correction"*. (iii) It is symmetric with S2, which already requires `3/3` on
S1's accuracy leg; making S6 weaker than S2 would be arbitrary. Keeping S6 binding at `3/3` is
correct and costs nothing.

## §15.4 — S4's statistic and the resolution floor

**S4's statistic is correctly pre-registered; the resolution-floor statement is wrong (H-1); `B`
should not rise.**

Checked against C01: the resample form matches `paired_bootstrap`'s item resampling; `5 %` is
`bootstrap_lower_quantile`; the one-sided `p = (1 + #{Δ_b ≤ 0})/(B+1)` is `:1769` verbatim; the
Holm step-down is `holm_adjust:1775-1784` (`min(1, (total − rank) · p)` with a running max), and I
executed it. §5.4's departure from C01's function is correctly identified and correctly justified —
`paired_bootstrap` evaluates `metric_value` on sampled *scores* with no seed axis, so it genuinely
cannot be reused, and putting the seed average inside `c̄_X(i)` keeps the seed axis out of the
multiplicity. Sharing draw indices across comparators within a `(dataset, lineage)` is the right
choice for a paired family. **The specification is unambiguous enough for a code lineage to check**
— with one gap the request did not ask about but which belongs here: it does not say whether the
shared draw indices are also shared **across lineages** within a dataset, which matters because the
two lineages' comparisons live in one Holm family. One clause.

The `42 of 92` claim is wrong by ~2× (**H-1**); the true floor is 22–24 for the witness disjunct;
and under C-2's family question the floor changes again. **My ruling: `B = 2000` stays.**

## §15.5 — the corrected enumeration, and the hunt for the next uncounted loop

**The 60-cell counts are right and the whole of §8 re-multiplies.** I derived every count
independently from folds × seeds × lineages × datasets × arms × draws before looking at the table,
then diffed: Phase 1's `15/3` split per dataset ✓; Phase 1R's `15/15` ✓; 66 mints ✓; Phase 2's
`4×60 / 9×60` matching the measured key dims `{1024: 4, 2048: 9}` ✓; Phase 2b's `60` ✓; Phase 2z's
`120` votes and `60 × 2/13` construction share ✓ (the `2/13` share is the right accounting — two
extra arms built alongside thirteen, at the same per-arm cost, and the direction is conservative
since both guard arms are 2048-d while `U10` averages over the 4/9 split); Phase 2R's `10` raw fold
cells with no seed axis ✓ (correct — the raw leg has no head); Phase 2D's `2 + 60` ✓ (`ρ_raw` is
per `(arm, dataset)`, not per fold, which matches §6.1's 26-value table); Phase 3's
`256 × 3 × 2 × 2` with arms and folds inside `U4` ✓ and `256 = n_id_hash_permutations` ✓;
Phase 4's `23 × 2 × 2` ✓; Phase 7's `14 × 3 × 2 × 2 = 168` ✓.

**The hunt succeeds, but small.** Two loops appear in no phase — `GATE-FOLD`'s 66-`.npz` re-read
and the arena process's own ro-cache load — and I measured both at `0.033 s` each (**I-3**).
Nothing else in §5 or §6 iterates over folds without appearing in §8: I checked `GATE-NESTED`
(per item, inside Phase 7), `GATE-ZEROOP`/`GATE-ALGEBRA` (Phase 2z), S7 (Phase 7), the shuffle null
(folds inside `U4`), and the bootstrap (seeds inside the statistic). Phase 1b appears to
*over*-count rather than under-count, since `headspace_mint`'s own `keys_of(tr)` is already inside
the measured mint unit — the direction is conservative and I do not raise it.

## §15.6 — the tie diagnostic's head-space residual

**Pre-registration-safe: no decision can be tuned by it.** The residual is an *instrument*
quantity, not a decision quantity: it widens or narrows the tie set inside `GATE-ZEROOP`, whose only
effects are REPORT and HALT. Round 4's one-directional ruling is correct and I re-verified it
against §6.5's closed final bullet (*"any mismatch outside them HALTs"*) — so a larger measured
residual can only convert HALT → REPORT within the cap, and the cap can only convert REPORT → HALT.
Neither direction reaches SURVIVE or CLOSE. Binding the criterion to a run-time measurement rather
than a frozen constant is therefore safe **and** more correct than v4's frozen raw number, which
measurement (η) shows understates the head-space value by ~1.6×.

Two conditions on that ruling. (i) The residual must be **recorded** on the verdict face alongside
the tie-casualty count, so it is auditable after the fact rather than a hidden run-time free
parameter; §6.5 does not say it is. (ii) The **granularity** must be fixed (**H-3**) — a run-time
residual with an undefined denominator is tunable in effect even if not in intent.

## §3.C — does anything in §5 still contain two decision rules at once?

**Yes — §5.6, and it is C-2.** The combination rule and the absence rule give opposite outcomes on
the lineage-drop path. The specific v4 defect round 4 found (§5.2's table vs its S6 footnote) is
**gone**: §5.2's table now lists S1–S7 as the conjunction and every accompanying note reinforces
rather than contradicts it. Two further near-misses that are *not* findings: A2's disjunction over
arms and §5.2's *"there exists `A ∈ R`"* agree; §5.3's CLOSE and §5.6's rule 2 agree word for word.

## §3.F — honesty and completeness

**Does v5 claim any repair the artifact does not contain?** **One, and it is C-2(a):** §5.6 and
§14's H-3 row claim that a failing lineage is *"dropped, not the battery"*, which the absence rule
in the same subsection undoes. Every other claimed repair is physically present and I verified all
18 by execution (Part D).

**Blindness (§7.3): intact.** I repeated round 4's method across **all five** drafts and classified
the union of 97 distinct decimals in `[0.6, 0.99]`. Every one is a `ρ` (arena or trained-head), a
`‖head_f(0,0)‖` magnitude (`0.58`–`0.65`), a v2-era cos/`‖Δ‖` geometry figure (`0.674`–`0.773`), a
banked `GATE-FLOOR` anchor, a published C01 dev-arena accuracy from §1's table, a majority/band/
threshold constant, or a unit-time arithmetic string (`0.04674`). **No arm accuracy produced by
this battery appears anywhere in v1–v5.** §6.1's trained-head reference reads `K_train` only; I
reproduced it and it computes no accuracy.

**Emitter- and weight-dependent quantities: correctly ranged.** §7.4(a) gives `0.58–0.65` scoped to
its emitter with the alternative convention recorded, and only *"non-zero"* load-bearing — I
measured `0.581950 / 0.597144 / 0.584977`, inside the band. §7.5 records all three rounds' values
for the one-block `paired` with only the invariant claim carrying weight. Both adoptions are real.
Measurement (η) adds a third quantity of this class (the head-space algebra residual) which should
be recorded the same way (**M-4**).

**Hard constraints: none touched.** I re-read
`iteration_8_stage0_bounded_extraction_amendment.amended_rule.conditions.d_no_other_relaxation` and
checked each: no OCR; no cross-dataset mixing (the two-dataset requirement is a conjunction of
independently computed verdicts, and I confirmed no pooled object survives across datasets — every constant in
§3.7 is per dataset); no external API; single-dataset train split; parent-video binary label only;
no ensemble (`avg_score` is C01's own frozen `gain_control`, verified in
`decision.gain_controls`); no size scaling; SLURM-only, no `--time`. §10.4's ban analysis is
unchanged and correct against the Gate-0 record's `why_gated_not_struck` text, which I read
verbatim. §10.3's statement that a SURVIVE authorises no GPU is correct against condition `a`.

**§10.2's scope.** Round-4 I-8's Givens bullet is present and is the strongest narrowing available;
I confirmed `orthogonal_blocks:1272` is a Givens mixing, that the `θ = 0` and `θ = 45` identities
hold at `8.94e-08`/`1.19e-07` on the raw features, and that §1's table does show 4 of 6 HateMM and
2 of 6 ZH rotations below the primary. The only thing §10.2 does not yet scope is the outcome of
C-1/C-2 — which lineage(s) and which dataset(s) the verdict rests on when a drop occurred.

---

# PART D — DISPOSITION AUDIT OF §14's ROUND-4 BLOCK, BY EXECUTION

**18 of 18 VERIFIED ADOPTED. 0 NOT ADOPTED, 0 PARTIAL.** Each row checked against the primary
source or by execution, never against §14. Special attention to adoptions that could have broken
each other — the mechanism that produced round-4's C-2.

| finding | audit result |
|---|---|
| **C-1** `GATE-ARMVIAB` | **VERIFIED ADOPTED — retired, totally.** I grepped all 21 `GATE-` names across v5: `GATE-ARMVIAB` survives only in §6.2's retirement argument and §14's row. Absent from the gate table, §5.6's two scope lists, every verdict path, §8, §9 and §13. The redundancy argument is sound (§15.1) and its premise is measured (V2). §6.3's invariant is now literally true — I checked every §5/§6 clause for a re-entering lower bound on a real arm and found none. |
| **C-2** S6 binding | **VERIFIED ADOPTED.** *"Reported, not screening"* is gone; S6 is a conjunct in §5.2's table with no contradicting footnote; §5.9 item 4 is rewritten with the verified counterexample and the correct seed-mean/per-seed statement (V3). **Collision check:** this adoption is the one most likely to have broken another, since it sharpens the seed axis. It did not break §5.9's other items — but §6's `GATE-SELFTEST` was **not** brought along and still writes the seed-mean symbol (**I-1**), and S7, created in the same round, has no seed axis at all (**C-3** limb 4). |
| **C-3** fold axis + guard arms | **VERIFIED ADOPTED and arithmetically true.** Phase 2b `12 → 60`, Phase 2D `14 → 62`, new Phase 2z; total `2927.5 / 3659.4 s`. I re-derived every count independently and re-multiplied every product; the printed column sums exactly to the total (V4). The `+2.5 s` guard-arm construction round 4 offered as an option is taken, in the conservative direction. §9 carries the per-cell 2D line. |
| **H-1** `GATE-SMALLDISP` → S7 | **VERIFIED ADOPTED.** Removed from §6's table; present as S7 in §5.2 with C01's placement argued from the seven-entry `required_halt_only_validity_guards` list, which I read verbatim and which does not contain it. Arm scope and zero-fix convention pre-registered. **Residual: C-3** — the promotion raised the specification bar and four elements are still missing; **I-5** — the direction change is undisclosed. |
| **H-2** S4's statistic | **VERIFIED ADOPTED.** §5.4 pre-registers the resample, the per-resample delta, the `5 %` lower bound, the one-sided `p` and the Holm step-down, all matching C01's source. **Residual: H-1** — §5.5's resolution-floor inference is wrong; **H-2 (mine)** — the identical repair was not carried to S5. |
| **H-3** lineage scoping | **VERIFIED ADOPTED in text (option ii).** §5.6 scopes gates global/per-lineage, §6 carries a scope column, §5.7 names the third outcome, §10.2 names the lineage(s). **But the repair opened C-1 and C-2**, and under C-2(a) it does not deliver the behaviour §14 claims. |
| **I-1** `GATE-FOLD` under resume | **VERIFIED ADOPTED, by execution.** §3.2 and §6 carry the two-way discharge. I confirmed `:192-194` returns before `:203-216`, that `meta["fold_parity_vs_banked_vsw_ckpt"]` is present in every banked `.npz` (`[True]×5`), that `fold_of` is banked, and that `:321-325` writes only after the assertion. The re-read costs `0.5 ms` per file — free, as claimed, but uncounted in §8 (**I-3**). |
| **I-2** `GATE-DOMAIN` populations | **VERIFIED ADOPTED.** `maj_arena = 0.6003 / 0.6891` with `acc_ro`; `maj_full = 0.5995 / 0.6891` with the banked `acc_native`; both in §3.7's table and both re-derived exactly (V10). §6.4 states the pairing. |
| **I-3** quantile population | **VERIFIED ADOPTED — population named.** §3.7's table and §5.2's S7 both say the arena. **But the adoption named the population and not the space**, and H-1's adoption in the same round made S7 binding — the two together are **C-3**. This is the round-4 pattern (*"textually present but arithmetically false"*) recurring in a different form: textually present but under-determined. |
| **I-4** tie units | **VERIFIED ADOPTED.** §6.5 uses `‖Δk‖₂` or its `√d` bound, defines "collapse" as the worst case over orderings of every near-tie group, states the residual is the head-space one measured at run time, and takes the maximum over the two identities. All four prescriptions present. **Residual: H-3** (granularity) and **M-4** (stale illustration). |
| **I-5** `dev_path_opens` | **VERIFIED ADOPTED, by execution.** §12 reads `mints_executed + 0`. I grepped `headspace_fidelity.py`: **zero** occurrences of `dev_seen`; it takes `lab_dev` from the mint `.npz` at `:66`. |
| **I-6** `ρ_head` granularity | **VERIFIED ADOPTED.** Per fold, all 60 cells, HALT if any fires; `60 × 13 = 780` values stated; §8 Phase 2D counted at the same granularity; §9 carries a per-cell line. The conservative and structurally correct choice, as prescribed. |
| **I-7** `avg_score` | **VERIFIED ADOPTED.** §6 says *"every one of the 14 arms … including `avg_score`"*; §8 Phase 7 reads `14 × 3 × 2 × 2 = 168`. `avg_score ∈ gain_controls` verified in `c01_a0_v2.json`, so it is a comparator for both real arms and a Holm-family member, consistent with V11's `23`. |
| **I-8** Givens bullet | **VERIFIED ADOPTED.** §10.2's sixth bullet carries §1's round-14 sharpening, quotes the Gate-0 record's own adverse-reading sentence, and cites the 4-of-6 / 2-of-6 counts, which I verified against `C01_A0_OUT.json`. |
| **M-1** `classifier.py:81-82` | **VERIFIED ADOPTED** — `:80` is the comment, `:81-82` are the two biased `nn.Sequential(nn.Linear(...), nn.Dropout(...))` projections, `:115` is `def forward`. (A *different* citation in §3.4 is now wrong — **M-2**.) |
| **M-2** Phase 7 rounding note | **VERIFIED ADOPTED** — the stale note is retired and the printed column sums to the total directly (V4). |
| **M-3** emitter scope | **VERIFIED ADOPTED** — both emitter conventions recorded, only *"non-zero"* load-bearing; I measured within the stated band. |
| **M-4** heartbeat span / `GATE-ROWSUBSET` population | **VERIFIED ADOPTED** — §9 gives one line per `(gate, dataset)`; §3.7's table marks `GATE-ROWSUBSET` **HateMM only**, matching §8 Phase 2C's count of `1`. |

**Round-4 measurements α–ζ:** α, β, γ, δ, ε all folded in and all independently re-confirmed here
(V6, V10, V12). ζ is folded in but its inference is wrong (**H-1**).

**Rounds 1–3, spot-checked:** the direction of *"conservative"*; A7; per-arm retraining excluded;
`max` as `ρ*`'s order statistic; SLURM and the login-node dismissal; §5.9 item 1's inapplicability
reasoning (I verified the comparator: `historical_strict_devtrain.deployed_r0_accuracy_context_only`
= `0.8504672897` / `0.8589743590` from `READOUT_SCREEN_OUT.json` at `n_dev` 107/78 — a raw dev-arena
figure, as claimed); the tie cap's one-directionality; `GATE-ROWSUBSET`'s renaming; and §3.4's
account of what two-block parity does and does not buy. All present and still sound.

---

# PART E — FREEZE-READINESS AND THE RUN BOUNDARY

**Everything the document says exists, exists, and everything it says is absent, is absent.** All
ten `vsw_ckpt/{hatemm,zh}/f{0..4}.npz`; all six `headspace_arena_*_OUT.json`; exactly **36**
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`. All four items of new code
(`c06_falsifier_mint.py`, `c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`,
`c06_falsifier_cpu.sbatch`) are **absent**, as they must be. `TARGET_STATE.json`'s `falsifier_spec`
and `falsifier_design_constraints` quotations at §1 are **verbatim-correct**, character for
character. Every C01 constant §5 and §6 cite reproduces from `c01_a0_v2.json`, including
`n_id_hash_permutations = 256`, `holm_metrics = ['accuracy','macro_f1']`,
`bootstrap_lower_quantile = 0.05`, `minimum_net_fixes {HateMM: 3, MHC_zh: 2}`,
`max_small_displacement_fix_fraction = 0.5` (**which the draft does not carry — C-3**), and
`execution {require_slurm, cpu_only, required_cpus: 8}`.

**`rule_2_heartbeat`: satisfied at the corrected counts.** I checked every phase against the
`~15 s` claim. The longest un-instrumented span is one `GATE-C01PARITY` dataset at `11.27 s`
(`14.1 s` conservative), as stated. Phase 2D is now `0.62 s` per cell; Phase 2Ra is `4.63 s` per
dataset; Phase 2C's `GATE-ROWSUBSET` is `4.84 s`; Phase 3's `32`-draw blocks are `2.85 s`; Phase 6
is `3.70 s` per `(dataset, seed)`; a mint's per-epoch line is ~`1.3 s`; a process's pre-first-line
gap is imports (`3.05–3.18 s`) plus loads. **No interval exceeds ~15 s.** The `RuntimeError`
wrapper, the `buffering=1` handle, the unbuffered driver echo and the frozen `elapsed ÷ projected`
denominator are all specified consistently across §5.6, §9 and §13 item 12.

**`rule_1_compute_projection`: satisfied in substance, with I-3's clause outstanding.** Every unit
is measured on the real path at real scale — the mints by `/usr/bin/time -v` and `date` brackets,
`U4` as one full draw, `U3` at the real `B = 2000`, `U10` at real `n = 743` — and no figure is
extrapolated from a reduced-draw run. The `U9` exit-status defect and the `Phase 1e` non-addition
are both correctly reasoned.

**What is not yet freeze-ready.** Six things a context-free operator cannot execute unambiguously:
(1) whether a per-lineage gate failure on one dataset drops the lineage on both (**C-1**);
(2) whether a dropped lineage's absent decision quantities HALT, and what the Holm family becomes
(**C-2**); (3) S7's threshold, reference, space, reduction and seed axis (**C-3**); (4) S5's
statistic, family and feasibility (**H-2**); (5) `GATE-ZEROOP`'s cell granularity (**H-3**);
(6) `GATE-SELFTEST`'s per-seed accuracy symbol (**I-1**).

**The run boundary is otherwise unambiguous** — one `sbatch`, 8 CPU / 32 GB, no
`--gres`/`--time`/array/dependency/requeue, 73 processes in the order 66 mints → 6 fidelity →
1 arena, `GATE-SHA` once in the driver before any of them, `GATE-POP` before any
population-consuming gate — and the cloud-routing dismissal is correct: `GATE-FLOOR` anchors to six
floors measured locally on `foscsmlprd01`, so CLAUDE.md's same-table-same-hardware condition cannot
be met off-node.

# PART F — ADDITIONS TO §13's HANDOFF

The eighteen items are good and should be kept verbatim. Four to add, plus two extensions:

19. **The one-construction claim (I-6).** That the two-block and one-block builds are the **same
    function** with different block lists, that no separate head-space builder exists, and that
    `GATE-C01PARITY` runs against that function. This is the sole warrant transferring the parity
    guarantee into head space, and the head-space arms have no other anchor.
20. **The `(dataset, lineage)` cross (C-1).** That every per-lineage gate is evaluated per
    `(dataset, lineage)`, that the drop propagates across datasets as §5.6 will now specify, and
    that no verdict path can be reached with a lineage that passed on one dataset only.
21. **The dropped lineage's quantities (C-2).** That a dropped lineage's decision quantities are
    exempt from the absence rule, are excluded from the Holm family, and that the family size the
    code uses is the one §5.5 will now specify.
22. **The key-forward site.** That the ro-cache forwards producing `h_std`/`h_ow` happen inside the
    mint process — `headspace_mint` suppresses state-dict saves, so the head weights never leave it
    — and that §8 Phase 1b's `174` is priced against that placement.

*Extensions:* item **15** must cover S7's threshold `0.5`, its reference-selection rule, its space
and reduction, and its seed axis. Item **16** must cover S5's statistic, its family, and whether
bootstrap draw indices are shared across lineages as well as across comparators.

# PART G — MINIMAL SET OF CHANGES THAT WOULD EARN GO

1. **C-1** — one sentence in §5.6 fixing the dataset axis of the drop, in the conservative
   direction, plus the scope-column legend.
2. **C-2** — one clause exempting a dropped lineage from the absence rule, one sentence fixing the
   Holm family size on that path, and a corrected §5.6 summary and §14 H-3 row.
3. **C-3** — four constants and a space in §5.2's S7 row and §3.7's table, plus the §3.6
   reconciliation if S7 reads raw.
4. **H-1 … H-3** — correct the resolution floor to 22/24; give S5 the §5.4 treatment with its
   `n ≤ 12` feasibility note and the two orphaned C01 guards; state `GATE-ZEROOP`'s aggregation.
5. **I-1 … I-6** and **M-1 … M-5** as written.

**Can the falsifier still discharge its written condition at `$0`? Yes.** Nothing I found requires
a GPU, an extraction, new data or a redesign, and the corrected projection — `48.8` corroborating /
`61.0` conservative CPU-minutes on 8 CPU / 32 GB — is right; I re-derived it from the design's own
structure rather than from its table. The instrument is, after five rounds, verified end to end by
execution: the three-population contract, the mask convention, the row-subset identity, the arena
constants, the `ρ*` calibration, the two-block anchor, the head-space build and its dimensions, and
now the head-space algebra residual. **Every remaining defect is in the decision layer** — how two
lineages combine, and what S5 and S7 actually say. That is a narrower failure surface than any
previous round, and none of it is expensive to fix.

---

*Read-only review. No GPU, SLURM, Modal, model load, head training, arena run, cache write,
test-split access, job submission or commit occurred. `TARGET_STATE.json`, all five drafts, all
configs and all four prior reviews were read and not modified. All computation was `sha256sum`,
file reads, and numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked
mint checkpoints, in the session scratchpad. No arm accuracy was computed at any point. A GO on
this lineage would authorise nothing to run: the design would still require freeze with hashes, a
**separate** independent code/resource review lineage over the executable reaching its own
`0C/0H/0I`, and main-dialogue authorization.*
