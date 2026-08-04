# C06 `$0` falsifier — independent design review, **ROUND 3**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V3.md` (DRAFT v3, 2026-08-04)
**Reviewer posture:** fresh, independent of rounds 1 and 2 and of the designer. Read-only. No
GPU, SLURM, Modal, model load, head training, arena run, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json`, all three drafts and all configs were read
and not modified. Nothing heavier than `sha256sum`, file reads, and numpy/torch-CPU
re-derivations on already-banked **train-split** caches was executed.

---

# VERDICT

## **REVISE (2C / 1H / 6I)** — plus 4 Minor

Not `GO (0C/0H/0I)`.

**Disposition audit: 16 of the 16 round-2 findings verified as truly adopted, and the three
reopened round-1 items (C-3, the C-1 companion, I-3) are genuinely repaired.** I found no
disguised rebuttal and no claimed repair that the artifact does not contain. That is a clean
sweep and a real change from round 2, which found three of twenty-three not adopted.

The ceremony floor is clean and I re-derived it rather than reading it: **all 21 sha256 digests
reproduce**, both provenance chains are sha-gated in source, **all 26 `ρ_raw` values reproduce
to 4 dp**, the six `GATE-FLOOR` anchors match the banked JSON on both metrics, the Holm
arithmetic is right, and §8 re-multiplies. Most importantly, **v3's two load-bearing new
measurements are true and I reproduced both from scratch**: the null-contract defect is real
(`head_f(0,0)` non-zero at three seeds; `common_displacement` unbuildable in head space under
either mask), and the row-subset identity is **bit-exact at `0.000e+00` on all 13 arms**, with
every `ρ` unchanged. The three-population contract is the right design.

**Both Criticals are new, both were found by executing C01's frozen code rather than by reading
the document, and both live in the seam the new contract opened.**

**C-1 is the round-2 pattern recurring one gate over.** v3's null contract is written in terms
of `zero_mask = None`, and `GATE-C01PARITY` specifies invoking `prepare_views` at
*"`n = 579, None` (ZH)"*. **`prepare_views` cannot be called with `zero_mask = None` on any
dataset.** Its own derived-mask check (`c01_policy_contrast_a0.py:1381-1386`) compares an
exact-zero boolean array against the raw `zero_mask` argument, and `np.array_equal(array, None)`
is `False`, so the call dies unconditionally. I measured it: on MHC-ZH,
`prepare_views(..., None)` dies with *"derived exact-zero mask preservation failed"*; C01 itself
never passes `None`, it always builds `np.zeros(n, dtype=bool)` (`:2224`, call site `:2303`).
`GATE-C01PARITY` is a HALT gate on the verdict path, so as written the battery HALTs on MHC-ZH
and the two-dataset conjunction can never complete.

**C-2 is the population leak the round asked me to hunt for, and it is in the one constant
nobody re-derived.** Removing row 355 changes HateMM's class balance: the arm arena's majority
rate is **`0.600269 → 0.6003`**, not the `0.5995` measured on 744 rows. `GATE-ARENA`'s lower
bound and `GATE-ARMVIAB` both apply `majority + 0.02` — a **744-banked constant** — to
accuracies computed on the **743** population. `GATE-POP` cannot detect this, because it checks
realised populations and not whether population-dependent *bars* were derived on them. That is
the direct answer to §15.1: the three-population split does leak, in exactly one place, and
`GATE-POP` is not sufficient to catch it.

Neither Critical requires a GPU, an extraction, or a redesign. C-1 is a calling convention;
C-2 is one constant. The instrument, once they are fixed, does measure what `falsifier_spec`
asks.

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All **21** digests recomputed with `sha256sum` and matched character-for-character: 7 imported modules, 6 read-for-definition files, 8 input caches. No mismatch. Spot values: `headspace_mint.py` `cefdf8dc…0916612`; `c01_policy_contrast_a0.py` `d2b9c2ff…8db1b855`; HateMM `ro_L24` `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f`. |
| **V2** | **VERIFIED** | Config chain in JSON: `v4.frozen_v3` → `c01_a0_v3.json` (`4ddb0f6f…`) → `scientific_base` → `c01_a0_v2.json` (`f3997bdd…`), `scientific_thresholds_exact: true` at both hops. **In source:** `c01_policy_contrast_a0_v4.py:52-55` `load_frozen_v3()` raises *"frozen v3 analysis source SHA256 drift"* unless `_v3.py` matches `V3_SOURCE_SHA256`; `_v3.py:48-51` `load_frozen_base()` raises *"frozen v2 analysis source SHA256 drift"* unless `c01_policy_contrast_a0.py` matches `BASE_SOURCE_SHA256 = d2b9c2ff…8db1b855` — the file this battery imports. Both hops gated in code, not only in config. |
| **V3** | **VERIFIED — the defect is real, and I reproduced every leg** | `classifier_hateClipper.__init__` (`src/model/classifier.py:81-82`) builds `img_proj = nn.Sequential(nn.Linear(3584,1024), nn.Dropout(...))` with the default bias. Using `headspace_arena.py:9`'s own emitter definition `head_f = mlp[:-2](l2n(img_proj(img)) * l2n(text_proj(text)))`: **`‖head_f(0,0)‖ = 0.581950 / 0.597144 / 0.584977`** at torch seeds 0/1/2 — non-zero at every seed (see **M-2** on v3's `0.634676`). `h_std[355] == h_ow[355]` **exactly**, and `h_std[355]` is **not** all-zero (`‖·‖ = 0.581950`). Row 355 is the **only** row with `std == ow` bit-identically in both modalities, so no second row creates the same condition. `l2_rows`: endpoint block DIES at `{355}`, OK at `None`; common block DIES at `{355}`, OK at `None`; displacement block OK at `{355}`, DIES at `None`. The full 13-arm head-space build **dies under both masks** — at `{355}` on the first endpoint block, at `None` on the first displacement block. `common_displacement` is therefore unbuildable in head space under either mask. |
| **V4** | **VERIFIED** | All **13** head-space arms build at `n = 743, zero_mask = None` through the imported `l2_rows`, at torch seeds 0, 1 and 2, `dtype = float32`. Headroom against `normalization_epsilon = 1e-12` is enormous: min head-displacement row norm `1.05e-02 … 1.12e-02` across seeds, i.e. `≈ 1e10 ×` epsilon, so the fail-closed epsilon leg of `l2_rows` is not a live risk even under substantial training-induced contraction. |
| **V5** | **VERIFIED — the load-bearing measurement is exact** | Raw arms at `n = 743` vs `n = 744, {355}` restricted to the 743 surviving rows: **`max|diff| = 0.000e+00` on all 13 arms**, both via `prepare_views` (with the admissible all-`False` array) and via v3 §3.4's builder at `None`. The `n = 743` algebra guard is bit-identical to the `n = 744` guard (`8.940696716308594e-08` / `1.1920928955078125e-07`). Every `ρ` is unchanged to **`0.000e+00`**. **One caveat the document should state:** the invariance holds when the null is excluded from the mean. Computing `ρ` over the realised 744-row array *including* the masked zero row shifts values by up to **`1.301e-03`** (`endpoint_std`), which would fail `GATE-ORBITDISP`'s own 4-dp reproduction leg — fail-safe, but see the code-lineage list. |
| **V6** | **VERIFIED** | I re-implemented §3.4's `fuse` / `paired` / `build_views` from the prose alone, calling C01's imported `l2_rows`, and compared against `prepare_views` on the real raw L24 features: **`max|diff| = 0.000e+00`, all 13 arms, both datasets**, `dtype float32` on both sides. The spec is reproducible by a third party from the document text — the second independent confirmation of this, after round 2's measurement (α). |
| **V7** | **VERIFIED — all 26 values, exact** | Per-dataset maxima `0.968176` (HateMM, `endpoint_std`) and `0.977223` (MHC-ZH, `endpoint_std`); runner-ups `0.964446` and `0.969686`, both `common` — matching §6.1's `0.9644` / `0.9697`. All 26 `ρ_raw` entries reproduce at 4 dp exactly as tabulated, including `common_interaction`'s cross-dataset asymmetry (`0.9138` / `0.9682`) and `displacement` as the least concentrated arm on both datasets (`0.8917` / `0.9091`). Unit-norm verified on every arm. See **I-2** on the truncation. |
| **V8** | **VERIFIED, with a consequence v3 does not draw** | HateMM `n = 744`, `pos = 298`, `posrate 0.400538` ⇒ majority **`0.599462 → 0.5995`**; MHC-ZH `n = 579`, `pos = 180`, `posrate 0.310881` ⇒ majority **`0.689119 → 0.6891`**. `ids[355] == "hate_video_95"`, `label = 1`, and under `StratifiedKFold(5, shuffle=True, random_state=0)` — fold parity `[True]×5` against the banked `vsw_ckpt/hatemm/f{0..4}.npz` — row 355 is held out in **fold 4**. **On the arm-arena population it is `n = 743`, `pos = 297`, majority `0.600269 → 0.6003`.** → **C-2**. |
| **V9** | **VERIFIED** | Re-read from all six `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`: `acc_deployed` HateMM `0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946`; `mF1_deployed` HateMM `0.8838 / 0.8811 / 0.8812`, ZH `0.8747 / 0.8710 / 0.8765`. Identical to §6's anchors and to C09's banked table (`C09_A0_V17_RECORD.md:1511-1522`). Every `fold_acc_deployed` array is present in the same files. |
| **V10** | **VERIFIED** | `gain_controls` = 5; `bootstrap_comparisons.primary_vs_controls` = 6 (those plus `displacement`). Union for `common_displacement` = 6, for `displacement` = 5. `(6+6) + (5+6) = 23`; `23 × 2 metrics = 46` per `(dataset, lineage)`; `× 2 lineages = 92` per dataset. §8 Phase 4's `23 × 2 ds × 2 lineages = 92` also equals 92 — a coincidence of two different products (see **M-4**). |
| **V11** | **VERIFIED (one stated convention)** | Every product re-multiplied: `15×40.39=605.85→605.9`; `3×49.30=147.9`; `15×34.40=516.0`; `3×38.87=116.61→116.6`; `15×40.39=605.9`; `15×34.40=516.0`; `174×0.0461=8.0214→8.0`; `66×0.033=2.178→2.2`; `1×0.12→0.1`; `240×0.00305=0.732→0.7`; `540×0.00629=3.3966→3.4`; `30×0.00305=0.0915→0.1`; `12×0.1873=2.2476→2.2`; `40×0.04239=1.6956→1.7`; `90×0.08098=7.2882→7.3`; `2×4.63=9.26→9.3`; `2×11.27=22.54→22.5`; `1×4.84→4.8`; `14×0.62=8.68→8.7`; `3072×0.08908=273.654→273.7` (M-1 adopted); `92×0.126=11.592→11.6`; `3×3.70+3×3.49=21.57→21.6`. **No product is wrong.** The printed column sums to `2886.2`; `2886.3` follows only if Phase 7's *"< 0.1 s"* is counted as `0.1` (**M-1** below). `×1.25 = 3607.875 → 3607.9` ✓; `48.105 → 48.1 min` ✓; `60.132 → 60.1 min` ✓. Mint share `2508.3/2886.3 = 86.90 %` ✓. Phase 3 share `9.48 % → 9.5 %` ✓. Sensitivity `3160.0` and `3981.1` ✓. **No phase is ratio-derived.** |
| **V12** | **VERIFIED** | `C09_A0_V17_RECORD.md:1569-1572` reads *"**Pooled native accuracy** must satisfy `majority_rate + 0.02 ≤ acc ≤ 0.98` on both datasets (C02 `ARENA2` convention): HateMM majority `0.5995` ⇒ band `[0.6195, 0.98]`; MHC-ZH majority `0.6891` ⇒ `[0.7091, 0.98]`."* It is scoped to the **floor arm** and never to a treatment arm, so v3 §6.3's restriction of the lower bound to `endpoint_std` matches the precedent it cites. **But C09's constant is the majority of the population C09 scores**, and v3 changes that population without changing the constant — **C-2**. |

## Additional measurements I made, which v3 does not report

**(a) `prepare_views` and `zero_mask = None`.** MHC-ZH `n = 579`, `zero_mask = None` ⇒
**DIES**, *"zh/probe derived exact-zero mask preservation failed"*. With
`np.zeros(579, dtype=bool)` ⇒ **OK**. HateMM with `None` dies earlier, at the first endpoint
block, as expected when a real null is present. → **C-1**.

**(b) The arm-arena majority rate.** HateMM at `n = 743`: `pos = 297`, `neg = 446`, majority
**`0.600269`**. The `GATE-ARENA` band under the document's constant is `[0.6195, 0.98]`; under
the arena's own majority it is `[0.6203, 0.98]`. Seed-mean accuracies are multiples of
`1/(3 × 743) = 0.000449`, and **two achievable values — `1381/2229 = 0.619560` and
`1382/2229 = 0.620009` — fall inside the gap**, so the discrepancy is capable of converting a
warranted instrument HALT into a published verdict. → **C-2**.

**(c) `ρ` on the *banked trained* deployed-head keys — and it vindicates `GATE-ORBITDISP`.**
I was concerned that a bar calibrated on the raw feature family would fire routinely on head
keys, because at an **untrained** head I measure `ρ_head(endpoint_std) = 0.99997` and **10 of
13 arms above `ρ*`** on both datasets. That turns out to be an initialisation artifact, exactly
as round 1's I-10 warned. On the **real trained deployed heads** banked at
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_{ds}_s{0,1,2}_f{0..4,full}.npz::K_train`
(18 cells per dataset), `ρ = ‖mean_i l2n(K_i)‖` is:

| dataset | min | median | max | `ρ*` | cells above `ρ*` |
|---|---|---|---|---|---|
| HateMM | 0.447803 | 0.562434 | 0.632996 | 0.9681 | **0 / 18** |
| MHC-ZH | 0.340179 | 0.574247 | 0.667326 | 0.9772 | **0 / 18** |

A trained deployed head does **not** concentrate; it sits at roughly *half* the bar. So
`GATE-ORBITDISP` will not fire on ordinary head-induced concentration, and a head-space `ρ`
above `0.9681` really would be anomalous. Round 2's §15.2 ruling is correct and now has a
measurement behind it rather than only an argument. **This measurement is label-free and
computes no arm accuracy, so it does not touch §7.3's blindness discipline.** I recommend v3
cite it in §6.1 — it is the strongest available defence of the bar and it costs nothing.

**(d) The one-block instantiation, on my own head weights.** `fuse([b])` differs from `l2(b)`
by **`7.451e-09`** (matching v3 §7.5 exactly); the one-block `paired` differs from v1's rejected
`pair` by **`3.725e-09`** (v3 records `1.118e-08`, round 2 measured `7.451e-09`). All three are
a fraction of a `float32` eps (`1.192e-07`), so the shared conclusion holds; the spread is head-
weight dependent → **M-3**.

**(e) `headspace_mint.py` opens `dev_seen` on *every* mint.** `:199`
`dv = load_split(cache_dir, "dev_seen", model_name)` is unconditional and `:322` writes
`lab_dev=dv[3]` into every `.npz`. `model_name` comes from the frozen dataset table, so a
`--train-cache` override leaves the dev load pointed at the **native** dev cache. → **H-1**.

---

# PART B — FINDINGS

## CRITICAL

### C-1. `zero_mask = None` is not an admissible argument to `prepare_views`, so `GATE-C01PARITY` **cannot execute as written on MHC-ZH** — and the `None` convention is the vocabulary the whole null contract is written in.
*Attaches to:* §6 `GATE-C01PARITY` clause (i) and `GATE-DUALPATH`; §3.7's contract table; §7.4 rows (g)/(h); §7.6.

§6 specifies `GATE-C01PARITY` as *"(i) the two-block builder reproduces `prepare_views`
**bit-exactly** at `n = 744, zero_mask = {355}` (HateMM) and **`n = 579, None`** (ZH)"*. That
second call dies.

`prepare_views` (`c01_policy_contrast_a0.py:1381-1386`) closes with

```python
derived_masks = {
    arm: bool(np.array_equal(np.all(value == 0, axis=1), zero_mask))
    for arm, value in views.items()
}
if not strict_all(derived_masks.values(), context + "/derived_zero_masks"):
    die("{} derived exact-zero mask preservation failed".format(context))
```

and it compares against the **raw `zero_mask` argument**, not against the normalised array
`l2_rows` builds internally at `:1187-1188`. `np.array_equal(<bool array>, None)` is `False` for
every arm, so the gate raises `RuntimeError` on **any** dataset when handed `None` —
independently of whether that dataset has a null row. Measured on MHC-ZH at `n = 579`:
`zero_mask = None` ⇒ *"zh/probe derived exact-zero mask preservation failed"*;
`np.zeros(579, dtype=bool)` ⇒ **OK**. C01 itself never passes `None`: `validate_raw_zero_contract`
builds `expected_mask = np.zeros(n, dtype=bool)` (`:2224`) and `analyse_dataset` hands that array
to `prepare_views` (`:2304-2306`).

This is round 2's C-1 pattern in a new location: a HALT gate on the verdict path whose written
invocation cannot run. Its consequences are the same. Implemented literally, the battery HALTs
on MHC-ZH at `GATE-C01PARITY`, the two-dataset conjunction never completes, and the `$0`
falsifier cannot discharge its written condition. Implemented "sensibly" by a coder who
substitutes the array, the executed gate differs from the pre-registered gate — which is the
thing a hash freeze exists to prevent.

**Two things this finding does *not* impugn.** The head-space builder is unaffected: it calls
`l2_rows` directly, and `l2_rows:1187-1188` *does* convert `None` to a zeros array, which is why
V4's 13-arm build at `n = 743, None` genuinely succeeds. And §7.6's claim that the parity holds
*"both datasets"* is true — I reproduced it — which means the designer's own dry check must have
used the admissible form. **The measurement is right; the specification of it is wrong.**

**Repair.** State the convention once, in §3.7's contract table, and use it everywhere: the
`zero_mask` argument is **always an explicit `np.zeros(n, dtype=bool)`**, with the single
exception of HateMM's `n = 744` build where it is the one-hot `{355}`. Replace *"`n = 579,
None`"* in `GATE-C01PARITY` and *"`zero_mask = None`"* in `GATE-DUALPATH`, §3.7 and §7.4(g)/(h)
with that wording, and add one sentence recording **why** `None` is inadmissible to
`prepare_views` (`:1381-1386` compares against the raw argument) while it is admissible to
`l2_rows` (`:1187-1188` normalises it) — otherwise the distinction will be lost again in the
code lineage. Then re-state §7.4(g)/(h) as what was actually executed.

### C-2. The `GATE-ARENA` / `GATE-ARMVIAB` majority rate is a **744-population constant applied to the 743 arm arena**, and `GATE-POP` cannot detect it. The instrument-health lower bound is `0.0008` looser than the arena's own data supports, in the anti-conservative direction.
*Attaches to:* §6 `GATE-ARENA`, `GATE-ARMVIAB`, `GATE-POP`; §6.2; §6.3; §7.1; §3.7's contract table; §15.1.

§7.1 measures *"HateMM `posrate 0.4005` ⇒ majority **`0.5995`**"* — correct, and I reproduce it
(`pos = 298`, `n = 744`, `446/744 = 0.599462`). §6 then uses that number as the bar for two
gates that are evaluated on the **arm arena**, which §3.7 sets at `n = 743`:

* `GATE-ARENA`: *"**lower** bound `majority + 0.02 ≤ acc` on **`endpoint_std` only**… Majority
  `0.5995` / `0.6891`"*;
* `GATE-ARMVIAB`: *"if head-space accuracy fails `majority + 0.02`, HALT **iff** the same arm
  clears it in the raw space"*.

Row 355 carries **label 1**. Removing it removes a positive, so the arena's class balance is
`pos = 297`, `neg = 446`, `n = 743`, and its majority rate is **`0.600269 → 0.6003`**. The
correct band is `[0.6203, 0.98]`; the document's is `[0.6195, 0.98]`.

**Three reasons this is Critical rather than cosmetic.**

1. **It is a wrong-verdict path, not merely an imprecision.** `GATE-ARENA`'s lower bound is the
   gate round 2 ruled *substantively discharges* round 1's C-2 — the OOD-transplant fidelity
   check (§6.4, endorsed in round 2's §3.D ruling). It is therefore *expected to operate near
   its bar*: the whole point is that the `ro_L24` forward may collapse `endpoint_std` toward the
   majority rate. A bar `0.0008` too low means a collapsed instrument can pass and publish a
   CLOSE. And the gap is reachable: seed-mean accuracy is a multiple of `1/(3×743) = 0.000449`,
   and `1381/2229 = 0.619560` and `1382/2229 = 0.620009` both sit inside `[0.6195, 0.6203)`.
2. **It runs the wrong way under §4's declared lean.** Loosening a HALT gate makes closure
   *easier*, and §4 fixes "conservative" as *hardest for the falsifier to deliver the `$0`
   closure*. Round 1's ruling on that lean attached the condition that it *"must not be allowed
   to excuse an arithmetic error"*.
3. **`GATE-POP` does not cover it, and §15.1 asks exactly this.** `GATE-POP` asserts *"the
   realised populations equal §3.7's table exactly"* — it checks row sets, not whether a
   population-dependent **bar** was derived on the population it is applied to. The majority
   rate is the only such bar in the design, and it is precisely the one that silently kept its
   744-row value. §3.7's own closing sentence — *"the folds, the heads, the floors and every
   threshold are per-dataset already"* — is true and beside the point: the defect is
   per-*population*, not per-dataset.

**Repair (three lines, no measurement needed).** (i) Add a row to §3.7's contract table naming
the **arena majority rate** as a population-derived constant, frozen at **`0.6003` (HateMM,
`446/743`)** and **`0.6891` (MHC-ZH, `399/579`, unchanged)**, with the full-population `0.5995`
retained *only* for any statement about the 744-row population. (ii) Update `GATE-ARENA` and
`GATE-ARMVIAB` to cite the arena constant, and update §6.3's band. (iii) Extend `GATE-POP` — or
add one clause to it — to assert that the realised arena class counts equal `(297, 446)` /
`(180, 399)`, so the constant is checked against the population at run time rather than assumed.
That clause is what makes `GATE-POP` sufficient for the question §15.1 poses, and it is free.

---

## HIGH

### H-1. §12's *"Head-R mints open no dev file"* is **false under §3.3's own shared-driver claim**, and the two are mutually exclusive: either the ledger's declared dev counts are wrong, or the driver is not shared.
*Attaches to:* §3.3; §12 `GATE-LEDGER` rows `dev_path_opens` and `dev_label_materialisations_outside_decisions`; §14 row **H-2**.

§3.3's repair for round 2's H-2 — and the entire basis on which §15.5 asks round 3 to confirm
the sufficiency ruling — is:

> **ONE driver, `scripts/analysis/c06_falsifier_mint.py`, serves both lineages.** It imports
> `headspace_mint` … and reuses its dataset table, deployed CLI, fold assignment, fold-parity
> assertion, dummy-dataloader construction, monkeypatches, seeding and DET-1 contract
> **unchanged** … **Its only lineage-varying argument is `--train-cache`.**

§12 then states: *"Head-R mints open no dev file."* Both cannot be true.
`headspace_mint.py:199` is `dv = load_split(cache_dir, "dev_seen", model_name)` — **unconditional
on every mint**, before the `fold` branch — and `:322` writes `lab_dev=dv[3].numpy().astype(int)`
into **every** `.npz`. `model_name` is read from the frozen dataset table
(`mechnov_pairverify.DATASETS`), so a `--train-cache` override that changes only the *training*
cache leaves the dev load pointed at the **native** `dev_seen_*.pt`. Under a driver that reuses
that code unchanged, all **66** mints open a dev file and materialise dev labels, not 36.

Consequently §12's declared counts are wrong on two rows: `dev_path_opens` (*"36 Head-N mints +
`GATE-DEVFID` reads"*) and `dev_label_materialisations_outside_decisions` (*"36, one per Head-N
mint"*) should read **66**. Neither is a binding leg, so this does not HALT and does not reach a
decision quantity — the binding `dev_or_test_labels_into_decision_quantities = 0` is unaffected,
and no dev label enters any Head-R science, because at `fold ≥ 0` `dev_sp` is a slice of the
fitting pool (`:223-226`) and only `dv[3]` is written to disk. So this is **not** a wrong-verdict
path.

It is High for two other reasons. First, it is the **same defect round 1 raised as H-4** —
*"§12's dev-label claim is false"* — recorded as VERIFIED adopted by round 2, now recurring for
the lineage v2 introduced after that audit. Second, and more importantly, it is a **live test of
§3.3's load-bearing sentence**. If the designer's intent is that Head-R skips the dev load, then
`--train-cache` is *not* the only lineage-varying argument, `headspace_mint`'s code is *not*
reused unchanged, and the anchor argument that §15.5 rests on is weaker than stated — which is
precisely the failure mode round 2 found in v2 (*"that sentence was false of v2's dry check"*).

**Repair.** Keep the shared driver, correct §12: state that all **66** mints open the **native**
`dev_seen` cache and write `lab_dev`, that Head-R opens no `dev_seen_*-ro_*` file (true, and
worth saying), and set the two declared counts to 66. Then either make both counts **binding**
under §5.6's absence rule — they are now exactly predictable — or say why they remain reported.
If instead the driver is to suppress the dev load on Head-R, say so explicitly, name it as a
second lineage-varying behaviour, and re-argue §15.5 without the *"only variable"* sentence.

---

## IMPORTANT

### I-1. `GATE-DUALPATH` is a **different property wearing C01's name**. C01's `displacement_registered_null_exclusion` is a masked-vs-removed **prediction** equivalence; v3's gate is a **key-level row-subset identity**.
*Attaches to:* §6 `GATE-DUALPATH`; §3.7 point 3; §15.4.

`c01_a0_v2.json::transforms.displacement_registered_null_exclusion` is
`"with_null_masked_vs_physically_remove_null_dual_path_exact"`, and `zero_contract_v2` pairs it
with `require_remove_null_exact_equivalence: true`,
`require_displacement_null_exclusion_dual_path_exact: true` and
`require_null_absent_from_all_top20: true`. C01's property is that the **votes and metrics** of
the masked path and the physically-removed path agree exactly, and it holds in the raw space for
a reason v3 never states: **in raw space the null's key is exactly zero in all thirteen arms**,
not just in `displacement`, because `prepare_views`' `derived_masks` check (`:1381-1386`) forces
every arm to preserve the exact-zero mask — which is why `require_null_absent_from_all_top20`
holds and the two paths cannot diverge.

v3's `GATE-DUALPATH` asserts something else: that the 743-row arm matrices equal the 744-row
matrices restricted to 743 rows. That is **stronger at the key level** and I verified it exactly
(`0.000e+00`, V5). But it is not C01's gate, and calling it *"C01's
`displacement_registered_null_exclusion` dual-path equivalence, applied where it is defined"* is
an over-claim: C01's version is defined at the *vote* level, and v3 runs no vote at `n = 744`.

**Ruling for §15.4:** the substance is sound and I would keep the gate exactly as specified —
but it should be **renamed** (`GATE-ROWSUBSET` is the honest name) and cited as *"strictly
stronger than C01's dual-path property at the key level, and sufficient to license the
population change"*, rather than as C01's property. **Is C01's original property still needed
anywhere?** No, and the document should say so: the only population where a masked path exists
is the `n = 744` build inside `GATE-C01PARITY`, which is a parity comparison and never votes, so
C01's vote-level equivalence has no object in this design.

### I-2. `ρ*`'s downward truncation makes `endpoint_std` — the instrument-health arm — **structurally exempt from `GATE-ORBITDISP` on both datasets**, and §6.1's two tables give the same arm two different 4-dp values.
*Attaches to:* §6.1 both tables; §6 `GATE-ORBITDISP`; §14 row **I-7**.

`GATE-ORBITDISP` HALTs iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D`. §6.1 freezes `ρ*` = **`0.9681`**
(HateMM) and **`0.9772`** (MHC-ZH), truncated down from the measured maxima per round 2's I-7
prescription. But the arm that *supplies* the maximum is `endpoint_std`, whose measured `ρ_raw`
is `0.968176 > 0.9681` and `0.977223 > 0.9772`. So the second conjunct is **false by
construction for `endpoint_std` on both datasets**, and that arm can never trigger the
degeneracy HALT however concentrated its head-space keys become.

That is a coverage hole in the one arm `GATE-ARENA`'s lower bound and `GATE-DOMAIN` both single
out as the instrument-health arm. Measurement (c) above says the hole is unlikely to matter for
a healthy trained head (`ρ ≈ 0.5`), so I am not raising it higher — but it is coverage the
document believes it has and does not.

Separately, §6.1's **`ρ_raw` table lists `endpoint_std` HateMM at `0.9682`** while the `ρ*` row
describes the same arm as *"`0.9681` (`endpoint_std`, `0.968176`)"*. `GATE-ORBITDISP` also
requires *"`ρ_raw` reproduces §6.1's frozen values at 4 dp"* — under rounding that is `0.9682`,
so the document asks the run to reproduce `0.9682` and to compare it against a bar of `0.9681`
derived from the same number.

**Repair.** Freeze `ρ*` at the **full measured precision** — `0.968176` / `0.977223` — which
removes the exemption (`ρ_raw ≤ ρ*` becomes true for `endpoint_std` by equality), removes the
two-table inconsistency, and keeps the calibration label-free and banked exactly as round 2
required. Keep the runner-ups on the face of the document as they are.

### I-3. **S6 is vacuous**: `GATE-SELFTEST`'s own identity makes S3 imply S6 by arithmetic, so the net-fix minima can never bind and §5.2's claim to import the Gate-0 record's currency is decorative.
*Attaches to:* §5.2 S3/S6; §6 `GATE-SELFTEST`; §14 round-1 row **I-5**.

S3 requires `acc(A) − max_{c∈C} acc(c) ≥ 0.02`, and `endpoint_std ∈ C` on both real arms'
comparator sets (§5.1). S6 requires net fixes `≥ 3` (HateMM) / `≥ 2` (MHC-ZH) **against
`endpoint_std`**. `GATE-SELFTEST` asserts `net(A) = n · (acc(A) − acc(endpoint_std))` exactly.
Therefore S3 ⇒ `net(A) ≥ 0.02 × 743 = 14.86 ⇒ ≥ 15` on HateMM and `≥ 0.02 × 579 = 11.58 ⇒ ≥ 12`
on MHC-ZH — five to six times the frozen minima. **S6 cannot fail whenever S3 holds.**

The reason is a scale transfer nobody priced: C01 froze `minimum_net_fixes` on an arena of
`n_dev` 107 / 78, where `+3` net fixes is `+2.8 %` accuracy; carried unchanged to `n = 743`, the
same integer is `+0.40 %`. Round 1's I-5 asked for the dropped C01 conditions to be listed and
justified; S6 is *carried*, not dropped, so it escaped that audit — but the carrying is what
made it inert.

This is not a wrong-verdict path (a vacuous conjunct cannot cause a false SURVIVE). It is an
honesty defect: §5.2 and round 2's Part C both present S6 as bringing the battery *"into the
currency the Gate-0 record demands — NET ITEMS"*, and it does not.

**Repair.** Keep S6 — it costs nothing and `GATE-SELFTEST` needs its object — but say plainly in
§5.8 that at `n = 743 / 579` the frozen minima are **implied by S3** and therefore report rather
than screen, and that the net-item figure is carried as a **reported quantity in the Gate-0
currency**, not as a binding bar. If a binding net bar is wanted, it must be re-derived at the
arena's scale, which is a new threshold and needs its own justification.

### I-4. Two decision quantities are population- and axis-ambiguous: `GATE-SELFTEST`'s `n`, and whether S6's net count is per-seed or on the seed mean.
*Attaches to:* §5.2 S6; §6 `GATE-SELFTEST`; §5.1.

`GATE-SELFTEST` is stated as *"`net(A) = n · (acc(A) − acc(endpoint_std))` holds **exactly** for
every arm, seed, dataset and lineage"*. With three coexisting populations, `n` is now
under-determined: the identity holds **only** at `n = 743` on HateMM, and fails by one item's
worth if a coder reaches for the banked `744`. Because the gate demands *exactness*, the wrong
`n` produces a guaranteed HALT — fail-safe, but a HALT that publishes no verdict is exactly the
outcome §6.2 and §6.3 were rewritten to avoid.

Separately, §5.1 defines `acc(A,D,L)` as the **seed mean**, while `GATE-SELFTEST` is stated
per-seed. S6 says neither, and the two readings differ: a seed-mean net is non-integer, so
*"net fixes ≥ 3"* is ill-typed under §5.1's own definition.

**Repair.** Write `n = |arena(D)|` explicitly in `GATE-SELFTEST` and pin it to §3.7's table; and
state in S6 whether the count is the per-seed net required in `3/3` seeds, or `n × (seed-mean
Δacc)` rounded. Given I-3 this changes no outcome, but both are un-preregistered elements
touching a gate and a decision condition.

### I-5. `GATE-ZEROOP`'s tie diagnostic is under-specified in two places and **does not cover the dominant flip mechanism** it was introduced to excuse, so round 2's I-5 repair is fail-safe but largely inert.
*Attaches to:* §6.5; §6 `GATE-ZEROOP`; §15.6.

The diagnostic reads: *"emit the number of affected items whose 20th/21st neighbour similarities
differ by less than the measured `GATE-ALGEBRA` residual. A mismatch confined to such items is
REPORTED, not HALTed."*

**(a) Two referents are missing.** *Whose* 20th/21st neighbours — `orthrot_0`'s ranking or
`endpoint_concat`'s? They are the two rankings under comparison and they differ precisely on the
items in question. And *which* `GATE-ALGEBRA` residual — the gate measures two
(`endpoint_concat_vs_theta0` and `common_displacement_vs_theta45`; I measure
`8.941e-08` / `1.192e-07` on HateMM and `8.941e-08` / `8.941e-08` on MHC-ZH). Neither is
recoverable from the text, and both change which items qualify.

**(b) It watches the wrong boundary.** `mechfix_ops.deployed_vote` weights neighbours by
**descending integers `[20…1]`**, so the score changes whenever *any adjacent pair inside the
top-20* reorders, not only when the 20/21 boundary swaps. A `~1e-7` key perturbation reorders
in-set adjacent pairs at least as often as boundary pairs, and an in-set reordering of two
neighbours with opposite labels moves the vote by `~2 × sim`, which is the larger effect. Such
items are **not** in the diagnostic's tie set, so they HALT. The design is therefore fail-safe —
good — but the false-HALT probability round 2's I-5 identified is only partly reduced, and §6.5
implies otherwise.

**(c) No cap.** Nothing bounds how many items may be excused. A systematic defect that happens
to leave boundary near-ties would be reported rather than HALTed.

**Ruling for §15.6: the branch cannot be *widened* by an implementer — the criterion is
narrower than the failure it excuses, which is the safe direction — but it is not sharp enough
to be checkable by the code lineage as written.**

**Repair.** Name the ranking (I recommend the **union** of the two arms' top-21 sets, which is
the conservative choice), name the residual (the **maximum** of the two, likewise conservative),
and replace the 20/21 criterion with the one that matches the operator: an item is a **tie
casualty** iff recomputing its rank-weighted vote after collapsing every pair of neighbours whose
signed similarities differ by less than the residual leaves the two arms' predictions equal.
Add a pre-registered cap — a mismatch on more than a stated fraction of items HALTs regardless.

### I-6. A `die()` raised inside `l2_rows` during arm construction is an **unhandled `RuntimeError`, not a gate failure**, and no §5.6 or §9 path records it as `INSTRUMENT_INCONCLUSIVE` naming a gate.
*Attaches to:* §5.6; §9; §6.

§5.6 makes every gate failure a HALT *"recorded as `INSTRUMENT_INCONCLUSIVE`"*, and §9's free
addition is that *"the HALT path names **which gate** failed in its final line, so a HALT is
distinguishable from a crash without reading the JSON"*. But `l2_rows` and `prepare_views`
signal by `die()` → `RuntimeError` (`c01_policy_contrast_a0.py:392-393`), which is a crash, not a
gate result. C-1's `None` defect and any epsilon or exact-zero surprise inside arm construction
all arrive by that route.

The headroom measurement in V4 (`1e10 ×` epsilon) says the epsilon leg is not a live risk, so
this is about the *record*, not the science: as written, the single most likely instrument
failure in this battery would be indistinguishable from a wedged process, which is the precise
observability defect `rule_2_heartbeat` was adopted to prevent.

**Repair.** Pre-register that every call into the imported C01 algebra is wrapped, that a
`RuntimeError` from it is recorded as `INSTRUMENT_INCONCLUSIVE` with the `context` string
(`l2_rows`' first argument already carries the arm and block name) written to both the decision
JSON and the final heartbeat line, and add the wrapper to §9's code-lineage list.

---

## MINOR

* **M-1.** §8's corroborating total is `2886.3 s`, but the printed product column sums to
  **`2886.2 s`**; the difference is Phase 7's *"< 0.1 s"*, which the total counts as `0.1`.
  Every downstream figure (`×1.25`, the shares, the sensitivities) is consistent with `2886.3`,
  so nothing moves — state the convention in one clause (*"Phase 7 is carried at `0.1 s`, its
  upper bound"*) so a reader who re-adds the column is not left with a `0.1 s` discrepancy.
* **M-2.** §7.4(a) records `‖head(0,0)‖ = 0.634676` to six digits. That value is
  **initialisation-dependent**: I measure `0.581950 / 0.597144 / 0.584977` at torch seeds
  0 / 1 / 2 with the deployed dims. §7.3 already says the structural result is weight-independent;
  §7.4(a) should say *"non-zero at every seed tested; the value is initialisation-dependent
  (`0.58–0.64` observed)"* rather than quoting six digits that no reviewer will reproduce.
* **M-3.** §7.5's *"the one-block `paired` differs from v1's rejected `pair` by `1.118e-08`"* is
  likewise head-weight dependent — round 2 measured `7.451e-09`, I measure `3.725e-09`. Only the
  invariant claim (*"a fraction of a `float32` eps"*) should carry digits. `fuse([b])` vs `l2(b)`
  at `7.451e-09` **does** reproduce exactly, so that one is safe to quote.
* **M-4.** §5.5's `92` (hypotheses per dataset = `23 comparisons × 2 metrics × 2 lineages`) and
  §8 Phase 4's `92` (bootstrap cells = `23 × 2 datasets × 2 lineages`, both metrics inside `U3`)
  are two different products that coincide. Both are correct — I verified each — but a reader
  will assume one is derived from the other. One clause distinguishing them prevents a future
  round from "reconciling" them into an error.

---

# PART C — REQUIRED RULINGS

## §3.A (deliverable 6) — is the null-removal contract verdict-neutral, and is the three-population arrangement free of silent mixing?

**Verdict-neutral: YES, and I tested the symmetry rather than accepting it. Free of silent
mixing: NO — in exactly one place, C-2.**

**(i) The removal is a pure row-subset, provably.** V5: `0.000e+00` on all 13 arms, both build
paths, with the `n = 743` algebra guard bit-identical to the `n = 744` guard and every `ρ`
unchanged. There is no algebraic content to the population change. This is the strongest form
the claim could take and v3 earns it.

**(ii) The symmetry argument holds, and it is stronger than §3.7 states.** The question §15.2
poses is whether removing one item from a 743-item bank can shift a top-20 neighbourhood in a
way that favours either lane. It cannot, for three independent reasons. The row is removed from
the bank **and** the query set of **every arm identically**, so both lanes lose the same item
from the same index. Its own query row is dropped from all 13 arms' scoring, so all arms are
scored on the identical 743 items. And because row 355 is held out in **fold 4** (V8), its
removal from the bank touches only queries in folds 0–3 — again identically across arms. A
per-lane bias would require the removal to be arm-dependent, and it is not.

**(iii) Leaving the null in really would bias the comparison, and v3 under-states why.** In
**raw** space there is no asymmetry at all: `prepare_views`' `derived_masks` check
(`:1381-1386`) forces *every* arm to carry an exact-zero key at 355, so the null is absent from
every top-20 in every arm. The asymmetry is purely a **head-space** phenomenon, created by the
bias term. And it is worse than *"contributes nothing"*: a zero key has inner product `0` with
every query, which under `faiss.IndexFlatIP` ranks **above** any negatively-similar candidate,
so in the head-space `displacement` arm the null would not merely abstain — it would **displace
a real neighbour out of the top-20** while contributing `(2y-1) × 0 = 0` to the vote. Removal
dissolves that. §3.7 point 2 is right and could be argued harder.

**(iv) Label-freeness and pre-registration: clean.** Row 355 is selected by an exact-zero
*feature* property, is C01's pre-existing frozen `authorized_null`
(`c01_a0_v2.json::zero_contract_v2`: `authorized_row_index 355`, `authorized_id hate_video_95`,
both policies, both modalities), and I confirmed it is the **only** row on either dataset with
`std == ow` bit-identically in both modalities — so the selection is forced, not chosen. Nothing
in the contract depends on a trained-head number: the defect turns on the *presence* of a bias
term, which I confirmed is weight-independent by reproducing it at three seeds.

**(v) The mixing that does occur.** Fold assignments are computed on the full `n` and asserted
against the banked `vsw_ckpt` — correct, and I verified parity `[True]×5` on both datasets. The
six floors are anchored on the full `n` with native keys — correct. `GATE-C01PARITY` runs at
`n = 744` and `GATE-DUALPATH` bridges to 743 — correct. `ρ_raw`, `GATE-ARMVIAB`'s raw-vs-head
comparison and the head leg are all on 743 — correct, and §3.6 says so. **The single leak is the
majority rate**, a 744-derived constant used as the bar for the two gates that are evaluated on
743 (**C-2**). `GATE-POP` as written cannot see it, because it validates populations and not the
provenance of population-dependent constants.

## §15.1 — do the three populations mix, and is `GATE-POP` sufficient to detect it?

**They mix in one place, and `GATE-POP` is not sufficient.** See C-2. The repair is one clause
inside `GATE-POP` (assert the realised arena class counts `(297, 446)` / `(180, 399)`), which
converts the gate from a population check into a population-**and**-constant check and makes it
sufficient. **One quantity remains population-ambiguous in the text after that fix:**
`GATE-SELFTEST`'s `n` (**I-4**).

## §15.2 — is the removal verdict-neutral?

**Yes.** See §3.A(ii)–(iii). I could not construct a lane-favouring mechanism, and the raw-space
bit-exactness closes the algebraic route entirely.

## §15.3 — the dataset asymmetry (HateMM 743, MHC-ZH 579)

**Contained, and I rule for v3.** The two-dataset requirement is a **conjunction of
independently computed verdicts** (§3.1), never a pooled number; every threshold except the one
C-2 names is per-dataset; the MHC-ZH contract is vacuous because that dataset has no exact-zero
row, which I confirmed directly (no zero row in any modality of any of its caches). Round 2's
I-7 removed the last cross-dataset object, the pooled `ρ*`. The asymmetry therefore never
crosses a decision boundary. **One condition:** the fix for C-2 must be applied per dataset —
HateMM's arena majority moves, MHC-ZH's does not — which is itself an instance of the discipline
that contains the asymmetry.

## §15.4 — `GATE-DUALPATH`'s new role

**A different gate wearing C01's name; keep the gate, change the name and the citation.** See
**I-1**. C01's original property is not needed anywhere else, because no arm is voted at
`n = 744`, and the document should say that explicitly rather than leave a reader to wonder what
happened to `require_null_absent_from_all_top20`.

## §15.5 — does round 2's Head-R sufficiency ruling survive now that the driver is shared in fact?

**Yes, with one named residual and subject to H-1.** The shared driver is a genuinely stronger
answer than round 2's fallback: it anchors the *harness* on six banked numbers at zero cost
rather than at `264.5 s`, and it makes true of the battery the sentence that was false of v2's
dry check. Re-pricing Head-R at Head-N's measured units (`+146.9 s`) is the right direction and
the reasoning is sound — the scratchpad harness skipped the fold-parity `npz` loads, the native
`dev_seen` load and the `npz` save, so its `37.46 / 27.54 s` under-states the real driver.

**What `GATE-FLOOR` does not anchor.** It exercises the driver **only at
`--train-cache = <native>`**. Anything conditional on the cache *path or suffix* — a
`model_name`-derived branch, a shape or dtype assumption, a filename-keyed lookup — is
unexercised by the floor and invisible to this review. The residual is small because the two
caches are both `(n, 3584)` float32 with identical `ids` and `labels` (`GATE-IDPARITY` covers
that leg, and I confirmed the property directly), but it is exactly what the code lineage must
check.

**The reason this ruling is conditional on H-1:** the sufficiency argument is *"one driver, only
`--train-cache` varies"*, and §12 asserts a behavioural difference between the lineages that
contradicts it. Resolve H-1 and the ruling stands unconditionally.

## §15.6 — can `GATE-ZEROOP`'s tie diagnostic be widened to swallow a genuine mismatch?

**Not by an implementer — the criterion is narrower than the failure class, so it errs toward
HALT — but it is not sharp enough for the code lineage to check, and it does not do the job it
was added for.** See **I-5**.

## §3.B — the three round-2 Criticals

**C-1 → §3.7: ADOPTED, and it reaches every consumer of the old contract.** I enumerated them:
the head leg (now buildable, V4), the raw leg (moved to the same rows, §3.6), `GATE-ZEROMASK`
(restated feature-space only), `GATE-DUALPATH` (re-scoped), `GATE-SHUFFLEFIX` (deleted),
`GATE-NULLREMOVED` (added), Phase 5 (deleted), §5.1's population, §10.2's scope sentence. None
still assumes the masked head-space path. The *notation* in which the repair is written is
defective (**C-1**), and one constant did not follow the population (**C-2**), but the repair
itself lands.

**C-2 → §6.3: ADOPTED, and the self-defeat is gone.** I traced the run round 2 asked for. Both
real arms sit near the majority rate ⇒ `GATE-ARENA` no longer looks at them (lower bound is
`endpoint_std`-only); `GATE-ARMVIAB` fires only if the **raw** counterpart clears the same bar,
and if C06's premise is false the raw counterpart will not; `GATE-ORBITDISP` needs
`ρ_head > ρ*`, which measurement (c) says a trained head does not approach; `GATE-ARENA`'s upper
bound `≤ 0.98` cannot fire downward. **The run CLOSES.** The converse — is there now a path
where a genuinely broken real arm escapes every gate? — is covered: a broken real arm that
*collapses* is caught by `GATE-ARMVIAB`'s raw discriminator, one that *saturates* by the `≤ 0.98`
upper bound which v3 correctly kept on both real arms, one that is *degenerate in direction* by
`GATE-ORBITDISP`, and one that is *algebraically wrong* by `GATE-C01PARITY`/`GATE-ALGEBRA`/
`GATE-ZEROOP`. The residual is a real arm that is wrong but healthy-looking and correctly
normalised, which no `$0` design can catch and which §10.2 does not claim to.

**C-3 → §6, §3.7: ADOPTED, and nothing from C01's `zero_contract_v2` is left unguarded that
should be.** I checked each clause against the design that now runs.
`input_state: exact_numeric_zero` → `GATE-ZEROMASK` (feature space, where it is true).
`normalization_output_state` and `require_derived_mask_preservation` → enforced inside
`prepare_views:1381-1386` on the only population where they hold, and asserted by
`GATE-C01PARITY`. `require_null_absent_from_all_top20` → **enforced rather than assumed**, by
removal, and checked by `GATE-NULLREMOVED`. `require_remove_null_exact_equivalence` and
`require_displacement_null_exclusion_dual_path_exact` → superseded by the row-subset identity
(I-1's naming caveat). `require_fixed_null_in_shuffle` / `shuffle_fixed_point_bijection` →
**has no object**: the shuffle runs on a population from which the null is absent, so there is no
row to hold fixed. v3's deletion of `GATE-SHUFFLEFIX` is correct and is the F118 erratum lesson
applied properly — I checked that `required_halt_only_validity_guards` contains no other guard
that survives into this design unguarded.

## §3.C — the decision rule and multiplicity

**§5.5's family of 92 is the right correction, and it interacts correctly with the surrounding
structure.** SURVIVE is `∃ lineage ∃ arm (S1∧…∧S6)` on **both** datasets. The false-positive
event is the disjunction over `(arm, lineage)` = 4 disjuncts, and one Holm family per dataset
spanning `23 comparisons × 2 metrics × 2 lineages = 92` covers exactly those four disjuncts'
bootstrap legs. The two datasets remain a conjunction, which only tightens control. **S5's
shuffle rejections are not in the family, and that is correct**: they are *conjunctive* within
each disjunct, so `P(∃ disjunct : all conditions) ≤ P(∃ disjunct : its bootstrap legs all
reject) ≤ α` under Holm on the 92. Adding them would tighten further but is not required.

**§5.6's absence path is closed.** The rule (*"an absent decision or gate quantity HALTs on the
same footing as a non-finite one"*) plus a **binding** process count (`66 + 6 + 1`, and I verified
`66 = 36 + 30` against §8's own Phase 1/1R counts and §3.3's mint table) closes the lane where a
silently missing lineage makes SURVIVE vacuously false and supplies half of CLOSE. **The one
lane that remains open is a lineage that runs and produces plausible-but-wrong numbers**, which
is §15.5's residual and belongs to the code lineage. H-1 sits on top of this: two of
`GATE-LEDGER`'s declared counts are wrong, though neither is binding.

**§5.8's disclosure list is not complete.** It carries three items; **I-3** names a fourth (S6's
vacuity at arena scale) that belongs there.

## §3.D — Head-R and the shared driver

Ruled at §15.5 above. **Adding to the code-lineage list:** `GATE-FLOOR` anchors the driver only
on the native cache path, so the ro path's only unexercised surface is cache selection — which
is why `GATE-SHA` over the ro caches and `GATE-IDPARITY` are load-bearing here and not merely
hygiene.

## §3.E — the process rules

**`rule_1_compute_projection`: discharged.** No phase is ratio-derived (round 2's I-2 is
discharged by deletion), every unit is attributed to a dataset with the conservative-application
convention stated once (round 2's I-3), and every product re-multiplies (V11). **§7.2's argument
that interpreter/import cost is already inside the mint units is sound and the evidence settles
it**: the units are full-process wall (`/usr/bin/time -v` `0:40.39` and `date +%s.%N` brackets),
the same run's internal timer reads `33.0 s`, and measured startup is `3.05–3.18 s` — comfortably
inside the `7.4 s` gap alongside the cache loads and the `npz` save. **Adding a Phase 1e line
would double-count**, and round 2's I-1 is correctly repaired by statement-plus-measurement
rather than by an extra line.

**I hunted for an uncounted loop, as rounds 1 and 2 were told to.** I found **none that is
material**. I checked, and each is covered: the raw leg's 13 arms × 10 seed-free cells (Phase 2R
correctly omits the seed and lineage axes); the 12 head-space arm-construction cells (Phase 2b,
round 2's I-4); the 14 `ρ` cells (`2 raw + 12 head`, correctly decomposed); the 66 per-process
ro loads and the once-in-driver `GATE-SHA`; the 174 key forwards, whose decomposition
`(30×3)+(6×4)+(30×2)` is coherent against the mint table (Head-N fold mints need `ro_L24`,
`ro_ow_L24` and native; full mints add the dev forward; Head-R needs only the two ro forwards).
The **only** uncounted work I can name is per-gate arithmetic on already-materialised vectors —
`GATE-NESTED`'s per-item check count, `GATE-SMALLDISP`'s quantile at `0.1`, `GATE-POP`,
`GATE-NULLREMOVED`, `GATE-IDPARITY` — all in the same sub-`0.1 s` class as Phase 7, which §8
does count. One line grouping them at Phase 7's granularity would make the enumeration literally
exhaustive; I am not raising it as a finding because the rule's object is the unit **cost**
measurement and these carry none.

**`rule_2_heartbeat`: adequate, with one addition.** I re-checked every interval under the new
phase structure. Longest un-instrumented span is `GATE-C01PARITY` at `11.27 s` (`14.1 s` under
`×1.25`); the mints carry a per-epoch line (30 epochs over `40–49 s`), whose worst gap is the
`≈ 7 s` of pre-training cache loads (`8.8 s` conservative); Phase 3 emits every 32 draws
(`2.85 s`); Phase 2's whole span is `4.2 s`; `U9` is `3.70 s`. **Nothing exceeds ~15 s.** §9's
list of what the code lineage must verify carries all five items round 1 named plus round 2's
HALT-line addition; **I-6** adds the sixth (the `RuntimeError`-to-`INSTRUMENT_INCONCLUSIVE`
wrapper).

**§7.7's `U9` correction, and could another unit carry it?** The correction is sound and the
disclosure is in the right place. On the residual question: the mints are corroborated by fold
parity passing in all seven (and I independently reproduced the fold assignment and its parity
against the banked `vsw_ckpt` on both datasets); `U5a`, `U6` and `U10` are corroborated because
**I independently reproduced their outputs** (V5, V6, V7, V4), so those processes certainly ran
and produced what is claimed. `U2a`–`U2d`, `U3`, `U4`, `U7`, `U8` and `U11` have no independent
corroboration; v3's commitment that *"the freeze record will state the exit-status discipline
under which each was timed"* is the right instrument and the code/resource lineage should hold
it to that.

## §3.F — scope and honesty

**§10.2 now says what a CLOSE is scoped to.** It carries the population (`743 / 579`), both
lineages, the `GATE-DOMAIN` recovery fraction on the face of the verdict, and five exclusions
including round 2's post-fusion-contrast bullet (H-3), which is stated accurately against
`contrast_blocks:1242-1270` and `classifier.py:115-124`. I checked whether anything a CLOSE is
scoped to is still missing and found nothing material.

**Hard constraints: none touched.** Re-checked against
`iteration_8_stage0_bounded_extraction_amendment.amended_rule.conditions.d_no_other_relaxation`
read verbatim — no OCR; no cross-dataset mixing (the conjunction structure holds and round 2's
last cross-dataset object, the pooled `ρ*`, is now per-dataset); no external API; single-dataset
train split; parent-video binary label only; no ensemble (`avg_score` is C01's frozen
`gain_control`); no size scaling; SLURM-only with no `--time`. §10.4's ban analysis is unchanged
and stands. Test-split non-contact is sound **by construction**: `headspace_mint.py:106-116`'s
`torch.load` guard, the driver's `split == "train"` assertion, and the frozen `c09_guard`
`sitecustomize` open()-level predicate — three layers, and no `dev_seen_*-ro_*` or `test_seen`
path is reachable from any phase.

**Does v3 claim a repair the artifact does not contain?** **No.** Every one of the 16 round-2
dispositions and all three reopened round-1 items are genuinely present — see Part D. The two
Criticals are defects v3 *introduced* while making repairs that did land, not repairs it claims
falsely.

---

# PART D — DISPOSITION AUDIT

Each row checked against the primary source or by execution, never against §14.

## Round 2 — 13 findings + 3 Minor: **16 of 16 VERIFIED ADOPTED**

| finding | v3 claim | audit result |
|---|---|---|
| **C-1** head-space arms unbuildable | ADOPTED | **VERIFIED ADOPTED** — I reproduced the defect independently (V3: all three block families, both mask choices, `common_displacement` dead under both) and then the repair (V4: all 13 arms build at `n = 743`, three seeds, `float32`). Three populations are named in §3.7 and gated by `GATE-POP`. The notation used to express it is defective (**C-1**) and one constant did not follow the population (**C-2**); the repair itself is real. |
| **C-2** `GATE-ARENA` lower bound self-defeats | ADOPTED | **VERIFIED ADOPTED** — §6.3 restricts the lower bound to `endpoint_std`, matching C09's scope which I confirmed verbatim at `C09_A0_V17_RECORD.md:1569-1572` (V12); `≤ 0.98` kept on all three arms; the real arms' lower side belongs to `GATE-ARMVIAB`. I traced the near-majority run and it CLOSES (Part C §3.B). |
| **C-3** C01's zero contract not portable | ADOPTED | **VERIFIED ADOPTED** — `GATE-DUALPATH` re-scoped to the raw leg and verified bit-exact by me (V5); `GATE-ZEROMASK` restated feature-space only; `GATE-SHUFFLEFIX` deleted with a correct vacuity argument; `GATE-NULLREMOVED` added; Phase 5 deleted. I checked every `zero_contract_v2` clause for orphaned coverage and found none (Part C §3.B). Naming caveat only: **I-1**. |
| **H-1** lineage disjunction uncorrected | ADOPTED (b) | **VERIFIED ADOPTED** — one Holm family of 92 per dataset spanning both lineages; arithmetic re-derived from the two frozen C01 lists (V10) and the structure ruled correct (Part C §3.C). |
| **H-2** Head-R has no anchor, does not run the banked script | ADOPTED, stronger | **VERIFIED ADOPTED** — one shared driver, `GATE-FLOOR` anchoring it for both lineages at zero cost, and Head-R re-priced at Head-N's units (`+146.9 s`), which I re-multiplied (V11). Stronger than round 2's `264.5 s` fallback, as claimed. **But §12 contradicts the "only variable" sentence it rests on: H-1.** |
| **H-3** post-fusion contrast undisclosed | ADOPTED | **VERIFIED ADOPTED** — §10.2 bullet and the §3.4 clause, both accurate against `contrast_blocks:1242-1270` (per-modality, pre-fusion) and `classifier.py:115-124` (internal Hadamard fusion). |
| **I-1** round-1 I-3 leg (c) unrepaired | ADOPTED | **VERIFIED ADOPTED** — §7.2 states the units are full-process wall, says how they were timed, and gives the `40.39` vs `33.0 s` evidence plus measured startup `3.05–3.18 s`. **The "no line added, because adding one would double-count" reasoning is correct**, not a dodge. |
| **I-2** Phase 5 derived by ratio | ADOPTED by deletion | **VERIFIED ADOPTED** — I checked every §8 row: none is ratio-derived. |
| **I-3** units not attributed to a dataset | ADOPTED | **VERIFIED ADOPTED** — §7.7 labels all 14 units and states the conservative-application convention once, with `U9`, `U7`, `U11` correctly excepted. |
| **I-4** head-space arm construction uncounted | ADOPTED | **VERIFIED ADOPTED** — Phase 2b, `12 cells × U10 = 2.2 s`; the 12 decomposes correctly as `2 ds × 3 seeds × 2 lineages`. |
| **I-5** `GATE-ZEROOP` not "strictly stronger" | ADOPTED | **VERIFIED ADOPTED** — the wording is corrected to logically independent, in both directions, and the tie diagnostic is pre-registered with report-not-HALT semantics. **Under-specified: I-5**, but the adoption is present and the direction is safe. |
| **I-6** absence not a HALT; process count not binding | ADOPTED | **VERIFIED ADOPTED** — §5.6 carries the absence rule; §12 marks the process count **binding — HALT on any mismatch**, and the `66 + 6 + 1` decomposition checks out against §3.3 and §8. |
| **I-7** pooled cross-dataset `ρ*` | ADOPTED | **VERIFIED ADOPTED** — per-dataset `ρ*` with runner-ups recorded; both maxima and both runner-ups reproduce exactly (V7). The truncation prescribed by round 2 has a side effect: **I-2**. |
| **M-1** Phase 3 truncated | ADOPTED | **VERIFIED** — `273.7 s`, rounded like every other product (V11). |
| **M-2** `displacement`'s comparator asymmetry | ADOPTED | **VERIFIED** — disclosed as §5.8 item 2 with the correct reasoning (C01 froze a comparator list only for its primary). |
| **M-3** overrun framing | ADOPTED | **VERIFIED** — §7.8 now reads *"disclosed at the same time as the result"* and concedes the conflict was knowable from C09's banked mint costs before the first burn. |

**Round-2 free strengthenings, both adopted:** `GATE-SELFTEST` is present (with the `n` ambiguity
of **I-4**), and §9's HALT line names the failing gate (with **I-6**'s gap for crashes).

## The three reopened round-1 items: **all three genuinely repaired**

| finding | round-2 audit | round-3 result |
|---|---|---|
| **C-3** unanchored arm algebra | NOT ADOPTED ON THE VERDICT PATH | **REPAIRED.** The verdict path now executes — I built all 13 head-space arms through the imported `l2_rows` at `n = 743` (V4) — and §3.4 states correctly what two-block parity does and does not buy, adopting round 2's ruling rather than paraphrasing it, including the concession that the outer normalisation is **numerically vacuous at one block** and that what was actually fixed is the dtype. I reproduced both figures (measurement (d)). The *"forced by the head's architecture, not selected"* argument is right and is the reason round 2 gave. |
| **C-1 companion** both real arms clear majority | NOT ADOPTED IN EFFECT | **REPAIRED.** §6.3 removes the blocking lower bound and `GATE-ARMVIAB`'s two-case form is now reachable; I traced the run and it CLOSES rather than HALTs. The gate is now operative — though the bar it applies is the wrong constant (**C-2**). |
| **I-3** three per-process loops | PARTIAL, 2 of 3 | **REPAIRED.** Leg (c) is settled in §7.2 by measurement rather than by an added line, and the no-double-count argument is correct. |

**Round-1 and round-2 rulings carried without change, all still sound:** the direction of
"conservative"; A7; per-arm retraining excluded; the `max` as `ρ*`'s order statistic (now with
measurement (c) behind it); SLURM and the login-node dismissal; the untrained-head blindness
discipline, which v3 observes throughout — **no arm accuracy appears anywhere in v3**, and I
found nothing inconsistent with that; HALT semantics; §5.8's inapplicability reasoning for
`require_accuracy_gain_over_deployed_r0_context`; S6's net-fix reference (correct as to
reference — see **I-3** as to force).

---

# PART E — WHAT THE SEPARATE CODE-REVIEW LINEAGE MUST VERIFY

A design review cannot reach these. Listed in the order they can produce a wrong verdict.

**The shared mint driver (§3.3).**
1. That `c06_falsifier_mint.py` really imports `headspace_mint` with its sha256 asserted, and
   that **no** behaviour outside `--train-cache` differs between the two lineages — in
   particular that the fold-parity assertion (`:203-216`), the dummy construction (`:219-227`),
   the `torch.load` guard (`:106-116`), the seeding and the DET-1 contract are the frozen ones
   and not re-implemented.
2. That `--train-cache` overrides **only** the training cache and cannot reach `model_name`, the
   dev load (`:199`) or the dataset table — and that whichever answer H-1 receives is the one
   the code implements and the ledger declares.
3. That there is **no branch conditional on the cache filename or suffix** anywhere in the
   driver. `GATE-FLOOR` exercises the native path only; such a branch would be invisible to it.
4. That the six `GATE-FLOOR` mints and the 30 Head-R mints go through the *same* function, not
   two copies.

**Populations and constants (C-2, I-4).**
5. That the majority rate used by `GATE-ARENA` and `GATE-ARMVIAB` is computed from the **arena's
   own labels**, not read from a constant, and that it equals the frozen `0.6003 / 0.6891`.
6. That `GATE-SELFTEST`'s `n` is the arena size, and that no banked `744` leaks into any
   per-item denominator.
7. That `ρ` is computed over the **743/579-row** arm matrices and not over a 744-row array with
   a masked row left in — I measure a `1.301e-03` shift on `endpoint_std` if it is, which would
   fail the 4-dp reproduction leg (fail-safe, but it would present as an unexplained HALT).

**The `zero_mask` convention (C-1).**
8. That every `prepare_views` call passes an explicit boolean array, and every `l2_rows` call's
   mask matches the population it is handed — with an assertion, not a comment.
9. That the `n = 744` build exists **only** inside `GATE-C01PARITY`/`GATE-DUALPATH` and that
   nothing votes on it.

**The tie diagnostic (I-5).**
10. Which ranking and which residual the implementation uses, that its item set is the
    pre-registered one, and that the report-not-HALT branch cannot be reached by any item
    outside it.

**`GATE-POP` (§15.1).**
11. That `GATE-POP` is evaluated **before** any gate that consumes a population-dependent
    constant, and that it asserts row identity between the head leg and the raw leg by **index
    set**, not merely by count.

**Heartbeat and failure recording (I-6, §9).**
12. All six items in §9's own list, plus: that a `RuntimeError` from the imported C01 algebra is
    caught, recorded as `INSTRUMENT_INCONCLUSIVE` with its `context` string, and written to the
    final heartbeat line before exit.

---

# PART F — MINIMAL SET OF CHANGES THAT WOULD EARN GO

1. **C-1** — replace `zero_mask = None` with an explicit `np.zeros(n, dtype=bool)` everywhere
   `prepare_views` is invoked (`GATE-C01PARITY` clause (i), `GATE-DUALPATH`, §3.7, §7.4(g)/(h)),
   and record why the two functions differ.
2. **C-2** — freeze the **arena** majority rate at `0.6003` (HateMM) / `0.6891` (MHC-ZH), cite it
   in `GATE-ARENA` and `GATE-ARMVIAB`, and add the class-count clause to `GATE-POP`.
3. **H-1** — correct §12's Head-R dev-file sentence and the two declared counts, or declare the
   dev-load suppression as a second lineage-varying behaviour and re-argue §15.5 without the
   "only variable" sentence.
4. **I-1 … I-6** as written — rename `GATE-DUALPATH`; freeze `ρ*` at full precision; disclose
   S6's vacuity in §5.8; pin `GATE-SELFTEST`'s `n` and S6's seed axis; sharpen the tie
   diagnostic; wrap the C01 algebra's `die()` path.
5. **M-1 … M-4** — the Phase 7 rounding convention, the two initialisation-dependent digits, and
   the two-`92` clarification.

**Can the falsifier still discharge the written condition at `$0`? Yes — and v3 is close.**
Nothing I found requires a GPU, an extraction, new data or a redesign. C-1 is a calling
convention; C-2 is one constant plus one gate clause; H-1 is a sentence and two integers. The
three-population contract is the right answer to round 2's C-1/C-3 and its central claim is
**provable and proved**: the population change is a pure row-subset with `max|diff| = 0.000e+00`
on all 13 arms and no movement in any `ρ`. With the five repairs above the instrument measures
what `falsifier_spec` asks, on the arena the registry requires, under both a native-trained head
applied out of domain and an `ro_L24`-trained in-domain head — with §10.2's post-fusion caveat on
the face of the verdict.

---

*Read-only review. No GPU, SLURM, Modal, model load, head training, arena run, cache write,
test-split access, job submission or commit occurred. `TARGET_STATE.json`, all three drafts and
all configs were read and not modified. All computation was `sha256sum`, file reads, and
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint
checkpoints. No arm accuracy was computed at any point. A GO on this lineage would authorise
nothing to run: the design would still require freeze with hashes, a **separate** independent
code/resource review lineage over the executable reaching its own `0C/0H/0I`, and main-dialogue
authorization.*
