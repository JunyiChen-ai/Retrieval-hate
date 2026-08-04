# C06 `$0` falsifier — ERRATUM 2, PROPOSAL v6: ADJUDICATION (round 6)

**Target:** `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V6.md`,
sha256 `05c93599b9ee45450a685632d6ea057ddbbb0545f08fdb546090556f3f3dc722`, 920 lines.
**Against:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`,
sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
**Reviewer:** fresh, no part in rounds 1–5. Judged from documents and repository only.

---

## VERDICT

> ## REVISE — 0C / 1H / 3I / 5M

The subtraction machine is now **verified, not trusted**. Every one of the ten printed commands
reproduces its printed hit list **byte-identically including sort order**; the global partition
`265 → 240 = 128 + 112`, `UNCHARGED = 0` recomputes exactly from the raw lists; the meta-check
reproduces at `102 = 30 + 72`; the Phase 1c ruling holds at source; the fourth live-wrong claim is
real and correctly diagnosed; and §9's consolidated delta covers every editing row exactly once.
I found nothing wrong inside the method v6 built.

**What I found is one site class the method cannot see, and it is v6's own primary family.** Sweep A
keys on the numerals `72|73|74`. Two sites state the process inventory as a **decomposition without
the total** — `V15E1:1814` (§12's `processes reporting | 66 mints + 6 fidelity + 1 arena | yes —
HALT on any mismatch`) and `config:42` (`"process_order": "66 mints -> 6 fidelity -> 1 arena"`).
Neither carries a `72|73|74` numeral, neither carries a `Phase N` label or a unit symbol, so neither
is returned by any of the ten sweeps **or** by the §6 meta-check, and neither has ever been named in
six rounds of this erratum. Both are flatly wrong the moment v6 lands. The §12 one is the design's
own declaration of the **binding, HALT-on-mismatch predicate that round 1's H-1 opened this erratum
on**.

This is the eleventh family §6 was built to prove could not exist. It exists because §6 sweeps §8's
*unit/phase* vocabulary, and the process inventory is not a §8 unit or phase.

---

## 1. PROCESS INTEGRITY — CLEAN

**Five artifacts, re-verified now:**

| path | sha256 | matches §11 |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` | ✓ |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` | ✓ |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` | ✓ |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` | ✓ |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` | ✓ |

**v1–v5 byte-unmodified**, each matching the prefix v6's header declares:
`f063c388…`, `4225bea3…`, `48f4e015…`, `0b494041…`, `c41a0223…`. ✓

**Blindness:** `artifacts/c06_falsifier/` does not exist (`ls` → No such file or directory). No arena
run, no `--gate-sha-only` leg, no mint, no job, no edit outside my scratchpad. My compute:
`sha256sum`, file/source reads, the ten sweeps + the meta-check + twenty wider sweeps of my own
construction, `c09guard` predicate probes, and arithmetic.

---

## 2. SWEEP DIFF — ten for ten, byte-identical

Run from the printed commands verbatim, with the `$F` list exactly as printed, each ending in
`| sort -t: -k1,1 -k2,2n`. I compared the printed `file:line` column against my raw output
**element-wise, in order**:

| sweep | printed | mine | order | verdict |
|---|---|---|---|---|
| A process counts | 18 | **18** | identical | EXACT |
| B ordinals | 3 | **3** | identical | EXACT |
| C projection/denominator | 38 | **38** | identical | EXACT |
| D design pointer | 15 | **15** | identical | EXACT |
| E pass-count idiom | 22 | **22** | identical | EXACT |
| F §8 equation/units | 25 | **25** | identical | EXACT |
| G ledger/blocked predicate | 82 | **82** | identical | EXACT |
| H artifact count/scope | 29 | **29** | identical | EXACT |
| I progress coverage | 16 | **16** | identical | EXACT |
| J Phase 1d/1g re-price | 17 | **17** | identical | EXACT |

**M-5 discharged literally.** The lists are the commands' output in the commands' order — I checked
position by position, not as sets.

**Per-sweep subtractions all close**, recomputed from the printed charge column:
`A 18=11+7`, `B 3=1+2`, `C 38=15+23`, `D 15=11+4`, `E 22=9+13`, `F 25=15+10`, `G 82=56+26`,
`H 29=13+16`, `I 16=8+8`, `J 17=7+10`. Ten for ten.

**Cross-sweep charge consistency: 0 divergent.** Every site appearing in more than one sweep carries
a byte-identical charge string in every table it appears in.

---

## 3. GLOBAL PARTITION — recomputed from the raw lists

| quantity | v6 | mine |
|---|---|---|
| hit-instances across the ten sweeps | 265 | **265** |
| distinct sites (deduplicated) | 240 | **240** |
| excess hit-instances (`265 − 240`) | 25 | **25** |
| distinct sites returned by more than one sweep | 21 | **21** |
| distinct sites charged to a row | 128 | **128** |
| distinct sites charged to a declaration | 112 | **112** |
| `UNCHARGED` | 0 | **0** |

`240 = 128 + 112`. **Residue: none, in either direction.** M-1 is discharged: the two quantities are
genuinely different (25 vs 21) and both are now printed under correct labels.

**Rows.** §2 defines **67** rows (`1`–`66` plus `26†`), no duplicates, no gaps in `1..66`.
Sweeps return sites for **66** of them; **row 48 is the only siteless row**, as declared.

**Multi-line rows.** Independently derived from the sweeps, the rows with more than one *charged*
site are exactly
`2, 5, 7, 12, 15, 16, 19, 23, 24, 32, 34, 35, 36, 39, 41, 43, 44, 46, 47, 49, 53, 57, 58, 59, 61, 65`
— **26 rows, identical to §1's list, zero discrepancy in either direction.** Round 5's I-3 (row 19)
is folded in.

**The named extent-only list — three items, each reason verified.**

* `V15E1:1633` (row 19) — verified: the line reads *"`c06_falsifier_arena.py` and
  `configs/c06/c06_falsifier.json` is updated with each correction.)\*"*. It names **where the
  literal lives**, states no quantity, and so falls outside every quantity family by construction.
  Reason holds.
* `arena:470` (row 39) — verified at source: the line is `            fails.append(`, a bare
  continuation. Reason holds.
* **row 48** — verified: the sbatch's only exports are `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`
  (`:28-31`), `CUDA_VISIBLE_DEVICES` (`:32`), `PYTHONPATH` (`:37`), `C09_LEDGER_DIR` (`:42`).
  `C06_MINTS_EXECUTED` and `C06_PROJECTED_SECONDS` appear nowhere in `scripts/` or `configs/` except
  the *read* at `arena:1419`. A grep cannot return a line that does not exist. Reason holds.

Replacing v5's contorted `ORPHAN = 0` with this named list is the right call and it is honestly
argued. Its **completeness claim** is nevertheless false — see I-2.

---

## 4. THE META-CHECK — reproduced, and audited against the delta

```bash
grep -nE '\bU[0-9]+[a-d]?\b|\bU_(acc|mF1|tie)\b|[Pp]hase [0-9]+[a-zA-Z]?' $F | sort -t: -k1,1 -k2,2n
```

**102 hits. 30 already charged by A–J. 72-hit residue.** Reproduced exactly.

The residue's inventory, extracted independently:

* **12 phases**: `1b, 1c, 1e, 1f, 2, 2b, 2z, 2D, 3, 4, 5, 7` — identical to §6's list.
* **16 units**: `U1, U2a, U2b, U2c, U2d, U3, U4, U5a, U5b, U6, U8, U9, U10, U_acc, U_mF1, U_tie` —
  identical to §6's list (§6's *"U3–U6"* shorthand is a range covering `U5a`/`U5b`; the set and the
  count are right).

**"Of §8's 21 phase labels and 18 unit symbols" — verified.** §8's table (`V15E1:1542-1567`) carries
26 rows bearing **21 distinct phase labels** (`1, 1R, 1b, 1c, 1d, 1e, 1f, 1g, 2, 2b, 2z, 2R, 2Ra,
2C, 2D, 3, 4, 5, 6, 7z, 7`) and **18 distinct unit symbols** (`U1, U2a–U2d, U3, U4, U5a, U5b, U6,
U7, U8, U9, U10, U11, U_acc, U_mF1, U_tie`). Both counts correct.

**"Exactly two phases move, zero units move" — I checked all 21 against the delta and agree.** The
only per-*process* phases are 1c, 1d and 1g. 1d and 1g move. 1e is per-mint (66), 1f per-cell (150),
6 per-fidelity (3+3), and every Phase-2/3/4/5/7 row is per-arm, per-draw or per-gate — none is
sensitive to the process inventory. `U7` and `U11` keep their values (§6's measurement plus round
5's independent re-measure); `U9`'s object and value are untouched.

**Phase 1c's `67` — the one growth candidate — ruled out correctly, and I confirmed it at source.**

* `main()` order verified verbatim: `bat.gate_det1()` (`:1269`) → `bat.assert_guard_active()`
  (`:1271`) → `bat.gate_sha()` (`:1276`) → `bat.load_frozen()` (`:1277`) →
  `if args.gate_sha_only:` (`:1278`) / `hb(...)` (`:1279`) / `return 0` (`:1280`).
* `load_ro` is **first called at `:1284`** (inside the `dry_parity_only` branch) and in the main path
  at `:1300` and `:1358` — **all after the early return**. Exactly as §6 states.
* `load_frozen()` (`:482-547`) opens **one** file, `configs/c01/c01_a0_v2.json` (`:483`), then
  imports modules and asserts eighteen C01 constants. **It opens no `.pt`.** Phase 1c's `67` stands.

---

## 5. THE FOURTH LIVE-WRONG CLAIM — confirmed at source, conclusion survives

**`headspace_fidelity.py:66-68` reads exactly this:**

```python
66:  z = np.load(os.path.join(a.mintdir, "mint_{}_s{}_ffull.npz".format(a.dataset, s)),
67:              allow_pickle=True)
68:  meta = json.loads(str(z["meta"]))
```

`z["meta"]` is the **only** subscript of `z` in the file. `grep -nEi 'lab|label'` over the whole file
returns **zero hits** — no `lab_dev`, no label array of any kind. Its only other opens are `:42`
(`with open(path)`, the banked `.trainlog` inside `floor_dev_curve`, matched by
`is_banked_trainlog`) and `:113` (its own output JSON).

So both claims are false as stated:

* `V15E1:1810`: *"`headspace_fidelity.py` opens **no** `dev_seen` file, **reading `lab_dev` out of
  the banked mint `.npz`** (`:66`)"*
* `sbatch:103`: *"# It reads lab_dev out of the banked mint .npz and opens no dev_seen file (§12)."*

**The conclusion survives untouched** — fidelity opens no `dev_seen` file, so its contribution to the
second term is genuinely zero — and rows 43 and 66 are the right repairs, worded correctly (`:1810`'s
new text points at `:68`, which is where `z["meta"]` actually is). §0's reading is exact.

I also confirmed the repair does not depend on the false mechanism: **`expected_sha_dev_opens = 4`**
is driven entirely by `GATE-SHA`. `gate_sha` (`:562-563`) iterates
`frozen_sha256` (13 entries) `+ frozen_sha256_input_caches` (8 entries) and `sha256_of`s each. Under
`c09guard.is_dev_like` the dev-like count over that concatenated iterable is **exactly 2**
(`config:247`, `config:249`), test-like **0**; appending the design document under either its V15E1
or V15E2 name leaves it at **2** (both probed: `is_dev_like` False, `is_test_like` False). Two passes
× two files = **4**. Row 65's *"checkable 2"* is exactly right, and `V15E1:1753`/`:1755` are the §11
twins of those two config lines.

---

## 6. ROW 4, ROW 64 AND §8-INTERNAL CONSISTENCY

**Row 4 produces a consistent §8.** `V15E1:1550` is a single physical table line carrying count cell,
unit cell and cost cell, so moving `1 → 2` and `3.8 s → 7.6 s` on one line is coherent. The result
matches row 34's summand:

`2642.4 + 1.0 + 0.7 + 0.1 + 1.3 + 7.6 + 7.0 + 1013.8 = 3673.9` ✓ (I recomputed)
`3673.9 × 1.25 = 4592.375 → 4592.4` ✓
`3673.9/60 = 61.23 → 61.2 min` and `4592.4/60 = 76.54 → 76.5 min` — **unchanged at one decimal** ✓
mint share `2508.3/3673.9 = 68.27 % → 68.3 %` ✓, Phase 3 `1013.8/3673.9 = 27.59 % → 27.6 %` ✓ (row 35
and row 57 correctly marked unchanged)
2× miss `3673.9 + 1013.8 = 4687.7 s = 78.1 min` ✓, 5× miss `3673.9 + 4×1013.8 = 7729.1 s = 128.8 min` ✓
`7729.1/4592.4 = 1.68×` — the ratio on `:1610` is unchanged, correctly left alone ✓
margin `3.8 − 3.717 = 0.083` ✓ — and splitting it into a **unit** statement is the right repair.

**Row 64 names the count correctly but is one line short of the words it fixes.** See I-1.

---

## 7. R5's 3I / 5M — LIMB-LEVEL DISPOSITION

| finding | limb | disposition | verified |
|---|---|---|---|
| **I-1** | extend row 39 over `arena:432-441` | extent is `:432-441` **and** `:468-475`; `correct` column states the docstring rewrite; sweep G widened so the paragraph is charged, not asserted | ✓ source-verified: `:432` *"ONE PREDICATE IS BLOCKED…NOT ADJUSTED HERE"*, `:434-437` the `+2/+4` derivation, `:439-441` *"implemented exactly as frozen"* / *"not this lineage's call"* — all ten lines charged to row 39 |
| **I-2** | name `c06_falsifier_mint.py:336` | §3's warrant now names **both** write sites and is extended with the fidelity fact | ✓ adopted and correctly extended |
| **I-3** | add row 19 to §1's list; charge `:1632-1633` | row 19 in the 26-row list; `:1632` charged by sweep C's `denominator`; `:1633` **substituted** as a named extent-only line | ✓ adopted; the substitution is reasoned and disclosed, but its completeness claim fails (I-2 below) |
| **M-1** | mislabelled cell | excess (25) and multi-sweep (21) printed separately | ✓ both reproduce |
| **M-2** | `total_§11_digests` naming | row 49 gains *"whose digest §11 names and `config:6` carries"* | ✓ and the arithmetic checks: `frozen_sha256`=13, `input_caches`=8 → 21; `21+16=37 → 22+16=38` |
| **M-3** | assertion placement | §7 item 4 pins it to the pre-`gate_sha_only` block, before `arena:1278` | ✓ such a block exists (`:1269-1277`); it does fire in process 1 of 74 |
| **M-4** | `design_sha256_note` on the verdict face | §5 prescribes the string; rows 23/24 carry *"plus M-4's caveat label"* | ✓ on both publication paths (`emit_halt` and the verdict face) |
| **M-5** | sorted output | every command ends in the sort | ✓ and the lists are now literally the commands' output |

**All nine limbs adopted.** R5's obligation-6 carry-forward list is honoured, with one exception
noted at I-3 below.

**Row 45 independently re-verified** (it is load-bearing for the whole `74`): `c09guard.install()`
registers `atexit.register(_flush)` (`:143`), and `_ledger_path()` (`:110-120`) writes one file per
`(job, pid, t0)`. The driver leg is a python process with the guard installed — `arena:417` asserts
`_INSTALLED` — so it writes a ledger. `len(procs)` = 66 mints + 6 fidelity + 1 driver leg = **73**,
`+ 1` for the not-yet-flushed arena = **74**. Row 45's *"unchanged — CORRECT"* is right.

---

## 8. CUMULATIVE LANDING COHERENCE — §9's delta is complete

I extracted §9's consolidated delta and matched it against all 67 rows. **Every editing row appears
exactly once**, either by row id or by line citation:

* `arena`: 12, 14, 23, 24, 39, 50, 58, 59 by id; 15 (`:29-30`), 26 (`:4`), 32 (`:559-560`),
  33 (`:1232`), 26† (`delete :1418`) by line.
* `config`: 10, 11, 16, 20, 21, 22, 38, 40, 41, 42, 49.
* `sbatch`: 9, 25, 30, 31, 48, 66.
* `mint`: 13, 17, 27, 55, 60, 62.
* `V15E2`: 1, 2, 3, 4, 5, 6, 7, 8, 19, 28, 29, 34, 35, 36, 37, 43, 44, 52, 53, 54, 56, 57, 61, 63,
  64, 65.

The only rows absent are **18, 45, 46, 47, 51** — all marked *"unchanged — CORRECT"*, correctly
carrying no delta entry. **No duplication, no contradiction, nothing endorsed silently dropped.**

**Cross-round check.** R2's H-1 offered two branches — re-price, or bound the omission explicitly —
and required that *"§7.2's '73rd process', §9's 'the one span' and the sbatch header must be
corrected with the rest"* under either. v6 takes the re-price branch and does all three (rows 2, 5,
9). R3's H-3 (row 2 must not contradict row 6) is discharged: row 2 now carries **no ordinal**, and
`72` (row 1, verified correct: 66 + 6) `+ 2 = 74` agrees with rows 6, 10, 11, 12. R4's I-2 is adopted
whole in §7. One cross-round contradiction survives — see I-3.

---

## FINDINGS

### H-1. Two process-inventory sites state the decomposition without the total, so no sweep and no meta-check can reach them — and both are flatly wrong at landing. One is §12's declaration of the binding HALT predicate this erratum exists to repair.

**Site 1 — `V15E1:1814`, §12's predicate table:**

```
| processes reporting | **66 mints + 6 fidelity + 1 arena** | yes — HALT on any mismatch |
```

**Site 2 — `configs/c06/c06_falsifier.json:42`:**

```json
"process_order": "66 mints -> 6 fidelity -> 1 arena",
```

**Both are invisible to the whole apparatus.** Neither appears in any of the ten sweeps' 240 sites,
neither appears among the meta-check's 102 hits, neither is in any row's extent, neither is in the
extent-only list, and **neither has been named in any of the six proposals or five reviews.**
`V15E1:1814` is the *only* occurrence of the string `processes reporting` in V15E1.

**Why the method misses them.** Sweep A is `grep -nE '\b7[234]\b'` — it charges by the **total**.
Both lines state the inventory as a **sum of parts with no total written down**, so they carry no
`72|73|74` numeral. Sweep G matches `processes_reporting` with an **underscore**; the §12 table row
writes it with a **space**. And §6's meta-check sweeps §8's unit/phase vocabulary, which the process
inventory is not.

**Why they are wrong after landing.**

* `V15E1:1814` declares the expected value of `processes_reporting` as `66 + 6 + 1` = **73**, marked
  *"yes — HALT on any mismatch"*, while row 11 sets `config:222` to `{"expected":74,
  "decomposition":"1+66+6+1"}`, row 12 makes `arena:465-467` assert `!= 74` read from
  `cfg["ledger"]`, and row 45 confirms the runtime value is 74. A landed V15E2 would carry **74 in
  §7.2, §8, §13 and the config, and 73 in §12** — on a binding predicate. That is precisely the
  trade round 2's H-2 refused and round 1's H-1 opened this erratum on. It is also the exact failure
  shape round 5 called High one row over: a count moved in one cell and left standing in its
  neighbour.
* `config:42` sits **between** `config:41` (row 10, → `total:74` with `gate_sha_driver:1`) and
  `config:43` (row 16, the projection single source). After landing, `:41` would enumerate four
  process classes and the next line would declare an order over three — contradicting row 6's own
  new §13 text (*"74 processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1
  arena"*) and row 9's identical sbatch text.

**Repair.** Add two rows and one sweep.

1. **Row 67** — `V15E1:1814` → *"`1` `GATE-SHA` driver leg + 66 mints + 6 fidelity + 1 arena"*,
   binding, HALT on mismatch.
2. **Row 68** — `config:42` → `"process_order": "1 gate-sha driver leg -> 66 mints -> 6 fidelity -> 1 arena"`.
3. **Sweep K**, keyed on the decomposition rather than the total, with hit list, charges and
   subtraction. One candidate, which I ran:

   ```bash
   grep -nE '66 ?(mints|\+)|6 ?(fidelity|\+)|1 ?arena|processes? reporting' $F | sort -t: -k1,1 -k2,2n
   ```

   **16 hits: 6 already charged** — `config:255` (row 49), `V15E1:226` (declared, sweep G),
   `V15E1:1550` (row 4), `V15E1:1566` (declared, sweep F), `V15E1:1839` (row 6), `sbatch:16`
   (row 9) — **and 10 uncharged**, of which exactly the two named above are targets:

   * **`config:42`** and **`V15E1:1814`** — the two new rows.
   * `V15E1:1546` — Phase 1c, correctly a non-target; §6 already rules it out at source.
   * `V15E1:1923` — `GATE-FOLD` under resume, unaffected by the inventory.
   * `V15E1:1518` (a CPU-minute drafting sum), `V15E1:701` and `arena:384` (the p-value's
     `(256 + 1)`) — pattern artifacts, declarable as such.
   * `sbatch:67`, `:101`, `:138` — the three block banners (`66 mints`, `6 fidelity`, `1 arena`).
     If the inventory now leads with a driver leg, a fourth banner above `sbatch:62` is the
     consistent completion. Cosmetic, but it is the same family.

   Any pattern that reaches `config:42` and `V15E1:1814` will do; this one is offered as a worked
   example, not as a prescription.
4. **Extend §6's meta-check statement** so it says what it actually covers. It proves no eleventh
   family exists *among §8's priced units and phases*. It cannot, and does not, prove one does not
   exist among the document's **prose and config statements of the process inventory** — which is
   where both of these live.

---

### I-1. Row 64's site is one line short, and the uncharged line is the one carrying the false words. This is round 5's I-3 defect reproduced at the row created to answer round 5's H-1.

The sentence round 5 called *"the sentence that determines the count"* spans **two** lines:

```
1368| be under `0.4 s` ... and it already contains the six fidelity processes' startup. **So §8 Phase 1g's
1369| count is `1`, the arena alone, and it is determined by this measurement rather than inferred.**
```

Row 64 declares its site as **`V15E1:1368`**, singular, and quotes the full two-line sentence. Sweep J
returns `:1368` (it matches `Phase 1[dg]`); **`:1369` carries no `3.8`, no `7.6` and no phase label,
so no sweep returns it** — and row 64 is **not** in §1's 26-row multi-line list. The words that are
false — *"count is `1`, the arena alone"* — are on the uncharged line.

Round 5's I-3 was: *"one three-line row whose two uncharged lines happen to be the two lines it
exists to delete."* This is the same defect, at the row v6 added to fix round 5's High.

**Repair.** Row 64's site becomes `V15E1:1368-1369`; add 64 to §1's multi-line list; declare `:1369`
as a named extent-only line with its reason (it is the sentence's continuation and carries no
quantity-family token).

---

### I-2. The extent-only list's completeness claim is false: two further rows have extents wider than their charged sites, and neither extra line is named.

§10 states: *"these are the only extent lines no pattern returns."* Three more exist.

* **Row 5 — `V15E1:1644`.** The subject the rewrite replaces is on `:1644`, outside the declared
  extent `:1645-1646`:
  ```
  1644| single `GATE-C01PARITY` dataset at `11.27 s` (`14.1 s` conservative). The **arena's own startup**
  1645| is the one span that precedes any python-side line — **`3.094–3.717 s` measured over 35
  1646| arena-class runs by three parties** (§7.7), `3.8 s` as carried at §8 Phase 1g (round-11 I-1,
  ```
  Row 5's prescribed text begins *"the `--gate-sha-only` driver leg's startup is the first such span
  and the arena's the second"* — that cannot grammatically follow *"The **arena's own startup**"*, so
  the edit necessarily touches `:1644`. `:1644` is returned by no sweep, is in no meta-check hit, and
  is named nowhere in v6. **Round 2's H-1 item 4 originally scoped this site as `V15E1:1643-1651`**;
  it has been silently narrowed to two lines across versions.
* **Row 54 — `V15E1:2440`.** The sentence spans two lines:
  ```
  2439| ... **No `.py` source moved** — all 37
  2440| §11 digests recompute. Nothing else was written outside `refine-logs/` ...
  ```
  Row 54 declares `:2439` alone. The `37 → 38` edit lands on `:2439`, but the appended clause
  (*"as of this document's own freeze; the 38th is this document…"*) must go where *"§11 digests
  recompute"* is — `:2440`, which no sweep returns.
* **Row 44's label.** The header reads `V15E1:1817-1821` **(four lines)**; that range is **five**
  lines. `:1820` is inside the declared extent yet is charged by sweep E to a **declaration**
  (*"'removed twice elsewhere' — ordinary English"*). §1's rule is that a site is charged to exactly
  one row **or** one declaration, and that extent ⊇ charged sites; here one line is simultaneously
  inside a row's edit extent and declared a non-target. The two dispositions contradict.

None of these loses a site at landing — an implementer working from the row will see the sentence.
But they are the difference between an extent-only list a reader can verify and one a reader must
trust, which is the entire warrant for replacing `ORPHAN` with it.

**Repair.** Extend rows 5 and 54 to `:1644-1646` and `:2439-2440`, add both to §1's multi-line list,
and name `:1644` and `:2440` in the extent-only list. Either restate row 44's extent as the four
lines it charges or add `:1820` to the extent and remove it from sweep E's declaration.

---

### I-3. "The driver leg emits three lines before returning" is wrong — it emits four — and round 2 had already enumerated all four by name.

Row 5's `why` column: *"the driver leg instantiates `Heartbeat` at `arena:1266` and emits **three**
lines before returning, and runs `load_frozen()` first (rounds 4 and 5 both verified)"*. Round 5's
obligation 6 repeats *"the driver leg's **three** heartbeat lines"*.

Traced at source. `Heartbeat.__call__` (`:63-71`) writes exactly one line per call
(`self.fh.write(line + "\n")`), and `Battery.__init__` (`:395-398`) stores the same instance as
`self.hb`. The calls reachable before `return 0` at `:1280` are:

1. `:1270` `hb("GATE-DET1", …)`
2. `:1272` `hb("GUARD", …)`
3. `:586` `self.hb("GATE-SHA", n, n, …)`, inside `gate_sha()` called at `:1276`
4. `:1279` `hb("GATE-SHA-ONLY", …)`

The next `hb` in the main path is `:629` (`GATE-FOLD`), reached from `:1295` — after the return.
**Four lines.**

**Round 2's H-1 item 4 got this right and named them individually**: *"the `--gate-sha-only` process
emits its own `GATE-DET1` / `GUARD` / `GATE-SHA` / `GATE-SHA-ONLY` heartbeat lines"*. v6 carries
round 5's incorrect count forward against round 2's correct enumeration.

The load-bearing limb of row 5's argument — the driver leg runs `load_frozen()` first, so its startup
is arena-class — is **correct** and I verified it. Only the line count is wrong. But it is a
checkable count in a justification attributed to two prior rounds, which is exactly the class round 5
flagged as I-2 against v5's *"`lab_dev` occurs exactly once"*.

---

### M-1. `V15E1:1581-1582` asserts in the present tense that §7.7 establishes the count row 64 changes.

```
1581| I-1 added Phase 1g at `1 × U11`, moving the total `2930.7 → 2933.9` on the `count = 1` reading
1582| that §7.7's `U9` measurement establishes; ...
```

`:1581` is charged as *"provenance 2930.7->2933.9 -- historical"*; `:1582` is returned by no sweep and
sits in the meta residue as a `U9` mention, where §6's question (*does its value move?*) correctly
answers no. But the clause is present tense: after row 64 lands, §7.7 establishes count **2**, and
this sentence still says §7.7 establishes count **1**. The numbers are unambiguously historical; the
attribution is not. Row 56 exists for exactly this ambiguity one paragraph up. Give `:1582` the same
treatment — an explicit historical marking, or *"on the `count = 1` reading §7.7 then carried"*.

### M-2. Sweep J's declared reason for `:1583` is sweep E's reason and does not address the hit.

Sweep J charges `V15E1:1583` as *"declared — 'once sklearn is restored' -- ordinary English"* —
verbatim sweep E's reason for the same line. In sweep J the line matched on `\b3\.8\b`
(*"`3.8 s` once `sklearn` is restored…"*), not on `once`. The **disposition is correct** (that `3.8`
is the unit, which does not move), but the printed reason explains a different pattern's hit.
Cross-sweep charge identity is otherwise a virtue of the method; here it hides a hole.

### M-3. §10's extent-only heading counts two and lists three.

*"**Extent-only lines — 2, named**"* is followed by three bullets and then *"These three are stated
rather than patterned into existence."* The third bullet (row 48) is a siteless **row**, not an
extent-only **line**. The counts are individually defensible; the heading and the body disagree.

### M-4. §6's residue unit list writes `U3–U6` as a range spanning `U5a`/`U5b`.

The set and the count (16) are correct and I verified both, but no symbol `U5` exists in §8; the
range notation implies one. Write the sixteen out, as §6 does for the twelve phases.

### M-5. The delta lists some `unchanged — CORRECT` rows and omits others.

§9 names rows 1, 35, 56, 57, 58, 65 (all CORRECT) but omits 18, 45, 46, 47, 51 (also CORRECT). No
edit is dropped — I checked all 67 — but a reader reconstructing the row set from §9 will not get a
partition. State the convention, or list all of them.

---

## OBLIGATIONS FOR A V7 THAT WOULD CARRY GO

1. **H-1**: rows 67 (`V15E1:1814`) and 68 (`config:42`); **sweep K** keyed on the process
   decomposition, with hit list, charges and subtraction; §6's meta-check conclusion re-scoped to
   what it proves.
2. **I-1**: row 64 → `:1368-1369`, added to §1's multi-line list, `:1369` named extent-only.
3. **I-2**: rows 5 → `:1644-1646` and 54 → `:2439-2440`, both added to §1's list, `:1644` and
   `:2440` named extent-only; row 44's extent/label reconciled with `:1820`'s charge.
4. **I-3**: *"three lines"* → **four**, naming them (`GATE-DET1`, `GUARD`, `GATE-SHA`,
   `GATE-SHA-ONLY`), with round 2's enumeration cited rather than round 5's count.
5. **The five minors.**
6. **Carry forward at full strength, everything I re-derived independently**: the ten sweeps'
   byte-identical reproduction including order; the partition `265 → 240 = 128 + 112`,
   `UNCHARGED = 0`, 25 excess / 21 multi-sweep; the ten per-sweep subtractions; zero divergent
   charges; the 26-row multi-line list; row 48 as the only siteless row; the three extent-only
   reasons; the meta-check `102 = 30 + 72` with its 12 phases and 16 units; **21 phase labels and 18
   unit symbols**; *"two phases move, zero units move"*; **Phase 1c's `67` confirmed at source**
   (`load_frozen` opens no `.pt`; `load_ro` first at `:1284`, all after `:1278-1280`); the fourth
   live-wrong claim and rows 43/66; `expected_sha_dev_opens = 2 × 2 = 4` under `is_dev_like`, with
   the design document neither dev- nor test-like under both names; `21 + 16 = 37 → 22 + 16 = 38`;
   row 45's `73 + 1 = 74` via `atexit`-flushed per-pid ledgers; §8's full arithmetic (`3673.9`,
   `4592.4`, shares and minutes unchanged, `4687.7`/`7729.1`, margin `0.083`); and §9's delta
   verified complete and non-duplicative over all 67 rows.

**The delta's substance still does not grow.** Two rows, one sweep, three site-range extensions and a
corrected numeral. No code change beyond what v6 already prescribes.

---

## WHAT V6 STILL GETS WRONG — SUMMARY

v6 does the thing this lineage has been reaching for since v1: it stops arguing that the enumeration
is complete and builds an instrument that can be re-run, then submits to having that instrument run
against it. Ten commands, ten byte-identical lists, a partition that closes in both directions, and a
meta-check that reproduces to the hit — I checked every one and found no discrepancy anywhere in the
machine. The Phase 1c ruling is right at source, the fourth live-wrong claim is a real find that no
obligation asked for, and replacing v5's contorted `ORPHAN = 0` with three named lines and their
reasons is the honest move. What v6 still gets wrong is that it answered *"could a family be missed?"*
by sweeping **§8's units and phases**, when the family it missed twice is the **process inventory** —
and the process inventory has two sites that write the decomposition and never write the total, so
`\b7[234]\b` cannot see them and neither can `\bU[0-9]+\b|[Pp]hase [0-9]+`. One of those two is
`V15E1:1814`, §12's declaration of `processes reporting` as *"66 mints + 6 fidelity + 1 arena"*
marked **"yes — HALT on any mismatch"**: the design-side statement of the very binding predicate
round 1's H-1 opened this erratum on, never charged in six rounds, and left declaring **73** in a
document that would assert **74** in four other sections. The other, `config:42`, sits one line below
a config line the erratum already moves. The three smaller findings are of a piece with each other:
row 64, row 5 and row 54 each declare a site one line narrower than the sentence their own
prescription rewrites, and in row 64's case the uncharged line is the one carrying the words *"count
is `1`, the arena alone"* — which is round 5's I-3 defect, reproduced at the row v6 added to answer
round 5's H-1. None of them loses a site at landing. All of them are the gap between a delta a reader
can verify and one a reader has to trust, which is the whole of what §1 claims and very nearly all of
what v6 delivers.

---

## BLINDNESS AND EDIT STATEMENT

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no arena run; no `--gate-sha-only` leg run; no GPU, no
SLURM job, no commit, no `TARGET_STATE.json` edit. `artifacts/c06_falsifier/` does not exist and was
not created (verified by `ls`).

**Compute used:** `sha256sum` over the five artifacts and the six proposals; file, review and source
reads; the ten `grep` sweeps of §10 run verbatim from their printed commands; the §6 meta-check
sweep; **twenty wider sweeps of my own construction** (spelled-out numerals; the `U11` measurement
band and the `0.083` margin; `0.1 s`/`0.2 s`; startup/arena-class vocabulary; the process
decomposition; `\b6[78]\b`; unanchored `3\.8|7\.6`; unanchored `2642|2508|3670|3673|4587|4592`;
shares and miss multiples; `GATE-SHA`; `lab_dev`/`headspace_fidelity`; `driver`; `elapsed|ratio|
denominator`; `mints_executed`; `every|all|each process`; `design|sha256|digest`; `\b7[0-9]\b`;
`priced|re-price`; `U11|U7|U9`; `process(es)?`), each diffed against the 240 charged sites;
`c09guard.is_dev_like` / `is_test_like` / `is_banked_trainlog` probes over the design document under
both names and over the full 21-entry concatenated iterable; static reads of
`c06_falsifier_arena.py`, `c06_falsifier_mint.py`, `headspace_fidelity.py`, `c09guard.py`, the config
and the sbatch; and arithmetic. Every count, list and partition figure above is produced by a script
over the sweep output, not transcribed.

**Nothing was edited.** All five artifacts and all six proposals carry the hashes recorded in §1,
re-verified after this review. My only write is this file.

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`. **The
battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under v1–v6 as
specified.**
