# C06 `$0` falsifier — independent design review, **ROUND 12**

**Reviewer:** fresh, independent of rounds 1–11 and of the designer.
**Artifact:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V12.md`, sha256
`3504ce8a5850336fb1a4b28c5fefb21e43b52f71c9591100ea260d298a65eb1b`, 159793 bytes, 2177 lines —
**recomputed on disk, matches the review request.**
**Compute:** read-only. No GPU, no SLURM, no Modal, no arena run, no mint, no cache write, no
test-split access, no commit. `TARGET_STATE.json` read, not modified. **No repository file was
written or edited except this review**; the draft, the configs and the state are untouched, and every
temporary artifact — script extractions, spliced counterfactual drafts, timing probes — went to the
session scratchpad. The one `--out` target of the `headspace_fidelity.py` timing runs was the
scratchpad.

---

# VERDICT

## **REVISE — 0C / 0H / 1I + 3M**

Zero Critical. Zero High. **One Important.** Three Minor, none of which touches the verdict path.

The science layer is closed and I re-derived it rather than inheriting it: 13/13 arms rebuilt
bit-exactly, 26/26 `ρ` at 6 dp, 16/16 C01 accuracies and 16/16 net-fix integers, 12/12 `GATE-FLOOR`
anchors, 37/37 digests, all twenty gates re-derived as unable to fire on a warranted CLOSE. **The
record is clean: 4/4 limbs faithful and complete, both Repair paragraphs subtracting to bare
scaffolding, no undisclosed deviation, no repair claimed that the artifact does not contain, no
repair landed narrower than prescribed, and zero stale totals.** Round 11's two findings are both
fully discharged.

**The one Important is in §8 Phase 1g — the row round 11's I-1 created.** Not in its count, which
I re-measured and confirm is `1`, and not in the number it carries, which is `3.2 s` and is right to
within `0.2 s`. It is in the **measured basis** the row states for that number. Both "arena-class"
interpreter+import measurements — round 11's `1.84–1.91 s` and v12's `1.82–1.85 s`, which I
reproduced at `1.81–2.03 s` — were taken over an import set that **omits `sklearn`**, and `sklearn`
is pulled in by `headspace_mint.py:68`, a §11 frozen import, and imported directly by
`headspace_arena.py:35-36`, the module §8 derives its own Phase 1b and 1f counts from. Restoring it
moves the arena's startup to **`3.00–3.27 s`** on this node. So `3.2 s` does not *bound* the arena at
`≈ 1.7×` as §14 states; it **approximates** it, and my slowest observation exceeded it. The projection
is unaffected in substance — the residual is `≤ 0.2 s` against a `733.5 s` margin — but §8's own
institutionalised lesson, printed nine lines above the row, is *"state the timing boundary, not just
the number,"* and the boundary here is the import set, which the document nowhere pins and which
swings the number by 65 %.

I considered and rejected both softer and harsher gradings, and I did so before knowing it would be
the only finding. It is not Critical: the brief's Critical column admits *"any un-counted loop in
§8"*, and this loop **is** counted — Phase 1g exists, its count is right, and no verdict quantity
moves. It is not High: nothing about the verdict's authority or scope changes, and the repair round
11 prescribed (*"× `U11`"*) was landed exactly as written, not narrower. It is not Minor: it is the
stated evidentiary basis of a row in the section `rule_1_compute_projection` governs, and the
same-class defect — a measurement whose enclosure is narrower than the thing it prices — is what
rounds 7, 8, 9, 10 and 11 each graded Important. **Softening it because a GO is one finding away
would be grading on trajectory, which the brief forbids in both directions; so would inventing a
finding to avoid looking captured.** I have tried to do neither.

---

# PART A — AUDITING THE AUDITOR

## A.1 The §14.2 script, re-run against final on-disk v12: **byte-identical, exit 0**

I extracted the script from the v12 fence, ran it unmodified, and captured stdout.

* **Exit code `0`.**
* Embedded transcript **1733 bytes**; my run **1733 bytes**; **`BYTE-IDENTICAL: True`**
  (sha256 of both: `c48c256d7f7e50c11d98b6997652a120e28b78b913b03763cafb065e2fc2e3de`).

The transcript is a **verified fixed point**, the fifth consecutive version for which that holds.

## A.2 `CHANGED §14.2 +0 chars` — the explanation verified, not accepted

v12 §14.1 (`v12:1953-1958`) explains this as same-length substitutions with `CHANGED` computed from
content (`SA[k] != SB[k]`) rather than size. I diffed §14.2 between v11 and v12 directly:
**`len(v11 §14.2) = len(v12 §14.2) = 7247`, `identical = False`.** The differing lines are exactly
**five**, and every one is a same-length version label:

```
-"""Mechanical disposition verification, C06 falsifier v11.     +... v12.
-(1) section diff v10->v11 ...                                  +(1) section diff v11->v12 ...
-V_OLD='...DRAFT_V10.md'                                        +V_OLD='...DRAFT_V11.md'
-V_NEW='...DRAFT_V11.md'                                        +V_NEW='...DRAFT_V12.md'
-print('=== (1) SECTION DIFF v10 -> v11 ===')                   +print('=== (1) SECTION DIFF v11 -> v12 ===')
-print('=== (5) ... (round-10 prescriptions) ===')              +print('=== (5) ... (round-11 prescriptions) ===')
```

The four substitution classes v12 names are the four that occur. **The explanation is exact.**

## A.3 Breaking the self-exclusion — and a fact about v12 the construction exposes

I spliced v11's §14.1 into a copy of v12 and re-ran, as rounds 10 and 11 did. The script printed
**`UNCHANGED §14.1 (self, size not reported)`** and dropped `14.1` from the changed-but-uncited
list — **but no row failed and it exited `0`.**

That is **not** a regression. **The construction is vacuous in v12 for a structural reason: no v12
row and no v12 limb cites §14.1.** Round 11's two findings land in §8, §7.7, §7.2, §9 and §7.9, so
there is nothing for the check to bite on. Rounds 9–11 could run it only because their findings
happened to land in §14.1.

**So I proved the mechanism live by construction instead.** I took the spliced document and inserted
one synthetic disposition row citing §14.1. Result:

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  X-9   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 2 ; rows failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**The self-exclusion covers size only and still fails a §14.1-citing row when §14.1 did not change.**
v12 §14.1's claim is *"The logic is unchanged in v12"* — which is precisely what A.2 and this test
establish, and it is notable that v12 states exactly that and does **not** claim the counterfactual
reproduces in v12. That is an honest and well-drawn sentence. **No finding.**

## A.4 Section deltas, recomputed with my own splitter

My splitter reproduces every printed delta exactly: `§8 +1176`, `§9 +249`, `§14 −2690`, `§15 +71`,
`§7.2 +887`, `§7.3 +268`, `§7.7 +1144`, `§7.9 +576`, `§14.2 +0`, `header −293`, `UNCHANGED: 47`.

**Independent corroboration of v12's `41 → 23` item-site attribution.** v12 §14.1 attributes the drop
by section, `§14.1 −9` being the largest term. My splice — which restores v11's §14.1 and nothing
else — printed **32 sites** against v12's 23. `32 − 23 = 9`, exactly the `−9` v12 claims for §14.1.
**The attribution is measured, not guessed.**

---

# PART B — DISPOSITION AUDIT OF ROUND 11'S TWO FINDINGS, AT LIMB LEVEL

## B.1 The four limbs: **4 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded**

Each quotation was normalised for emphasis and quote-glyph only, then tested for containment inside
the exact `R11:NNN-NNN` line range it cites, read out of round 11's file.

| # | finding | limb (opening) | cited range | verbatim | inside range | ruling |
|---|---|---|---|---|---|---|
| 1 | I-1 | *"Add one §8 row — "1g interpreter + imports, non-mint processes" …"* | `R11:308-311` | ✓ | ✓ | **FAITHFUL** |
| 2 | I-1 | *"State `U9`'s timing boundary in §7.7, one clause …"* | `R11:311-312` | ✓ | ✓ | **FAITHFUL** |
| 3 | I-1 | *"Amend §7.2's "already inside every unit" to name the scope …"* | `R11:312-314` | ✓ | ✓ | **FAITHFUL** (widening ruled at B.3) |
| 4 | M-1 | *""No separate interpreter line is added", or cite the label the row now carries"* | `R11:328-329` | ✓ | ✓ | **FAITHFUL** |

No word is dropped from any of the four, including every qualifying clause — limb 1 carries both
branches (`1` **and** `7`) and both totals (`2933.9` **and** `2953.0`), which is the place a
narrowing would have been most convenient.

## B.2 The residues, by subtraction

**I-1's Repair paragraph (`R11:308-314`), three limbs removed:**

> `Repair, three lines. (1) ⟦LIMB⟧. (2) ⟦LIMB⟧. (3) ⟦LIMB⟧.`

**Nothing but enumerative scaffolding and connective punctuation.** No prescriptive content survives.

**M-1's Repair sentence (`R11:328-330`), one limb removed:**

> `… Repair: ⟦LIMB⟧. (This is one sentence away from I-1's third line and is naturally fixed with it.)`

One clause remains, and it is **not a prescription** — it is a locational remark about where the fix
belongs, of the same class round 11 correctly ruled out of the limb column. **And it was in fact
obeyed:** both the M-1 sentence and I-1's third line land in the *same paragraph*, `v12:1092-1096`,
with the collision recorded at `v12:1098-1103`. A directive that is satisfied and has no landing
section of its own cannot occupy a *landed in* column.

**Residue verdict: clean. Nothing prescribed by round 11 is missing from v12.**

## B.3 Ruling on the §7.2 widening (§4.A): **WARRANTED WIDENING, correctly disclosed**

Round 11's limb 3 asked §7.2 to *"name the scope it is true of — **the mint units**"*. v12 names
*"every one of the 66 mint units **and inside `U9`**"* (`v12:1090`).

**I rule this a warranted widening, not a substitution requiring a deviation label**, on three
grounds.

1. **Round 11 could not have written the wider scope, and said so.** Its own line (2) exists because
   *"the document nowhere says it, and the count cannot be derived from what is written"*
   (`R11:296-297`). The narrow scope was round 11's best available formulation under an
   acknowledged unknown, not a prescription that `U9` be excluded.
2. **Executing line (2) determines line (3).** Round 11 explicitly coupled them — *"so the count in
   (1) is determined by"*. Once `U9`'s boundary is measured as a full-process wall, naming only the
   mint units would leave the six fidelity processes unaccounted in §7.2 while §8 Phase 1g's count
   of `1` depends on their being accounted. The narrow scope would have been true-but-incomplete in
   exactly the way that hid the tenth item for eleven rounds.
3. **The widened claim is true, and I verified it independently.** See B.4.

**And it is disclosed in place.** The limb cell itself reads *"The scope named is the 66 mint units
**and `U9`**, which is what the measurement establishes and is strictly more informative than the
mint units alone"* (`v12:1811`). A deviation that is stated in the limb row, in the row's own words,
with its warrant, is disclosed to the standard the record uses.

## B.4 §7.2 is true of every one of the 72 processes it now covers — checked, not inherited

* **66 mints.** Units are full-process walls measured around the `python …` invocation; the `40.39 s`
  unit contains a `33.0 s` internal timer and a `7.4 s` gap, and the gap exceeds the `3.05–3.18 s`
  startup. Adding a separate line double-counts. ✓
* **6 fidelity.** `U9 = 3.70 / 3.49 s` is a full-process wall — established at B.5 by my own
  measurement. Adding a separate line double-counts. ✓
* **The 73rd, the arena**, is excluded by name and priced at Phase 1g. ✓

**`66 + 6 + 1 = 73` accounts for every process §13 declares, and I found no seventy-fourth.** The
sbatch driver's own bash process is the only other process in the job; it is not a python process,
its startup is milliseconds, and §8's `30 s` declared slack covers ledger aggregation and JSON emit.
I do not raise it.

## B.5 Ruling on the count (§4.B): **`1` is correct, and the inference is sound — re-measured**

I re-measured both quantities the ruling turns on, on this node, under
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=""`, four runs each:

| quantity | v12's measurement | **mine** |
|---|---|---|
| `headspace_fidelity.py --seeds 0`, full process wall | `3.13 / 3.46 / 3.12 s` | **`3.158 / 3.116 / 3.146 / 3.084 s`** |
| the same module's interpreter + imports **alone** | `3.06–3.16 s` | **`3.055 / 3.055 / 3.088 / 3.074 s`** |
| bare interpreter | — | `0.016 s` |

**The inference is sound, and by a wider margin than v12 claims.** The payload of a `--seeds 0`
fidelity run is the *difference* — `≈ 0.03–0.10 s`. A frozen `U9 = 3.70 s` is **~40–70×** that. It
cannot be an internal timing. It is a full-process wall, conservative by `≈ 17 %` over my slowest
observed wall, and it therefore already contains the six fidelity startups.

**Phase 1g's count is `1`. The `7`-branch (`22.4 s`, total `≈ 2953`) is correctly not taken.** I note
that v12 reached this by measurement and explicitly declined the `--seeds 0` anecdote round 11 had
correctly refused to rely on — the right method, and `U9`'s row now states the timed region
(`v12:1182`), which is what round 11 asked for.

## B.6 Ruling on the unit (§4.B): **`3.2 s` is the right number, on a basis the document states wrongly**

Two separable judgements, and they part company.

**The choice of `U11` over the arena measurement is correct.** `rule_1` demands *"measured unit ×
explicit count"*; `U11` is a unit already in §7.7, measured on this node, and round 11's own repair
text specified *"× `U11`"*. Carrying above a measurement is the convention every other carried row
uses. **I would have ruled against using the `1.82–1.85 s` figure as the unit** — and B.7 now shows
that choice was not merely conventional but load-bearing, because the smaller figure is the wrong
one.

**The stated basis does not hold.** See I-1 below. This is the round's one finding.

---

# PART C — MY OWN VERIFICATION OF ALL TWELVE §3 ITEMS

| # | claim | result |
|---|---|---|
| **V1** | 37 sha256 match disk; elided cache rows resolved | **PASS.** Exactly **37** digest rows; **37/37 recompute identically.** Note: **eight** rows carry the `…` ellipsis, not four as the request states — all eight resolve unambiguously under `data/CLIP_Embedding/<ds>/` (`-ro_L24`, `-ro_ow_L24`, native `train`, native `dev_seen`, per dataset); the `-ro_L28` / `-ro_ow_L28` siblings exist and are demonstrably not the digested files. The four new-code paths of `v12:1563-1565` are **absent** from the tree, as claimed. |
| **V2** | `U9` is a full-process wall | **PASS**, re-measured — B.5. |
| **V3** | Phase 1g and the whole re-multiplied column | **PASS**, re-multiplied independently — C.1. |
| **V4** | The four limb quotations, by subtraction | **PASS** — B.1, B.2. |
| **V5** | Re-run the audit; byte-compare; verify the `+0 chars` explanation | **PASS** — A.1, A.2. |
| **V6** | Break the self-exclusion | **PASS**, by construction — A.3. The stock construction is vacuous in v12; I built one that bites and it exits `1`. |
| **V7** | Rebuild the arms; one bit-exact predicate; misreading under `2e-6` | **PASS.** All 13 arms rebuilt at `n = 744` one-hot and at the arena `n = 743`/`579`: `GATE-C01PARITY` **`0.000e+00`** both datasets, `GATE-ROWSUBSET` bridge **`0.000e+00`**. Un-normalised misreading **`1.878e-06` / `1.609e-06`**, both under `2e-6` — reproduced to the digit. `GATE-C01PARITY` states **one** predicate. See C.3 on what the prose does and does not determine. |
| **V8** | `ρ*`; all 26 `ρ_raw` at 6 dp; trained-head `0/18` | **PASS.** `ρ* = 0.968176 / 0.977223`; **26/26** `ρ_raw` reproduce at 6 dp; both runners-up reproduce; the `float64`-accumulation claim is exactly right — `orthrot_83p8` is the **only** one of the 26 whose sixth decimal moves (`2.48e-07`), and all 26 agree at 4 dp under both reductions. Trained-head **0/18 on both**, min/median/max reproducing to the digit. The `1.301e-03` shift from including the masked row reproduces. |
| **V9** | Holm counterexample; `n ≤ 12`; §3.7's two blocks, two verbs | **PASS.** The 92-family re-derives (`(12+11) × 2 × 2 = 92`); `92 × 2 / 2001 = 0.091954 > 0.05` and `46 × 2 / 2001 = 0.045977 ≤ 0.05`, so the resolution floor is real. §3.7 has two blocks with two distinct verbs — *computed from the arena* vs *read and asserted equal* — with the `<=` operator correctly moved to the read block. |
| **V10** | §7.9's sum | **PASS.** Heading reads *"v1–v12"*. `7+1+4+0+0+0+0+0 = 12` ✓; `22+4+2+1+1+1+1 = 32` ✓; `89+21+6+3+3+3+3 = 128` ✓. Agrees with §7.8, with §7.2's *"all seven mints"*, and with the footer's *"twelve CPU head mints across v1–v12"* and *"v12 trained no heads"*. |
| **V11** | §6 has 20 rows, `12 G / 6 L / 2 R`; §13.1 defines 26 contiguous items | **PASS.** Exactly **20** rows; scope column **12 G / 6 L / 2 R**; the G-set and L-set match §5.6's two lists **name for name**, symmetric difference empty both directions. §13.1 defines `(1)…(26)` contiguously — no gap, no repeat. Items 10, 15, 19 and 22 carry their round-7/8/10 repairs, including item 22's *"`GATE-FLOOR`'s vote is computed in the arena process, not in the mint"*. |
| **V12** | §7.2 true of all 72; §9's arena clause conflicts with nothing | **PASS on §7.2** (B.4). **PASS on §9's conflict question** (C.2) — but §9 inherits the same understated measurement as §8, and is cited in I-1. |

## C.1 §8 re-multiplied, every row, by my own arithmetic

I re-entered the entire printed product column and summed it without reference to the stated total:

```
printed-column sum = 2933.9000        min = 48.8983
× 1.25             = 3667.3750 s      = 61.1229 min
mint sum           = 2508.3   share   = 85.4937 %   -> 85.5 %
Phase 3 share      = 9.3289 %                       -> 9.3 %
2× miss on Phase 3 = 3207.6000 s = 53.4600 min
5× miss on Phase 3 = 4028.7000 s = 67.1450 min
Phase 1c  67 × 0.033 = 2.211 -> 2.2      Phase 1f  150 × 0.0041 = 0.615 ; 150 × 0.0044 = 0.66
Phase 1g  1 × 3.2 = 3.2                  2933.9 − 3.2 = 2930.7   (round 11's pre-repair total)
stated base: 2927.6 + 1.0 + 0.7 + 0.1 + 1.3 + 3.2 = 2933.9  ✓
```

**Every figure in V3 is confirmed**, including that `2933.9 × 1.25 = 3667.375` rounds to `3667.4`
and that the `7`-branch would have given `3691.25 → 3691.3`. **§8 has exactly 26 rows.**

**Stale-total sweep: zero, as in round 11.** `85.6 %` occurs **0** times; `85.5 %` three times. Every
occurrence of `2930.7`, `2930.4` and `3663.4` is either §8's own provenance narrative (`v12:1363-1366`)
or inside the verbatim round-11 limb quotation (`v12:1809`) or the header's change summary — all
correctly historical. Counts 174, 150, 92, 3072, 168, 72, 240/540 all re-derive.

## C.2 §9: no conflict with the `~15 s` bound or the echo discipline

The longest un-instrumented span remains a single `GATE-C01PARITY` dataset at `11.27 s`
(`14.1 s` conservative). The arena's startup — whether at v12's `1.82–1.85 s` or at my corrected
`3.00–3.27 s` — is far under `15 s`, so **the interval bound is untouched under either reading, with
`~12 s` of headroom.** `rule_2_heartbeat` changes in no respect. The new clause's assertion that the
bash driver's unbuffered echo *brackets* the arena startup is carried to the code lineage by §13.1
item 12, which binds *"all six §9 items"* — so it is checkable rather than merely asserted. **No
finding.**

## C.3 What §3.4's prose does and does not determine — checked, and closed

Rebuilding the 13 arms required four facts §3.4 itself does not state: the Givens sign convention
(`first = cos·std + sin·ow`, `second = −sin·std + cos·ow`); that the arm-name suffix *is* the angle
in degrees; that `l2_rows` turns a masked row into an exact-zero row; and six of the thirteen arm
formulas. **All four are recoverable from the document alone** — from §1's `θ = 0` / `θ = 45`
identities (`v12:106-110`), §3.5's arm-name table (`v12:277`) and §3.7 — and **nothing external was
supplied**, which is the load-bearing claim and which my rebuild's `0.000e+00` confirms. §13 item 23
already discloses the formula-map gap to the code lineage.

I checked whether the *undisclosed* three could produce a wrong verdict rather than a HALT, and they
cannot. A sign flip on `second` makes `orthrot_45` cease to reproduce `common_displacement`;
**`GATE-ZEROOP` requires identical predictions on exactly that pair** and §13.1 item 14 requires the
guard arms be built by the rotation route and never aliased. The failure mode is a gate firing —
`INSTRUMENT_FAILED`, then HALT under §5.6 — not a published CLOSE. **Fail-safe. No finding**, though
a future code lineage will want §13 item 23 read broadly.

## C.4 `GATE-IDPARITY`'s scope — considered, and dismissed

`GATE-IDPARITY` reads *"every ro cache's `ids` order and `labels` identical to the native bank"*
(`v12:875`), while §11 lists exactly four ro caches. I considered whether *"every"* is
under-determined. It is not: §3.1 states *"no `dev_seen_*-ro_*` file is opened by any phase; the
`test_seen` ro caches are opened by nothing"* (`v12:148-149`), and §11's input table is the
authority on which caches exist for this battery. The quantifier ranges over four files and an
operator can enumerate them. **No finding.**

---

# FINDINGS

## CRITICAL — none

## HIGH — none

## IMPORTANT

### I-1. §8 Phase 1g's unit is carried on a measurement whose import set omits `sklearn`. Both "arena-class" figures (`1.82–1.85 s`, `1.84–1.91 s`) were timed over a strict subset of the arena's imports; the arena's real startup is `3.00–3.27 s`, so `3.2 s` does not bound it at `≈ 1.7×` — it approximates it, and my slowest observation exceeded it.

*Attaches to:* §8 Phase 1g's unit cell (`v12:1341`); §14's unit paragraph (`v12:1789-1796`,
specifically *"`3.2 s` bounds them at `≈ 1.7×`"* at `v12:1791`); §9's arena-startup clause
(`v12:1422`); §15 item 2.

**What both parties measured.** Round 11 states it timed *"an arena-class interpreter+import (numpy,
torch, faiss, the frozen `c01_policy_contrast_a0` and `mechfix_ops` modules)"* at `1.91 / 1.84 /
1.84 s`; v12 reports `1.82–1.85 s` for the same class. **I reproduced that set exactly** —
`1.844 / 2.033 / 1.812 s`. The measurement is real and the two parties agree. **The question is not
the timer; it is what the timer enclosed.**

**The decomposition, measured on this node, three runs per rung.** I built the arena's import set up
one module at a time:

| import set | wall |
|---|---|
| bare interpreter | `0.017 s` |
| `+ numpy` | `0.11 s` |
| `+ torch` | `1.79 s` |
| `+ faiss` | `1.82 s` |
| `+ c01_policy_contrast_a0 + mechfix_ops` — **the set round 11 and v12 timed** | **`1.81–2.03 s`** |
| `+ sklearn.metrics, sklearn.model_selection` | **`3.00–3.03 s`** |
| `+ mechnov_pairverify + vsw_pregate + headspace_mint` — **`headspace_arena.py`'s actual set, plus c01** | **`3.05–3.09 s`** |
| `+ runtime_block()`'s deferred `threadpoolctl`/`scipy`/`sklearn` | **`3.12–3.27 s`** |

**One module accounts for essentially the whole gap: `sklearn`, at `≈ 1.2 s`.**

**Why `sklearn` is in the arena, from the document's own sources.**

1. **`headspace_mint.py` is a §11 frozen import** with its sha256 asserted at run time (`v12:1495`),
   and it imports `from sklearn.model_selection import StratifiedKFold` at **top level**
   (`headspace_mint.py:68`). Any process importing it pays `sklearn`.
2. **`headspace_arena.py` — the module §8 derives its own counts from** — imports
   `sklearn.metrics.roc_auc_score` and `sklearn.model_selection.StratifiedKFold` **directly**
   (`headspace_arena.py:35-36`) and calls `StratifiedKFold` at `:72`. §8's Phase 1b decomposition is
   justified by *"`headspace_arena.py:75-89` loads `mint_{ds}_s{seed}_f{fold}.npz` inside the fold
   loop"* (`v12:1326-1328`), and §2's primary-source list cites `headspace_arena.py:59, :75, :85,
   :89, :92-93`. **The document treats that file as the arena's structural model everywhere except
   when timing its startup.**
3. The arena also needs `torch` (`U8` is *"ro cache `torch.load`, 2 files"*) and computes macro-F1
   (`v12:450`, `v12:622`), whose natural source is `sklearn.metrics`.

**What is and is not wrong.** The **count** is right (`1`). The **number** is right to within
`0.2 s` — indeed `3.2 s` sits inside my measured `3.00–3.27 s` band, and choosing `U11` over the
`1.82–1.85 s` figure is what keeps the row approximately correct. What is wrong is the **stated
basis**: the row presents `1.82–1.85 s` as *the* arena measurement and `3.2 s` as conservative
*relative to it*, and §14 quantifies that conservatism at `≈ 1.7×`. There is no `1.7×` margin.
There is roughly none, and against `runtime_block()` there is a small deficit.

**On materiality, stated plainly so the severity is auditable.** Worst case the row moves
`3.2 → 3.4 s` and the total `2933.9 → 2934.1 s` — `0.007 %`, inside the `× 1.25` margin of `733.5 s`
and inside the `30 s` declared slack. **No heartbeat interval changes** (C.2), **no verdict quantity
is touched**, and the falsifier's `$0` character is unaffected. The defect is evidentiary, not
arithmetic.

**On severity.** Not Critical: Phase 1g is a **counted** row and no verdict quantity moves. Not High:
round 11's repair was landed as prescribed, not narrower, and the verdict's authority and scope are
untouched. Important is the grade the brief defines for *"an argument right for a weaker reason than
available"* and for completeness, and it is the grade rounds 7–11 gave the same class — a measured
quantity whose enclosure is narrower than the thing it prices. §8's own paragraph at `v12:1369-1377`
institutionalises exactly this: *"the spread is about **what each timer enclosed**, not about the
machine … **state the timing boundary, not just the number.**"* Phase 1g states a boundary and omits
the set, and the set is worth 65 % of the number.

**Repair — two lines, and the second is the durable one.**

1. **Re-state Phase 1g's basis honestly** (`v12:1341`, and correspondingly `v12:1422` and
   `v12:1789-1796`). Keep the unit at `U11 = 3.2 s` and keep the count at `1`. Replace the
   `1.82–1.85 s` / `≈ 1.7×` framing with the measured decomposition: an import set of
   `numpy + torch + faiss + c01 + mechfix_ops` measures `1.81–2.03 s`, and adding `sklearn` — which
   `headspace_mint.py:68` and `headspace_arena.py:35-36` require — takes it to `3.00–3.27 s`, so
   `3.2 s` is **approximately the measured arena startup rather than a bound on it**, with the
   residual `≤ 0.2 s` absorbed by the `× 1.25` margin and the `30 s` declared slack. Say that the
   direction is no longer strictly conservative and why it does not matter here.
2. **Pin the arena's import set in §13**, as one clause on item 12 or a new item: *the arena imports
   `headspace_mint` (hence `sklearn`), `c01_policy_contrast_a0`, `mechfix_ops`, `numpy`, `torch`,
   and calls `runtime_block()`* — whichever the executable will actually do. This is the exact
   analogue of round-11 I-1's second line: **the count was undeterminable until §7.7 stated `U9`'s
   boundary; the unit is undeterminable until §13 states the arena's import set.** With it stated,
   the code lineage can check the number instead of a reviewer having to guess the set.

---

## MINOR (non-blocking; none touches the verdict path)

* **M-1. §7.3's `97` does not reproduce, and cannot be right under the convention that makes its
  neighbour right** (`v12:1124`). Grepping every decimal in the closed interval `[0.6, 0.99]` across
  the drafts, I get **116** distinct for v1–v10 — reproducing round 10 exactly — and **118** as the
  total through v12 on that basis, but **98**, not 97, for v1–v5. `97` is obtainable only
  under a half-open interval, which would make round 10's figure 115. The cause is benign and
  identifiable: the tokens `0.6` and `0.99` occur in the whole corpus **only** inside the literal
  string `` `[0.6, 0.99]` `` — the audit's own description of its own interval, first appearing in
  v5. Excluding both self-references gives the consistent pair `(96, 114)`. **Non-blocking**, and
  emphatically so: **the load-bearing claim came out stronger than the document states.** The
  new-in-v12 set in `[0.6, 0.99]` is **empty** — v12 introduces no in-band decimal at all — and its
  twenty genuinely new decimals anywhere on the number line are `0.4`, `1.82`, `1.84`, `1.85`,
  `1.91`, `3.06`, `3.08`, `3.12`, `3.13`, `3.16`, `3.46` (seconds), `2933.9`, `2953.0`, `3207.6`,
  `3667.4`, `3691.3`, `4028.7` (second totals), `48.9`, `53.5` (minutes) and `85.5` (a share).
  **All timings or arithmetic on timings; no accuracy.** I also classified all 81 in-band values
  present in v12 and every one has a verified non-arm provenance — published C01 dev accuracies
  recomputed from confusion matrices, `GATE-FLOOR` anchors matched against the banked OUT JSONs, the
  26 `ρ`, majority/band constants, `‖Δ‖` geometry, `0.95` and `0.65`, timings, and the two
  self-referential interval endpoints. **No battery-arm accuracy anywhere in v1–v12.** Round 11's
  `0.615` / `0.66` and round 10's `0.8718` both reproduce exactly. Repair: print `98`, or state the
  convention once.

* **M-2. §6.1's trained-head `ρ` figures are not reproducible from the sentence that reports them**
  (`v12:943-946`). The banked `K_train` arrays are **not** unit-norm as stored; taken literally the
  36 matrices give `0.0586 / 0.0906 / 0.1824` (HateMM) and `0.0471 / 0.0899 / 0.1409` (MHC-ZH),
  5–10× under the quoted values. Row-renormalising reproduces all six figures to the digit and
  `0/18` on both. §6.1's opening does say *"over unit keys"* (`v12:892`), so this is **recoverable,
  not wrong** — but fifty lines separate the licence from the measurement. **Non-blocking and
  structurally incapable of touching the verdict:** §11 declares these 36 mints *"inputs to §6.1's
  **reference measurement** only — no gate reads them"*. Repair: three words at `v12:943`
  (*"on row-renormalised keys"*).

* **M-3. §6.2's clearance band is a decimal short** (`v12:955-956`). *"clearing by `0.15`–`0.23`"*;
  the four measured clearances are `0.1499 / 0.2302 / 0.2395 / 0.1755`, so the upper end is `0.24`.
  **Non-blocking**: §6.2 is the *retirement* rationale for `GATE-ARMVIAB`, a gate that no longer
  exists; every clearance clears; nothing reads the band.

---

# REQUIRED RULINGS

## 1. §4.D — can any gate fire on a warranted CLOSE? **No, for all twenty. Derived from the gate texts and from my own measurements.**

A *warranted CLOSE* is: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction.** `GATE-DET1` (thread env, and I
hit the CPU-only guard myself). `GATE-SHA` (37 digests over files no phase writes — **all 37
recomputed**). `GATE-FOLD` (banked parity flags + `fold_of`). `GATE-FLOOR` (native deployed
reproduction of six banked anchors — computed on **native** keys, so no ro-derived arm's outcome can
reach it; all twelve anchor values verified against the banked OUT JSONs). `GATE-POP` (populations,
class counts `(297,446)` / `(180,399)`, index-set identity, constants recomputed — all verified).
`GATE-C01PARITY` (the builder reproduces `prepare_views` bit-exactly — a property of the *builder*,
`0.000e+00` measured). `GATE-ROWSUBSET` (`743` vs `744` restriction — builder property,
`0.000e+00`). `GATE-RHORAW` (26 frozen `ρ_raw` at 4 dp — a property of the ro caches and the raw leg,
identical for both lineages; **26/26 reproduced at 6 dp**, and all 26 agree at 4 dp under both
reduction orders). `GATE-NULLREMOVED` / `GATE-ZEROMASK` (`{355}` / `{}`, verified as the sole
exact-zero row in both modalities of both ro caches). `GATE-IDPARITY` (ids/labels parity).
`GATE-LEDGER` (declared counts). **None reads which arm won.**

**The six per-lineage gates, one at a time.**

* **`GATE-ARENA`.** Its **lower** bound is on `endpoint_std` **only** — the reference arm, not a
  real-vs-rotation quantity. A warranted CLOSE says nothing about `endpoint_std`; if that arm cannot
  clear `majority + 0.02` the instrument is genuinely dead. Its **upper** bound (`≤ 0.98`) fires only
  on implausibly high accuracy, which is a leak signal, and the measured headroom is large.
* **`GATE-ORBITDISP`.** Fires iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D` — head space *more degenerate than
  the raw family*. I measured trained deployed heads at roughly **half** the bar, `0/18` on both
  datasets. Arm-outcome-independent.
* **`GATE-NESTED`.** The scoring head excluded its fold. Structural.
* **`GATE-SELFTEST`.** `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is an **identity**; it holds
  whatever the accuracies are.
* **`GATE-ZEROOP`.** `orthrot_0 ≡ endpoint_concat` and `orthrot_45 ≡ common_displacement` are
  **algebraic identities of the Givens family**, verified in §1 at residuals `8.941e-08` and
  `1.192e-07`. Independent of which arm wins.
* **`GATE-ALGEBRA`.** Key-level `≤ 2e-6` on the same two identities. Same argument.

**The two `R` gates** (`GATE-DOMAIN`, `GATE-DEVFID`) carry **no bar** and cannot fire at all.

**All twenty: no gate can fire on a warranted CLOSE.** I also checked the one recognised failure path
(§5.7's Head-N `GATE-ARENA` lower-bound miss): it drops Head-N, and §5.6 then yields HALT or a
Head-R-only SURVIVE — **never a CLOSE on one lineage.**

## 2. Verdict-path enumeration — mine, from the document alone: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure**

Let `G` = all twelve globals pass; for each lineage `L ∈ {Head-N, Head-R}` let `p_L` = passed all six
per-lineage gates **on both datasets** (§5.6's dataset-axis rule), and `c_L` = clears S1–S7 on both
datasets.

| `G` | `p_N` | `p_R` | outcome | rule |
|---|---|---|---|---|
| fail | any | any | **HALT** `INSTRUMENT_INCONCLUSIVE` | 3 |
| pass | ✓ | ✓ | **SURVIVE** if `c_N ∨ c_R`; else **CLOSE** | 1 / 2 |
| pass | ✓ | dropped | **SURVIVE** if `c_N`; else **HALT** (rule 2 needs *both* passed) | 1 / 3 |
| pass | dropped | ✓ | **SURVIVE** if `c_R`; else **HALT** | 1 / 3 |
| pass | dropped | dropped | **HALT** | 3 |

**Exactly one published state per combination; no unmapped outcome; no overlap.** `c_L` is never
evaluated for a dropped lineage — its quantities are `INSTRUMENT_FAILED` and enter the S4 family only
as `NOT_TESTED` with `p = 1`, the family staying frozen at 92.

**The declared-drop exemption is the only lawful absent-quantity path**, stated in terms at
`v12:763-764`: *"Absence by declared drop is lawful; absence by computation failure in a surviving
lineage still HALTs."*

**No gate failure is reportable as a closure.** CLOSE requires all twelve globals to pass **and**
both lineages to have passed every per-lineage gate on **both** datasets. A global failure HALTs; a
per-lineage failure drops that lineage on both datasets, falsifying rule 2's conjunct, so the only
reachable outcomes are SURVIVE-on-the-clean-lineage or HALT. **A CLOSE always rests on two clean
negatives, never one.**

## 3. Rulings on §15's five open issues

1. **The subtraction on four limbs, and the widening.** Executed in full — **4/4 FAITHFUL**, I-1's
   Repair paragraph subtracting to `Repair, three lines. (1) ⟦LIMB⟧. (2) ⟦LIMB⟧. (3) ⟦LIMB⟧.` and
   M-1's to its limb plus a non-prescriptive locational parenthetical that was obeyed anyway.
   **The widening is WARRANTED** and is disclosed in the limb cell — B.3.
2. **The count and the unit.** **Count `1`: confirmed by my own measurement**, with a wider margin
   than v12 claims (`U9` is ~40–70× the payload). **Unit: `U11` was the right choice** and the
   `3.2 s` figure is right to within `0.2 s` — **but the basis stated for it is not**, which is I-1.
3. **The eleventh item.** **I searched three axes and name them: per-output-line, per-process-deferred
   (function-level imports), and per-process-for-the-non-python-process.** Results: the heartbeat's
   own line writes — a genuinely new axis, `≈ 2416` lines across the battery — price at
   **`4.5 µs/line = 0.011 s` measured**, below the `sub-0.1 s` class §8 already carries at its upper
   bound; **not a finding, and I report the number rather than the impression.** Deferred imports
   (`c01_policy_contrast_a0:1050-1052`, `mechfix_ops:161`, `headspace_mint:83-85, :236`) are all
   either inside a priced full-process wall or already resident — `runtime_block()`'s `scipy`/
   `sklearn` cost `≈ 0.06 s` because `headspace_mint` has already imported `sklearn`; that same fact
   is what produced **I-1**, so this axis did yield, but as a *unit* defect rather than an uncounted
   item. The sbatch driver's bash process is real, millisecond-scale, and covered by the `30 s`
   declared slack. **No eleventh uncounted item.** `66 + 6 + 1 = 73` holds.
4. **Seams in v12's own repair.** §7.2's re-scoped sentence is **true of all 72 processes** (B.4).
   §9's new clause conflicts with **neither** the `~15 s` bound nor the echo discipline (C.2) — but
   §9 quotes the same understated measurement and is cited in I-1. **The round's finding is once
   again inside the previous round's repair, on the one axis that repair created.**
5. **Record sound, design freeze-ready?** **The record is sound** — nothing prescribed is missing,
   nothing claimed is absent, nothing narrowed, zero stale totals. **The design is freeze-ready on
   everything except I-1**, and I-1 is a two-line documentation repair that moves no quantity a
   reader would act on.

## 4. Process rules

* **`rule_1_compute_projection`.** Satisfied in form — every §8 row is a measured unit × an explicit
  count, re-multiplying exactly to `2933.9`, with no extrapolation from a reduced-scale dry run
  anywhere. **The one row whose measured unit does not enclose what it prices is Phase 1g** (I-1).
  Twelve rounds, ten items, and the eleventh search returns **no new uncounted item** on any of the
  three axes I searched.
* **`rule_2_heartbeat`.** **Unchanged and satisfied.** Line-buffered per-phase appends through a
  `buffering=1` handle, plus an unbuffered bash echo per mint; progress file path stated in full
  (`artifacts/c06_falsifier/progress/C06_PROGRESS.txt`); longest un-instrumented span `11.27 s`
  (`14.1 s` conservative) against a `~15 s` bound. **v12 changes no interval**, under either reading
  of the arena's startup.

## 5. Freeze-readiness, operationally

**Ready except for I-1**, judged as the document an operator with no context would execute.

* **No decision point on the run boundary.** One `sbatch`, 8 CPU / 32 GB, no `--gres`, no `--time`,
  no array, no dependency, no requeue. **The 73-process order is now stated** — `66 mints → 6
  fidelity → 1 arena` (`v12:1613`) — with `GATE-SHA` once in the driver before any of them and
  `GATE-POP` before any population-consuming gate.
* **Preconditions are checkable.** All 37 digests recompute (V1); all four new-code paths are absent
  from the tree; `mints_present_before_arena` is a declared predicate.
* **Exit and resume semantics are defined.** The HALT path names the failing gate in its final line;
  a `RuntimeError` out of the imported C01 algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE`
  with its `context` string, and written before exit. Resume is handled explicitly and consistently
  in three places (`v12:155-159`, `:1592-1597`, `:1696`), including the correct refusal to make
  `dev_path_opens == 66` binding because it would HALT a legitimate resume.
* **The `$0` character holds.** No GPU, no Modal, no test contact, no new data.

## 6. Can the falsifier discharge the written condition at `$0`? **Yes.**

The condition is: re-run C01's battery in the fold-head arena on already-banked caches and see
whether the matched-norm orthogonal rotations again match the real prompt displacement. Every input
exists and is digest-frozen; the head space is re-mintable on CPU at measured cost; the arms rebuild
bit-exactly; the decision rule is pre-registered with its multiplicity resolution floor proved
attainable; and the verdict combination is total with one lawful absence path. **I-1 does not bear on
this and neither do the three Minors.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Nothing here authorizes execution. Before any job: (1) freeze with hashes; (2) a **separate**,
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — and I
note for that lineage that §13 item 23 should be read broadly (C.3) and that I-1's second repair line
is addressed to it; (3) main-dialogue authorization. This document is not authority to write
`TARGET_STATE.json`.

---

# CLOSING

**The most severe finding is I-1**, and what makes it worth a REVISE at round 12 is not its size —
`0.2 s` against a `733.5 s` margin — but where it sits and what it is made of. §8's own text
institutionalises the rule that a timing is only as good as the statement of what the timer enclosed,
and Phase 1g, the row created one round ago to fix a cost that no unit enclosed, carries a unit whose
measurement encloses a strict subset of the arena's imports. **`sklearn` alone is `1.2 s`, 65 % of
the figure, and it enters through `headspace_mint.py:68` — a module §11 freezes and asserts by
sha256 — and through `headspace_arena.py:35-36`, the file §8 cites when it derives its own counts.**
Two independent reviewers measured the same wrong set and agreed with each other, which is precisely
how a subset measurement survives: agreement between parties who made the same omission is not
corroboration.

The repair is short and the second line is the durable one — **pin the arena's import set in §13, for
the same reason round 11 made the document state `U9`'s timing boundary.** The count was
undeterminable until §7.7 said what `U9` enclosed; the unit is undeterminable until §13 says what the
arena imports. With that stated, a code lineage can check the number instead of a reviewer having to
reconstruct the set from three source files.

**Everything else is clean, and I say so as plainly as the brief asks.** The record is faithful at
limb level with nothing left in the residue; the science reproduces on every axis I tested,
independently and at full precision; all twenty gates are unable to fire on a warranted CLOSE; the
verdict path is total, mutually exclusive and admits exactly one lawful absence; and the eleventh
uncounted item does not exist on any of the three axes I searched. **v12 is one documentation repair
away from a GO, and I have declined to grant it early for the same reason round 11 declined to
downgrade its own finding.**
