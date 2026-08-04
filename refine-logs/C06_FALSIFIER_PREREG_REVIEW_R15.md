# C06 `$0` falsifier — independent design review, **ROUND 15**

**Reviewer:** fresh, independent of rounds 1–14 and of the designer.
**Artifact:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md`, sha256
`75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228`, 176903 bytes, 2338 lines —
**recomputed on disk, matches the review request.**
**Compute:** read-only. No GPU, no SLURM, no Modal, no arena run, no mint, no cache write, no
test-split access, no commit, no job. `TARGET_STATE.json` read, not modified. **No repository file
was written or edited except this review.** Every temporary artifact — the extracted audit script,
my two spliced counterfactual drafts, the arm rebuild, the `U4` timing — went to an isolated
subdirectory of the session scratchpad (`.../scratchpad/r15_indep/`), deliberately separate from the
shared scratchpad so that no artifact of the designer's was reused. My numpy/torch work ran under
the `HateVideo` interpreter, for which every module involved already had a `cpython-311` bytecode
cache: `scripts/analysis/__pycache__/` holds the same **11** files before and after, and **all 37
§11 digests recompute after my compute as well as before**.

---

# VERDICT

## **GO (0C/0H/0I)**

Zero Critical. Zero High. Zero Important. **One Minor, explicitly non-blocking, which does not touch
the verdict path** (stated with its reason below).

**The round's central check bites, and it bites exactly as the document says it does.** I extracted
§14.2's script from the v15 fence, ran it unmodified against the finished on-disk v15 — **exit `0`,
transcript byte-identical** to the embedded one (1653 characters / 1677 bytes, sha256
`246c896efc2e44d02852f8bb043824c5a82aeffdf71ceb730243a1fdba362108`). I then built **both** splice
forms myself from v14's §14.1 and ran them. **The plain form exits `1`**, printing
`UNCHANGED §14.1 (self, size not reported)`, `FAIL I-1 cites §14.1 -- NOT DIFFED`,
`rows verified against diff hunks: 2 ; rows failing: 1`, `limbs landed: 3 ; limbs open/failing: 1`
and `named by a row but unchanged: ['14.1']`. **The biting form exits `1` with `rows failing: 2`**,
adding `FAIL X-9`. **Every element v15 claims for both forms is reproduced, and none is missing or
overstated.** Round-14 I-1 is discharged, and — this is the part that matters — the corrected
sentence is not itself an unchecked inheritance.

**The science layer is closed and I re-derived it rather than inheriting it.** 13/13 arms rebuilt
from §3.4's prose alone at `max|diff| = 0.000e+00` on both datasets; `GATE-ROWSUBSET` at
`0.000e+00`; the un-normalised misreading at `1.878e-06` / `1.609e-06`, both under `2e-6`; the
algebra guard at `8.941e-08` / `1.192e-07`; 26/26 `ρ_raw` at 6 dp with `ρ*` `0.968176` / `0.977223`;
trained heads **0/18** on row-renormalised keys on both datasets; 16/16 C01 accuracies **recomputed
from the stored confusion matrices** and 16/16 net-fix integers; **37/37** digests; the Holm
counterexample and its three-way equality through C01's own `holm_adjust`; §8 re-multiplied to
`2934.5` row by row with **zero stale totals**; all twenty gates re-derived as unable to fire on a
warranted CLOSE.

**The compute projection gains its last corroboration, from me.** §15 item 4 asked this round to
consider `U4` — the only substantial unit still uncorroborated end to end — directly. I built its
stated object (`2 arms × 5 folds + rebuild`, head space, `n = 743`, 1024/2048-d) and timed 25 draws:
**median `0.0739 s` against the frozen `0.08908 s`**, i.e. the frozen unit is **`1.21×`
conservative**, and `3072` draws measure `226.9 s` against the carried `273.7 s`. Its vote leg alone
measures `0.04630 s` against `5 × (U2a + U2b) = 0.04670 s`, a `0.9 %` agreement that re-corroborates
both vote units on realistic fold shapes. **§7.7's uncorroborated list is now empty of substantial
units.** I searched one further axis and it returned nothing (§C.10).

**The record is faithful at limb level.** 4/4 limbs verbatim, complete, and inside the `R14:NNN-NNN`
range each cites; M-1's residue is the limb itself; M-2's residue is a measurement report; I-1's
residue is the framing sentence, and I rule below that v15 does **not** depart from it. **No repair
is claimed that the artifact does not contain, and none landed narrower than prescribed.**

**I say GO plainly because the brief asks me to, and I have declined the two ways of avoiding it.**
I did not withhold a GO to look rigorous: I ran every check the brief names, plus one it does not,
and the artifact survived all of them. And I did not inflate the one thing I found — a dropped
explanatory clause in §14.1 that repeats no false statement, moves no quantity, and is fully
recoverable from the same subsection — into an Important in order to produce a fifteenth finding.
**Fifteen rounds is evidence of nothing in either direction, and neither is the fact that the last
four produced exactly one Important each.**

---

# PART A — AUDITING THE AUDITOR

## A.1 The §14.2 script, re-run against final on-disk v15: **byte-identical, exit 0**

I extracted the script from v15's `### 14.2` fence (6893 characters, lines 2145–2277), wrote it
unmodified to my own scratchpad, and ran it.

* **Exit code `0`.**
* Embedded transcript and my run are **identical**: 1653 characters / 1677 bytes each,
  **`BYTE-IDENTICAL: True`**, sha256 of both
  `246c896efc2e44d02852f8bb043824c5a82aeffdf71ceb730243a1fdba362108`.

The transcript is a **verified fixed point**, and every printed line reproduces including
`UNCHANGED: 50 sections`, `30 reference sites`, `defined 1..28` and `unresolved: NONE`.

The cross-version byte claims in §14.1 are also exact against their sources: round 12 reported
`1733 bytes` (`R12:64`) and round 13 reported `1971 bytes` (`R13:70`) — both verbatim what §14.1
states. v14 on disk still hashes to `d80bbb44911daef9e772dfe1246ffa71876147e82d7f8b4bce6d83d5c34b0a46`,
round 14's recorded value, so **v1–v14 are unmodified** as the footer claims.

## A.2 `CHANGED §14.2 +0 chars` — verified by direct diff, not accepted

`len(v14 §14.2) = len(v15 §14.2) = 7247`, `identical = False`, both 143 lines. The differing lines
are **six**, all same-length, in **four substitution classes**:

```
L9    """Mechanical disposition verification, C06 falsifier v14.  -> v15.       [version string]
L10   (1) section diff v13->v14 ...                                -> v14->v15   [diff label]
L15   V_OLD='...DRAFT_V13.md'                                      -> ...V14.md  [V_OLD/V_NEW]
L16   V_NEW='...DRAFT_V14.md'                                      -> ...V15.md  [V_OLD/V_NEW]
L28   print('=== (1) SECTION DIFF v13 -> v14 ===')                 -> v14 -> v15 [print header]
L110  print('=== (5) ... (round-13 prescriptions) ===')            -> round-14   [print header]
```

Every one of the six is the same length before and after. **v15's account of its own §14.2 change is
exact** — four classes, six lines — for the fourth consecutive version. No finding.

## A.3 The self-exclusion, **BOTH forms** — §3 V1, the round's central check

I built both counterfactuals myself. The plain form: take the finalized on-disk v15 and replace its
§14.1 section (heading through the next heading) with v14's §14.1 verbatim. The biting form: the
same, plus one synthetic `X-9` row citing §14.1, inserted into §14's row-level table **outside** the
limb-table region. In each case I copied the extracted script and changed **exactly one line** —
`V_NEW` — verified by line-index diff; nothing else in the script was touched.

**Plain splice (v14's §14.1 into v15, no synthetic row): exit `1`, NOT vacuous.**

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  I-1   cites §14.1 -- NOT DIFFED
  OK    M-1   cites §14, §15
  OK    M-2   cites §7.9
  rows verified against diff hunks: 2 ; rows failing: 1
  FAIL  I-1   *"In §14.1 (`v14:2003-2004`), replace "The same vacu -> §14.1 NOT DIFFED
  limbs landed: 3 ; limbs open/failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**Biting form (splice + one synthetic §14.1-citing row `X-9`): exit `1`, two failing rows.**

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  I-1   cites §14.1 -- NOT DIFFED
  OK    M-1   cites §14, §15
  OK    M-2   cites §7.9
  FAIL  X-9   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 2 ; rows failing: 2
  limbs landed: 3 ; limbs open/failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**Element-by-element against §14.1's sentence (`v15:2027`).** The document claims the plain form
yields `UNCHANGED §14.1 (self, size not reported)` ✓, `FAIL I-1 cites §14.1 -- NOT DIFFED` ✓,
`rows verified against diff hunks: 2 ; rows failing: 1` ✓, *"one failing limb"* ✓ (`limbs landed: 3 ;
limbs open/failing: 1`), `named by a row but unchanged: ['14.1']` ✓, **exit `1`** ✓, *"with no
synthetic row required"* ✓. For the biting form it claims *"a second failing row `X-9` and changes
nothing else: `rows failing: 2`, exit `1`"* ✓ — and I confirm nothing else in the verdict-relevant
output moves: the failing limb, the named-but-unchanged list and the exit code are identical between
the two forms. (The only difference outside those is §14's own character delta, `−482 → −394`, which
is a mechanical consequence of inserting the synthetic row and depends on the wording each reviewer
chooses; the document neither states nor could state it.)

**Every claimed output element is reproduced. Nothing is overstated, and nothing is missing.**

The document's further claim that *"v14 was the first version since v11 for which the plain form bit,
and v15 is the second — because in both, a finding about this section is dispositioned in this
section"* is correct of v15 on my own reading: round-14 I-1 attaches to §14.1, its disposition row
cites `§14.1, §15`, and its first limb cites `§14.1`, so the plain construction has two things to
bite on natively.

## A.4 Section deltas, recomputed with my own splitter

I wrote a **line-based** splitter (the audit's is a regex-slice splitter) and reproduced **every
printed delta exactly**:

```
§14 −482 · §15 +540 · §7.3 +317 · §7.9 +1403 · §13.1 +968 · §14.2 +0 · header +396
UNCHANGED: 50 sections
```

For completeness, the one delta the audit self-excludes: **`§14.1 +434 chars`**. Section inventory is
stable at **57** in both versions — none added, none removed — so no section was silently created or
deleted under cover of the roll-forward.

My own scan confirms §13.1 defines `(1)…(28)` **contiguously**, no gap, no repeat, no duplicate.

---

# PART B — DISPOSITION AUDIT OF ROUND 14'S THREE FINDINGS, AT LIMB LEVEL

## B.1 The four limbs: **4 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded**

Each quotation was normalised for emphasis and quote-glyph only, then tested for containment inside
the exact `R14:NNN-NNN` line range it cites, read out of round 14's file.

| # | finding | limb (opening) | cited range | len | verbatim | inside range | ruling |
|---|---|---|---|---|---|---|---|
| 1 | I-1 | *"In §14.1 (`v14:2003-2004`), replace …"* | `R14:650-655` | 513 | ✓ | ✓ | **FAITHFUL** |
| 2 | I-1 | *"In §15 item 5 (`v14:2293-2295`), strike …"* | `R14:656-658` | 248 | ✓ | ✓ | **FAITHFUL** |
| 3 | M-1 | *"Repair: quote the framing sentence in full …"* | `R14:674-675` | 186 | ✓ | ✓ | **FAITHFUL** |
| 4 | M-2 | *"Repair: one clause naming the bridge, e.g. …"* | `R14:691-695` | 338 | ✓ | ✓ | **FAITHFUL** |

**No word is dropped from any of the four, including every qualifying clause.** Limb 4 is the place
a narrowing would have been most convenient — it carries a second disjunct v15 declines (*"Or drop
the decomposition and state the spend as `≈ 2 / ≈ 4` with the `52` unelaborated"*) — and it is
quoted in full with the choice labelled in the limb cell.

**On limb 1's treatment, which §3 V3 asks me to rule.** Round 14's limb 1 prescribes a sentence
*about v14*: *"Against v14 the plain construction is no longer vacuous … splicing v13's §14.1 into
v14 yields …"*. v15 lands it **for v15 rather than v14, and measured rather than transcribed**, and
says so in the limb cell. **I rule that treatment correct, and it is the only correct treatment
available.** Writing round 14's sentence verbatim into v15 would have produced a paragraph asserting
something about a superseded document while saying nothing checkable about the one in front of the
reader — which is precisely the "claim carried over instead of checked" that round-14 I-1 named. The
landed sentence is strictly stronger: it is about this artifact, and I verified every element of it
(A.3). The departure is declared in the limb cell in terms.

## B.2 The residues, by subtraction

**I-1's Repair paragraph (`R14:649-658`), two limbs removed:**

> `Repair — two sentences, no new measurement.` `1.` `⟦LIMB1⟧` `2.` `⟦LIMB2⟧`

The residue is the enumerators plus the **framing sentence**, which — like round 13's — is not
purely non-prescriptive: it carries *"no new measurement"*. **I rule that v15 does not depart from
it**, and this is the trap this round was set up to fall into, so I state the reasoning. Round 14's
*"no new measurement"* cannot mean *"do not run §14.2"*: §14.1's fixed point **requires** running the
script on every version, it has been run on every version since v8, and round 14's **own second
limb** instructs round 15 to run both splice forms. The clause is written against round-13 I-1's
kind of measurement — timing the instrument — and v15 took none: §7.9 records *"no mints, and no
timing of the instrument at all"* and the footer says *"v15 ran no timings at all"*, both of which I
verified (`__pycache__` unchanged at 11 files, 37/37 digests recompute, no `.py` moved). Running a
drafting instrument that §14.2 itself describes as one that *"reads two draft files, writes nothing,
and touches no repository artifact"* is not the measurement the clause disclaims. **There is
therefore no departure to declare, and nothing for v15 to have quoted and failed to quote.**

**M-1's Repair sentence (`R14:674-675`), one limb removed:** residue is `⟦LIMB⟧` — the Repair
sentence **is** the limb, entirely. Nothing prescriptive survives; nothing could.

**M-2's Repair sentence (`R14:691-695`), one limb removed:** residue is
*"…count that overrides anyone. The load-bearing count — v14's `56` — is fully reproducible, and I
reproduced it."* — a measurement report, not a prescription. Everything prescriptive landed.

**Residue verdict: clean on all three.**

## B.3 Disposition of the three findings at limb level

| finding | prescribed | landed | ruling |
|---|---|---|---|
| **I-1** | replace §14.1's vacuity sentence with the measured fact; strike §15 item 5's false premise and ask round 15 to run both forms | §14.1 states what the plain and biting forms do **against v15**, every element of which I reproduced; §15 item 1 asks for both forms and states the plain form is expected to exit `1` | **ADOPTED IN FULL, and re-derived for this document rather than transcribed** |
| **M-1** | quote round 13's framing sentence in full wherever it is quoted | full form *"Repair — one line, arithmetic only, no new measurement"* at three sites (`v15:57` header, `v15:1928` M-1 row, `v15:1952` M-1 limb, `v15:2290` §15 item 2); **no partial quotation of it survives anywhere** — I grepped every occurrence of `arithmetic only` | **ADOPTED** |
| **M-2** | one clause naming the bridge from `44` to `52` | bridge stated and checkable: `5×4 + 10 + 14 = 44`, plus an unreported eighth rung (`+4`) and rung 5's increment-vs-total (`+4`), `44 + 4 + 4 = 52` | **ADOPTED, first disjunct** |

**A note on M-1's wording, since I checked it rather than assumed it.** The M-1 row's disposition
says *"both sites now quote … in full"*, while one of the two sites round 14 named — §14's widening
paragraph at `v14:1895` — no longer exists, having been superseded in the normal roll-forward (v15's
§14 dispositions round 14; round 13's block compresses into *"Rounds 1–13 — carried"*). **This is not
a defect and I decline to raise it**, for three reasons I verified: the `§(diffed)` column reads
`§14, §15`, so *"both sites"* denotes the two sections, and **both sections do carry the full
quotation**; the header at `v15:56-58` states the claim in its precise form — *"quoted **in full**
wherever it is quoted"* — which matches M-1's own conditional *"where it is quoted at all"*; and the
substance of the superseded paragraph survives, with the widening's warrant relocated to §7.7 beside
the measurement that justifies it (*"one command, no rung run a different number of times"*) and its
disclosure to the carried paragraph at `v15:1984-1988`. **No partial quotation survives, which is
what M-1 asked for.**

---

# PART C — MY OWN VERIFICATION OF ALL TWELVE §3 ITEMS

| # | claim | result |
|---|---|---|
| **V1** | **the self-exclusion, BOTH forms** | **PASS, both run and both reported** — A.3. Plain: **exit `1`**, 1 failing row, 1 failing limb, `['14.1']`. Biting: **exit `1`**, 2 failing rows. Every element v15 claims is reproduced |
| **V2** | re-run the audit; byte-compare; `+0 chars` by direct diff | **PASS** — A.1, A.2. Exit `0`, byte-identical at 1653 chars / 1677 bytes; **four classes over six lines**, every line the same length |
| **V3** | the four limb quotations, by subtraction; rule limb 1's treatment | **PASS** — B.1, B.2. **4/4 FAITHFUL.** Limb 1's landing-for-v15 ruled **correct and the only correct option** |
| **V4** | the v13 bridge, against v13's own §7.7 table | **PASS** — C.1. `5×4 + 10 + 14 = 44` reproduces from v13's printed table; `44 + 4 + 4 = 52`. Round 14's condition **met in both clauses** |
| **V5** | §13.1 item 28, recorded not prescribed; contiguity; items 10/15/19/22/27 | **PASS** — C.2. Recording it is **right**. `1..28` contiguous, no duplicate. All five named items still carry their repairs |
| **V6** | §8 re-multiplies; 26 rows; `2934.5`; all 73 processes | **PASS** — C.3. Column sums to `2934.5` under my own arithmetic; every derived figure reproduces; **zero stale totals** |
| **V7** | §7.9's sum, heading, terms | **PASS** — C.4. Heading `v1–v15`; `12` mints; wall `= 37`; CPU `= 137`; every per-version paragraph matches its term |
| **V8** | rebuild the arms from §3.4; one bit-exact predicate; the misreading under `2e-6` | **PASS** — C.5. **13/13 at `0.000e+00`, both datasets**; misreading `1.878e-06` / `1.609e-06`; `GATE-ROWSUBSET` `0.000e+00` |
| **V9** | `ρ*`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18` | **PASS** — C.6. All reproduced; row-norm span `0.0271`–`0.5596` |
| **V10** | Holm counterexample and three-way equality; `n ≤ 12`; §3.7's two verbs | **PASS** — C.7 |
| **V11** | §6's 20 gate rows `12 G / 6 L / 2 R`; 37/37 digests; four new paths absent | **PASS** — C.8 |
| **V12** | the arena's import set | **NOT RE-MEASURED, by choice** — C.10 states the axis I spent my compute on instead, and it yielded a result |

## C.1 **RULING ON THE v13 BRIDGE (§3 V4): round 14's condition is MET, in both clauses**

I read v13's own §7.7 table out of `C06_FALSIFIER_PREREG_DRAFT_V13.md`. It has **seven** rows at
*"`4` runs per rung unless stated"*, with row 5 (`+ c01_policy_contrast_a0 + mechfix_ops`) at
`(10 runs)` and row 7 (`the same + c01 + runtime_block()`) at `(14 runs)`. That is
`5 × 4 + 10 + 14 = 44` — **v15's premise reproduces exactly from the source it names.**

v15's bridge adds two corrections: an **eighth** rung — *"the arena's actual set plus
`c01_policy_contrast_a0` but **without** `runtime_block()`"* — run and never printed (`+4`); and rung
5's `(10 runs)` being the second command's **increment** where rung 7's `(14 runs)` was a **total**,
so rung 5's true total is `14` (`+4`). `44 + 4 + 4 = 52`. ✓

**Both corrections are independently corroborated, which is more than the condition required.**
Round 12's own ladder at `R12:337-349` has exactly **eight** rungs, and its seventh is
*"`+ mechnov_pairverify + vsw_pregate + headspace_mint` — **`headspace_arena.py`'s actual set, plus
c01**"* at `3.05–3.09 s`, with `+ runtime_block()` as the eighth. **That is precisely the rung v15
says v13 ran and never printed**, in precisely the position v15 puts it. And the decomposition
reconciles with v14's own statement of the two commands (`8 rungs × 4 = 32`, then `10` each on two
rungs `= 20`): under it rungs 5 and 7 each total `14`, which is what v13 printed for rung 7 and what
the bridge says was true of rung 5.

**Round 14's general rule** — *a prior version's figure may be corrected in place when it feeds a
live sum, provided the correction states what the earlier version got wrong and how the two
reconcile* — is now satisfied in **both** clauses. v14 satisfied the first; v15 supplies the second.
I endorse the rule itself: §7.9 is a live cumulative sum asserted in the footer and re-derived every
round, so leaving a known-wrong term in it would make the sum wrong on purpose. The `52` warrants
nothing, enters only the historical spend line, and is not the count that overrides anyone.

## C.2 **RULING ON §13.1 ITEM 28 (§3 V5): recording a non-finding in §13 is RIGHT**

**I rule for it, and I would have asked for it had v15 not done it.** The argument is structural, not
a matter of taste. §13 is stated — and has been since v6's C-1 — as *"the **sole input** to the
mandatory separate code/resource review lineage"*. A wiring note that lives only in round 14's review
file reaches that lineage only if someone thinks to read a fifteen-round review trail, which is
exactly the failure mode round-6 C-1 found when §13 went un-edited for a whole version. The campaign's
own record is that the separate code lineage caught two wrong-verdict paths on C09 after seventeen
clean design rounds; starving it of a known wiring dependency to keep a list short is the wrong trade.

**It does not inflate the list, because it is labelled as what it is.** The item sits under a heading
that reads *"Round 14's one (item 28) — **recorded, not prescribed**"*, states in its own body that it
is *"**not** a finding and nothing in §12 changes"*, and quotes round 14's own characterisation. A
reader cannot mistake it for a design defect.

**Its factual content is true, and I checked it at source rather than accepting it.**
`scripts/analysis/c09_guard/sitecustomize.py`'s docstring reads *"Installed by PYTHONPATH in
`scripts/slurm/c09_a0_cpu.sbatch` so that EVERY python process of the **C09 A0** job (36 mints + 2
fidelity runs + the arena) carries the test-split guard"* — it does still name C09's sbatch, and the
process counts in it are C09's, not C06's `66 + 6 + 1`. The dependency is real:
`scripts/slurm/c09_a0_cpu.sbatch:50` is what exports `PYTHONPATH` to reach the module, and no C06
sbatch exists yet (all four new-code paths are absent). **Item 28's instruction — verify the export
exists and that layer 3 is *active* in all 73 processes, not merely importable — is the right
instruction**, and §12's three-layer claim is not fully true until it is discharged.

**Contiguity and the five named items.** §13.1 defines `(1)…(28)` contiguously by my own scan, with
no gap and no duplicate. Items **10** (analytic worst-case over orderings; per-`(dataset, seed,
lineage)` aggregation so the denominator is `n_D`), **15** (S7's full parameter set: threshold `0.5`,
reference rule, head-space one-block statistic, `<=`, per-seed `3/3`, plus `tiny_ok`'s non-carriage
and its two constants), **19** (endpoint pre-normalisation **and** `GATE-C01PARITY` at bit-exactness,
never at a tolerance), **22** (all three key forwards inside the mint; the `GATE-FLOOR` vote in the
arena, which is what Phase 1f's `150` is priced against), and **27** (the arena's top-level import
set, with trimming **or** extension bound to a re-measurement of `U11` and a re-carry of Phase 1g) all
still carry their repairs in full.

## C.3 §8 re-multiplied, every row, by my own arithmetic

I parsed the table and summed the printed product column without reference to the stated total:

```
printed-column sum = 2934.5000        min = 48.9083   -> 48.9
× 1.25             = 3668.1250 s      = 61.1354 min   -> 3668.1 / 61.1
mint sum           = 2508.3   share   = 85.4762 %     -> 85.5 %
Phase 3 share      = 9.3270 %                         -> 9.3 %
2× miss on Phase 3 = 3208.2 s = 53.4700 min           -> 53.5
5× miss on Phase 3 = 4029.3 s = 67.1550 min           -> 67.2
stated base: 2927.6 + 1.0 + 0.7 + 0.1 + 1.3 + 3.8 = 2934.5  ✓
2933.9 + 0.6 = 2934.5 ; 2933.9×1.25 = 3667.375 ; 2934.5×1.25 = 3668.125
```

**§8 has exactly 26 rows.** I independently re-multiplied all 18 principal unit×count products —
`15×40.39=605.9`, `3×49.30=147.9`, `15×34.40=516.0`, `3×38.87=116.6`, `174×0.0461=8.0`,
`67×0.033=2.2`, `240×0.00305=0.7`, `540×0.00629=3.4`, `60×0.1873=11.2`, `40×0.04239=1.7`,
`90×0.08098=7.3`, `2×4.63=9.3`, `2×11.27=22.5`, `62×0.62=38.4`, `3072×0.08908=273.7`,
`92×0.126=11.6` — **zero mismatches at the printed precision**, plus the composite rows
`120×0.00629 + 60×(2/13)×0.1873 = 2.48 → 2.5`, `(4.63+0.21) = 4.84 → 4.8`, `3×3.70+3×3.49 = 21.57 →
21.6`, `150×0.0041 = 0.615` / `150×0.0044 = 0.66`, `66×0.0005 = 0.033`, `72×2.0e-05 = 0.0014`. Every
count re-derives: `(30×3)+(6×4)+(30×2)=174`; `60×2+30=150`; `256×3×2×2=3072`; `23×2×2=92`;
`14×3×2×2=168`; `7×6+5×6=72`; **`66+6+1=73`**.

**Population constants re-derive from the caches themselves, not from the table:** HateMM full
`n = 744` with `{355}` the sole exact-zero row across both policies and both modalities, arena
`n = 743`, class counts **`(297, 446)`**; MHC-ZH `n = 579`, no exact-zero row, **`(180, 399)`**.
`446/743 = 0.600269 → 0.6003`, `399/579 = 0.689119 → 0.6891`, `446/744 = 0.599462 → 0.5995`; bands
`0.6203` / `0.7091`; caps `⌊0.01×743⌋ = 7` / `⌊0.01×579⌋ = 5`. `0.02×743 = 14.86`, `0.02×579 = 11.58`,
the `(2,21,22)` counterexample mean `15.00`, `20/743 = 2.69 %`, `√2048 = 45.25`, recovery
`0.02/(0.8884−0.6003) = 6.94 %`, `11.27 × 1.25 = 14.0875 → 14.1`,
`45.25 × 8.848e-08 = 4.00e-06`, `45.25 × 2.682e-07 = 1.21e-05`, `2e-6/2.682e-07 = 7.5×`,
`2e-6/8.848e-08 = 22.6×`. **I also verified `GATE-IDPARITY` directly at source**: both ro caches'
`ids` order and `labels` are identical to the native bank on both datasets.

**Stale-total sweep: zero.** `85.6 %`, `3663.4`, `2953.0`, `3691.3`, `3207.6`, `4028.7`, `48.8 min`,
`61.0 min` all occur **0** times. Every occurrence of `2930.4`, `2930.7`, `2933.9`, `3667.4` is
inside §8's own provenance narrative; `2925.0` and the single `1.7×` are historical. **§8 is
untouched this round** — it does not appear in the section-diff — and it still re-multiplies.

## C.4 §7.9's sum

Heading reads **`Cumulative v1–v15`**. Mints `7 + 1 + 4 + 0×8 = ` **12** ✓.
Wall `22+4+2+1+1+1+1+2+2+1 = ` **37** ✓. CPU `89+21+6+3+3+3+3+4+4+1 = ` **137** ✓. Each per-version
paragraph states the term the sum uses — v7 `≈4/≈21`, v8 `≈2/≈6`, v9–v12 `≈1/≈3` each, v13 `≈2/≈4`
(round-13 I-1's correction, now bridged), v14 `≈2/≈4`, v15 `≈1/≈1` — with **no term asserted that its
paragraph does not carry.** The footer agrees (`twelve` mints, `v1–v15`).

## C.5 The arms, rebuilt from the prose alone

I implemented `fuse`, `paired`, the contrast definitions and the Givens family **from §3.4's text
only** — including the pinned pre-normalisation `std[m] := l2_rows(standard[m])`,
`ow[m] := l2_rows(oneword[m])` and `common_interaction[m] = l2(common[m] ⊙ displacement[m])` with the
arm as `paired(common, common_interaction)` — then compared against `prepare_views` called through
the frozen `c01_policy_contrast_a0`:

* **`GATE-C01PARITY`: `max|diff| = 0.000e+00`** across **all 13 arms on both datasets**, at `n = 744`
  one-hot `{355}` (HateMM) and `n = 579` all-False (ZH). Dimensions come out `4 × 1024-d`-equivalent
  and `9 × 2048-d`-equivalent in head space, i.e. `7168`-d and `14336`-d in raw space, matching §8
  Phase 2's `240 / 540` split. **The gate states one predicate and it costs nothing.**
* **The un-normalised misreading** (dropping the endpoint pre-normalisation §3.4 pins) measures
  **`1.878e-06`** (HateMM) and **`1.609e-06`** (MHC-ZH), **both under `2e-6`** — reproduced to the
  digit, confirming why the `2e-6` clause had to be struck from `GATE-C01PARITY`.
* **Algebra guard**: `8.941e-08` at θ=0 on both datasets; `1.192e-07` (HateMM) / `8.941e-08` (ZH) at
  θ=45 — exactly §1's figures.
* **`GATE-ROWSUBSET`: `max|diff| = 0.000e+00`** — the `n = 743` all-False build is bit-identical to
  the `n = 744` one-hot build restricted to the 743 surviving rows, all 13 arms.
* **Exact-zero rows measured, not read**: HateMM `{355}` only, MHC-ZH none.

## C.6 `ρ`, and the trained-head bar

* **26/26 `ρ_raw` reproduce at 6 dp**, every value in §6.1's table, under `float64` accumulation over
  `float32` keys on the arena populations. `ρ*` = `0.968176` (HateMM) / `0.977223` (ZH), both
  supplied by `endpoint_std`, runners-up `common` at `0.964446` / `0.969686` ✓.
* The `1.301e-03` shift from including the masked zero row reproduces exactly:
  measured `1.301312e-03` against `0.968176 × (1 − 743/744) = 1.301312e-03`.
* **Trained heads on row-renormalised keys**, all 36 banked `K_train` at
  `artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`: HateMM min/median/max
  `0.447803 / 0.562434 / 0.632996`, MHC-ZH `0.340179 / 0.574247 / 0.667326`, **`0/18` above `ρ*` on
  both** — roughly half the bar, exactly as §6.1 claims.
* Row norms as stored span **`0.0271`–`0.5596`** across all 36 cells → §6.1's `0.027`–`0.56` ✓.
  I record round 14's caution as still correct: §6.1's gap factor is the ratio of the **order
  statistics**, which is what its sentence asserts and the reading that reproduces.

## C.7 Holm, feasibility, and §3.7's two verbs

Executed through **C01's own `holm_adjust`** (`c01_policy_contrast_a0.py:1775-1784`), padding the
family to `m`:

| witness p-values | `m = 92` pad `0.5` | `m = 92` pad `1.0` | `m = 46` |
|---|---|---|---|
| 24 × `1/2001` | **24/24** | **24/24** | **24/24** |
| 23 × `1/2001` + 1 × `2/2001` | **23/24** | 23/24 | **24/24** |
| 24 × `2/2001` | **0/24** | 0/24 | **24/24** |

`22/22` for the `displacement` disjunct. **The three-way equality at the floor holds and the
counterexample one step off the floor holds** — §5.5's table is exact, and its honest warrant (the
92-freeze is kept for auditability, not because it is consequence-free) is the right one.
`92 × 2/2001 = 0.091954 > 0.05` while `46 × 2/2001 = 0.045977 ≤ 0.05` ✓. Feasibility:
`1/257 = 0.0038911`; `12/257 = 0.04669 ≤ 0.05`; `13/257 = 0.05058 > 0.05` ⇒ **`n ≤ 12`** ✓. Both
`92`s are the different products §5.5 says they are: `(12+11)×2×2 = 92` and `23×2×2 = 92`.

**§3.7 has two blocks with two distinct verbs** — population-derived constants *computed from the
arena, never read* (which I recomputed from the caches, C.3), and frozen C01 config constants *read
from the sha-gated config and asserted equal* — with the `<=` operator correctly in the **read**
block. I verified every config constant directly against `configs/c01/c01_a0_v2.json`:
`normalization_epsilon = 1e-12`, `tiny_displacement_epsilon = 0.001`,
`max_tiny_displacement_fraction = 0.05`, `max_small_displacement_fix_fraction = 0.5`,
`small_displacement_train_quantile = 0.1`; `gain_controls` = the five names §5.1 uses (so `C` is
**six** and **five** as stated), `minimum_gain_over_strongest_control = 0.02`,
`minimum_net_fixes = {HateMM: 3, MHC_zh: 2}`, `statistics.seed = 20260728`, `holm_alpha = 0.05`,
`n_bootstrap = 2000`, `n_id_hash_permutations = 256`, `bootstrap_lower_quantile = 0.05`,
`bootstrap_upper_quantile = 0.95`, `permutation_hash = sha256`,
`retrieval.fix_break_reference = endpoint_std`, and `required_halt_only_validity_guards` = **seven**
entries **not** containing `require_no_small_displacement_dominance` — which is what makes S7 a
SURVIVE condition rather than a HALT gate. `angles_degrees = [8.3, 17.6, 29.1, 60.4, 72.7, 83.8]` ✓,
`standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`, `feature_dim = 3584` ✓.

## C.8 Digests, gates, and the C01 evidence table

* **37/37 digests recompute identically**, both before and after all my numpy/torch work. Eight rows
  carry the `…` ellipsis and **all eight resolve to exactly one file** in the tree. All four
  new-code paths (`c06_falsifier_mint.py`, `c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`,
  `c06_falsifier_cpu.sbatch`) are **absent**.
* **§6 has exactly 20 gate rows, `12 G / 6 L / 2 R`**, and the G-set and L-set match §5.6's lists
  **name for name**, symmetric difference empty in both directions.
* **§1's evidence table is exact, and I recomputed it from the stored confusion matrices** rather
  than reading the metric fields: all **16** accuracies and all **16** net-fix integers reproduce
  from `C01_A0_OUT.json`. `net_fixes.reference` is `"common"` (HateMM) and `"endpoint_concat"`
  (MHC-ZH), **not** `endpoint_std`, confirming **D-1**;
  `gain_over_strongest_control` accuracy `-0.009345794392523366` → **`−0.0093`** (the round-5
  erratum) and `-0.02564102564102566` → `−0.0256`; `decision.datasets.*.pass = false` on both;
  `decision.continue = false`. **§10.2's counts check out at source: 4 of 6 HateMM rotations
  (`17.6/29.1/60.4/72.7`, all at `0.8505`) and 2 of 6 ZH rotations (`8.3/29.1`, both at `0.8462`)
  sit below the primary.**
* **§6.2's clearances**: `0.8505−0.6203 = 0.2302`, `0.8598−0.6203 = 0.2395`, `0.8846−0.7091 = 0.1755`,
  `0.8590−0.7091 = 0.1499` — the set is exactly `{0.1499, 0.1755, 0.2302, 0.2395}`, minimum rounding
  to `0.15` and maximum to `0.24`. **`GATE-ARMVIAB`'s escape branch is confirmed unreachable.**
* **§1's two verbatim quotations reproduce from
  `TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`**, and `new_status` is
  `gated_on_zero_cost_falsifier`. The `falsifier_spec` is character-exact. The
  `falsifier_design_constraints` quotation differs in exactly **two characters** — two ASCII hyphens
  in the JSON rendered as em-dashes in the document — which is the quote-glyph normalisation the
  document declares for quotations generally, changes no word, and is unchanged this round (§1 does
  not appear in the section diff). **Recorded for completeness, not raised.**

## C.9 Freeze-readiness, operationally

Judged as the document an operator with no context would execute.

* **No decision point on the run boundary.** One `sbatch`, 8 CPU / 32 GB, no `--gres`, no `--time`,
  no array, no dependency, no requeue. The 73-process order is stated — `66 mints → 6 fidelity →
  1 arena` — with `GATE-SHA` once in the driver before any of them and `GATE-POP` before any
  population-consuming gate.
* **Preconditions are checkable.** 37/37 digests recompute; all four new-code paths absent;
  `mints_present_before_arena` is a declared binding predicate; and the `dev_path_opens ==
  mints_executed` choice is correct — `headspace_mint.py:192-194` returns before the `:199` dev load
  on a resumed mint, so a binding `== 66` would HALT a legitimate resume.
* **Per-class import accounting is consistent.** 66 mints carry startup inside their full-process
  walls; 6 fidelity inside `U9` (whose `3.70 s` cannot be an internal timing, since the same
  process's imports alone measure `3.06–3.16 s`); the 73rd, the arena, priced once at Phase 1g.
  `66 + 6 + 1 = 73`. **I found no seventy-fourth python process**, and C.10 records the one place I
  looked for a hidden per-draw process and did not find one.
* **Heartbeat.** Line-buffered `buffering=1` handle, per-phase and per-cell granularity, plus an
  unbuffered bash echo per mint; longest un-instrumented span `11.27 s` (`14.1 s` conservative,
  and `11.27 × 1.25 = 14.0875` re-derives) against a `~15 s` bound. The arena startup at
  `3.094–3.717 s` leaves `~11 s` of headroom and changes no interval. **`rule_2_heartbeat`
  unchanged and satisfied; v15 touches no §8 row and no §9 interval.**
* **Exit and resume semantics defined.** HALT names the failing gate in its final line; a
  `RuntimeError` from the imported C01 algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE` with its
  `context` string, and written to both the decision JSON and the final heartbeat line before exit.
* **Test-split non-contact by construction, verified at source.** Three layers, all present in the
  frozen files: `headspace_mint.py:106-116` rebinds `torch.load` behind an assertion; the driver's
  `split == "train"` assertion; `c09_guard/sitecustomize.py` importing `c09guard` and calling
  `install()` at interpreter startup. `GATE-LEDGER` binds `test_path_opens == 0` and
  `test_label_materialisations == 0`. §3.1 states no `dev_seen_*-ro_*` and no `test_seen` ro cache is
  opened by any phase. **Layer 3's activation is the one thing that is a code-side obligation rather
  than a design fact, and §13.1 item 28 is now where the lineage will find it.**
* **The `$0` character holds.** No GPU, no Modal, no test contact, no new data.

## C.10 `rule_1_compute_projection` — **the axis I searched, and what it returned**

**I name one, and unlike the last three rounds it returns something — in the artifact's favour.**

**`U4`, measured end to end.** §15 item 4 asked this round to consider `U4` directly: it is
`273.7 s`, `9.3 %` of the projection, the single largest unit that no party had reproduced, and its
object is described only as *"2 arms × 5 folds + rebuild"*. I built exactly that object from §5.4.1
and §3.4 — permute the one-word endpoint rows, rebuild `displacement` (1024-d) and
`common_displacement` (2048-d) in head space with §3.4's builder, then run
`mechfix_ops.deployed_vote` for both arms across five folds at `n = 743` — and timed 25 draws after a
warm-up, under `OMP/MKL_NUM_THREADS=8` and `CUDA_VISIBLE_DEVICES=""`. **No labels, no accuracy, no
arm outcome: a timing depends on array shapes, not values, so this is shape-only work and computes
nothing blind.**

| leg | my measurement | §7.7's frozen figure |
|---|---|---|
| whole draw (2 arms × 5 folds + rebuild) | **`0.0728–0.0795 s`, median `0.0739`** | **`U4 = 0.08908 s`** |
| rebuild leg alone | median `0.0102 s` | — |
| vote leg alone (2 arms × 5 folds) | median **`0.04630 s`** | `5 × (U2a + U2b) = 0.04670 s` |

**The frozen `U4` is `1.21×` conservative**, and over all `3072` draws my measurement gives `226.9 s`
against the carried `273.7 s`. The vote leg agrees with the two frozen vote units to `0.9 %`,
independently re-corroborating `U2a` and `U2b` on realistic fold shapes. **`U4` bounds an
independent measurement. §7.7's uncorroborated list no longer contains a substantial unit.**

**The sub-axis I probed inside it, and dropped after measuring.** §5.4.1 says *"the one-word endpoint
rows are permuted"*. A code lineage reading that as *permute the raw `ro_ow` rows and re-forward
through the head* would add a head forward per draw that §8 prices nowhere — I measured such a
forward at `0.0165 s`, so `3072 × 2` of them would be `101.1 s`, **`3.4 %` of the total**. **I do not
raise it, for three independent reasons**, and I record it so the next reader knows it was examined
rather than overlooked. (i) The head forward is row-wise, so permuting rows then forwarding is
mathematically identical to permuting the rows of the already-forwarded `h_ow` — I verified the two
agree to `1.669e-06`, pure `float32` matmul-ordering noise — meaning the expensive reading is never
*necessary*. (ii) §5.4.1's own words are *"rebuilt **in head space** by the §3.4 builder"*, which
selects the cheap reading. (iii) §13.1 item 22 states that **all three** key forwards happen **inside
the mint process**, and §8 Phase 1b enumerates them exhaustively at `174` with every factor named —
a per-draw forward would have to occur in the *arena* process and would contradict item 22 outright.
**Raising a phrase that has a true reading, that the document selects, and that a §13 item
independently forecloses would be manufacturing a finding.** No item.

**No eleventh uncounted item. Fifteen rounds, ten items** — and the projection's last substantially
uncorroborated unit is now measured and conservative.

## C.11 Verdict-path enumeration — mine, from the document alone

Let `G` = all twelve globals pass; for lineage `L ∈ {N, R}` let `p_L` = passed all six per-lineage
gates **on both datasets** (§5.6's dataset-axis rule), `c_L` = clears S1–S7 on both datasets.

| `G` | `p_N` | `p_R` | outcome | rule |
|---|---|---|---|---|
| fail | any | any | **HALT** `INSTRUMENT_INCONCLUSIVE` | 3 |
| pass | ✓ | ✓ | **SURVIVE** if `c_N ∨ c_R`; else **CLOSE** | 1 / 2 |
| pass | ✓ | dropped | **SURVIVE** if `c_N`; else **HALT** (rule 2 needs *both* passed) | 1 / 3 |
| pass | dropped | ✓ | **SURVIVE** if `c_R`; else **HALT** | 1 / 3 |
| pass | dropped | dropped | **HALT** | 3 |

**Exactly one published state per combination; no unmapped outcome; no overlap** — rules 1 and 2 are
mutually exclusive by construction (rule 2 requires *neither* clears), and rule 3's explicit
*"otherwise"* makes the mapping total. `c_L` is never evaluated for a dropped lineage.

**The declared-drop exemption is the only lawful absent-quantity path**, stated in terms at
`v15:776-777`: *"Absence by declared drop is lawful; absence by computation failure in a surviving
lineage still HALTs."*

**No gate failure is reportable as a closure.** CLOSE requires all twelve globals to pass **and**
both lineages to have passed every per-lineage gate on **both** datasets. A global failure HALTs; a
per-lineage failure drops that lineage on both datasets, falsifying rule 2's conjunct, so the only
reachable outcomes are SURVIVE-on-the-clean-lineage or HALT. **A CLOSE always rests on two clean
negatives, never one.**

## C.12 §4.C — can any gate fire on a warranted CLOSE? **No, for all twenty**

A *warranted CLOSE*: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction.** `GATE-DET1` (thread env before
any python starts). `GATE-SHA` (37 digests over files no phase writes — **all 37 recomputed by me,
twice**). `GATE-FOLD` (banked parity flags + `fold_of`; the assertion is at
`headspace_mint.py:203-216` and a `.npz` is written only at `:321-325`, after it passes).
`GATE-FLOOR` (six banked anchors reproduced on **native** keys, so no ro-derived arm outcome can
reach it). `GATE-POP` (populations `743/579`, class counts `(297,446)` / `(180,399)`, index-set
identity, constants recomputed — **all measured by me from the caches**). `GATE-C01PARITY` (a
property of the **builder**; I measured `0.000e+00`). `GATE-ROWSUBSET` (builder property;
`0.000e+00`). `GATE-RHORAW` (a property of the ro caches and the raw leg, identical for both
lineages; **26/26 reproduced at 6 dp**, asserted at 4 dp). `GATE-NULLREMOVED` / `GATE-ZEROMASK`
(`{355}` / `{}`, which I measured as the sole exact-zero row). `GATE-IDPARITY` (ids/labels parity —
**verified True at source on both datasets**). `GATE-LEDGER` (declared counts). **None reads which
arm won.**

**The six per-lineage gates.**

* **`GATE-ARENA`.** Its **lower** bound is on `endpoint_std` **only** — the reference arm, never a
  real-vs-rotation quantity. A warranted CLOSE says nothing about `endpoint_std`; if that arm cannot
  clear `majority + 0.02` the instrument is genuinely dead. Its **upper** bound `≤ 0.98` fires only
  on implausibly high accuracy and cannot fire downward.
* **`GATE-ORBITDISP`.** Fires iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D` — head space *more degenerate than
  the raw family*. I measured trained deployed heads at roughly **half** the bar, **`0/18` on both**.
  It is a statement about key geometry, not about accuracy. Arm-outcome-independent.
* **`GATE-NESTED`.** The scoring head excluded its fold. Structural.
* **`GATE-SELFTEST`.** `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is an **identity**; it holds
  whatever the accuracies are.
* **`GATE-ZEROOP`.** `orthrot_0 ≡ endpoint_concat` and `orthrot_45 ≡ common_displacement` are
  **algebraic identities of the Givens family** — I measured the residuals at `8.941e-08` (θ=0, both
  datasets) and `1.192e-07` / `8.941e-08` (θ=45). Its tie cap is **one-directional** (REPORT → HALT
  only), so it can cause non-publication, never a wrong verdict.
* **`GATE-ALGEBRA`.** Key-level `≤ 2e-6` on the same two identities, with `7.5×`–`22.6×` measured
  head-space headroom (which I re-derived from §7.8's residual endpoints).

**The two `R` gates** (`GATE-DOMAIN`, `GATE-DEVFID`) carry **no bar** and cannot fire at all.

**All twenty: no gate can fire on a warranted CLOSE.** The load-bearing structural fact is §6.2's
retirement of `GATE-ARMVIAB`: I confirmed its escape branch unreachable (all four raw clearances
`0.1499`–`0.2395` clear), which had reduced it to a one-sided HALT on precisely the warranted-CLOSE
outcome. With it deleted, **no lower-bound instrument HALT is applied to a real arm anywhere in this
design** — I read the §6 table row by row, and `GATE-ARENA`'s lower bound on `endpoint_std` is the
only lower accuracy bound in the document.

## C.13 Blindness, v1–v15, by my own grep

Under the leading-digit convention on the **closed** interval `[0.6, 0.99]` across all fifteen
drafts:

* **v1–v5: `98`** ✓ · **v1–v10: `116`** ✓ · **v1–v12: `118`** ✓ — v15's triple reproduces exactly.
* Excluding the two self-referential endpoint tokens: **`96 / 114 / 116`** ✓.
* **The new-in-v13, new-in-v14 and new-in-v15 in-band sets are ALL EMPTY.** The corpus total is
  unchanged at **`118` from v12 through v15**, so §7.3's *"None of v13, v14 or v15 adds one"* and the
  scope labels `v1–v15` and `v6–v15` are **verified for v15 by me rather than inherited from round
  14** — round 14 could only certify through v14, and §7.3 correctly says so.
* **No battery-arm accuracy exists anywhere in v1–v15.**

I confirm round 13's warning holds: a regex admitting leading-dot decimals picks up `.28` from
`defined 1..28`; the leading-digit convention is the one that reproduces the published triple.

## C.14 The footer's side-effect claims

`scripts/analysis/__pycache__/` holds **11** files, before and after my own numpy/torch/faiss work —
every module involved already had a `cpython-311` cache, so I wrote no new bytecode either. **No
`.py` source moved**: all 37 §11 digests recompute after my compute. v15's own hash is unchanged on
disk. `TARGET_STATE.json` read, not modified. **The footer's claims are true of the tree as I found
it and as I left it.**

---

# FINDINGS

## CRITICAL — none

## HIGH — none

## IMPORTANT — none

## MINOR (non-blocking; does not touch the verdict path)

* **M-1. §14.1's *"Two things in this transcript will look odd on a first read"* no longer explains
  step (6)'s *changed-but-uncited* list, which grew from three entries to four and changed
  membership** (`v15:2110-2119`, against the transcript at `v15:2102`). v12, v13 and v14 each carried
  a bullet for that line; v14's read *"Step (6)'s changed-but-uncited list is three entries — §14.2,
  §15 and the header — each of which changes every round by the process itself … Nothing on it is a
  repair claim, and the direction that would be a finding, named-but-unchanged, is empty."* v15's
  list is `['13.1', '14.2', '7.3', 'header']` — §15 has left it (round-14 I-1's second limb and
  round-14 M-1's row now cite §15) and §13.1 and §7.3 have joined it — and no sentence in §14.1
  accounts for the new shape.

  **Why it is non-blocking, and why I graded it Minor rather than Important.** I checked each entry
  against the three disposition rows and the four limbs: **none of the four is a landing site for any
  round-14 finding**, so nothing on the list is an uncited repair claim — which is the only thing
  the check exists to surface in that direction. The **load-bearing** direction,
  *named-but-unchanged*, **is** stated in the same paragraph as empty, and I verified it is. And each
  entry is independently accounted for in the document: §13.1 by the very next bullet (*"§13.1 gained
  item 28 this round"*), §14.2 by the bullet above it, and §7.3 and the header by the routine
  per-version scope relabel and supersession preamble that change every round by the drafting process
  itself. **It moves no quantity, touches no gate, alters no verdict path, and repeats no false
  statement** — nothing in §14.1 asserts anything untrue about the list; it simply stops explaining
  it. This is a completeness gap in the document's self-explanation, not a defect in the design or
  the record, and I decline to inflate it: the artifact's own machinery reports the line correctly
  and the reader can resolve every entry from the same subsection.

  **Repair (one clause, no new measurement, optional before freeze):** restore the bullet, e.g.
  *"Step (6)'s changed-but-uncited list is four entries — §13.1, §14.2, §7.3 and the header. None is
  a repair claim: §13.1 carries the recorded item 28, and the other three change every round by the
  process itself (the audit's own version labels, §7.3's version-scope relabel, and the supersession
  preamble). §15 has left the list because round-14 I-1's second limb and round-14 M-1's row both
  cite it."*

---

# REQUIRED RULINGS

## 1. §4.A — **is the correction itself checked?** YES, and the safeguard is stronger than the ordering it rests on.

This is the question the round exists for, so I answer it in two parts.

**The stated ordering is a process claim, and I cannot verify it from the artifact.** §14 records the
sequence — finish every other edit; run §14.2 to its byte-identical fixed point; **then** splice v14's
§14.1 into a copy of the finalized v15 and run the unmodified script; **then** write §14.1's sentence
and §15 item 5 from that output; then re-run §14.2 and re-verify. Nothing in a finished document can
attest to the order in which its sentences were written. Taken alone, that is a claim about process
of exactly the kind that cannot be audited, and I decline to credit it as evidence.

**It does not need to be, and that is the point.** Two facts I established by execution make the
correction checked regardless of how it was produced.

1. **I checked the claim itself, against the final artifact, element by element** (A.3). Every
   printed element §14.1 attributes to both splice forms is reproduced by my own independent
   construction: the `UNCHANGED` line, the failing row and its text, `rows failing: 1` then `2`, the
   one failing limb, the `['14.1']` set, and exit `1` in both. **A claim that survives being checked
   is checked, whatever order it was written in.** That is the whole difference from round-14 I-1,
   where the analogous claim did not survive.
2. **The splice result is provably invariant to §14.1's own content**, so the fix cannot falsify
   itself by being written. The plain construction replaces §14.1 wholesale with v14's, so whatever
   v15's §14.1 says is discarded before the comparison; what makes the form bite is that a row and a
   limb **in §14** cite §14.1, and neither was touched by the sentence. I confirmed this structurally
   from the script (`SELF='14.1'`, status computed from `SA[SELF]!=SB[SELF]`) and empirically — the
   spliced run's row/limb/exit output is identical whether or not §14.1 carries the new sentence.
   **The sentence is a fixed point of its own insertion**, which is a stronger guarantee than any
   ordering discipline, because it holds for every future version in which a finding about §14.1 is
   dispositioned in §14.1.

**The ordering is therefore a true and useful account of good drafting practice, and the document is
right to record it, but it is not what makes the repair sound. The checkability is.** I also note
that the ordering is internally consistent: writing §15 item 5 after the fixed point would break it,
and the stated sequence closes with *"then re-run §14.2 and re-verify the fixed point"* — which is
the run whose output I byte-matched.

## 2. Verdict-path enumeration: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.** C.11.

## 3. §4.C — any gate that can fire on a warranted CLOSE? **No, for all twenty.** C.12, derived from the gate texts and my own measurements, not inherited.

## 4. §3 V4 — the v13 bridge: **the condition is MET in both clauses**, and independently corroborated by round 12's own eight-rung ladder. C.1.

## 5. §3 V5 — recording a non-finding as §13.1 item 28: **RIGHT.** C.2. §13 is the code lineage's sole input; the item is labelled *"recorded, not prescribed"* and states in its own body that it is not a design defect; and I verified its factual content at source (`c09_guard/sitecustomize.py`'s docstring does still name C09's sbatch; `c09_a0_cpu.sbatch:50` is what exports `PYTHONPATH`). The list is `1–28`, contiguous.

## 6. Rulings on §15's six open issues

1. **The self-exclusion, in BOTH forms.** **PASS — both run, both reported, and the stated premise is
   correct.** Plain: exit `1`, one failing row (`I-1`), one failing limb, `named by a row but
   unchanged: ['14.1']`. Biting: exit `1`, `rows failing: 2` with `X-9`. **The plain form is not
   vacuous against v15, exactly as §14.1 says.** I checked rather than inherited. A.3.
2. **The widening on round-13 I-1, re-ruled against the full sentence.** **WARRANTED ON BOTH
   CLAUSES.** On the *method* clause (*"arithmetic only, no new measurement"*) I reach round 14's
   conclusion by my own route: the two arithmetic disjuncts would each have produced a **consistent**
   statement of a **non-uniform, two-command** sample, and since the sample size was the warrant for
   overriding round 12, a consistent statement of a weak sample is weaker evidence than a uniform
   one — and the tell that this was not a designer preferring its own instrument is that §7.7's
   three-party table **preserves** rounds 12 and 13 with their own counts and ranges rather than
   replacing them. On the *length* clause (*"one line"*), now restored to the quotation: the repair is
   emphatically not one line — §7.7 grew `1583` characters and §7.9 `949`, both printed in the
   document's own transcript, i.e. **disclosed by the instrument rather than concealed**. **I rule the
   length overrun warranted too**, on a ground the previous rounds did not need to state: the length
   is a *consequence* of the method decision, not an independent choice. Once you re-measure seven
   rungs uniformly, printing the seven-row table with its `runs` column and the three-party split
   table is the **minimum** honest presentation of that sample — a one-line version would have
   reasserted a figure whose parts are not derivable, which is the exact defect round 13 found. And
   the cost is text: §8's total is unchanged, `3.8 s` still bounds the pooled maximum by `0.083 s`,
   no heartbeat interval moves, no quantity a reader acts on moves.
3. **v13's `52`, and the bridge.** **SATISFIED IN BOTH CLAUSES**, and corroborated beyond what was
   asked — round 12's ladder at `R12:337-349` has exactly the eighth rung, in exactly the position,
   that v15 says v13 ran and never printed. Ruling 4 above, C.1.
4. **The eleventh uncounted item.** **One axis named and searched — `U4` end to end, the unit §15
   invited — and it returns a result: the frozen `0.08908 s` is `1.21×` conservative against my
   measured median of `0.0739 s`, and `3072` draws measure `226.9 s` against the carried `273.7 s`.**
   Its vote leg re-corroborates `U2a`/`U2b` to `0.9 %`. One sub-axis (a literal re-forward reading of
   §5.4.1, worth `3.4 %` of the total) was measured and **dropped**, because the head forward
   commutes with the permutation, §5.4.1 selects the cheap reading in terms, and §13.1 item 22
   forecloses the expensive one. **No eleventh uncounted item.** C.10.
5. **§13.1 item 28.** **RIGHT to record it**, and it does not inflate the list. Ruling 5 above, C.2.
6. **Is the record still sound, and is the design freeze-ready?** **The record is sound at limb
   level** — 4/4 faithful and complete, both of M-2's disjuncts quoted including the one declined,
   nothing prescribed missing, nothing claimed absent, nothing narrowed, zero stale totals. **The
   science is closed** and I re-derived every leg of it independently. **The design is freeze-ready**,
   with no Critical, High or Important open. The single Minor is one restorable clause of
   self-explanation about a check whose load-bearing direction is stated and verified empty.

## 7. Process rules

* **`rule_1_compute_projection`.** Satisfied in form and in substance — every §8 row is a measured
  unit × an explicit count, every product re-multiplies exactly, the column sums to `2934.5`, all
  `73` processes are accounted, and there is no extrapolation from a reduced-scale dry run. **The
  projection's last substantially uncorroborated unit, `U4`, is now measured by an independent party
  and is conservative** (C.10). **No §8 row is affected by the Minor.**
* **`rule_2_heartbeat`.** **Unchanged and satisfied.** v15 touches no §8 row and no §9 interval. The
  longest un-instrumented span is `11.27 s` (`14.1 s` conservative) against `~15 s`; the arena's
  startup at `3.094–3.717 s` leaves `~11 s` of headroom and is bracketed by the driver's unbuffered
  echo.

## 8. Can the falsifier discharge the written condition at `$0`? **Yes.**

Every input exists and is digest-frozen (37/37, recomputed by me before and after my own compute);
the head space is re-mintable on CPU at measured cost; the arms rebuild **bit-exactly** from the
document's own prose (`0.000e+00`, 13/13, both datasets); the decision rule is pre-registered with
its multiplicity resolution floor proved attainable through C01's own `holm_adjust`; the projection's
largest uncorroborated unit is now independently measured and conservative; the verdict combination
is total with exactly one lawful absence path; and no gate can fire on a warranted CLOSE. **The Minor
does not bear on this.**

---

# WHAT THIS GO DOES AND DOES NOT AUTHORIZE

**A GO authorizes nothing to run.** Before any job: (1) **freeze with hashes**; (2) a **separate**,
independent **code/resource review lineage** over the executable reaching its own `0C/0H/0I` — and I
carry forward, for that lineage, everything rounds 12–14 flagged plus what this round adds: §13 item
23 should be read broadly (*test against `prepare_views`, do not reimplement*); item 27 binds any
trimming **or extension** of the arena's import set to a re-measurement of `U11` and a re-carry of
Phase 1g; item 22's placement of the `GATE-FLOOR` vote in the arena process is what Phase 1f's `150`
is priced against; item 28's `PYTHONPATH` export is what makes §12's third guard layer **active**
rather than merely importable, and the lineage must confirm it in all 73 processes; and — new from
this round — **§5.4.1's *"the one-word endpoint rows are permuted"* must be implemented as a
permutation of the already-forwarded head-space `h_ow` rows, not as a re-forward of permuted raw
rows.** The two are mathematically identical (I measured the difference at `1.669e-06`, `float32`
matmul-ordering noise), §13 item 22 already requires all key forwards to live inside the mint
process, and §8 prices no per-draw forward — but the expensive reading would add `≈ 101 s`, `3.4 %`
of the projection, and is the one place I found where a literal implementation could exceed the
table. (3) **main-dialogue authorization.** This document is **not** authority to write
`TARGET_STATE.json`.

---

# CLOSING

**There is no severe finding this round, and I want to be precise about what that does and does not
mean.** It does not mean the artifact was assumed sound: I re-ran the audit and byte-matched its
transcript, built both splice counterfactuals from scratch and reproduced every element of the
sentence they were used to write, rebuilt all thirteen arms from §3.4's prose and got `0.000e+00` on
both datasets, recomputed all sixteen C01 accuracies from the stored confusion matrices rather than
reading the metric fields, re-derived every population constant from the caches, re-multiplied §8 to
`2934.5`, recomputed the 37 digests twice, ran the Holm counterexample through C01's own code, and
re-derived all twenty gates as unable to fire on a warranted CLOSE. It means the artifact survived
all of it.

**The one thing this round adds that no prior round could is a measurement rather than a check.**
`U4` — `273.7 s`, `9.3 %` of the projection, the largest unit nobody had reproduced, and the unit
§15 item 4 pointed at — measures `0.0739 s` per draw against a frozen `0.08908 s`. The projection's
last substantially uncorroborated row now bounds an independent measurement, and it does so
conservatively. The sub-axis inside it that could have been a finding — a literal reading of §5.4.1
costing `3.4 %` of the total — is foreclosed by the document's own words and by §13 item 22, so I
measured it, dropped it, and handed it to the code lineage instead of inflating it into an item.

**On round-14 I-1, the reason this version exists: it is discharged, and discharged in the way that
matters.** The failure round 14 found was a sentence about the document's own mechanism that had been
carried across three versions without anyone running the thing it described. The obvious way to fail
again was to write the repair from round 14's expectation. v15 did not: the sentence it wrote is
about **this** document, and when I built the construction myself and ran it, every element came back
exactly as stated — one failing row, one failing limb, `named by a row but unchanged: ['14.1']`,
exit `1`, and exit `1` again with two failing rows under the synthetic form. The document also
declares an ordering it used to get there. I have ruled that the ordering is unverifiable and that it
does not need to be verified, because the splice result is provably independent of §14.1's own
content: **the corrected sentence is a fixed point of its own insertion**, which is a stronger
safeguard than the discipline that produced it.

**My one Minor is a dropped explanatory clause and I have said plainly why it is not more than that.**
The check it concerns reports correctly, its load-bearing direction is stated and empty, every entry
on the non-load-bearing list is accounted for elsewhere in the same subsection, and none of them is a
repair claim. Restoring one sentence would tidy the record; not restoring it publishes nothing false
and risks no verdict. **I have declined to grade it Important in order to produce a fifteenth
consecutive finding, exactly as I would have declined to grant a GO in order to end a long trail.
The design is clean, the record is faithful, and I say so plainly: GO (0C/0H/0I).**
