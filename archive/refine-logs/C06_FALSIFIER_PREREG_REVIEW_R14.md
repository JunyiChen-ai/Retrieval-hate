# C06 `$0` falsifier — independent design review, **ROUND 14**

**Reviewer:** fresh, independent of rounds 1–13 and of the designer.
**Artifact:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V14.md`, sha256
`d80bbb44911daef9e772dfe1246ffa71876147e82d7f8b4bce6d83d5c34b0a46`, 173368 bytes, 2321 lines —
**recomputed on disk, matches the review request.**
**Compute:** read-only. No GPU, no SLURM, no Modal, no arena run, no mint, no cache write, no
test-split access, no commit, no job. `TARGET_STATE.json` read, not modified. **No repository file
was written or edited except this review.** Every temporary artifact — the extracted audit script,
the spliced counterfactual drafts, the timing ladder — went to the session scratchpad. My timings
ran under the `HateVideo` interpreter (Python 3.11.8), for which every module involved already had a
`cpython-311` bytecode cache, so **my 56 timed starts wrote no new `.pyc`**:
`scripts/analysis/__pycache__/` holds the same **11** files before and after, and all 37 §11 digests
recompute.

---

# VERDICT

## **REVISE — 0C / 0H / 1I + 2M**

Zero Critical. Zero High. **One Important.** Two Minor, neither touching the verdict path.

**The run-count accounting is fixed, and I verified it by re-running the measurement, not by reading
it.** I built the seven rungs from `headspace_arena.py:28-46` and §13.1 item 27, ran them **8 times
each in one command — 56 timed starts, 95.7 seconds of wall**, and every count in v14 reconciles
under my own arithmetic: `7 × 8 = 56` here, and an arena-class pool of `8 + 3 + 24 = 35`. Both
pooled endpoints are arena-class observations attributable to a named party's stated rung —
round 12's `3.12–3.27 s` over three runs at `R12:349` and round 13's `3.094–3.717 s` over 24 at
`R13:237` are **exact**. My own rung 7 is `3.070–3.540 s`; `3.8 s` bounds it with `0.26 s` to spare.
Round 13's I-1 is discharged in full, and the widening that discharged it is **warranted** (§C.1).

**The science layer is closed and I re-derived it rather than inheriting it.** 13/13 arms rebuilt
from §3.4's prose alone at `max|diff| = 0.000e+00` on both datasets; `GATE-ROWSUBSET` at
`0.000e+00`; 26/26 `ρ_raw` reproduced at 6 dp; `ρ*` `0.968176` / `0.977223`; trained-head **0/18** on
row-renormalised keys on both; 16/16 C01 accuracies and 16/16 net-fix integers; **37/37** digests;
the Holm counterexample and its three-way equality through C01's own `holm_adjust`; §8 re-multiplied
to `2934.5` row by row; all twenty gates re-derived as unable to fire on a warranted CLOSE.

**The record is faithful at limb level.** 6/6 limbs verbatim, complete, and inside the `R13:NNN-NNN`
range each cites; both Minors' residues non-prescriptive; I-1's residue is the framing sentence,
whose method clause v14 quotes and rules on rather than hides. **Zero stale totals.**

**The one Important is not a number and not a count — it is a false factual claim about the
document's own verification mechanism, and the document's own transcript refutes it two paragraphs
away.** §14.1 states *"The same vacuity holds for v14, whose limbs land elsewhere"* and §15 item 5
instructs this round that *"no row or limb cites §14.1"*. **Both are false.** Round-13 M-2 lands in
§14.1; v14's M-2 disposition row (`v14:1893`) and its M-2 limb (`v14:1922`) both cite §14.1, and the
embedded transcript prints `OK M-2 cites §14.1` and `OK M-2 … -> §14.1`. I ran the plain splice: it
does **not** exit `0` — it exits `1`, failing the M-2 row, failing the M-2 limb, and reporting
`named by a row but unchanged: ['14.1']`. **The mechanism is stronger than v14 claims for it**, and
the claim is wrong in the direction that understates the artifact.

I considered grading it Minor and rejected that: §14.1 exists so the disposition record is checkable
by machine rather than asserted, and a statement about what that machine does on this document, in
that section, contradicted by that section's own printed output, is exactly the class rounds 7–13
have graded Important — a stated evidentiary basis that does not survive being checked. I considered
High and rejected it: nothing is narrowed, round 13's prescriptions all land, no verdict quantity
moves, no gate is touched, and the error runs in the conservative direction. **The irony is worth
stating once: the false claim is an unchecked inheritance from v12/v13, which is the precise defect
round-13 M-2 named — and it sits in the paragraph that lands round-13 M-2's repair.**

---

# PART A — AUDITING THE AUDITOR

## A.1 The §14.2 script, re-run against final on-disk v14: **byte-identical, exit 0**

I extracted the script from the v14 fence, ran it unmodified, and captured stdout.

* **Exit code `0`.**
* Embedded transcript **1969 bytes**; my run **1969 bytes**; **`BYTE-IDENTICAL: True`**
  (sha256 of both: `7446c1d2e28e4f3fbf343078e8b7db1069c00a648d57a74bdef8245050d810cc`).

The transcript is a **verified fixed point**, and every printed line reproduces including the
`28 reference sites`, `defined 1..27` and `unresolved: NONE`.

## A.2 `CHANGED §14.2 +0 chars` — verified by direct diff, not accepted

`len(v13 §14.2) = len(v14 §14.2) = 7247`, `identical = False`. The differing lines are **six**, all
same-length, in **four substitution classes**:

```
L8   """Mechanical disposition verification, C06 falsifier v13.  -> v14.        [version string]
L9   (1) section diff v12->v13 ...                               -> v13->v14    [diff label]
L14  V_OLD='...DRAFT_V12.md'                                     -> ...V13.md   [V_OLD/V_NEW]
L15  V_NEW='...DRAFT_V13.md'                                     -> ...V14.md   [V_OLD/V_NEW]
L27  print('=== (1) SECTION DIFF v12 -> v13 ===')                -> v13 -> v14  [print header]
L109 print('=== (5) ... (round-12 prescriptions) ===')           -> round-13    [print header]
```

**v14's account of its own §14.2 change is exact**: four classes, six lines, every line the same
length. **Round-13 M-2 is landed and verified.** No finding.

## A.3 Breaking the self-exclusion — both forms, and the result is this round's finding

**Plain splice (v13's §14.1 into v14): exit `1`, NOT vacuous.**

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  M-2   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 2 ; rows failing: 1
  FAIL  M-2   *"Repair: print `six`, or say "four substitution cla -> §14.1 NOT DIFFED
  limbs landed: 5 ; limbs open/failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**Biting form (splice + one synthetic §14.1-citing row `X-9`): exit `1`, two failing rows.**

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  X-9   cites §14.1 -- NOT DIFFED
  FAIL  M-2   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 2 ; rows failing: 2
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**The mechanism is live, and it is live natively — no synthetic row is required against v14.** The
self-exclusion covers size only and still fails a §14.1-citing row when §14.1 did not change. This
is **I-1**.

## A.4 Section deltas, recomputed with my own splitter

I wrote a line-based splitter (the audit's is a regex-slice splitter) and reproduced **every printed
delta exactly**: `§8 +178`, `§9 +94`, `§14 −881`, `§15 −29`, `§6.1 +113`, `§7.3 +87`, `§7.7 +1583`,
`§7.9 +949`, `§14.2 +0`, `header −129`, `UNCHANGED: 47`. For completeness, the one delta the audit
self-excludes: **`§14.1 +125 chars`**.

My own scan confirms §13.1 defines `(1)…(27)` **contiguously**, no gap, no repeat.

---

# PART B — DISPOSITION AUDIT OF ROUND 13'S THREE FINDINGS, AT LIMB LEVEL

## B.1 The six limbs: **6 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded**

Each quotation was normalised for emphasis and quote-glyph only, then tested for containment inside
the exact `R13:NNN-NNN` line range it cites, read out of round 13's file.

| # | finding | limb (opening) | cited range | verbatim | inside range | ruling |
|---|---|---|---|---|---|---|
| 1 | I-1 | *"Reconcile the counts against §7.7's table …"* | `R13:569-570` | ✓ | ✓ | **FAITHFUL** |
| 2 | I-1 | *"Concretely: in §7.9 and the footer, either print `44` …"* | `R13:570-573` | ✓ | ✓ | **FAITHFUL** |
| 3 | I-1 | *"In §7.7's `U11` row and §9, replace "24 timed runs …"* | `R13:573-576` | ✓ | ✓ | **FAITHFUL** |
| 4 | I-1 | *"§7.3's "`24` … timings" follows whichever is chosen."* | `R13:576` | ✓ | ✓ | **FAITHFUL** |
| 5 | M-1 | *"Repair: print `0.027`–`0.56` and `3.5×`–`7.6×` …"* | `R13:597-598` | ✓ | ✓ | **FAITHFUL** |
| 6 | M-2 | *"Repair: print `six`, or say "four substitution classes" …"* | `R13:608` | ✓ | ✓ | **FAITHFUL** |

**No word is dropped from any of the six, including every qualifying clause.** Limb 2 is the place a
narrowing would have been most convenient — it is the limb v14 declines to execute — and it carries
both disjuncts in full, with the departure labelled in the limb cell itself.

## B.2 The residues, by subtraction

**I-1's Repair paragraph (`R13:569-576`), four limbs removed:**

> `**Repair — one line, arithmetic only, no new measurement.**` `⟦LIMB1⟧` `⟦LIMB2⟧` `⟦LIMB3⟧` `⟦LIMB4⟧`

The residue is the **framing sentence**, and unlike round 12's *"Repair — two lines, and the second
is the durable one"* it is **not purely non-prescriptive**: it carries the method constraint
(*"one line, arithmetic only, no new measurement"*) that v14's repair does not satisfy. **v14 does
not hide it.** §14's paragraph is titled *"Why I-1 was answered by measurement when round 13 said
'arithmetic only, no new measurement'"* (`v14:1895`) and §15 item 2 puts the same clause to me for
ruling. What the document does **not** quote is the words *"one line"* — see **M-1**.

**M-1's Repair sentence (`R13:597-598`), one limb removed:** residue is the finding body and the
non-blocking justification — a measurement report, not a prescription. Everything prescriptive
landed.

**M-2's Repair sentence (`R13:608`), one limb removed:** residue is the finding body and the
non-blocking justification. Nothing prescriptive survives.

**Residue verdict: clean on the two Minors; on I-1 the one prescriptive residue is quoted, labelled
and ruled on rather than dropped.**

## B.3 Disposition of the three findings at limb level

| finding | prescribed | landed | ruling |
|---|---|---|---|
| **I-1** | reconcile the counts; either print `44` or scope the `24`; replace *"24 timed runs by two parties"*; §7.3 follows | re-measured as `7 × 8 = 56` in one command; three-party split table `8 + 3 + 24 = 35`; every mention carries one accounting | **ADOPTED, WIDER than prescribed, and the widening is warranted** (C.1) |
| **M-1** | print `0.027`–`0.56` and `3.5×`–`7.6×` | both printed; the per-dataset triples re-derived, and they reproduce round 13's exactly under my own measurement | **ADOPTED** |
| **M-2** | print `six`, or say *"four substitution classes"* | **both**: four classes named **and** six lines stated | **ADOPTED, both disjuncts** |

**No repair is claimed that the artifact does not contain, and no repair landed narrower than
prescribed.** One landed wider — I-1 — and it is declared as a widening in the row, in the limb
cell, in a dedicated paragraph and in §15.

---

# PART C — MY OWN VERIFICATION OF ALL TWELVE §3 ITEMS

| # | claim | result |
|---|---|---|
| **V1** | every run-count reconciles | **PASS** — C.2. My own grep and arithmetic: `7 × 8 = 56`, `8 + 3 + 24 = 35`, `32 + 20 = 52`, cumulative `12 / 36 / 136`. **No figure asserts a total its stated parts cannot produce.** One provenance gap in the *v13* term is **M-2** |
| **V2** | re-measure the arena's import set, rung by rung | **PASS** — C.1. My 56 starts; rung 7 `3.070–3.540 s`. `3.8 s` bounds the whole pool |
| **V3** | Phase 1g and the re-multiplied column | **PASS** — C.3. **26 rows**, all **73** processes, column sums to `2934.5`, every product re-derives, **zero stale totals** |
| **V4** | the six limb quotations, by subtraction; rule the widening | **PASS** — B.1, B.2. **6/6 FAITHFUL.** Widening ruled **warranted**, C.1 |
| **V5** | re-run the audit; byte-compare; verify `+0 chars` by direct diff | **PASS** — A.1, A.2. Exit `0`, byte-identical at 1969 bytes; **four classes over six lines** confirmed by direct diff |
| **V6** | break the self-exclusion in the form that bites | **FAIL of the premise** — A.3. The plain splice is **not** vacuous against v14; it exits `1`. **This is I-1** |
| **V7** | the two Minors | **PASS on both** — C.4, A.2 |
| **V8** | rebuild the arms from §3.4; one bit-exact predicate; the misreading under `2e-6` | **PASS** — C.5 |
| **V9** | `ρ*`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18` on row-renormalised keys | **PASS** — C.5, C.4 |
| **V10** | Holm counterexample; `n ≤ 12`; §3.7's two blocks with two verbs | **PASS** — C.6 |
| **V11** | §7.9's sum, and the corrected v13 terms | **PASS on the sums** — C.2. Heading `v1–v14`; `12` mints, `36` wall, `136` CPU all re-derive; §7.8 and the footer agree. The *provenance* of the v13 term is **M-2** |
| **V12** | §6's 20 gate rows `12 G / 6 L / 2 R`; §13.1's 27 contiguous items; 37/37 digests | **PASS** — C.7 |

## C.1 **RULING ON THE WIDENING: warranted — and I would have done the same**

I built the rungs myself from `headspace_arena.py:28-46` plus §13.1 item 27, under
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8` and `CUDA_VISIBLE_DEVICES=""`, timing the full process wall
with `/usr/bin/time -f '%e'` around each `python -c`, **8 runs per rung, seven rungs, one shell
command**:

| rung | my runs | my range (median) | v14's stated |
|---|---|---|---|
| 1 bare interpreter | 8 | `0.010–0.010 s` | `0.01 s` |
| 2 `+ numpy` | 8 | `0.090–0.100 s` (0.090) | `0.09–0.10 s` |
| 3 `+ torch` | 8 | `1.760–1.870 s` (1.780) | `1.75–1.98 s` |
| 4 `+ faiss` | 8 | `1.790–1.860 s` (1.830) | `1.79–1.95 s` |
| 5 `+ c01 + mechfix_ops` | 8 | `1.800–1.900 s` (1.850) | `1.81–1.94 s` |
| 6 arena's actual top-level set | 8 | `3.040–3.260 s` (3.210) | `3.02–3.21 s` |
| **7 the arena class** (`+ c01` `+ runtime_block()`) | **8** | **`3.070–3.540 s`** (3.250) | **`3.10–3.70 s`** |

`sklearn` alone (rung 6 − rung 5) measures `≈ 1.32 s` against v14's *"≈ 1.2 s"* — the same
two-thirds share. **My whole ladder cost 95.7 seconds**, against v14's *"about `96` seconds"*
(`v14:1900`), which is the closest agreement between two parties anywhere in this file.

**The ruling.** Round 13 prescribed *"one line, arithmetic only, no new measurement"* with two
disjuncts. v14 took neither and re-measured. **I rule the widening warranted**, on three grounds.

1. **The prescribed outcome is delivered anyway.** No figure in v14 asserts a total its stated parts
   cannot produce — I grepped every one (C.2). The prescription's *purpose* is met in full; only its
   *method* was exceeded.
2. **The prescribed method could not have produced a uniform sample, and uniformity is what the
   defect was about.** Round 13's own grading note is *"when the warrant for overriding a reviewer is
   'I measured more than you did,' the count is load-bearing."* Printing `44` would have made a
   consistent statement of a sample that was `4` runs on five rungs and `10` on two, assembled from
   two commands. The consistent statement of an uneven, two-command sample is weaker evidence than a
   uniform one-command sample, and the *sample* is the warrant. Re-measuring converted an
   accounting exercise into a reproducible one — which is why I could re-run it in a single command
   and get the same structure.
3. **It cost 96 seconds, wrote nothing, and moved no quantity.** `3.8 s` is unchanged; the pooled
   maximum is unchanged; §8's total is unchanged; no heartbeat interval moves.

**What would have made me rule the other way**: if the re-measurement had moved `U11`, moved Phase
1g, moved §8's total, or replaced round 12's and round 13's samples instead of pooling with them.
None of that happened — the three-party table *preserves* both prior parties' observations with
their own counts and their own ranges, which is the opposite of a designer preferring its own
instrument. **A designer preferring its own instrument would have dropped the other two rows.**

**On the pooled endpoints.** Both are now correctly attributed and I checked both against the source
reviews: round 12's `+ runtime_block()` rung at `R12:349` is `3.12–3.27 s` over *"three runs per
rung"* (**exact**), and round 13's rung 7 at `R13:237` is `3.094–3.717 s` over 24 runs (**exact**).
v13's mis-attribution — its `3.00` was round 12's **sklearn-only** rung (`R12:348`, `3.00–3.03 s`),
not an arena-class one — is correctly diagnosed and corrected. **Recorded for completeness:** my own
rung-7 minimum, `3.070 s`, sits `0.024 s` *below* the pooled minimum, so the stated `3.094` is not a
bound on a fourth party's sample at the low end. That is not a defect — the pooled range is a report
of three parties' observations, only its **maximum** is load-bearing, and a lower minimum is
conservative for the heartbeat.

## C.2 The run-counts, grepped and re-derived — **every figure reconciles**

I grepped every occurrence of `56`, `52`, `44`, `35`, `24`, `8 runs`, `rungs`, `timed starts`,
`timed runs`, `two parties` and `three parties`, and reconstructed each figure from its stated parts
without reading the total:

| where | figure | its stated parts | my arithmetic |
|---|---|---|---|
| §7.7 decomposition table (`v14:1226-1234`) | — | 7 rows, `runs` column = `8,8,8,8,8,8,8` | **56** ✓ |
| §7.7 prose (`v14:1221`) | `56` | `7` rungs × `8` runs | **56** ✓ |
| §7.7 three-party table (`v14:1244-1249`) | pooled `35` | `8` + `3` + `24` | **35** ✓ |
| §7.7 `U11` row (`v14:1209`) | `35` runs by three parties | `8` / `3` / `24` split named in the row | **35** ✓ |
| §7.3 (`v14:1153`) | `52` in v13, `56` in v14 | matches §7.9's two terms | ✓ consistent |
| §8 Phase 1g (`v14:1435`) | `35` runs | `7 × 8` here + round 12's `3` + round 13's `24` | **35** ✓ |
| §9 (`v14:1517-1518`) | `35` arena-class runs, three parties | same | **35** ✓ |
| §7.9 v14 term (`v14:1393-1394`) | `56`, one command | `7 × 8` | **56** ✓ |
| §7.9 v13 term (`v14:1382-1384`) | `52`, two commands | `8 × 4 = 32` + `2 × 10 = 20` | **52** ✓ arithmetically |
| §7.9 cumulative (`v14:1399-1404`) | mints `12`, wall `36`, CPU `136` | `7+1+4+0×6`; `22+4+2+1+1+1+1+2+2`; `89+21+6+3+3+3+3+4+4` | **12 / 36 / 136** ✓ |
| footer (`v14:2307-2308`) | `56` timed starts, `7 × 8`, one command | same | ✓ |
| §14 I-1 row (`v14:1891`) | `56`, `35`, `52` | same | ✓ |
| §15 item 1 (`v14:2272-2273`) | `7 × 8 = 56`, `8 + 3 + 24 = 35` | same | ✓ |

**Round-13 I-1 is discharged.** Every mention now carries one accounting, and the pooled figure is
derivable from its parts at every site. The one thing that does **not** reconcile against a primary
source is the *v13* term's decomposition — see **M-2**, which is Minor because the `52` warrants
nothing and enters only a historical spend line.

`3.8 − 3.717 = 0.083` ✓, the stated headroom.

## C.3 §8 re-multiplied, every row, by my own arithmetic

I re-entered the entire printed product column and summed it without reference to the stated total:

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

**Every figure in V3 is confirmed. §8 has exactly 26 rows.** I additionally re-multiplied **every
one of the 23 unit×count products** independently — `15×40.39=605.9`, `3×49.30=147.9`,
`174×0.0461=8.0`, `67×0.033=2.2`, `240×0.00305=0.7`, `540×0.00629=3.4`, `60×0.1873=11.2`,
`120×0.00629 + 60×(2/13)×0.1873=2.5`, `40×0.04239=1.7`, `90×0.08098=7.3`, `2×4.63=9.3`,
`2×11.27=22.5`, `(4.63+0.21)=4.8`, `62×0.62=38.4`, `3072×0.08908=273.7`, `92×0.126=11.6`,
`3×3.70+3×3.49=21.6` — **zero mismatches at the printed precision.** Every count re-derives:
`(30×3)+(6×4)+(30×2)=174`; `60×2+30=150`; `256×3×2×2=3072`; `23×2×2=92`; `14×3×2×2=168`;
`2×60=120`; `2+60=62`; `4×60=240`/`9×60=540`; `7×6+5×6=72` and `72×2.0e-05=0.0014 s`; `66+6+1=73`.
Population constants re-derive: `446/743=0.600269`, `399/579=0.689119`, `446/744=0.599462`, caps
`⌊0.01×743⌋=7` / `⌊0.01×579⌋=5`, `0.02×743=14.86`, `0.02×579=11.58`, the `(2,21,22)` counterexample
mean `15.00`, `20/743=2.69 %`, `√2048=45.25`, recovery `0.02/(0.8884−0.6003)=6.94 %`.

**Stale-total sweep: zero.** `85.6 %`, `3663.4`, `2953.0`, `3691.3`, `3207.6`, `4028.7`, `48.8 min`,
`61.0 min` all occur **0** times. Every occurrence of `2930.4`, `2930.7`, `2933.9`, `3667.4` is
inside §8's own provenance narrative; every surviving `3.2 s` (five sites) and the single `1.7×` is
historical, quotational, or the deviation record.

## C.4 The two Minors, and the blindness grep across v1–v14

**§6.1's row-norm span and gap factor (round-13 M-1) — exactly reproduced.** Over all 36 banked
`K_train` matrices at `artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`:

* row norms span **`0.0271`–`0.5596`** (HateMM `0.0410`–`0.5596`, MHC-ZH `0.0271`–`0.2882`) → v14
  prints `0.027`–`0.56` ✓, and §14's per-dataset figures are exact.
* The **gap factor** is the ratio of the *order statistics*, which is what §6.1's sentence
  ("reading **these six figures** off the stored arrays literally") actually asserts. Measured:
  HateMM `7.64 / 6.21 / 3.47`, MHC-ZH `7.22 / 6.39 / 4.74` — **round 13's figures to the digit** —
  giving the range `3.47×`–`7.64×` → v14's `3.5×`–`7.6×` ✓. *(A caution for any later round: the
  per-cell ratio is a different quantity and spans `2.45×`–`9.45×`. The order-statistic reading is
  the one the sentence states and the one that reproduces.)*
* All six trained-head `ρ` figures reproduce to the digit under `np.median` (the average of the 9th
  and 10th of 18): HateMM `0.447803 / 0.562434 / 0.632996`, MHC-ZH `0.340179 / 0.574247 / 0.667326`,
  and **0/18 above `ρ*` on both**. **M-1 landed correctly.**

**Blindness, v1–v14, by my own grep.** Under the leading-digit convention on the **closed** interval
`[0.6, 0.99]` across all fourteen drafts:

* **v1–v5: `98`** ✓ · **v1–v10: `116`** ✓ · **v1–v12: `118`** ✓ — v14's triple reproduces exactly.
* Excluding the two self-referential endpoint tokens: **`96 / 114 / 116`** ✓.
* Half-open: `97` for v1–v5, `115` for v1–v10 — confirming round 12's diagnosis.
* **The new-in-v13 in-band set is EMPTY and the new-in-v14 in-band set is EMPTY.** The corpus total
  is unchanged at `118` from v12 through v14, so *"Neither v13 nor v14 adds one"* is verified rather
  than inherited, and **no battery-arm accuracy exists anywhere in v1–v14.**

## C.5 The arms, the algebra, and `ρ` — rebuilt from the prose alone

I implemented `fuse`, `paired` and the contrast/Givens definitions **from §3.4's text only**, then
compared against `prepare_views` called through the frozen `c01_policy_contrast_a0`:

* **`GATE-C01PARITY`: `max|diff| = 0.000e+00`** across all 13 arms on **both** datasets, at
  `n = 744` one-hot `{355}` (HateMM) and `n = 579` all-False (ZH). The gate states **one** predicate.
* **The un-normalised misreading** (dropping the endpoint pre-normalisation §3.4 pins) measures
  **`1.878e-06`** (HateMM) and **`1.609e-06`** (MHC-ZH), **both under `2e-6`** — reproduced to the
  digit, and confirming why the `2e-6` clause had to be struck from `GATE-C01PARITY`.
* **Algebra guard**: `8.941e-08` at θ=0 on both datasets; `1.192e-07` (HateMM) / `8.941e-08` (ZH) at
  θ=45 — exactly §1's figures.
* **`GATE-ROWSUBSET`: `max|diff| = 0.000e+00`** — the `n = 743` all-False build is bit-identical to
  the `n = 744` one-hot build restricted to the 743 surviving rows, all 13 arms.
* **Exact-zero rows**: HateMM `{355}` only, MHC-ZH none — measured, not read.
* **26/26 `ρ_raw` reproduce at 6 dp**, every value in §6.1's table, under `float64` accumulation over
  `float32` keys. `ρ*` = `0.968176` (HateMM) / `0.977223` (ZH), supplying arm `endpoint_std`,
  runners-up `common` at `0.964446` / `0.969686` ✓.
* The `1.301e-03` shift from including the masked zero row reproduces exactly
  (`0.968176 × (1 − 743/744) = 1.301312e-03`).

## C.6 Holm, feasibility, and §3.7's two verbs

Executed through **C01's own `holm_adjust`** (`c01_policy_contrast_a0.py:1775-1784`), padding the
family to `m`:

| witness p-values | `m = 92` pad `0.5` | `m = 92` pad `1.0` | `m = 46` |
|---|---|---|---|
| 24 × `1/2001` | **24/24** | **24/24** | **24/24** |
| 23 × `1/2001` + 1 × `2/2001` | **23/24** | 23/24 | **24/24** |
| 24 × `2/2001` | **0/24** | 0/24 | **24/24** |

`22/22` for the `displacement` disjunct. **The three-way equality at the floor holds and the
counterexample one step off the floor holds** — §5.5's table is exact, and its honest warrant
(the 92-freeze is kept for auditability, not because it is consequence-free) is the right one.
`92 × 2/2001 = 0.091954 > 0.05` while `46 × 2/2001 = 0.045977 ≤ 0.05` ✓.
Feasibility: `1/257 = 0.0038911`; `12/257 = 0.04669 ≤ 0.05`; `13/257 = 0.05058 > 0.05` ⇒ **`n ≤ 12`** ✓.

**§3.7 has two blocks with two distinct verbs** — population-derived constants *computed from the
arena, never read*, and frozen C01 config constants *read from the sha-gated config and asserted
equal* — with the `<=` operator correctly in the **read** block. I verified all five config
constants directly against `configs/c01/c01_a0_v2.json`: `normalization_epsilon = 1e-12`,
`tiny_displacement_epsilon = 0.001`, `max_tiny_displacement_fraction = 0.05`,
`max_small_displacement_fix_fraction = 0.5`, `small_displacement_train_quantile = 0.1`; plus
`gain_controls` = the five names §5.1 uses (so `C` is **six** and **five** as stated),
`minimum_gain_over_strongest_control = 0.02`, `minimum_net_fixes = {HateMM: 3, MHC_zh: 2}`,
`statistics.seed = 20260728`, `holm_alpha = 0.05`, `n_bootstrap = 2000`,
`n_id_hash_permutations = 256`, `bootstrap_lower_quantile = 0.05`, `bootstrap_upper_quantile = 0.95`,
`permutation_hash = sha256`, `retrieval.fix_break_reference = endpoint_std`, and
`required_halt_only_validity_guards` = **seven** entries **not** containing
`require_no_small_displacement_dominance` — which is what makes S7 a SURVIVE condition rather than a
HALT gate. `orthogonal_rotation_control.angles_degrees = [8.3, 17.6, 29.1, 60.4, 72.7, 83.8]` ✓,
`inputs.standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`, `feature_dim = 3584` ✓.

## C.7 Digests, gates, items — and the C01 evidence table

* **37/37 digests recompute identically.** Eight rows carry the `…` ellipsis and **all eight resolve
  to exactly one file** in the tree. All four new-code paths (`c06_falsifier_mint.py`,
  `c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`, `c06_falsifier_cpu.sbatch`) are
  **absent**.
* **§6 has exactly 20 gate rows, `12 G / 6 L / 2 R`**, and the G-set and L-set match §5.6's lists
  **name for name**, symmetric difference empty in both directions.
* **§13.1 defines `(1)…(27)` contiguously**, verified by my own scan.
* **§1's evidence table is exact.** All **16** accuracies and all **16** net-fix integers reproduce
  from `C01_A0_OUT.json`; `net_fixes.reference` is `"common"` (HateMM) and `"endpoint_concat"`
  (MHC-ZH), **not** `endpoint_std`, confirming **D-1**;
  `gain_over_strongest_control` accuracy `-0.009345794392523366` → **`−0.0093`** (the round-5
  erratum) and `-0.02564102564102566` → `−0.0256`; `decision.datasets.*.pass = false` on both;
  `decision.continue = false`. §10.2's counts check out at source: **4 of 6** HateMM rotations
  (`17.6/29.1/60.4/72.7` at `0.8505`) and **2 of 6** ZH rotations (`8.3/29.1` at `0.8462`) sit below
  the primary.
* **§6.2's clearances**: `0.8505−0.6203 = 0.2302`, `0.8598−0.6203 = 0.2395`, `0.8846−0.7091 = 0.1755`,
  `0.8590−0.7091 = 0.1499` — the set is exactly `{0.1499, 0.1755, 0.2302, 0.2395}`, minimum rounding
  to `0.15` and maximum to `0.24`. **`GATE-ARMVIAB`'s escape branch is confirmed unreachable.**
* **§1's two verbatim quotations match `TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`
  exactly**, and `new_status` is `gated_on_zero_cost_falsifier`.

Every source citation I checked resolves as stated: `headspace_mint.py:106-116` (the `torch.load`
guard asserting `"test_seen" not in s and "/test" not in s`), `:192-194` (early return),
`:199` (unconditional native `dev_seen` load), `:203-216` (fold parity vs banked `vsw_ckpt`, with
`.npz` written only after at `:321-325`, `lab_dev` at `:323`), `:68` (top-level `StratifiedKFold`),
`:82-94` (`runtime_block` deferring `threadpoolctl`/`scipy`/`sklearn`); `classifier.py:80` comment
and `:81-82` default-bias projections, `:140-141` `torch.mul` under `align`, `:146`
`embed = self.mlp[:-2](x)`; `c01_policy_contrast_a0.py:1725` (reporting `fix_break`),
`:2702-2714` (decision `net_fixes` via `select_strongest_ordinary_control`), `:2724` (the
consistency `die`), `:1940-1948` / `:1955-1962` (guards and the `(accuracy, macro_f1, −index)`
ranking), `:2036` (`small_mask = dev_min <= threshold`), `:2049`/`:2050` (`"source_rows"` /
`"registered_null_rows_excluded"`, exactly as round-9 M-1 corrected), `:1372-1377` (the `2e-6`
algebra guard), `:1787` (`id_hash_permutation`); `mechfix_ops.py:45-49` (`_flat_ip` builds and adds
**inside** the timed function), `:82`, `:94` (the vote formula);
`headspace_arena.py:28-46`, `:72`, `:75-89` (per-fold `mint_{ds}_s{seed}_f{fold}.npz` load);
`headspace_fidelity.py:31/:33/:66` (**no `dev_seen` at all**); `c09_guard/sitecustomize.py` imports
`c09guard` and calls `install()`.

## C.8 Verdict-path enumeration — mine, from the document alone

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
`v14:771-772`: *"Absence by declared drop is lawful; absence by computation failure in a surviving
lineage still HALTs."*

**No gate failure is reportable as a closure.** CLOSE requires all twelve globals to pass **and**
both lineages to have passed every per-lineage gate on **both** datasets. A global failure HALTs; a
per-lineage failure drops that lineage on both datasets, falsifying rule 2's conjunct, so the only
reachable outcomes are SURVIVE-on-the-clean-lineage or HALT. **A CLOSE always rests on two clean
negatives, never one.**

## C.9 Freeze-readiness, operationally

Judged as the document an operator with no context would execute.

* **No decision point on the run boundary.** One `sbatch`, 8 CPU / 32 GB, no `--gres`, no `--time`,
  no array, no dependency, no requeue. The 73-process order is stated — `66 mints → 6 fidelity →
  1 arena` — with `GATE-SHA` once in the driver before any of them and `GATE-POP` before any
  population-consuming gate.
* **Preconditions are checkable.** 37/37 digests recompute; all four new-code paths absent;
  `mints_present_before_arena` is a declared binding predicate; the `dev_path_opens ==
  mints_executed` choice is correct and would otherwise HALT a legitimate resume.
* **Per-class import accounting is consistent.** 66 mints carry startup inside their full-process
  walls; 6 fidelity inside `U9` (whose `3.70 s` cannot be an internal timing, since the same
  process's imports alone measure `3.06–3.16 s`); the 73rd, the arena, priced once at Phase 1g.
  `66 + 6 + 1 = 73`. **I found no seventy-fourth python process.**
* **Heartbeat.** Line-buffered `buffering=1` handle, per-phase and per-cell granularity, plus an
  unbuffered bash echo per mint; longest un-instrumented span `11.27 s` (`14.1 s` conservative,
  and `11.27 × 1.25 = 14.0875` re-derives) against a `~15 s` bound. **The arena startup at
  `3.094–3.717 s` — and my own maximum of `3.540 s` — leaves `~11 s` of headroom and changes no
  interval.** `rule_2_heartbeat` unchanged and satisfied.
* **Exit and resume semantics defined.** HALT names the failing gate in its final line; a
  `RuntimeError` from the imported C01 algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE` with its
  `context` string, and written before exit.
* **Test-split non-contact by construction, verified at source.** Three layers, all present in the
  frozen files: `headspace_mint.py:106-116` rebinds `torch.load` behind an assertion; the driver's
  `split == "train"` assertion; `c09_guard/sitecustomize.py` installing `c09guard` at interpreter
  startup. `GATE-LEDGER` binds `test_path_opens == 0` and `test_label_materialisations == 0`.
  §3.1 states no `dev_seen_*-ro_*` and no `test_seen` ro cache is opened by any phase.
* **The `$0` character holds.** No GPU, no Modal, no test contact, no new data.

## C.10 `rule_1_compute_projection` — the axis I searched

**I name one, and it returns nothing.**

**The shuffle draw's permutation construction — whether `U4`'s timed region encloses it.** `U4` is
the single largest uncorroborated unit (`273.7 s`, `9.3 %` of the total) and §7.7 flags it as such.
§5.4.1 pre-registers the null as C01's `id_hash_permutation(ids, dataset, split, draw, seed,
fixed_indices=())` — a per-draw sha256 over every id plus a sort — while §7.7 describes `U4`'s
object as *"2 arms × 5 folds + rebuild"*, which does **not** name the permutation. If the timed
region were rebuild-and-vote only, all `3072` draws would carry an unpriced companion, and that is
exactly the defect class rounds 11, 12 and 13 found (a per-process fixed cost, a per-process unit
basis, a reported sample size).

**Measured:** `id_hash_permutation` at `n = 743` costs **`1.216–1.273 ms`** (median `1.243 ms`) over
20 timed calls through C01's own function. Over `3072` draws that is **`3.82 s`** — `1.4 %` of Phase
3, **`0.13 %` of the `2934.5 s` total**, inside the row's own printed precision and two orders inside
the declared `× 1.25` margin. Worse, the permutation depends on `(seed, draw, dataset, split)` and
not on lineage, so at most `1536` distinct permutations exist and the true figure is likely half
that. **No item.**

**Incidental corroboration, recorded because §7.7 asks for it.** Two further uncorroborated units
are bounded by their frozen values under my independent measurement: `U8` (two ro-cache
`torch.load`s, warm) measures `0.0106–0.0287 s` against the frozen **`0.033 s`**; and a `B = 2000`
resample over `n = 743` with two metric legs measures `0.023–0.028 s` against `U3`'s frozen
**`0.126 s`**. Both frozen units are conservative. Together with round 13's `U2a`–`U2d` and `U7`,
that leaves `U4` as the only substantial unit still uncorroborated end to end — and its one
plausible unpriced companion is now measured at `0.13 %` of the total.

**No eleventh uncounted item on this axis. Fourteen rounds, ten items.**

## C.11 §4.C — can any gate fire on a warranted CLOSE? **No, for all twenty**

A *warranted CLOSE*: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction.** `GATE-DET1` (thread env before
any python starts). `GATE-SHA` (37 digests over files no phase writes — **all 37 recomputed by me**).
`GATE-FOLD` (banked parity flags + `fold_of`; the assertion is at `headspace_mint.py:203-216` and a
`.npz` is written only after it passes). `GATE-FLOOR` (six banked anchors reproduced on **native**
keys, so no ro-derived arm outcome can reach it). `GATE-POP` (populations `743/579`, class counts
`(297,446)` / `(180,399)`, index-set identity, constants recomputed — all verified by me).
`GATE-C01PARITY` (a property of the **builder**; I measured `0.000e+00`). `GATE-ROWSUBSET` (builder
property; `0.000e+00`). `GATE-RHORAW` (a property of the ro caches and the raw leg, identical for
both lineages; **26/26 reproduced at 6 dp**, asserted at 4 dp). `GATE-NULLREMOVED` / `GATE-ZEROMASK`
(`{355}` / `{}`, which I measured as the sole exact-zero row). `GATE-IDPARITY` (ids/labels parity).
`GATE-LEDGER` (declared counts). **None reads which arm won.**

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
  head-space headroom. Same argument.

**The two `R` gates** (`GATE-DOMAIN`, `GATE-DEVFID`) carry **no bar** and cannot fire at all.

**All twenty: no gate can fire on a warranted CLOSE.** The load-bearing structural fact is §6.2's
retirement of `GATE-ARMVIAB`: I confirmed its escape branch unreachable (all four raw clearances
`0.1499`–`0.2395` clear), which had reduced it to a one-sided HALT on precisely the warranted-CLOSE
outcome. With it deleted, **no lower-bound instrument HALT is applied to a real arm anywhere in this
design** — I read the §6 table row by row, and `GATE-ARENA`'s lower bound on `endpoint_std` is the
only lower accuracy bound in the document.

---

# FINDINGS

## CRITICAL — none

## HIGH — none

## IMPORTANT

### I-1. §14.1's *"The same vacuity holds for v14"* and §15 item 5's *"no row or limb cites §14.1"* are false of v14, are contradicted by §14.1's own embedded transcript, and understate the mechanism they exist to defend. Measured: the plain splice exits `1`, not `0`.

*Attaches to:* §14.1 (`v14:2003-2004`); §15 item 5 (`v14:2293-2295`).
*Contradicted within the artifact by:* §14.1's own step-(6) note (`v14:2102`, *"§14.1 has left that
list because round-13 M-2 lands there"*); the M-2 disposition row (`v14:1893`, landing column
`§14.1`); the M-2 limb (`v14:1922`, landing column `§14.1`); and the embedded transcript's own lines
`OK M-2 cites §14.1` (`v14:2063`) and `OK M-2 … -> §14.1` (`v14:2078`).

**What the document claims.** §14.1: *"Round 13 ran both forms against v13 and reported both: the
plain splice vacuous at exit `0`, the biting form failing one row at exit `1`. **The same vacuity
holds for v14, whose limbs land elsewhere**; §15 carries the biting form forward."* §15 item 5:
*"The plain splice is **vacuous** against v14 as it was against v12 and v13 — **no row or limb cites
§14.1**. Do what rounds 12 and 13 did: splice **and** insert a synthetic §14.1-citing row."*

**What I measured.** I spliced v13's §14.1 into v14 — the plain construction, no synthetic row — and
ran the §14.2 script unmodified:

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  M-2   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 2 ; rows failing: 1
  FAIL  M-2   *"Repair: print `six`, or say "four substitution cla -> §14.1 NOT DIFFED
  limbs landed: 5 ; limbs open/failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**One row and one limb cite §14.1**, so the plain construction bites natively. The biting form adds
a second failing row (`X-9`) and changes nothing else. **v14 is the first version since v11 in which
the plain counterfactual is non-vacuous**, and v11 is the version whose round reported exactly this
shape (*"two failing rows, two failing limbs"*).

**Why this is not merely a wording slip.** Three distinct things go wrong at once.

1. **A false factual claim about the artifact**, in the section whose entire purpose is to make the
   disposition record machine-checkable rather than asserted.
2. **A self-contradiction inside one subsection.** §14.1 says the limbs *"land elsewhere"* three
   paragraphs before its own reading note explains that §14.1 left the changed-but-uncited list
   *"because round-13 M-2 lands there"*. Both sentences cannot be true.
3. **A misdirection of the next reviewer.** §15 item 5 tells round 15 that the plain form has nothing
   to bite on and to go straight to the synthetic construction. A reviewer who complies never
   discovers that the mechanism now fires without help; a reviewer who checks gets a result the
   document says is impossible. The review request for this round inherited the same false sentence
   verbatim, which is how far an unchecked claim travels in one hop.

**The claim is also weaker than the truth.** Rounds 12 and 13 had to *manufacture* a citing row to
prove the mechanism live. v14 does not: its own repair supplies one. The strongest available
statement — *the self-exclusion is proved live by this document's own disposition record, with no
synthetic construction* — is exactly the statement §14.1 declines to make.

**On severity, stated so the grade is auditable.** Not **Critical**: no verdict quantity moves, no
§8 loop is uncounted, no gate is touched, no test-split exposure, no un-preregistered threshold, and
**the M-2 repair itself is landed and independently verified** (A.2) — the claimed repair *is* in the
artifact. Not **High**: nothing is narrowed, all three of round 13's findings land in full, and the
verdict's authority and scope are untouched; the error runs in the conservative direction. **Important**
is the grade the brief defines for *"completeness, reproducibility, or an argument right for a weaker
reason than available"*, and this is both — a checkable assertion about the document that does not
survive being checked, made in place of a stronger one that does. It is the same family rounds 7–13
have graded Important, and it is the fourteenth consecutive round in which the finding sits inside
the previous round's repair: **the vacuity sentence is an unchecked inheritance from v12 and v13,
which is the precise defect round-13 M-2 named — *"the line count was inherited from round 12 without
being checked"* — sitting in the paragraph that lands round-13 M-2's repair.**

**Repair — two sentences, no new measurement.**
1. In §14.1 (`v14:2003-2004`), replace *"The same vacuity holds for v14, whose limbs land elsewhere;
   §15 carries the biting form forward"* with the fact: *"Against v14 the plain construction is no
   longer vacuous — round-13 M-2's row and limb both cite §14.1 — so splicing v13's §14.1 into v14
   yields `UNCHANGED §14.1`, `FAIL M-2 cites §14.1 -- NOT DIFFED`, one failing limb,
   `named by a row but unchanged: ['14.1']` and exit `1`, with no synthetic row required. v14 is the
   first version since v11 for which that holds."*
2. In §15 item 5 (`v14:2293-2295`), strike *"The plain splice is vacuous against v14 as it was
   against v12 and v13 — no row or limb cites §14.1"* and ask round 15 to run **both** forms and
   report both, noting that the plain form is expected to exit `1`.

---

## MINOR (non-blocking; neither touches the verdict path)

* **M-1. §14's widening paragraph quotes round 13's prescription with the words *"one line"*
  dropped** (`v14:1895`, and the same partial quotation at `v14:2277-2278`). Round 13 wrote
  *"**Repair — one line, arithmetic only, no new measurement.**"* v14 quotes it as *"arithmetic only,
  no new measurement"*. The omitted clause is a **second dimension** of the same deviation: the
  landed repair is not one line — §7.7 grew by `1583` characters and §7.9 by `949`, both printed in
  the document's own transcript. **Non-blocking, and emphatically so.** The four sentence-level limbs
  are quoted verbatim and in full (B.1); the framing sentence is not a limb, and round 13 itself
  treated the analogous *"Repair — two lines, and the second is the durable one"* as non-prescriptive
  framing when it subtracted round 12's paragraph. The size of the repair is disclosed by the audit
  transcript rather than concealed, and the widening is ruled warranted on its merits (C.1).
  **Repair:** quote the framing sentence in full where it is quoted at all — *"Repair — one line,
  arithmetic only, no new measurement"* — so the widening is declared against both of its clauses.

* **M-2. §7.9's decomposition of v13's `52` starts cannot be checked against v13's own §7.7, and v14
  states no bridge** (`v14:1382-1384`). The correction reads *"`52` timed interpreter starts, from
  two commands (`8` rungs × `4` runs = `32`, then `10` runs each on two rungs = `20`)"*. The
  arithmetic is right (`32 + 20 = 52`), but v13's §7.7 prints a **seven**-rung table at *"`4` runs
  per rung unless stated"* with rung 5 at *"(10 runs)"* and rung 7 at *"(14 runs)"* — which is round
  13's `44`. Reconciling `44` to `52` requires two unstated corrections: an **eighth** rung v13 never
  printed (`+4`), and rung 5's true total being `14` rather than the printed `10`, i.e. v13 recorded
  one rung by its *increment* and the other by its *total* (`+4`). Both are plausible — round 12's
  own ladder at `R12:337-349` has exactly **eight** rungs, which is presumably what v13 replicated,
  and round-13 I-1's second complaint (*"'10 each' contradicts §7.7's own `10` and `14`"*) is
  precisely the increment-versus-total confusion — but neither is written down, so a reader who
  follows §15 item 3's instruction to *"check the `52` against the two commands it describes"* can
  verify the addition and nothing else. **Non-blocking:** the `52` is a historical spend line that
  warrants nothing, enters only the `≈ 36 / ≈ 136` cumulative (which re-derives), and is not the
  count that overrides anyone. The **load-bearing** count — v14's `56` — is fully reproducible, and I
  reproduced it. **Repair:** one clause naming the bridge, e.g. *"v13's table printed `44` across
  seven rungs; the executed sample was `52`, because an eighth rung went unreported and rung 5's
  `(10 runs)` was the second command's increment rather than its total of `14`."* Or drop the
  decomposition and state the spend as `≈ 2 / ≈ 4` with the `52` unelaborated.

---

# REQUIRED RULINGS

## 1. §4.C — any gate that can fire on a warranted CLOSE? **No, for all twenty.** Derived at C.11 from the gate texts and my own measurements, not inherited.

## 2. Verdict-path enumeration: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.** C.8.

## 3. The widening (§4.A): **WARRANTED.** C.1. The prescribed outcome is delivered in full, the prescribed method could not have produced the uniform sample the finding was about, the cost was 96 seconds, and — decisively — the re-measurement **pooled with** round 12's and round 13's samples rather than replacing them, preserving each party's own count and range. That is not a designer preferring its own instrument. **What I would have done:** the same thing, and for the same reason round 13 gave when it overrode round 12 — a count that warrants an override has to be one a reader can re-run, and the two-command `4`/`10`/`14` sample was not.

## 4. §15 item 3 — correcting **v13's** recorded spend from `≈ 1 / ≈ 3` to `≈ 2 / ≈ 4`: **RIGHT, and it is the campaign's discipline, not an unusual step.**

Two reasons and one condition.

* §7.9 is a **live cumulative sum**, not an archive. Its terms are summed to `≈ 36 / ≈ 136` and those
  totals are asserted in the footer and re-derived by every round. A term known to be wrong cannot be
  left in a sum that is presented as checkable — the alternative (an erratum beside a figure that
  still enters the total) would make the sum wrong on purpose. Round-9 H-1 already moved the
  discharge mint's `≈ 1 / ≈ 4` from v7's column to v6's for exactly this reason, and every round
  since has endorsed that move.
* The correction is in the **conservative** direction (spend revised *upward*) and moves nothing a
  reader would act on: `2 + 2 = 4` wall-minutes and `4 + 4 = 8` CPU-minutes across v13 and v14, on a
  `$0` budget with no GPU.
* **The condition is that the corrected figure be re-derivable, and this one is only half
  re-derivable** — hence **M-2**. The principle is right; the execution stops one clause short. The
  campaign's numeric-provenance discipline is *"never transcribe numbers without re-reading the
  source log"*, and the corrected `52` is stated without the bridge that would let a reader
  re-read it against v13's printed table.

The general rule I would write down for later rounds: **a prior version's figure may be corrected in
place when it feeds a live sum, provided the correction states what the earlier version got wrong and
how the two reconcile.** v14 satisfies the first clause and not the second.

## 5. Rulings on §15's six open issues

1. **The one accounting, checked end to end.** **PASS.** C.2. I grepped every run-count site and
   re-derived each from its stated parts: `7 × 8 = 56`, `8 + 3 + 24 = 35`, `32 + 20 = 52`,
   `12 / 36 / 136`. **No figure asserts a total its stated parts cannot produce.** Round-13 I-1 is
   discharged. The one residual provenance gap is **M-2**, on the v13 term only.
2. **The widening.** **WARRANTED.** Ruling 3 above, C.1.
3. **v13's corrected spend term.** **RIGHT**, with the reconciliation clause missing — ruling 4
   above, **M-2**.
4. **The eleventh uncounted item.** **One axis named and searched — whether `U4`'s timed region
   encloses `id_hash_permutation` — and it does not yield.** Measured at `1.243 ms` per draw, `3.82 s`
   over all `3072`, `0.13 %` of the total and inside the row's printed precision. Two further
   uncorroborated units, `U8` and `U3`, are incidentally shown conservative by my own measurements.
   **No eleventh uncounted item.** C.10.
5. **The self-exclusion, in the biting form.** **Both forms run and both reported — and the premise
   of the question is false.** The plain splice against v14 exits `1`, failing the M-2 row and the
   M-2 limb; the biting form exits `1` with two failing rows. **This is I-1.** A.3.
6. **Is the record still sound, and is the design freeze-ready?** **The record is sound at limb
   level** — 6/6 faithful and complete, nothing prescribed missing, nothing claimed absent, nothing
   narrowed, zero stale totals. **The science is closed** and I re-derived every leg of it
   independently. **The design is freeze-ready on everything except I-1**, and I-1 is two sentences
   of prose in §14.1 and §15 that require no new measurement and move no quantity a reader would act
   on.

## 6. Process rules

* **`rule_1_compute_projection`.** Satisfied in form and in substance — every §8 row is a measured
  unit × an explicit count, all 23 products re-multiply exactly, the column sums to `2934.5`, all
  `73` processes are accounted, and there is no extrapolation from a reduced-scale dry run. One axis
  searched (C.10); it returns nothing and incidentally corroborates two more units. **No §8 row is
  affected by I-1 or by either Minor.**
* **`rule_2_heartbeat`.** **Unchanged and satisfied.** Line-buffered `buffering=1` appends plus an
  unbuffered bash echo per mint; the longest un-instrumented span is `11.27 s` (`14.1 s`
  conservative) against `~15 s`. The arena's startup at `3.094–3.717 s` — and my own maximum of
  `3.540 s` — leaves `~11 s` of headroom. **v14 changes no interval.**

## 7. Can the falsifier discharge the written condition at `$0`? **Yes.**

Every input exists and is digest-frozen (37/37, recomputed by me); the head space is re-mintable on
CPU at measured cost; the arms rebuild **bit-exactly** from the document's own prose (`0.000e+00`,
13/13, both datasets); the decision rule is pre-registered with its multiplicity resolution floor
proved attainable through C01's own `holm_adjust`; the verdict combination is total with exactly one
lawful absence path; and no gate can fire on a warranted CLOSE. **Neither I-1 nor either Minor bears
on this.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Nothing here authorizes execution. Before any job: (1) freeze with hashes; (2) a **separate**,
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — and I note
for that lineage, in addition to what rounds 12 and 13 flagged, that §13 item 23 should be read
broadly; that item 27 is addressed to it and binds any trimming **or extension** of the arena's
import set to a re-measurement of `U11` and a re-carry of Phase 1g; that item 22's placement of the
`GATE-FLOOR` vote in the arena process is what Phase 1f's `150` is priced against; and that §12's
three-layer test guard depends on the C06 sbatch exporting `PYTHONPATH` to reach `c09_guard`, whose
`sitecustomize.py` docstring still names **C09's** sbatch — a code-side wiring item, not a design
defect, since both guard files are sha-frozen in §11. (3) main-dialogue authorization. This document
is not authority to write `TARGET_STATE.json`.

---

# CLOSING

**The most severe finding is I-1, and it is a claim about the document rather than a number in it.**
Round 13 asked v14 to make its counts reconcile, and v14 did — I re-ran the measurement in one
command, got 56 starts in 95.7 seconds against the document's *"about 96 seconds"*, and every
run-count in the file reconciles under my own arithmetic with both pooled endpoints traceable to a
named party's stated rung. The widening that achieved this is warranted, and the tell is that it
**pooled with** the two prior parties instead of replacing them. What did not survive checking is a
sentence about the audit itself: §14.1 says the plain self-exclusion counterfactual is vacuous
against v14 and §15 tells round 15 the same, and both are false — round-13 M-2's own row and limb
cite §14.1, so the plain splice exits `1` with a failing row, a failing limb and
`named by a row but unchanged: ['14.1']`. The document's own transcript prints the citations that
refute the claim, on the page above it. **The sentence is an unchecked inheritance from v12 and v13 —
the exact defect round-13 M-2 named — and it sits in the paragraph that lands round-13 M-2's repair.**
It understates the artifact: v14 is the first version since v11 whose mechanism fires without a
synthetic construction, and that is the stronger statement it declined to make.

**Everything else is clean, and I say so as plainly as the brief asks.** The science reproduces on
every axis I tested, independently and at full precision: 13/13 arms at `0.000e+00` from the prose
alone, the un-normalised misreading at `1.878e-06` / `1.609e-06` under `2e-6`, `GATE-ROWSUBSET` at
`0.000e+00`, 26/26 `ρ` at 6 dp, 37/37 digests, 16/16 accuracies and 16/16 net-fix integers, the Holm
counterexample and its three-way equality through C01's own code, `0/18` trained heads on
row-renormalised keys, and both of round 13's Minors reproducing to the digit — M-1's gap factors
only under the order-statistic reading its own sentence states, which is the reading that works.
§8 re-multiplies to `2934.5` row by row with zero stale totals. All twenty gates are unable to fire
on a warranted CLOSE; the verdict path is total, mutually exclusive and admits exactly one lawful
absence; the blindness grep across all fourteen drafts finds the new-in-v13 and new-in-v14 in-band
sets both **empty**. A newly-searched cost axis returns nothing and incidentally shows two more
frozen units conservative. **v14 is two sentences of prose away from a GO. I have declined to grant
it early, and I have declined to inflate the finding I did have: I-1 is graded Important and not
High because nothing is narrowed and no quantity moves, and I raised no second finding I could not
measure.**
