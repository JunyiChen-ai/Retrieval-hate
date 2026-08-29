# C06 `$0` falsifier — independent design review, **ROUND 13**

**Reviewer:** fresh, independent of rounds 1–12 and of the designer.
**Artifact:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V13.md`, sha256
`308578cc8087f430a8cb0d9a520b67144e272338870fe39514ee86fabcb7db97`, 171270 bytes, 2281 lines —
**recomputed on disk, matches the review request.**
**Compute:** read-only. No GPU, no SLURM, no Modal, no arena run, no mint, no cache write, no
test-split access, no commit. `TARGET_STATE.json` read, not modified. **No repository file was
written or edited except this review.** Every temporary artifact — extracted script, spliced
counterfactual drafts, timing rungs — went to the session scratchpad. My import timings ran under
Python 3.11.8 (`HateVideo`), for which every module involved already had a `cpython-311` bytecode
cache, so **my measurements wrote no new `.pyc`**; `scripts/analysis/__pycache__/` holds the same 11
files before and after.

---

# VERDICT

## **REVISE — 0C / 0H / 1I + 2M**

Zero Critical. Zero High. **One Important.** Two Minor, neither touching the verdict path.

**The deviation is forced by measurement, and I rule for v13.** I rebuilt the arena's import set from
`headspace_arena.py:28-46` and §13.1 item 27, added `c01_policy_contrast_a0` and the `runtime_block()`
call, and timed it 24 times: **`3.094–3.717 s`**, median `3.172`, mean `3.254`. Ten of 24 exceed
`3.2 s`; five exceed `3.4 s`. Round 12's prescribed *"residual `≤ 0.2 s`"* is falsified by my own
data — my maximum exceeds `3.2 s` by **`0.517 s`**. Carrying `3.2 s` would have written a residual
bound that my measurement and v13's both refute, and made Phase 1g the second §8 row carried below
its measurement. `3.8 s` bounds my whole sample with `0.083 s` to spare. **This is a measurement
compelling a departure, not a preference dressed as one**, and v13 disclosed it in the limb cell, in
the paragraph above the limb table, and in §7.7's parenthetical.

**The science layer is closed and I re-derived it rather than inheriting it.** 13/13 arms rebuilt
from §3.4's prose alone at `max|diff| = 0.000e+00` on both datasets; `GATE-ROWSUBSET` at
`0.000e+00`; 26/26 `ρ_raw` at 6 dp; `ρ*` `0.968176` / `0.977223`; trained-head **0/18** on both;
16/16 C01 accuracies and 16/16 net-fix integers; 37/37 digests; the Holm counterexample reproduced
through C01's own `holm_adjust`; all twenty gates re-derived as unable to fire on a warranted CLOSE.
**The record is faithful:** 5/5 limbs verbatim, complete, and inside the `R12:NNN-NNN` range each
cites, with both Repair paragraphs subtracting to bare enumerative scaffolding, and **zero stale
totals**.

**The one Important is a run-count, and it sits inside v13's own repair — the seventh consecutive
round for which that holds.** §7.7's decomposition table specifies `4` runs per rung with two rungs
stated at `10` and `14`, which is **44** timed starts. §7.9 and the footer both report **`24` …
in total**, and §7.9's own breakdown (*"`4` per rung and `10` each for the two rungs the finding
turns on"*) reconciles with neither `44` nor `24` — it sums to `40`, and its *"10 each"* contradicts
§7.7's own `10` and `14`. Separately, §7.7's `U11` row and §9 both state the arena-class range rests
on *"24 timed runs by two parties"*, which requires round 12 to have contributed 10 arena-class runs,
whereas round 12 documents *"three runs per rung"* (`R12:337-349`). **The range is right and I
corroborate it; the sample sizes reported around it do not reconcile.** That matters because the
sample size is the entire warrant for departing from a prior review's explicit prescription.

I considered grading it Minor and rejected that. §7.9 exists because round-8 I-1 asked for the spend
to be *"shown as a sum so it is checkable rather than asserted"*, and this count is not checkable —
it contradicts the table it summarises and itself. I also considered High and rejected it: nothing is
narrowed, round 12's prescription is landed in full, no verdict quantity moves, and the carried
`3.8 s` is independently corroborated. **Softening a finding because a GO is one repair away would be
grading on trajectory, which the brief forbids in both directions; so would manufacturing one. I have
tried to do neither, and I dropped one candidate finding on exactly that ground** (§C.6).

---

# PART A — AUDITING THE AUDITOR

## A.1 The §14.2 script, re-run against final on-disk v13: **byte-identical, exit 0**

I extracted the script from the v13 fence, ran it unmodified, and captured stdout.

* **Exit code `0`.**
* Embedded transcript **1971 bytes**; my run **1971 bytes**; **`BYTE-IDENTICAL: True`**
  (sha256 of both: `eebc4829203aea509ac3e465fcbfd4c8d829ed56927b212bdda29575fe10efa4`).

The transcript is a **verified fixed point**, the sixth consecutive version for which that holds.

## A.2 `CHANGED §14.2 +0 chars` — verified by direct diff, not accepted

I diffed §14.2 between v12 and v13 directly rather than accepting §14.1's explanation:
**`len(v12 §14.2) = len(v13 §14.2)`, `identical = False`.** The differing lines are **six**, and
every one is a same-length version label:

```
-"""Mechanical disposition verification, C06 falsifier v12.   +... v13.
-(1) section diff v11->v12 ...                                +(1) section diff v12->v13 ...
-V_OLD='...DRAFT_V11.md'                                      +V_OLD='...DRAFT_V12.md'
-V_NEW='...DRAFT_V12.md'                                      +V_NEW='...DRAFT_V13.md'
-print('=== (1) SECTION DIFF v11 -> v12 ===')                 +print('=== (1) SECTION DIFF v12 -> v13 ===')
-print('=== (5) ... (round-11 prescriptions) ===')            +print('=== (5) ... (round-12 prescriptions) ===')
```

The **four substitution classes v13 names are exactly the four that occur**, and they cover all six
lines. v13's own account of its own change is therefore accurate. What does not reproduce is v13's
restatement of round 12's count — see **M-2**.

## A.3 Breaking the self-exclusion — both forms, as the brief requires

**Plain splice (v12's §14.1 into v13): vacuous, exit `0`.** The script printed
`UNCHANGED §14.1 (self, size not reported)` and dropped `14.1` from the changed-but-uncited list, but
no row failed. **v13's claim that the plain construction is vacuous against v13 is correct** — none
of its four rows and none of its five limbs cites §14.1; they land in §8, §7.7, §9, §13.1, §14, §7.9,
§6.1, §6.2 and §7.3.

**Biting form (splice + one synthetic §14.1-citing row): exit `1`, as designed.**

```
  UNCHANGED §14.1     (self, size not reported)
  FAIL  X-9   cites §14.1 -- NOT DIFFED
  rows verified against diff hunks: 4 ; rows failing: 1
  named by a row but unchanged:    ['14.1']
EXIT=1
```

**The self-exclusion covers size only and still fails a §14.1-citing row when §14.1 did not change.**
Mechanism live. §14.1's claim *"The logic is unchanged in v13"* is what A.2 and this test establish,
and v13 correctly does **not** claim the plain counterfactual reproduces. **No finding.**

## A.4 Section deltas, recomputed with my own splitter

I wrote a line-based splitter (the audit's is a regex-slice splitter) and reproduced **every printed
delta exactly**: `§8 +448`, `§9 +258`, `§14 +2541`, `§15 +945`, `§6.1 +570`, `§6.2 +183`,
`§7.3 +894`, `§7.7 +2264`, `§7.9 +540`, `§13.1 +1331`, `§14.2 +0`, `header +596`, `UNCHANGED: 45`.
For completeness, the one delta the audit self-excludes: **`§14.1 +731 chars`.**

The `defined 1..27` and `35 reference sites` lines both reproduce; my own scan confirms §13.1 defines
`(1)…(27)` **contiguously**, no gap, no repeat.

---

# PART B — DISPOSITION AUDIT OF ROUND 12'S FOUR FINDINGS, AT LIMB LEVEL

## B.1 The five limbs: **5 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded**

Each quotation was normalised for emphasis and quote-glyph only, then tested for containment inside
the exact `R12:NNN-NNN` line range it cites, read out of round 12's file.

| # | finding | limb (opening) | cited range | verbatim | inside range | ruling |
|---|---|---|---|---|---|---|
| 1 | I-1 | *"Re-state Phase 1g's basis honestly (`v12:1341`, …"* | `R12:392-399` | ✓ | ✓ | **FAITHFUL** |
| 2 | I-1 | *"Pin the arena's import set in §13, as one clause …"* | `R12:400-405` | ✓ | ✓ | **FAITHFUL** |
| 3 | M-1 | *"Repair: print `98`, or state the convention once."* | `R12:429-430` | ✓ | ✓ | **FAITHFUL** |
| 4 | M-2 | *"Repair: three words at `v12:943` …"* | `R12:439-440` | ✓ | ✓ | **FAITHFUL** |
| 5 | M-3 | *"the four measured clearances are `0.1499 / …"* | `R12:443-444` | ✓ | ✓ | **FAITHFUL** |

**No word is dropped from any of the five, including every qualifying clause.** Limb 1 is the place a
narrowing would have been most convenient — it is the limb v13 deviates from — and it carries the
prescription **in full**, including *"Keep the unit at `U11 = 3.2 s`"*, *"the residual `≤ 0.2 s`"* and
*"Say that the direction is no longer strictly conservative"*, all three of which v13's own deviation
contradicts. **Quoting the prescription that convicts you is the opposite of a narrowing**, and it is
what made this round's central judgement possible from the artifact alone.

## B.2 The residues, by subtraction

**I-1's Repair paragraph (`R12:390-405`), two limbs removed:**

> `Repair — two lines, and the second is the durable one.` `1. ⟦LIMB⟧.` `2. ⟦LIMB⟧.`

**Nothing but enumerative scaffolding and a non-prescriptive framing sentence.** No prescriptive
content survives, and both lines landed.

**M-1's Repair sentence (`R12:429-430`), one limb removed:**

> `` `0.615` / `0.66` and round 10's `0.8718` both reproduce exactly. ⟦LIMB⟧ ``

One clause remains; it is a **measurement report, not a prescription** — and v13 records it anyway
(§7.3 carries round 11's `0.615` / `0.66` and round 10's `0.8718`).

**M-2's Repair sentence (`R12:439-440`), one limb removed:**

> `… reference measurement only — no gate reads them". ⟦LIMB⟧`

Residue is the tail of the preceding sentence, and §6.1 restates *"no gate reads them"* verbatim.

**M-3 (`R12:442-445`), one limb removed:**

> `⟦LIMB⟧. Non-blocking: §6.2 is the retirement rationale for GATE-ARMVIAB, a gate that no longer
> exists; every clearance clears; nothing reads the band.`

Residue is the **severity justification**, not a prescription.

**On M-3's treatment — I rule it CORRECT.** Round 12 states no *"Repair:"* sentence for M-3. v13
quotes its prescriptive clause instead and **says so in the limb cell**. The alternative — leaving the
limb column empty, or inventing a Repair sentence round 12 did not write — would be worse on both
the fidelity and the completeness axis. A Minor whose prescription is embedded in its finding sentence
is correctly dispositioned by quoting that sentence and labelling it.

**Residue verdict: clean. Nothing prescribed by round 12 is missing from v13.**

## B.3 Disposition of the four findings at limb level

| finding | prescribed | landed | ruling |
|---|---|---|---|
| **I-1 line 1** | restate the basis; keep `U11`; keep count `1`; print the decomposition; cite `sklearn`'s two entry points | all landed; `1.82–1.85 s` / `≈ 1.7×` gone from §8, §9, §14 except as historical or quotational | **ADOPTED with a stated, warranted DEVIATION on the number** |
| **I-1 line 2** | pin the arena's import set in §13 | §13.1 item 27, list now `1–27` contiguous | **ADOPTED, and wider than prescribed in the right direction** (see C.3) |
| **M-1** | print `98`, or state the convention once | **both**: `98` printed and the closed interval named | **ADOPTED** |
| **M-2** | three words, *"on row-renormalised keys"* | landed verbatim at `v13:952`, plus added context | **ADOPTED**; the added context carries **M-1** below |
| **M-3** | `0.24` upper end, four clearances | `0.15`–`0.24`, all four printed | **ADOPTED** |

**No repair is claimed that the artifact does not contain, and no repair landed narrower than
prescribed.** Two landed *wider* — item 27 and §6.1's parenthetical. The first widening is warranted
and verified (C.3); the second is where my Minor M-1 lives.

---

# PART C — MY OWN VERIFICATION OF ALL TWELVE §3 ITEMS

| # | claim | result |
|---|---|---|
| **V1** | the arena's startup, re-measured by me; **rule the deviation** | **PASS / FORCED.** My 24 runs: **`3.094–3.717 s`** (median `3.172`, mean `3.254`). Inside v13's pooled `3.00–3.75`; `3.8 s` bounds it. Ruling at C.1. |
| **V2** | Phase 1g and the whole re-multiplied column | **PASS.** Re-multiplied independently — C.2. **26 rows**, all **73** processes accounted, **zero stale totals**. |
| **V3** | §13.1 item 27 against source | **PASS** — C.3. Complete for the non-stdlib set; list `1–27` contiguous. |
| **V4** | the five limb quotations, by subtraction | **PASS** — B.1, B.2. **5/5 FAITHFUL.** M-3's treatment ruled correct. |
| **V5** | re-run the audit; byte-compare; verify `+0 chars` by direct diff | **PASS** — A.1, A.2. Exit `0`, byte-identical. The `+0` is verified; the *count* v13 attributes to round 12 is **M-2**. |
| **V6** | break the self-exclusion in the form that bites | **PASS** — A.3. Plain splice vacuous (exit `0`); biting form exits `1`. Both reported. |
| **V7** | the three Minors | **PASS on all three, with one subsidiary defect.** §7.3's `98`: reproduced — see C.4. §6.1's `ρ` on row-renormalised keys: **all six figures reproduce to the digit**, `0/18` both — but two subsidiary numbers do not (**M-1**). §6.2's `0.15`–`0.24`: **exactly reproduced** — C.5. |
| **V8** | rebuild the arms; one bit-exact predicate; the misreading under `2e-6` | **PASS.** All 13 arms rebuilt from §3.4's prose alone: `GATE-C01PARITY` **`0.000e+00`** both datasets; `GATE-ROWSUBSET` bridge **`0.000e+00`**; un-normalised misreading **`1.878e-06` / `1.609e-06`**, both under `2e-6`, reproduced to the digit. `GATE-C01PARITY` states **one** predicate. Algebra guard reproduced at `8.941e-08` (θ=0, both) and `1.192e-07` / `8.941e-08` (θ=45). |
| **V9** | `ρ*`; 26/26 `ρ_raw` at 6 dp; trained-head `0/18` | **PASS.** `ρ* = 0.968176 / 0.977223`; both runners-up reproduce; **26/26** `ρ_raw` at 6 dp under `float64` accumulation; the `1.301e-03` shift from including the masked row reproduces; trained-head **0/18 on both**, min/median/max to the digit. |
| **V10** | Holm counterexample; `n ≤ 12`; §3.7's two blocks | **PASS.** Run through C01's own `holm_adjust`: `24×1/2001` → 24/24 at both `m`; `23×1/2001 + 1×2/2001` → **23/24 at `m=92`**, 24/24 at `m=46`; `24×2/2001` → **0/24 at `m=92`**, 24/24 at `m=46`; `22/22` for the other disjunct; the three-way equality (`m=92` padded `0.5`, padded `1.0`, `m=46`) holds. `1/257 = 0.0038911`; `12/257 = 0.04669 ≤ 0.05`; `13/257 = 0.05058 > 0.05`. §3.7 has two blocks with two distinct verbs, the `<=` operator correctly in the *read* block. |
| **V11** | §7.9's sum | **PASS on the sums.** Heading reads *"v1–v13"*. `7+1+4+0+0+0+0+0+0 = 12` ✓; `22+4+2+1+1+1+1+1 = 33` ✓; `89+21+6+3+3+3+3+3 = 131` ✓. Agrees with §7.8 and the footer. **The run-count inside the v13 term is I-1.** |
| **V12** | §6's 20 rows, `12 G / 6 L / 2 R`; items 10/15/19/22/27; 37 digests | **PASS.** Exactly **20** rows, **12 G / 6 L / 2 R**; the G-set and L-set match §5.6's lists **name for name**, symmetric difference empty both directions. **37/37 digests recompute identically**; **eight** rows carry the `…` ellipsis and all eight resolve to exactly one file. All four new-code paths **absent** from the tree. Items 10, 15, 19, 22 and 27 carry their repairs. |

## C.1 **RULING ON THE DEVIATION: forced by measurement**

I built the rungs myself. Rung 7 replicates `headspace_arena.py:28-46` — `argparse/json/os/sys/time`,
`numpy`, `sklearn.metrics.roc_auc_score`, `sklearn.model_selection.StratifiedKFold`, `mechfix_ops`,
`mechnov_pairverify`, `vsw_pregate`, `from headspace_mint import det1_assert, runtime_block,
sha256_of`, `torch` — plus `c01_policy_contrast_a0` and a `runtime_block()` call, under
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=""`, full process wall via
`date +%s.%N` brackets.

| rung | my runs | my range | v13 |
|---|---|---|---|
| bare interpreter | 4 | `0.0165–0.0190 s` | `0.01 s` |
| `+ numpy` | 4 | `0.1046–0.1104 s` | `0.09–0.10 s` |
| `+ torch` | 4 | `1.790–1.824 s` | `1.79–1.91 s` |
| `+ faiss` | 4 | `1.859–2.011 s` | `1.81–1.84 s` |
| `+ c01 + mechfix_ops` (the set v12 and round 11 timed) | 4 | `1.839–1.922 s` | `1.78–1.91 s` (10 runs) |
| `headspace_arena.py`'s actual top-level set | 16 | `3.080–3.437 s` | `3.02–3.35 s` |
| **the same `+ c01` `+ runtime_block()`** | **24** | **`3.094–3.717 s`** | `3.02–3.75 s` (14 runs) |

**The ruling.** My maximum is `3.717 s`. Round 12's prescription — *"Keep the unit at `U11 = 3.2 s`"*
with *"the residual `≤ 0.2 s`"* — is **falsified by my own sample**, which exceeds `3.2 s` by
`0.517 s` and does so on 10 of 24 runs. Round 12's prescription was exactly right on round 12's own
maximum of `3.27 s`; it is not right on a larger sample, and v13 says so and shows why. Keeping
`3.2 s` would have written a residual bound the document's own measurement refutes and made Phase 1g
the second §8 row carried below its measurement — the first, Phase 1d, being one-decimal rounding
that rounds 8 and 9 ruled acceptable and that I independently confirm is conservative (C.2).

**`3.8 s` is not over-conservative.** It bounds my maximum by `0.083 s`. Had my maximum landed near
round 12's `3.27 s` I would have said so, as the brief requires; it did not. **Forced by measurement.**

I also record that v13 did the *harder* thing: it quoted the prescription it could not follow **in
full and verbatim**, including the two clauses that convict it, and labelled the departure a
deviation in three separate places rather than quietly carrying a different number.

## C.2 §8 re-multiplied, every row, by my own arithmetic

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
Phase 1c 67×0.033 = 2.211 -> 2.2 ; Phase 1f 150×0.0041 = 0.615 ; 150×0.0044 = 0.66
```

**Every figure in V2 is confirmed. §8 has exactly 26 rows.** Every count re-derives independently:
`(30×3)+(6×4)+(30×2) = 174`; `60×2+30 = 150`; `256×3×2×2 = 3072`; `23×2×2 = 92`; `14×3×2×2 = 168`;
`2×3×5×2 = 60`; `4×60 = 240` / `9×60 = 540`; `2×60 = 120`; `2+60 = 62`; `7×6+5×6 = 72` and
`72×2.0e-05 = 0.0014 s`. Population constants re-derive: `446/743 = 0.600269`, `399/579 = 0.689119`,
`446/744 = 0.599462`, caps `⌊0.01×743⌋ = 7` / `⌊0.01×579⌋ = 5`, `0.02×743 = 14.86`,
`0.02×579 = 11.58`, the `(2,21,22)` counterexample mean `15.00`, `20/743 = 2.69 %`, `√2048 = 45.25`,
recovery fraction `0.02/(0.8884−0.6003) = 6.94 %`.

**Stale-total sweep: zero.** `85.6 %` occurs **0** times; `3663.4`, `2953.0`, `3691.3`, `3207.6`,
`4028.7` all **0**. Every occurrence of `2933.9`, `3667.4`, `2930.7`, `2930.4` is inside §8's own
provenance narrative or the header's change summary; every surviving `3.2 s` and `1.7×` is
historical, quotational, or the deviation record. `3667.4` occurs once, in the sentence recording the
move to `3668.1`.

**Phase 1d checked against reality.** I timed `GATE-SHA` over all 37 artifacts (120.8 MB): **`0.094–
0.100 s`** warm, against the frozen `U7 = 0.13 s` carried at `0.1 s`. The sixteen banked artifacts are
`1.21 MB` and hash in `1.2 ms` against the `5 ms` claimed. Both conservative.

## C.3 §13.1 item 27, checked against source as it actually is

`headspace_arena.py:28-46` imports, at top level: `argparse`, `json`, `os`, `sys`, `time`; `numpy`;
`from sklearn.metrics import roc_auc_score`; `from sklearn.model_selection import StratifiedKFold`;
`mechfix_ops`; `mechnov_pairverify`; `vsw_pregate`; `from headspace_mint import det1_assert,
runtime_block, sha256_of`; `torch`. `headspace_mint.py:68` is
`from sklearn.model_selection import StratifiedKFold` at top level — **confirmed**, so `sklearn` is
unavoidable even if the two direct imports were dropped, exactly as item 27 states.
`runtime_block()` (`headspace_mint.py:82-94`) defers `threadpoolctl`, `scipy`, `sklearn` —
**confirmed**.

**Item 27 names every non-stdlib module in the actual set, plus the battery's own addition.** It
omits the five stdlib imports, which the bare-interpreter rung prices at `0.017 s` and which cannot
move the number. It does not name `faiss` explicitly, but names `mechfix_ops`, whose line 29 is
`import faiss` — so a lineage following item 27 gets `faiss` transitively, and §7.7's rung table
prices it separately.

**Ruling: item 27 says enough.** I wrote my rung 7 from item 27 plus the source, and a lineage could
have written it from item 27 alone. The `U11` number is now checkable without reconstructing the set
from three source files, which is what round 12 asked for. **The list is `1–27`, contiguous, verified
by my own scan.** Item 27 is also correctly addressed to the separate code/resource lineage: it binds
a lineage that trims *or extends* the set to re-measure and re-carry Phase 1g.

## C.4 §7.3's `98`, and the blindness grep across v1–v13

Grepping every decimal in the **closed** `[0.6, 0.99]` across all thirteen drafts, under the
convention that a decimal carries a leading digit:

* **v1–v5: `98`** ✓ · **v1–v10: `116`** ✓ · **v1–v12: `118`** ✓ — v13's triple reproduces exactly.
* Excluding the two self-referential endpoint tokens inside the literal `` `[0.6, 0.99]` ``:
  **`96 / 114 / 116`** ✓.
* Under a half-open interval: `97` for v1–v5 and `115` for v1–v10 — confirming round 12's diagnosis
  of where `97` came from and that the closed convention is the one that makes the triple coherent.
* **The new-in-v13 in-band set is EMPTY.** So is the new-in-v12 set, corroborating round 12.

*(A caution for any later round: a regex admitting leading-dot decimals picks up the fragment `.27`
inside the audit's own `defined 1..27` line and returns `119`. That is a grep artifact, not a
decimal in the document. My first pass hit it; the leading-digit convention is the correct one and it
is the one that reproduces.)*

v13's twenty-eight genuinely new decimals anywhere on the number line are `0.011`, `0.04`, `0.07`,
`0.09`, `0.10`, `0.2`, `0.27`, `0.55`, `1.78`, `1.81`, `2.03`, `3.00`, `3.02`, `3.27`, `3.35`,
`3.75`, `3.8` (timings and import-rung bounds), `0.1499`, `0.1755`, `0.2302`, `0.2395`, `0.24`
(M-3's clearances — *differences* of two published figures, and all below the band), and `67.2`,
`2934.5`, `3208.2`, `3668.1`, `4029.3` (second and minute totals). **All timings or arithmetic on
timings. No battery-arm accuracy anywhere in v1–v13**, and I verified this by classification rather
than inheriting it.

## C.5 §6.2's clearance band — exactly reproduced

From `C01_A0_OUT.json` against the arena bars `0.6203` / `0.7091`:
`displacement` HateMM `0.8505 − 0.6203 = 0.2302`; `common_displacement` HateMM
`0.8598 − 0.6203 = 0.2395`; `displacement` ZH `0.8846 − 0.7091 = 0.1755`; `common_displacement` ZH
`0.8590 − 0.7091 = 0.1499`. **The set is exactly `{0.1499, 0.2302, 0.2395, 0.1755}`**, minimum
rounding to `0.15` and maximum to `0.24`. **M-3 is landed correctly.** Every clearance clears, and
`GATE-ARMVIAB`'s escape branch is confirmed unreachable.

I also re-verified from the OUT file, recomputed rather than read: `net_fixes.reference = "common"`
(HateMM) and `"endpoint_concat"` (MHC-ZH) — **not** `endpoint_std`, confirming **D-1**;
`gain_over_strongest_control` accuracy `-0.009345794392523366` → **`−0.0093`**, confirming the
round-5 erratum, and `-0.02564102564102566` → `−0.0256`; `pass: false`; `decision.continue = false`.
**4 of 6** HateMM rotations and **2 of 6** ZH rotations sit below the primary, matching §10.2's
attribution to the OUT file exactly.

## C.6 A candidate finding I searched, measured, and dropped

§7.7's `U11` row reads *"Both are **inside the mint units and inside `U9`**; the arena's is priced
once at §8 Phase 1g"*. Read as *"both **classes**"*, that is false of the arena class and
self-contradictory within its own sentence. **I did not raise it**, because a plainly available
reading is true and is the one the document uses elsewhere: the first class is labelled
*"mint/fidelity"*, and *"the mint units and … `U9`"* are precisely the two enclosures for its two
process kinds — the distributive reading, word for word §7.2's *"already inside every one of the 66
mint units and inside `U9`"*. §7.2 and §8 Phase 1g both state the arena's exclusion unambiguously.
Raising a compressed table cell that has a true reading and is stated correctly twice elsewhere would
be manufacturing a finding, which the brief forbids as squarely as softening one.

## C.7 Verdict-path enumeration — mine, from the document alone

Let `G` = all twelve globals pass; for lineage `L ∈ {N, R}` let `p_L` = passed all six per-lineage
gates **on both datasets** (§5.6's dataset-axis rule), `c_L` = clears S1–S7 on both datasets.

| `G` | `p_N` | `p_R` | outcome | rule |
|---|---|---|---|---|
| fail | any | any | **HALT** `INSTRUMENT_INCONCLUSIVE` | 3 |
| pass | ✓ | ✓ | **SURVIVE** if `c_N ∨ c_R`; else **CLOSE** | 1 / 2 |
| pass | ✓ | dropped | **SURVIVE** if `c_N`; else **HALT** (rule 2 needs *both* passed) | 1 / 3 |
| pass | dropped | ✓ | **SURVIVE** if `c_R`; else **HALT** | 1 / 3 |
| pass | dropped | dropped | **HALT** | 3 |

**Exactly one published state per combination; no unmapped outcome; no overlap.** Rule 3's explicit
*"otherwise"* makes the mapping total by construction. `c_L` is never evaluated for a dropped
lineage.

**The declared-drop exemption is the only lawful absent-quantity path**, stated in terms at
`v13:772`: *"Absence by declared drop is lawful; absence by computation failure in a surviving
lineage still HALTs."*

**No gate failure is reportable as a closure.** CLOSE requires all twelve globals to pass **and** both
lineages to have passed every per-lineage gate on **both** datasets. A global failure HALTs; a
per-lineage failure drops that lineage on both datasets, falsifying rule 2's conjunct, so the only
reachable outcomes are SURVIVE-on-the-clean-lineage or HALT. **A CLOSE always rests on two clean
negatives, never one.**

## C.8 Freeze-readiness, operationally

Judged as the document an operator with no context would execute.

* **No decision point on the run boundary.** One `sbatch`, 8 CPU / 32 GB, no `--gres`, no `--time`,
  no array, no dependency, no requeue. The 73-process order is stated — `66 mints → 6 fidelity →
  1 arena` — with `GATE-SHA` once in the driver before any of them and `GATE-POP` before any
  population-consuming gate.
* **Preconditions are checkable.** 37/37 digests recompute; all four new-code paths absent;
  `mints_present_before_arena` is a declared binding predicate.
* **Per-class import accounting is consistent.** 66 mints carry startup inside their full-process
  walls; 6 fidelity inside `U9`; the 73rd, the arena, priced once at Phase 1g. `66 + 6 + 1 = 73`.
  I found no seventy-fourth python process.
* **Heartbeat.** Line-buffered `buffering=1` handle, per-phase and per-cell granularity, plus an
  unbuffered bash echo per mint; longest un-instrumented span `11.27 s` (`14.1 s` conservative)
  against a `~15 s` bound. **The corrected arena startup of `3.00–3.75 s` leaves `~11 s` of headroom
  and changes no interval.** `rule_2_heartbeat` unchanged and satisfied.
* **Exit and resume semantics defined.** HALT names the failing gate in its final line; a
  `RuntimeError` from the imported C01 algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE` with its
  `context` string, and written before exit. The refusal to make `dev_path_opens == 66` binding is
  correct and would otherwise HALT a legitimate resume.
* **Test-split non-contact by construction, verified at source.** `headspace_mint.py:106-116`
  replaces `torch.load` with a guard asserting `"test_seen" not in s and "/test" not in s`;
  `c09_guard/sitecustomize.py` imports `c09guard` and calls `install()`, which rebinds
  `builtins.open` at interpreter startup in every process; `GATE-LEDGER` binds `test_path_opens == 0`
  and `test_label_materialisations == 0`. §3.1 states no `dev_seen_*-ro_*` and no `test_seen` ro cache
  is opened by any phase.
* **The `$0` character holds.** No GPU, no Modal, no test contact, no new data.

Every source citation I checked resolves as stated: `headspace_mint.py:192-194` (early return),
`:199` (unconditional native `dev_seen` load), `:203-216` (fold parity vs banked `vsw_ckpt`),
`:321-325` (`np.savez` with `lab_dev` at `:323`); `K_FOLDS = 5`, `FOLD_SEED = 0`;
`classifier.py:81-82` (default-bias projections, `:80` the comment), `:140-141`
(`torch.mul` under `align`), `:146` (`embed = self.mlp[:-2](x)`); `c01_policy_contrast_a0.py:1725`
(reporting `fix_break`), `:2702-2714` (decision `net_fixes` via
`select_strongest_ordinary_control`), `:2724` (the consistency `die`), `:1940-1948` / `:1955-1962`
(guards and the `(accuracy, macro_f1, −index)` ranking), `:2036` (`small_mask = dev_min <= threshold`),
`:1989-1996` (zero-fix convention), `:1372-1377` (algebra guard at `2e-6`);
`headspace_arena.py:75-89`, `:72`; `headspace_fidelity.py:31/:33/:66` (no `dev_seen` at all);
`mechfix_ops.py:94`. §1's two verbatim quotations match
`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]` exactly, and `new_status` is
`gated_on_zero_cost_falsifier`. F118's erratum lesson is as §2 describes.

## C.9 `rule_1_compute_projection` — the axes I searched

**I name two, and both return nothing.**

1. **The `GATE-SHA` hashing cost.** Phase 1d prices `U7` over 37 artifacts at `0.1 s`, and six of
   those artifacts are 16–21 MB caches. If `U7` had been timed over the modules only, Phase 1d would
   be understated by seconds. **Measured: `0.094–0.100 s` for all 37 (120.8 MB), three trials.** The
   frozen `0.13 s` bounds it; the carried `0.1 s` is one-decimal rounding of a bound. **No item.**
2. **The vote units' timed region — whether `U2a`–`U2d` enclose the faiss index construction.**
   §7.7 lists all four as *uncorroborated*, and they price 930 votes across Phases 2, 2z and 2R. If
   the timed region were search-and-vote only, every one would carry an unpriced
   `IndexFlatIP` build and `add`. **It does not arise:** `mechfix_ops.deployed_vote` calls
   `_flat_ip`, which constructs the index and adds the bank **inside** the function (`:45-49`, called
   at `:82`), so any timing of a vote necessarily encloses it. I then measured all four on realistic
   fold shapes (bank 594, queries 149): `U2a` **`0.00292 s`** vs frozen `0.00305`; `U2b`
   **`0.00542 s`** vs `0.00629`; `U2c` **`0.01995 s`** vs `0.04239`; `U2d` **`0.05738 s`** vs
   `0.08098`. **All four frozen units bound my independent measurement including index
   construction** — so §7.7's four uncorroborated units are now corroborated and conservative.
   **No item.**

**No eleventh uncounted item on either axis.** Thirteen rounds, ten items. I note that the last two
findings were a per-process fixed cost and a per-process *unit basis* — neither a payload loop — and
that mine is a *sample size*, continuing that drift away from loops and toward the evidentiary
statements around them.

## C.10 §4.C — can any gate fire on a warranted CLOSE? **No, for all twenty**

A *warranted CLOSE*: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction.** `GATE-DET1` (thread env before
any python starts). `GATE-SHA` (37 digests over files no phase writes — **all 37 recomputed by me**).
`GATE-FOLD` (banked parity flags + `fold_of`; the assertion is at `headspace_mint.py:203-216` and a
`.npz` is written only after it passes). `GATE-FLOOR` (six banked anchors reproduced on **native**
keys, so no ro-derived arm outcome can reach it). `GATE-POP` (populations `743/579`, class counts
`(297,446)` / `(180,399)`, index-set identity, constants recomputed — all verified). `GATE-C01PARITY`
(a property of the **builder**; I measured `0.000e+00`). `GATE-ROWSUBSET` (builder property;
`0.000e+00`). `GATE-RHORAW` (a property of the ro caches and the raw leg, identical for both
lineages; **26/26 reproduced at 6 dp**). `GATE-NULLREMOVED` / `GATE-ZEROMASK` (`{355}` / `{}`,
verified as the sole exact-zero row). `GATE-IDPARITY` (ids/labels parity). `GATE-LEDGER` (declared
counts). **None reads which arm won.**

**The six per-lineage gates.**

* **`GATE-ARENA`.** Its **lower** bound is on `endpoint_std` **only** — the reference arm, never a
  real-vs-rotation quantity. A warranted CLOSE says nothing about `endpoint_std`; if that arm cannot
  clear `majority + 0.02` the instrument is genuinely dead. Its **upper** bound `≤ 0.98` fires only
  on implausibly high accuracy and cannot fire downward.
* **`GATE-ORBITDISP`.** Fires iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D` — head space *more degenerate than
  the raw family*. I measured trained deployed heads at roughly **half** the bar, `0/18` on both.
  Arm-outcome-independent.
* **`GATE-NESTED`.** The scoring head excluded its fold. Structural.
* **`GATE-SELFTEST`.** `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is an **identity**; it holds
  whatever the accuracies are.
* **`GATE-ZEROOP`.** `orthrot_0 ≡ endpoint_concat` and `orthrot_45 ≡ common_displacement` are
  **algebraic identities of the Givens family** — I measured the residuals at `8.941e-08` and
  `1.192e-07` / `8.941e-08`. Its tie cap is **one-directional** (REPORT → HALT only), so it can cause
  non-publication, never a wrong verdict.
* **`GATE-ALGEBRA`.** Key-level `≤ 2e-6` on the same two identities, with `7.5×`–`22.6×` measured
  head-space headroom. Same argument.

**The two `R` gates** (`GATE-DOMAIN`, `GATE-DEVFID`) carry **no bar** and cannot fire at all.

**All twenty: no gate can fire on a warranted CLOSE.** The load-bearing structural fact is §6.2's
retirement of `GATE-ARMVIAB`: I confirmed its escape branch unreachable (all four raw clearances
`0.1499`–`0.2395` clear), which had reduced it to a one-sided HALT on precisely the warranted-CLOSE
outcome. With it deleted, **no lower-bound instrument HALT is applied to a real arm anywhere in this
design** — I checked the §6 table row by row, and `GATE-ARENA`'s lower bound on `endpoint_std` is the
only lower accuracy bound in the document.

---

# FINDINGS

## CRITICAL — none

## HIGH — none

## IMPORTANT

### I-1. §7.9's and the footer's *"`24` timed interpreter starts in total"* contradicts §7.7's own decomposition table, which specifies **44**, and does not reconcile with its own stated breakdown; and the *"24 timed runs by two parties"* on which §7.7 and §9 rest the arena-class range does not reconcile with round 12's documented run counts.

*Attaches to:* §7.9's v13 term (`v13:1358-1360`); the footer (`v13:2270-2271`); §7.3
(`v13:1154`); §7.7's `U11` row (`v13:1209`); §9's arena-startup clause (`v13:1482`); §14's I-1
disposition row (`v13:1855`).

**What §7.7 specifies.** *"this document reproduces the decomposition on the same node, `4` runs per
rung unless stated"* (`v13:1220-1221`), over a **seven**-rung table in which rung 5 is stated at
`(10 runs)` (`v13:1229`) and rung 7 at `(14 runs)` (`v13:1231`). That is
`5 × 4 + 10 + 14 = ` **44** timed interpreter starts.

**What §7.9 and the footer report.** *"the seven-rung arena import decomposition of §7.7 (`24` timed
interpreter starts **in total** — `4` per rung and `10` **each** for the two rungs the finding turns
on)"*, and *"the `24` interpreter starts of §7.7's seven-rung arena import decomposition"*.

**Three separate failures to reconcile.**

1. **The total.** `24` against §7.7's `44`.
2. **The per-rung figure.** *"`10` each for the two rungs"* against §7.7's `10` **and** `14` — and
   §7.7 states the `14` twice more, at `v13:1233` (*"Pooling this document's 14 observations"*) and
   `v13:1864` (*"Fourteen timed runs of the arena's actual set here"*), the second being the
   deviation's own warrant sentence.
3. **Internal incoherence.** The stated breakdown does not produce the stated total under any
   reading: `5 × 4 + 2 × 10 = 40`, `7 × 4 = 28`, and no assignment of `4`/`10` across seven rungs
   yields `24`. The one arithmetic that *does* give `24` is `10 + 14` — the two decisive rungs alone
   — which is exactly what *"in total"* denies.

**The second, independent limb.** §7.7's `U11` row and §9 both state the arena-class range as
*"`3.00–3.75 s` over 24 timed runs by two parties"*. This document contributes 14, so round 12 must
have contributed 10. Round 12 documents *"three runs per rung"* (`R12:337-349`) across eight rungs;
its sklearn-bearing rungs number three, giving 9, and the single rung that is the true arena class
including `runtime_block()` gives 3. **`14 + 3 = 17` or `14 + 9 = 23`, not 24.** Round 12's own
*total* across all eight rungs is `8 × 3 = 24`, which is the most likely provenance of the figure and
is a different quantity entirely.

**What is and is not wrong.** The **range is right** and I corroborate it independently: round 12's
minimum `3.00` and this document's maximum `3.75` are both real, my own 24 runs land at
`3.094–3.717 s` inside them, and `3.8 s` bounds everything measured by all three parties. **No
verdict quantity moves, no gate is touched, no heartbeat interval changes, and the deviation of C.1
stands on its merits.** What is wrong is the **stated sample size** — in a paragraph whose entire
purpose is to make the dry-check spend checkable, and beneath a row whose sample size is the warrant
for departing from a prior review's explicit prescription.

**On severity, stated so the grade is auditable.** Not **Critical**: no §8 loop is uncounted, Phase
1g's count and unit are right, and the carried number is independently corroborated. Not **High**:
round 12's prescription is landed in full and wider rather than narrower, and the verdict's authority
and scope are untouched. **Important** is the grade the brief defines for completeness and
reproducibility, and it is the grade rounds 7–12 gave the same family — a stated evidentiary basis
that does not survive being checked. §8's own institutionalised lesson at `v13:1436` is *"state the
timing boundary, not just the number"*; the sibling obligation, and the one round 12's *"agreement
between parties who made the same omission is not corroboration"* turns on, is **state the sample
correctly, because the sample is what makes a larger measurement authoritative over a smaller one.**
It also has a small knock-on: at 44 starts the decomposition alone costs `≈ 1.5` wall-minutes against
§7.9's *"≈ 1 wall-minute"* for the round's whole work, though the `≈`-rounded cumulative `33` / `131`
is not materially affected and both sums re-derive.

**Repair — one line, arithmetic only, no new measurement.** Reconcile the counts against §7.7's table
and say which quantity each figure is. Concretely: in §7.9 and the footer, either print **`44`** for
this document's decomposition, or scope the `24` to what it actually counts —
*"`24` timed starts on the two rungs the finding turns on (`10` and `14`), out of `44` across the
seven rungs"* — and correct *"`10` each"* to *"`10` and `14`"*. In §7.7's `U11` row and §9, replace
*"24 timed runs by two parties"* with the count that reconciles with round 12's stated
*"three runs per rung"*, or state the pooled figure as *"14 here, pooled with round 12's"* without a
total. §7.3's *"`24` … timings"* follows whichever is chosen.

---

## MINOR (non-blocking; neither touches the verdict path)

* **M-1. §6.1's two subsidiary numbers in the M-2 repair do not reproduce** (`v13:956-957`).
  The parenthetical states *"measured row norms run `0.04`–`0.27`"* and that the literal reading gives
  values *"`5`–`10×` lower"*. Measured over all 36 banked `K_train` matrices: **row norms span
  `0.0271`–`0.5596`** (HateMM `0.0410`–`0.5596`, MHC-ZH `0.0271`–`0.2882`), and the gap factors are
  **`3.47×`–`7.64×`** (`7.64 / 6.21 / 3.47` HateMM, `7.22 / 6.39 / 4.74` ZH). I tested every natural
  subset — fold-only, full-only, per-dataset, per-cell median, per-cell mean, native dtype — and
  **none yields `0.04`–`0.27`**; the closest is HateMM's global minimum `0.0410`, whose companion
  maximum is `0.5596`. The stated range understates the top by `2×` and excludes real rows below its
  floor. **Non-blocking, and emphatically so.** The prescribed repair is landed verbatim, and
  everything load-bearing reproduces exactly under it: all six trained-head `ρ` figures
  (`0.447803 / 0.562434 / 0.632996` and `0.340179 / 0.574247 / 0.667326`) to the digit, and **`0/18`
  above `ρ*` on both datasets**. §11 declares these 36 mints digest-free *"inputs to this reference
  measurement only — no gate reads them"*, which I verified: `ρ*` and the 26 `ρ_raw` come from the
  ro caches and the raw leg, which I rebuilt independently. Nothing reads the row-norm range or the
  gap factor. This is the same class round 12 graded Minor for §6.2's band — *"nothing reads the
  band"*. **Repair:** print `0.027`–`0.56` and `3.5×`–`7.6×`, or drop both figures and keep the
  qualitative claim, which is what the repair needs.

* **M-2. §14.1's *"exactly five same-length version-label substitutions"* is six** (`v13:1972`).
  The brief asked me to verify the `+0 chars` line by direct diff rather than accept the explanation.
  I did, both ways: `len(v11 §14.2) = len(v12 §14.2)` with `identical = False` and **six** differing
  lines, and the same for v12→v13. Round 12 wrote *"exactly five"* while displaying six in its own
  block, and v13 restates the five as an established fact. **Non-blocking:** the `CHANGED` verdict is
  computed from a content comparison, never from size; the transcript is a verified byte-identical
  fixed point at exit `0`; and v13's account of **its own** §14.2 change is accurate — it names four
  substitution *classes*, and those four classes are exactly the four that occur and do cover all six
  lines. **Repair:** print `six`, or say *"four substitution classes"* rather than a line count.

---

# REQUIRED RULINGS

## 1. §4.C — any gate that can fire on a warranted CLOSE? **No, for all twenty.** Derived at C.10 from the gate texts and my own measurements, not inherited.

## 2. Verdict-path enumeration: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.** C.7.

## 3. The deviation (§4.A): **FORCED BY MEASUREMENT.** My own range is `3.094–3.717 s` over 24 runs; round 12's prescribed residual `≤ 0.2 s` is falsified by `0.517 s`; `3.8 s` bounds my sample. C.1.

## 4. Rulings on §15's six open issues

1. **The deviation.** **Forced by measurement, not a substitution.** My maximum did *not* land near
   round 12's — it landed at `3.717 s`, within `0.033 s` of v13's. `3.8 s` is **not** over-conservative;
   it has `0.083 s` of headroom over my slowest observation. C.1.
2. **The subtraction, on five limbs.** Executed — **5/5 FAITHFUL**, residues bare. **M-3's treatment
   is the right one**: quoting the prescriptive clause and saying it is not a *"Repair:"* sentence is
   better than an empty limb column or an invented prescription. B.1, B.2.
3. **The self-exclusion, in the biting form.** Both run and both reported: the plain splice is
   **vacuous against v13** (exit `0`), and the biting construction fails one row and exits `1`. A.3.
4. **The eleventh uncounted item.** **Two axes named and searched — `GATE-SHA`'s hashing cost and the
   vote units' timed region — and neither yields.** Both are now measured and both frozen units bound
   my measurements. §8 has **26** rows, accounts for all **73** processes, and re-multiplies to
   `2934.5`. **No eleventh uncounted item.** C.9.
5. **§13.1 item 27, and whether it is checkable.** **It says enough.** It names every non-stdlib
   module in `headspace_arena.py`'s actual set plus the battery's addition and the `runtime_block()`
   call; `faiss` comes transitively through the named `mechfix_ops`; the omitted stdlib imports are
   priced at `0.017 s` by the bare-interpreter rung. The list is `1–27` and **contiguous**. C.3.
6. **Record sound, design freeze-ready?** **The record is sound** — nothing prescribed is missing,
   nothing claimed is absent, nothing narrowed, zero stale totals, 5/5 limbs faithful. **The design is
   freeze-ready on everything except I-1**, and I-1 is an arithmetic reconciliation of stated run
   counts that requires no new measurement and moves no quantity a reader would act on.

## 5. Process rules

* **`rule_1_compute_projection`.** Satisfied in form — every §8 row is a measured unit × an explicit
  count, re-multiplying exactly to `2934.5`, with no extrapolation from a reduced-scale dry run. Two
  axes searched for an eleventh item; neither yields. **The one row whose stated evidentiary sample
  does not reconcile is Phase 1g's, and that is I-1** — a defect in the reported sample size, not in
  the unit, the count, or the product.
* **`rule_2_heartbeat`.** **Unchanged and satisfied.** Line-buffered `buffering=1` appends plus an
  unbuffered bash echo per mint; progress path stated in full; longest un-instrumented span `11.27 s`
  (`14.1 s` conservative) against `~15 s`. The corrected arena startup of `3.00–3.75 s` — and my own
  maximum of `3.717 s` — leaves `~11 s` of headroom. **v13 changes no interval.**

## 6. Can the falsifier discharge the written condition at `$0`? **Yes.**

Every input exists and is digest-frozen (37/37); the head space is re-mintable on CPU at measured
cost; the arms rebuild bit-exactly from the document's own prose; the decision rule is pre-registered
with its multiplicity resolution floor proved attainable through C01's own `holm_adjust`; the verdict
combination is total with exactly one lawful absence path; and no gate can fire on a warranted CLOSE.
**Neither I-1 nor either Minor bears on this.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Nothing here authorizes execution. Before any job: (1) freeze with hashes; (2) a **separate**,
independent code/resource review lineage over the executable reaching its own `0C/0H/0I` — and I note
for that lineage that §13 item 23 should be read broadly, that item 27 is addressed to it and binds
any trimming or extension of the import set to a re-measurement, and that item 22's placement of the
`GATE-FLOOR` vote in the arena process is what Phase 1f's `150` is priced against; (3) main-dialogue
authorization. This document is not authority to write `TARGET_STATE.json`.

---

# CLOSING

**The most severe finding is I-1, and it is a sample size, not a number.** v13 was asked to justify
departing from an explicit prescription, and it did so on the strength of having measured more than
round 12 did. I re-ran that measurement independently and **v13 is right** — my 24 runs reach
`3.717 s`, ten of them exceed the prescribed `3.2 s`, and round 12's *"residual `≤ 0.2 s"`* cannot
survive either my sample or v13's. The deviation is forced, disclosed three times, and carried in the
conservative direction. But the document reports the sample that forces it as `24` *"in total"* where
its own table specifies `44`, as *"`10` each"* where its own table says `10` and `14`, and as
*"24 timed runs by two parties"* where round 12 documents three runs per rung. The breakdown does not
even sum to its own total. **When the warrant for overriding a reviewer is "I measured more than you
did," the count is load-bearing**, and here it is the one quantity in the repair that does not survive
being checked — the fourth consecutive round in which the finding lives inside the previous round's
repair, and on the very axis that repair created.

**Everything else is clean, and I say so as plainly as the brief asks.** The science reproduces on
every axis I tested, independently and at full precision: 13/13 arms at `0.000e+00` from the prose
alone, 26/26 `ρ` at 6 dp, 37/37 digests, 16/16 accuracies and net-fix integers, the Holm
counterexample through C01's own code, `0/18` trained heads, and both of round 12's Minors landing
correctly on their load-bearing content. The record is faithful at limb level with nothing in the
residue, and v13 quoted verbatim the prescription its own deviation contradicts — which is the
behaviour that made this round decidable from the artifact. All twenty gates are unable to fire on a
warranted CLOSE; the verdict path is total, mutually exclusive and admits exactly one lawful absence;
two newly-searched cost axes return nothing and incidentally corroborate four previously
uncorroborated units. **v13 is one arithmetic reconciliation away from a GO, and I have declined to
grant it early for the same reason round 12 declined — and I have declined to invent a second finding
for the same reason, dropping one candidate at C.6 after measuring it.**
