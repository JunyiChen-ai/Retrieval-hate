# C06 `$0` falsifier — ERRATUM 2, PROPOSAL v7: ADJUDICATION (round 7)

**Target:** `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V7.md`,
sha256 `9576da0d12ce3e2a2cdcda4d57116dcf50a2e2b2bf31fc0c77feb54b97b3f262`, 983 lines.
**Against:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`,
sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
**Reviewer:** fresh, no part in rounds 1–6. Judged from documents and repository only.

---

## VERDICT

> ## REVISE — 0C / 2H / 3I / 5M

The re-keying works. I attacked the quantity list and the form list as instructed — I invented six
forms outside the nine and swept every one of them over all five artifacts, and I derived the moved
set independently from §9's delta — and the 10×9 table survives both attacks: **no site anywhere in
the five artifacts states a moved quantity in a form the table does not cover and no sweep reaches.**
The twelve sweeps reproduce byte-identically including sort order, the partition
`305 → 267 = 137 + 130`, `UNCHARGED = 0` recomputes exactly from the raw lists, all twelve
subtractions balance, there are zero divergent cross-sweep charges, and all 71 rows appear exactly
once in §3 and exactly once in §9. Rows 67, 68 and 31 are right and land consistently with every
other inventory statement. Round 6's H-1 and I-1 are fully discharged.

What blocks GO is not the completeness domain. It is that **two of the numbers the erratum would land
are wrong, and one of them is the number this erratum exists to make consistent.** §8's Phase 1d
re-price doubles an already-rounded product instead of recomputing it, so the prescribed total
`3673.9` — the heartbeat denominator §7 makes a single source and asserts three ways under HALT — is
low by `0.1 s` against §8's own product column. And row 48 lost, between v6 and v7, the clause that
made the `C06_MINTS_EXECUTED` export resume-safe; the sbatch's only mint counter is the wrong one,
and exporting it would HALT a legitimate resume — the precise self-defeating gate §12's warrant
exists to prevent.

Neither is a completeness failure. Both are inside rows this lineage has carried since v3 and v5 and
that rounds 3–6 each endorsed. The instrument v7 built is sound; what it measures was not re-derived.

---

## 1. PROCESS INTEGRITY — CLEAN

All five artifacts carry their post-CODE-R1 hashes, re-verified before and after my sweeps:

| path | sha256 | matches §11 |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aa…0d58f7d` | yes |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0…2db0742` | yes |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6…9b4ad7f9` | yes |
| `configs/c06/c06_falsifier.json` | `e2678431…41e5adeb` | yes |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173…6fa2fc4d` | yes |

`…_PROPOSAL.md` through `…_V6.md` all match the digests v7's header cites, and every mtime precedes
v7's. **Byte-unmodified confirmed.** `artifacts/c06_falsifier/` is absent. No arena run, no mint, no
`--gate-sha-only` leg, no job, no commit, nothing edited outside my scratchpad.

---

## 2. THE TWELVE SWEEPS — twelve for twelve, byte-identical

I ran all twelve printed commands against the same `$F` list and compared **position by position**,
not as sets:

| sweep | hits | order-identical | subtraction |
|---|---|---|---|
| A | 18 | yes | `18 = 11 + 7` ✓ |
| B | 3 | yes | `3 = 1 + 2` ✓ |
| C | 38 | yes | `38 = 15 + 23` ✓ |
| D | 15 | yes | `15 = 11 + 4` ✓ |
| E | 22 | yes | `22 = 10 + 12` ✓ |
| F | 25 | yes | `25 = 15 + 10` ✓ |
| G | 82 | yes | `82 = 56 + 26` ✓ |
| H | 29 | yes | `29 = 13 + 16` ✓ |
| I | 16 | yes | `16 = 8 + 8` ✓ |
| J | 32 | yes | `32 = 10 + 22` ✓ |
| **K** | 16 | yes | `16 = 9 + 7` ✓ |
| **L** | 9 | yes | `9 = 6 + 3` ✓ |

Every hit-table index runs `1..n` with no gaps. The two new sweeps are real instruments, not
decoration: **K** returns `V15E1:1814` (row 67), `config:42` (row 68) and the three sbatch banners
(row 69) exactly as claimed, and **L** returns nine hits of which the two uncharged spelled-out ones
are `V15E1:613` and `:1405`, both *"sixty"* meaning the 60 `ρ` cells, as §1 states.

---

## 3. GLOBAL PARTITION — recomputed from the raw lists

| quantity | v7 declares | I compute |
|---|---|---|
| hit-instances across the twelve sweeps | 305 | **305** ✓ |
| distinct sites | 267 | **267** ✓ |
| excess | 38 | **38** ✓ |
| sites returned by more than one sweep | 32 | **32** ✓ |
| sites charged to a row | 137 | **137** ✓ |
| sites charged to a declaration | 130 | **130** ✓ |
| `UNCHARGED` | 0 | **0** ✓ |

`267 = 137 + 130` closes in both directions. **Zero divergent cross-sweep charges**: no site is
charged to two different rows, and no site is charged to a row by one sweep and to a declaration by
another. 70 distinct rows are reached by a sweep; row 48 is the only siteless one; `70 + 1 = 71`.

**Row set.** §3 contains rows `1`–`70` plus `26†`, each exactly once, no duplicates, none missing.
§9's two columns cover the same 71 rows with no row absent — round-6 M-5 is discharged at the level
of coverage (but see **M-1** for its arithmetic).

**Multi-line rows and declared==computed.** Every `(**N lines**)` label matches its computed charged
count except at exactly the five extent-only lines §10 names, plus one it does not (see **I-1**).
Row 41's label `(**5 lines**)` matches its 5 charged sites; row 43's `(**7 lines**)` matches 7; row
39's `(**17 charged lines**)` matches **neither** its extent (18) nor its charged count (16).

---

## 4. THE QUANTITY-LIST ATTACK — no eleventh quantity, but the domain under-describes the erratum

I derived the moved set myself from §3's subsection structure and §9's delta rather than reading
§1's table. Every one of the 71 rows is assigned to a quantity, and the assignment partitions:
`13 + 12 + 8 + 8 + 6 + 15 + 6 + 3 = 71`. **I found no edit whose quantity is absent from Q1–Q10.**

But four rows do not move a *quantity* at all, and §1's framing — *"the erratum moves ten
quantities"* — does not describe them:

* rows **43** (`:1810` limb) and **66** correct a **mechanism** claim: *"`headspace_fidelity.py`
  reads `lab_dev` out of the banked mint `.npz`"* → *"reads only `meta`"*. Not a count.
* row **39**'s docstring limb corrects an **authority** claim: *"IS NOT ADJUSTED HERE"*,
  *"implemented exactly as frozen"*, *"is not this lineage's call"*.
* row **19** deletes a **mechanism** claim: *"the denominator is pinned to §8 by name, so it tracks
  automatically."*

So I checked each family by direct sweep instead of trusting the table:

* `lab_dev|label array|reads .*npz|out of the banked` over all five artifacts returns **6** sites.
  The two false ones are exactly `V15E1:1810` and `sbatch:103`, both charged (rows 43, 66). The other
  four — `V15E1:224`, `:1826`, `mint:31`, `:336` — describe `headspace_mint`/`c06_falsifier_mint`
  **writing** `lab_dev`, which is true and unchanged. **No third site.**
* the authority family is swept explicitly by G
  (`NOT ADJUSTED|BLOCKED ON|blocked_on|implemented exactly as frozen|lineage.s call|ERRATUM REQUIRED`)
  — and this is where the domain's blind spot bites. See **I-1**: `arena:474` states the same policy
  in a **string literal**, lowercase, in words G's pattern cannot match.

**Verdict on the quantity list: no eleventh quantity.** The gap is that a *fact* is not a *quantity*,
and the one uncharged consequence of that gap is `arena:474`.

---

## 5. THE FORM-LIST ATTACK — six invented forms, all swept, one hit

I invented forms outside the nine and swept each over all five artifacts.

**X1 — pronoun / anaphoric reference with a numeric antecedent.**
`all of them|each of them|both of them|the remaining|the rest of|the others|every one of them|all four|all three|these processes|those processes` → 16 hits. **None** states the process inventory or any
other moved quantity: they are round counts, limb counts, cache counts and draft-version counts
(`:24` *"All fourteen remain on disk"*, `:109/:148/:196/:361/:423/:1527/:1705/:1791` *"all four
rounds"*, `:416/:725/:1526/:1952/:2211` *"all three"*). The two live anaphoric statements of the
inventory — `V15E1:1840` *"before any of them"* and `sbatch:62` *"before any other process"* — are
already charged as rows 29 and 31. **Clean.**

**X2 — arithmetic expression stating the inventory.**
`6[0-9] ?\+|\+ ?1\b|= ?7[234]\b|7[234] ?=|1 ?\+ ?66|66 ?\+ ?6` → the only inventory arithmetic in the
corpus is `V15E1:1550`'s `66 + 6 + 1 = 73` (row 4, charged) and `V15E1:1814` (row 67, charged).
Everything else is p-value forms `(256 + 1)`, loop indices, `MINT_N+1`, and drafting-cost sums.
**Clean.**

**X3 — singular or alternative decomposition wording.** Sweep K's `66 ?(mints|\+)` cannot match the
singular *"66 mint"*, and four such sites exist — `V15E1:1195`, `:1621`, `sbatch:68`, `:106` — but
every one counts `.npz` files, mint *units* or head configurations, **not processes**. `six fidelity|one arena|6 full heads|the six` adds nothing new. **The pattern is narrower than the form it
claims to cover, but nothing hides in the gap.**

**X4 — percentages of the inventory.** None exist. The only percentages in the corpus are budget
shares (`68.3`, `27.6`, `85.6`, `9.3`) and float statistics (`38.8`), all inside sweep F or declared.
**Clean.**

**X5 — spaced identifiers beyond `processes reporting`.** This is the form sweep K was built for, and
K hard-codes the single string `processes? reporting`, so the F6 column is checked for one quantity
rather than nine. I swept the spaced form of every ledger key myself
(`dev path|test path|process(es) reporting|mints executed|banked trainlog|predicate coverage|label materialisation|design sha|projected seconds|gate sha` in table position) across all five artifacts:
**three hits.** §12's `GATE-LEDGER` table has exactly two unbackticked prose keys — `:1814`
(row 67 ✓) and `:1815` *"predicate coverage | re-derived in-job | reported"*, which is true at 74 and
needs no edit — plus `:1273`, a read-map row that is correct as written. **Clean, but by luck rather
than by pattern.**

**X6 — code comment vs string literal vs identifier as distinct carriers.** This is the one that
pays. Sweep G's authority terms are keyed on the **docstring**'s uppercase `NOT ADJUSTED` and on
`implemented exactly as frozen`. `arena:474` carries the same policy inside a **format-string
literal**, lowercase and reworded: `"implemented as frozen and is NOT adjusted here."`. Neither term
matches. See **I-1**.

---

## 6. EMPTY-CELL VERIFICATION — including the Q1×F8 challenge

**Q1 × F8, scrutinised as instructed.** §1 asserts the cell is empty *"because the process inventory
has no alias and no superseded value: it has been `73` since v1."* Under v7's own F8 definition — *"a
paraphrase, or a prior value of the same quantity"* — the cell is **correctly empty in the pre-landing
corpus**. `73` is the *current* value, not a superseded one, and it is already the key of sweep A, so
F8 has nothing to add for Q1 that A does not reach. I looked for a superseded inventory value (any
provenance line of the form *"N → 73"*) and there is none: sweep A's ten V15E1 hits are five charged
rows plus five declared non-targets (two line citations `:73-89`, two rotation angles `72.7`, one
tie-cap product `7×6+5×6=72`), and no historical process count exists anywhere in the five artifacts.
I also swept the paraphrase forms — `process count|number of processes|process total|total processes|process inventory|reporting processes` — and every hit (`V15E1:849`, `:975`, `arena:17`,
`:429`) states *that* the count binds without stating its **value**, so all are true at 74 and none is
a site. **The claim holds.** (It will stop holding the moment V15E2 is written, since V15E2 must carry
the `73 → 74` provenance — a point for the next erratum, not this one.)

**The other empty cells.** §1 promises that *"every EMPTY cell is an explicit claim with a stated
reason."* Four reasons are stated: F3 empty for Q5–Q10; F2 empty for Q7–Q10; F9 empty for Q5–Q10;
F8 empty for Q1. Those cover 4 groups. **Twenty empty cells have no stated reason at all**:

`Q2×{F3,F6,F7}`, `Q3×{F3,F6,F8}`, `Q4×{F2,F6,F8}`, `Q5×{F5,F6}`, `Q6×F6`, `Q7×{F6,F7}`,
`Q8×{F6,F7}`, `Q9×F7`, `Q10×{F6,F7}`.

Note that Q2's and Q3's F3 cells are empty for the reason §1 gives about ordinals, but the sentence
scopes it to *"Q5–Q10"* and therefore does not reach them; the same for Q4's F2 against the *"Q7–Q10"*
sentence. I swept the most exposed of these myself (X5 for the F6 column, X4 for F7 on Q7, and the
superseded-value form for Q3's F8 against `\b21\b|\b22\b|twenty-one|twenty-two` in digest context,
which returns only `config:253` and `:255`, both charged to row 49) and **found no falsifying site**.
The table survives; the promise does not. See **M-5**.

---

## 7. ROWS 67 / 68 / 31 — right, and consistent with everything downstream

I checked the three prescribed texts against every other inventory statement the erratum leaves
behind, post-landing:

| site | after landing | order | total |
|---|---|---|---|
| `V15E1:1814` (row 67) | `1 GATE-SHA driver leg + 66 mints + 6 fidelity + 1 arena` | driver first | 74 |
| `config:42` (row 68) | `1 gate-sha driver leg -> 66 mints -> 6 fidelity -> 1 arena` | driver first | 74 |
| `V15E1:1839` (row 6) | `74 processes in the order 1 GATE-SHA driver leg → 66 mints → …` | driver first | 74 |
| `sbatch:16` (row 9) | `74 processes, in this order: 1 GATE-SHA driver leg -> …` | driver first | 74 |
| `config:41` (row 10) | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | driver first | 74 |
| `config:222` (row 11) | `{"expected":74,"decomposition":"1+66+6+1",…}` | driver first | 74 |
| `V15E1:1197-1199` (rows 1, 2) | `72` unchanged + *"the remaining two processes"* | — | `72+2=74` |
| `arena:465-467` (row 12) | `!= 74`, read from `cfg["ledger"]` | — | 74 |
| `arena:449` (row 45) | `len(procs) + 1` | — | `73+1=74` |

All six decompositions write the same four classes in the same order. `config:41`'s key is
`gate_sha_driver` where the prose says *"GATE-SHA driver leg"* — cosmetic, not a mismatch.

**Row 31's fourth banner.** I confirmed at source that `sbatch:62` is the GATE-SHA banner comment and
`:63-64` the invocation, that the driver leg therefore precedes the `66 mints` block at `:67`, and
that the other two banners are at `:101` (`6 fidelity`) and `:138` (`1 arena`). Row 31 replacing `:62`
with a `1 GATE-SHA driver leg` block banner gives **four banners over four process classes** in the
same order as `config:41` and `config:42`. Row 69 is correctly CORRECT: each existing banner is
accurate for its own block.

**Phase 1c stays 67, verified at source.** `load_frozen()` opens exactly one file,
`configs/c01/c01_a0_v2.json`, and no `.pt`; `load_ro` is first reachable at `arena:1284`, after the
early return at `:1278-1280`. The driver leg loads no ro cache, so `66 mints + the arena` is right.

**§7's `68` ratio-computing processes.** `74 − 6 fidelity = 68` ✓, and the driver leg is correctly
inside the 68 since it emits heartbeat lines. This is consistent — and it is what makes **I-3** below
a contradiction rather than a quibble.

---

## 8. ROUND 6's 1H / 3I / 5M — LIMB-LEVEL DISPOSITION

| finding | limb | disposition |
|---|---|---|
| **H-1** | row 67 = `V15E1:1814` | **adopted**, verified at source |
| | row 68 = `config:42` | **adopted**, verified at source |
| | sweep K with list, charges, subtraction | **adopted**, reproduces byte-identically |
| | §6's conclusion re-scoped | **adopted verbatim** |
| | *(beyond the obligation)* domain re-keyed to quantities | **delivered**, and it survives attack |
| **I-1** | row 64 → `:1368-1369` | **adopted**; `:1369` confirmed at source as the line carrying *"count is `1`, the arena alone"* |
| | added to the multi-line list | **adopted** |
| | `:1369` named extent-only | **exceeded** — J's new `count is` term *charges* it (sweep J hit 19) |
| **I-2** | row 5 → `:1644-1646` | **adopted**; `:1644` confirmed to carry *"The arena's own startup"* |
| | row 54 → `:2439-2440` | **adopted**; `:2440` confirmed |
| | row 44's extent/label reconciled, `:1820` charged | **adopted**; `:1820`→E, `:1822`→J, both to row 44; the contradiction is gone |
| | §10 prints the **verified** extent-only list | **NOT discharged** — the list is still one line short (**I-1** below) |
| **I-3** | *"three"* → **four** | **adopted** |
| | named `GATE-DET1`/`GUARD`/`GATE-SHA`/`GATE-SHA-ONLY` | **adopted**; verified at `arena:1270`, `:1272`, `:586`, `:1279` — `:586` is `self.hb("GATE-SHA", n, n, …)`, reached from the call at `:1276` |
| | round 2's enumeration cited rather than round 5's count | **adopted, and the credit is accurate** — R2's H-1 item 4 does list all four by name |
| | *(propagation)* | **NOT propagated to row 13** (**I-3** below) |
| **M-1** | `:1582`'s present tense → row 70 | **adopted**; the replaced phrase lies wholly on `:1582`, so single-site is right |
| **M-2** | sweep J's reason for `:1583` | **adopted**; the declaration now names both patterns, identically in E and J |
| **M-3** | extent-only heading counts separately | **adopted in form**; the count is wrong (**I-1**) |
| **M-4** | `U3–U6` range → sixteen units written out | **adopted**; I re-derived the residue vocabulary and it is exactly the 12 phases and 16 units v7 lists, item for item |
| **M-5** | §9 lists all 71 rows | **adopted in coverage**; the split is wrong (**M-1**) |

---

## 9. CUMULATIVE LANDING COHERENCE — one prescription lost, three compressed away

I diffed every row's site cell and target cell from v6 to v7. **No row was dropped** (v6's 67 rows all
persist; 67, 68, 69, 70 are new). Thirty-five site cells changed, all of them either the round-6
site-range extensions or the addition of an `(N lines)` label — no site silently narrowed.

Thirty target cells changed. Twenty-six are pure compression that loses nothing recoverable. **Four
lose prescription:**

* **row 48** — v6: *"export `C06_MINTS_EXECUTED` **from an executed-vs-skipped counter** and
  `C06_PROJECTED_SECONDS` **from `config:43`**"*. v7: *"export `C06_MINTS_EXECUTED` and
  `C06_PROJECTED_SECONDS`"*. §7 retains only the `PROJECTED_SECONDS` provenance. This one is
  material — see **H-2**.
* **row 49** — v6 spelled the replacement note as *"§11 declares **38** = 7 + 6 + 8
  imported/read/cached digests **plus the design document…**"*; v7 elides `= 7 + 6 + 8`. See **M-3**.
* **row 52** — v6 pinned *"(8 caches + 13 modules/configs + 16 banked + **the design document**)"*;
  v7 elides to *"… + the design document"*. See **M-3**.
* **row 66** — v6 gave the whole replacement comment including *"and opens no dev_seen file (§12)"*;
  v7 truncates at an ellipsis. See **M-4**.

Everything else endorsed across the seven proposals survives intact: rows 23/24's dual-digest
publication and the M-4 caveat string; row 39's docstring rewrite; row 59's verbatim retention of the
two uninstrumented assertions; row 44's resume warrant; the rejections of (ii) and (iii); (iv)
declined on TOCTOU; the *argv ≠ launch argv* note recorded and not built on.

---

# FINDINGS

### H-1. §8's Phase 1d re-price is arithmetically wrong, and the error lands in the heartbeat denominator that §7 makes a HALT-on-mismatch single source.

`U7` is **`0.13 s`** (`V15E1:1300`: *"0.12 s + `0.005 s` measured for the sixteen = **0.13 s**"*), and
§6 confirms it does not move — the added artifact costs `0.000164 s`.

The design document states its own convention, with a worked example of exactly this operation, at
`V15E1:1595`:

> *"Phase 1c's count rises `66 → 67` and its product is `67 × 0.033 = 2.211 → 2.2`, unchanged at one
> decimal."*

Product = count × unit, then rounded to one decimal. Applying it to Phase 1d as the count rises
`1 → 2`:

* `1 × 0.13 = 0.13 → **0.1 s**` — which is what `V15E1:1547` carries today, and which `:1350`
  explicitly names as *"one-decimal rounding that rounds 8 and 9 ruled acceptable."*
* `2 × 0.13 = 0.26 → **0.3 s**`.

**v7 prescribes `0.2 s`** (row 3, and §6's summary table as `2 × U7 = 0.2 s`). `0.2` is obtainable
only by doubling the already-rounded `0.1`, which the document's own example forbids. No rounding
convention in §8 yields `0.2`: nearest gives `0.3`, and the conservative-up convention the sub-`0.1 s`
rows use (Phase 1e `0.033 → 0.1`, Phase 7z `0.0014 → 0.1`, Phase 1f `0.615 → 1.3`) also gives `0.3`.
Phase 1g escaped the same error only because `2 × 3.8 = 7.6` needs no re-rounding.

**It propagates, because Phase 1d is a summand of the residue.** I summed §8's product column: every
row except the seven named in the total line comes to **exactly `2642.3`**, and the full sum to
**exactly `3670.0`** — so `V15E1:1569-1571` is verified as printed, and Phase 1d's `0.1` is inside
`2642.3`. With Phase 1d at `0.3` the delta is `+0.2`, not `+0.1`:

| figure | v7 prescribes | correct |
|---|---|---|
| Phase 1d product | `0.2 s` | **`0.3 s`** |
| §8 residue term | `2642.4` | **`2642.5`** |
| §8 total | `3673.9 s` | **`3674.0 s`** |
| `× 1.25` | `4592.4 s` | **`4592.5 s`** |
| `2×` miss | `4687.7 s` | **`4687.8 s`** |
| `5×` miss | `7729.1 s` | **`7729.2 s`** |

Minutes (`61.2 / 76.5 / 78.1 / 128.8`), shares (`68.3 % / 27.6 %`) and the `1.68×` ratio are
**unchanged under the correction**, so rows 35, 55 and 57 stay CORRECT and row 36's *"the `1.68×`
ratio on `:1610` is unchanged"* survives.

**Why this is High rather than cosmetic.** `3673.9` is not just a number in §8. §7 makes
`config:43` the **single source** for it, exports it, and has the arena assert environment, module
constant and `cfg["projected_seconds"]` all agree, **HALTing on mismatch**. Landing v7 as specified
writes `3673.9` into `config:43`, `arena:46`, `mint:112`'s fallback and V15E1 §8 and §9 (rows 14, 15,
16, 17, 19, 34, 36 and §6's table and §7 item 3), while §8's own printed product column sums to
`3674.0`. That is a design document internally inconsistent on the single quantity this erratum
exists to make consistent — the same species as row 49's *"otherwise a false arithmetic statement in
one of the five artifacts."*

**Provenance.** The `0.2` has been in the proposal since **v3**; rounds 3, 4, 5 and 6 each passed
over it, and round 6 listed *"§8's full arithmetic (`3673.9`, `4592.4` … `4687.7`/`7729.1`)"* among
the things v7 must **carry forward at full strength**. Carrying forward is how it survived six
rounds. This is not v7's origination — but v7 re-asserts it, and §9 is the landing list.

**Repair.** Row 3's cost → `0.3 s`; row 34 → `2642.5 … = 3674.0`, `× 1.25 = 4592.5`; row 36 →
`4687.8 s = 78.1 min`, `7729.2 s = 128.8 min`; rows 14, 15, 16, 17, 19 and §7 item 3 → `3674.0`
(and `4592.4 → 4592.5` where the conservative figure appears); §6's table → `2 × U7 = 0.3 s`,
total `3674.0`, `× 1.25 = 4592.5`, `4687.8 / 7729.2`. Print the re-multiplied sum term by term, as
v6's row 34 did, so the next reviewer can check it without reconstructing the residue.

---

### H-2. Row 48 lost the clause that made the `C06_MINTS_EXECUTED` export resume-safe, and the sbatch's only mint counter is the wrong one.

v6's row 48 read: export **`C06_MINTS_EXECUTED`** *"**from an executed-vs-skipped counter**"* and
`C06_PROJECTED_SECONDS` *"from `config:43`"*. v7's row 48 reads, in full:

> export **`C06_MINTS_EXECUTED`** and **`C06_PROJECTED_SECONDS`**

§7 item 2 retains the provenance of the second export only (*"from that key"*). **Nothing anywhere in
v7 pins the semantics of the first.**

The semantics are load-bearing, and the obvious implementation is the wrong one:

* The sbatch's only mint counter is `MINT_N` (`:77`), incremented **unconditionally** at `:81` and
  `:89`, printed as `mint ${MINT_N}/66`, and asserted `-eq 66` at `:99`. It counts **attempts**, and
  it is 66 on every run including a resume.
* `c06_falsifier_mint.py:218-220` returns at `MINT-SKIP` **before** the dev load whenever `--out`
  exists. On a resumed run, executed < attempted and `dev_path_opens` < 66.
* After this erratum, `gate_ledger` binds `dev_path_opens == mints_executed + expected_sha_dev_opens`
  with `expected_sha_dev_opens = 4` (row 38). Exporting `MINT_N` sets `mints_executed = 66` always, so
  a resumed run fails the binding predicate and **HALTs**.

That is precisely the failure §12's *"Why `mints_executed` and not `66`"* paragraph exists to prevent
— *"A binding `dev_path_opens == 66` would HALT a legitimate resume — the same class of self-defeating
gate this lineage has removed twice elsewhere"* — the warrant row 44 amends and keeps, on round 2's
resume-stability fact. The arena's fallback does not save it either: `arena:1419-1420` falls back to
`mints_present_before_arena`, which is also 66 on a complete resume.

**Repair.** Restore v6's limb and make it concrete: row 48 must name the counter's semantics
(incremented only on the executed path, not on `MINT-SKIP`) and, if the sbatch cannot distinguish the
two without reading the mint's exit signal, say how. Also restore `from config:43` for the second
export. This is the one place in v7 where a dropped clause changes what gets built.

---

### I-1. The extent-only list is still incomplete — `arena:474` is a sixth line — and row 39's `(17 charged lines)` label matches neither number.

Round 6's I-2 found v6's extent-only list claimed two lines where three more existed. v7 answers with
*"§10 prints the verified extent-only list — **five lines, not two**"* and *"this is the verified
list."* I derived the set myself, row by row, by parsing each site cell into an extent and
subtracting its charged sites. **Six lines, not five.**

Row 39's extent is `arena:432-441` (10 lines) **plus** `:468-475` (8 lines) = **18**. Sweep G returns
16 of them. Two are returned by nothing:

* **`arena:470`** — `fails.append(` — named by §10 ✓
* **`arena:474`** — `"implemented as frozen and is NOT adjusted here.".format(` — **not named**

I verified `:474` at source and confirmed why no sweep reaches it: G's terms are `NOT ADJUSTED`
(uppercase) and `implemented exactly as frozen`. The line writes `NOT adjusted` and `implemented as
frozen`. It is a **string literal**, where G's authority terms were written against the **docstring**
at `:432-441` — the carrier distinction §1's form list does not make.

And it is inside the edit, not merely inside the range: row 39's own *"why"* column calls this exact
clause false at landing, and its prescription says *"the message's decomposition becomes the
derivation"* — the message is `:471-475`.

Consequently the label `(**17 charged lines**)` is wrong in both directions: the extent is 18, the
charged count is 16. It is the only row whose label uses the words *"charged lines"*, and it is the
only one where the label matches neither quantity.

**Repair.** Name `arena:474` as a sixth extent-only line; relabel row 39 as `(18 lines, 16 charged)`
or split the two counts; and restate §10's heading as **6 lines plus 1 siteless row**. Optionally
widen G with `NOT adjusted|implemented as frozen` so the line is charged rather than declared — that
would also close the carrier gap the form list leaves open.

---

### I-2. §6's meta-check subtraction is stale: recomputed against v7's own twelve sweeps it is `102 = 32 + 70`, not `30 + 72`.

I reran the meta-check command. **102 hits ✓.** But v7 charges them against *its* sweeps, and v7 has
twelve where v6 had ten, with J widened:

| | charged | residue |
|---|---|---|
| v7 declares | 30 | 72 |
| against v6's ten sweeps (J unwidened) | 30 | 72 |
| against A–J with v7's widened J | 31 | 71 |
| **against v7's twelve sweeps** | **32** | **70** |

Sweep K charges `V15E1:1546` (Phase 1c's *"66 mints + the arena process itself"*), and J's new terms
charge one more. The `30 / 72` split is v6's arithmetic carried into a section whose inputs changed
in the same revision.

**The conclusion is unaffected**, and I verified that independently: I re-derived the residue
vocabulary and it is **exactly** the 12 phases (`1b, 1c, 1e, 1f, 2, 2b, 2z, 2D, 3, 4, 5, 7`) and 16
units (`U1, U2a, U2b, U2c, U2d, U3, U4, U5a, U5b, U6, U8, U9, U10, U_acc, U_mF1, U_tie`) v7 lists,
item for item. *"Two phases move, zero units move"* stands.

I grade this an Issue rather than a Minor because §6 is presented as a **check** whose subtraction
closes, and a printed subtraction that does not reproduce against its own inputs is a broken
instrument — which is the one thing this lineage has said it will not accept on trust.

---

### I-3. Row 13's `73 of 74` contradicts v7's own I-3 adoption; the correct figure is `72 of 74`.

`mint:117` reads *"**72 of 73** processes previously wrote nothing"*, inside the docstring recording
CODE-R1 H-3's rationale. Row 13 prescribes *"**73 of 74**"* — a mechanical `+1/+1` with no *"why"*
note.

But the count is of processes that wrote **nothing** to the progress file before H-3, and H-3 added
the heartbeat to `c06_falsifier_mint.py`, not to the arena. Before H-3 the writers were the arena
**and** any `c06_falsifier_arena.py` invocation with `--progress` — which is exactly what the driver
leg is (`sbatch:63-64`). v7 establishes this itself: its adoption of round-6 I-3 says the driver leg
emits **four** heartbeat lines before returning, and names them.

So at the corrected inventory of 74, **two** processes wrote and **72** did not:

* 66 mints — wrote nothing ✓
* 6 fidelity — write nothing; the bash driver writes their line (`sbatch:128-129`, rows 60/61/62) ✓
* arena — wrote ✓
* `--gate-sha-only` driver leg — **wrote**, four lines ✓

`72 of 74`, not `73 of 74`. v7's figure asserts that only one process wrote, which its own §8 I-3
paragraph and its own row 5 refute, and which §7's *"All **68** ratio-computing processes"* also
refutes — the driver leg is inside that 68 precisely because it computes and emits ratios.

This is a **new** wrong number, created by this erratum, in a live code file. Rows 55 and 66 are in
the delta because comment-level falsehoods in `c06_falsifier_mint.py` are worth an erratum; landing
row 13 as written adds one.

**Repair.** Row 13 → *"**72 of 74**"*, with a *"why"* note recording that the driver leg is one of
the two that wrote, so the next reader does not re-derive `73`.

---

### M-1. §9's *"55 editing, 16 CORRECT"* split matches nothing in §9.

Counted from §9's own columns: **59** distinct editing rows and **13** distinct rows in the CORRECT
column. Twelve rows are fully CORRECT (1, 18, 35, 45, 46, 47, 51, 56, 57, 58, 65, 69); row 41 appears
in both columns because it is partial; row 65 appears twice in the CORRECT column because its four
lines split across `config` and V15E2. `59 ∪ 13` with `|∩| = 1` gives **71** ✓ — the total is right.

Neither addend is: not 12 (fully CORRECT rows), not 13 (distinct CORRECT-column rows), not 14
(CORRECT-column entries), not 15 (entries including the `(18 is mint)` pointer). That `55 + 16 = 71`
suggests `55` was obtained by subtracting a wrong CORRECT count from the correct total rather than
counted — the addends were never checked against the columns above them. This is the section
round-6 M-5 asked to be made a visible partition.

### M-2. §2's stated criterion for the 29-row list is not the criterion the list uses.

§2: *"The **29** rows with more than one **charged site** are …"*. Computed from the sweeps, **28**
rows have more than one charged site. Row 54 has exactly one (`V15E1:2439`); `:2440` is extent-only
and §10 says so. The membership is right under *"more than one line in its extent"* — which yields
exactly 29, and which is what §8's I-2 paragraph intends when it puts rows 5 and 54 in the list. One
word in the criterion, not the list.

### M-3. Rows 49 and 52 were compressed to ellipses that drop the decompositions they must repair.

`config:255` currently reads *"§11 declares 37 = 7 + 6 + 8 imported/read/cached digests plus the 16
banked artifacts … 21 + 16 = 37"*. `7 + 6 + 8 = 21` and must become 22. v7's row 49 pins
*"22 + 16 = 38"* but elides the first decomposition; v6 spelled it out. Likewise `V15E1:1300` reads
*"over all 37 §11 artifacts (8 caches + 13 modules/configs + 16 banked; …)"*, and `8 + 13 + 16 = 37`
must gain a fourth term; v6's row 52 pinned the parenthetical, v7 elides it. Both are
**decomposition sites whose totals move** — the eleventh family this version was built to close — and
both now depend on the implementer reconstructing what v6 had already written down.

### M-4. Row 66's prescription was truncated past the clause that must survive.

v6: *"# It reads only `meta` out of the banked mint .npz **and opens no dev_seen file (§12)**."*
v7: *"# It reads only `meta` out of the banked mint .npz…"*. The elided half is **true** and is the
limb §12 preserves (row 43 keeps the corresponding sentence). An implementer following v7 literally
could drop it. Row 61 likewise lost v6's `(sbatch:128-129)` citation, though §3.8's preamble still
carries it.

### M-5. §1's *"what the empty cells assert"* leaves twenty empty cells with no stated reason.

§1 claims *"every EMPTY cell is an explicit claim with a stated reason"* and then states four:
F3 for Q5–Q10, F2 for Q7–Q10, F9 for Q5–Q10, F8 for Q1. Twenty cells are unexplained (enumerated in
§6 above), including `Q2×F3` and `Q3×F3`, which the ordinal sentence would cover if it were not
scoped to Q5–Q10, and `Q4×F2`, which the sum-of-parts sentence would cover if it were not scoped to
Q7–Q10. I swept the exposed ones and found no falsifying site, so this costs the table nothing
substantive — but the promise is the whole basis on which a reviewer is invited to attack the table,
and it is not kept.

---

## OBLIGATIONS FOR A V8 THAT WOULD CARRY GO

1. **H-1**: Phase 1d → `0.3 s`; residue `2642.5`; total `3674.0`; `× 1.25 = 4592.5`;
   `4687.8 / 7729.2`; the denominator literal `3674.0` in rows 14, 15, 16, 17, 19 and §7 item 3;
   §6's table. Print row 34's re-multiplied sum **term by term**, as v6 did.
2. **H-2**: restore row 48's *"from an executed-vs-skipped counter"* and *"from `config:43`"*, and
   state how the sbatch obtains a count that excludes `MINT-SKIP`.
3. **I-1**: name `arena:474`; relabel row 39 (extent 18, charged 16); §10's heading → **6 lines**.
4. **I-2**: recompute §6's subtraction against the twelve sweeps — `102 = 32 + 70`. The vocabulary
   lists and the conclusion are correct as printed; carry them unchanged.
5. **I-3**: row 13 → *"72 of 74"*, with the driver leg named as the second writer.
6. **The five minors.**
7. **Carry forward at full strength, everything I re-derived independently:** the twelve sweeps'
   byte-identical reproduction including sort order; the partition `305 → 267 = 137 + 130`,
   `UNCHARGED = 0`, 38 excess / 32 multi-sweep; the twelve per-sweep subtractions; zero divergent
   charges; all 71 rows exactly once in §3 and once in §9; the 29-row list's **membership**; sweeps
   K and L as instruments; rows 67, 68, 69 and 31 and their consistency with all nine inventory
   statements; the four heartbeat lines at `arena:1270`, `:1272`, `:586`, `:1279` and round 2's
   credit; Phase 1c's `67` (`load_frozen` opens one JSON and no `.pt`; `load_ro` first at `:1284`);
   `len(procs) + 1 = 74`; `21 + 16 = 37 → 22 + 16 = 38`; the residue vocabulary of 12 phases and 16
   units; the Q1×F8 emptiness; and **the quantity-list and form-list attacks, which found no
   eleventh quantity and no form outside the nine that hides a site.**

**The delta's substance does not grow.** Two corrected figures, one restored clause, one extent-only
line, one recomputed subtraction. No new row, no new sweep, no code change beyond what v7 already
prescribes.

---

## WHAT V7 STILL GETS WRONG — SUMMARY

v7 does what round 6 asked and more: it stops defending the enumeration and re-keys it onto the
quantities, then prints the form list as a table and invites the attack. I ran that attack — six
invented forms, every one swept over all five artifacts, plus the paraphrase and spaced-identifier
families the table names — and **the domain holds.** There is no eleventh quantity and no form
outside the nine that conceals a site. That question, open since round 1, is closed.

What v7 gets wrong is on the other side of the instrument. Having built a machine that verifies
*which sites* the erratum touches, it did not re-derive *what numbers* it writes there. `2 × U7` is
`0.26`, which is `0.3` at one decimal by the document's own worked example at `V15E1:1595`; v7 carries
`0.2`, inherited unexamined from v3 through four adjudications, and with it a §8 total of `3673.9`
against a product column that sums to `3674.0` — an inconsistency in the heartbeat denominator, which
is the single quantity §7 elevates to a three-way HALT-on-mismatch assertion, in an erratum whose
purpose is to stop the five artifacts disagreeing about a number. Row 13 makes the same kind of
mistake in the other direction: `72 → 73` is the arithmetic of adding a process, but the process being
added is one that **wrote**, so the count of those that wrote nothing does not move, and v7's own
adoption of round-6 I-3 is what proves it. And row 48, alone among 71, is weaker in v7 than in v6:
the clause that kept the `mints_executed` export resume-safe was compressed away, leaving the sbatch's
`MINT_N` — a counter that is 66 on every run by construction — as the obvious thing to export and a
legitimate resume as the thing that would HALT.

The three findings share a shape. Each is a place where the *form* of the change was checked and the
*content* was assumed: a product doubled instead of recomputed, a count incremented instead of
re-derived, a prescription shortened instead of restated. The completeness domain v7 built cannot
catch any of them, because all three sites are correctly identified, correctly charged and correctly
listed in §9. They are the residue that a sweep-based method leaves behind, and the next version
should say so — the domain proves the delta is *complete*, never that it is *right*.

---

## BLINDNESS AND EDIT STATEMENT

No battery-arm accuracy or macro-F1 computed; `deployed_vote` called zero times; no arm built; no
mint, no arena run, no `--gate-sha-only` leg; no GPU, no SLURM job, no commit, no `TARGET_STATE.json`
edit. `artifacts/c06_falsifier/` verified absent.

**Compute used:** `sha256sum` over the five artifacts and all seven proposals; file and source reads;
the twelve `grep` sweeps of §10 plus the §6 meta-check and roughly twenty wider adversarial variants
(the six invented forms, the paraphrase family, the spaced-identifier family, the `lab_dev` mechanism
family and the digest-count family); python arithmetic over the sweep output and §8's product column.
Every count, partition and extent computation in this review is produced by a script over the raw
sweep output, not transcribed.

**Nothing edited.** The five artifacts carry the same digests before and after, matching §11 exactly.
`…_PROPOSAL.md` through `…_V6.md` are byte-unmodified and match the digests v7's header cites.
