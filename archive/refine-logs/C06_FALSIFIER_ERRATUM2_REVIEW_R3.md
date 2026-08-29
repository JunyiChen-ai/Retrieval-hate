# C06 `$0` falsifier — **ERRATUM 2, INDEPENDENT REVIEW — ROUND 3**

*Target:* `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V3.md`
(sha256 `48f4e0153103cc608884b8fb70b5fccf965b355e8ae86a9b1c2431e235a34aee`)
*Supersedes as review target:* `…_V2.md` (`4225bea3cc9907d38e2e3f5815448f7d0b291195523b781be33367feb132e040`
— **verified, matches** v3's citation) and `…_PROPOSAL.md`
(`f063c388c4afabdb7964360eda2fe1ef7d6fee611b2a287ca9817929c7f670f5` — **verified, matches**)
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`
(`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` — **verified, matches**)
*Prior adjudications:* `C06_FALSIFIER_ERRATUM2_REVIEW.md` (REVISE 0C/3H/3I/3M);
`C06_FALSIFIER_ERRATUM2_REVIEW_R2.md` (REVISE 0C/2H/3I/3M)
*Reviewer:* fresh and independent; no part in the fifteen design rounds, the implementation,
Erratum 1, code-review round 1, or either prior erratum-2 round. Judged from documents, repository
and execution only.

---

## VERDICT

> **REVISE — 0 Critical, 3 High, 3 Important, 3 Minor.**

**The erratum's substance is right and I reproduced its arithmetic core at source.** The
concatenated iterable `gate_sha` consumes is 21 files, exactly 2 are dev-like and 0 are test-like,
`frozen_sha256` alone contributes 0 — so `expected_sha_dev_opens = 2 × 2 = 4` on both factors, and
the first factor is genuinely derived from the config's own digest table rather than asserted. The
`74 = 1 + 66 + 6 + 1` inventory holds. The uniform counter criterion is correct, uniformly applied,
and matches the C09 precedent in the way that matters; I audited the code paths and confirm the
demotion removes **no reachable tripwire** — `c09guard._guarded_open` increments exactly three
counters (`:97`, `:102`, `:106`) and the other three are written by nothing in C06's process tree, a
fact that is itself digest-pinned because `c09guard.py` **is** in `frozen_sha256`. Every one of round
2's seven obligations is engaged and six are discharged at full strength. §4's re-price arithmetic
checks out to the last decimal.

**And the `\b7[234]\b` subtraction claim is exactly, verifiably true.** I ran the sweep myself over
all five artifacts: 18 hits. Ten rows of §1 account for eleven of them (row 12 covers two lines), and
the seven the proposal declares as non-process — three line citations, three rotation angles, one
tie-cap product — are precisely the seven that remain. Nothing is left over in either direction.
**That leg of the enumeration is the strongest verification work in this lineage's record**, and I
say so plainly because what follows is not a criticism of it.

**What blocks GO is that the enumeration is complete for the numeral `73`/`74` and is not complete
for "every site stating a quantity this erratum moves" — which is the promise §1 actually makes.**
The `73`/`74` sweep was run and its subtraction is stated. The other sweeps §1 claims — *"for each
quantity this erratum moves"* — were not carried through with the same rigour, and no subtraction is
stated for them. My own sweeps find **ten sites** outside the table, in all five artifacts. Two of
them are already carrying wrong numbers today, independently of this erratum: the config's
`design_sha256` names a V15E1 state that has not existed since CODE-R1 and is **published on the
verdict face**, and `c06_falsifier_mint.py:112` runs 66 of the 74 processes' heartbeat ratio off a
denominator `740.1 s` below the arena's. §4's sentence *"`PROJECTED_SECONDS` moves in **both**
`c06_falsifier_arena.py:46` and the config"* is a completeness claim about a quantity with four
sites.

**And one prescribed repair contradicts another prescribed repair in the same table.** Row 2 renames
the driver leg *"the 73rd process"* while row 6 declares the order *"1 `GATE-SHA` driver leg → 66
mints → 6 fidelity → 1 arena"*, under which the driver leg is the **first** process and the 73rd is a
fidelity process. Landing both lands the internal contradiction this erratum exists to end, written
by the erratum.

---

## THE ARITHMETIC CORE — MY OWN MEASUREMENT

Run under an imported `c09guard`, over the config's own digest tables, opening no cache:

```
concatenated iterable (frozen_sha256 + frozen_sha256_input_caches): 21 files
  dev-like:  2   data/CLIP_Embedding/HateMM/dev_seen_…-LoRA-curric_HF.pt
                 data/CLIP_Embedding/MHC_zh/dev_seen_…-LoRA_HF.pt
  test-like: 0
frozen_sha256 alone: 13 files, 0 dev-like
frozen_sha256_input_caches alone: 8 files, 2 dev-like
configs/c06/c06_falsifier.json in frozen_sha256:  False   (I-2's premise, confirmed)
scripts/analysis/c09_guard/c09guard.py in frozen_sha256:  True
```

`2 dev-like × 2 passes = 4`. **Both factors reproduce.** The two derivations (over the concatenated
iterable vs. over `frozen_sha256_input_caches` alone) agree today because `frozen_sha256` contributes
zero, and v3's reason for preferring the concatenated one is correct rather than cosmetic.

**§4's re-price, checked line by line.** Phase 1d `1 × U7 → 2 × U7`, carried `0.1 → 0.2 s` on §8's
existing (down-)rounding of `U7 = 0.13 s`; Phase 1g `1 × U11 → 2 × U11`, `3.8 → 7.6 s`. Total
`3670.0 + 0.1 + 3.8 = 3673.9` ✓. `× 1.25 = 4592.375 → 4592.4` ✓. Minutes `61.23 → 61.2` and
`76.54 → 76.5` ✓ unchanged at one decimal. Mint share: the six mint rows sum to `2508.3 s`;
`2508.3 / 3673.9 = 68.27 %` → **`68.3 %` holds** ✓. Phase 3: `1013.8 / 3673.9 = 27.59 %` →
**`27.6 %` holds** ✓. `2×` miss `3673.9 + 1013.8 = 4687.7 s = 78.1 min` ✓; `5×` miss
`3673.9 + 4 × 1013.8 = 7729.1 s = 128.8 min` ✓. **Every figure in §4's table is right.**

**The `\b7[234]\b` subtraction, verified by my own sweep.**

| file | hits | accounted by |
|---|---|---|
| `V15E1` | `:106`, `:310`, `:1197`, `:1550`, `:1566`, `:1658`, `:1691`, `:1839`, `:1904`, `:2017` | rows 1, 4, 6, 7, 8 + 5 declared non-process |
| `config` | `:41`, `:65`, `:222`, `:264` | rows 10, 11 + 2 declared non-process |
| `mint` | `:117` | row 13 |
| `sbatch` | `:16` | row 9 |
| `arena` | `:465`, `:466` | row 12 |

18 hits, 11 in rows, 7 declared. **The declared residue is exactly right**: the line citation
`generate_…:73-89` occurs at `V15E1:106`, `V15E1:1658` and `config:264` (×3 as stated); `72.7` at
`V15E1:310`, `V15E1:1691` and `config:65` (×3 as stated); the tie-cap `7×6 + 5×6 = 72` at
`V15E1:1566`. Nothing left over.

*One methodological note, which is Minor and recorded as such below:* `\b7[234]\b` **cannot** match
`73rd`, which is row 2's own site (`V15E1:1198`). Row 2 exists, so nothing was lost — I swept
`[0-9](st|nd|rd|th)` across all five files and the only ordinals are `V15E1:1198` (row 2),
`arena:286` (*"the 21st"*) and `V15E1:458` (*"95th percentile"*), neither of the last two relevant.
But the pattern the subtraction rests on has a structural hole in it that the document does not
disclose.

---

## THE UNIFORM CRITERION (round 2's H-2) — DISCHARGED IN FULL

**The criterion is genuinely uniform.** §2 states one rule, applies it to all six counters in one
table, dispositions each, and takes the harder branch on rows 2 and 6 rather than accepting round 2's
offered exception. There is no unexplained carve-out. Its argument for refusing the exception — that
a vacuous belt *"presents as a live check, and the failure mode this whole erratum documents is a
predicate that looks checked and is not"* — is the right reading of this lineage's own defect record,
and I endorse it.

**No reachable tripwire is removed. Audited at source.** `c09guard._guarded_open` (`:95-107`)
increments exactly `test_path_opens` (`:97`), `dev_path_opens` (`:102`) and `banked_trainlog_opens`
(`:106`). The other three keys are initialised at `:41`, `:43`, `:44` and incremented nowhere.
Repo-wide, the only other code that touches those three names is **C09's** arena
(`c09_a0_arena.py:1940`, `:1944`, `:1949`, all writes into C09's own output block) and **C06's** three
read-only predicates (`c06_falsifier_arena.py:458-462`). There is no path — defect-induced or
otherwise — by which a C06 process can increment them, and the guard that would have to change to
create one is **digest-pinned in `frozen_sha256`**, so `GATE-SHA` HALTs before any modified guard
could run. The three integer predicates are guaranteed-false comparisons and nothing else. **The
demotion is safe and the erratum's warrant for it is sound.**

**The C09 precedent's form, read from the artifact rather than the source.**
`artifacts/c09_topo/v1/a0/C09-A0-v1/C09_A0_DECISION.json::GATE_LEDGER` carries a `measured` block
with exactly `test_path_opens`, `dev_path_opens`, `banked_trainlog_opens`, and a
`by_construction_zero` block with exactly the other three as narrative strings. **v3's prescribed
structure matches it.** It also confirms two things v3 relies on: C09's `measured_expectations` for
`dev_path_opens` is itself a *string* that declines to bind, and its `evidence.resume_note` refuses to
bind `39` — so §0's claim that C06's gate is stronger than the precedent's, and safe only because of
the spawn/skip split, is correct as stated.

**§5's I-3 repair is implementable exactly as described.** C09's call site is
`c09_a0_arena.py:1901` (`cov = c09guard.verify_predicate()`) publishing into
`predicate_coverage_measured_this_run` at `:1902-1911` — four lines, copyable, and I confirm the
published block exists in C09's decision artifact with `n_repo_files_matched: 983`. §12's row 9
becomes true as written.

**The audit mechanism's input exists.** `c09guard._flush` (`:129`) records `sys.argv` per process, so
`#{ledger argv containing --gate-sha-only}` is computable with no edit to the sha-frozen guard, and
I-1's `int("GATE-SHA" in self.gates)` is available because `gate_sha()` sets `self.gates["GATE-SHA"]`
at `arena:584`.

---

## DISPOSITION AUDIT OF ROUND 2 — LIMB LEVEL

| finding | limb | disposition in v3 | strength |
|---|---|---|---|
| **H-1** | the five named sites become delta rows | rows 2 (`:1197-1199`), 4 (`:1550`), 3 (`:1547`), 5 (`:1645`), 9+16 (`sbatch:16-17`) | **full for the five named** |
| | state explicitly whether §8 is re-priced | §4 chooses branch (a) on the record, with a stated reason | **full** |
| | `PROJECTED_SECONDS` moves in **both** files | §4 + row 25 | **incomplete — four sites, not two; see H-2** |
| | *(the obligation's own scope: "every process-count and pass-count site in the document")* | §1's enumeration | **incomplete — ten sites outside the table; see H-1, H-2, I-1, I-2, I-3** |
| **H-2** | all three uninstrumented counters as warranted strings, out of the measured block | §2's table + rows 23, 24 | **full** |
| | mark rows 2, 5, 6 by-construction on §12's face | §7's V15E2 row (*"by-construction marks"*) | **full** |
| **I-1** | own-process term derived, not literal `1` | §3, adopted in terms with the failure mode spelled out | **full** |
| **I-2** | one cross-checked copy of each expectation | §5, `gate_ledger` asserts against `cfg["ledger"]`, covering the `74`, the `66`, the three zeros and the new formula | **full** |
| **I-3** | call `verify_predicate` or amend the row | §5, adopted (call it) | **full** |
| **M-1** | fold the two prose-only §12 prescriptions into the delta | §5, adopted | **full** |
| **M-2** | ledger `argv` ≠ launch argv for an executed mint | §5, recorded, not built on | **full** |
| **M-3** | sbatch activates no environment | §5, referred to the code lineage | **full** |
| **obl. 7** | carry the discharged core unchanged | §0 + §3 + §6 | **full, except row 2's ordinal — see H-3** |

**Round 2's own additions are carried and correctly attributed**: the resume-stability of `74`, the
C09 non-binding contrast, the spawn/skip split at `c06_falsifier_mint.py:217-220`. §1's row 26 (the
dead `arena:1418` line) is a real find, correctly characterised, and is evidence the grep discipline
works when it is actually run.

---

## FINDINGS

### H-1. The config's published design pointer is wrong **today**, and the erratum — which necessarily moves it — has no row for it. Four sites name the design; none is in the table.

*Attaches to:* §1 (the enumeration's stated scope); §7's config delta row; `configs/c06/c06_falsifier.json:5-7`; `c06_falsifier_arena.py:29-30`, `:1249-1250`, `:1433-1434`; `c06_falsifier_cpu.sbatch:14`.

```
configs/c06/c06_falsifier.json:6  "design_sha256": "0b446b91675fd4ff8aea15f2648401d6ce589d089eadad34846f885b2ec9c2ab"
V15E1 actual sha256              : 8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d
```

The config names `C06_FALSIFIER_PREREG_DRAFT_V15E1.md` at the sha it had **when Erratum 1 landed**.
CODE-R1 then edited that document — the implementation record's own table (`:328`) records
`0b446b91… → 8cde58aa… (§8 correction only)` — and the config's `design_sha256` was not updated with
it. The stale digest is not inert: `c06_falsifier_arena.py:1433-1434` writes
`"design": {"document": cfg["design_document"], "sha256": cfg["design_sha256"]}` onto the face of
**every verdict artifact**, and `emit_halt` (`:1249-1250`) writes the same onto every HALT artifact.
Any reader verifying a C06 verdict against the design it cites gets a digest mismatch on a document
that has not existed in that state since CODE-R1.

Landing Erratum 2 produces **V15E2**, so `design_document` and `design_sha256` must move regardless —
this is not an optional cleanup, and Erratum 1 set the precedent explicitly (*"The literals in
`c06_falsifier_arena.py:29,45` and the config are updated"*, implementation record `:294`). Three
further sites name the design by version and are equally absent from §1 and from §7's delta:

* `config:7` — `design_status`, a narrative naming *"GO at round 15 … + ERRATUM 1 landed"*; Erratum 2
  is not in it. Row 22 replaces `config:211`'s `blocked_on_erratum_2` narrative with an `erratum_2`
  block, which is right, but `design_status` is a **different** key in a different block.
* `arena:29-30` — the module docstring, *"§8 projects 3670.0 s (4587.5 s conservative) under v15 +
  ERRATUM 1 + CODE-R1 H-4 (`…V15E1.md`)"*. Both the figures and the document name move. Row 25 names
  only `arena:46`.
* `sbatch:14` — *"Frozen design: `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` (GO at round 15)"*,
  already two revisions stale, in a file this erratum edits for rows 9, 16, 17 and obligation 7.

**Why this is High.** It is a wrong number on the verdict face, in one of the five artifacts, of a
quantity the erratum moves — the precise category §1 promises to enumerate exhaustively. It was found
by a routine sweep of the same five files. And unlike the process count, it is wrong *now*: it is not
a defect the erratum would introduce but one the erratum's own sweep should have caught and did not.

**Repair.** Add rows for `config:5`, `config:6`, `config:7`, `arena:29-30` and `sbatch:14`; state
that `design_document` → `…_V15E2.md` and `design_sha256` → V15E2's digest, computed **after** V15E2
is written, and record in §7 that this is the last edit in the landing order.

### H-2. `PROJECTED_SECONDS` has four sites, not two — and one of them, `c06_falsifier_mint.py:112`, is already `740.1 s` stale and is the live heartbeat denominator for 66 of the 74 processes.

*Attaches to:* §1 row 25; §4's closing sentence; `c06_falsifier_mint.py:112`, `:127`;
`c06_falsifier_arena.py:29`, `:46`; `configs/c06/c06_falsifier.json:43-44`; `V15E1:1631-1633`,
`§13.1 item 12`.

Every site in the five artifacts carrying §8's projection:

| site | current | in §1's table? |
|---|---|---|
| `arena:46` `PROJECTED_SECONDS = 3670.0` | correct at V15E1 | **yes**, row 25 |
| `config:43-44` `projected_seconds` / `_conservative` `3670.0` / `4587.5` | correct at V15E1 | **yes**, row 25 |
| `arena:29` docstring *"projects `3670.0 s` (`4587.5 s` conservative)"* | correct at V15E1 | **no** |
| `mint:112` `PROJECTED_SECONDS = 2929.9` | **wrong today** — the pre-CODE-R1-H-4 figure | **no** |

`c06_falsifier_mint.py:127` divides by that constant on every heartbeat line the mints write. So on a
clean run today, 66 processes publish `elapsed / 2929.9` while the arena publishes `elapsed / 3670.0`
— a `25 %` disagreement in the ratio column of the progress file, in the phase that is `68.3 %` of the
budget. §9 says the denominator *"is pinned to §8 by name, so it tracks automatically"* and §13.1
item 12 makes *"the frozen `elapsed ÷ projected` denominator"* a handoff item the code lineage must
verify. It does not track, and the erratum's re-price widens the gap to `744.0 s`.

**§4's own sentence is the completeness claim that fails:** *"`PROJECTED_SECONDS` moves in **both**
`c06_falsifier_arena.py:46` and the config, because §9 pins the heartbeat denominator to §8 by
name."* The word *both* asserts a closed set over a quantity with four sites.

**And `V15E1:1631-1633` is a fifth site, in §9, absent from the table**, which states the projection
*and* repeats the same two-file claim: *"(ERRATUM 1 set it to `2929.9 s`; CODE-R1 H-4 sets it to
`3670.0 s`. The denominator is pinned to §8 by name, so it tracks automatically; the literal in
`c06_falsifier_arena.py` and `configs/c06/c06_falsifier.json` is updated with each correction.)"* At
landing this sentence must carry `3673.9` **and** name the mint, or it re-publishes the error the
erratum just corrected.

**Why this is High.** Same reason as H-1, and one more: this is the quantity §4 re-prices. The
erratum's most visible act is moving `3670.0 → 3673.9`, and its instruction for doing so names two of
the four places the number lives, one of which is already carrying a different number.

**Repair.** Rows for `arena:29`, `mint:112` and `V15E1:1631-1633`. `mint:112` → `3673.9` with its
comment updated off *"§8 under ERRATUM 1"*. §7's mint delta row grows from *"1 line"* to two.

### H-3. Row 2's replacement text contradicts row 6's replacement text: the driver leg cannot be both *"the 73rd process"* and the first in the declared order.

*Attaches to:* §1 rows 2 and 6; `V15E1:1197-1199` (§7.2); `V15E1:1839` (§13); row 11's
`"decomposition": "1+66+6+1"`.

Row 6 prescribes §13: *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6
fidelity → 1 arena."* Under that order the driver leg is process **1**, the mints are 2–67, the
fidelity processes 68–73 and the arena **74**.

Row 2 prescribes §7.2: *"The **73rd and 74th** processes — the `--gate-sha-only` driver leg and the
arena — are a different case and are priced separately at §8 Phase 1g."* By its own apposition the
driver leg is the **73rd**. Under row 6's order the 73rd process is a fidelity process and the driver
leg is the 1st.

Row 11 makes it worse rather than better: `"decomposition": "1+66+6+1"` puts the driver leg's `1`
first, matching row 6 and contradicting row 2. So the erratum prescribes the driver leg's position
three times and gets it wrong once.

**Why this is High rather than Minor.** The defect this erratum documents is a document asserting two
incompatible things about the same object because nobody re-derived one of them. Row 2 is the repair
for exactly that defect in §7.2, and it lands a fresh instance of it — this time between two rows of
the same table, which is worse than the §12/§7.2 split round 2 refused, because a reader who checks
one row against the other inside the erratum finds the contradiction before the document is even
written. §7.2's paragraph is also the one whose scope clause round-11 M-1 already repaired once for
this exact reason (*"v11's 'already inside every unit' was true of the mint units this section is
titled for and false of the arena's"*).

**Repair.** Row 2's replacement should not use ordinals at all — the paragraph's job is to say which
processes are *not* priced inside the mint units. E.g. *"The remaining two processes — the
`--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8
Phase 1g."* That is true under any ordering and cannot drift against §13.

### I-1. §7.7's `U11` row is a sixth site of the §7.2 accounting defect, in the section that **defines** the unit §8 Phase 1g prices twice.

*Attaches to:* `V15E1:1304`; §1 row 4; §4's Phase 1g re-price.

`U11`'s row closes: *"Both are **inside the mint units and inside `U9`**; the arena's is priced once
at §8 Phase 1g."* That sentence is an exhaustive accounting of where every `U11` instance in the job
is charged — the mint/fidelity class inside the mint units and `U9`, the arena class at Phase 1g —
and after the re-price Phase 1g charges **two** arena-class startups, of which the sentence names one
and does not mention the driver leg at all.

It is not literally false (the arena's own startup is indeed priced once there), which is why I rate
it Important rather than High. But §7.2 and §8 Phase 1g both cite §7.7 as the authority for the unit
and its class, so a V15E2 in which Phase 1g reads *"`2` — the `--gate-sha-only` driver leg and the
arena"* while its own unit definition accounts for one instance is the same instrument-vs-run gap that
round 2's I-3 caught in §12 row 9. §1's own standard settles the disposition: even a **CORRECT** site
gets a row *"because completeness is verified by subtraction"* (row 1 is listed on exactly that
ground). This site is neither listed nor declared out of scope.

**Repair.** A row for `V15E1:1304` amending the closing clause to name both arena-class startups.

### I-2. `c06_falsifier_arena.py:1232` is the fifth `ONCE` site, in a file the erratum edits, and it is the one a user actually sees.

*Attaches to:* §1b (rows 14–18); `c06_falsifier_arena.py:1230-1232`.

```python
ap.add_argument("--gate-sha-only", action="store_true",
                help="run GATE-DET1 and GATE-SHA and exit; the sbatch driver calls "
                     "this ONCE before any other process (§13)")
```

My `once|ONCE|one span|twice|single pass` sweep over the five artifacts returns eleven relevant hits.
Nine are rows 3, 5, 14, 15, 16, 17, 18 or irrelevant idiom (`arena:55` *"opened once and never
re-wrapped"*, `arena:1039` *"once per (arm, seed)"*, `V15E1:1583` *"once sklearn is restored"*). The
two that are neither are `V15E1:1304` (I-1) and this one.

Its claim is defensible on a strict reading — the driver does invoke `--gate-sha-only` exactly once —
but it is the same uppercase `ONCE` idiom as `sbatch:62` (row 17) and `arena:560` (row 18), both of
which the erratum rewrites, and it **cites §13** for a claim §13's amended sentence no longer makes in
that form. It is also `--help` output: after landing, `gate_sha`'s docstring 670 lines earlier reads
*"the first of two passes"* and the flag's own help still reads `ONCE`. Unlike §1's declared
non-process residue, this site was never adjudicated either way.

**Repair.** A row, with either disposition — amended, or marked **CORRECT** with the reason. §1's
subtraction standard requires it to appear.

### I-3. §8's total equation and risk row have no site rows, and landing §4's numbers without moving `2642.3` breaks §8's own stated summation discipline.

*Attaches to:* `V15E1:1569-1571`, `:1609-1610`; §1c; §4's table; §7's V15E2 row (*"§8 (1d, 1g, totals, risk row)"*).

`V15E1:1569-1571` prints the total as a sum: `2642.3 + 1.0 + 0.7 + 0.1 (Phase 7z) + 1.3 (Phase 1f) +
3.8 (Phase 1g) + 7.0 (Phase 4) + 1013.8 (Phase 3) = 3670.0`, *"where `2642.3` is the sum of every row
except the seven named."* Phase 1d's `0.1 s` is inside that residue and Phase 1g's `3.8` is a named
term, so the re-price moves **two** literals: `2642.3 → 2642.4` and `3.8 → 7.6`.

`2642.3 + 1.0 + 0.7 + 0.1 + 1.3 + 7.6 + 7.0 + 1013.8 = 3673.8`
`2642.4 + 1.0 + 0.7 + 0.1 + 1.3 + 7.6 + 7.0 + 1013.8 = 3673.9` ✓

An implementer landing from §1 (which has no row for either literal) plus §4 (which gives the new
totals with no sites and no `2642.3`) writes an equation that sums to `3673.8` beneath a stated total
of `3673.9`, breaking §8's own **M-2, adopted:** *"the printed product column now sums to the total
directly."* The risk row at `:1609-1610` (`4683.8`, `7725.2`, `78.1 min`, `128.8 min`) is in the same
position: §4's table carries its replacements, §1 carries no site.

For completeness of my own subtraction: §12's predicate rows at `V15E1:1810`, `:1811` and the *"Why
`mints_executed` and not `66`"* paragraph at `:1817-1821` are also sites stating quantities this
erratum moves and are also absent from §1 — but unlike the above they **are** covered by §7's V15E2
row (*"§12 (rows 4, 5, 8, 9 …)"*, and I confirm §12's rows number 4 = `dev_path_opens`,
5 = `dev_label_materialisations_outside_decisions`, 8 = processes reporting, 9 = predicate coverage),
so they will not be lost at landing. I record them so the gap is measured rather than asserted.

**Repair.** Rows for `V15E1:1569-1571` (naming `2642.3 → 2642.4` explicitly) and `:1609-1610`.

### M-1. §2 overstates the C09 precedent's form: C09 does publish all six counters as integers, just not in its aggregate `measured` block.

§2 says the C09 block puts three counters in `by_construction_zero` *"as narrative strings, **never as
integers**."* Read from the artifact rather than the source, `GATE_LEDGER` also carries a `per_process`
list and an `arena_process_counts` dict, both of which publish **all six** keys as integers for every
one of the 39 processes. The claim is true of the aggregate `measured` block, which is the block v3's
prescription actually restructures (`c06_falsifier_arena.py:447`, `self.ledger = dict(tot)`), so
nothing in the repair changes. But the erratum cites the precedent as literal and it is not.

### M-2. The subtraction rests on a pattern that structurally cannot match ordinals, and the document does not say so.

`\b7[234]\b` cannot match `73rd` — row 2's own site. I confirmed by an independent ordinal sweep that
no ordinal is in fact missed, so this costs nothing today. But §1 offers the residue *"stated so the
subtraction is checkable"*, and a reader checking it with the stated pattern would not reproduce row
2. One clause naming the second pattern closes it.

### M-3. §7's delta does not say the arena **derives** the dev-like factor rather than reading the config's literal `4`.

§3 is explicit that the first factor is *"[d]erived from the config's own digest table, so it cannot
drift from it"*, and I verified that derivation reproduces. But the landing instrument is §7, whose
arena row says only *"two-term `dev_path_opens`"* and *"expectations asserted against
`cfg["ledger"]`"*, while row 20 puts `"expected_sha_dev_opens": 4` into the config as a literal. An
implementer reading §7 alone can satisfy both by having the arena *read* the `4` — which reintroduces
the trusted literal §3 exists to remove, one key lower down, exactly as round 2's I-1 found for the
`+1`. One sentence in §7: the arena derives the count from the digest tables and **asserts** it
against the config's declared `4`.

---

## OBLIGATIONS FOR A V4 THAT WOULD CARRY GO

1. **Design pointer** (H-1): rows for `config:5`, `:6`, `:7`, `arena:29-30`, `sbatch:14`;
   `design_document` → V15E2, `design_sha256` → V15E2's digest computed after V15E2 is written, and
   the landing order stated.
2. **Projection literal** (H-2): rows for `arena:29`, `mint:112` (`2929.9 → 3673.9`, comment updated)
   and `V15E1:1631-1633`; §4's *"both"* becomes the full set.
3. **Row 2's ordinal** (H-3): drop the ordinal; make §7.2's replacement ordering-independent so it
   cannot drift against §13's declared order or row 11's `1+66+6+1`.
4. **`V15E1:1304`** (I-1): a row amending `U11`'s closing accounting clause to cover both
   arena-class startups.
5. **`arena:1232`** (I-2): a row, with either disposition, stated.
6. **§8's summation and risk row** (I-3): rows for `:1569-1571` (naming `2642.3 → 2642.4`) and
   `:1609-1610`.
7. **Three minors** (M-1, M-2, M-3): correct the C09 form claim; disclose the ordinal hole in the
   stated pattern; put "derives, asserts against" into §7's arena row.
8. **Carry forward unchanged and at full strength**, everything I verified above: the two-term
   `dev_path_opens` with both factors derived over the concatenated iterable; the `74 = 1+66+6+1`
   inventory and its resume-stability; the uniform criterion and all six counter dispositions with
   their source-verified warrants; `int("GATE-SHA" in self.gates)`; `cfg["ledger"]` cross-check;
   `verify_predicate` called and published; option (iv) declined with the TOCTOU reason written into
   §6; the `argv`-based pass audit and its HALT; the sbatch's `C06_MINTS_EXECUTED` export; row 26's
   dead-line deletion; the rejections of (ii) and (iii); the M-2 referral; and §4's branch-(a) figures,
   every one of which I re-derived.

Obligations 1–3 and 6 are design-lineage; 4, 5 and 7 are a row each plus one sentence. The delta stays
bounded: the code change grows by two literals (`mint:112`, `arena:29`) and one config key pair.

---

## ENUMERATION DIFF — MY HIT LIST AGAINST THE TABLE

**Sites I found that §1's 26 rows do not contain:**

| # | site | quantity | finding |
|---|---|---|---|
| 1 | `config:5` `design_document` | names V15E1 | H-1 |
| 2 | `config:6` `design_sha256` | `0b446b91…` — **wrong today**, published on the verdict face | H-1 |
| 3 | `config:7` `design_status` | *"GO at round 15 + ERRATUM 1"* | H-1 |
| 4 | `arena:29-30` | `3670.0` / `4587.5` + names V15E1 | H-1, H-2 |
| 5 | `sbatch:14` | *"Frozen design: …V15.md"* | H-1 |
| 6 | `mint:112` | `PROJECTED_SECONDS = 2929.9` — **wrong today**, 66 processes' denominator | H-2 |
| 7 | `V15E1:1631-1633` | §9's `2929.9`/`3670.0` + the two-file claim | H-2 |
| 8 | `V15E1:1304` | §7.7 `U11` — *"the arena's is priced once at §8 Phase 1g"* | I-1 |
| 9 | `arena:1232` | the fifth `ONCE` | I-2 |
| 10 | `V15E1:1569-1571`, `:1609-1610` | §8's total equation (`2642.3`, `3.8`) and risk row | I-3 |

**Sites absent from §1 but covered by §7's delta** (recorded, not charged): `V15E1:1810`, `:1811`,
`:1817-1821` (§12 rows 4 and 5 and their warrant paragraph).

**The table's `correct text` column, checked for new contradictions:** one row fails — **row 2 against
rows 6 and 11** (H-3). Rows 1, 3, 4, 5, 7–26 produce text consistent with their neighbours and with
each other; row 1's *"72"* is correctly left alone (66 mints + 6 fidelity is right at 74); rows 10, 11,
12 and 6 agree on `1+66+6+1`; row 20's two-term formula agrees with §3's derivation and with row 12's
`cfg["ledger"]` read; row 3's `2 × U7 = 0.2 s` and row 4's `2 × U11` agree with §4's `3673.9`.

---

## WHAT V3 STILL GETS WRONG — SUMMARY PARAGRAPH

v3 is right about everything that decides the erratum, and it discharged round 2's hardest obligation
— the uniform counter criterion — better than round 2 asked, taking the strict branch on the two
vacuous belts and giving the reason. Its `73`/`74` sweep is exact: I ran it and the residue subtracts
to zero in both directions, which is the first time in this lineage's record that a completeness claim
has been verifiable rather than asserted. Its error is that it ran **one** of the sweeps §1 promises.
§1's heading says *every site stating a quantity this erratum moves*; the pattern it actually carried
through, and the only one whose residue it states, is `\b7[234]\b`. The consequence is not a fourth
predicate collision — the process count is now consistent everywhere it appears — but it is the same
shape one register down: the erratum re-prices §8 and names two of the four places `PROJECTED_SECONDS`
lives, one of the two it omits having drifted `740.1 s` from the arena at CODE-R1 and never been
reconciled, so 66 of the 74 processes are today publishing a heartbeat ratio against a superseded
denominator; and the erratum necessarily moves the design pointer to V15E2 without noticing that the
pointer is *already* wrong, naming V15E1 at a digest V15E1 lost at CODE-R1, on the face of every
verdict and HALT artifact the battery can emit. Both were found by sweeping the same five files for
the other quantities the erratum moves. And in the one place where the table prescribes the process
inventory in prose rather than in a number, it prescribes it wrongly: row 2 calls the driver leg *"the
73rd process"* while rows 6 and 11 put it first in the order — a contradiction between two repairs in
the same table, in the erratum whose entire subject is a document contradicting itself about that
process. None of this touches the arithmetic, the demotion, the audit mechanism or the `74`. It is
still the difference between an erratum that ends this defect and a fourth round.

---

## BLINDNESS AND EDIT STATEMENT

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` was
called zero times; no arm was built; no mint was run or read; no GPU, no SLURM job, no commit, no
`TARGET_STATE.json` edit; `artifacts/c06_falsifier/` was never created and no mint, arena or fidelity
process was launched. **I did not run the `--gate-sha-only` leg** — round 2's measurement of it is
carried on its record, not re-taken.

Compute used: `sha256sum` over the five artifacts and the three proposals; file reads; `grep` sweeps
over the five artifacts for `7[234]` (bounded and unbounded), ordinals, `once`/`ONCE`/`one span`/
`twice`/`single pass`, process-count phrasing without a numeral, `mints_executed`/`dev_path_opens`/
`+ 0`, `gate_sha`, and every projection literal; one repo-wide grep for the three uninstrumented
counter names; one enumeration of the config's own digest tables under `c09guard.is_dev_like` /
`is_test_like`, which opens the config and no cache; one `json.load` of C09's banked
`C09_A0_DECISION.json`; and arithmetic.

**No file outside this review was edited.** All five artifacts still carry their post-CODE-R1 hashes,
verified against the implementation record's table at `:324-328`:

| path | sha256 | matches record |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` | ✓ |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` | ✓ |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` | ✓ |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` | ✓ |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` | ✓ |

The arena still implements `dev_path_opens == mints_executed + 0` (`:469-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`. **The
battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under v1, v2
or v3 as specified.**
