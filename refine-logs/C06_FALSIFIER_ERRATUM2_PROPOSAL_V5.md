# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v5**

**Supersedes** `…_PROPOSAL_V4.md` (`0b4940416abd1fb4bf79…`) → `…_V3.md` (`48f4e0153103cc608884…`) →
`…_V2.md` (`4225bea3cc9907d38e2e…`) → `…_PROPOSAL.md` (`f063c388c4afabdb7964…`).
**All four stay on disk, byte-unmodified.**
*Against:* `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudications:* round 1 (0C/3H/3I/3M) → v2; round 2 (0C/2H/3I/3M) → v3; round 3 (0C/3H/3I/3M) → v4;
**round 4 (0C/2H/3I/4M)** → this.
*Status:* **PROPOSAL. Nothing is landed.** All five artifacts carry their post-CODE-R1 hashes.

---

## 0. What round 4 settled, and the one thing it blocked

**Re-derived independently by round 4 and carried forward unchanged:** the two-term `dev_path_opens`
with both factors over the concatenated iterable (21 files, 2 dev-like, 0 test-like, `frozen_sha256`
contributing zero — `4 = 2 × 2`, and `sha256_of` at `arena:81-86` opening each path exactly once,
which is what makes the second factor a *pass* count); the `74 = 1 + 66 + 6 + 1` inventory read off
the sbatch; the uniform counter criterion with all six dispositions and the digest-pinned
`c09guard` increment map (`:97`, `:102`, `:106`, and `c09guard.py` itself inside `frozen_sha256`);
`int("GATE-SHA" in self.gates)` plus the `argv` pass audit (`_flush` records `sys.argv` at `:129`);
the `cfg["ledger"]` cross-check; `verify_predicate` called; option (iv) declined with the TOCTOU
reason; the rejections of (ii) and (iii); `C06_MINTS_EXECUTED`; row 26†'s dead-line deletion; row 2's
ordinal-free §7.2 text; rows 26 and 27 (`arena:4`, `mint:4`); and **every figure in §4, exact to the
last decimal**, including the `3673.8`-vs-`3673.9` point that makes row 34 necessary.

**Round 4's own verdict on round 3: 9-for-9, three of them past what was asked.** Nothing in that
tranche is reopened here.

**Blocked, for the fourth time in this line, on the same delivery gap.** v4 promised a stated,
reproducible subtraction across seven sweep families and delivered it for two. Sweeps C, D, E and G
did not close under their own declared patterns; sweep F declared no pattern at all. Round 4 traced
every unaccounted hit and confirmed **no site is lost at landing** — which is why it was High and not
Critical — but §1's standard is *"completeness is verified by subtraction, not by trust,"* and four
of seven subtractions could not be reproduced by a reader doing exactly what §1 invites.

**And round 4 found the quantity that gap was hiding.** §3's new gate moves the `GATE-SHA` artifact
count `37 → 38`; v4 wrote that down and disposed of it as *"a reported figure … not a binding one"*
— against §1's own rule that even a **CORRECT** site gets a row. One of its unrecorded sites is
`V15E1:1300`, **the definition of the unit row 3 re-prices as `2 × U7`**.

**v5's answer is §8: nine sweeps, every one re-run against the files, every hit printed, every hit
charged, every subtraction stated — and a global partition that closes in both directions.** There
is no design judgement in that appendix; it is grep, paste, subtract, and its numbers are produced
by a script rather than transcribed from a table.

**What the re-run found that v4 did not.** Running the patterns wider rather than narrower produced
**three** further sites, two of them wrong today:

* **`mint:118`** — the `heartbeat()` docstring states the mint phase is *"`85.6 %` of §8's budget"*.
  CODE-R1 H-4 moved that share to `68.3 %` (`V15E1:1576`, `:1607`). **Wrong today**, in the same
  docstring as the live-wrong `PROJECTED_SECONDS`, and its neighbour `mint:117` is already row 13.
* **`mint:116`** and **`mint:209`** — the same false process-coverage claim round 4 charged at
  `V15E1:1628-1629` (I-3), in the mint's own docstring and in its `--progress` help text. My first
  sweep-I pattern was case-sensitive and missed `mint:116`'s *"EVERY python process"*; the recorded
  pattern is `grep -i`.
* **`config:217`** (`mints_present_before_arena`) is a `ledger` key my sweep-G pattern did not name,
  and **`arena:559`** and **`V15E1:1786-1787`** are the two places other than §6 that state
  `GATE-SHA`'s *scope*, which §3 enlarges. All three are now in the patterns and in rows.

---

## 1. The accounting rule, stated before the rows

> **Every site returned by any sweep is charged to exactly one row or exactly one declaration.**
> A row may collect sites returned by several sweeps — `V15E1:1904` is a `73` site *and* a
> `projected` site — so the same row appears in more than one sweep's table with the **same** charge.
> Because the charge is unique, a site counted in two sweeps is not counted twice: the per-sweep
> subtraction is `hits = rows + declarations` with no residue, and the **global** subtraction is over
> the deduplicated union.

This is the repair for round-4 H-1's mechanism finding — *"the row-to-sweep assignment is not a
partition."* It now is one, at the level of the charge. §8 prints both subtractions, and the two
emptiness checks that make the partition an identity rather than a claim: **no site returned by a
sweep is charged to nothing**, and **no row is charged to a site no sweep returns**.

**Multi-line rows are declared everywhere they occur**, not only where v4 happened to notice: rows
7, 12, 15, 16, 23, 24, 32, 34, 35, 36, 39, 41, 43, 44, 46, 47, 49, 53, 57, 58, 59 and 61 each span
more than one line, and every line of each is a separate charged site in §8.

**One row has no site by construction.** Row 48 *adds* a line to the sbatch (the
`C06_PROJECTED_SECONDS` export). A sweep over the current files cannot return a line that does not
exist yet; §8's orphan check therefore excludes row 48 and says so.

---

## 2. FULL ENUMERATION — 63 rows

Rows marked **CORRECT** need no edit and are listed anyway, because completeness is verified by
subtraction, not by trust.

### 2.1 The process inventory — `73 → 74`

| # | site | current | correct | why |
|---|---|---|---|---|
| 1 | `V15E1:1197` §7.2 | *"those **72** processes"* | **unchanged — CORRECT** | 66 mints + 6 fidelity = 72 is exactly right at 74 |
| 2 | `V15E1:1198` §7.2 | *"The **73rd** process, the arena…"* | *"**The remaining two processes** — the `--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8 Phase 1g"* | **round-3 H-3.** No ordinal: the paragraph's job is to say which processes are *not* priced inside the mint units, and that is true under any ordering, so it cannot drift against §13 again. `72` (row 1) `+ 2` `= 74`, agreeing with rows 6, 10, 11 and 12 |
| 6 | `V15E1:1839` §13 | *"**73** processes in the order 66 mints → 6 fidelity → 1 arena"* | *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena"* | |
| 7 | `V15E1:1903-1904` §13.1 item 12 | *"the `buffering=1` handle never re-wrapped … append-without-interleaving across all **73** processes"* | *"…across all **74** processes, **of which the 68 this lineage authors append through a handle opened `buffering=1`; the six `headspace_fidelity.py` processes are sha-frozen and third-party and the bash driver writes their line (`sbatch:128-129`)**"* | **round-4 I-3.** The count moves and the coverage assertion it sits inside is corrected in the same breath, rather than restated over a larger set |
| 8 | `V15E1:2017` §13.1 item 28 | *"active in all **73** processes"* | *"in all **74**"* | the driver leg is where the guard first installs |
| 9 | `sbatch:16` | *"73 processes…"* | *"74 processes: 1 GATE-SHA driver leg → 66 mints → 6 fidelity → 1 arena"* | |
| 10 | `config:41` | `{"mints":66,"fidelity":6,"arena":1,"total":73}` | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | |
| 11 | `config:222` | `"processes_reporting": {"expected":73,"binding":true}` | `{"expected":74,"decomposition":"1+66+6+1","binding":true}` | |
| 12 | `arena:465-467` | `!= 73` and its message (**three lines**) | `!= 74`, read from `cfg["ledger"]` (§2.6) | `:465` and `:466` are sweep-A hits; `:467` is the message's continuation line and is a sweep-G hit. All three are row 12 |
| 13 | `mint:117` | *"**72 of 73** processes previously wrote nothing"* | *"**73 of 74**"* | docstring narrating the pre-CODE-R1 state |

### 2.2 The `--gate-sha-only` leg is a second priced process

| # | site | current | correct | why |
|---|---|---|---|---|
| 3 | `V15E1:1547` §8 Phase 1d | *"`GATE-SHA`, once in the driver \| `1` \| `U7` \| `0.1 s`"* | *"`GATE-SHA`, **twice** — driver leg and arena \| `2` \| `U7` \| `0.2 s`"* | the only *"once"* site carrying a numeric count column. **`U7` itself is unchanged** — see §5 |
| 4 | `V15E1:1550` §8 Phase 1g | *"**`1`** — the arena process alone … `66+6+1 = 73` accounts for **every** process"* | *"**`2`** — the `--gate-sha-only` driver leg and the arena … `1+66+6+1 = 74` accounts for every process §13 declares"* | the compute-accounting twin of the §12 defect |
| 5 | `V15E1:1645` §9 | *"the **arena's own startup** is **the one span**…"* | *"the `--gate-sha-only` driver leg's startup is the first such span and the arena's the second; both bounded by the same arena-class band"* | measured false — the driver leg instantiates `Heartbeat` at `arena:1266` and emits three lines before `--gate-sha-only` returns, and it runs `load_frozen()` first (round 4 verified both) |
| 28 | `V15E1:966` §6 `GATE-SHA` row | *"every frozen import, input cache **and the sixteen banked artifacts of §11** … **once in the sbatch driver**"* | **both limbs move**: *"…the sixteen banked artifacts of §11 **and the design document itself (§3)**"*, and *"**twice** — once in the sbatch driver before any other process, and again in the arena at the point of use"*, **with the TOCTOU reason** | round-1 H-2 for the pass count; §3 for the scope. The reason makes the second pass a decision, not an accident |
| 29 | `V15E1:1840` §13 | *"`GATE-SHA` **once in the driver** before any of them"* | *"…in the driver leg before any of them **and again in the arena**"* | |
| 30 | `sbatch:17` | *"GATE-SHA runs ONCE in this driver"* | *"…runs in this driver before any of them, and again inside the arena"* | |
| 31 | `sbatch:62` | *"# GATE-SHA, ONCE, before any other process"* | *"# GATE-SHA, first of two passes, before any other process"* | |
| 32 | `arena:559-560` | docstring *"§6: every frozen import, input cache AND the sixteen banked artifacts. / ONCE, in the driver, before any other process"* | *"…the sixteen banked artifacts **and the design document (§3)**. / the first of two passes; the arena repeats it at the point of use"* | **two lines, both limbs** — the same pair as row 28, in the code |
| 33 | `arena:1232` | `--gate-sha-only` help: *"the sbatch driver calls this **ONCE** before any other process (§13)"* | *"…calls this once, as the **first of two** `GATE-SHA` passes (§6, §13)"* | **round-3 I-2.** `--help` output citing §13 for a claim §13 no longer makes in that form |

### 2.3 The projection literals

| # | site | current | correct | why |
|---|---|---|---|---|
| 14 | `arena:46` | `PROJECTED_SECONDS = 3670.0` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`, **asserted three-way** (§6) | the heartbeat denominator |
| 15 | `arena:29-30` | docstring *"§8 projects `3670.0 s` (`4587.5 s` conservative) … (`…V15E1.md`)"* | **`3673.9 s` / `4592.4 s`**, and the document name → V15E2 | **two lines**: `:29` is a sweep-C hit, `:30` a sweep-D hit |
| 16 | `config:43-44` | `3670.0` / `4587.5` | **`3673.9` / `4592.4`** — **and `:43` becomes the single source** (§6) | **two lines** |
| 17 | **`mint:112`** | `PROJECTED_SECONDS = 2929.9  # §8 under ERRATUM 1` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))` (§6) | **WRONG TODAY.** `740.1 s` below the arena's — 66 of 74 processes publish `elapsed/2929.9` against the arena's `elapsed/3670.0`, in the phase that is `68.3 %` of the budget |
| 18 | `mint:127` | `elapsed / PROJECTED_SECONDS` | **unchanged — CORRECT** | the divide site is right; only its constant was stale |
| 58 | `arena:57`, `:61`, `:68` | `projected=PROJECTED_SECONDS`, `self.projected = float(projected)`, `elapsed / self.projected` | **unchanged — CORRECT** (**three lines**) | the arena's consumption chain. It carries no literal of its own and moves with row 14. Listed for the same reason row 18 is: the defect lives at the constant and is *consumed* here |
| 19 | `V15E1:1631-1633` §9 | *"(ERRATUM 1 set it to `2929.9 s`; CODE-R1 H-4 sets it to `3670.0 s`… **the denominator is pinned to §8 by name, so it tracks automatically**; the literal in `c06_falsifier_arena.py` and `configs/c06/…json` is updated with each correction.)"* | carries **`3673.9 s`**, **deletes the false *"tracks automatically"* clause**, and states the single source and the assertion (§6) | **round-3 H-2.** This sentence re-published the two-file claim that caused the drift, and its *"tracks automatically"* was false for exactly the reason round-4 I-2 gives |
| 55 | **`mint:118`** | *"the mint phase — **`85.6 %`** of §8's budget — dark for its whole span"* | **`68.3 %`** | **WRONG TODAY, found by this round's sweep F.** CODE-R1 H-4 moved the mint share from `85.6 %` to `68.3 %` (`V15E1:1576`, `:1607`); row 13 is already making the process counts in this same docstring current, so leaving the share stale would produce a half-current sentence |

### 2.4 The design pointer

| # | site | current | correct | why |
|---|---|---|---|---|
| 20 | `config:5` | `"design_document": "…DRAFT_V15E1.md"` | `"…DRAFT_V15E2.md"` | |
| 21 | **`config:6`** | `"design_sha256": "0b446b91675fd4ff8aea…"` | V15E2's digest, **derived at freeze time** (§3) | **WRONG TODAY.** Names V15E1 at the sha it had when Erratum 1 landed; CODE-R1 moved it to `8cde58aa…` and this was not updated |
| 22 | `config:7` | `design_status`: *"GO at round 15 … + ERRATUM 1 landed"* | *"… + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2 landed"* | |
| 23 | `arena:1249-1250` | `emit_halt` writes `cfg.get("design_document")` / `["design_sha256"]` | **publishes BOTH**: `sha256_declared` from the config and `sha256_derived` from `self.reports.get("design_sha256_derived", "NOT_DERIVED")` (**two lines**) | **round-4 I-1.** On the one occasion the new gate fires, declared ≠ derived *by construction*; v4 marked this site CORRECT while giving as its reason that this is the path putting a stale digest on every HALT artifact |
| 24 | `arena:1433-1434` | the verdict face writes the same pair | same repair (**two lines**) | same reason, for every verdict artifact |
| 25 | `sbatch:14` | *"Frozen design: …DRAFT_V15.md (GO at round 15)"* | *"…DRAFT_V15E2.md"* | two revisions stale |
| 26 | **`arena:4`** | *"Frozen design: …DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | v4's own find, confirmed at source by round 4 |
| 27 | **`mint:4`** | *"Frozen design: …DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | v4's own find, confirmed at source by round 4 |

### 2.5 §8's equation, shares, risk row and the two units

| # | site | current | correct | why |
|---|---|---|---|---|
| 34 | `V15E1:1569-1571` | `2642.3 + 1.0 + 0.7 + 0.1 + 1.3 + **3.8** + 7.0 + 1013.8 = 3670.0`; `× 1.25 = 4587.5` | **`2642.4`** + 1.0 + 0.7 + 0.1 + 1.3 + **`7.6`** + 7.0 + 1013.8 = **`3673.9`**; `× 1.25 = **4592.4`** (**three lines**) | **round-3 I-3.** **Two** literals move: Phase 1d's `0.1 → 0.2` lives *inside* the `2642.3` residue, and Phase 1g's `3.8` is a named term. Landing §4's totals without moving `2642.3` writes an equation summing to `3673.8` under a stated `3673.9` |
| 35 | `V15E1:1576-1577` | *"mints fall from `85.6 %` to `68.3 %` and Phase 3 rises from `9.3 %` to `27.6 %`"* | **unchanged — CORRECT** (**two lines**) | `2508.3/3673.9 = 68.27 %` and `1013.8/3673.9 = 27.59 %` still round to `68.3`/`27.6` |
| 36 | `V15E1:1609-1610` risk row | *"2× miss `4683.8 s = 78.1 min`, 5× miss `7725.2 s = 128.8 min`"* | **`4687.7 s = 78.1 min`**, **`7729.1 s = 128.8 min`** (**two lines**) | second-order consequence; minutes unchanged at one decimal |
| 37 | `V15E1:1304` §7.7 `U11` | *"Both are inside the mint units and inside `U9`; **the arena's is priced once at §8 Phase 1g**"* | *"…the **two arena-class startups — the `--gate-sha-only` driver leg and the arena — are priced at §8 Phase 1g**"* | **round-3 I-1.** The section that *defines* the unit Phase 1g now prices **twice** accounted for one instance |
| 56 | `V15E1:1574` | *"`2929.9 − 273.7 + 1013.8 = 3670.0`"* | **unchanged — CORRECT, and declared historical** | **round-4 M-3.** v4 called this *"inside row 34's replacement block"*; row 34's block is `:1569-1571` and this is three lines outside it, so an implementer editing 1569–1571 would not touch it. It is CODE-R1 H-4's provenance arithmetic, true of the total *as it then stood*, and correct to leave alone — but its status is now on the record instead of mis-described |
| 57 | `V15E1:1607-1608` | *"Mints are **`68.3 %`** … **Phase 3 is now `27.6 %`**"* | **unchanged — CORRECT** (**two lines**) | v4 declared these as *"row 35"*; row 35's site is `:1576-1577`. This is a **second, separate** statement of the same two shares in §8's risk paragraph, and it needs its own row for the same reason row 35 does |

### 2.6 The ledger

| # | site | current | correct | why |
|---|---|---|---|---|
| 38 | `config:218` | `"dev_path_opens": {"expected":"mints_executed","binding":true}` | `{"expected":"mints_executed + expected_sha_dev_opens","expected_sha_dev_opens":4,"derivation":"(dev-like files in the concatenated iterable gate_sha hashes = 2) × (GATE-SHA passes = 2)","binding":true}` | **the core repair** |
| 39 | `arena:433`, `:438`, `:468`, `:469`, `:471`, `:475` | the frozen `+ 0` predicate and its `ERRATUM REQUIRED` message (**six lines**) | the two-term predicate; **the message's decomposition becomes the derivation**. **Round-3 M-3:** the arena **derives** the dev-like count from the digest tables and **asserts** it against the config's declared `4`; it must not merely *read* the `4` | |
| 40 | `config:219` | `dev_label_materialisations_outside_decisions` binding `mints_executed` | moved to a `by_construction` block as a **warranted string** | §3's criterion |
| 41 | `config:215`, `:216`, `:217`, `:220`, `:221` | all published as `{"expected": N, "binding": …}` (**five lines**) | `:215` `test_path_opens` and `:221` `banked_trainlog_opens` **unchanged — CORRECT** (both instrumented); `:217` `mints_present_before_arena` **unchanged — CORRECT** (computed at `arena:628`); `:216` and `:220` move to `by_construction` strings | §3's criterion applied to every key in the block, so the reader sees the whole partition rather than the two that move |
| 42 | `config:211` | `_code_review_r1.blocked_on_erratum_2` narrative | replaced by an `erratum_2` block recording what landed | |
| 43 | `V15E1:1807-1813` §12 rows | the six counter rows plus `mints_present_before_arena` (**seven lines**) | the two-term `dev_path_opens`, the by-construction marks on the three uninstrumented counters, and M-3's *"top-level processes only"* sentence | **round-3 I-3 recorded these as covered by §7's delta; they are rows** |
| 44 | `V15E1:1817`, `:1819`, `:1821` §12 | *"Why `mints_executed` and not `66`"* — the resume warrant (**three lines**) | amended to carry the two-term form **and** the spawn/skip fact that makes an exact `74` safe where C09 refused to bind `39` | the warrant paragraph must move with the predicate |
| 45 | `arena:449` | `self.ledger["processes_reporting"] = len(procs) + 1` | **unchanged — CORRECT** | round 4 checked this and it needs no edit: the 6 fidelity processes and the `--gate-sha-only` driver leg all write `c09guard` ledger files, so `len(procs)` is **73** and the `+ 1` for the not-yet-flushed arena yields **74** after row 12's repair. It is the line that *computes* what row 12 compares, and v4 left it unrecorded |
| 46 | `arena:456-457` | `if tot.get("test_path_opens", 0) != 0` and its message (**two lines**) | **unchanged — CORRECT** | `test_path_opens` **is** instrumented (`_guarded_open:97`), so it stays measured and binding; §3's criterion does not touch it |
| 59 | `arena:458-462` | the `test_label_materialisations` and `dev_or_test_labels_into_decision_quantities` assertions (**five lines**) | **RETAINED VERBATIM** as vacuous defence-in-depth; only the *publication* moves to `by_construction` | stated explicitly so the code lineage does not read §3's *"never binding"* as licence to **delete a tripwire**. Nothing increments these keys, so `tot.get(k, 0)` is `0` and the assertions cannot fire; retaining them costs nothing and preserves the guard if a future lineage ever instruments them |
| 47 | `arena:1419-1420` | `mints_executed = int(os.environ.get("C06_MINTS_EXECUTED", bat.reports["mints_present_before_arena"]))` (**two lines**) | **unchanged — CORRECT** | the arena **already** reads the export with a fallback. What is missing is the export itself — row 48 |
| 48 | `sbatch` — **an addition, no existing site** | the sbatch never sets `C06_MINTS_EXECUTED`, so `arena:1419` silently falls back to `mints_present_before_arena` | export `C06_MINTS_EXECUTED` from an executed-vs-skipped counter in the mint loop | this is why row 47 is CORRECT and this row exists. **No sweep can return this site**, because the line does not exist yet; §8's orphan check excludes it by name |
| 26† | `arena:1418` | `mints_executed = sum(1 for _ in [None])   # placeholder` | **delete** | v3's own find, carried |

### 2.7 The `GATE-SHA` artifact count and scope — **round-4 H-2**

**The branch is stated: the design document DOES increment `n`.** §3's mechanism hashes it through
the same `sha256_of`, in the same loop, and compares it against a declared digest — that is exactly
what the other 21 hashed artifacts get, so it is one of them. The count moves `21 + 16 = 37` to
`22 + 16 = 38`, and **every site of that count moves with it.**

| # | site | current | correct | why |
|---|---|---|---|---|
| 49 | `config:251-255` | `_gate_sha_count`: `imported_modules 7, read_for_definitions 6, input_caches 8`, `total_§11_digests 21`, `banked_unhashed_artifacts 16`, note *"§11 declares 37 = 7 + 6 + 8 … 21 + 16 = 37"* (**five lines**) | gains `design_document: 1`; `total_§11_digests` **`21 → 22`**; note becomes *"§11 declares **38** = 7 + 6 + 8 imported/read/cached digests **plus the design document (ERRATUM 2 §3)** plus the 16 banked artifacts … **22 + 16 = 38**"* | without this the config carries a **false arithmetic statement** in one of the five artifacts |
| 50 | `arena:563` | `+ list(self.cfg["frozen_sha256_input_caches"].items()):` | the design document is appended to the concatenated iterable — **this is the code site implementing §3** | round 4 verified this is the iterable whose dev-like membership fixes `expected_sha_dev_opens`; adding a `.md` in `refine-logs/` leaves that count at **2**, so the second factor and the `4` are unchanged |
| 51 | `arena:585` | `self.reports["gate_sha_artifacts"] = n` | **unchanged — CORRECT, but its published value moves `37 → 38`** | **round-4 M-4.** `_gate_sha_count` is the *config key*; `gate_sha_artifacts` is the *runtime symbol*, published in the verdict's reports block. Both move; naming only one hides the other |
| 52 | `V15E1:1300` §7.7 `U7` | *"`GATE-SHA` over **all 37 §11 artifacts** (8 caches + 13 modules/configs + 16 banked…)"* | *"over **all 38** (8 caches + 13 modules/configs + 16 banked + **the design document**)"*, cost **unchanged at `0.13 s`** — see §5 | **the unit definition row 3 re-prices as `2 × U7`.** v4 moved the price twice while enlarging the object, in neither place |
| 53 | `V15E1:1786-1787` §11 | *"`GATE-SHA`'s scope is stated in §6 as the frozen imports and the input caches plus the sixteen banked artifacts above."* (**two lines**) | *"…plus the sixteen banked artifacts above **and the design document itself (ERRATUM 2 §3)**"* | §11's own scope sentence. Found by this round's sweep H; in none of v4's seven |
| 54 | `V15E1:2439` | *"**No `.py` source moved** — all **37** §11 digests recompute."* | *"— all **38** §11 digests recompute **as of this document's own freeze**; the 38th is this document, whose digest is by construction the one `config:6` carries"* | §16's compute record. Left alone it is a present-tense claim that is false the moment §11 has 38 artifacts, one of which is the document making the claim |

### 2.8 The progress-coverage claims — **round-4 I-3**

Round 4 measured this and I confirm at source: `headspace_fidelity.py` has **no** progress handle —
it opens at `:42`, writes JSON at `:113`, prints at `:115`, nothing else — and it is sha-frozen
(`config:229`) and run UNMODIFIED, so it is not fixable in code. The 6 fidelity processes' progress
lines are written by the **bash** driver at `sbatch:128-129` with a literal `-` in both the elapsed
and ratio columns. The claim *"every python process appends"* is therefore false for 6 of the 74,
and it is stated in **four** places, not one.

| # | site | current | correct | why |
|---|---|---|---|---|
| 61 | `V15E1:1627-1629` §9 | *"every python process appends through a handle opened `buffering=1`"* (**three lines**) | *"every python process **that this lineage authors** appends through a handle opened `buffering=1` — the 68 mint, driver-leg and arena processes; the six `headspace_fidelity.py` processes are sha-frozen and third-party, and the bash driver writes their line (`sbatch:128-129`) with `-` in the elapsed and ratio columns"* | round 4's own prescribed text, adopted |
| 60 | `mint:116` | *"§9 requires **EVERY** python process to append through a handle opened `buffering=1`"* | the same qualification | **found by this round's sweep I**, and only after the pattern was made case-insensitive |
| 62 | `mint:209` | `--progress` help: *"§9 progress file; every python process appends to it (H-3)"* | *"…every python process this lineage authors appends to it (H-3)"* | `--help` output carrying the same false claim |

---

## 3. The uniform counter criterion — unchanged, discharged in full at round 3

> **A ledger quantity is published as a measured integer on the verdict face if and only if some code
> path in this job increments it. A quantity no code path increments is published as a
> by-construction narrative string carrying its warrant, in a separate block, never an integer and
> never binding.**

| counter | incremented by | disposition | warrant (source-verified) |
|---|---|---|---|
| `test_path_opens` | `_guarded_open:97` | **measured, binding `0`** | — |
| `dev_path_opens` | `_guarded_open:102` | **measured, binding** at the two-term formula | — |
| `banked_trainlog_opens` | `_guarded_open:106` | **measured, reported** | — |
| `test_label_materialisations` | **nothing** | by-construction string | `_guarded_open` **raises** on a test path, so no test file is opened at all |
| `dev_label_materialisations_outside_decisions` | **nothing** | by-construction string | `lab_dev` occurs **exactly once** in the executed corpus — `headspace_mint.py:323` — and in neither the arena nor `headspace_fidelity.py` |
| `dev_or_test_labels_into_decision_quantities` | **nothing** | by-construction string | same write, read by no decision path; the arena never iterates `.npz` keys generically |

Round 3 audited this and confirmed **no reachable tripwire is removed**, adding that the guard which
would have to change to create one is itself **digest-pinned in `frozen_sha256`**, so `GATE-SHA`
HALTs before a modified guard could run. **Row 59 makes that concrete in the code**: the two
uninstrumented assertions are retained verbatim, so the criterion changes what is *published*, never
what is *checked*.

**Round-3 M-1, carried.** C09's `GATE_LEDGER` also carries `per_process` and `arena_process_counts`
blocks publishing **all six** keys as integers per process. The *"never as integers"* claim is true
only of the aggregate `measured` block — which **is** the block this prescription restructures
(`arena:447`).

---

## 4. The `design_sha256` mechanism, and its residual — **round-4 I-1**

**Round 3's H-1 is a stale-hash defect, and hand-updating it would only reset the clock.**

> **`GATE-SHA` gains one artifact: the design document itself.** The arena reads
> `cfg["design_document"]`, hashes the on-disk file, and compares against `cfg["design_sha256"]`,
> **HALTing on mismatch** with both digests in the message. Any future edit to the design not
> accompanied by a config update is caught **before any battery computation**, in the
> `--gate-sha-only` process, at zero cost.

Round 4 verified the ordering in `main()` — `gate_det1 → assert_guard_active → gate_sha →
load_frozen`, and only then `if args.gate_sha_only: return 0` — and that the driver leg is process 1
of 74 (`sbatch:63-64`, before the mint loop at `:67`). **The HALT is before the first mint, at zero
compute cost.**

**The residual, stated — this is I-1 and v4 overstated it.** `configs/c06/c06_falsifier.json` is
**not** in `frozen_sha256` (verified by enumeration by rounds 3 and 4). So `cfg["design_sha256"]` is
pinned by nothing inside the job, and the gate establishes **config↔disk parity only**:

* **Uncoordinated drift** — the design edited, the config not — is caught. This is exactly the
  CODE-R1 failure that produced the live-wrong `config:6`. ✔
* **Coordinated drift** — a post-freeze edit to the design *with* a matching config update — passes
  startup parity silently. ✘

**§3 must therefore not say *"removes the class"*; it removes the observed subclass.** What anchors
the declared digest outside the job is the freeze table in
`refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md`, which **no code path reads** — so the residual
is closed by procedure (the freeze record plus the separate code/resource review lineage), not by the
battery. V15E2 states that in those words.

**The publication path is closed** (rows 23, 24). The arena records `self.reports["design_sha256_derived"]`
at the moment it hashes, and both `emit_halt` and the verdict face publish `sha256_declared` **and**
`sha256_derived`, the latter via `.get(…, "NOT_DERIVED")` because `emit_halt` is reachable from
`GATE-DET1` before `gate_sha` has run. No artifact can then carry an unverified design hash without
saying so on its face. Round 4's incidental note is adopted into §3: `--out` defaults to
`artifacts/c06_falsifier/C06_VERDICT.json` and the sbatch invokes the driver leg without `--out`, so
a design-drift HALT in process 1 writes to the canonical verdict path — correct behaviour, newly
reachable at startup.

**Landing order, stated because the digest is self-referential.** V15E2 is written **first**; its
sha256 is computed **after** it is final; the config's `design_sha256` is the **last** edit of the
landing.

**Interaction with `expected_sha_dev_opens`, checked twice.** The design document is a `.md` in
`refine-logs/` — **not dev-like and not test-like** under `c09guard`'s predicates (round 4 ran the
predicates and confirms `False`/`False`) — so the dev-like count in the concatenated iterable stays
at **2** and `expected_sha_dev_opens` at **4 = 2 × 2**.

---

## 5. §8: re-priced — and what `U7` does and does not do

| row | before | after |
|---|---|---|
| Phase 1d `GATE-SHA` | `1 × U7 = 0.1 s` | **`2 × U7 = 0.2 s`** |
| Phase 1g arena-class startups | `1 × U11 = 3.8 s` | **`2 × U11 = 7.6 s`** |
| §8 residue term (row 34) | `2642.3` | **`2642.4`** — Phase 1d's `0.1 → 0.2` lives inside it |
| **total** | `3670.0 s` | **`3673.9 s`** |
| `× 1.25` | `4587.5 s` | **`4592.4 s`** |
| minutes | `61.2 / 76.5` | **unchanged at one decimal** |
| mint / Phase 3 share | `68.3 % / 27.6 %` | **unchanged** |
| `2×` / `5×` miss | `4683.8 / 7725.2 s` | **`4687.7 / 7729.1 s`** (`78.1 / 128.8 min`) |

**`U7`'s object grows and its price does not — measured, not asserted.** Round-4 H-2's obligation is
that if the design document increments `n`, *"row 3's `2 × U7` and `U7`'s stated object must move
together."* They do: row 52 moves the object `37 → 38`, row 3 prices `2 × U7`, and the added artifact
is a **188 061-byte** markdown file whose sha256 I measured on the login node over 7 repetitions —
**median `0.000164 s`** (min `0.000158`, max `0.000371`). Against `U7`'s `0.13 s` that is `0.13 %` of
the unit and **invisible at two decimals**, so `U7 = 0.13 s` is unchanged, `2 × U7 = 0.2 s` stands,
and **§8's total is unaffected by the §3 mechanism**. The measurement is stated rather than waved at
because the alternative is exactly the defect H-2 names.

**`PROJECTED_SECONDS` has FOUR literal sites, not two** (round-3 H-2): `arena:46`, `arena:29`,
`config:43-44` and **`mint:112`** — rows 14–17 — plus §9's sentence at `V15E1:1631-1633` (row 19).

**Branch (a) chosen, reason unchanged:** Phase 1g exists *because* round-11 I-1 found a per-process
cost no row priced; leaving a second one knowingly unpriced in the row created to fix the first is
the same defect one iteration later.

**For the code lineage, not repaired here:** the `--gate-sha-only` leg runs `load_frozen()` before
returning, which is what makes it a *full* arena-class startup. Pricing the full startup is
conservative and is what this erratum carries.

---

## 6. The heartbeat denominator gets a single source — **round-4 I-2**

Round 4's objection is correct and is adopted rather than argued with. §4 diagnoses hand-carried
digests as a class and builds a structural fix; applying the opposite remedy to the one quantity in
this erratum where hand-carrying has **already measurably drifted**, across 66 of 74 processes, is
not defensible three sections later.

**Prescription — derive-and-assert, the same shape as §4:**

1. **`configs/c06/c06_falsifier.json:43` (`projected_seconds`) is the single source.** Row 16.
2. **`sbatch` exports `C06_PROJECTED_SECONDS`**, parsed from that key, before any process launches —
   one line beside the `C06_MINTS_EXECUTED` export of row 48.
3. **`mint:112` and `arena:46`** become `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`.
   The literal survives **only** as a hand-run fallback and V15E2 says so in those words.
4. **The arena asserts all three agree** — environment, module constant, `cfg["projected_seconds"]` —
   and HALTs on mismatch, so a future re-price that touches one place cannot reach a run.
5. **Row 19** names the single source and the assertion, and **deletes** §9's false *"the denominator
   is pinned to §8 by name, so it tracks automatically"*.

Round 4 traced all four legs and all **68** ratio-computing processes agree after the repair (the 6
fidelity processes compute no ratio; the bash driver writes `-`), so this changes no number — it
changes what keeps the number true.

---

## 7. Round 4's remaining findings

**H-1 — four subtractions that did not close, and a fifth with no pattern. Adopted in full.** §8 is
nine sweeps, each with its exact command, its raw hit list printed `file:line`, a charge for every
hit, and its stated subtraction, plus a global partition with both emptiness checks. Sweep F now has
a regex. Multi-line rows are declared wherever they occur. **Every count in this document is printed
by the script in §8, not transcribed.**

**H-2 — the `GATE-SHA` artifact count. Adopted, with the branch decided:** the design document
**does** increment `n`; sweep H is the eighth sweep; rows 49, 50, 51, 52, 53 and 54 carry it; and §5
measures the `U7` consequence instead of assuming it.

**I-1 — §3's residual and the publication path. Adopted:** §4 states the config↔disk-not-disk↔freeze-record
limit and names what anchors the digest outside the job; rows 23 and 24 publish the derived digest.

**I-2 — the third hand-carried literal. Adopted:** §6, single source with a three-way assertion.

**I-3 — the coverage claim over the enlarged inventory. Adopted and widened:** row 61 is round 4's
own prescribed text; rows 60 and 62 are the two further sites my sweep I found in `mint.py`; and row
7 carries the qualification rather than restating a false coverage assertion over a larger set.

**M-1 — sweep E's *"twelve in V15E1"* over a list of eleven.** Dissolved: §8's lists are generated,
so no count in this document is written by hand.

**M-2 — `once` is unanchored and matches inside words.** The pattern is **retained unanchored** (a
narrower pattern is the failure mode this lineage keeps being caught in) and the substring hit is
**declared as such**: `V15E1:995` is charged *"SUBSTRING ARTIFACT: `once` inside `concentration`."*

**M-3 — `V15E1:1574` mis-described as inside row 34's block.** Adopted: row 56, declared historical,
which is what it is.

**M-4 — `_gate_sha_count` is the config key, `gate_sha_artifacts` the runtime symbol.** Adopted:
rows 49 and 51 respectively.

**Carried from earlier rounds, unchanged:** the archaeology (latent from **v2**, frozen by **round-4
I-5**, re-affirmed by **round-8 H-1**, zero mentions in rounds 6–9, rounds 10–15 verifying only the
fidelity-side clause — which is true); the rejections of **(ii)** (launders the audit; and
`_guarded_open` is the only thing that *raises* on a test path) and **(iii)** (falsifies §3.1's
*"covered by `GATE-SHA`"* and leaves a file entering 66 processes unverified); **(iv)** declined with
the TOCTOU reason written into §6; round 2's resume-stability fact; `M-2`'s *argv ≠ launch argv*
recorded and not built on; and `M-3`'s *"the sbatch activates no environment"* referred to the
code/resource lineage.

---

## 8. Implementation delta

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | two-term `dev_path_opens` **derived and asserted** (39); `processes_reporting → 74` (12); `gate_sha_passes` audit; three counters to `by_construction` publication with `:458-462` **retained verbatim** (59); expectations asserted against `cfg["ledger"]`; `verify_predicate` called; **design-digest gate + derived-digest publication** (23, 24, 50); `PROJECTED_SECONDS` from the export with a three-way assertion (14, 58); docstrings `:4`, `:29-30`, `:559-560`, `:1232`; delete `:1418` | **≈ 60 lines** |
| `configs/c06/c06_falsifier.json` | rows 10, 11, 16, 20, 21, 22, 38, 40, 41, 42, 49 | **≈ 30 lines** |
| `c06_falsifier_cpu.sbatch` | rows 9, 25, 30, 31 + **`C06_MINTS_EXECUTED`** (48) + **`C06_PROJECTED_SECONDS`** (§6) | **≈ 14 lines** |
| `c06_falsifier_mint.py` | rows 4→27, 13, 17, 55, 60, 62 — **including two live-wrong numbers** (`PROJECTED_SECONDS`, `85.6 %`) | **6 lines** |
| **V15E2** | §6 (28), §7.2 (1, 2), §7.7 (37, 52), §8 (3, 4, 34, 35, 36, 56, 57), §9 (5, 19, 61), §11 (53), §12 (43, 44), §13 (6, 29), §13.1 items 12 and 28 (7, 8), §16 (54) | text |

**Landing order:** V15E2 written → its sha256 computed → every code/config edit → `design_sha256`
set **last** → full dry-check battery re-run (GATE-SHA **38/38**, GATE-C01PARITY `max|diff| = 0.0`
on both datasets, the blindness grep) → implementation record updated.

---

## 9. APPENDIX — the nine sweeps, printed

**Everything below is generated output.** Each sweep gives its exact command, its raw hit list as
`file:line` for every hit, the charge for every hit, and its subtraction. The two emptiness checks in
the global table are what make the partition an identity: **`UNCHARGED = 0`** means no site a sweep
returned is unaccounted, and **`ORPHAN = 0`** means no row points at a site no sweep returns (row 48
excepted by name, being an addition).

**File list, bound once and used by every command below:**

```bash
F="refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md \
   scripts/analysis/c06_falsifier_arena.py \
   scripts/analysis/c06_falsifier_mint.py \
   configs/c06/c06_falsifier.json \
   scripts/slurm/c06_falsifier_cpu.sbatch"
```

### Sweep A — process counts

```bash
grep -nE '\b7[234]\b' $F
```

**Raw hit list — 18 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:41` | **row 10** |
| 2 | `configs/c06/c06_falsifier.json:65` | declared — rotation angle 72.7 -- not a process count |
| 3 | `configs/c06/c06_falsifier.json:222` | **row 11** |
| 4 | `configs/c06/c06_falsifier.json:264` | declared — line citation generate_...:73-89 -- not a process count |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:106` | declared — line citation :73-89 |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:310` | declared — rotation angle 72.7 |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1197` | **row 1** |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1550` | **row 4** |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1566` | declared — tie-cap product 7x6+5x6=72 |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1658` | declared — line citation :73-89 |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1691` | declared — rotation angle 72.7 |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1839` | **row 6** |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1904` | **row 7** |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2017` | **row 8** |
| 15 | `scripts/analysis/c06_falsifier_arena.py:465` | **row 12** |
| 16 | `scripts/analysis/c06_falsifier_arena.py:466` | **row 12** |
| 17 | `scripts/analysis/c06_falsifier_mint.py:117` | **row 13** |
| 18 | `scripts/slurm/c06_falsifier_cpu.sbatch:16` | **row 9** |

**Subtraction A.** my sweep: **18** hits; rows account for **11**; declared non-targets: **7**; `18 = 11 + 7`; **residue: none.**

### Sweep B — ordinals

```bash
grep -nE '[0-9](st|nd|rd|th)' $F
```

**Raw hit list — 3 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:458` | declared — '95th percentile' -- S5's p95, not a process ordinal |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1198` | **row 2** |
| 3 | `scripts/analysis/c06_falsifier_arena.py:286` | declared — 'rank 20 (the 21st)' -- tie-window comment |

**Subtraction B.** my sweep: **3** hits; rows account for **1**; declared non-targets: **2**; `3 = 1 + 2`; **residue: none.**

### Sweep C — projection literals

```bash
grep -nE 'projected|PROJECTED|3670|4587|2929|2933|2934|3673|4592' $F
```

**Raw hit list — 17 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:43` | **row 16** |
| 2 | `configs/c06/c06_falsifier.json:44` | **row 16** |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1570` | **row 34** |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1571` | **row 34** |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1574` | **row 56** |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1581` | declared — provenance 2930.7->2933.9 -- historical |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1584` | declared — provenance 2933.9->2934.5 -- historical |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1631` | **row 19** |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1904` | **row 7** |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2080` | declared — round-14's id_hash_permutation share, frozen at the total it was taken against -- historical |
| 11 | `scripts/analysis/c06_falsifier_arena.py:29` | **row 15** |
| 12 | `scripts/analysis/c06_falsifier_arena.py:46` | **row 14** |
| 13 | `scripts/analysis/c06_falsifier_arena.py:57` | **row 58** |
| 14 | `scripts/analysis/c06_falsifier_arena.py:61` | **row 58** |
| 15 | `scripts/analysis/c06_falsifier_arena.py:68` | **row 58** |
| 16 | `scripts/analysis/c06_falsifier_mint.py:112` | **row 17** |
| 17 | `scripts/analysis/c06_falsifier_mint.py:127` | **row 18** |

**Subtraction C.** my sweep: **17** hits; rows account for **14**; declared non-targets: **3**; `17 = 14 + 3`; **residue: none.**

### Sweep D — design pointer

```bash
grep -nE 'design_document|design_sha256|design_status|PREREG_DRAFT|Frozen design|V15E|"design"' $F
```

**Raw hit list — 15 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:5` | **row 20** |
| 2 | `configs/c06/c06_falsifier.json:6` | **row 21** |
| 3 | `configs/c06/c06_falsifier.json:7` | **row 22** |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:3` | declared — supersession header naming v15 -- historical, correct as written |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:23` | declared — supersession chain V14->V13->... -- historical |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2257` | declared — V_OLD in the SS14.2 drafting-audit script -- an instrument for diffing drafts |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2258` | declared — V_NEW in the same script -- changing it would break that script's own fixed point |
| 8 | `scripts/analysis/c06_falsifier_arena.py:4` | **row 26** |
| 9 | `scripts/analysis/c06_falsifier_arena.py:30` | **row 15** |
| 10 | `scripts/analysis/c06_falsifier_arena.py:1249` | **row 23** |
| 11 | `scripts/analysis/c06_falsifier_arena.py:1250` | **row 23** |
| 12 | `scripts/analysis/c06_falsifier_arena.py:1433` | **row 24** |
| 13 | `scripts/analysis/c06_falsifier_arena.py:1434` | **row 24** |
| 14 | `scripts/analysis/c06_falsifier_mint.py:4` | **row 27** |
| 15 | `scripts/slurm/c06_falsifier_cpu.sbatch:14` | **row 25** |

**Subtraction D.** my sweep: **15** hits; rows account for **11**; declared non-targets: **4**; `15 = 11 + 4`; **residue: none.**

### Sweep E — pass-count idiom

```bash
grep -nE 'ONCE|once|one span|twice|single pass' $F
```

**Raw hit list — 22 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:128` | declared — 'load-bearing twice over' -- ordinary English |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:194` | declared — 'held out exactly once' -- fold construction |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:274` | declared — 'once corrected' -- ordinary English |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:629` | declared — 'items once' -- resample unit |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:966` | **row 28** |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:970` | declared — 'two criteria at once' -- ordinary English |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:995` | declared — SUBSTRING ARTIFACT: 'once' inside 'concentration' (M-2) |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1229` | declared — 'stated here once' -- ordinary English |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1295` | declared — 'the draws matrix is built once' -- U_acc's timed region |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1304` | **row 37** |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1547` | **row 3** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1583` | declared — 'once sklearn is restored' -- ordinary English |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1645` | **row 5** |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1820` | declared — 'removed twice elsewhere' -- ordinary English |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1840` | **row 29** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2420` | declared — 'held both lines at once' -- ordinary English |
| 17 | `scripts/analysis/c06_falsifier_arena.py:55` | declared — 'opened once and never re-wrapped' -- the heartbeat handle |
| 18 | `scripts/analysis/c06_falsifier_arena.py:560` | **row 32** |
| 19 | `scripts/analysis/c06_falsifier_arena.py:1039` | declared — 'once per (arm, seed)' -- the mF1 precompute |
| 20 | `scripts/analysis/c06_falsifier_arena.py:1232` | **row 33** |
| 21 | `scripts/slurm/c06_falsifier_cpu.sbatch:17` | **row 30** |
| 22 | `scripts/slurm/c06_falsifier_cpu.sbatch:62` | **row 31** |

**Subtraction E.** my sweep: **22** hits; rows account for **9**; declared non-targets: **13**; `22 = 9 + 13`; **residue: none.**

### Sweep F — SS8 equation, shares, risk row, and the two units it re-prices

```bash
grep -nE '2642\.|2508\.|1013\.8|4683\.|7725\.|4687\.|7729\.|68\.3|27\.6|61\.2|76\.5|85\.6|\bU7\b|\bU11\b' $F
```

**Raw hit list — 25 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1300` | **row 52** |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1304` | **row 37** |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1308` | declared — unit list 'U7 and U11 (dataset-independent)' -- no count, no figure |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1310` | declared — 'why U11 is class-dependent' heading -- no figure |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1373` | declared — corroboration list naming U7/U11 -- historical |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1374` | declared — corroboration list continuation -- historical |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1547` | **row 3** |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1550` | **row 4** |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1560` | declared — Phase 3 row 1013.8 s -- unchanged by this erratum |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1569` | **row 34** |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1570` | **row 34** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1571` | **row 34** |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1573` | declared — provenance 11.6->7.0 and 273.7->1013.8 -- historical |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1574` | **row 56** |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1576` | **row 35** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1577` | **row 35** |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1581` | declared — provenance 2930.7->2933.9 -- historical |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1607` | **row 57** |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1608` | **row 57** |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1609` | **row 36** |
| 21 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1610` | **row 36** |
| 22 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2002` | declared — handoff instruction to re-measure U11 -- carries no figure |
| 23 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2084` | declared — corroboration list naming U7 -- historical |
| 24 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2408` | declared — corroboration list naming U7 -- historical |
| 25 | `scripts/analysis/c06_falsifier_mint.py:118` | **row 55** |

**Subtraction F.** my sweep: **25** hits; rows account for **15**; declared non-targets: **10**; `25 = 15 + 10`; **residue: none.**

### Sweep G — ledger quantities

```bash
grep -nE 'dev_path_opens|mints_executed|processes_reporting|expected_sha_dev_opens|test_path_opens|banked_trainlog_opens|dev_label_materialisations_outside_decisions|test_label_materialisations|dev_or_test_labels_into_decision_quantities|mints_present_before_arena' $F
```

**Raw hit list — 54 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:211` | **row 42** |
| 2 | `configs/c06/c06_falsifier.json:215` | **row 41** |
| 3 | `configs/c06/c06_falsifier.json:216` | **row 41** |
| 4 | `configs/c06/c06_falsifier.json:217` | **row 41** |
| 5 | `configs/c06/c06_falsifier.json:218` | **row 38** |
| 6 | `configs/c06/c06_falsifier.json:219` | **row 40** |
| 7 | `configs/c06/c06_falsifier.json:220` | **row 41** |
| 8 | `configs/c06/c06_falsifier.json:221` | **row 41** |
| 9 | `configs/c06/c06_falsifier.json:222` | **row 11** |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1567` | declared — Phase 7 row naming mints_present_before_arena among gate names -- that counter is unchanged |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1807` | **row 43** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1808` | **row 43** |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1809` | **row 43** |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1810` | **row 43** |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1811` | **row 43** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1812` | **row 43** |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1813` | **row 43** |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1817` | **row 44** |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1819` | **row 44** |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1821` | **row 44** |
| 21 | `scripts/analysis/c06_falsifier_arena.py:411` | declared — docstring 'still publish test_path_opens: 0' -- that counter stays measured and binding |
| 22 | `scripts/analysis/c06_falsifier_arena.py:424` | declared — gate_ledger signature -- unchanged |
| 23 | `scripts/analysis/c06_falsifier_arena.py:433` | **row 39** |
| 24 | `scripts/analysis/c06_falsifier_arena.py:438` | **row 39** |
| 25 | `scripts/analysis/c06_falsifier_arena.py:449` | **row 45** |
| 26 | `scripts/analysis/c06_falsifier_arena.py:450` | declared — copies mints_present_before_arena into the ledger -- unchanged |
| 27 | `scripts/analysis/c06_falsifier_arena.py:451` | declared — continuation of :450 |
| 28 | `scripts/analysis/c06_falsifier_arena.py:452` | declared — copies mints_executed into the ledger -- unchanged |
| 29 | `scripts/analysis/c06_falsifier_arena.py:456` | **row 46** |
| 30 | `scripts/analysis/c06_falsifier_arena.py:457` | **row 46** |
| 31 | `scripts/analysis/c06_falsifier_arena.py:458` | **row 59** |
| 32 | `scripts/analysis/c06_falsifier_arena.py:459` | **row 59** |
| 33 | `scripts/analysis/c06_falsifier_arena.py:460` | **row 59** |
| 34 | `scripts/analysis/c06_falsifier_arena.py:461` | **row 59** |
| 35 | `scripts/analysis/c06_falsifier_arena.py:462` | **row 59** |
| 36 | `scripts/analysis/c06_falsifier_arena.py:463` | declared — mints_present_before_arena != 66 assertion -- unchanged |
| 37 | `scripts/analysis/c06_falsifier_arena.py:464` | declared — continuation of :463 |
| 38 | `scripts/analysis/c06_falsifier_arena.py:465` | **row 12** |
| 39 | `scripts/analysis/c06_falsifier_arena.py:466` | **row 12** |
| 40 | `scripts/analysis/c06_falsifier_arena.py:467` | **row 12** |
| 41 | `scripts/analysis/c06_falsifier_arena.py:468` | **row 39** |
| 42 | `scripts/analysis/c06_falsifier_arena.py:469` | **row 39** |
| 43 | `scripts/analysis/c06_falsifier_arena.py:471` | **row 39** |
| 44 | `scripts/analysis/c06_falsifier_arena.py:475` | **row 39** |
| 45 | `scripts/analysis/c06_falsifier_arena.py:607` | declared — gate_fold docstring naming mints_present_before_arena -- unchanged |
| 46 | `scripts/analysis/c06_falsifier_arena.py:626` | declared — the mints_present_before_arena != 66 message -- unchanged |
| 47 | `scripts/analysis/c06_falsifier_arena.py:628` | declared — the site that counts present mints -- unchanged |
| 48 | `scripts/analysis/c06_falsifier_arena.py:1418` | **row 26†** |
| 49 | `scripts/analysis/c06_falsifier_arena.py:1419` | **row 47** |
| 50 | `scripts/analysis/c06_falsifier_arena.py:1420` | **row 47** |
| 51 | `scripts/analysis/c06_falsifier_arena.py:1421` | declared — the gate_ledger call -- unchanged |
| 52 | `scripts/analysis/c06_falsifier_mint.py:217` | declared — comment citing SS12's clause title 'Why mints_executed and not 66' -- row 44 keeps that title |
| 53 | `scripts/analysis/c06_falsifier_mint.py:346` | declared — prints guard.LEDGER['dev_path_opens'] -- reads the counter, declares no expectation |
| 54 | `scripts/slurm/c06_falsifier_cpu.sbatch:36` | declared — narrative on why sitecustomize matters for test_path_opens -- unchanged |

**Subtraction G.** my sweep: **54** hits; rows account for **39**; declared non-targets: **15**; `54 = 39 + 15`; **residue: none.**

### Sweep H — GATE-SHA artifact count and scope

```bash
grep -nE '\b37\b|\b38\b|gate_sha_artifacts|_gate_sha_count|total_.11_digests|banked_unhashed_artifacts|imported_modules|read_for_definitions|input_caches|sixteen banked|banked artifacts|scope is stated' $F
```

**Raw hit list — 29 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:14` | declared — Erratum-1 float statistic 38.8% -- not an artifact count |
| 2 | `configs/c06/c06_falsifier.json:241` | declared — frozen_sha256_input_caches table key; the table gains no entry (the design digest lives at config:6) |
| 3 | `configs/c06/c06_falsifier.json:251` | **row 49** |
| 4 | `configs/c06/c06_falsifier.json:252` | **row 49** |
| 5 | `configs/c06/c06_falsifier.json:253` | **row 49** |
| 6 | `configs/c06/c06_falsifier.json:254` | **row 49** |
| 7 | `configs/c06/c06_falsifier.json:255` | **row 49** |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:51` | declared — round-15's own verification record (37/37 digests) -- historical |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:230` | declared — timing 37.46/27.54 s |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:602` | declared — '18x-38x above the epsilon' |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:678` | declared — 38.8% float statistic |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:966` | **row 28** |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1213` | declared — timing 38.87 s |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1214` | declared — timing 37.46 s |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1300` | **row 52** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1425` | declared — '18x-38x above C01's epsilon' |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1463` | declared — round-8's GATE-SHA widening-cost narrative (5 ms for the sixteen) -- historical, and unchanged: the sixteen are existence-checked, not re-hashed |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1517` | declared — '~37 wall-minutes' -- drafting spend, not an artifact count |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1543` | declared — timing 38.87 s |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1559` | declared — Phase 2D span 38.4 s |
| 21 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1636` | declared — Phase 2D span 38.4 s |
| 22 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1786` | **row 53** |
| 23 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1787` | **row 53** |
| 24 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2065` | declared — an earlier round's verification record (37/37 digests) -- historical |
| 25 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2439` | **row 54** |
| 26 | `scripts/analysis/c06_falsifier_arena.py:328` | declared — 38.8% float statistic |
| 27 | `scripts/analysis/c06_falsifier_arena.py:559` | **row 32** |
| 28 | `scripts/analysis/c06_falsifier_arena.py:563` | **row 50** |
| 29 | `scripts/analysis/c06_falsifier_arena.py:585` | **row 51** |

**Subtraction H.** my sweep: **29** hits; rows account for **13**; declared non-targets: **16**; `29 = 13 + 16`; **residue: none.**

### Sweep I — progress-coverage claims (grep -i)

```bash
grep -niE 'every python process|buffering=1|appends through|append-without-interleaving|progress file|append-only' $F
```

**Raw hit list — 16 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1622` | declared — lists the progress file among SS9 outputs -- carries no coverage claim |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1627` | **row 61** |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1628` | **row 61** |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1629` | **row 61** |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1903` | **row 7** |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1904` | **row 7** |
| 7 | `scripts/analysis/c06_falsifier_arena.py:16` | declared — arena module docstring describing SS9 heartbeat -- true of the arena |
| 8 | `scripts/analysis/c06_falsifier_arena.py:55` | declared — 'opened once and never re-wrapped' -- the heartbeat handle |
| 9 | `scripts/analysis/c06_falsifier_arena.py:59` | declared — the arena's buffering=1 handle open -- unchanged |
| 10 | `scripts/analysis/c06_falsifier_mint.py:116` | **row 60** |
| 11 | `scripts/analysis/c06_falsifier_mint.py:117` | **row 13** |
| 12 | `scripts/analysis/c06_falsifier_mint.py:119` | declared — 'line-buffered, append-only' -- true of this handle |
| 13 | `scripts/analysis/c06_falsifier_mint.py:131` | declared — the mint's own buffering=1 handle open -- unchanged |
| 14 | `scripts/analysis/c06_falsifier_mint.py:209` | **row 62** |
| 15 | `scripts/slurm/c06_falsifier_cpu.sbatch:18` | declared — the progress file is created before the first python process -- true, unchanged |
| 16 | `scripts/slurm/c06_falsifier_cpu.sbatch:54` | declared — the creation site itself -- unchanged |

**Subtraction I.** my sweep: **16** hits; rows account for **8**; declared non-targets: **8**; `16 = 8 + 8`; **residue: none.**

### Global partition — the nine sweeps together

| quantity | value |
|---|---|
| hit-instances across the nine sweeps | **199** |
| distinct sites (deduplicated) | **183** |
| sites returned by more than one sweep | **16** |
| distinct sites charged to a row | **107** |
| distinct sites charged to a declaration | **76** |
| sites returned by a sweep and charged to nothing | **0** |
| rows charged to a site that no sweep returns | **0** |

`183 = 107 + 76`. **Residue: none, in either direction.**

**Rows: 63 defined** (`1`–`62` plus `26†`). **1 has no site and is excluded from the orphan check by name: row 48**, which adds the `C06_PROJECTED_SECONDS`/`C06_MINTS_EXECUTED` exports to the sbatch. A grep over the current files cannot return a line that does not exist yet.

**Reproduction.** The nine commands above, run against the five artifacts at their post-CODE-R1 hashes, reproduce every hit list in this appendix line for line. Any hit a reviewer's wider pattern returns that is absent from these lists is a defect in this document, not in the pattern.


---

## 10. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no GPU, no SLURM job, no commit, no `TARGET_STATE.json`
edit; `artifacts/c06_falsifier/` never created. I did **not** run the `--gate-sha-only` leg — rounds
2 and 3 carry that measurement on their record.

**Compute used:** file and review reads; the nine `grep` sweeps of §9 plus wider variants of each
(the wider variants are what produced `mint:118`, `mint:116`, `mint:209`, `config:217`, `arena:559`
and `V15E1:1786-1787`); one `sha256sum` over the five artifacts; **one login-node timing of
`hashlib.sha256` over the 188 061-byte design document, 7 repetitions, reported in §5**; and
arithmetic. The charge table and every count in §9 are produced by a script over the sweep output.

**Nothing is edited.** All five artifacts carry their post-CODE-R1 hashes, re-verified before and
after:

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` |

`…_PROPOSAL.md`, `…_V2.md`, `…_V3.md` and `…_V4.md` stay on disk byte-unmodified.

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`.
**The battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under
v1, v2, v3 or v4 as specified.**
