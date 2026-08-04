# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v4**

**Supersedes** `…_PROPOSAL_V3.md` (`48f4e0153103cc608884…`) → `…_V2.md` (`4225bea3cc9907d38e2e…`) →
`…_PROPOSAL.md` (`f063c388c4afabdb7964…`). **All three stay on disk.**
*Against:* `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudications:* round 1 (0C/3H/3I/3M) → v2; round 2 (0C/2H/3I/3M) → v3; **round 3 (0C/3H/3I/3M)** → this.
*Status:* **PROPOSAL. Nothing is landed.** All five artifacts carry their post-CODE-R1 hashes.

---

## 0. What round 3 settled, and what it blocked

**Verified at source and carried forward unchanged:** the two-term `dev_path_opens` with both factors
over the concatenated iterable (21 files, 2 dev-like, 0 test-like, `frozen_sha256` contributing zero
— `4 = 2 × 2`); the `74 = 1 + 66 + 6 + 1` inventory and its resume-stability; **the uniform counter
criterion, discharged in full** with a source audit confirming `_guarded_open` increments exactly
three counters (`:97`, `:102`, `:106`) and that the demoted three are digest-pinned unreachable
because `c09guard.py` **is** in `frozen_sha256`; `int("GATE-SHA" in self.gates)`; the `cfg["ledger"]`
cross-check; `verify_predicate` called; option (iv) declined with the TOCTOU reason; the `argv` pass
audit; `C06_MINTS_EXECUTED`; row 26's dead-line deletion; and §4's branch-(a) re-price **to the last
decimal**.

**And the `\b7[234]\b` subtraction was verified exactly** — 18 hits, eleven in rows, seven declared,
nothing left over in either direction. Round 3 called it *"the strongest verification work in this
lineage's record."*

**Blocked because that rigour was applied to one sweep out of seven.** §1 promised *"every site
stating a quantity this erratum moves"*; only the `73`/`74` pattern was carried through with a stated
residue. Round 3's own sweeps found **ten sites outside the table**, two of them **wrong today**:

* `config:6`'s `design_sha256` names a V15E1 state that has not existed since CODE-R1 — **and it is
  published on the face of every verdict and HALT artifact**;
* `mint:112`'s `PROJECTED_SECONDS = 2929.9` is `740.1 s` below the arena's, and `mint:127` divides
  every heartbeat line of **66 of the 74 processes** by it.

**And one prescribed repair contradicted another in the same table** (H-3): row 2 called the driver
leg *"the 73rd process"* while rows 6 and 11 put it first in the order.

**v4's answer is §1: seven sweeps, each with its hit list, its rows, its declared residue, and its
subtraction stated.** My sweeps reproduce all ten of round 3's sites and find **two more** it did not
list — `arena:4` and `mint:4`, both reading *"Frozen design: …DRAFT_V15.md"*, two revisions stale.

---

## 1. FULL ENUMERATION — seven sweeps, each subtractable

**44 rows.** Rows marked **CORRECT** need no edit and are listed anyway, because completeness is
verified by subtraction, not by trust.

### Sweep A — `\b7[234]\b` · 18 hits · rows 1–13 · residue 7

| # | site | current | correct | why |
|---|---|---|---|---|
| 1 | `V15E1:1197` §7.2 | *"those **72** processes"* | **unchanged — CORRECT** | 66 mints + 6 fidelity = 72 is exactly right at 74 |
| 2 | `V15E1:1198` §7.2 | *"The **73rd** process, the arena…"* | *"**The remaining two processes** — the `--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8 Phase 1g"* | **round-3 H-3.** v3's *"73rd and 74th"* contradicted rows 6 and 11, which put the driver leg **first**. **No ordinal**: the paragraph's job is to say which processes are *not* priced inside the mint units, and that is true under any ordering, so it cannot drift against §13 again |
| 3 | `V15E1:1547` §8 Phase 1d | *"`GATE-SHA`, once in the driver \| `1` \| `U7` \| `0.1 s`"* | *"`GATE-SHA`, **twice** — driver leg and arena \| `2` \| `U7` \| `0.2 s`"* | the only *"once"* site carrying a numeric count column |
| 4 | `V15E1:1550` §8 Phase 1g | *"**`1`** — the arena process alone … `66+6+1 = 73` accounts for **every** process"* | *"**`2`** — the `--gate-sha-only` driver leg and the arena … `1+66+6+1 = 74` accounts for every process §13 declares"* | the compute-accounting twin of the §12 defect |
| 5 | `V15E1:1645` §9 | *"the **arena's own startup** is **the one span**…"* | *"the `--gate-sha-only` driver leg's startup is the first such span and the arena's the second; both bounded by the same arena-class band"* | measured false — the driver leg emits its own heartbeat lines |
| 6 | `V15E1:1839` §13 | *"**73** processes in the order 66 mints → 6 fidelity → 1 arena"* | *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena"* | |
| 7 | `V15E1:1904` §13.1 item 12 | *"across all **73** processes"* | *"across all **74**"* | handoff item the code lineage checks |
| 8 | `V15E1:2017` §13.1 item 28 | *"active in all **73** processes"* | *"in all **74**"* | the driver leg is where the guard first installs |
| 9 | `sbatch:16` | *"73 processes…"* | *"74 processes: 1 GATE-SHA driver leg → 66 mints → 6 fidelity → 1 arena"* | |
| 10 | `config:41` | `{"mints":66,"fidelity":6,"arena":1,"total":73}` | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | |
| 11 | `config:222` | `"processes_reporting": {"expected":73,"binding":true}` | `{"expected":74,"decomposition":"1+66+6+1","binding":true}` | |
| 12 | `arena:465-466` | `!= 73` (×2) | `!= 74`, read from `cfg["ledger"]` (sweep G) | |
| 13 | `mint:117` | *"**72 of 73** processes previously wrote nothing"* | *"**73 of 74**"* | docstring narrating the pre-CODE-R1 state |

**Subtraction A.** 18 hits − 11 in rows (row 12 covers two lines) = **7 declared non-process**:
`generate_…:73-89` at `V15E1:106`, `V15E1:1658`, `config:264` (×3); `72.7` at `V15E1:310`,
`V15E1:1691`, `config:65` (×3); the tie-cap product `7×6 + 5×6 = 72` at `V15E1:1566`. **Nothing left
over in either direction.**

### Sweep B — ordinals `[0-9](st|nd|rd|th)` · 3 hits · row 2 · residue 2

**Round-3 M-2 adopted:** `\b7[234]\b` **structurally cannot match `73rd`**, which is row 2's own
site, so sweep A alone could not have found it. Sweep B closes that hole and is stated so a reader
reproducing the subtraction is not misled.

**Subtraction B.** 3 hits − 1 in row 2 (`V15E1:1198`) = **2 declared**: `arena:286` (*"rank 21st"*,
the tie-window comment) and `V15E1:458` (*"95th percentile"*, S5's p95). Neither is a process count.

### Sweep C — projection literals (`3670.0|4587.5|2929.9|PROJECTED_SECONDS|projected_seconds`) · 12 hits · rows 14–19 · residue 6

| # | site | current | correct | why |
|---|---|---|---|---|
| 14 | `arena:46` | `PROJECTED_SECONDS = 3670.0` | **`3673.9`** | the heartbeat denominator |
| 15 | `arena:29-30` | docstring *"§8 projects `3670.0 s` (`4587.5 s` conservative) … (`…V15E1.md`)"* | **`3673.9 s` / `4592.4 s`**, and the document name → V15E2 | **round-3 H-2 + H-1**, absent from v3 |
| 16 | `config:43-44` | `3670.0` / `4587.5` | **`3673.9` / `4592.4`** | |
| 17 | **`mint:112`** | `PROJECTED_SECONDS = 2929.9  # §8 under ERRATUM 1` | **`3673.9  # §8 under ERRATUM 1 + CODE-R1 H-4 + ERRATUM 2`** | **WRONG TODAY.** `740.1 s` below the arena's, i.e. 66 of 74 processes publish `elapsed/2929.9` against the arena's `elapsed/3670.0` — a `25 %` disagreement in the ratio column, in the phase that is `68.3 %` of the budget |
| 18 | `mint:127` | `elapsed / PROJECTED_SECONDS` | **unchanged — CORRECT** | the divide site is right; only its constant was stale. Listed because the defect lives at `:112` and is *consumed* here |
| 19 | `V15E1:1631-1633` §9 | *"(ERRATUM 1 set it to `2929.9 s`; CODE-R1 H-4 sets it to `3670.0 s`… the literal in `c06_falsifier_arena.py` and `configs/c06/…json` is updated with each correction.)"* | carries **`3673.9 s`** and names **all four** literal sites including `mint:112` | **round-3 H-2.** This sentence re-publishes the two-file claim that caused the drift |

**Subtraction C.** 12 hits − 6 in rows = **6 declared**: `arena:57` (`projected=PROJECTED_SECONDS`,
the parameter default — follows `:46`); `V15E1:1570`, `:1571`, `:1574`, `:1584` (§8's own provenance
narrative — **covered by sweep F rows 34–36**, not lost); `V15E1:2080` (round-14's
`id_hash_permutation` figure `0.13 %` of the total, a **historical** measurement correctly frozen at
the total it was taken against, and explicitly not re-derived).

### Sweep D — design pointer (`design_document|design_sha256|design_status|V15E1|DRAFT_V15|Frozen design`) · 12 hits · rows 20–27 · residue 4

| # | site | current | correct | why |
|---|---|---|---|---|
| 20 | `config:5` | `"design_document": "…DRAFT_V15E1.md"` | `"…DRAFT_V15E2.md"` | |
| 21 | **`config:6`** | `"design_sha256": "0b446b91675fd4ff8aea…"` | V15E2's digest, **derived at freeze time** (§3) | **WRONG TODAY.** Names V15E1 at the sha it had when Erratum 1 landed; CODE-R1 moved it to `8cde58aa…` and this was not updated |
| 22 | `config:7` | `design_status`: *"GO at round 15 … + ERRATUM 1 landed"* | *"… + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2 landed"* | a different key in a different block from row 30's `erratum_2` |
| 23 | `arena:1249-1250` | `emit_halt` writes `cfg["design_document"]` / `["design_sha256"]` | **unchanged — CORRECT**, and it is *why* row 21 is High: the stale digest reaches **every HALT artifact** | |
| 24 | `arena:1433-1434` | the verdict face writes the same pair | **unchanged — CORRECT**, same reason for **every verdict artifact** | |
| 25 | `sbatch:14` | *"Frozen design: …DRAFT_V15.md (GO at round 15)"* | *"…DRAFT_V15E2.md"* | two revisions stale |
| 26 | **`arena:4`** | *"Frozen design: …DRAFT_V15.md (GO at round 15)"* | *"…DRAFT_V15E2.md"* | **found by this sweep, not in round 3's ten-site list** |
| 27 | **`mint:4`** | *"Frozen design: …DRAFT_V15.md (GO at round 15)"* | *"…DRAFT_V15E2.md"* | **found by this sweep, not in round 3's ten-site list** |

**Subtraction D.** 12 hits − 8 in rows = **4 declared**: `arena:30` (the V15E1 mention inside the
docstring — **row 15**, counted there not twice); `V15E1:3` (the supersession header's own reference
to v15, correct as a historical statement); `V15E1:2258` (`V_NEW` inside §14.2's **drafting audit
script**, an instrument for diffing drafts, not a design pointer — and changing it would break that
script's own fixed point).

### Sweep E — pass-count idiom (`ONCE|once|one span|twice|single pass`) · 21 hits · rows 28–33 · residue 15

| # | site | current | correct | why |
|---|---|---|---|---|
| 28 | `V15E1:966` §6 `GATE-SHA` row | *"**once in the sbatch driver**"* | *"**twice** — once in the sbatch driver before any other process, and again in the arena at the point of use"*, **with the TOCTOU reason** | round-1 H-2; the reason makes the second pass a decision, not an accident |
| 29 | `V15E1:1840` §13 | *"`GATE-SHA` **once in the driver** before any of them"* | *"…in the driver leg before any of them **and again in the arena**"* | |
| 30 | `sbatch:17` | *"GATE-SHA runs ONCE in this driver"* | *"…runs in this driver before any of them, and again inside the arena"* | |
| 31 | `sbatch:62` | *"# GATE-SHA, ONCE, before any other process"* | *"# GATE-SHA, first of two passes, before any other process"* | |
| 32 | `arena:560` | docstring *"ONCE, in the driver, before any other process"* | *"the first of two passes; the arena repeats it at the point of use"* | |
| 33 | **`arena:1232`** | `--gate-sha-only` help: *"the sbatch driver calls this **ONCE** before any other process (§13)"* | *"…calls this once, as the **first of two** `GATE-SHA` passes (§6, §13)"* | **round-3 I-2.** Defensible on a strict reading — the driver *does* invoke it once — but it is `--help` output citing §13 for a claim §13 no longer makes in that form, and it is the same uppercase idiom as rows 31–32 which this erratum rewrites |

**Subtraction E.** 21 hits − 6 in rows = **15 declared irrelevant idiom**: `arena:55` (*"opened once
and never re-wrapped"*, the heartbeat handle), `arena:1039` (*"once per (arm, seed)"*, the mF1
precompute), and twelve in V15E1 that are ordinary English or unrelated technical uses — `:128`
(*"load-bearing twice over"*), `:194` (*"held out exactly once"*), `:274` (*"once corrected"*), `:629`
(*"items once"*), `:970` (*"ONE predicate"*), `:1229` (*"stated here once"*), `:1295`, `:1304`
(**`:1304` is row 37**, listed under sweep F), `:1583` (*"once sklearn is restored"*), `:1820`
(*"removed twice elsewhere"*), `:2420` (round-15 narrative).

### Sweep F — §8's equation, shares and risk row · 8 hits · rows 34–37 · residue 4

| # | site | current | correct | why |
|---|---|---|---|---|
| 34 | `V15E1:1569-1571` | `2642.3 + 1.0 + 0.7 + 0.1 + 1.3 + **3.8** + 7.0 + 1013.8 = 3670.0`; `× 1.25 = 4587.5` | **`2642.4`** + 1.0 + 0.7 + 0.1 + 1.3 + **`7.6`** + 7.0 + 1013.8 = **`3673.9`**; `× 1.25 = **4592.4**` | **round-3 I-3.** **Two** literals move: Phase 1d's `0.1 → 0.2` lives *inside* the `2642.3` residue, and Phase 1g's `3.8` is a named term. Landing §4's totals without moving `2642.3` writes an equation summing to `3673.8` under a stated `3673.9`, breaking §8's own *"the printed product column now sums to the total directly"* |
| 35 | `V15E1:1576-1577` | *"mints fall from `85.6 %` to `68.3 %` and Phase 3 rises from `9.3 %` to `27.6 %`"* | **unchanged — CORRECT** | re-derived: `2508.3/3673.9 = 68.27 %` and `1013.8/3673.9 = 27.59 %` still round to `68.3`/`27.6` |
| 36 | `V15E1:1609-1610` risk row | *"2× miss `4683.8 s = 78.1 min`, 5× miss `7725.2 s = 128.8 min`"* | **`4687.7 s = 78.1 min`**, **`7729.1 s = 128.8 min`** | the second-order consequence of the re-price; minutes unchanged at one decimal |
| 37 | `V15E1:1304` §7.7 `U11` | *"Both are inside the mint units and inside `U9`; **the arena's is priced once at §8 Phase 1g**"* | *"…the **two arena-class startups — the `--gate-sha-only` driver leg and the arena — are priced at §8 Phase 1g**"* | **round-3 I-1.** The section that *defines* the unit Phase 1g now prices **twice** accounts for one instance. Not literally false, which is why round 3 rated it Important — but §7.2 and §8 both cite §7.7 as the authority for the unit and its class |

**Subtraction F.** 8 hits − 4 in rows = **4 declared**: `V15E1:1571`'s `4587.5` and `:1574`'s
`2929.9`/`2933.9` are **inside row 34's replacement block**; `V15E1:1584`'s `2933.9 → 2934.5` is the
**Erratum-1 provenance narrative**, correctly historical; `V15E1:1607-1608`'s `68.3 %`/`27.6 %` is
row 35.

### Sweep G — the ledger quantities (`dev_path_opens|mints_executed|processes_reporting|expected_sha_dev_opens`) · 14 hits · rows 38–44 · residue 0

| # | site | current | correct | why |
|---|---|---|---|---|
| 38 | `config:218` | `"dev_path_opens": {"expected":"mints_executed","binding":true}` | `{"expected":"mints_executed + expected_sha_dev_opens","expected_sha_dev_opens":4,"derivation":"(dev-like files in the concatenated iterable gate_sha hashes = 2) × (GATE-SHA passes = 2)","binding":true}` | the core repair |
| 39 | `arena:433`, `:438`, `:468-475` | the frozen `+ 0` predicate and its `ERRATUM REQUIRED` message | the two-term predicate; **the message's decomposition becomes the derivation**. **Round-3 M-3:** the arena **derives** the dev-like count from the digest tables and **asserts** it against the config's declared `4` — it must not merely *read* the `4`, which would reintroduce the trusted literal one key lower down | |
| 40 | `config:219` | `dev_label_materialisations_outside_decisions` binding `mints_executed` | moved to a `by_construction` block as a **warranted string** | §2's criterion |
| 41 | `arena:447` (`self.ledger = dict(tot)`) + `config` | all six counters published as measured integers | three instrumented counters in `measured`; three uninstrumented as `by_construction` strings | §2's criterion, uniformly |
| 42 | `config:211` | `_code_review_r1.blocked_on_erratum_2` narrative | replaced by an `erratum_2` block recording what landed | |
| 43 | `V15E1:1810`, `:1811` §12 rows | the `+ 0` predicate and the binding dev-label row | rows 4, 5 and 8 of §12's table + the by-construction marks + M-3's *"top-level processes only"* sentence | **round-3 I-3 recorded these as covered by §7's delta; they are now rows** |
| 44 | `V15E1:1817-1821` §12 | *"Why `mints_executed` and not `66`"* — the resume warrant | amended to carry the two-term form **and** the spawn/skip fact that makes an exact `74` safe where C09 refused to bind `39` | the warrant paragraph must move with the predicate |
| 26† | `arena:1418` | `mints_executed = sum(1 for _ in [None])   # placeholder` | **delete** | v3's own find, carried |

**Subtraction G.** 14 hits, all in rows 38–44 plus row 26†. **Residue 0.**

---

## 2. The uniform counter criterion — unchanged, discharged in full at round 3

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

Round 3 audited this and confirmed **no reachable tripwire is removed**, adding the point that the
guard which would have to change to create one is itself **digest-pinned in `frozen_sha256`**, so
`GATE-SHA` HALTs before a modified guard could run.

**Round-3 M-1 adopted, correcting v3's overstatement.** v3 said C09 publishes the three *"never as
integers"*. Read from the artifact, `GATE_LEDGER` also carries `per_process` and
`arena_process_counts` blocks that publish **all six** keys as integers per process. The claim is
true only of the aggregate `measured` block — which **is** the block this prescription restructures
(`arena:447`) — so the repair is unchanged, but the precedent is cited accurately here.

---

## 3. The `design_sha256` mechanism — derived at freeze time, never hand-carried

**Round 3's H-1 is a stale-hash defect, and hand-updating it would only reset the clock.** The
mechanism removes the class:

> **`GATE-SHA` gains one artifact: the design document itself.** The arena reads
> `cfg["design_document"]`, hashes the on-disk file, and compares against `cfg["design_sha256"]`,
> **HALTing on mismatch** with both digests in the message. The design pointer becomes a gated
> quantity like every other frozen artifact, and any future edit to the design that is not
> accompanied by a config update is caught **before any battery computation**, in the
> `--gate-sha-only` process, at zero cost.

**Landing order, stated because the digest is self-referential.** V15E2 is written **first**; its
sha256 is computed **after** it is final; the config's `design_sha256` is the **last** edit of the
landing. §7 records this ordering.

**Interaction with `expected_sha_dev_opens`, checked.** The design document is a `.md` in
`refine-logs/` — **not dev-like and not test-like** under `c09guard`'s predicates — so adding it to
`GATE-SHA`'s scope leaves the dev-like count at **2** and `expected_sha_dev_opens` at **4**. The
`GATE-SHA` artifact count moves `37 → 38`, which is a reported figure (`_gate_sha_count`), not a
binding one. **Stated because this erratum's whole subject is a repair that broke a second
predicate.**

---

## 4. §8: re-priced, branch (a) — figures re-derived by round 3 to the last decimal

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

**`PROJECTED_SECONDS` has FOUR sites, not two** (round-3 H-2): `arena:46`, `arena:29`, `config:43-44`
and **`mint:112`** — rows 14–17 — plus §9's sentence at `V15E1:1631-1633` (row 19) which must name
all four or re-publish the drift it caused.

**Branch (a) chosen, reason unchanged:** Phase 1g exists *because* round-11 I-1 found a per-process
cost no row priced; leaving a second one knowingly unpriced in the row created to fix the first is
the same defect one iteration later.

**For the code lineage, not repaired here:** the `--gate-sha-only` leg runs `load_frozen()` before
returning, which is what makes it a *full* arena-class startup; returning right after `gate_sha()`
would make it far cheaper. Pricing the full startup is conservative and is what this erratum carries.

---

## 5. Round 3's remaining findings

**H-3 — the ordinal contradiction. Adopted, and repaired by removing ordinals entirely** (row 2).
v3 prescribed the driver leg's position three times and got it wrong once; the replacement text is
ordering-independent and cannot drift against §13's declared order or row 11's `1+66+6+1`.

**I-1 (`V15E1:1304`), I-2 (`arena:1232`), I-3 (§8's equation and risk row): adopted as rows 37, 33,
34 and 36.**

**M-1, M-2, M-3: adopted** — the C09 form claim corrected (§2), the ordinal hole in the stated
pattern disclosed (sweep B), and *"derives, asserts against"* put into row 39 and §7's arena row.

**Carried from earlier rounds, unchanged:** the archaeology (latent from **v2**, frozen by **round-4
I-5**, re-affirmed by **round-8 H-1**, zero mentions in rounds 6–9, rounds 10–15 verifying only the
fidelity-side clause — which is true); the rejections of **(ii)** (launders the audit; and
`_guarded_open` is the only thing that *raises* on a test path) and **(iii)** (falsifies §3.1's
*"covered by `GATE-SHA`"* and leaves a file entering 66 processes unverified); **(iv)** declined with
the TOCTOU reason written into §6; round 2's resume-stability fact; `M-2`'s *argv ≠ launch argv*
recorded and not built on; and `M-3`'s *"the sbatch activates no environment"* referred to the
code/resource lineage.

---

## 6. Implementation delta

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | two-term `dev_path_opens` **derived and asserted**; `processes_reporting → 74`; `gate_sha_passes` audit with `int("GATE-SHA" in self.gates)`; three counters to `by_construction`; expectations asserted against `cfg["ledger"]`; `verify_predicate` called; **design-digest gate (§3)**; literals `:46`, `:29-30`; docstrings `:4`, `:560`, `:1232`; delete `:1418` | **≈ 50 lines** |
| `configs/c06/c06_falsifier.json` | rows 10, 11, 19, 20, 21, 22, 38, 40, 41, 42 | **≈ 25 lines** |
| `c06_falsifier_cpu.sbatch` | rows 9, 25, 30, 31 + **`C06_MINTS_EXECUTED`** export with an executed-vs-skipped counter | **≈ 12 lines** |
| `c06_falsifier_mint.py` | rows 13, 17, 27 — **including the live-wrong `PROJECTED_SECONDS`** | **3 lines** |
| **V15E2** | §6 (row 28), §7.2 (rows 1, 2), §7.7 (row 37), §8 (rows 3, 4, 34, 35, 36), §9 (rows 5, 19), §12 (rows 43, 44 + by-construction marks + top-level-processes sentence), §13 (rows 6, 29), §13.1 items 12 and 28 (rows 7, 8) | text |

**Landing order:** V15E2 written → its sha256 computed → every code/config edit → `design_sha256`
set **last** → full dry-check battery re-run → implementation record updated.

---

## 7. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no GPU, no job, no commit, no `TARGET_STATE.json`
edit; `artifacts/c06_falsifier/` never created. Compute: file and review reads; the seven greps of
§1 over the five artifacts; arithmetic. I did **not** re-run the `--gate-sha-only` leg — rounds 2 and
3 carry that measurement on their record.

**Nothing is edited.** `C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (`8cde58aa…`), `c06_falsifier_arena.py`
(`0cdfd4f0…`), `c06_falsifier_mint.py` (`98f7b4a6…`), `configs/c06/c06_falsifier.json`
(`e2678431…`) and `c06_falsifier_cpu.sbatch` (`c3647173…`) carry their post-CODE-R1 hashes. The
arena still implements `dev_path_opens == mints_executed + 0` and `processes_reporting != 73` exactly
as frozen, failing with `ERRATUM REQUIRED`. **The battery cannot pass `GATE-LEDGER` before this
erratum lands, and could not have passed under v1, v2 or v3 as specified.**
