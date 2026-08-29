# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v6**

**Supersedes** `…_PROPOSAL_V5.md` (`c41a0223bdf6db709114…`) → `…_V4.md` (`0b4940416abd1fb4bf79…`) →
`…_V3.md` (`48f4e0153103cc608884…`) → `…_V2.md` (`4225bea3cc9907d38e2e…`) → `…_PROPOSAL.md`
(`f063c388c4afabdb7964…`). **All five stay on disk, byte-unmodified.**
*Against:* `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudications:* round 1 (0C/3H/3I/3M) → v2; round 2 (0C/2H/3I/3M) → v3; round 3 (0C/3H/3I/3M) → v4;
round 4 (0C/2H/3I/4M) → v5; **round 5 (0C/1H/3I/5M)** → this.
*Status:* **PROPOSAL. Nothing is landed.** All five artifacts carry their post-CODE-R1 hashes.

---

## 0. What round 5 settled, and the one thing it blocked

**Round 5 re-ran all nine of v5's printed commands and reproduced all nine hit lists exactly**, then
recomputed the partition independently from the printed charge tables and confirmed
`183 = 107 + 76` with `UNCHARGED = 0` and `ORPHAN = 0`, all 15 cross-sweep sites identically charged,
row 48 the only siteless row, and the multi-line declaration matching the charged extents of all 22
rows it named in both directions with zero discrepancy. It confirmed the three new live findings at
source, re-measured H-2's `U7` cost and reached the same conclusion, verified I-1's honest scoping on
both its load-bearing premises, and confirmed row 59's retained assertions and the `c09guard`
increment map. **All nine of round 4's limbs are discharged.** None of that is reopened here.

**Blocked on the tenth family.** The erratum re-prices Phase 1g from `1 × U11 = 3.8 s` to
`2 × U11 = 7.6 s`, and **no sweep pattern contained `3\.8`, `7\.6`, or `Phase 1[dg]`**. Sweep F was
built for §8's equation and catches `U11` *mentions*, so `U7`'s object grew and got rows 49–54 while
`U11`'s row-level cost moved and got none. Two consequences, both correct as round 5 states them:

* **Row 4 edited the count column of `V15E1:1550` and left `3.8 s` standing in the cost column of the
  same line** — against row 34's `7.6` in the total nineteen lines below. That is the
  *"equation summing to `3673.8` under a stated `3673.9`"* failure row 34 exists to prevent,
  reproduced one row up in the same table.
* **`V15E1:1368`'s bolded *"So §8 Phase 1g's count is `1`, the arena alone"*** — the sentence that
  *determines* the count — was charged to nothing. That is the `U11` analogue of `V15E1:1300`, the
  site round-4 H-2 singled out for `U7`.

**v6 adds sweep J and, because this is now the second time a family was missed, a meta-check over
§8's whole row/unit structure so that no eleventh family can exist.** §6 states that check's result.

**What this round's work found beyond the ask — a fourth live-wrong claim, on the warrant for the
very term this erratum repairs.** Round 5's I-2 corrected §3's `lab_dev` warrant from *"exactly
once"* to two write sites. Verifying that at source, I found the neighbouring claim is also false:

> `V15E1:1810`: *"`headspace_fidelity.py` opens **no** `dev_seen` file, **reading `lab_dev` out of
> the banked mint `.npz`** (`:66`)"* — and `sbatch:103`: *"It reads lab_dev out of the banked mint
> .npz and opens no dev_seen file (§12)."*

Measured: `headspace_fidelity.py` reads **only** `z["meta"]` (`:68`) out of that `.npz`, and the file
contains **no `lab`/`label` reference of any kind**. It does not read `lab_dev`. The **conclusion**
survives untouched — fidelity opens no `dev_seen` file, so the second term's fidelity contribution is
genuinely zero — but the stated mechanism is wrong in the design (`:1810`, row 43) and in the sbatch
(`:103`, now row 66). This does not change the repair: `expected_sha_dev_opens = 4` is driven by
`GATE-SHA` hashing the two `dev_seen` caches, not by fidelity.

---

## 1. The accounting rule, stated before the rows

> **Every site returned by any sweep is charged to exactly one row or exactly one declaration.**
> A row may collect sites returned by several sweeps — `V15E1:1904` is a `73` site, a `projected`
> site *and* a coverage-claim site — so the same row appears in more than one sweep's table with the
> **same** charge. Because the charge is unique, a site counted in several sweeps is not counted
> twice: the per-sweep subtraction is `hits = rows + declarations` with no residue, and the
> **global** subtraction is over the deduplicated union.

**Two distinct concepts, separated at round 5's I-3.** A row's **extent** is the set of lines its
edit touches; its **charged sites** are the lines a sweep returns. Extent ⊇ charged sites always.
§9 prints the three **extent-only lines** — lines inside a row's extent that no pattern returns — by
name, rather than contorting a pattern to manufacture a hit. Stating them is the honest form: it
names a limit of the method instead of hiding one.

**Multi-line rows are declared everywhere they occur.** The 26 rows with more than one charged site
are **2, 5, 7, 12, 15, 16, 19, 23, 24, 32, 34, 35, 36, 39, 41, 43, 44, 46, 47, 49, 53, 57, 58, 59,
61, 65**, and every line of each is a separate charged site in §9. Round 5 audited v5's list of 22
in both directions and found one omission — **row 19** — which is in the list above, together with
rows 2, 5, 39, 43, 44 and 65 whose extents this round enlarges.

---

## 2. FULL ENUMERATION — 67 rows

Rows marked **CORRECT** need no edit and are listed anyway, because completeness is verified by
subtraction, not by trust.

### 2.1 The process inventory — `73 → 74`

| # | site | current | correct | why |
|---|---|---|---|---|
| 1 | `V15E1:1197` §7.2 | *"those **72** processes"* | **unchanged — CORRECT** | 66 mints + 6 fidelity = 72 is exactly right at 74 |
| 2 | `V15E1:1198-1199` §7.2 | *"The **73rd** process, the arena, is a different case and is priced separately at §8 Phase 1g"* (**two lines**) | *"**The remaining two processes** — the `--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8 Phase 1g"* | **round-3 H-3.** No ordinal: the paragraph's job is to say which processes are *not* priced inside the mint units, true under any ordering. `72` (row 1) `+ 2` `= 74`, agreeing with rows 6, 10, 11, 12. **Round-5 H-1(d): `:1199` is the sentence's second line and is now charged (sweep J)** |
| 6 | `V15E1:1839` §13 | *"**73** processes in the order 66 mints → 6 fidelity → 1 arena"* | *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena"* | |
| 7 | `V15E1:1903-1904` §13.1 item 12 | *"the `buffering=1` handle never re-wrapped … append-without-interleaving across all **73** processes"* (**two lines**) | *"…across all **74** processes, **of which the 68 this lineage authors append through a handle opened `buffering=1`; the six `headspace_fidelity.py` processes are sha-frozen and third-party and the bash driver writes their line (`sbatch:128-129`)**"* | **round-4 I-3.** The count moves and the coverage assertion it sits inside is corrected in the same breath |
| 8 | `V15E1:2017` §13.1 item 28 | *"active in all **73** processes"* | *"in all **74**"* | the driver leg is where the guard first installs |
| 9 | `sbatch:16` | *"73 processes…"* | *"74 processes: 1 GATE-SHA driver leg → 66 mints → 6 fidelity → 1 arena"* | |
| 10 | `config:41` | `{"mints":66,"fidelity":6,"arena":1,"total":73}` | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | |
| 11 | `config:222` | `"processes_reporting": {"expected":73,"binding":true}` | `{"expected":74,"decomposition":"1+66+6+1","binding":true}` | |
| 12 | `arena:465-467` | `!= 73` and its message (**three lines**) | `!= 74`, read from `cfg["ledger"]` (§2.6) | `:465`/`:466` are sweep-A hits, `:467` the message continuation and a sweep-G hit |
| 13 | `mint:117` | *"**72 of 73** processes previously wrote nothing"* | *"**73 of 74**"* | docstring narrating the pre-CODE-R1 state |

### 2.2 The `--gate-sha-only` leg is a second priced process

| # | site | current | correct | why |
|---|---|---|---|---|
| 3 | `V15E1:1547` §8 Phase 1d | *"`GATE-SHA`, once in the driver \| `1` \| `U7` \| `0.1 s`"* | *"`GATE-SHA`, **twice** — driver leg and arena \| `2` \| `U7` \| **`0.2 s`**"* | the whole line moves, count column **and** cost column. `U7` itself is unchanged — §6 |
| 4 | `V15E1:1550` §8 Phase 1g | count column *"**`1`** — the arena process alone … `66+6+1 = 73` accounts for **every** process"*; **cost column `3.8 s`**; and inside the same cell *"`3.8 s` is carried **above the pooled maximum** by `0.083 s`"* | count → *"**`2`** — the `--gate-sha-only` driver leg and the arena … `1+66+6+1 = 74` accounts for every process §13 declares"*; **cost → `7.6 s`**; and the margin clause **split into a unit statement**: *"the **unit** `3.8 s` is carried above the pooled maximum `3.717 s` by `0.083 s`; the row carries `2 ×` that unit"* | **round-5 H-1(a).** v5 moved only the count column and left `3.8 s` in the cost column of the same line, against row 34's `7.6` in the total below. Row 3, one table row up, already moved its whole line; the asymmetry was not deliberate. **The `0.083 s` margin is a property of the unit, not of the row**, so it is restated rather than renumbered |
| 5 | `V15E1:1645-1646` §9 | *"the **arena's own startup** is **the one span** … `3.094–3.717 s` measured over 35 arena-class runs by three parties (§7.7), **`3.8 s` as carried at §8 Phase 1g**"* (**two lines**) | *"the `--gate-sha-only` driver leg's startup is the first such span and the arena's the second; both bounded by the same arena-class band … **`2 × 3.8 s = 7.6 s` as carried at §8 Phase 1g**"* | measured false — the driver leg instantiates `Heartbeat` at `arena:1266` and emits three lines before returning, and runs `load_frozen()` first (rounds 4 and 5 both verified). **Round-5 H-1(c): `:1646` is the sentence's second line and carries the Phase 1g figure** |
| 63 | `V15E1:1346` §7.7 | *"§8 Phase 1g carries **`3.8 s`**, above the pooled maximum by `0.083 s`."* | *"§8 Phase 1g carries **`2 × 3.8 s = 7.6 s`**; the **unit** `3.8 s` is above the pooled maximum by `0.083 s`."* | **round-5 H-1(c).** Live-wrong at landing. The `0.083 s` is a unit property and survives; the row figure does not |
| 64 | **`V15E1:1368`** §7.7 | *"**So §8 Phase 1g's count is `1`, the arena alone, and it is determined by this measurement rather than inferred.**"* | *"**So §8 Phase 1g's count is `2` — the `--gate-sha-only` driver leg and the arena — and the `U9` boundary that excludes the six fidelity processes is determined by this measurement rather than inferred.**"* | **round-5 H-1(b).** Bolded, and it is the sentence that *determines* the count. Flatly false at landing, and it contradicts row 2's new §7.2 text and row 4's new count. **This is the `U11` analogue of `V15E1:1300`** — the exact site class round-4 H-2 named for `U7` |
| 28 | `V15E1:966` §6 `GATE-SHA` row | *"every frozen import, input cache **and the sixteen banked artifacts of §11** … **once in the sbatch driver**"* | **both limbs move**: *"…and the design document itself (§3)"*, and *"**twice** — once in the sbatch driver before any other process, and again in the arena at the point of use"*, **with the TOCTOU reason** | round-1 H-2 for the pass count; §5 for the scope |
| 29 | `V15E1:1840` §13 | *"`GATE-SHA` **once in the driver** before any of them"* | *"…in the driver leg before any of them **and again in the arena**"* | |
| 30 | `sbatch:17` | *"GATE-SHA runs ONCE in this driver"* | *"…runs in this driver before any of them, and again inside the arena"* | |
| 31 | `sbatch:62` | *"# GATE-SHA, ONCE, before any other process"* | *"# GATE-SHA, first of two passes, before any other process"* | |
| 32 | `arena:559-560` | docstring *"§6: every frozen import, input cache AND the sixteen banked artifacts. / ONCE, in the driver, before any other process"* (**two lines**) | *"…and the design document (§3). / the first of two passes; the arena repeats it at the point of use"* | the code twin of row 28 |
| 33 | `arena:1232` | `--gate-sha-only` help: *"the sbatch driver calls this **ONCE** before any other process (§13)"* | *"…calls this once, as the **first of two** `GATE-SHA` passes (§6, §13)"* | **round-3 I-2** |

### 2.3 The projection literals

| # | site | current | correct | why |
|---|---|---|---|---|
| 14 | `arena:46` | `PROJECTED_SECONDS = 3670.0` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`, **asserted three-way** (§7) | the heartbeat denominator |
| 15 | `arena:29-30` | docstring *"§8 projects `3670.0 s` (`4587.5 s` conservative) … (`…V15E1.md`)"* (**two lines**) | **`3673.9 s` / `4592.4 s`**, document name → V15E2 | `:29` is a sweep-C hit, `:30` a sweep-D hit |
| 16 | `config:43-44` | `3670.0` / `4587.5` (**two lines**) | **`3673.9` / `4592.4`** — and **`:43` becomes the single source** (§7) | |
| 17 | **`mint:112`** | `PROJECTED_SECONDS = 2929.9` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))` (§7) | **WRONG TODAY.** `740.1 s` below the arena's — 66 of 74 processes publish `elapsed/2929.9` against the arena's `elapsed/3670.0`, in the phase that is `68.3 %` of the budget |
| 18 | `mint:127` | `elapsed / PROJECTED_SECONDS` | **unchanged — CORRECT** | the divide site is right; only its constant was stale |
| 58 | `arena:57`, `:61`, `:68` | the arena's consumption chain (**three lines**) | **unchanged — CORRECT** | carries no literal of its own; moves with row 14 |
| 19 | `V15E1:1631-1633` §9 | *"(ERRATUM 1 set it to `2929.9 s`; CODE-R1 H-4 sets it to `3670.0 s`. **The denominator is pinned to §8 by name, so it tracks automatically**; the literal in `c06_falsifier_arena.py` and `configs/c06/…json` is updated with each correction.)"* (**three lines**) | carries **`3673.9 s`**, **deletes the false *"tracks automatically"* clause** (`:1632`), **replaces the two-file claim** (`:1633`) with the single source and the assertion (§7) | **round-5 I-3.** v5 charged only `:1631` and did not declare the row multi-line — and the two uncharged lines were precisely the two the row exists to fix. `:1632` is now charged by sweep C's widened pattern (`denominator`); `:1633` is a named extent-only line |
| 55 | **`mint:118`** | *"the mint phase — **`85.6 %`** of §8's budget — dark for its whole span"* | **`68.3 %`** | **WRONG TODAY.** CODE-R1 H-4 moved the mint share (`V15E1:1576`, `:1607`); row 13 makes the counts in this same docstring current, so leaving the share yields a half-current sentence |

### 2.4 The design pointer

| # | site | current | correct | why |
|---|---|---|---|---|
| 20 | `config:5` | `"design_document": "…DRAFT_V15E1.md"` | `"…DRAFT_V15E2.md"` | |
| 21 | **`config:6`** | `"design_sha256": "0b446b91675fd4ff8aea…"` | V15E2's digest, **derived at freeze time** (§5) | **WRONG TODAY.** CODE-R1 moved V15E1 to `8cde58aa…` and this was not updated |
| 22 | `config:7` | `design_status`: *"GO at round 15 … + ERRATUM 1 landed"* | *"… + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2 landed"* | |
| 23 | `arena:1249-1250` | `emit_halt` writes `cfg.get("design_document")` / `["design_sha256"]` (**two lines**) | publishes **both** `sha256_declared` and `sha256_derived` (via `.get(…, "NOT_DERIVED")`), **plus M-4's caveat label** | **round-4 I-1** and **round-5 M-4** |
| 24 | `arena:1433-1434` | the verdict face writes the same pair (**two lines**) | same repair, same label | |
| 25 | `sbatch:14` | *"Frozen design: …DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | two revisions stale |
| 26 | **`arena:4`** | *"Frozen design: …DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | v4's own find |
| 27 | **`mint:4`** | *"Frozen design: …DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | v4's own find |

### 2.5 §8's equation, shares, risk row and the two units

| # | site | current | correct | why |
|---|---|---|---|---|
| 34 | `V15E1:1569-1571` | `2642.3 + 1.0 + 0.7 + 0.1 + 1.3 + **3.8** + 7.0 + 1013.8 = 3670.0`; `× 1.25 = 4587.5` (**three lines**) | **`2642.4`** + 1.0 + 0.7 + 0.1 + 1.3 + **`7.6`** + 7.0 + 1013.8 = **`3673.9`**; `× 1.25 = **4592.4`** | **round-3 I-3.** Phase 1d's `0.1 → 0.2` lives *inside* the `2642.3` residue; Phase 1g's `3.8` is a named term |
| 35 | `V15E1:1576-1577` | *"mints fall from `85.6 %` to `68.3 %` … Phase 3 rises from `9.3 %` to `27.6 %`"* (**two lines**) | **unchanged — CORRECT** | `2508.3/3673.9 = 68.27 %`, `1013.8/3673.9 = 27.59 %` |
| 36 | `V15E1:1609-1610` risk row | *"2× miss `4683.8 s`, 5× miss `7725.2 s`"* (**two lines**) | **`4687.7 s = 78.1 min`**, **`7729.1 s = 128.8 min`** | minutes unchanged at one decimal |
| 37 | `V15E1:1304` §7.7 `U11` | *"the arena's is **priced once** at §8 Phase 1g"* | *"the **two arena-class startups — the `--gate-sha-only` driver leg and the arena — are priced at §8 Phase 1g**"* | **round-3 I-1** |
| 56 | `V15E1:1574` | *"`2929.9 − 273.7 + 1013.8 = 3670.0`"* | **unchanged — CORRECT, declared historical** | **round-4 M-3.** CODE-R1 H-4's provenance arithmetic, true of the total as it then stood, three lines outside row 34's block |
| 57 | `V15E1:1607-1608` | *"Mints are **`68.3 %`** … **Phase 3 is now `27.6 %`**"* (**two lines**) | **unchanged — CORRECT** | a second, separate statement of the same shares in §8's risk paragraph |

### 2.6 The ledger

| # | site | current | correct | why |
|---|---|---|---|---|
| 38 | `config:218` | `"dev_path_opens": {"expected":"mints_executed","binding":true}` | `{"expected":"mints_executed + expected_sha_dev_opens","expected_sha_dev_opens":4,"derivation":"(dev-like files in the concatenated iterable gate_sha hashes = 2) × (GATE-SHA passes = 2)","binding":true}` | **the core repair** |
| 39 | `arena:432-441` and `:468-475` | the blocked-predicate docstring **and** the frozen `+ 0` predicate with its message (**seventeen charged lines**) | the two-term predicate, **derived and asserted** against `cfg["ledger"]`; the message's decomposition becomes the derivation; **and the docstring paragraph is rewritten** | **round-5 I-1.** `:432` says *"ONE PREDICATE IS BLOCKED ON A DESIGN ERRATUM AND IS NOT ADJUSTED HERE"*, `:439-441` say it is *"implemented exactly as frozen"* and that adjusting it *"is not this lineage's call"*, and `:434-437` carry the `+2/+4` derivation the new `expected_sha_dev_opens` block supersedes. **Every one of those claims is false the moment this erratum lands.** v5 charged two lines out of the paragraph and prescribed nothing for the rest. **Round-3 M-3 retained:** the arena **derives** the dev-like count from the digest tables and **asserts** it against the declared `4`; it must not merely read it |
| 65 | `config:247`, `:249`; `V15E1:1753`, `:1755` | the two native `dev_seen_*.pt` digests, in the config and in §11 (**four lines**) | **unchanged — CORRECT** | **these four lines ARE the `2` in row 38's derivation.** Listed because the derivation is only checkable if a reader can see *which two files* it counts. Their digests do not move; their dev-likeness is what makes `expected_sha_dev_opens = 2 × 2` |
| 40 | `config:219` | `dev_label_materialisations_outside_decisions` binding `mints_executed` | moved to a `by_construction` block as a **warranted string** (corrected warrant, §3) | §3's criterion |
| 41 | `config:215`, `:216`, `:217`, `:220`, `:221` | all published as `{"expected": N, "binding": …}` (**five lines**) | `:215`, `:217`, `:221` **unchanged — CORRECT** (all instrumented or computed); `:216` and `:220` move to `by_construction` strings | the whole partition is shown, not only the two that move |
| 42 | `config:211` | `_code_review_r1.blocked_on_erratum_2` narrative | replaced by an `erratum_2` block recording what landed | |
| 43 | `V15E1:1807-1813` §12 rows (**seven lines**) | the six counter rows plus `mints_present_before_arena` — **including `:1810`'s false *"reading `lab_dev` out of the banked mint `.npz`"*** | the two-term `dev_path_opens`; the by-construction marks; M-3's *"top-level processes only"* sentence; **and `:1810`'s mechanism corrected to *"`headspace_fidelity.py` opens no `dev_seen` file — it reads only `meta` out of the banked mint `.npz` (`:68`) and references no label array at all — so the second term's fidelity contribution is zero"*** | **round-3 I-3** for the rows; the mechanism correction is this round's find (§0). The **conclusion** — fidelity opens no `dev_seen` — is unchanged and is what the `+ 0` warrant rested on |
| 44 | `V15E1:1817-1821` §12 (**four lines**) | *"Why `mints_executed` and not `66`"* — the resume warrant | amended to carry the two-term form **and** the spawn/skip fact that makes an exact `74` safe where C09 refused to bind `39` | `:1818` is now charged by sweep G's widened pattern |
| 66 | **`sbatch:103`** | *"# It reads lab_dev out of the banked mint .npz and opens no dev_seen file (§12)."* | *"# It reads only `meta` out of the banked mint .npz and opens no dev_seen file (§12)."* | **WRONG TODAY**, the same false mechanism as `:1810`, found by sweep G's widened pattern. The clause that matters — *"opens no dev_seen file"* — is true and is retained |
| 45 | `arena:449` | `self.ledger["processes_reporting"] = len(procs) + 1` | **unchanged — CORRECT** | `len(procs)` is **73** (the 6 fidelity processes and the driver leg all write ledger files) and the `+ 1` for the not-yet-flushed arena yields **74** after row 12 |
| 46 | `arena:456-457` | the `test_path_opens != 0` assertion (**two lines**) | **unchanged — CORRECT** | `test_path_opens` **is** instrumented (`_guarded_open:97`); §3's criterion does not touch it |
| 59 | `arena:458-462` | the two uninstrumented assertions (**five lines**) | **RETAINED VERBATIM** as vacuous defence-in-depth; only the *publication* moves | stated so *"never binding"* is not read as licence to delete a tripwire. Round 5 verified the call and the increment map |
| 47 | `arena:1419-1420` | reads `C06_MINTS_EXECUTED` with a fallback (**two lines**) | **unchanged — CORRECT** | what is missing is the export — row 48 |
| 48 | `sbatch` — **an addition, no existing site** | the sbatch sets neither export (round 5 confirmed: its only exports are the four thread caps, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `C09_LEDGER_DIR`) | export **`C06_MINTS_EXECUTED`** from an executed-vs-skipped counter **and `C06_PROJECTED_SECONDS`** from `config:43` (§7) | **no sweep can return this site** |
| 26† | `arena:1418` | `mints_executed = sum(1 for _ in [None])   # placeholder` | **delete** | v3's own find |

### 2.7 The `GATE-SHA` artifact count and scope

**The design document DOES increment `n`** — it is hashed through the same `sha256_of`, in the same
loop, against a declared digest, which is what the other 21 hashed artifacts get. `21 + 16 = 37`
becomes `22 + 16 = 38`. Round 5 re-derived the arithmetic and confirmed there is **no circularity**:
§11 is a table of *other* artifacts' digests and does not contain its own, the design document's
expected digest lives at `config:6`, and the config is not itself hashed — so the arena hashes a file
that does not contain its own hash.

| # | site | current | correct | why |
|---|---|---|---|---|
| 49 | `config:251-255` | `_gate_sha_count` fields and the note *"§11 declares 37 … 21 + 16 = 37"* (**five lines**) | gains `design_document: 1`; `total_§11_digests` **`21 → 22`**; note → *"§11 declares **38** = 7 + 6 + 8 imported/read/cached digests **plus the design document, whose digest §11 names and `config:6` carries**, plus the 16 banked artifacts … **22 + 16 = 38**"* | otherwise a **false arithmetic statement** in one of the five artifacts. **Round-5 M-2:** the added clause removes the implication that §11 stores a digest it does not store |
| 50 | `arena:563` | `+ list(self.cfg["frozen_sha256_input_caches"].items()):` | the design document is appended to the concatenated iterable — **the code site implementing §5** | adding a `.md` in `refine-logs/` leaves the dev-like count at **2**, so `expected_sha_dev_opens = 4` is unchanged |
| 51 | `arena:585` | `self.reports["gate_sha_artifacts"] = n` | **unchanged — CORRECT, published value moves `37 → 38`** | **round-4 M-4.** `_gate_sha_count` is the config key; this is the runtime symbol |
| 52 | `V15E1:1300` §7.7 `U7` | *"`GATE-SHA` over **all 37 §11 artifacts**"* | *"over **all 38** (8 caches + 13 modules/configs + 16 banked + **the design document**)"*, cost **unchanged at `0.13 s`** — §6 | the unit definition row 3 re-prices |
| 53 | `V15E1:1786-1787` §11 | *"`GATE-SHA`'s scope is stated in §6 as the frozen imports and the input caches plus the sixteen banked artifacts above."* (**two lines**) | *"…**and the design document itself (ERRATUM 2 §3)**"* | §11's own scope sentence |
| 54 | `V15E1:2439` | *"**No `.py` source moved** — all **37** §11 digests recompute."* | *"— all **38** §11 digests recompute **as of this document's own freeze**; the 38th is this document, whose digest is by construction the one `config:6` carries"* | round 5 called this the right formulation: it names the artifact without quoting a digest into the document that would then have to contain it |

### 2.8 The progress-coverage claims

`headspace_fidelity.py` has no progress handle — it opens at `:42`, writes JSON at `:113`, prints at
`:115`, nothing else — and is sha-frozen and run UNMODIFIED. The 6 fidelity processes' progress lines
are written by the **bash** driver at `sbatch:128-129` with a literal `-` in the elapsed and ratio
columns. *"Every python process appends"* is false for 6 of the 74, in **four** places.

| # | site | current | correct | why |
|---|---|---|---|---|
| 61 | `V15E1:1627-1629` §9 (**three lines**) | *"every python process appends through a handle opened `buffering=1`"* | *"every python process **that this lineage authors** appends … the six `headspace_fidelity.py` processes are sha-frozen and third-party, and the bash driver writes their line (`sbatch:128-129`)"* | round-4 I-3's own prescribed text |
| 60 | `mint:116` | *"§9 requires **EVERY** python process to append…"* | the same qualification | found only once sweep I was made case-insensitive |
| 62 | `mint:209` | `--progress` help: *"every python process appends to it (H-3)"* | *"…every python process this lineage authors appends to it (H-3)"* | `--help` output carrying the claim |

---

## 3. The uniform counter criterion — carried, with one warrant corrected

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
| `dev_label_materialisations_outside_decisions` | **nothing** | by-construction string | **CORRECTED (round-5 I-2, extended):** `lab_dev` is written **twice** in the executed corpus — `headspace_mint.py:323` and **`c06_falsifier_mint.py:336`**, the latter a live `np.savez` into the banked `.npz` the arena later loads — and is read by **no** path in the arena or `headspace_fidelity.py`: the arena contains no `lab_dev` reference and no generic `.npz` key iteration (`.files`, `for k in z`, `z.keys` all absent), and `headspace_fidelity.py` reads only `z["meta"]` and references no label array at all |
| `dev_or_test_labels_into_decision_quantities` | **nothing** | by-construction string | same writes, read by no decision path |

**Why the correction matters.** §3 prescribes that the string **carries its warrant** onto the
verdict face. v5's *"`lab_dev` occurs exactly once in the executed corpus"* was a checkable count
claim and it was wrong — `c06_falsifier_mint.py:336` is the more relevant of the two writes for this
counter, and it was the one omitted. **The disposition is unchanged**: the safety limb (*"written,
read by no decision path"*) holds and I verified every clause of it at source. Round 5's own repair
text is adopted and extended with the `headspace_fidelity.py` fact from §0.

Round 3 audited this criterion and confirmed **no reachable tripwire is removed**; the guard that
would have to change is itself digest-pinned in `frozen_sha256`. **Row 59 makes that concrete**: the
two uninstrumented assertions are retained verbatim, so the criterion changes what is *published*,
never what is *checked*.

**Round-3 M-1, carried.** C09's `GATE_LEDGER` also carries `per_process` and `arena_process_counts`
blocks publishing all six keys as integers per process; the *"never as integers"* claim is true only
of the aggregate `measured` block, which is the block this prescription restructures.

---

## 4. — *(merged into §5; numbering retained from v5 for the reviewer's cross-references)*

---

## 5. The `design_sha256` mechanism, and its residual

> **`GATE-SHA` gains one artifact: the design document itself.** The arena reads
> `cfg["design_document"]`, hashes the on-disk file, and compares against `cfg["design_sha256"]`,
> **HALTing on mismatch** with both digests in the message.

Round 5 verified `main()`'s order — `gate_det1 → assert_guard_active → gate_sha → load_frozen`, then
`if args.gate_sha_only: return 0` at `:1278-1280` — and that the driver leg is process 1 of 74. **The
HALT is before the first mint, at zero compute cost.**

**The residual, stated.** `configs/c06/c06_falsifier.json` is **not** in `frozen_sha256` (rounds 3, 4
and 5 each enumerated it), so `cfg["design_sha256"]` is pinned by nothing inside the job and the gate
establishes **config↔disk parity only**. Uncoordinated drift — the CODE-R1 failure that produced the
live-wrong `config:6` — is caught; **coordinated drift is not.** What anchors the declared digest
outside the job is the freeze table in `C06_FALSIFIER_IMPLEMENTATION_RECORD.md`, which **no code path
reads** (round 5: `grep -rn IMPLEMENTATION_RECORD scripts/ configs/` returns zero). The residual is
closed by procedure, not by the battery. §3 therefore says *"removes the observed subclass"*, not
*"removes the class"*.

Round 5 judged this scoping **acceptable with no further anchor required**, on the reasoning that the
sbatch is no more pinned than the config, so a second in-job declaration would only raise coordinated
drift from two coordinated edits to three. That reasoning is adopted and recorded.

**The publication path is closed** (rows 23, 24). The arena records
`self.reports["design_sha256_derived"]` when it hashes, and `emit_halt` and the verdict face publish
`sha256_declared` **and** `sha256_derived`, the latter via `.get(…, "NOT_DERIVED")` because
`emit_halt` is reachable from `GATE-DET1` before `gate_sha` runs.

**Round-5 M-4 adopted.** In coordinated drift both fields carry the same drifted value and equality
*reads* like verification. The pair therefore ships with a caveat string beside it:

> `"design_sha256_note": "declared digest is not pinned inside the job; the external anchor is the
> freeze record in refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md"`

One string, and it puts the caveat where the reader is rather than only in §5.

**Landing order, stated because the digest is self-referential.** V15E2 written **first**; its sha256
computed **after** it is final; `config:6` the **last** edit of the landing.

**Interaction with `expected_sha_dev_opens`.** The design document is a `.md` in `refine-logs/` —
not dev-like and not test-like under `c09guard` (rounds 4 and 5 both ran the predicates, under both
its V15E1 and V15E2 names) — so the dev-like count stays **2** and `expected_sha_dev_opens` stays
**4 = 2 × 2**.

---

## 6. §8: re-priced, and the meta-check over §8's whole structure

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

**`U7`'s object grows and its price does not — measured.** The added artifact is a **188 061-byte**
markdown file; sha256 over it, 7 repetitions on the login node, **median `0.000164 s`** (min
`0.000158`, max `0.000371`). Round 5 re-measured independently and got median `0.000148 s`. Against
`U7`'s `0.13 s` that is `0.11 %`–`0.13 %` — **invisible at two decimals**. `U7 = 0.13 s` unchanged,
`2 × U7 = 0.2 s` stands, §8's total unaffected by the §5 mechanism.

**`U11`'s value does not move either.** The unit stays `3.094–3.717 s` measured, carried at `3.8 s`,
`0.083 s` above the pooled maximum. What moves is the **count** — `1 → 2` — and therefore the row
`3.8 → 7.6 s`. Rows 4, 63 and 64 keep the unit statement and the row statement separate, which is
what round-5 H-1 asked for.

### The meta-check — is there an eleventh family?

Twice now a family has been missed, so rather than add sweep J and stop, I swept **§8's entire
row/unit structure** and asked, for every phase label and every unit symbol, whether this erratum
moves its count or its cost.

```bash
grep -nE '\bU[0-9]+[a-d]?\b|\bU_(acc|mF1|tie)\b|[Pp]hase [0-9]+[a-zA-Z]?' $F | sort -t: -k1,1 -k2,2n
```

**102 hits.** 30 are already charged by sweeps A–J. The **72-hit residue** names 12 phases
(`1b`, `1c`, `1e`, `1f`, `2`, `2b`, `2z`, `2D`, `3`, `4`, `5`, `7`) and 16 units (`U1`, `U2a`–`U2d`,
`U3`–`U6`, `U8`–`U10`, `U_acc`, `U_mF1`, `U_tie`), across 13 §8 table rows and their supporting prose.

**Result: no eleventh family. Of §8's 21 phase labels and 18 unit symbols, exactly two phases have a
count or cost this erratum moves — `1d` and `1g` — and zero units have a value that moves.** Every
other §8 row's count, unit and cost are untouched: Phase 3's `1013.8 s` and Phase 4's `7.0 s` were
moved by CODE-R1 H-4 and ERRATUM 1 respectively and are already landed in V15E1, and this erratum
does not touch them.

**The one candidate I had to rule out at source, because the process inventory grows.** Phase 1c is
*"ro cache loads, per process — **66 mints + the arena process itself**"*, count `67`. If the new
`--gate-sha-only` driver leg loaded an `ro` cache, that count would move to `68` and would be an
eleventh family. **It does not.** `load_frozen()` (`arena:482-…`) reads the C01 config and imports
modules; it opens no `.pt`. The arena's `load_ro` is first *called* at `arena:1284`, inside the
`dry_parity_only` branch, and in the main path at `:1300`/`:1358` — **all after the
`if args.gate_sha_only: return 0` at `:1278-1280`**. The driver leg loads no `ro` cache, so
**Phase 1c's `67` stands unchanged**, and so do Phases 1e, 1f and 6, whose counts are per-mint or
per-fidelity rather than per-process.

---

## 7. The heartbeat denominator gets a single source

Round 4's I-2 is adopted rather than argued with: §5 diagnoses hand-carried values as a class and
builds a structural fix, so applying the opposite remedy to the one quantity that has **already
measurably drifted**, across 66 of 74 processes, is not defensible.

1. **`configs/c06/c06_falsifier.json:43` (`projected_seconds`) is the single source.** Row 16.
2. **`sbatch` exports `C06_PROJECTED_SECONDS`**, parsed from that key, before any process launches —
   beside the `C06_MINTS_EXECUTED` export (row 48).
3. **`mint:112` and `arena:46`** become `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`; the
   literal survives **only** as a hand-run fallback and V15E2 says so.
4. **The arena asserts all three agree** — environment, module constant, `cfg["projected_seconds"]` —
   and HALTs on mismatch. **Round-5 M-3: the assertion is placed in the same pre-`gate_sha_only`
   block as the design-digest gate**, i.e. before `arena:1278`, so it fires in **process 1 of 74** at
   zero cost. Placed later it would fire only in process 74, after all 66 mints had already published
   ratios against a drifted denominator — the same failure one run later. §5 pins the design gate's
   ordering for exactly this reason; this one is pinned the same way and in the same words.
5. **Row 19** names the single source and the assertion, and **deletes** §9's false *"the denominator
   is pinned to §8 by name, so it tracks automatically"*.

Round 4 traced all four legs and all **68** ratio-computing processes agree after the repair (the 6
fidelity processes compute no ratio; the bash driver writes `-`).

---

## 8. Round 5's findings

**H-1 — the tenth family. Adopted in full**, plus the meta-check (§6) so a missed family is no longer
possible by the same mechanism. Sweep J is in §9 with its hit list and subtraction; row 4 now moves
`V15E1:1550`'s cost cell and splits the `0.083 s` clause into a unit statement; rows 63 and 64 are
new; `:1646` is folded into row 5 and `:1199` into row 2, both declared multi-line; `:42`, `:1350`,
`:1362`, `:1993`, `:970`, `:1047`, `:1282` are declared as surviving non-targets with their reasons.

**I-1 — the blocked-predicate docstring. Adopted:** row 39's extent is `arena:432-441` and `:468-475`,
its `correct` column states the docstring rewrite, and sweep G's pattern is widened so the paragraph's
lines are charged rather than merely asserted to be covered.

**I-2 — the `lab_dev` warrant. Adopted and extended:** §3 now names both write sites, and this round's
own verification found the neighbouring *"fidelity reads `lab_dev`"* claim false in two further
places (row 43's `:1810` and the new row 66 at `sbatch:103`) — `headspace_fidelity.py` reads only
`z["meta"]` and references no label array at all.

**I-3 — row 19's three-line extent. Adopted:** row 19 is in §1's multi-line list; `:1632` is charged
by sweep C's widened pattern; `:1633` is a named extent-only line, together with `arena:470` and
row 48's absence.

**M-1 — the mislabelled cell. Adopted:** §9 prints **excess hit-instances** and **distinct sites
returned by more than one sweep** as two separately labelled, separately computed quantities.

**M-2 — `total_§11_digests` naming. Adopted:** row 49's note gains *"whose digest §11 names and
`config:6` carries"*.

**M-3 — the assertion's placement. Adopted:** §7 item 4 pins it to the pre-`gate_sha_only` block.

**M-4 — the verdict face's caveat. Adopted:** §5's `design_sha256_note` string.

**M-5 — sorted output. Adopted:** every printed command in §9 ends in `| sort -t: -k1,1 -k2,2n`.

**Carried unchanged from earlier rounds:** the archaeology; the rejections of **(ii)** and **(iii)**;
**(iv)** declined with the TOCTOU reason; round 2's resume-stability fact; `M-2`'s *argv ≠ launch
argv* recorded and not built on; `M-3`'s *"the sbatch activates no environment"* referred to the
code/resource lineage; and every figure round 5 re-derived.

---

## 9. Implementation delta

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | two-term `dev_path_opens` derived and asserted + **docstring rewrite** (39); `processes_reporting → 74` (12); `gate_sha_passes` audit; three counters to `by_construction` with `:458-462` **retained verbatim** (59); expectations asserted against `cfg["ledger"]`; `verify_predicate` called; **design-digest gate, derived-digest publication and the caveat string** (23, 24, 50); `PROJECTED_SECONDS` from the export with a three-way assertion **in the pre-`gate_sha_only` block** (14, 58); docstrings `:4`, `:29-30`, `:559-560`, `:1232`; delete `:1418` | **≈ 70 lines** |
| `configs/c06/c06_falsifier.json` | rows 10, 11, 16, 20, 21, 22, 38, 40, 41, 42, 49 | **≈ 30 lines** |
| `c06_falsifier_cpu.sbatch` | rows 9, 25, 30, 31, 66 + **both exports** (48) | **≈ 15 lines** |
| `c06_falsifier_mint.py` | rows 27, 13, 17, 55, 60, 62 — **including two live-wrong numbers** | **6 lines** |
| **V15E2** | §6 (28), §7.2 (1, 2), §7.7 (37, 52, 63, 64), §8 (3, 4, 34, 35, 36, 56, 57), §9 (5, 19, 61), §11 (53, 65), §12 (43, 44), §13 (6, 29), §13.1 items 12 and 28 (7, 8), §16 (54) | text |

**Landing order:** V15E2 written → its sha256 computed → every code/config edit → `design_sha256`
set **last** → full dry-check battery re-run (GATE-SHA **38/38**, GATE-C01PARITY `max|diff| = 0.0`
on both datasets, the blindness grep) → implementation record updated.

---

## 10. APPENDIX — the ten sweeps, printed

**Everything below is generated output.** Each sweep gives its exact command — now ending in a sort,
per M-5 — its raw hit list as `file:line` for every hit, the charge for every hit, and its
subtraction. `UNCHARGED = 0` means no site any sweep returned is unaccounted; the extent-only lines
are named rather than counted as zero.

**File list, bound once and used by every command below:**

```bash
F="refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md \
   scripts/analysis/c06_falsifier_arena.py \
   scripts/analysis/c06_falsifier_mint.py \
   configs/c06/c06_falsifier.json \
   scripts/slurm/c06_falsifier_cpu.sbatch"
```

**Round-5 M-5 adopted:** every command below ends in `| sort -t: -k1,1 -k2,2n`, so a reader who
pastes it gets the printed list **in the printed order**, not in `$F` order. v5's lists were
set-identical but sorted; the claim is now literally true.

### Sweep A — process counts

```bash
grep -nE '\b7[234]\b' $F | sort -t: -k1,1 -k2,2n
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
grep -nE '[0-9](st|nd|rd|th)' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 3 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:458` | declared — '95th percentile' -- S5's p95, not a process ordinal |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1198` | **row 2** |
| 3 | `scripts/analysis/c06_falsifier_arena.py:286` | declared — 'rank 20 (the 21st)' -- tie-window comment |

**Subtraction B.** my sweep: **3** hits; rows account for **1**; declared non-targets: **2**; `3 = 1 + 2`; **residue: none.**

### Sweep C — projection literals and the denominator

**widened at round 5 (I-3): `denominator|heartbeat` added, which is what charges `V15E1:1632`.**

```bash
grep -nE 'projected|PROJECTED|3670|4587|2929|2933|2934|3673|4592|denominator|heartbeat' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 38 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:43` | **row 16** |
| 2 | `configs/c06/c06_falsifier.json:44` | **row 16** |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:158` | declared — R2 rule row naming the line-buffered heartbeat -- carries no projection literal |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:397` | declared — GATE-DOMAIN's denominator -- a different quantity |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:853` | declared — the final heartbeat line as a HALT output -- no literal |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1141` | declared — 'near-zero denominator' in the rho discussion |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1156` | declared — the rho denominator n_D |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1437` | declared — 'near-zero denominator' in the rho discussion |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1570` | **row 34** |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1571` | **row 34** |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1574` | **row 56** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1581` | declared — provenance 2930.7->2933.9 -- historical |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1584` | declared — provenance 2933.9->2934.5 -- historical |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1592` | declared — heartbeat interval statement -- no literal |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1631` | **row 19** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1632` | **row 19** |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1880` | declared — 'per-item denominator' in the leakage items |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1898` | declared — the mismatch-rate denominator |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1901` | declared — SS13.1's 'GATE-POP and heartbeat' heading |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1904` | **row 7** |
| 21 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1905` | declared — the closing word of item 12's sentence; the elapsed/projected denominator reference is unchanged by row 7 |
| 22 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2080` | declared — round-14's id_hash_permutation share, frozen at the total it was taken against -- historical |
| 23 | `scripts/analysis/c06_falsifier_arena.py:16` | declared — arena module docstring describing SS9 heartbeat -- true of the arena |
| 24 | `scripts/analysis/c06_falsifier_arena.py:29` | **row 15** |
| 25 | `scripts/analysis/c06_falsifier_arena.py:46` | **row 14** |
| 26 | `scripts/analysis/c06_falsifier_arena.py:53` | declared — the heartbeat section banner |
| 27 | `scripts/analysis/c06_falsifier_arena.py:57` | **row 58** |
| 28 | `scripts/analysis/c06_falsifier_arena.py:61` | **row 58** |
| 29 | `scripts/analysis/c06_falsifier_arena.py:68` | **row 58** |
| 30 | `scripts/analysis/c06_falsifier_arena.py:90` | declared — GateFailure docstring naming the final heartbeat line |
| 31 | `scripts/analysis/c06_falsifier_arena.py:1240` | declared — CODE-R1 I-1 docstring on the RuntimeError context in the final heartbeat line |
| 32 | `scripts/analysis/c06_falsifier_arena.py:1471` | declared — writes the RuntimeError context to the final heartbeat line |
| 33 | `scripts/analysis/c06_falsifier_mint.py:112` | **row 17** |
| 34 | `scripts/analysis/c06_falsifier_mint.py:115` | declared — def heartbeat(...) -- the function definition |
| 35 | `scripts/analysis/c06_falsifier_mint.py:127` | **row 18** |
| 36 | `scripts/analysis/c06_falsifier_mint.py:214` | declared — heartbeat call site MINT-START |
| 37 | `scripts/analysis/c06_falsifier_mint.py:219` | declared — heartbeat call site MINT-SKIP |
| 38 | `scripts/analysis/c06_falsifier_mint.py:343` | declared — heartbeat call site MINT-DONE |

**Subtraction C.** my sweep: **38** hits; rows account for **15**; declared non-targets: **23**; `38 = 15 + 23`; **residue: none.**

### Sweep D — design pointer

```bash
grep -nE 'design_document|design_sha256|design_status|PREREG_DRAFT|Frozen design|V15E|"design"' $F | sort -t: -k1,1 -k2,2n
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

Deliberately unanchored; the substring hit is declared, not patterned away.

```bash
grep -nE 'ONCE|once|one span|twice|single pass' $F | sort -t: -k1,1 -k2,2n
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

### Sweep F — §8 equation, shares, risk row, and the two units it re-prices

```bash
grep -nE '2642\.|2508\.|1013\.8|4683\.|7725\.|4687\.|7729\.|68\.3|27\.6|61\.2|76\.5|85\.6|\bU7\b|\bU11\b' $F | sort -t: -k1,1 -k2,2n
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

### Sweep G — ledger quantities and the blocked-predicate warrant

**widened at round 5 (I-1): the docstring idiom and the dev-like vocabulary added, which is what charges `arena:432-441` and the two `dev_seen` caches.**

```bash
grep -nE 'dev_path_opens|mints_executed|processes_reporting|expected_sha_dev_opens|test_path_opens|banked_trainlog_opens|dev_label_materialisations_outside_decisions|test_label_materialisations|dev_or_test_labels_into_decision_quantities|mints_present_before_arena|ERRATUM REQUIRED|NOT ADJUSTED|BLOCKED ON|blocked_on|implemented exactly as frozen|lineage.s call|is_dev_like|dev_seen|builtins\.open|gate-sha-only' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 82 hits, every one accounted:**

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
| 10 | `configs/c06/c06_falsifier.json:247` | **row 65** |
| 11 | `configs/c06/c06_falsifier.json:249` | **row 65** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:180` | declared — SS3.1: the native dev_seen is opened by headspace_mint.py:199 and covered by GATE-SHA -- true |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:181` | declared — continuation of :180 -- true, unchanged |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:224` | declared — the unconditional dev load at :199 and the lab_dev write at :322-324 -- true, unchanged |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:226` | declared — 'All 66 mints open the native dev_seen' -- true, unchanged |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1273` | declared — read-map row (l): fidelity opens the mint npz and the trainlogs, no dev_seen -- CORRECT as written |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1567` | declared — Phase 7 row naming mints_present_before_arena among gate names -- that counter is unchanged |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1753` | **row 65** |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1755` | **row 65** |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1807` | **row 43** |
| 21 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1808` | **row 43** |
| 22 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1809` | **row 43** |
| 23 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1810` | **row 43** |
| 24 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1811` | **row 43** |
| 25 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1812` | **row 43** |
| 26 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1813` | **row 43** |
| 27 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1817` | **row 44** |
| 28 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1818` | **row 44** |
| 29 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1819` | **row 44** |
| 30 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1821` | **row 44** |
| 31 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1825` | declared — SS12's dev-label paragraph: headspace_mint.py:199 loads the native dev_seen unconditionally -- true |
| 32 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1827` | declared — 'None reaches a decision quantity' -- true, and is the limb row 43 preserves |
| 33 | `scripts/analysis/c06_falsifier_arena.py:26` | declared — module docstring: no dev_seen_*-ro_* and no test_seen file reachable -- true |
| 34 | `scripts/analysis/c06_falsifier_arena.py:411` | declared — docstring 'still publish test_path_opens: 0' -- that counter stays measured and binding |
| 35 | `scripts/analysis/c06_falsifier_arena.py:424` | declared — gate_ledger signature -- unchanged |
| 36 | `scripts/analysis/c06_falsifier_arena.py:432` | **row 39** |
| 37 | `scripts/analysis/c06_falsifier_arena.py:433` | **row 39** |
| 38 | `scripts/analysis/c06_falsifier_arena.py:434` | **row 39** |
| 39 | `scripts/analysis/c06_falsifier_arena.py:435` | **row 39** |
| 40 | `scripts/analysis/c06_falsifier_arena.py:436` | **row 39** |
| 41 | `scripts/analysis/c06_falsifier_arena.py:437` | **row 39** |
| 42 | `scripts/analysis/c06_falsifier_arena.py:438` | **row 39** |
| 43 | `scripts/analysis/c06_falsifier_arena.py:439` | **row 39** |
| 44 | `scripts/analysis/c06_falsifier_arena.py:440` | **row 39** |
| 45 | `scripts/analysis/c06_falsifier_arena.py:441` | **row 39** |
| 46 | `scripts/analysis/c06_falsifier_arena.py:449` | **row 45** |
| 47 | `scripts/analysis/c06_falsifier_arena.py:450` | declared — copies mints_present_before_arena into the ledger -- unchanged |
| 48 | `scripts/analysis/c06_falsifier_arena.py:451` | declared — continuation of :450 |
| 49 | `scripts/analysis/c06_falsifier_arena.py:452` | declared — copies mints_executed into the ledger -- unchanged |
| 50 | `scripts/analysis/c06_falsifier_arena.py:456` | **row 46** |
| 51 | `scripts/analysis/c06_falsifier_arena.py:457` | **row 46** |
| 52 | `scripts/analysis/c06_falsifier_arena.py:458` | **row 59** |
| 53 | `scripts/analysis/c06_falsifier_arena.py:459` | **row 59** |
| 54 | `scripts/analysis/c06_falsifier_arena.py:460` | **row 59** |
| 55 | `scripts/analysis/c06_falsifier_arena.py:461` | **row 59** |
| 56 | `scripts/analysis/c06_falsifier_arena.py:462` | **row 59** |
| 57 | `scripts/analysis/c06_falsifier_arena.py:463` | declared — mints_present_before_arena != 66 assertion -- unchanged |
| 58 | `scripts/analysis/c06_falsifier_arena.py:464` | declared — continuation of :463 |
| 59 | `scripts/analysis/c06_falsifier_arena.py:465` | **row 12** |
| 60 | `scripts/analysis/c06_falsifier_arena.py:466` | **row 12** |
| 61 | `scripts/analysis/c06_falsifier_arena.py:467` | **row 12** |
| 62 | `scripts/analysis/c06_falsifier_arena.py:468` | **row 39** |
| 63 | `scripts/analysis/c06_falsifier_arena.py:469` | **row 39** |
| 64 | `scripts/analysis/c06_falsifier_arena.py:471` | **row 39** |
| 65 | `scripts/analysis/c06_falsifier_arena.py:472` | **row 39** |
| 66 | `scripts/analysis/c06_falsifier_arena.py:473` | **row 39** |
| 67 | `scripts/analysis/c06_falsifier_arena.py:475` | **row 39** |
| 68 | `scripts/analysis/c06_falsifier_arena.py:560` | **row 32** |
| 69 | `scripts/analysis/c06_falsifier_arena.py:607` | declared — gate_fold docstring naming mints_present_before_arena -- unchanged |
| 70 | `scripts/analysis/c06_falsifier_arena.py:626` | declared — the mints_present_before_arena != 66 message -- unchanged |
| 71 | `scripts/analysis/c06_falsifier_arena.py:628` | declared — the site that counts present mints -- unchanged |
| 72 | `scripts/analysis/c06_falsifier_arena.py:1230` | declared — the --gate-sha-only argument definition; row 33 carries its help string |
| 73 | `scripts/analysis/c06_falsifier_arena.py:1418` | **row 26†** |
| 74 | `scripts/analysis/c06_falsifier_arena.py:1419` | **row 47** |
| 75 | `scripts/analysis/c06_falsifier_arena.py:1420` | **row 47** |
| 76 | `scripts/analysis/c06_falsifier_arena.py:1421` | declared — the gate_ledger call -- unchanged |
| 77 | `scripts/analysis/c06_falsifier_mint.py:166` | declared — mint docstring: no dev_seen or test_seen file reachable -- true |
| 78 | `scripts/analysis/c06_falsifier_mint.py:217` | declared — comment citing SS12's clause title 'Why mints_executed and not 66' -- row 44 keeps that title |
| 79 | `scripts/analysis/c06_falsifier_mint.py:346` | declared — prints guard.LEDGER['dev_path_opens'] -- reads the counter, declares no expectation |
| 80 | `scripts/slurm/c06_falsifier_cpu.sbatch:36` | declared — narrative on why sitecustomize matters for test_path_opens -- unchanged |
| 81 | `scripts/slurm/c06_falsifier_cpu.sbatch:64` | declared — the driver-leg invocation itself; rows 30 and 31 carry its comments |
| 82 | `scripts/slurm/c06_falsifier_cpu.sbatch:103` | **row 66** |

**Subtraction G.** my sweep: **82** hits; rows account for **56**; declared non-targets: **26**; `82 = 56 + 26`; **residue: none.**

### Sweep H — GATE-SHA artifact count and scope

```bash
grep -nE '\b37\b|\b38\b|gate_sha_artifacts|_gate_sha_count|total_.11_digests|banked_unhashed_artifacts|imported_modules|read_for_definitions|input_caches|sixteen banked|banked artifacts|scope is stated' $F | sort -t: -k1,1 -k2,2n
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

### Sweep I — progress-coverage claims

Run with `grep -i`; the case-sensitive form misses `mint:116`'s uppercase `EVERY`.

```bash
grep -niE 'every python process|buffering=1|appends through|append-without-interleaving|progress file|append-only' $F | sort -t: -k1,1 -k2,2n
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

### Sweep J — Phase 1g / Phase 1d re-pricing

**NEW at round 5 (H-1)** — the tenth family.

```bash
grep -nE '\b3\.8\b|\b7\.6\b|[Pp]hase 1[dg]' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 17 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:42` | declared — round-13's ruling that the arena class is bounded by the UNIT 3.8 s -- survives the row re-price |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:970` | declared — 'two criteria at once' -- ordinary English |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1047` | declared — a `3.5x`-`7.6x` ratio in the row-renormalisation argument -- not a cost |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1199` | **row 2** |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1282` | declared — the section heading '### 7.6 GATE-C01PARITY' -- a section number, not a cost |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1304` | **row 37** |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1346` | **row 63** |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1350` | declared — about the UNIT and Phase 1d's one-decimal rounding -- both survive the re-price |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1362` | declared — 'the count in SS8 Phase 1g turns on this clause' -- still true at count 2 |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1368` | **row 64** |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1550` | **row 4** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1569` | **row 34** |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1581` | declared — provenance 2930.7->2933.9 -- historical |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1583` | declared — 'once sklearn is restored' -- ordinary English |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1646` | **row 5** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1993` | declared — 'SS8 Phase 1g's unit is undeterminable without it' -- about the UNIT, survives |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2002` | declared — handoff instruction to re-measure U11 -- carries no figure |

**Subtraction J.** my sweep: **17** hits; rows account for **7**; declared non-targets: **10**; `17 = 7 + 10`; **residue: none.**

### Global partition — the ten sweeps together

| quantity | value |
|---|---|
| hit-instances across the ten sweeps | **265** |
| distinct sites (deduplicated) | **240** |
| **excess hit-instances** (`265 − 240`) | **25** |
| distinct sites returned by more than one sweep | **21** |
| distinct sites charged to a row | **128** |
| distinct sites charged to a declaration | **112** |
| sites returned by a sweep and charged to nothing (`UNCHARGED`) | **0** |

`240 = 128 + 112`. **Residue: none, in either direction.**

**Round-5 M-1 adopted.** v5 printed one row labelled *"sites returned by more than one sweep"*
carrying the value `16`, which was the count of **excess hit-instances**; the two are different
quantities because `V15E1:1904` is returned by three sweeps and contributes two excess
instances by itself. Both are now printed, separately labelled, and computed by the script.

**Extent-only lines — 2, named.** A row's *extent* is the lines its edit touches; its
*charged sites* are the lines a sweep returns. Every charged site is in its row's extent, and
these are the only extent lines no pattern returns:

* **`refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1633`** — row 19.
* **`scripts/analysis/c06_falsifier_arena.py:470`** — row 39.
* **row 48** has no site at all: it *adds* the two sbatch exports, and a grep over the current
  files cannot return a line that does not exist yet.

These three are stated rather than patterned into existence. `V15E1:1633` names *where the literal
lives* rather than stating the quantity, so it falls outside every quantity family by
construction; `arena:470` is a bare `fails.append(` continuation. Both are inside ranges whose
other lines are charged, so neither can be missed by an implementer working from the row.

**Rows: 67 defined** (`1`–`66` plus `26†`), of which **1** is siteless (row 48).


---

## 11. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no arena run; no `--gate-sha-only` leg run; no GPU, no
SLURM job, no commit, no `TARGET_STATE.json` edit; `artifacts/c06_falsifier/` never created.

**Compute used:** file and review reads; the ten `grep` sweeps of §10 plus wider variants of each and
the §6 meta-check sweep; `sha256sum` over the artifacts; **one login-node timing of `hashlib.sha256`
over the 188 061-byte design document, 7 repetitions, reported in §6**; static reads of
`headspace_fidelity.py`, `c06_falsifier_arena.py`, `c06_falsifier_mint.py` and the sbatch; and
arithmetic. The charge table, every hit list and every count in §10 are produced by a script over the
sweep output.

**Nothing is edited.** All five artifacts carry their post-CODE-R1 hashes, re-verified before and
after:

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` |

`…_PROPOSAL.md`, `…_V2.md`, `…_V3.md`, `…_V4.md` and `…_V5.md` stay on disk byte-unmodified.

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`.
**The battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under
v1–v5 as specified.**
