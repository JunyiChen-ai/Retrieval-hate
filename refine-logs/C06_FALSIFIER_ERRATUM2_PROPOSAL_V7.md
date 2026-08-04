# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v7**

**Supersedes** `…_PROPOSAL_V6.md` (`05c93599b9ee45450a68…`) → `…_V5.md` (`c41a0223bdf6db709114…`) →
`…_V4.md` (`0b4940416abd1fb4bf79…`) → `…_V3.md` (`48f4e0153103cc608884…`) → `…_V2.md`
(`4225bea3cc9907d38e2e…`) → `…_PROPOSAL.md` (`f063c388c4afabdb7964…`).
**All six stay on disk, byte-unmodified.**
*Against:* `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudications:* r1 (0C/3H/3I/3M) → v2; r2 (0C/2H/3I/3M) → v3; r3 (0C/3H/3I/3M) → v4;
r4 (0C/2H/3I/4M) → v5; r5 (0C/1H/3I/5M) → v6; **r6 (0C/1H/3I/5M)** → this.
*Status:* **PROPOSAL. Nothing is landed.** All five artifacts carry their post-CODE-R1 hashes.

---

## 0. What round 6 settled, and the one thing it blocked

**Round 6 ran every one of v6's ten printed commands and reproduced every hit list byte-identically
including sort order**, checked position by position rather than as sets. It recomputed the partition
from the raw lists (`265 → 240 = 128 + 112`, `UNCHARGED = 0`, 25 excess / 21 multi-sweep), reproduced
the ten per-sweep subtractions, found **zero divergent cross-sweep charges**, independently derived
the 26-row multi-line list and matched it exactly, verified all three extent-only reasons at source,
reproduced the meta-check at `102 = 30 + 72` with its 12 phases and 16 units, confirmed §8's 21 phase
labels and 18 unit symbols, agreed that two phases move and zero units move, **confirmed Phase 1c's
`67` at source**, confirmed the fourth live-wrong claim and both its repairs, re-derived
`expected_sha_dev_opens = 2 × 2 = 4`, re-verified row 45's `73 + 1 = 74` through `atexit`-flushed
per-pid ledgers, recomputed §8's full arithmetic, and matched §9's delta against all 67 rows finding
no duplication and nothing silently dropped. Its words: *"I found nothing wrong inside the method v6
built."* **None of that is reopened here.**

**Blocked on the eleventh family — the one §6 was built to prove could not exist.** Two sites state
the process inventory as a **decomposition with no total written down**:

* **`V15E1:1814`** — §12's own predicate table: `| processes reporting | **66 mints + 6 fidelity + 1
  arena** | yes — HALT on any mismatch |`
* **`config:42`** — `"process_order": "66 mints -> 6 fidelity -> 1 arena",`

Neither carries a `72|73|74` numeral, so sweep A cannot see them. Sweep G matches
`processes_reporting` with an **underscore**; §12's table writes it with a **space**. And §6's
meta-check sweeps §8's *unit and phase* vocabulary, which the process inventory is not. Both are
flatly wrong at landing, and `V15E1:1814` is the design's own declaration of the **binding,
HALT-on-mismatch predicate that round 1's H-1 opened this erratum on** — never named in six proposals
and five reviews.

**Why the method missed them, stated plainly, because it is the lesson.** Every completeness domain
this lineage has built has been keyed on a **form**: v5's ten patterns on the forms the quantities
happened to take at the sites already known, and v6's meta-check on the forms of *one section's*
vocabulary. A form-keyed domain can only find sites written in forms someone already thought of.

**v7 re-keys the domain on the quantities.** §1 enumerates every quantity this erratum moves, then —
for each — every **linguistic form** that quantity can be stated in, and gives each form its sweep.
The form list is printed as a table so a reviewer can attack the *form enumeration* directly rather
than re-deriving it from twelve regexes. Two sweeps are new: **K** for the decomposition and the
spaced identifier, and **L** for the spelled-out numeral, added so that form is *checked* rather than
assumed absent.

---

## 1. THE COMPLETENESS DOMAIN — quantities × forms

**The claim this table makes:** the erratum moves **ten** quantities; a statement of a quantity in
these artifacts takes one of **nine** linguistic forms; every occupied `(quantity, form)` cell is
covered by at least one sweep; and every sweep's every hit is charged in §10. **An empty cell is a
claim that the quantity is never written in that form**, and it is checkable by grepping for it.

| # | quantity this erratum moves | F1 numeral | F2 decomposition, no total | F3 ordinal | F4 prose idiom | F5 identifier | F6 spaced identifier | F7 unit / phase symbol | F8 alias or superseded value | F9 spelled-out |
|---|---|---|---|---|---|---|---|---|---|---|
| **Q1** | process inventory `73 → 74` | **A** | **K** | **B** | **E**, **J** | **G** | **K** | **J** | — | **L** |
| **Q2** | dev-open expectation `+0 → +4` | **G** | **G** | — | **G** | **G** | — | — | **G** | **L** |
| **Q3** | `GATE-SHA` scope and count `37 → 38` | **H** | **H**, **K** | — | **H** | **H** | — | **F** | — | **L** |
| **Q4** | `GATE-SHA` pass count `1 → 2` | **G**, **H** | — | **E** | **E** | **G** | — | **J** | — | **L** |
| **Q5** | Phase 1d / 1g counts and 1g cost | **J**, **F** | **A**, **J**, **K** | — | **J** | — | — | **F**, **J** | **J** | — |
| **Q6** | §8 total and second-order figures | **C**, **F** | **F** | — | **F** | **C** | — | **F** | **C** | — |
| **Q7** | heartbeat denominator | **C** | — | — | **C**, **I** | **C** | — | — | **C** | — |
| **Q8** | design pointer and digest | **D** | — | — | **D** | **D** | — | — | **D** | — |
| **Q9** | ledger counter dispositions | **G** | — | — | **G** | **G** | **K** | — | **G** | — |
| **Q10** | progress-coverage claims | **A** | — | — | **I** | **I** | — | — | **C** | — |

**The forms, defined so the table is falsifiable.**

| form | definition | example at source |
|---|---|---|
| **F1 numeral** | the quantity written as a number | `73`, `37`, `3670.0`, `3.8 s` |
| **F2 decomposition** | a sum of parts with **no total written** | `66 mints + 6 fidelity + 1 arena`; `21 + 16 = 37`; `+2 per GATE-SHA process, +4 total` |
| **F3 ordinal** | position rather than count | *"the **73rd** process"*, *"the **first of two**"* |
| **F4 prose idiom** | stated in words with no numeral | *"once in the driver"*, *"the one span"*, *"every python process"*, *"count is `1`, the arena alone"* |
| **F5 identifier** | a code or config key naming the quantity | `processes_reporting`, `PROJECTED_SECONDS`, `design_sha256`, `_gate_sha_count` |
| **F6 spaced identifier** | the same key written as prose in a table | *"processes reporting"* (`V15E1:1814`), `process_order`'s value |
| **F7 unit / phase symbol** | reference through §8's vocabulary | `U7`, `U11`, `Phase 1d`, `Phase 1g` |
| **F8 alias / superseded value** | a paraphrase, or a prior value of the same quantity | *"denominator"*, *"heartbeat"*, `3.2 s` (U11's superseded unit), `2929.9` |
| **F9 spelled-out** | the numeral written in words | *"sixteen banked"*, *"two processes"*; **`seventy-*` / `sixty-*` are absent — verified by sweep L, not assumed** |

**What the empty cells assert.** Q5–Q10 have no F3 (ordinal) because only the process inventory and
the pass count have an order to be ordinal about. Q7–Q10 have no F2 because none is a sum of parts.
Q5–Q10 have no F9: sweep L returns 9 hits over all five artifacts and **not one of them states a
moved quantity in a spelled-out form that some other sweep does not already charge** — the two
uncharged are `V15E1:613` and `:1405`, both *"sixty"* meaning the 60 `ρ` cells. Q1's F8 cell is empty
because the process inventory has no alias and no superseded value: it has been `73` since v1.

**This is the repair for round-6 H-1's diagnosis**, which is worth quoting because it is the whole
lesson: *"the eleventh family exists because §6 sweeps §8's unit/phase vocabulary, and the process
inventory is not a §8 unit or phase."* A form-keyed domain cannot close. A quantity-keyed domain with
an explicit form enumeration can be attacked at the form list — which is the object a reviewer should
be attacking.

**§6's meta-check is re-scoped accordingly** and no longer claims more than it proves.

---

## 2. The accounting rule

> **Every site returned by any sweep is charged to exactly one row or exactly one declaration.** A row
> may collect sites returned by several sweeps — `V15E1:1904` is a `73` site, a `projected` site and a
> coverage-claim site — so the same row appears in several sweeps' tables with the **same** charge.
> Because the charge is unique, a site returned by several sweeps is not counted twice.

A row's **extent** is the lines its edit touches; its **charged sites** are the lines a sweep returns;
extent ⊇ charged sites. §10 names every **extent-only** line rather than contorting a pattern to
manufacture a hit for it. Round 6 found v6's extent-only list incomplete — it claimed two lines and
three more existed — so §10 now prints the verified list with its two categories counted separately
(round-6 M-3).

**Multi-line rows are declared everywhere they occur.** The **29** rows with more than one charged
site are **2, 5, 7, 12, 15, 16, 19, 23, 24, 32, 34, 35, 36, 39, 41, 43, 44, 46, 47, 49, 53, 54, 57,
58, 59, 61, 64, 65, 69**. Rows **5, 44, 54 and 64** join the list this round, all four found by
round 6; row **69** is new and multi-line (three sbatch banners). Rows **67**, **68** and **70** are
new and single-site, and are correctly absent from this list.

---

## 3. FULL ENUMERATION — 71 rows

Rows marked **CORRECT** need no edit and are listed anyway, because completeness is verified by
subtraction, not by trust. **All 71 appear in the consolidated delta at §9, including the CORRECT
ones (round-6 M-5).**

### 3.1 Q1 — the process inventory, `73 → 74`

| # | site | current | correct | why |
|---|---|---|---|---|
| 1 | `V15E1:1197` | *"those **72** processes"* | **unchanged — CORRECT** | 66 + 6 = 72 is exactly right at 74 |
| 2 | `V15E1:1198-1199` (**2 lines**) | *"The **73rd** process, the arena, is a different case and is priced separately at §8 Phase 1g"* | *"**The remaining two processes** — the `--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8 Phase 1g"* | round-3 H-3; no ordinal, so it cannot drift against §13 again. `72` (row 1) `+ 2 = 74` |
| 6 | `V15E1:1839` | *"**73** processes in the order 66 mints → 6 fidelity → 1 arena"* | *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena"* | |
| 7 | `V15E1:1903-1904` (**2 lines**) | *"…append-without-interleaving across all **73** processes"* | *"…across all **74**, **of which the 68 this lineage authors append through a handle opened `buffering=1`; the six `headspace_fidelity.py` processes are sha-frozen and third-party and the bash driver writes their line (`sbatch:128-129`)**"* | round-4 I-3 |
| 8 | `V15E1:2017` | *"active in all **73** processes"* | *"in all **74**"* | |
| 9 | `sbatch:16` | *"73 processes, in this order: 66 mints -> 6 fidelity -> 1 arena"* | *"74 processes, in this order: 1 GATE-SHA driver leg -> 66 mints -> 6 fidelity -> 1 arena"* | **both limbs** — the numeral and the decomposition |
| 10 | `config:41` | `{"mints":66,"fidelity":6,"arena":1,"total":73}` | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | |
| 11 | `config:222` | `"processes_reporting": {"expected":73,"binding":true}` | `{"expected":74,"decomposition":"1+66+6+1","binding":true}` | |
| 12 | `arena:465-467` (**3 lines**) | `!= 73` and its message | `!= 74`, read from `cfg["ledger"]` | |
| 13 | `mint:117` | *"**72 of 73** processes previously wrote nothing"* | *"**73 of 74**"* | |
| **67** | **`V15E1:1814`** §12 | `\| processes reporting \| **66 mints + 6 fidelity + 1 arena** \| yes — HALT on any mismatch \|` | `\| processes reporting \| **1 `GATE-SHA` driver leg + 66 mints + 6 fidelity + 1 arena** \| yes — HALT on any mismatch \|` | **round-6 H-1.** §12's own declaration of the binding predicate this erratum exists to repair. Invisible to sweeps A–J: no `72\|73\|74` numeral, and the key is written **with a space** where sweep G matches an underscore. Left alone, a landed V15E2 carries **74** in §7.2, §8, §13 and the config and **73** in §12 — on a HALT-on-mismatch predicate |
| **68** | **`config:42`** | `"process_order": "66 mints -> 6 fidelity -> 1 arena"` | `"process_order": "1 gate-sha driver leg -> 66 mints -> 6 fidelity -> 1 arena"` | **round-6 H-1.** Sits **between** `config:41` (row 10, → four process classes) and `config:43` (row 16). Left alone, `:41` enumerates four classes and `:42` declares an order over three, contradicting rows 6 and 9 |
| **69** | `sbatch:67`, `:101`, `:138` (**3 lines**) | the three block banners `66 mints` / `6 fidelity` / `1 arena` | **unchanged — CORRECT** | each banner is accurate **for its own block**. What was missing is a fourth banner over the driver-leg block, which **row 31 now adds** — so the four banners and the four process classes agree |

### 3.2 Q4/Q5 — the second `GATE-SHA` pass and the two priced phases

| # | site | current | correct | why |
|---|---|---|---|---|
| 3 | `V15E1:1547` §8 Phase 1d | *"`GATE-SHA`, once in the driver \| `1` \| `U7` \| `0.1 s`"* | *"`GATE-SHA`, **twice** — driver leg and arena \| `2` \| `U7` \| **`0.2 s`**"* | the whole line, count **and** cost |
| 4 | `V15E1:1550` §8 Phase 1g | count *"**`1`** — the arena process alone … `66+6+1 = 73` accounts for **every** process"*; cost **`3.8 s`**; and *"`3.8 s` is carried above the pooled maximum by `0.083 s`"* | count → *"**`2`** … `1+66+6+1 = 74` accounts for every process §13 declares"*; cost → **`7.6 s`**; margin split into a **unit** statement: *"the **unit** `3.8 s` is carried above the pooled maximum `3.717 s` by `0.083 s`; the row carries `2 ×` that unit"* | round-5 H-1. Round 6 verified the result is §8-internally consistent |
| 5 | `V15E1:1644-1646` (**3 lines**) | *"The **arena's own startup** is **the one span** that precedes any python-side line … `3.8 s` as carried at §8 Phase 1g"* | *"the `--gate-sha-only` driver leg's startup is the first such span and the arena's the second; both bounded by the same arena-class band … **`2 × 3.8 s = 7.6 s`** as carried at §8 Phase 1g"* | **round-6 I-2:** the subject the rewrite replaces (*"The **arena's own startup**"*) is on `:1644`, outside v6's declared `:1645-1646`; round 2's H-1 item 4 originally scoped this as `:1643-1651` and it had been silently narrowed. **The driver leg emits FOUR heartbeat lines before returning, not three** — `GATE-DET1` (`arena:1270`), `GUARD` (`:1272`), `GATE-SHA` (`:586`, from the call at `:1276`) and `GATE-SHA-ONLY` (`:1279`) — **as round 2's H-1 item 4 enumerated them by name** (round-6 I-3; v5 and v6 carried *"three"*) |
| **63** | `V15E1:1346` | *"§8 Phase 1g carries **`3.8 s`**, above the pooled maximum by `0.083 s`."* | *"§8 Phase 1g carries **`2 × 3.8 s = 7.6 s`**; the **unit** `3.8 s` is above the pooled maximum by `0.083 s`."* | round-5 H-1(c) |
| **64** | **`V15E1:1368-1369`** (**2 lines**) | *"**So §8 Phase 1g's count is `1`, the arena alone, and it is determined by this measurement rather than inferred.**"* | *"**So §8 Phase 1g's count is `2` — the `--gate-sha-only` driver leg and the arena — and the `U9` boundary that excludes the six fidelity processes is determined by this measurement rather than inferred.**"* | round-5 H-1(b). **Round-6 I-1: the false words — *"count is `1`, the arena alone"* — are on `:1369`, which v6 left uncharged.** Sweep J now returns it on the `count is` term |
| **70** | `V15E1:1582` | *"on the `count = 1` reading **that §7.7's `U9` measurement establishes**"* | *"on the `count = 1` reading **§7.7 then carried**"* | **round-6 M-1.** The numbers on `:1581-1582` are unambiguously historical; the **attribution is present tense**, so after row 64 lands the sentence says §7.7 establishes count `1` when it establishes `2`. Row 56 exists for exactly this ambiguity one paragraph up |
| 28 | `V15E1:966` | *"…the sixteen banked artifacts of §11 … **once in the sbatch driver**"* | **both limbs**: *"…**and the design document itself (§5)**"*, and *"**twice** — once in the sbatch driver before any other process, and again in the arena at the point of use"*, with the TOCTOU reason | |
| 29 | `V15E1:1840` | *"`GATE-SHA` **once in the driver** before any of them"* | *"…in the driver leg before any of them **and again in the arena**"* | |
| 30 | `sbatch:17` | *"GATE-SHA runs ONCE in this driver"* | *"…runs in this driver before any of them, and again inside the arena"* | |
| 31 | `sbatch:62` | *"# ---- GATE-SHA, ONCE, before any other process (§6, §13) ----"* | *"# ==================== 1 GATE-SHA driver leg ===================="* followed by *"# GATE-SHA, first of two passes, before any other process (§6, §13)"* | the pass-count fix **plus the fourth block banner** row 69 refers to, so the sbatch's four banners match its four process classes |
| 32 | `arena:559-560` (**2 lines**) | *"§6: every frozen import, input cache AND the sixteen banked artifacts. / ONCE, in the driver…"* | *"…and the design document (§5). / the first of two passes; the arena repeats it at the point of use"* | |
| 33 | `arena:1232` | `--gate-sha-only` help: *"calls this **ONCE** before any other process (§13)"* | *"…calls this once, as the **first of two** `GATE-SHA` passes (§6, §13)"* | |

### 3.3 Q7 — the heartbeat denominator

| # | site | current | correct | why |
|---|---|---|---|---|
| 14 | `arena:46` | `PROJECTED_SECONDS = 3670.0` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`, **asserted three-way** (§7) | |
| 15 | `arena:29-30` (**2 lines**) | *"§8 projects `3670.0 s` (`4587.5 s` conservative) … (`…V15E1.md`)"* | **`3673.9 s` / `4592.4 s`**, name → V15E2 | |
| 16 | `config:43-44` (**2 lines**) | `3670.0` / `4587.5` | **`3673.9` / `4592.4`**, and **`:43` becomes the single source** | |
| 17 | **`mint:112`** | `PROJECTED_SECONDS = 2929.9` | `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))` | **WRONG TODAY** — `740.1 s` below the arena's, in the phase that is `68.3 %` of the budget |
| 18 | `mint:127` | `elapsed / PROJECTED_SECONDS` | **unchanged — CORRECT** | |
| 58 | `arena:57`, `:61`, `:68` (**3 lines**) | the arena's consumption chain | **unchanged — CORRECT** | |
| 19 | `V15E1:1631-1633` (**3 lines**) | *"…**The denominator is pinned to §8 by name, so it tracks automatically**; the literal in `c06_falsifier_arena.py` and `configs/c06/…json` is updated with each correction."* | carries **`3673.9 s`**, **deletes the false *"tracks automatically"* clause** (`:1632`), replaces the two-file claim (`:1633`) with the single source and the assertion | |
| 55 | **`mint:118`** | *"the mint phase — **`85.6 %`** of §8's budget"* | **`68.3 %`** | **WRONG TODAY** |

### 3.4 Q8 — the design pointer

| # | site | current | correct | why |
|---|---|---|---|---|
| 20 | `config:5` | `"…DRAFT_V15E1.md"` | `"…DRAFT_V15E2.md"` | |
| 21 | **`config:6`** | `"design_sha256": "0b446b91…"` | V15E2's digest, **derived at freeze time** (§5) | **WRONG TODAY** |
| 22 | `config:7` | `design_status`: *"… + ERRATUM 1 landed"* | *"… + ERRATUM 1 + CODE-R1 §8 correction + ERRATUM 2 landed"* | |
| 23 | `arena:1249-1250` (**2 lines**) | `emit_halt` writes the declared pair | publishes **both** `sha256_declared` and `sha256_derived` (via `.get(…, "NOT_DERIVED")`), **plus the M-4 caveat string** | round-4 I-1, round-5 M-4 |
| 24 | `arena:1433-1434` (**2 lines**) | the verdict face writes the same pair | same repair, same string | |
| 25 | `sbatch:14` | *"…DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | |
| 26 | **`arena:4`** | *"…DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | |
| 27 | **`mint:4`** | *"…DRAFT_V15.md"* | *"…DRAFT_V15E2.md"* | |

### 3.5 Q6 — §8's equation, shares, risk row and the two units

| # | site | current | correct | why |
|---|---|---|---|---|
| 34 | `V15E1:1569-1571` (**3 lines**) | `2642.3 + … + **3.8** + … = 3670.0`; `× 1.25 = 4587.5` | **`2642.4`** + … + **`7.6`** + … = **`3673.9`**; `× 1.25 = **4592.4`** | round 6 recomputed every term |
| 35 | `V15E1:1576-1577` (**2 lines**) | the `85.6→68.3` / `9.3→27.6` sentence | **unchanged — CORRECT** | |
| 36 | `V15E1:1609-1610` (**2 lines**) | *"2× miss `4683.8 s`, 5× miss `7725.2 s`"* | **`4687.7 s = 78.1 min`**, **`7729.1 s = 128.8 min`** | the `1.68×` ratio on `:1610` is unchanged |
| 37 | `V15E1:1304` §7.7 `U11` | *"the arena's is **priced once** at §8 Phase 1g"* | *"the **two arena-class startups … are priced at §8 Phase 1g**"* | |
| 56 | `V15E1:1574` | *"`2929.9 − 273.7 + 1013.8 = 3670.0`"* | **unchanged — CORRECT, declared historical** | |
| 57 | `V15E1:1607-1608` (**2 lines**) | *"Mints are `68.3 %` … Phase 3 is now `27.6 %`"* | **unchanged — CORRECT** | |

### 3.6 Q2/Q9 — the ledger

| # | site | current | correct | why |
|---|---|---|---|---|
| 38 | `config:218` | `{"expected":"mints_executed","binding":true}` | `{"expected":"mints_executed + expected_sha_dev_opens","expected_sha_dev_opens":4,"derivation":"(dev-like files in the concatenated iterable gate_sha hashes = 2) × (GATE-SHA passes = 2)","binding":true}` | **the core repair** |
| 39 | `arena:432-441`, `:468-475` (**17 charged lines**) | the blocked-predicate docstring **and** the frozen `+ 0` predicate with its message | the two-term predicate, **derived and asserted** against `cfg["ledger"]`; the message's decomposition becomes the derivation; **and the docstring paragraph is rewritten** | round-5 I-1. `:432` says the predicate *"IS NOT ADJUSTED HERE"*, `:439-441` that it is *"implemented exactly as frozen"* and that adjusting it *"is not this lineage's call"* — all false at landing |
| 65 | `config:247`, `:249`; `V15E1:1753`, `:1755` (**4 lines**) | the two native `dev_seen_*.pt` digests | **unchanged — CORRECT** | **these four lines ARE the `2` in row 38's derivation**; listed so the derivation is checkable |
| 40 | `config:219` | binding `mints_executed` | a `by_construction` warranted string (corrected warrant, §4) | |
| 41 | `config:215`–`:217`, `:220`, `:221` (**5 lines**) | all as `{"expected":N,"binding":…}` | `:215`, `:217`, `:221` **unchanged — CORRECT**; `:216`, `:220` → `by_construction` strings | the whole partition shown |
| 42 | `config:211` | `blocked_on_erratum_2` narrative | an `erratum_2` block recording what landed | |
| 43 | `V15E1:1807-1813` (**7 lines**) | the counter rows — including `:1810`'s false *"reading `lab_dev` out of the banked mint `.npz`"* | the two-term form, the by-construction marks, the *"top-level processes only"* sentence, **and `:1810`'s mechanism corrected to *"it reads only `meta` out of the banked mint `.npz` (`:68`) and references no label array at all"*** | round 6 confirmed at source: `z["meta"]` is the only subscript of `z`, and `grep -i 'lab\|label'` over the file returns zero |
| 44 | `V15E1:1817-1823` (**7 lines**) | *"Why `mints_executed` and not `66`"* — the resume warrant | amended to carry the two-term form **and** the spawn/skip fact that makes an exact `74` safe where C09 refused to bind `39` | **round-6 I-2:** v6 labelled this *"`:1817-1821` (four lines)"* — a five-line range charging four — and `:1820` was simultaneously inside the extent and charged to a sweep-E declaration. The extent is the paragraph, `:1817-1823`; `:1820` and `:1822` are now charged here and removed from sweep E's declarations; `:1823` is a named extent-only line |
| 66 | **`sbatch:103`** | *"# It reads lab_dev out of the banked mint .npz…"* | *"# It reads only `meta` out of the banked mint .npz…"* | **WRONG TODAY**, the same false mechanism as `:1810` |
| 45 | `arena:449` | `self.ledger["processes_reporting"] = len(procs) + 1` | **unchanged — CORRECT** | round 6 re-verified via `atexit`-flushed per-pid ledgers: `len(procs) = 73`, `+1 = 74` |
| 46 | `arena:456-457` (**2 lines**) | the `test_path_opens != 0` assertion | **unchanged — CORRECT** | it **is** instrumented (`_guarded_open:97`) |
| 59 | `arena:458-462` (**5 lines**) | the two uninstrumented assertions | **RETAINED VERBATIM**; only the *publication* moves | so *"never binding"* is not read as licence to delete a tripwire |
| 47 | `arena:1419-1420` (**2 lines**) | reads `C06_MINTS_EXECUTED` with a fallback | **unchanged — CORRECT** | |
| 48 | `sbatch` — **an addition, no site** | neither export is set | export **`C06_MINTS_EXECUTED`** and **`C06_PROJECTED_SECONDS`** | round 6 verified the sbatch's only exports are the four thread caps, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`, `C09_LEDGER_DIR` |
| 26† | `arena:1418` | the dead `sum(1 for _ in [None])` placeholder | **delete** | |

### 3.7 Q3 — the `GATE-SHA` artifact count and scope

**The design document DOES increment `n`**: `21 + 16 = 37` becomes `22 + 16 = 38`. Round 6 confirmed
the arithmetic and that there is **no circularity** — §11 holds other artifacts' digests, the design
document's lives at `config:6`, and the config is not itself hashed.

| # | site | current | correct | why |
|---|---|---|---|---|
| 49 | `config:251-255` (**5 lines**) | `_gate_sha_count` fields and the note *"§11 declares 37 … 21 + 16 = 37"* | gains `design_document: 1`; `total_§11_digests` **`21 → 22`**; note → *"§11 declares **38** … **plus the design document, whose digest §11 names and `config:6` carries** … **22 + 16 = 38**"* | otherwise a false arithmetic statement in one of the five artifacts |
| 50 | `arena:563` | the concatenated iterable | the design document is appended — **the code site implementing §5** | a `.md` in `refine-logs/` leaves the dev-like count at 2 |
| 51 | `arena:585` | `self.reports["gate_sha_artifacts"] = n` | **unchanged — CORRECT, published value `37 → 38`** | the runtime symbol, distinct from the config key |
| 52 | `V15E1:1300` §7.7 `U7` | *"over **all 37 §11 artifacts**"* | *"over **all 38** … + **the design document**"*, cost **unchanged at `0.13 s`** | the unit definition row 3 re-prices |
| 53 | `V15E1:1786-1787` (**2 lines**) | *"`GATE-SHA`'s scope is stated in §6 as … the sixteen banked artifacts above."* | *"…**and the design document itself (ERRATUM 2 §5)**"* | §11's own scope sentence |
| 54 | `V15E1:2439-2440` (**2 lines**) | *"**No `.py` source moved** — all **37** §11 digests recompute."* | *"— all **38** … **as of this document's own freeze**; the 38th is this document, whose digest is by construction the one `config:6` carries"* | **round-6 I-2:** the sentence spans two lines and the appended clause lands on `:2440`, which v6 did not charge |

### 3.8 Q10 — the progress-coverage claims

`headspace_fidelity.py` has no progress handle and is sha-frozen; the 6 fidelity processes' lines are
written by the bash driver at `sbatch:128-129` with `-` in the elapsed and ratio columns.

| # | site | current | correct | why |
|---|---|---|---|---|
| 61 | `V15E1:1627-1629` (**3 lines**) | *"every python process appends through a handle opened `buffering=1`"* | *"every python process **that this lineage authors** appends … the six `headspace_fidelity.py` processes are sha-frozen and third-party, and the bash driver writes their line"* | round-4 I-3's own text |
| 60 | `mint:116` | *"§9 requires **EVERY** python process to append…"* | the same qualification | |
| 62 | `mint:209` | `--progress` help carrying the claim | *"…every python process this lineage authors appends to it (H-3)"* | |

---

## 4. The uniform counter criterion — carried, warrant corrected

> **A ledger quantity is published as a measured integer on the verdict face if and only if some code
> path in this job increments it. A quantity no code path increments is published as a
> by-construction narrative string carrying its warrant, in a separate block, never an integer and
> never binding.**

| counter | incremented by | disposition | warrant (source-verified) |
|---|---|---|---|
| `test_path_opens` | `_guarded_open:97` | **measured, binding `0`** | — |
| `dev_path_opens` | `_guarded_open:102` | **measured, binding** at the two-term formula | — |
| `banked_trainlog_opens` | `_guarded_open:106` | **measured, reported** | — |
| `test_label_materialisations` | **nothing** | by-construction string | `_guarded_open` **raises** on a test path |
| `dev_label_materialisations_outside_decisions` | **nothing** | by-construction string | `lab_dev` is written **twice** in the executed corpus — `headspace_mint.py:323` and **`c06_falsifier_mint.py:336`**, the latter a live `np.savez` into the banked `.npz` — and read by **no** path in the arena or `headspace_fidelity.py`: the arena has no `lab_dev` reference and no generic `.npz` key iteration, and `headspace_fidelity.py` reads only `z["meta"]` and references no label array at all |
| `dev_or_test_labels_into_decision_quantities` | **nothing** | by-construction string | same writes, read by no decision path |

Round 3 confirmed **no reachable tripwire is removed**, and the guard that would have to change is
digest-pinned in `frozen_sha256`. **Row 59 makes that concrete in the code.**

---

## 5. The design-digest mechanism and its residual

> **`GATE-SHA` gains one artifact: the design document itself** — hashed on disk, compared against
> `cfg["design_sha256"]`, **HALTing on mismatch** with both digests in the message.

Round 6 re-verified `main()`'s order (`gate_det1 → assert_guard_active → gate_sha → load_frozen`,
then `if args.gate_sha_only: return 0` at `:1278-1280`) and that the driver leg is process 1 of 74.
**The HALT is before the first mint, at zero compute cost.**

**The residual, stated.** The config is **not** in `frozen_sha256`, so the gate pins **config↔disk
parity only**: uncoordinated drift — the CODE-R1 failure — is caught; **coordinated drift is not**.
What anchors the digest outside the job is the freeze table in
`C06_FALSIFIER_IMPLEMENTATION_RECORD.md`, which **no code path reads** (round 5:
`grep -rn IMPLEMENTATION_RECORD scripts/ configs/` returns zero). §5 says *"removes the observed
subclass"*, not *"removes the class"*. Round 5 judged this scoping acceptable with no further anchor
required, since the sbatch is no more pinned than the config; that reasoning is recorded.

**The publication path is closed** (rows 23, 24): both `emit_halt` and the verdict face publish
`sha256_declared` **and** `sha256_derived`, plus

> `"design_sha256_note": "declared digest is not pinned inside the job; the external anchor is the
> freeze record in refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md"`

**Landing order:** V15E2 written first → its sha256 computed → every code/config edit →
`design_sha256` **last**.

**`expected_sha_dev_opens` is unaffected:** the design document is neither dev-like nor test-like
under `c09guard` (rounds 4, 5 and 6 each probed it, under both its V15E1 and V15E2 names), so the
dev-like count over the concatenated iterable stays **2** and the expectation stays **4 = 2 × 2**.

---

## 6. §8 re-priced, and the meta-check — re-scoped to what it proves

| row | before | after |
|---|---|---|
| Phase 1d `GATE-SHA` | `1 × U7 = 0.1 s` | **`2 × U7 = 0.2 s`** |
| Phase 1g arena-class startups | `1 × U11 = 3.8 s` | **`2 × U11 = 7.6 s`** |
| §8 residue term (row 34) | `2642.3` | **`2642.4`** |
| **total** | `3670.0 s` | **`3673.9 s`** |
| `× 1.25` | `4587.5 s` | **`4592.4 s`** |
| minutes / shares | `61.2 / 76.5`; `68.3 % / 27.6 %` | **unchanged** |
| `2×` / `5×` miss | `4683.8 / 7725.2 s` | **`4687.7 / 7729.1 s`** |

**`U7`'s object grows and its price does not — measured.** The added artifact is **188 061 bytes**;
sha256 over it, 7 repetitions, **median `0.000164 s`** (round 5 independently: `0.000148 s`). Against
`U7`'s `0.13 s` that is `0.11 %`–`0.13 %`, invisible at two decimals. **`U11`'s value does not move
either** — `3.094–3.717 s` measured, carried at `3.8 s`, `0.083 s` above the pooled maximum. What
moves is the **count**.

### The meta-check, and what it does and does not prove

```bash
grep -nE '\bU[0-9]+[a-d]?\b|\bU_(acc|mF1|tie)\b|[Pp]hase [0-9]+[a-zA-Z]?' $F | sort -t: -k1,1 -k2,2n
```

**102 hits, 30 charged by the sweeps, 72-hit residue** naming 12 phases (`1b, 1c, 1e, 1f, 2, 2b, 2z,
2D, 3, 4, 5, 7`) and **16 units, written out** (round-6 M-4 — v6's *"U3–U6"* range notation implied a
`U5` that does not exist): `U1, U2a, U2b, U2c, U2d, U3, U4, U5a, U5b, U6, U8, U9, U10, U_acc, U_mF1,
U_tie`.

**What it proves.** Of §8's 21 phase labels and 18 unit symbols, exactly two phases have a count or
cost this erratum moves — `1d` and `1g` — and zero units have a value that moves. The one growth
candidate, **Phase 1c's `67`** (*"66 mints + the arena process itself"*), is ruled out at source and
round 6 confirmed it: `load_frozen()` opens one file, `configs/c01/c01_a0_v2.json`, and **no `.pt`**;
`load_ro` is first reachable at `arena:1284`, after the early return at `:1278-1280`.

**What it does NOT prove — round-6 H-1, adopted verbatim.** This check sweeps **§8's priced units and
phases**. It cannot, and does not, prove that no family exists among the documents' **prose and
config statements of a moved quantity** — which is exactly where `V15E1:1814` and `config:42` live.
**That gap is what §1 closes**, by keying the completeness domain on the quantities and enumerating
their forms, rather than on one section's vocabulary.

---

## 7. The heartbeat denominator's single source

1. **`config:43` (`projected_seconds`) is the single source.**
2. **`sbatch` exports `C06_PROJECTED_SECONDS`** from that key, beside `C06_MINTS_EXECUTED` (row 48).
3. **`mint:112` and `arena:46`** become `float(os.environ.get("C06_PROJECTED_SECONDS", 3673.9))`; the
   literal survives **only** as a hand-run fallback, and V15E2 says so.
4. **The arena asserts all three agree** — environment, module constant, `cfg["projected_seconds"]` —
   and HALTs on mismatch, **placed in the same pre-`gate_sha_only` block as the design-digest gate**
   (before `arena:1278`), so it fires in **process 1 of 74**. Placed later it would fire only in
   process 74, after all 66 mints had published ratios against a drifted denominator.
5. **Row 19** names the single source and deletes §9's false *"tracks automatically"*.

All **68** ratio-computing processes agree after the repair; the 6 fidelity processes compute none.

---

## 8. Round 6's findings

**H-1 — the eleventh family. Adopted, and answered structurally.** Rows 67 and 68 are the two sites;
row 69 records the three sbatch banners as CORRECT and row 31 adds the fourth; **sweep K** is keyed on
the decomposition and the spaced identifier; **sweep L** checks the spelled-out form rather than
assuming it absent; §6's meta-check conclusion is re-scoped; and **§1 re-keys the whole completeness
domain from forms to quantities**, with the form enumeration printed as an attackable table.

**I-1 — row 64's uncharged line. Adopted:** row 64 is `:1368-1369`, in §2's multi-line list, and
sweep J's new `count is` term returns `:1369` so it is charged rather than extent-only.

**I-2 — the extent-only list's completeness. Adopted:** rows 5 → `:1644-1646` and 54 →
`:2439-2440`, both in the multi-line list; row 44's extent restated as the paragraph `:1817-1823`
with `:1820` and `:1822` moved from sweep E's declarations into the row, resolving the contradiction
round 6 identified; and §10 prints the verified extent-only list — **five lines, not two** — with the
siteless row counted separately.

**I-3 — *"three lines"* → four. Adopted:** row 5 now names `GATE-DET1` (`arena:1270`), `GUARD`
(`:1272`), `GATE-SHA` (`:586`) and `GATE-SHA-ONLY` (`:1279`), **citing round 2's H-1 item 4, which
enumerated them correctly**, rather than round 5's count. The load-bearing limb — the driver leg runs
`load_frozen()` first, so its startup is arena-class — is unaffected.

**M-1 — `V15E1:1582`'s present-tense attribution. Adopted:** row 70.

**M-2 — sweep J's borrowed reason for `:1583`. Adopted:** its charge now explains **both** patterns
that return it — sweep E on `once`, sweep J on the literal `3.8` — and states that the unit does not
move.

**M-3 — the extent-only heading. Adopted:** §10 counts lines and siteless rows separately.

**M-4 — the `U3–U6` range. Adopted:** §6 writes all sixteen units out.

**M-5 — the delta's CORRECT rows. Adopted:** §9 lists **all 71** rows, editing and CORRECT alike, so
a reader reconstructing the row set from the delta gets a partition.

**Carried from earlier rounds, unchanged:** the archaeology; the rejections of **(ii)** and **(iii)**;
**(iv)** declined with the TOCTOU reason; round 2's resume-stability fact; *argv ≠ launch argv*
recorded and not built on; *"the sbatch activates no environment"* referred to the code/resource
lineage; and every figure rounds 5 and 6 re-derived.

---

## 9. Implementation delta — all 71 rows, editing and CORRECT

**Convention (round-6 M-5):** rows marked **CORRECT** carry no edit and are listed in parentheses, so
this section is a partition of the row set rather than a subset of it.

| file | editing rows | CORRECT rows (no edit) |
|---|---|---|
| `c06_falsifier_arena.py` | 12, 14, 15, 23, 24, 26, 32, 33, 39, 50, 59, 26† | (18 is mint), 46, 47, 51, 58 |
| `configs/c06/c06_falsifier.json` | 10, 11, 16, 20, 21, 22, 38, 40, 41 (partial), 42, 49, **68** | 41's `:215`/`:217`/`:221`, 65's `:247`/`:249` |
| `c06_falsifier_cpu.sbatch` | 9, 25, 30, 31, 48, 66 | **69** |
| `c06_falsifier_mint.py` | 13, 17, 27, 55, 60, 62 | 18 |
| **V15E2** | 2, 3, 4, 5, 6, 7, 8, 19, 28, 29, 34, 36, 37, 43, 44, 52, 53, 54, 61, 63, 64, **67**, **70** | 1, 35, 45, 56, 57, 65's `:1753`/`:1755` |

**Total: 71 rows** — 55 editing, 16 CORRECT. Round 6 verified v6's delta covered every editing row
exactly once; this one extends that to the CORRECT rows so the partition is visible.

**Landing order:** V15E2 written → its sha256 computed → every code/config edit → `design_sha256`
set **last** → full dry-check battery re-run (GATE-SHA **38/38**, GATE-C01PARITY `max|diff| = 0.0` on
both datasets, the blindness grep) → implementation record updated.

---

## 10. APPENDIX — the twelve sweeps, printed

**Everything below is generated output.** Each sweep gives its exact command ending in a sort, its raw
hit list as `file:line`, the charge for every hit, and its subtraction.

**File list, bound once and used by every command below:**

```bash
F="refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md \
   scripts/analysis/c06_falsifier_arena.py \
   scripts/analysis/c06_falsifier_mint.py \
   configs/c06/c06_falsifier.json \
   scripts/slurm/c06_falsifier_cpu.sbatch"
```

### Sweep A — process inventory — **numeral**

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

### Sweep B — process inventory / pass count — **ordinal**

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

### Sweep C — heartbeat denominator — numeral, identifier and **alias**

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

### Sweep D — design pointer — identifier, literal and prose

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

### Sweep E — pass count — **prose idiom**

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
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1583` | declared — round-12's re-price of the UNIT from 3.2 to 3.8 s -- historical; matched by sweep E on `once` and by sweep J on the literal `3.8`. The UNIT does not move; only Phase 1g's count does (M-2) |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1645` | **row 5** |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1820` | **row 44** |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1840` | **row 29** |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2420` | declared — 'held both lines at once' -- ordinary English |
| 17 | `scripts/analysis/c06_falsifier_arena.py:55` | declared — 'opened once and never re-wrapped' -- the heartbeat handle |
| 18 | `scripts/analysis/c06_falsifier_arena.py:560` | **row 32** |
| 19 | `scripts/analysis/c06_falsifier_arena.py:1039` | declared — 'once per (arm, seed)' -- the mF1 precompute |
| 20 | `scripts/analysis/c06_falsifier_arena.py:1232` | **row 33** |
| 21 | `scripts/slurm/c06_falsifier_cpu.sbatch:17` | **row 30** |
| 22 | `scripts/slurm/c06_falsifier_cpu.sbatch:62` | **row 31** |

**Subtraction E.** my sweep: **22** hits; rows account for **10**; declared non-targets: **12**; `22 = 10 + 12`; **residue: none.**

### Sweep F — §8 total and second-order figures — numeral and **unit symbol**

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

### Sweep G — ledger quantities — **identifier**, decomposition and warrant prose

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

### Sweep I — progress-coverage claims — **prose idiom**

Run with `grep -i`.

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

### Sweep J — Phase 1d / 1g counts and cost — numeral, phase label, **inline count idiom** and **superseded value**

**Widened at round 6:** `\b3\.2\b` (the unit's superseded value) and `count = N|count is` (the count stated inline). Those two terms are what charge `V15E1:1369` and `:1582`, which round 6 found uncharged.

```bash
grep -nE '\b3\.8\b|\b7\.6\b|\b3\.2\b|[Pp]hase 1[dg]|count = [0-9]|count is' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 32 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:42` | declared — round-13's ruling that the arena class is bounded by the UNIT 3.8 s -- survives the row re-price |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:184` | declared — the section heading '### 3.2 The head, the folds, and the vote' -- a section number |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:288` | declared — 'The block count is forced by the head's architecture' -- a different count |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:849` | declared — 'GATE-LEDGER's process count is binding (SS12)' -- states that it binds, carries no value; true at 74 |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:935` | declared — 'adverse count is' -- an ERRATUM-1 float statistic |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:967` | declared — GATE-FOLD's gate row, matched on a SS3.2 cross-reference |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:970` | declared — 'two criteria at once' -- ordinary English |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1047` | declared — a `3.5x`-`7.6x` ratio in the row-renormalisation argument -- not a cost |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1199` | **row 2** |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1282` | declared — the section heading '### 7.6 GATE-C01PARITY' -- a section number, not a cost |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1304` | **row 37** |
| 12 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1336` | declared — 'split so the count is checkable' -- about how the U11 measurement is presented, no count value |
| 13 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1346` | **row 63** |
| 14 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1347` | declared — round-12's superseded unit value 3.2 s -- historical |
| 15 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1350` | declared — about the UNIT and Phase 1d's one-decimal rounding -- both survive the re-price |
| 16 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1352` | declared — 'ten of which exceed 3.2 s' -- round-13's measurement record, historical |
| 17 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1362` | declared — 'the count in SS8 Phase 1g turns on this clause' -- still true at count 2 |
| 18 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1368` | **row 64** |
| 19 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1369` | **row 64** |
| 20 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1428` | declared — '3.2x above' -- a ratio, not a cost |
| 21 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1550` | **row 4** |
| 22 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1569` | **row 34** |
| 23 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1581` | declared — provenance 2930.7->2933.9 -- historical |
| 24 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1582` | **row 70** |
| 25 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1583` | declared — round-12's re-price of the UNIT from 3.2 to 3.8 s -- historical; matched by sweep E on `once` and by sweep J on the literal `3.8`. The UNIT does not move; only Phase 1g's count does (M-2) |
| 26 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1646` | **row 5** |
| 27 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1822` | **row 44** |
| 28 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1993` | declared — 'SS8 Phase 1g's unit is undeterminable without it' -- about the UNIT, survives |
| 29 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2002` | declared — handoff instruction to re-measure U11 -- carries no figure |
| 30 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2144` | declared — SS14's drafting-audit narrative on class count vs line count |
| 31 | `scripts/analysis/c06_falsifier_arena.py:47` | declared — TOPK = 20, matched on the SS3.2 cross-reference in its comment |
| 32 | `scripts/analysis/c06_falsifier_arena.py:605` | declared — gate_fold docstring, matched on the SS3.2 cross-reference |

**Subtraction J.** my sweep: **32** hits; rows account for **10**; declared non-targets: **22**; `32 = 10 + 22`; **residue: none.**

### Sweep K — process inventory — **DECOMPOSITION WITHOUT THE TOTAL, and the spaced identifier**

**NEW at round 6 (H-1)** — the eleventh family. Keyed on the sum of parts and on `processes reporting` written with a **space**, neither of which sweep A's `\b7[234]\b` nor sweep G's underscored `processes_reporting` can reach.

```bash
grep -nE '66 ?(mints|\+)|6 ?(fidelity|\+)|1 ?arena|processes? reporting|process_order|mints ?->|fidelity ?->' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 16 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:42` | **row 68** |
| 2 | `configs/c06/c06_falsifier.json:255` | **row 49** |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:226` | declared — 'All 66 mints open the native dev_seen' -- true, unchanged |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:701` | declared — the p-value form (256 + 1) -- matched as '6 +', a pattern artifact |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1518` | declared — a CPU-minute drafting sum -- matched as '6 + 3', a pattern artifact |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1546` | declared — Phase 1c, '66 mints + the arena process itself', count 67 -- CORRECT and unchanged: SS6 rules out the driver leg at source (load_frozen opens no .pt; load_ro is first reachable after the early return) |
| 7 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1550` | **row 4** |
| 8 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1566` | declared — tie-cap product 7x6+5x6=72 |
| 9 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1814` | **row 67** |
| 10 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1839` | **row 6** |
| 11 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1923` | declared — GATE-FOLD under resume, 'all 66 mints including skipped ones' -- a mint count, unaffected by the process inventory |
| 12 | `scripts/analysis/c06_falsifier_arena.py:384` | declared — the p-value form (256 + 1) -- the same pattern artifact as V15E1:701 |
| 13 | `scripts/slurm/c06_falsifier_cpu.sbatch:16` | **row 9** |
| 14 | `scripts/slurm/c06_falsifier_cpu.sbatch:67` | **row 69** |
| 15 | `scripts/slurm/c06_falsifier_cpu.sbatch:101` | **row 69** |
| 16 | `scripts/slurm/c06_falsifier_cpu.sbatch:138` | **row 69** |

**Subtraction K.** my sweep: **16** hits; rows account for **9**; declared non-targets: **7**; `16 = 9 + 7`; **residue: none.**

### Sweep L — every moved quantity — **spelled-out numeral**

**NEW at round 6.** Run with `grep -i`. Added so the spelled-out form is *checked* rather than assumed absent.

```bash
grep -niE 'seventy|sixty|thirty-(seven|eight)|twenty-(one|two)|(one|two|three|four|five|six) (processes|passes|python)|(sixteen|twenty-one|twenty-two|thirty-seven|thirty-eight) (banked|artifacts|digests)' $F | sort -t: -k1,1 -k2,2n
```

**Raw hit list — 9 hits, every one accounted:**

| # | `file:line` | charge |
|---|---|---|
| 1 | `configs/c06/c06_falsifier.json:211` | **row 42** |
| 2 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:613` | declared — 'four cells of sixty' -- the 60 rho cells, not a process count |
| 3 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:966` | **row 28** |
| 4 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1405` | declared — 'cell of sixty' -- the same 60 cells |
| 5 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1463` | declared — round-8's GATE-SHA widening-cost narrative (5 ms for the sixteen) -- historical, and unchanged: the sixteen are existence-checked, not re-hashed |
| 6 | `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1787` | **row 53** |
| 7 | `scripts/analysis/c06_falsifier_arena.py:436` | **row 39** |
| 8 | `scripts/analysis/c06_falsifier_arena.py:473` | **row 39** |
| 9 | `scripts/analysis/c06_falsifier_arena.py:559` | **row 32** |

**Subtraction L.** my sweep: **9** hits; rows account for **6**; declared non-targets: **3**; `9 = 6 + 3`; **residue: none.**

### Global partition — the twelve sweeps together

| quantity | value |
|---|---|
| hit-instances across the twelve sweeps | **305** |
| distinct sites (deduplicated) | **267** |
| excess hit-instances (`305 − 267`) | **38** |
| distinct sites returned by more than one sweep | **32** |
| distinct sites charged to a row | **137** |
| distinct sites charged to a declaration | **130** |
| sites returned by a sweep and charged to nothing (`UNCHARGED`) | **0** |

`267 = 137 + 130`. **Residue: none, in either direction.**

### Extent-only lines — **5 lines**, plus **1 siteless row**, all named (round-6 M-3)

A row's *extent* is the set of lines its edit touches; its *charged sites* are the lines a
sweep returns. Extent ⊇ charged sites always. Round 6 found v6's list of these incomplete;
this is the verified list, and its two categories are counted separately.

**Extent-only lines (5):**

* **`refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1633`** — row 19.
* **`refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1644`** — row 5.
* **`refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:1823`** — row 44.
* **`refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md:2440`** — row 54.
* **`scripts/analysis/c06_falsifier_arena.py:470`** — row 39.

**Siteless rows (1):** **row 48** adds the two sbatch exports; a grep over the current files
cannot return a line that does not exist yet.

**Rows: 71 defined** (`1`–`70` plus `26†`), of which **1** is siteless.


---

## 11. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no arena run; no `--gate-sha-only` leg run; no GPU, no
SLURM job, no commit, no `TARGET_STATE.json` edit; `artifacts/c06_falsifier/` never created
(verified absent).

**Compute used:** file, review and source reads; the twelve `grep` sweeps of §10 plus wider variants
of each and the §6 meta-check sweep; `sha256sum` over the artifacts and proposals; **one login-node
timing of `hashlib.sha256` over the 188 061-byte design document, 7 repetitions, reported in §6**;
static reads of `headspace_fidelity.py`, `c06_falsifier_arena.py`, `c06_falsifier_mint.py`, the
config and the sbatch; and arithmetic. The charge table, every hit list and every count in §10 are
produced by a script over the sweep output, not transcribed.

**Nothing is edited.** All five artifacts carry their post-CODE-R1 hashes, re-verified before and
after:

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` |

`…_PROPOSAL.md` through `…_V6.md` stay on disk byte-unmodified.

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`.
**The battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under
v1–v6 as specified.**
