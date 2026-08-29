# C06 `$0` falsifier — **ERRATUM 2, INDEPENDENT REVIEW — ROUND 4**

*Target:* `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V4.md`
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`
(`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` — **verified, matches** v4's citation)
*Prior adjudications:* round 1 (0C/3H/3I/3M) → v2; round 2 (0C/2H/3I/3M) → v3;
round 3 (0C/3H/3I/3M) → v4.
*Reviewer:* fresh and independent; no part in the fifteen design rounds, the implementation,
Erratum 1, code-review round 1, or any of the three prior erratum-2 rounds. Judged from documents,
repository and execution only.

---

## VERDICT

> **REVISE — 0 Critical, 2 High, 3 Important, 4 Minor.**

**Round 3's nine findings are 9-for-9 discharged, several beyond the ask.** H-3's ordinal
contradiction is gone and cannot recur — row 2 carries no ordinal, and row 1's `72` plus row 2's
*"remaining two"* sums to `74` in agreement with rows 6, 10, 11 and 12. The design pointer went
further than round 3 asked: instead of hand-updating `config:6`, §3 makes the digest a gated
quantity. Both live-wrong numbers have rows. The `PROJECTED_SECONDS` site set went from two to
four. `V15E1:1304`, `arena:1232`, §8's equation and the risk row are rows 37, 33, 34 and 36. §12's
three prose-only sites were promoted from *"covered by §7"* to rows 43 and 44. And v4's own sweep
found two sites round 3 missed — `arena:4` and `mint:4`, both reading *"Frozen design:
…DRAFT_V15.md"* — which I confirm at source.

**I re-derived the entire carried core and every figure, and all of it holds.** My own run over the
config's digest tables under an imported `c09guard`: the concatenated iterable is 21 files, exactly
2 dev-like, 0 test-like, `frozen_sha256` alone contributing 0, so `expected_sha_dev_opens = 2 × 2 =
4` on both factors. `sha256_of` (`arena:81-86`) opens each path exactly once through the wrapped
builtin, which is what makes the second factor a pass count. `74 = 1 + 66 + 6 + 1` reads off the
sbatch directly. `_guarded_open` increments exactly three counters (`:97`, `:102`, `:106`) and
`c09guard.py` **is** in `frozen_sha256`. `_flush` records `sys.argv` (`:129`), so the pass audit is
computable. §4's re-price is exact to the last decimal, including the `3673.8`-vs-`3673.9` point
that makes row 34 necessary. Row 5 is measured-true: the driver leg instantiates `Heartbeat` at
`arena:1266` and emits three lines before `--gate-sha-only` returns, and it runs `load_frozen()`
first, so §4's conservative pricing is right.

**What blocks GO is that v4 promises the one thing round 3 credited it for — a stated, reproducible
subtraction — across seven families, and delivers it for two.** Sweep A subtracts exactly; I
reproduce 18 hits, 11 in rows, 7 declared, nothing left over. Sweep B subtracts exactly. **Sweeps
C, D, E and G do not close under their own declared patterns**, and **sweep F declares no pattern at
all**. In each failing case the stated identity is self-consistent only because two or three errors
compensate: sweep D reads *"12 hits − 8 in rows = 4 declared"* where the pattern returns **13**,
the rows cover **10** line-hits, the residue is **3**, and exactly 3 are named. I traced every
unaccounted hit and **no site is lost at landing** — which is why this is High and not Critical —
but §1's standard is *"completeness is verified by subtraction, not by trust,"* and four of seven
subtractions cannot be reproduced by a reader doing exactly what §1 invites.

**And §3's new mechanism moves a quantity that no sweep covers and no row records.** §3 states that
the `GATE-SHA` artifact count goes `37 → 38` and dismisses it as *"a reported figure … not a binding
one."* §1's own standard forecloses that defence — row 1 is listed **CORRECT** on precisely the
ground that completeness is verified by subtraction. One of the unrecorded sites is `V15E1:1300`,
the definition of `U7` — **the unit row 3 re-prices as `2 × U7`**. The erratum prices a unit twice
while enlarging the object that unit is defined over, in neither place. That is round-3 I-1's defect
(§7.7's `U11` row versus Phase 1g) one row up in the same table, and v4 adopted the repair for `U11`
as row 37.

---

## SWEEP-DIFF RESULTS PER QUANTITY FAMILY

Every sweep below was run by me over all five artifacts with the pattern **as v4 declares it**, and
then again wider. Hit counts are line-hits, which is the unit v4 itself uses when it notes that
*"row 12 covers two lines."*

### Sweep A — `\b7[234]\b` — **REPRODUCES EXACTLY**

18 hits: `V15E1:106, 310, 1197, 1550, 1566, 1658, 1691, 1839, 1904, 2017`; `config:41, 65, 222,
264`; `mint:117`; `sbatch:16`; `arena:465, 466`. Rows 1, 4, 6, 7, 8, 9, 10, 11, 12 (×2), 13 account
for 11. The declared residue is exactly the remaining 7: the line citation
`generate_…:73-89` at `V15E1:106`, `V15E1:1658`, `config:264`; the rotation angle `72.7` at
`V15E1:310`, `V15E1:1691`, `config:65`; the tie-cap product at `V15E1:1566`. **Nothing left over in
either direction.** This leg is as good as v4 claims.

### Sweep B — `[0-9](st|nd|rd|th)` — **REPRODUCES EXACTLY**

3 hits: `V15E1:1198` (row 2), `arena:286` (*"the 21st"*), `V15E1:458` (*"95th percentile"*). Residue
2, both correctly declared non-process. Round-3 M-2 properly adopted.

### Sweep C — projection literals — **DOES NOT CLOSE**

Declared pattern `3670\.0|4587\.5|2929\.9|PROJECTED_SECONDS|projected_seconds` returns **11 hits**,
not the 12 claimed:

`config:43`, `config:44`, `mint:112`, `mint:127`, `arena:29`, `arena:46`, `arena:57`,
`V15E1:1570`, `V15E1:1571`, `V15E1:1574`, `V15E1:1631`.

Rows 14–19 cover **7** of them, not 6 — **row 16 is `config:43-44`, two lines**, and v4 does not
declare the doubling here although it declared exactly this for row 12 in sweep A. True residue is
**4**: `arena:57`, `V15E1:1570`, `:1571`, `:1574`.

v4 declares **6**, and two of the six — **`V15E1:1584` and `V15E1:2080`** — are **not hits of the
declared pattern at all**. They carry `2933.9`/`2934.5`, which match none of the five alternatives.
Declaring as residue two lines the sweep never returned, while under-counting the hits by one and
the row coverage by one, is what makes `12 − 6 = 6` look sound.

*Wider pattern* (`3670|4587|2929|2933|2934|3673|4592|projected`) additionally returns `arena:61`,
`arena:68` (the arena's `self.projected` assignment and divide, correctly downstream of `arena:57`)
and `V15E1:1581` (historical provenance). None needs a row; all three are consequences of sites that
have one.

### Sweep D — design pointer — **DOES NOT CLOSE**

Declared pattern `design_document|design_sha256|design_status|V15E1|DRAFT_V15|Frozen design` returns
**13 hits**, not 12:

`config:5, 6, 7`; `mint:4`; `sbatch:14`; `arena:4, 30, 1249, 1250, 1433, 1434`; `V15E1:3, 2258`.

Rows 20–27 are 8 rows but cover **10 line-hits** — **rows 23 and 24 are `arena:1249-1250` and
`arena:1433-1434`, two lines each**, again undeclared. True residue is **3**: `arena:30`, `V15E1:3`,
`V15E1:2258`. v4 names exactly those 3 while stating the count as **4**.

Substantively this sweep is v4's best original work: `arena:4` and `mint:4` are real, are outside
round 3's ten-site list, and I confirm both read `DRAFT_V15.md` — two revisions stale. The
arithmetic around them is simply wrong.

### Sweep E — pass-count idiom — **DOES NOT CLOSE, AND THREE HITS ARE NAMED NOWHERE**

Declared pattern `ONCE|once|one span|twice|single pass` returns **22 hits**, not 21:

`sbatch:17, 62`; `arena:55, 560, 1039, 1232`; `V15E1:128, 194, 274, 629, 966, 970, 995, 1229, 1295,
1304, 1547, 1583, 1645, 1820, 1840, 2420`.

Rows 28–33 cover 6. That leaves **16** non-row hits. v4 claims 15 and **names 13** — and its own
prose says *"twelve in V15E1"* over a list of **eleven**, one of which (`:1304`) it simultaneously
identifies as row 37, i.e. not residue at all. Ten genuinely-irrelevant V15E1 sites are named.

**Three hits are accounted for by no subtraction in the document:**

* **`V15E1:1547`** — row 3's own site (*"`GATE-SHA`, once in the driver"*).
* **`V15E1:1645`** — row 5's own site (*"is the one span that precedes any python-side line"*).
* **`V15E1:995`** — a substring artifact: the unanchored `once` matches inside
  *"c**once**ntration"*.

Rows 3 and 5 are printed under **Sweep A's** table but are not `\b7[234]\b` hits and are explicitly
excluded from sweep A's *"11 in rows"*; they are sweep-E hits but are not in sweep E's rows (28–33)
or its residue. **They fall between the two accountings.** Nothing is lost — both are rows, so both
land — but this is the exact mechanism by which the subtractions fail, and it is the finding.

### Sweep F — §8's equation, shares and risk row — **NO PATTERN DECLARED**

Sweep F is the only one of the seven given in prose rather than as a regex. Its *"8 hits · rows
34–37 · residue 4"* is therefore unverifiable by construction: a reader cannot reproduce a hit list
without knowing what was swept. Its *"4 in rows"* also counts rows where sweeps A, C and D count
line-hits, and rows 34, 35 and 36 each span multiple lines.

The **content** of sweep F is correct, and I verified every figure independently
(`2642.4 + 1.0 + 0.7 + 0.1 + 1.3 + 7.6 + 7.0 + 1013.8 = 3673.9`; the same sum with `2642.3`
gives `3673.8`, which is why row 34 must move it; `× 1.25 = 4592.375 → 4592.4`; `61.23 → 61.2` and
`76.54 → 76.5` unchanged; `2508.3/3673.9 = 68.27 % → 68.3`; `1013.8/3673.9 = 27.59 % → 27.6`;
`4687.7 s = 78.13 → 78.1 min`; `7729.1 s = 128.82 → 128.8 min`). Row 35's **CORRECT** disposition is
right. Only the claim of subtractability fails.

### Sweep G — ledger quantities — **DOES NOT CLOSE; "residue 0" IS FALSE**

Declared pattern `dev_path_opens|mints_executed|processes_reporting|expected_sha_dev_opens` returns
**26 hits**, not 14:

`mint:217, 346`; `config:211, 218, 219, 222`; `arena:424, 433, 438, 449, 452, 465, 466, 467, 468,
469, 471, 475, 1418, 1419, 1421`; `V15E1:1810, 1811, 1817, 1819, 1821`.

Rows 38–44 plus row 26† cover roughly 15 line-hits, leaving about 11 unaccounted — among them
`config:222` and `arena:465-467`, which are sweep A's rows 11 and 12 cross-listed without saying so,
and `arena:424`, `:449`, `:452`, `:1419`, `:1421`, `mint:217`, `mint:346`.

**I checked every one of these for substance and none is a missed repair.** In particular
`arena:449` (`self.ledger["processes_reporting"] = len(procs) + 1`) is the line that *computes* the
quantity row 12 re-compares. It needs no edit: the 6 fidelity processes and the `--gate-sha-only`
driver leg all write `c09guard` ledger files, so `len(procs)` is 73 and the `+ 1` for the
not-yet-flushed arena yields 74 after the repair. The claim *"residue 0"* is nonetheless false by a
wide margin.

### My own additional sweep — **GATE-SHA artifact count** — see H-2

`\b37\b|\b38\b|gate_sha_artifacts|_gate_sha_count` returns sites in **no** v4 sweep: `config:251`,
`config:255`, `V15E1:1300`, `V15E1:51`, `V15E1:2065`, `V15E1:2439`, `arena:585`.

---

## THE DESIGN-POINTER MECHANISM — ATTACKED

**What it pins.** Measured: `configs/c06/c06_falsifier.json` is **not** in `frozen_sha256`
(verified by enumeration; round 3 found the same). So `cfg["design_sha256"]` is a value nothing in
the job pins. §3's gate compares the on-disk design against that unpinned declaration:

* **Uncoordinated drift** — the design edited, the config not — is caught. This is exactly the
  CODE-R1 failure that produced the live-wrong `config:6`, so the mechanism closes the observed
  failure mode. ✔
* **Coordinated drift** — a post-freeze edit to the design *with* a matching config update — passes
  startup parity silently. Nothing outside the config anchors the digest: the freeze record that
  would do so (`C06_FALSIFIER_IMPLEMENTATION_RECORD.md:328`, which is where I verified all five
  post-CODE-R1 hashes) is read by no code path.

So §3's *"The mechanism removes the class"* is an overstatement. It removes a subclass — the
important one — and leaves a residual that §3 does not state. In a lineage whose entire method is
stating the residue, that is the objection. This is I-1.

**Is the HALT before any compute is spent?** **Yes, verified.** `main()` runs
`gate_det1 → assert_guard_active → gate_sha → load_frozen`, and only then
`if args.gate_sha_only: return 0`. The `--gate-sha-only` leg is process 1 of 74
(`sbatch:63-64`, before the mint loop at `:67`). A mismatch therefore HALTs before the first mint at
zero compute cost, as §3 claims. ✔

**Does `emit_halt` carry the DERIVED digest?** **No — and v4 marks the site CORRECT.** `emit_halt`
(`arena:1249-1250`) writes `cfg.get("design_sha256")` and the verdict face (`:1433-1434`) writes
`cfg["design_sha256"]`. v4 rows 23 and 24 mark both *"unchanged — **CORRECT**"*. On the one occasion
the new gate fires, declared ≠ derived **by construction**, and the HALT artifact's machine-readable
`design.sha256` field publishes the declared — i.e. wrong — value. The prose message carries both
digests, so it is recoverable; but rows 23 and 24's stated reason for existing is that these are the
sites putting a stale digest *"on the face of every verdict and HALT artifact."* The erratum
identifies the publication path as the reason row 21 is High, then leaves that path emitting the
unverified number. Also worth stating in §3: `--out` defaults to
`artifacts/c06_falsifier/C06_VERDICT.json` and the sbatch invokes the driver leg without `--out`, so
a design-drift HALT in process 1 writes to the canonical verdict path — correct behaviour, newly
reachable at startup.

**§3's two side-claims both check out.** The design document is a `.md` in `refine-logs/`, and I
confirm `c09guard.is_dev_like` and `is_test_like` both return `False` for it, so the dev-like count
stays at 2 and `expected_sha_dev_opens` at 4. ✔ And the artifact count is genuinely 37 today
(`13 + 8 + 6 + 10`, computed from the config), so `37 → 38` is right — which is precisely the
problem in H-2.

---

## THE HEARTBEAT DENOMINATOR — ALL 74 PATHS TRACED

**The premise of this obligation does not hold: v4 prescribes no single-source read.** Row 17
replaces `mint:112`'s literal with another literal, `3673.9`. After landing there are **three**
independent hand-carried copies — `arena:46`, `config:43-44`, `mint:112` — plus a docstring
(`arena:29-30`) and §9's sentence (`V15E1:1631-1633`).

Each cited path, verified at source:

| leg | count | code path | denominator after v4 |
|---|---|---|---|
| `--gate-sha-only` driver | 1 | `Heartbeat(args.progress)` `arena:1266` → default `projected=PROJECTED_SECONDS` `arena:57` → `arena:46` | `3673.9` ✔ |
| mints | 66 | `heartbeat()` `mint:114` → `elapsed / PROJECTED_SECONDS` `mint:127` → `mint:112` | `3673.9` ✔ |
| fidelity | 6 | **no ratio computed** — `headspace_fidelity.py` has no progress handle at all; the *bash* driver writes the line with literal `-` in the elapsed and ratio columns (`sbatch:128-129`) | n/a |
| arena | 1 | same `Heartbeat` object | `3673.9` ✔ |

So 68 of the 74 processes compute a ratio and, after the repair, **all 68 agree**. The repair is
correct and the `740.1 s` disagreement is genuinely eliminated. The objection is doctrinal and is
I-2: §3 argues that for the design digest *"hand-updating it would only reset the clock"* and
installs a derive-and-HALT mechanism, then applies the opposite remedy to the one quantity in this
erratum where hand-carrying has **already measurably drifted**, across 66 of 74 processes. The
sbatch is gaining a `C06_MINTS_EXECUTED` export in this very erratum, so the pattern for a real
single source already exists at the same cost.

---

## ROUND-3 DISPOSITION AUDIT — LIMB LEVEL

| finding | limb | disposition in v4 | strength |
|---|---|---|---|
| **H-1** | rows for `config:5`, `:6`, `:7`, `arena:29-30`, `sbatch:14` | rows 20, 21, 22, 15, 25 | **full** |
| | `design_sha256` computed after V15E2 is written; landing order stated | §3 + §6's landing order (`design_sha256` set **last**) | **full** |
| | *(beyond the ask)* | §3's derive-and-HALT gate; rows 26, 27 (`arena:4`, `mint:4`) found by v4's own sweep — **both confirmed at source** | **over-delivers**; residual → I-1, and the gate creates H-2 |
| **H-2** | rows for `arena:29`, `mint:112`, `V15E1:1631-1633` | rows 15, 17, 19 | **full** |
| | §4's *"both"* becomes the full set | §4's *"FOUR sites, not two"* | **full** |
| **H-3** | drop the ordinal; make §7.2 ordering-independent | row 2, no ordinal; `72` (row 1) `+ 2` = `74`, agreeing with rows 6, 10, 11, 12 | **full — contradiction gone and structurally unable to recur** |
| **I-1** | row amending `V15E1:1304`'s `U11` accounting | row 37 | **full** |
| **I-2** | row for `arena:1232`, either disposition | row 33, amended | **full** |
| **I-3** | rows for `:1569-1571` naming `2642.3 → 2642.4`, and `:1609-1610` | rows 34, 36 | **full** |
| | *(round 3's recorded-not-charged §12 sites)* | promoted to rows 43, 44 | **over-delivers** |
| **M-1** | correct the C09 form claim | §2's closing paragraph | **full** |
| **M-2** | disclose the ordinal hole | sweep B, stated | **full** — though the same class of pattern defect returns at sweep E (my M-2) |
| **M-3** | *"derives, asserts against"* into §7's arena row | row 39 + §6's arena row | **full** |
| **obl. 8** | carry the discharged core unchanged | §0, §2, §3, §6 | **full — I re-derived all of it** |

**Round 3 is 9-for-9. Every finding below is new.**

---

## FINDINGS

### H-1. Four of the seven subtractions do not close under their own declared patterns, and a fifth declares no pattern. §1's promise — the one thing round 3 credited v3 for — is delivered for two families out of seven.

*Attaches to:* §1's heading and its per-sweep subtraction lines; sweeps C, D, E, F, G.

| sweep | v4's stated identity | measured |
|---|---|---|
| A | 18 − 11 = 7 | **exact** ✔ |
| B | 3 − 1 = 2 | **exact** ✔ |
| C | 12 − 6 = 6 | hits **11**; rows cover **7**; residue **4**; 2 declared items are not hits |
| D | 12 − 8 = 4 | hits **13**; rows cover **10**; residue **3**; 3 named |
| E | 21 − 6 = 15 | hits **22**; rows cover 6; residue **16**; **13** named; **3 named nowhere** |
| F | 8 − 4 = 4 | **no pattern declared — unverifiable** |
| G | 14 − 14 = 0 | hits **26**; ~11 unaccounted; *"residue 0"* false |

**The mechanism is that the row-to-sweep assignment is not a partition.** Rows 2, 3 and 5 sit under
sweep A's table but are not sweep-A hits; sweep A's *"11 in rows"* correctly excludes them; row 2 is
then credited to sweep B, and **rows 3 and 5 are credited to nothing**. Row 37 is printed under
sweep F and simultaneously listed inside sweep E's residue. Sweep G silently re-uses sweep A's rows
11 and 12. And multi-line rows are declared in sweep A (row 12) and then not in sweep C (row 16) or
sweep D (rows 23, 24), which is where two of the three arithmetic errors come from.

**Why this is High and not Critical.** I traced every unaccounted hit in all five artifacts. **No
site is lost at landing.** The two substantive ones — `V15E1:1547` and `V15E1:1645` — are rows 3 and
5 and will be edited; the rest are either already rows, code the rows' edits necessarily touch, or
irrelevant idiom. The delta is not wrong; the completeness argument for it is not reproducible.

**Why it is High and not Minor.** Round 3 blocked v3 for running one sweep out of seven and
identified the fix as *"seven sweeps, each with its hit list, its rows, its declared residue, and
its subtraction stated."* v4 adopts the form for all seven and the rigour for two. A reader
verifying sweep D with v4's own pattern gets 13 where the document says 12 and cannot tell from the
document whether the extra hit is a bookkeeping slip or a site the erratum missed — which is the
precise epistemic state §1 exists to eliminate.

**Repair.** Assign every hit to exactly one accounting home; declare multi-line rows wherever they
occur (rows 16, 23, 24, 34, 36 at minimum); give sweep F a regex; re-run all seven and restate the
counts from the output rather than from the table.

### H-2. §3's own mechanism moves the `GATE-SHA` artifact count `37 → 38`. That quantity has no sweep and no rows — including at `V15E1:1300`, the definition of the unit row 3 re-prices.

*Attaches to:* §3's closing paragraph; §1 (all seven sweeps); row 3; `config:251-256`;
`V15E1:1300`, `:51`, `:2065`, `:2439`; `arena:585`.

§3 states the move and disposes of it in half a sentence: *"which is a reported figure
(`_gate_sha_count`), not a binding one."* My sweep for the count family returns sites in none of
v4's seven patterns:

* **`config:251-256`** — a structured block with `total_§11_digests: 21`,
  `banked_unhashed_artifacts: 16`, and a note reading *"§11 declares 37 = 7 + 6 + 8 imported/read/
  cached digests plus the 16 banked artifacts … 21 + 16 = 37."* Adding the design document makes
  that note a false arithmetic statement in one of the five artifacts.
* **`V15E1:1300`** — §7.7's `U7` row: *"`GATE-SHA` over **all 37 §11 artifacts** (8 caches + 13
  modules/configs + 16 banked; round-8 H-1 — v8 said '8 caches + 6 modules' while §11 listed 37)."*
  **This is the unit definition, and row 3 re-prices §8 Phase 1d as `2 × U7 = 0.2 s`.** The erratum
  prices a unit twice while enlarging the object that unit is defined over, and says so in neither
  place.
* `V15E1:51`, `:2065` (*"37/37 digests"*), `:2439` (*"all 37"*), and the runtime symbol
  `arena:585` (`self.reports["gate_sha_artifacts"] = n`).

**§1's own standard forecloses the "reported, not binding" defence.** Row 1 is listed **CORRECT**
explicitly *"because completeness is verified by subtraction, not by trust,"* and rows 18, 23, 24
and 35 are all listed **CORRECT**. Binding-ness has never been the criterion for getting a row;
being a site of a moved quantity has.

**And either branch is a defect.** If the design document counts toward `n`, the `37` sites are
wrong after landing. If it does not, §3's own *"`37 → 38`"* sentence is wrong. v4 does not say which
implementation it intends.

**Why this is High.** It is the same category round 3 rated High twice — a quantity the erratum
moves, with sites in the five artifacts, absent from every sweep and every row — with the
aggravation that this quantity is moved *by the erratum's own new mechanism*, and that one of its
sites is the unit definition another row re-prices. That last part is structurally identical to
round-3 I-1, whose repair v4 adopted as row 37; the same defect one row up went unexamined because
no sweep looked.

**Repair.** An eighth sweep (`\b37\b|\b38\b|gate_sha_artifacts|_gate_sha_count`) with its hit list
and subtraction; rows for `config:251-256` (fields **and** note), `V15E1:1300`, and a stated
disposition for `V15E1:51`, `:2065`, `:2439`; and one sentence in §3 saying whether the design
document increments `n`.

### I-1. §3's gate narrows the class rather than closing it, and the residual is unstated; and `emit_halt` — marked **CORRECT** by row 23 — publishes the declared digest, so the one artifact the mechanism exists to produce still carries the unverified number.

*Attaches to:* §3; rows 21, 23, 24; `arena:1249-1250`, `:1433-1434`.

Measured: the config is **not** in `frozen_sha256`, so `cfg["design_sha256"]` is pinned by nothing.
The gate establishes config↔disk parity only. Uncoordinated drift — the CODE-R1 failure — is caught
before any compute, in process 1, at zero cost; I verified the ordering in `main()` and it is as §3
claims. **Coordinated drift passes.** The freeze record that would anchor the digest independently
is outside the config and is read by no code path. §3 claims the mechanism *"removes the class"*; it
removes the observed subclass.

Separately, on the one occasion the gate fires, `emit_halt` writes `cfg.get("design_sha256")` — the
declared value, which by construction differs from the derived one — into the HALT artifact's
`design.sha256` field. Row 23 marks this **CORRECT — unchanged**, giving as its reason that this is
*why* row 21 is High. The reason is right and the disposition does not follow from it.

**Repair.** State the residual (the gate pins config↔disk, not disk↔freeze-record) and name what
anchors the declared digest outside the job. Have `emit_halt` and the verdict face publish the
**derived** digest, or both, so no artifact can ever carry an unverified design hash.

### I-2. The `PROJECTED_SECONDS` repair installs a third hand-carried literal — the opposite of the remedy §3 adopts three sections earlier, on the one quantity in this erratum where hand-carrying has measurably drifted.

*Attaches to:* row 17; §3's *"hand-updating it would only reset the clock"*; `mint:112`,
`arena:46`, `config:43-44`.

After landing, `3673.9` exists as three independent literals plus a docstring plus a prose sentence,
kept in step by nothing. That is the configuration that produced the `740.1 s` divergence across 66
of 74 processes which this erratum is fixing. §9's *"the denominator is pinned to §8 by name, so it
tracks automatically"* was false for the same reason and is being corrected at row 19.

The repair itself is correct — I traced all four legs and all 68 ratio-computing processes agree
after landing (the 6 fidelity processes compute no ratio; the bash driver writes `-`). The objection
is that §3 diagnoses this exact class for the design hash, calls hand-updating insufficient, and
builds a structural fix, then declines the same reasoning here. `mint.py` does not read the config
(no `--config`, no `json.load` of it), so a single source needs one export — and the sbatch is
already gaining a `C06_MINTS_EXECUTED` export in this erratum, so the cost is the same as
synchronising three literals.

**Repair.** Either source the denominator from `configs/c06/c06_falsifier.json` in both processes
(e.g. a `C06_PROJECTED_SECONDS` export from the sbatch, mirroring `C06_MINTS_EXECUTED`), or state
explicitly why three synchronised literals are acceptable here and not for the design digest.

### I-3. Row 7's replacement asserts heartbeat coverage over the enlarged inventory, and that coverage claim is false for 6 of the 74 — as its unlisted neighbour `V15E1:1628-1629` states in the strongest form.

*Attaches to:* row 7; `V15E1:1904`, `:1628-1629`; `headspace_fidelity.py`; `sbatch:128-129`.

Measured: `scripts/analysis/headspace_fidelity.py` contains no progress handle, no heartbeat and no
append — it opens at `:42`, writes JSON at `:113`, prints at `:115`, nothing else. It is sha-frozen
(`config:229`) and the sbatch states it is *"run UNMODIFIED"*, so this is not fixable in code. The
6 fidelity processes' progress lines are written by the **bash** driver via `>>` (`sbatch:128-129`),
with a literal `-` in both the elapsed and ratio columns.

`V15E1:1628-1629` reads: *"every python process appends through a handle opened `buffering=1`."*
That is false for 6 of the 74. Row 7 moves §13.1 item 12's companion phrase from *"across all 73
processes"* to *"across all 74"* — the count is right, the coverage assertion it sits inside is not,
and it is being restated over a larger set. `:1628-1629` is in §9, a section this erratum edits at
rows 5 and 19, and is in none of the seven sweeps.

This is pre-existing rather than introduced, which is why it is Important. But it is a claim about
the same process inventory the erratum re-states, two lines from a site the erratum edits, and it is
what obligation 1's *"verify each row's correct text introduces no new contradiction with unlisted
neighbours"* is for.

**Repair.** A row for `V15E1:1628-1629` (and a matching qualification in row 7) reading, e.g.,
*"every python process **that this lineage authors** appends through a handle opened `buffering=1`;
the six `headspace_fidelity.py` processes are sha-frozen and third-party, and the bash driver writes
their line."*

### M-1. Sweep E's residue says *"twelve in V15E1"* over a list of eleven, one of which (`:1304`) it simultaneously identifies as row 37 — so ten genuinely-irrelevant sites are named against a claim of twelve.

### M-2. Sweep E's `once` is unanchored and matches inside words: `V15E1:995` is *"c**once**ntration"*. Round-3 M-2 asked for pattern holes to be disclosed; this is the same class, introduced in the sweep added to close it. Anchor it (`\bonce\b`) or declare substring hits.

### M-3. Sweep C's residue says `V15E1:1574` is *"inside row 34's replacement block"*, but row 34's stated site is `V15E1:1569-1571` and `:1574` is three lines outside it. The line (`2929.9 − 273.7 + 1013.8 = 3670.0`) is historical provenance and is correct to leave alone — but an implementer editing 1569–1571 will not touch it, so the declaration mis-describes what covers it. Either extend row 34's stated range to the whole paragraph or declare `:1574` as historical, which is what it is.

### M-4. §3 names the moved count `_gate_sha_count`, which is the **config key** (`config:251`). The runtime symbol is `self.reports["gate_sha_artifacts"]` (`arena:585`) and is published in the verdict's reports block. Both move; naming one hides the other.

---

## OBLIGATIONS FOR A V5 THAT WOULD CARRY GO

1. **Re-run all seven sweeps and restate every count from the output** (H-1). Assign each hit to
   exactly one accounting home; declare multi-line rows wherever they occur; give sweep F a regex;
   fold `V15E1:1547` and `:1645` into a stated subtraction; anchor or declare the `once` substring
   behaviour.
2. **An eighth sweep for the `GATE-SHA` artifact count** (H-2), with rows for `config:251-256`
   (fields and note) and `V15E1:1300`, dispositions for `V15E1:51`, `:2065`, `:2439`, and one
   sentence in §3 saying whether the design document increments `n`. If it does, row 3's `2 × U7`
   and `U7`'s stated object must move together.
3. **State §3's residual and close the publication path** (I-1): the gate pins config↔disk, not
   disk↔freeze-record, because the config is outside `frozen_sha256`; and `emit_halt` and the
   verdict face publish the **derived** digest.
4. **Decide the `PROJECTED_SECONDS` remedy on the record** (I-2): a single source, or a stated
   reason why three literals are acceptable where a hand-carried design hash was not.
5. **A row for `V15E1:1628-1629`** and a matching qualification in row 7 (I-3).
6. **The four minors** (M-1 – M-4).
7. **Carry forward unchanged and at full strength**, everything I re-derived: the two-term
   `dev_path_opens` with both factors derived over the concatenated iterable (21 files, 2 dev-like,
   0 test-like, `frozen_sha256` contributing zero, `sha256_of` opening each path exactly once);
   `74 = 1 + 66 + 6 + 1`; the uniform counter criterion with all six dispositions and the
   digest-pinned `c09guard` increment map; `int("GATE-SHA" in self.gates)` and the `argv` audit;
   the `cfg["ledger"]` cross-check; `verify_predicate` called; option (iv) declined with the TOCTOU
   reason; the rejections of (ii) and (iii); `C06_MINTS_EXECUTED`; row 26†'s dead-line deletion;
   row 2's ordinal-free §7.2 text and its agreement with rows 1, 6, 10, 11, 12; rows 26 and 27
   (`arena:4`, `mint:4`), which are v4's own find; and every figure in §4, all of which I
   re-derived to the last decimal.

Obligations 1 and 2 are the round's work. Obligations 3–6 are a row or a sentence each. **The
delta's substance does not grow**: the code change gains at most one export (obligation 4) and one
publication-site change (obligation 3); everything else is documentation and bookkeeping.

---

## WHAT V4 STILL GETS WRONG — SUMMARY PARAGRAPH

v4 is right about everything that decides the erratum, and it discharged all nine of round 3's
findings, three of them past what was asked — the ordinal contradiction is not merely fixed but made
structurally unable to recur, the design pointer went from a hand-patched literal to a gated
quantity, and v4's own sweep found two stale design references round 3 had missed. Its error is the
same shape as v3's, one level of generality up. Round 3 blocked v3 because it promised seven
subtractions and delivered one; v4 promises seven and delivers two, having adopted the *form* of the
stated residue for all seven — hit count, rows, declared leftovers — without re-running the patterns
against the files. Sweep D says twelve hits where its own pattern returns thirteen and eight rows
where its own rows cover ten; sweep E says twenty-one where the pattern returns twenty-two, names
thirteen residue items while claiming fifteen, says "twelve in V15E1" over a list of eleven, and
leaves rows 3 and 5 — its own sites — accounted for by no subtraction anywhere in the document,
because they are printed under sweep A's table while being excluded from sweep A's arithmetic and
absent from sweep E's; sweep G declares residue zero over a pattern that returns twenty-six hits
against roughly fifteen accounted; and sweep F, alone of the seven, states no pattern at all, so its
figures cannot be checked even in principle. None of this loses a site — I traced every unaccounted
hit and the delta lands complete — which is exactly why it is worth one more round rather than a
concession: the numbers are right and the argument that they are complete is not reproducible, in
the artifact whose entire thesis is that completeness must be subtracted rather than trusted. The
one place where the failure has teeth is §3, whose new gate moves the `GATE-SHA` artifact count from
37 to 38 and whose author noticed, wrote it down, and reasoned that a reported figure needs no row —
against §1's own rule that even a **CORRECT** site gets one — leaving unrecorded both a config note
that becomes false arithmetic and `V15E1:1300`, the definition of the very unit row 3 re-prices two
sections later. That is round-3 I-1's defect one row up in the same table, reached by the same kind
of sweep v4 did not run.

---

## BLINDNESS AND EDIT STATEMENT

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote` was
called zero times; no arm was built; no mint was run or read; no GPU, no SLURM job, no commit, no
`TARGET_STATE.json` edit. `artifacts/c06_falsifier/` **does not exist** and was never created; no
mint, fidelity, arena or `--gate-sha-only` process was launched. **I did not run the
`--gate-sha-only` leg** — rounds 2 and 3 carry that measurement on their record.

Compute used: `sha256sum` over the five artifacts; file reads; the seven declared `grep` sweeps plus
wider variants of each, and one additional sweep for the `GATE-SHA` artifact count; one
`c09guard`-instrumented enumeration of the config's own digest tables under `is_dev_like` /
`is_test_like` (which opens the config and the guard module, and no cache); static reads of
`c06_falsifier_arena.py`, `c06_falsifier_mint.py`, `headspace_fidelity.py`, `c09guard.py` and the
sbatch; and arithmetic.

**No file outside this review was edited.** All five artifacts carry their post-CODE-R1 hashes,
re-verified against the CODE-R1 table in `C06_FALSIFIER_IMPLEMENTATION_RECORD.md:324-328` both
before and after my probes:

| path | sha256 | matches record |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` | ✓ |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` | ✓ |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` | ✓ |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` | ✓ |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` | ✓ |

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen, failing with `ERRATUM REQUIRED`. **The
battery cannot pass `GATE-LEDGER` before this erratum lands, and could not have passed under v1, v2,
v3 or v4 as specified.**
