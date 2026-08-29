# C06 `$0` falsifier — **ERRATUM 2, INDEPENDENT REVIEW — ROUND 5**

*Target:* `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V5.md`, sha256
`c41a0223bdf6db7091148f0f38bd66707f4baff2a310553bc22b31e7174a2d32`, 793 lines.
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256
`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Reviewer:* fresh, no part in rounds 1–4. Judged from documents + repository only.

---

## VERDICT

> ## **REVISE** — 0 Critical / **1 High** / **3 Issue** / 5 Minor

**What v5 delivered.** The subtraction is real this time. All nine printed commands reproduce
their printed hit lists exactly (as sets, ordering noted at M-5); the global partition recomputes
to `183 = 107 + 76` with `UNCHARGED = 0` and `ORPHAN = 0` confirmed by construction from the hit
lists; every one of the 15 cross-sweep sites carries an identical charge in every table it appears
in; row 48 is the only siteless row and its by-construction reason holds; and the multi-line
declaration in §1 matches the charged extents of all 22 rows it names, in both directions, with
zero discrepancy. The three new live findings are confirmed at source and their corrections are
right. H-2's branch is decided correctly and its cost claim survives my own re-measurement. I-1's
honest scoping is accurate and I verified its load-bearing premise independently. Row 59's
assertions are retained and nothing in the delta weakens them. All nine of round 4's limbs are
discharged. Process integrity is clean.

**Why it does not carry GO.** The subtraction closes over the *patterns v5 chose*, and one family
is missing from that choice: **Phase 1g's `3.8 s`**. The erratum re-prices Phase 1g from
`1 × U11 = 3.8 s` to `2 × U11 = 7.6 s` (§5, and row 34 puts `7.6` into §8's total equation), but no
sweep pattern contains `3\.8`, `7\.6`, or `Phase 1[dg]`. The consequences are that **row 4 does not
move the cost cell on its own site's line** — so §8's Phase 1g row would say `3.8 s` while §8's
total nineteen lines below sums `7.6` for it — and that **four further Phase-1g sites are charged to
nothing**, one of them the bolded sentence at `V15E1:1368` that *determines* the count. That is the
U11 analogue of exactly what round-4 H-2 caught for U7 at `V15E1:1300`; v5 repaired the U7 case in
full (rows 49–54) and reproduced the identical defect one unit over.

This is High rather than Critical for the same reason round 4 gave: nothing is *silently* lost —
an implementer would hit the contradiction on the first arithmetic check. But §1's standard is
"completeness is verified by subtraction, not by trust," and §9's own closing sentence invites
precisely this test: *"Any hit a reviewer's wider pattern returns that is absent from these lists
is a defect in this document, not in the pattern."* My wider patterns returned them.

---

## 1. SWEEP DIFF — the nine printed commands, re-run

Bound `F` exactly as §9 prints it and ran all nine commands verbatim.

| sweep | v5 prints | I measure | set-identical to printed list |
|---|---|---|---|
| A `\b7[234]\b` | 18 | **18** | ✔ |
| B `[0-9](st\|nd\|rd\|th)` | 3 | **3** | ✔ |
| C projection literals | 17 | **17** | ✔ |
| D design pointer | 15 | **15** | ✔ |
| E pass-count idiom | 22 | **22** | ✔ |
| F §8 equation/shares/units | 25 | **25** | ✔ |
| G ledger quantities | 54 | **54** | ✔ |
| H `GATE-SHA` count/scope | 29 | **29** | ✔ |
| I coverage claims (`grep -i`) | 16 | **16** | ✔ |

**Nine for nine on membership.** No site appears in a printed list that the command does not
return, and no site the command returns is missing from the printed list. Sweep F now has a regex
and sweep I is case-insensitive, both as round 4 required; making sweep I case-insensitive is what
surfaced `mint:116`, which I confirm the case-sensitive pattern misses (`EVERY` is uppercase at
source).

**Ordering (M-5).** The printed tables are the command output **sorted by `(file, line)`**; the
commands as printed emit in `$F` order (`V15E1`, `arena`, `mint`, `config`, `sbatch`). I verified
`printed == sorted(actual)` for all nine and `printed == sorted(printed)` for all nine. Lossless
and deterministic, but §9's *"reproduce every hit list in this appendix line for line"* is not
literally true of a reader who pastes the command. One `| sort -t: -k1,1 -k2,2n` appended to each
printed command makes the claim exact.

---

## 2. MY OWN WIDER SWEEPS — what they returned

I ran a wider variant of every family: case-insensitive throughout, loose number patterns
(`3[0-9]{3}\.[0-9]`, `\b(6[0-9]|7[0-9])\b`, `[0-9]+\.[0-9] *%`), spelled ordinals and quantities
(`seventy`, `sixty`, `thirty-(seven|eight)`, `twenty-(one|two)`), the quantities' aliases
(`denominator`, `heartbeat`, `elapsed`, `digest`, `sha256`, `interleav`, `line-buffer`), and the
symbol names (`gate_sha_artifacts`, `self.projected`, `projected_seconds`, `_opens`,
`materialisation|materialization`). Then I diffed every hit against the 183 charged sites and
triaged the residue by hand against the quantities the erratum actually moves.

**The targeted families that come back clean — zero uncharged, verified exhaustively:**

| quantity | pattern | hits | uncharged |
|---|---|---|---|
| `3670` | `3670` | 6 | **0** |
| `4587` | `4587` | 3 | **0** |
| `2929` | `2929` | 3 | **0** |
| `2642` | `2642` | 2 | **0** |
| `1013.8` | `1013\.8` | 4 | **0** |
| `4683`/`7725` | `4683\|7725` | 2 | **0** |
| `85.6` | `85\.6` | 2 | **0** |
| `68.3`/`27.6` | `68\.3\|27\.6` | 4 | **0** |
| process counts | `\b73\b\|\b74\b`, `\b72\b` | 13, 6 | **0**, **0** |
| artifact count | `\b37\b\|\b38\b` | 17 | **0** |
| `once`/`twice` anchored | `\bonce\b\|\btwice\b` | 20 | **0** |
| `design_sha` | `design_sha\|0b446b91` | 3 | **0** |
| `U7`/`U11` | `\bU7\b\|\bU11\b` | 12 | **0** |
| projection symbols | `PROJECTED_SECONDS\|projected_seconds\|self\.projected\|projected=` | 8 | **0** |

That is a strong result and it is the substance of what round 4 asked for. The residue in the
looser sweeps (`\b21\b|\b22\b`, `V15|DRAFT_V`, `GATE-SHA`, `every process`) is overwhelmingly
irrelevant — `TIE_RANK_WINDOW = 21`, the top-21 tie window, `22/24` witness comparators, timings
like `38.87 s`, historical `v1–v15` self-references, `GateFailure("GATE-SHA", …)` call sites,
`assert_guard_active`'s "ACTIVE in every process" (true, and about the c09guard ledger rather than
the progress handle). I checked each class and none carries a quantity this erratum moves.

**The one family my wider sweeps found that v5's do not cover is Phase 1g's `3.8 s`.** See H-1.

---

## 3. GLOBAL PARTITION — recomputed from the hit lists

Parsed all nine charge tables out of §9 and recomputed every quantity independently.

| quantity | v5 prints | I recompute | |
|---|---|---|---|
| hit-instances across the nine sweeps | 199 | **199** | ✔ |
| distinct sites (deduplicated) | 183 | **183** | ✔ |
| sites returned by more than one sweep | 16 | **15** | ✘ — see M-1 |
| distinct sites charged to a row | 107 | **107** | ✔ |
| distinct sites charged to a declaration | 76 | **76** | ✔ |
| `UNCHARGED` | 0 | **0** | ✔ |
| `ORPHAN` | 0 | **0** | ✔ |

`183 = 107 + 76` holds. Every one of the 183 sites carries exactly one charge, and the charge is
identical across every table the site appears in — **15 cross-sweep sites, zero inconsistencies**:

```
config:222        [A,G]    row 11        V15E1:1574   [C,F]    row 56
V15E1:966         [E,H]    row 28        V15E1:1581   [C,F]    declaration (identical text)
V15E1:1300        [F,H]    row 52        V15E1:1904   [A,C,I]  row 7
V15E1:1304        [E,F]    row 37        arena:55     [E,I]    declaration (identical text)
V15E1:1547        [E,F]    row 3         arena:465    [A,G]    row 12
V15E1:1550        [A,F]    row 4         arena:466    [A,G]    row 12
V15E1:1570        [C,F]    row 34        mint:117     [A,I]    row 13
V15E1:1571        [C,F]    row 34
```

**Row 48 is the only siteless row and its reason holds.** Of rows `1`–`62` plus `26†`, 62 have at
least one sweep site; only row 48 has none. It adds `C06_PROJECTED_SECONDS` / `C06_MINTS_EXECUTED`
exports to the sbatch, and I confirmed the sbatch sets neither today (its only exports are the four
thread caps, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH` and `C09_LEDGER_DIR`). A grep cannot return a line
that does not exist. The by-name exclusion is correct.

**Multi-line declaration — audited both directions.** Rows whose charged sites number more than one
are exactly `{7, 12, 15, 16, 23, 24, 32, 34, 35, 36, 39, 41, 43, 44, 46, 47, 49, 53, 57, 58, 59, 61}`,
identical to §1's declared list. Nothing declared is single-site; nothing multi-site is undeclared.
Stated extents match charged extents line for line (row 12 = `arena:465-467`; row 39's six lines;
row 43's seven; row 49's five; row 59's five; row 58's three). The **one** exception is row 19,
whose stated site is a three-line range — see I-3.

---

## 4. THE THREE NEW LIVE FINDINGS — confirmed at source

| site | source text (verbatim) | v5's disposition | my finding |
|---|---|---|---|
| `mint:118` | *"85.6 % of §8's budget -- dark for its whole span"* | row 55: → `68.3 %` | **CONFIRMED live-wrong.** `V15E1:1576` (*"mints fall from `85.6 %` to `68.3 %`"*) and `:1607` (*"Mints are `68.3 %`"*) both carry the post-CODE-R1-H-4 share. Row 13 already makes the process counts in this same docstring current, so leaving the share would produce the half-current sentence v5 describes. Correction right. |
| `mint:116` | *"§9 requires **EVERY** python process to append through a handle opened `buffering=1`"* | row 60: qualify | **CONFIRMED.** Uppercase `EVERY`; a case-sensitive `every python process` pattern misses it, as v5 says. False for the 6 fidelity processes. Correction right. |
| `mint:209` | `--progress` help: *"§9 progress file; every python process appends to it (H-3)"* | row 62: qualify | **CONFIRMED.** `--help` output carrying the claim. Correction right. |
| `config:217` | `"mints_present_before_arena": {"expected": 66, "binding": true}` | row 41: **CORRECT**, computed | **CONFIRMED.** Computed at `arena:628` and compared at `:463`. Stays measured and binding; the by-construction criterion correctly does not touch it. |
| `arena:559-560` | gate_sha docstring, both limbs | row 32: both limbs, two lines | **CONFIRMED.** `:559` is the scope limb, `:560` the pass-count limb. Two lines, exactly the pair row 28 changes in the design document. |
| `V15E1:1786-1787` | *"`GATE-SHA`'s scope is stated in §6 as the frozen imports and the input caches plus the sixteen banked artifacts above."* | row 53: add the design document | **CONFIRMED.** §11's own scope sentence, spanning two lines, in none of v4's seven sweeps. |

The **supporting** findings also check out at source: `mint:112` is `PROJECTED_SECONDS = 2929.9`
against the arena's `3670.0` (row 17, live-wrong); `config:6` is `0b446b91675fd4ff8aea…` while
V15E1's actual digest is `8cde58aa…` (row 21, live-wrong); `arena:1418` is the dead
`sum(1 for _ in [None])` placeholder (row 26†); `arena:1419-1420` already reads
`C06_MINTS_EXECUTED` with a fallback (row 47 CORRECT, and row 48 is genuinely needed).

---

## 5. H-2's DECISION — 37 → 38, and the circularity question

**The branch is decided correctly and the mechanism is where v5 says it is.** I read `gate_sha`
(`arena:558-586`) in full. `n` is incremented once per artifact in the concatenated digest loop
(`frozen_sha256` **13** entries + `frozen_sha256_input_caches` **8** = **21**, enumerated from the
config) and once per banked artifact in the existence loop (`2 ds × 3 seeds = 6` OUT JSONs +
`2 ds × 5 folds = 10` vsw_ckpt npz = **16**), so `n = 37` today and `n = 38` once the design
document joins the concatenated iterable at `arena:563`. The 16 banked are existence-checked, not
re-hashed — which is what makes sweep H's declaration at `V15E1:1463` correct and what makes the
added cost exactly one `sha256_of` call.

**Every site of the count has a row.** `config:251-255` → row 49 (I confirm `:256` is the closing
brace, so v5's five-line extent is right where round 4 wrote `:251-256`); `V15E1:1300` → row 52;
`arena:585` → row 51; `V15E1:2439` → row 54; `V15E1:51` and `:2065` declared historical in sweep H.
Row 52's arithmetic checks: `8 caches + 13 modules/configs + 16 banked + 1 design = 38`, and
`7 + 6 = 13` matches the config's `imported_modules 7 + read_for_definitions 6`.

**The cost claim survives my re-measurement.** I re-hashed the 188 061-byte design document 7 times
on the login node:

```
size 188061 bytes   digest 8cde58aa…  (matches the declared artifact hash)
reps=7   min 0.000148 s   median 0.000148 s   max 0.000316 s
```

v5 reports median `0.000164` (min `0.000158`, max `0.000371`). Same order, same conclusion: **0.11 %
of `U7`'s `0.13 s`** on my numbers, `0.13 %` on v5's, invisible at two decimals either way. `U7 =
0.13 s` unchanged, `2 × U7 = 0.2 s` stands, §8's total unaffected by the §3 mechanism. The claim is
measured, not asserted, exactly as H-2 required.

**No circularity.** I checked the shape directly. §11 (`V15E1:1715`ff) is a table of *other*
artifacts' digests; it does **not** contain its own. The design document's expected digest lives at
`config:6`, and the config is not hashed by `GATE-SHA` (see §6 below). So the arena hashes a file
that does not contain its own hash and compares it to a value stored elsewhere — no fixed point to
solve. v5's landing order (V15E2 written → its sha256 computed → code/config edits →
`design_sha256` set **last**) is sufficient and is stated. Row 54's replacement text is the right
formulation: *"the 38th is this document, whose digest is by construction the one `config:6`
carries"* — it names the artifact without quoting a digest into the document that would have to
contain it. **The self-reference is resolved by storing the digest outside the hashed file, and
that is exactly what the design does.**

---

## 6. I-1's HONEST SCOPING — verified, and judged acceptable

Both load-bearing premises hold under my own measurement.

* **The startup gate pins config↔disk only.** I enumerated `frozen_sha256` (13 entries, all `.py`
  and `configs/c01/*.json`) and `frozen_sha256_input_caches` (8). `configs/c06/c06_falsifier.json`
  is in **neither**, and neither is the design document. So `cfg["design_sha256"]` is pinned by
  nothing inside the job, and v5's *"removes the observed subclass, not the class"* is the accurate
  statement. Uncoordinated drift — the CODE-R1 failure that produced the live-wrong `config:6` — is
  caught; coordinated drift is not.
* **The implementation record is read by no code path.** `grep -rn IMPLEMENTATION_RECORD scripts/
  configs/` returns **zero** hits. The only references are in `refine-logs/` prose. Confirmed.
* **The HALT is free and early.** `main()` runs `gate_det1 → assert_guard_active → gate_sha →
  load_frozen` and only then `if args.gate_sha_only: return 0` (`arena:1269-1280`), and the sbatch
  invokes the driver leg at `:63-64` before the mint loop at `:67`. A design-drift HALT fires in
  process 1 of 74, before any mint. `--out` defaults to
  `artifacts/c06_falsifier/C06_VERDICT.json` (`arena:1225-1226`) and the driver leg is invoked
  without `--out`, so that HALT lands on the canonical verdict path — v5's incidental note is
  right.
* **Row 5's premise measured.** `Heartbeat` is instantiated at `arena:1266` and the driver leg
  emits three lines (`GATE-DET1` `:1270`, `GUARD` `:1272`, `GATE-SHA-ONLY` `:1279`) before
  returning, and it runs `load_frozen()` first. `V15E1:1645`'s *"the one span"* is measurably false.

**Is publishing both digests sufficient for the both-drift case? No — and v5 does not claim it is.**
In coordinated drift, `sha256_declared == sha256_derived`, both equal to the drifted value, and
equality *reads* like verification. Detection then requires the external anchor, which v5 names
(the freeze table in `C06_FALSIFIER_IMPLEMENTATION_RECORD.md`) together with the reason it works
(no code path reads it, and the separate code/resource review lineage checks it). That is the
correct epistemics, and I judge **the honest scoping acceptable — no further anchor required**.

Adding one is not worth it inside the job: the sbatch is no more pinned than the config, so a
second declaration there only raises coordinated drift from two coordinated edits to three. The
residual is irreducible within a job none of whose five artifacts is self-pinning, which is why
closing it by procedure is the right call. The one cheap improvement is reader-side, not
mechanism-side — see M-4.

---

## 7. ROW 59 — the deliberate call, verified

`arena:458-462` is exactly the two uninstrumented assertions:

```
458:  if tot.get("test_label_materialisations", 0) != 0:
459:      fails.append("test_label_materialisations = {} != 0".format(
460:          tot["test_label_materialisations"]))
461:  if tot.get("dev_or_test_labels_into_decision_quantities", 0) != 0:
462:      fails.append("dev_or_test_labels_into_decision_quantities != 0")
```

Row 59 marks these **RETAINED VERBATIM** with only the *publication* moving, and §8's arena cell
repeats it in the delta itself (*"three counters to `by_construction` publication with `:458-462`
**retained verbatim** (59)"*). The two statements agree; nothing else in the delta touches
`:458-462`; and the neighbouring `test_path_opens` assertion at `:456-457` (row 46) stays measured
and binding, which is right because `_guarded_open:97` is the only one of the three that
increments. I confirm the c09guard increment map exactly as §3's table states it — `LEDGER[…] += 1`
occurs at **`:97` `test_path_opens`**, **`:102` `dev_path_opens`**, **`:106`
`banked_trainlog_opens`**, and nowhere else — and that both `c09guard.py` and `sitecustomize.py`
are in `frozen_sha256`, so the digest-pinning argument holds. **The call is sound and correctly
recorded.** The §3 warrant text attached to one of these counters is not — see I-2.

---

## 8. ROUND-4 DISPOSITION AUDIT — LIMB LEVEL

| R4 finding | limb | v5's answer | disposition |
|---|---|---|---|
| **H-1** | assign every hit to exactly one accounting home | §1's charge rule; partition recomputes `183 = 107 + 76`, 15 cross-sweep sites all consistent | **full** |
| | declare multi-line rows wherever they occur (16, 23, 24, 34, 36 minimum) | §1 lists 22 rows; all five named are present; declared set == charged-multi set exactly | **full for the named; row 19 missed → I-3** |
| | give sweep F a regex | sweep F has one and reproduces at 25 | **full** |
| | re-run all seven, restate counts from output | nine sweeps, all nine reproduce | **full** |
| | fold `V15E1:1547` and `:1645` into a stated subtraction | sweep E hits 11 and 13, rows 3 and 5 | **full** |
| | anchor or declare the `once` substring behaviour | `V15E1:995` declared *"SUBSTRING ARTIFACT: `once` inside `concentration`"*; pattern deliberately retained unanchored | **full — and the reasoning for retaining it is right** |
| **H-2** | eighth sweep with hit list and subtraction | sweep H, 29 hits, reproduces | **full** |
| | rows for `config:251-256` (fields **and** note) and `V15E1:1300` | rows 49 (five lines incl. note) and 52 | **full** |
| | dispositions for `V15E1:51`, `:2065`, `:2439` | `:51`/`:2065` declared historical; `:2439` → row 54 | **full** |
| | one sentence in §3 saying whether the design document increments `n` | §2.7: *"the design document DOES increment `n`"* | **full** |
| | if it does, row 3's `2 × U7` and `U7`'s object must move together | row 52 moves the object, row 3 prices `2 × U7`, §5 **measures** the cost | **full — past what was asked** |
| **I-1** | state the residual (config↔disk, not disk↔freeze-record) | §4, with the config-not-in-`frozen_sha256` fact | **full** |
| | name what anchors the digest outside the job | the freeze record, and that no code path reads it | **full** |
| | `emit_halt` and the verdict face publish the derived digest | rows 23, 24 publish **both**, with `.get(…, "NOT_DERIVED")` for the pre-`gate_sha` reachability | **full** |
| **I-2** | decide the `PROJECTED_SECONDS` remedy on the record | §6: `config:43` single source, sbatch export, three-way assertion, row 19 deletes the false clause | **full — see M-3 on placement** |
| **I-3** | a row for `V15E1:1628-1629` and a matching qualification in row 7 | row 61 (`:1627-1629`, R4's own text) + row 7 rewritten to qualify rather than restate | **full, and widened by rows 60, 62** |
| **M-1** | correct the *"twelve over eleven"* count | dissolved: lists generated | **full in form — see M-1 below on one cell that is not** |
| **M-2** | `once` unanchored / substring | declared at `:995` | **full** |
| **M-3** | `V15E1:1574` mis-described | row 56, declared historical | **full** |
| **M-4** | `_gate_sha_count` vs `gate_sha_artifacts` | rows 49 and 51 | **full** |
| **7** | carry forward everything re-derived | see below | **full** |

**Obligation 7 re-derived independently by me, not taken on trust.** The concatenated iterable is
21 files; running `c09guard.is_dev_like` / `is_test_like` over it gives **2 dev-like** (the two
`dev_seen_*.pt` input caches) and **0 test-like**, `frozen_sha256` contributing zero; the design
document under both its V15E1 and V15E2 names returns `False`/`False` on both predicates, so
`expected_sha_dev_opens = 2 × 2 = 4` is unchanged by §3's mechanism. `arena:449` computes
`len(procs) + 1`, and with the 6 fidelity processes and the driver leg all writing c09guard ledger
files that yields **74** after row 12's repair — row 45's CORRECT disposition is right and v4 was
wrong to leave it unrecorded. `74 = 1 + 66 + 6 + 1` reads off the sbatch. Row 2's ordinal-free §7.2
text agrees with rows 1, 6, 10, 11, 12 (`72 + 2 = 74`). Rows 26/27 (`arena:4`, `mint:4`) confirmed.
`arena:1418`'s dead line confirmed.

---

## FINDINGS

### H-1. The Phase 1g re-pricing has no sweep, row 4 leaves the cost cell on its own site's line, and four Phase-1g sites — including the bolded sentence that determines the count — are charged to nothing.

*Attaches to:* §5's re-pricing table; rows 3, 4, 5, 34; §9's nine patterns; `V15E1:1550`, `:1346`,
`:1362`, `:1368`, `:1646`, `:1199`, `:42`, `:1350`, `:1993`.

**No sweep pattern contains `3\.8`, `7\.6`, or `Phase 1[dg]`.** Sweep F was built for §8's equation,
shares and risk row and catches `U11` *mentions*, but the Phase 1g literal itself is unswept. Every
consequence below follows from that one gap, and it is the same shape as round-4 H-2: a quantity the
erratum moves, with sites in the artifacts, absent from every pattern.

**(a) Row 4 is incomplete for its own site.** `V15E1:1550` is the §8 Phase 1g table row. That one
line carries the count column *and* the cost column, and `3.8` occurs on it **twice** — the cell
`| U11, arena class | **3.8 s** (…` and, inside the same cell, *"`3.8 s` is carried **above the
pooled maximum** by `0.083 s`"*. Row 4's `current` column quotes only *"**`1`** — the arena process
alone … `66+6+1 = 73` accounts for **every** process"* — a span that ends at the close of the count
column — and its `correct` column replaces exactly that. The cost cell is untouched.

Contrast row 3, one table row up. `V15E1:1547` is compact (`| **1d** GATE-SHA, once in the driver |
1 | U7 | 0.1 s |`) and row 3 moves the whole line including `0.1 s → 0.2 s`. The asymmetry is not
deliberate: §5's own re-pricing table says *"Phase 1g arena-class startups | `1 × U11 = 3.8 s` |
**`2 × U11 = 7.6 s`**"*, and row 34 puts **`7.6`** into §8's total equation. §8's implementation
delta lists the §8 edits as exactly rows 3, 4, 34, 35, 36, 56, 57 — so **no row moves Phase 1g's
cost cell.** At landing, §8's Phase 1g row reads `3.8 s` and §8's total nineteen lines below sums
`7.6` for Phase 1g. That is precisely the *"equation summing to `3673.8` under a stated `3673.9`"*
failure row 34 exists to prevent, reproduced one row up in the same table.

**(b) `V15E1:1368` is the sentence that determines the count, and it is uncharged.**

> **So §8 Phase 1g's count is `1`, the arena alone, and it is determined by this measurement rather
> than inferred.**

Bolded, in §7.7, the conclusion of the `U9`-boundary argument. Returned by none of the nine sweeps
and covered by no row. It becomes flatly false at landing, and it contradicts row 2's new §7.2 text
(*"the remaining two processes … are priced separately at §8 Phase 1g"*) and row 4's new count of
`2`. **This is the U11 analogue of `V15E1:1300`** — the site round-4 H-2 singled out as *"the
definition of the unit row 3 re-prices"*. v5 repaired the U7 case completely and left the U11 case
untouched because no pattern looked.

**(c) Two more live-wrong Phase 1g sites, both uncharged.**

* `V15E1:1346` — *"§8 Phase 1g carries **`3.8 s`**, above the pooled maximum by `0.083 s`."* After
  landing Phase 1g carries `7.6 s`. The `0.083 s` margin is a property of the **unit** (`3.8` vs the
  pooled max `3.717`), not of the row, so the sentence needs splitting rather than renumbering.
* `V15E1:1646` — *"…`3.8 s` as carried at §8 Phase 1g…"*. This is the continuation line of the very
  sentence row 5 rewrites (row 5's site is `:1645`), so row 5 is also a multi-line row that is not
  declared as one.

**(d) Four further uncharged sites that survive but are undeclared:** `V15E1:1199` (the second line
of row 2's own replacement text — row 2 is a multi-line row not declared as one), `:1362` (*"the
count in §8 Phase 1g turns on this clause"* — still true), `:42` and `:1350` (about the **unit**
`3.8 s` and Phase 1d's rounding — both survive), `:1993` (*"§8 Phase 1g's unit is undeterminable
without it"* — unit, survives).

I checked the neighbouring provenance block `V15E1:1581-1584` (*"Round-11 I-1 added Phase 1g at
`1 × U11` … round-12 I-1 corrects that row's unit from `3.2` to `3.8 s`"*) and it is **correctly
left alone**: it is historical, true of the total as it then stood, and already charged as a
declaration in sweeps C, E and F.

**Repair.** A tenth sweep — `grep -nE '\b3\.8\b|\b7\.6\b|[Pp]hase 1[dg]' $F` — with its hit list,
charges and subtraction; **extend row 4 to move `V15E1:1550`'s cost cell `3.8 s → 7.6 s` and split
its `0.083 s` clause into a unit statement**, matching what row 3 already does for Phase 1d; rows
for `V15E1:1368` and `:1346`; fold `:1646` into row 5 and `:1199` into row 2, declaring both as
multi-line; and declare `:42`, `:1350`, `:1362`, `:1993` as surviving non-targets.

### I-1. `arena:432-441` states that the predicate is blocked and NOT adjusted; row 39 charges two lines out of that paragraph and prescribes no edit to the rest, so the docstring is false the moment the erratum lands.

*Attaches to:* row 39; `arena:432`, `:434-437`, `:439-441`.

Row 39's six sites are `:433, :438, :468, :469, :471, :475`. Lines `:433` and `:438` sit inside a
ten-line docstring paragraph whose surrounding lines are returned by no sweep — sweep G's pattern
matches them only where `dev_path_opens` or `mints_executed` appears literally. Those surrounding
lines say:

```
432:  ONE PREDICATE IS BLOCKED ON A DESIGN ERRATUM AND IS NOT ADJUSTED HERE.  §12 binds
439:  predicate is implemented exactly as frozen; the failure message carries the
440:  decomposition and the words ERRATUM REQUIRED.  Adjusting it to pass would be a
441:  design change made by the implementation lineage, which is not this lineage's call.
```

Every one of those claims is false once ERRATUM 2 lands — the predicate *is* adjusted, by exactly
the design change the paragraph says is not this lineage's call. Row 39's `correct` column says only
*"the two-term predicate; the message's decomposition becomes the derivation"* and does not reach
the docstring. An implementer editing `:433`/`:438` would very likely rewrite the paragraph, so
this is unlikely to ship — but "unlikely to ship" is the standard §1 exists to replace, and
`:434-437` also carry the `+2 per GATE-SHA process, +4 total` derivation that row 39's new
`expected_sha_dev_opens` block supersedes.

**Repair.** Extend row 39's site list to `arena:432-441` (or state the docstring rewrite in its
`correct` column) and charge those lines in sweep G's table.

### I-2. §3's warrant for `dev_label_materialisations_outside_decisions` is factually wrong at source — and §3 prescribes publishing that warrant on the verdict face.

*Attaches to:* §3's counter table, row 40, row 41.

The table's column is headed **warrant (source-verified)**, and for this counter it reads:

> `lab_dev` occurs **exactly once** in the executed corpus — `headspace_mint.py:323` — and in
> neither the arena nor `headspace_fidelity.py`

Measured: `lab_dev` occurs in the executed corpus **three times**, in two files.

```
headspace_mint.py:323        lab_dev=dv[3].numpy().astype(int), fold_of=fold_of,
c06_falsifier_mint.py:31     (docstring) "... writes K_train / K_dev / lab / lab_dev / ..."
c06_falsifier_mint.py:336    lab=frozen["lab"], lab_dev=frozen["lab_dev"],
```

`c06_falsifier_mint.py` is the executed mint driver — the sbatch invokes it at `:83` and `:91`, 66
times — and `:336` is a live `np.savez` that **re-materialises the dev labels into the banked
`.npz` the arena subsequently loads**. That is the more relevant of the two writes for this
counter's warrant, and it is the one the warrant omits.

**The disposition survives; the warrant does not.** I verified the safety limb independently: the
arena contains no `lab_dev` reference and no generic `.npz` key iteration (`.files`, `for k in z`,
`z.keys` all absent), and `headspace_fidelity.py` has no `lab_dev` at all. So "written, read by no
decision path" holds and the counter is correctly published as a by-construction string rather than
an integer. But §3 says the string **carries its warrant**, which means this checkable, false count
claim goes onto the verdict face as the justification for not measuring the counter. Given that the
adjacent warrants in the same table are exact (`_guarded_open` raises on a test path; the increment
map at `:97`/`:102`/`:106`), this one should be too.

**Repair.** Restate as: *"`lab_dev` is written twice in the executed corpus —
`headspace_mint.py:323` and `c06_falsifier_mint.py:336`, the latter into the banked `.npz` — and is
read by no path in the arena or `headspace_fidelity.py`, neither of which references the key, and
the arena never iterates `.npz` keys generically."* That is both true and stronger.

### I-3. Row 19 is a three-line row that §1 does not declare, and the clause its own prescription deletes lives on one of the two uncharged lines.

*Attaches to:* §1's multi-line paragraph; row 19; `V15E1:1632`, `:1633`.

§1 promises, verbatim: *"Multi-line rows are declared everywhere they occur, not only where v4
happened to notice: rows 7, 12, … each span more than one line, and **every line of each is a
separate charged site in §8**."* I parsed all 63 rows' stated sites and compared against that list.
The audit is otherwise perfect — every declared row is multi-site, every multi-site row is
declared. **Row 19 is the single exception:** its stated site is `V15E1:1631-1633`, three lines, and
it appears nowhere in §1's list. Only `:1631` is a charged site; `:1632` and `:1633` are returned by
no sweep.

That matters more than the bookkeeping, because of what is on the uncharged lines:

```
1631:  projected** value. *(ERRATUM 1 set it to `2929.9 s`; **CODE-R1 H-4 sets it to `3670.0 s`**. The
1632:  denominator is pinned to §8 by name, so it tracks automatically; the literal in
1633:  `c06_falsifier_arena.py` and `configs/c06/c06_falsifier.json` is updated with each correction.)*
```

Row 19's prescription is to carry `3673.9 s`, **delete the false *"tracks automatically"* clause**,
and state the single source. The clause it deletes is on `:1632`; the two-file claim it replaces is
on `:1633`. So the two lines the row exists to fix are the two the partition does not see. Nothing
is lost at landing — the prescription names the clause explicitly — but this is round-4 H-1's own
defect class, recurring in the document that claims to have closed it everywhere.

Row 2 (`:1198`, whose prescribed replacement text visibly runs onto `:1199`) and row 5 (`:1645`,
whose sentence runs onto `:1646`) are the same pattern; both are folded into H-1's repair.

**Repair.** Add row 19 to §1's multi-line list and charge `:1632` and `:1633` in sweep C's table
(both match `projected`, so the existing pattern already returns the range once the row is stated
over it — I confirm `:1631` is the only one of the three the current pattern hits, so the rows'
extent, not the pattern, is what needs stating).

### M-1. The global partition's *"sites returned by more than one sweep: 16"* is wrong; the value is 15. It is the one count in §9 that is not what its label says.

199 hit-instances − 183 distinct sites = **16 excess instances**, but only **15 distinct sites** are
returned by more than one sweep, because `V15E1:1904` is returned by **three** (A, C and I) and so
contributes two excess instances by itself. The identity `183 = 107 + 76` and both emptiness checks
are unaffected and verified. But §7's M-1 disposition rests on *"§8's lists are generated, so no
count in this document is written by hand"* — and a cell whose label and value describe different
quantities is the counterexample to that defence. Either relabel it *"excess hit-instances"* or
print `len({s : multiplicity(s) > 1})`.

### M-2. `total_§11_digests: 22` counts a digest that §11 does not carry.

§11's table holds 21 digests plus 16 banked artifact names; the design document's digest lives at
`config:6`. Row 49's prescribed note (*"§11 declares **38** = 7 + 6 + 8 … plus the design document
… **22 + 16 = 38**"*) and the key name `total_§11_digests` therefore attribute to §11 a digest
stored outside it. Row 53 does amend §11's scope sentence to name the design document and row 54
says correctly that its digest *"is by construction the one `config:6` carries"*, so this is
naming looseness rather than an arithmetic error — but one clause in the note (*"…plus the design
document, whose digest §11 names and `config:6` carries"*) removes the ambiguity for the code
lineage that reads the key name alone.

### M-3. §6's *"cannot reach a run"* holds only for one placement of the three-way assertion, and §6 does not say which.

§6 item 4: *"The arena asserts all three agree — environment, module constant,
`cfg["projected_seconds"]` — and HALTs on mismatch, **so a future re-price that touches one place
cannot reach a run**."* Measured: `main()` runs `gate_det1 → assert_guard_active → gate_sha →
load_frozen` and only then `if args.gate_sha_only: return 0` (`arena:1269-1280`). The guarantee is
true if the assertion sits **before** `:1278`, where it fires in process 1 of 74 at zero cost.
Placed anywhere later it fires only in process 74, after all 66 mints have already published ratios
against the drifted denominator — the same failure mode, one run later. Given that §4 pins the
design-digest gate's ordering explicitly and for exactly this reason, §6 should pin this one too:
*"asserted in the same pre-`gate_sha_only` block as the design-digest gate."*

### M-4. The verdict face's digest pair does not tell an auditor that `sha256_declared` is unpinned inside the job.

Rows 23 and 24 publish `sha256_declared` and `sha256_derived`, which is the right repair and closes
the uncoordinated-drift case completely. In the coordinated-drift case both fields carry the same
drifted value, and an auditor reading the artifact alone sees equality — which reads as
verification. §4 states the limitation, but §4 is not on the artifact. One label beside the pair
(*"declared digest is not pinned inside the job; the external anchor is the freeze record in
`C06_FALSIFIER_IMPLEMENTATION_RECORD.md`"*) costs one string and puts the caveat where the reader
is. Hardening, not a defect — the honest scoping itself is correct and I do not require another
anchor.

### M-5. §9's printed lists are the commands' output sorted by `(file, line)`, not the commands' output.

All nine are set-identical and `printed == sorted(actual)` for all nine, so nothing is lost and the
transformation is deterministic. But §9 claims the commands *"reproduce every hit list in this
appendix line for line,"* and a reader who pastes a command gets `$F` order. Append
`| sort -t: -k1,1 -k2,2n` to each printed command and the claim becomes literally true.

---

## OBLIGATIONS FOR A V6 THAT WOULD CARRY GO

1. **A tenth sweep for the Phase 1g quantity** (H-1): `grep -nE '\b3\.8\b|\b7\.6\b|[Pp]hase 1[dg]'`
   with hit list, charges and subtraction; **extend row 4 to move `V15E1:1550`'s cost cell
   `3.8 s → 7.6 s`** and split its `0.083 s` clause into a unit statement; rows for `V15E1:1368`
   and `:1346`; `:1646` folded into row 5 and `:1199` into row 2, both declared multi-line;
   `:42`, `:1350`, `:1362`, `:1993` declared as surviving non-targets.
2. **Extend row 39 over `arena:432-441`** or state the docstring rewrite in its `correct` column
   (I-1).
3. **Correct the `lab_dev` warrant** to name `c06_falsifier_mint.py:336` (I-2).
4. **Add row 19 to §1's multi-line list** and charge `V15E1:1632-1633` (I-3).
5. **The five minors** (M-1 – M-5).
6. **Carry forward unchanged and at full strength, everything I re-derived independently:** the
   nine sweeps' reproduction, nine for nine; the partition `183 = 107 + 76` with `UNCHARGED = 0`
   and `ORPHAN = 0`; the 15 cross-sweep sites and their identical charges; row 48 as the only
   siteless row; the multi-line declaration for all 22 rows it names; the three new live findings
   and their corrections; H-2's branch, its `21 + 16 = 37 → 22 + 16 = 38` arithmetic and its
   measured `U7` consequence; the absence of circularity; I-1's honest scoping and both its
   premises (config not in `frozen_sha256`, freeze record read by no code path); the `main()`
   ordering and the driver leg's three heartbeat lines; row 59's retained assertions and the
   c09guard increment map; `expected_sha_dev_opens = 2 × 2 = 4` under `is_dev_like`; `arena:449`
   yielding 74; and every source confirmation in §4 of this review.

**The delta's substance still does not grow.** Obligation 1 is one sweep, one extended row and two
new rows; obligations 2–4 are a site-list edit, a sentence and a list entry. No code change beyond
what v5 already prescribes.

---

## WHAT V5 STILL GETS WRONG — SUMMARY

v5 finally delivers the method this lineage has been promising since v1: nine sweeps that reproduce
exactly, a partition that closes in both directions, charges that agree across every table a site
appears in, and — this is the part that deserves saying plainly — a multi-line declaration that
matches the charged extents of all 22 rows it names with zero discrepancy either way. Widening its
own patterns rather than narrowing them found three live-wrong sites v4 had no mechanism to see.
The subtraction is no longer a claim. What it still gets wrong is that the subtraction closes over
the patterns v5 chose, and the choice has one hole: **Phase 1g's `3.8 s`**, the second of the two
units this erratum re-prices. Sweep F was written for §8's equation and catches `U11` mentions, so
`U7`'s object grew and got a row (52) while `U11`'s row-level cost moved and got none — with the
consequence that row 4 edits the count column of `V15E1:1550` and leaves `3.8 s` standing in the
cost column of the same line, against row 34's `7.6` in the total nineteen lines below, and with
`V15E1:1368`'s bolded *"§8 Phase 1g's count is `1`, the arena alone"* left as the document's own
contradiction of its new count. That is the same defect round-4 H-2 named for `U7` — a quantity
moved by the erratum, sites in the artifacts, absent from every pattern and every row — reproduced
one unit over in the document that fixed the first instance. The other three findings are smaller
and of a piece: a docstring paragraph that will say the predicate was not adjusted after it was, a
source-verified warrant that names one of two write sites and is destined for the verdict face, and
one three-line row whose two uncharged lines happen to be the two lines it exists to delete. None
of them loses a site at landing. All of them are the difference between a delta a reader can verify
and one a reader has to trust, which is the whole of what §1 claims.

---

## BLINDNESS AND EDIT STATEMENT

**No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm.** `deployed_vote`
called zero times; no arm built; no mint run or read; no arena run; no `--gate-sha-only` leg run; no
GPU, no SLURM job, no commit, no `TARGET_STATE.json` edit. `artifacts/c06_falsifier/` does not exist
and was not created (verified).

**Compute used:** file and review reads; the nine `grep` sweeps of §9 run verbatim plus wider
variants of every family; python parsing of §9's charge tables and §2's row tables for the partition
recompute; `sha256sum` over nine files; one login-node timing of `hashlib.sha256` over the
188 061-byte design document, 7 repetitions; `c09guard.is_dev_like` / `is_test_like` over the 21
concatenated-iterable paths and the design document; `json` enumeration of the config's digest
tables; static reads of the sbatch. No edits outside the scratchpad; this review file is the only
thing written.

**All five artifacts unchanged**, re-verified at the start and end of this review:

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` |

**v1–v4 byte-unmodified**, and each matches the digest v5's supersession header declares:
`f063c388c4afabdb7964…` (v1), `4225bea3cc9907d38e2e…` (v2), `48f4e0153103cc608884…` (v3),
`0b4940416abd1fb4bf79…` (v4).

The arena still implements `dev_path_opens == mints_executed + 0` (`:468-475`) and
`processes_reporting != 73` (`:465-467`) exactly as frozen. **The battery cannot pass `GATE-LEDGER`
before this erratum lands, and could not under v1–v5 as specified.**
