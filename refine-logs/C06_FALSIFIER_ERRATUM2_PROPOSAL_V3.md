# C06 `$0` falsifier — **ERRATUM 2, PROPOSAL v3**

**Supersedes** `C06_FALSIFIER_ERRATUM2_PROPOSAL_V2.md` (`4225bea3cc9907d38e2e…`), which supersedes
`C06_FALSIFIER_ERRATUM2_PROPOSAL.md` (`f063c388c4afabdb7964…`). **Both stay on disk** as the record
of what rounds 1 and 2 reviewed.
*Against:* `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`, sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`.
*Adjudications:* `C06_FALSIFIER_ERRATUM2_REVIEW.md` (0C/3H/3I/3M) → v2;
`C06_FALSIFIER_ERRATUM2_REVIEW_R2.md` (0C/2H/3I/3M) → this document.
*Status:* **PROPOSAL. Nothing is landed.** All five artifacts carry their post-CODE-R1 hashes.

---

## 0. What round 2 settled, and what it blocked

**Settled and carried forward unchanged:** the `74` inventory (the reviewer's own static sbatch
count `1 + 36 + 30 + 6 + 1 = 74`, with the fork and grandchild channels closed **at source** — no
`subprocess`/`multiprocessing` anywhere on the path, every RAC `DataLoader` hard-coding
`num_workers=0`, and `wandb.init` never called); the four-part decomposition; the `+4` and both its
factors over the concatenated iterable; `torch.load` = exactly one `builtins.open` per file; the
demotion's warrant; the audit-over-declaration principle; and the rejections of (ii), (iii) and (iv).

**Round 2 added one fact that strengthens the case and belongs in the erratum:** `74` is
**resume-stable**. A resumed mint still *spawns* — `c06_falsifier_mint.py:217-220` returns early
**inside** the process, after `assert_guard_active()` and the `MINT-SKIP` heartbeat — so it still
flushes a ledger. This is why C06 can bind an exact `74` where C09 explicitly refused to bind its
`39` (`c09_a0_arena.py:1927-1958`, *"a resume legitimately reports FEWER … The gate requires >= 1
reporting process"*). C06's gate is **stronger** than the precedent's, and it is safe only because
of the spawn/skip split.

**Blocked on two things, both the same class:**

* **H-1 — the completeness boundary.** v2 enumerated all nine §12 predicates and then prescribed a
  delta leaving **five more sites** asserting 73. Landing it would produce a V15E2 saying 74 in
  §12/§13 and 73 in §7.2/§8 — the trade round 1's H-2 refused. **Third recurrence of one omission.**
* **H-2 — the demotion's second limb.** Three of six counters are incremented by nothing; v2 demoted
  one, left two binding, and took none of the three off the measured face, so the face still
  publishes an integer no code produces. C09's own block publishes all three as narrative strings.

**v3's answer to H-1 is the section that follows.** The completeness obligation is upgraded from
*"every predicate in §12"* to *"every site in every file that states a quantity this erratum
moves"*, enumerated by grep so a reviewer verifies it **by subtraction rather than by trust**.

---

## 1. FULL-DOCUMENT ENUMERATION — every site stating a moved quantity

Produced by grepping all five artifacts for `\b7[234]\b`, for `once`/`in the driver` pass-count
language, for process-count language carrying no numeral, and for each quantity this erratum moves.
**26 rows.** Rows marked **CORRECT** need no edit and are listed anyway, because completeness is
verified by subtraction.

### 1a. Process-count sites

| # | site | current text | correct text | why |
|---|---|---|---|---|
| 1 | `V15E1:1197` §7.2 | *"those **72** processes"* | **unchanged** — **CORRECT** | 66 mints + 6 fidelity = 72 is still exactly right at 74 |
| 2 | `V15E1:1198` §7.2 | *"The **73rd** process, the arena, is a different case and is priced separately at §8 Phase 1g"* | *"The **73rd and 74th** processes — the `--gate-sha-only` driver leg and the arena — are a different case and are priced separately at §8 Phase 1g"* | the arena is the 74th; the `--gate-sha-only` process is a **third class** whose interpreter cost is inside no mint unit, not inside `U9`, and (before this erratum) not in Phase 1g |
| 3 | `V15E1:1547` §8 Phase 1d | *"`GATE-SHA`, once in the driver \| `1` \| `U7` \| `0.1 s`"* | *"`GATE-SHA`, **twice** — driver leg and arena \| `2` \| `U7` \| `0.2 s`"* | the fourth *"once"* site, and the only one carrying a numeric count column |
| 4 | `V15E1:1550` §8 Phase 1g | *"**`1`** — the arena process alone … `66 + 6 + 1 = 73` accounts for **every** process §13 declares"* | *"**`2`** — the `--gate-sha-only` driver leg and the arena … `1 + 66 + 6 + 1 = 74` accounts for every process §13 declares"* | the compute-accounting twin of the §12 defect, false in the same words |
| 5 | `V15E1:1645` §9 | *"The **arena's own startup** is **the one span** that precedes any python-side line"* | *"The **`--gate-sha-only` driver leg's** startup is the first such span and the arena's is the second; both are bounded by the same arena-class band"* | measured false — the driver leg emits its own `GATE-DET1`/`GUARD`/`GATE-SHA` heartbeat lines, so **that** span precedes the job's first python-side line |
| 6 | `V15E1:1839` §13 | *"**73** processes in the order 66 mints → 6 fidelity → 1 arena"* | *"**74** processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena"* | the declared order omits the process it names in the very next clause |
| 7 | `V15E1:1904` §13.1 item 12 | *"append-without-interleaving across all **73** processes"* | *"… across all **74** processes"* | the handoff item the code lineage checks against |
| 8 | `V15E1:2017` §13.1 item 28 | *"layer 3 is actually active in all **73** processes"* | *"… in all **74** processes"* | item 28 is the guard-activation item; the driver leg is precisely where the guard first installs |
| 9 | `sbatch:16` | *"73 processes, in this order: 66 mints -> 6 fidelity -> 1 arena"* | *"74 processes: 1 GATE-SHA driver leg -> 66 mints -> 6 fidelity -> 1 arena"* | in the file that spawns the 74th process two lines of code later |
| 10 | `config:41` | `"processes": {"mints":66,"fidelity":6,"arena":1,"total":73}` | `{"gate_sha_driver":1,"mints":66,"fidelity":6,"arena":1,"total":74}` | |
| 11 | `config:222` | `"processes_reporting": {"expected": 73, "binding": true}` | `{"expected": 74, "decomposition": "1+66+6+1", "binding": true}` | |
| 12 | `arena:465-466` | `!= 73` (twice) | `!= 74`, read from `cfg["ledger"]` per I-2 | |
| 13 | `mint:117` | *"**72 of 73** processes previously wrote nothing"* | *"**73 of 74** …"* | a docstring narrating the pre-CODE-R1 state; the ratio moves with the inventory |

### 1b. `GATE-SHA` pass-count sites

| # | site | current text | correct text | why |
|---|---|---|---|---|
| 14 | `V15E1:966` §6 `GATE-SHA` row | *"**once in the sbatch driver**"* | *"**twice** — once in the sbatch driver before any other process, and again in the arena at the point of use"*, **with the TOCTOU reason** | round-1 H-2's obligation; the reason must be recorded so the second pass is a decision, not an accident |
| 15 | `V15E1:1840` §13 | *"`GATE-SHA` **once in the driver** before any of them"* | *"`GATE-SHA` in the driver leg before any of them **and again in the arena**"* | |
| 16 | `sbatch:17` | *"GATE-SHA runs ONCE in this driver, before any of them"* | *"GATE-SHA runs in this driver before any of them, and again inside the arena"* | |
| 17 | `sbatch:62` | *"# ---- GATE-SHA, ONCE, before any other process (§6, §13)"* | *"# ---- GATE-SHA, first of two passes, before any other process (§6, §13)"* | |
| 18 | `arena:560` | docstring *"ONCE, in the driver, before any other process"* | *"the first of two passes; the arena repeats it at the point of use"* | |
| 19 | *(new)* `config` | — | `"gate_sha_passes": 2` **declared for the audit to check against**, never trusted | §3's audit mechanism needs a declared value to HALT against |

### 1c. Sites stating the other moved quantities

| # | site | current | correct | why |
|---|---|---|---|---|
| 20 | `config:218` | `"dev_path_opens": {"expected": "mints_executed", "binding": true}` | `{"expected":"mints_executed + expected_sha_dev_opens","expected_sha_dev_opens":4,"derivation":"(dev-like files in the concatenated iterable gate_sha hashes = 2) x (GATE-SHA passes = 2)","binding":true}` | the erratum's core repair |
| 21 | `arena:433`, `:438`, `:468-475` | the frozen `+ 0` predicate and its `ERRATUM REQUIRED` message | the two-term predicate; the message's decomposition **becomes** the derivation | round-1 obligation 1 |
| 22 | `config:211` | `_code_review_r1.blocked_on_erratum_2` narrative | replaced by an `erratum_2` block recording what landed | |
| 23 | `config:219` | `"dev_label_materialisations_outside_decisions": {"expected":"mints_executed","binding":true}` | moved to a `by_construction` block as a **warranted string**, not an integer | §2's criterion |
| 24 | `config:220`-ish + `arena:447` | `test_label_materialisations`, `dev_or_test_labels_into_decision_quantities` published as measured integers | same treatment — `by_construction` strings | §2's criterion, applied uniformly |
| 25 | `arena:46` + `config.projected_seconds` | `3670.0` / `4587.5` | **`3673.9` / `4592.4`** | §4's re-price; §9 pins the denominator to §8 **by name**, so both literals move together |
| 26 | `arena:1418` | `mints_executed = sum(1 for _ in [None])   # placeholder replaced below` | **delete** | my own dead line, found by this enumeration; it is overwritten on the next statement and computes nothing |

**What the enumeration did *not* find, stated so the subtraction is checkable.** No other `73`/`74`
occurs in any of the five files: the remaining `\b7[234]\b` hits are `generate_…:73-89` (a line
citation, ×3), the rotation angles `72.7` (×3), and `§8 Phase 7z`'s `7×6 + 5×6 = 72` tie-cap items —
none is a process count. `§13.1 item 12`'s *"all 73"* and item 28's *"all 73"* are rows 7 and 8.
**Row 26 is the only defect this enumeration found that no reviewer had raised**, which is the point
of doing it by grep rather than from the delta list.

---

## 2. The uniform counter criterion (H-2)

**One criterion, stated once and applied to all six:**

> **A ledger quantity is published as a measured integer on the verdict face if and only if some
> code path in this job increments it. A quantity that no code path increments is published as a
> by-construction narrative string carrying its warrant, in a separate block, and is never an
> integer and never binding.**

The second clause follows from the first: a binding comparison against a counter nothing increments
can only ever fire in the **false** direction, so it is not a tripwire — it is a guaranteed-false
test that reads like one. Round 2 confirmed this in terms (*"There was no tripwire … Demotion
removes a guaranteed-false comparison and nothing else"*).

| counter | incremented by | disposition | warrant (source-verified) |
|---|---|---|---|
| `test_path_opens` | `c09guard._guarded_open` | **measured integer, binding `0`** | — |
| `dev_path_opens` | `_guarded_open` | **measured integer, binding** at the two-term formula | — |
| `banked_trainlog_opens` | `_guarded_open` | **measured integer, reported** | — |
| `test_label_materialisations` | **nothing** | **by-construction string, not binding** | `_guarded_open` **raises** on a test path, so no test file is opened at all and no test label can be materialised |
| `dev_label_materialisations_outside_decisions` | **nothing** | **by-construction string, not binding** | `lab_dev` occurs **exactly once** in the executed corpus — `headspace_mint.py:323`, the `.npz` write — and appears in neither the arena nor `headspace_fidelity.py` |
| `dev_or_test_labels_into_decision_quantities` | **nothing** | **by-construction string, not binding** | same write, read by no decision path; the arena never iterates `.npz` keys generically (no `z.files`, no `.keys()`, no key loop) — callers name `meta`, `fold_of`, `K_train`, `K_dev`, `lab`, `fit_idx` |

**This matches C09's form literally** (`c09_a0_arena.py:1913-1953`): a `"measured"` block containing
exactly the three instrumented counters, and the other three in `"by_construction_zero"` as
narrative strings, never as integers.

**Where I go one step further than round 2 offered, and why.** Round 2 said rows 2 and 6 *"may keep
their `fails` entries as free vacuous belts if the lineage prefers"*. **Under this criterion they do
not.** A vacuous belt costs nothing to execute and something to read: it presents as a live check,
and the failure mode this whole erratum documents is a predicate that *looks* checked and is not.
Keeping two guaranteed-false comparisons in the same gate whose defect is "a predicate no one
re-derived" would be the wrong lesson. **The criterion is uniform and the reviewer may overrule it,
but it should be overruled explicitly rather than by exception.**

---

## 3. `expected_sha_dev_opens` — both factors, one audited

**First factor — 2 dev-like files, derived over the concatenated iterable `gate_sha` consumes**
(`frozen_sha256` + `frozen_sha256_input_caches` = 21 files; 2 dev-like; the 16 banked artifacts
contribute 0). Derived from the config's own digest table, so it cannot drift from it.

**Second factor — 2 passes, audited not declared.** §6 and §13 say *once*; the `2` is a fact about
`c06_falsifier_arena.py:1276`. The arena measures it:

```
gate_sha_passes = int("GATE-SHA" in self.gates) + #{ledger argv containing --gate-sha-only}
```

and **HALTs if it differs from the declared `2`**. **Round-2 I-1 adopted:** the own-process term is
`int("GATE-SHA" in self.gates)`, **not a literal `1`** — otherwise a future cleanup landing option
(iv) would still compute `1 + 1 = 2`, match the declared value, pass, and leave `dev_path_opens`
HALTing by exactly 2 with a message pointing at the wrong term. This closes both directions.

**Option (iv) — make `GATE-SHA` run once, as §6 and §13 say — considered and declined on the
record.** The arena runs ~1 h after the driver leg; re-hashing at the point of use closes a
time-of-check/time-of-use window on the frozen inputs. Real benefit, currently an implementation
accident — so this erratum keeps two passes **and writes the reason into §6** (row 14).

---

## 4. §8: re-priced, branch (a)

**Round 2 offered (a) re-price or (b) a bounded statement, and did not require (a). I choose (a).**

The reason is §8's own discipline. Phase 1g exists *because* round-11 I-1 found a per-process cost
that no row priced; leaving a second such cost knowingly unpriced — in the row created to fix the
first — is the same defect one iteration later, and §8 Phase 1g's sentence would have to be softened
from *"accounts for every process"* to a hedge. A `0.11 %` correction is cheap; the precedent of a
"bounded statement" in a projection whose whole claim is *measured unit × explicit count* is not.

| row | before | after |
|---|---|---|
| Phase 1d `GATE-SHA` | `1 × U7 = 0.1 s` | **`2 × U7 = 0.2 s`** |
| Phase 1g arena-class startups | `1 × U11 = 3.8 s` | **`2 × U11 = 7.6 s`** |
| **total** | `3670.0 s` | **`3673.9 s`** |
| `× 1.25` | `4587.5 s` | **`4592.4 s`** |
| minutes | `61.2 / 76.5` | **`61.2 / 76.5`** (unchanged at one decimal) |
| mint share | `68.3 %` | `68.3 %` |
| Phase 3 share | `27.6 %` | `27.6 %` |
| `2×` / `5×` Phase-3 miss | `4683.8 / 7725.2 s` | **`4687.7 / 7729.1 s`** (`78.1 / 128.8 min`) |

`PROJECTED_SECONDS` moves in **both** `c06_falsifier_arena.py:46` and the config (row 25), because
§9 pins the heartbeat denominator to §8 **by name**.

**One observation for the code lineage, not repaired here.** The `--gate-sha-only` leg currently
runs `load_frozen()` — importing `c01_policy_contrast_a0` and calling `import_compute_modules` —
before returning, which is what makes it a *full* arena-class startup. Returning immediately after
`gate_sha()` would make it far cheaper. Pricing it at the full startup is the conservative choice
and is what this erratum carries; the cheaper variant is a code decision with a §8 consequence and
should be taken deliberately, not as a side effect.

---

## 5. Round-2's remaining findings

**I-2 — one cross-checked copy of each expectation. Adopted.** `cfg["ledger"]` is currently read by
nothing (grep-verified): every expectation in `gate_ledger` is a code literal while the config
declares the same set independently, and `configs/c06/c06_falsifier.json` is **not** in
`frozen_sha256`, so no digest catches a divergence. v2 would have edited `73 → 74` in two files with
nothing binding them — the exact drift class §3 invokes to justify auditing rather than declaring,
applied inconsistently inside one erratum. **Repair:** `gate_ledger` asserts its expectations equal
`cfg["ledger"]`'s before evaluating, so a future edit that misses one file HALTs instead of
drifting. Covers the `74`, the `66`, the three zeros and the new two-term formula.

**I-3 — §12 row 9 is the fourth collision. Adopted.** §12 declares *"predicate coverage | re-derived
in-job"*; the only caller of `c09guard.verify_predicate` is **C09's** arena (`c09_a0_arena.py:1901`).
No C06 process calls it. v2's completeness table marked row 9 *"change: none"* on the ground that
the function **exists and is read-only** — a fact about the instrument, not about the run, which is
the same conflation that let row 5 through fifteen rounds. **Repair: call it**, as C09 does in four
copyable lines, and publish the result. §12's row is then true as written and needs no amendment.

**M-1 — fold the prose-only prescriptions into the delta. Adopted.** §12 gains M-3's sentence
(*"`GATE-LEDGER`'s guarantee covers **top-level processes only**"*) and the by-construction warrants
for the three uninstrumented rows. Round 2 is right that a code lineage working from a delta table
will not land what lives only in prose.

**M-2 — the ledger's `argv` is not an executed mint's launch argv. Recorded, not built on.**
`headspace_mint.py:287-290` overwrites `sys.argv` before `run_rac.main` and never restores it, so an
executed mint's ledger records **run_rac's** argv while a *skipped* mint records the real one.
Harmless for the pass audit (neither string contains `--gate-sha-only`, and the driver leg's own
argv is intact — measured), but the alternative route for obligation 7 (*"read the per-mint ledger
argv to distinguish executed from skipped"*) works **by accident, on `argv[0]`**. This erratum keeps
the sbatch-export route and records why the alternative must not be built on unknowingly.

**M-3 — the sbatch activates no environment. Referred, out of scope.** No `conda activate`, no
`module load`, no absolute interpreter path; five bare `python` calls relying on SLURM's
`--export=ALL` carrying an activated `HateVideo` `PATH`. Round 2 reproduced the failure —
`ModuleNotFoundError: No module named 'torch'` inside `load_frozen`, **after** GATE-SHA passed.
Fail-fast and one process cheap, but unverified in the record. **Code/resource lineage, not this
erratum** — flagged so it is not lost.

---

## 6. The defect, the archaeology, and the rejected options

Carried from v2 unchanged: the collision (§12's `+0` against `GATE-SHA`'s coverage of two `dev_seen`
input caches, `+2` per pass × 2 passes); the archaeology (latent from **v2** of the draft, frozen by
**round 4 I-5**, re-affirmed by **round 8 H-1**, zero mentions in rounds 6–9, and rounds 10–15
verifying only the fidelity-side clause — which is **true**); and the finding that **every round
that examined the `+0` verified the warrant offered for it and never re-derived the term itself**.

Options **(ii)** (hash through a non-counted path) and **(iii)** (drop the two dev caches from
`GATE-SHA`) remain rejected on the grounds v1 gave plus round 1's addition — `_guarded_open` is the
only thing that *raises* on a test path, so bypassing it for dev-side convenience removes the hard
stop on the test side of the same function.

---

## 7. Implementation delta

| file | change | size |
|---|---|---|
| `c06_falsifier_arena.py` | two-term `dev_path_opens`; `processes_reporting → 74`; `gate_sha_passes` audit with `int("GATE-SHA" in self.gates)`; three counters moved to a `by_construction` block; expectations asserted against `cfg["ledger"]`; `verify_predicate` called and published; `PROJECTED_SECONDS → 3673.9`; delete `:1418`'s dead line; docstring `:560` | **≈ 40 lines**, one method + two literals |
| `configs/c06/c06_falsifier.json` | rows 10, 11, 19, 20, 22, 23, 24, 25 of §1 | **≈ 20 lines** |
| `c06_falsifier_cpu.sbatch` | header rows 9, 16, 17; **plus obligation 7**: count executed-vs-skipped mints and export `C06_MINTS_EXECUTED` | **≈ 10 lines** |
| `c06_falsifier_mint.py` | row 13 (docstring) | 1 line |
| **V15E2** | §6, §7.2, §8 (1d, 1g, totals, risk row), §9, §12 (rows 4, 5, 8, 9 + the top-level-processes sentence + by-construction marks), §13, §13.1 items 12 and 28 | text |

---

## 8. Blindness and no-edit statement

**No battery-arm accuracy or macro-F1 was computed on any ro-derived arm.** `deployed_vote` called
zero times; no arm built; no mint run or read; no GPU, no job, no commit, no `TARGET_STATE.json`
edit. Compute: file and review reads; greps across the five artifacts for the §1 enumeration; one
dev-like enumeration over the config's digest tables under an active `c09guard`; one
`--gate-sha-only` invocation writing to a scratchpad ledger dir; arithmetic. All scratch artifacts
live in the session scratchpad; `artifacts/c06_falsifier/` was never created.

**Nothing is edited.** `C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (`8cde58aa…`), `c06_falsifier_arena.py`
(`0cdfd4f0…`), `c06_falsifier_mint.py` (`98f7b4a6…`), `configs/c06/c06_falsifier.json`
(`e2678431…`) and `c06_falsifier_cpu.sbatch` (`c3647173…`) carry their post-CODE-R1 hashes. The
arena still implements `dev_path_opens == mints_executed + 0` and `processes_reporting != 73`
exactly as frozen, failing with `ERRATUM REQUIRED`. **The battery cannot pass `GATE-LEDGER` before
this erratum lands, and could not have passed under v1 or v2 as specified either.**
