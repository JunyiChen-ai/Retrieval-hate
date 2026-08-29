# C06 `$0` falsifier — **ERRATUM 2, INDEPENDENT REVIEW — ROUND 2**

*Target:* `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL_V2.md`
(sha256 `4225bea3cc9907d38e2e3f5815448f7d0b291195523b781be33367feb132e040`)
*Supersedes as review target:* `C06_FALSIFIER_ERRATUM2_PROPOSAL.md`
(`f063c388c4afabdb7964360eda2fe1ef7d6fee611b2a287ca9817929c7f670f5` — **verified, matches** v2's
citation)
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`
(`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` — **verified, matches**)
*Prior adjudication:* `C06_FALSIFIER_ERRATUM2_REVIEW.md` — REVISE 0C/3H/3I/3M
*Reviewer:* fresh and independent; no part in the fifteen design rounds, the implementation,
Erratum 1, code-review round 1, or erratum-2 round 1. Judged from documents, repository and
execution only.

---

## VERDICT

> **REVISE — 0 Critical, 2 High, 3 Important, 3 Minor.**

**The process inventory is right and I confirm it independently: 74.** My own count off the
CODE-R1-fixed sbatch, with the fork and grandchild channels closed at source rather than by probe,
is `1 + 36 + 30 + 6 + 1 = 74`. The four-part decomposition v2 settles on is correct, its driver-leg
measurement reproduces exactly, the `+4` arithmetic and both of its factors reproduce exactly, and
`torch.load` opens each file exactly once on this build — so the first term's multiplier stands.
Every one of the prior review's eight obligations is engaged, and six are discharged at full
strength.

**What blocks GO is that v2's *completeness* is scoped to §12 while the defect is scoped to the
document.** v2 reconciles all nine §12 predicates — and then prescribes a V15E2 delta that leaves
**five more sites** asserting the number the erratum exists to correct: §7.2's *"the 73rd process,
the arena"*, §8 Phase 1g's *"`66 + 6 + 1 = 73` accounts for **every** process §13 declares"*, §8
Phase 1d's *"`GATE-SHA`, once in the driver | `1`"*, §9's *"the arena's own startup is **the one
span** that precedes any python-side line"*, and the sbatch's own header — in a file this erratum
edits anyway. Landing v2 as specified produces a V15E2 that says **74** in §12/§13 and **73** in
§7.2/§8, which is the trade the prior review's H-2 named: repairing one internal contradiction by
creating another. Third recurrence of one omission.

**And the demotion is adopted for the binding limb but not the publishing limb.** Three of the six
ledger counters are incremented by nothing in the job. v2 demotes one and leaves two binding; none
of the three is taken off the measured verdict face, so the face still publishes
`dev_label_materialisations_outside_decisions: 0` against a §12 expectation of `66`. C09 — the
precedent v2 cites for exactly this demotion — publishes all three as narrative strings and **never
as integers in its `measured` block**. Matching the precedent's form closes it; v2 stops one step
short.

---

## THE PROCESS INVENTORY — MY OWN COUNT

**74, and complete.** Derived statically from `scripts/slurm/c06_falsifier_cpu.sbatch`
(`c3647173…`, `bash -n` clean), by enumerating every `python` invocation site and multiplying by its
enclosing loop:

| sbatch line | invocation | multiplicity | n |
|---|---|---|---|
| `:63-64` | `c06_falsifier_arena.py --gate-sha-only` | straight-line | **1** |
| `:83-85` | `c06_falsifier_mint.py … --lineage N` | `2 ds × 3 seeds × 6 folds` | **36** |
| `:91-94` | `c06_falsifier_mint.py … --lineage R --train-cache` | `2 ds × 3 seeds × 5 folds` | **30** |
| `:130-132` | `headspace_fidelity.py` | `2 ds × 3 seeds` | **6** |
| `:144-146` | `c06_falsifier_arena.py` (aggregator) | straight-line | **1** |
| | | | **74** |

The mint total is asserted by the driver itself at `:99` (`test "$MINT_N" -eq 66`) and the fidelity
total at `:136`. **No other python is spawned.** Every remaining command in the file is a shell
builtin or a coreutil — `printf`, `echo`, `date`, `hostname`, `mkdir`, `ln`, `test`. In particular
the two questions worth asking explicitly both come back negative: **the heartbeat lines are written
by `printf`** (`:56-58`, `:128-129`), not by a python helper, and **the `DRIVER-DONE` line is
`printf` too** (`:149-150`). There is no aggregation step beyond the arena.

**The grandchild channel is closed at source, which is stronger than the round-1 fork probe.**

* No `subprocess`, `multiprocessing`, `Pool`, `os.system` or `joblib` anywhere on the executed path
  — grep-clean across `c06_falsifier_mint.py`, `headspace_mint.py`, `headspace_fidelity.py`,
  `c06_falsifier_arena.py`, `c01_policy_contrast_a0.py`, `mechfix_ops.py`, `mechnov_pairverify.py`.
* `headspace_mint.py:287-290` runs `run_rac.main` **in-process** with `sys.argv` patched, including
  `--num_workers 0`; and every `DataLoader` on the RAC feature path
  (`RA-HMD/Stage2/src/data_loader/rac_dataloader.py:104-110`, `:136`, `:148-151`) hard-codes
  `num_workers=0` regardless of the flag. There are no worker processes to leak, not merely no
  worker processes that report.
* `run_rac.py` imports `wandb` (`:7`) but **never calls `wandb.init`** — the only `wandb.init` in
  `src/` is `run_linear.py:119`, off this path. A `wandb` service process would have been a real
  74th-plus python process inheriting `PYTHONPATH`; it does not exist here.

**And one ledger file per process, exactly.** `c09guard._ledger_path` is deterministic in
`(SLURM_JOB_ID, pid, t0)`, so the `atexit` flush and the test-path pre-raise flush write the **same**
path; `aggregate` (`:154-181`) filters to the current `SLURM_JOB_ID` prefix, files from an earlier
attempt landing in `stale` rather than in `procs`. So `len(procs) + 1 = 73 + 1 = 74` on a clean run.

**74 is also resume-stable, which v2 does not say and which strengthens its own case.** A resumed
mint still *spawns*: `c06_falsifier_mint.py:217-220` returns early **inside** the process, after
`assert_guard_active()` and the `MINT-SKIP` heartbeat, so it still flushes a ledger. The count is
`74` on fresh, resumed and partially-resumed runs alike. This matters because the C09 block v2 cites
declines to bind its own count (`"pass": bool(… and len(procs) >= 1)`, with
`"resume_note": "a resume legitimately reports FEWER … The gate requires >= 1 reporting process, not
39."` — `c09_a0_arena.py:1927-1958`). C06 binds an exact `74` where C09 refused to bind `39`; that is
a *stronger* gate than the precedent, and it is safe **only** because of the spawn/skip split above.
The erratum should say so, since it is the difference between the two designs.

---

## WHAT I VERIFIED BY EXECUTION

**1. The driver leg, verbatim as `sbatch:63-64` invokes it** (conda `HateVideo`, real config, real
caches, scratch ledger dir, `CUDA_VISIBLE_DEVICES=""`):

```
GATE-DET1 1/1 | GUARD 1/1 | GATE-SHA 37/37 all frozen digests match | GATE-SHA-ONLY 1/1 2.8s
rc=0 ; ledger files written: 1
led_nojob_650706_850146873.json
  counts: test_path_opens 0 | dev_path_opens 2 | dev_label_materialisations_outside_decisions 0
          banked_trainlog_opens 0 | dev_or_test_labels_into_decision_quantities 0
  argv[-1] = --gate-sha-only
  dev_paths: HateMM/dev_seen_…-curric_HF.pt , MHC_zh/dev_seen_…-LoRA_HF.pt
```

**Every cell of v2 §2's driver row reproduces**: one ledger file, `argv[-1] = --gate-sha-only`,
`dev_path_opens = 2`, `test_path_opens = 0`. And v2 §1's "I searched for a fourth" claim checks out
on this leg specifically — the driver process touches no test path, materialises nothing, and leaves
`banked_trainlog_opens` at `0`.

**2. Both factors of `expected_sha_dev_opens`, over the right iterable** (I-2 as adopted):

```
files in the concatenated iterable gate_sha hashes (frozen_sha256 + input_caches): 21
dev-like among them: 2
subset frozen_sha256_input_caches alone: 8 files, 2 dev-like
the 16 banked artifacts: 0 dev-like
```

`frozen_sha256` is 13 module/config paths, none dev-like; so the two derivations agree today and
v2's reason for preferring the concatenated one is correct rather than cosmetic. The banked-artifact
row also confirms v2's M-2 disposition: the existence-check deviation cannot move this arithmetic.

**3. `torch.load` — one `builtins.open` per file, confirmed on this build.** Called exactly as
`headspace_mint.load_split` calls it (`map_location="cpu"`, `weights_only=False`), under an active
`c09guard` with a second spy on `builtins.open`:

```
torch 2.6.0+cu124 | python 3.11.8 | both dev caches zipfile-format (PK header)
HateMM dev_seen   dev_delta=1  builtins.open=1
MHC_zh dev_seen   dev_delta=1  builtins.open=1
HateMM train      dev_delta=0  builtins.open=1
```

**4. `headspace_mint.py:198-199` is unconditional**, both `load_split` calls preceding the
`if a.fold >= 0:` branch at `:218` — so every executed mint contributes `+1`, both lineages, all
folds. `mints_executed` is the right variable with the right multiplier, as v2 carries forward.

**5. The `lab_dev` warrant, at source.** `lab_dev` occurs **exactly once** in the entire executed
corpus — `headspace_mint.py:323`, the `.npz` write. It appears nowhere in `c06_falsifier_arena.py`
or `headspace_fidelity.py`, and the arena never iterates `.npz` keys generically (no `z.files`, no
`.keys()`, no key loop): `load_mint` returns the handle and callers name `meta`, `fold_of`,
`K_train`, `K_dev`, `lab`, `fit_idx`. **Row 6's by-construction warrant holds, and so does the
demotion's.**

**6. The audit mechanism's inputs exist.** `aggregate` records each process's `argv`
(`c09guard.py:129`), so `#{ledger files whose argv contains --gate-sha-only}` is computable with no
edit to the sha-frozen guard — confirmed by the ledger file above.

**7. Process integrity: clean.** All five artifacts carry their post-CODE-R1 hashes, matching the
implementation record's table at `:324-328` exactly — arena `0cdfd4f0…`, mint `98f7b4a6…`, config
`e26784319…`, sbatch `c3647173…`, V15E1 `8cde58aa…`. The arena still implements
`dev_path_opens == mints_executed` and `processes_reporting != 73` exactly as frozen
(`c06_falsifier_arena.py:463-475`), failing with `ERRATUM REQUIRED` in the message. Nothing is
edited. My own compute: `sha256sum`; file reads; one `--gate-sha-only` invocation, which returns at
`arena:1278-1280` **before** any battery computation; a metadata-level `torch.load` of two dev caches
and one train cache under an open-counting guard; greps. `deployed_vote` called zero times, no arm
built, no mint read or run, no GPU, no job, no commit. Scratch artifacts live in the session
scratchpad; `artifacts/c06_falsifier/` was never created.

---

## DISPOSITION AUDIT OF THE PRIOR REVIEW — LIMB LEVEL

| finding | limb | disposition in v2 | strength |
|---|---|---|---|
| **H-1** | `processes_reporting → 74` with the decomposition written down | §1 row 8, §2's four-part table, §7 delta | **full** |
| | *not* by suppressing the driver's ledger | §2 closing paragraph, named as option (ii) under another name | **full** |
| | §12's row **and §13's sentence** | §7 delta names both | **full for §12/§13; incomplete document-wide — see H-1 below** |
| **H-2** | the `2` is a fact about code, not a static derivation | §0 and §3 concede it explicitly | **full** |
| | amend §6 and §13's *"once"* in the same erratum | §3's closing paragraph + §7 delta | **full for §6/§13; §8 Phase 1d missed — see H-1** |
| | audit the factor from ledger `argv`, HALT on mismatch | §3 second factor | **substantially — but the `+1` stays a literal, see I-1** |
| **H-3** | resolve the binding-but-uninstrumented predicate | §1 row 5 + §7 delta: demote to reported | **binding limb full** |
| | correct §4.1's "correct as it stands" | §0 retracts it in terms | **full** |
| | *(implicit)* stop the face publishing a number no code produces | not addressed | **not adopted — see H-2 below** |
| **I-1** | consider option (iv) explicitly, record why declined | §3's option-(iv) block, with the TOCTOU reason and the instruction to write it into §6 | **full** |
| **I-2** | derive over the concatenated iterable | §3, with the measurement | **full** |
| **I-3** | make `mints_executed` measured | §7: sbatch counts executed-vs-skipped and exports `C06_MINTS_EXECUTED`, ≈6 lines | **full** — and I confirm the premise: repo-wide, `C06_MINTS_EXECUTED` has exactly one hit, `arena:1419` |
| **M-1** | refresh line citations | `sbatch:63-64` ✓ (verified), `arena:1276` ✓ (verified — the only `gate_sha()` call site in the arena) | **full** |
| **M-2** | existence-check ≠ hashing | recorded, referred to the code lineage, arithmetic impact shown null | **full** |
| **M-3** | ledger blind to forked children | recorded; the sentence prescribed for §12 | **full in substance** — though the §7 delta row does not list it (M-1 below), and the bound is tighter than stated: there are no workers at all on this path |

**Everything v1 got right is preserved**: option (i)'s form, the rejections of (ii) and (iii) with
the reviewer's added `_guarded_open`-raises ground, the archaeology (now with round 3), the C09
precedent, the drift-proof derivation of the dev-like count from the digest table rather than a
literal, and a bounded delta. The `torch.load` settlement is carried with its build particulars.

---

## FINDINGS

### H-1. The delta is complete for §12 and incomplete for the document. Five more sites still assert the omitted process away — including two in §8, whose projection §9 pins by name.

*Attaches to:* v2 §7's V15E2 delta row (*"§12's rows 4, 5, 8; §13's "73" → "74"…; §6's and §13's
"once" → "twice, and why"; §9's process-count statements"*).

I grepped V15E1 for every process-count assertion, as the completeness obligation requires of the
delta and not only of §12. Five survive v2's prescription:

1. **`V15E1:1197-1199` (§7.2).** *"No separate interpreter line is added for those 72 processes …
   **The 73rd process, the arena, is a different case and is priced separately at §8 Phase 1g**."*
   At 74 the `72` is still right (66 mints + 6 fidelity), but the arena is the **74th**, and the
   `--gate-sha-only` process is a **third class** — its interpreter cost is inside no mint unit, is
   not inside `U9`, and is not Phase 1g.
2. **`V15E1:1550` (§8 Phase 1g).** *"**`1`** — the arena process alone … `66 + 6 + 1 = 73` accounts
   for **every** process §13 declares."* This is the compute-accounting twin of the §12 defect, and
   it is false in the same words.
3. **`V15E1:1547` (§8 Phase 1d).** *"`GATE-SHA`, once in the driver | `1` | `U7` | `0.1 s`."* The
   *"once → twice"* amendment has a **fourth** site, and unlike §6 and §13 this one carries a
   numeric count column.
4. **`V15E1:1643-1651` (§9).** *"The **arena's own startup** is **the one span** that precedes any
   python-side line … and the bash driver's unbuffered echo brackets it."* Measured false: the
   `--gate-sha-only` process emits its own `GATE-DET1` / `GUARD` / `GATE-SHA` / `GATE-SHA-ONLY`
   heartbeat lines, so it has a startup span of its own and it is **that** span, not the arena's,
   that precedes the job's first python-side line.
5. **`c06_falsifier_cpu.sbatch:16-17`.** *"73 processes, in this order: 66 mints -> 6 fidelity -> 1
   arena / GATE-SHA runs ONCE in this driver, before any of them."* In the file v2 edits anyway for
   obligation 7, and the file from which the 74th process is spawned two lines of code later.

**Why this is High rather than Minor.** v2's own §1 exists because the lineage was caught twice by
one omission surfacing one predicate at a time, and its remedy was to enumerate exhaustively. The
enumeration was scoped to §12. A V15E2 landed as prescribed asserts **74** in §12 and §13 and
**73** in §7.2 and §8 — the trade the prior review's H-2 refused (*"repairing a §11/§6/§12
contradiction by creating a §12/§6/§13 one is the wrong trade, particularly in a document whose
defect record is 'the predicate was frozen and never re-derived after the sections it depends on
changed'"*). It also matters that §9 states its heartbeat denominator is *"pinned to §8 by name, so
it tracks automatically"*: leaving §8's accounting at 73 pins the denominator to a projection that
provably does not cover every process.

**Numerically the gap is negligible and I do not require a re-price.** The unpriced 74th process is
one arena-class startup — §7.7's measured band is `3.094–3.717 s` over 35 runs, and my
`--gate-sha-only` leg reported `2.8 s` to its last line — plus one extra `U7` at `0.1 s`. Against
`3670.0 s` that is `~0.1 %`, immaterial to §9's `~15 s` interval claim and to the `× 1.25` bound.

**Repair — either branch, but explicitly.** (a) Re-price §8 Phase 1d to `2 × U7` and Phase 1g to
`2` arena-class startups, and update `PROJECTED_SECONDS` in **both** `c06_falsifier_arena.py:46` and
the config, since §9 pins the denominator by name; or (b) leave the numbers and replace §8 Phase
1g's *"accounts for every process §13 declares"* with a bounded statement — the 74th process's
startup is knowingly unpriced, is bounded by the same arena-class band, and is `~0.1 %` of the
total. Either way §7.2's *"73rd process"*, §9's *"the one span"* and the sbatch header must be
corrected with the rest.

### H-2. The demotion stops the HALT but not the publication. Three of six counters are incremented by nothing; v2 demotes one, leaves two binding, and takes none of the three off the measured face — which is not the form of the C09 precedent it cites.

*Attaches to:* v2 §1 rows 2, 5, 6; §7's delta (*"demote … to reported with the by-construction
warrant recorded on the face"*); `c06_falsifier_arena.py:447-467`;
`configs/c06/c06_falsifier.json::ledger`.

Verified at source: `c09guard._guarded_open` increments **three** counters — `test_path_opens`,
`dev_path_opens`, `banked_trainlog_opens`. The other three — `test_label_materialisations`,
`dev_label_materialisations_outside_decisions`, `dev_or_test_labels_into_decision_quantities` — are
initialised at `c09guard.py:41-45` and incremented by nothing in this job (repo-wide, the middle key
appears only there and in **C09's** arena at `c09_a0_arena.py:1944`).

v2 handles the three asymmetrically, and the asymmetry is defensible as far as it goes: rows 2 and 6
expect **zero**, so a dead counter satisfies them vacuously, while row 5 expected **66**, which a
dead counter can never produce. That is a real distinction and it justifies demoting row 5 and only
row 5. **What it does not justify is what the verdict face publishes.** `gate_ledger` does
`self.ledger = dict(tot)` at `:447`, so all six keys reach the face as integers. After the demotion
the face carries `dev_label_materialisations_outside_decisions: 0` while §12 declares
`mints_executed` — which is, verbatim, the defect the prior review's H-3 raised (*"any downstream
reader comparing the two finds a contradiction with no gate having fired"*). Demoting the predicate
removes the gate that would have fired; it does not remove the contradiction. And rows 2 and 6 stay
**binding integers that no code produces**, which is CODE-R1 H-2's original complaint in two more
rows.

**The cited precedent gets this right and is worth copying literally.** In
`c09_a0_arena.py:1913-1953`, C09's `GATE_LEDGER` face has a `"measured"` block containing exactly the
three **instrumented** counters, and puts the other three in `"by_construction_zero"` as narrative
strings — never as integers. That is why C09's face cannot be misread.

**Repair.** Publish the three uninstrumented quantities the way C09 does: as warranted strings in a
`by_construction` block, removed from the measured integer block; and say in §12, on the face of the
table, that rows 2, 5 and 6 are by-construction rather than measured. Rows 2 and 6 may keep their
`fails` entries as free vacuous belts if the lineage prefers — the guarantee they nominally carry is
real and I verified both warrants (row 2 rests on `_guarded_open` **raising**; row 6 on `lab_dev`
being written at `headspace_mint.py:323` and read by nothing) — but they must not appear on the face
as measured integers.

**On the team lead's question — does the demotion remove a tripwire?** **No.** There was no
tripwire: the counter was never incremented, so the binding predicate could only ever have fired in
the *false* direction. Demotion removes a guaranteed-false comparison and nothing else. The
guarantee itself is carried by row 6 plus the source facts above, both of which I verified
independently.

### I-1. The audit's `+1` is itself a trusted literal, so it closes the drift case in one direction only.

*Attaches to:* v2 §3's `gate_sha_processes = 1 + #{ledger files whose argv contains --gate-sha-only}`.

The mechanism does close the case the review asked about, and I confirm it: a future sbatch that
adds a second `--gate-sha-only` leg makes the count `3`, which differs from the declared `2`, and
HALTs. Good.

But the `1` is the arena's **own** pass, asserted rather than observed. If a later cleanup lands
option (iv) — the option v2 itself keeps on the table by name — and drops `arena:1276`, the formula
still returns `1 + 1 = 2`, matches the declared `2`, and **passes**, while the real dev term is
`mints_executed + 2`. `dev_path_opens` then HALTs by exactly `2` with a message that points at the
wrong term. Fail-safe in outcome, misleading in diagnosis, and it reintroduces the trusted literal
the audit exists to remove — one line lower down.

**Repair (one line).** The arena knows whether it ran the pass: `gate_sha()` sets
`self.gates["GATE-SHA"]` and `self.reports["gate_sha_artifacts"]`. Derive the own-process term as
`int("GATE-SHA" in self.gates)` rather than a literal `1`. That closes both directions and keeps the
declared `2` as the cross-check v2 intends.

### I-2. The arena reads none of the config's `ledger` block, so v2's delta creates two hand-maintained copies of `74` with nothing binding them.

*Attaches to:* v2 §7's first two delta rows; `c06_falsifier_arena.py:455-475`;
`configs/c06/c06_falsifier.json::ledger`.

Grep-verified: `cfg["ledger"]` is never read. Every expectation in `gate_ledger` is a literal in
code — `!= 0` three times, `!= 66`, `!= 73`, and `!= mints_executed` — while the config declares the
same set independently. The config's `ledger` block is pure declaration that no code consumes.

v2's delta therefore edits `73 → 74` **twice**, in two files, with no mechanism tying them. That is
precisely the drift class §3 invokes to justify auditing `gate_sha_processes` instead of declaring
it, applied inconsistently within the same erratum. Note also that `configs/c06/c06_falsifier.json`
is **not** in `frozen_sha256` (13 entries, verified), so no digest catches a divergence either.

**Repair.** One copy, cross-checked: have `gate_ledger` read its expectations from
`self.cfg["ledger"]`, or — cheaper and sufficient — assert the config's declared values equal the
arena's before evaluating, so a future edit to one that misses the other HALTs instead of drifting.
The same argument covers the `66` and the three `0`s.

### I-3. §12's row 9 is unimplemented, and the completeness table certifies it by checking that the instrument *exists* rather than that it *runs*.

*Attaches to:* v2 §1 row 9 (*"`predicate coverage` | re-derived in-job | §12 |
`c09guard.verify_predicate` exists and is read-only | no (reported) | **none***"); `V15E1:1815`.

§12 declares *"predicate coverage | re-derived in-job | reported"*. Repo-wide, the only caller of
`c09guard.verify_predicate` is **C09's** arena, `c09_a0_arena.py:1901`, which publishes the result
as `predicate_coverage_measured_this_run`. `c06_falsifier_arena.py` never calls it. So nothing in
C06 re-derives the coverage claim in-job, and the verdict face will carry no coverage evidence at
all.

This is the row the completeness obligation was built to catch, and the table's own reasoning is
where it slips: the *measured* column records that the function **exists and is read-only**, which
is a fact about the instrument, not about the run. That is the same conflation — an instrument
present, a measurement absent — that let row 5 through fifteen rounds and one erratum. v2 asserts a
search for a fourth collision and marks row 9 *"change: none"*; row 9 is the fourth.

**Repair.** Either call it (C09's block at `:1901-1911` is four lines and copyable, and
`verify_predicate` walks the tree read-only), or amend §12's row to state that the coverage claim is
carried by the freeze record rather than re-derived in-job. It must not stay a declared in-job
derivation that no process performs.

### M-1. v2 prescribes two §12 edits in prose that its own delta row omits.

§7's M-3 paragraph rules that *"`GATE-LEDGER`'s guarantee should be stated as covering **top-level
processes only**, and that sentence belongs in §12"*, and §1's prose supplies the by-construction
warrants for rows 2 and 6 — but the delta row lists only *"§12's rows 4, 5, 8"*. A code lineage
working from the delta table will not land either. Fold them into the row.

### M-2. The ledger's `argv` is not the process's launch argv for an executed mint.

`headspace_mint.py:287-290` overwrites `sys.argv` with `["run_rac.py", …]` before `run_rac.main` and
never restores it, so an executed mint's ledger records **run_rac's** argv while a *skipped* mint —
which returns at `c06_falsifier_mint.py:217-220`, before the patch — records the real one. Harmless
for v2's audit (neither string contains `--gate-sha-only`, and I verified the driver leg records its
own argv intact), but load-bearing for whoever discharges obligation 7: the prior review offered
*"have the arena read the per-mint ledger `argv` records, which distinguish an executed mint from a
skipped one"* as the alternative. That alternative happens to work — by accident, on `argv[0]` — and
should not be built on without knowing why. v2's chosen route (the sbatch export) is unaffected.

### M-3. Out of scope for this erratum, flagged for the code lineage: the sbatch activates no environment.

`c06_falsifier_cpu.sbatch` contains no `conda activate`, no `module load` and no absolute
interpreter path; it calls bare `python` at five sites and relies on SLURM's default `--export=ALL`
carrying an already-activated `HateVideo` `PATH` from the submitting shell. I reproduced the failure
mode by running the driver leg under a bare interpreter: `ModuleNotFoundError: No module named
'torch'` at `c01_policy_contrast_a0.py:1051`, inside `load_frozen`, **after** GATE-SHA had already
passed. `CODE_REVIEW_R1` has zero mentions of conda,
`PATH` or interpreter resolution. The failure is fail-fast and costs one process, so it is cheap;
but it is unverified in the record and belongs in the code/resource lineage's checklist, not in this
erratum.

---

## OBLIGATIONS FOR A V2.1 THAT WOULD CARRY GO

1. **Extend the delta to every process-count and pass-count site in the document** (H-1):
   `V15E1:1197-1199` (§7.2), `:1550` (§8 Phase 1g), `:1547` (§8 Phase 1d, the fourth *"once"*),
   `:1641-1648` (§9's *"the one span"*), and `c06_falsifier_cpu.sbatch:16-17`. State explicitly
   whether §8 is re-priced; if it is, `PROJECTED_SECONDS` moves in **both**
   `c06_falsifier_arena.py:46` and the config, because §9 pins the denominator to §8 by name. If it
   is not, §8 Phase 1g's *"accounts for every process"* becomes a bounded statement naming the
   unpriced startup and its `~0.1 %` share.
2. **Match the C09 precedent's form for all three uninstrumented counters** (H-2): publish them as
   warranted strings in a by-construction block, out of the measured integer block, and mark rows 2,
   5 and 6 by-construction on the face of §12's table.
3. **Derive the audit's own-process term** (I-1): `int("GATE-SHA" in self.gates)`, not a literal `1`.
4. **One cross-checked copy of each expectation** (I-2): the arena reads `cfg["ledger"]`, or asserts
   equality against it, before evaluating.
5. **Resolve §12 row 9** (I-3): call `c09guard.verify_predicate` as C09 does, or amend the row.
6. **Fold M-3's top-level-processes sentence and rows 2/6's warrants into the delta row** (M-1).
7. **Carry forward, unchanged and at full strength**, everything already discharged: option (i)'s
   two-term form with the dev-like count derived over the concatenated iterable; the `74`
   decomposition at §12/§13/§9/config/arena/heartbeat; §6 and §13's *"once → twice"* with the TOCTOU
   reason recorded; option (iv) declined on the record; the `argv`-based `gate_sha_processes` audit
   with its HALT; the sbatch's executed-mint counter and `C06_MINTS_EXECUTED` export; the refreshed
   citations; and the M-2 referral.

Obligations 1, 2 and 5 are design-lineage; 3, 4 and 6 may be discharged by the code lineage under the
amended §12. The delta stays bounded: obligations 3 and 4 are ~5 lines between them, obligation 2 is
a restructuring of one dict, and the rest is text.

---

## WHAT V2 STILL GETS WRONG — SUMMARY PARAGRAPH

v2 is a substantially better document than v1 and it is right about the things that decide the
erratum: the `+4` and both its factors, the `74` and its four-part decomposition, the demotion's
target, the audit-over-declaration principle, and the rejections of (ii), (iii) and (iv) — all of
which I reproduced or verified at source. Its error is that it drew the completeness boundary around
**§12** when the defect's boundary is the **document**. Having been caught twice by one omitted
process surfacing one predicate at a time, it enumerated every predicate and then prescribed a delta
that leaves §7.2 calling the arena *"the 73rd process"*, §8 Phase 1g asserting that `66 + 6 + 1 = 73`
*"accounts for every process §13 declares"*, §8 Phase 1d pricing `GATE-SHA` at one pass, §9 calling
the arena's startup *"the one span that precedes any python-side line"*, and the sbatch's own header
announcing 73 processes and a single `GATE-SHA` — the last of these in the file that spawns the
seventy-fourth process two lines later. The same table also certifies a row it has not actually
checked: §12 promises predicate coverage *"re-derived in-job"*, the table marks it *"change: none"*
on the ground that `verify_predicate` **exists**, and no C06 process **calls** it — instrument
present, measurement absent, which is the exact shape of the defect the table was built to prevent.
And the demotion, whose warrant I independently confirm is sound, is adopted for the limb that stops
the HALT and not for the limb that makes the face honest: three of six counters are incremented by
nothing, v2 demotes one, leaves two binding, and lets all three reach the verdict face as measured
integers — where the C09 block it cites for the demotion publishes the same three as narrative
strings and no integer at all. None of this is a wrong-verdict path and none of it touches the
arithmetic. It is the difference between an erratum that ends this defect and a third one.

---

## BLINDNESS AND EDIT STATEMENT

No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm. `deployed_vote` was called
zero times; no arm was built; no mint was run or read; no GPU, no SLURM job, no commit, no
`TARGET_STATE.json` edit. Compute used: `sha256sum`; file reads; `bash -n` and static analysis of
the sbatch; one `--gate-sha-only` invocation of the arena under a scratch `C09_LEDGER_DIR`, which
returns at `arena:1278-1280` before any battery computation; a metadata-level `torch.load` of the two
banked `dev_seen` caches and one banked `train` cache under an open-counting `c09guard`; one
enumeration of the config's digest tables under `is_dev_like`; and greps. All scratch artifacts were
written to the session scratchpad; `artifacts/c06_falsifier/` was never created. **No file outside
this review was edited** — `C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (`8cde58aa…`),
`c06_falsifier_arena.py` (`0cdfd4f0…`), `c06_falsifier_mint.py` (`98f7b4a6…`),
`configs/c06/c06_falsifier.json` (`e2678431…`) and `c06_falsifier_cpu.sbatch` (`c3647173…`) all
still carry the hashes recorded in the implementation record's CODE-R1 table.
