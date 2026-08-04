# C06 `$0` falsifier — **ERRATUM 2, INDEPENDENT REVIEW**

*Target:* `refine-logs/C06_FALSIFIER_ERRATUM2_PROPOSAL.md`
(sha256 `f063c388c4afabdb7964360eda2fe1ef7d6fee611b2a287ca9817929c7f670f5`)
*Against:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`
(sha256 `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` — **verified, matches**)
*Reviewer:* fresh and independent; no part in the fifteen design rounds, the implementation,
Erratum 1, or code-review round 1. Judged from documents, repository and execution only.

---

## VERDICT

> **REVISE — 0 Critical, 3 High, 3 Important, 3 Minor.**

**Option (i) is the right repair and its arithmetic is correct.** I reproduced the `+4` by
execution, and the one number the proposer asked to have measured — `torch.load`'s
`builtins.open` count per file — comes back **exactly 1 on this build**, so the first term's
multiplier is right and `mints_executed` is the right variable. The rejections of options (ii)
and (iii) are both sound and code-anchored. The archaeology is accurate. Process integrity is
clean.

**What blocks GO is not the repair but its scope.** The proposal's closing claim — *"The battery
cannot pass `GATE-LEDGER` until this erratum lands"* — reads as *and can once it does*. It cannot.
**`processes_reporting` is a second binding predicate in the same gate, broken by the same root
cause, and this erratum does not touch it.** I measured a clean run at **74** reporting processes
against a binding **73**. Landing erratum 2 exactly as specified leaves `GATE-LEDGER` HALTing on a
clean run, one gate later, after the full 73-process battery has been spent. The GATE-SHA driver
process is invisible to §12 in *two* places; the proposal found one.

---

## THE LOAD-BEARING NUMBER

**`torch.load` performs exactly ONE `builtins.open` per file on this build.** Measured under an
active `c09guard`, with a second `builtins.open` spy recording every call and its mode:

```
torch 2.6.0+cu124 | python 3.11.8 | all three caches are zipfile-format checkpoints

HateMM dev_seen          dev_path_opens delta = 1 ; builtins.open calls = 1
      open('…/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt', 'rb')
MHC_zh dev_seen          dev_path_opens delta = 1 ; builtins.open calls = 1
      open('…/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt', 'rb')
HateMM train (native)    dev_path_opens delta = 0 ; builtins.open calls = 1
```

Called exactly as `headspace_mint.load_split:160-161` calls it (`map_location="cpu"`,
`weights_only=False`, through `_ORIG_TORCH_LOAD`). The concern that motivated the request — that
zipfile-format checkpoints commonly get a header probe plus a zip-reader open — does **not**
materialise here: `torch.load` resolves the path to a file **object** once via `_open_file_like`
and hands that object to the zipfile reader, so the path is opened once and the reader never sees
it again. **The first term's multiplier is 1, not 2 or 3.**

**And `mints_executed` is the right variable, not `6`.** The premise floated in the review request
— that only the 6 full mints load dev and the 60 fold mints do not — is **false at source**.
`headspace_mint.py:198-199` performs both `load_split` calls **unconditionally**, before the
`if a.fold >= 0:` branch at `:218`; the fold branch then overwrites `dev_sp` with a slice of the
fitting pool, but the native `dev_seen` has already been opened. Those two lines are the *only*
`load_split` call sites in the module (grep-verified). The Head-R override
(`c06_falsifier_mint.py:230-238`) fires only on `split == "train"`; `"dev_seen"` falls through to
the frozen loader, so both lineages open the native cache. **Every executed mint contributes
exactly +1, both lineages, all folds.** §12's *"Dev labels"* paragraph states this correctly and
is confirmed.

So the first term needs **no** static-derivation treatment. The proposer was right to flag it and
right about its value.

---

## WHAT I VERIFIED AND CONFIRMED

**1. The defect, reproduced.** Executing the arena's own `sha256_of` over exactly the iterable
`gate_sha` hashes (`frozen_sha256` + `frozen_sha256_input_caches`), with `c09guard.install()`
active:

```
files hashed by gate_sha's loop: 21
digest mismatches: NONE (all 21 match)
dev_path_opens incurred by ONE GATE-SHA pass: 2
  …/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt
  …/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt
```

§11's input-cache table carries **8** entries, of which exactly **2** match `is_dev_like`
(basename contains `dev_seen`) — confirmed against
`configs/c06/c06_falsifier.json::frozen_sha256_input_caches`. Then I ran the driver's first
process verbatim as `c06_falsifier_cpu.sbatch:63-64` invokes it
(`--gate-sha-only`, real config, real caches, scratch ledger dir):

```
GATE-SHA | 37/37 | all frozen digests match
led_999999_348554_849013600.json  counts: {… 'dev_path_opens': 2 …}
```

Then, with 66 synthetic mint ledgers (+1 dev each, per the measurement above) and 6 fidelity
ledgers, I called the arena's **actual** `gate_ledger` after a **real** in-process `gate_sha()`:

```
dev_path_opens  measured = 70   (frozen predicate wants 66)
under the PROPOSED erratum (mints_executed + 4) -> SATISFIED
```

**The `+4` is real, is exactly `2 dev-like files × 2 GATE-SHA processes`, and option (i) settles
it.** The proposal's core claim survives execution intact.

**2. The archaeology is accurate.** Every row checks out against the actual files:

* **v1** (`C06_FALSIFIER_PREREG_DRAFT.md:656-667`) — the input-cache table is headed *"train split
  only"* and carries eight `train_*` rows, **zero** `dev_seen`. No collision.
* **v2** — `:908` and `:910` add the two `dev_seen` digests; `:410`'s `GATE-SHA` row already reads
  *"every frozen import and input cache matches §11"*, and `:138` already carries §3.1's warrant
  (*"…is listed in §11 and covered by `GATE-SHA`"*). **Latent from v2, as claimed.**
* **round 4, I-5** (`R4:437`, `:446`) — heading verbatim *"§12's `dev_path_opens` is **binding**
  with an unquantified term. Measured, that term is `0`"*; repair verbatim *"Write the term:
  `dev_path_opens == mints_executed + 0`"*. `R4:114` confirms v4's wording was *"`mints_executed` +
  `GATE-DEVFID` reads"*. **Frozen in round 4, as claimed.**
* **rounds 6–9** — `dev_path_opens` occurs **0** times in each. Confirmed by count.
* **rounds 10–15** — 1, 2, 1, 1, 1, 1 occurrences; each verifies the fidelity-side clause.
  Round 14's `:481-482` is the clearest instance: *"the `dev_path_opens == mints_executed` choice is
  correct and would otherwise HALT a legitimate resume"* — the resume warrant re-checked, the
  second term never re-derived.
* **round 8, H-1** (`R8:175`) — reason confirmed verbatim: §11 asserted a `GATE-SHA` scope §6 did
  not carry, leaving `GATE-FLOOR`'s anchors and `GATE-FOLD`'s parity files unverified.

One omission, immaterial: the table skips **round 3**, which is where `dev_path_opens` first
appears (`R3:229`, `:247` — H-2 correcting the Head-N/Head-R count from 36 to 66). It does not
change the account.

**3. Option (ii)'s rejection is correct.** All three grounds hold. `c09guard.py:28-30` states the
instrument's purpose in exactly the words quoted (*"the arena aggregates them into `GATE-LEDGER`,
so the ledger reports **MEASURED** opens rather than literals"*). The directionality point is real:
`is_dev_like` matches on `"dev_seen" in basename`, so it does cover `dev_seen_*-ro_*`, and §3.1
states *"no `dev_seen_*-ro_*` file is opened by any phase"* — a guarantee that a guard-invisible
`sha256_of` would silently stop certifying. I add one ground the proposal does not make: hashing
through `_ORIG_OPEN` would also make the **`test`** limb blind on the same call path, and
`_guarded_open` is the only thing that *raises* on a test path. Bypassing it for convenience on the
dev side removes the hard stop on the test side of the same function. **(ii) is worse than the
proposal argues.**

**4. Option (iii)'s rejection is correct.** §3.1 (`V15E1:180-182`) says in terms that the native
`dev_seen` *"is opened by `headspace_mint.py:199` on every mint and is covered by `GATE-SHA`"* —
verified at source, and the file is opened by all 66 mints. Dropping the two digests falsifies that
sentence and leaves a file entering 66 processes unverified. Round-8 H-1's widening principle is as
characterised.

**5. The C09 precedent is as characterised — and I verified the quotes.**
`C09_A0_DECISION.json::GATE_LEDGER.measured_expectations` reads:

```
dev_path_opens: "36 on a fresh run (one dev_seen load per mint); LOWER on a resume…"
banked_trainlog_opens: "6 …; design 8.1 counts these separately from the 36,
                        for a declared dev-side total of 42"
declared_dev_side_total_8_1: 42
```

and `C09_A0_V17_RECORD.md:1549-1553` and `:376-379` match the proposal's quotations verbatim.
**The house pattern is declare-and-expect with the decomposition written down, not prohibition.**
Option (i) is squarely in it.

**6. Process integrity: clean.** All five artifacts carry their post-CODE-R1 hashes, matching the
implementation record's table at `:324-328` exactly — arena `0cdfd4f0…`, mint `98f7b4a6…`, config
`e2678431…`, sbatch `c3647173…`, V15E1 `8cde58aa…`. Nothing was edited for this defect. The arena
does implement `dev_path_opens == mints_executed + 0` exactly as frozen and fails with `ERRATUM
REQUIRED` in the message (`c06_falsifier_arena.py:468-475`) — left failing rather than adjusted, as
claimed. No decision quantity, no arm, no accuracy appears anywhere in the proposal's work, and
none in mine.

---

## FINDINGS

### H-1. The same root cause breaks a **second** binding §12 predicate, and this erratum does not fix it. A clean run measures `processes_reporting == 74` against a binding `73`.

*Attaches to:* §12's *"processes reporting"* row; §13's *"73 processes"* sentence;
`configs/c06/c06_falsifier.json::ledger.processes_reporting`;
`c06_falsifier_arena.py:449`, `:465-467`; proposal §5's closing sentence.

`GATE-SHA` runs in its own **python process** (`sbatch:63-64`, `--gate-sha-only`). The sbatch
exports `PYTHONPATH` (`:37`) and `C09_LEDGER_DIR` (`:42`) **before** that call, so the frozen
`sitecustomize` installs the guard in it and its `atexit` `_flush` writes a ledger file — which
`c09guard.aggregate` then counts. Measured, not inferred:

```
ledger files after the driver's GATE-SHA process alone:            1
ledger files present when the arena aggregates:  73   (1 gate-sha-only + 66 mints + 6 fidelity)
processes_reporting measured = 74   (config binds 73)
  FAIL: processes_reporting = 74 != 73
```

The arena computes `len(procs) + 1`, adding itself. So the true clean-run count is
**1 GATE-SHA + 66 mints + 6 fidelity + 1 arena = 74**, and §12/§13/the config all say 73.

I ruled out the obvious inflation channel by measurement: forked children do **not** report.
Four `DataLoader` workers plus three `multiprocessing.Process` children wrote **0** ledger files
(they exit via `os._exit`, skipping `atexit`); only the parent wrote one. **74 is exact.**

**This is not a coincidence of two defects — it is one omission surfacing twice.** §13 declares
*"73 processes in the order 66 mints → 6 fidelity → 1 arena, with `GATE-SHA` once in the driver
before any of them"*: the GATE-SHA pass is described as something the *driver* does, not as a
process, so it contributes neither to the process count nor to the dev-open count. It does both.

**The corpus already contains the contradiction, twice, unnoticed.** Round 14 (`R14:486`) wrote
*"`66 + 6 + 1 = 73`. **I found no seventy-fourth python process.**"* — while the same bullet's own
quotation of §13 names GATE-SHA running before all of them. Code-review round 1 (`CODE-R1:694-698`)
then enumerated, under the heading **"73 processes in §13's order"**, the list *"`1 ×
--gate-sha-only`, then 66 mints …, then 6 fidelity, then 1 arena"* — **74 items counted as 73**,
and the very next line correctly notes that `--gate-sha-only` hashes 37 artifacts. Two independent
lineages held both halves and neither summed them.

**Consequence if this erratum lands as specified.** `dev_path_opens` passes; `processes_reporting`
HALTs. The HALT lands in the arena, i.e. after all 66 mints and 6 fidelity processes have run —
the whole battery spent, no verdict, and a third erratum required. The proposal's §5 assertion
that *"the battery cannot pass `GATE-LEDGER` until this erratum lands"* is true but misleading by
omission; the accurate statement is *"…and still cannot after."*

**Repair (belongs in this erratum, not the next).** §12's process row and §13's sentence become
**74 = 1 GATE-SHA + 66 mints + 6 fidelity + 1 arena**, with the decomposition written down exactly
as C09 wrote its own — and note that C09's `evidence.n_processes_expected_breakdown` is
*"**1 version/preflight heredoc** + 36 mints + 2 GATE-DEVFID runs"*: **the precedent the proposal
invokes counts its driver-side helper process.** The answer to the collision this proposal leaves
behind was in the same JSON block it quoted from.

Do **not** repair this by unsetting `C09_LEDGER_DIR` for the GATE-SHA process — that is option
(ii)'s laundering under a different name, and it would discard the counts of the process that
performs the largest new read surface.

### H-2. The second factor is **not** statically derivable from the frozen design. §6 and §13 both say `GATE-SHA` runs **once**; the `2` is a fact about code, and landing the erratum as written makes §12 contradict §6 and §13.

*Attaches to:* proposal §2 option (i) bullet 2, §3 supporting point 2, §4.1's config row, §4.2's
replacement text.

The proposal's warrant for `gate_sha_processes = 2` is *"the process count from §13's declared
73-process order"*, and it calls both factors *"readable from frozen artifacts … derived, not
measured-then-blessed. That distinction is what keeps it an audit rather than a rubber stamp."*

Applying that derivation to the frozen documents yields **1, not 2**:

* §6's row: *"`GATE-SHA` | G | every frozen import, input cache and the sixteen banked artifacts of
  §11 … **once in the sbatch driver**"*
* §13: *"…with `GATE-SHA` **once in the driver** before any of them"*

The `2` comes from `c06_falsifier_arena.py:1276`, where `main()` calls `bat.gate_sha()`
unconditionally on **every** invocation, so the arena repeats the pass the driver already made.
That is a property of the executable, not of §13. The first factor is genuinely derived (the
config's own digest table); the second is a **hand-maintained literal whose ground truth lives in
code and which nothing checks**.

Two consequences, both live:

1. **The erratum would introduce a fresh internal contradiction.** §4.2's replacement §12 text
   asserts *"`GATE-SHA` runs in **two** processes"* while §3.1, §6 and §11 are explicitly left
   unchanged — and §6 and §13 say *once*. Repairing a §11/§6/§12 contradiction by creating a
   §12/§6/§13 contradiction is the wrong trade, particularly in a document whose defect record is
   *"the predicate was frozen and never re-derived after the sections it depends on changed."*
2. **The literal can go stale in the self-defeating direction.** If any later cleanup makes the
   code obey §6's *"once"* — a plausible reading of the frozen text — the config's `2` over-expects
   by 2 and `GATE-LEDGER` HALTs on a clean run. Fail-safe in direction, self-defeating in effect,
   and exactly the class §12 says this lineage *"has removed twice elsewhere."*

**Repair.** Whichever process count is adopted, **§6's and §13's `GATE-SHA` clauses must be
amended in the same erratum** so all four sections state the same thing. And make the factor
audited rather than declared: the aggregated ledger already carries each process's `argv`, so the
arena can measure `gate_sha_processes = 1 + #{ledger files whose argv contains --gate-sha-only}`
and **HALT if it differs from the declared 2**. That is ~3 lines, needs no edit to the sha-frozen
`c09guard.py`, and converts the one remaining literal into a cross-checked measurement — which is
the principle CODE-R1's H-2 was raised to enforce.

### H-3. §4.1's claim that the rest of the H-2 wiring "is correct as it stands" is false. A third §12 predicate is declared **binding** and is incremented by nothing in the job.

*Attaches to:* §12's `dev_label_materialisations_outside_decisions` row;
`configs/c06/c06_falsifier.json::ledger.dev_label_materialisations_outside_decisions`
(`"expected": "mints_executed"`, `"binding": true`); `c06_falsifier_arena.py:455-476`;
proposal §4.1's closing paragraph.

The proposal bounds its delta with: *"the H-2 wiring already landed … evaluates each §12 predicate
as a pass-condition … is correct as it stands — this erratum changes one comparison."* Verified
against source, that is not so:

* `gate_ledger`'s `fails` list evaluates **six** conditions: `test_path_opens`,
  `test_label_materialisations`, `dev_or_test_labels_into_decision_quantities`,
  `mints_present_before_arena`, `processes_reporting`, `dev_path_opens`.
* `dev_label_materialisations_outside_decisions` — **binding in §12 and in the config** — is not
  among them.
* Nothing in the job increments it. Repo-wide grep: the key appears only in `c09guard.py:43`
  (initialised to `0`) and in `c09_a0_arena.py:1944` (**C09's** arena, not C06's). `_guarded_open`
  increments only the test, dev-path and trainlog counters.

So on a real clean run the aggregate is **0**, against a declared expectation of **66**. Today it
does not HALT only because the predicate is silently omitted — which is precisely CODE-R1 H-2's
original complaint (*"a published CLOSE will carry … literals no code produced"*) surviving in one
row. The verdict face would publish `0` while §12 declares `mints_executed`, and any downstream
reader comparing the two finds a contradiction with no gate having fired.

Note the contrast with the precedent: C09 handled this quantity in a `by_construction_zero` block
with a narrative warrant (*"dev labels are materialised once per mint inside `headspace_mint.py`
(36 mints) … the arena never loads a dev label"*), **not** as a binding integer. C06 promoted it to
a binding count without building an instrument that can produce it.

**Repair.** Either demote the row to *reported / by construction* with C09's warrant written out —
`headspace_mint.py:322-324` writes `lab_dev` into every `.npz`, which is the materialisation, and
no `open()`-level instrument can see it — or build a counter for it. It must not stay a binding
integer that nothing measures, and §4.1's bound on the delta must be corrected before a reviewer
relies on it.

### I-1. The option space is under-enumerated: "make `GATE-SHA` actually run once, as §6 and §13 say" was never considered.

Options (i)–(iii) all take *two GATE-SHA passes* as given. A fourth option exists and is arguably
the most conservative one available: drop the arena's redundant `gate_sha()` call (`:1276`) so the
gate runs once as the frozen design already states, making the dev term `mints_executed + 2` and
leaving §6 and §13 true as written.

I do **not** recommend it, and the reason should be stated rather than left implicit: the arena
runs roughly an hour after the driver's pass, and re-hashing at the moment of use closes a
time-of-check/time-of-use window on the frozen inputs that a single up-front pass leaves open. That
is a genuine provenance benefit and it argues for keeping two passes — but it is currently an
accident of implementation, not a design decision, and §6's *"once in the sbatch driver"* actively
contradicts it. **Whichever way the design lineage rules, the reason belongs in §6.**

### I-2. Derive the dev-like count over exactly the iterable `gate_sha` hashes, not over `frozen_sha256_input_caches` alone.

`gate_sha` (`:562-563`) hashes `frozen_sha256` **+** `frozen_sha256_input_caches` — 21 files.
§4.1 specifies the derivation over `frozen_sha256_input_caches` only (8 files). Today the two agree,
because no entry in `frozen_sha256` is dev-like (verified: 13 module/config paths, none matching
`is_dev_like`), and the sixteen banked artifacts are not dev-like either (verified). But the
proposal's whole warrant for this factor is that it *"cannot drift from the digest table"*, and a
derivation over a strict subset of what the loop hashes can drift from it. Deriving over the same
concatenated iterable the loop consumes costs nothing and makes the claim true.

### I-3. `mints_executed` is not measured. `C06_MINTS_EXECUTED` is never exported, so the erratum's **first** term is a literal in disguise and §12's resume-safety rationale is unimplemented.

`c06_falsifier_arena.py:1419-1420`:

```python
mints_executed = int(os.environ.get("C06_MINTS_EXECUTED",
                                    bat.reports["mints_present_before_arena"]))
```

Repo-wide grep for `C06_MINTS_EXECUTED` returns **exactly one hit — that line**. The sbatch never
sets it, and never counts executed-versus-skipped mints. So `mints_executed` always falls back to
`mints_present_before_arena`, which is a count of `.npz` files **present**, not mints **executed**.

On a fresh run the two coincide (66) and nothing is wrong. On a **resumed** run — the case §12
spends four sentences justifying — the skipped mints open no dev file
(`c06_falsifier_mint.py:218-220` returns before `HM.main()`), so measured dev opens fall while
`mints_executed` stays pinned at 66, and the predicate HALTs. That is the precise failure §12 says
it avoided: *"A binding `dev_path_opens == 66` would HALT a legitimate resume — the same class of
self-defeating gate this lineage has removed twice elsewhere."* As wired, the predicate **is**
`== 66`.

This is code-lineage territory and I raise it as Important rather than High for that reason — but
it is the erratum's own first term, the erratum is the moment §12's predicate is rewritten, and the
proposal's §4.2 text re-asserts the resume warrant while the mechanism behind it does not exist.
The honest fix is small: have the sbatch count executions and export the count (or have the arena
read the per-mint ledger `argv` records, which distinguish an executed mint from a skipped one).
C09 avoided the problem by not binding an exact integer at all — its declared expectation is *"36 on
a fresh run …; **LOWER on a resume**"*.

### M-1. Stale line citations.

The proposal cites the driver's GATE-SHA call as `c06_falsifier_cpu.sbatch:57`; in the current
sbatch (`c3647173…`, the hash the proposal itself certifies as unchanged) that line is a `printf`
and the call is at **`:63-64`**. The `:57` is CODE-R1's number, from before the H-2 wiring added the
`C09_LEDGER_DIR` export. Likewise the arena's second pass is at `:1276`, not CODE-R1's `:1038`.
Non-substantive, but this lineage's convention is that a quoted line number resolves in the file
whose hash is quoted beside it.

### M-2. `gate_sha` **existence-checks** the sixteen banked artifacts; it does not hash them.

`:571-583` calls `os.path.exists` and increments `n`, computing no digest and comparing nothing —
so §6's row (*"…and the sixteen banked artifacts of §11 matches its §11 digest"*) and round-8 H-1's
whole point are met only for presence. The config records this deviation openly
(`_gate_sha_count.banked_unhashed_artifacts: 16`, note: *"whose presence GATE-SHA asserts"*), so it
is declared rather than hidden. It does not affect this erratum's arithmetic — none of the sixteen
is dev-like, verified — but the proposal describes *"§11's GATE-SHA scope"* as though all 37 were
hashed, and a reader deriving the dev-like count from that description rather than from the config
would be reasoning about a scope the code does not implement. Flagged for the code lineage, not for
this erratum.

### M-3. The ledger is blind to anything a forked child opens.

Measured above: `DataLoader` workers and `multiprocessing` children skip `atexit` and write no
ledger file, so their in-memory counts are discarded. `src/run_rac.py:192` defaults
`--num_workers` to 24. The exposure is small in practice — the mint's loaders operate on in-memory
tensors, and a *test* open would raise inside the worker and propagate — but a **dev** open inside a
worker would be silently uncounted, and `GATE-LEDGER`'s guarantee should be stated as covering
top-level processes only. Informational; bounds what the instrument certifies.

---

## OBLIGATIONS FOR A V15E2 THAT WOULD CARRY GO

1. **Land option (i)** as specified for `dev_path_opens`: `mints_executed + expected_sha_dev_opens`,
   `expected_sha_dev_opens = (dev-like files in the iterable `gate_sha` hashes) × (GATE-SHA
   processes) = 2 × 2 = 4`. The arithmetic is verified; the reasoning is sound; the C09 precedent
   supports it.
2. **Fold `processes_reporting` into the same erratum** (H-1): §12's row and §13's sentence become
   **74**, with the decomposition `1 GATE-SHA + 66 mints + 6 fidelity + 1 arena` written down, and
   the config's `73` updated. Not by suppressing the GATE-SHA process's ledger.
3. **Amend §6 and §13's *"once in the sbatch driver"*** (H-2) so that the number of GATE-SHA passes
   is stated identically in §6, §12 and §13 — and state the reason for two passes (re-verification
   at the point of use, ~1 h after the driver's pass) rather than leaving it an implementation
   accident. Consider option (iv) explicitly and record why it was declined (I-1).
4. **Audit the process factor instead of declaring it** (H-2): derive
   `gate_sha_processes = 1 + #{ledger argv containing --gate-sha-only}` from the aggregated ledger
   and HALT if it differs from the declared value. ~3 lines, no frozen-module edit.
5. **Resolve `dev_label_materialisations_outside_decisions`** (H-3): demote to reported/by-construction
   with C09's warrant, or instrument it. It must not remain a binding integer no code produces — and
   §4.1's claim that every other predicate is already wired as a pass-condition must be corrected.
6. **Derive the dev-like count over the same iterable `gate_sha` hashes** (I-2).
7. **Make `mints_executed` measured, or stop binding it exactly** (I-3): export a real executed
   count from the sbatch, or adopt C09's non-exact form. As wired the predicate is `== 66` and will
   HALT a legitimate resume.
8. **Refresh the line citations** to resolve in the hashed files (M-1).

Obligations 1–5 are design-lineage; 6–8 may be discharged by the code lineage under the amended
§12. The implementation delta remains bounded and the sbatch remains untouched under all of them
except 7, which adds an export and a counter.

---

## WHAT THE PROPOSAL GOT WRONG — SUMMARY PARAGRAPH

The proposal's substantive analysis is right: the `+4` is real, option (i) is the correct repair,
and options (ii) and (iii) are correctly rejected on grounds I verified independently. Its error is
one of scope and of one overstated claim. It diagnosed the omission — that §12 never accounts for
the `GATE-SHA` driver process — and then fixed only the first of the two binding predicates that
omission breaks, leaving `processes_reporting == 73` to HALT a clean run at 74 after the entire
battery has been spent; and it did so while quoting the very C09 block whose process breakdown
(*"1 version/preflight heredoc + 36 mints + 2 GATE-DEVFID runs"*) shows the house pattern counting
exactly that class of process. It also claims both factors of the new expectation are statically
derivable from the frozen tables when only one is: §6 and §13 say `GATE-SHA` runs **once**, so the
factor `2` is read off the code, and the proposed §12 text would assert *two* against two sections
left explicitly unchanged saying *once*. Finally, §4.1's assurance that the rest of the H-2 wiring
"is correct as it stands" does not survive checking — `dev_label_materialisations_outside_decisions`
is binding in both §12 and the config, is incremented by nothing in the job, and is not evaluated,
so it is the same literal-published-as-measurement defect H-2 existed to close. The one thing the
proposer most wanted measured is the one thing that came back clean: `torch.load` opens each file
exactly once on this build, every executed mint opens the native `dev_seen` exactly once regardless
of fold or lineage, and `mints_executed` is therefore the right variable with the right multiplier.
The first term is sound. It is the accounting *around* it that is still one process short.

---

## BLINDNESS AND EDIT STATEMENT

No battery-arm accuracy or macro-F1 was computed on any `ro`-derived arm. `deployed_vote` was
called zero times; no arm was built; no mint was run or read; no GPU, no job, no commit. Compute
used: `sha256sum`; file reads; `torch.load` of the two banked `dev_seen` caches and one banked
`train` cache at metadata level under an open-counting guard; one `--gate-sha-only` invocation of
the arena (the permitted `sha256_of` path, which returns before any arena computation); one
in-process `gate_sha()` + `gate_ledger()` evaluation against synthetic ledger files; and a
fork-behaviour probe. All scratch artifacts (`led`, `led2`, `led3`, `prog*.txt`) were written to the
session scratchpad; `artifacts/c06_falsifier/` was never touched. **No file outside this review was
edited** — design, arena, mint, config and sbatch all still carry the hashes recorded in the
implementation record's CODE-R1 table.
