# C06 `$0` falsifier — CODE/RESOURCE REVIEW, ROUND 2 (final pre-freeze verification)

*Reviewer:* independent code/resource lineage, round 2. No part in any prior round; judged from
the documents and the repository only.
*Date:* 2026-08-05.
*Mandate:* the last review before freeze + submission. Block **only** on defects that could
produce a wrong published verdict, a HALT on a clean run, or a resource / test-contact violation.
Everything else is recorded as a non-blocking note.

---

## VERDICT: **NO-GO**

**0 wrong-verdict defects. 2 HALT-on-a-clean-run defects. 0 resource or test-contact violations.**

| severity | count | items |
|---|---|---|
| **C — blocking** | **2** | C-1 `GATE-COMPLETENESS` HALTs every clean run; C-2 `sbatch:70` makes `processes_reporting = 75` and `GATE-LEDGER` HALTs every clean run |
| H | 0 | — |
| I — informational | 3 | I-1 `I-4`'s *"the bound is exact"* is overstated; I-2 `M-10`'s runtime provenance never reaches the verdict; I-3 `M-6`'s consistency assertion is vacuous |
| M — minor | 4 | M-1 `load_ro` docstring; M-2 one stale denominator in `V15E2`; M-3 the dry-check protocol note omits `--progress`; M-4 the sbatch trailer conflates two HALT exit codes |

Both blocking defects fire **at the very end of the battery** — C-1 after S1–S7 and both Holm
families, C-2 in the last gate before the emitter. A clean run therefore spends all 74 processes and
the full ~61-minute projection and then publishes `HALT / INSTRUMENT_INCONCLUSIVE` with **no decision
content**: `emit_halt` writes `gates`, `dropped_lineages`, `ledger` and `scope`, and **not**
`decision`, `per_dataset`, `lineages_passed`, `gate_domain` or `gate_devfid`. The run would have to
be repeated in full. Neither defect can produce a *wrong* verdict — both are fail-closed — but both
sit squarely inside the blocking class the mandate names.

Everything else in scope verified clean, including the four legs' substantive content: Erratum 2's
six obligations, the five carried CODE-R1 items, the two disclosures, the resource envelope, and —
on four end-to-end synthetic drives of the real arena — the arm builder, the twenty gates, S1–S7, the
92-hypothesis Holm family, the shuffle null, the drop-and-continue path and the verdict emitter.

---

## The two blocking defects

### C-1 — `GATE-COMPLETENESS` HALTs every clean run: 7 of the 20 gate names are never written

`c06_falsifier_arena.py:1581-1587` requires all twenty declared gate names on the verdict face:

```python
want = set(bat.cfg["gates"]["global"] + bat.cfg["gates"]["per_lineage"]
           + bat.cfg["gates"]["reporting"])
missing = sorted(want - set(bat.gates))
if missing:
    raise GateFailure("GATE-COMPLETENESS", "verdict face missing gates: {}".format(missing))
```

Enumerating every write into that dict — `c06_falsifier_arena.py` lines 554, 642, 687, 730, 829,
834, 907, 1049, 1149, 1464, 1465, 1480, 1566, 1574 — gives **thirteen** names. Seven are never
written by any code path:

> `GATE-ALGEBRA`, `GATE-ARENA`, `GATE-C01PARITY`, `GATE-NESTED`, `GATE-ORBITDISP`, `GATE-RHORAW`,
> `GATE-ZEROOP`

Five of the six per-lineage gates (all but `GATE-SELFTEST`) record their results only by appending to
`drop_reasons`, and `GATE-C01PARITY` / `GATE-RHORAW` report only through the heartbeat
(`:813`, `:845`) — neither touches `self.gates`. The set difference is therefore non-empty on
**every** path that reaches the emitter, including the maximally clean one.

**Measured, end to end.** World 1 below, run unpatched with all sixty-six mints present, every gate
it evaluates passing and both lineages passing their per-lineage gates:

```
2026-08-04T16:44:55 | PER-LINEAGE-GATES | 2/2 | ... | passed=[]
2026-08-04T16:44:55 | S4 | 92/92 | ... | hatemm Holm family frozen at 92
2026-08-04T16:44:55 | S4 | 92/92 | ... | zh Holm family frozen at 92
2026-08-04T16:44:55 | HALT | - | 47.1s | INSTRUMENT_INCONCLUSIVE | gate=GATE-COMPLETENESS |
      verdict face missing gates: ['GATE-ALGEBRA', 'GATE-ARENA', 'GATE-C01PARITY',
      'GATE-NESTED', 'GATE-ORBITDISP', 'GATE-RHORAW', 'GATE-ZEROOP']
[drive] arena rc=2
```

The `gates` dict on that HALT face carried exactly the thirteen writable names.

*Minimal repair (not applied):* give each of the seven a write at the point where it is decided —
the two global ones next to their existing heartbeat call, the five per-lineage ones set once per
lineage from `drop_reasons` (`"PASS"` / `"FAILED (lineage dropped)"`). Note the per-lineage gates are
evaluated **twice**, once per lineage, so the face needs a per-lineage value or a documented
aggregation rule; that is a design question the seven blank names have so far concealed.

### C-2 — the sbatch's `projected_seconds` helper is a 75th ledger-writing process

`scripts/slurm/c06_falsifier_cpu.sbatch:40` exports `PYTHONPATH` with the `c09_guard` directory and
`:45` exports `C09_LEDGER_DIR`. Line 70 then runs an inline helper:

```bash
C06_PROJECTED_SECONDS="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['execution']['projected_seconds'])" "$CONFIG")"
```

That interpreter imports `sitecustomize` → `c09guard.install()` → `atexit.register(_flush)`, and
`_ledger_path()` resolves because `C09_LEDGER_DIR` is already exported. It therefore writes a ledger
file. Verified directly:

```
ledger files written by that one helper process:
led_TEST70_3990980_862981867.json
  led_TEST70_3990980_862981867.json | argv= ['-c', '.../c06_falsifier.json'] | dev_opens= 0
```

`gate_ledger` computes `processes_reporting = len(procs) + 1` (`:505`) and binds it to the config's
declared `74` (`"decomposition": "1+66+6+1"`). A clean run leaves **74** ledger files — the helper,
the `--gate-sha-only` leg, 66 mints, 6 fidelity — so the arena publishes **75**.

**Measured, end to end**, on an otherwise-clean run with C-1 patched out:

```
[drive] fabricated 74 ledger files, mints_executed=66
2026-08-04T17:04:42 | HALT | - | 67.5s | INSTRUMENT_INCONCLUSIVE | gate=GATE-LEDGER |
      processes_reporting = 75 != 74 (1+66+6+1)
[drive] arena rc=2
```

The helper does **not** disturb the other two predicates: its `argv` carries no `--gate-sha-only`,
so `GATE_SHA_PASSES` stays 2 and `expected_sha_dev_opens` derives to 4; and it opens no dev-like
path, so `dev_path_opens` stays `mints_executed + 4`. Only the process count breaks — and it is
binding.

This process was **introduced by Erratum 2 itself**: §7's single-source clause is what put the
`python -c` on line 70, in the same landing that moved `processes_reporting` from 73 to 74. The
count was raised for the `--gate-sha-only` leg and the new helper was not counted.

*Minimal repair (not applied):* either declare 75 with the decomposition `1+1+66+6+1` (and update
§13's process order and §8 Phase 1g's `count = 2`, which prices interpreter startup for the non-mint
side and likewise omits this process), or keep 74 by removing the extra interpreter — e.g. read
`projected_seconds` with the same `python` that already runs the `--gate-sha-only` leg and have it
print the value, or export it from a non-python reader. Whichever is chosen, §8 Phase 1g's count and
§13's "74 processes" both move with it.

---

## LEG 1 — Erratum-2 landing fidelity

**H-1, the re-multiplied total — verified by independent re-multiplication.** I summed §8's printed
product column myself, row by row, without reference to the stated total:

```
753.8 + 632.6 + 1121.9 + 8.0 + 2.2 + 0.3 + 0.1 + 1.3 + 7.6 + 4.1 + 0.1 + 11.2 + 2.5
     + 9.0 + 9.3 + 22.5 + 4.8 + 38.4 + 1013.8 + 7.0 + 0.0 + 21.6 + 1.0 + 0.7 + 0.1 + 0.1
= 3674.0
```

Exact. Every derived figure re-derived and confirmed: residue `3674.0 − 1031.5 = 2642.5`;
`× 1.25 = 4592.5`; `61.233 → 61.2` and `76.54 → 76.5` minutes; 2× Phase-3 miss `4687.8 = 78.1 min`;
5× miss `7729.2 = 128.8 min`; `7729.2 / 4592.5 = 1.6831 → 1.68`; mint share
`2508.3 / 3674.0 = 68.271 % → 68.3`; Phase 3 `1013.8 / 3674.0 = 27.593 % → 27.6`. Phase 1d's
`2 × 0.13 = 0.26 → 0.3` is the correct re-multiplication and is the value carried in the row.

**H-2, resume-safe `MINTS_EXECUTED` — bit-exact semantics confirmed.** `sbatch:100` and `:109`
increment on `[ -f "$OUT" ] ||`, i.e. the `.npz` absent at call time. `c06_falsifier_mint.py:225`
tests `os.path.exists(a.out)` and returns at `MINT-SKIP` (`:226-227`) **before** the frozen
`HM.main()` and therefore before `headspace_mint.py:199`'s `dev_seen` load. The two conditions test
the same predicate on the same path, and the loop is serial with `set -e`, so no third party can
create or remove the file between the shell test and the interpreter's test. The counter counts
executed mints, never attempts. `MINT_N` correctly stays at 66 and is correctly **not** exported.

A resume is also safe on the ledger axis: `c09guard.aggregate` (`:157-168`) sums only files whose
name carries the current `SLURM_JOB_ID` and reports the previous attempt's counts under `stale`.
Skipped mints still start an interpreter and still write a ledger file, so `processes_reporting`
stays constant across a resume while `dev_path_opens` falls with `mints_executed` — exactly what the
two-term predicate needs. (C-2 shifts that constant to 75; it does not break the resume logic.)

**The design-digest parity gate — falsifier-tested by me.** Scratchpad copy of the config with
`design_sha256` set to 64 zeros:

```
GATE-DET1  1/1 | GUARD 1/1 | PROJECTION 1/1
HALT | INSTRUMENT_INCONCLUSIVE | gate=GATE-SHA | design document
   refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md on-disk digest 254c0547… != declared 0000…
exit code = 2
```

Fires **before** `load_frozen`, i.e. before any C01 module is imported or any compute happens. The
HALT artifact carries `sha256_declared` **and** `sha256_derived` plus the external-anchor note.

**The denominator three-way assertion — falsifier-tested by me, both directions.**

```
config drifted : HALT | gate=GATE-DET1 | projected_seconds disagree:
                        config 9999.0 / module 3674.0 / env 3674.0      exit 2
env drifted    : HALT | gate=GATE-DET1 | projected_seconds disagree:
                        config 3674.0 / module 9999.0 / env 9999.0      exit 2
```

Fires in the pre-`gate_sha_only` block, before `GATE-SHA`, and `sha256_derived` correctly reads
`NOT_DERIVED` on that path. Undrifted control: `GATE-SHA 38/38 … all frozen digests match`, exit 0 —
so all 38 declared digests match the tree as it stands.

**Own stale-figure sweep over the five landed artifacts** (`3670.0`, `3667.4`, `3668.1`, `2933.9`,
`2934.5`, `2930.x`, `2929.9`, `2642.3`, `273.7`, `11.6 s`, `85.6`, `9.3 %`, `3.2 s`, bare `73`):
every hit sits inside an explicitly historical or provenance clause, with one exception recorded at
**M-2** below. No live figure is stale.

---

## LEG 2 — the carried CODE-R1 items

**I-4 (union tie-ranking) — mechanism correct, exactness claim overstated. See I-1.**
The narrow claim the LANDED doc makes *is* sound: guard arms retrieve to depth `2 × 21 = 42`
(`:978-984`), so a union member outside an arm's top-42 is outranked by ≥ 41 items in that arm. And
as the brief anticipated, the `≥ 41` bound is far stronger than needed — `vote_bounds_over_orderings`
builds `w_full = [20..1]` for ranks 0–19 and zero-fills the rest (`:303-305`), so **rank ≥ 20 already
carries weight 0**. The claim is valid but not tight.

I also confirmed the restriction preserves the *nominal* weights exactly: `_union_ranked` masks in
descending order, and an arm's own top-21 is a subset of `keep_ids` by construction, so positions
0–20 of the restricted list are that arm's true ranks 0–20 and carry their true weights.

Where the claim over-reaches is a *different* omission: `_union_ranked` also drops the entries at an
arm's own ranks 21–41 that are **not** union members. Those carry weight 0 nominally, but dropping
them changes the *gap structure* on which `vote_bounds_over_orderings` cuts its near-tie groups
(`:309-313`), so an admissible ordering available on the full 42-list can be unavailable on the
restricted one. Constructed counterexample, run against the deployed function:

```
full 42-list      lo=-0.864365 hi=-0.857703
rank-21 dropped   lo=-0.864365 hi=-0.864363
bounds identical  : False
```

So the bound is **not exact** in general; it is exact under the nominal ordering and conservative
otherwise. Non-blocking, on three independent grounds: the perturbation is bounded by the similarity
window, itself bounded by `GATE-ALGEBRA`'s bar (`2e-6 × √2048 = 9.05e-5`); the error direction
under-counts casualties, which can only make a lineage *more* likely to drop, never more likely to
publish `SURVIVE`; and the path runs at all only when two algebraically identical arms disagree.

The path **was** exercised in the drives — `halt/hatemm/R` seed 1 raised 1 casualty against 1
mismatch and `drop/zh/R` seed 0 raised 2 against 2; in both, `outside = 0` and no drop was appended.

**M-6 — landed; assertion is vacuous. See I-3.** `cell["selected_control"]` (`:1107-1108`) is a
second call to `C01.select_strongest_ordinary_control` with the *same* `evaluations` dict and the
*same* `gain_controls` list as the call that produced `cell["reference"]` (`:1030`). The function is
pure, so the `:1299` comparison cannot fail. §13.1 item 24 required the assertion to exist and it
does; it is not an independent check.

**M-7 — landed and complete.** `finite()` guards `net_s` (`:1047`), S4's `lower` /
`one_sided_raw_p` (`:1217-1218`), S5's `observed` / `p95` / `one_sided_raw_p` (`:1284-1285`) and
S7's `fixed_fraction` / `threshold` (`:1350`), in addition to `acc_s` / `mf1_s` (`:1023`). The S4
call sits inside the `TESTED` branch only, so a `NOT_TESTED` record's `lower = None` is never passed
to it — correct, and see the note on `None` in Leg 4.

**M-10 — half-landed. See I-2.** `headspace_mint` and `vsw_pregate` are imported and
`HM.runtime_block()` is called and stored at `self.reports["runtime"]` (`:569-575`). But `reports`
is never published: the emitted dict (`:1589-1617`) has no runtime key, and neither does
`emit_halt`. My schema audit confirms the verdict's top-level keys are exactly
`verdict, design, gates, dropped_lineages, lineages_passed, per_dataset, decision, gate_domain,
gate_devfid, ledger, scope`. The LANDED doc's *"so the verdict carries runtime provenance"* is not
true of the artifact. Informational — no verdict depends on it.

A worthwhile side effect of M-10's import that is nowhere recorded: importing `headspace_mint`
installs its module-level `torch.load = _guarded_torch_load` (`headspace_mint.py:106-116`) into the
arena process, so layer 1 of the test guard is live for `load_ro`'s direct `torch.load` call. That
is load-bearing given M-1 below, and it depends on `load_frozen` running before `load_ro` — which it
does on both the full path and `--dry-parity-only`.

**M-12 — memoisation scope is sound; the C-1 ghost is unreachable.** `load_ro` (`:734-762`) keys
`self.ro_cache` on `ds` alone, and that is complete: the cached value holds **both** policies
(`"standard"` / `"oneword"`) under their own sub-keys, the path is built from
`PV.DATASETS[ds]["cache_dir"]` and `["model"]` — both functions of `ds` — and the split is the
string literal `"train"`, with a `basename().startswith("train_")` assertion. Nothing lineage-varying
enters the key or the value, and correctly so: the ro caches are **inputs**, identical for Head-N and
Head-R; only the head keys (read from the mints, keyed on the full quadruple at `:698`) vary by
lineage. The C-1 ghost required a *loader override* to be live at call time; `load_ro` never routes
through `HM.load_split`, so there is no override to be stale. Confirmed at run time: all four drives
ran both lineages against one memoised ro per dataset with `GATE-IDPARITY`, `GATE-C01PARITY`,
`GATE-ROWSUBSET` and `GATE-RHORAW` passing.

---

## LEG 3 — the two disclosures

**(a) Five heartbeat lines — confirmed at source and by execution.** The `--gate-sha-only` leg emits
`GATE-DET1` (`:1416`), `GUARD` (`:1418`), `PROJECTION` (`:1430`), `GATE-SHA` (`:689`, inside
`gate_sha`) and `GATE-SHA-ONLY` (`:1438`). Executed against the real config:

```
GATE-DET1 1/1 | GUARD 1/1 | PROJECTION 1/1 | GATE-SHA 38/38 | GATE-SHA-ONLY 1/1     (rc 0)
```

Five. Grep over `V15E2` for any line-count claim returns nothing — the design carries no such number,
so no design text asserts four. The one nearby count, `mint.py:121`'s *"72 of 74 processes previously
wrote nothing"*, is consistent with §9's *"68 of the 74"* appenders and with round-7 I-3's correction.

**(b) The stray artifact — absent now, and the protocol note landed.** `artifacts/c06_falsifier/`
does not exist (checked at the start of this review, again after all three falsifier tests, and again
after all five drives; `git status` shows no `artifacts/` entry). The note landed at
`C06_FALSIFIER_ERRATUM2_LANDED.md:188-191`. See **M-3** for its one gap.

---

## LEG 4 — end-to-end synthetic verdict drive

**Method.** I generated synthetic mint `.npz` files in the scratchpad matching
`headspace_mint.py:321-325`'s schema plus the driver's `h_std` / `h_ow`, with the real C06 geometry:
real labels and ids from the ro caches, the real frozen `StratifiedKFold(5, shuffle=True,
random_state=0)` assignment, arena populations 743 / 579 after the frozen zero-row removal, 1024-d
keys, nonzero displacements, `fold_parity_vs_banked_vsw_ckpt = [True]×5`, 3 seeds × 5 folds for both
lineages plus Head-N's six full mints — 66 per world.

`GATE-FLOOR` is the one leg whose expectations are pinned to real banked numbers, so rather than stub
it I made the synthetic `K_train` **reproduce it exactly**: keys cluster by `g_i = lab_i XOR err_i`,
the per-fold error counts are recovered from the banked `fold_acc_deployed` integer grid, the class
split of the errors is chosen to hit `mf1_deployed`, and each fold's matrix is accepted only when its
own OOF predictions equal `g[ho]` item-wise — which is how `gate_floor` actually votes (each fold
with its *own* mint's `K_train`). All six `(ds, seed)` cells reproduce `per_fold`, `acc` and `mf1` at
4 dp. `GATE-FLOOR` therefore ran for real, banked comparison included.

I also fabricated a realistic `C09_LEDGER_DIR` for the 73 non-arena processes so
`c09guard.aggregate` and the whole Erratum-2 predicate set ran against realistic inputs rather than
being bypassed.

**Two disclosed overrides.** (i) `n_id_hash_permutations` 256 → 8/16/96, applied *after*
`load_frozen` so `GATE-SHA` still hashes the untouched `configs/c01/c01_a0_v2.json`; it reduces the
S5 shuffle-null draw count only. `n_bootstrap` was left at 2000 — reducing it would make S4's Holm
threshold unreachable and no `SURVIVE` could be produced. (ii) `--patch-gates` pre-seeds the seven
names of **C-1** so the drive can get past `GATE-COMPLETENESS` and exercise everything downstream.
Neither real config on disk was modified; both are switchable flags in the harness.

**Outcomes — four drives, all reaching the emitter, one published state each.**

| world | shape | lineages passed | clears | verdict | rc |
|---|---|---|---|---|---|
| **CLOSE** | displacement uninformative (`h_ow − h_std` = an isotropic perturbation small against `‖h_std‖`) | N ✓ R ✓ | none, all four cells | **CLOSE** | 0 |
| **SURVIVE** | class-dependent offset in `h_ow` only, so `l2(h_ow) − l2(h_std)` isolates the class direction while both endpoints keep the full per-item noise | N ✓ R ✓ | `common_displacement`, both datasets, both lineages | **SURVIVE** | 0 |
| **HALT** | degenerate orbit both lineages (all keys ≈ one direction) | N ✗ R ✗ (11/11/10/10 drop reasons) | — | **HALT** | 0 |
| **DROP** | R degenerate, N survive-shaped | N ✓ **R ✗** | `common_displacement` (hatemm), `displacement`+`common_displacement` (zh) | **SURVIVE** | 0 |

In the HALT world the drops were raised by `GATE-ORBITDISP` (`ρ_head > ρ*` with `ρ_raw ≤ ρ*`) and
`GATE-ARENA` lower; the verdict came out of §5.6 rule 3 through the **normal** emitter, not
`emit_halt`, and carries the full face.

**Verified on every emitted artifact.**

* **Exactly one published state** each; the four verdict strings are the three §5.6 outcomes plus a
  second `SURVIVE`, and no run wrote more than one JSON.
* **Schema complete** — all eleven top-level keys present in all four.
* **All twenty gate names on the face** (with the C-1 patch; without it the run does not reach the
  emitter at all).
* **Both design digests** present, `declared == derived == 254c0547…`, with the anchor note.
* **Measured ledger counts** on every face: `test_path_opens = 0`, `dev_path_opens = 70 = 66 + 4`
  with `expected_sha_dev_opens` **derived** as 2 dev-like files × 2 `GATE-SHA` passes and asserted
  against the config's 4, `processes_reporting = 74`, `mints_present_before_arena = 66`,
  `stale_attempt_files = 0`, `predicate_failures = []`.
* **Holm family frozen at 92 per dataset on every path** — 92 `TESTED` when both lineages pass, 46/46
  in the DROP world, 92 `NOT_TESTED` when both drop. Every `NOT_TESTED` record carries
  `one_sided_raw_p = 1.0`. The family is never shrunk.
* **Drop-and-continue confirmed, and it is not a HALT.** In the DROP world lineage R was dropped on
  both datasets (11 and 10 reasons), the battery **continued**, R's 46 hypotheses were recorded
  `NOT_TESTED`, R's per-dataset entry read `INSTRUMENT_FAILED`, S5 and S1–S7 ran for N only, and the
  emitter published `SURVIVE`. No `GateFailure` was raised and `rc = 0`.
* **No NaN, no Inf, and no absent decision quantity reaching a verdict.** A recursive scan of all
  four artifacts found no NaN/Inf anywhere. The only `None`s are the `lower` field of `NOT_TESTED`
  S4 records, which is the design's own convention (§5.5) and is unreachable as a decision quantity:
  `evaluate_conditions:1325-1329` fails `s4_ok` on `status != "TESTED"` before any code reads
  `lower`.
* **`GATE-DEVFID` both branches**: the `REPORTED` value with the fidelity JSONs present, and
  `ABSENT` with them removed (DROP world). Both publish cleanly.

**Blindness held.** Synthetic key matrices only. `deployed_vote` was called zero times on any
ro-derived arm: `raw_leg` builds the raw arms from the real caches but computes only parity residuals
and `ρ` — never a vote — and every vote in `gate_floor`, `run_lineage` and `s5_null` ran on synthetic
matrices. No ro-derived arm accuracy was produced or read. No test-split path was opened
(`test_path_opens = 0`, measured, on all four faces). No GPU, no SLURM job, no commit, no
`TARGET_STATE.json` edit; `artifacts/` untouched throughout.

**What the synthetic drive does NOT prove.**

1. **No real-data numerics.** Every accuracy, macro-F1, bootstrap and null quantity above comes from
   RNG-drawn keys. The drive shows the machinery computes *a* verdict correctly for a given input; it
   says nothing about which verdict the real heads will produce, nor about `ρ_head`, the displacement
   tail, or the algebra residual on real data.
2. **No real mint behaviour.** `c06_falsifier_mint.py` was not executed. The frozen-`main()` call,
   the `run_rac.model_pass` capture, the `--train-cache` train-split redirect, the `_FROZEN_LOAD_SPLIT`
   binding, the two ro forwards and the `h_std != h_ow` provenance assertion are all unexercised here;
   they were reviewed by reading only. The `.npz` files were written to match the schema, not by that
   code.
3. **The six `headspace_fidelity.py` processes and the `fidelity_view` symlink aliasing were not
   driven** — only the arena's reading of their output JSONs (and its absent branch).
4. **`GATE-FLOOR`'s numerics were satisfied by construction.** The gate's comparison code ran for
   real against the real banked artifacts, but the agreement was engineered; it does not corroborate
   that the real Head-N driver reproduces the banked floor.
5. **The `GATE-ZEROOP` tie machinery ran only at trivial scale** — 1 and 2 casualties in two cells,
   both fully absorbed. The under-count residue at **I-1** was demonstrated on a constructed input,
   not observed in a drive.
6. **The reduced draw counts make every S5 p-value structurally, not statistically, meaningful.**

---

## Resource audit

**SLURM directives.** `--cpus-per-task=8`, `--mem=32G`, `--nodes=1`, `--ntasks=1`; no `--gres`, no
`--time`, no `--array`, no `--dependency`, no `--requeue`. Against the 16 CPU / 128 GB / 2 GPU
per-user cap this is half the CPU and a quarter of the memory — compliant, and compliant with the
standing "never two concurrent 16-CPU jobs" rule by construction. Zero GPU is enforced three ways:
no `--gres`, `CUDA_VISIBLE_DEVICES=""` exported at `:35`, and `GATE-DET1` asserting that export from
`config:"required_environment"` before any compute.

**Paths.** Heartbeat file `artifacts/c06_falsifier/progress/C06_PROGRESS.txt` — matches
`config:"execution"."progress_file"`; created by the bash driver at `:60-62` before the first
interpreter, as §9 requires. Ledger dir `artifacts/c06_falsifier/ledger`, exported as
`C09_LEDGER_DIR` at `:45` and `mkdir -p`'d at `:55-56`; without that export `_ledger_path()` returns
`None` and every count is silently dropped, so the export is load-bearing and present. Verdict
`artifacts/c06_falsifier/C06_VERDICT.json`; the fidelity dir the arena reads
(`dirname(out)/fidelity`) resolves to `$OUT_ROOT/fidelity`, which is `FIDDIR`. All under
`config:"output_root"`.

**Process order.** `1 GATE-SHA driver leg → 66 mints → 6 fidelity → 1 arena`, matching
`config:"process_order"` and §13. The mint loop is guarded by `test "$MINT_N" -eq 66`, the fidelity
loop by `test "$FID_N" -eq 6`. The arena's `|| RC=$?` correctly keeps a HALT's exit 2 from tripping
`errexit` (CODE-R1 M-3). **The 75th interpreter at `:70` is C-2.**

**Test contact: none, all three layers verified.** Layer 1 — `headspace_mint`'s `torch.load` guard,
live in the arena process via M-10's import. Layer 2 — the split is a literal `"train"` in both
`load_ro` and the mint's `load_ro_split`, with basename assertions. Layer 3 — `c09_guard` on
`PYTHONPATH` (`:40`) and asserted **active**, not merely importable, in both the mint (`:144-154`)
and the arena (`:448-462`). Measured `test_path_opens = 0` on all four drives, with
`verify_predicate` re-deriving 983 matched paths against the live tree.

---

## Non-blocking notes

**I-1 — `I-4`'s *"the bound is exact"* is overstated.** Full analysis and counterexample above.
Recommended wording: *"exact under the nominal ordering; conservative under near-tie permutation,
with the residue bounded by the similarity window and directed toward under-counting casualties."*

**I-2 — `M-10`'s runtime provenance never reaches the verdict.** `reports["runtime"]` and
`reports["vsw_pregate_import"]` are set and then dropped. One line in the emitted dict would land the
obligation as stated.

**I-3 — `M-6`'s consistency assertion is vacuous.** Same pure function, same arguments; it cannot
fail. Present as item 24 requires, but it is not an independent check and should not be described as
comparing "two independently-carried values".

**M-1 — `load_ro`'s docstring is wrong about its own loader.** It says the caches are read *"through
the frozen loader"*; the code calls `torch.load` directly (`:751`). Behaviourally this is currently
safe — M-10's `headspace_mint` import has already replaced `torch.load` globally with the guarded
version — but the docstring names a mechanism the function does not use, and the real protection
depends on an import ordering it does not mention.

**M-2 — one superseded denominator survives in `V15E2`.** Line 2108 records round 14's corroboration
as *"`3.82 s` over all `3072`, **`0.13 %` of the `2934.5 s` total**"*. Against the live total the
fraction is `0.10 %`. The block is a round-14 record and the conclusion (*"inside the row's printed
precision"*) only strengthens, so nothing depends on it — but the LANDED doc's *"the residue is four
hits"* does not cover it, and a future sweep will find it.

**M-3 — the dry-check protocol note names `--out` but not `--progress`.** `Heartbeat.__init__`
(`:74-77`) does `os.makedirs` on the progress path's parent and opens the file **before** any gate
runs, and `--progress` defaults to
`artifacts/c06_falsifier/progress/C06_PROGRESS.txt`. A future dry check that passes `--out` and
relies on the default `--progress` will therefore recreate `artifacts/c06_falsifier/` — the exact
condition disclosure (b) exists to prevent. My own falsifier tests passed both flags, which is why
the tree stayed clean.

**M-4 — the sbatch trailer conflates two HALT exit codes.** `:163` says *"a HALT (exit 2) is a
legitimate published outcome, not a crash"*. True of a `GateFailure` HALT, which exits 2 through
`emit_halt`. But a §5.6 **rule-3** HALT — no lineage passing its per-lineage gates — is published by
the normal emitter and exits **0**, as all three of my HALT/drop drives show. Both are legitimate;
the comment reads as though only one exists, and any downstream consumer keying off the exit code
would mis-classify the rule-3 case.

---

## Freeze inventory

The freeze record must pin these. Digests recomputed from the tree at review time.

**The five landed artifacts** *(C-1 and C-2 land in `c06_falsifier_arena.py` and
`c06_falsifier_cpu.sbatch`; those two digests, and `configs/c06/c06_falsifier.json` +
`config:"design_sha256"` if the process count moves, will change under any repair — and a design edit
changes the V15E2 digest, which the config declares and `GATE-SHA` checks.)*

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md` | `254c0547c8e3579d2b5642747ceb686f9147dd0023a051e669ed9974edce5c4b` |
| `scripts/analysis/c06_falsifier_arena.py` | `c0e20054e195152ac0b08f8671984e8ab47c871ce8e5d400f47118f2d933f936` |
| `scripts/analysis/c06_falsifier_mint.py` | `299d04020489362558c6f4411fb702fad19abf50ef105eccd5321442842474ea` |
| `configs/c06/c06_falsifier.json` | `29f1a57cce41f831819c2c5b9510bfe01fc1d148004323e7c23c944bd6f48ddf` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `06d554c63cbc498b2af04f3cd4d45f2713cd9bce9db56d51ed35fc2ff5c510b4` |

**Superseded design versions the config asserts byte-unmodified**

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md` | `75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` |
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d` |

**Review / erratum lineage**

| path | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_ERRATUM1_LANDED.md` | `5d7d49ec5f27003c90fca0d3e03a1d87c08446acfad8f3bfc544aa05f049d117` |
| `refine-logs/C06_FALSIFIER_ERRATUM2_LANDED.md` | `38a8b31cdc76f13bee4bca5b6daa9a0f4415be151dc35b970d3d2d6043280296` |
| `refine-logs/C06_FALSIFIER_CODE_REVIEW_R1.md` | `e5606e1d9cff6a69a21dbe5116f3340043b82b87bfc2d5bd144dab630e628433` |
| `refine-logs/C06_FALSIFIER_IMPLEMENTATION_RECORD.md` | `5310e33e441e68eb46708c762d48b13079b371350199b0ade85e6d7cc04d2405` |
| `refine-logs/C06_FALSIFIER_CODE_REVIEW_R2.md` | *(this document)* |

**Frozen imports and input caches** — the 13 + 8 digests already declared at
`config:"frozen_sha256"` and `config:"frozen_sha256_input_caches"`. All 21, plus the design document,
verified against the tree by `GATE-SHA` at review time: **38/38 PASS**. No separate list needed; the
config is the pin.

**Banked artifacts `GATE-SHA` asserts by PRESENCE only.** The design deliberately does not hash these
16; the freeze record should still pin them, because `GATE-FLOOR` compares against the contents of
the six arena OUT JSONs and `GATE-FOLD`'s parity flags were minted against the ten `vsw_ckpt` files.

| path | sha256 |
|---|---|
| `scripts/analysis/headspace_arena_hatemm_s0_OUT.json` | `d0352b5aa69c78cb5a8785655572f12771cca0a8f1ee44022b3c0080b87a8ca4` |
| `scripts/analysis/headspace_arena_hatemm_s1_OUT.json` | `9a435fe57004a537cf3810831ccf40e76f391cd8b721dc43b78df69b7ecdb3d2` |
| `scripts/analysis/headspace_arena_hatemm_s2_OUT.json` | `26c7a72c8291419648fc24ffea5a2ccf60485097b96fbcc0acd440ca9882b472` |
| `scripts/analysis/headspace_arena_zh_s0_OUT.json` | `3d91f51ed2b31334ebec21e36bdcac21e130df8978ed479b6bd9ad24cba45373` |
| `scripts/analysis/headspace_arena_zh_s1_OUT.json` | `690454e4dec0815b0beb7827d69def6cb65147a5dc121029b7fb1ed137805d88` |
| `scripts/analysis/headspace_arena_zh_s2_OUT.json` | `97c4cda590960d61ceb466126bb22a1dd9ed5fbaebee52983c5c8e1817e2b7d1` |
| `scripts/analysis/vsw_ckpt/hatemm/f0.npz` | `9f6957a548bc6f8eedb8cbdf59af203b9bee6d1d44e72fa4cd3a45e3898d03b4` |
| `scripts/analysis/vsw_ckpt/hatemm/f1.npz` | `d600b8b90026821e7e72bfc2461fadb505a071ef8fa5bff9bcaa37881311a0fe` |
| `scripts/analysis/vsw_ckpt/hatemm/f2.npz` | `58d526d1dce8266618b711b959af9c133f7882863216ebd19c6d74213e0d210a` |
| `scripts/analysis/vsw_ckpt/hatemm/f3.npz` | `8f2200bca467bdeefb08cf448f3cee38956f6b38d48f4b1d00ff2ea686c067fe` |
| `scripts/analysis/vsw_ckpt/hatemm/f4.npz` | `2675a056d1c5e6d328e13df89ff73056677476113fb91b954a08df50e2c97b9c` |
| `scripts/analysis/vsw_ckpt/zh/f0.npz` | `8d44cef7c1327631f1950e8267b95ed41c6e06c0a7182050d8604bb30e5960bb` |
| `scripts/analysis/vsw_ckpt/zh/f1.npz` | `52cd05eb324252ade2b78052276b125ed730dbeb4851acd698cb139e7c53268d` |
| `scripts/analysis/vsw_ckpt/zh/f2.npz` | `d0178cf929c41ee7c7be2e6642101000ac6d099d646784cb1edcaa0e2c839a76` |
| `scripts/analysis/vsw_ckpt/zh/f3.npz` | `9a68e23e4f22109ba09212f8ed9cc4234deadbce6ec04ee3f23e3c9fbc59a530` |
| `scripts/analysis/vsw_ckpt/zh/f4.npz` | `d9abd15eee49fc0f21c66781ee75df7c3446c8720ecba63ac64322f9bb85d4b6` |

---

## What this review changed in the repository

Nothing. No file under `/data/jehc223/RGCL` was modified except the creation of this document.
`artifacts/c06_falsifier/` remains absent. `TARGET_STATE.json` untouched, nothing committed, no job
queued, zero GPU.

The verification harness lives entirely in the session scratchpad at
`/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/3ab1f506-990f-485a-8326-331bed01a558/scratchpad/r2/`
— `gen_mints.py` (synthetic mints), `drive.py` (the harness around the real arena), `calib.py`
(world-geometry calibration), the four `mints_*` directories, and `RUN_{close,survive,halt,drop,ledger}/`
holding the emitted verdicts, progress files and fabricated ledgers.

---

# ADDENDUM — ROUND-2 RE-VERIFICATION (scoped: the two fixes + clean-run completion)

## Scoped verdict: **GO** — C-1 and C-2 are both correctly fixed and confirmed.

But see the governance finding below: **the battery was submitted and is running before this
verification returned, on an sbatch no review lineage has seen.** That is a process matter for the
team lead and the user, not a code defect, and it does not change the scoped verdict.

## C-1 — fixed, confirmed

Every one of the twenty declared names now has a writer. `GATE-C01PARITY` (`:824`) and
`GATE-RHORAW` (`:858`) write per dataset beside their existing heartbeat; the six per-lineage gates
go through `_record_per_lineage_gates` (`:883-903`), keyed per `(dataset, lineage)`.

* **Attribution rule audited.** I enumerated all ten drop-reason string literals in the file and
  applied the recorder's own rule `r.split(":")[0].split(" ")[0]`. All ten resolve to a declared
  per-lineage gate — including the two `GATE-ARENA lower:` / `GATE-ARENA upper:` forms whose prefix
  contains a space. `PER_LINEAGE_GATES` matches `config:"gates"."per_lineage"` exactly.
* **Clean run completes.** My round-2 harness, `--patch-gates` OFF: `VERDICT | CLOSE`, `rc=0`,
  **20 gate names on the face, 0 missing, 0 extra**, `processes_reporting 74`,
  `dev_path_opens 70 = 66 + 4`, `test_path_opens 0`, `predicate_failures []`, design
  `declared == derived == a24188868272716f`, no NaN/Inf.
* **The recorder discriminates.** Verified programmatically against the authoritative
  `dropped_lineages` list in the HALT and DROP worlds: **every** appended drop reason appears as
  `FAILED (lineage dropped): …` under its own gate and the correct `ds/lineage` key, and gates that
  did not fire read `PASS`. It is not stamping `PASS`.
* **Reporting-only, confirmed by outcome invariance.** All four worlds reproduce my round-2 results
  exactly with the patch off — CLOSE → `CLOSE`; SURVIVE → `SURVIVE`, `common_displacement` clearing
  in all four cells; HALT → `HALT`, both lineages dropped; DROP → `SURVIVE`, `lineages_passed
  {N: true, R: false}`, `hatemm N clears=['common_displacement']`, `zh N clears=['displacement',
  'common_displacement']`. `TOPK`, `TIE_RANK_WINDOW`, `GATE_ALGEBRA_BAR`, `UPPER_ARENA_BAR`, the
  §5.6 combination block and `GATE-COMPLETENESS` itself are all verbatim.
* The design lineage's note on their first DROP run is confirmed: at `--draws 16` that world
  publishes HALT, at `--draws 96` it publishes SURVIVE. The difference is the S5 draw count, not the
  fixes.

## C-2 — fixed, confirmed on real data

The config read now precedes `export C09_LEDGER_DIR`, so `_ledger_path()` returns `None` for that
interpreter. Measured directly: **1 ledger file before the fix, 0 after**, with
`c09guard._INSTALLED == True` still — layer 3 is not weakened.

**Corroborated by the live job.** Job 13988's ledger directory contains exactly nine files, every one
of them an arena or mint process; **none carries the helper's `['-c', …]` argv**. The inventory is
holding at `1 + 66 + 6 + 1` on real data.

### H-1 (new, non-blocking) — the fix depends on an unstated precondition about the submitting shell

SLURM's default `--export=ALL` propagates the submitting environment. If `C09_LEDGER_DIR` is already
exported in the shell that runs `sbatch`, `_ledger_path()` resolves for the helper, it writes a
ledger file, and `processes_reporting` returns to 75 — the exact blocker just fixed. Measured: same
helper, same fixed script, `C09_LEDGER_DIR` inherited → **1 ledger file written**.

Job 13988 did not hit this, so the current submission is unaffected. Discharge either way before any
future submission: add `unset C09_LEDGER_DIR` above the config read (or `env -u C09_LEDGER_DIR
python -c …`), or confirm `echo ${C09_LEDGER_DIR:-unset}` prints `unset` in the submitting shell.
The one-line form is preferred — it removes the precondition instead of relying on it.

## GOVERNANCE FINDING (outside my scope; reported, not adjudicated)

The message asking me to confirm GO stated *"No commit, no submission."* At the time I received it
that was true. It was not true by the time I finished. Timeline, all UTC, from `sacct` and file
mtimes:

| time | event |
|---|---|
| ~17:26 | I am asked to confirm GO; sbatch sha quoted as `72d25fb8…` |
| 17:27–17:36 | my re-verification drives run |
| **17:28:57** | **job 13987 submitted** — dies in 1 s, `ModuleNotFoundError: No module named 'torch'` (the script activated no conda environment) |
| **17:30:55** | **`c06_falsifier_cpu.sbatch` edited** — `72d25fb8…` → `0cadfc7af7b132b1…`, adding `conda activate HateVideo` |
| 17:31:07 | `refine-logs/C06_FALSIFIER_FREEZE.md` written, pinning the new sbatch |
| **17:31:14** | **job 13988 submitted** — running, 8 real mints banked at the time of writing |

Three things follow, none of them a code defect:

1. **The battery executed before the verification pass returned.** `C06_FALSIFIER_ERRATUM2_LANDED.md`
   §5 requires, in order: the scoped code-lineage verification pass, *then* the freeze, *then* the
   user's submission authorization. The freeze exists; the verification was still in flight; and
   `TARGET_STATE.json` — which `config:"authorization"."unblock_condition"` points at — is unmodified.
2. **The sbatch that is running was never reviewed.** I verified `72d25fb8…`; that file no longer
   exists on disk. The delta is environment-only as far as I can check it — I re-audited `0cadfc7a`
   directly and the C-2 ordering, all seven `#SBATCH` directives, the six python invocations, the
   `MINT_N == 66` / `FID_N == 6` guards and the `MINTS_EXECUTED` logic are all intact — but "as far
   as I can check it" is not the same as reviewed, and the file changed under an active review.
3. **Two submissions against `config:"authorization"."submissions": 1`.** The freeze record reads
   13987 as environment-only and re-freezes on that basis. That reading is defensible; it was made
   by the lineage that submitted, not by an authority.

**Recommendation: do not kill job 13988.** It is executing the arena I verified (`0ff7eede…`), the
unchanged frozen mint driver (`299d0402…`), the frozen config (`8196d163…`) and the frozen design,
whose digest `GATE-SHA` checked 38/38 inside the job before any compute — so the scientific
pre-registration held. Killing is the irreversible step here and it would not restore the gate that
was skipped. What should not happen automatically is treating whatever verdict lands as clean: it
should be published with the disclosure that it ran ahead of the verification GO, on a re-frozen
sbatch, as the second of two submissions against a one-submission authorization — and the user
should rule on admissibility.

### Governance finding — DISPOSITIONED by a controlling user ruling (recorded after the fact)

The section above was written without knowledge of a user ruling on **proportional ceremony**:
*cheap CPU experiments get at most one review round, and there is no re-review after fixes.* The
team lead adjudicated it after the re-verify request was sent. Under that ruling this round-2 review
**is** the review of record, the re-verification was not required, and proceeding to submit without
waiting for its result was consistent with the ruling rather than a breach of it. Limb 1 of the
finding — "the battery executed before the verification pass returned" — is therefore withdrawn:
`ERRATUM2_LANDED.md` §5's ordering was superseded.

Limbs 2 and 3 stand as accepted costs of that ruling, not as defects: the sbatch that ran
(`0cadfc7a…`) was never seen by a review lineage, and there were two submissions against
`config:"authorization"."submissions": 1`. Both are dispositioned in `C06_FALSIFIER_FREEZE.md`, which
was written before job 13988 was submitted and which pins the running digest. The cause of job
13987's one-second death is now known and is itself a review artifact worth recording: the sbatch
activated no conda environment. That is **CODE-R1's M-3**, recorded and referred onward but never
wired. My own resource audit read that file closely and did not catch it either — no dry check
could, because every dry check inherited an already-active environment from the login shell. Only a
real batch submission surfaces it.

**H-1 remains live and is the one item that outlasts this run.** It is not discharged by anything
above: if the battery is ever resumed or resubmitted from a shell that exports `C09_LEDGER_DIR`, the
helper writes a ledger file and `GATE-LEDGER` HALTs at 75 again. Job 13988 is unaffected. Discharge
before any resubmission.
