# C06 $0 FALSIFIER — SEPARATE CODE/RESOURCE REVIEW LINEAGE, ROUND 1

**Verdict: REVISE — 1 Critical, 4 High, 5 Important, 12 Minor.**

Reviewer had no part in the 15 design rounds, the implementation, or the erratum. Judged solely
from the frozen design, the four implementation files, the frozen modules they import, and
measurement.

**Artifacts, sha256 re-verified at review time — all five match the handoff:**

| file | sha256 |
|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `0b446b91675fd4ff8aea15f2648401d6ce589d089eadad34846f885b2ec9c2ab` |
| `scripts/analysis/c06_falsifier_mint.py` | `1084b5be8c11ad60085115504e999b338db481801614452526084b87d1b3a1d0` |
| `scripts/analysis/c06_falsifier_arena.py` | `6ba6a14e4120e683121f93d234f5794f7bab514dfe2f51a779c87246f484e7a8` |
| `configs/c06/c06_falsifier.json` | `3ebcc36c74b759d28612e0974227c08dea98f6ba72e09f36ca047f35d7f5087e` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `76d061daf62c51dae584387160924cae482ca7ea20710423b443424b2a21b634` |

---

## 0. What this review executed, and what that does and does not prove

Three probes were run on the login node under DET-1, zero GPU, zero SLURM, zero commit, zero edit
outside the session scratchpad. No mint of the 66, no battery-arm accuracy on any ro-derived arm.

**Probe 1 — `probe_mint_redirect.py`.** Calls the **real** `c06_falsifier_mint.main()` with the real
`_load_split_with_override` closure and the real `load_ro_split`, on four synthetic `.pt` caches
whose feature values are distinct constants (native `1.0`, ro-standard `2.0`, ro-one-word `3.0`,
dev `4.0`) and a pass-through dummy head, so the emitted key matrix names the file it came from.
`headspace_mint.main` and `run_rac` are stubbed.
*Proves:* which file each of the driver's post-`main()` ro forwards actually opens, on both
lineages. *Does not prove:* anything about the real head, real features, training, or the frozen
`headspace_mint.main()` body.

**Probe 2 — `probe_arena_bits.py`.** Runs the **real** `ArmBuilder`, the **real**
`resampled_macro_f1`, and the **real** `mechfix_ops.macro_f1` on synthetic and adversarial inputs.
*Proves:* the downstream behaviour of the Critical finding through C01's own `l2_rows`; bit-equality
of the vectorised macro-F1 path over 12,200 resamples spanning five regimes; the measured cost ratio
of the S5 arm build. *Does not prove:* that the real arena's numbers are correct — no gate upstream
of `run_lineage` executed.

**Probe 3 — `probe_decision.py`.** Drives the **real** `Battery.load_frozen`, `Battery.s4_family`,
`Battery.evaluate_conditions` and C01's own `holm_adjust` / `select_strongest_ordinary_control` on
fabricated cells at `n = 120`, plus a replication of `main()`'s verdict block.
*Proves:* family assembly and the drop path's padding, the S4 key wiring, each conjunct's veto
power, and the verdict truth table. *Does not prove:* that `run_lineage` fills those cells
correctly — it never ran.

**The F114/C09 lesson, stated rather than assumed.** A synthetic harness hides real-path defects.
Everything below that rests on a harness says so; everything that rests on reading says so. The
Critical finding rests on a harness that exercised the real closure, and its downstream consequence
rests on C01's real `l2_rows` — but neither ran against the real caches, so the *mechanism* is
measured and the *magnitude* (all 30 Head-R mints) is inferred from the code being unconditional.

---

## 1. CRITICAL

### C-1 — Head-R's one-word key matrix is the **standard** cache. All 30 Head-R mints carry `h_std == h_ow`, and the battery HALTs where the design requires CLOSE.

**Files:** `scripts/analysis/c06_falsifier_mint.py:115-125` and `:158-170`, interacting.

The driver installs its train-split override **globally on the frozen module** and never removes it:

```
158    _frozen_load_split = HM.load_split
160    def _load_split_with_override(cache_dir, split, model):
161        if split == "train" and override is not None:
162            directory, filename = os.path.split(override)
166            stem = filename[len("train_"):-len(".pt")]
167            return _frozen_load_split(directory, "train", stem)
168        return _frozen_load_split(cache_dir, split, model)
170    HM.load_split = _load_split_with_override
```

`HM.main()` is then called (`:196`), and **after** it returns the driver performs its own ro
forwards through `load_ro_split` (`:225`, `:227`), which routes through the *same patched symbol*:

```
115    def load_ro_split(dataset, which, split="train"):
121        assert split == "train", ...
125        return HM.load_split(cfg["cache_dir"], split, model_with_suffix)
```

`split` is the literal `"train"` (§12 layer 2 requires it to be), and on Head-R `override` is not
`None`, so the branch at `:161` fires and the `model_with_suffix` argument — carrying `ro_ow_L24` —
is **discarded**. The function returns `train_<model>-ro_L24.pt`, the standard cache, for both
`which="std"` and `which="ow"`.

§13.1 item 3 requires "no branch conditional on the cache filename or suffix", and the
implementation satisfies that literally — the substitution is unconditional. That unconditionality
is exactly the mechanism: the guard the design asked for is what makes the one-word load
unreachable.

**Measured (probe 1, real closure, synthetic caches):**

```
===== N =====                              ===== R =====
files opened by the frozen loader:         files opened by the frozen loader:
    train_SYNTH.pt                             train_SYNTH-ro_L24.pt
    train_SYNTH-ro_L24.pt                      train_SYNTH-ro_L24.pt
    train_SYNTH-ro_ow_L24.pt
K_train marker : 1.0                       K_train marker : 2.0
h_std   marker : 2.0                       h_std   marker : 2.0
h_ow    marker : 3.0                       h_ow    marker : 2.0     <-- expected 3.0
h_std == h_ow  : False                     h_std == h_ow  : True
```

Head-N is correct. Head-R's `h_ow` is the standard cache, bit-for-bit identical to `h_std` (same
file, same head, same eval-mode forward). Nothing in the mint catches it: the only post-forward
assertion is `h_std.shape == h_ow.shape == frozen["K_train"].shape` (`:229`), which holds.
`meta["c06"]["ro_ow_path"]` records the *intended* path, so the `.npz` provenance block asserts a
file that was never read.

**Wrong-verdict path, measured downstream.** In the arena, `head_cell_arms` (`:596-610`) builds
`displacement = l2(h_ow − h_std)` on an all-zero matrix. C01's `l2_rows` compares `exact_zero`
against the authorised `zero_mask` (all-False on the arena population) and calls `die()`:

```
A. h_std == h_ow (the measured Head-R state) fed to the real ArmBuilder
   RuntimeError: probe/degenerate/0/displacement exact-zero mask diverged from authorized mask
```

`die()` raises `RuntimeError` (`c01_policy_contrast_a0.py:392-393`), which
`c06_falsifier_arena.py:1173-1177` catches and reports as
`HALT | INSTRUMENT_INCONCLUSIVE | c01_algebra RuntimeError`. So on **every** run the battery burns
all 66 mints, the raw leg, `GATE-FLOOR` and the entire Head-N lineage — §8's Phases 1–2, ~2,500 s of
the 2,929.9 s budget — and then HALTs at the first Head-R cell. §5.7 pre-declares CLOSE as the
expected outcome; the code cannot publish it, or SURVIVE, on any input.

The failure is also **silent in the direction that matters**: it presents as an algebra crash, not
as a data-provenance error, so a reader of the heartbeat's final line would look at `l2_rows` and
the arm builder rather than at the mint.

**Resolution path.** Restore the frozen symbol before the driver's own loads, e.g. set
`HM.load_split = _frozen_load_split` immediately after the `try/finally` around `HM.main()`
(`:198`), or have `load_ro_split` call `_frozen_load_split` directly. Either way, add an
assertion that binds the returned object to the file that was asked for — the natural one is to have
`load_ro_split` return the resolved path alongside the split and assert it equals
`ro_cache_path(dataset, which)`, and to record the **resolved** path in `meta["c06"]` rather than the
intended one. A second, independent check worth adding at the same site: assert
`not np.array_equal(h_std, h_ow)`, which is a one-line falsifier for this entire defect class and
cannot fire on a correct run (§7.8 measures `min_i d_i` at `0.018`–`0.038`).

---

## 2. HIGH

### H-1 — Two per-lineage gates HALT the whole battery instead of dropping their lineage. Round-4 H-3 is reintroduced inside §5.6's repair.

**Files:** `c06_falsifier_arena.py:667`, `:699-704`, `:730-735`, `:230`.

§5.6 scopes six gates as per-lineage — `GATE-ARENA`, `GATE-ORBITDISP`, `GATE-NESTED`,
`GATE-SELFTEST`, `GATE-ZEROOP`, `GATE-ALGEBRA` — and a failure "drops that lineage on **both**
datasets", never voids the battery. The code implements four of them as appends to `drop`
(`:742-784`) but the other two as `raise GateFailure`:

```
667        raise GateFailure("GATE-NESTED", "empty fold {}".format(fold))
700        raise GateFailure("GATE-NESTED", "{} {} unscored items".format(arm, lineage))
704        raise GateFailure("GATE-NESTED", "check count != item count")
733        raise GateFailure("GATE-SELFTEST", "{} {} s{} net {} != n_D*(dacc) {:.6f}" ...)
```

and `rho_of` (`:230`) raises `GateFailure("GATE-ORBITDISP", ...)` on a non-finite or zero key norm,
which is likewise a per-lineage gate.

`GateFailure` is caught only at `main()`'s top level (`:1170`) and produces
`HALT | INSTRUMENT_INCONCLUSIVE`.

**Wrong-verdict path.** Head-N is the out-of-domain transplant §5.7 names as the live
instrument-failure path. If Head-N trips `GATE-SELFTEST` or `GATE-ORBITDISP`'s norm guard on one
dataset, §5.6 rule 1 requires: drop Head-N on both datasets, and if Head-R clears S1–S7 on both,
publish **SURVIVE**. The code publishes HALT. This is the precise inversion round-4 H-3 identified
and §5.6 was written to repair — "A clean Head-R SURVIVE is therefore not voided by the transplant
lineage's failure."

Reachability is low (`GATE-SELFTEST` is an arithmetic identity at `1e-6` tolerance; `GATE-NESTED`'s
`:703` predicate is tautological, see M-5), but §5.6's scoping is the design's central verdict-
combination repair and the code contradicts it for two of the six gates. §13.1 item 20 assigns this
lineage the check "no verdict path can be reached with a lineage that passed on one dataset only";
the converse — no verdict path may HALT the battery on a per-lineage failure — is not satisfied.

**Resolution path.** Convert the four sites to `drop.append(...)` and make `rho_of` return a sentinel
the caller turns into a drop. `GATE-NESTED`'s "empty fold" at `:667` is arguably a population defect
rather than a lineage defect; if it is to stay global, §5.6's twelve-gate list must say so.

### H-2 — `GATE-LEDGER` is hardcoded `PASS`. Seven of §12's eight predicates are never measured, the verdict face publishes unmeasured literals as counts, and `c09_guard`'s own measured ledger is not wired. Implemented as frozen, one binding predicate would fail.

**Files:** `c06_falsifier_arena.py:362-364`, `:1136-1139`, `:1161`; `scripts/slurm/c06_falsifier_cpu.sbatch:37`.

§6's gate table: `GATE-LEDGER | G | C09's full declared-count predicate set, process count **binding**
(§12)`. §12 tabulates eight predicates, six binding plus "processes reporting … yes — HALT on any
mismatch". The code:

```
362        self.ledger = {"test_path_opens": 0, "test_label_materialisations": 0,
363                       "dev_path_opens": 0, "banked_trainlog_opens": 0,
364                       "dev_or_test_labels_into_decision_quantities": 0}
...
1137        bat.ledger["mints_present_before_arena"] = bat.reports["mints_present_before_arena"]
1139        bat.gates["GATE-LEDGER"] = "PASS"
```

Those five counters are initialised to `0` and **never incremented anywhere**. No predicate is
evaluated. `GATE-LEDGER` is set to `PASS` unconditionally. `processes_reporting` (73, binding) and
`dev_label_materialisations_outside_decisions` are not in the dict at all. The only real predicate
is `mints_present_before_arena == 66`, raised from `gate_fold_and_ledger_presence` (`:482-484`).

The verdict JSON emits `"ledger": bat.ledger` (`:1161`), so a published CLOSE will carry
`"test_path_opens": 0` and `"dev_or_test_labels_into_decision_quantities": 0` — read by any
downstream consumer as measurements, when they are literals no code produced.

**The measured mechanism the design intended exists and is not used.** `scripts/analysis/c09_guard/
c09guard.py` maintains a live `LEDGER`, writes a per-process file to `$C09_LEDGER_DIR` at exit
(`_flush`), and exposes `aggregate(ledger_dir)` whose docstring says: *"the arena aggregates them
into GATE-LEDGER, so the ledger reports MEASURED opens rather than literals."* Measured: the C06
sbatch never exports `C09_LEDGER_DIR`, so `_ledger_path()` returns `None` and every process's counts
are discarded; and `grep -n "c09guard\|C09_LEDGER_DIR\|aggregate"` over the arena, the mint and the
sbatch returns nothing.

**§13.1 item 28's limb is unmet.** The item requires this lineage to verify "that layer 3 is
actually **active** in all 73 processes, **not merely importable**." The export exists and resolves
correctly — measured: with `PYTHONPATH` set, `importlib.util.find_spec("sitecustomize").origin` is
`/data/jehc223/RGCL/scripts/analysis/c09_guard/sitecustomize.py`, no competing module shadows it.
But that file swallows every failure by design (`except Exception … never break the interpreter over
the guard`), and **no C06 process asserts that `install()` succeeded**. A silent guard failure is
indistinguishable from a clean run, and `GATE-LEDGER` would still publish `test_path_opens: 0`.

**Wrong-verdict path.** §5.6 lists `GATE-LEDGER` among the twelve global gates whose failure HALTs
the battery. As coded it cannot fail. If the guard were inactive and a test path were opened, the
design requires HALT; the code publishes CLOSE with a `PASS` and a zero.

**A design/code collision this lineage must surface, not paper over.** §12's binding predicate is
`dev_path_opens == mints_executed + 0`, warranted on "`headspace_fidelity.py` opens **no** `dev_seen`
file". But round-8 H-1 widened `GATE-SHA` to the input caches, and `frozen_sha256_input_caches`
(config `:239`, `:241`) contains two `dev_seen_*.pt` files. `sha256_of` opens them with
`builtins.open`, which `c09guard.is_dev_like` counts (`dev_seen` in basename). `GATE-SHA` runs in
**two** processes — the `--gate-sha-only` driver call (sbatch `:57`) and the arena itself (`:1038`)
— so a faithful implementation would measure `dev_path_opens == mints_executed + 4` and **HALT the
battery on a clean run**. The code avoids this only by not implementing the gate.

**Resolution path.** Wire `c09guard.aggregate` (export `C09_LEDGER_DIR` in the sbatch, import
`c09guard` in the arena, add the arena's own live counts as `aggregate`'s docstring instructs), add
a hard assertion that `c09guard._INSTALLED` is true at the top of every C06 process, and evaluate
each §12 predicate as a pass-condition. Before any of that, the `dev_path_opens` expectation needs a
design ruling: either it becomes `mints_executed + 4` with the `GATE-SHA` term named, or the two dev
caches are dropped from the digest list. This is a design question, not a code choice, and it should
go back to the design lineage as an erratum rather than be decided here.

### H-3 — 72 of 73 processes never write to the progress file. The mint phase — 85.6 % of the budget — is dark in it.

**Files:** `c06_falsifier_mint.py` (no heartbeat at all); `c06_falsifier_cpu.sbatch:50-52`, `:76`,
`:121`.

§9: "One progress file … created by the sbatch driver before the first python process starts;
**every python process appends through a handle opened `buffering=1`**. The bash driver **also**
echoes a line per mint, unbuffered." §13.1 item 12 requires this lineage to check
"append-without-interleaving across **all 73 processes**", and §9's granularity list opens with "one
line per mint (66), **one per training epoch within a mint**".

As written, the progress file receives exactly one `DRIVER` line from the sbatch (`:50-52`), then
nothing until the arena starts. The 66 mint processes and the 6 fidelity processes never open it —
`c06_falsifier_mint.py` has no `Heartbeat`, only a single `print()` to stdout at `:255`, and the
sbatch's per-mint `echo` (`:76`) goes to stdout, i.e. the SLURM `.out` file, not to `$PROGRESS`.

§8 prices the mints at `2,508.3 s` of the `2,929.9 s` total. Under the code the progress file is
silent for that entire span, against a specification whose stated guarantee is that "no interval
exceeds ~15 s". The arena's own `Heartbeat` is correct in isolation — one handle, `buffering=1`,
opened once, never re-wrapped, frozen denominator `PROJECTED_SECONDS = 2929.9` matching the
erratum'd §8 and the config — so this is a coverage gap, not a mechanism defect.

**Resolution path.** Give the mint driver and the fidelity invocation the same `Heartbeat` handle in
append mode (a `--progress` argument, one line on entry and one on exit at minimum), or route the
sbatch's `echo` to `$PROGRESS` as well as stdout. The per-epoch line inside a mint requires a hook
into `headspace_mint`'s existing `_metrics_spy` curve, which the C06 driver cannot reach without
editing a frozen module — that limb should go back to the design lineage.

### H-4 — S5 builds all 15 arms per draw when it reads 2. Measured 10.8×; §8 Phase 3 is under-priced by ~5,700 s and the realised total would exceed the conservative bound by ~2.4×.

**File:** `c06_falsifier_arena.py:914-917`.

```
914                    views = self.builder.build_views(
915                        [h_std], [h_ow[order]],
916                        "c06/null/{}/{}/s{}/f{}/d{}".format(...), mask)
...
920                    for a in real:            # real = ["displacement", "common_displacement"]
```

`build_views` constructs the 7 named arms, 6 rotation arms and 2 guard arms — 15 in all — inside a
loop of `256 draws × 3 seeds × 5 folds` per `(dataset, lineage)`. Thirteen of the fifteen are
discarded on every iteration.

**Measured (probe 2, real `ArmBuilder`, `n = 743`, `d = 1024`, 8 threads, 3 repeats):**

```
   full build_views (13 key arms + 2 guards): 0.3907 s   arms=15
   displacement + common_displacement only : 0.0361 s
   ratio full/two = 10.81x
   as coded : 3072 x 5 x 0.3907 =   6001.3 s (construction alone)
   two-arm  : 3072 x 5 x 0.0361 =    555.1 s
   §8 priced Phase 3 total (build + vote) = 273.7 s
```

§8 prices Phase 3 at `3072 × U4 = 273.7 s`, i.e. `U4 = 0.0891 s` for one `(draw, seed, dataset,
lineage)` cell including its five folds' builds **and** its ten votes — a unit consistent only with
a two-arm build. As coded the construction alone is `6,001 s`. Carrying the rest of §8 unchanged,
the realised total lands near `8,650 s ≈ 2.4 h` against the corroborating `2,929.9 s` and the
conservative `3,662.4 s` — a `2.36×` overrun of the conservative figure, which trips §8's own
clause: *"If the realized cost exceeds the conservative total by more than 2×, that is itself a
reportable process finding."*

Two smaller instances of the same class, both Minor on their own and folded here for the
resource picture: the arena loads the two ro caches **six** times (`:1060` twice, `:1109` four
times) against Phase 1c's count of one for the arena process; and it materialises arrays out of the
banked mint `.npz` roughly **270** times (120 in `run_lineage`, 30 in `gate_floor`, 120 in
`s5_null`'s cache) against Phase 1f's `150`.

**Resolution path.** S5 needs only `std`, `ow`, `common`, `displacement` and the `common_displacement`
pairing. Either add an optional arm allow-list to `build_views` — keeping it **one construction**, as
§13.1 item 19 requires, since the parity warrant transfers only through that single function — or
call the builder's `l2`/`fuse`/`paired` primitives directly for the two arms, which is what the
timing above does and what §8's `U4` prices. Note that a builder change touches
`GATE-C01PARITY`'s anchor and should be re-run against `--dry-parity-only` before freeze.

---

## 3. IMPORTANT

### I-1 — No decision JSON is written on any HALT path, and §5.6 requires the `RuntimeError` context in both places.

**File:** `c06_falsifier_arena.py:1170-1177`.

§5.6: "a `RuntimeError` is recorded as `INSTRUMENT_INCONCLUSIVE` with `l2_rows`' `context` string —
which carries the arm and block name — written to **both** the decision JSON and the **final
heartbeat line**." Both `except` clauses write the heartbeat line and `return 2`; `args.out` is
never created. A HALT therefore leaves no verdict artifact at all — no gates dict, no
`dropped_lineages`, no ledger, no scope block — only a line in a text file. Given C-1 this is the
path every run currently takes.

**Resolution path.** Emit the verdict JSON on both exception paths with
`verdict = "HALT"`, the failing gate name, the context string, and whatever gate outcomes were
recorded before the failure.

### I-2 — Both reporting gates are missing from the verdict face; `GATE-DOMAIN`'s quantity is computed nowhere.

**Files:** `c06_falsifier_arena.py:1141-1163`; nothing implements `GATE-DOMAIN` or reads the
fidelity outputs.

§6's table declares twenty gates. `grep -o 'GATE-[A-Z0-9]*'` over the arena returns eighteen names:
`GATE-DOMAIN` and `GATE-DEVFID` appear nowhere.

* **`GATE-DOMAIN`** (§6.4) requires `recovery fraction = (acc_ro − maj_arena) / (acc_native −
  maj_full)` for `endpoint_std` under Head-N, plus the raw-vs-head `endpoint_std` comparison, and
  says both "appear on the verdict face and in §10.2's scope sentence", and that "on a Head-N-drop
  SURVIVE or HALT it is reported as `INSTRUMENT_FAILED` rather than silently omitted." Neither
  figure is computed. `population_derived_constants.*.full_majority` sits in the config unread —
  its only consumer is this gate.
* **`GATE-DEVFID`** is produced by the six fidelity processes into `$FIDDIR/devfid_*.json`, but the
  arena never opens them and `bat.gates` carries no entry, so the verdict face reports 18 of 20
  gates with no indication that two are missing.

No bar attaches to either, so neither can flip CLOSE/SURVIVE/HALT. The consequence is that a
published verdict cannot support §10.2's scope sentence, which the design makes mandatory.

**Resolution path.** Compute the two `GATE-DOMAIN` figures in `run_lineage` for Head-N (all inputs —
`acc["endpoint_std"]`, the banked `acc_deployed`, both majorities — are already in hand), and have
the arena read the six fidelity JSONs into a `GATE-DEVFID` block. Add a completeness assertion that
`bat.gates` covers all twenty names before emit.

### I-3 — §13.1 item 25's per-cell displacement tail is not computed, so `tiny_ok`'s non-carriage rests on the four-cell extrapolation §5.2.3 explicitly refused.

**File:** `c06_falsifier_arena.py:607-610`, `:786-793`, `:1141-1163`.

Item 25 requires `min_i d_i` and `frac(d_i ≤ 0.001)` "computed and recorded per `(dataset, seed,
fold, lineage)` cell alongside the `GATE-ALGEBRA` residual, so the max-versus-median gap §7.8
measures is auditable at run time rather than assumed." §5.2.3 names this as the control that keeps
`tiny_ok`'s non-carriage honest: "**What bounds the four-cell evidence.** Four cells of sixty bound
nothing formally … The control is §13 item 25, which requires **every** cell to record its own
`min_i d_i` and `frac(d_i ≤ 0.001)` at run time, so the assumption is auditable per cell rather than
extrapolated from four." §8's Phase 7z prices it at `0.7 s` over 60 cells.

The code computes `d_i` (`:607-609`) and keeps it as `displacement_norm` for S7's quantile only. No
minimum, no tiny fraction, no per-fold breakdown, and `displacement_norm` never reaches the verdict
JSON. A SURVIVE would therefore rest on the four-cell extrapolation the design refused, and a CLOSE
would leave the assumption unaudited.

**Resolution path.** Record `min`, the `≤ 0.001` fraction and the median per `(dataset, seed, fold,
lineage)` in `head_cell_arms`, carry them in the cell, and emit them beside `algebra_residual` in
`per_dataset`. The `d_i` vector is already materialised per fold, so the cost is the `0.7 s` §8
already carries.

### I-4 — `GATE-ZEROOP`'s near-tie ranking is each arm's own top-21, not the union of the two arms' top-21 sets.

**Files:** `c06_falsifier_arena.py:673-677`, `:268-277`.

§6.5, first bullet: "**Ranking:** the **union** of the two arms' top-21 sets." Config `:178` repeats
it verbatim. The code retrieves each arm's own top-21 independently (`:674-677`) and
`tie_casualties` (`:268-277`) forms near-tie groups within each arm's own neighbourhood
(`vote_bounds_over_orderings(sim_a[i], lab_a[i], window)` and separately for `b`).

The two arms' keys differ by at most the `GATE-ALGEBRA` residual, so their top-21 sets differ only
at the rank boundary — which is precisely the case the 21-wide window exists to catch. A neighbour
that arm `b` ranks 21st and arm `a` ranks 22nd is admissible under reordering for `a` but invisible
to `a`'s own window, so `hi_a`/`lo_a` are computed over too small a set and the item is **not**
classified as a tie casualty.

**Wrong-verdict path.** Under-counting casualties inflates `outside = mm & ~cas` (`:774`), which
appends a `GATE-ZEROOP` drop, which drops the lineage on both datasets, which by §5.6 rule 3 turns a
CLOSE into a **HALT**. This runs in the same one-directional sense §5.9 item 5 records for the cap
("it can only convert REPORT → HALT") — so it cannot manufacture a SURVIVE — but it can suppress the
closure the falsifier exists to publish.

The analytic machinery itself is correct: the weight vector `[20…1, 0]` at `:248-249` matches
`mechfix_ops.deployed_vote`'s `w = _rank_weights(20)` and `w.sum() = 210`; `signed = (2·lab−1)·sim`
matches the deployed statistic; and the rearrangement-inequality pairing at `:260-263` gives the
true extrema, satisfying §13.1 item 10's "ANALYTICALLY, not by enumerating the `g!` orderings".

**Resolution path.** Retrieve on the union: take the two `gi` index sets, union them per item, and
score both arms' votes over that common neighbour set, or widen the retrieval `topk` enough that
each arm's window provably contains the other's top-21.

### I-5 — `GATE-POP` runs after four gates that consume the arena population.

**Files:** `c06_falsifier_arena.py:1055-1079`, `:622-641`.

§13: "`GATE-SHA` once in the driver before any of them and **`GATE-POP` before any
population-consuming gate**." §13.1 item 11 repeats it and adds "asserts row identity by **index
set**."

`GATE-POP` is implemented inside `run_lineage` (`:622-641`), which `main()` reaches at `:1079`.
Before it: `gate_idparity_zeromask_nullremoved` (`:1062`), `raw_leg` (`:1068`) — which runs
`GATE-C01PARITY`, `GATE-ROWSUBSET` and `GATE-RHORAW`, the last of which computes `ρ` over the arena
rows and asserts against 26 frozen values — `GATE-NULLREMOVED` (`:1070`), and `GATE-FLOOR`
(`:1075`). All four consume the realised population.

Separately, `GATE-POP` as coded asserts counts (`n_arena`, class counts, majority, band, tie cap)
but never asserts that the head leg and the raw leg sit on **identical row index sets**, which §6's
gate row names explicitly. In this implementation both legs derive from the same `keep` vector so
identity holds by construction, but it is not asserted, and §13.1 item 8's standard is "with an
assertion, not a comment".

**Resolution path.** Hoist the population computation into a standalone `gate_pop(ds, ro, keep)`
called immediately after `gate_idparity_zeromask_nullremoved` and before `raw_leg`, and add the
index-set identity assertion between the raw-leg `keep` and the head-leg `keep`.

---

## 4. MINOR

**M-1 — Thirteen of eighteen declared C01 constants are read but not asserted.**
`c06_falsifier_arena.py:384-395` asserts five of the entries under the config key
`frozen_c01_constants_read_and_asserted`. The other thirteen — `minimum_gain_over_strongest_control`,
`minimum_net_fixes`, `gain_controls`, `angles_degrees`, `rotation_arm_prefix`, `statistics_seed`,
`n_bootstrap`, `bootstrap_lower_quantile`, `bootstrap_upper_quantile`, `n_id_hash_permutations`,
`holm_alpha`, `holm_metrics`, `topk_deployed`, `small_set_comparison_operator` — are read straight
off the C01 config with no comparison to the C06 config's copies. `GATE-SHA` covers the C01 config's
digest so a drift is caught, but the config key name over-claims what the code does, and §13.1 item
5b's verb is "asserted equal to it". *(All thirteen were verified equal by hand at review time.)*

**M-2 — The C01 module and config are imported and used before `GATE-SHA` verifies their digests.**
`main()` calls `load_frozen()` (`:1037`) before `gate_sha()` (`:1038`), and `load_frozen` imports
`c01_policy_contrast_a0`, calls its `import_compute_modules`, and reads `c01_a0_v2.json`. This holds
in the `--gate-sha-only` process too, so no process checks before importing. The mint driver does it
correctly (`assert_frozen()` at module scope, `:97`, before `import headspace_mint`). Contained,
because `gate_sha` still HALTs before any decision quantity exists.

**M-3 — `RC=$?` is unreachable and the driver's last three lines are dead code.**
`scripts/slurm/c06_falsifier_cpu.sbatch:10` sets `set -euo pipefail`; `:132-138` runs the arena, then
`RC=$?`, then echoes and exits. Measured: under `set -e` a non-zero command aborts the script before
the assignment — a minimal reproduction printed nothing and exited `2`. So on a HALT the driver's
`[c06] driver done rc=…` line never appears; on success `RC` is always `0`. The job's exit status is
still correct.

**M-4 — Dead first assignment of `window`.** `c06_falsifier_arena.py:758-759` computes
`window = algebra_res * sqrt(max(...) if False else 1.0)` and immediately overwrites it at
`:760-761`. The surviving value is correct (`√2048 · residual`, §6.5's unit correction). The dead
expression should go.

**M-5 — `GATE-NESTED`'s check-count predicate is tautological.** `:702` accumulates
`checked += n_arena` inside `for seed … for arm in self.all_arms`, and `:703` tests
`checked != n_arena * len(self.all_arms) * len(self.SEEDS)` — the same arithmetic. It can never fire.
§6's gate row asks that "check count equals the item count", which needs an independent count (e.g.
the number of items actually assigned a prediction by a fold's head).

**M-6 — §13.1 item 24's `:2724`-equivalent consistency assertion is not implemented.** Item 24
requires "C01's `:2724`-equivalent consistency assertion (S7's reference must equal the selected
control) is implemented". `evaluate_conditions` reads `cell["reference"]` for S7 (`:990`) and
`run_lineage` uses the same value for S6 and `GATE-SELFTEST`, so the property holds by construction
— but the assertion the item names does not exist.

**M-7 — Finiteness is asserted on two quantities, not on the decision quantities.** `finite()`
(`:98-103`) is called once, on `acc_s`/`mf1_s` (`:713`). §13.1 item 16's last limb requires "no
**decision** quantity — not only no gate quantity — reaches a comparison non-finite": S4's `lower`
and `one_sided_raw_p`, S5's `observed`/`p95`, `net_s`, and S7's `fixed_fraction` and `threshold` are
never checked. All are structurally finite in the current code (every division in
`resampled_macro_f1` is `np.where`-protected), so this is a missing assertion rather than a live
path.

**M-8 — The S5 `keep` is read from the config rather than computed from the arena.**
`main():1110-1112` rebuilds `keep` from `population_derived_constants[ds]["removed_zero_rows"]`,
whereas the earlier `keep` (`:1062-1064`) derives from the measured zero rows. The two agree because
`GATE-ZEROMASK` asserts it upstream in the same run, but §13.1 item 5a's verb for population-derived
constants is "computed from the arena, **not read**". Simplest fix: keep the first `keep` in `cells`
and reuse it.

**M-9 — No runtime fidelity assertion on the vectorised macro-F1 (advisory; the path measures
bit-identical).** See §5 below for the measurement. ERRATUM 1 obligation 3 pins the function as
`mechfix_ops.macro_f1` and `resampled_macro_f1` (`:300-337`) is a vectorised replica — the
re-implementation-without-anchor class round 2's C-3 named. It is bit-identical over every input
tried, but nothing in the run would notice if a future numpy changed the association. A spot-check
of, say, 32 sampled draws against direct `mechfix_ops.macro_f1` calls, HALTing on any non-zero
difference, costs milliseconds and turns the docstring's claim into a run-time fact.

**M-10 — The arena's import set is a strict subset of §13.1 item 27's.** Measured after
`load_frozen()`: `numpy`, `torch`, `faiss`, `sklearn`, `scipy`, `threadpoolctl`, `mechfix_ops`,
`mechnov_pairverify`, `c01_policy_contrast_a0` are resident; **`headspace_mint` and `vsw_pregate`
are not**, and `runtime_block()` is never called, so the verdict carries no runtime provenance
block. Item 27: "A lineage that trims this set must re-measure `U11`'s arena class and re-carry
Phase 1g." The direction is cost-conservative and `sklearn` — the term round-12 I-1 corrected the
unit for — is still pulled in via `mechnov_pairverify:46-49`, so `U11 = 3.8 s` remains an upper
bound. Flagged because item 27 makes the set itself the deliverable.

**M-11 — Four decision-relevant constants are module literals duplicating config entries.**
`c06_falsifier_arena.py:47-50`: `TOPK = 20` duplicates `frozen_c01_constants_read_and_asserted.
topk_deployed`; `GATE_ALGEBRA_BAR = 2e-6` duplicates `gates["GATE-ALGEBRA"]["bar"]`;
`TIE_RANK_WINDOW = 21` and `UPPER_ARENA_BAR = 0.98` have no config entry at all. All four currently
agree with the design. This is the stale-literal class the implementation's own first dry run caught
in the config; reading them from the sha-gated config closes it. (`PROJECTED_SECONDS = 2929.9` is a
deliberate exception — §9 states the literal is carried in both files and it matches.)

**M-12 — Resource accounting drift beyond §8 (folded into H-4's resolution).** Six arena-side ro
cache loads against Phase 1c's one; ~270 mint-array materialisations against Phase 1f's 150. Both
are sub-second and neither affects a verdict; listed so §8's counts and the code can be reconciled
in one pass.

---

## 5. Priority-target dispositions

### Priority 1 — the vectorised macro-F1 replication: **CLEARS on measurement.**

`resampled_macro_f1` (`:300-337`) is bit-identical to direct `mechfix_ops.macro_f1` calls over
**12,200 resamples across five regimes**, including every adversarial case the erratum and its
review name:

```
   n=743 realistic, B=600             max|diff|=0.000e+00  non-bit-identical 0/600
   exactly-tied predictions           max|diff|=0.000e+00  non-bit-identical 0/600
   n=6 degenerate/absent-class draws  max|diff|=0.000e+00  non-bit-identical 0/4000
   gold all-negative                  max|diff|=0.000e+00  non-bit-identical 0/3000
   n=40, single positive              max|diff|=0.000e+00  non-bit-identical 0/4000
   TOTAL non-bit-identical draws across all cases: 0
```

The four-rounded-ops association §5.9 item 10 quantifies is reproduced exactly, not approximately:
`pr = tp/(tp+fp)`, `rc = tp/(tp+fn)`, `2·pr·rc/(pr+rc)`, then `(f₀+f₁)/2` in that order and that
association, on operands that are exactly-representable small integers, so the tie-escape behaviour
at `±1.11e-16` is preserved. The seed mean is inside the statistic and is taken with the same
reduction order (`np.mean` over axis 0 of a `(3, B)` array vs `np.mean` of a 3-list — both
`((a+b)+c)/3`). The accuracy leg (`:281-297`) is §5.4's frozen expression verbatim, un-re-associated,
as obligation 1 requires. `one_sided_p` (`:340-343`) matches
`c01_policy_contrast_a0.py:1769` exactly, and the lower bound uses `np.quantile(d, lower_q)` with
C01's own `bootstrap_lower_quantile = 0.05` — the same call C01's `paired_bootstrap` makes.

No wrong-verdict path in S4's adverse count. The only residue is **M-9**, the absent runtime
assertion.

### Priority 2 — implementation note (a), the pre-patched `model_pass`: **patch mechanics sound; the ro-forward routing is C-1.**

The patch protocol is correct and I could not break it. The driver saves the genuine
`run_rac.model_pass` at `:174` **before** installing `_capture` at `:181`; `HM.main()` then reads
`run_rac.model_pass` (now `_capture`) as its own `_ORIG_MODEL_PASS` (`headspace_mint.py:242`) and
chains `_model_pass_spy → _capture → genuine`. Both spies therefore hold **the same object**, and
because Python holds a reference rather than a copy, the post-training parameter state is what both
see. `headspace_mint.py:296-297` calls `model.eval()` on that object before computing `K_train`, and
the C06 driver calls `model.eval()` again at `:203`; the forward at `:206-208` is
`torch.no_grad()`, `model(sp[1], sp[2], return_embed=True)`, `.astype("float64")` — character-for-
character the frozen `keys_of` at `headspace_mint.py:300-303`. So there is no train/eval asymmetry
and no dtype drift between `K_train` and the driver's ro keys, which matters because on Head-R
`h_std` **is** `K_train`.

The one-final-`.npz` layout matches what the arena reads: `K_train`, `K_dev`, `lab`, `lab_dev`,
`fold_of`, `fit_idx`, `h_std`, `h_ow`, `meta` (`:246-251`); the arena reads `h_std`/`h_ow`
(`:601-602`), `K_train` (`:808`), `fold_of` (`:643`, `:803`) and `meta` (`:473`), and
`headspace_fidelity.py:65-68` reads `meta["eval_curve"]`, `meta["secs"]`, `n_dev`, `n_train`,
`head_dim` — all preserved by the driver's re-dump. `os.replace` makes the file atomic, so the
resume predicate at `:148` ("its presence is a complete record") holds.

`ro_std_is_k_train` (`:218-220`) is sound: `mechnov_pairverify.DATASETS[*]["cache_dir"]` is absolute
(`os.path.join(REPO, …)`) and `headspace_mint` does `os.chdir(REPO)` at import, so the `realpath`
comparison against the sbatch's absolute `--train-cache` resolves correctly — measured `True` on the
Head-R path, `False` on Head-N.

What fails is downstream of all of this: **C-1**.

### Priority 3 — implementation note (b), the fidelity symlink aliases: **CLEARS.**

`ln -sfn <abs target> <alias>` (sbatch `:112-113`) with an **absolute** target, `-f` to overwrite and
`-n` so an existing symlink is replaced rather than dereferenced. Re-running the driver re-points the
aliases unconditionally, so no stale resolution survives a resume; and because the mints themselves
are written by `os.replace`, an alias can never point at a partial file. The six aliases name Head-N's
six full mints and nothing else: `mint_${DS}_N_s${SEED}_ffull.npz → mint_${DS}_s${SEED}_ffull.npz` for
`DS ∈ {hatemm, zh}`, `SEED ∈ {0,1,2}`. The naming matches the frozen reader:
`headspace_fidelity.py:65` formats `mint_{}_s{}_ffull.npz` with `--dataset`, whose `choices` are
`sorted(FLOOR)` = `{hatemm, zh}` (`:28-33`) — the same keys the sbatch loops over. `--seeds "$SEED"`
makes each process read exactly one alias. No Head-R mint is reachable through `$FIDVIEW`.

Split discipline holds: `floor_dev_curve` (`:36-51`) is a hard filter that drops any line failing
`VAL_RE` before parsing, and `proxy` comes from `meta["eval_curve"]`, so no `dev_seen` file is
opened — §12's "`+ 0`" term. The six `.trainlog` opens are what `c09guard.is_banked_trainlog` would
count as §12's `banked_trainlog_opens = 6`, if the ledger were wired (H-2).

### Priority 4 — S4/Holm machinery: **CLEARS, measured.**

Probe 3, driving the real `s4_family` and C01's real `holm_adjust`:

```
both lineages live   family=92 TESTED=92 NOT_TESTED=0  (p==1: 0)  NOT_TESTED holm_reject=0
Head-N dropped       family=92 TESTED=46 NOT_TESTED=46 (p==1: 46) NOT_TESTED holm_reject=0
```

* **Family assembly is exactly 92 on both paths.** `(6+6)×2 + (5+6)×2 = 46` per lineage from
  `cfg["arms"]["comparators"]` plus `rotation_family` across `holm_metrics`, doubled over lineages —
  §5.5's `23 comparisons × 2 metrics × 2 lineages`. The `len(entries) != 92` guard at `:884` is a
  real predicate, not a tautology.
* **The drop path pads rather than shrinks.** A dropped lineage's 46 hypotheses are `NOT_TESTED`
  with `one_sided_raw_p = 1.0` and `lower = None`, exactly §5.5 and §13.1 item 21. `holm_adjust`
  sorts ascending, so the `p = 1` entries land last and cannot raise the running maximum for the
  witness's hypotheses; none of them is marked `holm_reject`.
* **The one-sided p is C01's own form.** `(1 + Σ[Δ ≤ 0]) / (B + 1)` at `:343` against
  `c01_policy_contrast_a0.py:1769`.
* **The shared draws matrix is built once per dataset** (`:869`, guarded by `draws is None`) from
  `np.random.default_rng(c01cfg["statistics"]["seed"])` = 20260728, and is reused across every
  comparator **and** both lineages — §5.4's draw-sharing clause. It is also correct on the drop path:
  `rng` is constructed fresh per `s4_family` call and consumed only by that one `integers` call, so
  the matrix is identical whether lineage N or lineage R is the first to instantiate it.
* **The lower-bound quantile** is `np.quantile(d, bootstrap_lower_quantile)` = C01's own call.
* **S1–S7 wiring.** Every conjunct was individually falsified on a cell that otherwise clears:

```
baseline clears: True
break displacement's S5   -> common_displacement S5 = False   (§5.2 S5 spans BOTH real arms — correct)
one hypothesis NOT_TESTED -> S4 = False
one hypothesis lower=None -> S4 = False
net_s[0]=0                -> S6 = False
all fixes in the small set-> S7 = False  {'n_fixed': 35, 'fixed_fraction': 1.0, 'dominated': True}
```

  The `str((lineage, A, comp, metric))` key format used by `evaluate_conditions` (`:970`) matches
  what `s4_family` emits (`:887`) — verified by lookup, not by inspection. S6 reads per-seed integer
  nets against the `select_strongest_ordinary_control` reference. S7's five frozen parameters are all
  present and all read from the sha-gated C01 config: quantile `0.1`, dominance `0.5`, the `<=`
  operator (`:988`), the `common_displacement`-only arm scope (`:982`), the 3/3 seed axis, the
  head-space one-block statistic (`:607-609`), and the zero-fix convention (`:993-994`).
* **The S5 "both real arms" reading is correct, not a partner-veto defect.** `s5_ok` (`:977-979`)
  iterates `cfg["arms"]["real"]` rather than the candidate arm `A`, so `displacement`'s null failure
  vetoes `common_displacement`. This looked like the C09 lineage's "undefined p vetoing its partner"
  but is what §5.2's S7 row requires verbatim: "**both** real arms exceed the 95th percentile … and
  the shuffle comparison Holm-rejects", restated at §5.4.1's p95 bullet. **No finding.**
* S5's own family is 4 per `(dataset, lineage)` with its own `holm_adjust` (`:944-946`), the p95
  convention is `bootstrap_upper_quantile = 0.95`, the p-form is `(1 + #{null ≥ obs}) / 257`, and
  `id_hash_permutation` is called with `fixed_indices=()` — §5.4.1 on every clause.

### Priority 5 — the verdict emitter: **truth table CLEARS; two escape routes found.**

The §5.6 combination block (`:1121-1134`) was replicated and exercised over all five reachable
states:

```
  passed={N:True, R:True}  clears={}          -> CLOSE    design CLOSE    OK
  passed={N:True, R:True}  clears={R:True}    -> SURVIVE  design SURVIVE  OK
  passed={N:False,R:True}  clears={}          -> HALT     design HALT     OK
  passed={N:False,R:True}  clears={R:True}    -> SURVIVE  design SURVIVE  OK
  passed={N:False,R:False} clears={}          -> HALT     design HALT     OK
```

Rules 1–3 are implemented exactly, including the case §5.6 works through by hand ("Head-N is dropped
on both datasets; rule 2's *both lineages passed* is false; **the run HALTs**"). The dataset-axis
drop rule is right: `main():1086-1094` pools `drop_reasons` across **both** datasets before setting
`passed[lineage]`, so a lineage failing on one dataset is dropped on both, and SURVIVE at `:1126`
requires `clears` on **both** datasets for the same arm. `INSTRUMENT_FAILED` is written for dropped
lineages (`:1107`) and they are excluded from S5 and S1–S7 (`:1105-1108`). No absent or NaN quantity
reaches a verdict outside the declared-drop exemption: dropped lineages never enter
`evaluate_conditions`, and a `NOT_TESTED` S4 record fails the `status` test before its `None` lower
bound is dereferenced (`:972-976`) — measured above.

The two escapes are **H-1** (per-lineage gate failures that HALT instead of dropping, which can turn
a lawful SURVIVE into HALT) and **I-1** (no verdict artifact on any HALT path).

### Priority 6 — resource and run boundary: **process order and split guards clear; ledger, heartbeat and §8 do not.**

* **73 processes in §13's order** — `1 × --gate-sha-only`, then 66 mints (36 Head-N incl. six
  `fold=-1`, 30 Head-R), then 6 fidelity, then 1 arena. The sbatch asserts `MINT_N == 66` and
  `FID_N == 6`. No `--gres`, no `--time`, no `--array`, no `--dependency`, no `--requeue`; 8 CPU /
  32 GB matching the config. `GATE-SHA` runs once in the driver before anything else, and
  `--gate-sha-only` hashes 37 artifacts (21 digests + 6 arena OUT JSONs + 10 `vsw_ckpt` npz),
  reconciling with `_gate_sha_count`.
* **`PYTHONPATH` layer 3** — exported at `:37` before every python process; measured to resolve to
  `c09_guard/sitecustomize.py` with no competing module. But activation is never asserted (H-2).
* **§12's three layers, tested adversarially.** I could not construct a code path that opens a test
  or `dev_seen`-ro file. Layer 1 is `headspace_mint`'s `torch.load` guard, inherited unchanged and
  used by `load_split`. Layer 2: the mint's `split == "train"` assertion is on a literal
  (`c06_falsifier_mint.py:121`), the `--train-cache` filename must match `train_*.pt` (`:163`), and
  the arena's `load_ro` builds its paths from the frozen `DATASETS` table with a `train_` basename
  assertion (`:498`). The arena reads only: the two train ro caches, the banked mint `.npz`, the six
  banked arena OUT JSONs, the ten `vsw_ckpt` npz, and the sha-gated configs. `--dry-parity-only`
  opens no mint. §13.1 item 26 holds: no ro-derived arm is voted outside the arena phase, and
  `gate_floor` votes only on Head-N `K_train`. The dev caches are *hashed* by `GATE-SHA`, never
  loaded — which is nonetheless the collision H-2 records.
* **Resume semantics** — the mint's `os.path.exists(a.out)` skip (`:148`) is sound because the final
  `.npz` is atomic and complete; `GATE-FOLD` re-reads parity from all 66 banked files including
  skipped ones (§13.1 item 18). The arena has no resume and needs none. The fidelity aliases survive
  re-execution (priority 3).
* **Failures:** H-2 (ledger), H-3 (heartbeat coverage), H-4 (§8 Phase 3), M-3 (`RC=$?`).

### Priority 7 — config vs design: **clean on values; two verbs over-claim.**

Every constant in `configs/c06/c06_falsifier.json` was checked against V15E1 and against the frozen
C01 config. All match: the 13 `rho_raw_frozen_6dp` rows against §6.1's table; `rho_star`
`0.968176`/`0.977223`; `GATE-FLOOR`'s six accuracies and six macro-F1s against §6's row; the
population block (`743`/`579`, `(297,446)`/`(180,399)`, majorities `0.6003`/`0.6891`, bands
`[0.6203, 0.98]`/`[0.7091, 0.98]`, caps `7`/`5`, removed rows `{355}`/`{}`); the arm lists and both
comparator sets (6 and 5, `avg_score` in both via `gain_controls`); `holm_family_size_per_dataset`
92; `s5_family_size_per_cell` 4; the erratum block's `accuracy_leg_moved: false`; `design_sha256`
matching the frozen document byte-for-byte; and `projected_seconds` `2929.9` matching both §8 and the
arena literal. The `_gate_sha_count` arithmetic (7+6+8 = 21, +16 = 37) matches the code's count.

Two entries describe behaviour the code does not have: `frozen_c01_constants_read_and_asserted`
(M-1, 5 of 18 asserted) and the entire `ledger` block, whose eight predicates the code neither
measures nor compares (H-2). `population_derived_constants.*.full_majority` has no reader (I-2).

---

## 6. §13.1 handoff — all 28 items, disposition

| # | item | disposition |
|---|---|---|
| 1 | mint imports `headspace_mint` sha-asserted; nothing outside `--train-cache` differs | **VERIFIED** — `assert_frozen()` at module scope; `HM.main()` called, never re-implemented |
| 2 | `--train-cache` overrides only the training cache; §12's counts match the code | **FAILED** — the override survives into the driver's own ro loads (**C-1**); §12's counts are unimplemented (**H-2**). `model_name` and the dev load are untouched |
| 3 | no branch on cache filename or suffix | **VERIFIED** — and the unconditionality is C-1's mechanism |
| 4 | `GATE-FLOOR` mints and Head-R mints use the same function | **VERIFIED** — one driver, one code path |
| 5 | (a) population constants computed; (b) C01 constants read **and asserted** | **PARTIAL** — (a) M-8; (b) M-1 |
| 6 | `GATE-SELFTEST`'s `n` is the arena size; no banked 744 leaks | **VERIFIED** — `n_arena * (acc_s[A] − acc_s[ref])` |
| 7 | ρ over the 743/579-row matrices | **VERIFIED** — `raw_arena` is the row-subset build on HateMM; head arms are `[keep]` |
| 8 | explicit boolean mask everywhere with an assertion; ε = 1e-12 from the sha-gated config | **VERIFIED** — `ArmBuilder.l2:126-130` asserts dtype **and** shape; ε from `transforms.normalization_epsilon` |
| 9 | the `n = 744` build exists only inside `GATE-C01PARITY`/`ROWSUBSET`; nothing votes on it | **VERIFIED** — nothing votes on any raw arm at all |
| 10 | tie diagnostic: ranking, residual, analytic worst case, bounded report branch, per-`(ds,seed,lineage)` aggregation | **PARTIAL** — analytic ✓, residual `√2048·max\|Δk\|` ✓, aggregation ✓ (matches §8's `7×6 + 5×6`); ranking is per-arm, not the union (**I-4**) |
| 11 | `GATE-POP` before any population-consuming gate; row identity by index set | **FAILED** — **I-5**, both limbs |
| 12 | all six §9 items, the `RuntimeError` wrapper, `buffering=1` never re-wrapped, driver echo, append across all 73, frozen denominator | **FAILED** — **H-3** (72/73 silent), **I-1** (context not in the JSON). The arena's own handle, the denominator and the driver echo are correct |
| 13 | the fold axis — arms and ρ rebuilt from each of the 60 fold matrices; no cross-fold vote | **VERIFIED** — `head_cell_arms` per fold; bank `X[fit]`, query `X[ho]`, both under fold `f`'s head |
| 14 | guard arms via the rotation route, never aliased, all four voted | **VERIFIED** — `_rotation` at `:191-193`; all four in `views`, hence voted at `:670` |
| 15 | S7's full parameter set + `tiny_ok`'s non-carriage | **PARTIAL** — all six parameters correct; `tiny_ok` not computed ✓, but its run-time warrant is missing (**I-3**) |
| 16 | statistics: §5.4 bootstrap (erratum'd), one-sided p, Holm; §5.4.1's S5; per-seed `GATE-SELFTEST`; no non-finite decision quantity | **PARTIAL** — the statistics all verified and the mF1 leg measured bit-identical (§5 above); the finiteness limb is **M-7** |
| 17 | population constants, extended | **PARTIAL** — as item 5 |
| 18 | `GATE-FOLD` under resume, all 66 including skipped | **VERIFIED** — `:462-487` re-reads `meta` + `fold_of` from every file |
| 19 | one construction; endpoint pre-normalisation; `GATE-C01PARITY` at bit-exactness | **VERIFIED** — one `build_views`; `std`/`ow` are `l2_rows`-normalised before every contrast (`:157-160`); predicate is `worst != 0.0`, no tolerance |
| 20 | the `(dataset, lineage)` cross; drop propagates across datasets; no one-dataset verdict path | **PARTIAL** — propagation ✓ (measured), but **H-1** breaks the scoping for two gates |
| 21 | the dropped lineage's quantities: exempt, excluded, `NOT_TESTED` p = 1, family frozen at 92 | **VERIFIED** — measured (§ priority 4) |
| 22 | all three key forwards inside the mint; one `.npz` per mint; `GATE-FLOOR`'s vote in the arena; Phase 1f's 150 | **PARTIAL** — placement ✓ on every limb; the `h_ow` forward reads the wrong file (**C-1**); materialisations ~270 vs 150 (**M-12**) |
| 23 | the arm→formula map pinned by `GATE-C01PARITY` and nothing else | **VERIFIED** — `common_interaction` is `paired(common, l2(common ⊙ displacement))` (`:168-169`, `:182`), the instance two reviewers mis-derived |
| 24 | the reference arm is run-time; S6/S7/`GATE-SELFTEST` share it; `:2724`-equivalent assertion; name on the verdict face | **PARTIAL** — shared ✓, on the face ✓ (`reference_arm`), assertion absent (**M-6**) |
| 25 | the head-space displacement tail per cell | **FAILED** — **I-3** |
| 26 | the trained-head blindness boundary | **VERIFIED** — no ro-derived vote outside the arena; `GATE-FLOOR` votes only on native keys |
| 27 | the arena's import set | **PARTIAL** — **M-10**; measured, cost-conservative, `sklearn` still resident |
| 28 | `PYTHONPATH` wiring **and** layer 3 active in all 73, not merely importable | **PARTIAL** — export verified and resolution measured; activation never asserted (**H-2**) |

**Could not verify: none.** Every item was reached by reading or by measurement. Items 12, 15 and 25
were partly assessed by absence — the code contains no site implementing them, which is a stronger
statement than a failed check.

---

## 7. What a re-review must re-execute

C-1's fix touches the mint driver, so probe 1 must be re-run and must show markers `2.0` / `3.0` on
the Head-R path. H-4's fix touches `ArmBuilder`'s call surface, so `--dry-parity-only` must be re-run
and `GATE-C01PARITY` must still measure `max|diff| = 0.000e+00` on both datasets — that gate is the
sole warrant transferring the parity guarantee into head space (§13.1 item 19) and no change to the
builder may be landed without re-firing it. H-2's `dev_path_opens` collision needs a design ruling
before code.

Nothing in this review was executed against the real ro caches, the real heads, or the real banked
mints; a GO from this lineage clears the implementation for freeze and submission, not the numbers.
