# C06 `$0` falsifier — IMPLEMENTATION RECORD

*Frozen design:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md`, sha256
`75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228` — **GO at round 15
(0C/0H/0I)**, verified unchanged at the end of this work.
*Phase:* implementation of the four new-code artifacts §11 declares confined to the battery.
*Date:* 2026-08-04. *Node:* `foscsmlprd01`, conda `HateVideo`, Python 3.11.8.
*Status:* **files written, `$0` dry checks passed, NOT frozen, NOT submitted, NOT authorized.**

**No SLURM submission, no arena run, no mint, no commit, no `TARGET_STATE.json` edit.**
`artifacts/c06_falsifier/` does not exist; nothing was written into the repository outside the
four files below. The next step is the **separate independent code/resource review lineage**
(house rule from C09, where that lineage caught two wrong-verdict paths after 17 clean design
rounds), then freeze, then main-dialogue authorization.

---

## 1. The four files, as written

| # | path | sha256 | bytes | lines |
|---|---|---|---|---|
| 1 | `scripts/analysis/c06_falsifier_mint.py` | `1084b5be8c11ad60085115504e999b338db481801614452526084b87d1b3a1d0` | 11806 | 260 |
| 2 | `scripts/analysis/c06_falsifier_arena.py` | `3e423bc66d93d9da549f777c1941d53dbbde74e55da101c89c21d470e6a9eada` | 57730 | 1115 |
| 3 | `configs/c06/c06_falsifier.json` | `a0ebe0dc29e3e820edc258bf96551fa5d68618f2a960ee010f9e49650da4bc56` | 13039 | 246 |
| 4 | `scripts/slurm/c06_falsifier_cpu.sbatch` | `76d061daf62c51dae584387160924cae482ca7ea20710423b443424b2a21b634` | 5942 | 138 |

---

## 2. Dry-check results — all `$0`, login-node, payload-real

Run under `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=""`,
`PYTHONPATH` including `scripts/analysis/c09_guard`. Every output below was produced by the
**actual new files**, not by a probe script standing in for them.

### A. Syntax and import — 4/4 clean

| check | target | result |
|---|---|---|
| `python -m py_compile` | both `.py` files | **OK** |
| `bash -n` | the sbatch | **OK** |
| `json.load` | the config | **OK** |
| module exec (`importlib`, real import of every dependency) | both `.py` files | **OK** |

The module-exec check is the load-bearing one: it executes the frozen-sha assertions at
`c06_falsifier_mint.py` import time and pulls in `headspace_mint`, `mechnov_pairverify`, `torch`
and `run_rac`'s dependency chain, so an import-order or deferred-import defect surfaces here
rather than at run time.

### B. `GATE-DET1` + `GATE-SHA` — **37/37**, from the actual arena file

```
2026-08-04T11:45:59+00:00 | GATE-DET1 | 1/1 | 0.0s | 0.000x | thread env verified
2026-08-04T11:46:02+00:00 | GATE-SHA  | 37/37 | 2.8s | 0.001x | all frozen digests match
2026-08-04T11:46:02+00:00 | GATE-SHA-ONLY | 1/1 | 2.8s | 0.001x | driver precondition satisfied
exit=0
```

**This check caught a defect in my own config on its first run.** It reported `33/33`, because
`configs/c06/c06_falsifier.json` listed **four** input caches where §11 declares **eight** — the
four native caches (`train` and `dev_seen` per dataset) were missing, so `GATE-FLOOR`'s and the
mint driver's own inputs would have been unhashed. Fixed by adding the four digests from
§11:1647-1650; the re-run reports 37/37. The count decomposes as §11 states: 7 imported modules
+ 6 read-for-definitions + 8 input caches = 21 digests, plus 16 banked artifacts (6 arena OUT
JSONs + 10 `vsw_ckpt` npz) whose presence the gate asserts. `21 + 16 = 37`.

### C. `GATE-C01PARITY` / `GATE-ROWSUBSET` / `GATE-RHORAW` — payload-real, both datasets

**This is the single most load-bearing verification in the implementation**, and it is the one
§3.4 and §13.1 item 23 exist to force: the arm→formula map is pinned by `GATE-C01PARITY` against
`prepare_views` **and by nothing else**, so a lineage that reimplements from prose produces a
wrong arm and a build that still looks like it passes.

```
2026-08-04T11:46:18+00:00 | GATE-C01PARITY | 1/1 | 15.2s | 0.005x | hatemm max|diff|=0.0 in 12.3s
2026-08-04T11:46:24+00:00 | GATE-ROWSUBSET | 1/1 | 21.3s | 0.007x | hatemm bit-exact bridge
2026-08-04T11:46:24+00:00 | GATE-RHORAW    | 13/13 | 21.6s | 0.007x | hatemm 13 arms at 4 dp
2026-08-04T11:46:30+00:00 | GATE-C01PARITY | 1/1 | 26.9s | 0.009x | zh max|diff|=0.0 in 5.0s
2026-08-04T11:46:30+00:00 | GATE-RHORAW    | 13/13 | 27.1s | 0.009x | zh 13 arms at 4 dp
2026-08-04T11:46:30+00:00 | DRY-PARITY-COMPLETE | 2/2 | 27.2s | 0.009x | no arm accuracy computed
exit=0
```

* **`GATE-C01PARITY`: `max|diff| == 0.0`**, the single predicate §6 requires, over all 13
  key-space arms on **both** datasets, at `n = 744` one-hot `{355}` (HateMM) and `n = 579`
  all-False (MHC-ZH). Passed **first attempt**, which means the two wrong-verdict paths the
  design records are both avoided: `common_interaction = paired(common, l2(common ⊙ displacement))`
  (round-6 C-1) and the **endpoint pre-normalisation** `std[m] := l2_rows(standard[m])`
  (round-7 C-1, the one a `2e-6` tolerance would have admitted).
* **`GATE-ROWSUBSET`**: the HateMM `n = 743` all-False build is bit-identical to the `n = 744`
  one-hot build restricted to the 743 surviving rows, all 13 arms.
* **`GATE-RHORAW`**: all 26 `ρ_raw` reproduce §6.1's frozen table at 4 dp under the frozen
  float64-over-float32 reduction.

### D. §7.4 mask convention and §3.7 population constants — re-derived

| dataset | `zero_mask=None` into `prepare_views` | arena `n` | class counts | majority | tie cap | zero rows | config match |
|---|---|---|---|---|---|---|---|
| HateMM | **DIES** (`exact-zero mask diverged`) | 743 | (297 pos, 446 neg) | `0.600269 → 0.6003` | 7 | `{355}` | **yes** |
| MHC-ZH | **DIES** (`derived exact-zero mask preservation failed`) | 579 | (180, 399) | `0.689119 → 0.6891` | 5 | `{}` | **yes** |

Full-population majorities `0.599462 → 0.5995` / `0.689119 → 0.6891`, matching §3.7's
`GATE-DOMAIN` companion. Every population-derived constant in the config was **computed from the
caches** here, never read — which is the verb §13.1 item 5a requires and which `GATE-POP` asserts
again at run time.

### E. What the dry checks could **not** exercise

Everything downstream of the 66 mints, because the mints need the SLURM run: the head-leg
assembly, `GATE-FLOOR` / `GATE-POP` / `GATE-NESTED` / `GATE-SELFTEST` / `GATE-ARENA` /
`GATE-ORBITDISP` / `GATE-ZEROOP` / `GATE-ALGEBRA` / `GATE-LEDGER`, S1–S7, the 92-hypothesis Holm
family, the 256-draw shuffle null, `GATE-DOMAIN`, `GATE-DEVFID` and the verdict emitter. They are
written and syntax-clean but have executed on nothing. **Stated plainly so no reader infers more
coverage than exists.**

---

## 3. Blindness statement

**No arm accuracy was computed, printed or logged at any point in the dry checks.** Verified by
machine, not asserted: a grep of the dry-check progress file for every decimal in the closed
interval `[0.6, 0.99]` returns **NONE**, and the only line matching `acc|accur|mF1` is the
battery's own `no arm accuracy computed`. The `--dry-parity-only` path opens no mint, constructs
no head-space arm and calls `deployed_vote` zero times; the `--gate-sha-only` path hashes files.
No test-split file was opened by any check, and no `dev_seen_*-ro_*` path is reachable from the
code paths exercised.

---

## 4. Design ambiguities hit, and what was done

Three points. **None was improvised around**: two are implementation decisions taken because the
design as written cannot be satisfied literally, both documented in the source at the point of
decision; the third is a question I could not resolve from the document and which should be ruled
on **before** the run.

### (a) §13.1 item 22 vs `headspace_mint.py:321-325` — the mint `.npz` contents

**The conflict.** Item 22 requires all key forwards to happen inside the mint process and each
mint to write its key matrices into its own `.npz`, citing that `np.savez` as the pattern. §8
Phase 1b prices three key forwards per Head-N fold mint `{native, ro_std, ro_ow}` and two per
Head-R fold mint. But the frozen savez writes only `K_train`/`K_dev`/`lab`/`lab_dev`/`fold_of`/
`fit_idx`/`meta`, and `headspace_mint.main()` neither returns the trained model nor exposes a
hook — so `h_std`/`h_ow` cannot be added without editing a frozen module, which **§13.1 items 1
and 4 forbid** ("not re-implemented"; "the same function, not two copies").

**Resolution used.** The driver (i) captures the trained head by pre-patching
`run_rac.model_pass` **before** calling `main()` — `main()` then wraps this driver's spy as its
own `_ORIG_MODEL_PASS`, so both capture and nothing frozen changes; (ii) calls the frozen
`main()` unmodified into a per-mint staging path; (iii) forwards the ro caches in the same
process; (iv) writes **one** final `.npz` carrying every array the arena reads. One process, one
file, the pattern preserved — but the second `savez` is the driver's, not `headspace_mint`'s.
Head-R's `h_std` **is** its `K_train` (its training cache is the ro_L24 cache), so it costs one
extra forward and Head-N costs two, exactly reproducing Phase 1b's `(30×3)+(6×4)+(30×2) = 174`.
Recorded in the file's docstring as IMPLEMENTATION NOTE (a).

### (b) §9's mint filenames vs `headspace_fidelity.py:66` — a filename that cannot exist

**The conflict.** §9 requires each of the 66 mint `.npz` to name its
`(dataset, lineage, seed, fold)` quadruple. `headspace_fidelity.py` is sha-frozen in §11 and
therefore run **unmodified**, and its `:66` hard-codes `mint_{dataset}_s{seed}_ffull.npz` — which
has **no lineage slot**. The two requirements cannot both be met by one filename.

**Resolution used.** The mints keep their quadruple names; the sbatch driver builds a read-only
**alias directory of symlinks** in the shape the frozen reader expects, pointing at Head-N's six
full mints, and gives the fidelity processes that directory as `--mintdir`. Nothing frozen is
edited, no mint is duplicated, and only the `--mintdir` argument differs. Recorded in the sbatch
as IMPLEMENTATION NOTE (b).

### (c) **OPEN — needs a ruling before the run.** §5.4's bootstrap statistic has no macro-F1 leg

**The problem.** §5.4 pre-registers the per-resample statistic as
`Δ_b = mean_{i ∈ draw_b}[c̄_A(i)] − mean_{i ∈ draw_b}[c̄_c(i)]`, where `c̄_X(i)` is the seed-mean
**0/1 correctness of item `i`**. That is an accuracy decomposition and it has no macro-F1
analogue: macro-F1 is not a mean of per-item quantities, so "the same statistic, on macro-F1"
is undefined. Yet S4 is scoped over `statistics.holm_metrics = ['accuracy', 'macro_f1']`, and
§5.5's family arithmetic **counts both metrics** — `(12 + 11) × 2 metrics × 2 lineages = 92`.

**What the implementation currently does, and why it is provisional.** Both metric legs resample
the same correctness vectors, so the macro-F1 leg is presently a duplicate of the accuracy leg.
That keeps the family at the frozen 92 and is non-anti-conservative (it cannot make S4 easier
than a genuine macro-F1 test would, since it neither adds nor weakens a rejection requirement) —
but it is **not** what a reader would take "likewise `mF1`" to mean, and I am not willing to let
it stand as a silent reading. The two defensible alternatives are (i) recompute macro-F1 on each
resample from the resampled predictions and labels, C01's `paired_bootstrap` shape at
`:1742-1772`, which is a different statistic from the one §5.4 pre-registers; or (ii) an explicit
erratum scoping S4's bootstrap to accuracy and re-deriving the family size. **This is a design
question, not an implementation choice, and I have not decided it.**

---

## 5. File → design-section traceability

### `scripts/analysis/c06_falsifier_mint.py`

| implements | design section |
|---|---|
| ONE shared driver, both lineages, `headspace_mint` imported with sha asserted | §3.3; §13.1 items 1, 4 |
| `--train-cache` the only lineage-varying argument; cannot reach `model_name`, the dev load or the dataset table | §3.3; §13.1 item 2 |
| no branch conditional on cache filename or suffix | §13.1 item 3 |
| all key forwards inside the mint process; every key matrix in the mint `.npz` | §8 Phase 1b; §13.1 item 22 |
| `split == "train"` assertion on every ro-cache load (guard layer 2) | §12 |
| resume-safe skip on an existing final `.npz` | §12 ("why `mints_executed` and not 66") |
| the ro cache paths, built from the frozen dataset table; L28 unreachable | §3.1 |

### `scripts/analysis/c06_falsifier_arena.py`

| implements | design section |
|---|---|
| `ArmBuilder.fuse` / `.paired` / `.build_views` — one construction, block-list parameterised | §3.4 |
| endpoint pre-normalisation `std[m] := l2_rows(standard[m])`; `common_interaction = paired(common, l2(common ⊙ displacement))` | §3.4 (round-6 C-1, round-7 C-1) |
| explicit boolean `zero_mask` everywhere, never `None`; `normalization_epsilon = 1e-12` passed explicitly to the one-block build | §3.7; §13.1 item 8 |
| `GATE-DET1`, `GATE-SHA` (37 artifacts, once in the driver) | §6 |
| `GATE-C01PARITY` — ONE predicate, `max|diff| == 0.0`; `GATE-ROWSUBSET`; `GATE-RHORAW` | §6; §3.7 |
| `GATE-FOLD` by re-reading the banked parity flag from all 66 `.npz` (resume-safe) | §3.2; §6 |
| `GATE-IDPARITY`, `GATE-ZEROMASK`, `GATE-NULLREMOVED`, `GATE-POP` | §6; §3.7 |
| `GATE-FLOOR` — Head-N native keys, full `n`, both metrics, every `fold_acc_deployed` entry | §6 |
| `GATE-ORBITDISP` per fold over all 60 head cells, `ρ` in float64 over float32 keys | §6.1 |
| `GATE-NESTED`, `GATE-SELFTEST` (`net_s = n_D·Δacc_s`, all 14 arms, per seed) | §6 |
| `GATE-ARENA` bands `[0.6203, 0.98]` / `[0.7091, 0.98]`, lower bound on `endpoint_std` only | §6.3 |
| `GATE-ZEROOP` + `GATE-ALGEBRA`; tie casualties **analytic** (rearrangement inequality, not `g!`) | §6.5; §13.1 item 10 |
| tie aggregation per `(dataset, seed, lineage)` pooling folds; cap `⌊0.01 n_D⌋`, one-directional | §6.5; §5.9 item 5 |
| `avg_score` = mean of the two endpoint vote **scores** | §3.5 |
| guard arms built by the rotation route, never aliased, and voted | §3.5; §13.1 item 14 |
| reference arm via `select_strongest_ordinary_control`, per `(dataset, lineage)` cell | §5.2.1 (D-1) |
| S1–S7, seed axes `3/3`, S7 `common_displacement`-only with `<=` and zero-fix convention | §5.2, §5.2.2 |
| `tiny_ok` **not** carried | §5.2.3 |
| S4 bootstrap, shared draws, Holm family frozen at 92, dropped lineage `NOT_TESTED` at `p = 1` | §5.4, §5.5 |
| S5 shuffle null, `id_hash_permutation`, 256 draws, family of 4 | §5.4.1 |
| verdict combination rules 1/2/3; dataset axis of the drop | §5.6 |
| `RuntimeError` from the C01 algebra → `INSTRUMENT_INCONCLUSIVE` with the `context` string | §5.6 |
| heartbeat: one `buffering=1` handle, never re-wrapped; `elapsed ÷ 2934.5` | §9; §13.1 item 12 |
| `GATE-LEDGER` predicate set, process count binding | §12 |
| `import_compute_modules` before touching the algebra | §3.4 deferred-import note |

### `configs/c06/c06_falsifier.json`

| implements | design section |
|---|---|
| frozen C01 constants, **read and asserted equal** (`<=`, `0.001`, `0.05`, `0.5`, `0.1`, `1e-12`) | §3.7 second block; §13.1 item 5b |
| population-derived constants recorded for `GATE-POP` to assert against a **computed** value | §3.7 first block; §13.1 item 5a |
| `ρ*` per dataset at full precision; the 26 frozen `ρ_raw` at 6 dp | §6.1 |
| `GATE-FLOOR` anchors, both metrics | §6 |
| all 37 sha256 | §11 |
| ledger predicates; verdict rules; scope sentences | §12, §5.6, §10.2 |

### `scripts/slurm/c06_falsifier_cpu.sbatch`

| implements | design section |
|---|---|
| one submission, 8 CPU / 32 GB, no `--gres` / `--time` / array / dependency / requeue | §13 |
| 73 processes in the order 66 mints → 6 fidelity → 1 arena | §13 |
| `GATE-SHA` once in the driver before any of them | §6; §13 |
| DET-1 thread env exported **before** any python starts | §6 |
| progress file created before the first python process; unbuffered per-mint echo | §9 |
| `PYTHONPATH` export reaching `c09_guard` — the wiring guard layer 3 depends on | §13.1 item 28 |
| output root `artifacts/c06_falsifier/` | §9 |

---

## 5b. ERRATUM 1 — landed 2026-08-05

Ambiguity **(c)** of §4 below was adjudicated a **design erratum**, not an implementation choice.
Proposal `C06_FALSIFIER_ERRATUM1_PROPOSAL.md` → independent review
`C06_FALSIFIER_ERRATUM1_REVIEW.md` (**REVISE — 1C/1H/2I/2M**, endorsing the direction, refuting the
proposal's *"decisive technical fact"*, binding six obligations) → landed record
`C06_FALSIFIER_ERRATUM1_LANDED.md`. **6/6 obligations satisfied.**

**The design revision is `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md`** (sha256
`0b446b91675fd4ff8aea15f2648401d6ce589d089eadad34846f885b2ec9c2ab`). **v15 stays on disk unmodified**
at `75e3aa84…` as the GO'd record.

**What the erratum does.** §5.4's accuracy bullet is **retained verbatim** — the reviewer measured
that re-associating it into C01's per-seed form is algebraically identical but **not** bit-identical
(`5.55e-16`), and that S4's `lower > 0` and zero-adverse-count predicates are the only ulp-sensitive
predicates in the design (`38.8 %` of near-identical arm pairs get a different `one_sided_raw_p`,
`1.2 %` flip `lower > 0`, and neither form dominates on ties). A **macro-F1 bullet is added**, using
C01's recompute-per-resample form via `mechfix_ops.macro_f1` (named in the design text, with its
degenerate-draw behaviour stated), and §5.9 gains **item 10** disclosing the tie direction
(`19.3 %` of tied draws escape the adverse count → S4 easier → **CLOSE harder** → conservative).

**Updated file hashes.**

| path | sha256 | bytes |
|---|---|---|
| `scripts/analysis/c06_falsifier_mint.py` | `1084b5be8c11ad60085115504e999b338db481801614452526084b87d1b3a1d0` *(unchanged)* | 11806 |
| `scripts/analysis/c06_falsifier_arena.py` | **`6ba6a14e4120e683121f93d234f5794f7bab514dfe2f51a779c87246f484e7a8`** | 61574 |
| `configs/c06/c06_falsifier.json` | **`3ebcc36c74b759d28612e0974227c08dea98f6ba72e09f36ca047f35d7f5087e`** | 14554 |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `76d061daf62c51dae584387160924cae482ca7ea20710423b443424b2a21b634` *(unchanged)* | 5942 |

**Re-priced compute.** Phase 4 `11.6 → 7.0 s` on the settled units `U_acc = 0.0049 s` (per
comparison) and `U_mF1 = 0.0384 s` (per `(arm, seed)`), i.e. `168 × U_mF1 + 92 × U_acc = 6.90 s`
carried at `7.0`. Total **`2934.5 → 2929.9 s`**, `× 1.25 = 3662.4`, mint share `85.6 %`,
`2×` miss `3203.6 s`, `5×` miss `4024.7 s`. `U3` is retired. The literals in
`c06_falsifier_arena.py:29,45` and the config are updated.

**Dry checks re-run against the edited files — all pass.**

| check | result |
|---|---|
| `py_compile` / `bash -n` / `json.load` / module exec | **all OK** |
| `GATE-DET1` + `GATE-SHA` | **37/37**, exit `0` — *unchanged by the erratum; the c06 config is not in its own digest table, so no §11 digest moves* |
| `GATE-C01PARITY` both datasets | **`max|diff| = 0.0`**, exit `0` |
| `GATE-ROWSUBSET` (HateMM) | bit-exact bridge |
| `GATE-RHORAW` | 13 arms at 4 dp, both datasets |
| **NEW** `resampled_macro_f1` vs scalar `mechfix_ops.macro_f1`, 200 draws | **`0.000e+00` — BIT-IDENTICAL** |
| `PROJECTED_SECONDS` as imported | `2929.9` |
| blindness grep of the re-run progress file, `[0.6, 0.99]` | **NONE** |

**Ambiguity (c) is closed.** (a) and (b) remain implementation decisions for the code lineage to
audit.

---

## 5c. CODE REVIEW ROUND 1 — fixes landed 2026-08-05

`refine-logs/C06_FALSIFIER_CODE_REVIEW_R1.md` — **REVISE 1C / 4H / 5I / 12M**, found by
**execution** (three probes driving the real functions) after fifteen clean design rounds and the
erratum adjudication. The C09 house rule vindicated again.

### Updated hashes

| path | before CODE-R1 | after |
|---|---|---|
| `scripts/analysis/c06_falsifier_mint.py` | `1084b5be…` | **`98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9`** |
| `scripts/analysis/c06_falsifier_arena.py` | `6ba6a14e…` | **`0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742`** |
| `configs/c06/c06_falsifier.json` | `3ebcc36c…` | **`e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb`** |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `76d061da…` | **`c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d`** |
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E1.md` | `0b446b91…` | **`8cde58aade0d04873f5313a89d6f321c59423e602a85fd88d6505593b0d58f7d`** (§8 correction only) |

### Per-finding disposition

* **C-1 (Critical) FIXED.** `c06_falsifier_mint.py`: the frozen loader is bound once at import
  (`_FROZEN_LOAD_SPLIT`), `load_ro_split` calls **that object** rather than the module attribute, and
  the train-split override is removed in the `finally` around `HM.main()` so nothing after it can
  see the patch. **No suffix-conditional branch was added**, so §13.1 item 3 still holds literally.
  Added: a resolved-path assertion, resolved paths recorded in `meta["c06"]` instead of intended
  ones, and `assert not np.array_equal(h_std, h_ow)` — a one-line falsifier for the whole class that
  cannot fire on a correct run (§7.8 measures `min_i d_i` at `0.018`–`0.038`).
  **Probe 1 re-run, Head-R: `h_ow` marker `3.0` (was `2.0`), `h_std == h_ow: False`.**
* **H-1 FIXED.** The four `raise GateFailure` sites for `GATE-NESTED`/`GATE-SELFTEST` became
  `drop_early` appends merged into §5.6's drop list; `rho_of` returns `NaN` as a sentinel instead of
  raising. **Probe 3 re-run: 0 `raise GateFailure` sites left in `run_lineage`.**
* **H-2 PARTIALLY FIXED — one limb is ERRATUM TERRITORY and is reported, not adjusted.** The
  measured ledger is wired: `C09_LEDGER_DIR` exported in the sbatch, `c09guard.aggregate` called,
  every §12 predicate evaluated as a pass-condition, the verdict face carrying measured counts, and
  a hard `c09guard._INSTALLED` assertion at the top of every C06 process (§13.1 item 28's
  "active, not merely importable"). **BLOCKED:** §12's binding `dev_path_opens == mints_executed + 0`
  is unsatisfiable on a clean run — measured `+2` per `GATE-SHA` process × 2 processes = `+4`,
  because round-8 H-1 widened `GATE-SHA` to input caches including two `dev_seen_*.pt`. Implemented
  **exactly as frozen**; the failure message carries the decomposition and the words
  `ERRATUM REQUIRED`. **The battery cannot pass GATE-LEDGER until Erratum 2 lands.**
* **H-3 FIXED.** The mint driver gained a `--progress` argument and a line-buffered `heartbeat()`
  writing `MINT-START` / `MINT-SKIP` / `MINT-DONE`; the sbatch passes it to all 66 mints and writes
  a `FIDELITY` line per fidelity process and a `DRIVER-DONE` line. All 73 processes now appear.
* **H-4 FIXED IN CODE, AND §8 RE-PRICED.** `build_views` gained an `only=` allow-list that changes
  **which** arms are emitted, never **how** any is built — verified **bit-identical** to the full
  build, so §13.1 item 19's one-construction warrant holds and `GATE-C01PARITY` still measures
  `max|diff| = 0.0`. S5 now builds 2 arms, not 15: measured **11.11×**. Separately, `U4` was
  re-measured against its own stated object and is **`0.33 s`, not `0.08908 s` — low by `3.64×`**;
  Phase 3 re-priced `273.7 → 1013.8 s`, total **`2929.9 → 3670.0 s`** (`× 1.25 = 4587.5`), mint
  share `85.6 → 68.3 %`, Phase 3 `9.3 → 27.6 %`.
* **I-1 FIXED** — `emit_halt()` writes a verdict JSON on **both** exception paths with the failing
  gate, the `l2_rows` context string, the gates dict, drops and ledger.
* **I-2 FIXED** — `GATE-DOMAIN`'s recovery fraction and the raw-vs-head comparison are computed for
  Head-N and emitted; the six `GATE-DEVFID` JSONs are read; a completeness assertion requires all
  **twenty** gate names on the face before emit.
* **I-3 FIXED** — §13.1 item 25's per-cell `min_i d_i`, median, max and `frac(d_i ≤ 0.001)` are
  recorded per `(dataset, seed, fold, lineage)` and emitted beside `algebra_residual`.
* **I-4 NOT FIXED — carried to round 2.** `GATE-ZEROOP`'s near-tie ranking is still each arm's own
  top-21 rather than the union of the two. The direction is one-sided (it can only convert
  REPORT → HALT, never manufacture a SURVIVE), so it cannot publish a wrong verdict; it is recorded
  here rather than claimed fixed.
* **I-5 FIXED** — `gate_pop()` is standalone and runs before `raw_leg`, `GATE-NULLREMOVED` and
  `GATE-FLOOR`; it also asserts head-leg/raw-leg **index-set identity** and the full-population
  majority.
* **Minors:** M-1 (all 18 C01 constants asserted), M-2 (`GATE-SHA` before the C01 import),
  M-3 (`|| RC=$?` — the trailer was dead under `set -e`), M-4 (dead `window` assignment removed),
  M-5 (independent scored-item count), M-8 (S5 uses the measured `keep`), M-9 (32-draw runtime
  fidelity spot-check against `mechfix_ops.macro_f1`, HALTing on any difference), M-11 (decision
  literals read from the sha-gated config). **M-6, M-7, M-10, M-12 carried to round 2.**

### Dry-check re-run — all invariants preserved

`py_compile` / `bash -n` / `json.load` / module exec **OK**; `GATE-SHA` **37/37** exit `0`;
**`GATE-C01PARITY max|diff| = 0.0` on both datasets** (the check §7 of the review demands after any
builder change); `GATE-ROWSUBSET` bit-exact; `GATE-RHORAW` 13 arms at 4 dp; `PROJECTED_SECONDS`
imports as `3670.0`; blindness grep of the progress file returns **NONE**.

---

## 6. What has not happened

* **Not frozen.** No hash of these four files is registered anywhere as a freeze record.
* **Not reviewed.** The separate independent code/resource review lineage has not run. Its sole
  input is §13.1's 28 items. Two places worth its sharpest attention: the analytic tie-casualty
  bound in `vote_bounds_over_orderings` (§6.5's "worst case over orderings, computed
  analytically"), and open question (c) above.
* **Not authorized, not submitted.** No job exists. `TARGET_STATE.json` is untouched and clean
  against `HEAD`; the five tracked modifications in the working tree all predate this work
  (`AGENTS.md`, `CLAUDE.md`, `RA-HMD/…`, `TARGET_LOOP.md`, `TARGET_REVIEW_RAW.md`).

---

## §5d. ERRATUM 2 — LANDED (2026-08-05)

Full obligations checklist, dry-check transcripts and the seven-round design lineage:
**`refine-logs/C06_FALSIFIER_ERRATUM2_LANDED.md`**. Final adjudication:
`refine-logs/C06_FALSIFIER_ERRATUM2_REVIEW_R7.md` (REVISE 0C/2H/3I/5M); landed in one pass under the
user directive that round 7 is the last review of this erratum.

### Files touched — sha256 before → after

| path | before | after |
|---|---|---|
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md` | *(new file)* | `254c0547c8e3579d2b5642747ceb686f9147dd0023a051e669ed9974edce5c4b` |
| `scripts/analysis/c06_falsifier_arena.py` | `0cdfd4f0c9f7095c355a0f0df5389a619576a5254311fdbb6bb9c956d2db0742` | `c0e20054e195152ac0b08f8671984e8ab47c871ce8e5d400f47118f2d933f936` |
| `scripts/analysis/c06_falsifier_mint.py` | `98f7b4a6f10cb39cd180541f91ffd319406c7048254566c4be2f731d9b4ad7f9` | `299d04020489362558c6f4411fb702fad19abf50ef105eccd5321442842474ea` |
| `configs/c06/c06_falsifier.json` | `e26784319a8ea7cd6c4c7b3011515f4c33ccf091d9d7b305d775b84a41e5adeb` | `29f1a57cce41f831819c2c5b9510bfe01fc1d148004323e7c23c944bd6f48ddf` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `c3647173ff1556801c4c492f05e49e27e7fdb14ef80c1def8b9e492b6fa2fc4d` | `06d554c63cbc498b2af04f3cd4d45f2713cd9bce9db56d51ed35fc2ff5c510b4` |

`C06_FALSIFIER_PREREG_DRAFT_V15E1.md` (`8cde58aa…`) and `…_V15.md` (`75e3aa84…`) are
**byte-unmodified**, as are all seven erratum-2 proposals. Landing order: V15E2 written first, its
digest computed, all code/config edits applied, `design_sha256` set **last**.

### What changed, in one paragraph

`dev_path_opens` becomes the two-term predicate with `expected_sha_dev_opens` **derived** from the
digest tables and **asserted** against the config; `processes_reporting` 73 → 74 (the
`--gate-sha-only` driver leg is a process); `GATE-SHA` runs twice and gains the design document as a
38th artifact, HALTing on a drifted digest in process 1 at zero compute cost; §8's Phase 1d and 1g
counts each rise 1 → 2, with Phase 1d's product **re-multiplied** (`2 × 0.13 = 0.26 → 0.3`, not
`2 × 0.1`) so the total moves `3670.0 → 3674.0 s`; the heartbeat denominator gets a single source in
the config, an sbatch export and a three-way HALT assertion; three uninstrumented ledger counters
move to `by_construction` publication while their runtime assertions are **retained verbatim**; and
the sbatch gains a resume-safe `C06_MINTS_EXECUTED` counted on the mint driver's own skip condition.

### CODE-R1 carried items landed in the same pass

`I-4` (GATE-ZEROOP ranks on the union of the two arms' top-21 sets, retrieval widened to `2 × 21`),
`M-6` (the `:2724`-equivalent S7 consistency assertion, made non-vacuous by recording the selector's
own return), `M-7` (finiteness on S4/S5/S7/`net_s` decision quantities), `M-10` (`headspace_mint` and
`vsw_pregate` imported, `runtime_block()` recorded), `M-12` (`load_ro` memoised per dataset, removing
the six-vs-one drift at its cause).

### Dry checks re-run — all pass

`GATE-SHA 38/38` (was 37/37; the design document is the 38th) · `GATE-C01PARITY max|diff| = 0.0` on
**both** datasets · `GATE-ROWSUBSET` bit-exact · `GATE-RHORAW 13/13` both datasets ·
`expected_sha_dev_opens` derived `2 × 2 = 4`, matching the declaration · both new HALT gates
falsifier-tested and firing before any battery compute · blindness grep clean (the only "accuracy"
in the outputs is the line asserting none was computed).

### Two disclosures for the verification lineage

1. **The driver leg now emits five heartbeat lines, not four** — round-7 I-3's corrected count of
   four predates this landing, which adds `PROJECTION` per round-5 M-3. V15E2 carries no line-count
   claim, so no design text is stale.
2. **`artifacts/c06_falsifier/C06_VERDICT.json` was created and removed during the dry checks.** The
   first design-gate falsifier run omitted `--out`, whose default is that path, so the HALT wrote
   there. Provenance checked (one file, my own run, this session), directory removed, both falsifier
   tests re-run with `--out` in the scratchpad, absence re-verified. The dry-check protocol must pass
   `--out` explicitly from now on.

**Not authorized to run.** The scoped code-lineage verification pass, the freeze, and the user's
submission authorization all remain outstanding. `TARGET_STATE.json` unmodified; nothing committed.

---

## §5e. FINAL-VERIFICATION FIXES (CODE-REVIEW ROUND 2) — 2026-08-05

`refine-logs/C06_FALSIFIER_CODE_REVIEW_R2.md` returned **NO-GO** with exactly two blocking defects,
both HALT-on-a-clean-run wiring, **zero wrong-verdict paths and zero resource or test-contact
violations**. Both are fixed here, with the four minors and the three informational notes.

### sha256 before → after

| path | before (erratum-2 landing) | after (R2 fixes) |
|---|---|---|
| `scripts/analysis/c06_falsifier_arena.py` | `c0e20054e195152ac0b08f8671984e8ab47c871ce8e5d400f47118f2d933f936` | `0ff7eedea2932d12303a7268ec0d4daa609f77c02b16b8b0bec1b82e1a692372` |
| `scripts/slurm/c06_falsifier_cpu.sbatch` | `06d554c63cbc498b2af04f3cd4d45f2713cd9bce9db56d51ed35fc2ff5c510b4` | `72d25fb8665150d629ec714321eeeb2603040fc0e6e4c078f91ed0144f654bd2` |
| `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15E2.md` | `254c0547c8e3579d2b5642747ceb686f9147dd0023a051e669ed9974edce5c4b` | `a24188868272716ffcedcfc0dbb9769f5843f9fe34ebff66ecfc713e118337ae` |
| `configs/c06/c06_falsifier.json` | `29f1a57cce41f831819c2c5b9510bfe01fc1d148004323e7c23c944bd6f48ddf` | `8196d163f39299273040dc8532a022433afb6c6289ca642b9360cdf99c6615db` |
| `refine-logs/C06_FALSIFIER_ERRATUM2_LANDED.md` | `38a8b31cdc76f13bee4bca5b6daa9a0f4415be151dc35b970d3d2d6043280296` | `232545261be50f74c993b16974997cc69240b2c4c438787c2b2e05719181d198` |

`c06_falsifier_mint.py` is unchanged at `299d04020489362558c6f4411fb702fad19abf50ef105eccd5321442842474ea`.
`design_sha256` was re-set **last** after the V15E2 edit; startup parity re-verified.

### C-1 — `GATE-COMPLETENESS` HALTed every clean run

Seven of the twenty declared gate names were never written into `bat.gates`: five per-lineage gates
recorded only into `drop_reasons`, and `GATE-C01PARITY` / `GATE-RHORAW` reported only through the
heartbeat. **Reporting wiring only — no gate's predicate, threshold or drop semantics was touched.**

* `GATE-C01PARITY` and `GATE-RHORAW` now write beside their existing heartbeat call, keyed **per
  dataset** (each runs once per dataset).
* The six per-lineage gates are recorded by a new `_record_per_lineage_gates(ds, lineage, drop)`,
  keyed **per `(dataset, lineage)`** — the aggregation rule the seven blank names had concealed.
  A reason is attributed to its gate by the `"GATE-NAME"` prefix every append in the file already
  uses. `GATE-SELFTEST` moved into the same recorder so all six are uniform.

### C-2 — the `projected_seconds` helper was a 75th ledger-writing process

Erratum 2 §7 introduced a `python -c` config read on the sbatch, **after** `C09_LEDGER_DIR` was
exported. `c09_guard`'s `sitecustomize` therefore installed in it and its `atexit` flush wrote a
ledger file, so `len(procs) + 1` published **75** against the declared 74 and `GATE-LEDGER` HALTed.

Fixed by **re-siting the read above the `C09_LEDGER_DIR` export**, so `_ledger_path()` returns `None`
for that interpreter and it reports nothing. The battery's inventory stays exactly `1 + 66 + 6 + 1 =
74`, so §8 Phase 1g's count, §12's decomposition and §13's process order all stay as landed — no
design edit was needed. The sbatch carries a `DO NOT move this below the C09_LEDGER_DIR export`
comment. Layer 3 is not weakened: the guard still installs in that interpreter (verified), and it
opens only the config. Measured directly: **1 ledger file before the fix, 0 after.**

### Minors

* **M-1** `load_ro`'s docstring no longer claims the frozen loader; it names the direct `torch.load`
  and states the import ordering (`load_frozen` imports `headspace_mint`, installing the guarded
  `torch.load` process-wide) that makes the direct call safe.
* **M-2** `V15E2`'s round-14 record now marks `0.13 % of the 2934.5 s total` as the then-current
  figure and gives `0.10 %` against the live total.
* **M-3** the dry-check protocol note now requires **both** `--out` and `--progress` in the
  scratchpad — `Heartbeat.__init__` creates the progress directory before any gate runs, so `--out`
  alone would recreate `artifacts/c06_falsifier/`.
* **M-4** the sbatch trailer distinguishes a `GateFailure` HALT (exit 2) from a §5.6 rule-3 HALT
  (published by the normal emitter, **exit 0**) and says not to key on the exit code.

### Informational — recorded, non-blocking, with dispositions

* **I-1 — the `I-4` bound claim was overstated.** Corrected in the landed doc: exact under the
  nominal ordering, **conservative** under near-tie permutation, because `_union_ranked` also drops
  an arm's own non-union ranks 21–41 and that changes the gap structure the near-tie groups are cut
  on. Residue bounded by the similarity window (`≤ 9.05e-5`), directed toward **under-counting**
  casualties, so it can only make a lineage more likely to drop — never more likely to publish
  `SURVIVE`. **Disposition: wording corrected, code unchanged.**
* **I-2 — `M-10`'s runtime provenance never reaches the verdict.** `reports["runtime"]` is set and
  not emitted. **Disposition: left as is for this run; no verdict depends on it. Post-run item.**
* **I-3 — `M-6`'s consistency assertion is vacuous** (same pure function, same arguments, so it
  cannot fail). §13.1 item 24 required the assertion to exist and it does. **Disposition: recorded;
  it must not be described as comparing two independently-carried values. Post-run item.**

### Clean-run drive — the defect the fixes exist to remove is gone

Re-run using the reviewer's own harness **unmodified**, with `--patch-gates` **off** (so C-1 must be
fixed in the code) and the line-70 helper ledger **absent** (reflecting the re-sited sbatch):

```
CLOSE world, draws=16
  PER-LINEAGE-GATES 2/2 passed=['N','R'] · S4 92/92 both datasets · S1-S7 all four cells
  VERDICT 1/1 | CLOSE -> C06_VERDICT.json          [drive] arena rc=0
  gate names on face : 20   missing: []   extra: []
  processes_reporting: 74 (expected 74) · dev_path_opens 70 = 66 + 4 · test_path_opens 0
  predicate_failures : []  · design declared == derived
```

The seven previously-blank names now carry real values, e.g.
`GATE-ALGEBRA {"hatemm/N":"PASS","hatemm/R":"PASS","zh/N":"PASS","zh/R":"PASS"}` and
`GATE-C01PARITY {"hatemm":"PASS max|diff|=0.0","zh":"PASS max|diff|=0.0"}`.

**The recorder discriminates, it does not merely stamp PASS.** DROP world (lineage R degenerate):
`GATE-ARENA {"hatemm/R":"FAILED (lineage dropped): GATE-ARENA lower: endpoint_std 0.5657 < 0.6203", …}`
with `hatemm/N` and `zh/N` `PASS`, 20 names on the face and `processes_reporting = 74`.

### Standard dry-check battery — re-run, all pass

```
GATE-SHA 38/38 all frozen digests match · PROJECTION single source 3674.0 agrees config/module/env
GATE-C01PARITY hatemm max|diff|=0.0 · zh max|diff|=0.0
GATE-ROWSUBSET bit-exact · GATE-RHORAW 13/13 both datasets · "no arm accuracy computed"
artifacts/c06_falsifier/ absent after the battery (both --out and --progress in the scratchpad)
```

**Blindness held.** No ro-derived arm accuracy computed; every drive vote ran on synthetic matrices;
`test_path_opens = 0` measured on every face. No GPU, no SLURM job, no commit, no `TARGET_STATE.json`
edit.
