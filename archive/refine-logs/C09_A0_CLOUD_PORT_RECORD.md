# C09 A0 — CLOUD PORT RECORD (Modal), science frozen, substrate changed

**Status:** `PORT_BLOCKED_BY_PLATFORM_TIMEOUT_CAP — CLOUD A0 NOT LAUNCHED, AND NOT LAUNCHABLE`
(the local job took the tie-break's first move, and the port is independently infeasible
on this Modal plan — see §8 and §9)
**Date:** 2026-08-02 (Pacific/Auckland)
**Run ID (cloud):** `C09-A0-v1-MODAL`
**Design of record:** `refine-logs/C09_A0_V17_RECORD.md` (GO 0C/0H/0I, design round 17)
**Freeze record of record:** `refine-logs/C09_A0_RECORD.md` (frozen sha256 table,
measured-claims register)

This file governs **the substrate only**. It changes **no** science. Where it appears to
say anything about an arm, a threshold, a null, a gate predicate, a decision rule or a
scope, the two records above govern and this file is void in that respect.

---

## 0. Authority for the port

**USER DIRECTIVE 2026-08-02** (main dialogue, relayed by the team lead): the local SLURM
approval gate has released nothing since Saturday 2026-08-01 15:49 — a weekend approval
gap that may run until Monday — and the user has **explicitly ordered this A0 moved to
cloud**. That directive is the authority for this port and for nothing else.

Standing cloud policy this port runs under (`CLAUDE.md`, user ruling 2026-07-31):

- formal validation **may** run on cloud, **provided** the whole comparison table is
  same-hardware: every arm, every floor, every control on one pinned GPU/CPU SKU and one
  pinned image, with the SKU and image nailed down in the preregistration;
- cloud and local numbers **never** mix inside one table;
- **large data files (raw video) never leave the cluster**; derived float feature caches
  (`.pt`) and label JSONs may;
- `modal_probe_runner.assert_uploadable`'s media guard stays in force.

**Job 13885 (local, `JobHeldUser`) is NOT cancelled, NOT released, NOT touched.**

---

## 1. TIE-BREAK RULE — PRE-REGISTERED BEFORE EITHER RUN COMPLETES

> **The binding verdict is the FIRST of the two runs to COMPLETE with all validity gates
> passing: local SLURM job `13885`, or the Modal run `C09-A0-v1-MODAL`.**
>
> **The other run's outputs are VOID.** They must not be opened, not compared, not
> quoted, not reported as corroboration, not used to check the winner, and not used to
> explain the winner. A second run that finishes later is not a replication and not a
> disagreement — it is a run that has no standing at all.
>
> **A HALT does not win.** A run that HALTs publishes no verdict (§9 of the design), so
> it does not take the tie-break; the other run may then take it by completing with all
> gates passing.
>
> **"COMPLETE" means:** the run wrote `C09_A0_DECISION.json` and
> `DECISION.verdict ∈ {CONTINUE, KILL}` — i.e. the nine §8.1 HALT gates and the
> `SHUFFLE-POP` band all passed. `HALT_NO_VERDICT` is not a completion.
>
> **Ordering is by ABSOLUTE CLOCK TIME of the `C09_A0_DECISION.json` write**, read from
> the artifacts themselves and compared on the same scale on both sides: for the local
> job, `sacct -j 13885 -o End` or the `mtime` of
> `artifacts/c09_topo/v1/a0/C09-A0-v1/C09_A0_DECISION.json`; for the cloud run, the
> Modal container log's completion timestamp. Never from memory, and never by comparing
> a duration (`sacct` *Elapsed*) against an instant — an earlier draft of this clause did
> exactly that and is corrected here (port review I-8).
>
> **Exactly ONE cloud invocation** (port review H-1, carrying the freeze record's own
> preconditions 6 and 7 — *"exactly ONE CPU submission"*, *"never resubmit"* — onto the
> cloud substrate, where `modal run` has no approval gate to enforce them). A cloud A0
> that reaches the science and HALTs is **spent** and may not be re-invoked: because
> Modal offers no CPU-SKU selector, repeated attempts are independent host draws, and
> re-invoking after a GATE-FLOOR HALT would be a retry-until-the-host-passes loop that
> selects the host post hoc on a criterion this preregistration does not bound.
> **The boundary, named explicitly:** a container that dies *before* PORT-CHECK-1…5 and
> the startup guard have all completed has not begun the A0 — that is an infrastructure
> failure and may be relaunched. Once the first mint starts, the invocation is spent.
>
> **If the loser is the local job**, it stays in whatever state SLURM leaves it; it is
> neither cancelled nor released by this port, and its eventual output is simply never
> read.

Written here, and into `TARGET_STATE.json::c09_a0_cloud_port`, **before** the Modal run
is submitted. Recorded at a moment when **no** A0 result of any kind exists on either
substrate.

**Why a tie-break is needed rather than two runs read together.** Reading both would be
a two-substrate comparison the same-table-same-hardware ruling forbids, and — worse —
would create a post-hoc choice between two verdicts. The rule removes the choice before
either number exists.

---

## 2. What changes and what does not

**Unchanged (verified by sha256, §4):** the arena script, the config, the guard, the
four imported frozen modules, every arm, every threshold, the Feldman discriminator, the
decision rule, the label-use discipline (`H-L1`…`H-L4`), and the HALT-only validity
gates. **No frozen file is edited by this port.**

**Changed (substrate only):**

| | local (13885) | cloud (`C09-A0-v1-MODAL`) |
|---|---|---|
| scheduler | SLURM, 8 CPU / 32 GB, no `--time` | Modal `@app.function(cpu=8, memory=32768, timeout=…)` |
| driver | `scripts/slurm/c09_a0_cpu.sbatch` | `scripts/cloud/c09_a0_modal.py::a0` — a **transliteration** of the sbatch's five steps into Python `subprocess` calls, same order, same fail-closed semantics |
| repo root | `/data/jehc223/RGCL` | `/data/jehc223/RGCL` — **identical absolute path**, mandatory (§3) |
| env | conda `HateVideo` | pinned Modal image (§5) |

**Why the driver is a transliteration and not the sbatch itself.** `sbatch` is a SLURM
verb; there is no SLURM in the container. The Python driver reproduces the sbatch's
ordered steps 1–5 exactly — DET-1 thread export before any interpreter, `PYTHONPATH` at
the guard, the startup guard/zero-GPU assert, the four sha256 checks *before* the mints,
the 36 mints, the two reporting-only `GATE-DEVFID` runs wrapped so they cannot abort, the
arena — and fails closed on any non-zero return code, which is what `set -euo pipefail`
buys. The sbatch's own sha256 is carried and re-verified as a **provenance anchor**
(§4): the port asserts it is transliterating the reviewed file, unchanged.

---

## 3. The `/data/jehc223/RGCL` path constraint — why the port has no other shape

Three frozen, hash-pinned modules hard-code the repo root as an absolute literal:

- `headspace_mint.py:62-65` — `REPO = "/data/jehc223/RGCL"`, two `sys.path.insert`s, and
  `os.chdir(REPO)` at import;
- `mechnov_pairverify.py:51` — `REPO`, and `DATASETS[*]["cache_dir"]` built from it, which
  is how the operative feature caches are located;
- `c09_a0_arena.py:42-44` — `REPO`, `sys.path`, `os.chdir(REPO)`.

`c09guard.py:38` pins the **guard predicate's scope** to the same literal: a path is
test-like only if it is under `/data/jehc223/RGCL`.

Editing any of them would break the frozen sha256 and void the code review. Therefore the
container mounts the code subset and the data at **exactly** `/data/jehc223/RGCL/...`.
This is not a convenience: mounting anywhere else would silently **disable the guard**
(every path would fail the `startswith(REPO)` test and be judged non-test-like), which is
the single most dangerous failure mode this port has. The image mount path is therefore a
**load-bearing port invariant**, and the driver asserts the guard is live and correctly
scoped before the mints (§6, PORT-CHECK-1/2).

---

## 4. Gate portability audit

Every validity gate in the frozen design, classified as the task brief requires:
**(a)** file-hash anchor / in-run internal — portable as-is; **(b)** bit-exact anchor to a
**locally**-recomputed quantity — needs re-derivation in-container or carries substrate
risk.

### 4.1 The nine §8.1 HALT gates

| gate | what it is anchored to | class | port disposition |
|---|---|---|---|
| **GATE-FLOOR** | freshly-minted pooled deployed `acc`/`mF1` **vs banked** `headspace_arena_<ds>_s<seed>_OUT.json::result.acc_deployed/mF1_deployed`, **equal at 4 dp**, 6 cells | **(b)** | **carried verbatim, NOT re-derived.** See §4.3 — this is the port's one real risk, and it is fail-closed. |
| **GATE-PARITY-FOLD** | freshly-minted per-fold deployed acc **vs banked** `result.fold_acc_deployed`, 4 dp, 30 fold-cells | **(b)** | as above; strictly tighter (a single item in a ~149-item fold moves 0.0067) |
| **GATE-FIXK20** | in-run only: `M.deployed_vote(topk=k')` recomputed independently against the in-line truncation, every `k'` on the grid | **(a)** internal | portable unchanged; both sides computed in the same container |
| **GATE-BLIND** | in-run read-counters on `lab_query` / `is_inversion[seed]` / `is_stable_inversion` + `inspect`-level signature check | **(a)** internal | portable unchanged |
| **GATE-LEDGER** | `pass = (test_path_opens == 0) and (n_processes_reporting >= 1)` — both **measured in-run** across the job's processes | **(a)** internal **for the decisional conjuncts** | portable unchanged. **Emission caveat, declared:** `predicate_coverage_measured_this_run` re-walks the live tree (`c09guard.verify_predicate()`), so its `n_repo_files_matched` / `n_unmatched_paths_containing_test` will **not** equal the freeze record's `983 / 14` — the container holds only the input closure, not the 3-plus-TB repo. This is an **emission, read by no predicate**, and the difference is in the **safe** direction: the container contains **no test-split artifact at all**, which is strictly stronger than a guard that refuses to open one. Likewise `n_processes_expected_fresh_run: 39` is a declared expectation, not a conjunct. **Second port-specific caveat, repaired rather than declared (port review I-1):** `c09guard._ledger_path` names ledger files `led_{SLURM_JOB_ID or "nojob"}_…` and `aggregate()` sums only files matching the *current* job id, routing the rest to `stale`. With no SLURM in the container both sides would see the literal `"nojob"`, collapsing that partition so a previous attempt's processes could satisfy `n_processes_reporting >= 1` — the exact conjunct the design added so that a ledger reading zero because nothing reported cannot pass. The driver therefore sets a **per-invocation `SLURM_JOB_ID` token**, restoring the partition exactly. No gate text changes. |
| **GATE-NESTED** | in-run per-item checks against the in-run `StratifiedKFold(5, shuffle=True, random_state=0)` partition | **(a)** internal | portable unchanged (fold assignment is a pure function of the label vector; labels come from the byte-identical uploaded cache) |
| **GATE-SELFTEST** | `net_s == n · Δacc_s` exactly | **(a)** internal arithmetic | portable unchanged |
| **GATE-ZEROOP** | `S = ∅ ⇒ Δacc = ΔmF1 = 0, net = 0` | **(a)** internal | portable unchanged |
| **GATE-ARENA** | `majority_rate + 0.02 ≤ pooled acc ≤ 0.98`, majority computed in-run from `lab.mean()` | **(a)** internal + uploaded cache | portable unchanged |
| *(10th HALT)* **SHUFFLE-POP band** | in-run ASL in `[0.45, 0.55]` | **(a)** internal | portable unchanged |

### 4.2 In-job asserts that also fail closed

| assert | anchored to | class | disposition |
|---|---|---|---|
| four frozen-module sha256, twice (driver, then arena) + arena self-hash + config hash | **file hashes** | **(a)** | portable as-is; verified byte-identical **before and after** upload (§4.4) |
| mint `meta.script_sha256 == headspace_mint.py`'s frozen sha | file hash | **(a)** | portable |
| fold parity vs banked `vsw_ckpt/<ds>/f{0..4}.npz` (`headspace_mint.py:209-216`) | **uploaded `.npz` bytes** | **(a)** | portable; hashes verified both ends |
| `n == cfg["n_items"]` (744 / 579) | uploaded cache | **(a)** | portable |
| `mint fit_idx == splits[f][0]` | in-run splitter + uploaded labels | **(a)** | portable (sklearn pinned 1.5.2) |
| gt-order parity (`data/gt/<ds>/train.jsonl` order == cache id order) | uploaded bytes | **(a)** | portable |
| **GATE-NULL(1)** census: HateMM row `355` exact-zero in **both** streams; MHC-ZH none | uploaded cache | **(a)** | portable; recomputed in-container from the byte-identical cache |
| `banked_floors` config anchor == banked arena JSON | file-to-file, no computation | **(a)** | portable, trivially |
| `raw["banked_raw_acc_matches_recomputed"]` (0.8441 / 0.8480) | recomputed raw floor vs banked | **(b)** | carried verbatim. **Explicitly `non_decisional: True` in the code and in no `pass` predicate** — it is a parity emission. Lower risk than GATE-FLOOR: the raw space involves **no training**, only `l2n` + faiss on the uploaded cache. |

### 4.3 The one real risk, stated plainly

**GATE-FLOOR and GATE-PARITY-FOLD are class (b): they compare quantities this run
computes against quantities a LOCAL run computed.** The banked
`headspace_arena_*_OUT.json` were minted on the cluster (AMD EPYC 7742, conda
`HateVideo`, numpy-bundled OpenBLAS reporting `architecture: "Zen"`, torch 2.6.0). The
cloud run must re-mint 36 heads — 30 epochs of CPU training each — and land on the **same
4-dp deployed accuracy in all 36 cells**.

**They cannot be re-derived in-container without destroying them.** Re-minting the *bank*
inside the container and comparing the run against itself would make GATE-FLOOR a
tautology: its entire content is *"the head I just minted is the head the banked arena was
built on"*. Weakening it is forbidden by the port rules and would be a scientific change.
So it is **carried verbatim**.

**The residual hazard is floating-point, and it is fail-closed.** A different host ISA
(AVX-512 Intel vs AVX2 Zen) changes GEMM kernel dispatch, hence summation order, hence the
trained head, hence possibly an item flip — and one item is `1/744 = 0.00134`, far above
the 4-dp tolerance. If that happens the gates **fail**, the run **HALTs**, and a HALT
publishes no verdict and is evidence neither for nor against C09.

**Stated at its true strength (port review I-7, corrected and completed by r2 I-2).**
GATE-FLOOR and GATE-PARITY-FOLD constrain only **vote-derived discrete aggregates** —
pooled accuracy, pooled macro-F1, five fold accuracies — not item-level identity. Two
separate residuals follow, and the first draft named only the first:

1. **A compensating-flip coincidence**: a cloud-minted head that differs from the banked
   head yet leaves pooled accuracy, pooled macro-F1 **and** all five fold accuracies
   invariant at 4 dp simultaneously. The macro-F1 conjunct kills most such cases, since a
   0→1 / 1→0 swap moves the confusion matrix.
2. **The continuous quantities, which neither gate constrains at any tolerance**:
   `D-FELDMAN`'s `ΔAUC`, both permutation p-values, the fit-conditional item-bootstrap
   lower bound, the per-fold `τ_hi` medians, and the `dens50` / `class_gap` design
   columns. A key perturbation too small to move any vote but large enough to move a
   statistic across a bar is **not a coincidence** — it is the generic consequence of a
   different BLAS reduction order.

**What actually bounds (2) is a magnitude argument, which the first draft omitted:** a
head that reproduces 6 pooled-accuracy + 6 pooled-macro-F1 + 30 fold-accuracy cells at
4 dp is numerically **near-identical**, so the residual perturbation on any continuous
statistic is ~`1e-12`-scale. That is the honest bound. Tightening it further would require
an item-level prediction digest in the banked comparison — **a change to a frozen gate,
therefore out of scope for a substrate port**. Recorded, not repaired.

**Mitigation, and its status as non-science.** A **numerics preflight**
(`c09_a0_modal.py::preflight`) mints HateMM seed 0 folds 0–4 in the pinned image and
checks exactly GATE-FLOOR + GATE-PARITY-FOLD for that one seed cell, using **the frozen
arena's own `build_features` / `acc` / `mf1`** rather than a reimplementation, so it
cannot diverge from what the gates will compute. It computes **no A0 quantity**, renders
**no verdict**, writes to a **separate namespace**, and its mints are **discarded** — the
real run mints all 36 fresh in one container, as rule 2 requires.

**What a preflight `PROCEED` does NOT mean (port review I-6).** Modal exposes no CPU-SKU
selector, so the preflight is **one host draw and the real run gets an independent one**.
`PROCEED` is **cost-avoidance only** — it is not a portability guarantee for the run, and
it licenses no relaxation of either gate. **Preflight retry budget: 2.** Beyond that the
port STOPS and is reported rather than re-rolled.

**If the preflight fails, the port STOPS and is reported** — it is not repaired by
relaxing a tolerance.

### 4.4 Upload integrity

`sha256` of every uploaded file is recorded **before** upload and **re-read inside the
container after** upload; any mismatch aborts before the mints. Recorded in §7.

---

## 5. Image pin

See §7 for the resolved digest. Base and dependency pins:

```
modal.Image.debian_slim(python_version="3.11")
  torch==2.6.0  faiss-cpu==1.13.2  scikit-learn==1.5.2  numpy==1.26.4
  scipy==1.17.1  pandas==2.3.3  pillow==11.1.0  tqdm==4.67.3  easydict==1.13
  rank-bm25==0.2.2  torchmetrics==1.9.0  wandb==0.28.0  threadpoolctl==3.6.0
```

These are the **same pins the banked local `HateVideo` env carries** for the five members
DET-3/`RUNTIME_DRIFT` tracks (python 3.11 / numpy 1.26.4 / scipy 1.17.1 / sklearn 1.5.2 /
torch 2.6.0). Job 13885's own startup line confirms the local side reads
`python 3.11.8 numpy 1.26.4 scipy 1.17.1 sklearn 1.5.2 torch 2.6.0+cu124 faiss 1.13.2`,
and the local env is entirely pip-installed, so the pip-vs-conda BLAS-provenance trap does
not apply. `RUNTIME_DRIFT` is emitted by GATE-FLOOR either way and is **reported, not
gated**.

**Pin gaps, declared rather than glossed (port review H-2).** `tqdm` was unpinned in the
first draft and the aborted build resolved `4.70.0` against the local `4.67.3`; it is now
pinned. Two gaps remain and are **not** closed by a version list:

1. `debian_slim(python_version="3.11")` pins **no base-image digest and no Python patch
   level** — local is `3.11.8`, the container gets whatever `3.11.x` Modal's base ships.
2. **Transitive** dependencies (`joblib`, `packaging`, `networkx`, `fsspec`, …) are
   resolved at build time and are not pinned.

CLAUDE.md's 2026-07-31 ruling requires the image to be nailed down **in the
preregistration**. Therefore, **before** `a0` is invoked, §7 must record the **resolved
image id** and the container's **full `pip freeze`** plus its Python patch version — a
pin on the built artifact rather than on the recipe. Until §7 carries them, the image is
**not** pinned to the standard the ruling sets, and the run is **not** authorised.
An earlier draft of §5 asserted in the present tense that the image digest "covers the
input closure byte-for-byte"; that was a claim about an artifact that did not yet exist,
and it is withdrawn until §7 is filled.

**CPU-only.** `gpu` is not passed to the function; `CUDA_VISIBLE_DEVICES=""` is exported
before any interpreter and `torch.cuda.device_count() == 0` is asserted in the preflight
heredoc, exactly as the sbatch does.

**Threads pinned to 8, not 1** — DET-1 pins `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 8`
and `headspace_mint.det1_assert("8")` hard-fails otherwise. Pinning to 1 would violate
the frozen design and change the numerics. `cpu=8` is requested so the pinned thread count
has cores to sit on.

**Same-table-same-hardware — with the substitution named (port review r2 I-9).** The
2026-07-31 ruling asks for the SKU *"nailed down in the preregistration"*. **Modal exposes
no CPU-SKU selector** (§4.3), so that literal requirement **cannot be met** and a
different property is substituted: the entire A0 — 36 mints, both `GATE-DEVFID` runs, the
arena, every arm, every floor, every control, both `τ`, both datasets, the head /
remove-null / raw spaces — runs in **ONE** `@app.function` invocation, in **one**
container, on **one** host. That substituted property is the operative protection and it
does hold; it is stated here rather than left for a reader of §5 alone to discover. Every internal comparison is same-host by construction. No local number
enters any table this run produces, with the two declared exceptions that are *banked
anchors the frozen design itself gates against* (GATE-FLOOR / GATE-PARITY-FOLD) and one
non-decisional raw parity emission — all three are properties of the frozen science, not
of the port.

---

## 6. Port-specific checks (additive, HALT-direction only)

These are **not** new science gates. They are substrate assertions that can only stop the
run earlier than it would otherwise stop; none can turn a HALT into a verdict, and none is
read by `adjudicate()`.

**Order, as the code has it** (port review r2 I-1 — an earlier draft of this section
described the driver as guarded, which is the state that crash-looped, and §2 listed the
sha checks after the startup heredoc, matching the sbatch but not the port): the driver
runs `_port_checks` (PORT-CHECK-3, 2, 5) **first**, then the startup heredoc
(PORT-CHECK-1, 4 + the import probe), then the mints. The sbatch's order is heredoc then
`check_sha`. **Both orders put every check before the first mint and both fail closed**,
so nothing is weakened — but the record should not assert an order the code does not have.

- **PORT-CHECK-1** — `builtins.open.__name__ == "_guarded_open"`, asserted **in a child
  process**, not in the driver (the sbatch's own heredoc assert, carried). The driver is
  deliberately **unguarded** — that is the round-1 C-1 repair, and it is exact parity with
  SLURM, where the sbatch's own bash carries no guard either. The heredoc also asserts
  `__debug__` and imports the entire mint chain, so a stripped assert or a missing package
  is caught before the invocation is spent rather than inside mint #1.
- **PORT-CHECK-2** — `c09guard.REPO == "/data/jehc223/RGCL"` **and** that literal is the
  live repo root, so the guard predicate is in scope (§3). A guard mounted at the wrong
  path would silently pass everything; this makes that a hard stop.
- **PORT-CHECK-3** — post-upload sha256 of every input-closure file equals the pre-upload
  value (§4.4).
- **PORT-CHECK-4** — `torch.cuda.device_count() == 0` (carried from the sbatch).
- **PORT-CHECK-5** — the frozen set's nine sha256 re-verified **inside** the container
  against the freeze record's table, including `scripts/slurm/c09_a0_cpu.sbatch` as the
  provenance anchor for the transliteration (§2).

---

## 7. Execution record

*(filled from the published artifact and the Modal logs only, never from memory)*

- **input closure:** 37 files, 46.1 MB, all guarded; plus 43 `src/*.py` = 80 staged files.
  Contains **no** test-split artefact and **no** media — verified by walking the staging
  tree for both patterns (`NONE (clean)` on both). Per-file sha256 in
  `MANIFEST` (`c09_a0_modal.py`), re-verified in-container by PORT-CHECK-3.
- **frozen set:** all nine sha256 re-verified locally against `C09_A0_RECORD.md` §3 at
  port time — **9/9 match**.
- **preflight attempt 1 (2026-08-02 08:05 NZST):** did **not** run the science.
  Every container crash-looped at Modal's own bootstrap with
  `ModuleNotFoundError: No module named 'grpclib'`, before any user code. Cause: the
  image-level `PYTHONPATH` (port review C-1). Under §1's boundary this is an
  **infrastructure failure, not a spent invocation** — the startup guard and the
  PORT-CHECKs never completed and no mint began. Client terminated; `modal app list`
  confirms 0 running tasks. Images built and discarded:
  `im-Dm71nXhusU4HaKGyIE6j0p`, `im-nqdZLQXk7iztIIOGHdKiRD`, `im-umJtb0aWD06IZXR30a42qd`.
  **Cost: image-build time only; no compute function ever executed.**
- image id / container `pip freeze` / Python patch: **NOT YET RESOLVED — and §5 makes
  the run unauthorised until they are recorded here.**
- container host CPU / `threadpoolctl` architecture: **not measured** (no container
  reached user code).
- preflight numerics outcome: **NOT MEASURED.**
- Modal app + call id: **none — the cloud A0 was never invoked (§8).**
- wall time / cost: **no compute function executed.**
- gate outcomes / verdict: **none. This port produced no A0 quantity of any kind.**

---

## 8. Operational disposition — why the cloud A0 was NOT launched

**The directive's premise expired while the port was being built.**

`sbatch` job **13885** left `JobHeldUser` and entered `RUNNING` at
**2026-08-02T08:14:15** on `foscsmlprd01` — the approval gate released. The user
directive that authorised this port (§0) rested on the gate having released nothing
since 2026-08-01 15:49 and possibly not until Monday. That premise is now false, as a
matter of record rather than of expectation: the local job is executing the frozen A0.

**§1's tie-break then decides, and it decides against launching.** The rule binds the
verdict to the **first** run to complete with all gates passing, and declares the other
run's outputs **void** — not to be opened, compared or quoted. At the time of this entry
the local job had minted 17 of 36 heads in 13 min 41 s (~48 s/mint), putting it on track
for ~30 min of mints plus a ~20–30 min arena. The cloud path from here would require
fixing the two Criticals (done), a fresh image build, a preflight, and then ~115 CPU-min
of run. **The local job wins the tie-break by a wide margin.**

Launching the cloud A0 anyway would spend money and Modal capacity to produce an
artifact this record has **already pre-registered as void before it exists**, and whose
only possible uses — corroboration, comparison, cross-checking — are the exact uses §1
forbids and the same-table-same-hardware ruling forbids. **So it was not launched.**
That is the tie-break rule being applied, not waived: nothing in §1 requires a run to be
started, only that if two complete, the first binds.

**The port is left BUILT, REVIEWED and ARMED, not discarded.** If job 13885 HALTs, dies,
or is killed without publishing a verdict, the tie-break is unclaimed and the cloud run
becomes the live path. Before it may be invoked, two things must happen, in order:

1. §5's pin gap closed — the resolved image id, the container `pip freeze` and the
   Python patch version recorded in §7;
2. the numerics preflight run (retry budget 2), with `PROCEED = true`.

**No relaxation of any gate is available on that path**, and a preflight failure means
the port STOPS and is reported (§4.3).

---

## 9. BLOCKER — this port cannot execute A0 on this Modal plan (port review round 2, C-1)

**Measured in this repo, twice, and not by this port.**
`refine-logs/W2A_PROBE_RECORD.md:32-38` and `refine-logs/W2A_CHUNK_LOG.md:1-6`:

> *"Modal clamps the effective function timeout to **~3600 s server-side** (VERIFIED:
> `MODAL_PROBE_TIMEOUT=43200` reaches the child and computes 43200 — no stray 3600 in
> code — yet every single-container attempt was killed at ~3600 s function-time)."*

Two apps died at **~62 min** (`ap-93KHpJNP9yDSui6fhIlOLs`, `ap-qNF5v5HekPTrGvwjPWGIp0`),
attributed to the **Starter-plan cap**. So `timeout=28800` in the function decorator
proves nothing: W2-A verified exactly that failure mode.

**Against the measured cost of this A0 — and this is now MEASURED, not projected.**
Job 13885, on the *faster* substrate (AMD EPYC 7742, 8 threads), started 08:14:15, had all
36 mints and both `C09_FIDELITY_*.json` written by 08:42 (**~28 min of mints**), and
entered the arena at 08:42:10. **At 10:09 it had been running 1 h 54 min and the arena was
still executing.**

The first draft of this section estimated the arena at 20–30 min and the total at 50–58
min. **That understated it.** The design's own itemised budget (V17 §2) sums to ~115
CPU-min, of which the *arena alone* is ~79 min: two 12-min permutation nulls, 5 min
`SHUFFLE-POP`, 5 min `RANDOM-POP`, 5 min bootstrap, 8 min raw leg, and a 24-min
remove-null sensitivity. The observed run is consistent with that, not with the estimate.

**So the local run has already exceeded the ~62 min wall — on the faster machine, without
having finished.** A Modal container could not have completed this A0. The blocker is not
a projection from a marginal estimate; it is a fact with ~2× headroom against it.

**Why this blocks rather than inconveniences.** A kill at ~62 min lands *after the first
mint*, which is exactly where §1's boundary counts the single pre-registered cloud
invocation as **SPENT** — no verdict, and no second attempt available. The repo's existing
workaround for this cap (`modal_probe_runner._execute`'s soft-budget chunk loop) resumes
in a **new container**, which violates this port's invariant 2 (one container, one host)
and therefore the same-table-same-hardware ruling. **It is not available here.**

**Disposition.** `a0` **must not be invoked** on the present plan. The blocker is recorded
in `c09_a0_modal.py` at the `TIMEOUT_S` definition so a future caller meets it before
spending the invocation. Lifting it requires one of:

1. a Modal plan whose effective container lifetime is verified **≥ 2×** the budgeted wall
   (and §7 must then record the *measured* lifetime, not the requested one); or
2. a re-preregistration on a chunked-resume substrate — which is a **substrate change that
   breaks same-table-same-hardware** and would need its own independent review; or
3. a different cloud provider (see `refine-logs/CLOUD_PROVIDER_RECON.md`).

**None of these is attempted here.** This port stops and reports, which is what the port
rules require when the substrate cannot carry the frozen design unweakened.
