# C09 A0 cloud port — independent PORT-DELTA review lineage

Scope of every round: the **port delta only** — substrate fidelity, image pin, gate
portability classification, the tie-break rule, and the data boundary. The **science is
out of scope**: the design was GO 0C/0H/0I at design round 17 and the executables GO
0C/0H/0I at code round 7, and neither is changed by this port. Each round is a fresh
independent reviewer (opus) with no exposure to the port reasoning. Raw text appended to
`TARGET_REVIEW_RAW.md`.

| round | verdict | load-bearing findings |
|---|---|---|
| 1 | **REVISE 2C/2H/8I** | C-1 image-level `PYTHONPATH` crash-loops every container (MEASURED); C-2 module-level local-filesystem work executes container-side and cannot import; H-1 the frozen one-submission precondition was not carried onto the cloud substrate; H-2 the image was not actually pinned while the record claimed a pin |
| 2 | **REVISE 1C/2H/9I** | **C-1 (NEW, decisive): Modal clamps this plan's function timeout to ~3600 s server-side — verified twice in this repo — while A0's measured wall is ~50-58 min on the FASTER local substrate. The port cannot execute A0 in one container.** H-1 the one-invocation rule was prose with no mechanism, and a second invocation would erase the evidence of the first; H-2 the §7 authorisation gate demanded three things the port collected none of. Round 1's twelve findings re-derived and confirmed CLOSED.|

---

## Round 1 — REVISE 2C/2H/8I

> *"The port is a careful, largely faithful transliteration, and its gate-portability
> classification is substantively **correct**. But the driver as written **cannot start a
> container** — and this is not a theoretical claim: a preflight was already launched and
> crash-looped 10+ times. There is also a second, independent container-start blocker
> that the crash-loop masked. Neither can produce a wrong verdict, but the artifact under
> review is currently non-functional and one frozen submission-discipline precondition
> has been dropped in the move."*

### C-1 — image `env()` sets `PYTHONPATH`, breaking Modal's own container bootstrap. Measured.

`scripts/cloud/c09_a0_modal.py` `.env({...})`. The image baked
`PYTHONPATH=/data/jehc223/RGCL/scripts/analysis/c09_guard`; the build log shows it landing
as `Step 11: ENV PYTHONPATH=…`, and every container then died before user code:

```
File "/pkg/modal/exception.py", line 42, in <module>
    import grpclib
ModuleNotFoundError: No module named 'grpclib'
Function c09_a0_modal.preflight is crash-looping: containers are repeatedly failing to start.
```

Two consistent mechanisms, both caused by that ENV: it **replaces** the `PYTHONPATH`
Modal's runtime uses to reach its vendored client deps, and it forces `site` to import the
C09 `sitecustomize` into *Modal's own runner process*, monkeypatching `builtins.open`
before Modal is loaded.

**Fix applied.** `PYTHONPATH` removed from the image env; the four DET-1 thread variables
and `CUDA_VISIBLE_DEVICES` **stay** there (the clause requires them set before an
interpreter starts). `_env()` sets `PYTHONPATH` for every child with the sbatch's
`${PYTHONPATH:+:$PYTHONPATH}` prepend semantics. This costs the port nothing: the guard is
needed in the job's python processes, not in the driver — exactly as on SLURM, where the
sbatch's own shell carries no guard either. PORT-CHECK-1 still verifies
`builtins.open.__name__ == "_guarded_open"` in a child before any mint runs.

### C-2 — module-level `from modal_probe_runner import guard_reason` cannot resolve container-side.

Modal 1.5.2 executes `importlib.import_module("c09_a0_modal")` inside the container, so
**all module-level code runs container-side**. Automounting is fully removed in Modal 1.x:
only `/root/c09_a0_modal.py` is mounted, and `scripts/cloud/` has no `__init__.py`, so the
package branch does not apply. The `sys.path.insert` pointed at a directory that does not
exist in the container. The same defect made `build_manifest()` re-hash 46 MB and
`build_stage()` re-copy the closure on every container start.

**Fix applied.** The guard import, `build_manifest()`, `build_stage()` and the data image
layer are all under `if modal.is_local():`. The container receives the authoritative
manifest as a function argument, so nothing is lost. Verified by simulating a
container-side import: `MANIFEST {} STAGED []`, both functions present.

### H-1 — the frozen "exactly ONE submission, never resubmit" precondition was not carried onto the cloud substrate.

`modal run …::a0` has no approval gate, `RUN_ID` is hard-coded, and the resume-skip makes
a re-run cheap. Since Modal offers no CPU-SKU control, repeated attempts are independent
host draws — a literal retry-until-GATE-FLOOR-passes loop, selecting the host post hoc on
a criterion the prereg does not bound. (The reviewer notes this does not reach a *biased*
verdict, because GATE-FLOOR is verdict-blind: it asks only whether the minted instrument
matches the banked one.)

**Fix applied.** Pre-registered in the port record §1 and in `TARGET_STATE`: exactly one
`a0` invocation; a cloud A0 that reaches the science and HALTs is spent; and the boundary
is named — a container dying **before** the startup guard and PORT-CHECK-1…5 complete has
not begun the A0 and may be relaunched.

### H-2 — the image is not actually pinned, and the record claimed a pin it did not have.

`tqdm` unpinned (the aborted build resolved `4.70.0` against the local `4.67.3`);
`debian_slim(python_version="3.11")` pins no base digest and no Python patch level (local
is `3.11.8`); transitive deps unpinned. Meanwhile §5 asserted in the present tense that
"the image digest covers the input closure byte-for-byte" while §7 read `image digest: TBD`.

**Fix applied.** `tqdm==4.67.3` pinned. The remaining two gaps are declared, the
present-tense claim is withdrawn, and the record now makes the run **unauthorised** until
§7 carries the resolved image id, the container `pip freeze` and the Python patch version.

The reviewer verified as **sound**: every version-pinned member matches the local env
exactly, and the local env is entirely pip-installed, so the pip-vs-conda BLAS-provenance
trap does not apply; thread pinning at 8 is preserved.

### Important findings (all addressed)

- **I-1** `c09guard`'s ledger current-vs-stale partition is keyed on `SLURM_JOB_ID` and
  collapses to the literal `"nojob"` in a container, so a previous attempt's processes
  could satisfy GATE-LEDGER's `n_processes_reporting >= 1` conjunct — the exact conjunct
  the design added so a ledger reading zero because nothing reported cannot pass.
  **Fixed:** per-invocation `SLURM_JOB_ID` token; no gate text changed.
- **I-2** the 36 mint `.npz` and the ledger files died with the container, leaving a
  winning run's instrument unauditable. **Fixed:** `scratch/` and `ledger/` persisted.
- **I-3** the allowlist widened by four extensions while the code declared three
  (`.sbatch` undeclared). **Fixed:** declared.
- **I-4** record staleness; and the reviewer independently observed job 13885 had gone
  `RUNNING`. **Fixed and acted on** — see the port record §8.
- **I-5** PORT-CHECK-5 compared the locally-computed manifest against the frozen table
  rather than hashing container bytes (sound only transitively, via PORT-CHECK-3).
  **Fixed:** it now hashes the container's own files directly.
- **I-6** the preflight was framed as *the* mitigation for the class-(b) risk, but Modal
  exposes no CPU-SKU selector, so it is one host draw and the run gets an independent one.
  **Fixed:** reframed as cost-avoidance only, with a retry budget of 2.
- **I-7** *"the port cannot produce a wrong verdict, only a HALT"* is very nearly but not
  exactly true: GATE-FLOOR / GATE-PARITY-FOLD compare **aggregates at 4 dp**, so a
  compensating flip pair inside one fold could in principle survive — though the macro-F1
  conjunct kills most such cases. **Fixed by softening the claim.** Tightening it would
  need an item-level prediction digest in the banked comparison, which is a change to a
  frozen gate and therefore out of scope for a substrate port.
- **I-8** the tie-break ordered a duration (`sacct` *Elapsed*) against a log timestamp.
  **Fixed:** absolute clock times on both sides.

### What round 1 checked hardest and found SOUND

- **The upstream media guard is unweakened.** `guard_reason` evaluates the media-extension
  blocklist first, then the forbidden-media-directory check, and only then the allowlist;
  the port's tolerance clause matches only `reason.startswith("extension ")` **and**
  requires a declared extra suffix, so it can tolerate only the third clause. **No media
  file can pass.** The added C09-TEST-GUARD is strictly stronger than upstream.
- **GATE-LEDGER's `pass` is not contaminated by the coverage emission.**
  `c09_a0_arena.py:1958`: `pass = bool(tot["test_path_opens"] == 0 and len(procs) >= 1)`.
  The coverage block and `n_processes_expected_fresh_run: 39` are read by no predicate.
  The port record's classification is **correct**.
- **The guard does not become tautological in a container with no test files.**
  `c09guard.is_test_like` is purely path-based and never stats the file, so an attempted
  test open still raises and still increments the counter. The detector is intact; the
  container merely also removes the objects.
- **The REPO-path invariant is structurally guaranteed *and* runtime-checked**, not
  asserted in prose — PORT-CHECK-2 runs a positive probe (`…/test_seen_x.pt` → True) and a
  negative probe (the operative train cache → False) and fails closed. *"This is the
  port's best piece of work."*
- **Step-for-step sbatch fidelity**: order, loop nesting, the `full`/`f{n}` tag, output
  filenames, `allow_fail` on both GATE-DEVFID runs matching `if ! …; then`, the
  resume-skip, and the fail-closed `rc != 0` semantics of `set -euo pipefail` all match.
  The expected **39** reporting processes is preserved exactly.
- **DET-1** thread variables set before any interpreter starts, and re-asserted per child.
- **ONE container** for the whole A0.
- **Input closure complete and minimal** — verified against the code, not the prose:
  nothing the job reads is missing, and nothing is uploaded that should not be
  (no `test_seen`, no `_shards`, no `data/gt/*/test.jsonl`, no media; 46.1 MB total).

---

## Round 2 — REVISE 1C/2H/9I (over the corrected delta)

The reviewer formed an independent view first (substrate diff, gate-by-gate trace,
closure verification, a live check of job 13885), then read round 1 last, and
**re-derived each round-1 fix rather than taking the record's word**: all twelve are
genuinely closed. One new Critical neither round had addressed, and two Highs where a
round-1 repair had been written as prose without a mechanism.

### C-1 (NEW, decisive) — Modal clamps this plan's function timeout to ~3600 s; A0 does not fit.

Verified twice in this repo, not by this port: `W2A_PROBE_RECORD.md:32-38` and
`W2A_CHUNK_LOG.md:1-6` — `MODAL_PROBE_TIMEOUT=43200` reached the child and computed
43200, yet **both** single-container attempts were killed at **~62 min**, attributed to
the Starter-plan cap. Measured A0 wall on the *faster* local substrate: ~28 min of mints
+ a 20–30 min arena ≈ **50–58 min**. A kill at ~62 min lands *after the first mint*,
which §1 counts as a **SPENT** invocation — no verdict, no relaunch. The repo's existing
chunk-loop workaround resumes in a **new container**, violating invariant 2.

**Disposition: the port STOPS and reports.** Recorded as §9 of the port record, and at
the `TIMEOUT_S` definition in the driver so a future caller meets it before spending the
invocation. Lifting it needs a plan with a verified ≥2× lifetime, a re-preregistration on
a chunked substrate (which breaks same-table-same-hardware and needs its own review), or
a different provider. **None attempted.**

### H-1 — "exactly ONE cloud invocation" was prose only; a second invocation would erase the first's evidence.

Three concrete gaps: the volume destination was the constant `RUN_ID`, so invocation #2
would overwrite #1's decision, manifest, mints and ledger — destroying the one artifact
proving the rule was broken, by breaking it; there was no spent-invocation sentinel
(SLURM has `sacct` plus an approval gate, `modal run` has neither); and `retries` was
left to the platform default, so a retry would re-run the science on a fresh independent
host draw.

**Fixed:** destination keyed on the per-invocation `job_token`; a
`/c09out/INVOCATION_<token>.json` sentinel written and committed **before the first
mint** (exactly where §1 draws the boundary) with a hard refusal if one already exists;
`retries=0` pinned explicitly.

### H-2 — the §7 authorisation gate demanded three things the port collected none of.

The record makes the run unauthorised until §7 carries the resolved image id, the
container `pip freeze` and the Python patch version — but `_hostinfo()` captured only CPU
model and uname, and `C09_PORT_MANIFEST.json` carried no `pip freeze` and no
`sys.version`, so the stated path to authorisation could not be walked by running the
port as built. The reviewer also notes a post-hoc freeze is a **description, not a pin**:
`debian_slim(python_version="3.11")` carries no base digest, so a cache-evicted rebuild
could differ and nothing would detect it, since PORT-CHECK-3/5 hash the input closure and
never the environment.

**Fixed as far as a substrate port can:** `_envpin()` now captures `pip freeze`,
`sys.version`, the Python patch version, the Modal task/image identifiers and
`threadpool_info()` into the port checks and the manifest, in both `a0` and `preflight`.
The record states plainly that this is a record of one build, not a constraint on the
next. **The blocker in C-1 makes the point moot in practice.**

### Important findings (addressed)

- **I-1** the record mis-described where PORT-CHECK-1 runs (it runs in a *child*, which
  **is** the C-1 repair) and listed the check order as sha-then-heredoc while the code
  does the reverse. Both are before the mints and both fail closed, so nothing is
  weakened — but a fidelity record must not assert an order the code does not have.
- **I-2** round 1's I-7 residual was still mis-enumerated: GATE-FLOOR/GATE-PARITY-FOLD
  constrain only **vote-derived discrete aggregates**, while every *continuous* decision
  quantity from the same keys (ΔAUC, both permutation p-values, the bootstrap bound, the
  per-fold τ_hi medians, the `dens50`/`class_gap` columns) is unconstrained at any
  tolerance. The real bound is a **magnitude** argument the record omitted: a head
  reproducing 6 pooled-acc + 6 pooled-mF1 + 30 fold-acc cells at 4 dp is numerically
  near-identical, so the residual perturbation is ~1e-12-scale.
- **I-3** `outvol.commit()` sat only on the success path, so any crash/OOM/preemption/
  timeout-kill lost all 36 mints and the ledger — precisely the state §1 calls spent.
  **Fixed:** incremental snapshots after the mints and after DEVFID, plus a `finally`.
- **I-4** no import probe before the first mint: the startup heredoc imported 5 packages
  while the mint chain needs a dozen more, so a missing `threadpoolctl` would surface
  only inside mint #1 after ~52 s of training. **Fixed:** the heredoc now imports the
  whole chain. (The reviewer traced the chain and confirmed the pin list **is** complete —
  `transformers` is genuinely not needed — but the port did not *prove* it.)
- **I-5** `PYTHONOPTIMIZE` was neither neutralised nor classified: every guard in
  `headspace_mint.py` is an `assert` and `-O` strips them all; the arena refuses `-O` but
  only after 36 unprotected mints. **Fixed:** `_env()` scrubs it and the heredoc asserts
  `__debug__`.
- **I-6** importing the driver locally `rmtree`s and rebuilds the 46 MB staging tree, and
  `STAGE` defaults to a session-specific scratchpad path. Accepted as a known local-only
  side effect of a blocked port.
- **I-7** the preflight retry budget is unmechanised and its output path is fixed, so
  attempt 3 is byte-indistinguishable from attempt 1. (Not a bias hazard — the run draws
  an independent host — an auditability one.)
- **I-8** `_env()` starts from `os.environ.copy()`, so children inherit a superset of the
  sbatch's environment; `WANDB_DISABLED` is set in the image and not by the sbatch
  (harmless, both disable wandb, but it contradicts the "verbatim" docstring).
- **I-9** §5's same-table-same-hardware compliance claim omits §4.3's own disclosure that
  **Modal exposes no CPU-SKU selector**, substituting the one-container/one-host property.
  That substitution is the operative protection and does hold, but it is not what the
  ruling literally asks for.

### What round 2 checked hardest and found SOUND

- **The C-1 guard trace, end to end.** `PYTHONPATH` absent from the image env, set per
  child by `_env()` with prepend semantics; every job process spawned through `_run` with
  that dict — heredoc, 36 mints, 2 fidelity runs, arena. `sitecustomize.py` is found
  because PYTHONPATH precedes site-packages, and the repo contains **no other
  `sitecustomize.py`** to shadow it. PORT-CHECK-1 runs before the first mint. The driver
  is unguarded — correctly, in exact parity with the sbatch's bash. `n_processes_reporting
  = 39` preserved exactly.
- **The C-2 container-side import path.** Every module-level statement walked; the guard
  import, `MANIFEST`, `STAGED` and the data layer are all local-only, and **nothing else
  at module level reads the local filesystem**.
- **The `SLURM_JOB_ID` token genuinely restores `aggregate`'s partition.** `job_token`
  contains hyphens but **no underscore**, so the `led_{job}_` prefix match is exact and
  unambiguous; writers and reader share one `env`; the self-exclusion still works; pid
  reuse cannot collide because the filename also carries `int(_T0*1000)%10**9`.
- **PORT-CHECK-5 hashes container bytes, before the first mint.** All nine re-hashed live:
  **9/9 match** the freeze record §3.
- **Input closure complete**, traced through the read path rather than the prose; the
  image pip set covers the entire chain; `src/moka` and `src/logging` correctly excluded.
- **Data boundary intact.** `git diff HEAD` on `modal_probe_runner.py` is **empty** — the
  upstream guard is byte-identical to the committed version. Live staging tree walked:
  **80 files, 45 MB, zero test-like names, zero media**
  (`json 7 / jsonl 2 / npz 10 / pt 4 / py 50 / sbatch 1 / trainlog 6`).
- **No gate weakened, relaxed, made tautological or dropped.** §4.1's ten rows match
  `adjudicate`'s `halts` dict one-for-one; GATE-LEDGER's `pass` is exactly
  `test_path_opens == 0 and len(procs) >= 1`; the guard does not go tautological in a
  container with no test files; GATE-FLOOR/GATE-PARITY-FOLD are carried **verbatim** and
  not re-derived in-container.
- **The tie-break is sound, unambiguous and genuinely pre-registered.** "COMPLETE" is
  exactly `adjudicate`'s two-valued outcome — the code's definition, not a paraphrase.
  **Not launching does not violate anything §1 pre-registers**: the rule binds *if* two
  runs complete and never requires a second run to start, and *"the strongest anti-gaming
  position is the one taken, since no cloud number exists to be peeked at."*
