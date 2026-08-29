# PREGATE DETERMINISM CLAUSE (DET-1 … DET-4) + COMMIT-ATOMICITY CLAUSE (GIT-1)

**Two standing, pre-registerable infrastructure clauses. Adopt by citing this file in a prereg's
bars/provenance section.** Cost of the whole determinism clause: **`$0`, one extra line per driver and
six keys per output JSON.**

**Date:** 2026-07-28 NZST · **Author:** closeout/hardening agent · **Cost of the evidence below: `$0`**
— CPU only, login node, **train split only** (`data/CLIP_Embedding/*/train_*.pt`), zero GPU / SLURM /
Modal / training of any deployed arm, **zero test contact**.

**Companion clause:** `refine-logs/PREGATE_CALIBRATION_CLAUSE.md` (owned separately) governs whether
the raw train arena *predicts deployment*. **This** clause governs whether a run reproduces **itself**.
They are orthogonal and both should be cited.

---

## §1. THE INCIDENT, AND THE CORRECTED DIAGNOSIS

`VSW_PREGATE_RECORD.md` §4.3 recorded a campaign-level erratum: re-running the **frozen** F95 module
`scripts/analysis/mechnov_pairverify.py` (sha256 `77b0defd…b7240d`) unmodified — same node, same env,
same caches, same seeds — reproduced every closed-form quantity but **drifted on 44 of 48 trained
quantities**. Four diagnostics exonerated the harness and the residual cause was attributed to
*"oneDNN/MKL kernel selection … varying between sessions"*, from which the record drew the rule that
G-repro parity is unattainable for trained components and that future records must gate against a
same-session re-run rather than against banked JSON.

**That diagnosis is wrong, and the rule it implies is unnecessary. The drift is fully deterministic and
fully controllable.** Measured today, independently, on the frozen module imported unmodified:

### 1.1 The cause is one unpinned environment variable

| run configuration | fold-0 `auc_mlp`, fused space — HateMM / MHC-ZH / MHC-EN |
|---|---|
| `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` | **0.7589 / 0.7908 / 0.6900** = the **recorded F95** cell, exactly |
| unset (⇒ 64) | **0.7584 / 0.7911 / 0.6902** = the **VSW `--stage anchor`** cell, exactly |

**6 of 6 predicted values hit.** `scripts/analysis/mechnov_drive.sh:9` and `mechnov_drive_diag.sh:4`
carry `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; `scripts/analysis/vsw_drive.sh` carries no such
export. Nothing drifted: **two different environments produced two different, each perfectly
reproducible, answers.**

### 1.2 Under the pinned environment, parity is complete — including every trained quantity

The frozen `run_space` re-run unmodified with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, diffed key-by-key
against the recorded F95 fused cell (wall-clock `secs` excluded, being not a result):

| dataset | closed-form | logistic-trained | **MLP-trained** | total |
|---|---|---|---|---|
| HateMM | 0/86 differ | 0/62 | **0/62** | **210/210 exact** |
| MHC-ZH | 0/86 differ | 0/62 | **0/62** | **210/210 exact** |
| MHC-EN | 0/86 differ | 0/62 | **0/62** | **210/210 exact** |

**630 / 630.** A fresh-process repeat at the pinned setting reproduces the score array **bit-for-bit**
(sha256 of the float64 eval-score vector identical). **G-repro is attainable for trained components.
It was never the training that was irreproducible.**

### 1.3 The perturbation enters UPSTREAM of the estimator, and the estimator only amplifies it

The sha256 of `Phi_fit` — the standardised pair-feature matrix, a **closed-form** quantity — takes a
**distinct value for every thread count**:

| `OMP_NUM_THREADS` | 1 | 2 | 4 | 8 | 16 | 32 | 64 (= unset) |
|---|---|---|---|---|---|---|---|
| sha256(`Phi_fit`)[:8] | `7aee94eb` | `3dc60ed7` | `74d31f2f` | `6ef43497` | `3ed62d21` | `2190bf3a` | `723c9239` |
| `auc_mlp` (torch) | 0.6904776 | 0.6908249 | 0.6894453 | 0.6899822 | 0.6898976 | 0.6901854 | 0.6902186 |
| `auc_logistic` @4dp | 0.6428 | 0.6428 | 0.6428 | 0.6428 | 0.6428 | 0.6428 | 0.6428 |

`Phi_fit` is produced by `PCA(svd_solver="full")` → LAPACK/OpenBLAS, whose reduction blocking depends
on the thread count. The rotation therefore differs at ~1e-16, `Zn` differs, and the **float32** pair
features differ in their last bits. **Every closed-form quantity absorbs this below the 4-dp reporting
threshold** — which is why they looked "bit-exact"; they are 4-dp-exact, not bit-exact. **The convex
logistic arm also absorbs it** (0 of 186 quantities moved at 4 dp across the two sessions, and
`auc_logistic` is constant at 4 dp over the whole thread grid). **A 30-epoch Adam MLP does not**: 4 500
non-convex update steps amplify a last-bit input perturbation into 5-8 flipped items on 744.

### 1.4 What this means for the four exonerating diagnostics

All four were **correct and all four tested the wrong knob.** Within-process determinism, call
ordering, and nominated-vs-full-matrix scoring are genuinely irrelevant. Diagnostic 3 varied
`torch.set_num_threads` ∈ {1,4,8} — which controls ATen's OpenMP pool and **not** the OpenBLAS/LAPACK
path that `PCA` uses. Reproduced today: `torch.set_num_threads` ∈ {1,2,4,8,16} with the environment
unset gives **bit-identical** scores across six separate processes. The one uncontrolled knob was the
**process environment**, and no diagnostic looked at it.

### 1.5 The environment, for the record

`python 3.11.8` · `numpy 1.26.4` · `scipy 1.17.1` · `scikit-learn 1.5.2` · `torch 2.6.0+cu124` ·
`threadpoolctl 3.6.0` · `faiss-cpu 1.13.2`. Site-packages mtimes: the whole numerics stack
**2026-03-27**, faiss 2026-04-01, torchmetrics 2026-07-01 — **unchanged across the two sessions**, so
no library moved. `torch` is built against **MKL 2024.2 + oneDNN v3.5.3**; `numpy`/`scipy`/`sklearn`
go through **OpenBLAS**. One process loads **three** distinct OpenBLAS builds (`libopenblas 0.3.23.dev`
= numpy, `libscipy_openblas 0.3.30` = scipy, `libopenblas 0.3.15` = faiss) and **three** `libgomp`
OpenMP runtimes (8 / 256 / 256 threads). Node default: 256 cores,
`std::thread::hardware_concurrency()=256`, `at::get_num_threads()=128`, `mkl_get_max_threads()=128`,
OpenBLAS capping at 64.

**Every one of these is unpinned.** `requirement.txt` (12 lines) names `scikit-learn` with **no
version**, does not list numpy, scipy, faiss at all, comments out pytorch, and carries exactly one
constraint anywhere (`torchmetrics>=1.0`). Reproducibility currently rests on nobody touching the
conda env. That is a real exposure and is **not** fixed by this clause; it is flagged for a ruling.

---

## §2. THE CLAUSE

### DET-1 — Pin the thread environment (mandatory, `$0`)

> Every pregate driver **must** export, before any Python process starts:
> ```bash
> export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
> ```
> `8` is the campaign's standing value (`mechnov_drive.sh`, and the `≤8 threads` every record already
> claims). Any value is admissible **provided it is declared in the prereg and recorded in the output
> JSON**; `8` is the default and should be used unless there is a reason. Setting only
> `torch.set_num_threads()` **does not satisfy DET-1** — it does not reach the LAPACK/BLAS path that
> dominates the perturbation (§1.4).

### DET-2 — Record the runtime numerics environment in every output JSON (mandatory, `$0`)

> Every pregate's `meta` block **must** carry:
> ```python
> "runtime": {
>   "env": {k: os.environ.get(k) for k in
>           ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
>   "threadpools": threadpoolctl.threadpool_info(),   # BLAS build + version + num_threads, per library
>   "versions": {"python":…, "numpy":…, "scipy":…, "sklearn":…, "torch":…},
>   "torch_num_threads": torch.get_num_threads(),
>   "node": socket.gethostname(),
> }
> ```
> `threadpoolctl` is installed (3.6.0) and this costs milliseconds. **No pregate in the campaign
> currently records any of it** — checked: `vga_`, `aggnet_`, `restrans_`, `vsw_pregate_OUT.json` all
> have no thread or environment key in `meta`. That absence is why §1 took a forensic reconstruction
> instead of a one-line diff.

### DET-3 — The parity requirement, restated in three tiers (replaces "bit-exact 4 dp or HALT")

> The purpose of G-repro is to catch **silent re-runs, instrument defects and transcription errors** —
> not to assert that floating-point arithmetic is thread-invariant. It is therefore restated as:
>
> **Tier A — closed-form quantities: bit-exact at 4 dp, unconditionally. HALT on any mismatch.**
> Unchanged from the existing discipline. Measured 0/258 mismatches across the two F95 sessions even
> with the environment differing, so this tier costs nothing and loses nothing.
>
> **Tier B — trained quantities, environment reproduced: bit-exact at 4 dp. HALT on any mismatch.**
> This is the normal case and it is **achievable** — measured 630/630 in §1.2. A record that satisfies
> DET-1 and DET-2 is entitled to gate against **banked JSON**, and the VSW rule *"must gate against a
> same-session re-run, never against the recorded JSON"* is **withdrawn as a general rule**; it applies
> only where the anchor's environment is unknown or unreproducible.
>
> **Tier C — trained quantities, environment NOT reproducible** (anchor predates DET-2, or a library
> moved): parity is asserted against a **measured tolerance band**, not a point. The band is
> established by an **N ≥ 5 self-variance run** of the same cell across the declared configuration
> grid, reported in the record, and the re-run must land inside it. A quantity outside the band is a
> **defect**, not drift, and HALTs. Any *verdict* whose margin is smaller than the band is
> **not bankable** and must be re-run under Tier B.
>
> **Reference band, measured, for the F95 pair-verifier arena** (fused space, EN fold 0, thread grid
> {1,2,4,8,16,32,64}): `auc_mlp` range **0.0014**; pooled `acc_mlp_*` excursion vs the recorded cell
> **≤ 0.0067** (5 items on 744); integer fixed/broke counts move by **≤ 2 each**. `auc_logistic` and
> every closed-form quantity: **0.0000**. Use these until a tighter cell-specific band is measured.

### DET-4 — Prefer the estimator that does not amplify (recommended, not mandatory)

> Where the science permits a choice, prefer a **convex / deterministic** estimator: measured over the
> whole thread grid, the sklearn `LogisticRegression(lbfgs)` arm of the same harness is invariant at
> 4 dp while the torch Adam MLP is not. If a non-convex estimator is scientifically required, it must
> carry a DET-3 Tier-C band, and **no verdict may rest on a margin inside that band.**

---

## §3. BLAST RADIUS OF THE 2026-07-27/28 EPISODE

Recorded as a ledger erratum (append-only, F77/F104 precedent) and **no banked verdict is rewritten**.
Summary: **every verdict survives; two non-load-bearing claims need a scoping note.** See the ledger
row and `VSW_PREGATE_RECORD.md` §4.3 (which carries an appended correction).

| finding | verdict rests on | survives? | scoping note needed |
|---|---|---|---|
| **F95** MECHNOV | control-2b shape cost + deployed floor — **closed-form** | **YES** | the headline **count** *"0 of 36 cells"* is environment-conditioned. It is exact under `OMP=8` (its own run configuration, now identified); under an unpinned env one **secondary** cell clears. Quote as *"0 of 4 PRIMARY cells"*, or quote the count **with** the environment. |
| **F96** RESTRANS | degeneracy control fires; parity 81/81 was against F95's **deployed-floor** (closed-form) quantities; treatment base model is **sklearn logistic** | **YES** | none |
| **F97** VGA/VNQ | K-VGA-3, a **relative** within-session comparison | **YES** | the *"78/78 parity"* claim is **restored, not lost**: its 48 trained cells matched F95's recorded values, which is only possible at `OMP=8`, so F97 ran pinned and **re-asserts whenever DET-1 is honoured**. The VSW record's *"would not re-assert today"* is **too strong** and holds only for an unpinned re-run. |
| **F98** AGGNET | decisive bar missed by **>2×**, both degeneracy controls fire; parity was against closed-form floor quantities | **YES** | its own arm is torch-trained, so its point numbers are environment-conditioned; the >2× margin is ~10× the measured band, so nothing is at risk. |
| **F105** VSW | PARITY-λ0 54/54 is **closed-form**; every Δ is paired against a same-session closed-form floor | **YES** | K-VSW-1 margins (ZH +0.0069, EN +0.0164 hindsight ceilings vs a +0.030 bar) are 2-3× the band. The **ER = 6.0000** datum is an integer ratio at 21 changed items and is band-sensitive in *magnitude* (±2 items ⇒ ER ≈ 4-9) but **not in direction** — the refutation of the ≤1.2 law survives at every point of the band. Quote ER as *"≳4, measured 6.0"*. |

**Not at risk anywhere:** every test-side number in the campaign (produced under SLURM, which sets its
own thread environment per job and does not share this defect), and every closed-form train-side
quantity.

---

## §4. CLAUSE GIT-1 — COMMIT ATOMICITY UNDER CONCURRENT AGENTS (mandatory)

**This is the same class of defect as §1 — a process that silently produces a wrong record — and it
has fired twice today.** It is recorded here as a measured finding, not a style preference.

**The defect.** `git add <path>` followed by a bare `git commit` is **not path-scoped and not atomic**.
The bare commit takes *whatever is in the index at commit time*, including paths staged by another
agent in the window between the `add` and the `commit`.

**Evidence — two confirmed incidents, 2026-07-28, both on `main`:**

| commit | what happened | outcome |
|---|---|---|
| `c290180` (streamcomp) | staged only its own path; swept up **four** files concurrently staged by other agents | repaired via `git reset --soft HEAD~1` + a pathspec commit; all four verified byte-identical afterwards (`sha256sum -c` 4/4) |
| `06c4719` (litsweep-8 erratum) | swept up **three** files it never staged — `refine-logs/DISK_FORENSICS_2026-07-28.md`, `scripts/disk_guard.sh`, `src/run_rac.py` | all three verified intact in HEAD (507 / 584 / 1563 lines) |

**No data was lost in either incident.** The damage is **provenance**: work is attributed to the wrong
commit message, and a future reader tracing a number from `git log` is misled about which agent, which
run and which record produced it. A third, milder instance occurred in this very session in the
opposite direction: commit `7fd207d`'s message says it banks two sweeps, but
`LITSWEEP8_PATHOLOGY_MATCH.md` had already been committed at `2e2805f` by another agent, so only
`LITSWEEP7_LANDING_SITE.md` actually landed — the message over-claims. Same root cause: the index is
shared state and a commit message written before the commit cannot describe it.

> ### GIT-1 — the rule
>
> 1. **Always commit with an explicit pathspec:** `git commit -- <paths>`. A pathspec commit takes only
>    the named paths and **leaves the rest of the index staged as its owners left it**. Never
>    `git add <path>` followed by a bare `git commit`.
>    *Practical note:* `git commit -- <path>` refuses a path git has never seen, so a **new** file must
>    still be `git add`-ed first. That is fine — `git add <newfile> && git commit -- <newfile> <others>`
>    is still path-scoped and still safe. The rule is about the **bare** commit, not about `add`.
> 2. **Never** `git add -A`, `git add .`, `git commit -a`, or any wildcard that can reach another
>    agent's paths.
> 3. **Verify after committing:** `git show --stat HEAD` and confirm that **only your own paths
>    landed**, and that **all** of them landed. If a path you intended is missing, another agent
>    committed it first — do not re-commit it; correct your commit message's claim instead.
> 4. **Never rewrite history** (`rebase`, `reset --hard`, amend of a pushed commit) while other agents
>    are writing to `main`. Mis-attribution is cosmetic; a rebase under concurrent writers is not.
>    The `c290180` repair was safe only because it was performed immediately and verified by hash.

---

## §5. WHAT THIS CLAUSE DOES **NOT** RELAX

* It does not weaken Tier A. Closed-form parity stays bit-exact-at-4-dp-or-HALT.
* It does not license quoting a number without re-reading its source log (the 0.8732 discipline).
* It does not license a verdict whose margin sits inside a Tier-C band.
* It does not fix the **unpinned dependency stack** (§1.5). Pinning `requirement.txt` / exporting a
  conda lockfile is a separate, still-open exposure and needs a user ruling; until then, DET-2's
  version block is the only thing that will let a future reader detect that a library moved.

---

## §6. FILE MANIFEST / REPRODUCTION

| artefact | role |
|---|---|
| `<scratchpad>/det_probe.py` | imports the frozen `mechnov_pairverify.py` unmodified (sha asserted), rebuilds one fold's fitting problem, hashes `Phi_fit` and both estimators' eval scores |
| `<scratchpad>/full_parity.py` | re-runs the frozen `run_space` unmodified and diffs every emitted quantity against the recorded F95 cell |

**Read-only inputs:** `scripts/analysis/mechnov_pairverify.py` (sha256 `77b0defd…b7240d`, asserted at
run time), `scripts/analysis/mechfix_ops.py`, `scripts/analysis/mechnov_pairverify_{hatemm,zh,en}_OUT.json`,
`scripts/analysis/vsw_f95anchor_{hatemm,zh,en}_OUT.json`, `data/CLIP_Embedding/*/train_*.pt`.
**No frozen script was modified. No banked verdict was rewritten. Zero GPU, zero SLURM, zero Modal,
zero test contact.**
