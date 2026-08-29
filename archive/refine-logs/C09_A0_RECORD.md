# C09 A0 — freeze record, with the measured-claims register

**Status:** `RUN COMPLETE — VERDICT PUBLISHED: KILL` (job `13885`, adjudicated 2026-08-04;
§8 below is the post-run record. Everything above §8 is the prospective freeze text and is
left exactly as frozen.)
**Date:** 2026-08-01 (Pacific/Auckland); post-run fields filled 2026-08-04
**Candidate:** `C09 Stable-Inversion Topology Surgery`
**Run ID:** `C09-A0-v1`
**Design of record (specification):** `refine-logs/C09_A0_V17_RECORD.md`
(sha256 `b6f33ea8210a5c4e20547b2642133ebd58b9643fb7c700a79516e410d6ebc1ca`), GO 0C/0H/0I
at design-review round 17.

Prospective at the moment of writing. No A0 job, no result, no metric and no verdict
exists. This file is the **freeze record**: it fixes the identity of the executable set,
registers every empirical claim the set makes, and names every place where the
implementation had to read the design rather than merely transcribe it.

**Reading order.** `C09_A0_V17_RECORD.md` §§1–11 is the design and governs; this file
supersedes it in **no** respect. Where this file says "declared reading" or "declared
deviation", it is naming an implementation choice the design did not fully determine —
never overriding a choice the design did determine.

---

## 1. Review lineage

**Design lineage (17 rounds, closed).** `C09_A0_PREREG_DRAFT.md` → round-1
`REVISE 4C/8H/10I` → `V2`…`V17`, each round a fresh independent reviewer with no
exposure to the repair reasoning, raw text appended to `TARGET_REVIEW_RAW.md`. Round 17
returned **GO (0C/0H/0I)** and named the remaining work explicitly: *"implement the
analysis script and the sbatch driver, hash-freeze the frozen set, and pass a separate
independent code/resource review."*

**Code/resource lineage (7 rounds, closed).** A separate lineage over the executables,
again a fresh independent reviewer each round:

| round | verdict | the load-bearing findings |
|---|---|---|
| 1 | REVISE 1C/6H/13I | **C-1** `own_norm` measured on already-L2-normalised keys (BASE was 7 features, ΔAUC biased toward CONTINUE); GATE-BLIND a structural no-op; GATE-ZEROOP re-read the floor; GATE-NESTED counted rows with a `>0` predicate; GATE-LEDGER unmeasurable; guard confined to one process; GATE-NULL(3) never computed |
| 2 | REVISE 0C/3H/12I | **H-1** the test perimeter missed every path component named `test_seen` (754 shard artifacts); **H-2** GATE-LEDGER passed with zero cross-process evidence; **H-3** GATE-NULL(3) compared `p ≤ α` instead of the Holm-corrected K-rule |
| 3 | REVISE 0C/2H/12I | **H-1** K-DEG's FIXK twin computed on one scale only — the omitted per-item scale agrees *more*, so K-DEG could under-fire (the first identified wrong-verdict path); **H-2** GATE-NULL(3) compared neither the verdict nor the per-multiplier cells |
| 4 | REVISE 0C/0H/6I | instrument HALT on the sensitivity leg mis-reported as a verdict disagreement; row-level loss ungated; GATE-BLIND's counters tautological; GATE-FIXK20 satisfied by construction; `null_row_index` never checked against the census |
| 5 | REVISE 0C/2H/2I | **H-1** GATE-FIXK20's tolerance rested on a **false mechanism** (see claim 12); **H-2** `holm2` let an uncomputable τ_hi veto a rejecting τ_0 — a wrong verdict, not a HALT |
| 6 | REVISE 0C/0H/2I | SHUFFLE-POP's third reading emitted as a mean rather than an ASL; RANDOM-POP's declared reason stated a false arithmetic identity |
| 7 | **GO 0C/0H/0I** | — |

Two of the seven rounds found a path to a **wrong verdict** rather than a HALT (round 3
H-1, round 5 H-2). Both are closed and both are re-verified below.

---

## 2. MEASURED-CLAIMS REGISTER

Every empirical claim the frozen set makes or relies on. `[V]` verified by a static
reviewer without execution · `[D]` documentary, source checked but measurement not
reproduced · `[U]` asserted as measured, **not** re-derivable without execution.

| # | claim | how obtained | what depends on it | static-reviewer re-derivable? |
|---|---|---|---|---|
| 1 | `n = 744` (HateMM) / `579` (MHC-ZH) train rows | `P.load_cache` label length; asserted in-job against `cfg["n_items"]` | every pooled metric, the NET bars, `n_scored` | **[V]** yes |
| 2 | majority rate `0.599462` / `0.689119` ⇒ GATE-ARENA bands `[0.6195, 0.98]` / `[0.7091, 0.98]` | `lab.mean()` on the operative train caches | GATE-ARENA | **[V]** yes |
| 3 | banked deployed floors — HateMM acc `0.8884 / 0.8858 / 0.8858`, mF1 `0.8838 / 0.8811 / 0.8812`; MHC-ZH acc `0.8929 / 0.8895 / 0.8946`, mF1 `0.8747 / 0.8710 / 0.8765` | read from the six `headspace_arena_<ds>_s<seed>_OUT.json` | GATE-FLOOR; frozen into the config as `banked_floors` and **asserted in-job** against those same files | **[V]** yes |
| 4 | banked `raw_deployed_acc` `0.8441` / `0.8480` | same six files, `membership` block | the raw leg's parity emission (non-decisional) | **[V]** yes |
| 5 | structural-zero census: HateMM row `355` is an exact-zero vector in **both** streams, label 1; MHC-ZH has none | `np.abs(img/txt).sum(1) == 0` on the operative train caches | GATE-NULL, the remove-null leg, `null_row_index` | **[V]** yes — and **asserted in-job before the remove-null leg runs** |
| 6 | data-defect counts: whitespace-only `text` 39 / 0; `<em` markup 0 / 243 | `json.loads` over `data/gt/<ds>/train.jsonl` | `DATA_DEFECT_OVERLAP` enrichments | **[V]** yes |
| 7 | `data/gt/<ds>/train.jsonl` is in the same order as the train cache | id-sequence comparison | the positional defect flags | **[V]** yes — and **asserted in-job** |
| 8 | `StratifiedKFold(5, shuffle=True, random_state=0)` reproduces the ten banked `vsw_ckpt/<ds>/f*.npz` hold-out sets | re-ran the splitter | fold parity; every mint's own parity assert | **[V]** yes |
| 9 | key matrices are **float64** (`headspace_mint.py:304` `astype("float64")`; `l2n` preserves dtype; the raw leg's caches are float64) | dtype read | claim 12's closure | **[V]** yes — and **asserted in-job per fold** |
| 10 | `mechfix_ops._norm32` L2-normalises its input **in place** when handed a float32 C-contiguous array (`np.asarray` is then a no-op) and **copies** for float64 | read `mechfix_ops.py:38-41`; measured both dtypes | why every call site passes a fresh fancy-indexed slice | **[V]** yes |
| 11 | `R(118.5) = 119` while `np.round(118.5) = 118` | arithmetic | the NET `k` grid at the design's named banker's-rounding hazard | **[V]** yes |
| 12 | faiss is **bit-deterministic in the search `k`**: the independent `deployed_vote(topk=k')` reproduces the in-line truncation with `max\|Δvote\| = 0.0` at every `k'` | measured, both dtypes, fresh slices | GATE-FIXK20's independent conjunct | **[V]** on mechanism; the `0.0` itself is **[U]**. **See the erratum below** |
| 13 | raw HateMM contains exact-duplicate keys with **opposite labels** (item 537 has two bank neighbours at sim 1.0), so its top-`k'` set is genuinely non-unique | measured on the operative raw cache | GATE-FIXK20's tie exclusion | **[U]** needs a cache load. **Inert:** the tie mask is computed in-job from the realised sims, not from this claim; head space realises **zero** ties |
| 14 | guard predicate coverage at freeze time: **983** repo files matched, **14** unmatched paths containing "test" | `c09guard.verify_predicate()` on the live tree | GATE-LEDGER's coverage emission | **[V]** yes — and **recomputed in-job**, not asserted from this number |
| 15 | all 14 unmatched residues are non-test-split | enumerated: 4 LLaMA-Factory `tests/` sources, 5 `*selftest*` analysis outputs, `REDTEAM_UNTESTED_CELLS.md`, 4 `external/baselines` sources | the perimeter's completeness | **[V]** yes |
| 16 | the guard blocks all 60 `test_seen_*.pt` in the two operative cache dirs, all `data/gt/*/test.jsonl`, the `_shards/test_seen/*.pt` trees, and `clap_*_test.pt`; and blocks **none** of the 319 paths the job legitimately opens | enumerated both directions against the live tree | the H-L3 boundary | **[V]** yes |
| 17 | no competing `sitecustomize.py` exists on the `HateVideo` path, so the PYTHONPATH one wins | `python -c` under the sbatch's PYTHONPATH | the job-wide perimeter | **[V]** yes — and the sbatch **asserts** `builtins.open.__name__ == "_guarded_open"` before the mints |
| 18 | the four frozen modules' sha256 match the live files, and `headspace_mint.py`'s matches the `mint_script_sha256` pinned inside the banked mints | `sha256sum` | the whole import surface | **[V]** yes — and checked in-job twice (wrapper, then arena) |
| 19 | runtime ≈ 45–65 CPU-min (36 mints ≈ 20–36 min + arena ≈ 20–30 min), peak RSS < 1.2 GB | extrapolated from timed dry runs at reduced draw counts, twice, by two independent reviewers | the 8 CPU / 32 GB request; §2's ≈115-min budget | **[U]** wall clock is an estimate. **Inert:** no `--time` is set, and the memory margin is ~25× |
| 20 | the dry run exercises every code path and returns `HALT_NO_VERDICT` under synthetic head keys (GATE-FLOOR / GATE-PARITY-FOLD / GATE-ARENA fail ⇒ fail-closed) | this session's synthetic harness (real `n`, real labels, synthetic float64 keys) | confidence that the harness runs; **nothing in the decision** | **[U]** needs execution |

**Nothing in `[U]` is load-bearing for the verdict.** 12's mechanism is `[V]` and its
measured `0.0` is re-derived in-job by the gate itself; 13 is inert because the tie mask
is computed from realised sims; 19 is unbounded by design (no `--time`); 20 is about the
harness, not the science.

### Erratum carried forward, in place

An earlier draft of GATE-FIXK20 justified a `1e-6` tolerance by asserting that *"faiss
returns sims that differ by up to ~2.1e-7 depending on the search k even when the returned
neighbour IDs are bit-identical."* **That claim is false and is retracted.** faiss is
bit-deterministic in `k`. The observed `2.086e-07` was real but its cause was claim 10 —
`_norm32` renormalising a **reused float32 array in place**, so each extra `deployed_vote`
call drifted the caller's own keys by ~6e-8. That drift would have reached `dens50` and
`class_gap`, which are decision-path features. It is closed at source (fresh slices per
call, claim 9's dtype assertion), the tolerance is now `1e-9` with the decision conjunct
exact, and the emitted text in the artifact states the true mechanism and the retraction.
The synthetic harness was regenerated with float64 keys so that it matches the real minter
in exactly the dimension the repair depends on.

---

## 3. Frozen identity — sha256

**C09 A0 frozen set (new executables):**

| path | sha256 |
|---|---|
| `scripts/analysis/c09_a0_arena.py` | `7562e43477ed5d9705ea357d4815aaea5cddd3bc0c1db8741ea5a25a04b52844` |
| `configs/c09/c09_a0.json` | `21ffdc3ff59913cd91f9d001ca66664f56b3d7f54bc62a607a63583820c626da` |
| `scripts/slurm/c09_a0_cpu.sbatch` | `3f9f181cb635afc1eb15647aaeeee2ae963290651ac64de3763acdbd66f139c7` |
| `scripts/analysis/c09_guard/c09guard.py` | `aed50842c232105f1b06182aa89512ee89dd050bdcaedec2706062c9d745f062` |
| `scripts/analysis/c09_guard/sitecustomize.py` | `b238789fd80076b0b890c4894fd8b69255792af51c80cd9fe2d6db6c53383850` |

**Imported unmodified, sha256 verified by the wrapper before the mints and again by the
arena at import:**

| path | sha256 |
|---|---|
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` |

`scripts/analysis/headspace_drive.sh` is **not invoked** — the sbatch reproduces its
layout inline — and is therefore not sha-checked.

**Namespace absence at freeze time (verified):** `artifacts/c09*` does not exist.
**Queue at freeze time (verified):** `squeue -u jehc223` empty.

---

## 4. Declared readings and declared deviations

The design determines these only partly. Each is implemented as stated, emitted into the
artifact, and recorded here because the code points at this file for them.

**Declared readings** (the design is silent or admits more than one literal reading):

1. **RANDOM-POP's sampling domain** — the size-matched random sample is drawn from the
   analysis pool `P^(τ) = ∪_f A^(f)`, the only self-consistent domain: drawing from all
   `n` would put unstable errors (in neither class) and always-correct items (already the
   negatives) into the positive class at once.
2. **RANDOM-POP's scope** — only ΔAUC is recomputed, not "every reported quantity".
   O1/NET/K-DEG against a random population are deterministic functions of that sample's
   error content (`Δacc_s = k(2p−1)/n`, ≈ −0.78·k/n at the arena's measured per-seed error
   rate), so they price the sample's label content rather than identifiability. They are
   **not** §5.1's `|P_τ|/n` identity — that identity holds only because `P_τ` is wrong at
   *every* seed, which a random sample is not.
3. **THRESH-BEST's per-item band** is scored by `mean_s net_s` (the per-item analogue of a
   per-seed quantity the deployed rule averages). The alternative reading — scoring by the
   union of the three seeds' error sets — is computed and emitted with
   `read_by_K_DEG: false`. On synthetic data the two select different bands, so the choice
   is real and is therefore made auditable rather than silent.
4. **FIXK's per-item base** is the sign of the seed-mean `k=20` vote.
5. **FIXK's DEGENERATE exemption** is applied **per seed**: a seed with `|flip(k')| = 0` at
   every `k'` contributes no twin and is dropped from the mean, while the twin is still
   read by K-DEG if any seed is non-degenerate. The stricter reading would only make K-DEG
   fire more often, i.e. KILL more readily.
6. **`runs`** is emitted as label changes **+ 1**, so the name is literal. An affine shift
   on a z-scored column; inert.
7. **`fit_lr`'s standardisation guard** is `sd < 1e-12 → 1.0`, not `sd == 0`. A post-`l2n`
   norm column has `sd ≈ 7.5e-17`, which `== 0` would not catch, and the ULP noise would be
   amplified into an O(1) column. This is the round-1 Critical's companion repair.
8. **RNG child 4** carries the D-FELDMAN item bootstrap **and** UNSTABLE-POP's resample —
   one declared role ("item bootstrap"), one stream, fixed order. **Child 5 stays reserved
   and unused**, exactly as §5.2 pins it. Load-bearing for reproducibility: the six children
   are spawned once in `main()` and consumed across spaces in a fixed call order
   (head-hatemm → remove-null-hatemm → raw-hatemm → head-zh → raw-zh).
9. **`q_max`** is the accuracy-leg quantile only; the macro-F1 leg is not a function of
   `|P_τ|` alone, so no closed-form quantile exists for it. Descriptive; read by no rule.
10. **The NET bar is recomputed only when `n_scored ≠ n`** (the remove-null leg: `37.15` and
    `22.29`). A full-`n` space keeps the frozen `37.2 / 29.0 / 22.3 / 17.4` verbatim —
    those are rounded from `28.95 / 22.32 / 17.37` and must not be silently re-derived.
11. **GATE-FIXK20's tie exclusion** — a row whose `k'`-th and `(k'+1)`-th neighbour sims are
    exactly equal has no unique top-`k'` set, so its fixed-`k` vote is not a function of the
    data; such rows are excluded from the independent-recomputation equality and counted.
    Realised: zero in every head space, 42 across the grid in raw HateMM (claim 13).

**Declared deviations** (the design determines these and the implementation departs, in
the stated direction):

- **§8.1's "three read-counting guards"** is discharged **structurally**, not by counting.
  The feature phase lives in four top-level functions whose signatures are emitted into the
  artifact and admit no label or target array, so none of the three arrays is in scope and
  a counter over them would read 0 for *any* code whatsoever — a tautology, not a check.
  The binding, measured evidence is the `BankLabels` audit: exactly `seeds × folds` calls,
  every one served from the named fold's frozen fitting pool, zero refusals. A read would
  now be a `NameError`, which is strictly stronger than a counter.
- **GATE-NESTED's `all_folds_fit` conjunct** is an addition beyond §8.1's stated predicate:
  a fold that cannot be fitted HALTs. Deliberate and conservative — it can never yield a
  false CONTINUE — and `fit_failure_causes` names the cause so a data-caused failure
  (single-class training fold) is distinguishable from an instrument failure. Unreachable
  at the pre-declared `|P_τ|`.
- **GATE-FIXK20 is strengthened** beyond §8.1's letter: every `k'` on the grid is compared
  against an independent `M.deployed_vote(topk=k')`. §8.1's own text calls it *"the only
  gate that checks the fixed-`k` vote path, which K-DEG reads and which can KILL"*, and the
  `k'=20` identity alone cannot exercise the truncation path. HALT-direction only.
- **The remove-null and raw legs' structural checks do NOT gate.** §8.1 declares its nine
  gates plus the SHUFFLE-POP band *"the complete publication precondition of §9"*, so no
  HALT condition is added to that frozen list. The checks are computed and published under
  `other_spaces_structural_only`. §9 confines the raw leg to KILL corroboration and lets no
  raw number reach the decision; §8.2 makes the remove-null leg a sensitivity whose
  disagreement is already published as a first-class finding.
- **`holm2` treats an undefined `p` as `1.0`**, not as a blocking `None`. An uncomputable τ
  cannot reject itself and must not veto its partner; returning `(False, False)` let a
  degenerate τ_hi convert a clean τ_0 CONTINUE into a KILL — a wrong verdict, not a HALT.

**Known reporting limits, recorded rather than repaired** (none changes a verdict, rule,
threshold, gate, operating point or scope): `C09_A0_DECISION.json` carries
`identifiability_underpowered` per `(dataset, τ)` but not `identifiability_cell_marking`,
so on a KILL the "unreachable vs tested-and-failed" distinction is readable from
`C09_A0_OUT.json` (`STRATUM_OCCUPANCY.cell_marking`) but not from the decision file alone.
Both files are published together. This was raised **after** the round-7 GO and is
therefore left unchanged: the reviewed artifact is the submitted artifact.

**The eight `_OPEN` call sites** in the arena bypass the ledger by construction — all eight
are literal repo constants or the sbatch-pinned `--config` / `--outdir`. GATE-LEDGER's
`test_path_opens == 0` is a statement about every *other* open in the job.

---

## 5. Execution boundary

`sbatch scripts/slurm/c09_a0_cpu.sbatch` — **8 CPU / 32 GB, no `--time`, no `--gres`,
ZERO GPU.** Per-user cap is 16 CPU / 128 GB / 2 GPU.

Order of work, fail-closed at every step under `set -euo pipefail`:

1. DET-1 thread environment exported **before** any interpreter starts; `PYTHONPATH`
   points at `scripts/analysis/c09_guard` so `sitecustomize` installs the split guard at
   interpreter startup in **every** python process of the job; a preflight heredoc asserts
   the guard is installed and that `torch.cuda.device_count() == 0`.
2. sha256 of the four frozen modules (exit 4 on mismatch), before the mints, so a changed
   module costs seconds rather than ~25 minutes.
3. **36 CPU head mints** = 2 datasets × 3 seeds × {5 fitting-pool folds + 1
   deployed-configuration head}, via the unmodified `headspace_mint.py`. Resume-safe: an
   existing `.npz` is skipped, and each mint writes `.tmp.npz` then `os.replace`.
4. GATE-DEVFID via the unmodified `headspace_fidelity.py` — **reporting only** (§8.2), so
   it is wrapped in `if ! …; then` and cannot abort the job before a verdict exists.
5. The arena, which runs the battery and renders the verdict.

Outputs: `artifacts/c09_topo/v1/a0/C09-A0-v1/{C09_A0_OUT.json, C09_A0_DECISION.json,
C09_FIDELITY_<ds>.json}`, both JSONs written `.tmp` then `os.replace`.

**Data boundary.** Train split + `dev_seen` features only. The guard predicate is
component-wise and repo-scoped; `torch.load`, `np.load` and `json.load` all bottom out in
the guarded `open`, so a test read raises and, under `set -e`, fails the job. Each process
appends its measured counts to `$C09_LEDGER_DIR`; the arena aggregates them, and
GATE-LEDGER **requires** at least one reporting process besides the arena, so a ledger that
reads zero because nothing reported cannot pass.

---

## 6. Submission preconditions

| # | precondition | state at freeze |
|---|---|---|
| 1 | design review GO 0C/0H/0I | round 17 |
| 2 | **separate** code/resource review GO 0C/0H/0I | round 7 |
| 3 | `squeue -u jehc223` empty | verified empty |
| 4 | sha256 of the frozen set re-verified immediately before submission | to be done in the submit step |
| 5 | `artifacts/c09*` namespace absent | verified absent |
| 6 | exactly ONE CPU submission, no `--time`, no GPU | authorised by the task brief for exactly one |
| 7 | `JobHeldUser` waits are normal — never force-release, never resubmit | standing rule |

**Post-run fields, to be filled from `sacct` and the published artifact only, never from
memory:** job id, state, elapsed, the verdict, and the measured quantities quoted exactly.

---

## 7. Scope of any verdict (from §10 of the design, unchanged)

A **KILL** closes the C09 Stage-0 oracle under the frozen Stage-0 rule, at the τ values §9
scopes it to. It is not an impossibility proof for encoder-level work. A **CONTINUE** is a
Stage-0 pass only, and its Stage-1 precondition is VOID unless a proponent names an
operator that is (a) global and symmetric at inference, (b) not one of F75's three named
objectives, (c) not hard-example weighting alone, and (d) adjudicated afresh against F75,
F66, F98 and F99. A **HALT** publishes no verdict, consumes no scientific gate, and is
evidence neither for nor against C09.

**Pre-declared expectation: BAND B** (identifiability real, conversion sub-bar), from
F97's `+0.0269 / +0.0104 / +0.0182` and F98's banked `+0.0269` ceiling.

---

## 8. POST-RUN RECORD — filled 2026-08-04 from `sacct` and the published artifact only

§6 requires the post-run fields to be filled *"from `sacct` and the published artifact
only, never from memory"*. Every number below was re-read in the adjudicating session from
`sacct -j 13885`, `artifacts/c09_topo/v1/a0/C09-A0-v1/C09_A0_DECISION.json`,
`.../C09_A0_OUT.json` and `.../C09_A0.log`. Nothing above §8 is edited.

### 8.1 Job

| field | measured |
|---|---|
| job id | `13885` (`c09_a0`) |
| state / exit | `COMPLETED` `0:0` |
| start → end | `2026-08-02T08:14:15` → `2026-08-04T12:18:33` |
| elapsed | `2-04:04:18` = **187 458 s = 52.07 h** |
| TotalCPU | `17-02:15:40` = 1 476 940 s (≈ 7.88 of the 8 requested cores busy throughout) |
| MaxRSS | `1 282 076 K` = **1.22 GiB** (request 32 GB; ~26× margin, as claim 19 predicted) |
| GPU | **zero**, as frozen — CPU-only sbatch, no `--gres` |
| node | `foscsmlprd01.its.auckland.ac.nz` |

Per-phase wall times, from `C09_A0.log` (arena started `2026-08-02T08:42:10`):

| phase | secs | `P_0` | `P_τhi` |
|---|---|---|---|
| head hatemm | 60 762.9 | 73 | 36 |
| head_remove_null hatemm | 60 882.8 | 72 | 36 |
| raw hatemm | 284.2 | 116 | 57 |
| head zh | 63 596.3 | 50 | 26 |
| raw zh | 255.1 | 88 | 45 |
| **arena total** | **185 781.3 s = 51.61 h** | | |

Preflight + 36 mints + 2 GATE-DEVFID runs occupied `08:14:15 → 08:42:10` = **1 675 s
≈ 27.9 min**, inside claim 19's `20–36 min` band. **Claim 19's arena estimate was the one
that failed** — see §8.7.

### 8.2 Verdict

**`DECISION.verdict = KILL`**, `verdict_provisional = KILL`.
`kill_scope.rules_fired = ["K-FELDMAN@tau_0", "K-REACH@tau_hi"]`.
`kill_scope.closes = "tau in {tau_0, tau_hi} only; neither precision nor AUC is monotone
in tau (§9)"`.

Under §9 this is a KILL by **three independent routes**, not one: `K-FELDMAN` fails at
both `τ` on both datasets, `K-NET` clears in no cell anywhere, and `K-DEG` fires at `τ_0`.
`K-REACH` — the one necessary condition the candidate does satisfy — clears at `τ_0` on
both datasets and fails at `τ_hi` on both.

### 8.3 Publication precondition (§9) — all nine gates plus the SHUFFLE-POP band

`publication_precondition.publishes_a_verdict = true`. On **both** datasets:
`GATE_FLOOR`, `GATE_PARITY_FOLD`, `GATE_FIXK20`, `GATE_BLIND`, `GATE_NESTED`,
`GATE_SELFTEST`, `GATE_ZEROOP`, `GATE_ARENA`, `SHUFFLE_POP_band` — **all `true`**.
`GATE_LEDGER.pass = true`, and `other_spaces_all_pass = true` for the three
non-decisional legs (diagnostic, not gating, per §8.1).

`GATE_LEDGER` measured: **`test_path_opens = 0`**; `dev_path_opens = 36`;
`banked_trainlog_opens = 6` (declared dev-side total 42); **39 of 39 expected processes
reporting**; predicate coverage re-derived in-job at 983 repo files matched / 14 unmatched
paths containing "test", the residue enumerated in §2 claim 15. No stale ledger files.

`SHUFFLE-POP` null means: `0.4801 / 0.4798` (HateMM `τ_0 / τ_hi`),
`0.4699 / 0.4799` (MHC-ZH), all inside the `[0.45, 0.55]` band.
`GATE-DEVFID` (reporting only): `Δ_3seedmean` `−0.0093` (HateMM) / `−0.0086` (MHC-ZH),
`STOP_RULE_TRIGGERED = false` on both; no `PROXY_FIDELITY_FLAG`.

### 8.4 `K-REACH` — the pool is arithmetically big enough at `τ_0`, and only just short at `τ_hi`

Bar: `Δacc ≥ +0.050` **and** `ΔmF1 ≥ +0.050` on **both** datasets.

| dataset | τ | `Δacc_O1` | `ΔmF1_O1` | `K_REACH` |
|---|---|---|---|---|
| HateMM | `τ_0` | **+0.0981** | **+0.1021** | **clears** |
| MHC-ZH | `τ_0` | **+0.0864** | **+0.1010** | **clears** |
| HateMM | `τ_hi` | +0.0484 | +0.0505 | fails |
| MHC-ZH | `τ_hi` | +0.0449 | +0.0530 | fails |

**The `τ_hi` failure is on the accuracy leg alone**: both macro-F1 legs clear `+0.050`
(`+0.0505`, `+0.0530`); both accuracy legs fall short (`+0.0484`, `+0.0449`). §9's
`K-REACH`-cleared-at-`τ_0`-failed-at-`τ_hi` scoping applies verbatim: the co-primary is
closed **on reach alone**, and **that closes nothing about identifiability or conversion at
`τ_hi`**, both of which are still measured and reported there and must not be read as
adjudicated.

`q_max` (accuracy leg, descriptive): `0.4795` / `0.4200` head, `0.6724` / `0.6705` raw.

### 8.5 `K-FELDMAN` — the candidate's own discriminator, and it fails everywhere

`K_FELDMAN_per_dataset` is `false` at **both** `τ` on **both** datasets.

| dataset | τ | `AUC_strat` FULL | BASE | `ΔAUC` | `PERM_STRUCT` p | `PERM_STRUCT_COND` p |
|---|---|---|---|---|---|---|
| HateMM | `τ_0` | 0.50640 | 0.53487 | **−0.02848** | 0.85614 | 0.83017 |
| HateMM | `τ_hi` | 0.52009 | 0.48932 | +0.03076 | 0.26474 | 0.16384 |
| MHC-ZH | `τ_0` | 0.59704 | 0.61347 | **−0.01643** | 0.51349 | 0.48152 |
| MHC-ZH | `τ_hi` | 0.56391 | 0.56545 | −0.00155 | 0.35864 | 0.29870 |

Every `reject` is `false` at `α = 0.05` after Holm over the two `τ` within each dataset and
family; the smallest p in the whole table is `0.164`. **At `τ_0`, on both datasets, `ΔAUC`
is negative** — the label-free structural block does not merely fail to add signal over
`BASE`, it costs a little. The GBM capacity arm (`DET-4` Tier-C, read by no rule) does not
rescue it: `+0.0318 / +0.0207 / −0.0050 / −0.0205`.

Against §6.1's hypothesis table this is the **H-MEMORISATION** row on both of its
predictive columns at once (`ΔAUC ≈ 0`, `NET ≈ 0 or negative`). The pre-declared
expectation was **BAND B** (*identifiability real, conversion sub-bar*); the measured
outcome is **worse than BAND B** — identifiability is not real either, at this power, in
this feature family.

`IDENTIFIABILITY_UNDERPOWERED`: `false` on HateMM at both `τ`, `false` on MHC-ZH at `τ_0`,
**`true` on MHC-ZH at `τ_hi`**; `identifiability_underpowered_any = true`. So the `τ_hi`
identifiability leg carries §9's *"not identifiable at this power"* qualifier on MHC-ZH;
the `τ_0` leg, where `K-FELDMAN` fired, is **not** underpowered on either dataset. No cell
is marked `ARITHMETICALLY_DEAD_AT_THIS_POWER`; every identifiability cell reports `LIVE`.

Controls: `UNSTABLE-POP` `ΔAUC` `−0.0190` (HateMM, n=21, CI width 0.0423) and `+0.0170`
(MHC-ZH, n=26, CI width 0.0865), neither underpowered, both non-gating. `RANDOM-POP` mean
`ΔAUC` `+0.0085 / +0.0082 / +0.0151 / +0.0048`. The item bootstrap's one-sided 95 % lower
bound is negative in all four cells (`−0.0583 / −0.0312 / −0.0425 / −0.0548`).

### 8.6 `K-NET` and `K-DEG` — conversion is negative in every cell, and the operation is a near-no-op at `τ_0`

`K-NET` bar: `mean_s net_s ≥ 37.2` (HateMM) / `≥ 29.0` (MHC-ZH) **and**
`mean_s ΔmF1_s ≥ +0.050`, both datasets simultaneously.
**`K_NET_clears` is `false` in all six `(τ, k)` cells**, and
`secondary_030_cleared_anywhere = false` — the `+0.030`-sized secondary
(`22.3 / 17.4`) is not cleared either, so §5.3's "cleared the secondary but not the
binding pair" clause does not apply.

`mean_s net_s`, measured — **every value is negative**:

| τ | mult | HateMM `k` / `mean_net` | MHC-ZH `k` / `mean_net` |
|---|---|---|---|
| `τ_0` | 1.0 | 73 / **−8.33** | 50 / **−10.00** |
| `τ_0` | 1.5 | 110 / **−27.33** | 75 / **−15.67** |
| `τ_0` | 2.0 | 146 / **−53.33** | 100 / **−28.67** |
| `τ_hi` | 1.0 | 36 / **−4.00** | 26 / **−5.33** |
| `τ_hi` | 1.5 | 54 / **−11.33** | 39 / **−14.33** |
| `τ_hi` | 2.0 | 72 / **−16.00** | 52 / **−21.33** |

Mechanism, from the same cells: at HateMM `τ_0, k=73` the operation needs precision
`π* = 0.7548` and realises `0.4429`; the pattern holds everywhere (realised precision
`0.29–0.44` against `π*` `0.63–1.06`). Four cells are marked
`ARITHMETICALLY_DEAD_ON_NET` rather than tested-and-failed (HateMM `τ_0` `k=146` and
`τ_hi` `k=36`; MHC-ZH `τ_0` `k=100` and `τ_hi` `k=26`) — the prereg's
*"`K-NET` is arithmetically unreachable for `k ≥ 132 / 96`"* is confirmed at `k = 146 / 100`.

`K-DEG` (kill line `0.95` on the maximum prediction-vector agreement over both scales):

| τ | mult | HateMM `max_pred_agree` | MHC-ZH | aggregate `K_DEG_fires` |
|---|---|---|---|---|
| `τ_0` | 1.0 | **0.9677** | **0.9620** | **fires** |
| `τ_0` | 1.5 | **0.9570** | **0.9551** | **fires** |
| `τ_0` | 2.0 | **0.9516** | 0.9482 | **fires** |
| `τ_hi` | 1.0 | 0.9435 | 0.9413 | no |
| `τ_hi` | 1.5 | 0.9409 | 0.9171 | no |
| `τ_hi` | 2.0 | 0.9435 | 0.8964 | no |

At `τ_0` the surgered prediction vector is `95.2–96.8 %` identical to a bare threshold
twin's — the operation is very nearly the thing it was supposed to improve on. The R3 H-1
repair is load-bearing here: `K-DEG` reads the maximum over both scales, and it is the
**per-item** scale that supplies these maxima (`THRESH_SYM`, e.g. `0.9677` per-item vs
`0.9418` per-seed-mean at HateMM `τ_0, k=73`). Under the pre-repair per-seed-only reading
`K-DEG` would have under-fired at exactly these cells.

### 8.7 `GATE-NULL-AGREEMENT` — the verdict is null-row robust; the *cell pattern* is not, and it disagrees in the KILL direction

`ran = true`, `dataset = hatemm`, `row_removed = 355`, `remove_null_leg_halted = false`.

**`verdict_primary = KILL` and `verdict_remove_null = KILL`.** The verdict does not depend
on the structural-zero row.

`agrees = false`, with **four disagreements, all at `τ_hi`, all on `K-DEG`**: with row 355
removed, `K_DEG_fires` becomes `true` at all three multipliers (primary: `false` at all
three). `K_REACH`, `K_FELDMAN`, `K_NET`, `IDENTIFIABILITY_UNDERPOWERED` and the cell
markings agree at both `τ`. **Every disagreement is in the KILL direction** — the
sensitivity leg is strictly *more* hostile to C09 than the primary.

**ERRATUM — reporting defect in the decision file, recorded not repaired.**
`DECISION.kill_scope.scoped_to_null_row_disagreement` reads *"this KILL does NOT survive
removal of the structural-zero row: see GATE_NULL_AGREEMENT.disagreements"*. **That
sentence is false for this run and must not be quoted.** It is boilerplate emitted by
`c09_a0_arena.py:1615-1618` whenever `agrees is False`, keyed on the coarse agreement flag
without inspecting *what* disagreed; the measured legs both return `KILL`. The correct
§8.2 scoping statement is the one above: the verdict is null-row robust, and the
disagreement is confined to `K-DEG` at `τ_hi`, in the KILL direction. The frozen executable
is **not** edited — the reviewed artifact is the submitted artifact — so this erratum is
the repair, in the same form as §2's `1e-6`-tolerance erratum.

### 8.8 The prereg's own predictions, scored

| pre-declared | predicted | measured |
|---|---|---|
| `\|P_0\|` | ≈ 79 / 60 | **73 / 50** |
| `\|P_τhi\|` | ≈ 40 / 30 | **36 / 26** |
| `τ_hi` reach, *"live but marginal on both legs"* | `0.0538 / 0.0518` (above `+0.050`) | **`0.0484 / 0.0449` (below)** |
| `K-NET` unreachable for `k ≥ 132 / 96` | — | confirmed at `k = 146 / 100` |
| banked deployed floors | HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946` | reproduced exactly, `GATE-FLOOR` and `GATE-PARITY-FOLD` `ok` on every seed and fold |
| peak RSS | `< 1.2 GB` | `1.22 GiB` |
| **runtime** | **45–65 CPU-min** (mints 20–36 + arena 20–30) | **mints 27.9 min ✓; arena 3 096 min ✗; job 3 124 min** |

The pool-size and reach predictions erred in the same direction — the real pools are
smaller and the real `τ_hi` reach lower than the F88-transferred arithmetic projected — so
the `τ_hi` co-primary that was called *"live but marginal"* was in fact already below the
bar on accuracy.

**The runtime miss is the process finding.** Claim 19 was flagged `[U]` and declared
*"inert: no `--time` is set"*, which is true of job survival and false of everything else:
the arena over-ran its `20–30 min` estimate by **103×–155×**, the job over-ran its
`45–65 min` estimate by **48×–69×**, and the campaign's serial-execution lock was held for
2 days 4 hours instead of an hour. The mechanism is named in claim 19 itself — the estimate
was *"extrapolated from timed dry runs at reduced draw counts"*, and the realised draw
counts (`D_perm 1000`, `D_shuffle 200`, `D_random 200`, `B_boot 10 000`, per fold, per
seed, per `τ`, per space) were never re-multiplied through. The memory prediction, which
was *measured* rather than extrapolated, was right to within 2 %.

**Compounding it: a 51.6-hour observability blackout.** The arena's five per-phase lines
were written to a block-buffered stream and flushed only at process exit; measured, nothing
from the arena appeared between `2026-08-02T08:42:10` and the flush at
`2026-08-04T12:18:32`. The external watcher
(`artifacts/c09_port/watch_13885.log`) could therefore report only `state=RUNNING
mints=36/36 decision=no`, unchanged, from `08:32` on 2026-08-02 until the watcher itself
stopped at `12:31` the same day. For 51.6 h the run was indistinguishable from a hang.
The standing rule this motivates is recorded in `TARGET_STATE.json` under
`process_rule_compute_projection_and_heartbeat_2026_08_04`.

### 8.9 Non-decisional legs, published for the record

The raw fused-key leg (`confined_to: "KILL corroboration only (unified_pilot_gate.arena;
F113)"`, no raw number reaching the decision) prices a **larger** single-pass inversion
population — `P_0` `116` / `88`, `P_τhi` `57` / `45` — against banked deployed accuracies
`0.8441` / `0.8480`, and it corroborates rather than contradicts the KILL.
`DATA_DEFECT_FLAG_COUNTS` reproduce the freeze-time census.
`GATE-FIXK20` recomputed independently at every `k'` on the grid: `max_abs_diff = 0.0`,
predictions identical, `changed = 0`, `d_acc = 0.0` on every seed and space; zero ties
excluded in head space, ties present only in raw HateMM as claim 13 predicted.
`GATE-ZEROOP` returns `d_acc = d_mF1 = net = 0` on the empty operator in every cell.

### 8.10 What this KILL does and does not close

Per §7 and §10, unchanged: it closes the **C09 Stage-0 oracle under the frozen Stage-0
rule, at `τ ∈ {τ_0, τ_hi}`** — `K-FELDMAN` and the `K-NET`/`K-DEG` pair both scope to those
two `τ` only, since neither precision nor AUC is monotone in `τ`, and `K-REACH` did **not**
fire at `τ_0`, so the *"closes every `τ ≥ 0` by arithmetic"* clause does **not** apply here.
It is **not** an impossibility proof for encoder-level topology intervention: one feature
set, one estimator family, one stratification, one power regime. The §11 Stage-1 seam is
moot. No test-split artifact was opened at any point (`test_path_opens = 0`).
