# C02 A0 v5 — evidence-density orbit reachability, preregistration record

**Status:** `V5_FROZEN_READY_NOT_SUBMITTED_PENDING_DELTA_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v5` (A0)

Prospective. At freeze time no extraction job, no A0 job, no result, no decision, no
metric and no verdict exists, and none is claimed.

**Reading order.** `C02_A0_RECORD.md` (v1) carries the design of record — authority,
question, views, extraction contract, arena, arms, gates. `C02_A0_V2_RECORD.md` carries
the H1–H4 repairs, `C02_A0_V3_RECORD.md` the H-A/H-B repairs, `C02_A0_V4_RECORD.md` the
round-3 repairs, and this file the round-4 Info closure. **None of v1–v4 was ever
submitted; no artifact, result or decision exists for any of them.** Their executables are
removed; all hashes are preserved in §3.

---

## 1. Why there is a v5

The v4 freeze went to a **fourth, fresh** independent reviewer. Verdict:
**`GO (0C/0H/23I)`**, persisted at `refine-logs/C02_A0_V4_PREREG_REVIEW.md` —
**zero Critical, zero High**, with all four round-3 High findings verified `REPAIRED` in
the code rather than in the prose.

That reviewer verified all 7 v4 hashes plus 9 auxiliary/frozen-module hashes, confirmed
all nine v1–v3 executables and the `c02_edq` / `*c02den*` namespaces absent, found no
reachable non-`Halt` exception, and checked runtime survivability end to end (adapters
older than their caches, both banked caches, both P3 files, gt row counts by `wc -l`, no
broken video symlinks, scipy 1.17.1 with `.statistic`, numpy 1.26.4, faiss-cpu 1.13.2,
1.8 TB free, and F113's measured `B_fid` of 0.0093 / 0.0086 against this design's 0.050
stop bar).

**v5 exists because 23 Info findings is not a good state to run in.** It closes the list
rather than shipping with it open. **No threshold, bar, arm, metric, decision rule or
scientific constant changed in v5.** Every change is a hardening or a correction of a
statement that was inexact.

---

## 2. What v5 changes

### Code hardening

| # | change |
|---|---|
| I5 | The last member of the H2 family. A donor group of size 1 could still survive `shuffle_groups`' merge rule and produce a **bare `AssertionError` outside the `Halt` path** — no decision artifact, after the GPU is spent. It is now a `Halt`, so the run still publishes. (Unreachable at `n ≥ 579` with 5 stratified folds; fixed anyway because "unreachable" is what round 2 said about the oscillation that round 3 made reachable.) |
| I6 | A **video-decode-failure (zero-guard) row is now in the SHUFFLE degeneracy mask**. Its orbit is the identity in every space — all six views share one zero text vector, hence one head key — but the manifest's text-derived `identity_views` does not say so, so it sat in the *non-degenerate* donor class and received a real donated displacement under `SHUFFLE` while `FULL` left it untouched. Exactly the asymmetry the degeneracy grouping exists to prevent, in the direction that makes `FULL > SHUFFLE` easier. |
| I7 | The zero contract's **criterion 3** (no row with `0 < norm ≤ 1e-12`) is now applied to the **banked `img_feats` and `text_feats`** as well as the six view matrices. C01's own fail-closed halt, job `13712`, was on the **img** modality; the config claimed the four criteria were applied "verbatim" and this one was not. |
| I14 | Each view file's sha256 **recorded by the extractor in the manifest is recomputed and compared before use**, and the manifest path must equal the path opened. A stale or swapped view cache with a matching id/label vector could previously pass every gate. |
| I15 | A fail-closed HALT now **exits 3**. Previously `main()` returned 0 after catching a `Halt`, so SLURM reported `COMPLETED` for a run that produced no verdict. |
| I16 | `PARITY_NAT`'s "BIT-EQUAL on all 15 cells" is no longer a **hardcoded string**. The cells are counted, the count is asserted equal to `seeds × folds`, and the count is emitted. |
| I12 | The **secondary** raw arena can no longer destroy a completed **primary** measurement: it runs inside a `try/except Halt` that records the halt and returns the primary result. Previously a secondary halt would have propagated and turned the one permitted A0 submission into `HALT_FAIL_CLOSED_NO_DECISION`. |
| I13 | The **oracle self-test now runs first in the job**, before the 36 mints and GATE-FID. Its own stated justification — "seconds, not a queue slot" — was true inside the arena process and false for the job, where ~20–35 minutes elapsed first. |
| I4 | The **GATE-FID reader's sha256 is now enforced by the wrapper** before it is invoked. The record claimed it was "asserted at run time"; nothing asserted it. This was the third instance of a record asserting a property the code did not implement, which is the pattern that produced round 2's and round 3's H1. |

### Statements corrected

| # | correction |
|---|---|
| I1 | The arena docstring named the superseded filename. |
| I2 | `authority.record` pointed at the **v1** record. |
| I3 | `gates.GATE_FID` cited `c02_a0_cpu_v2.sbatch` — a file the freeze that wrote the citation had deleted. |
| I8 | The `k = topk` exactness claim is now stated **with its side condition**: exact whenever at most `topk` items attain `tau`; with `topk` or more exact float32 ties at `tau` inside one pair a boundary item could be dropped. The only exactly-tied rows reachable here are the structural all-zero rows, where every similarity is 0 and the prediction is invariant. The self-test uses tie-free matrices and therefore checks the generic case only — now said. |
| I9 | The tie exemption's stated reason was false in general: two tied neighbours with different labels at adjacent ranks move the vote by `2·s·(w_r − w_{r+1})/Σw`. The exemption is safe for a **different** reason, now recorded: predictions **and** sorted similarities are bit-checked on **every** row, so a tie-induced vote flip HALTs rather than passing silently. |
| I10 | The view module still claimed `C02_EXPERIMENT_PLAN.md §3.1` "already required" the length guard. That clause is a **tokenizer-truncation** rule; `L_MAX = 12000` characters is a new constant chosen here. Corrected in the module to match what the config already said. |
| I11 | `max_chars` is now defined as `len(json['text'])` in **Python characters** — exactly what `L_MAX` compares against — and re-measured a second time with identical results (80731 / 12275 / 708 / 343). A byte-length or gt-line-length count gives different numbers and is not the quantity the guard uses. No code path consumes it. |
| I20 | The `SHUFFLE` seed is now in the config alongside the noise and bootstrap seeds, together with the fact that the permutation is **head-seed-independent by construction** — deliberate, because it removes head seed as a source of control variance. |
| I23 | The `EMPTY_TEXT` justification described only `text == ""`, of which the frozen gt contains **zero** rows. The guard's actual trigger is `text.strip() == ""`, and the 39 HateMM-train rows that fire it hold whitespace, which is already truthy, so no `(none)` flip was ever being prevented. The behaviour was always correct and conservative; only the recorded reason was inexact for 100 % of the cases that fire it. Both sub-cases are now described. |
| I21 | The nulls' **linearity assumption is now named**: `SHUFFLE` and `NOISE` are vector arithmetic in head-key space while `FULL`'s views are genuine head outputs, so a materially nonlinear head handicaps both nulls — in the direction that makes both conjuncts *easier*, which can weaken a PASS but cannot manufacture a KILL. `orbit_radius_median_oof` is reported so it can be bounded post hoc; it is not gated. |
| I22 | The **uncovered residue of the `FULL > SHUFFLE` conjunct is now recorded**: a displacement with a shared component plus an item-specific component, where the movement is carried by the shared component, makes `FULL` and `SHUFFLE` differ only by noise so a strict `>` degenerates to a coin flip (`FULL > NOISE` does still hold, because NOISE destroys the shared direction). More generally, an orbit that is any deterministic function of the native key could clear the bar without the density channel carrying anything new. `retrieval_length_spearman` and `krr_length_probe` are the right instruments and are computed per arm, but they are **reported, not gated** — attaching a threshold now would be an unjustified post-hoc bar. **This bounds what a PASS may be read as and cannot affect a KILL**, and it is recorded so a Stage-1 design must address it. |
| I18 | The 4.0 GPU-hour cap is **not** enforced in-job; amendment condition (f) makes it a post-hoc `sacct` measurement and an overrun **voids** the result. Projected spend, re-derived: ~8 760 text forwards + 1 508 video decodes, ~1.5–2.5 GPU-h. ~12 % of the spend buys `dev_seen` views **no A0 code path reads**, declared as Stage-2 pre-payment. |
| I19 | Amendment condition (e) is enforced by a **manual `squeue` check** immediately before the extraction is submitted, not by an automated interlock. The check and its result are recorded in §5. |

### Accepted, not changed

**I17 — the frozen set is not git-tracked**, so the "changed only in the record pointer"
claims are not diff-verifiable and the sha256 chain is the whole audit mechanism. This is
**a repository-wide condition of this iteration, not a C02 defect**: every C01, C03 and
C04 artifact produced in iteration 8 is likewise untracked. Recorded rather than silently
fixed, because committing is not this task's authority. Mitigation in force: every frozen
file's sha256 is in §3, the extractor pins the view module's hash and asserts it at run
time, and the arena writes its own sha256 into the result artifact.

---

## 3. Frozen identity — sha256

**V5 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v5.json` | `2d90f7bd455dfcd7743fe3b7a0d0f99ff60b7e5e7ace7f449491f73d930aacf6` |
| `src/utils/c02_density_views.py` | `1db85dedc1753ce23e4267f6cb872e1f010c3a1f7b95f987bfe9ad41341e7227` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `d6fd01913f3be2a8a15c034096c95766c460da55a07c15dd38afd40823fd8884` |
| `scripts/slurm/c02_density_extract.sbatch` | `5f9cf4525f24a50180f872a88a9e3e4a234b9d4727e069f34ce5461763c01b8a` |
| `scripts/analysis/c02_a0_mint.py` | `360dc03f5301c2a85e65e4e13456f75183aec40cb3567f0435a758b407d430d5` |
| `scripts/analysis/c02_a0_arena_v5.py` | `4f5d9cff2bd7829e97c60d3902340bbf37f14c73989e320279afdd795cb08edc` |
| `scripts/slurm/c02_a0_cpu_v5.sbatch` | `d4c1783fabd7bce22ccd5c1ad144674db4dd47459f89de16b65ec6b66b824239` |

**Superseded (executables removed, never submitted):** v1 `0b8a8289…` / `92abe7d8…` /
`2b55c678…`, record `3c703b77…`; v2 `2d4b7148…` / `7315e323…` / `ccf9881c…`, record
`12c7e49e…`; v3 `3c552144…` / `7d04a8ad…` / `9463d642…`, record `f54f08d9…`; v4
`8ccd2464…` / `71bba0f1…` / `ae4a2375…`, record
`de2c631dcbbdd70256b067a1b62d41671d0a8f06a17c489ac5e65a2781813251`. Full-length values
are in the corresponding records.

**Imported unmodified, sha256 asserted at run time:**

| path | sha256 | asserted by |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | the extractor |
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` | mint + arena |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | mint + arena |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | arena |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` | **the wrapper, enforced (new in v5)** |

**Namespace absence at v5 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in `data/CLIP_Embedding/HateMM` or
`data/CLIP_Embedding/MHC_zh`.

---

## 4. Execution boundary

- **Two SLURM submissions, one each, in this order:**
  1. `sbatch scripts/slurm/c02_density_extract.sbatch` — 8 CPU / 1 A100 / 64 G, no
     `--time`.
  2. `sbatch scripts/slurm/c02_a0_cpu_v5.sbatch` — 8 CPU / 0 GPU / 32 G, no `--time`.
- No `--time`, dependency, array, singleton, requeue, chain, force or release path.
  `PENDING (JobHeldUser)` is normal, may last hours, and **must never be force-released**.
- 8 CPU each; the two jobs are strictly serial.
- **Executed at preparation time, on the login node, and nothing else:**
  `python -m py_compile`; `bash -n`; `json.load` on the config; the view module's
  pure-string `self_test()`; the arena's `oracle_self_test()` on synthetic arrays; a
  synthetic stress of `derangement_within` over 20 fold seeds on size-2 and size-3 groups;
  a synthetic dry run of the bootstrap, Holm, Spearman and KRR helpers; and `json`-parse
  counts over `data/gt/<DS>/{train,val}.jsonl`. **No `.pt` cache, model, video, teacher,
  GPU, SLURM job or test path was opened, and no scientific quantity exists.**

---

## 5. Serial-execution check (amendment condition (e))

*To be completed immediately before the extraction is submitted.*

| field | value |
|---|---|
| `squeue -u jehc223` at submit time | *(pending)* |
| other candidate GPU/teacher pilot running | *(pending)* |
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
