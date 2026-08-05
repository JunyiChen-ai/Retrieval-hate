# C02 A0 v9 — preregistration record, with the measured-claims register

**Status:** `V9_FROZEN_READY_NOT_SUBMITTED_PENDING_SCOPED_RE_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v9` (A0)

Prospective. No extraction job, no A0 job, no result, no decision, no metric and no
verdict exists.

**Reading order.** `C02_A0_RECORD.md` (v1) is the design of record; `V2` the H1–H4
repairs, `V3` the H-A/H-B repairs, `V4` the round-3 repairs, `V5` the Info closure, `V6`
the last four Info items, `V7` the three strings, `V8` the **erratum**, this file the
three changes below. **None of v1–v8 was ever submitted.**

---

## 1. What v9 changes — exactly three things

### 1.1 The last Info: the `O(topk)` claim was wrong about what `k` bounds

`k = topk` was justified partly as "`O(topk)` rather than `O(n_bank)` per view pair".
False under the natural reading: **a flat inner-product index computes every inner product
regardless of `k`**. What `k` bounds is the selection heap and the `(nq × topk)` result
width, not the scan. Reworded in the arena docstring and in the config's erratum entry to
say what is actually true, with the previous wording explicitly marked as corrected. The
other two reasons — provable sufficiency for the top-`topk`, and **literally** the deployed
call for a singleton orbit — are unaffected and over-determine the choice.

### 1.2 The GPU-hour cap is now enforced **in-job**, fail-closed

This was the largest-downside item on the v8 measured-claims register: condition (f) makes
an over-budget run a protocol violation whose result is **void**, and nothing enforced it.

- `scripts/slurm/c02_density_extract.sbatch` computes **one absolute deadline at job
  start** — `BUDGET_CAP_SECONDS = 14400` (4.0 GPU-h) minus `BUDGET_MARGIN_SECONDS = 600` —
  and passes it to **both** dataset invocations, so the second inherits whatever the first
  left. Basis is `SLURM_JOB_START_TIME` when exported, else the wrapper's own `date +%s` a
  few seconds later; the 600 s margin dominates that difference.
- `budget_check` refuses to **start** work whose completion could cross the deadline,
  requiring headroom of `max(2 × slowest item so far, 60 s)`, so one unusually slow item
  cannot straddle it.
- On breach: publish `BUDGET_BREACH_<dataset>.json` — an **accounting** record only — and
  exit **5**.

**The guard can only ever stop work.** View caches and the manifest are written only after
a whole split completes, so a breach leaves the in-progress dataset **entirely unwritten**
and any completed dataset intact; the A0 then fails closed on the missing manifest or view
cache rather than reading a truncated bank. It touches **no** scientific semantic,
threshold, arm, metric or decision rule.

Guard logic exercised on synthetic clocks: proceeds with 3600 s remaining and a 30 s
slowest item; halts at 59 s remaining with no history (60 s floor), at 100 s remaining with
a 90 s slowest item (180 s required), and at 1000 s remaining with a 600 s slowest item
(1200 s required); proceeds at 200 s remaining with a 30 s slowest item.

### 1.3 The measured-claims register is adopted (§3 below)

---

## 2. The habit this record institutionalises

> **When two defects are live in one dry run, re-measure every claim attributed to the
> first after fixing the second.**

That is the general rule the v8 erratum teaches, and it is written here because it is the
cheapest thing that would have caught it. The companion rule, for review requests: **name
the measurements a design claims to have made, not only the arguments it makes.** Six
review rounds verified a correct argument and never questioned the false empirical claim
standing beside it, because a claim phrased as already measured reads as settled context
and is exactly what a static reviewer cannot test.

---

## 3. MEASURED-CLAIMS REGISTER

Every empirical claim in the frozen set. `[V]` verified by a static reviewer without
execution · `[D]` documentary, source checked but measurement not reproduced · `[U]`
asserted as measured, **not** re-derivable without execution.

| # | claim | how obtained | what depends on it | static-reviewer re-derivable? |
|---|---|---|---|---|
| 1 | `n = 744 / 107 / 579 / 78` train and val rows | `wc -l` on the four gt files | fold sizes, every pooled metric | **[V]** yes |
| 2 | whitespace-only `text`: 39 / 9 / 0 / 0 | `awk` over gt | `EMPTY_TEXT` counts, `VIEW_SUPPORT` | **[V]** yes |
| 3 | `len(T) > 12000`: 9 / 1 / 0 / 0 | `awk`, stable across 12000/12052/12200-byte thresholds | `LENGTH_GUARD` counts | **[V]** yes |
| 4 | `max_chars` 80731 / 12275 / 708 / 343 | `json.loads` then `len(text)`; measured twice, identically | nothing — no code path reads it | **[V]** yes (round-2's 80732/12276/710 were raw-character counts that missed JSON escapes) |
| 5 | full-identity 48 / 10 / 0 / 0; text-only `view_support` 0.9355 / 0.9065 / 1.0 / 1.0 | derived from 2 + 3 | documentation; the **gate** computes its own | **[V]** yes |
| 6 | runtime `view_support` ≈ 0.9341 (HateMM) / 1.0000 (ZH) | `1 − 49/744` given the one known zero-guard row | prediction only; hedged "*if the extraction reports…*" | **[V]** arithmetic yes; the census itself is **[U]** (see 12) |
| 7 | banked `B_fid` 0.0093 / 0.0086 | read from `headspace_fidelity{,_zh}_OUT.json` | context for the 0.050 stop rule | **[V]** yes |
| 8 | `fixed − broken = n·Δacc` identically | algebra | the net-fix clause is discharged by the accuracy bar | **[V]** yes |
| 9 | a lone degenerate item's displacement is **exactly zero** | extractor copies one vector into every view slot; zero-guard writes `zero.clone()` six times | the `SHUFFLE` singleton-drop justification | **[V]** yes, bitwise |
| 10 | `mechfix_ops._norm32` can alias its input | read `mechfix_ops.py:37-42` | the arena's always-copy `_norm32` | **[V]** yes |
| 11 | the `k = topk` exactness argument | derivation | the oracle's correctness | **[V]** yes |
| 12 | structural-zero census "1 on HateMM train, 0 on MHC-ZH" | prior records | claim 6's prediction; the tie-reachability remark | **[U]** needs a `.pt` load. **Inert:** `ZERO_CONTRACT` computes the masks and halts on mismatch; `VIEW_SUPPORT` computes the real fraction |
| 13 | exhaustive `k = n_bank` **is** bit-equal to `k = 20`; max \|Δsim\| = 0.0 | this session's synthetic dry run | **nothing** — `k = topk` is retained on two non-empirical grounds | **[U]** needs execution. Plausible on mechanism; **decorative by construction**, which is the right structure for an erratum |
| 14 | KRR synthetic R²: 0.0087 pre-repair, 0.8788 post, −0.0163 null | synthetic dry run | whether the z-scoring repair was needed | **[U]**. Secondary, **ungated**, deferred to Stage-1 |
| 15 | GPU projection ~8760 forwards + 1508 decodes ⇒ ~1.5–2.5 GPU-h | forward count derived; wall clock **estimated** | the budget headroom | forward count **[V]** (independently re-derived ≈8758); wall clock **[U]** — **now bounded by the in-job guard of §1.2** |
| 16 | the deployed CLI is byte-identical to the banked sbatch files | inherited from `headspace_mint.py:126-153`, hash-pinned | proxy-head fidelity | **[U]** not diffed against the two banked sbatch files. Bounded by `GATE-FID`, which runs first |
| 17 | sacct budget basis (13468 = 02:00:08 etc.) | `sacct` | the 4.0-hour cap | **[D]** source checked |
| 18 | C01 zero-contract provenance for row 355 | prior records; the arena tags criteria 1 and 4 `DOCUMENTARY_CITATION_NOT_COMPUTED` | the zero-contract treatment | **[D]** |

**Nothing in `[U]` is load-bearing for the verdict.** 13 is decorative by construction, 14
and 12 touch only secondary or self-computing quantities, 16 is bounded by a gate that runs
before the arena, and 15 — previously the largest downside — is now enforced in-job.

---

## 4. Frozen identity — sha256

**V9 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v9.json` | `62b38cd95cf9cd4035ae2efac67123560d2cbb2d3889134c65a4083155e4d729` |
| `src/utils/c02_density_views.py` | `44fbb00bf88ed1cbe7df2346d0961a172e8cfadd202af49d8b75f8ad7def3a52` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `bb698ab97b4d58f1c7af16cdb2053e45891785cd4be97b21fa187ea13327fc86` |
| `scripts/slurm/c02_density_extract.sbatch` | `c50eed383b8ecb1829d13ffc5f298b427453dfd06c6371e274f0b28933048d4c` |
| `scripts/analysis/c02_a0_mint.py` | `e6430b76b7ccdd831ddb9939500aa24ea70d9662b62b955a2a11273a3b00ac1b` |
| `scripts/analysis/c02_a0_arena_v9.py` | `8cdaf1d3dc1e72ae2f4073a42b97ab9d88791a4df6ef920f1473080a526d8f93` |
| `scripts/slurm/c02_a0_cpu_v9.sbatch` | `f91875d89c61926fa28aedbd191c1b88bf80c1a1db0e9000003a5feb5f03807c` |

**Superseded (executables removed, never submitted):** v1 `0b8a8289…`; v2 `2d4b7148…`;
v3 `3c552144…`; v4 `8ccd2464…`; v5 `2d90f7bd…`; v6 `8b0572f1…`; v7 `4fac6050…`; v8 config
`280c7b81…`, arena `7f8f491e…`, wrapper `85576a24…`, record
`4745ff1bad6eb7c1b99391afd57077deaefc2cb1d3dcea6c8802e91b7f68901d`. Full values are in
each version's record.

**Imported unmodified, sha256 verified by the wrapper before the mints:**
`headspace_fidelity.py` `72fd8e0a…`, `mechfix_ops.py` `635c1312…`,
`mechnov_pairverify.py` `77b0defd…`, `headspace_mint.py` `cefdf8dc…`; plus
`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…` asserted by the extractor.

**Namespace absence at v9 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in either cache directory.

---

## 5. Execution boundary

`sbatch scripts/slurm/c02_density_extract.sbatch` (8 CPU / 1 A100 / 64 G, no `--time`),
then `sbatch scripts/slurm/c02_a0_cpu_v9.sbatch` (8 CPU / 0 GPU / 32 G, no `--time`). No
dependency, array, singleton, requeue, chain, force or release path. `JobHeldUser` is
normal and must never be force-released. Distinct exit codes: 1 generic, 3 A0 fail-closed
HALT, 4 frozen-module hash mismatch, 5 GPU-budget breach.

**Operator preconditions no code can enforce**, mandatory immediately before `sbatch`:
re-verify the seven sha256 values, and run the `squeue` check for amendment condition (e).
Recorded in §6.

**Executed at preparation time and nothing else:** `py_compile`, `bash -n`, `json.load`,
the view module's pure-string `self_test()`, the arena's synthetic `oracle_self_test()`,
synthetic stresses of the derangement/bootstrap/Holm/Spearman/KRR helpers and of the new
budget guard, and `json`-parse counts over `data/gt/<DS>/{train,val}.jsonl`. **No `.pt`
cache, model, video, teacher, GPU, SLURM job or test path was opened.**

---

## 6. Submission preconditions and post-run fields

| field | value |
|---|---|
| seven v9 sha256 re-verified at submit time | *(pending)* |
| `squeue -u jehc223` at submit time | *(pending)* |
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| budget guard fired? | *(not run)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
