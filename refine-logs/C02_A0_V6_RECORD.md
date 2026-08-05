# C02 A0 v6 — evidence-density orbit reachability, preregistration record

**Status:** `V6_FROZEN_READY_NOT_SUBMITTED_PENDING_FINAL_DELTA_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v6` (A0)

Prospective. No extraction job, no A0 job, no result, no decision, no metric and no
verdict exists, and none is claimed.

**Reading order.** `C02_A0_RECORD.md` (v1) is the design of record; `V2` carries the
H1–H4 repairs, `V3` the H-A/H-B repairs, `V4` the round-3 repairs, `V5` the Info closure,
this file the last four items. **None of v1–v5 was ever submitted.** Their executables are
removed; hashes are in §3.

---

## 1. Why there is a v6

The v5 freeze went to a delta review by the round-4 reviewer, who held the full Info list.
Verdict: **`GO (0C/0H/4I)`** (`refine-logs/C02_A0_V5_PREREG_REVIEW.md`) — 19 of 23 Info
findings closed in the code and the frozen statements, 3 accepted as declared operating
conditions, 1 partially closed. The reviewer wrote that none of the four residual items
"can change the verdict, corrupt a gated quantity, touch test data, or kill either job",
and **did not recommend a v6**.

v6 exists anyway, because the four items are three sentences and one line, and because the
registry amendment's condition (a) reads best with the literal `GO (0C/0H/0I)` as the
execution key. **No threshold, bar, arm, metric, arena or decision rule changed.**

---

## 2. The four items

| # | change |
|---|---|
| **I23** | `configs/c02/c02_a0_v5.json:35` still carried only the `(none)`-flip rationale for `EMPTY_TEXT`, which applies to `text == ""` — **zero rows in the frozen gt** — and not to any of the 39 HateMM-train rows that actually fire the guard. The view module had already been corrected in v5; the config now matches it and names both sub-cases. Documentation only; the behaviour was and is correct. |
| **N1** | The config pre-measured `view_support = 0.9355` from gt text alone, but v5's I6 fix makes the **runtime** gate additionally count video-decode-failure rows as identity orbits. The two numbers are now both stated: text-only 0.9355, and expected runtime `1 − 49/744 = 0.9341` on HateMM train if the extraction reports the one known zero-guard row, `1.0000` on MHC-ZH if it reports none. Both are far above the 0.60 bar, and the config now says which one the gate reads. |
| **N2** | `shuffle_groups` could **merge a lone identity item back into the non-degenerate donor class**, undoing v5's I6 fix for ≤1 item per partition per fold. A singleton class group is now **dropped** instead of merged. Dropping leaves that item carrying its **own** displacement in `SHUFFLE` — a trace of the treatment inside the control, which makes `FULL > SHUFFLE` **harder**, the conservative direction — whereas merging handed a degenerate item a real displacement it never has under `FULL`, which made the conjunct **easier**. Dropped singletons are counted and reported, and the self-test now asserts that a dropped item leaves the donor pool entirely rather than being regrouped. |
| **N3** | `mechfix_ops.py`'s frozen hash was first verified only inside the arena, **after** the 36 mints, as a bare `assert`. The wrapper now verifies **all four** frozen imported modules — `headspace_fidelity`, `mechfix_ops`, `mechnov_pairverify`, `headspace_mint` — before the mints, exiting 4 on any mismatch. A changed module now costs seconds instead of ~25 minutes. The mint and arena still re-assert their own subsets in-process. |

---

## 3. Frozen identity — sha256

**V6 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v6.json` | `8b0572f1ee8626de540417613eb9d3dc2d6bf1db3d867b7ad4aebb9472340982` |
| `src/utils/c02_density_views.py` | `3bf0783095a7aeaf978ef7c52fe59cac0d74256ee28f83c14d9de42c84ea7746` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `8c17bdb8e4660783618ac3f335cd98f43da14eba587324debb1e80b9b93c8101` |
| `scripts/slurm/c02_density_extract.sbatch` | `51dc4d9314180f5d65282d57e8daad18d07c9854db3d627bb8a563088ca8fdc4` |
| `scripts/analysis/c02_a0_mint.py` | `49abad31349897026668b711df79028f51ed9fb628ccdf9e22095f55fd7c65a2` |
| `scripts/analysis/c02_a0_arena_v6.py` | `d2f62adbd06ec9f286220b26f656a6026e3deb336d4537b6f6d71ff266784815` |
| `scripts/slurm/c02_a0_cpu_v6.sbatch` | `1de480c3bbfad52215fb781e676062eaf17e5a909a5f6a48f770f67616b14acb` |

**Superseded (executables removed, never submitted):** v1 `0b8a8289…`/`92abe7d8…`/`2b55c678…`, record `3c703b77…`; v2 `2d4b7148…`/`7315e323…`/`ccf9881c…`, record `12c7e49e…`; v3 `3c552144…`/`7d04a8ad…`/`9463d642…`, record `f54f08d9…`; v4 `8ccd2464…`/`71bba0f1…`/`ae4a2375…`, record `de2c631d…`; v5 `2d90f7bd…`/`4f5d9cff…`/`d4c1783f…`, record `62e19eb55d9a4f81e08a18d33269d94ca94e9c71595a44ca9e8b2094e6ba0f18`.

**Imported unmodified, sha256 verified by the wrapper before the mints (new in v6):**
`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…` (asserted by the extractor),
`headspace_mint.py` `cefdf8dc…`, `mechnov_pairverify.py` `77b0defd…`,
`mechfix_ops.py` `635c1312…`, `headspace_fidelity.py` `72fd8e0a…`.

**Namespace absence at v6 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in either cache directory.

---

## 4. Execution boundary

Two SLURM submissions, one each, in order:
`sbatch scripts/slurm/c02_density_extract.sbatch` (8 CPU / 1 A100 / 64 G, no `--time`),
then `sbatch scripts/slurm/c02_a0_cpu_v6.sbatch` (8 CPU / 0 GPU / 32 G, no `--time`).
No dependency, array, singleton, requeue, chain, force or release path. `JobHeldUser` is
normal and must never be force-released.

**Operator preconditions the code cannot enforce**, both carried from the reviewer:
re-verify the seven sha256 values immediately before `sbatch` (the frozen set is untracked
and the tree has concurrent writers), and run the `squeue` check for amendment condition
(e). Both are recorded in §5.

**Executed at preparation time and nothing else:** `py_compile`, `bash -n`, `json.load`,
the view module's pure-string `self_test()`, the arena's synthetic `oracle_self_test()`,
synthetic stresses of the derangement/bootstrap/Holm/Spearman/KRR helpers, and
`json`-parse counts over `data/gt/<DS>/{train,val}.jsonl`. **No `.pt` cache, model, video,
teacher, GPU, SLURM job or test path was opened.**

---

## 5. Submission preconditions and post-run fields

| field | value |
|---|---|
| seven v6 sha256 re-verified at submit time | *(pending)* |
| `squeue -u jehc223` at submit time | *(pending)* |
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
