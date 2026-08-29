# C02 A0 v7 — evidence-density orbit reachability, preregistration record

**Status:** `V7_FROZEN_READY_NOT_SUBMITTED_PENDING_CONFINED_DELTA_RECHECK`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v7` (A0)

Prospective. No extraction job, no A0 job, no result, no decision, no metric and no
verdict exists, and none is claimed.

**Reading order.** `C02_A0_RECORD.md` (v1) is the design of record; `V2` the H1–H4
repairs, `V3` the H-A/H-B repairs, `V4` the round-3 repairs, `V5` the Info closure, `V6`
the last four Info items, this file the three findings v6 itself created. **None of v1–v6
was ever submitted.** Their executables are removed; hashes are in §3.

---

## 1. Why there is a v7

The v6 freeze returned **`GO (0C/0H/3I)`**
(`refine-logs/C02_A0_V6_PREREG_REVIEW.md`). All four v5 items — I23, N1, N2, N3 — were
verified **CLOSED in the code**. The three remaining Info findings were created by v6's own
N2 edit and were documentation-only. The reviewer recommended **stopping there**, on the
grounds that a v7 for three strings would be the third consecutive version whose only
defects were introduced by the previous version's documentation edits.

v7 exists for one reason, and it is not the token: **finding F3 showed my stated
justification for the N2 drop rule was materially wrong**, and a wrong justification in a
hash-frozen preregistration is worth one more version even when the code it describes is
correct. The reviewer's own correction is adopted verbatim in substance.

---

## 2. The three findings

| # | change |
|---|---|
| **F3** | **A wrong justification, corrected.** v6 said dropping a singleton class group "makes `FULL > SHUFFLE` HARDER" because the dropped item keeps its own displacement — a trace of the treatment inside the control. That reasoning is wrong for the case the rule exists to handle: **a lone DEGENERATE item's displacement is ZERO by construction**, so dropping it leaves `SHUFFLE` **exactly matched** to `FULL` for that item, and it contributes to neither side of the conjunct. The conservative property is real but is a different one: **merging** would have handed that degenerate item a real displacement it never has under `FULL`, making the conjunct **easier**; dropping cannot. (For a lone *non*-degenerate item the drop does leave one unshuffled real displacement in the control, which can only make the conjunct harder — that sub-case is now stated separately.) The code is unchanged; only its description was wrong. |
| **F2** | A `Halt` message named "`shuffle_groups`' merge rule", a rule v6 deleted. It now names the size-≥2 rule that actually exists, and says the branch is unreachable by construction. |
| **F1** | The config claimed the self-test proves the grouping "merges a singleton class within its own partition". It proves the opposite — that a singleton is **dropped** and leaves the donor pool entirely. Corrected to match the assertion the self-test actually makes. |

No threshold, bar, arm, metric, arena, gate or decision rule changed. The only executable
change is the `Halt` message string in F2.

---

## 3. Frozen identity — sha256

**V7 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v7.json` | `4fac60501de74e8975d3bca0209837ce416c15bdeff00cbcdb3fdd1898a94ed1` |
| `src/utils/c02_density_views.py` | `6b2107b7a3a899492e68e735fe1e49c97de8c6214c6c3fa6440dfa268a5a0740` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `0223e885cbada6eed5866258c004376cbda966a2f3a0490d30ec7857b3abce47` |
| `scripts/slurm/c02_density_extract.sbatch` | `a2e12b9b5370f96a4fd531ef0ffd538ff9859f497ff06f63eae84747511e9c27` |
| `scripts/analysis/c02_a0_mint.py` | `3696addc260f137f5100761562072f6d93bc00912bc7e744e863f948a3833484` |
| `scripts/analysis/c02_a0_arena_v7.py` | `1548a7e330b3c05557cf86ebea6bbf60368a5ec2383950f066798a9e16258fd9` |
| `scripts/slurm/c02_a0_cpu_v7.sbatch` | `592bad52e2b6ee68a45fd1f54de00f7000fd8491fe7123205913e498f659ca81` |

**Superseded (executables removed, never submitted):** v1 `0b8a8289…`/`92abe7d8…`/`2b55c678…` rec `3c703b77…`; v2 `2d4b7148…`/`7315e323…`/`ccf9881c…` rec `12c7e49e…`; v3 `3c552144…`/`7d04a8ad…`/`9463d642…` rec `f54f08d9…`; v4 `8ccd2464…`/`71bba0f1…`/`ae4a2375…` rec `de2c631d…`; v5 `2d90f7bd…`/`4f5d9cff…`/`d4c1783f…` rec `62e19eb5…`; v6 `8b0572f1…`/`d2f62adb…`/`1de480c3…` rec `95d95c63abff081a84357f3b28a88ea0c988db33eadd64348e1af248ce16607a`.

**Imported unmodified, sha256 verified by the wrapper before the mints:**
`headspace_fidelity.py` `72fd8e0a…`, `mechfix_ops.py` `635c1312…`,
`mechnov_pairverify.py` `77b0defd…`, `headspace_mint.py` `cefdf8dc…`; plus
`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…` asserted by the extractor.

**Namespace absence at v7 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in either cache directory.

---

## 4. Execution boundary

Two SLURM submissions, one each, in order:
`sbatch scripts/slurm/c02_density_extract.sbatch` (8 CPU / 1 A100 / 64 G, no `--time`),
then `sbatch scripts/slurm/c02_a0_cpu_v7.sbatch` (8 CPU / 0 GPU / 32 G, no `--time`).
No dependency, array, singleton, requeue, chain, force or release path. `JobHeldUser` is
normal and must never be force-released.

**Operator preconditions no code can enforce**, carried from every review round: re-verify
the seven sha256 values immediately before `sbatch`, and run the `squeue` check for
amendment condition (e). Both are recorded in §5.

**Executed at preparation time and nothing else:** `py_compile`, `bash -n`, `json.load`,
the view module's pure-string `self_test()`, the arena's synthetic `oracle_self_test()`,
synthetic stresses of the derangement/bootstrap/Holm/Spearman/KRR helpers, and
`json`-parse counts over `data/gt/<DS>/{train,val}.jsonl`. **No `.pt` cache, model, video,
teacher, GPU, SLURM job or test path was opened.**

---

## 5. Submission preconditions and post-run fields

| field | value |
|---|---|
| seven v7 sha256 re-verified at submit time | *(pending)* |
| `squeue -u jehc223` at submit time | *(pending)* |
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
