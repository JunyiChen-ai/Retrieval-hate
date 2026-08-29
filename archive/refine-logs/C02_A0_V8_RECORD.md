# C02 A0 v8 — ERRATUM freeze: a false empirical claim, self-detected after GO

**Status:** `V8_FROZEN_READY_NOT_SUBMITTED_PENDING_CONFINED_ERRATUM_RECHECK`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v8` (A0)

Prospective. No extraction job, no A0 job, no result, no decision, no metric and no
verdict exists.

---

## 1. What was wrong

Every freeze from v1 to v7, and the v7 review that returned `GO (0C/0H/0I)`, carried this
sentence in the arena docstring and, in shorter form, in the config and in
`C02_A0_RECORD.md §9` / `C02_A0_V2_RECORD.md §4`:

> An exhaustive `k = n_bank` faiss search is **NOT** bit-equal to the deployed `k = 20`
> call: faiss selects a different code path for large `k` and the returned float32
> similarities differ at ulp level. Measured max `|delta sim| = 1.5e-07`, enough to break
> `PARITY-NAT`.

**It is false, and the number is misattributed.** Re-measured on synthetic arrays with
private, singly-normalised operands:

| comparison | similarities bit-equal | neighbour ids equal | max \|Δsim\| |
|---|---|---|---|
| exhaustive `k = n_bank` vs deployed `k = 20` | **True** | **True** | **0.0** |
| `k = 20` per pair (the frozen path) vs deployed `k = 20` | True | True | 0.0 |
| operands normalised **twice** (the aliasing defect) vs once | False | — | **1.4901161193847656e-07** |

The exhaustive path **is** bit-equal. The entire `1.4901161193847656e-07` discrepancy
belonged to the `mechfix_ops._norm32` **aliasing** defect — the operands had been
normalised twice — and I attributed it to the search width.

**How it happened.** Both defects were present in the same first dry run. I changed the
search width first, saw the discrepancy persist, then found the aliasing bug and fixed it,
after which parity was exact. I never re-tested the exhaustive path with the aliasing fixed
in place, and wrote the conclusion as if I had. **This is a fabricated companion
measurement in the exact sense the project's numeric-provenance rule prohibits.**

**Why six review rounds did not catch it.** Every review request asked whether the
`k = topk` exactness *argument* holds. It does, and each reviewer verified it
independently. The false statement was the empirical claim standing beside a correct
argument, presented as already measured — a class of error that a reviewer who is not
re-running the measurement cannot detect.

---

## 2. What changes, and what does not

**No numerical behaviour changes.** `k = topk` per view pair is retained, and it is still
the right choice — now for the reasons that are actually true:

- it is provably sufficient for the top-`topk` (if `tau` is the `topk`-th largest per-item
  maximum, any row `>= tau` forces its item into the top-`topk`, so each view pair
  contributes at most `topk` such rows and its own top-`topk` already holds all of them);
- it is `O(topk)` rather than `O(n_bank)` per view pair;
- for a singleton orbit it is **literally the deployed call**, rather than merely equal to
  it — which is the cleanest possible basis for `PARITY-NAT`.

The retraction is recorded in the frozen arena docstring, in
`configs/c02/c02_a0_v8.json::oracle.search_width_erratum`, and in the supersession reason,
so the false claim cannot be read anywhere in the v8 set without its retraction beside it.
The superseded records v1–v7 are left as written — they are the historical evidence of what
was believed when — and this file is the erratum that governs them.

`refine-logs/C02_A0_RECORD.md §9` and `refine-logs/C02_A0_V2_RECORD.md §4` are hereby
**corrected by reference**: item 1 in each is retracted; item 2 (the `_norm32` aliasing
defect) stands and is where the `1.4901161193847656e-07` figure belongs.

---

## 3. Frozen identity — sha256

**V8 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v8.json` | `280c7b81e905373579283684e25a961604a5a948940047a71fe8cb3624ea1ed7` |
| `src/utils/c02_density_views.py` | `b427c100ae14584dfcbe3b5330bbaa1a7703171378cde7ec467e252b540a42d8` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `6c5b9fd15e653ede183a1ad1d15f6f151f2f1dfc3760883d4b7c884e992ee235` |
| `scripts/slurm/c02_density_extract.sbatch` | `afda59d243db9a6e2c1af882f2338cacaf509497e824a50b3b7fbc65c835cb0d` |
| `scripts/analysis/c02_a0_mint.py` | `3340457278124aae715a44980c6c8e1f5bf6ce3ea31b874c4de7bff8b601e85b` |
| `scripts/analysis/c02_a0_arena_v8.py` | `7f8f491e5632775fe465555622c2560ae650b148059a85b91ef2d7c515b80349` |
| `scripts/slurm/c02_a0_cpu_v8.sbatch` | `85576a244d890ff143ea16abe78ebfa145f990daf8288d7455e45b25a3635aa4` |

**Superseded (executables removed, never submitted):** v1 `0b8a8289…`/`92abe7d8…`/`2b55c678…` rec `3c703b77…`; v2 `2d4b7148…`/`7315e323…`/`ccf9881c…` rec `12c7e49e…`; v3 `3c552144…`/`7d04a8ad…`/`9463d642…` rec `f54f08d9…`; v4 `8ccd2464…`/`71bba0f1…`/`ae4a2375…` rec `de2c631d…`; v5 `2d90f7bd…`/`4f5d9cff…`/`d4c1783f…` rec `62e19eb5…`; v6 `8b0572f1…`/`d2f62adb…`/`1de480c3…` rec `95d95c63…`; v7 `4fac6050…`/`1548a7e3…`/`592bad52…` rec `a3b4f1440e70e5997a2abc0aeed9c07cdac106c0064ea030fa351a94497797e4`.

**Imported unmodified, sha256 verified by the wrapper before the mints:**
`headspace_fidelity.py` `72fd8e0a…`, `mechfix_ops.py` `635c1312…`,
`mechnov_pairverify.py` `77b0defd…`, `headspace_mint.py` `cefdf8dc…`; plus
`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…` asserted by the extractor.

**Namespace absence at v8 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in either cache directory.

---

## 4. Execution boundary

Unchanged from v7 except the wrapper name: `sbatch scripts/slurm/c02_density_extract.sbatch`
then `sbatch scripts/slurm/c02_a0_cpu_v8.sbatch`. Operator preconditions unchanged: re-verify
the seven sha256 immediately before `sbatch`, and run the `squeue` check for amendment
condition (e).

---

## 5. Submission preconditions and post-run fields

| field | value |
|---|---|
| seven v8 sha256 re-verified at submit time | *(pending)* |
| `squeue -u jehc223` at submit time | *(pending)* |
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
