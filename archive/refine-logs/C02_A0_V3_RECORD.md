# C02 A0 v3 — evidence-density orbit reachability, preregistration record

**Status:** `V3_FROZEN_READY_NOT_SUBMITTED_PENDING_FRESH_INDEPENDENT_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v3` (A0)

Prospective. At freeze time no extraction job, no A0 job, no result, no decision, no
metric and no verdict exists, and none is claimed.

**Reading order.** `refine-logs/C02_A0_RECORD.md` (v1) carries the design of record —
authority, question, views, extraction contract, arena, arms, gates — and is superseded
only where v2 and v3 change it. `refine-logs/C02_A0_V2_RECORD.md` carries the H1–H4
repairs. This file carries the v3 repairs. **Neither v1 nor v2 was ever submitted; no
artifact, result or decision exists for either.** Their executables have been removed so
a superseded design cannot be submitted by mistake; all hashes are preserved in §4.

---

## 1. Why there is a v3

The v2 freeze went to a **second, fresh** independent reviewer. Verdict:
**`REVISE (0C/2H/19I)`**, persisted at `refine-logs/C02_A0_V2_PREREG_REVIEW.md`.

That reviewer re-derived the H2 counts independently with `grep`/`awk` and matched them
exactly; re-derived the `net_fix = n·Δacc` identity; matched all 7 v2 hashes and all 5
run-time-pinned module hashes; confirmed v1 executables and the
`artifacts/c02_edq` / `*c02den*` namespaces absent; found **no reachable test path**
anywhere including through imported modules; confirmed F113 is respected; and confirmed
the ≤4.0 GPU-hour budget is plausible.

Its repair verdicts on the previous round: **H1 partially repaired, H2 repaired, H3
repaired, H4 not repaired in the frozen executables.** Both remaining High findings are
repaired below. Neither was argued away.

---

## 2. The two High findings and their repairs

### H-A — the retracted upper-bound claim survived in two frozen files

*Finding:* v2's record said the "`s_Q` upper-bounds what any orbit-contracting
representation could buy … a failure is therefore decisive" claim had been replaced
"everywhere". It had not: the sentence survived verbatim in the frozen arena docstring
and in the config's `oracle_status`, where it **contradicted the corrected
`interpretation_boundary` that the same script emits**. The record's own claim of
completeness was therefore false.

*Repair:* the sentence is gone from **both** frozen files. `oracle_status` in
`configs/c02/c02_a0_v3.json` and the docstring of `scripts/analysis/c02_a0_arena_v3.py`
now carry the same corrected statement the decision emits, and the config records the old
wording as explicitly `RETRACTED`. The accurate formulation, in all three places:

> `s_Q` is optimistic in the ordinary sense — it may use the best view of every item on
> **both** sides of every comparison, which no deployable system may do, and it is not a
> router. It is **not** a proven supremum over all orbit-contracting representations; it
> is one particular orbit-invariant similarity, the canonical max-matching quotient
> pseudo-metric. A **KILL is a gate verdict under the registry's frozen Stage-0 rule**,
> not a proof that no orbit-contracting representation could ever help. A **PASS**
> authorises Stage-1 design plus a fresh review only.

### H-B — `SHUFFLE` was still degraded under the design's own null

*Finding:* v2 closed the cross-partition leak, but the *consequence* survived through a
bank-side channel. `SHUFFLE` donated item `π(i)`'s **absolute** view keys, so bank row `j`
became a near-duplicate of bank row `π(j)`. Every true neighbour was mirrored onto an
unrelated-label row, so `SHUFFLE` degraded **under H0 itself** — the `FULL > SHUFFLE`
conjunct was satisfiable when the orbit carries nothing, and therefore proved nothing.
The reviewer noted this is structurally the same vacuity as H3's `net_fix_rate`, which had
correctly been labelled in place.

*Repair — a fix, not a relabel.* `SHUFFLE` now donates the **displacement**:

```
z_i^v  :=  NAT_i + ( view_v(pi(i)) - NAT_pi(i) )
```

No component of `π(i)`'s **position** enters, so no spurious cross-item near-duplicate is
created; what is destroyed is exactly the correspondence between a density displacement
and the video it came from. **Under H0 — "the orbit carries nothing beyond `NAT`" —
`FULL` and `SHUFFLE` are exchangeable by construction**, which is what makes the conjunct
a real test. It is verified in-job by the new self-test case
`shuffle_donates_displacement`, which asserts that the offset is the donor's *and* that
the result is **not** the donor's absolute view.

Two supporting changes:

- **Donor grouping is now partition × degeneracy class.** The partition boundary (fitting
  pool vs held-out fifth) remains absolute. The degeneracy split stops an item whose own
  orbit is the identity from receiving a real displacement it would never have under
  `FULL` — 48 of 744 HateMM train items, 0 of 579 MHC-ZH. A class group with fewer than
  two members merges into the other class of the **same** partition, never across it; the
  merge count is reported. Verified by the self-test case `degeneracy_matched_groups`.
- **Control roles are now stated honestly.** `NOISE` and `SHUFFLE` are **both** fair nulls
  and both remain binding PASS conjuncts. `NOISE` fixes the number of extra vectors and
  the per-item displacement **norms** while randomising direction. `SHUFFLE` keeps **real
  displacement directions** but mismatches the donor video, which makes it the *stronger*
  control: if density displacements share a common direction, `SHUFFLE` approaches `FULL`
  and the conjunct fails. Neither creates a cross-item near-duplicate.

---

## 3. What v3 does not change

Everything else is v2 as frozen: the views and their subsequence contract; the
extraction contract, split scope and guards; the fold-head arena, `PARITY-NAT`,
`ARENA-1/2`, `GATE-FID`, `GATE-EXT`, the zero contract, `VIEW_SUPPORT`; the arms other
than `SHUFFLE`; the `+0.050` / `+0.050` bar and the full decision rule; the estimand-
consistent paired bootstrap; the Holm family; the tie-corrected Spearman; the KRR probe
and its declared repair and declared limitation; the `__debug__` and `torch.load` guards;
the early no-clobber; and the C14 dedup statement. The v2 record's §2 (H1–H4) and §3
(Info adoptions) stand as written, with H4 now completed by §2 H-A above.

The extraction wrapper, extractor, view module and mint changed **only** in their record
pointer (`C02_A0_V2_RECORD.md` → `C02_A0_V3_RECORD.md`) and, in the extractor, the
re-pinned view-module hash. No science, guard or constant in those four files changed.

---

## 4. Frozen identity — sha256

**V3 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v3.json` | `3c55214494372457fb8f2702f7ecf1c82c48b13c6b523d99e00272d2b0aa15ca` |
| `src/utils/c02_density_views.py` | `531d4574a6c678132cb76510af0570067891a64ab5aa0a751f638b7f99ffd2fc` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `128461322f0b8f8d66478fc5bc296dba282850bd56f9b4665f06733af99149a0` |
| `scripts/slurm/c02_density_extract.sbatch` | `001a089107e40f486a14eafd2daa52fb80033450b382dd41f204f46e8862de5a` |
| `scripts/analysis/c02_a0_mint.py` | `da8a49187b821f0da15c7cec28421317225213f78f40f20d5c62c58c0ab71d33` |
| `scripts/analysis/c02_a0_arena_v3.py` | `7d04a8ad8e644851fb8e25f77eee30ac12fbf7e33344dbc484bc5da240e21629` |
| `scripts/slurm/c02_a0_cpu_v3.sbatch` | `9463d642756269a77929dd3ffeb8afeab02f81c2b5bd77a20d1566d245bae399` |

**Superseded, preserved for the audit trail (executables removed, never submitted):**

| path | sha256 | round |
|---|---|---|
| `configs/c02/c02_a0_v1.json` | `0b8a8289e7438396ce081fdf872f7d18017f870640fa33a687099de4066b53d1` | v1 |
| `scripts/analysis/c02_a0_arena.py` | `92abe7d8157a54f89a47657fb1edaf4a8f90e55b873c3fd03840aa940593fa41` | v1 |
| `scripts/slurm/c02_a0_cpu.sbatch` | `2b55c67834fc6dfdaf9a932be634c735b5362edcd128cfd5aa6e3829fc82c281` | v1 |
| `refine-logs/C02_A0_RECORD.md` (as reviewed) | `3c703b77d7cd6ebeac965d60378fffba8714dadb66fe16c91cf45fbfc42e679b` | v1 |
| `configs/c02/c02_a0_v2.json` | `2d4b7148154caea6ed41ec95043c15295c63d1abf3c47467b9191d285bd98a6f` | v2 |
| `scripts/analysis/c02_a0_arena_v2.py` | `7315e3232a42c96f1bf943028bb852eb89c9d85acd902f3890fb83fcd110e01d` | v2 |
| `scripts/slurm/c02_a0_cpu_v2.sbatch` | `ccf9881ccae7019d261a393afec4e6504203b947d38a259bf4d68b0238eccbf0` | v2 |
| `refine-logs/C02_A0_V2_RECORD.md` (as reviewed) | `12c7e49e7cfc361a21b6d04903ffe8dd3677b872eecb952b5f4d924254e9949c` | v2 |

**Imported unmodified, sha256 asserted at run time:**

| path | sha256 | asserted by |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | the extractor |
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` | mint + arena |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | mint + arena |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | arena |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` | GATE-FID reader, unmodified |

**Namespace absence at v3 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in `data/CLIP_Embedding/HateMM` or
`data/CLIP_Embedding/MHC_zh`.

---

## 5. Execution boundary

- **Two SLURM submissions, one each, in this order:**
  1. `sbatch scripts/slurm/c02_density_extract.sbatch` — 8 CPU / 1 A100 / 64 G, no
     `--time`.
  2. `sbatch scripts/slurm/c02_a0_cpu_v3.sbatch` — 8 CPU / 0 GPU / 32 G, no `--time`.
- No `--time`, dependency, array, singleton, requeue, chain, force or release path.
  `PENDING (JobHeldUser)` is normal, may last hours, and **must never be force-released**.
- 8 CPU each, so the 16-CPU aggregate-cap wedge cannot occur; the two jobs are strictly
  serial.
- `squeue` must be checked immediately before the extraction is submitted and must show
  no other candidate's GPU or teacher pilot running.
- **Executed at preparation time, on the login node, and nothing else:**
  `python -m py_compile`; `bash -n`; `json.load` on the config; the view module's
  pure-string `self_test()`; the arena's `oracle_self_test()` on synthetic arrays; a
  synthetic-array dry run of the bootstrap, Holm, Spearman and KRR helpers; and a
  `json`-parse count of whitespace-only and over-length rows in
  `data/gt/<DS>/{train,val}.jsonl`. **No `.pt` cache, model, video, teacher, GPU, SLURM
  job or test path was opened, and no scientific quantity exists.**
- Execution requires a **fresh independent static review** of the v3 set returning
  `GO (0C/0H/0I)`.

---

## 6. Post-run fields — EMPTY at freeze time

| field | value |
|---|---|
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
