# C02 A0 v4 — evidence-density orbit reachability, preregistration record

**Status:** `V4_FROZEN_READY_NOT_SUBMITTED_PENDING_FRESH_INDEPENDENT_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v4` (A0)

Prospective. At freeze time no extraction job, no A0 job, no result, no decision, no
metric and no verdict exists, and none is claimed.

**Reading order.** `refine-logs/C02_A0_RECORD.md` (v1) carries the design of record —
authority, question, views, extraction contract, arena, arms, gates. `C02_A0_V2_RECORD.md`
carries the H1–H4 repairs, `C02_A0_V3_RECORD.md` the H-A/H-B repairs, and this file the
round-3 repairs. **None of v1, v2 or v3 was ever submitted; no artifact, result or
decision exists for any of them.** Their executables are removed so a superseded design
cannot be submitted by mistake; all hashes are preserved in §4.

---

## 1. Why there is a v4

The v3 freeze went to a **third, fresh** independent reviewer. Verdict:
**`REVISE (0C/4H/23I)`**, persisted at `refine-logs/C02_A0_V3_PREREG_REVIEW.md`.

That reviewer verified all 7 v3 hashes mechanically with `sha256sum -c`, confirmed the
v1/v2 executables and the `c02_edq` / `*c02den*` namespaces absent, traced `run_rac`'s
loader to the patched three-tuple branch to confirm **no reachable test path**, and
independently re-derived the ≤4 GPU-hour budget from the gt length distribution (~1–2 h).
It found the split scope, subsequence proof, fold-head arena, self-orbit exclusion,
`PARITY-NAT`, zero contract, bootstrap estimand, the `+0.050`/`+0.050` bar, F113
confinement, SLURM hygiene and the full config↔code constant audit sound.

Its repair verdicts on round 2: **H-A NOT REPAIRED**, **H-B PARTIALLY REPAIRED** — the
displacement-donation *mechanism* was correct in code, but the *claims* attached to it
were not. All four round-3 High findings are repaired below. None was argued away.

**A process failure is recorded here rather than buried.** H-A was reported as repaired in
v3 and was not. The cause: the v3 edit was a `str.replace` whose target string did not
match the file byte-for-byte, so it was a **silent no-op**, and the record asserted the
repair without re-reading the file. Every edit in v4 was applied by a helper that **exits
non-zero if its target does not match**, and every claimed removal was re-verified by
`grep` across the whole frozen set after the write. The claim "the sentence is gone from
both files" is, this time, a checked fact: `grep -rn "upper-bounds what any representation"`
over the entire v4 frozen set returns nothing.

---

## 2. The four High findings and their repairs

### H1 — the retracted upper-bound claim still survived in the arena docstring

*Finding:* the sentence "`s_Q` … upper-bounds what any representation that contracts this
orbit could buy … A failure is therefore decisive" was still at
`c02_a0_arena_v3.py:17-19`, contradicting the `RETRACTED` note in the config and the
`interpretation_boundary` the same script emits — for the second round running, while the
record claimed it had been removed.

*Repair:* removed, and verified absent by `grep` across every frozen file. The docstring
now carries the same corrected statement as the config and the emitted decision:

> `s_Q` is optimistic in the ordinary sense — it may use the best view of every item on
> **both** sides of every comparison, which no deployable system may do, and it is not a
> router. It is **not** a proven supremum over all orbit-contracting representations; it
> is one particular orbit-invariant similarity, the canonical max-matching quotient
> pseudo-metric. A **KILL is a gate verdict under the registry's frozen Stage-0 rule**,
> not a proof that no orbit-contracting representation could ever help. A **PASS**
> authorises Stage-1 design plus a fresh review only.

### H2 — the derangement could hang and then die outside the fail-closed path

*Finding:* v3's degeneracy-class grouping made **size-2 donor groups reachable**. On a
size-2 group, `rng.permutation(2)` draws the identity with probability 1/2, and the
pairwise-swap repair then oscillates `[0,1] → [1,0] → [0,1] → …` forever, exiting the
64-iteration loop with fixed points and raising a **bare `AssertionError` outside the
`Halt` path** — no result JSON, no decision JSON, roughly 25 minutes into the A0 job,
**after the GPU extraction is already spent**, under a one-submission-only rule. Round 2
had rated the same loop `Info` because it was then unreachable; v3 is what made it
reachable.

*Repair:* the repair loop is gone. The permutation is now built by **Sattolo's
algorithm**, which produces a uniformly random **single-cycle** permutation and is
therefore a derangement **by construction** for every group of size ≥ 2, in `O(m)`, with
no rejection and no loop that can fail to terminate. Stress-checked statically over 20
fold seeds on a size-2 and a size-3 group: zero fixed points, zero boundary crossings.

### H3 — "exchangeable by construction" was false

*Finding:* the v3 claim "under H0 `FULL` and `SHUFFLE` are exchangeable by construction"
is wrong. Counter-example: a purely **radial** displacement `d_i = eps * NAT_i`. The orbit
is then a function of `NAT` and adds nothing, so `FULL ≡ NATIVE` after L2 normalisation,
yet `SHUFFLE` still perturbs and degrades — so `FULL > SHUFFLE` is satisfied strictly at a
null where the orbit carries nothing.

*Repair:* the claim is **retracted** in the arena, the config and this record, and
replaced by the precise scope:

> `FULL` and `SHUFFLE` are exchangeable under the **exchangeability null** — displacements
> i.i.d. across items and independent of the item they attach to. They are **not**
> exchangeable under *every* null in which the orbit is uninformative; the radial null is
> a counter-example. **`FULL > SHUFFLE` is therefore necessary, not sufficient.**

And the reason this does not endanger the verdict, stated rather than assumed: **under the
radial null `FULL == NATIVE` exactly, so `delta_acc = 0` and the binding `+0.050` bar is
unreachable.** The bar against the paired native floor is what carries the verdict;
`SHUFFLE` and `NOISE` only exclude ways of clearing that bar which do not need the correct
within-video orbit.

### H4 — the self-test that "verified" the SHUFFLE repair could not fail

*Finding:* `shuffle_donates_displacement` re-typed the donation formula locally and then
asserted an algebraic identity of the line above it. It never called `build_arms`, so it
would have passed unchanged if `SHUFFLE` still donated the donor's absolute position —
yet the config claimed it "proves, in-job" the SHUFFLE construction, and that string is
written into the result artifact.

*Repair:* the case now builds synthetic keys, calls **`build_arms` itself**, and asserts
against the arm the production path actually returns: that `SHUFFLE[0]` is the untouched
native view; that each `SHUFFLE[v]` equals the donor's **displacement** applied to the
item's own native key; that it is **not** the donor's absolute view; and, added while
there, that `NOISE` is norm-matched to `FULL` per item and per view. The case is renamed
`shuffle_donates_displacement_via_build_arms` so the artifact string cannot outlive the
property it names.

---

## 3. What v4 does not change

Everything else is v3 as frozen: the views and their subsequence contract; the extraction
contract, split scope and guards; the fold-head arena, `PARITY-NAT`, `ARENA-1/2`,
`GATE-FID`, `GATE-EXT`, the zero contract, `VIEW_SUPPORT`; the arms other than `SHUFFLE`'s
permutation generator; the displacement-donation rule itself and the partition ×
degeneracy grouping; the `+0.050` / `+0.050` bar and the full decision rule; the
estimand-consistent paired bootstrap; the Holm family; the tie-corrected Spearman; the KRR
probe with its declared repair and declared limitation; the `__debug__` and `torch.load`
guards; the early no-clobber; the `net_fix` identity disclosure; and the C14 dedup
statement.

The extraction wrapper, extractor, view module and mint changed **only** in their record
pointer and, in the extractor, the re-pinned view-module hash. No science, guard or
constant in those four files changed.

---

## 4. Frozen identity — sha256

**V4 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v4.json` | `8ccd2464699a7029db3952bc18612ea1cfcc79ede2b946e67051df843b26a4a9` |
| `src/utils/c02_density_views.py` | `2ec193cdfa920a2d974db5c8468702614a54fa378a8df324ca5ba47b7d955592` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `66381c40a03c480bceab0af3d4c0497478e00da39a6de7ec0c33f6daf992a319` |
| `scripts/slurm/c02_density_extract.sbatch` | `a1523087253990ce4a38642214aabd2890c34e650007d844ee5b9b4e992d8a9f` |
| `scripts/analysis/c02_a0_mint.py` | `2afbe8b075aefb1cdd02669e0336c53d4306366deeed8714c7f11f58a65e78a5` |
| `scripts/analysis/c02_a0_arena_v4.py` | `71bba0f1bd47517ea8da1bbd922274f66d4b2ef6c62099ca17cc97c1364aba26` |
| `scripts/slurm/c02_a0_cpu_v4.sbatch` | `ae4a237508ebfccde51cd3552903991d60001aad89f483e4861c490a648a8cd6` |

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
| `configs/c02/c02_a0_v3.json` | `3c55214494372457fb8f2702f7ecf1c82c48b13c6b523d99e00272d2b0aa15ca` | v3 |
| `scripts/analysis/c02_a0_arena_v3.py` | `7d04a8ad8e644851fb8e25f77eee30ac12fbf7e33344dbc484bc5da240e21629` | v3 |
| `scripts/slurm/c02_a0_cpu_v3.sbatch` | `9463d642756269a77929dd3ffeb8afeab02f81c2b5bd77a20d1566d245bae399` | v3 |
| `refine-logs/C02_A0_V3_RECORD.md` (as reviewed) | `f54f08d9baa4c33863cacd452b365b857c844d5fbe7a078ec5c7124e4add8dbc` | v3 |

**Imported unmodified, sha256 asserted at run time:**

| path | sha256 | asserted by |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | the extractor |
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` | mint + arena |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | mint + arena |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | arena |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` | GATE-FID reader, unmodified |

**Namespace absence at v4 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in `data/CLIP_Embedding/HateMM` or
`data/CLIP_Embedding/MHC_zh`.

---

## 5. Execution boundary

- **Two SLURM submissions, one each, in this order:**
  1. `sbatch scripts/slurm/c02_density_extract.sbatch` — 8 CPU / 1 A100 / 64 G, no
     `--time`.
  2. `sbatch scripts/slurm/c02_a0_cpu_v4.sbatch` — 8 CPU / 0 GPU / 32 G, no `--time`.
- No `--time`, dependency, array, singleton, requeue, chain, force or release path.
  `PENDING (JobHeldUser)` is normal, may last hours, and **must never be force-released**.
- 8 CPU each, so the 16-CPU aggregate-cap wedge cannot occur; the two jobs are strictly
  serial.
- `squeue` must be checked immediately before the extraction is submitted and must show
  no other candidate's GPU or teacher pilot running.
- **Executed at preparation time, on the login node, and nothing else:**
  `python -m py_compile`; `bash -n`; `json.load` on the config; the view module's
  pure-string `self_test()`; the arena's `oracle_self_test()` on synthetic arrays; a
  synthetic-array stress of `derangement_within` over 20 fold seeds on size-2 and size-3
  groups; a synthetic dry run of the bootstrap, Holm, Spearman and KRR helpers; and a
  `json`-parse count of whitespace-only and over-length rows in
  `data/gt/<DS>/{train,val}.jsonl`. **No `.pt` cache, model, video, teacher, GPU, SLURM
  job or test path was opened, and no scientific quantity exists.**
- Execution requires a **fresh independent static review** of the v4 set returning
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
