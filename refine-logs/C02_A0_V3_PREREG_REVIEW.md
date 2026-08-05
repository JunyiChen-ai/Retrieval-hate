# C02 A0 v3 — fresh independent static preregistration review

**Reviewer:** fresh independent static reviewer (round 3; no access to the implementer's
reasoning, no contact with either prior reviewer)
**Date:** 2026-07-30 (Pacific/Auckland)
**Request:** `refine-logs/C02_A0_V3_REVIEW_REQUEST.md`
**Type:** read-only static review. Nothing was executed. See §6.

---

## 0. Hash and namespace verification

All seven declared sha256 values were verified. To avoid hand transcription, the declared
digests were extracted mechanically from the review request's table and, independently, from
`refine-logs/C02_A0_V3_RECORD.md:126-132`, and both extracted lists were fed to
`sha256sum -c`. **Both checks return `OK` on all seven files.** The verified digests:

| path | sha256 (verified against disk, request and record) | verdict |
|---|---|---|
| `configs/c02/c02_a0_v3.json` | `3c55214494372457fb8f2702f7ecf1c82c48b13c6b523d99e00272d2b0aa15ca` | MATCH |
| `src/utils/c02_density_views.py` | `531d4574a6c678132cb76510af0570067891a64ab5aa0a751f638b7f99ffd2fc` | MATCH |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `128461322f0b8f8d66478fc5bc296dba282850bd56f9b4665f06733af99149a0` | MATCH |
| `scripts/slurm/c02_density_extract.sbatch` | `001a089107e40f486a14eafd2daa52fb80033450b382dd41f204f46e8862de5a` | MATCH |
| `scripts/analysis/c02_a0_mint.py` | `da8a49187b821f0da15c7cec28421317225213f78f40f20d5c62c58c0ab71d33` | MATCH |
| `scripts/analysis/c02_a0_arena_v3.py` | `7d04a8ad8e644851fb8e25f77eee30ac12fbf7e33344dbc484bc5da240e21629` | MATCH |
| `scripts/slurm/c02_a0_cpu_v3.sbatch` | `9463d642756269a77929dd3ffeb8afeab02f81c2b5bd77a20d1566d245bae399` | MATCH |

**`refine-logs/C02_A0_V3_RECORD.md` (requested, recompute and report):**
`f54f08d9baa4c33863cacd452b365b857c844d5fbe7a078ec5c7124e4add8dbc`

**Superseded v1 and v2 executables — confirmed ABSENT:** `configs/c02/c02_a0_v1.json`,
`configs/c02/c02_a0_v2.json`, `scripts/analysis/c02_a0_arena.py`,
`scripts/analysis/c02_a0_arena_v2.py`, `scripts/slurm/c02_a0_cpu.sbatch`,
`scripts/slurm/c02_a0_cpu_v2.sbatch` all do not exist. `configs/c02/` contains **only**
`c02_a0_v3.json`.

**Namespace absence — confirmed:** `artifacts/c02_edq` does not exist; zero `*c02den*`
files exist anywhere outside `.git`, and zero in `data/CLIP_Embedding/{HateMM,MHC_zh}`.

**Run-time-asserted modules — all five recomputed and matching** the record §4 table:
`mechfix_ops.py` `635c1312…c83fc8d`, `mechnov_pairverify.py` `77b0defd…8598b7240d`,
`headspace_mint.py` `cefdf8dc…f0916612`, `generate_VideoMLLM_embedding_lora_HF.py`
`75bb8156…612399`, `headspace_fidelity.py` `72fd8e0a…08bf6598`. The four in-job sha pins
(`c02_a0_arena_v3.py:75-82,928-930`; `c02_a0_mint.py:55-56,92-95`;
`generate_c02_density_view_text_embedding_HF.py:63-64,139-142`) will therefore pass rather
than refuse. The extractor's `FROZEN_VIEWS_SHA256` (`:64`) equals the frozen view module's
actual digest — the re-pin the record §3 announces was applied.

**No hash mismatch. No Critical finding from §0.**

---

## 1. Repair verdicts for the two round-2 High findings

| # | prior finding | verdict | evidence |
|---|---|---|---|
| **H-A** | the retracted "`s_Q` upper-bounds / a failure is decisive" claim survived in two frozen files, and the record asserted it had been removed everywhere | **NOT REPAIRED** | The retracted sentence **survives verbatim in the v3 arena**, `scripts/analysis/c02_a0_arena_v3.py:17-19`: *"s_Q is deliberately OPTIMISTIC: it upper-bounds what any representation that contracts this orbit could buy, and it is not a deployable router. A failure is therefore decisive; a pass authorises Stage-1 design only."* I grepped `-i -E "upper[- ]bound\|decisive\|OPTIMISTIC"` across all seven frozen artifacts: the config **was** repaired (`configs/c02/c02_a0_v3.json:14`, with an explicit `RETRACTED` note), the other five files are clean, and the arena is not. `refine-logs/C02_A0_V3_RECORD.md:50-52` states *"the sentence is gone from **both** frozen files. `oracle_status` in `configs/c02/c02_a0_v3.json` **and the docstring of `scripts/analysis/c02_a0_arena_v3.py`** now carry the same corrected statement the decision emits"* — that is a positively false statement about a file whose bytes this preregistration pins. This is the **second consecutive round** in which the record claims this specific repair is complete and the artifact contradicts it. See **H1**. |
| **H-B** | `SHUFFLE` donated the donor's ABSOLUTE view keys, mirroring every true neighbour onto an unrelated-label row, so `FULL > SHUFFLE` was satisfied under the design's own null | **PARTIALLY REPAIRED** | *The named mechanism is genuinely and correctly fixed in code.* `build_arms` (`c02_a0_arena_v3.py:329-332`) now builds `shuffled = [keys["NAT"]] + [(nat_sh + (keys[v].astype("float64")[perm] - nat_sh[perm])).astype(keys["NAT"].dtype) for v in V.NON_NATIVE_VIEWS]`, i.e. exactly `z_i^v = NAT_i + (view_v(π(i)) − NAT_{π(i)})`. No component of `π(i)`'s absolute position enters; the first-order near-duplicate channel is closed. The formula is correct, deterministic, dtype-safe and allocation-only (see §3.B). **What is not repaired is the claim attached to it.** The frozen config (`:160`), the arena source comment (`:326-328`), the record (`:80-82`) and the value the job **emits** (`:828-831`) all assert that under H0 `FULL` and `SHUFFLE` are *"EXCHANGEABLE BY CONSTRUCTION"*. They are not; I give a concrete counterexample in **H3**. `FULL > SHUFFLE` is now a *real but insufficient* test: it is no longer guaranteed to be satisfied, but it remains satisfiable under nulls in which the orbit carries nothing beyond `NAT`. Separately, the in-job verification the record leans on is a tautology (**H4**), and the new degeneracy-class grouping introduces a reachable job-killing path (**H2**). |

---

## 2. Findings

Classification (the convention of rounds 1-2, retained): `Critical` = would void the result
or breach a hard registry constraint; `High` = would materially weaken a load-bearing
conjunct, a required control or a stated scientific claim, would kill the job after the GPU
spend, or is a false assertion inside the hash-frozen set about a load-bearing matter;
`Info` = correctness/robustness/documentation, non-blocking on its own.

| # | sev | file:line | finding |
|---|---|---|---|
| **H1** | **High** | `scripts/analysis/c02_a0_arena_v3.py:17-19`; `refine-logs/C02_A0_V3_RECORD.md:50-52` | **The H-A retraction was applied to the config and not to the arena, and the record asserts it was applied to both.** The arena's module docstring — the primary human-readable description of the instrument, and the file whose bytes carry the verdict — still declares that `s_Q` "upper-bounds what any representation that contracts this orbit could buy" and that "a failure is therefore decisive". That proposition was formally withdrawn in v2's `interpretation_boundary` and is explicitly marked `RETRACTED` four lines into the config this same script loads (`configs/c02/c02_a0_v3.json:14`). The frozen set therefore again contains two mutually contradictory statements of the same claim, and the arena will emit a corrected `interpretation_boundary` (`:1028-1037`) from a script whose own header contradicts it. Because a KILL is the expected outcome of a Stage-0 gate, the practical effect is unchanged from round 2: a KILL ships accompanied by a frozen artifact asserting a decisiveness the design has withdrawn. The aggravating element is the record: `:50-52` names `c02_a0_arena_v3.py` **by path** as one of the two files repaired. This is a repair verified in prose and not in the code, for the second round running, on the same sentence. Fix: replace `:17-19` with the corrected paragraph already present at `configs/c02/c02_a0_v3.json:14` and `c02_a0_arena_v3.py:1028-1037`, and correct the record. |
| **H2** | **High** | `scripts/analysis/c02_a0_arena_v3.py:234-258` (`shuffle_groups`) and `:261-287` (`derangement_within`), reached from `:762-764` and `:890-891` | **The new degeneracy-class grouping makes a pre-existing non-terminating branch REACHABLE, and it kills the A0 job with no artifact at all.** `derangement_within` repairs fixed points by the rotation `p[i], p[(i+1) % m] = p[(i+1) % m], p[i]` over a *pre-computed* `fx`. For a group of size **exactly 2** whose seeded draw is the identity, `fx = [0, 1]`; `i = 0` swaps to `[1, 0]` and `i = 1` swaps straight back to `[0, 1]`, so the 64-iteration loop exits unchanged and `:283` fires a **bare `AssertionError`**. That is outside the `Halt` path, so `main()`'s `except Halt` (`:966`) does not catch it: **no `C02_A0_OUT.json` and no `C02_A0_DECISION.json` are written**, the job dies after ~36 CPU mints (F113 prices a mint at 30.6-40.4 s, i.e. ~20-25 min elapsed) and after the entire GPU extraction budget has already been spent, and the record's execution boundary (`:165-166`, "two SLURM submissions, one each") leaves no re-run. Round 2 raised this branch as **I10 = Info** on the explicit ground that it was *unreachable*: with a single group per partition the sizes were ~460-595 and ~115-149. v3 changes exactly that. `shuffle_groups` now emits up to four groups per fold — `fit` × {non-degenerate, degenerate} and `ho` × {non-degenerate, degenerate} — and merges a class group only when it has **fewer than 2** members (`:247-254`), so a group of size exactly 2 is passed straight through. With 48 full-identity items in 744 HateMM train rows (which I re-derived independently, §3.A) and `ho` ≈ 149, the per-fold degenerate count is hypergeometric with mean ≈ 9.6, and `P(exactly 2)` ≈ 1.5-3 × 10⁻³ per fold; across the 5 folds, times the ½ probability that the seeded 2-permutation is the identity, the loss probability is order 0.4 %. It is a *deterministic* 0.4 %: `derangement_within` is seeded on `[SHUFFLE_SEED, fold, group]` with no seed axis, and the fold assignment is frozen, so the draw either is or is not the identity, and this review cannot tell which without opening a cache. Neither self-test case exercises it — case 4 (`:563-570`) uses groups of 70 and 50, case 5 (`:572-584`) tests only the size-1 merge. The same branch also still leaves the `|group| < 2` skip at `:272-273` feeding the global assert at `:286`. Two-line fix: for `m == 2` force `p = [1, 0]` (or roll instead of pairwise-swap), and convert both asserts to `halt(...)` so a failure still writes a fail-closed artifact. |
| **H3** | **High** | `configs/c02/c02_a0_v3.json:160`; `scripts/analysis/c02_a0_arena_v3.py:326-328` and the **emitted** `:828-831`; `refine-logs/C02_A0_V3_RECORD.md:80-82` | **"Under H0 FULL and SHUFFLE are EXCHANGEABLE BY CONSTRUCTION" is false, and it is the sentence that carries the H-B repair.** Exchangeability of the two arms would require the joint law of `(NAT_i, d_i)` to be invariant under permuting `d`. It is not, under any null the design actually states. Counterexample, entirely inside the design's own H0 ("the orbit carries nothing beyond `NAT`"): let the density displacement be radial, `d_i^v = ε·NAT_i`. Then the orbit is a deterministic function of `NAT` and adds *literally nothing*, and after the `_norm32` L2 normalisation every `FULL` view is bit-identical in direction to `NAT_i`, so `FULL ≡ NATIVE`. `SHUFFLE`, however, builds `NAT_i + ε·NAT_{π(i)}`, a genuinely different direction, which inflates `s_Q` in a label-random way and degrades. `FULL > SHUFFLE` is then satisfied **strictly**, at a null where the orbit carries nothing. The general statement is: `SHUFFLE` destroys the coupling between an item and its own displacement, and that coupling exists under *any* null in which the displacement is a function of the item's own text — which repetition of the item's own transcript necessarily is. This is a much weaker defect than v2's (it is second-order, not a guaranteed first-order mirroring, and it does not by itself let the seven-conjunct PASS fire — the counterexample gives `delta_acc = 0`), but the *claim* is not weakened: it is stated as a construction-level guarantee in a hash-frozen config, in the arena source, in the record, and in the `H0_behaviour` string the job writes into `C02_A0_OUT.json`. Given this project's retraction history, a frozen "by construction" that is refutable in three lines is the same genre of over-claim as H-A. Correct formulation: displacement donation removes the *systematic* degradation that made the conjunct vacuous; `FULL > SHUFFLE` is now a necessary condition that the orbit be item-specific, not a sufficient one that it be informative. See also **I23** for the class of artifact that survives both controls. |
| **H4** | **High** | `scripts/analysis/c02_a0_arena_v3.py:586-594` (self-test case 6); `configs/c02/c02_a0_v3.json:191`; `refine-logs/C02_A0_V3_RECORD.md:83-84` | **The in-job self-test that the record offers as proof of the H-B repair cannot fail, and never touches the repaired code.** Case 6 re-types the displacement formula locally — `sh = nat.astype("float64") + (vw.astype("float64")[pm] - nat.astype("float64")[pm])` — and then asserts `np.allclose(sh - nat, vw[pm] - nat[pm])`. That is an algebraic identity of the line immediately above it; it is true for *any* `nat`, `vw`, `pm` and can never fail. The second assert, `not np.allclose(sh, vw[pm])`, only says the random draw is non-degenerate. **`build_arms` is never called**, so if the frozen `build_arms` (`:329-332`) had donated the absolute position, or dropped the `[perm]`, or mis-ordered the subtraction, case 6 would still pass and still print `shuffle_donates_displacement`. The two sibling cases added in the same round do call the frozen functions (case 4 → `derangement_within`, case 5 → `shuffle_groups`), so the pattern was available. Meanwhile `configs/c02/c02_a0_v3.json:191` asserts the self-test *"proves, in-job: … that SHUFFLE donates a DISPLACEMENT and never the donor's absolute position"*, the record (`:83-84`) says the repair "is verified in-job by the new self-test case", and the string `shuffle_donates_displacement` is written into the result artifact (`:955`, `oracle_self_test_cases`). A downstream reader of either the config or the output JSON will conclude the deployed construction was verified when nothing about it was executed. I verified `build_arms` statically and it *is* correct — so this is a false verification claim, not a wrong number; but "the repair is verified in prose, not in code" is precisely what this review was instructed not to accept. One-line fix: call `build_arms` on a synthetic `keys` dict and assert on `arms["SHUFFLE"]`. |
| I1 | Info | `scripts/analysis/c02_a0_arena_v3.py:2` | The module docstring still names the superseded, now-deleted `c02_a0_arena.py`. The frozen artifact is `c02_a0_arena_v3.py` and its `run_id` is `C02-A0-v3`. Round-2 **I1**, unrepaired. |
| I2 | Info | `configs/c02/c02_a0_v3.json:11`; `refine-logs/C02_A0_RECORD.md:185` | `authority.record` still points at `refine-logs/C02_A0_RECORD.md`, the **v1** record. Round-2 **I2**, unrepaired — and it now costs more than it did: the v1 record is the declared "design of record", and at `:185` it defines `SHUFFLE` as *"`NAT` + item `pi(i)`'s non-native views"*, i.e. the **absolute-position rule this round retracted**. A reader following the config's own authority pointer lands on the retracted definition with nothing at that line to warn them. Every other frozen file's `Record:` pointer was updated to `C02_A0_V3_RECORD.md`; the config's was not. |
| I3 | Info | `configs/c02/c02_a0_v3.json:191` | The `oracle.self_test` field names `scripts/analysis/c02_a0_arena_v2.py::oracle_self_test`. That file was **deleted by this freeze**; the pointer is dangling. |
| I4 | Info | `configs/c02/c02_a0_v3.json:197` | `GATE_FID` says the `B_fid < 0.050` bar is "enforced in `scripts/slurm/c02_a0_cpu_v2.sbatch`". That wrapper was **deleted by this freeze**; the bar is actually enforced at `scripts/slurm/c02_a0_cpu_v3.sbatch:71-82`, which I verified reads `B_fid_abs_3seedmean` and asserts `< 0.050` for both datasets. |
| I5 | Info | `configs/c02/c02_a0_v3.json:229` | `interpretation_boundary` ends "The v1 draft overstated this as an upper bound; **corrected in v2**." That is false, and the v3 record's own §2 H-A says so: v2's config `:14` and v2's arena docstring both carried the retracted wording. The sentence should read "corrected in v3". |
| I6 | Info | `configs/c02/c02_a0_v3.json:49,57,65` | Three of the four `max_chars` values remain wrong. I re-measured independently with `gawk length()` under `en_US.UTF-8` on the raw gt substring: HateMM train **80732** (config 80731), HateMM val **12276** (config 12275), MHC-ZH train **710** (config 708), MHC-ZH val **343** (correct). Round-2 **I3**, unrepaired. Nothing consumes `max_chars`; every count the `L_MAX` guard and `VIEW_SUPPORT` depend on is exact (§3.A). Separately, round 2's supporting claim that no gt line carries a backslash escape is **wrong for MHC-ZH**: 244 of 579 train rows contain `<em class=\"keyword\">…</em>`, so for that dataset the raw substring is not the decoded string and `max_chars` cannot be settled by raw counting at all. Neither fact is load-bearing; both belong in a table that exists to repair a measured statement contradicted by the data. |
| I7 | Info | `src/utils/c02_density_views.py:46-50` | The `LENGTH_GUARD` docstring still asserts that "the original C02 plan already required that such items 'are excluded from this view and counted' (`C02_EXPERIMENT_PLAN.md §3.1`)". I read the plan: the clause (`refine-logs/C02_EXPERIMENT_PLAN.md:80-81`) is a *tokenizer-limit truncation* rule, and the deployed processor call passes no `truncation`/`max_length`, so `L_MAX = 12000` characters is a freely chosen new constant. The correction was applied to `configs/c02/c02_a0_v3.json:36` ("in that spirit but is not the same criterion") and not to the frozen module. Round-2 **I5**, unrepaired. |
| I8 | Info | `scripts/analysis/c02_a0_arena_v3.py:679-691` | GATE-EXT gates only the **median** row cosine at 0.99. `min_cos`, `mean_cos` and `max_abs_diff` are computed and emitted but compared to nothing, so up to half the rows may deviate arbitrarily and the gate still passes. 0.99 is also loose for a same-model, same-prompt, same-merged-adapter re-extraction. Round-2 **I7**, unrepaired. |
| I9 | Info | `scripts/analysis/c02_a0_arena_v3.py:882-895,918,965` | The SECONDARY raw arena runs **inside** `run_dataset`, after the primary read is complete, and `res` is only handed to `out["datasets"][ds]` on return (`:918` → `:965`). Any `Halt` raised there therefore discards a finished primary measurement and aborts the second dataset as well, inverting the primary/secondary hierarchy the design declares. Round-2 **I8**, unrepaired. |
| I10 | Info | `scripts/analysis/c02_a0_arena_v3.py:962-975,1038-1042` | `main()` still returns 0 after a `Halt`: the exception is caught, recorded, written and the process exits normally, so SLURM reports `COMPLETED` for a fail-closed run. Round-2 **I11**, unrepaired. |
| I11 | Info | `scripts/analysis/c02_a0_arena_v3.py:832-835` | `res["gates"]["PARITY_NAT"]["predictions_and_sorted_similarities"]` is a **hardcoded string**, `"BIT-EQUAL on all 15 (seed x fold) cells"`. The halts in `parity_native` make it true wherever the line is reached, but the artifact stores an assertion rather than a measurement. `tie_rows_total` beside it *is* counted. Round-2 **I12**, unrepaired. |
| I12 | Info | `scripts/analysis/c02_a0_arena_v3.py:915-917` vs `src/utils/generate_c02_density_view_text_embedding_HF.py:279-280` | The extractor records the sha256 of every view file it writes into the manifest, and the arena recomputes the sha256 of every view file it loads — but the two are **never compared**. The arena reads the same manifest at `:694-701` and checks only the id set. One `assert` would close it. Round-2 **I13**, unrepaired. |
| I13 | Info | `scripts/analysis/c02_a0_arena_v3.py:63,130` vs `src/utils/generate_c02_density_view_text_embedding_HF.py:67` | Token-list asymmetry: the extractor's `FORBIDDEN_TOKENS` carries four entries including `"test_"` and lowercases the path; the arena's `torch.load` wrapper and `guard_path` carry only three and omit `"test_"`; the extractor's own `torch.load` wrapper checks only two and does not lowercase. Defence-in-depth only — I traced every path all three scripts construct and found no reachable test artefact (§3.A). Round-2 **I15**, unrepaired. |
| I14 | Info | `configs/c02/c02_a0_v3.json:99`; `scripts/slurm/c02_density_extract.sbatch` (whole) | `budget_gpu_hours_cap: 4.0` is declared but nothing measures or enforces it in-job, and an overrun silently **voids** the result under amendment condition (f). My own arithmetic makes the cap comfortable (§4), so the residual risk is procedural. About 12 % of the spend (185 `dev_seen` items × up to 6 text forwards) buys view files that **no A0 code path reads**. Round-2 **I16**, unrepaired. |
| I15 | Info | `refine-logs/C02_A0_V3_RECORD.md:174`; `TARGET_STATE.json:84-96` | Amendment condition (e) (`one_candidate_at_a_time`, `parallel_gpu_or_teacher_pilots_forbidden`, "a bounded Stage-0 extraction counts as a GPU pilot for both rules") is deferred entirely to a manual `squeue` check at submission time, with no automated interlock. The registry's `serial_execution.current_design_boundary` presently reads `C04_IMPL_V5_CPU_PREFLIGHT_ENGINEERING_HALT_JOB_13805_V6_REPAIR_REQUIRED`, i.e. a different candidate with an open job lineage. I did not run `squeue`/`sacct` and take no view on the live queue. Round-2 **I17**, unrepaired. |
| I16 | Info | `configs/c02/c02_a0_v3.json:226` vs `:230` | `decision_rule.PASS` still enumerates `net_fix_rate >= 0.030` inline as one of five conjuncts, with the disclosure that it cannot bind living in `net_fix_clause` four lines later. The arena's own field name (`net_fix_rate_implied_by_acc_bar`, `:993,1017`) is the better pattern. Round-2 **I18**, unrepaired. |
| I17 | Info | `scripts/analysis/c02_a0_arena_v3.py:499-503`; `configs/c02/c02_a0_v3.json:194` | The general rationale printed for the PARITY-NAT tie exemption ("their vote is invariant to tie order") remains false in general: two tied neighbours with different labels at adjacent ranks change the vote by `2·s·(w_r − w_{r+1})/Σw ≈ 0.0095`. Self-test case 2 verifies only the all-zero query, where the vote is identically 0. The exemption stays **operationally safe** — `parity_native` bit-checks predictions *and* the sorted top-20 similarity vector on **every** row, tied or not; only neighbour IDs are exempted, and the count is reported. Round-2 **I4**, unrepaired. |
| I18 | Info | `scripts/analysis/c02_a0_arena_v3.py:182-194`; `configs/c02/c02_a0_v3.json:164` | The `k = topk` per-view-pair exactness argument is **correct as I re-derived it independently**, but only when the topk-th largest per-item maximum `τ` is attained by at most `topk` items; with exact float32 ties at `τ`, a pair's own top-20 need not contain every row at or above `τ` and a boundary item can be dropped. Neither the code nor the stated argument carries the caveat, and self-test case 3 uses tie-free random matrices. The one place ties at `τ` are certainly reachable — the all-zero structural-null query in the **raw** arena — is harmless (every similarity is 0, so the vote is 0 regardless of which 20 ids come back), and `parity_native` is never called on the raw arena. Round-2 **I6**, unrepaired. |
| I19 | Info | `scripts/analysis/c02_a0_arena_v3.py:508-513`; `scripts/slurm/c02_a0_cpu_v3.sbatch:52-84` | `oracle_self_test`'s docstring justifies itself as running "before any real data is opened so a numerical-contract break costs seconds, not a queue slot". True inside the arena process (`main:932` precedes the config load, the output-path checks and `run_dataset`), but the wrapper invokes the arena **last**, after 36 CPU mints and GATE-FID — ~20-25 minutes in at F113's measured mint cost. Round-2 **I9**, unrepaired. |
| I20 | Info | `refine-logs/C02_A0_V3_RECORD.md:134-145` | The "superseded, preserved for the audit trail" tables again list only the config, arena, wrapper and record for each of v1 and v2. Four further files in the frozen set (`c02_density_views.py`, `generate_c02_density_view_text_embedding_HF.py`, `c02_a0_mint.py`, `c02_density_extract.sbatch`) also changed between rounds and are not in either table. §3 now *describes* the change in prose ("only in their record pointer … and the re-pinned view-module hash"), which is an improvement on round 2's silence, and reading the current bytes I find that description consistent — but all four are untracked in git and were overwritten in place, so the byte-level delta still cannot be bounded from the record alone. Round-2 **I14**, partially addressed. |
| I21 | Info | `scripts/analysis/c02_a0_arena_v3.py:702-705` | `degen_mask` covers only **full-identity text** orbits (`len(identity_views) == 5`), so two other classes of null-displacement item fall into the *non*-degenerate donor group and therefore receive a real displacement under `SHUFFLE` that they can never have under `FULL`: (a) **video-decode-failure zero-guard rows**, whose six view vectors are bit-identical zeros by the extractor's guard (`generate_c02_density_view_text_embedding_HF.py:234-240`) — `C01_ZERO_CONTRACT_PROBE.md:7,13` puts this at 1 HateMM row (`hate_video_95`, row 355); and (b) **`EMPTY_WINDOW` partial-identity rows** (`len(T) < 4`), where some `RW_k` equal `NAT` — MHC-ZH's shortest train text is 3 characters, so this class is non-empty. Both are tiny and the effect is second-order, but they are exactly the asymmetry the degeneracy split was introduced to remove, and neither the record's "48 of 744 HateMM train items, 0 of 579 MHC-ZH" (`:92`) nor the config's `arms.SHUFFLE` text (`:151`) mentions them. |
| I22 | Info | `scripts/analysis/c02_a0_arena_v3.py:822-831` vs `:890-891` | `shuffle_group_merges` is accumulated only in the head arena (`:763`); the raw arena discards its merge count (`shuffle_groups(fit, ho, degen_mask)[0]`). The emitted field is named `n_group_merges_over_all_seeds_and_folds`, which over-states its coverage. (The two arenas do compute identical groups — see §3.B — so the number would be the same; the name is still wrong.) |
| I23 | Info | `scripts/analysis/c02_a0_arena_v3.py:329-342`; `configs/c02/c02_a0_v3.json:159` | **A class of artifact survives both controls, and it is a different class from the one round 2 identified.** v3's SHUFFLE *does* now cover the shared-direction ("verbosity axis") artifact round-2 **I19** raised — if displacements share a common direction, SHUFFLE approaches FULL and the conjunct fails, exactly as `control_roles` claims. What neither control covers is a displacement that is a **deterministic, item-specific function of the item's own native key**, `d_i = f(NAT_i)`: NOISE destroys it (wrong direction, norm preserved), SHUFFLE destroys it (right kind of direction, wrong item), and FULL keeps it. A PASS driven by such an `f` would be a statement about a fixed transform of `NAT` — reachable with zero extra extraction and no density signal at all — not about the density orbit. The `retrieval_length_spearman` and `krr_length_probe` diagnostics are computed per arm (`:811-812,818-819`) but neither is gated. This bounds what a PASS can mean; it does not touch a KILL. |

**Counts: 0 Critical, 4 High, 23 Info.**

---

## 3. What I verified independently and found sound

Recorded so the REVISE is not read as a rejection of the design. Everything below I checked
myself, from the artifacts, without relying on either prior review.

### A. Registry and contract compliance

- **No reachable test path anywhere, including through imported modules.** *Extractor:*
  `SPLIT_TO_OUTNAME` admits only `train`/`val` (`:66,147-150`), `assert_no_test_token` is
  applied to every gt path, every one of the 12 output paths and the manifest
  (`:156-169,213,273,304`), and a global `torch.load` wrapper is installed at import
  (`:69-80`). *Mint:* `load_view_text` hard-asserts `split == "train"` (`:64`);
  `headspace_mint` installs its own `torch.load` guard at import (`headspace_mint.py:109-116`);
  `run_rac.load_feats_from_CLIP` is replaced wholesale (`:146-149`) — and I confirmed at
  source that `src/run_rac.py:1128-1135` reaches the loader only through that module-global
  name, on the non-`FB` three-tuple branch that both `HateMM` and `MHC_zh` take, so the
  patch cannot be bypassed for either dataset; the fold head's dev and test dataloaders are
  stratified slices of the fitting pool (`c02_a0_mint.py:131-134`). *Arena:* I enumerated
  every file handle it opens — `train_<model>.pt`, the six `train_<model>-c02den-<VIEW>.pt`,
  `mint_<ds>_s<seed>_f{0..4}.npz`, the extract manifest, the **train** P3 jsonl and the
  config — at `:292,606,619,694,734,935`; every one of the data paths passes through
  `guard_path`. `mechfix_ops.py` and `mechnov_pairverify.py` have no import-time side
  effects and no test read. **I found no path, direct or transitive, that can reach a
  `test_seen` cache, a `test.jsonl` or a test label.** (`data/MLLM_scores/HateMM/` does
  contain `test_seen_segscoreK4_qwen.jsonl`; the arena constructs only the `train_` name.)
- **Split scope, bar, hard constraints.** Extraction is `--splits train,val` → `train` and
  `dev_seen` only, both datasets (`c02_density_extract.sbatch:49,60`). The `+0.050`/`+0.050`
  two-dataset bar is verbatim at `c02_a0_arena_v3.py:88-89` and enforced at `:1016`, and
  matches `configs/c02/c02_a0_v3.json:226` and `TARGET_STATE.json` amendment condition (c).
  No OCR; no cross-dataset mixing or cross-dataset training (each dataset's own cache, own
  adapter, own folds); no external API (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, local
  merged LoRA); single-dataset train split only; parent-video binary label only (`labels`
  from the gt jsonl; P3 window scores enter only the two control arms and never as gold);
  no size scaling; no cross-seed ensembling (seeds are averaged for the read at `:842-853`,
  never stacked into a predictor). The C14 boundary (`TARGET_STATE.json` candidate registry,
  "frozen mechanism/upper-bound diagnostic … ensemble predictions forbidden as the final
  method") is confronted explicitly at `configs/c02/c02_a0_v3.json:231`, and the A0 occupies
  precisely the permitted diagnostic role.
- **Amendment condition (a).** The design is hash-frozen and unsubmitted; the extraction
  order in the record §5 places this review before `sbatch`. Conditions (b), (c), (d), (g)
  are met as above; (e) and (f) are procedural — see **I14**, **I15**.
- **F113 compliance.** The primary arena is the fold-head / deployed-head path
  (`:726-820`); the raw fused arena is labelled `SECONDARY … may corroborate a KILL, may
  never promote` (`:897-899`) and **does not enter the decision at all** — `main()` reads
  only `summary_3seed` and `bootstrap_FULL_vs_NATIVE` (`:988-1005`). GATE-FID uses the
  frozen, unmodified `headspace_fidelity.py`, whose `--dataset/--mintdir/--out/--seeds`
  signature and `B_fid_abs_3seedmean` output key I checked at source, against the wrapper's
  `< 0.050` assertion (`c02_a0_cpu_v3.sbatch:71-82`). F113 §2.1's companion rule ("a Δ
  smaller than `B_fid` cannot carry a verdict") is not restated but is automatically
  satisfied: a PASS needs Δ ≥ 0.050 while the gate forces `B_fid < 0.050`.
- **ARENA-2 will not spuriously halt.** F113 §4.2/§5.2 measured pooled head-space deployed
  accuracy 0.8867 (HateMM) and 0.8923 (ZH), against majority rates ≈0.5995 / ≈0.60; both sit
  well inside `[maj + 0.02, 0.98]`.
- **Every required input exists on disk.** Both banked `train_*.pt` caches; both P3
  `train_segscoreK4_qwen.jsonl` (744 and 579 rows, `scores` length 4, so
  `load_p3_windows`'s `len(sc) == V.K_WINDOWS` filter will admit every row and
  `n_p3_missing` will be 0); all ten `vsw_ckpt/<ds>/f{0..4}.npz`. `artifacts/c02_edq` is
  absent, so both no-clobber paths are clean.
- **The H2 gt counts are exact and I re-derived them with `gawk` alone** (no Python, no
  cache, no test): whitespace-only rows **39 / 9 / 0 / 0**; `len(T) > 12000` rows
  **9 / 1 / 0 / 0**; disjoint, so full identity **48 / 10 / 0 / 0**; `n` = 744 / 107 / 579 /
  78; view support **0.9355 / 0.9065 / 1.0 / 1.0**, both train splits clearing the 0.60
  gate by a wide margin. The only measured value I disagree with is the decorative
  `max_chars` (**I6**).
- **SLURM hygiene.** Both wrappers: no `--time`, no `--dependency`, no `--array`, no
  `--singleton`, no `--requeue`, no release/force path, `conda activate HateVideo`, 8 CPU
  each (so the submit-time 16-CPU aggregate cap cannot wedge), one submission each, strictly
  serial. The A0 wrapper is CPU-only with `CUDA_VISIBLE_DEVICES=""` and exports the DET-1
  thread block before any python starts, which `HM.det1_assert("8")` re-checks in-process.
  `set -euo pipefail` on both.

### B. Numerics and implementation that hold

- **The displacement-donation formula is correct, deterministic, dtype-safe and free of
  in-place mutation of shared arrays** (the request's first question under B). `:329-332`
  promotes to float64, indexes with the fold-local `perm`, subtracts the donor's own native
  key and casts back to `keys["NAT"].dtype` (float32 in both arenas — the mint stores
  `astype("float32")` and `raw_keys` is built `.astype("float32")`). `astype` and fancy
  indexing both copy, so no input array is touched; `keys["NAT"]` is shared into
  `NATIVE`/`FULL`/`SHUFFLE`/`NOISE` as element 0, but the only consumers are `_norm32`
  (which **always** copies, `:162-173`) and `X[fit]` / `X[ho]` fancy indexes, so
  `faiss.normalize_L2`'s in-place behaviour can never reach a stored key. `M.deployed_vote`
  is the one caller of the aliasing `mechfix_ops._norm32`, and it is only ever handed
  `arms["NATIVE"][0][fit]` / `[ho]`, already fresh temporaries. Determinism: `perm` comes
  from `np.random.default_rng([SHUFFLE_SEED, fold, group_index])` with frozen constants and
  no seed axis, and `rngn` is re-created inside every `build_arms` call so the NOISE tangent
  set is identical across folds, seeds and arenas.
- **`shuffle_groups` is correct on the leakage question, and a donor can never straddle the
  boundary.** The outer loop runs over `fit` and `ho` separately and every emitted group is
  a subset of one of them (`:244-257`); the singleton merge is written to consume only the
  *other class of the same partition* (`:248,252`); self-test case 5 asserts subset
  containment for every group. `derangement_within` permutes strictly inside each supplied
  group (`:270-284`) and asserts globally that no covered index is a fixed point (`:286`).
  Together with the per-fold `np.intersect1d(ho, fit).size == 0` assert (`:759-760`), **full
  self-orbit exclusion holds in every arm including SHUFFLE**: a query's own orbit is never
  in the bank, and a query's donor is another query.
- **The singleton-merge rule leaves no index fixed point.** Every group that reaches
  `derangement_within` with `size ≥ 2` is deranged with zero fixed points, asserted twice.
  Degenerate items do keep their *own* (zero) displacement, but that is the intent of the
  matching, not a defect. The residual hazards are `size == 2` (**H2**) and `size < 2`
  (reachable only if a whole partition has ≤ 1 member, i.e. never here).
- **The degeneracy-matched grouping is identical in the head arena and the raw arena** (the
  request's third question under B). Both call `shuffle_groups(fit, ho, degen_mask)` with
  the same manifest-derived `degen_mask` (`:704`); the head arena's `fit` is the mint's
  `fit_idx`, which is sklearn's `splits[f][0]` under `StratifiedKFold(5, shuffle=True,
  random_state=0)` asserted item-for-item against the banked `vsw_ckpt/<ds>/f{0..4}.npz`
  inside every mint (`c02_a0_mint.py:115-126`), and the raw arena's is
  `np.flatnonzero(fold_of_ref != f)`, its exact ascending complement. Same groups, same
  `gi` ordering, same seeds, same `perm`. **`degen_mask` and `shuffle_group_merges` are both
  defined at `:704-705`, before every use** (`:762-763`, `:827`, `:890-891`) — the request's
  fourth question under B.
- **Subsequence contract and its pre-forward proof.** `build_views` only ever forms
  `T[:c_k] + " " + T[c_{k-1}:c_k] + T[c_k:]`, `T + " " + T`, or `T` itself, so `T` is an
  ordered subsequence by construction; `assert_subsequence` (`c02_density_views.py:135-144`)
  is the correct greedy-iterator test, is called per item per view at
  `generate_c02_density_view_text_embedding_HF.py:228-229` — **before**
  `BASE.load_video_frames` at `:231` and therefore before any forward — and is itself
  negatively tested (`deletion_rejected`). Degenerate identity is bit-exact by object reuse
  (`:245-254`), not by tolerance.
- **Prompt parity.** `build_text_prompt` (`:124-129`) reproduces the deployed assembly under
  the deployed defaults; I confirmed structurally that no gt line in any of the four
  train/val files carries a `title` key, so both the banked and the new extraction take the
  `(none)` branch identically.
- **`PARITY-NAT` is binding and the tie exemption is not a loophole.** Predictions and the
  sorted top-20 similarity vector are bit-checked on **every** row of all 15 (seed × fold)
  cells per dataset (`:495-498`); only neighbour IDs are exempted on tie rows and the count
  is reported. For a singleton orbit `orbit_vote` degenerates to the literal deployed
  `k = 20` faiss call over the same bank, both `_norm32` implementations produce
  byte-identical float32 C-contiguous normalised copies from the same float32 input, and the
  candidate id set is the *same* faiss return in both paths, so only intra-tie order can
  differ and the similarity vectors are equal regardless. See **I17** for the stated
  rationale, which is still wrong, and **I18** for the tie caveat.
- **Bootstrap estimand is the bar's estimand, and is paired.** `paired_bootstrap:383-417`
  computes the point estimate and every replicate as `mean_s[metric(arm) − metric(floor)]`
  over the three seeds on the resampled item set, algebraically identical to
  `delta_acc_3seed_mean` / `delta_mf1_3seed_mean` at `:848-849`. One index vector is applied
  to both arms and to `y`, so pairing is at item level. `M.macro_f1` is pure numpy, so
  B = 10000 is seconds.
- **`net_fix_rate` is disclosed rather than dressed up.** The per-arm field is
  `net_fix_rate_IDENTICAL_TO_delta_acc` (`:804`), the summary carries the identity note
  (`:854-856`), the decision key is `net_fix_rate_implied_by_acc_bar` (`:993,1017`) and the
  comment at `:1011-1015` says it can never bind. I re-derived the algebra: per seed
  `(fixed − broken)/n ≡ acc_arm − acc_native` by construction, so with
  `BAR_ACC 0.050 > BAR_NETFIX_RATE 0.030` the clause is discharged by the accuracy bar.
  `precision_on_changed` is added and explicitly not gated.
- **Zero contract.** Criteria 2 and 3 are computed and asserted across the banked native and
  all six views (`:641-659`); criteria 1 and 4 are honestly labelled
  `DOCUMENTARY_CITATION_NOT_COMPUTED` rather than emitted as hardcoded `true`. Structural
  nulls are retained identically in every arm and a sensitivity read excluding them is
  reported (`:868-880`).
- **Guards fire where claimed and cannot be stripped.** `c02_a0_arena_v3.py:55-56`,
  `c02_a0_mint.py:46-47` and `generate_c02_density_view_text_embedding_HF.py:53-54` are `if`
  statements, not asserts, so `python -O` hits them; all three precede any assert-guarded
  work. The arena's `torch.load` guard is installed at `:58-69`.
- **Early no-clobber covers every file the extraction job writes** — all 12 view paths plus
  the manifest are validated absent before `transformers`/`peft` are imported (`:152-170`),
  with late re-checks at `:274-275,306-307`. The A0 job re-checks both output paths before
  any work (`:941-942`) and writes atomically.
- **Determinism of the environment.** `scipy` in the `HateVideo` env is **1.17.1**, so
  `spearmanr(...).statistic` exists and the tie-corrected Spearman will not raise.
- **Wiring end to end.** The wrapper writes `mint_<ds>_s<seed>_f{full,0..4}.npz`;
  `headspace_fidelity.py:66` reads `mint_{ds}_s{s}_ffull.npz`; the arena reads
  `mint_{ds}_s{seed}_f{0..4}.npz`; the extractor writes
  `<split>_<base_tag>-c02den-<VIEW>.pt` into `data/CLIP_Embedding/<DS>/` and
  `manifest_<DS>.json` into `artifacts/c02_edq/v1/extract/C02-DEN-v1/`, which is exactly
  what `c02_a0_mint.py:61` and `c02_a0_arena_v3.py:619-620,694-695` construct. Every
  manifest key the arena consumes (`splits.train.per_item[*].{id,len_native,
  identity_views,degenerate}`, `n_degenerate_items`, `zero_guard_videos`) is written by the
  extractor. `run_id`, both schema versions and the output namespace are consistently `v3`
  across config, arena and wrapper. **I found no name mismatch that would kill the job after
  the 36 mints.**

### C. Config ↔ code constant audit

Every constant in `configs/c02/c02_a0_v3.json` was checked against the code and **all
matched**, except the claims raised as findings above (`self_test` path — I3; the GATE-FID
wrapper path — I4; `interpretation_boundary`'s "corrected in v2" — I5; `max_chars` — I6;
the exchangeability claim at `:160` — H3; the self-test claim at `:191` — H4). Spot list:
`k_windows 4` ↔ `K_WINDOWS`; `window_cut_rule (k*len)//4` ↔ `window_cuts`; `separator " "` ↔
`SEP`; `l_max_chars 12000` ↔ `L_MAX`; `identity_causes` ↔ the `DEGEN_*` branches and the
`EMPTY_WINDOW` path; `views.names` ↔ `VIEW_NAMES`; `topk 20` ↔ `TOPK`; `weights [20..1]` ↔
`M._rank_weights(20)`; `vote v ≥ 0` ↔ `(votes >= 0)`; `tie_rule lower bank index` ↔
`lexsort((idx, -best))`; `seeds [0,1,2]` ↔ `SEEDS`; all nine `arms` ↔ `build_arms`
(memberships exact, including the new `SHUFFLE` formula at `:151`); `argmin/argmax ties →
lowest index` ↔ `argmin_window`/`argmax_window`; `p3_missing_rule` ↔ `make_choices`;
`bootstrap B=10000 seed 20260730` ↔ `BOOTSTRAP_B`/`BOOTSTRAP_SEED`; `alpha 0.05` ↔ `ALPHA`;
`holm_family` ↔ the four `pv` keys; `ARENA2 [maj+0.02, 0.98]` ↔
`ARENA2_MARGIN`/`ARENA2_CEILING`; `GATE_EXT median cos ≥ 0.99` ↔
`EXT_PARITY_MEDIAN_COS_MIN`; `VIEW_SUPPORT ≥ 0.60` ↔ `VIEW_SUPPORT_MIN`; `TEST_PATH` tokens
↔ `guard_path`; `krr gamma=1/d, ridge=1, target log1p(len)` ↔ `krr_length_probe`;
`orbit_radius` median over items and non-native views, strict OOF ↔ the `rad_oof` stack
median; `membership` ↔ `mean_top20_overlap_with_native`; `det1_threads 8` ↔
`HM.det1_assert("8")` plus the wrapper exports; `extraction 8/1/64G`, `a0 8/0/32G`,
`time null` ↔ both sbatch headers; `extract_namespace`/`a0_namespace`/`result_file`/
`decision_file`/`atomic_json`/`no_clobber` ↔ the wrappers and `main()`; `supersedes.v1.*`
and `supersedes.v2.*` ↔ the record §4 values.

---

## 4. The adversarial questions, answered directly

**Is `FULL > SHUFFLE` now a real test at H0, or still satisfiable when the orbit carries
nothing?** It is **substantially, but not entirely, repaired**. The first-order mechanism —
bank row `j` becoming a near-duplicate of bank row `π(j)` at cosine ≈ 1, mirroring every
true neighbour onto an unrelated label — is gone, and I verified that in `build_arms`, not
in prose. What remains is that `SHUFFLE` destroys the coupling between an item and its own
displacement, and that coupling exists under nulls in which the orbit carries nothing: the
radial case `d_i = ε·NAT_i` gives `FULL ≡ NATIVE` (the views normalise to the native
direction) while `SHUFFLE` perturbs, so `FULL > SHUFFLE` holds strictly at a null where the
orbit is literally a function of `NAT` (**H3**). So the conjunct is a *necessary* condition
that the orbit be item-specific — a real and useful test — and not the construction-level
guarantee the frozen artifacts claim.

**Can the treatment still pass on an artifact that neither `SHUFFLE` nor `NOISE` covers?**
**Yes.** v3 closes round 2's shared-direction gap: if displacements share a common
direction, `SHUFFLE` retains it, approaches `FULL`, and the conjunct fails — that is a
genuine improvement and the config states it correctly (`:159`). The gap that remains is a
displacement that is a deterministic, item-specific function of the item's own native key.
`NOISE` randomises direction; `SHUFFLE` randomises the item; neither preserves
`d = f(own NAT)`, so `FULL` can beat both on a transform that requires no density
extraction at all. See **I23**. The two diagnostics that would expose it
(`retrieval_length_spearman`, `krr_length_probe`) are computed per arm and not gated.

**Is the `k = 20` per-view-pair exactness argument correct?** Correct as I re-derived it: if
`τ` is the 20th largest per-item maximum, every `(a, b, j)` row with `s ≥ τ` forces
`m_j ≥ τ`, hence `j` is in the top-20 item set, hence each view pair contributes at most 20
such rows and its own top-20 list already contains all of them. It carries one unstated
caveat — exact float32 ties at `τ` (**I18**) — which is unreachable on real head keys and
harmless in the one place it is reachable.

**Would a KILL from this design be sound?** **Yes**, as a gate verdict under the registry's
frozen Stage-0 rule — which is what the emitted `interpretation_boundary` (`:1028-1037`)
correctly says. None of the four High findings contaminates it: **H3** and **I23** only make
a *positive* easier; **H4** is a verification-hygiene defect on code I verified statically;
**H2** does not bias anything, it destroys the run outright; **H1** is a wording defect. The
problem with a KILL under this freeze is exactly round 2's: it would ship alongside a frozen
artifact (`c02_a0_arena_v3.py:17-19`) asserting a decisiveness the design has withdrawn, and
alongside a record that says that sentence is gone.

**Would a PASS from this design be sound?** **Not as stated, though it is materially better
than v2.** Of the seven per-dataset conjuncts, one (`net_fix_rate`) is an acknowledged
algebraic identity, correctly disclosed; two (`beats_shuffle_*`) are now real but necessary-
not-sufficient rather than "exchangeable by construction" (**H3**); the extraction-parity
gate constrains only the median row (**I8**); and no gate blocks the `d = f(NAT)` artifact
(**I23**). `NOISE`, the paired bootstrap CIs, the Holm family, `PARITY-NAT` and the
`+0.050`/`+0.050` bar are the parts of a PASS that would carry weight — and, since a PASS
"authorises Stage-1 design plus a fresh review only", the residual risk is bounded by the
next review rather than by this one.

**Budget plausibility (≤ 4.0 GPU-hours).** I measured the extraction load from the gt files
directly. Excluding the `LENGTH_GUARD` items (which take one forward each), HateMM train
carries 735 items totalling 837 308 characters, median 675, p90 2 800, max 11 741; HateMM
val 106 items / 94 711 chars; MHC-ZH train 579 items / 78 186 chars (median 106, max 710);
MHC-ZH val is smaller still. Each non-degenerate item costs `NAT + RFULL + 4×RW ≈ 8·L`
characters of text across six forwards, i.e. ≈ 6.7 M characters for HateMM train and ≈ 0.6 M
for MHC-ZH train — on the order of 2 M text tokens in total, plus ~1 500 video decodes and
~8 000 forwards' worth of visual tokens at `num_frames=8`, `max_pixels=151200`. Against the
amendment's own `sacct` table (comparable 8 CPU / 1 GPU / 48-64 G extraction jobs at 00:24
to 02:00, the largest being 13468 at 02:00:08 for a strictly heavier workload),
**~1-2 GPU-hours is a credible projection and the 4.0 cap carries ≈2× headroom.** The
residual risk is procedural (**I14**), not arithmetic.

---

## 5. Verdict

REVISE (0C/4H/23I)

`GO` is not available.

**H1** is the same finding as round 2's H-A, on the same sentence, in the same class of file,
with the record again asserting the repair is complete. The config was fixed and the arena
was not. **H2** is a defect the v3 repair itself introduced: the degeneracy-class grouping
makes a size-2 donor group possible, and a size-2 group with an identity draw sends
`derangement_within` into a swap loop that terminates in a bare `AssertionError` outside the
`Halt` path — no result JSON, no decision JSON, after the GPU extraction is spent and under a
one-submission-only rule. **H3** and **H4** are the two claims the H-B repair is presented
on: "exchangeable by construction", which I refute with a three-line counterexample inside
the design's own H0, and "verified in-job by the self-test", where the named case is an
algebraic tautology on a re-typed expression that never calls `build_arms`.

None of the four requires re-architecting. H1 is one paragraph in one file plus a record
line. H2 is two lines (force `p = [1, 0]` for `m == 2`; convert both derangement asserts to
`halt(...)`). H3 is a sentence in four places, replaced by the accurate statement that
`FULL > SHUFFLE` is necessary but not sufficient. H4 is one self-test case rewritten to call
`build_arms`. The 23 Info findings are individually non-blocking; **I2**, **I3**, **I4**,
**I5**, **I6** and **I7** are worth folding in at the same time because they are all
instances of the same pattern — a repair narrative, or a pointer, running ahead of the
artifacts — and three of them now point at files this freeze deleted.

The rest of the design I verified independently and found sound: the extraction contract and
its split scope, the subsequence proof and its placement before any forward, the absence of
any reachable test path including through `run_rac`, the fold-head arena and its parity
against the banked `vsw_ckpt`, full self-orbit exclusion in every arm including the new
`SHUFFLE`, `PARITY-NAT`, `ARENA-2`'s calibration against F113's measured 0.8867 / 0.8923, the
zero contract, GATE-FID's stop rule and its wiring, the bootstrap estimand, the Holm family,
the `net_fix` disclosure, the `__debug__` and `torch.load` guards, the early no-clobber path,
the aliasing and determinism discipline, the full config↔code constant audit, the SLURM
hygiene, and the ≤ 4.0 GPU-hour budget. The displacement-donation formula itself — the heart
of the H-B repair — is correct in code.

---

## 6. What I did and did not execute

**Executed (read-only, login node, no compute):** `sha256sum` on the eight frozen artifacts
and the five run-time-asserted modules; `ls`, `wc`, `grep`, `sed`, `awk`/`gawk`, `head`,
`cut` over repository text files; full reads of `CLAUDE.md`, `AGENTS.md`, the review request,
`refine-logs/C02_A0_V3_RECORD.md`, `refine-logs/C02_A0_V2_PREREG_REVIEW.md`,
`refine-logs/C01_ZERO_CONTRACT_PROBE.md`, all seven frozen artifacts in full, and targeted
reads of `TARGET_STATE.json` (`registry_update_2026_07_28` incl. `hard_constraints`,
`serial_execution`, `unified_pilot_gate`, `candidate_registry` C02 and C14; and
`iteration_8_stage0_bounded_extraction_amendment` in full),
`refine-logs/HEADSPACE_TRANSFER_PREGATE.md` (§0 and the ARENA-2 / gate-ledger lines),
`refine-logs/C02_A0_RECORD.md` (arms, controls, gates), `refine-logs/C02_EXPERIMENT_PLAN.md`
(§3.1 truncation clause, KRR spec), `refine-logs/C02_DESIGN_REVIEW.md`,
`scripts/analysis/mechfix_ops.py` (`_norm32`, `_rank_weights`, `macro_f1`, `deployed_vote`),
`scripts/analysis/mechnov_pairverify.py` (`DATASETS`, `K_FOLDS`, `FOLD_SEED`),
`scripts/analysis/headspace_mint.py` (guard, `CLI`, `load_split`, dummy loaders),
`scripts/analysis/headspace_fidelity.py` (argparse, `B_fid_abs_3seedmean`, mint filename),
and `src/run_rac.py:1120-1145` (loader dispatch). Structural `gawk` queries over
`data/gt/{HateMM,MHC_zh}/{train,val}.jsonl` (row counts, whitespace-only counts, `len(T)`
distribution and maxima, escape presence, length histogram for the budget arithmetic).
Directory listings of `configs/c02`, `artifacts`, `data/MLLM_scores/*`,
`data/CLIP_Embedding/*`, `scripts/analysis/vsw_ckpt/*` and the `HateVideo` conda
`site-packages` (to read the scipy version from its `dist-info` directory name).

**NOT executed:** no Python was run — not `py_compile`, not the view module's `self_test()`,
not the arena's `oracle_self_test()`, not any reviewed script, not `jq`. No `.pt` cache, no
`.npz` mint or `vsw_ckpt` file, and no model was loaded. No GPU, no Modal, no teacher call.
**No `test_seen` cache, no `test.jsonl`, no `test_seen_segscoreK4_qwen.jsonl` and no test
label was opened or read at any point.** No SLURM job was submitted, released, held,
cancelled or inspected — `sbatch`, `squeue`, `sacct` and `scontrol` were never invoked. No
file under review was modified; the only file this review writes is this one. The record's
§5 preparation-time execution list and the implementer's synthetic dry-run results are
reported as claims and are **not** verified here.
