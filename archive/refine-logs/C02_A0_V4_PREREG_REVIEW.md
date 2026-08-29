# C02 A0 v4 — fresh independent static review

**Reviewer:** fresh independent static reviewer, round 4. No prior context on this candidate
beyond the artifacts named in `refine-logs/C02_A0_V4_REVIEW_REQUEST.md`; the implementer's
reasoning was neither seen nor requested.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed. See §6.

**Verdict:** `GO (0C/0H/23I)` — see §5 for the exact reading of the verdict token.

---

## 0. Hash and namespace verification

Every sha256 recomputed with `sha256sum` on the working tree at review time.

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v4.json` | `8ccd2464…3b26a4a9` | `8ccd2464699a7029db3952bc18612ea1cfcc79ede2b946e67051df843b26a4a9` | **MATCH** |
| `src/utils/c02_density_views.py` | `2ec193cd…d7955592` | `2ec193cdfa920a2d974db5c8468702614a54fa378a8df324ca5ba47b7d955592` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `66381c40…f992a319` | `66381c40a03c480bceab0af3d4c0497478e00da39a6de7ec0c33f6daf992a319` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `a1523087…4e992d8a9f` | `a1523087253990ce4a38642214aabd2890c34e650007d844ee5b9b4e992d8a9f` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `2afbe8b0…5a65e78a5` | `2afbe8b075aefb1cdd02669e0336c53d4306366deeed8714c7f11f58a65e78a5` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v4.py` | `71bba0f1…7c1364aba26` | `71bba0f1bd47517ea8da1bbd922274f66d4b2ef6c62099ca17cc97c1364aba26` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v4.sbatch` | `ae4a2375…c648a8cd6` | `ae4a237508ebfccde51cd3552903991d60001aad89f483e4861c490a648a8cd6` | **MATCH** |
| `refine-logs/C02_A0_V4_RECORD.md` | *(recompute and report)* | `de2c631dcbbdd70256b067a1b62d41671d0a8f06a17c489ac5e65a2781813251` | **REPORTED** |

Auxiliary hashes in record §4, all recomputed and **matching**:
`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…`, `headspace_mint.py` `cefdf8dc…`,
`mechnov_pairverify.py` `77b0defd…`, `mechfix_ops.py` `635c1312…`, `headspace_fidelity.py`
`72fd8e0a…`, and the superseded-record hashes `3c703b77…` (v1), `12c7e49e…` (v2),
`f54f08d9…` (v3). The three runtime-asserted constants in `c02_a0_arena_v4.py:80-87` and
`c02_a0_mint.py:56` equal the files on disk, so neither job will die on its own freeze
assert.

**Superseded executables — all nine ABSENT** (`ls` returns *No such file or directory* for
each): `configs/c02/c02_a0_v{1,2,3}.json`, `scripts/analysis/c02_a0_arena{,_v2,_v3}.py`,
`scripts/slurm/c02_a0_cpu{,_v2,_v3}.sbatch`. `configs/c02/` contains exactly one file.

**Namespace absence confirmed:** `artifacts/c02_edq` does not exist;
`find . -name '*c02den*'` (excluding `.git`) returns nothing, so no view cache exists in
`data/CLIP_Embedding/HateMM` or `data/CLIP_Embedding/MHC_zh`.

---

## 1. Repair verdicts for the four round-3 High findings

| # | finding | verdict | evidence |
|---|---|---|---|
| **H1** | the retracted "`s_Q` upper-bounds … failure is decisive" claim survived in the arena docstring | **REPAIRED** | `grep -rn "upper-bounds what any representation\|failure is therefore decisive\|A failure is decisive\|upper-bounds"` over **all eight** v4 artifacts returns hits **only** inside the record's own H1 narrative (`C02_A0_V4_RECORD.md:43,52,53`), which quotes the sentence it removed. A broader `grep -rniE "upper[- ]bound\|decisive\|supremum"` returns, in the executables: `c02_a0_arena_v4.py:19,24` and `:1074` and `configs/c02/c02_a0_v4.json:14,230` — every one of which is the *corrected* statement or the explicit `RETRACTED` note; plus `c02_a0_v4.json:232`, which uses "upper-bound diagnostic" only to name the registry's own permitted C14 role. The docstring at `:16-24` now matches the config and the string the arena emits at `:1067-1076`. Verified by grep, not by the record. |
| **H2** | `derangement_within` could oscillate forever on a size-2 group and die with a bare `AssertionError` outside the `Halt` path | **REPAIRED** | `c02_a0_arena_v4.py:285-290` is Sattolo's algorithm: a bounded `for i in range(m-1, 0, -1)` with `j = rng.integers(0, i)`, i.e. `j < i` strictly. It has no rejection step and no repair loop, so it terminates in exactly `m-1` swaps for every `m`, and it produces a single `m`-cycle, hence a derangement **by construction** for every `m ≥ 2` — including `m = 2` (`i=1, j=0` → `[1,0]`) and `m = 3`. The assert at `:290` can therefore never fire. Groups reaching this code always have `size ≥ 2` (`:277-278` skips smaller ones) and, for every reachable partition size, `shuffle_groups` emits no group of size 1 unless a partition contains exactly one item — see **I5**; with `n ≥ 579` and 5 stratified folds the smallest partition is ≈115 items, so that path is unreachable. Every item is covered: `fit ∪ ho = [n]` because the mint's `fit_idx` is `StratifiedKFold(...).split(...)[fold][0]` (`c02_a0_mint.py:129`), the exact complement of `ho`, so no item silently keeps its own displacement. |
| **H3** | "under H0 FULL and SHUFFLE are EXCHANGEABLE BY CONSTRUCTION" was false | **REPAIRED** | The claim is absent from every artifact (`grep -rniE "exchangeab"` hits only the corrected text). The replacement appears in all three places that matter and in identical terms: `c02_a0_arena_v4.py:335-344` (the code comment), `:863-870` (the string written into the result artifact), `configs/c02/c02_a0_v4.json:161`, and record §2/H3. I checked the replacement itself rather than accepting it: (a) the exchangeability null as stated — displacements i.i.d. across items and independent of the item — does make `{(NAT_i, d_i)}` and `{(NAT_i, d_{π(i)})}` equal in joint law, so the arms *are* exchangeable there, and restricting π to within-group permutations preserves that; (b) the radial counter-example is valid: `d_i = ε·NAT_i` gives `view_v(i) = (1+ε)NAT_i`, which `_norm32` maps to the same unit vector as `NAT_i`, so `FULL ≡ NATIVE` while `SHUFFLE` still moves; (c) the "cannot manufacture a PASS" argument is correct **for that null** — `Δacc = 0` there and the binding `+0.050` bar is unreachable. The residual scope gap (an orbit that is a *general* deterministic function of the native key, of which radial is the degenerate case) is not covered by this argument and is recorded as **I22**; it bounds what a PASS may be read as, and cannot affect a KILL. |
| **H4** | the SHUFFLE self-test re-typed the formula and could not fail | **REPAIRED** | `c02_a0_arena_v4.py:606-629` now constructs synthetic keys for **every** name in `V.VIEW_NAMES`, calls **`build_arms(tk, tch, tperm)`** itself (`:612`), and asserts against what the production path returned: `SHUFFLE[0]` is the untouched native view (`:613`), each `SHUFFLE[v]` equals `NAT + view_v[π] − NAT[π]` (`:616-620`), and — the regression guard proper — that it is **not** the donor's absolute view (`:621-622`). I confirmed the guard bites: had `build_arms` regressed to `keys[v][perm]`, `got` would differ from `want` by `NAT − NAT[π]`, which for the standard-normal test data is `O(1)` per component, so the `atol=1e-5` allclose at `:619` fails and the case raises. The renamed case string `shuffle_donates_displacement_via_build_arms` is what gets written into the artifact (`:629`, `:994`), so the artifact string cannot outlive the property. The added NOISE norm-match check (`:624-628`) also passes only because `build_arms` is the producer. |

No repair was argued away, and no repair claim in the record was accepted without reading
the corresponding bytes.

---

## 2. Findings

**Critical: none. High: none.** Twenty-three Info findings follow. Twelve are unrepaired
carry-overs from round 2's Info list, which I re-verified against the v4 bytes rather than
copying.

| # | class | location | finding |
|---|---|---|---|
| I1 | Info | `scripts/analysis/c02_a0_arena_v4.py:2` | Module docstring still names the superseded filename (`"""c02_a0_arena.py -- …`). The frozen artifact is `c02_a0_arena_v4.py` and its `run_id` is `C02-A0-v4`. Round 2 I1, unrepaired for two rounds. |
| I2 | Info | `configs/c02/c02_a0_v4.json:11` | `authority.record` still points at `refine-logs/C02_A0_RECORD.md`, the **v1** record, not at the v4 record that freezes this config. All five executables cite `C02_A0_V4_RECORD.md` correctly. Round 2 I2, unrepaired. |
| I3 | Info | `configs/c02/c02_a0_v4.json:198` | `gates.GATE_FID` states that the 0.050 bar is "enforced in `scripts/slurm/c02_a0_cpu_v2.sbatch`" — a file this very freeze deliberately deleted. The enforcement actually lives in `scripts/slurm/c02_a0_cpu_v4.sbatch:71-82` and is correct there. A hash-frozen config should not point its only gate-enforcement citation at a non-existent file. |
| I4 | Info | `refine-logs/C02_A0_V4_RECORD.md:175-183` | The table header reads "**Imported unmodified, sha256 asserted at run time**", but **nothing in the frozen set asserts `headspace_fidelity.py`'s sha256**. `c02_a0_arena_v4.py:80-87` pins only `mechfix_ops`, `mechnov_pairverify`, `headspace_mint`; `c02_a0_mint.py:92-95` pins `headspace_mint` and `mechnov_pairverify`; the wrapper invokes the reader unguarded (`c02_a0_cpu_v4.sbatch:66-69`). This is the **third** instance of the record asserting a property the code does not implement — the pattern that produced round 2's and round 3's H1 — and it is why I checked it. It stays Info because I recomputed the hash and it **matches** (`72fd8e0a…`), so the identity is pinned in prose, and because the gate is 5× slack (see I-note below). Related: `HEADSPACE_TRANSFER_PREGATE.md:411` records that reader's as-run sha as `3e0a35cd…`, so the file *has* changed since F113's HateMM pass (consistent with the later addition of the `zh` `FLOOR` entry at `headspace_fidelity.py:32-33`); "unmodified" is true only relative to this design, not relative to F113. |
| I5 | Info | `scripts/analysis/c02_a0_arena_v4.py:252-259`, `:277-278`, `:293` | H2's oscillation is gone, but the size-`<2` residue that round 2 flagged (I10) survives. If a partition contained exactly one item, `shuffle_groups`' two merge branches hand it back and forth (`nd`→`dg`) and emit a **size-1 group**; `derangement_within` then `continue`s past it and the global assert at `:293` fires as a **bare `AssertionError` outside the `Halt` path** — no result JSON, after the GPU is spent. Unreachable at `n ≥ 579` with 5 stratified folds (`|ho| ≈ 115-149`). One line — dropping a residual singleton, or raising `Halt` instead of `assert` — closes the last member of the family H2 named. |
| I6 | Info | `scripts/analysis/c02_a0_arena_v4.py:738-739`, `:797` | `degen_mask` is built purely from the manifest's text-derived `identity_views`, so a **video-decode-failure (zero-guard) item is not in it**, even though its orbit is the identity in every space: all six views share one zero text vector, hence one head key. Such an item therefore sits in the *non-degenerate* donor class and receives a **real donated displacement under SHUFFLE while FULL leaves it untouched** — precisely the asymmetry the degeneracy-matched grouping exists to prevent, and in the direction that makes `FULL > SHUFFLE` *easier*. Bounded and tiny: `C01_ZERO_CONTRACT_PROBE.md` reports one such row on HateMM train (index 355, `hate_video_95`), i.e. ≤ 0.13% of items, far below the `+0.050` bar; the count is reported (`VIEW_SUPPORT.zero_guard_videos`) so a reader can bound it post hoc. `degen_mask |= (zero-row mask)` would remove it. |
| I7 | Info | `scripts/analysis/c02_a0_arena_v4.py:684-694` vs `configs/c02/c02_a0_v4.json:200` | Zero-contract **criterion 3** ("no non-structural row with `0 < norm ≤ 1e-12`") is computed only over the six *view* text matrices. The banked `img_feats` and the banked `text_feats` are never tiny-checked — only their exact-zero rows are recorded (`:677-683`). The config says the four C01 criteria are applied "verbatim", and C01's own fail-closed halt (job 13712) was on the **img** modality. Impact is limited: any such row is identical in every arm and in the paired floor, so it cannot create an arm-to-arm delta, and it would enter only the SECONDARY raw arena's `l2n` (`:922-924`). |
| I8 | Info | `scripts/analysis/c02_a0_arena_v4.py:184-196`; `configs/c02/c02_a0_v4.json:165` | The `k = topk`-per-view-pair exactness argument is **correct as I re-derived it** — if `m_j ≥ τ` then at most 19 items are strictly above `j` in the achieving pair, so `j` is inside that pair's top-20 — but only when at most `topk` items attain `τ`. With `≥ topk` exact float32 ties at `τ` inside one pair, a boundary item can be dropped. The only exactly-tied rows reachable here are duplicate keys, i.e. the structural all-zero rows (1 on HateMM train, 0 on ZH), where every similarity is 0, the vote is identically 0 and the prediction is invariant — so the gap is unreachable in practice. The claim is nevertheless stated unconditionally in both the docstring and the config, and self-test case 3 (`:557-577`) uses tie-free random matrices. Round 2 I6, unrepaired. |
| I9 | Info | `scripts/analysis/c02_a0_arena_v4.py:504-507`; `configs/c02/c02_a0_v4.json:195` | The tie-exemption's stated reason — "their vote is invariant to tie order" — is false in general: two tied neighbours with different labels at adjacent ranks change the vote by `2·s·(w_r − w_{r+1})/Σw ≈ 0.0095`. The exemption is nonetheless **operationally safe**, because `parity_native:511-514` bit-checks predictions *and* the sorted top-20 similarity vector on **every** row, tied or not; only IDs are exempt, and a tie-induced vote flip would therefore HALT rather than pass silently. Self-test case 2 (`:546-555`) verifies invariance only for the all-zero query, which is the one case where it is true. Round 2 I4, unrepaired. |
| I10 | Info | `src/utils/c02_density_views.py:46-50` | The `LENGTH_GUARD` provenance correction that v2 applied to the config (`c02_a0_v4.json:36`, "in that spirit but is not the same criterion") is still absent from the frozen view module, which continues to assert that "the original C02 plan **already required** that such items 'are excluded from this view and counted' (`C02_EXPERIMENT_PLAN.md §3.1`)". The plan's clause is a *tokenizer-truncation* rule; `L_MAX = 12000` characters is a freely chosen new constant. Round 2 I5, unrepaired. |
| I11 | Info | `configs/c02/c02_a0_v4.json:49,56,65,73` | The `max_chars` values in `measured_identity_counts_2026_07_30` are unchanged from v2 (80731 / 12275 / 708 / 343); round 2 measured 80732 / 12276 / 710 / 343 three independent ways. I independently re-confirmed the **load-bearing** counts — `n = 744 / 107 / 579 / 78` by `wc -l`, and 39 / 9 whitespace-only HateMM `text` fields by `awk` — but did not re-derive `max_chars` to single-character precision, so I record this as unrepaired rather than re-measured. Nothing in any code path consumes `max_chars`. |
| I12 | Info | `scripts/analysis/c02_a0_arena_v4.py:921-944` vs `:1004` | The SECONDARY raw arena still runs **inside** `run_dataset`, after the primary read. Any `Halt` it raised would propagate before `out["datasets"][ds] = run_dataset(...)` assigns, discarding a completed primary measurement and turning the one permitted A0 submission into `HALT_FAIL_CLOSED_NO_DECISION`. I traced the reachable halts there (`orbit_vote`'s `nb < topk`, faiss-id and non-finite checks) and found none reachable — `|fit| ≥ 460`, and finiteness is asserted upstream at `:671-674`. Structure still inverts the declared primary/secondary hierarchy. Round 2 I8, unrepaired. |
| I13 | Info | `scripts/slurm/c02_a0_cpu_v4.sbatch:52-84`; `c02_a0_arena_v4.py:524-529` | `oracle_self_test`'s own justification — "it runs before any real data is opened so a numerical-contract break costs seconds, not a queue slot" — is defeated by the wrapper's ordering: 36 CPU mints and GATE-FID run first, so ~20-35 minutes elapse before the self-test executes. True inside the arena process, false for the job. Round 2 I9, unrepaired. |
| I14 | Info | `scripts/analysis/c02_a0_arena_v4.py:954-956`, `:736` vs `src/utils/generate_c02_density_view_text_embedding_HF.py:279-280` | The extractor records each view file's sha256 in the manifest, and the arena recomputes each view file's sha256 into its diagnostics — but the two are **never compared**; the manifest is checked only for id-set equality. A stale or swapped view file with a matching id/label vector would pass every gate. One `assert` closes it. Round 2 I13, unrepaired. |
| I15 | Info | `scripts/analysis/c02_a0_arena_v4.py:1001-1015` | `main()` exits 0 after a HALT: the `Halt` is caught, recorded and written out, so SLURM reports `COMPLETED` for a fail-closed run. Round 2 I11, unrepaired. |
| I16 | Info | `scripts/analysis/c02_a0_arena_v4.py:871-874` | `PARITY_NAT.predictions_and_sorted_similarities` is a **hardcoded string** ("BIT-EQUAL on all 15 (seed x fold) cells") rather than a counted measurement. The halts make it true wherever the line is reached, and `tie_rows_total` beside it *is* counted. Round 2 I12, unrepaired. |
| I17 | Info | all seven frozen artifacts | **None of the frozen set is tracked in git** (`git ls-files --error-unmatch` fails for every one), and the v3 bytes were overwritten in place. Record §3's claim that the view module, extractor, extraction wrapper and mint "changed **only** in their record pointer and, in the extractor, the re-pinned view-module hash" is therefore **unverifiable by diff** — and all four hashes did change (v3 record §4: `531d4574…`, `12846132…`, `001a0891…`, `da8a4918…`). Mitigation, stated so the finding is not overweighted: I read all four files in full, they contain exactly one `C02_A0_V4_RECORD.md` pointer each, and the extractor's `FROZEN_VIEWS_SHA256` (`:64`) does equal the current view module hash — consistent with the claim. Related gap: nothing verifies the **arena/mint/config** hashes before the job runs; the arena writes its own sha256 into the artifact (`:988-992`), so tampering between review and submission is detectable only post hoc. Round 2 I14, unrepaired. |
| I18 | Info | `configs/c02/c02_a0_v4.json:99`; `scripts/slurm/c02_density_extract.sbatch` | `budget_gpu_hours_cap: 4.0` is declared but nothing measures or enforces it in-job, and an overrun **voids** the result under amendment condition (f). My own arithmetic makes the cap comfortable: ≈4 224 + 592 (HateMM) + 3 474 + 468 (MHC-ZH) ≈ **8 760 text forwards** plus 1 508 video decodes, i.e. ~1.5-2.5 GPU-h at the historical per-forward rate, against the 4.0 cap and the amendment's own 2:00:08 worst observed extraction. ~12% of the spend buys `dev_seen` views that **no A0 code path reads** (declared as Stage-2 pre-payment at `:87`). Round 2 I16, unrepaired. |
| I19 | Info | `refine-logs/C02_A0_V4_RECORD.md:201-202`; `TARGET_STATE.json:93-94` | Amendment condition (e) — `one_candidate_at_a_time`, `parallel_gpu_or_teacher_pilots_forbidden`, and "a bounded Stage-0 extraction counts as a GPU pilot for both rules" — is deferred entirely to a manual `squeue` check at submission time, with no automated interlock. `current_execution_candidate` is `null`, which is favourable, but `current_design_boundary` reads `C04_IMPL_V5_CPU_PREFLIGHT_ENGINEERING_HALT_JOB_13805_V6_REPAIR_REQUIRED`, i.e. another candidate with an open job lineage. I did not run `squeue`/`sacct` and take no view on the live queue. Round 2 I17, unrepaired. |
| I20 | Info | `configs/c02/c02_a0_v4.json:151` vs `c02_a0_arena_v4.py:99-100` | The config states the NOISE seed (20260730) but not the SHUFFLE seed, although the permutation is as much a frozen constant as the noise draw. Both are `20260730` in code. The permutation is also **head-seed-independent** by construction (`derangement_within(..., SHUFFLE_SEED, f)`), which is correct and desirable but undocumented. |
| I21 | Info | `scripts/analysis/c02_a0_arena_v4.py:345-358` | Both nulls are built by **vector arithmetic in head-key space**, whereas `FULL`'s non-native views are genuine head outputs. If the head is materially nonlinear at the measured orbit radius, `SHUFFLE`'s and `NOISE`'s extra points sit off the head's image manifold and are handicapped for reasons unrelated to the null — again in the direction that makes both conjuncts *easier*. The design already reports `orbit_radius_median_oof` and its per-view breakdown (`:849-852`), which is exactly the quantity needed to bound this post hoc; it is not gated, and the record does not name the assumption. |
| I22 | Info | `scripts/analysis/c02_a0_arena_v4.py:335-344`, `:846-847`, `:853`; `configs/c02/c02_a0_v4.json:159,161` | The `FULL > SHUFFLE` conjunct is a **strict inequality with no margin**. v4's scope statement correctly covers the pure-radial null and the config now correctly notes that a purely *shared* displacement direction makes `SHUFFLE ≈ FULL` so the conjunct fails. The uncovered residue is the mixed case: displacement = shared component + item-specific component, where the accuracy movement is carried by the shared component. Then `FULL` and `SHUFFLE` both carry it, they differ only by noise, and a strict `>` becomes a coin flip rather than a filter — while `FULL > NOISE` would genuinely hold, because NOISE destroys the shared direction. More generally, an orbit that is any deterministic function of the native key can clear the bar without the density channel carrying anything new. `retrieval_length_spearman` (`:846-847`) and `krr_length_probe` (`:853`) are the right instruments and are computed per arm, but neither is gated, so nothing blocks a PASS driven by this artifact. This bounds what a PASS may be read as; it cannot affect a KILL, and the config's own "NECESSARY, NOT SUFFICIENT" language already concedes the shape of it. Round 2 I19, restated against the v4 wording. |
| I23 | Info | `src/utils/c02_density_views.py:42-45`; `configs/c02/c02_a0_v4.json:35` | The stated mechanism for the `EMPTY_TEXT` guard — "the deployed prompt substitutes `(none)` for falsy text; `T + " " + T` would be `" "`, which is truthy, so the prompt would change from `(none)` to `" "`" — applies literally only to `text == ""`, of which the frozen gt contains **zero** rows. The guard's actual trigger is `text.strip() == ""`, and the 39 HateMM-train rows that fire it hold whitespace, which is already truthy, so their deployed prompt was never `(none)` and no `(none)` flip is prevented. The *behaviour* is fine and conservative (all six views = `T`, identity orbit, symmetric across arms, counted in `VIEW_SUPPORT = 0.9355`, which I re-derived as `1 − 48/744`); only the recorded justification is inexact for 100% of the cases that actually trigger it. |

**None of I1-I23 can change the verdict, corrupt a gated quantity, touch test data, or kill
either job.** I5 and I12 are the only two that could in principle waste a submission, and I
traced both to unreachable preconditions.

---

## 3. What I verified independently and found sound

Recorded so that a GO is not read as an unexamined GO.

**Test contact.** No reachable path to any test cache, test jsonl or test label, including
through imported modules. Traced by hand: (a) the extractor restricts `--splits` to
`{train, val}` by an explicit membership check before anything opens
(`generate_c02_density_view_text_embedding_HF.py:147-150`), maps them to `train`/`dev_seen`
(`:66`), validates every gt path, every output path and the manifest path against four
forbidden tokens *before the 7B model loads* (`:154-170`), and wraps `torch.load` (`:72-80`);
(b) the mint's `load_view_text` asserts `split == "train"` (`c02_a0_mint.py:64`) and
`HM.load_split` constructs only `train_*`/`dev_seen_*` names; `run_rac`'s loader is replaced
wholesale (`:146-149`) so the harness's "test" dataloader is a dummy stratified slice of the
fitting pool (`:131-134`); `headspace_mint`'s import-time `torch.load` guard is inherited;
(c) the arena installs its own `torch.load` guard *before* importing any project module
(`c02_a0_arena_v4.py:63-78`) and `guard_path`s every constructed path — the only files it
opens are `train_*.pt`, `train_*-c02den-*.pt`, `mint_*_f{0..4}.npz`, `manifest_<ds>.json`
and `data/MLLM_scores/<ds>/train_segscoreK4_qwen.jsonl`; (d) GATE-FID's reader is a
`Val_Retrieval`-only hard filter (`headspace_fidelity.py:37-51`) whose regex cannot emit a
`Test_Retrieval` line; (e) `mechfix_ops`, `mechnov_pairverify` and the deployed extractor
have no module-level side effects (`if __name__ == "__main__"` guards verified), so importing
them opens nothing. **A0 itself consumes the train split only**; `dev_seen` is touched only
by the fidelity instrument's own head.

**Bar and decision rule.** `BAR_ACC = BAR_MF1 = 0.050` (`:93-94`), applied as
`delta_acc >= 0.050 and delta_mf1 >= 0.050` on **both** datasets on the 3-seed mean of the
pooled `FULL − NATIVE` delta (`:1055-1062`), matching
`unified_pilot_gate.stage_0_reachability` and amendment condition (c) verbatim. `target_met`
is hardcoded `False` in both branches (`:1023`, `:1065`) — correct, an A0 cannot meet the
campaign target. `net_fix_rate` is retained but the code, config and record all state
plainly that it is algebraically `n·Δacc` and cannot bind at `0.050 > 0.030`
(`:1050-1054`, config `:231`); the registry's net-fix clause is thereby discharged by the
accuracy bar, honestly labelled rather than dressed up.

**F113.** The primary read is the fold-head arena (bank = the fitting pool's keys from a head
trained on that same pool; queries = the held-out fifth), and the raw fused space is computed
into `secondary_raw_arena` (`:936-944`) which **no term of `ok` reads**. A raw-arena result
therefore cannot promote. Amendment condition (g) satisfied.

**Self-orbit exclusion.** `np.intersect1d(ho, fit).size` asserted per fold (`:794-795`), and
`fit` comes from the mint's `fit_idx`, the exact complement of `ho` under the same
`StratifiedKFold(5, shuffle=True, random_state=0)`, whose `ho_idx` every mint asserts
item-for-item against the banked `vsw_ckpt/<ds>/f{0..4}.npz` (`c02_a0_mint.py:115-126`);
the arena re-checks the parity flags (`:772-773`) and that fold assignment is identical
across seeds (`:781-783`). A query's own orbit can never be retrieved.

**PARITY-NAT.** `orbit_vote` on a singleton orbit is byte-for-byte the deployed call:
`_norm32` → `IndexFlatIP` → `k = 20`, and `parity_native` asserts predictions **and** the
sorted top-20 similarity vector against `mechfix_ops.deployed_vote` on **every** row of
every one of the 15 (seed × fold) cells per dataset, exempting only neighbour **IDs** on
rows with an exact float32 tie (`:509-519`). The arena's local `_norm32` (`:167-178`)
correctly *always* copies, where the frozen `mechfix_ops._norm32` can alias a float32
C-contiguous input into an in-place `faiss.normalize_L2`; I confirmed the frozen version
does alias (`mechfix_ops.py:37-42`) and that every array the arena hands to faiss is
private, and that `arms["NATIVE"][0][fit]` is an advanced-index copy so the deployed call
cannot mutate the mint's stored keys either.

**Zero contract.** Criteria 1 and 4 are honestly labelled
`DOCUMENTARY_CITATION_NOT_COMPUTED`; 2 and 3 are computed and asserted with a fail-closed
`Halt` (`:684-694`). Zero rows are retained identically in every arm — so no arm gains or
loses from them — and a sensitivity read excluding them is emitted separately
(`:907-919`). I verified that `faiss.normalize_L2` leaves an exactly-zero row at zero
(`fvec_renorm_L2` guards on `nr > 0`), that this is *tested in-job* on synthetic data by
self-test case 2 rather than assumed, and that a zero query's vote is therefore identically
0 regardless of tie order. Caveat I7 above.

**View contract and its pre-forward proof.** Every view is `T` with one contiguous block of
`T` duplicated in place, so `T` is an ordered subsequence by construction;
`assert_subsequence` (a correct shared-iterator greedy check, valid because leftmost
matching is optimal) is called for all six views **before** `load_video_frames` and before
any `_encode` (`generate_c02_density_view_text_embedding_HF.py:227-232`), so a construction
bug cannot reach a forward pass. Degenerate identity is bit-exact by construction: one
forward per **distinct view string**, the rest copied (`:245-254`). This satisfies
`C02_DESIGN_REVIEW.md` blocking finding 2 in full.

**Prompt fidelity.** `build_text_prompt` (`:124-129`) reproduces the deployed assembly
(`generate_VideoMLLM_embedding_lora_HF.py:438-442`) exactly under the deployed defaults,
which I confirmed are the English `TEXT_INSTRUCTION`, `"Title: "`, `"Transcript: "`,
`"(none)"` (`:63-68,151-170`) and are what `gen_embed_lora.sbatch` used (it passes none of
those flags). `num_frames = 8` and `max_pixels = 360*420` match the banked run. The base
extractor's sha256 is asserted at start-up and **matches** the pinned constant. `_encode`,
`read_gt`, `load_video_frames` are called with the signatures they actually have.

**Registry carry-overs.** All eight of the C02 registry's
`still_binding_requirements_carried_forward` are present:
`RANDOM_WINDOW_REPEAT`/`MIN_WINDOW_REPEAT`/`REPEAT_ONLY`/`LOCALIZED_REPEAT_ONLY` arms
(`:363-368`), frozen orbit radius (`:849-852`), frozen KRR length metric (`:449-490`),
retrieval-length correlation (`:846-847`), frozen confidence/control thresholds
(`:93-106`), declared lambda selection (declared N/A at A0, `:951-953`), declared Holm
family (`:1045-1047`, matching config `:208-213`), full self-orbit exclusion, and an
explicit fail-closed account of identity-orbit items (`VIEW_SUPPORT` + degeneracy-matched
donor grouping).

**Hard constraints.** No OCR (the channel is the gt `text` field the deployed encoder
already consumes); no cross-dataset mixing (each dataset runs in its own `run_dataset`, own
cache, own folds); no external API; single-dataset train split only; the parent-video binary
label is the only label used, and P3 scores enter only the two declared control arms as view
selectors, never as targets, features or pooling weights; no ensemble as a final method (the
multi-view max is retrieval-time oracle machinery, declared and dedup'd against C14 at
config `:232`, and the method contract has no multi-view inference); no size scaling.

**SLURM hygiene.** Both wrappers: `--partition=slurmpartition` and (for the GPU job)
`--gres=gpu:a100:1`, matching the historical extraction jobs; **no `--time`**; 8 CPU each,
so the 16-CPU aggregate-cap wedge cannot occur; no dependency, array, singleton, requeue,
chain, force or release anywhere; `set -euo pipefail`; `conda activate HateVideo`;
`CUDA_VISIBLE_DEVICES=""` on the CPU job; DET-1 thread env exported before any python
starts and asserted in-process. One submission each, strictly serial.

**Runtime survivability** (the "does it die after the GPU is spent" pass). All assets exist
and resolve: both LoRA adapters (both **older** than their banked caches, so the
re-extraction is against the same weights), both banked native caches, both P3 score files,
all four gt files with `wc -l` = 744/107/579/78, and **zero broken video symlinks** in
either `All/` directory (1066 and 806 targets, all resolving). The environment supplies
`scipy 1.17.1` (so `spearmanr(...).statistic` exists), `numpy 1.26.4` (so
`default_rng([seed, fold, gi])`, `rng.integers`, and `np.lexsort(..., axis=1)` behave as
used) and `faiss-cpu 1.13.2`. 1.8 TB free against a working set of ~1 GB. Every symbol the
arena and mint import exists with the signature used; the mint's npz payload carries exactly
the keys `headspace_fidelity` and the arena read back; the wrapper's `f{TAG}` naming
(`ffull` for `--fold -1`) matches both consumers. GATE-FID's stop rule will not trip: F113
measured `B_fid` = 0.0093 (HateMM) and 0.0086 (ZH) against this design's 0.050 bar, and the
c02 mint reproduces the same training trajectory (same CLI, same seeds, same dataloaders;
the extra view loads happen outside the RNG stream). I found **no** reachable non-`Halt`
exception, no unserialisable value for `json.dump`, and no division-by-zero or empty-mean
path that is not guarded.

**Statistics.** The paired item bootstrap is on the same estimand as the bar — the 3-seed
mean of the pooled metric, for both the point estimate and every replicate (`:399-433`) —
`B = 10 000`, seed `20260730`, percentile CI, both lower bounds required `> 0`, Holm over
the declared 4-member family. The KRR probe's declared repair (fit-fold z-scoring so
`gamma = 1/d` is meaningful on L2-normalised keys) is applied inside the fold loop with
nothing selected on the held-out fold, and its known limitation (the OOF key matrix mixes
heads) is declared and correctly scoped to a secondary diagnostic.

**Config ↔ code constant audit.** Every constant matches: `TOPK`/`topk` 20 and the weight
vector `[20…1]` (= `_rank_weights(20)`), `BAR_ACC`/`BAR_MF1` 0.050, `BAR_NETFIX_RATE` 0.030,
`VIEW_SUPPORT_MIN` 0.60, `BOOTSTRAP_B` 10000, `BOOTSTRAP_SEED`/`NOISE_SEED`/`SHUFFLE_SEED`
20260730, `ALPHA` 0.05, `ARENA2_MARGIN` 0.02, `ARENA2_CEILING` 0.98,
`EXT_PARITY_MEDIAN_COS_MIN` 0.99, `TINY_NORM` 1e-12, `KRR_RIDGE` 1.0 with `gamma = 1/d`,
`L_MAX` 12000, `K_WINDOWS` 4, `SEP` `" "`, seeds `(0,1,2)`, `K_FOLDS` 5, the nine arm names,
the two dataset keys and their cache dirs/model tags (against the frozen
`mechnov_pairverify.DATASETS`), the four Holm family names, both output namespaces, both
schema versions, both resource blocks. The measured identity counts I could re-derive
(`n`, whitespace-only rows, `view_support = 0.9355`) are correct; see I11 for the one class
I did not re-measure.

---

## 4. The adversarial questions, answered directly

**Can the treatment pass on an artifact neither SHUFFLE nor NOISE covers?** Partly, and the
design now says so itself. NOISE fixes the vector count and the per-item displacement norms
while randomising direction, so it covers the "more vectors means a bigger max" inflation
that is the obvious artifact of a max-over-orbit similarity. SHUFFLE keeps real displacement
directions and destroys the item correspondence, so it covers item-specific-radius and
hub-formation artifacts, and — because donation is of the **displacement**, not the position
— it no longer manufactures cross-item near-duplicates, so it no longer degrades under the
design's own null. What neither covers is an orbit that is a deterministic function of the
native key: radial displacement is the degenerate case (correctly named and correctly shown
to be harmless, because there `FULL ≡ NATIVE`), and a shared-direction-plus-noise
displacement is the live case (I22). There the strict `FULL > SHUFFLE` test degenerates to a
coin flip. That is a bound on the meaning of a PASS, not a route to one: the binding
constraint is `+0.050`/`+0.050` against the paired native floor on both datasets.

**Is the `k = 20` per-view-pair exactness argument correct?** Yes, and I re-derived it
independently rather than reading it: if `τ` is the 20th largest per-item maximum and pair
`(a*,b*)` achieves `m_j ≥ τ`, then every item ranked above `j` in that pair also has
`m ≥ m_j`, of which there are at most 19, so `j` is inside that pair's own top-20. The one
caveat is a `≥ 20`-way exact float32 tie at `τ` inside a single pair, which requires
duplicate keys; the only duplicates reachable here are the structural all-zero rows (one on
HateMM train), where every similarity is 0 and the vote is 0 irrespective of which ids come
back. Recorded as I8 because the claim is stated unconditionally.

**Would a KILL from this design be sound?** Yes, as the *gate verdict it is now labelled*.
`s_Q` is the canonical max-matching quotient pseudo-metric: it may use the best view of both
items in every comparison, which no deployable system may do, and it is evaluated in the
deployed head's key space on strictly held-out fifths with the deployed vote reproduced
bit-exactly on the native orbit. If that cannot reach `+0.050`/`+0.050`, the registry's
frozen Stage-0 rule closes the candidate. What a KILL is *not* — a proof that no
orbit-contracting representation could help, since a trained contraction re-ranks rather
than ceilings — is now stated in the docstring, the config, the record and the emitted
decision artifact, in the same words. The head is trained on native keys only, so a KILL is
specifically "the frozen deployed head does not already linearise this orbit usefully";
that is the correct reading and the artifact does not overstate it.

**Would a PASS be sound?** Yes, within its declared boundary — "authorises Stage-1 design
plus a fresh review", not a training gain, not a development result, not a test result. The
PASS conjunction is genuinely demanding: `+0.050` on both metrics on both datasets, strictly
beating both nulls on both metrics, both bootstrap lower bounds `> 0`, both Holm rejections.
The two soft spots are that the null conjuncts carry no margin (I22) and are mildly biased
in the easy direction by two small construction asymmetries (I6, I21); neither can supply
the `+0.050`, so neither can convert a KILL into a PASS. A PASS should be read as "the orbit
is reachable in the deployed head space under an optimistic oracle", and the accompanying
`retrieval_length_spearman` and `krr_length_probe` rows should be read before Stage-1 is
designed.

---

## 5. Verdict

```
GO (0C/0H/23I)
```

This is a **GO**, not a REVISE, under the rule the request states: *GO is permitted with
zero Critical and zero High findings.* I found zero of each, and I did not soften anything
to get there — the four round-3 High findings are repaired **in the code**, verified by
grep and by reading the replacement algorithms, not by trusting the record.

The count is reported honestly rather than as the literal token `GO (0C/0H/0I)`, because 23
Info findings exist and asserting `0I` would be false. The project has precedent for a GO
carrying Info (`GO 0C/0H/1I`, C04 v6 unlock review request). If the registry's literal
`GO (0C/0H/0I)` string is required as the execution key under amendment condition (a), the
cheapest complete path is: **I2, I3, I4, I1, I20** — five one-line documentation edits that
touch no science, no constant and no code path — followed by the two one-line code
hardenings **I5** (drop a residual singleton group instead of asserting on it) and **I6**
(`degen_mask |= zero-row mask`), and the one-line **I14** (compare the manifest's view-file
sha256 to the recomputed one). Nothing in that list requires re-review of the design, and
nothing in it changes a frozen threshold.

If the intent is to submit now, the two conditions I would restate as operator
preconditions, because no code enforces them, are: (1) the `squeue` check for amendment
condition (e) (I19), and (2) re-verifying the seven frozen sha256 values immediately before
`sbatch`, since none of the artifacts is under version control and other agents write this
tree concurrently (I17).

---

## 6. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `sed`, `awk`, `wc`, `du`, `df`, `git ls-files`,
and file reads. Specifically: recomputed all 8 declared hashes plus 9 auxiliary ones; read
in full `c02_a0_arena_v4.py`, `c02_a0_mint.py`, `c02_density_views.py`,
`generate_c02_density_view_text_embedding_HF.py`, both sbatch wrappers, the v4 config and
the v4 record; read `mechfix_ops.deployed_vote`/`_norm32`/`_rank_weights`/`macro_f1`,
`headspace_mint.py` in full, `headspace_fidelity.py` in full, the relevant blocks of
`mechnov_pairverify.py` and `generate_VideoMLLM_embedding_lora_HF.py`; read `CLAUDE.md`,
`AGENTS.md`, the `TARGET_STATE.json` blocks named in the request,
`C02_DESIGN_REVIEW.md`, `C02_EXPERIMENT_PLAN.md`, `C01_ZERO_CONTRACT_PROBE.md`,
`HEADSPACE_TRANSFER_PREGATE.md` §0-§4.5, and the round-1 and round-2 reviews; counted gt
rows with `wc -l` and whitespace-only `text` fields with `awk`; listed the video symlink
trees and checked for broken links; checked package versions by directory listing.

**Did not:** run Python of any kind (including `py_compile`), import any module, load or
open any `.pt` cache, `.npz`, model, adapter or video; open any `test_seen` cache,
`test.jsonl`, or any test label or metric; run `squeue`, `sacct`, `sbatch` or any SLURM
command; touch a GPU or Modal; modify, move or delete any file under review; or write
anything other than this review file.
