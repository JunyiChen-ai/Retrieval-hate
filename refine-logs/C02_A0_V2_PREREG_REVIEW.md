# C02 A0 v2 — fresh independent static preregistration review

**Reviewer:** fresh independent static reviewer (no access to the implementer's reasoning,
no contact with the prior reviewer)
**Date:** 2026-07-30 (Pacific/Auckland)
**Request:** `refine-logs/C02_A0_V2_REVIEW_REQUEST.md`
**Type:** read-only static review. Nothing was executed. See §6.

---

## 0. Hash and namespace verification

All seven declared sha256 values were recomputed with `sha256sum`. **All seven match.**

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v2.json` | `2d4b7148…d98a6f` | `2d4b7148154caea6ed41ec95043c15295c63d1abf3c47467b9191d285bd98a6f` | MATCH |
| `src/utils/c02_density_views.py` | `f6209f04…785b34` | `f6209f04f04b88cfe47fadd5f7c7cd20b079f397a646fe824c8d2c3b35785b34` | MATCH |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `1e40e530…321b0803` | `1e40e53013a032e527853cc5e82ca53b054882774b315a6ea6f319ce321b0803` | MATCH |
| `scripts/slurm/c02_density_extract.sbatch` | `aaee1516…2e76e8` | `aaee1516f52ff2aabf508580b5451973a0484b3ca0875116be33a92c252e76e8` | MATCH |
| `scripts/analysis/c02_a0_mint.py` | `f93a9d33…466da6b` | `f93a9d336c2917ede8737a8a597b7c9e3f83d5173ef4163b2e62118ba466da6b` | MATCH |
| `scripts/analysis/c02_a0_arena_v2.py` | `7315e323…d110e01d` | `7315e3232a42c96f1bf943028bb852eb89c9d85acd902f3890fb83fcd110e01d` | MATCH |
| `scripts/slurm/c02_a0_cpu_v2.sbatch` | `ccf9881c…38eccbf0` | `ccf9881ccae7019d261a393afec4e6504203b947d38a259bf4d68b0238eccbf0` | MATCH |

**`refine-logs/C02_A0_V2_RECORD.md` (requested, recompute and report):**
`12c7e49e7cfc361a21b6d04903ffe8dd3677b872eecb952b5f4d924254e9949c`

**Superseded v1 executables — confirmed ABSENT:** `configs/c02/c02_a0_v1.json`,
`scripts/analysis/c02_a0_arena.py`, `scripts/slurm/c02_a0_cpu.sbatch` do not exist.
`configs/c02/` contains only `c02_a0_v2.json`; `scripts/analysis/` contains only
`c02_a0_arena_v2.py` and `c02_a0_mint.py` under the `c02` prefix.

**Namespace absence — confirmed:** `artifacts/c02_edq` does not exist; zero `*c02den*`
files exist anywhere outside `.git`, and zero in
`data/CLIP_Embedding/{HateMM,MHC_zh}`.

**Run-time-asserted modules — all five recomputed and matching** the record §5 table
(`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…612399`, `headspace_mint.py`
`cefdf8dc…f0916612`, `mechnov_pairverify.py` `77b0defd…8598b7240d`, `mechfix_ops.py`
`635c1312…c83fc8d`, `headspace_fidelity.py` `72fd8e0a…08bf6598`), so the four in-job sha
pins (`c02_a0_arena_v2.py:75-82,852-854`, `c02_a0_mint.py:55-56,92-95`,
`generate_c02_density_view_text_embedding_HF.py:63-64,139-142`) will pass rather than
refuse.

**No hash mismatch. No Critical finding from §0.**

---

## 1. Repair verdicts for the four prior High findings

| # | prior finding | verdict | evidence |
|---|---|---|---|
| **H1** | leaking `SHUFFLE` derangement | **PARTIALLY REPAIRED** | The named leak is genuinely gone, verified in code. `derangement_within` (`c02_a0_arena_v2.py:234-263`) permutes strictly inside each supplied index group; the caller passes `[fit, ho]` inside the fold loop (`:698` head arena, `:815` raw arena) and `build_arms` applies the result at `:294`, so `keys[v][perm][fit] = keys[v][perm[fit]]` with `perm[fit] ⊆ fit` and `perm[ho] ⊆ ho`. No held-out query's views can reach the bank, so the cross-partition near-identity channel is closed. It is deterministic (`np.random.default_rng([seed, fold, gi])`, `:251`), fold-local, and identical between the head and raw arenas because `fit_by_fold[f]` is sklearn's `splits[f][0]` (`c02_a0_mint.py:129`, ascending, exact complement of `ho`) and equals `np.flatnonzero(fold_of != f)`. Self-test case 4 (`:525-532`) asserts zero fixed points and no boundary crossing. **What is not repaired is the consequence H1 rested on:** the conjunct `FULL > SHUFFLE` is still satisfied under the design's own null, by a bank-side channel the repair leaves intact — see **H-B**. |
| **H2** | false `EMPTY_TEXT` expectation | **REPAIRED** on every load-bearing count | I re-derived the table independently from `data/gt/<DS>/{train,val}.jsonl` with `grep`/`awk` only (no Python, no cache, no test). Confirmed exactly: whitespace-only rows **39 / 9 / 0 / 0**; `len(T) > 12000` rows **9 / 1 / 0 / 0**; zero overlap between the two causes, so full-identity **48 / 10 / 0 / 0**; view support **0.9355 / 0.9065 / 1.0000 / 1.0000**; `text == ""` zero everywhere; `n` = 744 / 107 / 579 / 78. Every gt line in all four files matches `^\{"id": "[^"]*", "text": ".*", "label": [0-9]+\}$` and contains **no** backslash escape, so the raw substring is the decoded string and gawk `length()` in `en_US.UTF-8` is an exact code-point count. The withdrawal of the v1 sentence and of the byte-vs-character confusion is correct. One non-load-bearing element of the same table is wrong — see **I3**. |
| **H3** | tautological `net_fix_rate` gate | **REPAIRED** | The identity is now stated rather than hidden, in four places: the emitted per-arm field is renamed `net_fix_rate_IDENTICAL_TO_delta_acc` (`:738`), the 3-seed summary carries `net_fix_rate_note` "fixed − broken = n · delta_acc EXACTLY … not an independent quantity" (`:778-780`), the decision code carries the comment "it can never bind" (`:935-939`), and the decision key is `net_fix_rate_implied_by_acc_bar` (`:917-918,941`). I re-derived the algebra: per seed `(fixed−broken)/n ≡ acc_arm − acc_native` by construction (`:730-731`), and `net_fix_rate_3seed_mean = mean(nets)/n ≡ delta_acc_3seed_mean` (`:772,777`), so with `BAR_ACC 0.050 > BAR_NETFIX_RATE 0.030` it cannot bind. Amendment condition (c) is therefore discharged arithmetically by the accuracy bar, and the record (§2 H3) and config (`:229`) both say so in those words. `precision_on_changed` is added and explicitly **not** gated (`:740-741,781-787`). |
| **H4** | unproved upper-bound claim | **NOT REPAIRED in the frozen executables** | The corrected statement is present in the emitted `interpretation_boundary` (`:952-961`), in `configs/c02/c02_a0_v2.json:228` and in the record. But the retracted sentence survives **verbatim in two files of the frozen set**: `scripts/analysis/c02_a0_arena_v2.py:17-19` ("`s_Q` is deliberately OPTIMISTIC: it **upper-bounds what any representation that contracts this orbit could buy** … A failure is therefore **decisive**") and `configs/c02/c02_a0_v2.json:14` (`"oracle_status": "OPTIMISTIC UPPER BOUND … A failure is decisive …"`). The record's `:116-117` states the claim was "withdrawn and replaced **everywhere** — record, config and the emitted `interpretation_boundary`"; that is false of the config and never covered the arena. See **H-A**. |

---

## 2. Findings

Classification: `Critical` = would void the result or breach a hard registry constraint;
`High` = would materially weaken a load-bearing conjunct, a required control, or a stated
scientific claim, or is a false assertion inside the hash-frozen set about a load-bearing
matter; `Info` = correctness/robustness/documentation, non-blocking on its own.

| # | sev | file:line | finding |
|---|---|---|---|
| **H-A** | **High** | `scripts/analysis/c02_a0_arena_v2.py:17-19`; `configs/c02/c02_a0_v2.json:14`; `refine-logs/C02_A0_V2_RECORD.md:116-117` | **The H4 retraction was not applied to the code or the config, and the record asserts that it was.** The arena's module docstring — the primary human-readable description of the instrument, whose bytes this prereg pins — still declares that `s_Q` "upper-bounds what any representation that contracts this orbit could buy" and that "a failure is therefore decisive". The config's `oracle_status`, the first substantive field a reader meets, still declares "OPTIMISTIC UPPER BOUND … A failure is decisive." These are the exact proposition H4 established to be unproved and, as stated, incorrect: `s_Q(i,j) = max_{a,b} cos(z_i^a, z_j^b)` is the canonical quotient (min-distance) pseudo-metric — *one* orbit-invariant similarity — and no invariance or Lipschitz condition is asserted anywhere that would make it a supremum over the class of orbit-contracting representations. The config now contains two mutually contradictory statements of the same claim (`:14` versus `:228`), and the arena will emit a corrected `interpretation_boundary` from a script whose own header contradicts it. Because a KILL is the expected outcome of a Stage-0 gate, the practical effect is that a KILL will ship accompanied by two frozen artifacts asserting a decisiveness the design has formally withdrawn — which is exactly the over-claim pattern this project has had to retract before. The record's "replaced everywhere" makes this a positive false statement about the repair, not merely an omission. |
| **H-B** | **High** | `configs/c02/c02_a0_v2.json:151`; decision conjuncts at `scripts/analysis/c02_a0_arena_v2.py:920-923,942`; `build_arms` at `:294` | **The within-partition derangement does not restore discriminating power to `FULL > SHUFFLE`, and the config claims it does.** `:151` states that a global derangement is what would make "the load-bearing `FULL > SHUFFLE` conjunct … near-vacuous", implying the fold-local version is not. It still is. With `perm` deranging inside `fit`, bank row `j` carries the non-native views of another **bank** row `m = π(j)`; those views lie within one orbit radius of `NAT_m`, so `s_Q(q, j) ≳ s(q, m) − radius` for every query `q`. Every genuine neighbour `m` of a query is therefore mirrored onto row `π⁻¹(m)`, which carries an unrelated label at essentially the same similarity, so roughly half the top-20 slots of a SHUFFLE query carry random labels. Under the design's own null — the density orbit carries no signal — SHUFFLE degrades sharply while FULL is a near-rank-preserving inflation of each item's **own** keys. `beats_shuffle_acc` and `beats_shuffle_mf1` are thus satisfied under H0 and cannot answer the question the design of record assigns to them (`C02_A0_RECORD.md:185`, "does the gain need the correct within-video orbit?"; `:188-190`, "load-bearing, not decoration"). This is structurally the *same* defect as H3 — a conjunct that cannot fail — and v2 fixed H3 by labelling it in place while leaving this one presented with full standing in `decision_rule.PASS` (`:225`) and in the `ok = all(...)` conjunction at `:940-947`. Note that the v2 record's own §2 H1 closing paragraph ("NOISE is the primary anti-inflation control … SHUFFLE is the membership control") is the honest framing and is *inconsistent with* the config text at `:151`; NOISE is indeed leak-free and is the only control here that can fail. |
| I1 | Info | `scripts/analysis/c02_a0_arena_v2.py:2` | The module docstring still names the superseded file (`"""c02_a0_arena.py -- …`). The frozen artifact is `c02_a0_arena_v2.py`, and its own `run_id` is `C02-A0-v2`. |
| I2 | Info | `configs/c02/c02_a0_v2.json:11` | `authority.record` points at `refine-logs/C02_A0_RECORD.md`, the **superseded** v1 record, not at the v2 record that freezes this config. The arena, mint, extractor and both wrappers all correctly cite `C02_A0_V2_RECORD.md`. |
| I3 | Info | `configs/c02/c02_a0_v2.json:49,57,65`; `refine-logs/C02_A0_V2_RECORD.md:78,79,80` | **Three of the four `max_chars` values in the H2 repair table are wrong.** Measured independently three ways (gawk `length()` in UTF-8; `wc -m`; and byte arithmetic on the isolated record, cross-checked against the astral-character count): HateMM train **80732** (record says 80731; line 540, `non_hate_video_533`, 80840 text bytes − 36×3 for 36 astral chars); HateMM val **12276** (says 12275; line 56, all-ASCII, 12276 bytes); MHC-ZH train **710** (says 708; line 436, 740 bytes − 2×3 − 12×2); MHC-ZH val **343** (says 343, correct). Nothing consumes `max_chars` — every count the `L_MAX` guard and `VIEW_SUPPORT` actually depend on is exact (see H2 above) — but this is a measured statement contradicted by the data, inside the very table created to repair a measured statement contradicted by the data. |
| I4 | Info | `refine-logs/C02_A0_V2_RECORD.md:137-140`; `configs/c02/c02_a0_v2.json:190,193`; `scripts/analysis/c02_a0_arena_v2.py:450-453,492-501` | The self-test verifies tie-order invariance **only for the all-zero query**, where every similarity is exactly 0 and the vote is identically 0. The general rationale still printed in the config (`:193`, "their vote is invariant to tie order") and in `parity_native`'s docstring is false: two tied neighbours with different labels at adjacent ranks change the vote by `2·s·(w_r − w_{r+1})/Σw ≈ 0.0095`. The record describes the special case as verifying "the PARITY-NAT tie exemption … verified rather than assumed", which overstates it. The exemption remains **operationally safe** — `parity_native:457-460` bit-checks predictions *and* the sorted top-20 similarity vector on **every** row, tied or not; only neighbour IDs are exempted. The stated reason should be corrected; the guard should not. |
| I5 | Info | `src/utils/c02_density_views.py:46-50` | The `LENGTH_GUARD` citation correction announced in record §3 and applied in `configs/c02/c02_a0_v2.json:36` ("in that spirit but not the same criterion") was **not** applied to the frozen view module, which still asserts that "the original C02 plan already required that such items 'are excluded from this view and counted' (`C02_EXPERIMENT_PLAN.md §3.1`)". The plan's clause is a *tokenizer-limit truncation* rule; the deployed processor call passes no `truncation` and no `max_length`, so no tokenizer limit exists and `L_MAX = 12000` characters is a freely chosen new constant. |
| I6 | Info | `scripts/analysis/c02_a0_arena_v2.py:182-194`; `configs/c02/c02_a0_v2.json:163` | The `k = topk` per-view-pair exactness argument is **correct as I re-derived it**, but only when the topk-th largest per-item maximum `τ` is attained by at most `topk` items. If `|{j : m_j ≥ τ}| > topk` (exact float32 ties at `τ`), a pair's own top-20 need not contain every row at or above `τ` and a boundary item can be dropped. Neither the code nor the stated argument carries the caveat, and self-test case 3 (`:503-523`) uses tie-free random matrices, so it does not exercise it. The one place ties at `τ` are certainly reachable — the all-zero structural-null query in the raw arena, where every similarity ties at 0 — is harmless, because the vote is 0 regardless of which 20 ids are returned. |
| I7 | Info | `scripts/analysis/c02_a0_arena_v2.py:616-629` | GATE-EXT still gates only the **median** row cosine at 0.99; `min_cos`, `mean_cos` and `max_abs_diff` are computed and emitted but never compared to anything. Up to half the rows may deviate arbitrarily and the gate still passes. 0.99 is also loose for a same-model, same-prompt, same-merged-adapter re-extraction where ≳0.999 is expected. Unchanged from v1 and not mentioned in §3. |
| I8 | Info | `scripts/analysis/c02_a0_arena_v2.py:806-829,887-889` | The SECONDARY raw arena runs **inside** `run_dataset`, after the primary read is complete. Any `Halt` it raises propagates out before `out["datasets"][ds] = run_dataset(ds, …)` assigns, so a failure in a computation the record labels "may corroborate a KILL, may never promote" discards a finished primary measurement for that dataset and turns the whole job into `HALT_FAIL_CLOSED_NO_DECISION`. The reachable halts there (`orbit_vote`'s non-finite and faiss-id checks) are unlikely, but the structure inverts the primary/secondary hierarchy the design declares. |
| I9 | Info | `scripts/slurm/c02_a0_cpu_v2.sbatch:52-84`; `scripts/analysis/c02_a0_arena_v2.py:473-474`; `refine-logs/C02_A0_V2_RECORD.md:133-134` | `oracle_self_test`'s own docstring justifies it as running "before any real data is opened so a numerical-contract break costs seconds, not a queue slot", and the record repeats the "before any real data is opened" framing. True inside the arena process, but the wrapper invokes the arena **last**, after 36 CPU mints and GATE-FID. F113 measured 30.6-40.4 s per mint (`HEADSPACE_TRANSFER_PREGATE.md:404`), so ~20-25 minutes of the job elapse before the self-test runs. Running it once at the top of the wrapper would cost seconds and match the stated intent. |
| I10 | Info | `scripts/analysis/c02_a0_arena_v2.py:249-250,253-262` | `derangement_within` has two latent degeneracies. A group of size `< 2` is silently skipped, leaving its members as fixed points, which then trips the global assert at `:262`. A group of size exactly 2 whose initial permutation is the identity oscillates: `i=0` swaps to `[1,0]`, `i=1` swaps back to `[0,1]`, forever, so the 64-iteration loop exits with fixed points and `:260` fires. Neither is reachable with 5 stratified folds over `n ≥ 579` (group sizes ~460-595 and ~115-149), and both fail closed — but they fail as a bare `AssertionError` outside the `Halt` path, so no result JSON is written at all. |
| I11 | Info | `scripts/analysis/c02_a0_arena_v2.py:886-899,962-966` | `main()` still returns **0** after a HALT: the `Halt` is caught, recorded, written out and the process exits normally, so SLURM will report `COMPLETED` for a fail-closed run. Unchanged from v1. |
| I12 | Info | `scripts/analysis/c02_a0_arena_v2.py:756-759` | `res["gates"]["PARITY_NAT"]["predictions_and_sorted_similarities"]` is a **hardcoded string**, "BIT-EQUAL on all 15 (seed x fold) cells". The halts make the claim true wherever the line is reached, but the artifact stores an assertion rather than a measurement. `tie_rows_total` next to it *is* counted. Unchanged from v1. |
| I13 | Info | `scripts/analysis/c02_a0_arena_v2.py:839-841` vs `src/utils/generate_c02_density_view_text_embedding_HF.py:279-280` | The extractor records the sha256 of every view file it writes into the manifest, and the arena computes the sha256 of every view file it loads — but the two are never compared. The arena reads the same manifest at `:632-639` and checks only the id set. A stale or mismatched view file with a matching id/label vector would pass. One `assert` would close it. |
| I14 | Info | `refine-logs/C02_A0_V2_RECORD.md:212-219` | The "superseded v1 identity, preserved for the audit trail" table lists only the config, the arena, the wrapper and the record. Four further files in the v2 frozen set have hashes **different** from the v1 record's §10 table — `c02_density_views.py` (`e0cd2d2b…` → `f6209f04…`), `generate_c02_density_view_text_embedding_HF.py` (`9ebb80f4…` → `1e40e530…`), `c02_a0_mint.py` (`3b1b602b…` → `f93a9d33…`) and `c02_density_extract.sbatch` (`e5c29338…` → `aaee1516…`). §3 discloses the extractor and mint changes; the view module and extraction wrapper changes are disclosed nowhere. All four are untracked in git and were overwritten in place, so their v1 bytes are unrecoverable and the diff cannot be bounded from the v2 record alone. (Reading the current bytes, the undisclosed deltas appear to be the `Record:` pointer updates to `C02_A0_V2_RECORD.md` and the extractor's `FROZEN_VIEWS_SHA256` re-pin, both consistent — but that is my inference, not a documented fact.) |
| I15 | Info | `src/utils/generate_c02_density_view_text_embedding_HF.py:67,72-77,91-96`; `scripts/analysis/c02_a0_arena_v2.py:61-69,128-133` | Token-list asymmetry in the test guards. The extractor's `FORBIDDEN_TOKENS` has four entries including `"test_"` and lowercases the path; its `torch.load` wrapper checks only `("test_seen", "/test")` and does **not** lowercase. The arena's `torch.load` wrapper and `guard_path` both omit `"test_"`. Defence-in-depth only: I traced every path the three scripts construct and found **no** reachable test cache, test jsonl or test label (see §3). |
| I16 | Info | `configs/c02/c02_a0_v2.json:99`; `scripts/slurm/c02_density_extract.sbatch` (whole) | `budget_gpu_hours_cap: 4.0` is declared but nothing measures or enforces it in-job; an overrun silently **voids** the result under amendment condition (f). My own arithmetic makes ~2 h plausible (§4), so the cap carries ≈2× headroom — but ~12% of the spend (185 `dev_seen` items × up to 6 text forwards) buys view files that **no A0 code path reads**, and there is no interlock if the projection is wrong. |
| I17 | Info | `refine-logs/C02_A0_V2_RECORD.md:253-255`; `TARGET_STATE.json:94` | Amendment condition (e) (`one_candidate_at_a_time`, `parallel_gpu_or_teacher_pilots_forbidden`, and "a bounded Stage-0 extraction counts as a GPU pilot for both rules") is deferred entirely to a manual `squeue` check at submission time, with no automated interlock. The registry's own `serial_execution.current_design_boundary` presently reads `C04_IMPL_V5_CPU_PREFLIGHT_ENGINEERING_HALT_JOB_13805_V6_REPAIR_REQUIRED`, i.e. a different candidate with an open job lineage. I did not run `squeue`/`sacct` and take no view on the live queue. |
| I18 | Info | `configs/c02/c02_a0_v2.json:225` vs `:229` | `decision_rule.PASS` still enumerates `net_fix_rate >= 0.030` inline as one of five conjuncts, with the disclosure that it cannot bind living in a separate `net_fix_clause` field four lines later. The two are consistent, but a reader of `PASS` alone still sees five independent gates. The arena's own summary field name (`net_fix_rate_IDENTICAL_TO_delta_acc`) is the better pattern. |
| I19 | Info | `scripts/analysis/c02_a0_arena_v2.py:296-304` (NOISE), `:294` (SHUFFLE), `:745-746` (diagnostic) | **Neither control covers a shared-direction inflation artifact.** NOISE draws an isotropic tangent per item and per view, so it destroys any *common* component of the native→view displacement; SHUFFLE preserves the common component but destroys label association (H-B) and so degrades regardless. If the repetition displacement has a large shared "verbosity/length" component — which is exactly what repeating a transcript would be expected to produce — then `s_Q` behaves like a global re-centering of the key space and can move accuracy with no item-specific density signal at all, while beating both controls. The right diagnostics exist and are computed per arm (`retrieval_length_spearman` at `:745-746`, `krr_length_probe` at `:752-753`), but **neither is gated**, so nothing blocks a PASS driven by this artifact. This bounds what a PASS from this design can mean; it does not affect a KILL. |

**Counts: 0 Critical, 2 High, 19 Info.**

---

## 3. What I verified independently and found sound

Recorded so the REVISE is not read as a rejection of the design.

**A. Registry and contract compliance.**
- *No reachable test path anywhere.* Extractor: `SPLIT_TO_OUTNAME` admits only `train`/`val`
  (`:66,148-150`), `assert_no_test_token` on gt, every output path and the manifest
  (`:156-169,213,273,304`), global `torch.load` wrapper (`:69-80`). Mint: `load_view_text`
  hard-asserts `split == "train"` (`:64`), `headspace_mint` installs its own `torch.load`
  guard at import, `run_rac.load_feats_from_CLIP` is replaced wholesale (`:146-149`), and the
  harness's dev/test dataloaders for a fold head are stratified slices of the fitting pool
  (`:131-134`). Arena: only `train_*.pt`, `train_*-c02den-*.pt`, `mint_*_s*_f{0..4}.npz`, the
  extract manifest and the **train** P3 jsonl are opened, each through `guard_path`.
  `generate_VideoMLLM_embedding_lora_HF.py` and `mechnov_pairverify.py` have `__main__`
  guards and no import-time side effects; `headspace_mint`'s only import-time effect is
  installing the guard. I found **no path, direct or through an imported module, that can
  reach a `test_seen` cache, `test.jsonl` or a test label.**
- *Split scope, bar, hard constraints.* Extraction is `--splits train,val` →
  `train`/`dev_seen` only, on both datasets. The `+0.050`/`+0.050` two-dataset bar is verbatim
  in `c02_a0_arena_v2.py:88-89,940` and `configs/c02/c02_a0_v2.json:225`, unchanged from v1.
  No OCR, no cross-dataset mixing or cross-dataset training, no external API, own train split
  only, parent-video binary label only, no size scaling, no cross-seed ensembling (seeds are
  averaged for the read, never stacked into a predictor). The C14 boundary is now confronted
  explicitly (record §3; config `:230`) and the registry entry (`TARGET_STATE.json:240-247`)
  permits precisely the "frozen mechanism/upper-bound diagnostic" role this A0 occupies.
- *F113 compliance.* The primary arena is the fold-head/deployed-head path; the raw fused
  arena is labelled `SECONDARY … may corroborate a KILL, may never promote` (`:821-823`) and
  **does not enter the decision at all** — `main()` reads only `summary_3seed` and
  `bootstrap_FULL_vs_NATIVE` (`:912-929`). GATE-FID uses the frozen, unmodified
  `headspace_fidelity.py` and its dev-only `Val_Retrieval` hard filter; the wrapper's
  `B_fid < 0.050` bar is enforced at `c02_a0_cpu_v2.sbatch:71-82`. The instrument's own
  `raw_effect_under_test: 0.0255` / `STOP_RULE_TRIGGERED` fields are now disclosed as F105's,
  not this design's (record §3; config `:196`). F113 §2.1's companion rule ("a Δ smaller than
  `B_fid` cannot carry a verdict") is not restated, but it is **automatically satisfied**: any
  PASS needs Δ ≥ 0.050 while the gate forces `B_fid < 0.050`.
- *File-name wiring checked end to end.* The wrapper writes `mint_<ds>_s<seed>_f{full,0..4}.npz`
  (`:55-60`); `headspace_fidelity.py` reads `mint_{ds}_s{s}_ffull.npz` and requires
  `meta.eval_curve/secs/n_dev/n_train/head_dim`, all of which `c02_a0_mint.py:218-233` emits;
  the arena reads `mint_{ds}_s{seed}_f{0..4}.npz`. No name mismatch that would kill the job
  after the 36 mints.
- *SLURM hygiene.* Both wrappers: no `--time`, no `--dependency`, no `--array`, no
  `--singleton`, no `--requeue`, no release/force path, `conda activate HateVideo`, 8 CPU each
  (so the submit-time 16-CPU aggregate cap cannot wedge), one submission each, strictly serial.
  The A0 wrapper is CPU-only with `CUDA_VISIBLE_DEVICES=""` and exports the DET-1 thread block
  before any python starts, which `HM.det1_assert("8")` re-checks in-process.
- *All required inputs exist on disk:* both banked `train_*.pt` and `dev_seen_*.pt` caches,
  both LoRA adapter directories, all ten `vsw_ckpt/<ds>/f{0..4}.npz`, both P3
  `train_segscoreK4_qwen.jsonl`, and all six GATE-FID floor trainlogs (13241 × 3 seeds,
  13150 × 3 seeds). `artifacts/c02_edq` is absent, so both no-clobber paths are clean.

**B. Numerics and implementation that hold.**
- *Subsequence contract.* `build_views` only ever forms `T[:c_k] + " " + T[c_{k-1}:c_k] + T[c_k:]`,
  `T + " " + T`, or `T` itself, so `T` is an ordered subsequence by construction;
  `assert_subsequence` (`c02_density_views.py:135-144`) is the correct greedy-iterator test, is
  called per item per view **before** `BASE.load_video_frames`/`BASE._encode`
  (`generate_c02_density_view_text_embedding_HF.py:227-232`), and is itself negatively tested
  (`deletion_rejected`). Degenerate identity is bit-exact by object reuse (`:245-254`), not by
  tolerance.
- *Prompt parity.* `build_text_prompt` (`:124-129`) is byte-identical to the deployed assembly
  at `generate_VideoMLLM_embedding_lora_HF.py:438-442` under the deployed defaults, which I
  checked at source: `--text_instruction` default `TEXT_INSTRUCTION`, `--title_label` default
  `"Title: "`, `--transcript_label` default `"Transcript: "`, `--none_placeholder` default
  `"(none)"`, and `gen_embed_lora.sbatch` passes **no** prompt override. I also confirmed
  structurally that **no** gt line in any of the four train/val files carries a `title` key
  (every line matches `^\{"id": …, "text": …, "label": [0-9]+\}$`), so both the banked and the
  new extraction take the `(none)` branch identically.
- *Bootstrap estimand is now the bar's estimand, and is paired.* `paired_bootstrap:345-379`
  computes both the point estimate and every replicate as `mean_s[metric(arm) − metric(floor)]`
  over the three seeds on the resampled item set, which is algebraically identical to
  `delta_acc_3seed_mean` / `delta_mf1_3seed_mean` at `:772-773`. The same index vector is
  applied to both arms and to `y`, so pairing is at item level. `M.macro_f1` is pure numpy, so
  B = 10000 × 3 seeds × 2 arms is seconds, not hours.
- *`oracle_self_test` runs before any real data is opened* (`main:852-857`, after the frozen-module
  sha check and before the config load, the output-path checks and `run_dataset`). Case 1 proves
  singleton-orbit bit-exact parity with `mechfix_ops.deployed_vote` on predictions, sorted
  similarities, neighbour ids and votes at max-diff 0.0; case 3 proves the multi-view `k = topk`
  search reproduces a brute-force max over view pairs on the top-20 item set — this is the one
  code point v1 had no in-job verification for, and it is now covered; case 4 proves the
  derangement has no fixed point and never crosses the group boundary. These would catch a faiss
  code-path change, an aliasing regression, an accumulation-index error in
  `best[rows, I] = maximum(...)`, and a derangement regression.
- *The `__debug__` guards fire where claimed and cannot be stripped:* `c02_a0_arena_v2.py:55-56`,
  `c02_a0_mint.py:46-47`, `generate_c02_density_view_text_embedding_HF.py:53-54` are `if`
  statements, not asserts, so `python -O` hits them. All three precede any assert-guarded work.
  (The view module carries no such guard, and the record does not claim one; its asserts are
  reached only through the extractor, which does.)
- *The arena's `torch.load` guard is installed* (`:58-69`), closing the v1 gap.
- *The early no-clobber path covers every file the extraction job writes:* all 12 view paths
  (2 splits × 6 views) plus the manifest are validated absent before `transformers`/`peft` are
  even imported (`:152-170`), and the late checks at `:274-275,306-307` remain as a second line.
  The only unchecked write is the transient `manifest + ".tmp"`.
- *No aliasing and no in-place mutation of shared arrays.* The arena's `_norm32:162-173` always
  copies; every array handed to faiss is private; `build_arms` produces fresh arrays for SHUFFLE
  (fancy index) and NOISE (`astype` + arithmetic); `orbit_vote` receives fancy-indexed copies.
  `M.deployed_vote`'s aliasing `_norm32` is only ever handed `arms["NATIVE"][0][fit]` /
  `[ho]`, which are already fresh temporaries.
- *Determinism.* `NOISE_SEED`/`SHUFFLE_SEED`/`BOOTSTRAP_SEED` are frozen constants; `rngn` is
  re-created per `build_arms` call so the tangent set is identical across folds and seeds;
  `derangement_within` seeds on `[SHUFFLE_SEED, fold, group]`. `scipy` in the `HateVideo`
  env is **1.17.1**, so `spearmanr(...).statistic` exists (and the same call pattern is already
  used by `scripts/analysis/errpat_zh_remint_fidelity.py:66`) — the tie-corrected Spearman will
  not raise.
- *Arena leakage.* Bank/query disjointness is asserted per fold against the mint's own `fit_idx`
  (`:695-696`), which is sklearn's `splits[f][0]` under `StratifiedKFold(5, shuffle=True,
  random_state=0)` asserted item-for-item against the banked `vsw_ckpt/<ds>/f{0..4}.npz` inside
  **every** mint (`c02_a0_mint.py:115-126`, re-checked at `c02_a0_arena_v2.py:673-674`). The
  held-out fifth never enters the head's training, dev or test loaders. Self-orbit exclusion is
  complete in every arm, including SHUFFLE (donors stay inside the query partition). `fold_of`
  is asserted identical across seeds.
- *`ARENA-2` is calibrated correctly.* F113 §4.x measured pooled head-space native accuracy
  ≈0.8867 on HateMM against a majority rate ≈0.5995; `[maj + 0.02, 0.98]` will pass.
- *Zero contract.* Criteria 2 and 3 are computed and asserted across the banked native and all
  six views (`:587-597`); criteria 1 and 4 are now honestly labelled
  `DOCUMENTARY_CITATION_NOT_COMPUTED` (`:599-611`) instead of being emitted as hardcoded `true`,
  which is the correct fix. Structural nulls are retained identically in every arm and a
  sensitivity read excluding them is reported (`:792-804`).

**C. Config ↔ code constant audit.** Every constant in `configs/c02/c02_a0_v2.json` was checked
against the code and **all matched**, except the claims already raised as findings
(`oracle_status` at `:14` — H-A; the SHUFFLE rationale at `:151` — H-B; the tie rationale at
`:193` — I4; `max_chars` at `:49,57,65` — I3). Spot list: `k_windows 4` ↔ `K_WINDOWS`;
`window_cut_rule (k*len)//4` ↔ `window_cuts`; `separator " "` ↔ `SEP`; `l_max_chars 12000` ↔
`L_MAX`; `identity_causes` ↔ `DEGEN_*` branches; `views.names` ↔ `VIEW_NAMES`; `topk 20` ↔
`TOPK`; `weights [20..1]` ↔ `M._rank_weights(20)`; `vote v ≥ 0` ↔ `(votes >= 0)`; `tie_rule
lower bank index` ↔ `lexsort((idx, -best))`; `seeds [0,1,2]` ↔ `SEEDS`; all nine `arms` ↔
`build_arms` (memberships exact); `argmin/argmax ties → lowest index` ↔
`argmin_window`/`argmax_window`; `bootstrap B=10000 seed 20260730` ↔ `BOOTSTRAP_B`/`BOOTSTRAP_SEED`;
`alpha 0.05` ↔ `ALPHA`; `holm_family` ↔ the four `pv` keys; `ARENA2 [maj+0.02, 0.98]` ↔
`ARENA2_MARGIN`/`ARENA2_CEILING`; `GATE_EXT median cos ≥ 0.99` ↔ `EXT_PARITY_MEDIAN_COS_MIN`;
`VIEW_SUPPORT ≥ 0.60` ↔ `VIEW_SUPPORT_MIN`; `TEST_PATH` tokens ↔ `guard_path`;
`krr gamma=1/d, ridge=1, target log1p(len)` ↔ `krr_length_probe` (d = 1024 head keys);
`orbit_radius` median over items and non-native views, strict OOF ↔ `rad_oof` stack median;
`membership` ↔ `mean_top20_overlap_with_native`; `det1_threads 8` ↔ `HM.det1_assert("8")` +
wrapper exports; `extraction 8/1/64G`, `a0 8/0/32G`, `time null` ↔ both sbatch headers;
`extract_namespace`/`a0_namespace`/`result_file`/`decision_file`/`atomic_json`/`no_clobber` ↔ the
wrappers and `main()`; `supersedes.*_sha256` ↔ the v1 record's §10 values.

---

## 4. Adversarial questions the request asks, answered directly

**Can the treatment pass on an artifact neither `SHUFFLE` nor `NOISE` covers?** Yes — a
shared-direction (verbosity/length-axis) displacement. See **I19**. NOISE randomises the
direction per item and per view and so cannot reproduce a common component; SHUFFLE carries the
common component but is dominated by its own label-corruption bias (**H-B**). The two
diagnostics that would expose it are computed but not gated.

**Is the `k = 20` per-view-pair exactness argument correct, and do ties break it?** Correct as
re-derived, with a tie caveat the design does not state — see **I6**. Ties at `τ` are the only
break, they are essentially unreachable on real float32 head keys, and the one place they are
certainly reachable (the all-zero raw-arena query) is harmless.

**Is `PARITY-NAT` binding, and is the tie exemption a loophole?** Binding, and not a loophole.
Predictions and the sorted top-20 similarity vector are bit-checked on **every** row of every
one of the 15 (seed × fold) cells per dataset; only neighbour IDs are exempted on tie rows, and
the count is reported. For a singleton orbit `orbit_vote` degenerates to the literal deployed
`k = 20` faiss call, and both `_norm32` implementations produce byte-identical float32
C-contiguous normalised copies from the same float32 input, so bit-equality will reproduce. The
*rationale* printed for the exemption is still wrong (**I4**); the guard is right.

**Is the arena free of leakage?** Yes. Bank/query disjoint and asserted per fold; the held-out
fifth is never seen by its fold's head in any role (the harness's dev and test loaders are
stratified slices of the fitting pool); full self-orbit exclusion holds in every arm including
SHUFFLE after the v2 repair. The residual SHUFFLE issue (**H-B**) is a **bias**, not leakage:
no query-side information reaches the bank.

**Would a KILL be sound?** Yes, as a **gate verdict under the registry's frozen Stage-0 rule** —
which is exactly what the emitted `interpretation_boundary` now says. None of the defects above
contaminates it: H-B only makes a control easier to beat, I19 only bounds a positive, I3/I5/I14
are documentary, I7 only weakens a parity gate in the direction of admitting the run. The
problem is that the KILL will ship alongside two frozen artifacts asserting the retracted
"a failure is decisive" (**H-A**), which is precisely the over-claim the H4 repair was supposed
to remove.

**Would a PASS be sound?** Not as stated. One of the seven per-dataset conjuncts is satisfied
under the design's own null (**H-B**), a second is an acknowledged algebraic identity (H3, now
correctly disclosed), the extraction-parity gate constrains only half the rows (**I7**), and no
gate blocks the shared-direction artifact (**I19**). NOISE, the bootstrap CIs, the Holm family
and the `+0.050`/`+0.050` bar are the parts of a PASS that would carry weight.

**Budget plausibility (asked explicitly).** The amendment's own `sacct` table prices comparable
8 CPU / 1 GPU / 48-64 G extraction jobs at 00:24 to 02:00, with `13468 gen_embed_readout`
(4 readout cells × 2 datasets × 3 splits × 2 streams) the largest at 02:00:08. C02 does
1508 items × one video decode + up to 6 text forwards, with the 48 HateMM full-identity items
deduped to one forward each and the sequence growth confined to the text tokens (`RFULL` roughly
doubles transcript tokens; visual tokens at `num_frames=8`, `max_pixels=151200` dominate).
**~2 GPU-hours is a credible projection and the 4.0 cap carries ≈2× headroom.** The residual
risk is procedural (**I16**), not arithmetic.

---

## 5. Verdict

REVISE (0C/2H/19I)

`GO` is not available. **H-A** leaves the H4-retracted upper-bound claim standing verbatim in
two sha256-frozen artifacts while the record asserts it was removed everywhere — a repair
verified only in prose, not in the code, which is the specific failure mode this review was
instructed not to accept. **H-B** leaves a PASS conjunct that is satisfied under the design's
own null and adds a frozen claim that the v2 repair removed that property, when it did not.
Neither requires re-architecting: H-A is two sentences in two files plus a corrected record
line; H-B needs either the same in-place vacuity label that H3 correctly received, or the
removal of `beats_shuffle_*` from the conjunction with SHUFFLE retained as a reported
diagnostic. The nineteen Info findings are individually non-blocking; **I3**, **I5**, **I14**
and **I4** are worth folding in at the same time because they are all instances of a repair
narrative running ahead of the artifacts.

The remainder of the design — the extraction contract, the subsequence proof, the split scope,
the head arena, self-orbit exclusion, PARITY-NAT, the zero contract, the bootstrap estimand
repair, the in-job self-test, the `__debug__`/`torch.load` guards, the early no-clobber path,
the constant audit and the budget — I verified independently and found sound.

---

## 6. What I did and did not execute

**Executed (read-only, login node, no compute):** `sha256sum` on the eight frozen artifacts and
the five run-time-asserted modules; `ls`, `wc`, `grep`, `sed`, `awk`, `head`, `tail`, `od`,
`printf`, `tr` over repository text files; full reads of `CLAUDE.md`, `AGENTS.md`,
`TARGET_STATE.json` (the amendment block, `hard_constraints`, `serial_execution`,
`unified_pilot_gate`, `candidate_registry` incl. C02 and C14), the review request,
`C02_A0_V2_RECORD.md`, `C02_A0_RECORD.md`, `C02_A0_PREREG_REVIEW.md`, `C02_DESIGN_REVIEW.md`,
`C02_EXPERIMENT_PLAN.md`, `C01_ZERO_CONTRACT_PROBE.md`, `HEADSPACE_TRANSFER_PREGATE.md`
(§0-§3, §4.1-§4.5 and the GATE-FID/stop-rule text), all seven frozen artifacts in full, and
`mechfix_ops.py`, `headspace_mint.py`, `headspace_fidelity.py`, `mechnov_pairverify.py`
(constants/table) and `generate_VideoMLLM_embedding_lora_HF.py` (prompt assembly, argparse
defaults, `__main__` guard). Structural queries with `grep`/`awk` over
`data/gt/{HateMM,MHC_zh}/{train,val}.jsonl` (line shape, escape presence, whitespace-only
counts, `len(T)` distribution in characters, longest-record byte arithmetic). `ls` on the
`HateVideo` conda `site-packages` to read the scipy version from its `dist-info` directory
name. `git status --porcelain` and `git log --oneline -3` on the frozen paths. Three
intermediate text extracts were written to the session scratchpad only.

**NOT executed:** no Python was run — not `py_compile`, not the view module's `self_test()`,
not the arena's `oracle_self_test()`, not any reviewed script, not `jq`. No `.pt` cache, no
`.npz` mint or `vsw_ckpt` file, and no model was loaded. No GPU, no Modal, no teacher call.
**No `test_seen` cache, no `test.jsonl`, no `test_seen_segscoreK4_qwen.jsonl` and no test
label was opened or read at any point.** No SLURM job was submitted, released, held, cancelled
or inspected — `sbatch`, `squeue`, `sacct` and `scontrol` were never invoked. No file under
review was modified; the only file this review writes is this one. The record's §4 freeze-time
defect findings, its §6 preparation-time execution list and the implementer's synthetic dry-run
results are reported as claims and are **not** verified here.
