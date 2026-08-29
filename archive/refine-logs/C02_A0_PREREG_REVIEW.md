# C02 A0 — independent static preregistration review

**Reviewer:** independent static reviewer (no access to the implementer's reasoning)
**Date:** 2026-07-30 (Pacific/Auckland)
**Request:** `refine-logs/C02_A0_REVIEW_REQUEST.md`
**Type:** read-only static review. Nothing was executed. See §5.

---

## 0. Hash verification

All seven declared sha256 values were recomputed with `sha256sum` and **all seven match**.

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v1.json` | `0b8a8289…b53d1` | `0b8a8289e7438396ce081fdf872f7d18017f870640fa33a687099de4066b53d1` | MATCH |
| `src/utils/c02_density_views.py` | `e0cd2d2b…2eab72` | `e0cd2d2b920a4f5133f30d174d36865843fe23977ff1f8639eea0400d12eab72` | MATCH |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `9ebb80f4…c6832b` | `9ebb80f48d27fd14278b15692d45d3c925efc84ae61941fae1488574bc96832b` | MATCH |
| `scripts/slurm/c02_density_extract.sbatch` | `e5c29338…acafbc` | `e5c29338fab4b0ac1af4c57826e11bde9d96f29b111bd806b98ccc1658acafbc` | MATCH |
| `scripts/analysis/c02_a0_mint.py` | `3b1b602b…a0bbfa7` | `3b1b602b145fa362f270ba08a604a1b284ae153f0d22f9a15dafa5c3a0abbfa7` | MATCH |
| `scripts/analysis/c02_a0_arena.py` | `92abe7d8…93af41` | `92abe7d8157a54f89a47657fb1edaf4a8f90e55b873c3fd03840aa940593fa41` | MATCH |
| `scripts/slurm/c02_a0_cpu.sbatch` | `2b55c678…c82c281` | `2b55c67834fc6dfdaf9a932be634c735b5362edcd128cfd5aa6e3829fc82c281` | MATCH |

**`refine-logs/C02_A0_RECORD.md` (requested, recompute and report):**
`3c703b77d7cd6ebeac965d60378fffba8714dadb66fe16c91cf45fbfc42e679b`

The five "imported unmodified, sha256 asserted at run time" modules (record §10) were also
recomputed and **all five match** the record's table
(`generate_VideoMLLM_embedding_lora_HF.py` `75bb8156…612399`, `headspace_mint.py`
`cefdf8dc…f0916612`, `mechnov_pairverify.py` `77b0defd…8598b7240d`, `mechfix_ops.py`
`635c1312…c83fc8d`, `headspace_fidelity.py` `72fd8e0a…4bf6598`).

Namespace-absence claim (record §10) independently verified: `artifacts/c02_edq` does not exist
and zero `*-c02den-*` files exist in `data/CLIP_Embedding/{HateMM,MHC_zh}`.

**No hash mismatch. No Critical finding from §0.**

---

## 1. Findings

Classification: `Critical` = would void the result or breach a hard registry constraint;
`High` = would materially weaken a load-bearing conjunct, a required control, or a stated
scientific claim; `Info` = correctness/robustness/documentation, non-blocking on its own.

| # | sev | file:line | finding |
|---|---|---|---|
| **H1** | High | `scripts/analysis/c02_a0_arena.py:217-228,257,583-584` | **The SHUFFLE control is corrupted by a fold-blind derangement.** `derangement(n, SHUFFLE_SEED)` permutes over all `n` train items, and `build_arms` applies it to the full `[n,d]` key matrices *before* the per-fold `X[fit]` / `X[ho]` split. Consequently, in each fold ~80% of held-out queries carry the non-native views of an item that is **in the bank** (`P(perm[u] ∈ fit) = |fit|/n ≈ 0.80`), and ~20% of bank rows carry the non-native views of an item that is a **query**. Either way the query obtains a near-identity match `cos(RW_k^b, NAT^b) = 1 − orbit_radius` to a bank row whose label is unrelated to its own. Since the orbit radius is by design small, this ghost lands at or near rank 1 and injects a random-label neighbour worth `2·s·w_1/Σw ≈ 0.19` of vote mass into the majority of SHUFFLE queries. SHUFFLE is therefore biased *downward* by construction, the load-bearing conjunct `FULL > SHUFFLE` (record:188-190; decision `beats_shuffle_acc/mf1` at `c02_a0_arena.py:787-790`) becomes near-vacuous, and the design cannot answer its own stated question, "does the gain need the **correct** within-video orbit?" (record:185). `NOISE` is unaffected and remains a valid inflation control, but it does not cover the correct-orbit question. Remedy: derange within `fit` and within `ho` separately, inside the fold loop. |
| **H2** | High | `refine-logs/C02_A0_RECORD.md:104-106` (guard at `src/utils/c02_density_views.py:88`) | **A frozen design-time fact is false, and it understates the identity-orbit fraction by 5.2 pt on the primary dataset.** The record states: "`grep -c '"text": ""'` returns 0 on all four train/val files, so `EMPTY_TEXT` is expected to be empty — the guard exists because it must, not because it is expected to fire." The implemented condition is `text.strip() == ""`, not `text == ""`. Measured on the frozen gt with `jq`: **39 of 744 HateMM `train` rows and 9 of 107 HateMM `val` rows have `text` = `" "` (a single space)** — every one of the 39 length-1 train rows is whitespace-only. `EMPTY_TEXT` will therefore fire on **5.24%** of HateMM train, and combined with the 9 `LENGTH_GUARD` rows **48/744 = 6.5% of HateMM train items carry the full identity orbit**, i.e. `FULL ≡ NATIVE` for them. (MHC-ZH: 0 whitespace-only, max 708 chars, so support ≈ 1.0.) The mechanism is not biased by this — identity applies to every arm — and `VIEW_SUPPORT` (0.935) still clears 0.60, but a hash-frozen prereg must not assert a measured fact that its own data contradicts. |
| **H3** | High | `scripts/analysis/c02_a0_arena.py:612-620,802-809`; `refine-logs/C02_A0_RECORD.md:262`; `configs/c02/c02_a0_v1.json:116` | **The `net_fix_rate ≥ 0.030` conjunct is a tautology and can never bind.** By construction `fixed − broken = n·(acc_FULL − acc_NATIVE)` exactly, so `net_fix_rate_3seed_mean ≡ delta_acc_3seed_mean` (both are `mean_s(nets_s)/n`). The decision at `:802-809` therefore tests the *same number* against 0.050 and against 0.030; the second test is implied by the first and is unreachable. Registry amendment condition (c) — "with enough net corrected-minus-broken items for the `+0.030` final bar" — is thus discharged by an algebraic identity rather than by any independent quantity, and the frozen decision rule presents a dependent quantity as an independent gate. A non-vacuous reading would need a different statistic (e.g. `fixed/n`, or churn `changed/n` versus `net_fix`), which the code already computes but does not gate. |
| **H4** | High | `refine-logs/C02_A0_RECORD.md:58-61`; `configs/c02/c02_a0_v1.json:15`; `scripts/analysis/c02_a0_arena.py:11-19` | **"`s_Q` upper-bounds what any representation that contracts this orbit could buy … therefore a failure is decisive" is an unproved and, as stated, incorrect claim — and it is the sole bridge from the oracle to the mechanism.** `s_Q(i,j) = max_{a,b} cos(z_i^a, z_j^b)` is the canonical *quotient (min-distance) pseudo-metric* induced by the orbit: it is **one particular** orbit-invariant similarity, not a supremum over the class of orbit-contracting representations. No Lipschitz or invariance condition is stated that would make it a bound, and none holds for a learned projector. Concretely, a trained contraction *removes* the density direction from the geometry and therefore **re-ranks** neighbours, whereas the max operator leaves the density nuisance fully present in every pairwise cosine and only adds a ceiling; the oracle can read ≈0 while a genuine contraction gains, which is exactly the C02 hypothesis. A KILL from this design is therefore a kill of *this max-matching oracle*, not of the evidence-density mechanism, and record §8's "A KILL closes C02 and the serial loop advances" over-claims. (See also I16: the oracle is itself an eval-time multi-prompt ensemble, a construct the registry bans as a final method; this claim is the only thing carrying the result across that line.) |
| I1 | Info | `refine-logs/C02_A0_RECORD.md:223-225`; `configs/c02/c02_a0_v1.json:87` | The PARITY-NAT tie exemption is justified by "their vote is invariant to tie order". That is **false**: two tied neighbours with different labels at adjacent ranks change the vote by `2·s·(w_r − w_{r+1})/Σw ≈ 0.0095`. The exemption is nevertheless operationally safe, because `parity_native` (`c02_a0_arena.py:415-418`) still asserts bit-equality of predictions **and** of the sorted top-20 similarity vector on every row, tied or not. The stated reason should be corrected; the guard should not. |
| I2 | Info | `scripts/analysis/c02_a0_arena.py:419` | The tie detector only looks for duplicates *within* the returned top-20 (`len(np.unique(sim_ref[i])) < TOPK`). A tie straddling the 20th/21st position leaves `sim_ref` internally distinct while faiss's heap order and the arena's `lexsort` may select different bank ids — tripping the ID assert and HALTing the whole 36-mint job. Fail-closed, but it can waste the run. |
| I3 | Info | `scripts/analysis/c02_a0_arena.py:165-176`; `refine-logs/C02_A0_RECORD.md:284-293` | The `k = topk` per-view-pair exactness argument is **correct as I re-derived it**, but only when the topk-th largest per-item maximum `τ` is attained by at most `topk` items; exact float32 ties at `τ` can drop a boundary row, and the record states no such caveat. Separately, PARITY-NAT exercises only the **singleton** orbit path, so the multi-view `best[rows, I] = maximum(...)` accumulation — the one code point the record itself reports as a freeze-time defect — has **no in-job verification** and a defect there would be visible only in the treatment arms, never in the floor. A dense 6×6 brute-force cross-check on one `(seed, fold)` cell would cost milliseconds. |
| I4 | Info | `scripts/analysis/c02_a0_arena.py:501-512` | GATE-EXT gates only the **median** row cosine at 0.99; `min_cos`, `mean_cos` and `max_abs_diff` are computed and reported but never gated. Up to half the rows could deviate arbitrarily (e.g. a subset of videos decoding to different frames on the new run — `text_feats` attend over frames) and the gate still passes. 0.99 is also loose for a same-model, same-prompt, same-adapter re-extraction, where ≳0.999 is expected. |
| I5 | Info | `scripts/analysis/c02_a0_arena.py:488-496` | Zero-contract criteria **1** and **4** are emitted as hardcoded literals and are never verified: nothing asserts that the observed zero ids are the documented `hate_video_95` / row 355, and `banked_img_zero_rows` is reported but never compared to the text zero mask (C01's "other modality" enum). Criterion 2's cross-view mask assert does pin the observed set to the banked set, so the practical protection stands, but two of four criteria are guards that cannot fire. (Row index 355 / `hate_video_95` independently confirmed against `data/gt/HateMM/train.jsonl` line 356 and `slurm/logs/lora_embed_13329.out:88-91`.) |
| I6 | Info | `scripts/slurm/c02_a0_cpu.sbatch:79` vs `scripts/analysis/headspace_fidelity.py:102-103` | The enforced GATE-FID stop bar is `B_fid < 0.050`, but the "frozen, unmodified" instrument writes its own `raw_effect_under_test: 0.0255` and `STOP_RULE_TRIGGERED: B_fid >= 0.0255` into the same JSON. If `B_fid` lands in `[0.0255, 0.050)` the emitted artifact will read `STOP_RULE_TRIGGERED: true` while the job proceeds to a verdict. Neither the record nor the config discloses F113's own 0.0255 bar, nor carries over F113 §2.1's companion rule that a Δ smaller than `B_fid` cannot carry a verdict. F113 §4.1 measured `B_fid = 0.0093` on HateMM, so this is unlikely to bite in practice. |
| I7 | Info | `refine-logs/C02_A0_RECORD.md:95`; `configs/c02/c02_a0_v1.json:31` | `LENGTH_GUARD` is attributed to `C02_EXPERIMENT_PLAN.md §3.1`, whose clause is *"items that would truncate under the frozen native tokenizer limit are excluded from this view and counted"*. The deployed encoder's processor call (`generate_VideoMLLM_embedding_lora_HF.py:352-357`) passes no `truncation` and no `max_length`, so **no tokenizer limit exists** and the plan's condition never applies. `L_MAX = 12000` characters is a new, freely chosen constant, not the plan's rule; the record's "no snapping heuristic and no tunable parameter" applies to the window cut rule only. (The guard is nevertheless good engineering — the 80 731-char item would otherwise double to ~40 k tokens.) |
| I8 | Info | `refine-logs/C02_A0_RECORD.md:101-104` | The design-time length statistics are quoted in **bytes** (p50 745, p99 12 710, max 80 784) while `L_MAX` is compared against `len(T)` in **characters** (`c02_density_views.py:90`). Measured char values are p50 693, p90 3038, p95 3866, p99 12 274, max 80 731. The `>12000` count happens to agree at 9 rows, but the two units are not interchangeable near the threshold on UTF-8 text. |
| I9 | Info | `scripts/analysis/c02_a0_arena.py:393-401` | `spearman` uses `argsort(argsort(x))` ranks with **no tie averaging**. HateMM `lengths` has 39 items tied at 1 (plus further ties), so the reported retrieval-length correlation is computed on arbitrarily broken ranks. Diagnostic only; no gate depends on it. |
| I10 | Info | `scripts/analysis/c02_a0_arena.py:349-390` | The KRR length probe fits on rows produced by four *different* fold heads and predicts rows from a fifth. The heads share an initialisation and drift little (F113 §4.2 measures max abs key delta 0.026-0.040 across seeds), so this is probably benign, but the cross-head basis assumption is undeclared and the resulting R² is not comparable to any within-head Stage-1 number it may later be compared against. |
| I11 | Info | `scripts/analysis/c02_a0_arena.py:111-116`; `refine-logs/C02_A0_RECORD.md:130-133` | Unlike the extractor (`generate_c02_density_view_text_embedding_HF.py:66-76`) and the mint (via `headspace_mint.py:106-116`), the arena installs **no** global `torch.load` guard; it relies solely on `guard_path()` over the paths it constructs, and its token list omits the extractor's `"test_"`. The record's "`torch.load` is wrapped by the head-space instrument's guard" is true of the extractor and mint but not of the arena. No reachable test path was found in the arena, so this is a defence-in-depth gap, not a leak. |
| I12 | Info | `configs/c02/c02_a0_v1.json:49`; `scripts/slurm/c02_density_extract.sbatch` (whole) | `budget_gpu_hours_cap: 4.0` is declared but nothing measures or enforces it in-job; an overrun silently **voids** the result under amendment condition (f). My own arithmetic makes ~2 h plausible (see §3), so the cap has ≈2× headroom — but ~12% of the spend (185 `dev_seen` items × 6 text forwards) buys view files that **no A0 code path reads**, and there is no interlock if the projection is wrong. |
| I13 | Info | `src/utils/generate_c02_density_view_text_embedding_HF.py:241-242,271-273` | The no-clobber asserts fire only **after** every forward for a split has been computed. A rerun after a partial failure re-burns the entire HateMM split on the GPU before dying at the write step. A pre-flight existence check of all 6 output paths (and the manifest) before the item loop would be equally fail-closed and far cheaper. |
| I14 | Info | `src/utils/generate_c02_density_view_text_embedding_HF.py`, `scripts/analysis/c02_a0_mint.py`, both wrappers | Every fail-closed guard in the extractor and mint — including the split guard, the test-path guard, the sha256 pins and `assert_subsequence` — is a bare `assert`. Under `python -O` / `PYTHONOPTIMIZE` they are all stripped. Neither wrapper sets or explicitly clears `PYTHONOPTIMIZE`. |
| I15 | Info | `refine-logs/C02_A0_RECORD.md:357-360`; `TARGET_STATE.json` `serial_execution.current_design_boundary` | Amendment condition (e) (`one_candidate_at_a_time`, `parallel_gpu_or_teacher_pilots_forbidden`) is deferred entirely to a manual `squeue`/`sacct` check at submission time, with no automated interlock. The registry's own `current_design_boundary` presently reads `C04_IMPL_V5_CPU_PREFLIGHT_ENGINEERING_HALT_JOB_13805_V6_REPAIR_REQUIRED`, i.e. a different candidate with an open job lineage. |
| I16 | Info | `TARGET_STATE.json` `hard_constraints[4]`; `refine-logs/C02_A0_RECORD.md:52-61,174-190` | `hard_constraints` forbids "multi-prompt ensemble **as a final performance method**". The A0 oracle *is* an eval-time 36-way multi-prompt max ensemble. It is declared non-deployable and a PASS authorises only Stage-1 design, so the constraint is not violated on its face — but the record never confronts the constraint at all, and the only bridge from this banned construct to a legal single-encoder contraction is the unproved upper-bound claim in **H4**. |
| I17 | Info | `scripts/analysis/c02_a0_arena.py:283-301,271`; `refine-logs/C02_A0_RECORD.md:191-197` | The MIN/MAX P3 approximation is **adequately confined**: the primary `FULL` arm uses no P3, the fallback for a missing P3 row is the identity orbit, and `n_p3_missing` is reported. Both `train_segscoreK4_qwen.jsonl` files exist with full train coverage (HateMM 744 rows). But the approximation maps *temporal* ASR-window scores onto *character* quarters of a string, so `MIN_WINDOW_REPEAT` — a control the 2026-07-29 reviewer required **by name** — may be statistically indistinguishable from `RANDOM_WINDOW_REPEAT`; the requirement is met nominally rather than substantively. Separately, items whose selected window is empty (`len(T) < 4`; 53 HateMM train rows) silently collapse MIN/MAX to the identity orbit and are not counted anywhere. |
| I18 | Info | `scripts/analysis/c02_a0_arena.py:642-664,802-806` | The macro-F1 **point estimate** the bar is applied to (`delta_mf1_3seed_mean` = mean of per-seed mF1 deltas) and the macro-F1 **interval/p-value** (`paired_bootstrap` on the *3-seed majority* predictions) are two different estimands. The choice is declared in `configs/c02/c02_a0_v1.json:99`, and accuracy is consistent (`corr` mean ≡ `delta_acc_3seed_mean`), but the mF1 CI does not bound the mF1 statistic the decision rule thresholds. |
| I19 | Info | `scripts/analysis/c02_a0_arena.py:336-346,799-809` | Holm over `{hatemm_acc, hatemm_mf1, zh_acc, zh_mf1}` is applied to a decision that is a pure **conjunction** (intersection-union test), for which no multiplicity correction is required. The choice is conservative, so it cannot inflate a PASS; noted only so the family size is not later mistaken for a necessary correction. The bootstrap p-values are the standard CI-inversion form `(#{Δ* ≤ 0} + 1)/(B + 1)`, floored at 1e-4, which is below the smallest Holm threshold (0.0125). |
| I20 | Info | `scripts/analysis/c02_a0_arena.py:250,283,636-639,756-768`; `scripts/analysis/c02_a0_mint.py:193` | Housekeeping: `build_arms(keys, ids, p3, choices)` never uses `ids` or `p3` (dead parameters); `res["gates"]["PARITY_NAT"]` records "BIT-EQUAL on all 15 (seed × fold) cells" as a **hardcoded string** rather than a counted value (the halts make the claim true when reached, but the artifact stores an assertion, not a measurement); `main()` exits **0** on a HALT, so SLURM will report the job COMPLETED for a fail-closed run; and `run_rac`'s `--force True` (inherited verbatim from `headspace_mint`) collides in name with the config's `no_..._force` SLURM clause. |

**Counts: 0 Critical, 4 High, 20 Info.**

---

## 2. What I verified and found sound

Recorded so the REVISE is not read as a rejection of the whole design.

**A. Contract compliance.**
- *Subsequence contract holds and is proved before any forward.* `build_views` only ever forms
  `T[:c_k] + " " + T[c_{k-1}:c_k] + T[c_k:]` or `T + " " + T` or `T` itself, so `T` is an
  ordered subsequence of every view by construction. `assert_subsequence`
  (`c02_density_views.py:135-144`) is the correct greedy-iterator subsequence test, is called
  per item per view **before** `BASE.load_video_frames` / `BASE._encode`
  (`generate_c02_density_view_text_embedding_HF.py:195-199`), and is itself negatively tested
  (`self_test` case `deletion_rejected`). Nothing is deleted, reordered or paraphrased.
- *Degenerate identity is bit-exact, not tolerance-based.* The extractor dedupes by view string
  (`:210-222`) so an identity view reuses the same tensor object; degenerate causes are typed,
  counted per item and rolled up into `view_support`.
- *All four reviewer-named controls are present and genuinely distinct* from the treatment
  (`REPEAT_ONLY` ⊂ `FULL`, `LOCALIZED_REPEAT_ONLY` ⊂ `FULL`, `RANDOM_WINDOW_REPEAT` per-item
  hash-selected, `MIN_WINDOW_REPEAT` P3-argmin). `SHUFFLE` and `NOISE` match `FULL`'s orbit
  cardinality (6), so the max-inflation artefact is cardinality-matched. `NOISE`'s random
  direction in 1024-d is near-orthogonal to `NAT`, so its angular spread is at least that of the
  true displacement — i.e. `NOISE` is a **conservative** control. (`SHUFFLE` is not — see H1.)
- Orbit radius, KRR probe, retrieval-length Spearman, control thresholds, lambda-selection
  status, Holm family and self-orbit exclusion are each specified in the config **and**
  implemented in code (see §4 audit).

**B. Registry compliance.**
- *No reachable test path anywhere.* Extractor: `SPLIT_TO_OUTNAME` admits only `train`/`val`
  (`:63,136-138`), path guard on gt/output/manifest (`:87-90`), global `torch.load` guard
  (`:66-76`). Mint: `load_view_text` hard-asserts `split == "train"`, `headspace_mint`'s
  `torch.load` guard is installed at import, `load_feats_from_CLIP` is replaced wholesale, and
  the harness's "test" dataloader is a stratified slice of the fitting pool. Arena: only
  `train_*.pt`, `mint_*_f{0..4}.npz`, the extract manifest and the **train** P3 jsonl are
  opened, each through `guard_path`. `generate_VideoMLLM_embedding_lora_HF.py` has no import-time
  side effects (`__main__` guard at `:568`). I found **no path, direct or through an imported
  module, that can reach a `test_seen` cache, `test.jsonl` or a test label.**
- *Split scope, bar, and hard constraints.* Extraction is `train,val` → `train,dev_seen` only;
  the `+0.050`/`+0.050` two-dataset bar is verbatim in `configs/c02/c02_a0_v1.json:116` and in
  `c02_a0_arena.py:71-72,802-803`; no OCR, no cross-dataset mixing, no external API, own train
  split only, parent-video binary label only, no model-size scaling, no cross-seed ensembling
  (seeds are averaged for the read, never stacked into a predictor).
- *F113 compliance.* The **primary** arena is the fold-head / deployed-head path; the raw fused
  arena is computed, labelled `SECONDARY … may corroborate a KILL, may never promote`
  (`:694-696`), and — importantly — **does not enter the decision at all** (`main()` reads only
  `summary_3seed` and `bootstrap_FULL_vs_NATIVE` from the head arena). F113 §1.3's rule that a
  CPU-minted arm must be paired against a CPU-minted floor is honoured: the `NATIVE` floor is
  the same-session re-extracted `NAT` through the same CPU fold head.
- *SLURM hygiene.* Both wrappers: no `--time`, no `--dependency`, no `--array`, no `--singleton`,
  no `--requeue`, no release/force path, `conda activate HateVideo`, 8 CPU each (so the
  submit-time 16-CPU aggregate cap cannot wedge), one submission each, strictly serial. The A0
  wrapper is CPU-only with `CUDA_VISIBLE_DEVICES=""` and exports DET-1 threads before any python
  process starts, which `det1_assert` then re-checks in-process.

**C. Scientific validity that does hold.**
- *No arena leakage.* Bank/query disjointness is asserted per fold (`:578-579`) against the
  mint's own `fit_idx`, which comes from `StratifiedKFold(5, shuffle=True, random_state=0)`
  asserted item-for-item against the banked `vsw_ckpt/<ds>/f{0..4}.npz` inside **every** mint
  (`c02_a0_mint.py:112-123`, re-checked at `c02_a0_arena.py:556-557`). All ten `f*.npz` files
  exist. The held-out fifth is never seen by its fold's head in any role — the harness's dev and
  test loaders are dummy slices of the fitting pool. Self-orbit exclusion is complete: a query's
  own orbit is not in its own bank. `fold_of` is asserted identical across all three seeds.
- *PARITY-NAT binds and will reproduce.* For a singleton orbit, `orbit_vote` degenerates to
  exactly the deployed `k = 20` faiss call; `best` holds the 20 returned similarities and `-inf`
  elsewhere, and `lexsort((idx, -best))[:, :20]` recovers faiss's descending order. Both
  `_norm32` implementations produce byte-identical float32 C-contiguous normalised copies from
  the same float32 input, and both index the same fresh temporaries — so predictions, sorted
  similarities and neighbour ids will match bit-for-bit. The record's §9.2 aliasing repair is
  real and correctly implemented (`np.array(..., copy=True)` at `:154`).
- *`ARENA-2` is calibrated correctly.* F113 §4.2 measured pooled head-space native accuracy
  0.8867 on HateMM against a majority rate of ~0.5995; `[maj + 0.02, 0.98]` will pass.
- *No module shadowing* from the `sys.path.insert(0, …)` of `src/utils` and `scripts/analysis`
  in either the mint or the arena; no file in either directory collides with a stdlib or
  third-party module name used downstream.
- *A KILL is sound as a kill of this oracle* (subject to H4's scope limit) — it is not
  contaminated by any of the defects above: H1 only makes a control easier to beat, H2 only
  shrinks the effective n, H3 is vacuous in both directions. **A PASS is not sound as stated**:
  one of the two mechanism controls is broken (H1), one of the five conjuncts is a tautology
  (H3), and the interpretation the record attaches to it rests on H4.

**Budget plausibility (asked explicitly).** Job `13329` (`slurm/logs/lora_embed_13329.out`) is
the closest comparable: HateMM, same `num_frames=8`, same `max_pixels=151200`, same merged-LoRA
path, 744 + 107 + test items × **2** forwards, elapsed `00:27:18`. That is ≈1.5 s per item for
one decode plus two forwards. C02 does 1508 items × one decode + up to **6** text forwards
(deduped for identity items; the 9 worst-case HateMM rows are guarded to 1 forward), giving
≈3.9 s/item ≈ 1.6-1.7 h of compute plus two full base-model loads and LoRA merges. Sequence
growth is modest (visual tokens dominate; `RFULL` adds ~28% of the text tokens only).
**~2 GPU-hours is a credible projection and the 4.0 cap carries ≈2× headroom.** The residual
risk is procedural, not arithmetic (I12).

---

## 3. Config ↔ code constant audit

Every constant in `configs/c02/c02_a0_v1.json` was checked against the code. **All matched**;
no config claim was found that the code fails to implement, except the two already raised as
findings (`net_fix_rate` at `:116` is vacuous — H3; `PARITY_NAT`'s tie rationale at `:87` is
wrong — I1).

`k_windows 4` ↔ `K_WINDOWS`; `window_cut_rule (k*len)//4` ↔ `window_cuts`; `separator " "` ↔
`SEP`; `l_max_chars 12000` ↔ `L_MAX`; `identity_causes` ↔ `DEGEN_*` branches; `views.names` ↔
`VIEW_NAMES`; `topk 20` ↔ `TOPK`; `weights [20..1]` ↔ `M._rank_weights(20)`; `vote v≥0` ↔
`(votes >= 0)`; `tie_rule lower bank index` ↔ `lexsort((idx, -best))` with `fit` ascending;
`seeds [0,1,2]` ↔ `SEEDS`; `arms` ↔ `build_arms` (all nine, memberships exact); `argmin/argmax
ties → lowest index` ↔ `argmin_window`/`argmax_window`; `bootstrap B=10000 seed 20260730` ↔
`BOOTSTRAP_B`/`BOOTSTRAP_SEED`; `alpha 0.05` ↔ `ALPHA`; `holm_family` ↔ the four `pv` keys;
`ARENA2 [maj+0.02, 0.98]` ↔ `ARENA2_MARGIN`/`ARENA2_CEILING`; `GATE_EXT median cos ≥ 0.99` ↔
`EXT_PARITY_MEDIAN_COS_MIN`; `VIEW_SUPPORT ≥ 0.60` ↔ `VIEW_SUPPORT_MIN`; `TEST_PATH` tokens ↔
`guard_path`; `krr gamma=1/d, ridge=1, target log1p(len)` ↔ `krr_length_probe`; `orbit_radius`
median over items and non-native views, strict OOF ↔ `rad_oof` stack median; `membership` ↔
`mean_top20_overlap_with_native`; `det1_threads 8` ↔ `det1_assert("8")` + wrapper exports;
`extraction resources 8/1/64G`, `a0 8/0/32G`, `time null` ↔ both sbatch headers;
`extract_namespace` / `a0_namespace` / `result_file` / `decision_file` / `atomic_json` /
`no_clobber` ↔ the wrappers and `main()`.

Dataset wiring verified against ground truth: `mechnov_pairverify.DATASETS` gives
`hatemm → HateMM / Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` and
`zh → MHC_zh / Qwen2.5-VL-7B-Instruct-LoRA_HF`; both banked `train_*.pt` caches, both LoRA
adapter directories, both P3 `train_segscoreK4_qwen.jsonl` files and all six GATE-FID floor
trainlogs (13241 × 3 seeds, 13150 × 3 seeds, each with a parseable `Val_Retrieval Epoch 29`
line) exist on disk. `build_text_prompt` is byte-identical to the deployed assembly at
`generate_VideoMLLM_embedding_lora_HF.py:438-442` under the deployed English defaults
(`TEXT_INSTRUCTION` imported from `BASE`; `"Title: "`, `"Transcript: "`, `"(none)"` match the
argparse defaults used by `gen_embed_lora.sbatch`, which passes no prompt overrides). Neither
`data/gt/HateMM/train.jsonl` nor `data/gt/MHC_zh/train.jsonl` carries a `title` key, so both
datasets take the `(none)` branch identically in the banked and the new extraction.

---

## 4. Verdict

REVISE (0C/4H/20I)

`GO` is not available: **H1** breaks a control the design itself calls load-bearing, **H2** is a
false measured statement inside a hash-frozen preregistration, **H3** discharges a registry
condition with a tautology, and **H4** is the interpretation claim on which the KILL branch's
consequence rests. None of the four requires re-architecting the design; H1 is a few lines
inside the fold loop, H2 and H4 are text plus a recount, H3 needs one substantive statistic in
place of an identity.

---

## 5. What I did and did not execute

**Executed (read-only, login node, no compute):** `sha256sum` on the eight frozen artifacts and
the five imported modules; `ls`, `wc -l`, `grep`, `sed`, `head`, `tail`, `cut`, `awk`, `sort`,
`uniq` and `jq` over repository text files; full reads of `CLAUDE.md`, `AGENTS.md`,
`TARGET_STATE.json` (the amendment, `unified_pilot_gate`, `hard_constraints`,
`serial_execution`, `candidate_registry`), `refine-logs/C02_A0_REVIEW_REQUEST.md`,
`refine-logs/C02_A0_RECORD.md`, `refine-logs/C02_DESIGN_REVIEW.md`,
`refine-logs/C02_EXPERIMENT_PLAN.md`, `refine-logs/C01_ZERO_CONTRACT_PROBE.md`,
`refine-logs/HEADSPACE_TRANSFER_PREGATE.md` (§0-§3, §4.1-§4.5), all seven frozen artifacts,
`generate_VideoMLLM_embedding_lora_HF.py`, `headspace_mint.py`, `headspace_fidelity.py`,
`mechfix_ops.py`, `mechnov_pairverify.py` (dataset table), `src/model/classifier.py:108-150`,
`scripts/slurm/gen_embed_lora.sbatch`, `scripts/prep_mhc.py:70-80`, and the plain-text SLURM
logs `lora_embed_13302.out`, `lora_embed_13329.out` and the six `13241`/`13150` trainlogs.
Structural queries over `data/gt/{HateMM,MHC_zh}/{train,val}.jsonl` with `jq` (key sets,
`.text|length` distributions, whitespace-only counts, the row index of `hate_video_95`) and over
`data/MLLM_scores/HateMM/train_segscoreK4_qwen.jsonl` (schema, line count). Two intermediate
length lists were written to the session scratchpad only.

**NOT executed:** no Python was run — not `py_compile`, not the view module's `self_test`, not
the §9 synthetic dry run, not any of the reviewed scripts. No `.pt` cache, `.npz` mint or model
was loaded. No GPU, no Modal, no teacher call. **No `test_seen` cache, no `test.jsonl`, no
`test_seen_segscoreK4_qwen.jsonl` and no test label was opened or read at any point.** No SLURM
job was submitted, released, held, cancelled or inspected — `sbatch`, `squeue`, `sacct` and
`scontrol` were never invoked; the two job logs I read are ordinary text files on disk. No file
under review was modified; the only file this review writes is this one. The §9 defect
repairs and the §11 preparation-time executions are the implementer's claims and are reported,
not verified.
