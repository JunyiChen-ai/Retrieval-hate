# C02 A0 v9 — scoped re-review of the ninth freeze (round 9)

**Reviewer:** fresh, independent, no exposure to the implementer's reasoning.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed against project data. See §8.
**Scope:** the five items A–E in the v9 request.

**Verdict:** `GO (0C/0H/0I)`

---

## 0. Hashes, v8 absence, namespaces

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v9.json` | `62b38cd9…55e4d729` | `62b38cd95cf9cd4035ae2efac67123560d2cbb2d3889134c65a4083155e4d729` | **MATCH** |
| `src/utils/c02_density_views.py` | `44fbb00b…7def3a52` | `44fbb00bf88ed1cbe7df2346d0961a172e8cfadd202af49d8b75f8ad7def3a52` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `bb698ab9…327fc86` | `bb698ab97b4d58f1c7af16cdb2053e45891785cd4be97b21fa187ea13327fc86` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `c50eed38…33048d4c` | `c50eed383b8ecb1829d13ffc5f298b427453dfd06c6371e274f0b28933048d4c` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `e6430b76…55b93a1b` | `e6430b76b7ccdd831ddb9939500aa24ea70d9662b62b955a2a11273a3b00ac1b` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v9.py` | `8cdaf1d3…0a526d8f93` | `8cdaf1d3dc1e72ae2f4073a42b97ab9d88791a4df6ef920f1473080a526d8f93` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v9.sbatch` | `f91875d8…5f03807c` | `f91875d89c61926fa28aedbd191c1b88bf80c1a1db0e9000003a5feb5f03807c` | **MATCH** |
| `refine-logs/C02_A0_V9_RECORD.md` | `20da7533…d9009b8` | `20da75334f7f3522f549d52a40327750c444b12799bd62cab40eec75df9009b8` | **MATCH** |

**v8 executables ABSENT** (`configs/c02/c02_a0_v8.json`, `scripts/analysis/c02_a0_arena_v8.py`,
`scripts/slurm/c02_a0_cpu_v8.sbatch` all `No such file`). `configs/c02/` holds exactly one
config; `scripts/analysis/` exactly one C02 arena. **`artifacts/c02_edq` does not exist**;
`find` for `*c02den*` over the repo returns nothing. The extractor's start-up trap
(`generate_c02_density_view_text_embedding_HF.py:64`) pins
`FROZEN_VIEWS_SHA256 = 44fbb00b…`, which equals the v9 view module's recomputed hash. The
four wrapper-pinned frozen modules (`c02_a0_cpu_v9.sbatch:76-79`) were re-hashed this pass
and all four still match.

---

## A. Change 1 — the reworded complexity claim — **PASS**

**The new wording is TRUE.** `c02_a0_arena_v9.py:207-212` now reads that `k = topk`
"returns an `(nq x topk)` result and selects with a `topk`-sized heap instead of
materialising all `n_bank` similarities per view pair (**the SCAN is O(n_bank) either way**
— a flat inner-product index computes every inner product regardless of `k`, which bounds
only the heap and the result width, and an earlier wording that claimed an `O(topk)` search
cost is corrected here)". That is exactly right for `faiss.IndexFlatIP`
(`arena_v9.py:225-228`): `ix.search(qv, topk)` computes all `nb` inner products per query
and `k` governs only selection and the returned `(nq × k)` `D`/`I` arrays. The retraction of
the previous wording is explicit and in-place, which is the correct form for an erratum.
`configs/c02/c02_a0_v9.json:198` (`oracle.search_width_erratum`) carries the identical
corrected clause. No other file asserts the old form: `grep` for `O(topk)` across the frozen
set returns only these two corrected passages.

**The two other reasons are untouched and still correct.**

| reason | verdict |
|---|---|
| "provably sufficient for the top-topk (above)" | **Correct.** The exactness argument at `arena_v9.py:187-197` is byte-unchanged in substance: if `tau` is the topk-th largest per-item maximum, any row with `s >= tau` forces `m_j >= tau`, so each `(a,b)` pair contributes at most topk such rows and its own top-topk already contains them. Re-derived independently this pass. The tie side-condition (≥topk exact float32 ties at `tau`) is still stated adjacent, at `:192-196` and `config:197`. |
| "for a singleton orbit it is LITERALLY the deployed call rather than merely equal to it" | **Correct.** `orbit_vote` with one bank view and one query view issues `IndexFlatIP(d).add(bv); ix.search(qv, 20)` on `_norm32` private copies — the same call `mechfix_ops._flat_ip` makes inside `M.deployed_vote`. This remains the load-bearing reason and is the right one to lean on. |

The `_norm32` always-copy rule (`arena_v9.py:167-178`) and its stated justification are
unchanged, and the mechanism it names is real: `mechfix_ops._norm32:37-42` does
`np.ascontiguousarray(np.asarray(X, dtype="float32"))`, which returns the *same object* for
an already-float32 C-contiguous input, and `faiss.normalize_L2` is in-place.

**Net line accounting.** `orbit_vote`'s docstring grew by exactly 3 lines and **every**
anchor the round-8 review cited downstream shifted by exactly +3: `g.size >= 2` drop rule
277→**280**, banked tiny-row check 732→**735**, manifest sha comparison 801→**804**,
`degen_mask` union 820→**823**, parity-cell count 961→**964**, secondary-arena
`try/except Halt` 1038→**1041**, `SystemExit(3)` 1185→**1188**. The constants block is still
at `:90-110` (zero shift above the docstring). This is independent structural evidence that
the arena changed **only** inside that docstring, plus the v8→v9 identifier renames, which
are length-preserving.

---

## B. Change 2 — the in-job GPU-budget guard — **PASS**

### B.1 Correctness

**One deadline, computed once, passed to both invocations.** `c02_density_extract.sbatch:48-51`
computes `BUDGET_DEADLINE = JOB_START_EPOCH + 14400 - 600` a single time and passes the same
`"$BUDGET_DEADLINE"` to the HateMM invocation (`:66`) and the MHC_zh invocation (`:77`). It is
an **absolute** epoch, not a per-dataset duration, so the second dataset inherits exactly what
the first left. Correct.

**The `SLURM_JOB_START_TIME` fallback is sound.** `JOB_START_EPOCH=${SLURM_JOB_START_TIME:-$(date +%s)}`
(`:50`). `SLURM_JOB_START_TIME` is a Slurm output environment variable holding the job's start
as a Unix timestamp, which is the same basis `sacct Elapsed` reports — so the guard measures the
quantity amendment condition (f) audits. The `${VAR:-default}` form is safe under `set -u`
whether the variable is unset or empty. When it falls back, the anchor is the wrapper's own
`date +%s`, taken *after* `conda activate` and `disk_guard.sh`, i.e. **later** than true job
start, so the deadline is shifted later by the preamble duration. That preamble is bounded:
`disk_guard.sh` determines usage by parsing `quota -s` (`disk_guard.sh:139`), not by walking
the tree (`du -sb` appears only inside the destructive path, gated on the threshold), so it is
a seconds-scale operation. The 600 s margin dominates it, as the record states.

**The shell parses and exit 5 propagates.** `set -euo pipefail` at `:25`. Line 50 uses the
`set -u`-safe default form; line 51's arithmetic expansion references three variables all
assigned above; line 53's `$(date -d @… || echo n/a)` cannot fail. Each python invocation is a
simple command, so under `set -e` a non-zero status aborts the script **with that status** —
`raise SystemExit(BUDGET_EXIT_CODE)` (`extractor:419`) makes the process exit 5, the wrapper
exits 5, and Slurm records 5. A HateMM breach therefore also prevents the MHC_zh invocation
from starting, which is the desired fail-closed behaviour.

### B.2 Inertness — the load-bearing property — **HOLDS**

`grep` confirms the guard has **exactly two call sites**, and both are strictly before any
work:

* `extractor:214` — `budget_check(deadline, 0.0, "before model load")`, placed *after* the
  cheap no-clobber path validation (`:196-212`) and *before* the 7B load (`:218-231`);
* `extractor:275` — top of the per-item loop, **before** `V.build_views`, before
  `load_video_frames`, before any `BASE._encode`.

Every write happens outside that window:

| write | line | when |
|---|---|---|
| view caches (`torch.save`) | `:330` | after the item loop for a whole split completes |
| manifest (`tmp` → `os.replace`) | `:365-368` | after **all** splits complete |
| breach record | `:399-401` | only on the exception path |

There is no `budget_check` between `:322` and `:368`, so **the guard cannot fire during a
write**. It cannot alter, truncate, reorder or partially write any computed result: it raises
before an item is touched, `BudgetExceeded` propagates uncaught out of `main()` (no
intervening `except`), and is handled only at `:411` to publish and exit.

**Worst partial state, traced.** The finest granularity at which anything is written is
*split*. If the breach lands in HateMM's `val` split after `train` completed, HateMM's six
`train_*-c02den-*.pt` files exist **but no manifest does** (the manifest is written only at
`:365-368`, after both splits). The A0 then fails closed: `arena_v9.py:790-793` opens
`artifacts/c02_edq/v1/extract/C02-DEN-v1/manifest_<ds>.json` unconditionally, and its absence
raises `FileNotFoundError` — not a `Halt` — which is **not** caught by `main()`'s
`except Halt` (`:1108`), so the arena dies before `C02_A0_OUT.json` or `C02_A0_DECISION.json`
is written at all. No result, no decision, no truncated bank read. A completed dataset is
intact, and a resubmission is structurally blocked by the pre-model no-clobber sweep
(`extractor:206-207`), which refuses in seconds. Correct in every branch.

*Precision note, not a defect:* §1.2 of the record says a breach leaves the in-progress
dataset "entirely unwritten". At *split* granularity that is exact; if a dataset's first
split had completed, that split's six caches are on disk. The code's own breach note
(`extractor:390-392`) states this more precisely ("no partial view cache or manifest was
written for the in-progress **split**") and the breach record *enumerates* the caches that do
exist (`:386-389`), so the artifact set is self-disclosing. The consequence is nil, because
the manifest is the A0's hard dependency and no-clobber blocks a silent retry.

### B.3 Spurious firing / wasted allocation — **no**

Deadline is `start + 13800 s` (3 h 50 m) against a projected spend of ~1.5–2.5 h
(5400–9000 s), i.e. **1.3–2.3 h of slack**. The `:214` check needs 60 s and sees ~13800 s, so
it can never fire at job start and cannot waste an allocation up front. The per-item rule
`max(2 × slowest so far, 60 s)` is measured against per-item costs of ~5 s mean (≈8758
forwards over 1508 items) and a plausible worst item in the tens of seconds; even a 90 s item
requires only 180 s of headroom. `max_item_s` resets per split (`:269`), which only relaxes
the requirement to the 60 s floor at the start of the short second split — harmless at ~5 s
items. A fire therefore signals a genuine ~2× overrun of projection, not noise.

The residual tail is correctly absorbed rather than ignored: work not covered by a
sufficient-headroom check is (i) the wrapper preamble in the fallback-anchor case, (ii)
python import + 7B load + LoRA merge at the start of each invocation (gated at only 60 s by
`:214`), and (iii) the one item in flight when the deadline passes. Realistically ~200–500 s
combined, against the 600 s margin — and this only becomes relevant at all if the run is
already ~1.5 h past projection. The margin is doing exactly the job it is declared to do.

### B.4 Scientific inertness — **touches nothing**

The guard adds no arm, no threshold, no metric, no decision term. `max_item_s` is used only
by `budget_check` and by two **accounting** manifest fields (`slowest_item_seconds`,
`budget_seconds_remaining_at_split_end`, `:342-344`). The arena reads only
`man["splits"]["train"]["written"]`, `["per_item"]`, `["n_degenerate_items"]`,
`["zero_guard_videos"]` (`:797-834`) — no key was removed, no schema check rejects added keys,
and no timing field is compared to anything (`extract_manifest_sha256` at `:685` is reported,
never asserted). The feature tensors are untouched. Separately confirmed: the JSON config is
**not** a numerical input — the arena reads only `cfg["run_id"]` from it (`:1088,1120`, plus
hashing at `:1090`); the `cfg` at `:691` is `mechnov_pairverify.DATASETS`, a different object.

### B.5 Exit codes and the breach record

`BUDGET_EXIT_CODE = 5` (`extractor:76`) is distinct from **1** (generic), **3** (A0
fail-closed HALT, `arena_v9.py:1188` / `config:287`) and **4** (frozen-module mismatch,
`c02_a0_cpu_v9.sbatch:72`). Declared consistently at `RECORD:144` and `config:269-276`.

`BUDGET_BREACH_<dataset>.json` (`extractor:380-392`) contains run id, dataset, the guard
message (seconds remaining / required / slowest item), the deadline, the breach timestamp, a
`manifest_written` boolean, and the **basenames** of existing view caches. **No accuracy, no
metric, no feature, no label, no id, no gate value** — accounting only, as claimed. It is
written atomically via `tmp` + `os.replace`, its own failure is caught and cannot mask the
breach (`:403-405`), and it lands in the extract namespace declared at `config:288`, which the
arena never globs.

---

## C. Change 3 — the measured-claims register — **PASS (audited row by row)**

**Rows I re-derived from scratch this pass, without execution:**

* **Row 1** — `n = 744 / 107 / 579 / 78`. Reproduced by `wc -l`. **[V] correct.**
* **Row 2** — whitespace-only `text` = **39 / 9 / 0 / 0**. Reproduced by JSON-parsing the four
  gt files. **[V] correct.** Also confirms `text == ""` is **0** on all four splits, as
  `config:35` claims.
* **Row 3** — `len(T) > 12000` = **9 / 1 / 0 / 0**. Reproduced on Python-character length.
  **[V] correct.**
* **Row 4** — `max_chars` = **80731 / 12275 / 708 / 343**. Reproduced exactly by
  `len(json['text'])`. **[V] correct**, and the parenthetical explanation of round 2's
  80732/12276/710 as raw-character counts missing JSON escapes is consistent with what I
  measured.
* **Row 5** — full-identity **48 / 10 / 0 / 0**; text-only `view_support`
  **0.9355 / 0.9065 / 1.0 / 1.0**. Reproduced. **[V] correct.**
* **Row 6** — runtime `view_support ≈ 0.9341 = 1 − 49/744`. Arithmetic checks
  (48 text-degenerate + 1 zero-guard). Correctly split into a **[V]** arithmetic part and a
  **[U]** census dependency pointing at row 12. **Correct classification.**
* **Row 7** — banked `B_fid` **0.0093 / 0.0086**. Read directly out of
  `scripts/analysis/headspace_fidelity_OUT.json` and `…_zh_OUT.json`: both carry
  `"B_fid_abs_3seedmean"` equal to those values. **[V] correct.**
* **Row 8** — `fixed − broken = n·Δacc`. Algebraic identity; matches
  `arena_v9.py:917-925`. **[V] correct.**
* **Row 9** — a lone degenerate item's displacement is exactly zero. Verified bitwise from
  the extractor's one-forward-per-distinct-string rule (`:296-307`) and the zero-guard's
  `zero.clone()` into all six slots (`:289-291`). **[V] correct.**
* **Row 10** — `mechfix_ops._norm32` can alias. Verified by reading `:37-42`. **[V] correct.**
* **Row 11** — the `k = topk` exactness argument. Re-derived. **[V] correct.**
* **Rows 12, 13, 14** — correctly **[U]**: each needs a `.pt` load or execution. The stated
  inertness is verifiable and holds — `ZERO_CONTRACT` computes the masks and halts on
  mismatch (`arena_v9.py:745-755`), `VIEW_SUPPORT` computes the real fraction (`:826-838`),
  row 13 is decorative because `k = topk` rests on two non-empirical grounds, and the KRR
  probe is reported and never read by `dec` (`:1128-1165`). **Correct.**
* **Row 16** — deployed-CLI byte-identity, **[U]**, bounded by GATE-FID which runs first
  (`c02_a0_cpu_v9.sbatch:95-111`, before the arena at `:113`). **Correct.**
* **Rows 17, 18** — **[D]**. The sacct table is present verbatim in
  `TARGET_STATE.json::iteration_8_…amendment.budget_basis_sacct_measured_2026_07_30`
  (13468 = `02:00:08`); the C01 row-355 provenance is cited and the arena itself tags criteria
  1 and 4 `DOCUMENTARY_CITATION_NOT_COMPUTED` (`arena_v9.py:757-769`). **Correct — neither is
  claimed as reproduced.**

**Row 15 — tested hardest, as requested. The classification is correct and the new clause is
true.**

* The **forward count is [V], and I re-derived it independently**: HateMM train
  `696×6 + 48×1 = 4224`; HateMM val `97×6 + 10 = 592`; MHC-ZH train `579×6 = 3474`; MHC-ZH val
  `78×6 = 468`; total **8758**, against the config's "~8760" — using only the row-1/3/5 counts
  I verified above. Video decodes `744+107+579+78 = 1508`, matching exactly.
* The **wall clock is correctly still [U]** — it is an estimate, and the register does not
  upgrade it to [V] on the strength of the guard. That is the honest call.
* The added clause "**now bounded by the in-job guard of §1.2**" is **true**: §B above traces
  the ceiling to `start + 13800 s + (one unit of in-flight work)`, where the residual is
  covered by the 600 s margin. This is a genuine change of kind — v8 had *no* bound at all,
  and condition (f) voids an over-budget run — so calling it "bounded" is accurate, and the
  claim is deliberately not overstated as "cannot exceed".
* The concluding sentence "15 — previously the largest downside — is now enforced in-job" is
  therefore supported.

**Nothing classified [V] that I could not re-derive.** All eleven [V] rows above were
re-derived or re-read this pass. **Nothing classified [U] that I could.** Rows 12, 13, 14 each
genuinely require a `.pt` load or a Python run, both of which are forbidden to me; row 15's
[U] half is a wall-clock estimate that by definition cannot be re-derived statically, and its
[V] half I did re-derive.

**Completeness — one omission, disclosed and verified true (see §F.2).** `config:204` asserts
as measured that `headspace_fidelity.py` "also emits its own `raw_effect_under_test: 0.0255`
and `STOP_RULE_TRIGGERED` fields". That is an empirical claim about a banked artifact and is
not a register row. I checked it: **both** `headspace_fidelity_OUT.json` and
`headspace_fidelity_zh_OUT.json` contain `"raw_effect_under_test": 0.0255` and
`"STOP_RULE_TRIGGERED": false`. The claim is **true**, and by the design's own statement those
fields belong to F105 and are not consumed — only `B_fid_abs_3seedmean` is read
(`c02_a0_cpu_v9.sbatch:106`). It would be a [V] row 19. I did not count this as a finding; see
§F for the reasoning.

---

## D. Constants sweep — **CLEAN**

**Arena constants (`arena_v9.py:90-110`) — every value identical to v8, and each still matches
its config counterpart:** `TOPK 20` (`config:171`), `BAR_ACC / BAR_MF1 0.050`,
`BAR_NETFIX_RATE 0.030`, `VIEW_SUPPORT_MIN 0.60` (`config:207`), `BOOTSTRAP_B 10000`,
`BOOTSTRAP_SEED / NOISE_SEED / SHUFFLE_SEED` all `20260730` (`config:223-227`), `ALPHA 0.05`,
`ARENA2_MARGIN 0.02` / `ARENA2_CEILING 0.98` (`config:203`),
`EXT_PARITY_MEDIAN_COS_MIN 0.99` (`config:205`), `TINY_NORM 1e-12` (`config:206`),
`KRR_RIDGE 1.0` (`config:232`), and the same nine `ARM_NAMES` in the same order
(`config:116-158`).

**Decision rule — identical on both sides.** `arena_v9.py:1158-1165` requires, per dataset,
`delta_acc >= 0.050 AND delta_mf1 >= 0.050 AND net_fix_rate >= 0.030 AND` beats SHUFFLE and
NOISE in both metrics `AND` both bootstrap lower bounds `> 0 AND` both Holm rejections. That
is `config:240` verbatim. Holm family = `hatemm_acc, hatemm_mf1, zh_acc, zh_mf1`
(`arena_v9.py:1148-1149` vs `config:215-220`). `alpha = 0.05` both sides.

**View module vs config:** `VIEW_NAMES`, `K_WINDOWS = 4`, `SEP = " "`, `L_MAX = 12000`
(`c02_density_views.py:68-72`) all match `config:17-33`. The cut rule
`c_k = (k*len(T))//4` (`window_cuts:83`) and `RWk = T[:c_k] + " " + T[c_{k-1}:c_k] + T[c_k:]`
(`:130`) match `config:27,31` exactly; `RFULL = T + " " + T` (`:116`) matches `config:30`.

**Budget constants agree in all three places:** `14400 / 600` (`sbatch:48-49`),
`2.0 / 60.0 / 5` (`extractor:76-78`), and `config:269-276` lists all five identically.

**Self-test case lists unchanged.** The arena's `oracle_self_test` still has the same **six**
cases in the same order with the same names — `parity_singleton_bit_exact` (`:584`),
`zero_query_tie_invariant` (`:595`), `multiview_topk_exact` (`:617`),
`within_partition_derangement` (`:626`), `degeneracy_matched_groups` (`:642`),
`shuffle_donates_displacement_via_build_arms` (`:671`) — each exactly +3 from its v8 line, as
§A predicts. The view module's `self_test` still has its seven cases
(`ordinary_english`, `cjk`, `empty_text`, `length_guard`, `short_text_empty_window`,
`deletion_rejected`, `selectors`).

**Every earlier hardening intact:** `g.size >= 2` drop rule (`:280`), banked tiny-row check
(`:735`), manifest sha comparison (`:804`), `degen_mask` union (`:823`), parity-cell count
assertion (`:964`), secondary-arena `try/except Halt` (`:1041`), `SystemExit(3)` (`:1188`),
Sattolo derangement with the fail-closed `Halt` (`:308-321`), the always-copy `_norm32`
(`:176`), `VIEW_CACHE_INTEGRITY` sha re-check (`:798-806`), `if not __debug__` refusals in all
three python files.

**Nothing outside the three changes moved.** No threshold, bar, arm, seed, arena definition,
metric, gate or decision term differs from the v8 set as characterised by the round-8 review.

---

## E. Stale identifiers and regression sweep — **CLEAN**

**No stale v8 identifiers.** `grep` for `v8|V8` across all seven executables returns hits in
exactly one place: `config:363-371`, the `supersedes.v8` audit block, where it must appear.
`RUN_ID`, `CFG`, `--job-name`, the wrapper's self-test import (`c02_a0_cpu_v9.sbatch:59`) and
its arena invocation (`:113`), the output namespace (`:41`), and the result/decision schema
versions (`arena_v9.py:1087,1120` vs `config:283-284`) are all `v9` and mutually consistent.
All seven files carry the `C02_A0_V9_RECORD.md` pointer.

**Test-path isolation — intact and layered.** Extraction: splits restricted to `train`/`val`
by assertion (`extractor:190-192`), `assert_no_test_token` on every gt path, every one of the
12 planned output paths and the manifest (`:198-211`), plus a `torch.load` monkeypatch
(`:110-118`). Mint: `HM` installs its own guard, `load_view_text` re-asserts the token guard
**and** `split == "train"` (`c02_a0_mint.py:63-64`), and the patched loader replaces the
canonical one (`:146-149`). Arena: `guard_path` on the native cache, every view cache, the
manifest and every mint npz, plus its own `torch.load` guard (`:66-74`). The arena consumes
`train_*` only (`:696,710,797`). **Amendment condition (b) holds:** extraction covers
train + dev_seen, A0 reads train only, test is never named.

**View subsequence contract — intact.** `V.assert_subsequence(text, views[name])` is called
for all six views at `extractor:281-282`, *before* `load_video_frames` and therefore before
any GPU forward.

**PARITY-NAT — intact and unweakened.** `parity_native` (`:534-559`) still bit-checks
predictions **and** the sorted top-20 similarity vector on every row against
`M.deployed_vote`, exempting only neighbour IDs on tied rows, and the cell count is asserted
equal to `len(SEEDS) * P.K_FOLDS = 15` (`:964-966`). It never depended on the retracted claim;
it binds because a singleton orbit issues literally the deployed call.

**Zero contract — intact.** Exact zero-mask equality across the banked native and all six
views, with a `Halt` on any deviation (`:745-751`); no non-structural tiny rows on either the
banked arrays or any view (`:732-737, 752-755`); criteria 1 and 4 honestly tagged
`DOCUMENTARY_CITATION_NOT_COMPUTED`; structural nulls retained identically in every arm with a
separate sensitivity read (`:1009-1021`).

**Self-orbit exclusion — intact.** `np.intersect1d(ho, fit).size` asserted zero per fold
(`:879-880`), with bank drawn from `fit` and queries from `ho`.

**F113 confinement — intact.** The raw fused arena is built at `:1024-1046` and written only
into `res["secondary_raw_arena"]`. `raw_preds` never reaches `dec`, which reads exclusively
`summary_3seed` and `bootstrap_FULL_vs_NATIVE` (`:1128-1165`). Its `Halt`s are caught so it
cannot destroy a completed primary measurement. A raw arena can corroborate a KILL and can
never promote — condition (g) holds, and a PASS is rendered on the fold-head/deployed-head
path.

**SLURM hygiene — clean.** Neither sbatch sets `--time` (the only textual match is the comment
explaining why); no `--dependency`, `--array`, `--requeue`, `--singleton`, no `scontrol`, no
nested `sbatch`/`srun`. Extraction is 8 CPU / 1 A100 / 64 G, A0 is 8 CPU / 0 GPU / 32 G — both
at 8 CPUs, so the historical 16-CPU aggregate-cap wedge cannot occur. `conda activate
HateVideo` in both. The A0 wrapper exports DET-1 threads and `CUDA_VISIBLE_DEVICES=""` before
any python starts.

---

## F. Findings

| # | severity | file:line | statement |
|---|---|---|---|
| — | — | — | **No Critical, High or Info findings.** |

### F.1 Declared observation (not a finding) — stale cross-reference

`configs/c02/c02_a0_v9.json:267` — `serial_execution_interlock` says the condition-(e)
`squeue` check "and its result are recorded in `refine-logs/C02_A0_V5_RECORD.md`", a
superseded record. The operative instruction is `C02_A0_V9_RECORD.md:147` ("run the `squeue`
check for amendment condition (e). Recorded in §6") and the live pending field is
`C02_A0_V9_RECORD.md:163`. **Why this is not a finding:** the JSON config is not an operator
checklist and is not a numerical input — the arena reads only `cfg["run_id"]` from it; the
amendment does not require the squeue evidence to live in any particular file; the
operator-facing record is correct and complete; and the sentence is a superseded pointer, not
a false statement about the mechanism. It is unchanged, pre-existing state that v9's three
changes did not touch. Fold it in if a v10 ever happens for a real reason.

### F.2 Declared observation (not a finding) — one register omission, verified true

`configs/c02/c02_a0_v9.json:204` asserts `raw_effect_under_test: 0.0255` and
`STOP_RULE_TRIGGERED` as fields the frozen reader emits; the register's header claims to cover
"every empirical claim in the frozen set" and does not have this row. **I verified it this
pass and it is correct** in both banked JSONs, and the design explicitly excludes those fields
from its own rule. As a register row it would read `[V]`, load-bearing for nothing. Adding it
would change no number, gate, arm or verdict.

**On the calibration of both.** The round-8 Info was raised because a statement in the frozen
set was **false**. Neither item above is false: F.1 is a superseded pointer whose operative
counterpart is correct, and F.2 is a true, ungated, now-verified number missing from a
documentation table. Neither can change the verdict, corrupt a gated quantity, leak test data,
or kill a job after the GPU is spent. I record both with file:line so the reader can overrule
me, and I do not count either against the freeze.

---

## G. Verdict

```
GO (0C/0H/0I)
```

Eight hashes match; the v8 executables are gone; `artifacts/c02_edq` and every `*-c02den-*`
cache are absent. **Change 1** replaces a false complexity claim with a true one, marks the
correction in place in both the arena docstring and the config erratum, and leaves the two
over-determining reasons for `k = topk` intact and correct. **Change 2** is a genuinely inert
guard: one deadline computed at job start and shared by both invocations, a sound
`SLURM_JOB_START_TIME` fallback, clean shell semantics with exit 5 propagating through
`set -e`, exactly two call sites both strictly before any work and never between a computation
and its write, a breach that leaves no partial split and an A0 that fails closed on the
missing manifest, no plausible spurious fire against 1.3–2.3 h of slack, an accounting-only
breach record, and no contact with any scientific semantic. **Change 3**'s register classifies
all eighteen rows correctly — I independently re-derived every [V] row, including row 15's
forward count at 8758, and confirmed that each [U] genuinely requires execution and that row
15's "now bounded by the in-job guard" is true without being overstated. Constants, arms,
seeds, thresholds, self-tests and every v5–v8 hardening are unchanged, with the arena's uniform
+3 line shift giving independent structural proof that only the docstring moved. Test-path
isolation, the subsequence contract, PARITY-NAT, the zero contract, self-orbit exclusion, F113
confinement and SLURM hygiene are all intact.

Two operator preconditions remain unenforceable in code and are mandatory immediately before
`sbatch`: re-verify the eight sha256 values in §0, and run the `squeue` check for amendment
condition (e), recording it in `C02_A0_V9_RECORD.md` §6. Condition (f) is now enforced in-job
as well as audited post hoc, so the residual budget risk is a `sacct` reconciliation rather
than a live threat to the result.

---

## H. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `wc`, `sed`, `cut`, and file reads; plus read-only
JSON parsing of `data/gt/<DS>/{train,val}.jsonl` to re-derive register rows 1–5 (counts and
string lengths only — no cache, model, video or test path involved), and `grep` of two
dev-only GATE-FID gate outputs (`headspace_fidelity{,_zh}_OUT.json`) for
`B_fid_abs_3seedmean`, `raw_effect_under_test` and `STOP_RULE_TRIGGERED`, which contain no
test metric.

**Did not:** load or open any `.pt` cache, `.npz`, model, adapter or video; open any
`test_seen` cache, `test.jsonl`, or any test label or metric; run the arena, the mint, the
extractor or any module under review; run `squeue`, `sacct`, `sbatch` or any SLURM command;
touch a GPU or Modal; modify, move or delete any reviewed file; or write anything other than
this review.
