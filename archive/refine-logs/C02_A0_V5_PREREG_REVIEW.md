# C02 A0 v5 — delta review (round 5)

**Reviewer:** the round-4 reviewer, holding the v4 Info list in context. This is a **delta
review** of the v5 frozen set against `refine-logs/C02_A0_V4_PREREG_REVIEW.md`
(`GO (0C/0H/23I)`), not a fresh full review: the design, bars, arms, arena, gates and
decision rule were adjudicated in round 4 and are re-checked here only for regression.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed. See §5.

**Verdict:** `GO (0C/0H/4I)`

---

## 0. Hash and namespace verification

Every sha256 recomputed with `sha256sum`.

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v5.json` | `2d90f7bd…30aacf6` | `2d90f7bd455dfcd7743fe3b7a0d0f99ff60b7e5e7ace7f449491f73d930aacf6` | **MATCH** |
| `src/utils/c02_density_views.py` | `1db85ded…41e7227` | `1db85dedc1753ce23e4267f6cb872e1f010c3a1f7b95f987bfe9ad41341e7227` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `d6fd0191…23fd8884` | `d6fd01913f3be2a8a15c034096c95766c460da55a07c15dd38afd40823fd8884` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `5f9cf452…63c01b8a` | `5f9cf4525f24a50180f872a88a9e3e4a234b9d4727e069f34ce5461763c01b8a` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `360dc03f…407d430d5` | `360dc03f5301c2a85e65e4e13456f75183aec40cb3567f0435a758b407d430d5` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v5.py` | `4f5d9cff…5cb08edc` | `4f5d9cff2bd7829e97c60d3902340bbf37f14c73989e320279afdd795cb08edc` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v5.sbatch` | `d4c1783f…6b824239` | `d4c1783fabd7bce22ccd5c1ad144674db4dd47459f89de16b65ec6b66b824239` | **MATCH** |
| `refine-logs/C02_A0_V5_RECORD.md` | *(not declared)* | `62e19eb55d9a4f81e08a18d33269d94ca94e9c71595a44ca9e8b2094e6ba0f18` | **REPORTED** |

**v4 executables ABSENT:** `configs/c02/c02_a0_v4.json`,
`scripts/analysis/c02_a0_arena_v4.py`, `scripts/slurm/c02_a0_cpu_v4.sbatch` — `ls` returns
*No such file or directory* for all three, and all twelve v1-v4 executables are gone
(`configs/c02/` holds exactly one file; `scripts/analysis/` holds only `c02_a0_arena_v5.py`
and `c02_a0_mint.py`; `scripts/slurm/` only `c02_a0_cpu_v5.sbatch` and
`c02_density_extract.sbatch`). **`artifacts/c02_edq` does not exist**;
`find . -name '*c02den*'` returns nothing.

**The critical re-pin, checked because it would have killed the GPU job at start-up:**
`generate_c02_density_view_text_embedding_HF.py:64` pins
`FROZEN_VIEWS_SHA256 = 1db85ded…41e7227`, which **equals** the v5 view module's actual
hash. The extractor will not refuse to run. `FROZEN_BASE_SHA256:63` still equals the
deployed extractor's hash. `c02_a0_mint.py:56` still equals `headspace_mint.py`'s hash, and
`c02_a0_arena_v5.py:80-87` still equals all three frozen modules' hashes.

**Regression checks on the round-1..3 High findings:** the H1 phrases
(`upper-bounds what any representation`, `failure is therefore decisive`) are absent from
every v5 file; `EXCHANGEABLE BY CONSTRUCTION` appears only inside the explicit retraction
at `c02_a0_v5.json:162`. Sattolo (H2) is intact at `arena_v5.py:285-296`; the H3 scope
statement is intact at `:346-355`; the H4 `build_arms`-driven self-test case is intact at
`:626-653`. **No scientific constant moved**: I compared `arena_v5.py:90-110` line by line
against my round-4 read — `TOPK 20`, `BAR_ACC/BAR_MF1 0.050`, `BAR_NETFIX_RATE 0.030`,
`VIEW_SUPPORT_MIN 0.60`, `BOOTSTRAP_B 10000`, all three seeds `20260730`, `ALPHA 0.05`,
`ARENA2_MARGIN 0.02`, `ARENA2_CEILING 0.98`, `EXT_PARITY_MEDIAN_COS_MIN 0.99`,
`TINY_NORM 1e-12`, `KRR_RIDGE 1.0`, and the same nine `ARM_NAMES` — all unchanged. The
decision rule at `:1133-1140` is byte-equivalent to v4's. The record's claim that "no
threshold, bar, arm, metric, decision rule or scientific constant changed in v5" is
**verified**.

**Files claimed pointer-only** (`c02_a0_mint.py`, `c02_density_extract.sbatch`) are
byte-identical in size to their v4 versions (10 558 and 2 457 bytes) and their structural
scan reproduces v4's line numbers exactly (mint asserts at `:63,64,66,67,72,73,75,92-95,126`).
The extractor is still 316 lines with every function and guard at its v4 line number. The
view module grew 260 → 268 lines, entirely inside the degeneracy docstring; its code region
(`constants`, `window_cuts`, `build_views`, `assert_subsequence`, `random_window`,
`argmin/argmax_window`, `self_test`) is semantically identical, including
`L_MAX = 12000`, `SEP = " "`, `K_WINDOWS = 4` and the frozen
`RANDOM_WINDOW_SALT = b"C02-A0-v1/random-window/"` (correctly **not** bumped to v5 — it is a
salt, and changing it would silently redefine the `RANDOM_WINDOW_REPEAT` arm).

---

## 1. Status of every round-4 finding

19 CLOSED · 3 ACCEPTED-AS-DECLARED · 1 PARTIALLY CLOSED. All evidence is from the v5 code,
not from the record.

| # | status | evidence in v5 |
|---|---|---|
| I1 | **CLOSED** | `arena_v5.py:2` — docstring now names `c02_a0_arena_v5.py`. |
| I2 | **CLOSED** | `c02_a0_v5.json:11` — `authority.record` → `C02_A0_V5_RECORD.md`. All five executables cite the v5 record (`views:4`, `extractor:4`, `mint:5`, `arena:4`, `extract.sbatch:14`, `cpu_v5.sbatch:12`). |
| I3 | **CLOSED** | `c02_a0_v5.json:202` — now cites `scripts/slurm/c02_a0_cpu_v5.sbatch`, which exists and does enforce the bar at `:91-102`. |
| I4 | **CLOSED** | `cpu_v5.sbatch:63-70` — `FID_SHA` is compared to `sha256sum scripts/analysis/headspace_fidelity.py` and the job `exit 4`s on mismatch, **before** the reader is invoked at `:87`. I recomputed the reader's hash: it equals the pinned `72fd8e0a…`, so this will not fire spuriously. `RECORD §3:117` now attributes the row to "the wrapper, enforced (new in v5)" instead of the false "asserted at run time". Shell is correct: `GOT=$(sha256sum … \| cut -d" " -f1)` takes field 1 of `<hash>␣␣<path>`, and under `set -euo pipefail` a failing `sha256sum` aborts the job via the assignment's exit status. |
| I5 | **CLOSED** | `arena_v5.py:298-305` — the bare `assert` is replaced by `halt(HALT["ARENA"], …)` with a comment recording why the path is unreachable. A size-1 donor group now fails **closed through `Halt`**, so `main()` still writes both artifacts. The inner `assert` at `:295` remains, correctly: Sattolo makes it unfireable. |
| I6 | **CLOSED** | `arena_v5.py:801-805` — `degen_mask = degen_text \| degen_zero`. `zero_banked` is in scope (defined `:720`, used `:804`) and is the **right** mask: it is the banked-text zero set, the views' zero masks are asserted equal to it at `:731`, and img is constant across views, so a zero-text row is exactly a row whose six views share one head key. Effects, checked individually: **FULL** unaffected (`build_arms` never sees `degen_mask`); **NOISE** unaffected and already correct for these rows (`disp = 0 ⇒ noised = NAT`, `:365-369`); **metrics** affected only through SHUFFLE's predictions — plus one reporting consequence, N1 below. Two residuals recorded as N1/N2; the systematic defect is fixed. |
| I7 | **CLOSED** | `arena_v5.py:714-719` — criterion 3 is now applied to `banked_img` and `banked_text` before `zero_banked` is computed, halting via `HALT["ZERO"]`. Cannot fire spuriously: the deployed `_encode` L2-normalises every non-zero-guard row, so norms are ~1.0, and the test excludes exact zeros (`nrm0 > 0.0`), so the known structural zero at HateMM row 355 does not trip it. |
| I8 | **CLOSED** | `arena_v5.py:192-196` — the side condition is now stated **inline at the point of claim**, with the reachability argument (duplicate keys only; 1 on HateMM train, 0 on ZH; vote invariant there). `c02_a0_v5.json:196` adds `exactness_side_condition`. *Residual, non-blocking:* `c02_a0_v5.json:168` (`oracle.search`) still phrases the argument unconditionally; the qualification is the sibling key in the same `oracle` object, so a reader cannot encounter one without the other. |
| I9 | **CLOSED** | `arena_v5.py:521-531` and `c02_a0_v5.json:199` — both now state that tie order is **not** vote-invariant in general, give the exact perturbation `2·s·(w_r − w_{r+1})/Σw`, and locate the safety in the right place: predictions **and** sorted similarities are bit-checked on every row, so a tie-induced flip HALTs. |
| I10 | **CLOSED** | `c02_density_views.py:52-58` — "`L_MAX = 12000` is a **NEW constant chosen here, not an inherited one**"; the plan's §3.1 tokenizer-truncation clause is now named as "the closest precedent … but NOT the same criterion", matching what the config already said. |
| I11 | **CLOSED — and I resolved the dispute in v5's favour.** | `c02_a0_v5.json:76` defines `max_chars` as `len(json['text'])` in Python characters, which is exactly what `build_views` compares to `L_MAX` and what the manifest stores as `len_native`. I measured it independently with `wc -c`/`wc -m`/`grep -o`: HateMM train line 540 (`non_hate_video_533`) is 80 892 bytes / **80 784 characters**, of which 38 are the `{"id": …, "text": "` prefix and 14 the `", "label": 0}` suffix ⇒ **80 732 raw characters**, and it contains **exactly one** escape (`\"`) ⇒ decoded `len(text)` = **80 731** ✓. HateMM val line 56: 12 327 chars − 37 − 14 = 12 276 raw, one escape ⇒ **12 275** ✓. MHC-ZH train line 436: 756 chars − 32 − 14 = 710 raw, **two** escapes ⇒ **708** ✓. Round 2's 80732/12276/710 were the **raw-character** counts; they did not subtract the JSON escape sequences. The byte↔char gap (80 892 − 80 784 = 108 = 36 astral characters × 3) reproduces round 2's astral count exactly, which is why the two methods agreed everywhere except on the escapes. **v5's numbers are the correct ones**, and they are the quantity the guard uses. |
| I12 | **CLOSED** | `arena_v5.py:1005-1021` — the secondary arena runs in `try/except Halt`; the except path records `{"status": "SECONDARY_ARENA_HALTED_PRIMARY_UNAFFECTED", "halt": …}`, calls `_final_diagnostics` and returns. See §2 for the two questions asked about it: the early return **is** complete for the primary read, and it **cannot** mask a primary-relevant failure. |
| I13 | **CLOSED** | `cpu_v5.sbatch:52-61` — the self-test now runs in its own python process at the top of the job, before the 36 mints and before GATE-FID. The heredoc imports the arena as a module, so `main()` does not run (`__name__ != "__main__"`), and an assertion failure exits 1 and aborts under `set -e`. The self-test is consequently run twice (wrapper + `arena:1049`); it is pure and cheap, so this is defence in depth, not a defect. |
| I14 | **CLOSED** | `arena_v5.py:776-788` — for every view the manifest's recorded `path` must equal the path opened (compared through `os.path.abspath`) and its recorded `sha256` must equal the file's hash now. `c02_a0_v5.json:208` adds the `VIEW_CACHE_INTEGRITY` gate. See §2 for placement and spurious-fire analysis: it is sound. |
| I15 | **CLOSED** | `arena_v5.py:1160-1163` — `raise SystemExit(3)` after a HALT. **Both artifacts are already written**: result at `:1089-1092` (`os.replace` of the tmp file), decision at `:1155-1158`, print at `:1159`, exit at `:1163`. `c02_a0_v5.json:276` records `halt_exit_code: 3`. Under the wrapper's `set -e` the job is correctly reported as failed. |
| I16 | **CLOSED** | `arena_v5.py:831,876,939-951` — `parity_cells` is incremented per checked cell, asserted equal to `len(SEEDS)*K_FOLDS` through `Halt`, and both the checked and expected counts are emitted. The remaining prose is now explicitly conditioned on reachability rather than asserting a result. |
| I17 | **ACCEPTED-AS-DECLARED** | `RECORD §2 "Accepted, not changed"`. I spot-verified the repository-wide claim with `git ls-files --error-unmatch`: `c01_policy_contrast_a0_v4.py`, `c01_zero_contract_probe.py` and `c04_a0t_small_v1_v6_producer.py` are all **untracked**, while the inherited frozen modules `headspace_mint.py` and `mechfix_ops.py` **are** tracked — consistent with the claim as scoped ("every iteration-8 artifact"). Committing is outside this task's authority, and the mitigation (full sha256 chain in §3, the extractor's run-time view-module pin, the arena's self-hash into the artifact) is real. Accepting is the correct call; the operator precondition from my round-4 §5 stands: **re-verify the seven hashes immediately before `sbatch`**, because the tree is written by concurrent agents. |
| I18 | **ACCEPTED-AS-DECLARED** | `c02_a0_v5.json:264` + `RECORD §2`. Amendment condition (f) makes the cap a post-hoc `sacct` measurement; an overrun voids the result. The projected 1.5-2.5 GPU-h against a 4.0 cap matches my own round-4 arithmetic. |
| I19 | **ACCEPTED-AS-DECLARED** | `c02_a0_v5.json:265` + `RECORD §5`, which is now a table to be completed at submission time (`squeue -u jehc223`, "other candidate GPU/teacher pilot running"). No automated interlock exists and none is possible in-artifact; recording the check and its result is the correct resolution. I did not run `squeue` and take no view on the live queue. |
| I20 | **CLOSED** | `c02_a0_v5.json:221-226` — `seeds_frozen` lists all three seeds and records that the SHUFFLE permutation is head-seed-independent by construction, with the reason. Matches `arena_v5.py:98-100` and the `derangement_within(…, SHUFFLE_SEED, f)` call sites. |
| I21 | **CLOSED** | `arena_v5.py:371-377` + `c02_a0_v5.json:164` — the head-space-linearity assumption is now named, with the correct direction of the bias (both conjuncts *easier*, so it can weaken a PASS but cannot manufacture a KILL) and a pointer to `orbit_radius_median_oof` as the quantity that bounds it post hoc. I re-derived the direction and it is right. |
| I22 | **CLOSED** | `c02_a0_v5.json:163` `shuffle_conjunct_uncovered_residue` — states the shared-component-plus-noise case, that the strict `>` degenerates to a coin flip there, that `FULL > NOISE` still holds because NOISE destroys the shared direction, the general deterministic-function-of-native-key case, and that the two length instruments are reported and deliberately **not** gated. Every clause is correct as I re-derived it, and the "bounds what a PASS may be read as, cannot affect a KILL" framing is the right one. |
| I23 | **PARTIALLY CLOSED** | `c02_density_views.py:42-51` is fixed and is now the best statement in the frozen set: both sub-cases are described, the whitespace case is named as "the case that ACTUALLY fires (39 of 744 HateMM train rows, 0 of 579 MHC-ZH)", and the real reason is given (repeating whitespace changes token count without changing evidence density). **But `c02_a0_v5.json:35` is unchanged from v4** and still gives only the `(none)`-flip rationale, which applies to `text == ""` — zero rows in the frozen gt — and not to any of the 39 rows that actually trigger the guard. The finding cited both files; one was corrected. Documentation-only; the behaviour was and remains correct. |

---

## 2. Did any v5 edit introduce a new defect?

The five areas the request names, each answered from the code.

**`_final_diagnostics` and the `try/except Halt` (`arena_v5.py:657-667`, `:1005-1021`).**
Sound. The early-return path is **complete for the primary read**: everything `main()`
consumes — `summary_3seed` (`:980`) and `bootstrap_FULL_vs_NATIVE` (`:981`) — is assigned
*before* the `try`, as are `gates.{ZERO_CONTRACT, GATE_EXT, VIEW_SUPPORT, ARENA2,
PARITY_NAT}` , `seeds`, `diagnostics.shuffle_control` (`:925`) and
`sensitivity_excluding_structural_nulls` (`:984-996`). `_final_diagnostics` is called on
**both** paths (`:1019` and `:1033`) with identical arguments, so the two exits produce the
same `diagnostics` content — which is exactly what factoring it out was for. It **cannot
mask a primary-relevant failure**: the only `Halt`s reachable inside the block are
`orbit_vote`'s `nb < topk` / faiss-id / non-finite checks, which concern the raw key space
and whose preconditions (`|fit| ≥ 460`, finiteness asserted at `:708-711`) are already
established, and `derangement_within`'s new fixed-point `Halt` — and that one is raised by
the **primary** loop first, at `:866`, over the same groups with the same
`(SHUFFLE_SEED, fold)`, so it can never reach the secondary catch. A non-`Halt` exception
still propagates, correctly. `raw_preds` is assigned inside the `try` and read only after
it (`:1022`), so no `NameError`. One cosmetic consequence: `secondary_raw_arena` now has two
shapes, so a downstream consumer must branch on `"status"`; the record documents this.

**`degen_mask = degen_text | degen_zero` (`:801-805`).** `zero_banked` is in scope and is
the correct mask (argued in the I6 row). Effect on FULL: none. On NOISE: none — those rows
already had `disp = 0`. On the metrics: only through SHUFFLE, which is the intended repair.
Two real second-order consequences, both Info:

* **N1 — the reported `view_support` will no longer equal the config's pre-measured
  value.** `n_ident` (`:807`) is now `|degen_text ∪ degen_zero|`. I checked the one known
  zero row: `data/gt/HateMM/train.jsonl` line 356 is `hate_video_95` — index **355**,
  matching `C01_ZERO_CONTRACT_PROBE.md` — and its `text` is a long ordinary transcript, so
  it is **not** text-degenerate. HateMM's runtime `view_support` will therefore be
  `1 − 49/744 = 0.9341`, not the `0.9355` in
  `c02_a0_v5.json:48` (`measured_identity_counts_2026_07_30`, explicitly a gt-only parse
  that cannot see decode failures). Both clear the 0.60 gate by a mile and the gate
  definition at `:205` ("orbit is NOT the full identity") is if anything *better* served by
  the new number, but a reader comparing artifact to config will find a one-item
  discrepancy with nothing in the config warning them. One sentence in the config closes it.
* **N2 — the merge rule can still hand a lone identity item a real displacement.**
  `shuffle_groups:257-264` merges a degeneracy class with fewer than 2 members into the
  other class of the same partition. On HateMM the class now has ~49 members and splits
  ~39/~10 across the two partitions, so no merge fires. On MHC-ZH the text-degenerate count
  is **0**, so if the extraction reports exactly one or two decode failures the degenerate
  class becomes a singleton in one or both partitions, is merged back into the
  non-degenerate class, and that item receives a real donated displacement under SHUFFLE
  while FULL leaves it untouched — the very asymmetry I6 removed, for ≤1 item per partition
  per fold. Bounded at ≤2/579 = 0.35 % of items, affecting only the non-binding conjunct,
  and visible in the reported `n_group_merges_over_all_seeds_and_folds`. The clean fix is to
  **drop** a residual singleton from `groups` rather than merge it: an uncovered index keeps
  `perm[i] = i`, which for a degenerate item is exactly the desired `SHUFFLE = NAT`, and it
  is not in `covered` so the `:298` fixed-point `Halt` does not fire.

**The manifest-vs-file sha256 comparison (`:776-788`).** Correct and cannot fire spuriously.
The extractor computes `sha256_of(path)` *after* `torch.save` (`extractor:276-280`), so the
recorded digest is of the final bytes. The path comparison works because the extractor
stores the `--EXP_FOLDER`-relative `./data/CLIP_Embedding/<DS>/…` and both jobs run with
CWD = `/data/jehc223/RGCL` (`extract.sbatch:27`, `arena_v5.py:54`), so both sides of
`os.path.abspath` resolve to the same absolute path. A missing `written` entry degrades to
`rec = {}` → `abspath("")` = the repo root ≠ the view path → `Halt` with a readable message,
i.e. fail-closed. *Placement note:* the check runs **after** the six view files are already
`torch.load`ed (`:689-706`) and after GATE-EXT (`:757-769`), so a stale cache would most
likely be reported as an extraction-parity failure rather than as a hash mismatch. That is
not a correctness problem — the halt still precedes every gated use — but hoisting the check
above the loads would make the halt reason match the cause.

**`raise SystemExit(3)` (`:1160-1163`).** Both artifacts are written and `os.replace`d
before it, and the decision print precedes it. `SystemExit` from `main()` propagates
through the `if __name__ == "__main__"` call with nothing catching it, so the interpreter
exits 3 and `set -e` fails the job. Correct.

**The wrapper's new blocks (`cpu_v5.sbatch:52-70`).** Shell is correct. All three heredocs
(`:46/50`, `:55/61`, `:91/102`) use quoted delimiters terminated at column 0, so nothing
expands and nothing mis-parses under `set -euo pipefail`. `import c02_a0_arena_v5 as A`
executes only module-level code — path inserts, `os.chdir(REPO)` (already the CWD), the
third-party imports, the `torch.load` guard install, and three side-effect-free project
imports — and **not** `main()`, so no data is opened and no frozen-module check is
triggered there. Exit codes are distinct and meaningful: 1 (self-test assertion), 4 (frozen
reader changed), 3 (arena HALT). One asymmetry worth a line:

* **N3 — `mechfix_ops.py` is the only frozen module whose hash is first verified at the very
  end of the job.** v5 correctly moved the self-test and the GATE-FID reader hash to the top,
  and `headspace_mint.py` + `mechnov_pairverify.py` are already verified inside **every**
  mint (`c02_a0_mint.py:92-95`), i.e. within seconds of the job starting. But `mechfix_ops`
  — the module that defines the deployed vote the whole PARITY-NAT gate is built on — is
  checked only by `arena_v5.py:1045-1047`, after the 36 mints and GATE-FID, and as a bare
  `assert` that would produce no artifact. The wrapper already has the mechanism: the
  self-test heredoc has `A.FROZEN` in hand and could verify all three in two lines.

Beyond these, I re-checked the parts of the arena that v5 touched indirectly and found no
regression: the `+83`-line delta is fully accounted for by the twelve changes above; no
gated quantity, no arm, no seed and no threshold moved; `raw_keys` construction stays
outside the `try`; and memory, runtime and serialisation remain as analysed in round 4 (the
only new cost is one redundant sha256 pass over ~60 MB of view files, since `:785` and
`:666` both hash them).

---

## 3. What still stands from round 4

Everything in §3 of `C02_A0_V4_PREREG_REVIEW.md` — test-contact isolation, the bar and
decision rule, F113 confinement, self-orbit exclusion, PARITY-NAT, the zero contract, the
view contract and its pre-forward proof, prompt fidelity, the registry carry-overs, the
hard constraints, SLURM hygiene, runtime survivability, the statistics and the config↔code
constant audit — was re-checked for regression against the v5 bytes and holds unchanged.
The four round-3 High findings remain REPAIRED. The two adversarial answers stand: a KILL
from this design is sound as the gate verdict it is labelled, and a PASS is sound within its
declared boundary, with the residual now written into the frozen config at `:163-164`
instead of living only in a review file.

---

## 4. Verdict

```
GO (0C/0H/4I)
```

Zero Critical, zero High — as in round 4, and now with 19 of 23 Info findings closed **in
the code and the frozen statements**, 3 accepted as declared operating conditions with
reasoning I agree with, and 1 partially closed. The four remaining Info items are:

| # | item |
|---|---|
| I23 | `c02_a0_v5.json:35` still carries the `(none)`-flip rationale that applies to zero of the 39 rows that fire the guard; the view module is fixed. |
| N1 | `c02_a0_v5.json:48` pre-measures `view_support` 0.9355 from gt text alone, while the runtime gate will report ~0.9341 now that zero-guard rows count as identity orbits. |
| N2 | `shuffle_groups:257-264` can merge a lone identity item back into the non-degenerate donor class, undoing the I6 fix for ≤1 item per partition per fold — reachable on MHC-ZH iff the extraction reports 1-2 decode failures. |
| N3 | `mechfix_ops.py`'s frozen hash is first verified only by `arena_v5.py:1045-1047`, after the 36 mints, and as a bare `assert`. |

None can change the verdict, corrupt a gated quantity, touch test data, or kill either job:
I23 and N1 are documentation, N2 is bounded at ≤0.35 % of items in a non-binding conjunct
and is reported by the merge counter, and N3 is a detection-latency asymmetry on a module
whose hash I verified as correct today. **I do not recommend a v6 for them.** N2 is the only
one with a code consequence, and if it is ever fixed the change is one line in
`shuffle_groups` (drop a residual singleton instead of merging it); the other three are one
sentence each. If the registry's literal `GO (0C/0H/0I)` token is required as the execution
key under amendment condition (a), those four edits produce it — but on the merits this set
is ready to run as frozen.

Operator preconditions unchanged from round 4, because no code can enforce them: the
`squeue` check for amendment condition (e), and re-verifying the seven sha256 values
immediately before `sbatch`, since the frozen set is untracked and the tree has concurrent
writers.

---

## 5. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `sed`, `awk`, `cut`, `rev`, `sort`, `uniq`,
`wc`, `git ls-files`, and file reads. Specifically: recomputed all 7 v5 hashes plus the
record and the frozen-module/reader hashes; read `c02_a0_arena_v5.py` in full (both pages),
`c02_a0_cpu_v5.sbatch` in full, `c02_a0_v5.json` in full and `C02_A0_V5_RECORD.md` in full;
structurally re-scanned the view module, extractor, mint and extraction wrapper against my
round-4 reads; measured the three disputed `max_chars` values with `wc -c`/`wc -m` plus
escape and astral-character counts on `data/gt/{HateMM,MHC_zh}/{train,val}.jsonl`; located
`hate_video_95` at line 356 of the HateMM train gt; and spot-checked git tracking of C01/C04
iteration-8 artifacts.

**Did not:** run Python of any kind, import any module, load or open any `.pt` cache,
`.npz`, model, adapter or video; open any `test_seen` cache, `test.jsonl`, or any test label
or metric; run `squeue`, `sacct`, `sbatch`, `bash -n` or any SLURM command; touch a GPU or
Modal; modify, move or delete any reviewed file; or write anything other than this review.
