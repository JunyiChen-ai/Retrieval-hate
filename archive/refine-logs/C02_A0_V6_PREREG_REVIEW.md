# C02 A0 v6 — final delta review (round 6)

**Reviewer:** the round-4/round-5 reviewer, holding the I1-I23 and N1-N3 lists in context.
Delta review of the v6 frozen set against `refine-logs/C02_A0_V5_PREREG_REVIEW.md`
(`GO (0C/0H/4I)`), scoped to the four repairs plus a regression sweep.
**Date:** 2026-07-30 (Pacific/Auckland)
**Type:** read-only static review. Nothing was executed. See §5.

**Verdict:** `GO (0C/0H/3I)` — 0I was **not** reached. Details in §4; the three residuals
are one-sentence documentation edits and the code is correct in all three cases.

---

## 0. Hash and namespace verification

| path | declared | recomputed | verdict |
|---|---|---|---|
| `configs/c02/c02_a0_v6.json` | `8b0572f1…72340982` | `8b0572f1ee8626de540417613eb9d3dc2d6bf1db3d867b7ad4aebb9472340982` | **MATCH** |
| `src/utils/c02_density_views.py` | `3bf07830…84ea7746` | `3bf0783095a7aeaf978ef7c52fe59cac0d74256ee28f83c14d9de42c84ea7746` | **MATCH** |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `8c17bdb8…b93c8101` | `8c17bdb8e4660783618ac3f335cd98f43da14eba587324debb1e80b9b93c8101` | **MATCH** |
| `scripts/slurm/c02_density_extract.sbatch` | `51dc4d93…88ca8fdc4` | `51dc4d9314180f5d65282d57e8daad18d07c9854db3d627bb8a563088ca8fdc4` | **MATCH** |
| `scripts/analysis/c02_a0_mint.py` | `49abad31…fd7c65a2` | `49abad31349897026668b711df79028f51ed9fb628ccdf9e22095f55fd7c65a2` | **MATCH** |
| `scripts/analysis/c02_a0_arena_v6.py` | `d2f62adb…66784815` | `d2f62adbd06ec9f286220b26f656a6026e3deb336d4537b6f6d71ff266784815` | **MATCH** |
| `scripts/slurm/c02_a0_cpu_v6.sbatch` | `1de480c3…16b14acb` | `1de480c3bbfad52215fb781e676062eaf17e5a909a5f6a48f770f67616b14acb` | **MATCH** |
| `refine-logs/C02_A0_V6_RECORD.md` | `95d95c63…ce16607a` | `95d95c63abff081a84357f3b28a88ea0c988db33eadd64348e1af248ce16607a` | **MATCH** |

**v5 executables ABSENT** (`ls` → *No such file or directory*): `configs/c02/c02_a0_v5.json`,
`scripts/analysis/c02_a0_arena_v5.py`, `scripts/slurm/c02_a0_cpu_v5.sbatch`. The three
directories hold exactly one C02 A0 config, one arena, one wrapper. **`artifacts/c02_edq`
does not exist**; `find . -name '*c02den*'` returns nothing.

**The start-up trap, checked for the third time:**
`generate_c02_density_view_text_embedding_HF.py:64` pins
`FROZEN_VIEWS_SHA256 = 3bf07830…84ea7746`, which **equals** the v6 view module's hash. The
extraction job will not refuse. `FROZEN_BASE_SHA256:63` still equals the deployed
extractor's hash.

**Pointer-only files verified structurally:** `c02_a0_mint.py` is 10 558 bytes / 250 lines
and `c02_density_extract.sbatch` 2 457 bytes / 64 lines — byte-identical sizes to v4 and v5,
consistent with the equal-length `V5`→`V6` substitution. The view module is 11 067 bytes /
268 lines, again identical to v5, and its EMPTY_TEXT and LENGTH_GUARD docstrings are
unchanged from the v5 text I reviewed.

**Regression sweep.** `arena_v6.py:90-110`: every frozen constant is unchanged
(`TOPK 20`, `BAR_ACC/BAR_MF1 0.050`, `BAR_NETFIX_RATE 0.030`, `VIEW_SUPPORT_MIN 0.60`,
`BOOTSTRAP_B 10000`, all three seeds `20260730`, `ALPHA 0.05`, `ARENA2_MARGIN 0.02`,
`ARENA2_CEILING 0.98`, `EXT_PARITY_MEDIAN_COS_MIN 0.99`, `TINY_NORM 1e-12`,
`KRR_RIDGE 1.0`, the same nine `ARM_NAMES`). Every v5 hardening survives: banked-cache
tiny-row check `:719-722`, manifest sha comparison `:789`, parity-cell count `:946`,
`SystemExit(3)` `:1170`, `try/except Halt` `:1012-1028`, schema versions `c02_a0_result_v6`
/ `c02_a0_decision_v6`. The H1 phrases are absent and `EXCHANGEABLE BY CONSTRUCTION` appears
only inside the explicit retraction at `c02_a0_v6.json:163`. The Sattolo derangement,
the H3 scope statement and the `build_arms`-driven SHUFFLE self-test are intact.

---

## 1. Status of the four items

### I23 — **CLOSED.**

`c02_a0_v6.json:35` now reads, in substance and in numbers, what
`c02_density_views.py:42-51` says: both sub-cases are named; `T == ""` is identified as the
`(none)`-flip case and explicitly annotated "*the frozen gt contains ZERO rows with
T==''*"; whitespace-only `T` is named as "*the case that ACTUALLY fires (39 of 744 HateMM
train rows, 0 of 579 MHC-ZH)*" with the correct reason (repeating whitespace changes token
count without changing evidence density). I re-confirmed both counts independently with
`awk` on the frozen gt: **39** whitespace-only rows in `data/gt/HateMM/train.jsonl` and
**0** in `data/gt/MHC_zh/train.jsonl`. Config and module now agree; nothing is left that is
true only of a non-occurring case. **Does it match the view module? Yes.**

### N1 — **CLOSED. The arithmetic is right and the "which one the gate reads" claim is true of the code.**

`c02_a0_v6.json:76` (`scope_note`) states that the `measured_identity_counts` block is
text-derived from gt alone, that the runtime gate additionally counts zero-guard rows, and
gives the expected runtime values.

* **Arithmetic:** `1 − 49/744 = 0.9341398…` → **0.9341** ✓. The `49` is right: 48
  text-derived full-identity rows (39 whitespace-only + 9 over-length, both of which I
  re-measured) plus the one known zero-guard row. I re-verified that the zero row is *not*
  already in the 48: `hate_video_95` is line **356** of `data/gt/HateMM/train.jsonl`, i.e.
  index **355**, matching `C01_ZERO_CONTRACT_PROBE.md`, and its `text` is an ordinary
  5 762-byte transcript — neither whitespace-only nor over-length. MHC-ZH `1.0000` follows
  from 0 text-derived identities and 0 zero-guard rows.
* **Hedging:** both predictions are correctly conditioned — "*if the extraction reports the
  single known zero-guard row*" / "*if it reports none*" — so neither asserts an unverified
  cache fact. That is the right form for a prereg: the gate computes the value, the config
  predicts it.
* **"Which one the gate reads":** true of the v6 code. `arena_v6.py:804-811` builds
  `degen_mask = degen_text | degen_zero`, sets `n_ident = degen_mask.sum()` and
  `support = 1 − n_ident/n`; `:812-823` emits that `support` and halts on it. The config's
  pre-measured `0.9355` is consumed by nothing. The gate definition at `c02_a0_v6.json:206`
  ("orbit is NOT the full identity") is if anything better served by the runtime union,
  since a zero-guard row's orbit *is* the identity. No contradiction anywhere.

### N2 — **CLOSED** (with a residual on the recorded *reason*, F3 below).

`shuffle_groups` at `arena_v6.py:253-269` now appends a class group only when
`g.size >= 2` and counts `g.size == 1` as `dropped`. Checking each thing asked:

* **Is dropping the right direction?** Yes, and the *operative* claim — "dropping cannot
  make `FULL > SHUFFLE` easier; merging could" — is true. Merging (v5) handed a
  zero-displacement item a real donated displacement, perturbing SHUFFLE where FULL is
  untouched, which strictly favours the conjunct. Dropping cannot do that. See F3 for the
  one place the stated justification overshoots.
* **Is the dropped item genuinely excluded from `covered`, so the I5 Halt cannot fire on
  it?** **Yes.** `covered` at `:298` is `np.concatenate(groups)`, and `groups` is exactly
  the `out` list, which a size-1 group never enters. The dropped index therefore keeps
  `perm[i] = i` (`:280`) and is never tested at `:299`. Stronger: with v6's rule the guard
  is now **unreachable by construction** — every group in `out` has size ≥ 2 and Sattolo
  leaves no fixed point in any of them — which is a good state for a defensive check to be
  in, but see F2 about its wording.
* **The counter and its reported name:** `shuffle_singletons_dropped` is initialised at
  `:809`, accumulated at `:867-868`, and emitted at `:937` as
  `n_singleton_class_groups_dropped_head_arena`. The `_head_arena` suffix is accurate and
  well chosen: the secondary raw arena at `:1017-1018` calls `shuffle_groups(...)[0]` and
  discards its count, so the reported number really is head-arena-only.
* **Does the self-test assert the drop rather than a merge?** **Yes, and it would catch a
  regression.** `:621-626` marks a single degenerate index and asserts `ndrop1 == 1`
  ("a singleton class group must be dropped, not merged"), `sum(g.size) == 119`
  ("the dropped item must leave the donor pool") and `all(3 not in g)` ("must not be
  regrouped"). A reversion to merging would give `ndrop1 == 0` and `sum == 120` and fail the
  first two assertions immediately. The tie-free control case at `:613-620` still asserts
  `ndrop == 0` with all groups ≥ 2.

### N3 — **CLOSED.**

`c02_a0_cpu_v6.sbatch:67-79` verifies all four frozen imported modules before the 36 mints.

* **Shell correctness under `set -euo pipefail`:** correct, including the subtle part.
  `local got; got=$(sha256sum "$path" | cut -d" " -f1)` splits the declaration from the
  assignment — the idiom that matters here, because `local got=$(...)` would make `local`'s
  exit status mask the substitution's and `set -e` would *not* fire on a failed
  `sha256sum`. As written, the assignment carries the pipeline's status, `pipefail`
  propagates a `sha256sum` failure, and the job aborts. `cut -d" " -f1` correctly takes
  field 1 of `<hash>␣␣<path>`. The `[ … ]` test lives inside `if`, so a false result does
  not trip `set -e`. `exit 4` inside a function exits the shell, which is the intent.
  `set -u` is satisfied: `want`, `path`, `got` are all assigned before use, and the function
  is defined at `:67` before its first call at `:76`.
* **Are the four hashes right?** All four recomputed and **matching**:
  `headspace_fidelity.py` `72fd8e0a…`, `mechfix_ops.py` `635c1312…`,
  `mechnov_pairverify.py` `77b0defd…`, `headspace_mint.py` `cefdf8dc2f4a…`. They are also
  consistent with `arena_v6.py:80-87` (three of them) and `c02_a0_mint.py:56` — so no
  spurious `exit 4`, and the in-process asserts remain as a second line of defence.
* **The heredoc that follows:** `python - "$OUTDIR" <<'PY' … PY` at `:100-111` is unchanged
  from v5 and still well formed; all three heredocs (`:46/50`, `:55/61`, `:100/111`) use
  quoted delimiters terminated at column 0. Placing a shell *function definition* between
  the second heredoc and the loop introduces no parsing interaction.
* **Anything else regressed in the wrapper?** No. Diffing against v5 line by line: the
  SBATCH block, `set -euo pipefail`, `cd`, conda activation, the DET-1 exports,
  `CUDA_VISIBLE_DEVICES=""`, the mint loop with its `f{TAG}`/`ffull` naming, the GATE-FID
  reader invocation, the 0.050 stop-rule heredoc and the final arena call are byte-equivalent
  apart from `v5`→`v6` in `RUN_ID`, `CFG`, `--job-name`, the prereg comment and the two
  arena references (`:59` self-test import, `:113` invocation), which are mutually
  consistent. Still no `--time`, no `--gres`, no dependency/array/singleton/requeue/force/
  release, 8 CPU, one submission.

---

## 2. Did the four edits introduce a new defect?

**`shuffle_singletons_dropped` is defined before every use on every path, including the
secondary-arena early return.** It is initialised at `:809`, well before the seed loop; the
only write is `:868`; the only read is `:937`, inside the
`res["diagnostics"]["shuffle_control"]` assignment at `:928-945` — which executes **before**
the `try` at `:1012`. The early return at `:1028` therefore emits a `res` whose
`shuffle_control` block is already complete, and no path can reach a read of an unbound
name. I also re-walked the early return for completeness: `summary_3seed` (`:987`),
`bootstrap_FULL_vs_NATIVE` (`:988`), all five gate blocks, `seeds`, the sensitivity read and
`_final_diagnostics` are all present on both exits, and `main()` consumes only the first two.

Three new findings, all documentation, all created by the N2 edit itself:

| # | class | location | finding |
|---|---|---|---|
| **F1** | Info | `configs/c02/c02_a0_v6.json:196` | `oracle.self_test` still says the self-test proves "*that the degeneracy-matched grouping never straddles that boundary and **merges** a singleton class within its own partition*". The code does the **opposite**: `arena_v6.py:264-268` drops it, and the self-test at `:621-626` asserts the drop by name. The same config contradicts itself at `:153`, which correctly says "*A class group of exactly ONE member is DROPPED, not merged*". Nothing computes from this string — the artifact records `oracle_self_test_cases` (the returned case names) and `config_sha256`, not this prose — but it is a false statement about the design inside the hash-frozen prereg, and it is the one place a future auditor would look to learn what the self-test guarantees. |
| **F2** | Info | `scripts/analysis/c02_a0_arena_v6.py:300` and `:305` | The fixed-point guard's comment and its **halt message** still say the branch is reachable only if "*a donor group of size 1 survived shuffle_groups' **merge rule***". There is no merge rule in v6. Worse for the message than for the comment: `HALT_C02_A0_ARENA_DEGENERATE: derangement left a fixed point: a donor group of size 1 survived the merge rule` would be written into `C02_A0_OUT.json` and `C02_A0_DECISION.json` naming a mechanism that does not exist. Harmless in practice because the branch is now unreachable by construction (only groups of size ≥ 2 reach `out`; Sattolo leaves no fixed point in those; `covered` is exactly their concatenation), and keeping the check is correct defensive practice — it is the wording that is stale. |
| **F3** | Info | `scripts/analysis/c02_a0_arena_v6.py:257-263`; `configs/c02/c02_a0_v6.json:153`; `refine-logs/C02_A0_V6_RECORD.md:39` | The recorded justification — "*dropping leaves that single item carrying its OWN displacement in SHUFFLE … which makes `FULL > SHUFFLE` **HARDER** — the conservative direction*" — is literally true only in the branch that cannot occur. The reachable singleton is a **degenerate** class group (`dg.size == 1`, the MHC-ZH one-decode-failure case), and a degenerate item's own displacement is **exactly zero** by definition: for `degen_text` the extractor copies the NAT vector bit-exactly into every view slot, and for `degen_zero` all six view vectors are zero, so in both cases all six head keys coincide. Dropping it therefore gives `SHUFFLE_v(i) = NAT_i + (view_v(i) − NAT_i) = NAT_i =` FULL's own view — **exactly neutral, not "harder"**, and in fact the ideal matched behaviour. The "harder" reading holds only for a *non-degenerate* singleton (`nd.size == 1`), which needs a partition of exactly one item and is unreachable at `n ≥ 579`. The **decision is right and the operative claim ("dropping cannot make the conjunct easier, merging could") is true**; what overshoots is the claim of positive conservatism in the case that actually fires. Same species as the round-4 I23 finding, which is why I am not letting it pass silently. |

Nothing else changed behaviourally. The `shuffle_groups` rewrite is a strict simplification
— the two conditional merge branches are gone, so the v5 paths in which `nd` and `dg` were
concatenated and re-sorted no longer exist, and with them the last way a size-1 group could
reach the derangement. Group membership for every reachable configuration is otherwise
identical to v5: HateMM's degeneracy class holds ~49 items and splits roughly 39/10 across
the two partitions, so no drop fires there either.

---

## 3. What still stands

Everything verified in §3 of the v4 review and re-verified in §3 of the v5 review —
test-contact isolation, the `+0.050`/`+0.050` bar and decision rule, F113 confinement, self-
orbit exclusion, PARITY-NAT, the zero contract, the view subsequence contract and its
pre-forward proof, prompt fidelity, the registry carry-overs, the hard constraints, SLURM
hygiene, runtime survivability and the config↔code constant audit — holds unchanged against
the v6 bytes. The four round-3 High findings remain REPAIRED. I17/I18/I19 remain
ACCEPTED-AS-DECLARED on the same reasoning, and the two operator preconditions that no code
can enforce are unchanged: the `squeue` check for amendment condition (e), and re-verifying
the eight sha256 values immediately before `sbatch`.

---

## 4. Verdict

```
GO (0C/0H/3I)
```

Zero Critical, zero High. All four items I raised on v5 are closed: **I23 CLOSED, N1 CLOSED
(arithmetic correct, "which one the gate reads" claim true of the code), N2 CLOSED (drop is
the right rule, the dropped item is genuinely outside `covered`, the counter is correctly
named and scoped, and the self-test asserts the drop), N3 CLOSED (shell correct including
the `local got; got=$(…)` subtlety, all four hashes right, no wrapper regression).**

**0I was not reached, and I will not claim it was.** Three Info findings remain, all created
by the N2 edit and all documentation-only: F1 (config says the self-test proves a *merge*),
F2 (a halt message naming a rule that no longer exists), F3 (a justification that is true
only in the unreachable branch). None can change the verdict, corrupt a gated quantity,
touch test data, or kill either job — in F3's case the code is *better* than its own
description.

**My recommendation is to stop here and run with these three declared.** You asked me not to
soften, so I will also not inflate: a v7 for three strings would be the third consecutive
version whose only defects were introduced by the previous version's documentation edits —
v6's own N2 change is what created F1 and F2. If you nonetheless want the literal
`GO (0C/0H/0I)` token for amendment condition (a), the complete diff is three strings and
nothing else:

1. `c02_a0_v6.json:196` — "merges a singleton class within its own partition" → "drops a
   singleton class group rather than merging it".
2. `arena_v6.py:300,305` — "survived shuffle_groups' merge rule" → "survived
   `shuffle_groups`' size-≥2 rule (unreachable by construction)".
3. `arena_v6.py:257-263` + `c02_a0_v6.json:153` + `RECORD:39` — replace "makes
   `FULL > SHUFFLE` HARDER" with "cannot make `FULL > SHUFFLE` easier: a degenerate
   singleton's own displacement is zero, so dropping it leaves `SHUFFLE` exactly matched to
   `FULL` for that item; merging would instead have handed it a real displacement".

If a v7 is produced, my re-check would be confined to the eight hashes, those three strings
and a constants sweep — I would not expect to re-open anything else.

---

## 5. What I did and did not execute

**Did:** `sha256sum`, `ls`, `find`, `grep`, `sed`, `awk`, `cut`, `wc`, and file reads.
Specifically: recomputed all 8 v6 hashes and the four wrapper-pinned frozen-module hashes;
read `c02_a0_cpu_v6.sbatch` in full, `c02_a0_arena_v6.py`'s changed regions
(`244-307`, `612-651`, `795-834`, `925-955`, `1000-1049`) plus its constants and frozen
block; read the v6 config's `identity_causes`, `measured_identity_counts`, `arms`,
`oracle`, `gates` and `output` sections and `C02_A0_V6_RECORD.md`; grepped the whole v6 set
for `merge`, for the retracted H1/H3 phrases and for stale version references; re-measured
the HateMM/MHC-ZH whitespace-only counts with `awk` and located `hate_video_95` at gt line
356; and structurally compared the pointer-only files against their v4/v5 sizes.

**Did not:** run Python of any kind, import any module, load or open any `.pt` cache,
`.npz`, model, adapter or video; open any `test_seen` cache, `test.jsonl`, or any test label
or metric; run `squeue`, `sacct`, `sbatch`, `bash -n` or any SLURM command; touch a GPU or
Modal; modify, move or delete any reviewed file; or write anything other than this review.
