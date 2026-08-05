# C02 A0 v2 — evidence-density orbit reachability, preregistration record

**Status:** `V2_FROZEN_READY_NOT_SUBMITTED_PENDING_FRESH_INDEPENDENT_REVIEW`
**Date:** 2026-07-30 (Pacific/Auckland)
**Candidate:** `C02 Evidence-Density Quotient Geometry`
**Run IDs:** `C02-DEN-v1` (extraction), `C02-A0-v2` (A0)

This record is **prospective**. At freeze time no extraction job, no A0 job, no result,
no decision, no metric and no CONTINUE/KILL verdict exists, and none is claimed.

`refine-logs/C02_A0_RECORD.md` (v1) is **superseded** and is retained as the historical
first freeze. **V1 was never submitted, produced no artifact, no result and no
decision.** Its executable v1 files (`configs/c02/c02_a0_v1.json`,
`scripts/analysis/c02_a0_arena.py`, `scripts/slurm/c02_a0_cpu.sbatch`) have been removed
so that the superseded design cannot be submitted by mistake; their sha256 values are
preserved in §2 and in `configs/c02/c02_a0_v2.json::supersedes`.

Sections 1–8 of `C02_A0_RECORD.md` — authority, the question, the views, the extraction
contract, the arena, the arms and the gates — remain the design of record **except where
§2 and §3 below change them**. Read that file first, then this one.

---

## 1. Why there is a v2

The v1 freeze went to an independent static reviewer who had not seen the implementation
reasoning. Verdict: **`REVISE (0C/4H/20I)`**, persisted at
`refine-logs/C02_A0_PREREG_REVIEW.md`. The reviewer independently recomputed and matched
all 7 declared hashes plus the 5 imported-module hashes, independently confirmed
namespace absence, and independently confirmed: no reachable test path anywhere including
through imported modules; the subsequence contract and its pre-forward proof; full
self-orbit exclusion with fold parity asserted against the banked `vsw_ckpt`; that
`PARITY-NAT` binds; that the raw arena never enters the decision; that all config
constants match the code; and that the ≤4.0 GPU-hour budget is plausible.

Zero Critical. Four High. All four are repaired below. No High finding was argued away
and none was downgraded.

---

## 2. The four High findings and their repairs

### H1 — the `SHUFFLE` control leaked, and it is the load-bearing control

*Finding:* the derangement was **global over all `n` train items** and was applied to the
full key matrices **before** the per-fold split. In every fold roughly 80 % of held-out
queries therefore carried the non-native views of an item that was **in the bank**, so
each such query retrieved a near-identity match (`cos ~ 1 - orbit_radius`) to a bank row
with an unrelated label. `SHUFFLE` was biased downward by leakage rather than by loss of
the correct orbit, which made the `FULL > SHUFFLE` conjunct near-vacuous.

*Repair:* `derangement_within(groups, n, seed, fold)` deranges **within the fitting pool**
and **within the held-out fifth** separately, never across, with a fold-local seed
`[SHUFFLE_SEED, fold, group]`. A query's donor is now another query; a bank row's donor is
another bank row. The cross-partition near-identity channel is gone. Verified in-job by
the self-test case `within_partition_derangement`, which asserts zero fixed points **and**
that no donor crosses the boundary.

*Also recorded, because it is the honest framing:* **`NOISE` is the primary
anti-inflation control**, and it never had this problem — each item's tangent is generated
from its own `NAT` key, so it is leak-free by construction. It holds the number of extra
vectors and the displacement magnitude fixed while removing all semantic content, which is
exactly the artifact a max over a view-pair product could otherwise exploit. `SHUFFLE` is
the **membership** control: does the orbit have to belong to *this* video.

### H2 — the record's claim that `EMPTY_TEXT` would not fire was false

*Finding:* the v1 record said `EMPTY_TEXT` "is expected to be empty" on the basis of
`grep -c '"text": ""'` returning 0. The guard is `text.strip() == ""`, and many HateMM
rows carry a **whitespace-only** transcript, which `grep` for the empty string does not
match.

*Repair:* re-measured directly by parsing `data/gt/<DS>/{train,val}.jsonl` (no cache, no
model, no test). The corrected, verified counts:

| dataset / split | n | `text == ""` | `text.strip() == ""` | `len(T) > 12000` | full-identity orbit | view support | max chars |
|---|---|---|---|---|---|---|---|
| HateMM train | 744 | 0 | **39** | 9 | **48** | **0.9355** | 80731 |
| HateMM val | 107 | 0 | 9 | 1 | 10 | 0.9065 | 12275 |
| MHC-ZH train | 579 | 0 | 0 | 0 | 0 | 1.0000 | 708 |
| MHC-ZH val | 78 | 0 | 0 | 0 | 0 | 1.0000 | 343 |

`EMPTY_TEXT` is **not** a dormant guard: it fires on 5.2 % of HateMM train. Both datasets
still clear the `0.60` view-support gate with wide margin, and **A0 uses the train split
only** — the val rows are listed for completeness because the extraction covers them.

The v1 record's line "`EMPTY_TEXT` is expected to be empty" is **withdrawn**. The
character-length statistics quoted there were **line lengths in bytes**, while `L_MAX` is
in characters; the table above supersedes them with direct character counts.

### H3 — the `net_fix_rate` gate was a tautology dressed as an independent conjunct

*Finding:* `fixed - broken = n * delta_acc` **exactly**, so `net_fix_rate >= 0.030` cannot
bind under a `+0.050` accuracy bar. The registry amendment's net-fix clause was being
discharged by a restatement of the accuracy bar presented as a separate gate.

*Repair:* the identity is now **stated, not hidden**. The field is renamed
`net_fix_rate_IDENTICAL_TO_delta_acc` / `net_fix_rate_implied_by_acc_bar`, carries an
explicit note in every emitted record, and the decision code carries a comment saying it
can never bind. The check is retained defensively (it costs nothing and would catch an
accounting bug) but it is no longer claimed as independent evidence. The registry's
net-fix clause is discharged **by the accuracy bar itself**, and this record says so.

A genuinely independent fragility diagnostic is added and **reported, not gated**:
`precision_on_changed = fixed / (fixed + broken)`. An oracle that reaches `+0.050` by
fixing 100 and breaking 63 is far more fragile than one fixing 40 and breaking 3.
**No threshold is attached to it** — inventing a bar for a new quantity at this point
would itself be a defect.

### H4 — "`s_Q` upper-bounds what any orbit-contracting representation could buy" was unproved

*Finding:* max-cosine is the canonical quotient pseudo-metric, i.e. **one particular
orbit-invariant similarity**, not a supremum over all representations that contract the
orbit. The v1 record's "therefore a failure is decisive" overstated it.

*Repair:* the claim is withdrawn and replaced everywhere — record, config and the emitted
`interpretation_boundary` — by the accurate statement:

> A KILL is a **gate verdict under the registry's frozen Stage-0 rule**. `s_Q` is one
> particular orbit-invariant similarity, not a proven supremum over orbit-contracting
> representations, so a KILL closes C02 **under the rule** rather than proving that no
> such representation could ever help.

The oracle remains **optimistic** in the ordinary sense — it may use the best view of
every item on both sides of every comparison, which no deployable system may do — and that
is why it is the right instrument for a reachability gate. It is simply not a theorem.

---

## 3. Info-level changes carried into v2

Adopted from the review because they are cheap and make the instrument harder to fool:

- **In-job oracle self-test.** `c02_a0_arena_v2.py::oracle_self_test` runs on synthetic
  arrays **before any real data is opened** and proves: singleton-orbit bit-exact parity
  with `mechfix_ops.deployed_vote` (predictions, sorted similarities, neighbour IDs, and
  votes at max-diff `0.0`); that an all-zero query ties every similarity at 0 so its vote
  is 0 and its prediction is **invariant to tie order** — the justification for the
  `PARITY-NAT` neighbour-ID tie exemption, now *verified* rather than asserted; that the
  `k = topk` per-view-pair search reproduces a brute-force max over view pairs on the
  top-20 item set; and that the derangement never crosses the bank/query boundary.
- **Bootstrap estimand fixed.** v1 bootstrapped the macro-F1 of the 3-seed **majority**
  prediction while the bar is stated on the **mean of per-seed** macro-F1 — two different
  estimands. v2 computes both the point estimate and every replicate as the 3-seed mean of
  the pooled metric, identical to the bar.
- **Tie-corrected Spearman** (`scipy.stats.spearmanr`, average ranks). Text lengths tie
  heavily on short transcripts, so untied argsort ranks were wrong.
- **`torch.load` test guard installed in the arena** as well as in the mint and extractor.
- **`if not __debug__: raise SystemExit`** at the top of the arena, mint and extractor, so
  the assert-based guards cannot be silently stripped by `python -O`.
- **All output paths validated absent before the 7B model loads** in the extractor, so a
  name collision costs seconds rather than a full GPU pass; the test-path and split guards
  became explicit `RuntimeError` raises rather than bare asserts.
- **Zero-contract criteria 1 and 4 relabelled** `DOCUMENTARY_CITATION_NOT_COMPUTED`
  instead of being emitted as hardcoded `true`; criteria 2 and 3 remain
  `COMPUTED_AND_ASSERTED`.
- **`GATE-FID` scope note.** `headspace_fidelity.py` also emits its own
  `raw_effect_under_test: 0.0255` and `STOP_RULE_TRIGGERED` fields. Those belong to F105
  and are **not** this design's rule. The reader is used unmodified and only its measured
  `B_fid_abs_3seedmean` is consumed, against the `0.050` bar enforced in the wrapper.
- **`LENGTH_GUARD` citation corrected.** It is a **character** budget bounding sequence
  growth under doubling, not a tokenizer limit. `C02_EXPERIMENT_PLAN.md §3.1` (exclude and
  count items that would truncate under the frozen native tokenizer limit) is the closest
  precedent and is cited as such, not as the same criterion.
- **KRR limitation declared.** The OOF key matrix mixes heads — row `i` is read from
  fold(`i`)'s head, so a kernel entry can span two heads. This is a property of a
  **secondary** diagnostic; the gated arena never compares keys across folds, because bank
  and queries always come from one fold's head.
- **C14 dedup closed explicitly.** The A0 maxes over views at retrieval time, which is
  exactly the "frozen mechanism / upper-bound diagnostic" role the registry permits for
  `C14 Multi-prompt Representation Ensemble` and forbids as a final method. The C02
  **method** contract has no multi-view inference: inference uses `q_theta(x^0)`, the
  native view alone. No ensemble prediction is proposed at any stage.

---

## 4. Two defects found at v1 freeze time, before any real data was touched

Both were found by a synthetic-array dry run and are restated here because they are
load-bearing and remain in force in v2.

1. **An exhaustive `k = n_bank` faiss search is not bit-equal to the deployed `k = 20`
   call** — faiss takes a different code path for large `k`; measured max
   `|delta sim| = 1.5e-07`, enough to break `PARITY-NAT`. The oracle therefore searches
   `k = 20` **per view pair**, which is exact for the top-20 (if `tau` is the 20th largest
   per-item maximum, any row `>= tau` forces its item into the top-20, so each view pair
   contributes at most 20 such rows and its own top-20 already holds all of them) and is
   the literal deployed call for a singleton orbit.
2. **`mechfix_ops._norm32` can alias its input** — it returns the same buffer for an
   already-float32 C-contiguous array and `faiss.normalize_L2` works in place, so a second
   normalisation of the same buffer shifts similarities at float32 ulp level. The frozen
   module is not modified; the arena's own `_norm32` **always copies**.

The KRR standardisation repair described in `C02_A0_RECORD.md §9` also stands unchanged.

---

## 5. Frozen identity — sha256

**V2 frozen set:**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v2.json` | `2d4b7148154caea6ed41ec95043c15295c63d1abf3c47467b9191d285bd98a6f` |
| `src/utils/c02_density_views.py` | `f6209f04f04b88cfe47fadd5f7c7cd20b079f397a646fe824c8d2c3b35785b34` |
| `src/utils/generate_c02_density_view_text_embedding_HF.py` | `1e40e53013a032e527853cc5e82ca53b054882774b315a6ea6f319ce321b0803` |
| `scripts/slurm/c02_density_extract.sbatch` | `aaee1516f52ff2aabf508580b5451973a0484b3ca0875116be33a92c252e76e8` |
| `scripts/analysis/c02_a0_mint.py` | `f93a9d336c2917ede8737a8a597b7c9e3f83d5173ef4163b2e62118ba466da6b` |
| `scripts/analysis/c02_a0_arena_v2.py` | `7315e3232a42c96f1bf943028bb852eb89c9d85acd902f3890fb83fcd110e01d` |
| `scripts/slurm/c02_a0_cpu_v2.sbatch` | `ccf9881ccae7019d261a393afec4e6504203b947d38a259bf4d68b0238eccbf0` |

**Superseded v1 identity, preserved for the audit trail (files removed, never submitted):**

| path | sha256 |
|---|---|
| `configs/c02/c02_a0_v1.json` | `0b8a8289e7438396ce081fdf872f7d18017f870640fa33a687099de4066b53d1` |
| `scripts/analysis/c02_a0_arena.py` | `92abe7d8157a54f89a47657fb1edaf4a8f90e55b873c3fd03840aa940593fa41` |
| `scripts/slurm/c02_a0_cpu.sbatch` | `2b55c67834fc6dfdaf9a932be634c735b5362edcd128cfd5aa6e3829fc82c281` |
| `refine-logs/C02_A0_RECORD.md` (as reviewed) | `3c703b77d7cd6ebeac965d60378fffba8714dadb66fe16c91cf45fbfc42e679b` |

**Imported unmodified, sha256 asserted at run time:**

| path | sha256 | asserted by |
|---|---|---|
| `src/utils/generate_VideoMLLM_embedding_lora_HF.py` | `75bb8156705bff3c9bbce97542b90135c8f206f5bac30455f6987b0c48612399` | the extractor |
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` | mint + arena |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` | mint + arena |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` | arena |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` | GATE-FID reader, unmodified |

Chain: the view module pins nothing; the extractor pins the view module and the deployed
extractor; the mint and arena pin the frozen analysis modules; this record pins config,
sources and both wrappers; the TARGET records pin this record after its bytes are frozen.
**No file pins its own sha256.**

**Namespace absence at v2 freeze time (verified):** `artifacts/c02_edq` does not exist;
zero `*-c02den-*` files exist in `data/CLIP_Embedding/HateMM` or
`data/CLIP_Embedding/MHC_zh`.

---

## 6. Execution boundary

- **Two SLURM submissions, one each, in this order:**
  1. `sbatch scripts/slurm/c02_density_extract.sbatch` — 8 CPU / 1 A100 / 64 G, no
     `--time`.
  2. `sbatch scripts/slurm/c02_a0_cpu_v2.sbatch` — 8 CPU / 0 GPU / 32 G, no `--time`.
- Neither wrapper has `--time`, a dependency, array, singleton, requeue, chain, force or
  release path. `PENDING (JobHeldUser)` is normal, may last hours and **must never be
  force-released**.
- 8 CPU each, so the submit-time 16-CPU aggregate-cap wedge cannot occur, and the two jobs
  are strictly serial.
- `one_candidate_at_a_time` and `parallel_gpu_or_teacher_pilots_forbidden`: `squeue` must
  be checked immediately before the extraction is submitted and must show no other
  candidate's GPU or teacher pilot running.
- **Executed at preparation time, on the login node, and nothing else:**
  `python -m py_compile` on the four sources (byte-compile only); `bash -n` on both
  wrappers; `json.load` on the config; the view module's pure-string `self_test()`; the
  arena's `oracle_self_test()` on synthetic arrays; a synthetic-array dry run of the
  bootstrap, Holm and Spearman helpers; and a `json`-parse count of whitespace-only and
  over-length rows in `data/gt/<DS>/{train,val}.jsonl` for §2 H2. **No `.pt` cache, model,
  video, teacher, GPU, SLURM job or test path was opened, and no scientific quantity
  exists.**
- Execution requires a **fresh independent static review** of the v2 set returning
  `GO (0C/0H/0I)`.

---

## 7. Post-run fields — EMPTY at freeze time

| field | value |
|---|---|
| extraction job id | *(not submitted)* |
| extraction elapsed / GPU-hours vs the 4.0 cap | *(not measured)* |
| A0 job id | *(not submitted)* |
| GATE-FID `B_fid` per dataset | *(not measured)* |
| view support per dataset | *(not measured)* |
| `FULL - NATIVE` per dataset | *(not measured)* |
| verdict | *(none)* |
