# C06 `$0` CPU falsifier — preregistration **DRAFT v3** (2026-08-04)

**SUPERSESSION.** Supersedes `C06_FALSIFIER_PREREG_DRAFT_V2.md` (v2), which supersedes
`C06_FALSIFIER_PREREG_DRAFT.md` (v1). Both remain on disk **unmodified** as the record of
what each round reviewed. Reviews of record: `C06_FALSIFIER_PREREG_REVIEW.md` (round 1,
REVISE 3C/6H/10I+4M) and `C06_FALSIFIER_PREREG_REVIEW_R2.md` (round 2, REVISE 3C/3H/7I+3M).
This is a complete standalone document.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is
frozen, `TARGET_STATE.json` is untouched, nothing is committed.

**Disposition: all 13 round-2 findings ADOPTED, plus the 3 reopened round-1 findings
(C-3, the C-1 companion, I-3) now genuinely repaired. 0 rebutted.** Cumulative table in
**§14**.

**The one thing to read first.** Round 2 found, by executing C01's frozen code, that
**C01's zero contract is defined on a property the deployed head destroys**: `nn.Linear`
carries a bias, so `head(0,0)` is a *non-zero constant*, identical under both policies. In
head space the registered null therefore zeroes the *displacement* block while leaving every
*endpoint / common / rotation* block non-zero — and `l2_rows`' fail-closed mask assertion
cannot be satisfied by any single `zero_mask` across those arms. **I reproduced this myself:
C01's primary arm `common_displacement` cannot be built in head space on HateMM under either
mask choice.** §3.7 re-derives the null handling from first principles and §7.4 reports the
measurements, including the bit-exact row-subset identity that makes the new contract
provable rather than asserted.

---

## 1. What this falsifier is, and what authorizes it

C06 (*Prompt-Orbit Tangent/Curvature*) is **not an active candidate**; its registry status is
`gated_on_zero_cost_falsifier`
(`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`). What the queue has
reached is **C06's falsifier**, not C06.

**The unblock condition, verbatim** (`…dispositions.gated[0].falsifier_spec`):

> re-run C01's real-displacement-versus-matched-norm-orthogonal-rotation battery in the
> FOLD-HEAD ARENA on the already-banked `ro_*` caches. Zero GPU, zero extraction, minutes
> of CPU on `scripts/analysis/headspace_{mint,arena}.py`, which exist and are banked. If
> the rotations again match the real displacement in the deployed head space, C06 closes
> for `$0` and the `1.7-2.5 GPU-h` of extraction is never queued; if they do not, C06 has
> earned its extraction

**The two binding design constraints, verbatim** (`…falsifier_design_constraints`):

> its pre-registration must (i) use the per-dataset adapter lineage that ACTUALLY EXISTS
> — HateMM has only `-LoRA-curric` ro-caches, MHC_zh has only `-LoRA`, one lineage each,
> not a matched pair (correction V-8); and (ii) declare the prompt/readout-span confound,
> because `generate_VideoMLLM_embedding_readout_HF.py:73-89` shows the `ow_` cells change
> the readout span as well as the prompt — the same confound C01's review already narrowed
> its claim for

Both honoured — §3.1 and §10.1 — and verified as *honoured, not merely mentioned* by both
rounds.

**The evidence the gate rests on** — C01's A0, re-verified by the Gate-0 adjudicator against
`C01_A0_OUT.json` with every accuracy recomputed from the stored confusion matrices
(`GATE0_REOPEN_2026-07-31.md` §4.4):

| arm | HateMM acc / net (`n_dev` 107) | MHC-ZH acc / net (`n_dev` 78) |
|---|---|---|
| `endpoint_std` (reference) | `0.8411` / `0` | `0.8590` / `0` |
| `displacement` (real) | `0.8505` / `+1` | `0.8846` / `+2` |
| `common_displacement` (**primary**) | `0.8598` / `+2` | `0.8590` / `0` |
| `common_interaction` (secondary) | `0.8224` / `−2` | `0.8333` / `−2` |
| `common` | `0.8692` / `+3` | — |
| `endpoint_concat` | — | `0.8846` / `+2` |
| best rotation `orthrot_83p8` | `0.8692` / `+3` | `0.8974` / `+3` |
| `orthrot_72p7` | `0.8505` / `+1` | `0.8974` / `+3` |

`gain_over_strongest_control` `−0.0094` / `−0.0256`; `pass: false`;
`decision.continue = false`.

**Round-14's sharpening, adopted:** `orthogonal_blocks()` (`c01_policy_contrast_a0.py:1272`)
is a **Givens mixing of the two endpoint blocks**, so the six "random rotations" are six
angles on **one parameter family** that also contains the primary — `θ = 45°` **is**
`common_displacement`, `θ = 0` **is** `endpoint_concat`. Re-measured on the raw L24 features:
`8.941e-08` (θ=0, both datasets), `1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH) at θ=45,
matching the record's *"8.9e-08 to 1.2e-07"*.

**Why a re-run in a different space is the right instrument.** C01's arena is **raw dev
keys** (`n_dev` 107 / 78), not the fold-head path; the registry's `unified_pilot_gate.arena`
requires *"the actual fold-head/deployed-head path"* and F113 marks the raw-KILL direction
**NOT ESTABLISHED**. Both rounds ruled the arena reading correct.

---

## 2. The process rules that bind this design

`process_rule_compute_projection_and_heartbeat_2026_08_04` names this falsifier in
`applies_immediately_to`.

| rule | discharged in |
|---|---|
| **R1** measured-unit-cost × explicit-count projection; no reduced-scale extrapolation; no budgets inside the projection; **every unit enumerated** | **§8** — 16 unit types, each attributed to the dataset it was measured on; the ratio-derived phase is gone |
| **R2** line-buffered per-phase heartbeat | **§9** |
| **R3** (F114) dry execution exercises the **first real operation of the payload path** | **§7** |
| **R4** (`feedback-separate-code-review-lineage`) a design GO does not review the implementation | **§13** |
| **F118 erratum lesson** never let boilerplate describe a leg that did not run | **§3.7, §5.9, §8 Phase 5** — the head-space null-row sensitivity leg is *deleted*, not silently retained |

---

## 3. The arena and the instrument

### 3.1 Inputs — the lineage that actually exists

C01's frozen scientific configuration (`configs/c01/c01_a0_v2.json`) pins
`standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`, `feature_dim = 3584`.

| dataset | adapter lineage (the only one banked) | `expected.train.n` |
|---|---|---|
| HateMM | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | 744 |
| MHC_zh | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | 579 |

**Not a matched pair, and never treated as one.** No cross-dataset comparison of absolute
numbers is made; every decision quantity is within-dataset, within-seed, within-lineage. The
two-dataset requirement is a **conjunction of independently computed verdicts**. Round 2's
**I-7** caught the one place v2 broke this — a pooled cross-dataset HALT bar — and §6.1 now
freezes `ρ*` **per dataset**.

**Provenance.** The four L24 files are byte-identical to the ones C01 measured; the HateMM
`ro_L24` digest equals C01 v3's `diagnostic_train_cache_sha256` in full 64 hex. §11.

**L28 is not used** (round-1 I-8): at L28 the two endpoints are near-orthogonal
(`cos 0.147–0.396`) and the cell is `LAYER_FINAL`'s R0 clobber-guard, not a sibling of the
L24 grid. Removed from arms, gates, projection, manifest and verdict.

**Splits.** `train_*.pt` only for the ro caches. The native `dev_seen` cache is opened by
`headspace_mint.py:199` on every mint, is listed in §11 and is covered by `GATE-SHA`. No
`dev_seen_*-ro_*` file is opened by any phase; the `test_seen` ro caches are opened by
nothing.

### 3.2 The head, the folds, and the vote

* **Fold contract.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the
  **full** train split, asserted against the banked `vsw_ckpt/<ds>/f{0..4}.npz` by
  `headspace_mint.py:203-216`. **Unchanged by §3.7's null contract** — the fold assignment is
  computed on all `n` rows exactly as banked.
* **Head.** The deployed-recipe RGCL head, re-minted on CPU (deployed checkpoints are gone:
  F78). Bank = fitting pool, queries = held-out fifth, every item held out exactly once.
* **Vote.** `mechfix_ops.deployed_vote(..., topk=20)` — verified by both rounds to be
  numerically the operator C01's config specifies.
* **F88's caveat** — *"a CPU-trained arm must be paired against a CPU-TRAINED FLOOR"* —
  satisfied by construction.

### 3.3 Two head lineages, one driver

**Round 1's C-2**, measured: the head is trained on the native cache but forwarded over
`ro_L24` features that are **near-orthogonal** to it (median `cos(native_img, ro_L24_img)` =
`0.0234` HateMM / `0.0373` MHC-ZH, both caches unit-norm). v1's claim that this was "the
banked C02 house pattern" was **withdrawn in v2 and stays withdrawn**: `c02_a0_mint.py:214`
keeps `img_feats` **native** on every view and `:68` refuses any view file carrying an image
stream. C02 moved one stream inside one extraction family; C06 moves both to a different
readout cell.

**The battery therefore runs two lineages and requires C06 to fail on both:**

| lineage | head trained on | banked anchor | in-domain | mints |
|---|---|---|---|---|
| **Head-N** | native deployed cache | **`GATE-FLOOR`** — reproduces the six banked floors | no | 36 = 2 ds × 3 seeds × (5 folds + 1 full) |
| **Head-R** | `train_<model>-ro_L24.pt` | via the shared driver (below) | **yes** | 30 = 2 ds × 3 seeds × 5 folds |

Head-R needs no `fold = −1` head: the deployed-configuration head exists only to feed
`GATE-DEVFID`, which compares against banked **native** trainlogs that have no Head-R
counterpart.

**Round 2's H-2, and the repair.** v2 wrote *"both lineages share every other component …
the only variable is the training cache"*, while §7.2 showed Head-N minted by the frozen
`headspace_mint.py` and Head-R by new code. **That sentence was false of v2's dry check and
is now made true of the battery**, which is the stronger fix:

> **ONE driver, `scripts/analysis/c06_falsifier_mint.py`, serves both lineages.** It imports
> `headspace_mint` with its sha256 asserted and reuses its dataset table, deployed CLI, fold
> assignment, fold-parity assertion, dummy-dataloader construction, monkeypatches, seeding
> and DET-1 contract unchanged — the `c02_a0_mint.py` pattern. **Its only lineage-varying
> argument is `--train-cache`.**

Because Head-N runs through that same driver and Head-N must reproduce the six banked
`GATE-FLOOR` anchors at 4 dp on **both** metrics, **`GATE-FLOOR` anchors the driver, not
merely Head-N's science** — which is exactly the anchor round 2's §15.6 ruling asked for, at
**zero** additional cost. Round 2's fallback (six extra native-cache mints through a separate
Head-R driver, `264.5 s`) is not needed because there is no separate driver.

**Consequence for the projection, adopted conservatively.** v2 priced Head-R from a
scratchpad harness that skipped the fold-parity `npz` loads, the native `dev_seen` load and
the `npz` save (`37.46 / 27.54 s`). The real driver does all of those, so **v3 prices both
lineages at `headspace_mint.py`'s own measured units (`40.39 / 34.40 s`)** — `+146.9 s`
against v2, and the direction is honest.

### 3.4 The arm builder — one generic block-list construction

**Round 1's C-3.** v1 defined the arms afresh; the collapse from C01's two-modality
`paired_key` to one block was a *choice* no gate could check, and reading the source showed
the choice was wrong (it omitted `fuse_modalities`' outer per-block normalisation, and worked
in `float64` where `l2_rows` returns `float32`).

**The builder.** One construction, parameterised by an ordered list of blocks, in which every
normalisation is C01's `l2_rows` called through the **imported** `c01_policy_contrast_a0`:

```
fuse(blocks) = l2_rows(concat[ l2_rows(b) for b in blocks ])
paired(A,B)  = fuse([ l2_rows(concat[ l2_rows(A_m), l2_rows(B_m) ]) for m in blocks ])
build_views(std_blocks, ow_blocks, angles) -> the 13 arms
```

Two blocks `[img, text]` ⇒ C01's `prepare_views`, **bit-exactly** (`max|diff| = 0.000e+00`,
13 arms, both datasets — reproduced independently by round 2 from this prose alone). One
block (the fused head key) ⇒ the head-space arms.

**Round 2's ruling on how much that anchor buys, adopted verbatim rather than paraphrased.**
Two-block parity pins `l2_rows` itself, concatenation order, the contrast definitions, the
Givens mixing and its angle convention, the arm-name→formula map, the `float32` dtype and the
θ=0/θ=45 identities — *"a real anchor and a large advance on v1"*. It does **not** pin the one
operation C-3 named: at one block the outer `fuse` normalisation re-normalises an
already-unit vector. Measured here: `fuse([b])` differs from `l2(b)` by `7.451e-09` and the
one-block `paired` from v1's rejected `pair` by `1.118e-08` — both a fraction of a `float32`
eps (`1.192e-07`). **So v2's §3.4 overstated the repair, and v3 states it correctly: what the
restored normalisation actually fixed for the head-space arms is the *dtype*; the outer
normalisation is numerically vacuous at one block.**

**Why the block count is not a hidden choice.** `classifier_hateClipper` emits a single fused
1024-d vector, so no two-block head reading exists to get wrong. The block count is **forced
by the head's architecture**, not selected — which is what makes the one-block instantiation
derived rather than chosen, though for a different reason than v2 gave.

**Deferred-import note.** `c01_policy_contrast_a0.py:387` sets `np = torch = faiss = None`
and binds them only inside `import_compute_modules(config)`; the battery must call it before
touching the algebra.

### 3.5 The arms

Thirteen key-space arms plus one score-derived arm, at `ro_L24`.

| arm | role |
|---|---|
| `endpoint_std`, `endpoint_ow` | reference endpoints; C01 `gain_controls` |
| `avg_score` | derived score control (mean of the two endpoint vote scores) |
| `endpoint_concat` (≡ `orthrot_0`), `common` | ordinary controls |
| **`displacement`** | the real first-order prompt tangent — **real arm** |
| **`common_displacement`** (≡ `orthrot_45`) | C01's **primary** — **real arm** |
| `common_interaction` | C01's secondary |
| `orthrot_{8.3, 17.6, 29.1, 60.4, 72.7, 83.8}` | the matched-block-L2 rotation family |

### 3.6 The raw leg — gate-only, non-decisional, and on the same rows

The same builder runs on the raw L24 features (no head) and votes in the **same folds**, on
**the same population as the head leg** (§3.7). It renders no verdict and enters no decision
rule or multiplicity family. Its jobs are the gate discriminators: `ρ_raw` for
`GATE-ORBITDISP` and the raw accuracies for `GATE-ARMVIAB`. Round 2's §15.1 requirement —
that the raw leg, the head leg and the gate comparisons be on identical rows — is met by
construction and asserted by `GATE-POP`.

### 3.7 The head-space null contract — re-derived from first principles

**The defect, measured.** `classifier_hateClipper.__init__` (`src/model/classifier.py:80-81`)
builds `img_proj = nn.Sequential(nn.Linear(3584, 1024), …)` **with the default bias**, so
`head(0,0)` is a non-zero constant — measured `‖head(0,0)‖ = 0.6347`. HateMM train row 355
(`hate_video_95`, C01's registered `authorized_null`) is bit-identically zero in both
modalities of **both** ro caches, so `h_std[355] == h_ow[355]` **exactly** (measured). Hence
in head space:

* every **endpoint / common / rotation** block is **non-zero** at row 355 ⇒ needs
  `zero_mask = None`;
* the **displacement** block is **exactly zero** at row 355 ⇒ needs `zero_mask = {355}`.

`l2_rows:1193-1194` is fail-closed on precisely this. Measured, each choice kills the other,
and **`common_displacement` — C01's primary, one of the two arms whose comparison renders the
verdict — dies under both** (§7.4).

**It is not only an execution problem.** Round 2's C-3 identified the deeper asymmetry: with
the row *masked* rather than removed, its key in the eleven control arms is an ordinary unit
vector that faiss returns as a genuine top-20 neighbour and `deployed_vote` weights into other
items' votes, while in the `displacement` arm the same row has a zero key and contributes
nothing. An item whose features are a **known extraction failure** would vote in every control
arm and in no real arm — an asymmetry lying **exactly along the comparison that renders the
verdict**, on the dataset that carries the null. C01's own `zero_contract_v2` assumes the
opposite (`normalization_output_state: exact_numeric_zero`,
`require_null_absent_from_all_top20: true`); the head falsifies both.

**The contract.** Three populations, each named, each with its own justification:

| object | population | why |
|---|---|---|
| **head training** | full `n` (744 / 579), banked fold assignment, fold parity asserted | the deployed recipe is unchanged; `GATE-FLOOR` and the fold contract depend on it |
| **`GATE-FLOOR`** | full `n` (744 / 579), native keys | the six banked floors were computed on this population; anchoring on any other would be meaningless |
| **arm arena — head leg AND raw leg** | **`n = 743`** on HateMM (row 355 removed from bank *and* query sets), `n = 579` on MHC-ZH | the registered null is physically removed, so `zero_mask = None` is correct for every block and the imported `l2_rows` runs unmodified on all thirteen arms |

**Why this is legitimate, on four counts:**

1. **Label-free.** Row 355 is identified by an exact-zero *feature* property and is C01's own
   pre-existing frozen `authorized_null` (`c01_a0_v3.json::lineage_evidence.authorized_null`:
   `row_index 355`, `raw_id hate_video_95`, both policies, both modalities). No label is
   consulted to select it, and it was selected long before this design existed.
2. **Verdict-neutral, and it *removes* a bias rather than creating one.** The row is dropped
   identically from every arm, so neither lane can gain. More than that: leaving it in is what
   biases the comparison, by the C-3 asymmetry above. Removal *enforces* C01's own
   `require_null_absent_from_all_top20` instead of assuming it.
3. **Provably a pure row-subset.** Measured: raw arms built at `n = 743, zero_mask = None` are
   **bit-identical** to raw arms built at `n = 744, zero_mask = {355}` restricted to the 743
   surviving rows — `max|diff| = 0.000e+00` across all 13 arms — and every `ρ` value is
   unchanged to `0.000e+00`. The population change introduces **no algebraic difference
   whatever**. `GATE-DUALPATH` asserts exactly this identity at run time.
4. **Fixed before any trained-head number exists.** Nothing in this contract depends on an
   accuracy; it is settled by the zero-structure of the banked features and the presence of a
   bias term.

**The dataset asymmetry, stated plainly.** MHC-ZH has no exact-zero row, so its arena is
`n = 579` unchanged and the contract is vacuous there. The asymmetry is **contained by the
form of the two-dataset requirement**, which is a conjunction of two independently computed
verdicts, never a pooled number (§3.1). It touches nothing else: the folds, the heads, the
floors and every threshold are per-dataset already. `GATE-POP` asserts the realised
populations against this table, so a silent drift cannot occur.

**There is no head-space null-row sensitivity leg, and this document does not pretend
otherwise.** v2 carried one as Phase 5. It cannot exist: the alternative head-space
population is **unbuildable** (that is the whole of the defect above), so a leg comparing the
two would have nothing to compare. The sensitivity question is discharged where both
populations *are* defined — the raw two-block anchor — by `GATE-DUALPATH`. Saying this
explicitly is the F118 erratum lesson applied before the fact rather than after: no gate,
field or boilerplate in this design describes a leg that did not run.

---

## 4. Ambiguities in the written condition

**"Conservative" means *hardest for the falsifier to deliver the `$0` CLOSURE***, because
closure is the irreversible action. **Round 1 ruled this correct**, on two conditions — the
design must disclose what the lean buys (§5.8) and must not use it to excuse an arithmetic
error (§5.5). Round 2 held v2 to the second condition on the lineage axis; §5.5 now discharges
it there too.

| # | ambiguity | resolution |
|---|---|---|
| **A1** | *"the rotations"* — best, or family? | C01's `require_primary_above_all_rotation_controls`: the real arm must beat **every** rotation; the negation is the closure trigger |
| **A2** | *"the real displacement"* | **Both** `displacement` and `common_displacement`, disjunctively, multiplicity-corrected (§5.5) |
| **A3** | *"match"* — ties? | A tie **closes** C06; C01's own `>` semantics |
| **A4** | *"in the fold-head arena"* | Both rounds ruled the arena reading correct; the training-set half is resolved by running **both** lineages (§3.3). Round 2 reopened only *which rows* constitute it — settled in §3.7 |
| **A5** | which layer? | **L24 only** |
| **A6** | seeds | Seed-mean over 3 head seeds, plus **3/3** per-seed agreement on the rotation-dominance leg |
| **A7** | *"C06 closes"* — what? | The **first-order (tangent/chord) leg only**. **Round 1 ruled A7 is not an obstacle**: C01's battery *is* the two-point contrast |

---

## 5. The pre-registered decision rule

### 5.1 Notation and population

For dataset `D ∈ {HateMM, MHC-ZH}`, lineage `L ∈ {Head-N, Head-R}` and seed `s ∈ {0,1,2}`,
each arm `A` yields one OOF prediction vector over the **arm-arena population** of §3.7 —
`743` items on HateMM, `579` on MHC-ZH — scored against **train-split** labels held out from
the head that judged them. `acc(A,D,L)` is the mean over the three seeds.

* **Real arms** `R = {displacement, common_displacement}`.
* **Rotation family** `Θ` = the six frozen angles.
* **Ordinary controls**, per arm, from C01's own two frozen lists:
  * `A = common_displacement`: `C` = `gain_controls ∪ {displacement}` — **six**;
  * `A = displacement`: `C` = `gain_controls` — **five**.

### 5.2 SURVIVE

**On a given lineage `L`, C06 SURVIVES iff there exists `A ∈ R` such that S1–S6 all hold on
BOTH datasets:**

| | condition | frozen source |
|---|---|---|
| **S1** | `acc(A) > max_θ acc(orthrot_θ)` **and** `mF1(A) > max_θ mF1(orthrot_θ)` | `require_primary_above_all_rotation_controls` |
| **S2** | S1's accuracy leg holds in **3/3** seeds | A6 |
| **S3** | `acc(A) − max_{c∈C} acc(c) ≥ 0.02`, likewise `mF1` | `minimum_gain_over_strongest_control = 0.02` |
| **S4** | for every comparator in `C ∪ Θ`: paired item-level bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05` | `minimum_bootstrap_lower_bound`, `require_primary_bootstrap_holm_reject`, `require_rotation_bootstrap_holm_reject`, `n_bootstrap = 2000`, `statistics.seed = 20260728`, `holm_alpha = 0.05` |
| **S5** | **both** real arms exceed the 95th percentile of their shuffled-pair null, **and** the shuffle comparison Holm-rejects | `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject` |
| **S6** | **net fixes** `≥ 3` (HateMM) / `≥ 2` (MHC-ZH) against `endpoint_std` | `minimum_net_fixes`; reference is C01's own `retrieval.fix_break_reference = endpoint_std` |

Plus `GATE-SMALLDISP` (§6) for C01's `require_no_small_displacement_dominance`.

### 5.3 CLOSE

**C06 CLOSES iff the run publishes a verdict (§6) and SURVIVE is false on BOTH lineages.**
Equivalently, C06 survives if it clears S1–S6 on **either** lineage.

### 5.4 The bootstrap unit

Resample items once (`B = 2000`, C01's frozen `statistics.seed = 20260728`); inside each
resample, average the three seeds' per-item correctness. The seed axis is therefore **inside**
the statistic, not a hidden multiplicity.

### 5.5 Multiplicity — now corrected on all three axes

Round 1 caught the uncorrected **arm** disjunction; round 2 caught v2 re-committing the same
error on the **lineage** axis it had just introduced, and correctly refused v2's defence
(*"the two lineages are also a conjunction for CLOSE"* — CLOSE is, but **SURVIVE is a
disjunction**, and that is the event a false positive concerns).

**v3 adopts round 2's option (b): one Holm family per dataset spanning both lineages.**

`common_displacement` 6 comparators + 6 rotations = 12; `displacement` 5 + 6 = 11;
`(12 + 11) × 2 metrics = 46` per `(dataset, lineage)`; **× 2 lineages = 92 hypotheses per
dataset**, Holm at `α = 0.05`.

The two **datasets** remain a conjunction and are not pooled. CLOSE remains a conjunction over
lineages. Only SURVIVE is a disjunction, and it is the disjunction the family now covers.

### 5.6 Instrument failure, non-finiteness, and absence

Any HALT gate failing on either dataset in either lineage ⇒ **HALT: no verdict, in either
direction**, recorded as `INSTRUMENT_INCONCLUSIVE`; it **may not be reported as a closure**.

* Every gate quantity is asserted **finite before** comparison, and every gate is written in
  **pass-condition** form so a non-finite value HALTs.
* **Round 2's I-6, adopted:** an **absent** decision or gate quantity HALTs on the same
  footing as a non-finite one. This closes the path where a silently missing lineage, dataset
  or process makes SURVIVE vacuously false and thereby supplies half of CLOSE for free.
* `GATE-LEDGER`'s process-count predicate is **binding**, not merely declared (§12).

### 5.7 Pre-declared expectation

**CLOSE is expected**, on two grounds: C01 measured the premise rotation-indistinguishable at
the two-point case on both datasets, and the recon's structural objection (a fixed prompt
injects no per-item information) is unrebutted. v1's third ground — the `219×` untrained-head
contraction — remains **withdrawn** (round-1 I-10).

### 5.8 Disclosure: what the conservative lean and the frozen sets buy

1. `require_accuracy_gain_over_deployed_r0_context` is **not carried**. Its comparator
   `deployed_r0_accuracy_context_only` (`0.8505` / `0.8590`) sits in a block named
   `historical_strict_devtrain`, sourced from `READOUT_SCREEN_OUT.json`: a raw dev-arena
   figure at `n_dev` 107/78. Importing it would breach F88's CPU-arm/CPU-floor caveat.
   **Inapplicable across arenas, not waived** — round 2 verified the comparator and endorsed
   the reasoning.
2. **Round 2's M-2, adopted:** `displacement`'s comparator set omits `common_displacement`,
   while `common_displacement` must beat `displacement`. C01 froze a comparator list only for
   its primary, so this is not a violation — but the asymmetry **eases SURVIVE for one of the
   two disjuncts** and is disclosed here rather than left silent.
3. The two-real-arm disjunction (A2) is deliberately generous; its multiplicity is now
   corrected (§5.5).

### 5.9 What this design does not run

Stated affirmatively so no reader must infer it: **there is no head-space null-row sensitivity
leg** (§3.7), **no L28 leg** (§3.1), **no per-arm retrained head** (§15.3), and **no test-split
read of any kind**. No gate, output field or record sentence describes any of them.

---

## 6. Gates

Every quantity is asserted finite and present before comparison (§5.6).

| gate | asserts |
|---|---|
| `GATE-DET1` | thread env exported before any python starts (`headspace_mint.det1_assert`) |
| `GATE-SHA` | every frozen import and input cache matches §11; **once in the sbatch driver** |
| `GATE-FOLD` | fold assignment matches the banked `vsw_ckpt/<ds>/f{0..4}.npz`, on the **full** `n` |
| `GATE-FLOOR` | **Head-N through the shared driver**, native keys, **full `n`**, reproduces the banked floors at 4 dp on **both** metrics — acc HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946`; mF1 HateMM `0.8838/0.8811/0.8812`, ZH `0.8747/0.8710/0.8765`; and every `fold_acc_deployed` entry. **This anchors the driver for both lineages** (§3.3) |
| `GATE-POP` | the realised populations equal §3.7's table exactly — head training and `GATE-FLOOR` at full `n`; arm arena at `743 / 579`; head leg and raw leg on **identical row sets** |
| `GATE-C01PARITY` | (i) the two-block builder reproduces `prepare_views` **bit-exactly** at `n = 744, zero_mask = {355}` (HateMM) and `n = 579, None` (ZH); (ii) HALT on any residual above C01's `2e-6` |
| `GATE-DUALPATH` | **raw leg only** — the two-block build at `n = 743, zero_mask = None` is **bit-identical** to the `n = 744, zero_mask = {355}` build restricted to the 743 surviving rows, on all 13 arms. This is C01's `displacement_registered_null_exclusion` dual-path equivalence, applied where it is defined, and it is the run-time proof of §3.7's row-subset claim |
| `GATE-NULLREMOVED` | no arena population (head or raw, either dataset) contains an exact-zero row, and the removed row set equals `{355}` on HateMM / `{}` on MHC-ZH |
| `GATE-ORBITDISP` | for every arm: HALT iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D`, with **per-dataset** `ρ*` frozen in §6.1; and `ρ_raw` reproduces §6.1's frozen values at 4 dp |
| `GATE-ARMVIAB` | for `endpoint_std` and both real arms: if head-space accuracy fails `majority + 0.02`, HALT **iff** the same arm clears it in the raw space (§6.2) |
| `GATE-ARENA` | **lower** bound `majority + 0.02 ≤ acc` on **`endpoint_std` only** (C09's scope); **upper** bound `acc ≤ 0.98` on `endpoint_std` **and** both real arms. Majority `0.5995` / `0.6891` (§6.3) |
| `GATE-DOMAIN` | Head-N's recovery fraction `(acc_ro − maj)/(acc_native − maj)` for `endpoint_std` is computed and **printed on the verdict face**; reporting-only, no bar (§6.4) |
| `GATE-NESTED` | **per item**, the head that scored it excluded its fold; emitted as a check count equal to the item count |
| `GATE-SELFTEST` | the net-fix identity `net(A) = n · (acc(A) − acc(endpoint_std))` holds exactly for every arm, seed, dataset and lineage (**round 2's free strengthening**, which S6 gives an object) |
| `GATE-ZEROOP` | `orthrot_0` vs `endpoint_concat` and `orthrot_45` vs `common_displacement` produce identical predictions — with the **tie diagnostic** of §6.5 |
| `GATE-ALGEBRA` | key-level `max|orthrot_0 − endpoint_concat| ≤ 2e-6` and `max|orthrot_45 − common_displacement| ≤ 2e-6`. **Logically independent of `GATE-ZEROOP`, not weaker** (§6.5) |
| `GATE-IDPARITY` | every ro cache's `ids` order and `labels` identical to the native bank (`c02_a0_mint.py:69-71`, `:72-76`) |
| `GATE-ZEROMASK` | **feature space only** — the measured exact-zero row set of the ro inputs equals `{355}` (HateMM) / `{}` (MHC-ZH) on both policies. v2's head-space clause is **dropped**: it is vacuous once the row is removed |
| `GATE-SMALLDISP` | C01's `require_no_small_displacement_dominance` at `small_displacement_train_quantile = 0.1` |
| `GATE-LEDGER` | C09's full declared-count predicate set, with the process count **binding** (§12) |
| `GATE-DEVFID` | `headspace_fidelity.py` on Head-N's 6 full heads — **reporting only, does not gate** |

`GATE-SHUFFLEFIX` is **deleted**. C01's `shuffle_fixed_point_bijection` requires the
registered null to be a fixed point of every permutation; with the row removed from every
population that runs a shuffle, the requirement has no object. Its content is preserved
falsifiably by `GATE-NULLREMOVED`, which checks the removal actually happened. Deleting it
rather than retaining a gate that checks nothing is the F118 erratum lesson.

### 6.1 `GATE-ORBITDISP` — per-dataset bars

Every arm is `l2`-normalised before `deployed_vote` (and again by `_norm32`), so **the
retrieval key keeps only direction**. If the head's two policy outputs differ by a
near-constant offset, every `displacement` key is nearly the same vector, every rotation beats
it trivially, and a magnitude gate would see nothing. The quantity is therefore
`ρ = ‖mean_i k_i‖` over unit keys, where `ρ → 1` means every item's key points the same way.

**Round 2's ruling on the statistic, adopted with its reasoning.** The **max** is the right
order statistic, for a structural reason better than v2's: `GATE-ORBITDISP` is an *instrument*
gate, and its job is to fire only when the head space is **more degenerate than anything the
raw feature family produces**. A quantile bar would HALT on head-space arms whose
concentration sits inside the observed raw range, converting ordinary head-induced
concentration into `INSTRUMENT_INCONCLUSIVE` — the same self-defeat as §6.3.

**Round 2's I-7, adopted: the bar is frozen per dataset**, restoring §3.1's within-dataset
discipline. v2's single pooled `0.9772` was a MHC-ZH measurement setting HateMM's bar
`0.0090` looser than HateMM's own data supports.

| dataset | `ρ*` (max over 13 raw arms) | runner-up |
|---|---|---|
| HateMM | **0.9681** (`endpoint_std`, `0.968176`) | `0.9644` (`common`) |
| MHC-ZH | **0.9772** (`endpoint_std`, `0.977223`) | `0.9697` (`common`) |

Runner-ups are recorded so the permissiveness is on the face of the document.

**The gate, in the reviewer's case split:** `ρ_raw ≤ ρ*_D ∧ ρ_head > ρ*_D` ⇒ instrument
destroyed the object ⇒ **HALT**; both above ⇒ both spaces degenerate ⇒ C06's premise is false
⇒ **no HALT, CLOSE warranted**; otherwise no HALT. Applied to all 13 arms.

**`ρ_raw` frozen values** (measured on the arm-arena population of §3.7; the population change
moves them by `0.000e+00`, §7.4):

| arm | HateMM | MHC-ZH |
|---|---|---|
| `endpoint_std` | 0.9682 | 0.9772 |
| `common` | 0.9644 | 0.9697 |
| `orthrot_83p8` | 0.9569 | 0.9644 |
| `orthrot_72p7` | 0.9565 | 0.9651 |
| `endpoint_concat` | 0.9553 | 0.9624 |
| `orthrot_8p3` | 0.9514 | 0.9584 |
| `orthrot_60p4` | 0.9484 | 0.9587 |
| `orthrot_17p6` | 0.9448 | 0.9519 |
| `endpoint_ow` | 0.9422 | 0.9474 |
| `orthrot_29p1` | 0.9336 | 0.9418 |
| `common_displacement` | 0.9288 | 0.9399 |
| `common_interaction` | 0.9138 | 0.9682 |
| **`displacement`** | **0.8917** | **0.9091** |

In the raw space the displacement direction is the **most** item-dispersed arm of the
thirteen, on both datasets — which is why C01 could score it at all, and why a head-space
`ρ(displacement)` approaching `1.0` would be the head's doing rather than C06's.

### 6.2 `GATE-ARMVIAB` — the two-case form, endorsed by round 2

A one-sided majority-rate HALT on the real arms would fire on exactly the outcome the
falsifier exists to detect: if C06's premise is false, `displacement` in head space *should*
sit near the majority rate, and a one-sided gate would convert a **warranted CLOSE** into a
HALT, leaving C06 gated forever on an instrument that can never close it.

* head-space arm fails `majority + 0.02` **and** the raw counterpart also fails ⇒ genuine
  negative ⇒ **no HALT**;
* head-space arm fails **and** the raw counterpart clears ⇒ instrument destroyed it ⇒
  **HALT**.

**Round 2 endorsed this refinement explicitly**, including that it rests on a logical argument
rather than measured raw accuracies — those are decision-relevant inputs that §7.3's blindness
rule forbids measuring before freeze, and *"a gate that discriminates between two instrument
states does not need its own outcome measured in advance."*

### 6.3 `GATE-ARENA` — round 2's C-2, repaired

v2 extended `GATE-ARENA`'s **lower** bound to the two real arms. That HALTs unconditionally on
the same condition §6.2 exists to permit, one row above it in the same table — **so §6.2's
refinement was inoperative**. It was also over-scoped against its own precedent: C09's
`GATE-ARENA` (`C09_A0_V17_RECORD.md:1569-1572`) applies to **pooled native accuracy**, the
floor arm, never a treatment arm; and round 1's I-6 asked only that the **upper** bound be
added.

**v3 restricts the lower bound to `endpoint_std`** — the arm that measures instrument health
rather than the hypothesis — and keeps the `≤ 0.98` upper bound on `endpoint_std` and both
real arms, where it catches a leak and cannot fire on a warranted CLOSE. The real arms' lower
side belongs entirely to `GATE-ARMVIAB`.

**"Real arms lose badly" is a reportable scientific outcome — the falsifier working — and is
never an instrument HALT.** That sentence is the invariant §6.2 and §6.3 jointly enforce.

### 6.4 `GATE-DOMAIN` — reporting, with no invented bar

Round 2 ruled that C-2(a) is substantively discharged by `GATE-ARENA`'s lower bound on
`endpoint_std`, which *is* a head-space input-domain fidelity HALT gate anchored on a banked
number (the majority rate), and that refusing to invent a recovery-fraction bar is correct.
`GATE-DOMAIN` therefore computes `(acc_ro − maj)/(acc_native − maj)` for `endpoint_std` under
Head-N and requires it on the verdict face and in §10.2's scope sentence, with no threshold.

### 6.5 `GATE-ZEROOP` and `GATE-ALGEBRA` — independent, with a tie diagnostic

**Round 2's I-5, adopted.** v2 called `GATE-ZEROOP` *"strictly stronger"* than `GATE-ALGEBRA`.
That is wrong in both directions: identical predictions does not imply `≤ 2e-6` keys, and
`≤ 2e-6` keys does not imply identical predictions. The two are **logically independent** and
the value is their conjunction.

Worse, the θ=45 identity is **not exact** — measured `1.192e-07` (HateMM) / `8.941e-08`
(MHC-ZH) on the raw keys, from the `cos45 − sin45 = 1.11e-16` asymmetry — so a `~1e-7` key
perturbation can reorder a top-20 neighbourhood and `GATE-ZEROOP` has a real **false-HALT**
probability on a correct run. This is exactly why C01 gated keys at `2e-6` rather than
asserting equality.

**Pre-registered tie diagnostic.** On a `GATE-ZEROOP` mismatch, emit the number of affected
items whose 20th/21st neighbour similarities differ by less than the measured `GATE-ALGEBRA`
residual. **A mismatch confined to such items is REPORTED, not HALTed.** Any mismatch outside
them HALTs.

---

## 7. Dry-check — what was executed, and what it found

Login node `foscsmlprd01`, conda `HateVideo`, 8 threads, DET-1 exported, outputs only in the
session scratchpad. Zero GPU, zero SLURM, zero test-split file opened, zero write into
`data/`, `artifacts/` or `logging/`.

### 7.1 Real inputs

All 8 ro caches and 4 native caches loaded through the real `torch.load` path. `n = 744 /
579`; features `(n, 3584)`; `ids` order-identical and `labels` identical to the native bank on
every file. Exact-zero rows: HateMM `{355}` in both modalities of all four ro caches and the
native cache; MHC-ZH none. Class balance: HateMM `posrate 0.4005` ⇒ majority **`0.5995`**;
MHC-ZH `posrate 0.3109` ⇒ majority **`0.6891`**. Row 355 is held out in **fold 4**, label `1`.

### 7.2 Mint units — full-process wall, and how they were timed (round-2 I-1)

**Round 2's I-1: round-1 I-3 leg (c) — per-process interpreter and import cost — was never
enumerated, and v2 never said whether its mint figures were full-process wall or in-process
elapsed. It is now stated, with the measurement that settles it.**

Every mint figure below is **full-process wall**, measured around the `python …` invocation
itself — the first by `/usr/bin/time -v` (`Elapsed (wall clock) 0:40.39`) and the rest by
`date +%s.%N` bracketing the process. The interpreter-and-import cost is therefore **already
inside every unit**, and the evidence is direct: the same HateMM run reports `40.39 s`
full-process against the mint's own internal timer of `33.0 s`, a `7.4 s` gap that is
interpreter startup plus cache loads plus the `npz` save. Measured startup alone,
`python -c "import torch, numpy, faiss, sklearn.model_selection"`: **3.16 / 3.18 / 3.05 s** —
consistent with that gap and comfortably inside it. **No Phase 1e line is added, because
adding one would double-count.**

| lineage | unit | dataset | measured wall |
|---|---|---|---|
| Head-N | fitting-pool head | HateMM | **40.39 s** |
| Head-N | fitting-pool head | MHC-ZH | **34.40 s** |
| Head-N | full-train head | HateMM | **49.30 s** |
| Head-N | full-train head | MHC-ZH | **38.87 s** |
| Head-R | fitting-pool head, scratchpad harness | HateMM | 37.46 s — **not used**, see §3.3 |
| Head-R | fitting-pool head, scratchpad harness | MHC-ZH | 27.54 s — **not used** |

Fold parity passed in all seven mints. Peak RSS of the HateMM Head-N fold mint: **1 305 984 KB
= 1.25 GiB**, agreeing with C09's measured 1.22 GiB. §8 prices Head-R at the **Head-N units**,
because the shared driver does the fold-parity loads, the native `dev_seen` load and the `npz`
save that the scratchpad harness skipped.

### 7.3 Blindness

Every head used in any arm-building or voting dry check is **untrained**, so every operation is
real at real scale while the numbers are scientifically void. No arm accuracy was computed,
printed or recorded, on any dataset, in either lineage. Both rounds confirmed this discipline
and found nothing inconsistent with it. The `l2_rows` structural results in §7.4 are
weight-independent — they turn on the bias being non-zero, not on its value.

### 7.4 The null-contract measurements — the heart of v3

Reproducing round 2's C-1/C-3 and then testing the repair:

| # | measurement | result |
|---|---|---|
| **(a)** | is `head(0,0)` zero? | **No** — `‖head(0,0)‖ = 0.634676` |
| **(b)** | `h_std[355] == h_ow[355]`? | **Yes, exactly**; and `h_std[355]` is **not** all-zero |
| **(c)** | `l2_rows` on the endpoint block | `zero_mask = {355}` **DIES**; `None` **OK** |
| **(d)** | `l2_rows` on the common block | `zero_mask = {355}` **DIES**; `None` **OK** |
| **(e)** | `l2_rows` on the displacement block | `zero_mask = {355}` **OK**; `None` **DIES** |
| **(f)** | ⇒ `common_displacement = paired(common, displacement)` | **unbuildable under either mask** — round 2's C-1 confirmed independently |
| **(g)** | **the repair**: all 13 head-space arms at `n = 743, zero_mask = None` | **ALL BUILT** through the imported `l2_rows`, `dtype float32` |
| **(h)** | **the bridge**: raw arms at `n = 743, None` vs `n = 744, {355}` restricted to the 743 rows | **BIT-EXACT, `max|diff| = 0.000e+00`**, all 13 arms |
| **(i)** | every `ρ` under the population change | **unchanged, `0.000e+00`** |

(g), (h) and (i) are what license §3.7: the repair executes, and the population change is a
pure row-subset with no algebraic content whatever.

### 7.5 The one-block instantiation (round 2's γ, reproduced)

`fuse([b])` differs from `l2(b)` by **`7.451e-09`**; the one-block `paired` differs from v1's
rejected `pair` by **`1.118e-08`**. Both are a fraction of a `float32` eps (`1.192e-07`), so
round 2's conclusion holds on my own numbers: **the outer normalisation C-3 restored is
numerically vacuous at one block**, and what v2 actually fixed there was the dtype. §3.4 now
says so.

### 7.6 `GATE-C01PARITY`, re-verified

The two-block builder reproduces `prepare_views` **bit-exactly** on the raw L24 features,
`max|diff| = 0.000e+00`, all 13 arms, both datasets. C01's own algebra guard on the same call:
`8.941e-08` (θ=0, both) and `1.192e-07` / `8.941e-08` (θ=45).

### 7.7 v1's withdrawn units, and the current unit table

Round 1's H-6 found v1's `U4` smaller than its own constituents. Cause: v1 timed the arm
matrices as `float64` (paying a `_norm32` conversion the real battery will not) and repeated a
single fold five times. Re-measured with `float32` arms over the five real folds, it
reconciles: `5 × 0.00305 + 5 × 0.00629 = 0.04674 s` of votes, leaving `0.04234 s` for the
rebuild.

**Every unit, with the dataset it was measured on (round-2 I-3):**

| unit | what | dataset | measured |
|---|---|---|---|
| `U1` | head forward over one real ro cache | HateMM (`n = 744`) | 0.0461 s |
| `U2a` | vote, 1024-d, per fold-cell | HateMM | 0.00305 s |
| `U2b` | vote, 2048-d, per fold-cell | HateMM | 0.00629 s |
| `U2c` | vote, 7168-d (raw fused), per fold-cell | HateMM | 0.04239 s |
| `U2d` | vote, 14336-d (raw paired), per fold-cell | HateMM | 0.08098 s |
| `U3` | bootstrap `B = 2000`, one comparison, both metrics | HateMM | 0.126 s |
| `U4` | one shuffled-pair null draw (2 arms × 5 folds + rebuild) | HateMM | 0.08908 s |
| `U5a` | two-block build of 13 arms (`prepare_views` 6.427 + builder 4.626 + compare 0.213) | HateMM | 11.27 s |
| `U5b` | builder-only pass, 13 arms | HateMM | 4.63 s |
| `U6` | `ρ` over 13 arms | HateMM (raw, 7168/14336-d) | 0.62 s |
| `U7` | `GATE-SHA` over 8 caches + 6 modules | — | 0.12 s |
| `U8` | ro cache `torch.load`, 2 files | HateMM | 0.033 s |
| `U9` | `GATE-DEVFID`, per `(dataset, seed)` | HateMM / MHC-ZH | 3.70 s / 3.49 s |
| `U10` | head-space build of all 13 arms, one cell | HateMM (`n = 743`, 1024/2048-d) | 0.1873 s |
| `U11` | interpreter + `torch/numpy/faiss/sklearn` import | — | 3.05–3.18 s (**inside the mint units**, §7.2) |

**Convention, stated once and applied uniformly (round-2 I-3):** every unit above was measured
on **HateMM**, the larger dataset (`n = 744` vs `579`), and is applied to MHC-ZH unchanged.
Every such application therefore **over-states** the MHC-ZH cost. The two exceptions are `U9`,
measured separately per dataset, and `U7`/`U11`, which are dataset-independent.

**`U9` correction, disclosed.** The first `GATE-DEVFID` timing (`3.32 / 3.19 s`) was a
**failure path**: `headspace_fidelity.py` defaults to `--seeds 0,1,2` and only seed-0 full
mints existed, so it errored and wrote no file, while my shell captured `echo`'s exit status
rather than python's. Re-run with `--seeds 0`, both datasets exit `0` and write their JSON.
Round 2 asked whether another unit could carry the same defect: `U5a`, `U6` and the mints are
independently corroborated (round 2 reproduced the parity and `ρ` outputs; fold parity passed
in all seven mints), and for the remainder the freeze record will state the exit-status
discipline under which each was timed.

### 7.8 Dry-check cost, disclosed

v3's measurements added ≈ **4 wall-minutes / ≈ 12 CPU-minutes** (the null-contract battery, the
parity re-verification, the startup and build units), all `$0`, zero GPU. Cumulative across
v1–v3: ≈ 13 wall-minutes / ≈ 64 CPU-minutes. **Round 2's M-3, adopted:** the honest framing is
*disclosed at the same time as the result*, not *"raised rather than resolved silently"* — the
CPU-cap conflict was knowable from C09's banked mint costs before the first burn, and saying
otherwise overstated the process discipline. Both rounds ruled the underlying trade correct: a
standing `TARGET_STATE.json` rule beats a task brief's CPU cap.

---

## 8. Compute projection — measured unit × explicit count

Every unit is measured in §7 at real scale. No budget sits inside the projection, and **no
phase is derived by ratio** (round-2 I-2 is discharged by deletion — the phase that used a
ratio no longer exists).

| phase | count | unit | product |
|---|---|---|---|
| **1** Head-N mints, HateMM fold | `3 × 5 = 15` | 40.39 s | `605.9 s` |
| **1** Head-N mints, HateMM full | `3` | 49.30 s | `147.9 s` |
| **1** Head-N mints, ZH fold | `3 × 5 = 15` | 34.40 s | `516.0 s` |
| **1** Head-N mints, ZH full | `3` | 38.87 s | `116.6 s` |
| **1R** Head-R mints, HateMM | `3 × 5 = 15` | 40.39 s (§3.3) | `605.9 s` |
| **1R** Head-R mints, ZH | `3 × 5 = 15` | 34.40 s | `516.0 s` |
| **1b** key forwards `(30×3)+(6×4)+(30×2)` | `174` | `U1` 0.0461 s | `8.0 s` |
| **1c** ro cache loads, per process | `66` | `U8` 0.033 s | `2.2 s` |
| **1d** `GATE-SHA`, once in the driver | `1` | `U7` 0.12 s | `0.1 s` |
| **2** head-space votes, 1024-d arms | `4 × 60 = 240` | `U2a` 0.00305 s | `0.7 s` |
| **2** head-space votes, 2048-d arms | `9 × 60 = 540` | `U2b` 0.00629 s | `3.4 s` |
| **2** `GATE-FLOOR` native vote | `1 × 30 = 30` | `U2a` 0.00305 s | `0.1 s` |
| **2b** head-space arm construction (round-2 I-4) | `12` cells | `U10` 0.1873 s | `2.2 s` |
| **2R** raw votes, 7168-d arms | `4 × 10 = 40` | `U2c` 0.04239 s | `1.7 s` |
| **2R** raw votes, 14336-d arms | `9 × 10 = 90` | `U2d` 0.08098 s | `7.3 s` |
| **2Ra** raw arm construction | `2` datasets | `U5b` 4.63 s | `9.3 s` |
| **2C** `GATE-C01PARITY` | `2` datasets | `U5a` 11.27 s | `22.5 s` |
| **2C** `GATE-DUALPATH` (HateMM only: second build + compare) | `1` | `U5b + 0.21` = 4.84 s | `4.8 s` |
| **2D** `ρ`, raw + head cells | `2 + 12 = 14` | `U6` 0.62 s | `8.7 s` |
| **3** shuffled-pair null draws | `256 × 3 × 2 × 2 = 3072` | `U4` 0.08908 s | `273.7 s` |
| **4** bootstrap comparison-cells | `23 × 2 ds × 2 lineages = 92` | `U3` 0.126 s | `11.6 s` |
| **5** head-space null-row sensitivity | **0 — the leg does not exist** (§3.7) | — | `0.0 s` |
| **6** `GATE-DEVFID` | `3 + 3` | `U9` 3.70 / 3.49 s | `21.6 s` |
| **7** `GATE-SELFTEST` net identity | `13 arms × 12 cells = 156` | arithmetic on banked vectors | `< 0.1 s` |
| | | **corroborating total** | **`2886.3 s = 48.1 min`** |
| | | **conservative (× 1.25)** | **`3607.9 s = 60.1 min`** |

**Declared slack, outside the projection:** `30 s` for ledger aggregation and JSON emit —
labelled slack, not a measurement, deliberately not summed.

**Peak RSS ≈ 1.3 GiB** (measured 1.25 GiB for the dominant unit; raw arm matrices add
`744 × 14336 × 4 B = 43 MB` each). Request 32 GB.

**Where the risk sits.** Mints are `86.9 %` of the total (up from `82 %` in v2, because
Head-R is now priced at Head-N's units) and are measured directly. Phase 3 is `9.5 %`; a 2×
miss moves the total to `3160.0 s = 52.7 min`, a 5× miss to `3981.1 s = 66.4 min`.
**If the realized cost exceeds the conservative total by more than 2×, that is itself a
reportable process finding.**

**M-1, adopted:** Phase 3 is `3072 × 0.08908 = 273.654 → 273.7 s`, rounded like every other
product rather than truncated as v2 had it.

---

## 9. Heartbeat specification

* One progress file `$BASE/progress/C06_PROGRESS.txt`, created by the sbatch driver before the
  first python process starts.
* Every python process appends through a handle opened `buffering=1`.
* Each line: ISO-8601 timestamp · phase · units done / total · elapsed · elapsed ÷ **§8's
  frozen projected** value.
* Granularity: one line per mint (66), **one per training epoch within a mint**, one per
  `(dataset, seed, lineage)` arm block (12), one per 32 null draws (96), one per bootstrap
  block, one per gate, one per verdict field.
* The bash driver **also** echoes a line per mint, unbuffered.
* **Round 2's free addition, adopted: the HALT path names *which gate* failed in its final
  line**, so a HALT is distinguishable from a crash without reading the JSON.
* Round 2 checked the intervals: with the epoch line inside the mints, **no interval exceeds
  ~15 s** (longest un-instrumented span is `GATE-C01PARITY` at `11.27 s`, `14.1 s` under the
  `× 1.25` factor).
* **What the code-review lineage must verify** (a design review cannot): the handle is opened
  `buffering=1` and never re-wrapped; the driver's echo is unbuffered and does not inherit a
  block-buffered stdout under `sbatch`; all 66+ processes **append** rather than truncate and
  concurrent appends do not interleave partial lines; the HALT path writes its final line
  before exit; and the `elapsed ÷ projected` denominator is §8's frozen number.

---

## 10. Scope of any verdict

### 10.1 The prompt/readout-span confound

`generate_VideoMLLM_embedding_readout_HF.py:73-89` defines
`("ro_L24", "baseline", "prefix", "response", LAYER_MID)` versus
`("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`: the `ow_` cell changes
**the prompt kind and both readout spans**. **No result of this battery can attribute an
effect — or its absence — to the prompt alone.**

### 10.2 What a CLOSE would and would not close

A CLOSE closes **C06's first-order (tangent/chord) prompt-orbit route in the fold-head arena
at `ro_L24`, on `HateMM (-LoRA-curric)` at `n = 743` and `MHC_zh (-LoRA)` at `n = 579`, under
BOTH a native-trained deployed head applied out of domain AND an `ro_L24`-trained in-domain
head.** The `GATE-DOMAIN` recovery fraction must be stated in the same sentence. It does
**not** establish:

* anything about **curvature** — two prompt points give a chord; ≥ 3 require extraction;
* anything about a head **retrained per arm** — F66's trained-reshaping caveat stands;
* **anything about the per-modality contrast C01 measured** (round-2 H-3): the deployed head
  fuses image and text internally (`mlp(normalize(img_proj(·)) ⊙ normalize(text_proj(·)))`),
  so **every head-space arm is a post-fusion, one-block analogue** of C01's per-modality
  two-block contrast. C01 contrasts *before* fusion (`contrast_blocks:1242-1270`); this
  battery contrasts *after*. The one-block reading is **forced by the architecture, not
  chosen**, but a CLOSE will be read as "C01's battery re-run in the fold-head arena" and what
  it measures is not the transform C01 scored `0.8505 / 0.8846`;
* anything about a **different readout span**, **L28**, or the **test split**.

### 10.3 What a SURVIVE would license

Only that C06 *"has earned its extraction"*: the `1.7–2.5 GPU-h` bounded extraction may be
**proposed** under `iteration_8_stage0_bounded_extraction_amendment`, with its own
preregistration, design review, separate code/resource review lineage and authorization.
**A SURVIVE is not a Stage-0 PASS and authorizes no GPU.**

### 10.4 Bans checked

F80's object is prompt **language**; F70's is individual **readout cells**; C06's is the
**relation between** two cells. Both rounds tested and confirmed the object-mismatch warrant.
The multi-prompt **ensembling** carve-outs have C14 as their object and are not relied on.
`endpoint_std` and `endpoint_ow` **are** literally F70's two cells, entering here as
**controls**, not claims. No ensemble of prompt predictions is formed: `avg_score` is C01's
own frozen `gain_control`.

**Hard constraints: none touched** — no OCR, no cross-dataset mixing, no external API,
single-dataset train split, parent-video binary label only, no ensembles, no size scaling,
SLURM-only. Round 2 re-checked this against the amendment's `d_no_other_relaxation` and noted
the one residual cross-dataset object, the pooled `ρ*` bar — now per-dataset (§6.1).

---

## 11. Frozen imports and inputs — sha256, measured 2026-08-04

**Imported unmodified, sha256 asserted at run time:**

| module | sha256 |
|---|---|
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` |
| `scripts/analysis/c01_policy_contrast_a0.py` | `d2b9c2ff909c07518ae35526db9550df655fb4af395cc7a0899f83e48db1b855` |
| `scripts/analysis/c09_guard/c09guard.py` | `aed50842c232105f1b06182aa89512ee89dd050bdcaedec2706062c9d745f062` |
| `scripts/analysis/c09_guard/sitecustomize.py` | `b238789fd80076b0b890c4894fd8b69255792af51c80cd9fe2d6db6c53383850` |

**Read for definitions, thresholds and provenance:**

| file | sha256 |
|---|---|
| `configs/c01/c01_a0_v4.json` | `2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6` |
| `scripts/analysis/c01_policy_contrast_a0_v4.py` | `3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514` |
| `scripts/analysis/c01_policy_contrast_a0_v3.py` | `40b35eee2fb6fdbdb21fe9b4acfdcebf003c121c76492b898fbd2ea9b8c34dfb` |
| `configs/c01/c01_a0_v3.json` | `4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5` |
| `configs/c01/c01_a0_v2.json` | `f3997bddb4788d451ae5f90d9d03d096df3de383f8133a6d3818d97a241563f5` |
| `scripts/analysis/c02_a0_mint.py` | `e6430b76b7ccdd831ddb9939500aa24ea70d9662b62b955a2a11273a3b00ac1b` |

**The v4 → v2 chain.** `TARGET_STATE.json::c01_a0_v4_typed_audit_repair` pins v4's config and
analysis, namespace `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4`. v4's `frozen_v3` →
v3 (`reuse_policy: sha256_exact_import_then_v4_audit_schema_override_only`,
`scientific_thresholds_exact: true`); v3's `scientific_base` → v2, same flag. In source,
`c01_policy_contrast_a0_v4.py:44-52` sha-checks `_v3.py` and `_v3.py:29-50` sha-checks
`c01_policy_contrast_a0.py` — the file this battery **imports**. Round 2 verified both hops
are sha-gated in code, not only in config. **Config chain v4 → v3 → v2; algebra chain
v4 → v3 → base.**

**Input caches:**

| file | sha256 |
|---|---|
| `HateMM/train_…-LoRA-curric_HF-ro_L24.pt` | `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f` |
| `HateMM/train_…-LoRA-curric_HF-ro_ow_L24.pt` | `60054f3be1204ca7bf2ac55b9bae6a88dd84d9dda35b0225f1ca27ce61977f4e` |
| `MHC_zh/train_…-LoRA_HF-ro_L24.pt` | `1d33fe5d69083479f0b6968a924578770364c00ca78c37cfef664bb4b6221c06` |
| `MHC_zh/train_…-LoRA_HF-ro_ow_L24.pt` | `3ad1309dc75001820318e3e9a073b781d28ee0afc2b879571d063a276b8d2a23` |
| `HateMM/train_…-LoRA-curric_HF.pt` (native) | `5e80f39327a743144067857e6f8c9f0c909e3131bdc13bcb063be6abc333e7cf` |
| `HateMM/dev_seen_…-LoRA-curric_HF.pt` (native) | `46ee4fd9fcaec80b7859a5e4c18b76e84b4020fa242ced802f289f790e4d7cb0` |
| `MHC_zh/train_…-LoRA_HF.pt` (native) | `b2e8e78d19c71d2ca674903586d53ca171c33a539956ee37c1c61f44a5e01f1d` |
| `MHC_zh/dev_seen_…-LoRA_HF.pt` (native) | `4c07af75098391c999013e1cf6fb7ffe8fac29546d9ce329d51004a37e4f5d3c` |

Plus the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` and the ten banked
`vsw_ckpt/{hatemm,zh}/f{0..4}.npz`.

**New code, confined to the battery:** `scripts/analysis/c06_falsifier_mint.py` (**the single
shared driver**, §3.3), `scripts/analysis/c06_falsifier_arena.py`,
`configs/c06/c06_falsifier.json`, `scripts/slurm/c06_falsifier_cpu.sbatch`.

---

## 12. Label and split discipline

**Test contact: none, enforced in three layers** — `headspace_mint.py:106-116`'s `torch.load`
guard; the driver's `split == "train"` assertion on every ro-cache load; and the frozen
`c09_guard` `sitecustomize` installing an `open()`-level, component-wise, repo-scoped
predicate at interpreter startup in **every** process. Round 2 confirmed the `test_seen` ro
caches on disk create no opening.

**`GATE-LEDGER` — C09's declared-count predicate set, with the process count binding:**

| predicate | expected | binding? |
|---|---|---|
| `test_path_opens` | **0** | **yes** |
| `test_label_materialisations` | **0** | **yes** |
| `dev_path_opens` | nonzero, declared — 36 Head-N mints + `GATE-DEVFID` reads | reported |
| `dev_label_materialisations_outside_decisions` | 36, one per Head-N mint | reported |
| `dev_or_test_labels_into_decision_quantities` | **0** | **yes** |
| `banked_trainlog_opens` | declared, `GATE-DEVFID` only | reported |
| processes reporting | **66 mints + 6 fidelity + 1 arena** | **yes — HALT on any mismatch** (round-2 I-6) |
| predicate coverage | re-derived in-job | reported |

The process-count leg is binding because §5.6's absence rule needs an enforcer: a silently
missing lineage would otherwise make SURVIVE vacuously false and supply half of CLOSE.

**Dev labels, stated correctly.** `headspace_mint.py:199` loads `dev_seen_*.pt` on **every**
mint, `:322` writes `lab_dev` into every `.npz`, and at `fold == −1` (`:229`) the real dev
split **is** the training dev set, so dev labels enter `run_rac.main` on 6 of the 36 Head-N
mints. Head-R mints open no dev file. None reaches a decision quantity — which is why the
binding leg is the *into-decision* zero, not a blanket dev zero.

**No selection anywhere.** Every threshold in §5 and §6 is C01's frozen value, C09's banked
constant, or fixed in §4/§6.1 from banked label-free measurements before the run. §5.8 lists
what is deliberately not carried.

---

## 13. Execution boundary

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** Both rounds independently confirmed: F88's `$0` forensics name no
non-SLURM channel and price a 52-second process, no precedent for a ~48-minute job; and
CLAUDE.md's standing rule plus C01's frozen `execution.require_slurm = true, cpu_only = true,
required_cpus = 8` (with `required_memory: 32G` in v3) plus the C02/C09 precedents all agree.

**Cloud routing inapplicable:** `GATE-FLOOR` anchors to six floors measured locally on
`foscsmlprd01`, and the same-table-same-hardware ruling would require re-minting all six on
cloud hardware, costing more than the job. `squeue` is read at submission time. v1's third
reason (*"at 44 min this is not long-running"*) stays **withdrawn**.

**Not authorized by this document.** Required before anything runs: an independent design
review to GO (0C/0H/0I), a **separate** code/resource review lineage over the executable, and
main-dialogue authorization.

---

## 14. Cumulative disposition — rounds 1 and 2

### Round 2 (13 findings + 3 Minor) — all adopted

| finding | disposition | where in v3 |
|---|---|---|
| **C-1** head-space arms unbuildable through `l2_rows` | **ADOPTED** — null contract re-derived (§3.7); registered null physically removed from the arm arena; all 13 arms verified buildable at `n = 743, None`; three populations named and gated by `GATE-POP` | §3.7, §6, §7.4 |
| **C-2** `GATE-ARENA` lower bound self-defeats | **ADOPTED** — lower bound restricted to `endpoint_std` (C09's scope); `≤ 0.98` kept on all three; the real arms' lower side belongs to `GATE-ARMVIAB` | §6.3 |
| **C-3** C01's zero contract not portable | **ADOPTED** — `GATE-DUALPATH` re-scoped to the raw leg where it is defined and made the run-time proof of the row-subset identity; `GATE-ZEROMASK` restated as feature-space only; `GATE-SHUFFLEFIX` deleted as vacuous and replaced by `GATE-NULLREMOVED`; Phase 5 deleted | §3.7, §6, §8 |
| **H-1** lineage disjunction uncorrected | **ADOPTED (option b)** — one Holm family of **92** per dataset spanning both lineages | §5.5 |
| **H-2** Head-R has no anchor and does not run the banked script | **ADOPTED, and stronger than the fallback** — **one shared driver** for both lineages with `--train-cache` as its only lineage-varying argument, so `GATE-FLOOR` anchors the driver at zero cost; v2's "only variable" sentence made true of the battery; Head-R re-priced at Head-N's units (`+146.9 s`) | §3.3, §8 |
| **H-3** post-fusion contrast undisclosed | **ADOPTED** — scope bullet in §10.2 and the matching clause in §3.4 | §3.4, §10.2 |
| **I-1** round-1 I-3 leg (c) unrepaired | **ADOPTED** — §7.2 states the units are full-process wall, how they were timed, and gives the `40.39` vs `33.0 s` evidence plus the measured `3.05–3.18 s` startup; **no line added, because adding one would double-count** | §7.2 |
| **I-2** Phase 5 derived by ratio | **ADOPTED by deletion** — the phase no longer exists (§3.7); no phase in §8 is ratio-derived | §8 |
| **I-3** units not attributed to a dataset | **ADOPTED** — every unit labelled with its dataset, and the conservative-application convention stated once and applied uniformly | §7.7 |
| **I-4** head-space arm construction uncounted | **ADOPTED** — Phase 2b, `12 × U10` | §8 |
| **I-5** `GATE-ZEROOP` not "strictly stronger"; false-HALT risk | **ADOPTED** — wording corrected to logically independent; tie diagnostic pre-registered with report-not-HALT semantics | §6.5 |
| **I-6** absence not a HALT trigger; process count not binding | **ADOPTED** — §5.6 adds the absence rule; `GATE-LEDGER`'s process count is binding | §5.6, §12 |
| **I-7** pooled cross-dataset `ρ*` | **ADOPTED** — per-dataset `ρ*` (HateMM `0.9681`, ZH `0.9772`), runner-ups recorded | §6.1 |
| **M-1** Phase 3 truncated | **ADOPTED** — `273.7 s` | §8 |
| **M-2** `displacement`'s comparator asymmetry | **ADOPTED** — disclosed | §5.8 |
| **M-3** overrun framing | **ADOPTED** — "disclosed at the same time as the result" | §7.8 |

**Round-2 free strengthening adopted:** `GATE-SELFTEST` (the net-fix identity), which S6 gives
an object; and the HALT line naming which gate failed (§9).

### Round 1 (23 findings) — 20 verified adopted by round 2; the 3 reopened are now repaired

| finding | round-2 audit | v3 |
|---|---|---|
| **C-3** unanchored arm algebra | NOT ADOPTED ON THE VERDICT PATH | **repaired** — the path now executes (§3.7/§7.4-g), and §3.4 states correctly what two-block parity does and does not buy |
| **C-1 companion** both real arms clear majority | NOT ADOPTED IN EFFECT — unreachable behind `GATE-ARENA` | **repaired** — §6.3 removes the blocking gate; §6.2 is now operative |
| **I-3** three per-process loops | PARTIAL, 2 of 3 | **repaired** — leg (c) settled in §7.2 with measurement |
| C-1 dispersion, C-2 OOD transplant, H-1…H-6, I-1, I-2, I-4…I-10, M-1…M-4 | **VERIFIED adopted** | carried forward unchanged except where a round-2 finding refines them |

**Round-1 and round-2 rulings carried without change:** the direction of "conservative"; A7 is
not an obstacle; per-arm retraining stays excluded; the max is the right `ρ*` order statistic;
SLURM and the login-node dismissal; the untrained-head blindness discipline; HALT semantics;
§5.8's inapplicability reasoning; S6's net-fix reference.

---

## 15. Open issues for round 3

1. **The null contract (§3.7) is the substantive change and deserves the sharpest reading.**
   Three populations now coexist: full `n` for head training and `GATE-FLOOR`, `743 / 579` for
   the arm arena. Round 3 should check that no gate, statistic or comparison silently mixes
   them, and that `GATE-POP` is sufficient to detect it if one did.
2. **Is removal verdict-neutral?** §3.7 argues it removes a bias rather than creating one,
   because leaving the null in makes it a live neighbour in eleven control arms and a dead key
   in one real arm. Round 3 should test that argument, including whether removing one bank item
   from a 743-item bank can shift a top-20 neighbourhood in a way that favours either lane.
3. **The dataset asymmetry.** HateMM runs at `743`, MHC-ZH at `579` with no removal. §3.7 argues
   the conjunction-of-independent-verdicts structure contains it. Round 3 should rule.
4. **`GATE-DUALPATH`'s new role.** It is now the run-time proof of the row-subset identity
   rather than C01's masked-vs-removed prediction equivalence. Is that a faithful use of C01's
   `displacement_registered_null_exclusion`, or a different gate wearing its name?
5. **Head-R still has no *scientific* anchor**, only a harness anchor via the shared driver.
   Round 2 ruled the harness anchor sufficient *given* one driver. Round 3 should confirm that
   ruling survives now that the driver is shared in fact rather than in claim.
6. **`GATE-ZEROOP`'s tie diagnostic** (§6.5) introduces a report-not-HALT branch. Round 3
   should check it cannot be widened to swallow a genuine mismatch.

---

*No GPU, SLURM, Modal, teacher call, model load, training of any deployed arm, cache write,
test-split access, job submission or commit occurred in producing this document. Login-node
dry-check processes only (§7.8). `TARGET_STATE.json` was read and not modified. v1 and v2 are
unmodified.*
