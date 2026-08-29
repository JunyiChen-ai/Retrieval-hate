# C06 `$0` CPU falsifier — preregistration **DRAFT v4** (2026-08-04)

**SUPERSESSION.** Supersedes `C06_FALSIFIER_PREREG_DRAFT_V3.md`, which supersedes v2, which
supersedes v1. All three remain on disk **unmodified** as the record of what each round
reviewed. Reviews of record: `C06_FALSIFIER_PREREG_REVIEW.md` (round 1, REVISE 3C/6H/10I+4M),
`…_R2.md` (round 2, REVISE 3C/3H/7I+3M), `…_R3.md` (round 3, REVISE 2C/1H/6I+4M). This is a
complete standalone document.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is frozen,
`TARGET_STATE.json` is untouched, nothing is committed.

**Disposition: all 13 round-3 findings ADOPTED, 0 rebutted.** Round 3's disposition audit was a
clean sweep — 16 of 16 round-2 adoptions verified real, all three reopened round-1 items
genuinely repaired, no disguised rebuttal — so v4 changes nothing that rounds 1–2 settled.
Cumulative table in **§14**.

**What v4 changes.** Both round-3 Criticals live in the seam v3's null contract opened, and
neither needs a redesign. **C-1:** `zero_mask = None` is not admissible to `prepare_views` — its
derived-mask check compares against the raw argument, so `np.array_equal(…, None)` is always
`False` and the call dies on *any* dataset. The contract is now written in explicit boolean
arrays throughout. **C-2:** removing row 355 moves HateMM's majority rate from `0.5995` to
**`0.6003`**, and two gates were applying the 744-row constant to the 743-row arena; the arena
majority is now a named population-derived constant and `GATE-POP` checks the class counts that
produce it. Both fixes are **compute-neutral** — the projection is unchanged at `2886.3 s`.

---

## 1. What this falsifier is, and what authorizes it

C06 (*Prompt-Orbit Tangent/Curvature*) is **not an active candidate**; its registry status is
`gated_on_zero_cost_falsifier`
(`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`). What the queue has reached
is **C06's falsifier**, not C06.

**The unblock condition, verbatim** (`…dispositions.gated[0].falsifier_spec`):

> re-run C01's real-displacement-versus-matched-norm-orthogonal-rotation battery in the
> FOLD-HEAD ARENA on the already-banked `ro_*` caches. Zero GPU, zero extraction, minutes of
> CPU on `scripts/analysis/headspace_{mint,arena}.py`, which exist and are banked. If the
> rotations again match the real displacement in the deployed head space, C06 closes for `$0`
> and the `1.7-2.5 GPU-h` of extraction is never queued; if they do not, C06 has earned its
> extraction

**The two binding design constraints, verbatim** (`…falsifier_design_constraints`):

> its pre-registration must (i) use the per-dataset adapter lineage that ACTUALLY EXISTS —
> HateMM has only `-LoRA-curric` ro-caches, MHC_zh has only `-LoRA`, one lineage each, not a
> matched pair (correction V-8); and (ii) declare the prompt/readout-span confound, because
> `generate_VideoMLLM_embedding_readout_HF.py:73-89` shows the `ow_` cells change the readout
> span as well as the prompt — the same confound C01's review already narrowed its claim for

Both honoured — §3.1 and §10.1 — and verified as *honoured, not merely mentioned* by all three
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

`gain_over_strongest_control` `−0.0094` / `−0.0256`; `pass: false`; `decision.continue = false`.

**Round-14's sharpening, adopted:** `orthogonal_blocks()` (`c01_policy_contrast_a0.py:1272`) is a
**Givens mixing of the two endpoint blocks**, so the six "random rotations" are six angles on
**one parameter family** that also contains the primary — `θ = 45°` **is** `common_displacement`,
`θ = 0` **is** `endpoint_concat`. Re-measured on the raw L24 features: `8.941e-08` (θ=0, both
datasets), `1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH) at θ=45.

**Why a re-run in a different space is the right instrument.** C01's arena is **raw dev keys**
(`n_dev` 107 / 78), not the fold-head path; the registry's `unified_pilot_gate.arena` requires
*"the actual fold-head/deployed-head path"* and F113 marks the raw-KILL direction **NOT
ESTABLISHED**. All three rounds ruled the arena reading correct.

---

## 2. The process rules that bind this design

`process_rule_compute_projection_and_heartbeat_2026_08_04` names this falsifier in
`applies_immediately_to`.

| rule | discharged in |
|---|---|
| **R1** measured-unit-cost × explicit-count projection; no reduced-scale extrapolation; no budgets inside the projection; every unit enumerated | **§8** — round 3 hunted for an uncounted loop and found none material |
| **R2** line-buffered per-phase heartbeat | **§9** — no interval exceeds ~15 s (round-3 verified) |
| **R3** (F114) dry execution exercises the **first real operation of the payload path** | **§7** |
| **R4** (`feedback-separate-code-review-lineage`) a design GO does not review the implementation | **§13**, with round 3's twelve-item list |
| **F118 erratum lesson** never let boilerplate describe a leg that did not run | **§3.7, §5.9, §8 Phase 5** |

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
numbers is made; every decision quantity is within-dataset, within-seed, within-lineage, and the
two-dataset requirement is a **conjunction of independently computed verdicts**. Round 2 removed
the last cross-dataset object (the pooled `ρ*`); round 3 confirmed none remains.

**Provenance.** The four L24 files are byte-identical to the ones C01 measured; the HateMM
`ro_L24` digest equals C01 v3's `diagnostic_train_cache_sha256` in full 64 hex (§11).

**L28 is not used** (round-1 I-8). **Splits:** `train_*.pt` only for the ro caches; the native
`dev_seen` cache is opened by `headspace_mint.py:199` on every mint, is listed in §11 and is
covered by `GATE-SHA`. No `dev_seen_*-ro_*` file is opened by any phase; the `test_seen` ro
caches are opened by nothing.

### 3.2 The head, the folds, and the vote

* **Fold contract.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the **full**
  train split, asserted against the banked `vsw_ckpt/<ds>/f{0..4}.npz` by
  `headspace_mint.py:203-216`. Unchanged by §3.7's null contract.
* **Head.** The deployed-recipe RGCL head, re-minted on CPU (F78). Bank = fitting pool, queries =
  held-out fifth, every item held out exactly once.
* **Vote.** `mechfix_ops.deployed_vote(..., topk=20)` — all three rounds verified this is
  numerically the operator C01's config specifies.
* **F88's caveat** — *"a CPU-trained arm must be paired against a CPU-TRAINED FLOOR"* — satisfied
  by construction.

### 3.3 Two head lineages, one driver

**Round 1's C-2**, measured: the head is trained on the native cache but forwarded over `ro_L24`
features **near-orthogonal** to it (median `cos(native_img, ro_L24_img)` = `0.0234` HateMM /
`0.0373` MHC-ZH, both caches unit-norm). v1's claim that this was "the banked C02 house pattern"
is **withdrawn and stays withdrawn**: `c02_a0_mint.py:214` keeps `img_feats` **native** on every
view and `:68` refuses any view file carrying an image stream.

| lineage | head trained on | banked anchor | in-domain | mints |
|---|---|---|---|---|
| **Head-N** | native deployed cache | **`GATE-FLOOR`** | no | 36 = 2 ds × 3 seeds × (5 folds + 1 full) |
| **Head-R** | `train_<model>-ro_L24.pt` | via the shared driver | **yes** | 30 = 2 ds × 3 seeds × 5 folds |

Head-R needs no `fold = −1` head: the deployed-configuration head exists only to feed
`GATE-DEVFID`, which compares against banked **native** trainlogs that have no Head-R counterpart.

**The shared driver (round-2 H-2).**

> **ONE driver, `scripts/analysis/c06_falsifier_mint.py`, serves both lineages.** It imports
> `headspace_mint` with its sha256 asserted and reuses its dataset table, deployed CLI, fold
> assignment, fold-parity assertion, dummy-dataloader construction, monkeypatches, seeding and
> DET-1 contract unchanged. **`--train-cache` is its only lineage-varying argument.**

Because Head-N runs through that same driver and must reproduce the six banked `GATE-FLOOR`
anchors, **`GATE-FLOOR` anchors the driver, not merely Head-N's science** — the anchor round 2's
§15.6 ruling asked for, at zero additional cost.

**Round-3 H-1: the one behaviour that follows from "unchanged", stated rather than denied.**
v3's §12 said *"Head-R mints open no dev file."* That is **false under this very claim**, and the
claim is the one worth keeping. `headspace_mint.py:199` is
`dv = load_split(cache_dir, "dev_seen", model_name)` — **unconditional on every mint**, before the
`fold` branch — and `:322` writes `lab_dev` into every `.npz`; `model_name` comes from the frozen
dataset table, so a `--train-cache` override leaves the dev load pointed at the **native**
`dev_seen`. Under a driver that reuses that code unchanged, **all 66 mints open the native dev
cache and materialise `lab_dev`**, not 36. §12's declared counts are corrected to 66 and made
**binding**. `--train-cache` remains the only lineage-varying *argument*; the dev load is
identical on both lineages, which is what "unchanged" means and is why the sentence survives.
No dev label reaches any decision quantity (§12).

**Pricing.** v2 priced Head-R from a scratchpad harness that skipped the fold-parity `npz` loads,
the native `dev_seen` load and the `npz` save (`37.46 / 27.54 s`). The real driver does all of
those, so **both lineages are priced at `headspace_mint.py`'s own measured units
(`40.39 / 34.40 s`)**.

### 3.4 The arm builder — one generic block-list construction

**Round 1's C-3.** v1 defined the arms afresh; the collapse from C01's two-modality `paired_key`
to one block was a *choice* no gate could check, and the choice was wrong (it omitted
`fuse_modalities`' outer per-block normalisation and worked in `float64` where `l2_rows` returns
`float32`).

**The builder.** One construction, parameterised by an ordered list of blocks, in which every
normalisation is C01's `l2_rows` called through the **imported** `c01_policy_contrast_a0`:

```
fuse(blocks) = l2_rows(concat[ l2_rows(b) for b in blocks ])
paired(A,B)  = fuse([ l2_rows(concat[ l2_rows(A_m), l2_rows(B_m) ]) for m in blocks ])
build_views(std_blocks, ow_blocks, angles) -> the 13 arms
```

Two blocks `[img, text]` ⇒ C01's `prepare_views`, **bit-exactly** (`max|diff| = 0.000e+00`, 13
arms, both datasets) — independently reproduced from this prose alone by **both** round 2 and
round 3. One block (the fused head key) ⇒ the head-space arms.

**What the anchor buys, stated as round 2 ruled and round 3 confirmed.** Two-block parity pins
`l2_rows` itself, concatenation order, the contrast definitions, the Givens mixing and its angle
convention, the arm-name→formula map, the `float32` dtype and the θ=0/θ=45 identities. It does
**not** pin the one operation C-3 named: at one block the outer `fuse` normalisation
re-normalises an already-unit vector, differing from v1's rejected `pair` by a fraction of a
`float32` eps. **What the restored normalisation actually fixed for the head-space arms is the
dtype.** The block count is **forced by the head's architecture** — `classifier_hateClipper`
emits a single fused 1024-d vector, so no two-block head reading exists to get wrong.

**Deferred-import note.** `c01_policy_contrast_a0.py:387` sets `np = torch = faiss = None` and
binds them only inside `import_compute_modules(config)`; the battery must call it before touching
the algebra.

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

### 3.6 The raw leg — gate-only, non-decisional, on the same rows

The same builder runs on the raw L24 features (no head) and votes in the **same folds**, on **the
same arena population as the head leg** (§3.7). It renders no verdict and enters no decision rule
or multiplicity family. Its jobs are the gate discriminators: `ρ_raw` for `GATE-ORBITDISP` and
the raw accuracies for `GATE-ARMVIAB`. `GATE-POP` asserts the head leg and the raw leg run on
identical row **index sets**, not merely equal counts.

### 3.7 The null contract, and the mask convention

**The defect, measured.** `classifier_hateClipper.__init__` (`src/model/classifier.py:80-81`)
builds the projections **with the default bias**, so `head_f(0,0)` is a non-zero constant —
measured non-zero at every torch seed tested and under both emitter conventions
(`0.58–0.65` observed; the value is initialisation-dependent, the *sign of the result* is not).
HateMM train row 355 (`hate_video_95`, C01's registered `authorized_null`) is bit-identically
zero in both modalities of **both** ro caches, so `h_std[355] == h_ow[355]` **exactly**, and it
is the **only** row on either dataset with that property. Hence in head space every
**endpoint / common / rotation** block is non-zero at row 355 while the **displacement** block is
exactly zero — and `l2_rows:1193-1194` is fail-closed on precisely this. Measured, each mask
choice kills the other, and **`common_displacement`, C01's primary, dies under both** (§7.4).

**It is not only an execution problem.** With the row *masked* rather than removed, its key in
the eleven control arms is an ordinary unit vector that faiss returns as a genuine top-20
neighbour, while in `displacement` it is a zero key. Round 3 sharpened why that is worse than
"contributes nothing": a zero key has inner product `0` with every query, which under
`IndexFlatIP` ranks **above any negatively-similar candidate**, so the null would **displace a
real neighbour out of the top-20** while contributing `(2y−1) × 0 = 0` to the vote. An item whose
features are a known extraction failure would corrupt eleven control arms and one real arm
differently — an asymmetry lying exactly along the comparison that renders the verdict.

**The contract. Four objects, each with its population and its mask argument.**

| object | population | `zero_mask` argument | why |
|---|---|---|---|
| head training | full `n` (744 / 579) | — (no C01 algebra) | the deployed recipe and the fold contract are unchanged |
| `GATE-FLOOR` | full `n`, native keys | — | the six banked floors were computed on this population |
| `GATE-C01PARITY` / `GATE-ROWSUBSET` two-block build | `n = 744` (HateMM) / `579` (ZH) | **one-hot `{355}`** (HateMM) / **`np.zeros(579, dtype=bool)`** (ZH) | the only population where C01's masked contract is defined |
| **arm arena — head leg AND raw leg** | **`n = 743`** (HateMM) / `579` (ZH) | **`np.zeros(n, dtype=bool)`** | the null is physically absent, so the all-False mask is *correct*, not a workaround |

**The mask convention (round-3 C-1), stated once and binding everywhere.** The `zero_mask`
argument is **always an explicit boolean array** — never `None`. The two frozen functions differ
and the difference is not cosmetic:

* `l2_rows:1187-1188` **normalises** `None` into a zeros array, so `None` is admissible there;
* `prepare_views:1381-1386` compares its derived masks against the **raw argument**
  (`np.array_equal(<bool array>, None)` is `False` for every arm), so `None` is **inadmissible**
  and the call dies *on any dataset*, with or without a null row.

C01 itself never passes `None`: it builds `np.zeros(n, dtype=bool)` at `:2224` and hands that
array to `prepare_views` at `:2304-2306`. **Measured (§7.4): `prepare_views(…, None)` dies on
MHC-ZH; with the explicit all-False array it succeeds; and on the HateMM arena population
(`n = 743`) the all-False array succeeds** — because the zero row really is gone. v3 wrote the
contract in `None`; v4 writes it in explicit arrays, and this paragraph exists so the distinction
is not lost again in the code lineage.

**Why removal is legitimate, on four counts** (round 3 tested each and ruled for all four):

1. **Label-free.** Row 355 is selected by an exact-zero *feature* property, is C01's pre-existing
   frozen `authorized_null`, and is the only row with that property — the selection is forced,
   not chosen. No label is consulted.
2. **Verdict-neutral, and it removes a bias rather than creating one.** The row is dropped from
   the bank **and** the query set of **every arm identically**, so all arms are scored on the
   identical 743 items and a per-lane bias would require the removal to be arm-dependent. It is
   not. Round 3 attempted to construct a lane-favouring mechanism and could not.
3. **Provably a pure row-subset.** Raw arms built at `n = 743` are **bit-identical** to arms
   built at `n = 744` with the one-hot mask, restricted to the 743 surviving rows —
   `max|diff| = 0.000e+00` on all 13 arms, algebra guard bit-identical, every `ρ` unchanged.
   `GATE-ROWSUBSET` asserts this at run time.
4. **Fixed before any trained-head number exists.** The contract turns on the *presence* of a
   bias term, confirmed weight-independent at three seeds.

**The arena class balance is a population-derived constant (round-3 C-2).** Row 355 carries
**label 1**, so removing it changes HateMM's class balance and therefore its majority rate. This
is the one population-dependent *bar* in the design, and v3 left it at the 744-row value:

| dataset | population | pos / neg | majority | `GATE-ARENA` lower bound |
|---|---|---|---|---|
| HateMM | full `n = 744` | 298 / 446 | `0.599462 → 0.5995` | *(used only for statements about the 744-row population)* |
| **HateMM** | **arena `n = 743`** | **297 / 446** | **`0.600269 → 0.6003`** | **`0.6203`** |
| MHC-ZH | full = arena `n = 579` | 180 / 399 | `0.689119 → 0.6891` | `0.7091` |

The correction is `0.0008` and it runs **anti-conservatively**: v3's band `[0.6195, 0.98]` is
looser than the arena's own data supports, and the gap is *reachable* — seed-mean accuracy is a
multiple of `1/(3 × 743) = 0.000449`, and `1381/2229 = 0.619560` and `1382/2229 = 0.620009` both
fall inside `[0.6195, 0.6203)`. Since `GATE-ARENA`'s lower bound is the gate round 2 ruled
*substantively discharges* round 1's C-2 — the OOD-transplant fidelity check, which is *expected
to operate near its bar* — a bar `0.0008` too low is a path by which a collapsed instrument
publishes a CLOSE. **`GATE-POP` now asserts the realised arena class counts `(297, 446)` /
`(180, 399)`**, which converts it from a population check into a population-**and**-constant
check; round 3 named that as what makes it sufficient.

**The dataset asymmetry.** MHC-ZH has no exact-zero row, so its arena is `n = 579` unchanged and
the contract is vacuous there — including the majority rate, which does not move. Round 3 ruled
the asymmetry **contained**, because the two-dataset requirement is a conjunction of
independently computed verdicts and every threshold is per-dataset, **on the condition that the
C-2 fix be applied per dataset** — which the table above does.

**There is no head-space null-row sensitivity leg.** v2 carried one as Phase 5. It cannot exist:
the alternative head-space population is unbuildable. The sensitivity question is discharged
where both populations are defined — the raw two-block anchor — by `GATE-ROWSUBSET`. No gate,
field or boilerplate in this design describes a leg that did not run.

---

## 4. Ambiguities in the written condition

**"Conservative" means *hardest for the falsifier to deliver the `$0` CLOSURE***. Round 1 ruled
this correct on two conditions — disclose what the lean buys (§5.8), and never let it excuse an
arithmetic error. Rounds 2 and 3 each held the design to the second condition (the lineage
disjunction; the majority-rate constant), and both are now discharged.

| # | ambiguity | resolution |
|---|---|---|
| **A1** | *"the rotations"* | C01's `require_primary_above_all_rotation_controls`: the real arm must beat **every** rotation |
| **A2** | *"the real displacement"* | **Both** real arms, disjunctively, multiplicity-corrected (§5.5) |
| **A3** | *"match"* | A tie **closes** C06; C01's own `>` semantics |
| **A4** | *"in the fold-head arena"* | Arena reading ruled correct by all three rounds; the training-set half is resolved by running **both** lineages; *which rows* is settled in §3.7 |
| **A5** | which layer? | **L24 only** |
| **A6** | seeds | Seed-mean over 3 head seeds, plus **3/3** per-seed agreement on the rotation-dominance leg |
| **A7** | *"C06 closes"* | The **first-order (tangent/chord) leg only**. Round 1 ruled A7 is **not** an obstacle |

---

## 5. The pre-registered decision rule

### 5.1 Notation and population

For dataset `D`, lineage `L ∈ {Head-N, Head-R}` and seed `s ∈ {0,1,2}`, each arm `A` yields one
OOF prediction vector over the **arm-arena population** of §3.7 — `743` items on HateMM, `579` on
MHC-ZH — scored against **train-split** labels held out from the head that judged them.
`acc(A,D,L)` is the mean over the three seeds. Write `n_D = |arena(D)|` = `743 / 579`.

* **Real arms** `R = {displacement, common_displacement}`.
* **Rotation family** `Θ` = the six frozen angles.
* **Ordinary controls**, per arm, from C01's own two frozen lists: for `common_displacement`,
  `C = gain_controls ∪ {displacement}` (**six**); for `displacement`, `C = gain_controls`
  (**five**).

### 5.2 SURVIVE

**On a given lineage `L`, C06 SURVIVES iff there exists `A ∈ R` such that S1–S6 all hold on BOTH
datasets:**

| | condition | frozen source |
|---|---|---|
| **S1** | `acc(A) > max_θ acc(orthrot_θ)` **and** `mF1(A) > max_θ mF1(orthrot_θ)` | `require_primary_above_all_rotation_controls` |
| **S2** | S1's accuracy leg holds in **3/3** seeds | A6 |
| **S3** | `acc(A) − max_{c∈C} acc(c) ≥ 0.02`, likewise `mF1` | `minimum_gain_over_strongest_control = 0.02` |
| **S4** | for every comparator in `C ∪ Θ`: paired item-level bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05` | `minimum_bootstrap_lower_bound`, `require_primary_bootstrap_holm_reject`, `require_rotation_bootstrap_holm_reject`, `n_bootstrap = 2000`, `statistics.seed = 20260728`, `holm_alpha = 0.05` |
| **S5** | **both** real arms exceed the 95th percentile of their shuffled-pair null, **and** the shuffle comparison Holm-rejects | `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject` |
| **S6** | **per-seed net fixes** against `endpoint_std` `≥ 3` (HateMM) / `≥ 2` (MHC-ZH), **in 3/3 seeds** | `minimum_net_fixes`; reference is C01's `retrieval.fix_break_reference = endpoint_std` |

**S6's axis is pinned (round-3 I-4):** the count is the **per-seed** integer net, required in
`3/3` seeds. A seed-mean net is non-integer and would be ill-typed against `minimum_net_fixes`.
**S6 is reported, not screening — see §5.8 item 4.**

Plus `GATE-SMALLDISP` (§6) for C01's `require_no_small_displacement_dominance`.

### 5.3 CLOSE

**C06 CLOSES iff the run publishes a verdict (§6) and SURVIVE is false on BOTH lineages.**
Equivalently, C06 survives if it clears S1–S6 on **either** lineage.

### 5.4 The bootstrap unit

Resample items once (`B = 2000`, C01's frozen `statistics.seed = 20260728`); inside each
resample, average the three seeds' per-item correctness. The seed axis is **inside** the
statistic, not a hidden multiplicity.

### 5.5 Multiplicity

**One Holm family per dataset spanning both lineages**, `α = 0.05`:
`common_displacement` 6 comparators + 6 rotations = 12; `displacement` 5 + 6 = 11;
`(12 + 11) × 2 metrics = 46` per `(dataset, lineage)`; **× 2 lineages = 92 hypotheses per
dataset**.

Round 3 verified the structure: SURVIVE is `∃ lineage ∃ arm (S1∧…∧S6)` on both datasets, the
false-positive event is the disjunction over 4 `(arm, lineage)` disjuncts, and the family covers
exactly those disjuncts' bootstrap legs. The two **datasets** remain a conjunction (tightening
control). **S5's shuffle rejections are correctly outside the family**: they are *conjunctive*
within each disjunct, so `P(∃ disjunct : all conditions) ≤ P(∃ disjunct : its bootstrap legs all
reject) ≤ α` under Holm on the 92.

**M-4, adopted:** §5.5's `92` (hypotheses per dataset = `23 comparisons × 2 metrics × 2 lineages`)
and §8 Phase 4's `92` (bootstrap comparison-cells = `23 × 2 datasets × 2 lineages`, both metrics
inside `U3`) are **two different products that coincide**. Both are correct; neither is derived
from the other, and they must not be "reconciled".

### 5.6 Instrument failure, non-finiteness, and absence

Any HALT gate failing on either dataset in either lineage ⇒ **HALT: no verdict, in either
direction**, recorded as `INSTRUMENT_INCONCLUSIVE`; it **may not be reported as a closure**.

* Every gate quantity is asserted **finite and present** before comparison, and every gate is
  written in **pass-condition** form.
* An **absent** decision or gate quantity HALTs on the same footing as a non-finite one, closing
  the lane where a silently missing lineage makes SURVIVE vacuously false and supplies half of
  CLOSE. `GATE-LEDGER`'s process count is binding (§12).
* **Round-3 I-6, adopted.** `l2_rows` and `prepare_views` signal by `die()` → `RuntimeError`
  (`c01_policy_contrast_a0.py:392-393`), which is a **crash, not a gate result**, and it is the
  single most likely instrument failure in this battery. **Every call into the imported C01
  algebra is wrapped**; a `RuntimeError` from it is recorded as `INSTRUMENT_INCONCLUSIVE` with
  `l2_rows`' `context` string — which already carries the arm and block name — written to **both**
  the decision JSON and the **final heartbeat line**. Without this, C-1's own defect class would
  have been indistinguishable from a wedged process, which is the observability failure
  `rule_2_heartbeat` exists to prevent.

### 5.7 Pre-declared expectation

**CLOSE is expected**, on two grounds: C01 measured the premise rotation-indistinguishable at the
two-point case on both datasets, and the recon's structural objection (a fixed prompt injects no
per-item information) is unrebutted. v1's third ground — the untrained-head contraction — remains
**withdrawn**.

### 5.8 Disclosure

1. `require_accuracy_gain_over_deployed_r0_context` is **not carried**: its comparator is a raw
   dev-arena figure at `n_dev` 107/78 inside a block named `historical_strict_devtrain`, and
   importing it would breach F88's CPU-arm/CPU-floor caveat. **Inapplicable across arenas, not
   waived** — rounds 2 and 3 both verified the comparator and endorsed the reasoning.
2. `displacement`'s comparator set omits `common_displacement` while `common_displacement` must
   beat `displacement`. C01 froze a comparator list only for its primary, so this is not a
   violation, but the asymmetry **eases SURVIVE for one of the two disjuncts**.
3. The two-real-arm disjunction (A2) is deliberately generous; its multiplicity is corrected.
4. **S6 reports, it does not screen (round-3 I-3).** `GATE-SELFTEST` asserts
   `net(A) = n_D · (acc(A) − acc(endpoint_std))` exactly, and `endpoint_std ∈ C` for both real
   arms, so **S3 implies S6 by arithmetic**: `0.02 × 743 = 14.86 ⇒ net ≥ 15` on HateMM and
   `0.02 × 579 = 11.58 ⇒ net ≥ 12` on MHC-ZH, against frozen minima of **3** and **2** — five to
   six times over. The cause is a scale transfer: C01 froze `minimum_net_fixes` on an arena of
   `n_dev` 107/78 where `+3` is `+2.8 %` accuracy; at `n = 743` the same integer is `+0.40 %`.
   **S6 is retained** — it costs nothing, `GATE-SELFTEST` needs its object, and the net figure is
   the currency the Gate-0 record demands — **but it is carried as a reported quantity in that
   currency, not as a binding bar.** A binding net bar would have to be re-derived at arena scale
   and would be a new threshold needing its own justification.
5. **`GATE-ZEROOP`'s tie cap (§6.5) is a declared engineering choice**, not a frozen C01
   constant. It is fixed before the run and stated here so it is not mistaken for an inherited
   threshold.

### 5.9 What this design does not run

**No head-space null-row sensitivity leg** (§3.7), **no L28 leg**, **no per-arm retrained head**,
**no vote at `n = 744`**, and **no test-split read of any kind**. No gate, output field or record
sentence describes any of them.

---

## 6. Gates

Every quantity is asserted finite and present before comparison (§5.6).

| gate | asserts |
|---|---|
| `GATE-DET1` | thread env exported before any python starts |
| `GATE-SHA` | every frozen import and input cache matches §11; **once in the sbatch driver** |
| `GATE-FOLD` | fold assignment matches the banked `vsw_ckpt/<ds>/f{0..4}.npz`, on the **full** `n` |
| `GATE-FLOOR` | **Head-N through the shared driver**, native keys, **full `n`**, reproduces the banked floors at 4 dp on **both** metrics — acc HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946`; mF1 HateMM `0.8838/0.8811/0.8812`, ZH `0.8747/0.8710/0.8765`; and every `fold_acc_deployed` entry. **Anchors the driver for both lineages** |
| `GATE-POP` | the realised populations equal §3.7's table exactly; the head leg and raw leg run on **identical row index sets**; and **the realised arena class counts equal `(297, 446)` (HateMM) / `(180, 399)` (MHC-ZH)**, so §6.3's majority constant is checked against the population it gates rather than assumed |
| `GATE-C01PARITY` | the two-block builder reproduces `prepare_views` **bit-exactly** at `n = 744` with the **one-hot `{355}`** mask (HateMM) and `n = 579` with **`np.zeros(579, dtype=bool)`** (ZH); HALT above C01's `2e-6` |
| `GATE-ROWSUBSET` | *(renamed from `GATE-DUALPATH`, round-3 I-1)* the two-block build at `n = 743` with `np.zeros(743, dtype=bool)` is **bit-identical** to the `n = 744` one-hot build restricted to the 743 surviving rows, on all 13 arms. **This is strictly stronger than C01's `displacement_registered_null_exclusion` at the key level, and sufficient to license the population change — it is not C01's property**, which is defined at the *vote* level and has **no object here**, because no arm is ever voted at `n = 744` |
| `GATE-NULLREMOVED` | no arena population contains an exact-zero row, and the removed row set equals `{355}` (HateMM) / `{}` (MHC-ZH) |
| `GATE-ORBITDISP` | for every arm: HALT iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D`, with per-dataset `ρ*` frozen at **full measured precision** in §6.1; `ρ` is computed over the **arena rows only**; and `ρ_raw` reproduces §6.1's frozen values at 4 dp |
| `GATE-ARMVIAB` | for `endpoint_std` and both real arms: if head-space accuracy fails **`arena majority + 0.02`**, HALT **iff** the same arm clears it in the raw space (§6.2) |
| `GATE-ARENA` | **lower** bound `arena majority + 0.02 ≤ acc` on **`endpoint_std` only**; **upper** bound `acc ≤ 0.98` on `endpoint_std` **and** both real arms. Bands **`[0.6203, 0.98]`** (HateMM) / **`[0.7091, 0.98]`** (MHC-ZH) — §6.3 |
| `GATE-DOMAIN` | Head-N's recovery fraction `(acc_ro − maj)/(acc_native − maj)` for `endpoint_std`, computed and **printed on the verdict face**; reporting-only, no bar |
| `GATE-NESTED` | **per item**, the head that scored it excluded its fold; check count equals the item count |
| `GATE-SELFTEST` | `net(A) = n_D · (acc(A) − acc(endpoint_std))` holds exactly for every arm, seed, dataset and lineage, with **`n_D = |arena(D)|`** pinned to §3.7's table (round-3 I-4: the identity holds only at the arena size, and a banked `744` would produce a guaranteed HALT) |
| `GATE-ZEROOP` | `orthrot_0` vs `endpoint_concat` and `orthrot_45` vs `common_displacement` produce identical predictions — with the tie diagnostic of §6.5 |
| `GATE-ALGEBRA` | key-level `≤ 2e-6` on both identities. **Logically independent of `GATE-ZEROOP` in both directions**, not weaker; the value is their conjunction |
| `GATE-IDPARITY` | every ro cache's `ids` order and `labels` identical to the native bank |
| `GATE-ZEROMASK` | **feature space only** — the measured exact-zero row set of the ro inputs equals `{355}` / `{}` on both policies |
| `GATE-SMALLDISP` | C01's `require_no_small_displacement_dominance` at `small_displacement_train_quantile = 0.1` |
| `GATE-LEDGER` | C09's full declared-count predicate set, process count **binding** (§12) |
| `GATE-DEVFID` | `headspace_fidelity.py` on Head-N's 6 full heads — **reporting only, does not gate** |

### 6.1 `GATE-ORBITDISP` — per-dataset bars at full precision

Every arm is `l2`-normalised before `deployed_vote` (and again by `_norm32`), so **the retrieval
key keeps only direction**. If the head's two policy outputs differ by a near-constant offset,
every `displacement` key is nearly the same vector, every rotation beats it trivially, and a
magnitude gate would see nothing. The quantity is `ρ = ‖mean_i k_i‖` over unit keys.

**The max is the right order statistic**, for the structural reason round 2 gave and round 3
endorsed: this is an *instrument* gate, and its job is to fire only when the head space is more
degenerate than anything the raw feature family produces. A quantile bar would HALT on
head-space arms whose concentration sits inside the observed raw range — the same self-defeat as
§6.3.

**Round-3 I-2, adopted: `ρ*` is frozen at full measured precision.** v3 truncated down to 4 dp,
which made the second conjunct `ρ_raw ≤ ρ*` **false by construction for `endpoint_std`** — the
very arm `GATE-ARENA` and `GATE-DOMAIN` single out as the instrument-health arm — and left §6.1's
two tables quoting the same arm as `0.9682` and `0.9681`. At full precision the exemption
disappears (equality holds) and the inconsistency with it.

| dataset | **`ρ*`** | supplying arm | runner-up |
|---|---|---|---|
| HateMM | **`0.968176`** | `endpoint_std` | `0.964446` (`common`) |
| MHC-ZH | **`0.977223`** | `endpoint_std` | `0.969686` (`common`) |

**The gate:** `ρ_raw ≤ ρ*_D ∧ ρ_head > ρ*_D` ⇒ instrument destroyed the object ⇒ **HALT**; both
above ⇒ both spaces degenerate ⇒ C06's premise is false ⇒ **no HALT, CLOSE warranted**; otherwise
no HALT. Applied to all 13 arms.

**`ρ_raw`, frozen at 6 dp, measured on the arena population:**

| arm | HateMM | MHC-ZH |
|---|---|---|
| `endpoint_std` | 0.968176 | 0.977223 |
| `common` | 0.964446 | 0.969686 |
| `orthrot_83p8` | 0.956893 | 0.964384 |
| `orthrot_72p7` | 0.956491 | 0.965058 |
| `endpoint_concat` | 0.955291 | 0.962418 |
| `orthrot_8p3` | 0.951438 | 0.958355 |
| `orthrot_60p4` | 0.948430 | 0.958728 |
| `orthrot_17p6` | 0.944759 | 0.951882 |
| `endpoint_ow` | 0.942230 | 0.947382 |
| `orthrot_29p1` | 0.933575 | 0.941849 |
| `common_displacement` | 0.928799 | 0.939863 |
| `common_interaction` | 0.913840 | 0.968188 |
| **`displacement`** | **0.891728** | **0.909063** |

In the raw space the displacement direction is the **most** item-dispersed arm of the thirteen on
both datasets, which is why C01 could score it at all.

**`ρ` must be computed over the arena rows.** Computing it over a 744-row array with the masked
zero row left in shifts values by up to **`1.301e-03`** (`endpoint_std`; measured), which would
fail the 4-dp reproduction leg — fail-safe, but it would present as an unexplained HALT. This is
item 7 on §13's code-lineage list.

**The bar is defended by measurement, not only by argument (round-3 measurement (c), reproduced
here).** `ρ` on the **36 banked trained deployed-head key matrices**
(`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_{ds}_s{0,1,2}_f{0..4,full}.npz::K_train`,
18 cells per dataset) is:

| dataset | min | median | max | `ρ*` | cells above `ρ*` |
|---|---|---|---|---|---|
| HateMM | 0.447803 | 0.562434 | 0.632996 | 0.968176 | **0 / 18** |
| MHC-ZH | 0.340179 | 0.574247 | 0.667326 | 0.977223 | **0 / 18** |

**A trained deployed head does not concentrate; it sits at roughly half the bar.** So
`GATE-ORBITDISP` will not fire on ordinary head-induced concentration, and a head-space `ρ` above
the bar really would be anomalous. This measurement is **label-free and computes no arm
accuracy**, so it does not touch §7.3's blindness discipline. (At an *untrained* head, by
contrast, 10 of 13 arms sit above `ρ*` — an initialisation artifact, exactly as round 1's I-10
warned, and the reason the bar could not have been calibrated there.)

### 6.2 `GATE-ARMVIAB` — the two-case form

A one-sided majority-rate HALT on the real arms would fire on exactly the outcome the falsifier
exists to detect: if C06's premise is false, `displacement` in head space *should* sit near the
majority rate, and a one-sided gate would convert a **warranted CLOSE** into a HALT, leaving C06
gated forever on an instrument that can never close it.

* head-space arm fails `arena majority + 0.02` **and** the raw counterpart also fails ⇒ genuine
  negative ⇒ **no HALT**;
* head-space arm fails **and** the raw counterpart clears ⇒ instrument destroyed it ⇒ **HALT**.

Round 2 endorsed this refinement explicitly; round 3 traced the near-majority run and confirmed
it **CLOSES** rather than HALTs.

### 6.3 `GATE-ARENA` — scope, and the arena constant

The **lower** bound is restricted to `endpoint_std` — C09's own scope
(`C09_A0_V17_RECORD.md:1569-1572`, *"pooled native accuracy"*, the floor arm, never a treatment
arm) — and the `≤ 0.98` **upper** bound is kept on `endpoint_std` and both real arms, where it
catches a leak and cannot fire on a warranted CLOSE.

**The bar is the arena majority (round-3 C-2):** `0.6003` (HateMM, `446/743`) / `0.6891` (MHC-ZH,
`399/579`), giving bands **`[0.6203, 0.98]`** and **`[0.7091, 0.98]`**. The full-population
`0.5995` is retained in §3.7 **only** for statements about the 744-row population and is used by
no gate.

**"Real arms lose badly" is a reportable scientific outcome — the falsifier working — and is
never an instrument HALT.** That invariant is what §6.2 and §6.3 jointly enforce.

### 6.4 `GATE-DOMAIN` — reporting, with no invented bar

Round 2 ruled that round 1's C-2(a) is substantively discharged by `GATE-ARENA`'s lower bound on
`endpoint_std`, which *is* a head-space input-domain fidelity HALT gate anchored on a banked
number, and that refusing to invent a recovery-fraction bar is correct. `GATE-DOMAIN` computes
`(acc_ro − maj)/(acc_native − maj)` for `endpoint_std` under Head-N and requires it on the verdict
face and in §10.2's scope sentence, with no threshold.

### 6.5 `GATE-ZEROOP` and `GATE-ALGEBRA` — independent, with a sharpened tie diagnostic

The two gates are **logically independent in both directions**: identical predictions does not
imply `≤ 2e-6` keys, and `≤ 2e-6` keys does not imply identical predictions. The value is their
conjunction.

The θ=45 identity is **not exact** — measured `1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH) on the
raw keys — so a `~1e-7` key perturbation can reorder a top-20 neighbourhood and `GATE-ZEROOP` has
a real **false-HALT** probability on a correct run.

**Round-3 I-5, adopted in full. v3's diagnostic watched the wrong boundary.** `deployed_vote`
weights neighbours by **descending integers `[20…1]`**, so the score changes whenever *any
adjacent pair inside the top-20* reorders, not only when the 20/21 boundary swaps — and an in-set
reordering of two neighbours with opposite labels moves the vote by `~2 × sim`, the larger
effect. v3's 20/21 criterion therefore did not cover the dominant flip mechanism it was
introduced to excuse. The replacement:

* **Ranking:** the **union** of the two arms' top-21 sets (the conservative choice; v3 named no
  referent, and the two rankings differ precisely on the items in question).
* **Residual:** the **maximum** of the two `GATE-ALGEBRA` residuals (v3 named neither).
* **Criterion:** an item is a **tie casualty** iff recomputing its rank-weighted vote after
  collapsing every pair of neighbours whose signed similarities differ by less than the residual
  leaves the two arms' predictions **equal**. This matches the operator rather than a boundary.
* **Cap:** a mismatch on more than **1 % of the arena** (`≤ 7` items HateMM, `≤ 5` MHC-ZH) HALTs
  regardless. v3 had no cap, so a systematic defect that happened to leave near-ties would have
  been reported rather than HALTed. **This cap is a declared engineering choice, disclosed at
  §5.8 item 5.**
* A mismatch confined to tie casualties, under the cap, is **REPORTED, not HALTed**; any mismatch
  outside them HALTs.

---

## 7. Dry-check — what was executed, and what it found

Login node `foscsmlprd01`, conda `HateVideo`, 8 threads, DET-1 exported, outputs only in the
session scratchpad. Zero GPU, zero SLURM, zero test-split file opened, zero write into `data/`,
`artifacts/` or `logging/`.

### 7.1 Real inputs and class balance

All 8 ro caches and 4 native caches loaded through the real `torch.load` path. `n = 744 / 579`;
features `(n, 3584)`; `ids` order-identical and `labels` identical to the native bank on every
file. Exact-zero rows: HateMM `{355}` in both modalities of all four ro caches and the native
cache; MHC-ZH none. Row 355 is `hate_video_95`, **label 1**, held out in **fold 4**.

Class balance, both populations (the C-2 constants):

| dataset | population | n | pos | neg | majority |
|---|---|---|---|---|---|
| HateMM | full | 744 | 298 | 446 | `0.599462 → 0.5995` |
| **HateMM** | **arena** | **743** | **297** | **446** | **`0.600269 → 0.6003`** |
| MHC-ZH | full = arena | 579 | 180 | 399 | `0.689119 → 0.6891` |

### 7.2 Mint units — full-process wall

Every mint figure is **full-process wall**, measured around the `python …` invocation — the first
by `/usr/bin/time -v` (`Elapsed 0:40.39`), the rest by `date +%s.%N` brackets. Interpreter and
import cost is therefore **already inside every unit**: the same run's internal timer reads
`33.0 s`, a `7.4 s` gap that is startup plus cache loads plus the `npz` save, and measured startup
alone is `3.05–3.18 s`. **No Phase 1e line is added, because adding one would double-count** —
round 3 confirmed this reasoning is correct and not a dodge.

| lineage | unit | dataset | measured wall |
|---|---|---|---|
| Head-N | fitting-pool head | HateMM | **40.39 s** |
| Head-N | fitting-pool head | MHC-ZH | **34.40 s** |
| Head-N | full-train head | HateMM | **49.30 s** |
| Head-N | full-train head | MHC-ZH | **38.87 s** |
| Head-R | fitting-pool head, scratchpad harness | HateMM / MHC-ZH | 37.46 / 27.54 s — **not used**, §3.3 |

Fold parity passed in all seven mints; peak RSS **1.25 GiB**, agreeing with C09's measured
1.22 GiB.

### 7.3 Blindness

Every head used in any arm-building or voting dry check is **untrained**, so every operation is
real at real scale while the numbers are scientifically void. **No arm accuracy has been
computed, printed or recorded at any point in v1–v4**, on any dataset, in either lineage — round 3
audited this and confirmed it. §6.1's trained-head `ρ` measurement reads banked key matrices
only and computes no accuracy.

### 7.4 The null-contract and mask-convention measurements

| # | measurement | result |
|---|---|---|
| (a) | is `head_f(0,0)` zero? | **No** — non-zero at torch seeds 0/1/2 and under both emitter conventions tested; observed `0.58–0.65`. **The value is initialisation-dependent; the structural result is not** (round-3 M-2) |
| (b) | `h_std[355] == h_ow[355]`? | **Yes, exactly**; and not all-zero. It is the **only** such row on either dataset |
| (c)–(e) | `l2_rows` per block | endpoint DIES at `{355}` / OK at all-False; common the same; displacement OK at `{355}` / DIES at all-False |
| (f) | ⇒ `common_displacement` in head space | **unbuildable under either mask** |
| (g) | **the repair**: all 13 head-space arms at `n = 743`, all-False mask | **ALL BUILT** through the imported `l2_rows`, `float32` |
| (h) | **the bridge**: raw arms `n = 743` vs `n = 744` one-hot, restricted | **BIT-EXACT, `max\|diff\| = 0.000e+00`**, all 13 arms; every `ρ` unchanged |
| **(i)** | **`prepare_views` with `zero_mask = None`** | **DIES on MHC-ZH** (*"derived exact-zero mask preservation failed"*) and on HateMM. With `np.zeros(n, dtype=bool)`: **OK on MHC-ZH**; HateMM at `n = 744` needs the **one-hot `{355}`** |
| **(j)** | **`prepare_views` on the ARENA population with the explicit all-False mask** | **OK on BOTH datasets** — HateMM `n = 743`, MHC-ZH `n = 579`. This is the executed form of the §3.7 convention |
| (k) | `ρ` computed over 744 rows including the masked zero row | shifts by up to **`1.301e-03`** (`endpoint_std`) — would fail the 4-dp reproduction leg |

(g), (h), (i) and (j) are what license §3.7: the repair executes, the population change is a pure
row-subset, and the mask convention is the one that actually runs.

### 7.5 The one-block instantiation

`fuse([b])` differs from `l2(b)` by `7.451e-09` (reproduced identically by rounds 2 and 3). The
one-block `paired` differs from v1's rejected `pair` by a **head-weight-dependent** amount —
`1.118e-08` here, `7.451e-09` (round 2), `3.725e-09` (round 3). **Only the invariant claim carries
weight: all are a fraction of a `float32` eps (`1.192e-07`), so the outer normalisation restored
by round 1's C-3 is numerically vacuous at one block** (round-3 M-3).

### 7.6 `GATE-C01PARITY`

The two-block builder reproduces `prepare_views` **bit-exactly** on the raw L24 features,
`max|diff| = 0.000e+00`, all 13 arms, both datasets — with the admissible mask form of §3.7.
Independently reproduced from this document's prose by both round 2 and round 3.

### 7.7 Unit table

Round 1's H-6 found v1's `U4` smaller than its own constituents; cause: `float64` arms and one
repeated fold. Re-measured with `float32` arms over the five real folds it reconciles:
`5 × 0.00305 + 5 × 0.00629 = 0.04674 s` of votes, leaving `0.04234 s` for the rebuild.

| unit | what | dataset | measured |
|---|---|---|---|
| `U1` | head forward over one real ro cache | HateMM | 0.0461 s |
| `U2a` / `U2b` | vote, 1024-d / 2048-d, per fold-cell | HateMM | 0.00305 / 0.00629 s |
| `U2c` / `U2d` | vote, 7168-d / 14336-d (raw), per fold-cell | HateMM | 0.04239 / 0.08098 s |
| `U3` | bootstrap `B = 2000`, one comparison, both metrics | HateMM | 0.126 s |
| `U4` | one shuffled-pair null draw | HateMM | 0.08908 s |
| `U5a` / `U5b` | two-block build + compare / builder-only | HateMM | 11.27 / 4.63 s |
| `U6` | `ρ` over 13 arms | HateMM (raw) | 0.62 s |
| `U7` | `GATE-SHA` over 8 caches + 6 modules | — | 0.12 s |
| `U8` | ro cache `torch.load`, 2 files | HateMM | 0.033 s |
| `U9` | `GATE-DEVFID`, per `(dataset, seed)` | HateMM / MHC-ZH | 3.70 / 3.49 s |
| `U10` | head-space build of all 13 arms, one cell | HateMM | 0.1873 s |
| `U11` | interpreter + imports | — | 3.05–3.18 s (**inside the mint units**) |

**Convention, stated once:** every unit was measured on **HateMM**, the larger dataset, and
applied to MHC-ZH unchanged — every such application **over-states** the MHC-ZH cost. Exceptions:
`U9` (measured per dataset), `U7` and `U11` (dataset-independent).

**`U9` correction, disclosed.** The first `GATE-DEVFID` timing was a **failure path**:
`headspace_fidelity.py` defaults to `--seeds 0,1,2`, only seed-0 full mints existed, so it errored
and wrote no file while my shell captured `echo`'s exit status. Re-run with `--seeds 0`, both
datasets exit `0` and write their JSON. Round 3 noted that `U5a`, `U6`, `U10` and the mints are
independently corroborated (it reproduced their outputs; fold parity passed in all seven), while
`U2a`–`U2d`, `U3`, `U4`, `U7`, `U8`, `U11` are not — **the freeze record will state the
exit-status discipline under which each was timed**, and the code/resource lineage should hold
this document to that.

### 7.8 Dry-check cost, disclosed

v4's measurements added ≈ **3 wall-minutes / ≈ 9 CPU-minutes** (the mask-convention battery, the
arena class counts, the trained-head `ρ`, the population-sensitivity check). Cumulative across
v1–v4: ≈ 16 wall-minutes / ≈ 73 CPU-minutes, all `$0`, zero GPU. The honest framing is **disclosed
at the same time as the result**: the CPU-cap conflict was knowable from C09's banked mint costs
before the first burn. All three rounds ruled the underlying trade correct — a standing
`TARGET_STATE.json` rule beats a task brief's CPU cap.

---

## 8. Compute projection — measured unit × explicit count

**Unchanged from v3: both round-3 Criticals are a calling convention and a constant, and neither
adds or removes a unit.** Re-verified below.

| phase | count | unit | product |
|---|---|---|---|
| **1** Head-N mints, HateMM fold | `3 × 5 = 15` | 40.39 s | `605.9 s` |
| **1** Head-N mints, HateMM full | `3` | 49.30 s | `147.9 s` |
| **1** Head-N mints, ZH fold | `3 × 5 = 15` | 34.40 s | `516.0 s` |
| **1** Head-N mints, ZH full | `3` | 38.87 s | `116.6 s` |
| **1R** Head-R mints, HateMM | `15` | 40.39 s | `605.9 s` |
| **1R** Head-R mints, ZH | `15` | 34.40 s | `516.0 s` |
| **1b** key forwards `(30×3)+(6×4)+(30×2)` | `174` | `U1` | `8.0 s` |
| **1c** ro cache loads, per process | `66` | `U8` | `2.2 s` |
| **1d** `GATE-SHA`, once in the driver | `1` | `U7` | `0.1 s` |
| **2** head-space votes, 1024-d / 2048-d arms | `240` / `540` | `U2a` / `U2b` | `0.7` / `3.4 s` |
| **2** `GATE-FLOOR` native vote | `30` | `U2a` | `0.1 s` |
| **2b** head-space arm construction | `12` cells | `U10` | `2.2 s` |
| **2R** raw votes, 7168-d / 14336-d | `40` / `90` | `U2c` / `U2d` | `1.7` / `7.3 s` |
| **2Ra** raw arm construction | `2` datasets | `U5b` | `9.3 s` |
| **2C** `GATE-C01PARITY` | `2` datasets | `U5a` | `22.5 s` |
| **2C** `GATE-ROWSUBSET` (HateMM only) | `1` | `U5b + 0.21` | `4.8 s` |
| **2D** `ρ`, raw + head cells | `14` | `U6` | `8.7 s` |
| **3** shuffled-pair null draws | `256 × 3 × 2 × 2 = 3072` | `U4` | `273.7 s` |
| **4** bootstrap comparison-cells | `23 × 2 ds × 2 lineages = 92` | `U3` | `11.6 s` |
| **5** head-space null-row sensitivity | **0 — the leg does not exist** | — | `0.0 s` |
| **6** `GATE-DEVFID` | `3 + 3` | `U9` | `21.6 s` |
| **7** per-gate arithmetic on materialised vectors — `GATE-SELFTEST` (156 checks), `GATE-NESTED`, `GATE-SMALLDISP`, `GATE-POP` (incl. the class-count clause), `GATE-NULLREMOVED`, `GATE-IDPARITY` | all | sub-`0.1 s` class | `0.1 s` |
| | | **corroborating total** | **`2886.3 s = 48.1 min`** |
| | | **conservative (× 1.25)** | **`3607.9 s = 60.1 min`** |

**M-1, adopted:** the printed product column sums to `2886.2 s`; **Phase 7 is carried at its
upper bound of `0.1 s`**, which is the whole of the `0.1 s` difference. Every downstream figure
uses `2886.3`. Round 3 noted that grouping the per-gate arithmetic here makes the enumeration
literally exhaustive; Phase 7's description above now names each item.

**Declared slack, outside the projection:** `30 s` for ledger aggregation and JSON emit.

**Peak RSS ≈ 1.3 GiB.** Request 32 GB.

**Where the risk sits.** Mints are `86.9 %` of the total and are measured directly. Phase 3 is
`9.5 %`; a 2× miss moves the total to `3160.0 s = 52.7 min`, a 5× miss to `3981.1 s = 66.4 min`.
**If the realized cost exceeds the conservative total by more than 2×, that is itself a reportable
process finding.**

---

## 9. Heartbeat specification

* One progress file `$BASE/progress/C06_PROGRESS.txt`, created by the sbatch driver before the
  first python process starts; every python process appends through a handle opened
  `buffering=1`.
* Each line: ISO-8601 timestamp · phase · units done / total · elapsed · elapsed ÷ **§8's frozen
  projected** value.
* Granularity: one line per mint (66), **one per training epoch within a mint**, one per
  `(dataset, seed, lineage)` arm block (12), one per 32 null draws (96), one per bootstrap block,
  one per gate, one per verdict field. The bash driver **also** echoes a line per mint,
  unbuffered.
* **The HALT path names which gate failed in its final line** — and, per §5.6, a `RuntimeError`
  from the imported C01 algebra is caught, recorded as `INSTRUMENT_INCONCLUSIVE` with its
  `context` string, and written to that line before exit.
* Round 3 re-checked every interval under this phase structure: longest un-instrumented span is
  `GATE-C01PARITY` at `11.27 s` (`14.1 s` conservative); the mints' worst gap is the `≈ 7 s` of
  pre-training cache loads (`8.8 s` conservative); Phase 3 emits every `2.85 s`. **Nothing
  exceeds ~15 s.**

---

## 10. Scope of any verdict

### 10.1 The prompt/readout-span confound

`generate_VideoMLLM_embedding_readout_HF.py:73-89` defines
`("ro_L24", "baseline", "prefix", "response", LAYER_MID)` versus
`("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`: the `ow_` cell changes **the
prompt kind and both readout spans**. **No result of this battery can attribute an effect — or
its absence — to the prompt alone.**

### 10.2 What a CLOSE would and would not close

A CLOSE closes **C06's first-order (tangent/chord) prompt-orbit route in the fold-head arena at
`ro_L24`, on `HateMM (-LoRA-curric)` at `n = 743` and `MHC_zh (-LoRA)` at `n = 579`, under BOTH a
native-trained deployed head applied out of domain AND an `ro_L24`-trained in-domain head.** The
`GATE-DOMAIN` recovery fraction must be stated in the same sentence. It does **not** establish:

* anything about **curvature** — two prompt points give a chord; ≥ 3 require extraction;
* anything about a head **retrained per arm** — F66's trained-reshaping caveat stands;
* **anything about the per-modality contrast C01 measured**: the deployed head fuses image and
  text internally (`classifier.py:115-124`), so **every head-space arm is a post-fusion,
  one-block analogue** of C01's per-modality two-block contrast (`contrast_blocks:1242-1270`).
  The one-block reading is **forced by the architecture, not chosen**, but a CLOSE will be read as
  "C01's battery re-run in the fold-head arena" and what it measures is not the transform C01
  scored `0.8505 / 0.8846`;
* anything about a **different readout span**, **L28**, or the **test split**.

### 10.3 What a SURVIVE would license

Only that C06 *"has earned its extraction"*: the `1.7–2.5 GPU-h` bounded extraction may be
**proposed** under `iteration_8_stage0_bounded_extraction_amendment`, with its own
preregistration, design review, separate code/resource review lineage and authorization. **A
SURVIVE is not a Stage-0 PASS and authorizes no GPU.**

### 10.4 Bans checked

F80's object is prompt **language**; F70's is individual **readout cells**; C06's is the
**relation between** two cells. All three rounds tested and confirmed the object-mismatch
warrant. The multi-prompt **ensembling** carve-outs have C14 as their object and are not relied
on. `endpoint_std` and `endpoint_ow` **are** literally F70's two cells, entering here as
**controls**, not claims. No ensemble of prompt predictions is formed: `avg_score` is C01's own
frozen `gain_control`.

**Hard constraints: none touched** — no OCR, no cross-dataset mixing, no external API,
single-dataset train split, parent-video binary label only, no ensembles, no size scaling,
SLURM-only. Round 3 re-checked verbatim against `d_no_other_relaxation`.

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
analysis. v4's `frozen_v3` → v3 (`scientific_thresholds_exact: true`); v3's `scientific_base` →
v2, same flag. In source, `c01_policy_contrast_a0_v4.py` sha-checks `_v3.py`, which sha-checks
`c01_policy_contrast_a0.py` — the file this battery **imports**. Round 3 verified both hops raise
on drift. **Config chain v4 → v3 → v2; algebra chain v4 → v3 → base.**

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

Plus the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` (`GATE-FLOOR` anchors), the
ten banked `vsw_ckpt/{hatemm,zh}/f{0..4}.npz` (fold parity), and — **read-only, for §6.1's
trained-head `ρ` reference measurement** — the 36 banked
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`.

**New code, confined to the battery:** `scripts/analysis/c06_falsifier_mint.py` (**the single
shared driver**), `scripts/analysis/c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`,
`scripts/slurm/c06_falsifier_cpu.sbatch`.

---

## 12. Label and split discipline

**Test contact: none, enforced in three layers** — `headspace_mint.py:106-116`'s `torch.load`
guard; the driver's `split == "train"` assertion on every ro-cache load; and the frozen
`c09_guard` `sitecustomize` installing an `open()`-level, component-wise, repo-scoped predicate
at interpreter startup in **every** process. Round 3 confirmed no `dev_seen_*-ro_*` or
`test_seen` path is reachable from any phase.

**`GATE-LEDGER` (round-3 H-1 corrections in bold):**

| predicate | expected | binding? |
|---|---|---|
| `test_path_opens` | **0** | yes |
| `test_label_materialisations` | **0** | yes |
| `mints_present_before_arena` | **66** `.npz` (36 Head-N + 30 Head-R) | **yes** |
| `dev_path_opens` | **`mints_executed` + `GATE-DEVFID` reads**, where `mints_executed` is the number of mint processes that ran their body (`66` on a fresh run) | **yes, against `mints_executed`** |
| `dev_label_materialisations_outside_decisions` | **`mints_executed`**, one per executed mint | **yes, against `mints_executed`** |
| `dev_or_test_labels_into_decision_quantities` | **0** | yes |
| `banked_trainlog_opens` | declared, `GATE-DEVFID` only | reported |
| processes reporting | **66 mints + 6 fidelity + 1 arena** | yes — HALT on any mismatch |
| predicate coverage | re-derived in-job | reported |

**Dev labels, stated correctly (round-3 H-1).** `headspace_mint.py:199` loads `dev_seen_*.pt`
**unconditionally on every mint**, before the `fold` branch, and `:322` writes `lab_dev` into
every `.npz`; `model_name` comes from the frozen dataset table, so `--train-cache` does not
redirect it. Under the shared driver **all 66 mints** open the **native** `dev_seen` cache and
materialise `lab_dev` — not 36, as v3 wrongly said. **Head-R opens no `dev_seen_*-ro_*` file**,
which is the true and relevant statement. None of this reaches a decision quantity: at
`fold ≥ 0` `dev_sp` is a slice of the fitting pool (`:223-226`) and only `dv[3]` is written to
disk. Because both counts are now exactly predictable, they are **binding** under §5.6's absence
rule.

**Binding them to `mints_executed` rather than to the literal `66` is deliberate, and it repairs
a defect this document introduced.** `headspace_mint.py:192-194` returns **before** the
`dev_seen` load at `:199` whenever `--out` already exists, so on a **resumed** job a skipped mint
opens no dev file and materialises no `lab_dev`. A binding `dev_path_opens == 66` would therefore
HALT a legitimate resume — the same class of self-defeating gate round 3 caught at `GATE-ARENA`,
reintroduced one table over. Binding against the measured `mints_executed`, plus a separate
binding assertion that all **66** `.npz` are present before the arena runs, is exactly
predictable under both a fresh run and a resume, and still closes the absence lane §5.6 names.
The resume path is real and is the design's own (`§8`'s projection assumes a fresh run; a resumed
run costs strictly less).

**No selection anywhere.** Every threshold in §5 and §6 is C01's frozen value, C09's banked
constant, a population-derived constant frozen in §3.7/§6.1 from banked label-free arithmetic, or
a declared engineering choice disclosed in §5.8.

---

## 13. Execution boundary, and what the code lineage must verify

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** All three rounds confirmed: F88's `$0` forensics name no non-SLURM
channel and price a 52-second process, no precedent for a ~48-minute job; CLAUDE.md's standing
rule, C01's frozen `execution.require_slurm = true, cpu_only = true, required_cpus = 8` (with
`required_memory: 32G` in v3) and the C02/C09 precedents all agree. **Cloud routing
inapplicable:** `GATE-FLOOR` anchors to six floors measured locally on `foscsmlprd01`.

**Not authorized by this document.** Required before anything runs: an independent design review
to GO (0C/0H/0I), a **separate** code/resource review lineage over the executable, and
main-dialogue authorization.

**Round 3's twelve-item list for that lineage, adopted verbatim as the design's own handoff.**

*The shared mint driver:* (1) that it imports `headspace_mint` with its sha256 asserted and that
**no** behaviour outside `--train-cache` differs between lineages — fold-parity assertion, dummy
construction, `torch.load` guard, seeding and DET-1 the frozen ones, not re-implemented; (2) that
`--train-cache` overrides **only** the training cache and cannot reach `model_name`, the dev load
or the dataset table, and that §12's declared dev counts match what the code does; (3) that there
is **no branch conditional on the cache filename or suffix** — `GATE-FLOOR` exercises the native
path only, so such a branch would be invisible to it, which is why `GATE-SHA` over the ro caches
and `GATE-IDPARITY` are load-bearing here; (4) that the `GATE-FLOOR` mints and the Head-R mints
go through the *same* function, not two copies.

*Populations and constants:* (5) that the majority rate used by `GATE-ARENA` and `GATE-ARMVIAB`
is **computed from the arena's own labels**, not read from a constant, and equals `0.6003 /
0.6891`; (6) that `GATE-SELFTEST`'s `n` is the arena size and no banked `744` leaks into any
per-item denominator; (7) that `ρ` is computed over the **743/579-row** arm matrices and not over
a 744-row array with a masked row left in (a `1.301e-03` shift, fail-safe but presenting as an
unexplained HALT).

*The mask convention:* (8) that every `prepare_views` call passes an explicit boolean array and
every `l2_rows` call's mask matches the population it is handed — **with an assertion, not a
comment**; (9) that the `n = 744` build exists **only** inside `GATE-C01PARITY`/`GATE-ROWSUBSET`
and that nothing votes on it.

*The tie diagnostic:* (10) which ranking and which residual the implementation uses, that its
item set is the pre-registered one, and that the report-not-HALT branch cannot be reached by any
item outside it or above the cap.

*`GATE-POP`:* (11) that it is evaluated **before** any gate consuming a population-dependent
constant, and asserts row identity between the head and raw legs by **index set**, not count.

*Heartbeat and failure recording:* (12) all six items in §9's list, plus that a `RuntimeError`
from the imported C01 algebra is caught, recorded as `INSTRUMENT_INCONCLUSIVE` with its `context`
string, and written to the final heartbeat line before exit. Also: the progress handle opened
`buffering=1` and never re-wrapped; the driver's echo unbuffered under `sbatch`; all 73 processes
**append** rather than truncate without interleaving partial lines; and the `elapsed ÷ projected`
denominator being §8's frozen number.

---

## 14. Cumulative disposition — rounds 1 to 3

### Round 3 (13 findings + 4 Minor) — all adopted

| finding | disposition | where in v4 |
|---|---|---|
| **C-1** `zero_mask = None` inadmissible to `prepare_views` | **ADOPTED** — the contract is rewritten in explicit boolean arrays; §3.7's table carries a mask column; the `l2_rows`-vs-`prepare_views` difference is recorded with line cites; measured executing on **both** datasets at the arena population | §3.7, §6, §7.4(i)(j) |
| **C-2** 744-population majority applied to the 743 arena | **ADOPTED** — arena majority frozen at `0.6003 / 0.6891` as a named population-derived constant; `GATE-ARENA` bands `[0.6203, 0.98]` / `[0.7091, 0.98]`; `GATE-ARMVIAB` cites the arena constant; **`GATE-POP` gains the class-count clause** `(297,446)` / `(180,399)` | §3.7, §6, §6.3, §7.1 |
| **H-1** §12's Head-R dev sentence false | **ADOPTED** — corrected to **66** mints opening the native `dev_seen`; both counts made **binding**; the true statement (*Head-R opens no `dev_seen_*-ro_*`*) substituted; §3.3's "only variable" sentence retained and shown to be consistent | §3.3, §12 |
| **I-1** `GATE-DUALPATH` wears C01's name | **ADOPTED** — renamed **`GATE-ROWSUBSET`**, cited as strictly stronger at the key level, with the explicit statement that C01's vote-level property **has no object here** | §6 |
| **I-2** `ρ*` truncation exempts `endpoint_std` | **ADOPTED** — `ρ*` frozen at **full precision** (`0.968176` / `0.977223`); the `ρ_raw` table given at 6 dp; both the exemption and the two-table inconsistency gone | §6.1 |
| **I-3** S6 vacuous at arena scale | **ADOPTED** — disclosed as §5.8 item 4 with the arithmetic (`0.02 × 743 = 14.86 ⇒ ≥ 15` vs a frozen minimum of 3) and the scale-transfer cause; S6 retained as **reported, not screening** | §5.8 |
| **I-4** `GATE-SELFTEST`'s `n`, S6's axis | **ADOPTED** — `n_D = |arena(D)|` pinned to §3.7; S6 defined on the **per-seed** integer net in 3/3 seeds | §5.1, §5.2, §6 |
| **I-5** tie diagnostic under-specified, wrong boundary | **ADOPTED in full** — ranking = union of the two arms' top-21; residual = max of the two; criterion = collapse-and-recompute the rank-weighted vote (matching the operator, not the 20/21 boundary); **cap at 1 % of the arena**, disclosed as a declared choice | §6.5, §5.8 item 5 |
| **I-6** `die()` is a crash, not a gate result | **ADOPTED** — every call into the imported C01 algebra is wrapped; `RuntimeError` → `INSTRUMENT_INCONCLUSIVE` with the `context` string to the decision JSON **and** the final heartbeat line | §5.6, §9, §13 item 12 |
| **M-1** Phase 7 rounding | **ADOPTED** — carried at its `0.1 s` upper bound, convention stated; Phase 7 widened to name every per-gate arithmetic item | §8 |
| **M-2** `‖head(0,0)‖` six digits | **ADOPTED** — stated as non-zero at every seed and emitter convention tested, `0.58–0.65` observed | §7.4(a) |
| **M-3** one-block `paired` digits | **ADOPTED** — only the invariant claim carries weight; all three rounds' values recorded | §7.5 |
| **M-4** the two `92`s | **ADOPTED** — distinguished explicitly | §5.5 |

**Round-3 recommendation adopted beyond the findings:** the trained-head `ρ` measurement (0/18
cells above `ρ*` on both datasets) is now cited in §6.1 as the measured defence of the bar.

### Rounds 1 and 2 — carried unchanged

Round 3 audited **16 of 16** round-2 findings as truly adopted and confirmed all three reopened
round-1 items (C-3, the C-1 companion, I-3) genuinely repaired, with **no disguised rebuttal and
no claimed repair the artifact does not contain**. Those dispositions stand as recorded in v3
§14 and are not restated here except where a round-3 finding refines them: round-2 C-1 (§3.7),
C-2 (§6.3), C-3 (§6 renaming), H-2 (§3.3, §12), I-5 (§6.5), I-7 (§6.1).

**Rulings carried without change across all rounds:** the direction of "conservative"; A7 is not
an obstacle; per-arm retraining excluded; the `max` as `ρ*`'s order statistic (now with
measurement behind it); SLURM and the login-node dismissal; the untrained-head blindness
discipline; HALT semantics; §5.8 item 1's inapplicability reasoning; S6's net-fix reference.

---

## 15. Open issues for round 4

1. **The mask convention (§3.7).** Now stated in explicit arrays. Round 4 should confirm no
   `None` survives anywhere, and that the four objects' mask arguments are each correct for their
   population — in particular that the all-False mask is right at `n = 743` **because** the row is
   gone, not by coincidence.
2. **The arena majority constant (§6.3).** Round 4 should re-derive `446/743` and `399/579` and
   confirm `GATE-POP`'s class-count clause is sufficient to make the constant checkable at run
   time rather than assumed.
3. **`GATE-ZEROOP`'s 1 % cap (§6.5).** This is the only genuinely invented threshold in the
   design. It is disclosed as such. Round 4 should rule whether a cap is needed at all, and if so
   whether 1 % is defensible or should be derived from something banked.
4. **S6's retention (§5.8 item 4).** It is now disclosed as vacuous-under-S3. Round 4 should rule
   whether a condition that cannot bind belongs in a SURVIVE conjunction at all, or whether it
   should move out of S1–S6 into the reported quantities.
5. **§12's binding dev counts, and the resume path.** Binding `dev_path_opens` is stricter than
   C09, which reported it. Drafting this section surfaced that a literal `== 66` would HALT a
   legitimately **resumed** job, because `headspace_mint.py:192-194` returns before the
   `dev_seen` load at `:199`; §12 now binds against the measured `mints_executed` and separately
   asserts all 66 `.npz` are present before the arena. Round 4 should confirm that pair is
   exactly predictable under every code path that can legally run, and that nothing else in the
   design assumes a fresh run.

---

*No GPU, SLURM, Modal, teacher call, model load, training of any deployed arm, cache write,
test-split access, job submission or commit occurred in producing this document. Login-node
dry-check processes only (§7.8). `TARGET_STATE.json` was read and not modified. v1, v2 and v3
are unmodified.*
