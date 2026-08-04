# C06 `$0` CPU falsifier — preregistration **DRAFT v2** (2026-08-04)

**SUPERSESSION.** This document supersedes `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md`
(v1, same day), which stays on disk **unmodified** as the record of what round 1 reviewed.
v2 is a complete document, not a diff: read it standalone. Round-1 review of record:
`refine-logs/C06_FALSIFIER_PREREG_REVIEW.md` — **REVISE (3C / 6H / 10I + 4M)**.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is
frozen, `TARGET_STATE.json` is untouched, nothing is committed.

**Disposition summary: 23 of 23 findings ADOPTED, 0 rebutted** — one (C-1's
prediction-level companion) adopted in a **refined form** whose reason is stated at the
point of change and flagged for round 2. Full table in **§14**. All four requested rulings
in §15.

**What v2 changes that a reader should know before anything else.** Three things move:
(1) the battery now runs **two head lineages** — the native-trained deployed head *and* an
`ro_L24`-trained in-domain head — and a CLOSE requires C06 to fail on **both**; (2) the arm
builder is now a **generic block-list construction that reproduces C01's `prepare_views`
bit-exactly**, verified today at `max|diff| = 0.000e+00` on all 13 arms and both datasets,
and gated on that; (3) the **L28 leg is dropped entirely**. The compute projection moves
from 35.5 min to **47.8 min** corroborating (59.7 min conservative), every unit measured.

---

## 1. What this falsifier is, and what authorizes it

C06 (*Prompt-Orbit Tangent/Curvature*) is **not an active candidate**. Its registry status
is `gated_on_zero_cost_falsifier`
(`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`). What the campaign
queue has reached is **C06's falsifier**, not C06
(`iteration_8_queue_state_2026_08_04.next_item.IS_IT_AN_ACTIVE_CANDIDATE`: *"NO. C06
remains GATED."*).

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

Both honoured: (i) §3.1; (ii) §10.1. Round 1 verified both as *honoured, not merely
mentioned*.

**The evidence the gate rests on** — C01's A0, re-verified by the Gate-0 adjudicator
against `C01_A0_OUT.json` with every accuracy recomputed from the stored confusion
matrices (`GATE0_REOPEN_2026-07-31.md` §4.4):

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

**Round-14's sharpening, adopted:** `c01_policy_contrast_a0.py:1272`'s
`orthogonal_blocks()` is a **Givens mixing of the two endpoint blocks**, so the six
"random rotations" are six angles on **one parameter family** that also contains the
primary — `θ = 45°` **is** `common_displacement`, `θ = 0` **is** `endpoint_concat`. I
re-measured C01's own guard on the raw L24 features today: `8.94e-08` (θ=0, both datasets)
and `1.19e-07` / `8.94e-08` (θ=45), consistent with the record's *"`8.9e-08`–`1.2e-07`"*.

**Why a re-run in a different space is the right instrument.** C01's arena is **raw dev
keys** (`n_dev` 107 / 78), not the fold-head path. The registry's `unified_pilot_gate.arena`
reads *"strict train-OOF or untouched development split using the actual
fold-head/deployed-head path; raw-key arena may kill but may not promote a lead"*, and
F113 marks the raw-KILL direction **NOT ESTABLISHED** (correction V-4). Round 1 ruled the
arena reading correct (§15.1).

---

## 2. The process rules that bind this design

`process_rule_compute_projection_and_heartbeat_2026_08_04` names this falsifier in
`applies_immediately_to`.

| rule | discharged in |
|---|---|
| **R1** measured-unit-cost × explicit-count projection; no reduced-scale extrapolation; **UNKNOWN** rather than a band if unmeasurable | **§8** — 14 unit types measured at real scale, multiplied through enumerated counts. v1's `U2`/`U4` are **withdrawn and re-measured** (§7.6, finding H-6) |
| **R2** line-buffered per-phase heartbeat to a progress file | **§9**, now with a within-mint line (round-1 gap) |
| **R3** (F114) dry execution must exercise the **first real operation of the payload path** | **§7** |
| **R4** (`feedback-separate-code-review-lineage`) a design GO does not review the implementation | **§13** |

---

## 3. The arena and the instrument

### 3.1 Inputs — the lineage that actually exists, asserted not assumed

C01's frozen scientific configuration (`configs/c01/c01_a0_v2.json`) pins
`standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`, `feature_dim = 3584`. Directory
listing confirms **one adapter lineage per dataset** (correction V-8):

| dataset | adapter lineage (the only one banked) | `expected.train.n` |
|---|---|---|
| HateMM | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | 744 |
| MHC_zh | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | 579 |

*(M-1: the column is the config's `inputs.datasets.<ds>.expected.train.n`, named exactly.)*

**These are not a matched pair and this design never treats them as one.** No cross-dataset
comparison of absolute numbers is made; every decision quantity is a within-dataset,
within-seed, within-lineage arm comparison, and the two-dataset requirement is a
**conjunction of independently computed verdicts**.

**Provenance, re-measured today.** The four L24 files are byte-identical to the ones C01
measured — sha256 prefixes equal C01's frozen `*_provenance_sha16`, and the HateMM one
equals C01 v3's `diagnostic_train_cache_sha256` in **full 64 hex**. Full digests in §11.

**L28 is dropped (I-8, §15.4).** v1 ran a second orbit at `ro_L28`. Round 1 measured that
at L28 the two endpoints are **near-orthogonal** (`cos` `0.147–0.396`, `‖Δ‖` `1.10–1.31`
against a `√2` ceiling) versus `cos` `0.737–0.773`, `‖Δ‖` `0.674–0.726` at L24, and that
`generate_VideoMLLM_embedding_readout_HF.py` makes L28 the `LAYER_FINAL` **R0 clobber-guard
cell**, not a sibling of the L24 grid. It replicates nothing, it carries four files outside
C01's frozen 8-file manifest into a battery whose authority is that it reads C01's bytes,
and it was every `× 2 layers` factor in the projection. **Removed entirely** — from the
arms, the gates, the projection, §11 and the verdict.

**Splits.** `train_*.pt` only for the ro caches. The native `dev_seen` cache is opened by
`headspace_mint.py:199` on every mint and is listed in §11 and covered by `GATE-SHA`
(round-1 I-1). No `dev_seen_*-ro_*` file is opened by any phase. The `test_seen` ro caches
exist on disk and are opened by nothing (§12).

### 3.2 The head, the folds, and the vote

* **Fold contract.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the
  train split (`mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`), asserted against the
  banked `scripts/analysis/vsw_ckpt/<ds>/f{0..4}.npz` `ho_idx` by
  `headspace_mint.py:203-216`, which refuses to run on mismatch. Verified in all seven real
  mints run today.
* **Head.** The deployed-recipe RGCL head, re-minted on CPU (deployed checkpoints are gone:
  F78 / `HEADCOV_PREGATE_RECORD.md` §1.1). Bank = fitting pool, queries = held-out fifth,
  every item held out exactly once ⇒ one OOF prediction vector per arm over all `n` items.
* **Vote.** `mechfix_ops.deployed_vote(..., topk=20)` — round 1 verified this is
  numerically the operator C01's config specifies (`topk 20`, descending integer weights,
  signed cosine, cutoff `≥ 0`).
* **F88's binding caveat** — *"a CPU-trained arm must be paired against a CPU-TRAINED
  FLOOR"* — satisfied by construction: every arm and floor is minted in the same CPU arena.

### 3.3 Two head lineages — the C-2 repair

**The round-1 correction, stated first and without hedging.** v1 §3.3 claimed its
single-head construction was *"the banked house pattern, not an invention"*, citing
`c02_a0_mint.py`. **That citation was wrong and the claim is withdrawn.** `c02_a0_mint.py:214`
is `keys[view] = keys_of(tr[1], view_text[view])` — the **native** `img_feats` on every
view; `:21-23` says so (*"img_feats are byte-identical across views by construction … so
the view axis is the only thing that moves"*); and `load_view_text` at `:68` **refuses** any
view file carrying an image stream. C02 moved **one** stream **inside one extraction
family**. C06 moves **both** streams to a **different readout cell**. The precedent covers
forwarding a head over modified features; it does **not** cover this transplant.

Round 1 then measured what the transplant is: median `cos(native_img, ro_L24_img)` =
**`0.0234`** (HateMM) and **`0.0373`** (MHC-ZH), with text at `0.2300 / 0.2495` and both
caches unit-norm. The image streams are **essentially orthogonal representations**, and
forwarding them through `mlp(normalize(img_proj(·)) ⊙ normalize(text_proj(·)))` — projections
fitted on the native distribution — is an out-of-distribution transplant that plausibly
explains v1's unexplained `219×` orbit contraction.

**v2 runs both lineages and requires C06 to fail on both.**

| lineage | head trained on | keeps `GATE-FLOOR`? | in-domain? | mints |
|---|---|---|---|---|
| **Head-N** (native) | native deployed cache (`train_<model>.pt`) | **yes** — reproduces the six banked floors exactly | no (the transplant) | 36 = 2 ds × 3 seeds × (5 folds + 1 full) |
| **Head-R** (in-domain) | `train_<model>-ro_L24.pt` | no banked anchor | **yes** | 30 = 2 ds × 3 seeds × 5 folds |

Head-R needs no `fold = −1` head: the deployed-configuration head exists only to feed
`GATE-DEVFID`, which compares against **banked native trainlogs** that have no Head-R
counterpart. Stating that is cheaper and more honest than minting six heads nothing reads.

**Both lineages share every other component** — same wrapper, same fold contract, same
recipe CLI, same arm builder, same vote, same gates. The only variable is the training
cache. Head-N anchors the machinery on the banked floors; Head-R removes the OOD objection.

This adopts round-1 repairs **(a)**, **(b)** and **(c)** together: (a) is
`GATE-DOMAIN`/`GATE-ARENA` in §6, (b) is Head-R on **both** datasets rather than the one
the reviewer would have accepted, (c) is the §10.2 scope sentence.

### 3.4 The arm builder — a generic block-list construction, gated bit-exact against C01

**The round-1 finding (C-3):** v1 defined all thirteen arms afresh with
`pair(a,b) = l2(concat(l2(a), l2(b)))`, described as *"C01's `paired_key` with the modality
loop collapsed"*. That collapse was a **choice**, not a derivation, and `GATE-ALGEBRA`
could not detect a wrong one because both sides of each identity were built by the same
re-implementation. Round 1 was right, and reading the source confirms the collapse was in
fact **wrong**: C01's `paired_key` (`:1220-1239`) ends in `fuse_modalities`, which
`l2_rows` **each block again** before the final concat and row `l2` (`:1208-1217`) — an
outer normalisation v1's `pair` omitted. C01's pipeline is also **`float32`**
(`l2_rows:1200-1202` returns `astype("float32")`), which v1's `float64` construction did
not match either.

**v2's repair.** The battery defines **one** builder, parameterised by an ordered list of
blocks, in which every normalisation is C01's `l2_rows` called through the **imported**
`c01_policy_contrast_a0` module:

```
fuse(blocks)      = l2_rows(concat[ l2_rows(b) for b in blocks ])
paired(A, B)      = fuse([ l2_rows(concat[ l2_rows(A_m), l2_rows(B_m) ]) for m in blocks ])
build_views(std_blocks, ow_blocks, angles) -> the 13 arms
```

Instantiated with **two** blocks `[img, text]` it *is* C01's `prepare_views`. Instantiated
with **one** block (the fused head key) it yields the head-space arms — and the one-block
answer is now **derived by the same code path**, not chosen. This is what makes C-3's gate
possible: `GATE-C01PARITY` (§6) asserts the two-block instantiation reproduces
`prepare_views` **bit-exactly on the raw L24 features**, before any head.

**Measured today, on both datasets: bit-exact, `max|diff| = 0.000e+00` across all 13 arms**
(§7.3). The gate is not merely addable; it is verified satisfiable before freeze.

**Deferred-import note.** `c01_policy_contrast_a0.py:387` sets `np = torch = faiss = None`
and binds them only inside `import_compute_modules(config)` (`:1048-1065`). The battery must
call it before touching the algebra; calling `l2_rows` first raises
`AttributeError: 'NoneType' object has no attribute 'linalg'`, which is how I found it
(§7.3). Recorded because the campaign has a standing deferred-import audit lesson.

### 3.5 The arms

Thirteen key-space arms plus one score-derived arm, at `ro_L24` only.

| arm | role |
|---|---|
| `endpoint_std`, `endpoint_ow` | reference endpoints; also C01's `gain_controls` |
| `avg_score` | derived score control (mean of the two endpoint vote scores) |
| `endpoint_concat` (≡ `orthrot_0`) | ordinary control |
| `common` | ordinary control |
| **`displacement`** | the real first-order prompt tangent — **real arm** |
| **`common_displacement`** (≡ `orthrot_45`) | C01's **primary** — **real arm** |
| `common_interaction` | C01's secondary |
| `orthrot_{8.3, 17.6, 29.1, 60.4, 72.7, 83.8}` | the matched-block-L2 rotation family |

Angles are C01's frozen `orthogonal_rotation_control.angles_degrees`; `45°` and `0°` are
excluded from the grid **because they are the primary and `endpoint_concat`**.

### 3.6 The raw leg — gate-only, non-decisional

The same builder is run on the raw L24 features (no head) and voted in the **same folds**,
producing raw-space counterparts of every arm. It renders **no verdict** and enters **no
decision rule or multiplicity family**. Its two jobs are both gate discriminators (§6):
supplying the `ρ_raw` and raw-accuracy legs that separate *"C06's premise is false"* from
*"the instrument destroyed the object"*. This is consistent with F113 — a raw arena may
kill but may not promote, and here it does neither — and mirrors C09, whose raw leg was
confined to KILL corroboration.

---

## 4. Ambiguities in the written condition, and how each is resolved

**"Conservative" means *hardest for the falsifier to deliver the `$0` CLOSURE***, because
the closure is the irreversible action: it retires a candidate permanently and forgoes an
extraction `iteration_8_stage0_bounded_extraction_amendment` has authorized in principle,
whereas the other outcome merely queues a proposal that must still clear the unchanged
Stage-0 bar and its own two review lineages. **Round 1 ruled this correct** and attached two
conditions — the design must *disclose* what the lean buys (I-5, §5.8) and it must not
excuse an arithmetic error (H-2, §5.5). Both are now discharged.

| # | ambiguity | resolution | direction |
|---|---|---|---|
| **A1** | *"the rotations"* — best, or family? | C01's frozen `require_primary_above_all_rotation_controls`: the real arm must beat **every** rotation. Negation is the closure trigger. | as written |
| **A2** | *"the real displacement"* — `displacement` or `common_displacement`? | **Both**, disjunctively — with the disjunction now **multiplicity-corrected** (§5.5, H-2). | generous to C06, corrected |
| **A3** | *"match"* — ties? | A tie **closes** C06; the real arm must strictly exceed. C01's own `>` semantics. | as written |
| **A4** | *"in the fold-head arena"* | Round 1 ruled the arena reading correct and declined to raise it as Critical. What it did rule Critical was the **training set** (C-2). v2 runs **both** lineages (§3.3), so the reading no longer rests on one choice. | resolved by running both |
| **A5** | which layer? | **L24 only.** The L28 leg is dropped (§3.1). | narrowed |
| **A6** | seeds | Seed-mean primary over 3 head seeds, **plus 3/3 per-seed agreement** on the rotation-dominance leg. | tightens the bar |
| **A7** | *"C06 closes"* — closes **what**? | The **first-order (tangent/chord) leg only**; curvature needs ≥ 3 prompt points and is `$0`-impossible. **Round 1 ruled A7 is *not* an obstacle** — C01's battery *is* the two-point contrast, so re-running it is what the state file asks. §10.2 records curvature as unmeasured. | scope statement |

---

## 5. The pre-registered decision rule

### 5.1 Notation

For dataset `D ∈ {HateMM, MHC-ZH}`, lineage `L ∈ {Head-N, Head-R}` and head seed
`s ∈ {0,1,2}`, each arm `A` yields one OOF prediction vector over all `n` train items,
scored against **train-split** labels held out from the head that judged them.
`acc(A,D,L)` denotes the mean over the three seeds.

* **Real arms** `R = {displacement, common_displacement}`.
* **Rotation family** `Θ` = the six frozen angles.
* **Ordinary controls**, now per-arm and taken from C01's own two frozen lists
  (**H-1 repair** — v1 used only `gain_controls`, dropping `displacement` as a comparator
  for the primary, which C01's `bootstrap_comparisons.primary_vs_controls` requires):
  * for `A = common_displacement`: `C = {endpoint_std, endpoint_ow, avg_score,
    endpoint_concat, common, displacement}` — **six**;
  * for `A = displacement`: `C = {endpoint_std, endpoint_ow, avg_score, endpoint_concat,
    common}` — **five** (it cannot compare against itself).

### 5.2 SURVIVE

**On a given lineage `L`, C06 SURVIVES iff there exists `A ∈ R` such that all of
S1–S6 hold on BOTH datasets:**

| | condition | frozen source |
|---|---|---|
| **S1** | `acc(A) > max_θ acc(orthrot_θ)` **and** `mF1(A) > max_θ mF1(orthrot_θ)` | `require_primary_above_all_rotation_controls` |
| **S2** | S1's accuracy leg holds in **3/3** individual seeds | A6 |
| **S3** | `acc(A) − max_{c∈C} acc(c) ≥ 0.02` **and** the same on `mF1` | `minimum_gain_over_strongest_control = 0.02` |
| **S4** | for **every** comparator in `C ∪ Θ`: paired item-level bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05` | `minimum_bootstrap_lower_bound = 0.0`, `require_primary_bootstrap_holm_reject`, `require_rotation_bootstrap_holm_reject`, `n_bootstrap = 2000`, `statistics.seed = 20260728`, `bootstrap_lower_quantile = 0.05`, `holm_alpha = 0.05`, `holm_metrics = [accuracy, macro_f1]` |
| **S5** | **both** `displacement` **and** `common_displacement` exceed the 95th percentile of their shuffled-pair null, **and** the shuffle comparison Holm-rejects | `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject` (**I-5 repair** — v1 tested only the claimed arm and cited the wrong constant, **M-3**) |
| **S6** | **net fixes** `≥ 3` (HateMM) / `≥ 2` (MHC-ZH) against `endpoint_std`, i.e. items corrected minus items broken | `minimum_net_fixes = {HateMM: 3, MHC_zh: 2}` (**I-5 repair**) |

Plus C01's `require_no_small_displacement_dominance`, carried as `GATE-SMALLDISP` in §6
because C01 itself scopes the `endpoint_concat` small-displacement comparison as
`diagnostic_only`.

**S6 matters beyond bookkeeping.** The Gate-0 record's own closing strategic finding is
that Gate 0 *"must now screen on demonstrated conversion in the currency
`banned_constraints[10]` already names — NET ITEMS"*. v1 omitted it; a SURVIVE that never
counted net corrected-minus-broken items would have been out of step with the campaign's
current currency.

### 5.3 CLOSE

**C06 CLOSES iff the run publishes a verdict (§6) and SURVIVE is false on BOTH lineages.**

Equivalently: C06 survives if it clears S1–S6 on **either** Head-N **or** Head-R. This is
the C-2 repair at the decision level — a closure can no longer be attributed to the OOD
transplant, because the in-domain head closed it too.

### 5.4 The bootstrap unit (H-3 repair)

v1's S4 did not say what the bootstrap resamples, and its two readings implied different
families and different counts. **v2 pre-registers the reviewer's recommended unit:
resample items once (`B = 2000`, C01's frozen `statistics.seed = 20260728`), and inside
each resample average the three seeds' per-item correctness.** That yields exactly **one**
family per `(real arm, dataset, lineage)` and makes the seed axis part of the statistic
rather than a hidden multiplicity. §8 Phase 4's count is re-derived from it.

### 5.5 Multiplicity (H-2 repair)

v1 asserted the two-arm disjunction was *"absorbed"* by the S1–S6 conjunction. **That was
an arithmetic error and it is corrected, not defended:** a conjunction controls error
*within* an arm and says nothing about testing two arms and reporting SURVIVE if either
passes.

**v2 folds both real arms into a single Holm family per `(dataset, lineage)`:**

`common_displacement` 6 comparators + 6 rotations = 12; `displacement` 5 + 6 = 11;
`(12 + 11) × 2 metrics = **46 hypotheses**`, Holm at `α = 0.05`.

The two datasets remain a **conjunction** (conservative, not pooled). The two lineages are
also a conjunction for CLOSE (§5.3), which is conservative in the same direction.

### 5.6 Instrument failure

Any HALT gate failing on either dataset in either lineage ⇒ **HALT: no verdict, in either
direction**, recorded as `INSTRUMENT_INCONCLUSIVE`. It **may not be reported as a closure**;
C06's gate stays where it is and a re-run needs fresh authorization. Round 1 found no path
by which instrument failure could be reported as closure, and v2 changes nothing there.

**Finiteness (round-1 NaN note).** Every gate quantity must be asserted **finite before**
it is compared, and every gate is written in **pass-condition** form so a NaN comparison
evaluates `False` and HALTs. The specific hazard named: a fraction that is `0/0` when every
item is masked, under an implementation written as `if frac > bar: halt`, would **pass** on
NaN — the inverted-comparison form of the C09-lineage bug where an undefined `p` vetoed its
partner. The code-review lineage must check the comparison **direction** gate by gate.

### 5.7 Pre-declared expectation

**CLOSE is expected.** Grounds: C01 measured the premise rotation-indistinguishable at the
two-point case on both datasets, and the recon's structural objection (a fixed prompt
injects no per-item information) is unrebutted. **v1's third ground — the `219×`
contraction — is withdrawn (I-10):** it was measured at an untrained head, where
`Linear(3584→1024) → normalise → Hadamard → ReLU` concentrates essentially any two inputs,
so it is substantially a property of the initialisation and cannot be scored as a
successful prediction later.

### 5.8 C01 conditions deliberately not carried (I-5 disclosure)

v1 said *"every threshold in §5 is either C01's frozen value or fixed in §4"*, which read as
completeness. It was not. v2 restores four of the five omissions (S5's both-arms leg, the
shuffle Holm leg, S6's net fixes, and `require_no_small_displacement_dominance` as
`GATE-SMALLDISP`). **One is not carried, with its reason:**

* `require_accuracy_gain_over_deployed_r0_context` — its comparator
  `deployed_r0_accuracy_context_only` (`0.8505` HateMM / `0.8590` MHC-ZH) is a **raw dev-arena**
  figure at `n_dev` 107/78. Importing it into a CPU fold-head train-OOF arena would violate
  F88's binding caveat that *"a CPU-trained arm must be paired against a CPU-TRAINED
  FLOOR, never against the banked GPU floor."* The condition is **inapplicable across
  arenas**, not waived for convenience. The `GATE-FLOOR` anchors (§6) are the in-arena
  substitute.

---

## 6. Gates — HALT publishes no verdict

Every quantity is asserted finite before comparison (§5.6). All conditions must hold on
**both datasets** and, where the gate is lineage-scoped, on **both lineages**.

| gate | asserts | status |
|---|---|---|
| `GATE-DET1` | thread env exported before any python starts; `headspace_mint.det1_assert` | exercised in 7 real mints today |
| `GATE-SHA` | every frozen import and input cache matches §11; **run once in the sbatch driver**, not per process (I-3) | measured `0.12 s` |
| `GATE-FOLD` | fold assignment matches banked `vsw_ckpt/<ds>/f{0..4}.npz` | passed in all 7 mints |
| `GATE-FLOOR` | Head-N's **native-feature** OOF vote reproduces the banked fold-head floors at 4 dp on **accuracy AND macro-F1** (**H-5 repair**) — acc HateMM `0.8884/0.8858/0.8858`, MHC-ZH `0.8929/0.8895/0.8946`; **mF1** HateMM `0.8838/0.8811/0.8812`, MHC-ZH `0.8747/0.8710/0.8765`; and every `fold_acc_deployed` entry | anchors re-read from the 6 banked files today |
| `GATE-C01PARITY` | the two-block instantiation of the arm builder reproduces `c01_policy_contrast_a0.prepare_views` **bit-exactly** on the raw L24 features; any residual above C01's own `2e-6` HALTs (**C-3 repair**) | **measured `max\|diff\| = 0.000e+00`, both datasets, all 13 arms** |
| `GATE-ORBITDISP` | **(C-1 repair)** for every arm, with `ρ = ‖mean_i k_i‖` over unit keys on unmasked rows: HALT iff `ρ_head > ρ* AND ρ_raw ≤ ρ*`, with **`ρ* = 0.9772`** frozen below; and `ρ_raw` must reproduce today's frozen per-arm values at 4 dp | bar fixed from banked raw-space values, §6.1 |
| `GATE-ARMVIAB` | **(C-1 companion, refined)** for `endpoint_std` and both real arms: if head-space accuracy fails `majority + 0.02`, HALT **iff** the same arm clears it in the raw space | §6.2 |
| `GATE-ARENA` | two-sided band `majority + 0.02 ≤ acc ≤ 0.98` on `endpoint_std` and both real arms (**I-6**; C09 `:1569-1572`). Majority = `0.5995` (HateMM) / `0.6891` (MHC-ZH), measured today | upper bound is the leak catcher |
| `GATE-DOMAIN` | Head-N's `ro_L24` `endpoint_std` **recovery fraction** `(acc_ro − maj)/(acc_native − maj)` is computed and **printed on the verdict face**; reporting-only, no bar (**C-2(a)**) | §6.3 |
| `GATE-NESTED` | **per item**, the head that scored it excluded its fold; emitted as a check count equal to the item count (**I-6**; C09 `:1557-1563`) | catches a bank/query index slip that would inflate every arm |
| `GATE-ZEROOP` | `orthrot_0` and `endpoint_concat` produce **identical predictions**, and likewise `orthrot_45` and `common_displacement` — not merely keys agreeing at `2e-6`, since a `2e-6` key difference can reorder a top-20 neighbourhood (**I-6**; C09 `:1566-1568`) | strictly stronger than v1's `GATE-ALGEBRA` |
| `GATE-ALGEBRA` | key-level: `max\|orthrot_0 − endpoint_concat\| ≤ 2e-6` and `max\|orthrot_45 − common_displacement\| ≤ 2e-6`, **zero mask applied** | retained beneath `GATE-ZEROOP` |
| `GATE-IDPARITY` | every ro cache's `ids` order and `labels` identical to the native bank (`c02_a0_mint.py:69-71` unwrapping, `:72-76` assertions — **M-4**) | measured true on all 4 L24 files |
| `GATE-ZEROMASK` | measured exact-zero row set equals the pre-registered set — HateMM `{355}` (`hate_video_95`, C01's `authorized_null`), MHC-ZH `{}` — on both policies; and those rows give identical head keys under both policies | measured true today |
| `GATE-DUALPATH` | C01's `displacement_registered_null_exclusion`: masked path and physically-removed path agree **exactly** on every arm's predictions | HateMM only |
| `GATE-SHUFFLEFIX` | row 355 is a **fixed point** of every shuffle permutation and excluded from the remaining-source bijection (**I-7**; C01's `shuffle_fixed_point_bijection` + `permutation_pairing`) | HateMM only |
| `GATE-SMALLDISP` | C01's `require_no_small_displacement_dominance` at its frozen `small_displacement_train_quantile = 0.1` | §5.8 |
| `GATE-LEDGER` | C09's **full declared-count** predicate set (**H-4**), §12 | binding leg: dev-or-test labels into any decision quantity `= 0` |
| `GATE-DEVFID` | `headspace_fidelity.py` on Head-N's 6 full heads | **reporting only, does not gate** |

### 6.1 `GATE-ORBITDISP` — the C-1 repair, and why the bar is what it is

**The failure v1 could not catch.** Every arm is `l2`-normalised before `deployed_vote`, and
`mechfix_ops._norm32` normalises again at the index, so **the retrieval key discards `‖Δ‖`
entirely and keeps only direction**. v1's `GATE-ORBITSCALE` watched magnitude. If the
trained head's two policy outputs differ by an approximately constant offset —
`e_ow − e_std ≈ c·u + small residual`, exactly what "the head is invariant to this axis"
looks like, and exactly the mechanism §5.7 names as its structural objection — then every
item's `displacement` key is ≈ `u`, its neighbourhood is arbitrary, every rotation beats it
trivially, `‖Δ‖ ≈ c` sits comfortably above `1e-3`, and the design publishes a
clean-looking **CLOSE**. v1's gate also sat ~1000× above the `float32` noise floor it was
described as guarding, so no change of constant could have fixed it: the **quantity** was
wrong.

**The quantity v2 uses** is the reviewer's: `ρ_space = ‖mean_i k_i‖` over unit keys, where
`ρ → 1` means every item's key points the same way.

**The bar, fixed now from banked, label-free, already-frozen data.** I computed `ρ` for all
13 arms in the **raw** L24 space, both datasets, zero row masked (§7.3). Every one of those
arms is an arm the raw arena actually scored, so their `ρ` values bound what "still
informative" looks like in this feature family. **`ρ* = 0.9772`**, the measured maximum over
all 13 arms × 2 datasets (`endpoint_std` on MHC-ZH, `0.977223`, truncated down to 4 dp).

**The gate, in the reviewer's own case split:**

* `ρ_raw ≤ ρ*` and `ρ_head > ρ*` ⇒ the instrument destroyed the object ⇒ **HALT**;
* `ρ_raw > ρ*` and `ρ_head > ρ*` ⇒ both spaces degenerate ⇒ C06's premise is false ⇒ **no
  HALT, CLOSE is warranted**;
* otherwise ⇒ no HALT.

Applied to every arm, not only the real ones: a collapsed *rotation* arm would bias the
comparison toward SURVIVE, and catching it costs nothing.

**`ρ_raw` is also a free fidelity anchor.** Today's measured raw values are frozen into the
prereg and the gate re-derives them in-run at 4 dp:

| arm | `ρ_raw` HateMM | `ρ_raw` MHC-ZH |
|---|---|---|
| `endpoint_std` | 0.9682 | **0.9772** |
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

This reproduces round 1's measurement (c) to 4 dp and carries its point: **in the raw space
the displacement direction is the *most* item-dispersed arm of the thirteen** — more
dispersed than `endpoint_std` — which is why C01 could score it at `0.8505 / 0.8846` at all.
A head-space `ρ(displacement)` approaching `1.0` would therefore be **the head's doing**,
and without this gate the design would have attributed it to C06.

### 6.2 `GATE-ARMVIAB` — the C-1 companion, adopted in a refined form

Round 1 prescribed extending `GATE-VIABILITY` to *"require that both arms in `R` clear the
majority-class rate."* **I adopt the gate and refine its form, and this is the one place in
v2 where I have not taken a repair verbatim.**

The reason: a plain one-sided HALT on the real arms would fire on exactly the outcome the
falsifier exists to detect. If C06's premise is false, `displacement` in head space *should*
sit near the majority rate — that is a **warranted CLOSE**, and a one-sided gate would
convert it into a HALT, leaving C06 gated forever on an instrument that can never close it.

The refinement keeps the reviewer's intent and adds the discriminator the reviewer's own
C-1 argument supplies — the **raw** space:

* head-space arm fails `majority + 0.02` **and** the same arm fails it in the raw space ⇒
  genuine negative ⇒ **no HALT**;
* head-space arm fails **and** the raw arm clears ⇒ the instrument destroyed it ⇒ **HALT**.

This is structurally the same two-case logic as `GATE-ORBITDISP`, applied at the prediction
level instead of the key level, which is what the reviewer asked the companion to add. The
raw accuracies it needs are **not measured in this document** — computing them now would
measure a decision-relevant input before freeze — so the refinement rests on the logical
argument above, not on new data. **Round 2 should rule on it explicitly.**

### 6.3 `GATE-DOMAIN` — reporting, with no invented bar

C-2(a) asked for a head-space input-domain fidelity gate anchored on `GATE-FLOOR`. The
non-arbitrary parts of that are already gates: `GATE-ARENA`'s two-sided band and
`GATE-ARMVIAB`'s case split both fire on a collapse toward the majority rate. What remains
— *how much* degradation makes an arm comparison uninterpretable — has **no non-arbitrary
bar**, and inventing one would be exactly the un-preregistered threshold this house treats
as a High finding. So `GATE-DOMAIN` computes the recovery fraction
`(acc_ro − maj)/(acc_native − maj)` for `endpoint_std` under Head-N, and **requires it to
appear on the verdict face and in §10.2's scope sentence**, with no threshold. Head-R is the
structural answer to the same worry (§3.3), and it does carry a decision consequence.

---

## 7. Dry-check — what was executed, and what it found

Login node `foscsmlprd01`, conda `HateVideo`, 8 threads, DET-1 exported, outputs only in
the session scratchpad. Zero GPU, zero SLURM, zero test-split file opened, zero write into
`data/`, `artifacts/` or `logging/` (verified by `find -newermt`).

### 7.1 Real inputs

All 8 ro caches and 4 native caches loaded through the real `torch.load` path. `n = 744 /
579`; features `(n, 3584)`; `ids` order-identical and `labels` identical to the native bank
on every file. Exact-zero rows: HateMM `{355}` in both modalities of all four ro caches and
the native cache; MHC-ZH none. Class balance measured: HateMM `posrate 0.4005` ⇒ **majority
`0.5995`**; MHC-ZH `posrate 0.3109` ⇒ **majority `0.6891`**.

### 7.2 Seven real mint units, at real scale

`headspace_mint.py` run **unmodified**, sha256 verified, for Head-N; a scratchpad harness
mirroring `c02_a0_mint.py`'s monkeypatch structure for Head-R.

| lineage | unit | measured wall |
|---|---|---|
| Head-N | HateMM fitting-pool head | **40.39 s** |
| Head-N | MHC-ZH fitting-pool head | **34.40 s** |
| Head-N | HateMM full-train head | **49.30 s** |
| Head-N | MHC-ZH full-train head | **38.87 s** |
| Head-R | HateMM fitting-pool head (`ro_L24`-trained) | **37.46 s** |
| Head-R | MHC-ZH fitting-pool head (`ro_L24`-trained) | **27.54 s** |
| Head-R | HateMM full-train (timed, then **dropped** from the design — §3.3) | 38.38 s |

Fold parity passed in all seven. Peak RSS of the HateMM Head-N fold mint,
`/usr/bin/time -v`: **1 305 984 KB = 1.25 GiB**, agreeing with C09's measured 1.22 GiB.
**The Head-R harness trains and times only** — it computes no arm keys and no accuracy, so
no arm ordering was observable.

### 7.3 The payload path, executed end to end

* **`GATE-C01PARITY` verified satisfiable.** The generic block-list builder, instantiated
  with two blocks, was compared against `c01_policy_contrast_a0.prepare_views` on the real
  raw L24 features: **bit-exact, `max|diff| = 0.000e+00`, all 13 arms, both datasets.**
* **C01's own algebra guard re-measured** on the same call: `8.94e-08` (θ=0, both) and
  `1.19e-07` / `8.94e-08` (θ=45), matching the R14 record.
* **Deferred-import defect found and recorded** (§3.4): calling `l2_rows` before
  `import_compute_modules(config)` raises `AttributeError: 'NoneType' object has no
  attribute 'linalg'`, because `c01_policy_contrast_a0.py:387` sets `np = None` until the
  guards pass.
* **`ρ_raw` measured for all 13 arms, both datasets** (§6.1 table) — the frozen bar.
* **A real `classifier_hateClipper` forward** over the real ro caches, all 13 head-space
  arms built, and **every arm voted on a real fold-cell** through the frozen
  `deployed_vote`.

**Blindness.** Every head used in an arm-building or voting dry check is **untrained**, so
every operation is real at real scale while the numbers are scientifically void. No arm
accuracy was computed, printed or recorded, on any dataset, in either lineage. Round 1
confirmed this discipline was the right call and found nothing inconsistent with it.

### 7.4 The zero-row amplification finding (carried from v1, confirmed by round 1)

At `θ = 45°` an exact-zero displacement row is renormalised into a **full-norm arbitrary
direction** by the 1-ULP `cos45 − sin45 = 1.11e-16` asymmetry, producing an `8.31e-02`
algebra-guard violation attributable entirely to HateMM row 355. Masked, the guard passes at
`4.95e-15` / `5.24e-15`.

**Round 1 independently verified the mechanism and closed the follow-up question I had not
asked:** `l2` amplification requires `‖ow − std‖ ≲ 1.6e-16`, and no unmasked row on either
dataset comes within fifteen orders of magnitude of that, so the exactly-zero row is the
only case and `GATE-ZEROMASK` covers it. A near-zero-but-nonzero row is **not** a live risk
at these angles.

### 7.5 The magnitude finding, restated correctly (I-4)

v1 reported the head-space orbit contraction and said `GATE-ORBITSCALE` *"would pass at
`1/744 = 0.0013`"*. **That count is on a population the gate excludes.** Measured with the
mask applied: **`0/743`** on HateMM and `0/579` on MHC-ZH, i.e. `0.0000`. The magnitude
gate is gone from v2 anyway (C-1), but the misstatement is corrected here because v1's
number is on the record.

| space | median `‖e_ow − e_std‖` | median `cos` | unmasked rows below `1e-3` |
|---|---|---|---|
| raw fused key | `0.7016` | `0.753874` | `0 / 743` (HateMM), `0 / 579` (ZH) |
| head space (untrained head) | `0.0032` | `0.999995` | `0 / 743` |

The head-space row is retained as a **hazard note**, not as evidence: it is an untrained-head
measurement, and §5.7 no longer leans on it (I-10).

### 7.6 v1's `U2` and `U4` are withdrawn and re-measured (H-6)

Round 1 found that v1's `U4` (`0.1241 s`, defined as 2 arms × 5 folds + rebuild) was
**smaller than the sum of its own constituents** (`5×0.0042 + 5×0.0218 = 0.130 s`), leaving
negative time for the rebuild — `rule_1_compute_projection`'s falsified pattern.

**Cause, found on re-measurement: v1 timed the arm matrices as `float64` and repeated a
single fold five times instead of rotating the five real folds.** `deployed_vote` copies
its inputs to `float32` via `_norm32`, so a `float64` arm matrix pays a conversion on every
call that the real battery will not pay — C01's own `l2_rows` returns `float32`
(`:1200-1202`), so the arms are `float32` by construction.

**Re-measured with `float32` arms over the five real folds, and it now reconciles:**

| unit | v1 (withdrawn) | v2 (measured) |
|---|---|---|
| `U2a` vote, 1024-d, per fold-cell | 0.0042 s | **0.00305 s** |
| `U2b` vote, 2048-d, per fold-cell | 0.0218 s | **0.00629 s** |
| `U4` one null draw | 0.1241 s | **0.08908 s** |

Reconciliation: `5 × 0.00305 + 5 × 0.00629 = 0.04674 s` of votes, leaving
`0.08908 − 0.04674 = **0.04234 s**` for the arm rebuild — positive and plausible.

### 7.7 The remaining units, all measured

| unit | measured |
|---|---|
| `U1` head forward over one real ro cache, `n = 744` | 0.0461 s |
| `U2c` vote, 7168-d (raw fused), per fold-cell | 0.04239 s |
| `U2d` vote, 14336-d (raw paired), per fold-cell | 0.08098 s |
| `U3` bootstrap `B = 2000`, one comparison, both metrics | 0.126 s |
| `U5` `GATE-C01PARITY` per dataset (`prepare_views` 6.427 + builder 4.626 + compare 0.213) | 11.27 s |
| `U6` `ρ` over 13 raw arms, per dataset | 0.62 s |
| `U7` `GATE-SHA` over 8 caches + 6 modules, once | 0.12 s |
| `U8` ro cache `torch.load`, 2 files, per process | 0.033 s |
| `U9` `GATE-DEVFID` per `(dataset, seed)` | 3.70 s (HateMM) / 3.49 s (ZH) |

**`U9` correction, disclosed.** My first `GATE-DEVFID` timing (`3.32 / 3.19 s`) was a
**failure path**: `headspace_fidelity.py` defaults to `--seeds 0,1,2` and only seed-0 full
mints existed, so it errored and wrote no file — my shell captured `echo`'s exit status
rather than python's and I nearly recorded a crash as a measurement. Re-run with
`--seeds 0`, both datasets exit `0` and write their JSON; the table above is the corrected
per-`(dataset, seed)` unit, multiplied by 3 seeds in §8.

### 7.8 Dry-check cost, disclosed

Round 1 ruled v1's trade correct — when a brief's CPU budget and a standing
`TARGET_STATE.json` rule are incompatible, the standing rule wins and the overrun is
disclosed — while noting, fairly, that the incompatibility was **knowable before the burn**
from C09's banked mint costs and should have been raised as a conflict rather than resolved
unilaterally. **Recorded, and applied here: I am raising it now rather than afterwards.**
v2's dry check consumed a further **≈ 5.5 wall-minutes / ≈ 30 CPU-minutes** (seven mints
plus the parity, dispersion and timing runs), all `$0`, zero GPU, on a 64-core node. The
seven mint units are unavoidable under R1 — one unit *is* ~4 CPU-minutes — and the Head-R
units did not exist before v2 required them.

---

## 8. Compute projection — measured unit × explicit count

Every unit is measured in §7 at real scale on the real banked data. Nothing is
extrapolated from a reduced-scale run, and nothing budgeted sits **inside** the projection
(I-2).

### Phase 1 — Head-N mints, 36 units

| unit | count | measured | product |
|---|---|---|---|
| HateMM fitting-pool | 3 seeds × 5 folds = **15** | 40.39 s | `605.9 s` |
| HateMM full-train | **3** | 49.30 s | `147.9 s` |
| MHC-ZH fitting-pool | 3 × 5 = **15** | 34.40 s | `516.0 s` |
| MHC-ZH full-train | **3** | 38.87 s | `116.6 s` |
| | **36** | | **1386.4 s** |

### Phase 1R — Head-R mints, 30 units (no full-train heads, §3.3)

| unit | count | measured | product |
|---|---|---|---|
| HateMM fitting-pool | 3 × 5 = **15** | 37.46 s | `561.9 s` |
| MHC-ZH fitting-pool | 3 × 5 = **15** | 27.54 s | `413.1 s` |
| | **30** | | **975.0 s** |

### Phase 1b — key extraction inside the mints (I-1 repair)

Head-N per fold mint: native + `{std, ow}@L24` = **3** train forwards; per full mint: those
3 plus **1 native dev** forward = 4. **No `dev_seen` ro forward occurs** — v1's count of five
dev forwards implied four `dev_seen_*-ro_*` opens that §3.1 says never happen.
Head-R per mint: `{std, ow}@L24` = **2**.
`(30 × 3) + (6 × 4) + (30 × 2) = 90 + 24 + 60 = **174** forwards × 0.0461 s = **8.0 s**`
(the `U1` unit is HateMM-scale and applied to MHC-ZH too — conservative).

### Phase 1c / 1d — per-process loops (I-3 repair)

`66 mint processes × 0.033 s = **2.2 s**` of ro cache loads.
`GATE-SHA` **runs once in the sbatch driver**, not per process: `1 × 0.12 s = **0.1 s**`.

### Phase 2 — head-space arm votes. Fold-cells = 5 × 3 seeds × 2 ds × **2 lineages** = 60

| | count | unit | product |
|---|---|---|---|
| 1024-d arms (`endpoint_std`, `endpoint_ow`, `common`, `displacement`) | `4 × 60 = 240` | 0.00305 s | `0.7 s` |
| 2048-d arms (`endpoint_concat`, `common_displacement`, `common_interaction`, 6 × `orthrot`) | `9 × 60 = 540` | 0.00629 s | `3.4 s` |
| `GATE-FLOOR` native vote (Head-N only) | `1 × 30 = 30` | 0.00305 s | `0.1 s` |
| `avg_score` (derived from banked endpoint scores) | — | — | `0 s` |
| | | | **4.2 s** |

### Phase 2R — raw-space arm votes, gate leg. Fold-cells = 5 × 2 ds = 10 (seed-free)

`4 × 10 = 40 × 0.04239 = 1.7 s` + `9 × 10 = 90 × 0.08098 = 7.3 s` = **9.0 s**

### Phase 2C / 2D — the two new gates

`GATE-C01PARITY`: `2 datasets × 11.27 s = **22.5 s**`.
`ρ`: raw `2 × 0.62` + head `2 lineages × 2 ds × 0.62` = **3.7 s** (head-space arms are
1024/2048-d against the unit's 7168/14336-d, so this over-states).

### Phase 3 — shuffled-pair null, C01 frozen `n_id_hash_permutations = 256`

`256 draws × 3 seeds × 2 ds × 2 lineages = **3072** draw-units × 0.08908 s = **273.6 s**`

### Phase 4 — bootstrap, re-derived from §5.4's unit

The unit resamples items once and averages the three seeds inside each resample, so the
seed axis is **inside** the statistic and no longer multiplies the count (v1's error, H-3).
Comparisons per `(dataset, lineage)`: `12 + 11 = 23`.
`23 × 2 ds × 2 lineages = **92** comparison-cells × 0.126 s = **11.6 s**`

### Phase 5 — null-row sensitivity leg (HateMM only, `n = 743`, no re-mint)

Re-runs Phases 2, 2R, 3, 4 restricted to HateMM:
`(4.2 + 9.0 + 273.6 + 11.6) / 2 = **149.2 s**`

### Phase 6 — `GATE-DEVFID` (I-2 repair: measured, not budgeted)

`3 seeds × 3.70 s + 3 seeds × 3.49 s = 11.1 + 10.5 = **21.6 s**`

### Total

| phase | seconds |
|---|---|
| 1 Head-N mints | 1386.4 |
| 1R Head-R mints | 975.0 |
| 1b key extraction | 8.0 |
| 1c ro cache loads | 2.2 |
| 1d `GATE-SHA` (once) | 0.1 |
| 2 head-space arm votes | 4.2 |
| 2R raw-space arm votes | 9.0 |
| 2C `GATE-C01PARITY` | 22.5 |
| 2D dispersion `ρ` | 3.7 |
| 3 shuffled-pair null | 273.6 |
| 4 bootstrap | 11.6 |
| 5 null-row leg | 149.2 |
| 6 `GATE-DEVFID` | 21.6 |
| **corroborating total (all measured)** | **2867.1 s = 47.8 min** |
| **conservative (× 1.25, shared-node contention)** | **3583.9 s = 59.7 min** |

**Declared slack, outside the projection** (I-2): `30 s` for ledger aggregation and JSON
emit. It is labelled slack, not a measurement, and is deliberately **not** summed above.

**Projected peak RSS ≈ 1.3 GiB** (measured 1.25 GiB for the dominant unit; raw-space arm
matrices add `744 × 14336 × 4 B = 43 MB` each). Request 32 GB.

**Sensitivity.** Phase 3 is now `9.5 %` of the total (v1: 17.9 %), and its unit reconciles
against `U2a`/`U2b` (§7.6). A 2× miss on Phase 3 moves the total to
`2867.1 + 273.6 = 3140.7 s ≈ 52 min`; a 5× miss to `2867.1 + 4 × 273.6 = 3961.5 s ≈ 66 min`.
The dominant risk is now the mints, which are measured directly and account for `82 %` of
the total. **If the realized cost exceeds the conservative total by more than 2×, that is
itself a reportable process finding.**

---

## 9. Heartbeat specification

* One progress file `$BASE/progress/C06_PROGRESS.txt`, created by the sbatch driver before
  the first python process starts.
* Every python process appends through a handle opened `buffering=1` (line-buffered).
* Each line carries: ISO-8601 timestamp · phase · units done / units total · elapsed
  seconds · elapsed ÷ **§8's frozen projected** value.
* Granularity: one line per mint (66), **one line per training epoch within a mint**, one
  per `(dataset, seed, lineage)` arm block (12), one per 32 null draws (96), one per
  bootstrap block, one per gate, one per verdict field.
* **The within-mint epoch line is new (round-1 gap).** The longest unit is a 49.30 s
  full-train mint; under §8's `× 1.25` contention factor the worst-case silent interval was
  `61.6 s`, over the stated `~60 s`. `headspace_mint.py`'s `_metrics_spy` already produces
  one `eval_curve` row per epoch × split, so an epoch-level line is free and drops the
  worst-case interval to ~2 s.
* The bash driver **also** echoes a line per mint, unbuffered, so a phase boundary survives
  a wedged python process.
* **What the code-review lineage must verify** (round 1 correctly says a design review
  cannot): the handle is opened `buffering=1` and never re-wrapped; the driver's echo is
  unbuffered and does not inherit a block-buffered stdout under `sbatch`; all 66+ processes
  **append** rather than truncate and concurrent appends do not interleave partial lines;
  the HALT path writes a final line before exit; and the `elapsed ÷ projected` denominator
  is §8's frozen number, not something recomputed at run time.

---

## 10. Scope of any verdict

### 10.1 The prompt/readout-span confound — declared, as the gate requires

`src/utils/generate_VideoMLLM_embedding_readout_HF.py:73-89` defines the cells as
`("ro_L24", "baseline", "prefix", "response", LAYER_MID)` versus
`("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`: the `ow_` cell changes
**the prompt kind and both readout spans**. The two orbit endpoints therefore differ in
more than one respect. **No result of this battery can attribute an effect — or its absence
— to the prompt alone.** Round 1 verified the quotation against the source.

### 10.2 What a CLOSE would and would not close

A CLOSE closes **C06's first-order (tangent/chord) prompt-orbit route in the fold-head
arena at `ro_L24`, on `HateMM (-LoRA-curric)` and `MHC_zh (-LoRA)`, under BOTH a
native-trained deployed head applied out of domain AND an `ro_L24`-trained in-domain
head.** The measured `GATE-DOMAIN` recovery fraction (§6.3) must be stated in the same
sentence. It does **not** establish:

* anything about **curvature** — two prompt points give a chord; ≥ 3 require extraction and
  are `$0`-impossible (A7);
* anything about a head **retrained per arm** — F66's trained-reshaping caveat stands
  (§15.3);
* anything about a **different readout span**, or a prompt axis without the §10.1 confound;
* anything about **L28** or any layer other than L24 (§3.1);
* anything about the **test split** — no test artifact is opened at any point.

### 10.3 What a SURVIVE would license

Only what the gate says: that C06 *"has earned its extraction"*. The `1.7–2.5 GPU-h`
bounded extraction may then be **proposed** under
`iteration_8_stage0_bounded_extraction_amendment`, with its own preregistration, its own
independent design review, its own separate code/resource review lineage and its own
main-dialogue authorization. **A SURVIVE is not a Stage-0 PASS and authorizes no GPU.**

### 10.4 Bans checked

F80's object is prompt **language** (unconditional leg scoped to `MHC_zh`); F70's object is
individual **readout cells**. C06's object is the **relation between** two cells, which is
neither — round 1 tested this warrant and confirmed it. The multi-prompt **ensembling**
carve-outs in both bans have C14 as their object and are not relied on. This battery forms
**no ensemble of prompt predictions**: `avg_score` is C01's own frozen `gain_control`
serving as a control, not a method arm.

**M-2, added where a reader will look for it:** `endpoint_std` and `endpoint_ow` **are**
literally the two cells F70 priced. They enter here as **controls**, not as claims — which
is why the object-mismatch warrant is unaffected, and why saying so is better than leaving
the reader to notice it.

**Hard constraints: none touched** — no OCR, no cross-dataset mixing, no external API,
single-dataset train split, parent-video binary label only, no ensembles, no size scaling,
SLURM-only.

---

## 11. Frozen imports and inputs — sha256, measured 2026-08-04

**Imported unmodified, sha256 asserted at run time:**

| module | sha256 |
|---|---|
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` |
| **`scripts/analysis/c01_policy_contrast_a0.py`** (now **imported**, for `GATE-C01PARITY`) | `d2b9c2ff909c07518ae35526db9550df655fb4af395cc7a0899f83e48db1b855` |
| `scripts/analysis/c09_guard/c09guard.py` | `aed50842c232105f1b06182aa89512ee89dd050bdcaedec2706062c9d745f062` |
| `scripts/analysis/c09_guard/sitecustomize.py` | `b238789fd80076b0b890c4894fd8b69255792af51c80cd9fe2d6db6c53383850` |

**Read for definitions, thresholds and provenance (I-9 repair — the executed C01 A0 is
`v4`, and its two artifacts are now pinned):**

| file | sha256 |
|---|---|
| `configs/c01/c01_a0_v4.json` | `2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6` |
| `scripts/analysis/c01_policy_contrast_a0_v4.py` | `3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514` |
| `scripts/analysis/c01_policy_contrast_a0_v3.py` | `40b35eee2fb6fdbdb21fe9b4acfdcebf003c121c76492b898fbd2ea9b8c34dfb` |
| `configs/c01/c01_a0_v3.json` | `4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5` |
| `configs/c01/c01_a0_v2.json` | `f3997bddb4788d451ae5f90d9d03d096df3de383f8133a6d3818d97a241563f5` |
| `scripts/analysis/c02_a0_mint.py` | `e6430b76b7ccdd831ddb9939500aa24ea70d9662b62b955a2a11273a3b00ac1b` |

**The v4 → v2 chain, so a reader can verify the thresholds without tracing it themselves.**
`TARGET_STATE.json::c01_a0_v4_typed_audit_repair` pins v4's config and analysis (both
digests above, re-verified today) with namespace
`artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4`. v4's `frozen_v3` block points at v3 with
`reuse_policy: sha256_exact_import_then_v4_audit_schema_override_only` and
`scientific_thresholds_exact: true`; v3's `scientific_base` points at **v2** with the same
flag. Separately, `c01_policy_contrast_a0_v4.py:52-62` loads `_v3.py` under a sha256 check,
and `_v3.py` loads `base = c01_policy_contrast_a0.py` — the file this battery **imports**,
and where `orthogonal_blocks` sits at line 1272. **Config chain v4 → v3 → v2; algebra chain
v4 → v3 → base.** Every threshold quoted in §5 comes from v2 through that chain, and every
line of algebra from `base`.

**Input caches — `train` split only for ro; the native `dev_seen` is opened by the mint:**

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

The four `ro_L28` files are **no longer read** and are removed from the manifest (§3.1).

Plus the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` (`GATE-FLOOR` anchors)
and the ten banked `vsw_ckpt/{hatemm,zh}/f{0..4}.npz` (fold parity).

**New code, confined to the battery:** `scripts/analysis/c06_falsifier_mint.py`,
`scripts/analysis/c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`,
`scripts/slurm/c06_falsifier_cpu.sbatch`. Nothing frozen is edited or forked.

---

## 12. Label and split discipline

**Test contact: none, enforced in three layers.** (1) `headspace_mint.py:106-116` replaces
`torch.load` with a guard raising on any path containing `test_seen` / `/test`. (2) The
wrapper asserts `split == "train"` on every ro-cache load and refuses any non-`train_` path
(`c02_a0_mint.py:63-66` pattern). (3) The frozen `c09_guard` `sitecustomize` installs an
`open()`-level, component-wise, repo-scoped predicate at interpreter startup in **every**
process. Round 1 confirmed this is stronger than required and that the `test_seen` ro
caches on disk create no opening.

**`GATE-LEDGER` — C09's full declared-count predicate set (H-4 repair).** v1 asserted only
`test_path_opens == 0`. C09's ledger (`C09_A0_V17_RECORD.md:1549-1554`) declares counts
rather than asserting a single zero, and v2 adopts that shape:

| predicate | expected |
|---|---|
| `test_path_opens` | **0** |
| `test_label_materialisations` | **0** |
| `dev_path_opens` | **nonzero, declared** — 36 (every Head-N mint loads the native `dev_seen`) + `GATE-DEVFID` reads; measured and reported, not asserted zero |
| `dev_label_materialisations_outside_decisions` | **nonzero, declared** — 36, one per Head-N mint |
| `dev_or_test_labels_into_decision_quantities` | **0** — the binding leg |
| `banked_trainlog_opens` | declared, `GATE-DEVFID` only |
| processes reporting | **66 mints + 6 fidelity + 1 arena**, all reporting |
| predicate coverage | **re-derived in-job**, not asserted in a comment |

**v1's §12 sentence was false and is withdrawn.** It said *"`dev_seen` labels are
materialised only inside `headspace_fidelity.py`."* In fact `headspace_mint.py:199` loads
`dev_seen_*.pt` unconditionally on **every** mint, `:322` writes `lab_dev` into **every**
mint `.npz`, and at `fold == −1` (`:229`) the real dev split **is** the training dev set, so
dev labels enter `run_rac.main` and the `eval_curve` on 6 of the 36 Head-N mints. Head-R
mints open no dev file. None of this reaches a decision quantity — which is exactly why the
binding ledger leg is the *into-decision* zero, not a blanket dev zero.

**Labels.** Train-split labels only for every decision quantity. Every query's label is held
out from the head that judges it (`GATE-NESTED` checks this per item).

**No selection anywhere.** No arm, angle, layer, seed, lineage, threshold or `k` is chosen
by looking at a result. Every threshold in §5 and §6 is either C01's frozen value, C09's
banked constant, or fixed in §4/§6.1 from banked label-free measurements before the run.
§5.8 lists what is deliberately not carried.

---

## 13. Execution boundary

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** Round 1 independently confirmed both legs of this: F88's `$0`
forensics name no non-SLURM channel and price a 52-second process, which is no precedent
for a ~48-minute job; and CLAUDE.md's standing rule, C01's frozen
`execution.require_slurm = true, cpu_only = true, required_cpus = 8` (with
`required_memory: 32G` in v3), and the C02/C09 A0 precedents all point the same way.

**Queue routing.** The cloud route is inapplicable on the decisive reason round 1 endorsed:
`GATE-FLOOR` anchors to six floors measured **locally** on `foscsmlprd01`, and the
same-table-same-hardware precondition would require re-minting all six on cloud hardware,
costing more than the job. `squeue` is read at submission time, not now. **v1's third
reason — *"at 44 min this is not long-running"* — is withdrawn:** round 1 is right that a
projection may not be trusted for routing until measured, which is the very rule this
document is built around.

**Not authorized by this document.** No prereg is frozen, no hash is frozen, no job is
submitted. Required before anything runs: an independent design review to GO (0C/0H/0I), a
**separate** code/resource review lineage over the battery script, and main-dialogue
authorization.

---

## 14. Findings disposition — 23 of 23 adopted, 0 rebutted

| finding | disposition | where in v2 |
|---|---|---|
| **C-1** dispersion, not magnitude | **ADOPTED** — `GATE-ORBITSCALE` deleted; `GATE-ORBITDISP` added with `ρ* = 0.9772` fixed from banked raw values; applied to all 13 arms; `ρ_raw` doubles as a fidelity anchor | §6, §6.1 |
| **C-1 companion** both real arms clear majority | **ADOPTED, REFINED** — `GATE-ARMVIAB` as a two-case gate (head fails ∧ raw clears ⇒ HALT), because a one-sided HALT would fire on the warranted CLOSE. Reason stated; flagged for round 2 | §6.2, §15.5 |
| **C-2** OOD transplant; C02 precedent miscited | **ADOPTED (all three repairs)** — (a) `GATE-ARENA` + `GATE-ARMVIAB` + `GATE-DOMAIN`; (b) Head-R on **both** datasets, co-primary; (c) verdict-face re-scope. v1's precedent claim withdrawn in full | §3.3, §5.3, §6, §10.2 |
| **C-3** unanchored arm algebra | **ADOPTED** — generic block-list builder importing `c01_policy_contrast_a0`; `GATE-C01PARITY` bit-exact against `prepare_views`; **verified today at `0.000e+00`**. v1's `pair` was in fact wrong (missing outer `fuse` normalisation, wrong dtype) | §3.4, §6, §7.3 |
| **H-1** `displacement` missing as comparator | **ADOPTED** — per-arm control sets from C01's own two lists; 6 for the primary, 5 for `displacement` | §5.1 |
| **H-2** disjunction uncorrected | **ADOPTED** — single Holm family of **46** hypotheses per `(dataset, lineage)` | §5.5 |
| **H-3** bootstrap unit unstated | **ADOPTED** — resample items once, average seeds inside each resample; Phase 4 re-derived | §5.4, §8 |
| **H-4** dev-label sentence false | **ADOPTED** — sentence withdrawn; C09's full declared-count ledger adopted | §12 |
| **H-5** `GATE-FLOOR` accuracy only | **ADOPTED** — macro-F1 anchors added, all six cells | §6 |
| **H-6** `U4` below its constituents | **ADOPTED** — cause found (`float64` arms, one fold repeated); `U2a`/`U2b`/`U4` withdrawn and re-measured; now reconciles with a positive rebuild residual | §7.6 |
| **I-1** dev ro-forwards counted but never opened | **ADOPTED** — count corrected to 174; native `dev_seen` added to §11/`GATE-SHA` | §8, §11 |
| **I-2** Phase 6 scope; budgets inside a projection | **ADOPTED** — `GATE-DEVFID` measured per `(dataset, seed)` × 3 seeds; slack moved outside the projection | §7.7, §8 |
| **I-3** three uncounted per-process loops | **ADOPTED** — ro loads counted (Phase 1c); `GATE-SHA` moved to the driver, once | §8 |
| **I-4** gate value on an excluded population | **ADOPTED** — restated with the mask applied (`0/743`) | §7.5 |
| **I-5** five C01 conditions dropped silently | **ADOPTED** — four restored (S5 both-arms, shuffle Holm, S6 net fixes, `GATE-SMALLDISP`); the fifth disclosed with its reason | §5.2, §5.8 |
| **I-6** three C09 gates missing | **ADOPTED** — `GATE-NESTED`, `GATE-ZEROOP`, `GATE-ARENA` added | §6 |
| **I-7** shuffle fixed-point guard | **ADOPTED** — `GATE-SHUFFLEFIX` | §6 |
| **I-8** L28 not a sibling orbit | **ADOPTED** — leg dropped entirely | §3.1 |
| **I-9** C01 v4 artifacts absent | **ADOPTED** — v4 digests and the v4→v3→v2 / v4→v3→base chain recorded | §11 |
| **I-10** untrained contraction as evidence | **ADOPTED** — clause deleted from §5.7 | §5.7 |
| **M-1** `n_train` header | **ADOPTED** | §3.1 |
| **M-2** F70's two cells enter as controls | **ADOPTED** | §10.4 |
| **M-3** wrong constant cited for shuffle p95 | **ADOPTED** | §5.2 S5 |
| **M-4** `c02_a0_mint.py` line numbers | **ADOPTED** | §6 |

**Round-1 rulings carried without change:** the direction of "conservative" (§4); A7 is not
an obstacle (§10.2); `$0` closure need not survive per-arm retraining (§15.3); SLURM and the
login-node dismissal (§13); the untrained-head blindness discipline (§7.3); HALT semantics
(§5.6); §5.7 creates no tuning path.

---

## 15. Rulings and open issues for round 2

1. **The arena reading (§3.A).** Round 1 ruled the fold-head arena reading correct and
   declined to make A4 Critical; what it ruled Critical was the **training set**. v2 removes
   that by running both lineages. **No open question remains on the arena itself.**
2. **`GATE-ORBITDISP`'s bar.** `ρ* = 0.9772` is the measured maximum over 13 arms × 2
   datasets in the raw space — banked, label-free, frozen before the run. Round 2 should
   check the value reproduces and rule on whether the max (rather than, say, a quantile) is
   the right calibration statistic.
3. **Per-arm retraining stays excluded.** Round 1 ruled this correct *conditional on C-2
   being repaired by a gate* — noting that if it were not, the retrained reading would
   become the cheaper route to a defensible verdict. v2 repairs C-2 both by gates **and** by
   adopting an in-domain head lineage, so the condition is met twice over. The excluded
   variant remains priced at `195 × 40.39 + 195 × 34.40 = 14 584 s ≈ 4.05 h` of mints.
4. **L28 dropped**, per round 1's ruling on §14 issue 4.
5. **The one refinement, flagged for an explicit ruling: `GATE-ARMVIAB`'s two-case form**
   (§6.2). I did not adopt the one-sided majority-rate HALT verbatim, because it would fire
   on precisely the outcome the falsifier exists to detect and would leave C06 gated on an
   instrument that can never close it. The refinement is logical, not measured — the raw
   accuracies it would need are decision-relevant inputs I must not compute before freeze.
   **If round 2 disagrees, the one-sided form is a one-line change.**
6. **New, arising from v2's own construction:** Head-R has **no banked floor anchor**. Its
   instrument fidelity rests on sharing every component with Head-N except the training
   cache. Round 2 should rule whether that is sufficient, or whether Head-R needs an anchor
   of its own — and if so, what banked object could supply one.

---

*No GPU, SLURM, Modal, teacher call, model load, training of any deployed arm, cache write,
test-split access, job submission or commit occurred in producing this document. Login-node
dry-check processes only, itemised in §7 (≈ 5.5 wall-minutes, ≈ 30 CPU-minutes, `$0`).
`TARGET_STATE.json` was read and not modified. `refine-logs/C06_FALSIFIER_PREREG_DRAFT.md`
(v1) is unmodified.*
