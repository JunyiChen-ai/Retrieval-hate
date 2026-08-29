# C06 `$0` CPU falsifier — preregistration **DRAFT v5** (2026-08-04)

**SUPERSESSION.** Supersedes `C06_FALSIFIER_PREREG_DRAFT_V4.md` → V3 → V2 → v1. All four remain
on disk **unmodified** as the record of what each round reviewed. Reviews of record:
`C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3,_R4}.md` — REVISE 3C/6H/10I+4M, 3C/3H/7I+3M,
2C/1H/6I+4M, 3C/3H/8I+4M. Complete standalone document.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is frozen,
`TARGET_STATE.json` is untouched, nothing is committed.

**Disposition: all 18 round-4 findings ADOPTED, 0 rebutted.** Round 4's audit found 12 of 13
round-3 adoptions genuinely real and one (I-3) *textually present but arithmetically false*,
broken by the adoption of I-4 in the same round — that collision is its C-2 and is repaired here.
Cumulative table in **§14**.

**What v5 changes.** All three round-4 Criticals sit in the decision and accounting layers rather
than the instrument, and none needs a redesign. **C-1:** `GATE-ARMVIAB`'s escape branch required
the *raw* counterpart of a real arm to fail `majority + 0.02`, a bar C01 measured it clearing by
0.18–0.23 — so the gate degenerated into the one-sided HALT §6.2 exists to forbid, and it is now
**retired**, its function discharged by `GATE-ARENA`. **C-2:** S6 is **not** implied by S3 (S3
bounds the seed *mean*; S6 is pinned to the *per-seed* net), so v4's demotion of S6 to "reported,
not screening" deleted a conjunct that can fail — S6 is **binding** again. **C-3:** the head key
matrix is **per fold**, so two phases were short by ×5 and `GATE-ZEROOP`'s two guard arms were
counted nowhere; the corrected total is **`2927.5 s`**.

---

## 1. What this falsifier is, and what authorizes it

C06 (*Prompt-Orbit Tangent/Curvature*) is **not an active candidate**; its registry status is
`gated_on_zero_cost_falsifier`
(`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`). What the queue has reached
is **C06's falsifier**, not C06.

**The unblock condition, verbatim** (`…dispositions.gated[0].falsifier_spec`):

> re-run C01's real-displacement-versus-matched-norm-orthogonal-rotation battery in the FOLD-HEAD
> ARENA on the already-banked `ro_*` caches. Zero GPU, zero extraction, minutes of CPU on
> `scripts/analysis/headspace_{mint,arena}.py`, which exist and are banked. If the rotations again
> match the real displacement in the deployed head space, C06 closes for `$0` and the
> `1.7-2.5 GPU-h` of extraction is never queued; if they do not, C06 has earned its extraction

**The two binding design constraints, verbatim** (`…falsifier_design_constraints`):

> its pre-registration must (i) use the per-dataset adapter lineage that ACTUALLY EXISTS — HateMM
> has only `-LoRA-curric` ro-caches, MHC_zh has only `-LoRA`, one lineage each, not a matched pair
> (correction V-8); and (ii) declare the prompt/readout-span confound, because
> `generate_VideoMLLM_embedding_readout_HF.py:73-89` shows the `ow_` cells change the readout span
> as well as the prompt — the same confound C01's review already narrowed its claim for

Both honoured — §3.1 and §10.1 — and verified as *honoured, not merely mentioned* by all four
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
**This table is load-bearing twice over in v5** — it is the evidence the gate rests on, and it is
what refutes `GATE-ARMVIAB`'s escape branch (§6.2).

**Round-14's sharpening.** `orthogonal_blocks()` (`c01_policy_contrast_a0.py:1272`) is a **Givens
mixing of the two endpoint blocks**, so the six "random rotations" are six angles on **one
parameter family that also contains the primary** — `θ = 45°` **is** `common_displacement`,
`θ = 0` **is** `endpoint_concat`. Re-measured on the raw L24 features: `8.941e-08` (θ=0, both
datasets), `1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH) at θ=45. §10.2 now carries this as a scope
bullet (round-4 I-8), because it is the strongest narrowing available to a CLOSE.

**Why a re-run in a different space is the right instrument.** C01's arena is **raw dev keys**
(`n_dev` 107 / 78), not the fold-head path; `unified_pilot_gate.arena` requires *"the actual
fold-head/deployed-head path"* and F113 marks the raw-KILL direction **NOT ESTABLISHED**. All four
rounds ruled the arena reading correct.

---

## 2. The process rules that bind this design

| rule | discharged in |
|---|---|
| **R1** measured-unit-cost × explicit-count projection, multiplied through *"draws × **folds** × seeds × taus × spaces × datasets"* | **§8** — round 4 found the **folds** axis missing from two phases; corrected |
| **R2** line-buffered per-phase heartbeat | **§9** — with per-cell lines where the corrected counts create long spans |
| **R3** (F114) dry execution exercises the **first real operation of the payload path** | **§7** |
| **R4** (`feedback-separate-code-review-lineage`) a design GO does not review the implementation | **§13**, with round 4's six additions |
| **F118 erratum lesson** never let boilerplate describe a leg that did not run | **§3.7, §5.10, §8 Phase 5** |

---

## 3. The arena and the instrument

### 3.1 Inputs

C01's frozen scientific configuration (`configs/c01/c01_a0_v2.json`) pins
`standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`, `feature_dim = 3584`.

| dataset | adapter lineage (the only one banked) | `expected.train.n` |
|---|---|---|
| HateMM | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | 744 |
| MHC_zh | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | 579 |

**Not a matched pair, and never treated as one.** Every decision quantity is within-dataset,
within-seed, within-lineage; the two-dataset requirement is a **conjunction of independently
computed verdicts**. The four L24 files are byte-identical to the ones C01 measured (§11). **L28
is not used.** `train_*.pt` only for the ro caches; the native `dev_seen` is opened by
`headspace_mint.py:199` on every mint and is covered by `GATE-SHA`; no `dev_seen_*-ro_*` file is
opened by any phase; the `test_seen` ro caches are opened by nothing.

### 3.2 The head, the folds, and the vote

* **Fold contract.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the **full**
  train split. Discharged **two ways** (round-4 I-1): by `headspace_mint.py:203-216`'s assertion
  on every **executed** mint, **and** — because `:192-194` returns before `:203-216` on a resumed
  mint — by re-reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of` from **all 66**
  banked `.npz` before the arena runs. A `.npz` is written at `:321-325` only after the assertion
  passes, so the banked flag is a faithful record. This is exactly predictable on fresh, resumed
  and partially-resumed runs alike, and free.
* **Head.** The deployed-recipe RGCL head, re-minted on CPU (F78). Bank = fitting pool, queries =
  held-out fifth, every item held out exactly once.
* **Vote.** `mechfix_ops.deployed_vote(..., topk=20)` — `votes = ((lab*2−1) · sim · w).sum(1)/w.sum()`
  at `:94`, `w = [20…1]`. All four rounds verified this is C01's operator.
* **F88's caveat** — *"a CPU-trained arm must be paired against a CPU-TRAINED FLOOR"* — satisfied
  by construction.

### 3.3 Two head lineages, one driver

**Round 1's C-2**, measured: the head is trained on the native cache but forwarded over `ro_L24`
features **near-orthogonal** to it (median `cos(native_img, ro_L24_img)` = `0.0234` HateMM /
`0.0373` MHC-ZH, both caches unit-norm). v1's "banked C02 house pattern" claim stays **withdrawn**:
`c02_a0_mint.py:214` keeps `img_feats` native and `:68` refuses any view file carrying an image
stream.

| lineage | head trained on | banked anchor | in-domain | mints |
|---|---|---|---|---|
| **Head-N** | native deployed cache | **`GATE-FLOOR`** | no | 36 = 2 ds × 3 seeds × (5 folds + 1 full) |
| **Head-R** | `train_<model>-ro_L24.pt` | via the shared driver | **yes** | 30 = 2 ds × 3 seeds × 5 folds |

Head-R needs no `fold = −1` head: that head exists only to feed `GATE-DEVFID`, which compares
against banked **native** trainlogs.

**The shared driver.** **ONE driver, `scripts/analysis/c06_falsifier_mint.py`, serves both
lineages.** It imports `headspace_mint` with its sha256 asserted and reuses its dataset table,
deployed CLI, fold assignment, fold-parity assertion, dummy-dataloader construction, monkeypatches,
seeding and DET-1 contract unchanged. **`--train-cache` is its only lineage-varying argument.**
Because Head-N runs through that same driver and must reproduce the six banked `GATE-FLOOR`
anchors, **`GATE-FLOOR` anchors the driver, not merely Head-N's science**.

**The one behaviour that follows from "unchanged".** `headspace_mint.py:199` loads the native
`dev_seen` **unconditionally on every mint**, before the `fold` branch, and `:322` writes `lab_dev`
into every `.npz`; `model_name` comes from the frozen dataset table, so `--train-cache` does not
redirect it. **All 66 mints open the native `dev_seen`**; Head-R opens no `dev_seen_*-ro_*` file.
None reaches a decision quantity (§12).

**Pricing.** Both lineages are priced at `headspace_mint.py`'s own measured units
(`40.39 / 34.40 s`); the scratchpad Head-R harness (`37.46 / 27.54 s`) skipped the fold-parity
loads, the dev load and the `npz` save that the real driver performs.

### 3.4 The arm builder

One construction, parameterised by an ordered list of blocks, every normalisation being C01's
`l2_rows` called through the **imported** `c01_policy_contrast_a0`:

```
fuse(blocks) = l2_rows(concat[ l2_rows(b) for b in blocks ])
paired(A,B)  = fuse([ l2_rows(concat[ l2_rows(A_m), l2_rows(B_m) ]) for m in blocks ])
build_views(std_blocks, ow_blocks, angles) -> the 13 arms
```

Two blocks `[img, text]` ⇒ C01's `prepare_views`, **bit-exactly** (`max|diff| = 0.000e+00`, 13
arms, both datasets) — reproduced independently from this prose by rounds **2, 3 and 4**. One
block (the fused head key) ⇒ the head-space arms, whose dimensions are **`4 × 1024-d` and
`9 × 2048-d`** (round-4 measurement γ, the first independent confirmation of §8 Phase 2's
`240 / 540` split).

**What the anchor buys.** Two-block parity pins `l2_rows`, concatenation order, the contrast
definitions, the Givens mixing and its angle convention, the arm-name→formula map, the `float32`
dtype and the θ=0/θ=45 identities. It does **not** pin the outer `fuse` normalisation at one
block, which re-normalises an already-unit vector: **what round 1's C-3 actually fixed for the
head-space arms is the dtype.** The block count is **forced by the head's architecture** —
`classifier_hateClipper` emits a single fused 1024-d vector (`classifier.py:116-120`).

**Deferred-import note.** `c01_policy_contrast_a0.py:387` sets `np = torch = faiss = None`; the
battery must call `import_compute_modules(config)` before touching the algebra.

### 3.5 The arms

**Fourteen arms**: thirteen key-space arms plus `avg_score`, at `ro_L24`.

| arm | role |
|---|---|
| `endpoint_std`, `endpoint_ow` | reference endpoints; C01 `gain_controls` |
| `avg_score` | derived score control (mean of the two endpoint vote scores); **a C01 `gain_control`, hence a comparator in `C` for both real arms and a member of the Holm family** |
| `endpoint_concat` (≡ `orthrot_0`), `common` | ordinary controls |
| **`displacement`** | the real first-order prompt tangent — **real arm** |
| **`common_displacement`** (≡ `orthrot_45`) | C01's **primary** — **real arm** |
| `common_interaction` | C01's secondary |
| `orthrot_{8.3, 17.6, 29.1, 60.4, 72.7, 83.8}` | the matched-block-L2 rotation family |

**Plus two guard arms, built and voted but scoring nothing** (round-4 C-3(b)): `orthrot_0` and
`orthrot_45`, constructed by the **rotation** route and compared against `endpoint_concat` and
`common_displacement` by `GATE-ZEROOP`. They are outside the fourteen, are never aliased to their
counterparts, and are now counted in §8 (Phase 2z).

### 3.6 The raw leg — gate-only, non-decisional, on the same rows

The same builder runs on the raw L24 features (no head) and votes in the **same folds**, on **the
same arena population as the head leg**. It renders no verdict and enters no decision rule or
multiplicity family. Its remaining job is `ρ_raw` for `GATE-ORBITDISP` (§6.1) and the reported
raw-vs-head `endpoint_std` comparison of §6.4. `GATE-POP` asserts the head leg and the raw leg run
on identical row **index sets**.

### 3.7 The null contract, the mask convention, and the population-derived constants

**The defect.** `classifier_hateClipper.__init__` builds the projections **with the default bias**
at **`src/model/classifier.py:81-82`** (round-4 M-1; `:80` is the comment line), so `head_f(0,0)`
is a non-zero constant. HateMM train row 355 (`hate_video_95`, C01's registered `authorized_null`)
is bit-identically zero in both modalities of **both** ro caches, so `h_std[355] == h_ow[355]`
**exactly**, and it is the **only** such row on either dataset. Hence in head space every
endpoint / common / rotation block is non-zero at row 355 while the displacement block is exactly
zero — and `l2_rows:1193-1194` is fail-closed on precisely this. Measured, each mask choice kills
the other, and **`common_displacement`, C01's primary, dies under both**.

**Why masking is also wrong on the science.** A zero key has inner product `0` with every query,
which under `IndexFlatIP` ranks **above any negatively-similar candidate**; `deployed_vote`
(`mechfix_ops.py:94`) then weights `(2y−1) × 0 = 0` into the sum. So a masked null would
**displace a real neighbour out of the top-20 while contributing nothing** — corrupting eleven
control arms and one real arm differently, along the very comparison that renders the verdict.
Round 4 confirmed this at the operator (measurement δ).

**The contract. Four objects, each with its population and its mask argument.**

| object | population | `zero_mask` argument | why |
|---|---|---|---|
| head training | full `n` (744 / 579) | — (no C01 algebra) | the deployed recipe and fold contract are unchanged |
| `GATE-FLOOR` | full `n`, native keys | — | the six banked floors were computed on this population |
| `GATE-C01PARITY` two-block build | `n = 744` (HateMM) / `579` (ZH) | **one-hot `{355}`** / **`np.zeros(579, bool)`** | the only population where C01's masked contract is defined |
| `GATE-ROWSUBSET` two-block build | `n = 743`, **HateMM only** (round-4 M-4) | `np.zeros(743, bool)` | the bridge between the two populations |
| **arm arena — head leg AND raw leg** | **`n = 743`** (HateMM) / `579` (ZH) | **`np.zeros(n, bool)`** | the null is physically absent, so the all-False mask is *correct*, not a workaround |

**The mask convention, binding everywhere.** The `zero_mask` argument is **always an explicit
boolean array** — never `None`. `l2_rows:1187-1188` **normalises** `None` into a zeros array, so
`None` is admissible there; `prepare_views:1381-1386` compares its derived masks against the **raw
argument** (`np.array_equal(<bool array>, None)` is `False`), so `None` is **inadmissible** and the
call dies *on any dataset*. **Round 4 added the third cell v4 did not report: `None` dies at the
HateMM arena `n = 743` too**, where no null row exists — confirming the inadmissibility is a
property of the function, not of the data. C01 itself never passes `None` (`:2224`, `:2304-2306`).

**Why removal is legitimate** (all four counts tested and ruled on by rounds 3 and 4): the row is
selected by an exact-zero **feature** property and is C01's pre-existing frozen `authorized_null`
(label-free, and the only such row); it is dropped from the bank **and** query set of **every arm
identically**, so no lane can gain; the change is **provably a pure row-subset**
(`max|diff| = 0.000e+00` on all 13 arms, every `ρ` unchanged); and it turns on the *presence* of a
bias term, fixed before any trained-head number exists.

**Population-derived constants — the full list.** Round 3 believed the majority rate was the only
one; round 4 found two more. All are frozen here, all computed on the **arena** population, and
all are checked at run time rather than read (§13 item 5):

| constant | HateMM | MHC-ZH | consumed by |
|---|---|---|---|
| arena size `n_D` | **743** | **579** | `GATE-SELFTEST`, S6, the tie cap |
| arena class counts | **(297 pos, 446 neg)** | **(180, 399)** | `GATE-POP` |
| **arena majority** | **`446/743 = 0.600269 → 0.6003`** | **`399/579 = 0.689119 → 0.6891`** | `GATE-ARENA`'s lower bound |
| `GATE-ARENA` band | **`[0.6203, 0.98]`** | **`[0.7091, 0.98]`** | `GATE-ARENA` |
| **small-displacement quantile** (round-4 I-3) | `0.1` quantile of the **743** displacement norms | of the **579** | **S7** (§5.2) |
| **`GATE-DOMAIN` majorities** (round-4 I-2) | `maj_arena = 0.6003` with `acc_ro`; `maj_full = 0.5995` with the banked `acc_native` | `0.6891` for both (arena = full) | `GATE-DOMAIN` |
| tie cap | `⌊0.01 × 743⌋ = 7` | `⌊0.01 × 579⌋ = 5` | `GATE-ZEROOP` |

The full-population majority `0.5995` is used **only** as the `GATE-DOMAIN` denominator's
companion, where it is correct because `acc_native` is itself a full-population banked figure.

**The dataset asymmetry.** MHC-ZH has no exact-zero row, so its arena is `n = 579` unchanged and
every constant above coincides with its full-population value. Round 3 ruled the asymmetry
**contained** by the conjunction-of-independent-verdicts structure, on the condition that
population fixes be applied per dataset — which the table does.

**There is no head-space null-row sensitivity leg.** The alternative head-space population is
unbuildable, so a leg comparing the two would have nothing to compare. The sensitivity question is
discharged where both populations are defined — the raw two-block anchor — by `GATE-ROWSUBSET`.

---

## 4. Ambiguities in the written condition

**"Conservative" means *hardest for the falsifier to deliver the `$0` CLOSURE***. Round 1 ruled
this correct on two conditions — disclose what the lean buys (§5.9), and never let it excuse an
arithmetic error. Rounds 2, 3 and 4 each held the design to the second condition (the lineage
disjunction; the majority constant; **S6's demotion**), and all three are discharged.

| # | ambiguity | resolution |
|---|---|---|
| **A1** | *"the rotations"* | the real arm must beat **every** rotation |
| **A2** | *"the real displacement"* | **both** real arms, disjunctively, multiplicity-corrected |
| **A3** | *"match"* | a tie **closes** C06 |
| **A4** | *"in the fold-head arena"* | arena reading ruled correct by all four rounds; both lineages run; the row set is §3.7 |
| **A5** | which layer? | **L24 only** |
| **A6** | seeds | seed-mean primary, plus **3/3** per-seed agreement on the rotation-dominance leg and on S6 |
| **A7** | *"C06 closes"* | the **first-order (tangent/chord) leg only**; round 1 ruled A7 is **not** an obstacle |

---

## 5. The pre-registered decision rule

### 5.1 Notation and population

For dataset `D`, lineage `L ∈ {Head-N, Head-R}` and seed `s ∈ {0,1,2}`, each of the **fourteen**
arms yields one OOF prediction vector over the **arm-arena population** — `n_D` = `743 / 579` —
scored against **train-split** labels held out from the head that judged them. `acc(A,D,L)` is the
**mean over the three seeds**; `net_s(A)` is the **per-seed integer** net-fix count against
`endpoint_std`.

* **Real arms** `R = {displacement, common_displacement}`.
* **Rotation family** `Θ` = the six frozen angles.
* **Ordinary controls**, per arm, from C01's own two frozen lists: for `common_displacement`,
  `C = gain_controls ∪ {displacement}` (**six**); for `displacement`, `C = gain_controls`
  (**five**). `avg_score ∈ gain_controls`, so it is a comparator for both.

### 5.2 SURVIVE

**On a given lineage `L` that passed its per-lineage instrument gates (§5.6), C06 SURVIVES iff
there exists `A ∈ R` such that S1–S7 all hold on BOTH datasets:**

| | condition | frozen source |
|---|---|---|
| **S1** | `acc(A) > max_θ acc(orthrot_θ)` **and** `mF1(A) > max_θ mF1(orthrot_θ)` | `require_primary_above_all_rotation_controls` |
| **S2** | S1's accuracy leg holds in **3/3** seeds | A6 |
| **S3** | `acc(A) − max_{c∈C} acc(c) ≥ 0.02`, likewise `mF1` | `minimum_gain_over_strongest_control = 0.02` |
| **S4** | for every comparator in `C ∪ Θ`: bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05`, with the statistic pre-registered at **§5.4** | `minimum_bootstrap_lower_bound`, `require_primary_bootstrap_holm_reject`, `require_rotation_bootstrap_holm_reject`, `n_bootstrap = 2000`, `statistics.seed = 20260728`, `holm_alpha = 0.05` |
| **S5** | **both** real arms exceed the 95th percentile of their shuffled-pair null, **and** the shuffle comparison Holm-rejects | `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject` |
| **S6** | **`net_s(A) ≥ 3` (HateMM) / `≥ 2` (MHC-ZH) in 3/3 seeds** | `minimum_net_fixes`; reference `retrieval.fix_break_reference = endpoint_std` |
| **S7** | **no small-displacement dominance** for `common_displacement`, at C01's frozen `small_displacement_train_quantile = 0.1` computed on the **arena** displacement norms | `require_no_small_displacement_dominance` |

**S6 is BINDING (round-4 C-2).** v4 demoted it to *"reported, not screening"* on the strength of a
claimed implication `S3 ⇒ S6`. **That implication is false**, and the demotion deleted a conjunct
that can fail — the anti-conservative direction. See §5.9 item 4 for the corrected statement and
the counterexample.

**S7 is a SURVIVE condition, not a HALT gate (round-4 H-1).** C01 places
`require_no_small_displacement_dominance` in `decision`, and its
`required_halt_only_validity_guards` (seven entries) does **not** contain it. v4 listed it
unannotated in §6's gate table, where a real arm whose few fixes happened to concentrate in the
bottom displacement decile would have produced `INSTRUMENT_INCONCLUSIVE` instead of the CLOSE that
S1's failure warrants — the same class as C-1, in a gate no earlier round examined. Two further
pre-registrations it needs, neither inherited because the battery does not import
`displacement_audit`: its **arm scope is `common_displacement` only**, matching C01's hard-wired
`evaluations["common_displacement"]`; and the **zero-fix convention** is
`fixed_fraction := 0`, `dominated := false` when `fixed == 0`, mirroring
`c01_policy_contrast_a0.py:1989-1996`, so a CLOSE with zero fixes cannot divide by zero.

### 5.3 CLOSE

**C06 CLOSES iff the run publishes a verdict (§5.6) and SURVIVE is false on BOTH lineages, both
of which passed their per-lineage instrument gates.**

### 5.4 The bootstrap statistic — pre-registered (round-4 H-2)

v4 cited four C01 constants but never said **what `p` is**, and §5.4's seed-averaging makes C01's
own `paired_bootstrap` (`:1742-1772`, no seed axis, evaluates `metric_value` on sampled scores)
non-reusable. Pre-registered here:

* **Resample:** items once, `B = 2000` draws, C01's frozen `statistics.seed = 20260728`, the same
  draw indices shared across all comparators within a `(dataset, lineage)`.
* **Per-resample statistic:** `Δ_b = mean_{i ∈ draw_b}[ c̄_A(i) ] − mean_{i ∈ draw_b}[ c̄_c(i) ]`,
  where `c̄_X(i)` is the **mean over the three seeds** of item `i`'s 0/1 correctness under arm `X`
  — so the seed axis is inside the statistic, not a hidden multiplicity.
* **Lower bound:** the `5 %` quantile of `{Δ_b}` (C01's `bootstrap_lower_quantile`), required `> 0`.
* **One-sided p:** `p = (1 + #{b : Δ_b ≤ 0}) / (B + 1)`, C01's own form at `:1769`.
* **Holm:** the step-down of `c01_policy_contrast_a0.py:1775-1784` over the family of §5.5.

### 5.5 Multiplicity, and its resolution floor

**One Holm family per dataset spanning both lineages**, `α = 0.05`: `common_displacement`
6 comparators + 6 rotations = 12; `displacement` 5 + 6 = 11; `(12 + 11) × 2 metrics = 46` per
`(dataset, lineage)`; **× 2 lineages = 92 hypotheses per dataset**. The two **datasets** remain a
conjunction. S5's shuffle rejections are correctly outside the family: they are conjunctive within
each disjunct, so `P(∃ disjunct : all conditions) ≤ P(∃ disjunct : its bootstrap legs all reject)
≤ α`.

**The `B = 2000` resolution consequence, stated before freeze (round-4 H-2).** The smallest
achievable p is `1/2001 = 0.00049975`, against `α/92 = 0.00054348` — it clears Holm at rank 1. The
next achievable level, `2/2001 = 0.00099950`, first clears at **rank 43**. **Therefore at least
42 of the 92 comparators must show zero adverse resamples out of 2000 for S4 to pass.** That is
feasible but demanding, it is a property of the frozen `B` and family size rather than of the
data, and it is recorded here so a later round cannot discover it as a surprise.

**The two `92`s are different products** (round-3 M-4): §5.5's is `23 comparisons × 2 metrics ×
2 lineages` per dataset; §8 Phase 4's is `23 × 2 datasets × 2 lineages` comparison-cells with both
metrics inside `U3`. They must not be reconciled.

### 5.6 Verdict combination, instrument failure, and per-lineage gate scoping

**Round-4 H-3, adopted in its stronger form.** v4 combined lineages **disjunctively for SURVIVE**
but **conjunctively for HALT**, so an instrument failure confined to Head-N — the lineage the
design itself marks *"in-domain: no"* — would void a verdict Head-R could have delivered cleanly.
v5 scopes the instrument gates:

* **Global gates** (`GATE-DET1`, `GATE-SHA`, `GATE-FOLD`, `GATE-FLOOR`, `GATE-POP`,
  `GATE-C01PARITY`, `GATE-ROWSUBSET`, `GATE-NULLREMOVED`, `GATE-IDPARITY`, `GATE-ZEROMASK`,
  `GATE-LEDGER`) govern provenance, population, algebra and bookkeeping shared by both lineages —
  and `GATE-FLOOR` anchors the shared driver. **Any failure HALTs the whole battery.**
* **Per-lineage gates** (`GATE-ARENA`, `GATE-ORBITDISP`, `GATE-NESTED`, `GATE-SELFTEST`,
  `GATE-ZEROOP`, `GATE-ALGEBRA`) are evaluated within a lineage. **A lineage that fails one is
  marked `INSTRUMENT_FAILED` and is dropped**, not the battery.

**The combination rule, in the conservative direction:**

1. **SURVIVE** if any lineage that **passed** its per-lineage gates clears S1–S7 on both datasets.
2. **CLOSE** if **both** lineages passed their per-lineage gates and neither clears.
3. **HALT** (`INSTRUMENT_INCONCLUSIVE`) otherwise — i.e. whenever a global gate fails, or a
   lineage is dropped and no surviving lineage clears.

This is strictly better than v4 in both directions: a clean Head-R SURVIVE is no longer voided by
the transplant lineage's failure, and a CLOSE still requires **two** clean negatives, never one.
The lineage(s) that ran must be named in §10.2's scope sentence.

**Finiteness, absence, and crashes.**

* Every **gate and decision** quantity is asserted **finite and present** before comparison
  (round-4's extension of v4's gate-only clause), and every gate is written in **pass-condition**
  form. The concrete instance the extension covers is S7's `fixed_fraction` at zero fixes, now
  pre-registered in §5.2.
* An **absent** decision or gate quantity HALTs on the same footing as a non-finite one.
  `GATE-LEDGER`'s process count is binding (§12).
* `l2_rows` and `prepare_views` signal by `die()` → `RuntimeError` (`:392-393`), a **crash, not a
  gate result**. **Every call into the imported C01 algebra is wrapped**; a `RuntimeError` is
  recorded as `INSTRUMENT_INCONCLUSIVE` with `l2_rows`' `context` string — which carries the arm
  and block name — written to **both** the decision JSON and the **final heartbeat line**.

### 5.7 Pre-declared expectation

**CLOSE is expected**, on two grounds: C01 measured the premise rotation-indistinguishable at the
two-point case on both datasets, and the recon's structural objection (a fixed prompt injects no
per-item information) is unrebutted. v1's untrained-head contraction ground stays **withdrawn**.

**A recognised third outcome (round-4 H-3).** A Head-N-only instrument failure is a live path:
Head-N is an out-of-domain transplant by construction, `GATE-ARENA` asks its `endpoint_std` to
clear `majority + 0.02` — a recovery fraction of `0.02/(0.8884 − 0.6003) ≈ 6.9 %` — and §6.4
explicitly refuses to invent a bar for that quantity. Under §5.6 this now **drops Head-N** rather
than voiding the battery; if Head-R then clears S1–S7, C06 survives on Head-R alone, and if it
does not, the run HALTs rather than closing on one lineage. Either way the outcome is named in
advance and in §10.2.

### 5.8 What a CLOSE cannot be attributed to

Recorded here so §10.2 is not the only place: no CLOSE may be attributed to the prompt alone
(§10.1's confound), to curvature (two points give a chord), to a per-arm-retrained head, to a
per-modality contrast (§10.2), or to directions off the primary's own Givens family (§10.2's sixth
bullet).

### 5.9 Disclosure

1. `require_accuracy_gain_over_deployed_r0_context` is **not carried**: its comparator is a raw
   dev-arena figure at `n_dev` 107/78 from `historical_strict_devtrain`, and importing it would
   breach F88's CPU-arm/CPU-floor caveat. **Inapplicable across arenas, not waived** — rounds 2, 3
   and 4 all verified the comparator and endorsed the reasoning.
2. `displacement`'s comparator set omits `common_displacement` while `common_displacement` must
   beat `displacement`. C01 froze a comparator list only for its primary, so this is not a
   violation, but the asymmetry **eases SURVIVE for one of the two disjuncts**.
3. The two-real-arm disjunction is deliberately generous; its multiplicity is corrected (§5.5).
4. **S6 binds, and v4's vacuity claim was false (round-4 C-2).** S3 is defined on the **seed
   mean** and therefore bounds only `mean_s net_s ≥ 0.02 × 743 = 14.86` (HateMM) /
   `0.02 × 579 = 11.58` (MHC-ZH). It says **nothing** about `min_s net_s`. Counterexample,
   verified: `net = (2, 21, 22)` has mean `15.00 ≥ 14.86`, satisfying S3, while `net_0 = 2 < 3`
   fails S6. The required spread is `20` net items on 743 = **`2.69` accuracy points**, inside
   this campaign's ordinary seed noise rather than an exotic construction. **S6 therefore binds —
   through across-seed dispersion — and is a real tightening.** What remains true from v4's
   observation is the scale transfer: C01 froze `minimum_net_fixes` on an arena of `n_dev` 107/78
   where `+3` is `+2.8 %` accuracy, and at `n = 743` the same integer is `+0.40 %`, so S6's
   *mean-level* content is far below S3's. Its *dispersion-level* content is not.
5. **`GATE-ZEROOP`'s tie cap is a declared engineering choice**, fixed before the run. Round 4
   ruled it defensible and asked for one property to be explicit: **the cap is one-directional —
   it can only convert REPORT → HALT, never HALT → REPORT** — so it cannot cause a wrong verdict
   in either direction; it can only cause the falsifier not to publish.

### 5.10 What this design does not run

**No head-space null-row sensitivity leg**, **no L28 leg**, **no per-arm retrained head**, **no
vote at `n = 744`** (the sole full-`n` vote is `GATE-FLOOR`'s 30 native deployed-key votes, which
score no arm), and **no test-split read of any kind**. No gate, output field or record sentence
describes any of them.

---

## 6. Gates

**Eighteen gates.** `GATE-ARMVIAB` is **retired** (§6.2) and `GATE-SMALLDISP` has become **S7**
(§5.2). Every quantity is asserted finite and present before comparison. Scope column: **G** =
global (HALT the battery), **L** = per-lineage (drop that lineage), **R** = reporting only.

| gate | scope | asserts |
|---|---|---|
| `GATE-DET1` | G | thread env exported before any python starts |
| `GATE-SHA` | G | every frozen import and input cache matches §11; **once in the sbatch driver** |
| `GATE-FOLD` | G | fold parity vs the banked `vsw_ckpt`, discharged **both** on executed mints (`headspace_mint.py:203-216`) **and** by re-reading `meta["fold_parity_vs_banked_vsw_ckpt"]` + `fold_of` from all 66 banked `.npz` — resume-safe (§3.2) |
| `GATE-FLOOR` | G | **Head-N through the shared driver**, native keys, **full `n`**, reproduces the banked floors at 4 dp on **both** metrics — acc HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946`; mF1 HateMM `0.8838/0.8811/0.8812`, ZH `0.8747/0.8710/0.8765`; every `fold_acc_deployed` entry. **Anchors the driver for both lineages**, hence global |
| `GATE-POP` | G | realised populations equal §3.7's table; head leg and raw leg on **identical row index sets**; realised arena class counts equal **`(297,446)` / `(180,399)`**; **and every population-derived constant in §3.7's table is recomputed from the arena, not read** |
| `GATE-C01PARITY` | G | the two-block builder reproduces `prepare_views` **bit-exactly** at `n = 744` one-hot `{355}` (HateMM) and `n = 579` all-False (ZH); HALT above C01's `2e-6` |
| `GATE-ROWSUBSET` | G | **HateMM only** — the `n = 743` all-False build is **bit-identical** to the `n = 744` one-hot build restricted to the 743 surviving rows, all 13 arms. **Strictly stronger than C01's `displacement_registered_null_exclusion` at the key level; it is not C01's property**, which is vote-level and **has no object here** |
| `GATE-NULLREMOVED` | G | no arena population contains an exact-zero row; removed set `{355}` / `{}` |
| `GATE-IDPARITY` | G | every ro cache's `ids` order and `labels` identical to the native bank |
| `GATE-ZEROMASK` | G | **feature space only** — measured exact-zero row set equals `{355}` / `{}` on both policies |
| `GATE-LEDGER` | G | C09's full declared-count predicate set, process count **binding** (§12) |
| `GATE-ORBITDISP` | **L** | **per fold, all 60 head cells** (round-4 I-6): HALT iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D` for any arm in any fold; `ρ*` per dataset at full precision (§6.1); `ρ` over **arena rows only**; `ρ_raw` reproduces §6.1's frozen values at 4 dp |
| `GATE-ARENA` | **L** | **lower** bound `arena majority + 0.02 ≤ acc` on **`endpoint_std` only**; **upper** bound `acc ≤ 0.98` on `endpoint_std` **and** both real arms. Bands `[0.6203, 0.98]` / `[0.7091, 0.98]` (§6.3) |
| `GATE-NESTED` | **L** | **per item**, the head that scored it excluded its fold; check count equals the item count |
| `GATE-SELFTEST` | **L** | `net(A) = n_D · (acc(A) − acc(endpoint_std))` exactly for **every one of the 14 arms** (round-4 I-7, including `avg_score`), every seed, dataset and lineage, with `n_D` pinned to §3.7 |
| `GATE-ZEROOP` | **L** | `orthrot_0` vs `endpoint_concat` and `orthrot_45` vs `common_displacement` produce identical predictions — with the tie diagnostic of §6.5 |
| `GATE-ALGEBRA` | **L** | key-level `≤ 2e-6` on both identities. **Logically independent of `GATE-ZEROOP` in both directions** |
| `GATE-DOMAIN` | **R** | the recovery fraction of §6.4, on the verdict face; no bar |
| `GATE-DEVFID` | **R** | `headspace_fidelity.py` on Head-N's 6 full heads |

### 6.1 `GATE-ORBITDISP`

Every arm is `l2`-normalised before `deployed_vote` (and again by `_norm32`), so the retrieval key
keeps only **direction**; a near-constant offset orbit would make every `displacement` key nearly
the same vector while a magnitude gate saw nothing. The quantity is `ρ = ‖mean_i k_i‖` over unit
keys.

**The max is the right order statistic** — this is an *instrument* gate, and its job is to fire
only when the head space is more degenerate than anything the raw feature family produces; a
quantile bar would convert ordinary head-induced concentration into `INSTRUMENT_INCONCLUSIVE`.
**`ρ*` is frozen per dataset at full measured precision**, which removes the `endpoint_std`
self-exemption a 4-dp truncation created:

| dataset | **`ρ*`** | supplying arm | runner-up |
|---|---|---|---|
| HateMM | **`0.968176`** | `endpoint_std` | `0.964446` (`common`) |
| MHC-ZH | **`0.977223`** | `endpoint_std` | `0.969686` (`common`) |

**Cell granularity, pre-registered (round-4 I-6):** `ρ_head` is computed **per fold** — there are
`2 ds × 3 seeds × 5 folds × 2 lineages = 60` head key matrices and therefore `60 × 13 = 780`
values — and the gate **HALTs its lineage if any cell fires**. A degenerate head space in any
single fold destroys that fold's OOF predictions, which enter every decision quantity, so
per-fold is both the conservative and the structurally correct choice. §8 Phase 2D is counted at
the same granularity.

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

`ρ` must be computed over the arena rows: including the masked zero row shifts values by up to
`1.301e-03`, which would fail the 4-dp reproduction leg (§13 item 7).

**The bar is defended by measurement.** `ρ` on the **36 banked trained deployed-head key
matrices** (`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz::K_train`, 18 cells per
dataset): HateMM min/median/max `0.447803 / 0.562434 / 0.632996`; MHC-ZH
`0.340179 / 0.574247 / 0.667326`; **0/18 above `ρ*` on both**. A trained deployed head sits at
roughly **half** the bar. Reproduced to the digit by rounds 3 and 4. Label-free, computes no
accuracy.

### 6.2 `GATE-ARMVIAB` is retired — round-4 C-1

v4's `GATE-ARMVIAB` escaped a one-sided HALT only through case 1: *head-space arm fails
`majority + 0.02` **and the raw counterpart also fails** ⇒ no HALT*. **That branch is
unreachable.** §1's table records C01's measured raw `displacement` at `0.8505` / `0.8846` and
`common_displacement` at `0.8598` / `0.8590`, against arena bars of `0.6203` / `0.7091` — clearing
by `0.15`–`0.23`. The raw leg here is the same features and the same operator on a **larger**
arena, and this campaign's OOF train arenas run *higher* than its dev arenas, not lower
(`GATE-FLOOR`'s own native OOF accuracies are `0.8884 / 0.8929` against C01's dev-arena
`0.8411 / 0.8590`). So on the real arms the gate reduced to *head-space real arm fails
`majority + 0.02` ⇒ **HALT*** — precisely the one-sided gate §6.2 of v4 opened by rejecting, firing
on the outcome that opening sentence names as **the warranted CLOSE**.

**The discriminator was wrong in kind.** The question is *"is the head space alive?"*, and that is
answered by the **controls in the same space**, never by the same arm in a different space. A real
arm that collapses in head space while the head-space controls stay healthy is not an instrument
failure — **it is the strongest possible negative for C06**, and the design must be able to
publish it.

**Retirement, not restriction.** Round 4 offered restricting the gate to `endpoint_std`. Followed
through, that makes it *strictly redundant*: `GATE-ARENA`'s lower bound HALTs whenever
`acc_head(endpoint_std) < majority + 0.02`, while a two-case version would HALT only on that
condition **and** the raw arm clearing — a subset. Keeping a gate that can never fire when its
neighbour does not is the kind of decorative instrument this lineage has removed elsewhere, so
`GATE-ARMVIAB` is **deleted**. Its one informative residue — the raw-vs-head `endpoint_std`
comparison — survives as a **reported** diagnostic beside `GATE-DOMAIN` (§6.4).

**What still watches the real arms:** `GATE-ARENA`'s `≤ 0.98` upper bound (a leak catcher, which
cannot fire downward), `GATE-ORBITDISP` (direction degeneracy, all 13 arms, all 60 folds), and
`GATE-C01PARITY` / `GATE-ALGEBRA` / `GATE-ZEROOP` (algebra). **No lower-bound instrument HALT is
applied to a real arm anywhere in this design**, which makes §6.3's invariant literally true for
the first time.

### 6.3 `GATE-ARENA`

**Lower** bound restricted to `endpoint_std` — C09's own scope
(`C09_A0_V17_RECORD.md:1569-1572`, *"pooled native accuracy"*, the floor arm, never a treatment
arm). **Upper** bound `≤ 0.98` on `endpoint_std` and both real arms, where it catches a leak and
cannot fire on a warranted CLOSE. Bars are the **arena** majority: `[0.6203, 0.98]` (HateMM) /
`[0.7091, 0.98]` (MHC-ZH).

**"Real arms lose badly" is a reportable scientific outcome — the falsifier working — and is never
an instrument HALT.** With `GATE-ARMVIAB` retired, nothing in §6 contradicts this sentence.

### 6.4 `GATE-DOMAIN` — reporting, with both populations named

Round 2 ruled that round 1's C-2(a) is substantively discharged by `GATE-ARENA`'s lower bound on
`endpoint_std`, and that refusing to invent a recovery-fraction bar is correct. `GATE-DOMAIN`
therefore reports, with no threshold:

**recovery fraction = `(acc_ro − maj_arena) / (acc_native − maj_full)`** for `endpoint_std` under
Head-N, where **`acc_ro`** is a head-space accuracy on the **arena** (`maj_arena = 0.6003 /
0.6891`) and **`acc_native`** is `GATE-FLOOR`'s banked `acc_deployed`, measured on the **full**
population (`maj_full = 0.5995 / 0.6891`). Round-4 I-2: v4 wrote one `maj` for both terms, which
cannot be right for both; each term now carries its own population's majority, and the choice is
in §3.7's constant table.

Also reported here, inheriting `GATE-ARMVIAB`'s residue: the **raw-vs-head `endpoint_std`
comparison** on the arena rows. Both figures appear on the verdict face and in §10.2's scope
sentence.

### 6.5 `GATE-ZEROOP` and `GATE-ALGEBRA`

The two are **logically independent in both directions**; the value is their conjunction. The θ=45
identity is not exact, so a key perturbation can reorder a top-20 neighbourhood and `GATE-ZEROOP`
carries a real false-HALT probability on a correct run.

**The tie diagnostic, with round-4 I-4's unit correction.** v4 compared a **key-component**
residual against a **similarity** gap. `GATE-ALGEBRA`'s residual is a per-component max-abs on the
key matrices (`:1372-1377`); a key perturbation `Δk` with `max|Δk| = ε` changes an inner product
against a unit query by up to `‖Δk‖₂ ≤ √d · ε`, which at `d = 2048` and `ε = 1.192e-07` is
`5.394e-06` — **45× the threshold v4 applied**, leaving the tie set 45× too narrow.

* **Ranking:** the **union** of the two arms' top-21 sets.
* **Residual:** `‖Δk‖₂` measured directly on the head-space key difference, or its bound
  `√d · max|Δk|`; the **maximum** over the two compared identities. **It is the head-space
  residual, measured at run time on the same lineage whose predictions are being compared** — not
  the raw-key figures v4 quoted.
* **Criterion:** an item is a **tie casualty** iff recomputing its rank-weighted vote leaves the
  two arms' predictions equal under the **worst case over all orderings** of every near-tie group
  (this is what "collapsing" means; v4 left it undefined, and different readings give different
  tie sets).
* **Cap:** a mismatch on more than `⌊0.01 × n_D⌋` items (`7` HateMM, `5` MHC-ZH) HALTs regardless.
  The cap is **one-directional** — it can only convert REPORT → HALT (§5.9 item 5).
* A mismatch confined to tie casualties, under the cap, is **REPORTED, not HALTed**; any mismatch
  outside them HALTs.

---

## 7. Dry-check — what was executed, and what it found

Login node `foscsmlprd01`, conda `HateVideo`, 8 threads, DET-1 exported, outputs only in the
session scratchpad. Zero GPU, zero SLURM, zero test-split file opened, zero write into `data/`,
`artifacts/` or `logging/`.

### 7.1 Real inputs and the population constants

All 8 ro caches and 4 native caches loaded through the real `torch.load` path; `ids`
order-identical and `labels` identical to the native bank on every file (round-4 measurement β
confirmed `GATE-IDPARITY`'s property directly). Exact-zero rows: HateMM `{355}` in both modalities
of all four ro caches and the native cache; MHC-ZH none. Row 355 is `hate_video_95`, **label 1**,
held out in **fold 4**.

| dataset | population | n | pos | neg | majority |
|---|---|---|---|---|---|
| HateMM | full | 744 | 298 | 446 | `0.599462 → 0.5995` |
| **HateMM** | **arena** | **743** | **297** | **446** | **`0.600269 → 0.6003`** |
| MHC-ZH | full = arena | 579 | 180 | 399 | `0.689119 → 0.6891` |

### 7.2 Mint units — full-process wall

Every mint figure is **full-process wall**, measured around the `python …` invocation — the first
by `/usr/bin/time -v` (`Elapsed 0:40.39`), the rest by `date +%s.%N` brackets. Interpreter and
import cost is **already inside every unit**: the same run's internal timer reads `33.0 s`, a
`7.4 s` gap, and measured startup alone is `3.05–3.18 s`. **No Phase 1e line is added, because
adding one would double-count** — confirmed correct by rounds 3 and 4.

| lineage | unit | dataset | measured wall |
|---|---|---|---|
| Head-N | fitting-pool head | HateMM / MHC-ZH | **40.39 / 34.40 s** |
| Head-N | full-train head | HateMM / MHC-ZH | **49.30 / 38.87 s** |
| Head-R | scratchpad harness | HateMM / MHC-ZH | 37.46 / 27.54 s — **not used**, §3.3 |

Fold parity passed in all seven mints; peak RSS **1.25 GiB**.

### 7.3 Blindness

Every head used in any arm-building or voting dry check is **untrained**, so every operation is
real at real scale while the numbers are scientifically void. **No arm accuracy has been computed,
printed or recorded at any point in v1–v5.** Round 4 audited this by grepping every decimal in
`[0.6, 0.99]` across the draft and confirmed each is a `ρ`, a banked `GATE-FLOOR` anchor, a
published C01 dev-arena figure, or a majority/band constant.

### 7.4 The null-contract and mask-convention measurements

| # | measurement | result |
|---|---|---|
| (a) | is `head_f(0,0)` zero? | **No** — non-zero at torch seeds 0/1/2. **The magnitude is emitter- and initialisation-dependent** (round-4 M-3): `0.58–0.65` under the `mlp[:-2](l2n(img_proj) * l2n(text_proj))` emitter, `0.031` under the pre-MLP Hadamard convention. **Only "non-zero" is invariant**, and only "non-zero" is load-bearing |
| (b) | `h_std[355] == h_ow[355]`? | **Yes, exactly**; not zero; the **only** such row on either dataset |
| (c)–(e) | `l2_rows` per block | endpoint and common **DIE** at `{355}` / OK all-False; displacement OK at `{355}` / **DIES** all-False |
| (f) | ⇒ `common_displacement` in head space | **unbuildable under either mask** |
| (g) | **the repair**: 13 head-space arms at `n = 743`, all-False | **ALL BUILT**, `float32`, dims **`4 × 1024-d` + `9 × 2048-d`** |
| (h) | **the bridge**: raw arms `n = 743` vs `n = 744` one-hot, restricted | **BIT-EXACT, `max\|diff\| = 0.000e+00`**, all 13 arms; every `ρ` unchanged |
| (i) | `prepare_views` with `zero_mask = None` | **DIES on MHC-ZH `n = 579`, on HateMM `n = 744`, and — round-4 measurement α — on the HateMM arena `n = 743` too**, where no null row exists. The inadmissibility is a property of the function, not of the data |
| (j) | `prepare_views` on the arena population with the explicit all-False mask | **OK on BOTH datasets** — the executed form of §3.7's convention |
| (k) | `ρ` over 744 rows including the masked zero row | shifts by up to `1.301e-03` — would fail the 4-dp reproduction leg |
| (l) | `headspace_fidelity.py` | opens `mint_*_ffull.npz` (`:66`) and the banked trainlogs (`:31/:33`) — **no `dev_seen_*.pt` at all** (round-4 measurement ε ⇒ §12) |

### 7.5 The one-block instantiation

`fuse([b])` differs from `l2(b)` by `7.451e-09` (reproduced identically by rounds 2 and 3). The
one-block `paired` differs from v1's rejected `pair` by a **head-weight-dependent** amount —
`1.118e-08` here, `7.451e-09` (round 2), `3.725e-09` (round 3). **Only the invariant claim carries
weight: all are a fraction of a `float32` eps (`1.192e-07`).**

### 7.6 `GATE-C01PARITY`

Bit-exact, `max|diff| = 0.000e+00`, all 13 arms, both datasets, with §3.7's mask forms.
Independently reproduced from this document's prose by rounds 2, 3 and 4.

### 7.7 Unit table

| unit | what | dataset / space | measured |
|---|---|---|---|
| `U1` | head forward over one real ro cache | HateMM | 0.0461 s |
| `U2a` / `U2b` | vote, 1024-d / 2048-d, per fold-cell | HateMM, **head space** | 0.00305 / 0.00629 s |
| `U2c` / `U2d` | vote, 7168-d / 14336-d, per fold-cell | HateMM, **raw space** | 0.04239 / 0.08098 s |
| `U3` | bootstrap `B = 2000`, one comparison, both metrics | HateMM | 0.126 s |
| `U4` | one shuffled-pair null draw (2 arms × 5 folds + rebuild) | HateMM, **head space, 1024/2048-d** (round-4: the space is now named) | 0.08908 s |
| `U5a` / `U5b` | two-block build + compare / builder-only | HateMM | 11.27 / 4.63 s |
| `U6` | `ρ` over 13 arms | HateMM, raw | 0.62 s |
| `U7` | `GATE-SHA` over 8 caches + 6 modules | — | 0.12 s |
| `U8` | ro cache `torch.load`, 2 files | HateMM | 0.033 s |
| `U9` | `GATE-DEVFID`, per `(dataset, seed)` | HateMM / MHC-ZH | 3.70 / 3.49 s |
| `U10` | head-space build of all 13 arms, one cell | HateMM, `n = 743` | 0.1873 s |
| `U11` | interpreter + imports | — | 3.05–3.18 s (**inside the mint units**) |

**Convention:** every unit was measured on **HateMM**, the larger dataset, and applied to MHC-ZH
unchanged, so every such application **over-states** the MHC-ZH cost. Exceptions: `U9` (per
dataset), `U7` and `U11` (dataset-independent).

**Corroboration status, carried to freeze.** `U5a`, `U5b`, `U6`, `U10`'s object and the mints are
independently reproduced by rounds 3 and/or 4; `U2a`–`U2d`, `U3`, `U4`, `U7`, `U8`, `U11` are not.
**`U4` is the single largest uncorroborated unit** (`273.7 s`, `9.3 %` of the total) and its space
is now named above. The freeze record will state the exit-status discipline under which each unit
was timed — the instrument that catches the `U9` defect class, where a crashed process's `echo`
status was briefly recorded as a measurement before being re-run correctly with `--seeds 0`.

### 7.8 Dry-check cost

v5's measurements added ≈ **2 wall-minutes / ≈ 4 CPU-minutes** (the C-2 counterexample, the Holm
feasibility arithmetic, the fold recount, the line-cite checks). Cumulative v1–v5: ≈ 18
wall-minutes / ≈ 77 CPU-minutes, all `$0`, zero GPU. The CPU-cap conflict was knowable from C09's
banked mint costs before the first burn; all four rounds ruled the underlying trade correct — a
standing `TARGET_STATE.json` rule beats a task brief's CPU cap.

---

## 8. Compute projection — measured unit × explicit count

**Round-4 C-3 corrected the enumeration on the axis `rule_1_compute_projection` names by name.**
The head key matrix is **per fold** — `headspace_arena.py:75-89` loads
`mint_{ds}_s{seed}_f{fold}.npz` inside the fold loop, and §8's own Phase 1b decomposition proves
**60 fold mints** — so head-space arm construction and head `ρ` are `60`-cell loops, not `12`.
`GATE-ZEROOP`'s two guard arms were counted nowhere.

| phase | count | unit | product |
|---|---|---|---|
| **1** Head-N mints, HateMM fold / full | `15` / `3` | 40.39 / 49.30 s | `605.9` / `147.9 s` |
| **1** Head-N mints, ZH fold / full | `15` / `3` | 34.40 / 38.87 s | `516.0` / `116.6 s` |
| **1R** Head-R mints, HateMM / ZH | `15` / `15` | 40.39 / 34.40 s | `605.9` / `516.0 s` |
| **1b** key forwards `(30×3)+(6×4)+(30×2)` | `174` | `U1` | `8.0 s` |
| **1c** ro cache loads, per process | `66` | `U8` | `2.2 s` |
| **1d** `GATE-SHA`, once in the driver | `1` | `U7` | `0.1 s` |
| **2** head-space votes, 1024-d / 2048-d arms | `4×60 = 240` / `9×60 = 540` | `U2a` / `U2b` | `0.7` / `3.4 s` |
| **2** `GATE-FLOOR` native vote | `30` | `U2a` | `0.1 s` |
| **2b** head-space arm construction — **`2 ds × 3 seeds × 5 folds × 2 lin`** | **`60`** *(was 12)* | `U10` | **`11.2 s`** |
| **2z** `GATE-ZEROOP` guard arms — votes `2 × 60`, construction `60 × (2/13)` | **`120` votes + `60` partial builds** *(was 0)* | `U2b` / `U10` | **`2.5 s`** |
| **2R** raw votes, 7168-d / 14336-d | `4×10 = 40` / `9×10 = 90` | `U2c` / `U2d` | `1.7` / `7.3 s` |
| **2Ra** raw arm construction | `2` datasets | `U5b` | `9.3 s` |
| **2C** `GATE-C01PARITY` | `2` datasets | `U5a` | `22.5 s` |
| **2C** `GATE-ROWSUBSET` (HateMM only) | `1` | `U5b + 0.21` | `4.8 s` |
| **2D** `ρ` — **raw `2` + head `60`** | **`62`** *(was 14)* | `U6` | **`38.4 s`** |
| **3** shuffled-pair null draws | `256 × 3 × 2 × 2 = 3072` | `U4` | `273.7 s` |
| **4** bootstrap comparison-cells | `23 × 2 ds × 2 lin = 92` | `U3` | `11.6 s` |
| **5** head-space null-row sensitivity | **0 — the leg does not exist** | — | `0.0 s` |
| **6** `GATE-DEVFID` | `3 + 3` | `U9` | `21.6 s` |
| **7** per-gate arithmetic on materialised vectors — `GATE-SELFTEST` (**`14 × 3 × 2 × 2 = 168`**, round-4 I-7), `GATE-NESTED`, S7, `GATE-POP` (incl. class counts and constant recomputation), `GATE-NULLREMOVED`, `GATE-IDPARITY` | all | sub-`0.1 s` class | `0.1 s` |

**Total, re-multiplied:**
`2886.3 − 2.2 − 8.7 + 11.2 + 38.4 + 2.5 = ` **`2927.5 s = 48.8 min`** corroborating;
**`× 1.25 = 3659.4 s = 61.0 min`** conservative.

*(Round 4's headline figure was `2925.0 s`; the `2.5 s` difference is the guard-arm
**construction**, which it offered as an option and v5 counts. The direction is conservative.)*

**M-2, adopted:** the printed product column now sums to the total directly, with **Phase 7
carried at its `0.1 s` upper bound**; v4's rounding note described v3's table and is retired.

**Declared slack, outside the projection:** `30 s` for ledger aggregation and JSON emit.

**Peak RSS ≈ 1.3 GiB.** Request 32 GB.

**Where the risk sits.** Mints are `85.7 %` of the total and are measured directly. Phase 3 is
`9.3 %`; a 2× miss moves the total to `3201.2 s = 53.4 min`, a 5× miss to `4022.3 s = 67.0 min`.
**If the realized cost exceeds the conservative total by more than 2×, that is itself a reportable
process finding.**

---

## 9. Heartbeat specification

* One progress file `$BASE/progress/C06_PROGRESS.txt`, created by the sbatch driver before the
  first python process starts; every python process appends through a handle opened
  `buffering=1`. The bash driver **also** echoes a line per mint, unbuffered.
* Each line: ISO-8601 timestamp · phase · units done / total · elapsed · elapsed ÷ **§8's frozen
  projected** value.
* Granularity: one line per mint (66), **one per training epoch within a mint**, one per
  `(dataset, seed, lineage, fold)` arm block, **one per Phase 2D `ρ` cell** (round-4 C-3: at the
  corrected `62` cells Phase 2D is a `38.4 s` span, above the `~15 s` claim under a
  one-line-per-gate rule), one per 32 null draws, one per bootstrap block, **one per
  `(gate, dataset)`** (round-4 M-4: `GATE-C01PARITY` runs `11.27 s` per dataset, so one line per
  *gate* would leave a `22.5 s` span), one per verdict field.
* **The HALT path names which gate failed in its final line**, and a `RuntimeError` from the
  imported C01 algebra is caught, recorded as `INSTRUMENT_INCONCLUSIVE` with its `context` string,
  and written to that line before exit.
* Under this granularity **no interval exceeds ~15 s**: the longest un-instrumented span is a
  single `GATE-C01PARITY` dataset at `11.27 s` (`14.1 s` conservative).

---

## 10. Scope of any verdict

### 10.1 The prompt/readout-span confound

`generate_VideoMLLM_embedding_readout_HF.py:73-89` defines
`("ro_L24", "baseline", "prefix", "response", LAYER_MID)` versus
`("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`: the `ow_` cell changes **the
prompt kind and both readout spans**. **No result of this battery can attribute an effect — or its
absence — to the prompt alone.**

### 10.2 What a CLOSE would and would not close

A CLOSE closes **C06's first-order (tangent/chord) prompt-orbit route in the fold-head arena at
`ro_L24`, on `HateMM (-LoRA-curric)` at `n = 743` and `MHC_zh (-LoRA)` at `n = 579`, under the
lineage(s) that passed their instrument gates — named explicitly in the verdict**, together with
the `GATE-DOMAIN` recovery fraction and the raw-vs-head `endpoint_std` comparison. It does **not**
establish:

* anything about **curvature** — two prompt points give a chord; ≥ 3 require extraction;
* anything about a head **retrained per arm** — F66's trained-reshaping caveat stands;
* **anything about the per-modality contrast C01 measured**: the deployed head fuses image and
  text internally, so every head-space arm is a **post-fusion, one-block analogue** of C01's
  per-modality two-block contrast. The one-block reading is forced by the architecture, not
  chosen, but what it measures is not the transform C01 scored `0.8505 / 0.8846`;
* anything about a **different readout span**, **L28**, or the **test split**;
* **anything about directions off the primary's own family (round-4 I-8).** `orthogonal_blocks()`
  is a **Givens mixing of the two endpoint blocks**: `θ = 45°` *is* `common_displacement` and
  `θ = 0` *is* `endpoint_concat`, so the six angles are controls on the primary's **own
  one-parameter family**, not independent directions. A CLOSE therefore establishes *"the real
  displacement is not the best angle on its own family"* — **not** *"the prompt-orbit tangent
  carries nothing"*. The Gate-0 record insists on the same reading (*"a matched-norm random
  direction **can** reach or exceed the real displacement, not that every one does"*), and §1's
  table shows 4 of 6 HateMM rotations and 2 of 6 ZH rotations sitting **below** the primary.

### 10.3 What a SURVIVE would license

Only that C06 *"has earned its extraction"*: the `1.7–2.5 GPU-h` bounded extraction may be
**proposed** under `iteration_8_stage0_bounded_extraction_amendment`, with its own preregistration,
design review, separate code/resource review lineage and authorization. **A SURVIVE is not a
Stage-0 PASS and authorizes no GPU.**

### 10.4 Bans checked

F80's object is prompt **language**; F70's is individual **readout cells**; C06's is the **relation
between** two cells. All four rounds tested and confirmed the object-mismatch warrant. The
multi-prompt **ensembling** carve-outs have C14 as their object and are not relied on.
`endpoint_std` and `endpoint_ow` **are** literally F70's two cells, entering as **controls**. No
ensemble of prompt predictions is formed: `avg_score` is C01's own frozen `gain_control`.

**Hard constraints: none touched** — re-read verbatim against `d_no_other_relaxation` by rounds 3
and 4.

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

**The v4 → v2 chain**, sha-gated in source at both hops
(`c01_policy_contrast_a0_v4.py:52-55` → `_v3.py:48-51` → `c01_policy_contrast_a0.py`, the file this
battery imports). Config chain v4 → v3 → v2 with `scientific_thresholds_exact: true` at each hop.

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

Plus the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`, the ten banked
`vsw_ckpt/{hatemm,zh}/f{0..4}.npz`, and — read-only, for §6.1's reference measurement — the 36
banked `artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`.

**New code, confined to the battery:** `scripts/analysis/c06_falsifier_mint.py` (the single shared
driver), `scripts/analysis/c06_falsifier_arena.py`, `configs/c06/c06_falsifier.json`,
`scripts/slurm/c06_falsifier_cpu.sbatch`. All four are **absent** from the tree, as round 4
confirmed.

---

## 12. Label and split discipline

**Test contact: none, enforced in three layers** — `headspace_mint.py:106-116`'s `torch.load`
guard; the driver's `split == "train"` assertion on every ro-cache load; and the frozen
`c09_guard` `sitecustomize` installing an `open()`-level, component-wise, repo-scoped predicate at
interpreter startup in **every** process.

**`GATE-LEDGER`:**

| predicate | expected | binding? |
|---|---|---|
| `test_path_opens` | **0** | yes |
| `test_label_materialisations` | **0** | yes |
| `mints_present_before_arena` | **66** `.npz` (36 Head-N + 30 Head-R) | yes |
| `dev_path_opens` | **`mints_executed + 0`** — round-4 I-5: `headspace_fidelity.py` opens **no** `dev_seen` file, reading `lab_dev` out of the banked mint `.npz` (`:66`), so the second term is zero, not free | yes |
| `dev_label_materialisations_outside_decisions` | **`mints_executed`**, one per executed mint | yes |
| `dev_or_test_labels_into_decision_quantities` | **0** | yes |
| `banked_trainlog_opens` | `GATE-DEVFID` only, `2 × 3` | reported |
| processes reporting | **66 mints + 6 fidelity + 1 arena** | yes — HALT on any mismatch |
| predicate coverage | re-derived in-job | reported |

**Why `mints_executed` and not `66`.** `headspace_mint.py:192-194` returns **before** the
`dev_seen` load at `:199` whenever `--out` exists, so on a **resumed** job a skipped mint opens no
dev file. A binding `dev_path_opens == 66` would HALT a legitimate resume — the same class of
self-defeating gate this lineage has removed twice elsewhere. Binding against the measured
`mints_executed`, plus the separate `mints_present_before_arena == 66` assertion and
`GATE-FOLD`'s banked-flag re-read (§3.2), is exactly predictable on fresh, resumed and
partially-resumed runs and after a HALT, and still closes the absence lane.

**Dev labels.** `headspace_mint.py:199` loads the native `dev_seen` unconditionally on every mint
and `:322` writes `lab_dev` into every `.npz`; `--train-cache` does not redirect it. Head-R opens
no `dev_seen_*-ro_*` file. None reaches a decision quantity: at `fold ≥ 0` `dev_sp` is a slice of
the fitting pool (`:223-226`) and only `dv[3]` is written to disk.

**No selection anywhere.** Every threshold in §5 and §6 is C01's frozen value, C09's banked
constant, a population-derived constant frozen in §3.7 from banked label-free arithmetic, or a
declared engineering choice disclosed in §5.9.

---

## 13. Execution boundary, and what the code lineage must verify

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** 73 processes in the order 66 mints → 6 fidelity → 1 arena, with
`GATE-SHA` once in the driver before any of them and `GATE-POP` before any population-consuming
gate. All four rounds confirmed the channel; the cloud route is inapplicable because `GATE-FLOOR`
anchors to six floors measured locally on `foscsmlprd01`.

**Not authorized by this document.** Required before anything runs: an independent design review to
GO (0C/0H/0I), a **separate** code/resource review lineage over the executable, and main-dialogue
authorization.

**The handoff — round 3's twelve items plus round 4's six.**

*The shared mint driver:* (1) it imports `headspace_mint` with its sha256 asserted and **no**
behaviour outside `--train-cache` differs between lineages; (2) `--train-cache` overrides **only**
the training cache and cannot reach `model_name`, the dev load or the dataset table, and §12's
declared counts match what the code does; (3) **no branch conditional on the cache filename or
suffix** — `GATE-FLOOR` exercises the native path only, which is why `GATE-SHA` over the ro caches
and `GATE-IDPARITY` are load-bearing; (4) the `GATE-FLOOR` mints and the Head-R mints go through
the *same* function.

*Populations and constants:* (5) **every** population-derived constant in §3.7's table — the arena
size, class counts, majority, `GATE-ARENA` band, the small-displacement quantile, `GATE-DOMAIN`'s
**two** majorities and the tie cap — is **computed from the arena, not read**; (6)
`GATE-SELFTEST`'s `n` is the arena size and no banked `744` leaks into a per-item denominator;
(7) `ρ` is computed over the `743/579`-row matrices, not a 744-row array with a masked row left in
(a `1.301e-03` shift, fail-safe but presenting as an unexplained HALT).

*The mask convention:* (8) every `prepare_views` call passes an explicit boolean array and every
`l2_rows` call's mask matches the population it is handed — **with an assertion, not a comment**;
(9) the `n = 744` build exists **only** inside `GATE-C01PARITY`/`GATE-ROWSUBSET` and nothing votes
on it.

*The tie diagnostic:* (10) which ranking and which residual the implementation uses, that the
residual is the **head-space** `‖Δk‖₂` (or its `√d` bound), that "collapse" is the worst case over
orderings, and that the report branch cannot be reached outside the tie set or above the cap.

*`GATE-POP` and heartbeat:* (11) `GATE-POP` runs **before** any gate consuming a population-derived
constant and asserts row identity by **index set**; (12) all six §9 items plus the `RuntimeError`
wrapper, the `buffering=1` handle never re-wrapped, the unbuffered driver echo, append-without-
interleaving across all 73 processes, and the frozen `elapsed ÷ projected` denominator.

**Round 4's additions:** (13) **the fold axis** — the 13 arms and every `ρ` are rebuilt from
**each** of the 60 fold key matrices, and no arm built under head `f` is ever voted for a query
outside fold `f`'s held-out fifth; (14) **the guard arms** — `orthrot_0` and `orthrot_45` are built
by the *rotation* route and `endpoint_concat` / `common_displacement` by their own, never aliased,
and all four voted; (15) **S7** — its classification as a SURVIVE condition, its
`common_displacement`-only arm scope, and the zero-fix convention, which the battery does **not**
inherit because it does not import `displacement_audit`; (16) **the statistics** — the bootstrap
statistic, the one-sided p and the Holm step-down match §5.4/§5.5, and no **decision** quantity
(not only no gate quantity) reaches a comparison non-finite; (17) population-derived constants,
extended, as item 5 now states; (18) **`GATE-FOLD` under resume** — fold parity verified for all
66 mints including skipped ones, by reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of`
from the banked `.npz`.

---

## 14. Cumulative disposition

### Round 4 (14 findings + 4 Minor) — all adopted

| finding | disposition | where in v5 |
|---|---|---|
| **C-1** `GATE-ARMVIAB`'s escape unreachable ⇒ fires on a warranted CLOSE | **ADOPTED, followed through to retirement** — restriction to `endpoint_std` would be strictly redundant with `GATE-ARENA`, so the gate is **deleted**; its raw-vs-head residue becomes a reported diagnostic; §6.3's invariant is now literally true | §6.2, §6.3, §6.4, §6 |
| **C-2** `S3 ⇒ S6` false; S6 demoted | **ADOPTED (repair a)** — S6 is **binding** again; *"reported, not screening"* deleted; §5.9 item 4 rewritten with the verified counterexample `(2, 21, 22)` and the seed-mean/per-seed distinction | §5.2, §5.9 |
| **C-3** fold axis missing; guard arms uncounted | **ADOPTED** — Phase 2b `12 → 60`, Phase 2D `14 → 62`, new Phase 2z; total `2927.5 s` / `3659.4 s`; per-cell heartbeat for 2D | §8, §9 |
| **H-1** `GATE-SMALLDISP` classified as a HALT gate | **ADOPTED** — it becomes **S7**, a SURVIVE condition per C01's own placement; arm scope `common_displacement`; zero-fix convention pre-registered; removed from §6 | §5.2, §6 |
| **H-2** S4's statistic unregistered | **ADOPTED** — §5.4 pre-registers the resample, the per-resample delta, the `5 %` lower bound, the one-sided p and the Holm step-down; §5.5 records the `B = 2000` / 92-family resolution floor (**42 of 92 must show zero adverse resamples**) | §5.4, §5.5 |
| **H-3** global HALT vs per-lineage SURVIVE | **ADOPTED (option ii)** — gates are scoped **global / per-lineage**; a failing lineage is dropped, not the battery; CLOSE still requires **two** clean negatives; the lineage(s) that ran are named in the verdict | §5.6, §6, §10.2 |
| **I-1** `GATE-FOLD` undefined under resume | **ADOPTED** — discharged both on executed mints and by re-reading the banked parity flag from all 66 `.npz` | §3.2, §6 |
| **I-2** `GATE-DOMAIN` mixes populations | **ADOPTED** — `maj_arena` with `acc_ro`, `maj_full` with the banked `acc_native`; both in §3.7's table | §3.7, §6.4 |
| **I-3** small-displacement quantile population unnamed | **ADOPTED** — computed on the **arena** norms; in §3.7's table and in S7 | §3.7, §5.2 |
| **I-4** tie criterion compares key components to similarities | **ADOPTED** — `‖Δk‖₂` (or `√d · max\|Δk\|` = `5.394e-06` at `d = 2048`, **45×** v4's threshold); "collapse" defined as worst case over orderings; the residual is the **head-space** one | §6.5 |
| **I-5** `dev_path_opens` binding with a free term | **ADOPTED** — the term is `0`, measured | §12 |
| **I-6** `ρ_head` cell/aggregator undefined | **ADOPTED** — **per fold, all 60 cells, HALT if any fires** | §6, §6.1 |
| **I-7** `GATE-SELFTEST` excludes `avg_score` | **ADOPTED** — all **14** arms; Phase 7's count `156 → 168` | §3.5, §6, §8 |
| **I-8** §10.2 omits the Givens-family narrowing | **ADOPTED** — sixth scope bullet | §10.2 |
| **M-1** `classifier.py:80-81` off by one | **ADOPTED** — `:81-82` | §3.7 |
| **M-2** Phase 7 rounding note stale | **ADOPTED** — retired; the column now sums to the total | §8 |
| **M-3** `0.58–0.65` emitter-scoped | **ADOPTED** — both emitter conventions recorded; only *"non-zero"* is load-bearing | §7.4(a) |
| **M-4** `GATE-C01PARITY` heartbeat span; `GATE-ROWSUBSET` population | **ADOPTED** — one line per `(gate, dataset)`; §3.7's table marks `GATE-ROWSUBSET` **HateMM only** | §3.7, §9 |

**Round-4 measurements folded into the record:** α (`None` dies at the arena too, §7.4(i)),
β (`GATE-IDPARITY` holds directly, §7.1), γ (head-space dims `4 × 1024` + `9 × 2048`, §3.4/§7.4(g)),
δ (`deployed_vote` confirms the neighbour-displacement argument, §3.7), ε (`headspace_fidelity`
opens no `dev_seen`, §7.4(l)/§12), ζ (the Holm resolution floor, §5.5).

### Rounds 1–3 — carried

Round 4 audited **12 of 13** round-3 adoptions as genuinely real, with the thirteenth (I-3)
textually present but arithmetically broken by I-4 — repaired here as C-2. Round 3 had audited
16 of 16 round-2 adoptions clean and all three reopened round-1 items repaired. Those dispositions
stand as recorded in v3 §14 and v4 §14 and are not restated except where a round-4 finding refines
them: round-3 I-3/I-4 (§5.2, §5.9), I-5 (§6.5), I-6 (§6.1), C-2 (§3.7's constant table).

**Rulings carried without change across all rounds:** the direction of *"conservative"*; A7 is not
an obstacle; per-arm retraining excluded; `max` as `ρ*`'s order statistic; SLURM and the login-node
dismissal; the untrained-head blindness discipline; §5.9 item 1's inapplicability reasoning; S6's
net-fix reference; the tie cap's defensibility; `GATE-ROWSUBSET`'s renaming; and §3.4's account of
what two-block parity does and does not buy.

---

## 15. Open issues for round 5

1. **`GATE-ARMVIAB`'s retirement (§6.2).** The real arms now carry **no** lower-bound instrument
   HALT. Round 5 should confirm this is right — that a real arm collapsing while the head-space
   controls stay healthy is science and not instrument failure — and that nothing else in §6
   reintroduces a lower bound on them by another route.
2. **Per-lineage gate scoping (§5.6).** The global/per-lineage split is new. Round 5 should check
   the classification arm by arm: is any gate listed **global** that should be per-lineage, or
   vice versa? In particular `GATE-FLOOR` is global because it anchors the shared driver — is that
   the right reading?
3. **S6's binding form (§5.2, §5.9 item 4).** S6 now binds through across-seed dispersion.
   Round 5 should confirm the counterexample and rule whether requiring `3/3` seeds on an integer
   count frozen for an arena `7×` smaller is the right transfer, or whether it is now *too*
   tight.
4. **S4's statistic (§5.4) and the resolution floor (§5.5).** Newly pre-registered. Round 5 should
   check it against C01's `paired_bootstrap` and `holm_adjust`, and rule whether **42 of 92
   comparators showing zero adverse resamples** is an acceptable bar to freeze, or whether `B`
   should rise.
5. **The corrected enumeration (§8).** Round 5 should re-derive the 60-cell counts and hunt for
   any remaining uncounted loop — four rounds have found one each in Phases 1b, 2b, 2D and 2z.
6. **The tie diagnostic's head-space residual (§6.5).** It is now defined against a quantity
   measured at run time rather than a frozen constant. Round 5 should confirm that is
   pre-registration-safe, i.e. that no decision can be tuned by it.

---

*No GPU, SLURM, Modal, teacher call, model load, training of any deployed arm, cache write,
test-split access, job submission or commit occurred in producing this document. Login-node
dry-check processes only (§7.8). `TARGET_STATE.json` was read and not modified. v1–v4 are
unmodified.*
