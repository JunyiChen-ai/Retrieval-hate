# C06 `$0` CPU falsifier — preregistration **DRAFT v1** (2026-08-04)

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists. No hash is
frozen. `TARGET_STATE.json` is untouched. This document is the design of record offered
for independent review; the review request is
`refine-logs/C06_FALSIFIER_REVIEW_REQUEST.md`.

**What ran while writing this.** Zero GPU, zero SLURM, zero Modal, zero test-split
contact, zero cache write into `data/`, zero commit. Login-node dry-check processes only
(§7), all in the session scratchpad. Their arithmetic is reported in full in §7 and §8,
including one measured design finding (§7.4) and one instruction overrun (§7.6).

---

## 1. What this falsifier is, and what authorizes it

C06 (*Prompt-Orbit Tangent/Curvature*) is **not an active candidate**. Its registry
status is `gated_on_zero_cost_falsifier`
(`TARGET_STATE.json::gate0_reopen_2026_07_31.dispositions.gated[0]`). What the campaign
queue has reached is **C06's falsifier**, not C06
(`iteration_8_queue_state_2026_08_04.next_item.IS_IT_AN_ACTIVE_CANDIDATE`: *"NO. C06
remains GATED."*).

**The unblock condition, quoted verbatim** from
`gate0_reopen_2026_07_31.dispositions.gated[0].falsifier_spec`:

> re-run C01's real-displacement-versus-matched-norm-orthogonal-rotation battery in the
> FOLD-HEAD ARENA on the already-banked `ro_*` caches. Zero GPU, zero extraction, minutes
> of CPU on `scripts/analysis/headspace_{mint,arena}.py`, which exist and are banked. If
> the rotations again match the real displacement in the deployed head space, C06 closes
> for `$0` and the `1.7-2.5 GPU-h` of extraction is never queued; if they do not, C06 has
> earned its extraction

**The two binding design constraints, quoted verbatim** from the same entry
(`falsifier_design_constraints`):

> its pre-registration must (i) use the per-dataset adapter lineage that ACTUALLY EXISTS
> — HateMM has only `-LoRA-curric` ro-caches, MHC_zh has only `-LoRA`, one lineage each,
> not a matched pair (correction V-8); and (ii) declare the prompt/readout-span confound,
> because `generate_VideoMLLM_embedding_readout_HF.py:73-89` shows the `ow_` cells change
> the readout span as well as the prompt — the same confound C01's review already narrowed
> its claim for

Both are honoured: (i) §3.1 pins one adapter lineage per dataset and asserts it at run
time; (ii) §10.1 declares the confound as a limitation on the face of the verdict.

**The evidence the gate rests on** is C01's A0, re-verified by the Gate-0 adjudicator
directly against `C01_A0_OUT.json` with every accuracy recomputed from the stored
confusion matrices (`GATE0_REOPEN_2026-07-31.md` §4.4):

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

with `gain_over_strongest_control` `−0.0094` (HateMM) and `−0.0256` (MHC-ZH),
`pass: false`, `decision.continue = false`.

**Round-14's sharpening, which this design adopts** (`rotation_family_precision_R14`):
`c01_policy_contrast_a0.py:1272`'s `orthogonal_blocks()` is a **Givens mixing of the two
endpoint blocks**, so the six "random rotations" are six angles on **one parameter
family** that also contains the primary — `θ = 45°` **is** `common_displacement` and
`θ = 0` **is** `endpoint_concat`, confirmed by the code's own guards at max abs diff
`8.9e-08`–`1.2e-07`. The correct reading is *sharper and more adverse*: the real
displacement is one angle among many on a family where several other angles do better.
This design keeps that framing and re-asserts both algebra identities as a HALT gate in
the new space (§6, `GATE-ALGEBRA`).

**Why a re-run in a different space is the right instrument.** C01's arena is **raw dev
keys** (`n_dev` 107 / 78), not the fold-head path. The registry's own
`unified_pilot_gate.arena` reads *"strict train-OOF or untouched development split using
the actual fold-head/deployed-head path; raw-key arena may kill but may not promote a
lead"*, and F113 marks the raw-KILL direction **NOT ESTABLISHED** (correction V-4). So a
raw-arena negative is not, by itself, a closure — which is precisely why the disposition
is a falsifier and not a strike.

---

## 2. The three process rules from F118 that bind this design

`TARGET_STATE.json::process_rule_compute_projection_and_heartbeat_2026_08_04` names this
falsifier by name in `applies_immediately_to`:

> the C06 `$0` CPU falsifier, whose own spec says *"minutes of CPU on
> `scripts/analysis/headspace_{mint,arena}.py`"* — an estimate of exactly the falsified
> kind, on the SAME arena family that over-ran here.

| rule | source | discharged in |
|---|---|---|
| **R1** measured-unit-cost × explicit-count compute projection; no reduced-scale extrapolation; if the real-scale unit cannot be measured, say **UNKNOWN**, never name a band | `rule_1_compute_projection` | **§8** — four mint unit types and five arena unit types measured at real scale on the real banked data, multiplied through enumerated counts, multiplication shown |
| **R2** line-buffered per-phase heartbeat to a progress file; block-buffered stdout flushed at exit does **not** satisfy it | `rule_2_heartbeat` | **§9** |
| **R3** (F114, carried in the brief) the zero-cost dry execution must exercise the **first real operation of the payload path**, not a line-read | F114 / F114b | **§7** — real caches loaded, real mints run, real head forwards, real `deployed_vote` on real minted head keys |

The motivating incident is not abstract: C09 A0's arena was projected at 20–30 CPU-min
and measured **51.61 h** (103×–155× over), and its five per-phase log lines were
block-buffered into a **51.6-hour blackout** during which a healthy run was
indistinguishable from a hang (F118, `sacct` job `13885`, Elapsed `2-04:04:18`).

A fourth rule, from `feedback-separate-code-review-lineage`, is procedural and binds
after this document: a design GO does **not** review the implementation; the battery
script needs its **own, separate** code/resource review lineage.

---

## 3. The arena, and the one design choice that matters

### 3.1 Inputs — the lineage that actually exists, asserted not assumed

C01's frozen configuration (`configs/c01/c01_a0_v2.json`, sha256
`f3997bdd…41563f5`) pins `standard_suffix = ro_L24`, `oneword_suffix = ro_ow_L24`,
`feature_dim = 3584`, `allowed_splits = ["train","dev_seen"]`. Directory listing
confirms **one adapter lineage per dataset**, exactly as correction V-8 says:

| dataset | adapter lineage (the only one banked) | `n_train` |
|---|---|---|
| HateMM | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | 744 |
| MHC_zh | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | 579 |

**These are not a matched pair and this design never treats them as one.** No
cross-dataset comparison of absolute numbers is made anywhere; every decision quantity in
§5 is a *within-dataset, within-seed* comparison between arms built from the *same*
lineage, and the two-dataset requirement is a conjunction over two independently-computed
verdicts.

**Provenance, measured today.** The four L24 files this battery reads are byte-identical
to the ones C01 measured — their sha256 prefixes match C01's frozen
`*_provenance_sha16` fields exactly:

| file | sha256 | C01 v2 `provenance_sha16` | match |
|---|---|---|---|
| `HateMM/train_…-curric_HF-ro_L24.pt` | `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f` | `6a44cce4f65d4a60` | ✅ |
| `HateMM/train_…-curric_HF-ro_ow_L24.pt` | `60054f3be1204ca7bf2ac55b9bae6a88dd84d9dda35b0225f1ca27ce61977f4e` | `60054f3be1204ca7` | ✅ |
| `MHC_zh/train_…-LoRA_HF-ro_L24.pt` | `1d33fe5d69083479f0b6968a924578770364c00ca78c37cfef664bb4b6221c06` | `1d33fe5d69083479` | ✅ |
| `MHC_zh/train_…-LoRA_HF-ro_ow_L24.pt` | `3ad1309dc75001820318e3e9a073b781d28ee0afc2b879571d063a276b8d2a23` | `3ad1309dc7500182` | ✅ |

The HateMM `ro_L24` match is stronger still: its **full 64-hex** digest equals C01 v3's
`lineage_evidence.diagnostic_train_cache_sha256`, character for character.

*(Transcription note, per the campaign's numeric-provenance discipline: the HateMM
`ro_L24` digest in this table was mis-copied in the first pass of this draft — the tail of
the MHC-ZH digest was pasted onto the HateMM prefix — and was corrected by re-reading
`sha256sum` output before this document was offered for review. Every digest in this file
and in §11 was re-read from `sha256sum` at that point.)*

The L28 replication leg (§5.6) reads four further files that are **outside** C01's frozen
8-file manifest; their sha256 are given in §11 and the leg is non-decisional.

`train_*.pt` **only**. The `dev_seen` ro-caches are opened by no phase of this battery
(the arena is train-OOF); the `test_seen` ro-caches exist on disk and are opened by
nothing (§6 `GATE-LEDGER`).

### 3.2 The head, the folds, and the vote — all frozen, none rebuilt

* **Fold contract.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over the
  train split (`mechnov_pairverify.K_FOLDS = 5`, `FOLD_SEED = 0`), asserted against the
  banked `scripts/analysis/vsw_ckpt/<ds>/f{0..4}.npz` `ho_idx` by
  `headspace_mint.py:203-216`, which refuses to run on mismatch.
* **Head.** The deployed-recipe RGCL head, re-minted on CPU because the deployed
  checkpoints are gone (F78 / `HEADCOV_PREGATE_RECORD.md` §1.1). Bank = the fitting pool
  (4/5 of train); queries = the held-out fifth; every item is held out exactly once, so
  each arm yields one OOF prediction vector over all `n` train items.
* **Vote.** `mechfix_ops.deployed_vote(..., topk=20)` — the bit-faithful top-20
  rank-weighted signed-cosine vote, `w = [20…1]`, cutoff `≥ 0`. This is numerically the
  same object C01's frozen config specifies (`retrieval.topk = 20`,
  `rank_weights = descending_integer`, `prediction_cutoff = 0.0`,
  `similarity = signed_cosine`), so the *operator* is held fixed and **the key space is
  the only variable** — the discipline `headspace_arena.py`'s own header states.
* **Configuration.** 2 datasets × 3 head seeds × (5 fitting-pool folds + 1
  deployed-configuration head) = **36 mints**, the identical shape C09 and C02 ran.
* **F88's binding caveat is satisfied by construction:** *"a CPU-trained arm must be
  paired against a CPU-TRAINED FLOOR, never against the banked GPU floor"*
  (`ERRPAT_HateMM_2026-07-26.md` §8). Every arm and every floor here is minted inside the
  same CPU fold-head arena.

### 3.3 Where the arms are built — the load-bearing choice, and why

The C01 arms are not all the same width. `endpoint_std`, `endpoint_ow`, `common` and
`displacement` are *fused* keys (one block per modality); `endpoint_concat`,
`common_displacement`, `common_interaction` and all six `orthrot_θ` are **paired** keys
(*two* blocks per modality, `c01_policy_contrast_a0.py:1318-1352`). A paired key is
`2 × 3584 = 7168`-d per modality, and the deployed head's projections are
`Linear(3584 → 1024)`. **The paired arms therefore cannot be fed to the deployed head at
all.** Any design that claims otherwise is not implementable.

The resolution — and it is the banked house pattern, not an invention — is
`scripts/analysis/c02_a0_mint.py` (C02 A0 v9, executed as job `13847`): **one head per
`(dataset, seed, fold)`, trained on the NATIVE deployed cache, then the SAME trained head
forwarded over each view's features**, with the views compared in the head's output
space. C06's transplant is that pattern with the views replaced by C01's orbit endpoints:

1. Mint head `h_{D,s,f}` on the native deployed cache — `headspace_mint.py`'s recipe,
   reached through a thin wrapper exactly as C02 reached it.
2. Forward `h` over the two orbit endpoints to get **head-space endpoints**
   `e_std = h(img_std, txt_std)` and `e_ow = h(img_ow, txt_ow)`, both in `R^1024`.
3. Build **all fourteen C01 arms from `e_std`, `e_ow`** with C01's identical block
   algebra, one block instead of two per modality because after the head there is one
   fused key:

   | arm | head-space definition | dim |
   |---|---|---|
   | `endpoint_std` | `l2(e_std)` | 1024 |
   | `endpoint_ow` | `l2(e_ow)` | 1024 |
   | `common` | `l2(l2(e_std) + l2(e_ow))` | 1024 |
   | `displacement` | `l2(l2(e_ow) − l2(e_std))` | 1024 |
   | `endpoint_concat` | `pair(e_std, e_ow)` | 2048 |
   | `common_displacement` (**primary**) | `pair(common, displacement)` | 2048 |
   | `common_interaction` (secondary) | `pair(common, l2(common ⊙ displacement))` | 2048 |
   | `orthrot_θ`, θ ∈ {8.3, 17.6, 29.1, 60.4, 72.7, 83.8} | `pair(l2(cosθ·e_std + sinθ·e_ow), l2(−sinθ·e_std + cosθ·e_ow))` | 2048 |
   | `avg_score` | mean of the `endpoint_std` and `endpoint_ow` **vote scores** | — |

   where `l2 = mechnov_pairverify.l2n` and
   `pair(a,b) = l2(concat(l2(a), l2(b)))` — C01's `paired_key` with the modality loop
   collapsed. The angle grid is C01's frozen
   `orthogonal_rotation_control.angles_degrees`, verbatim; **45° and 0° are excluded from
   the grid because they *are* the primary and `endpoint_concat`**, and §6's
   `GATE-ALGEBRA` re-proves both identities in the new space rather than assuming them.
4. Run `deployed_vote` per arm, per fold, aggregate OOF, compare per §5.

**Why this reading and not the two alternatives** — enumerated as ambiguity **A4** in §4:

* It is the only reading whose instrument is *"`headspace_{mint,arena}.py`, which exist
  and are banked"*, as the condition requires.
* It is the only reading with a **free, banked fidelity anchor**: because `h` is trained
  on the native cache, its native-feature OOF accuracy must reproduce the six banked
  `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` floors **exactly**. That is §6's
  `GATE-FLOOR`, and no other reading has it.
* It is the only reading that costs *minutes* (§8: 36 mints = 1386.4 s ≈ 23 min), which is
  the condition's own cost model. The strongest alternative — retraining a head per arm —
  costs `13 arms × 5 folds × 3 seeds = 195` mints per dataset, i.e.
  `195 × 40.39 s = 7876 s` (HateMM) `+ 195 × 34.40 s = 6708 s` (MHC-ZH)
  `= 14 584 s ≈ **4.05 h**` of mints alone, ~10.5× this design, and is priced in §4/A4
  rather than hand-waved.

**What this reading does not test, stated here and again on the verdict (§10):** a head
**retrained** on orbit-geometry features. F66's caveat — raw-arena selection results do
not bind *trained* reshaping — therefore still applies to a per-arm-retrained
realisation, and any closure this battery renders is scoped accordingly.

---

## 4. Ambiguities in the written condition, and how each is resolved

The brief requires every ambiguity to be enumerated and resolved conservatively.
**"Conservative" is interpreted throughout as *hardest for the falsifier to deliver the
`$0` CLOSURE***, because the closure is the action with irreversible consequences: it
retires a candidate permanently and forgoes an extraction the
`iteration_8_stage0_bounded_extraction_amendment` has already authorized in principle. The
other outcome merely *queues* a measurement that must still clear its own Stage-0 bar. A
falsifier must earn its kill. **This interpretation is itself a reviewer decision point
and is flagged as such in the review request; the rule below is pre-registered either
way.**

| # | ambiguity in the written condition | resolution | direction |
|---|---|---|---|
| **A1** | *"the rotations"* — the best rotation, or the family? | C01's own frozen `decision.require_primary_above_all_rotation_controls = true`: the real arm must beat **every** rotation. Its negation is the closure trigger: the rotations "match" if **any** one of the six ties or beats. | as written; neither slackened nor tightened |
| **A2** | *"the real displacement"* — the `displacement` arm, or C01's primary `common_displacement`? The recon's own table compares the best rotation against both. | **Both**, disjunctively: C06 survives if **either** clears the full bar. | generous to C06, deliberately — a `$0` closure must not be an artifact of arm choice |
| **A3** | *"match"* — does a tie count? | Yes: the recon's phrase is *"matches or beats"*. The real arm must **strictly exceed**; a tie closes C06. This is C01's own `>` semantics. | as written |
| **A4** | *"in the fold-head arena"* — three implementable readings (native-trained head applied to the orbit endpoints; head trained on `ro_L24`; head retrained per arm) | The **C02 precedent**: native-trained head per `(seed, fold)`, arms built in its output space (§3.3). Decisive reasons: it is the only reading with a banked `GATE-FLOOR` anchor, the only one that uses the named banked instruments, and the only one inside the condition's own cost model. | see §3.3; the per-arm-retrained reading is the one that would be *more* generous to C06, and its omission is declared as a scope limit in §10, not buried |
| **A5** | which layer? `ro_L24` and `ro_L28` are both banked | **L24 is the primary and the sole decision surface** — it is C01's frozen `standard_suffix`. L28 is a free, independent second two-point orbit, run and reported as a **non-decisional** replication (§5.6). | as written |
| **A6** | seeds — C01's raw arena is seed-free; the fold-head arena is not | Seed-mean over 3 head seeds as the primary, **plus 3/3 per-seed agreement** on the rotation-dominance leg (the project's standing per-seed discipline, cf. B3's binding language). | tightens the bar C06 must clear |
| **A7** | *"C06 closes"* — closes **what**? | Two banked prompt points give a **chord, not a curvature** (recon §3.3). This battery adjudicates the **first-order (tangent/chord) leg only**. A curvature test needs ≥ 3 prompt points, i.e. extraction, and is `$0`-**impossible**. The state file's rule is executed as written; the curvature leg is recorded as **unmeasured**, not silently folded in. | scope statement — see §10.2, which the adjudicator must read before acting on a closure |

---

## 5. The pre-registered decision rule — written before any measurement

### 5.1 Notation

For dataset `D ∈ {HateMM, MHC-ZH}`, head seed `s ∈ {0,1,2}` and layer `L`, each arm `A`
yields one OOF prediction vector over all `n` train items (each item held out exactly
once), scored against **train-split** labels held out from the head that judged them.
`acc(A,D,s)` and `mF1(A,D,s)` follow; `acc(A,D)` denotes the mean over the three seeds.

* **Real arms** `R = {displacement, common_displacement}` (A2).
* **Rotation family** `Θ = {orthrot_8.3, 17.6, 29.1, 60.4, 72.7, 83.8}` — C01's frozen
  angles.
* **Ordinary controls** `C = {endpoint_std, endpoint_ow, avg_score, endpoint_concat,
  common}` — C01's frozen `decision.gain_controls`, verbatim.

### 5.2 SURVIVE — C06 earns its extraction

**C06 SURVIVES iff there exists `A ∈ R` such that all five conditions hold on BOTH
datasets at L24:**

| | condition | frozen source |
|---|---|---|
| **S1** | `acc(A) > max_θ acc(orthrot_θ)` **and** `mF1(A) > max_θ mF1(orthrot_θ)` | C01 `require_primary_above_all_rotation_controls` |
| **S2** | S1's accuracy leg holds in **3/3** individual seeds | A6 |
| **S3** | `acc(A) − max_c∈C acc(c) ≥ 0.02` **and** `mF1(A) − max_c∈C mF1(c) ≥ 0.02` | C01 `minimum_gain_over_strongest_control = 0.02` |
| **S4** | for **every** comparator in `C ∪ Θ`: paired item-level bootstrap lower bound on the difference `> 0`, and Holm rejects at `α = 0.05` | C01 `n_bootstrap = 2000`, `bootstrap_lower_quantile = 0.05`, `statistics.seed = 20260728`, `holm_alpha = 0.05`, `holm_metrics = [accuracy, macro_f1]`, `minimum_bootstrap_lower_bound = 0.0`, `require_primary_bootstrap_holm_reject = true` |
| **S5** | `acc(A)` exceeds the 95th percentile of `A`'s shuffled-pair null distribution | C01 `n_id_hash_permutations = 256`, `permutation_hash = sha256`, `bootstrap_upper_quantile = 0.95` |

### 5.3 CLOSE — C06 closes for `$0`

**C06 CLOSES iff the run publishes a verdict (§6) and SURVIVE is false.** There is no
third scientific outcome.

### 5.4 Multiplicity

The Holm family is the **11 comparators × 2 metrics = 22 hypotheses per `(real arm,
dataset)`**, `α = 0.05`. The two datasets are **not** pooled: S1–S5 must hold on both,
which is a conjunction and therefore conservative rather than anti-conservative. The two
real arms are an explicit **disjunction** (A2) — the single place where C06 gets two
shots — and its multiplicity is absorbed by requiring the *entire* S1–S5 conjunction of
whichever arm is claimed. The L28 leg enters no family because it enters no decision.

### 5.5 Instrument failure

Any HALT gate in §6 failing on either dataset ⇒ **HALT: no verdict published, in either
direction.** A HALT is an engineering outcome. It is recorded as
`INSTRUMENT_INCONCLUSIVE` and specifically **may not be reported as a closure** — C06's
gate stays exactly where it is and the falsifier is re-designed or re-run, subject to a
fresh authorization.

### 5.6 The L28 replication leg — declared non-decisional before it is run

`ro_L28` / `ro_ow_L28` give a second, independent two-point orbit at zero extra mint cost
(the head is the same; only the forward passes change). The entire battery is run and
reported at L28. It enters **no** decision rule and **no** multiplicity family. Its sole
formal obligation: if the L24 and L28 legs disagree on SURVIVE, the record must **say so
explicitly** and the verdict follows L24 alone. Declaring this now is what stops it from
becoming a second bite at the apple after the numbers are visible.

### 5.7 Pre-declared expectation, for prediction scoring

Recorded before measurement, per the campaign's practice of scoring its own predictions
(F118's *"Prediction scoring"*): **CLOSE is expected.** C01 measured the premise
rotation-indistinguishable at the two-point case on both datasets, the recon's structural
objection (a fixed prompt injects no per-item information) is unrebutted, and §7.4's
measured 219× orbit contraction points the same way. The falsifier is run because the
raw→head transfer is not established in the promote direction, not because the outcome is
in doubt. **A SURVIVE would be the surprise, and would need §6's gates to be clean before
anyone acts on it.**

---

## 6. Gates — HALT publishes no verdict

All must hold on **both** datasets.

| gate | what it asserts | dry-check status (§7) |
|---|---|---|
| `GATE-DET1` | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 8` exported **before** any python process starts; `headspace_mint.det1_assert` hard-fails otherwise | exercised, passes |
| `GATE-SHA` | every frozen import and every input cache matches its §11 sha256 | all 11 modules + 8 caches hashed today |
| `GATE-FOLD` | fold assignment matches banked `vsw_ckpt/<ds>/f{0..4}.npz` `ho_idx` | passed in all 4 real mints |
| `GATE-FLOOR` | the **native-feature** OOF vote in head space reproduces the banked fold-head floors at 4 dp: HateMM `0.8884 / 0.8858 / 0.8858`, MHC-ZH `0.8929 / 0.8895 / 0.8946`, and every `fold_acc_deployed` entry | anchor values read from the 6 banked `headspace_arena_*_OUT.json` |
| `GATE-IDPARITY` | every `ro_*` cache's `ids` list and `labels` vector are **order-identical** to the native bank (C02's assertion, `c02_a0_mint.py:70-77`) | **measured true on all 8 files today** |
| `GATE-ZEROMASK` | the measured exact-zero row set of the `ro_` inputs equals the pre-registered set — HateMM `{355}`, MHC-ZH `{}` — on both policies and both layers; and those rows are verified to give **identical** head keys under both policies | **measured true today**; row 355 = `hate_video_95`, C01's registered structural null |
| `GATE-ALGEBRA` | with the zero mask applied: `max|orthrot_0 − endpoint_concat| ≤ 2e-6` and `max|orthrot_45 − common_displacement| ≤ 2e-6` (C01's own bar) | **measured `1.39e-17` / `4.95e-15` (HateMM masked), `2.08e-17` / `5.24e-15` (ZH)** — see §7.4 |
| `GATE-DUALPATH` | C01's frozen `displacement_registered_null_exclusion` = *"with_null_masked_vs_physically_remove_null_dual_path_exact"*: masked path and physically-removed path agree **exactly** on every arm's predictions | HateMM only; vacuous on ZH |
| `GATE-ORBITSCALE` | fraction of unmasked items with `‖e_ow − e_std‖ < 1e-3` (C01 frozen `tiny_displacement_epsilon`) is `≤ 0.05` (C01 frozen `max_tiny_displacement_fraction`) | **the highest-risk gate — see §7.4** |
| `GATE-VIABILITY` | `endpoint_std`'s OOF accuracy strictly exceeds the majority-class rate | — |
| `GATE-LEDGER` | aggregated over **every** process: `test_path_opens == 0`, all expected processes reporting, predicate coverage re-derived in-job | reuses the frozen C09 guard, §12 |
| `GATE-DEVFID` | `headspace_fidelity.py` on the 6 deployed-configuration heads | **reporting only — does not gate**, exactly as in C09 §8.2 |

**`GATE-ORBITSCALE` is the gate that protects the closure**, and it exists because of a
measurement made today rather than an argument. See §7.4.

---

## 7. Dry-check — what was actually executed, and what it found

All on the login node `foscsmlprd01`, conda `HateVideo`, 8 threads, DET-1 environment
exported, outputs written only to the session scratchpad. Zero GPU (`CUDA_VISIBLE_DEVICES
= ""`), zero SLURM, zero test-split file opened, zero write into `data/` or `artifacts/`.

### 7.1 Real inputs, opened and checked

All 8 `train_*ro_*` caches on both datasets were loaded with the real `torch.load` path
and checked against the native bank. Measured: `n = 744 / 579`, `img_feats` and
`text_feats` both `(n, 3584)`, **`ids` order-identical and `labels` identical to the
native bank on all 8 files**. Exact-zero rows: HateMM `{355}` in **both** modalities of
**all four** ro caches *and* of the native cache; MHC-ZH **none**. The four L24 files'
sha256 match C01's frozen `provenance_sha16` (§3.1) — this battery reads the same bytes
C01 read.

### 7.2 Four real mint units, at real scale, on the real banked data

`scripts/analysis/headspace_mint.py` run **unmodified**, sha256 verified:

| unit | command | measured wall |
|---|---|---|
| HateMM fitting-pool head | `--dataset hatemm --seed 0 --fold 0` | **40.39 s** |
| MHC-ZH fitting-pool head | `--dataset zh --seed 0 --fold 0` | **34.40 s** |
| HateMM full-train head | `--dataset hatemm --seed 0 --fold -1` | **49.30 s** |
| MHC-ZH full-train head | `--dataset zh --seed 0 --fold -1` | **38.87 s** |

Fold parity against the banked `vsw_ckpt` passed in all four. Peak RSS of the HateMM
fold mint, from `/usr/bin/time -v`: **1 305 984 KB = 1.25 GiB** (C09's whole job measured
1.22 GiB — the two agree, and it is the *measured* resource prediction that F118 records
as having been accurate to 2 %). These four figures span the whole mint loop: the
projection in §8 uses them directly and extrapolates nothing.

### 7.3 Five real arena units, at real scale

Measured on the real minted head keys (`mint_hatemm_s0_f0.npz`, `K_train` `(744, 1024)`,
`n_fit = 595`, `n_ho = 149`) and the real ro caches:

| unit | measured |
|---|---|
| `U1` head forward over one real ro cache, `n = 744` | **0.0461 s** |
| `U2a` `deployed_vote`, one fold-cell, 1024-d | **0.0042 s** |
| `U2b` `deployed_vote`, one fold-cell, 2048-d | **0.0218 s** |
| `U3` bootstrap `B = 2000` for one comparison (accuracy + macro-F1) | **0.126 s** |
| `U4` one shuffled-pair null draw (rebuild 2 arms in head space + 2 arms × 5 folds of votes) | **0.1241 s** |

### 7.4 The payload path was executed end to end — and it found two things

The full arm construction was run on **real** head-space endpoints obtained by forwarding
a real `classifier_hateClipper` (constructed with the deployed CLI's own `proj_dim`,
`map_dim`, `fusion_mode`, `dropout`, `num_layers`) over the **real** `ro_L24` and
`ro_ow_L24` caches on both datasets: all 13 key-space arms built, both algebra guards
computed, and **every arm voted on one real fold-cell** through the frozen
`mechfix_ops.deployed_vote`. That is the first real operation of the payload path, per R3.

**The head is deliberately UNTRAINED.** A trained head would have revealed an arm ordering
before this document was frozen, which is a preregistration violation. An untrained head
makes every *operation* real at real scale while making the *numbers* scientifically void
— so no accuracy was computed or printed by the dry check, by design.

**Finding 1 — `GATE-ALGEBRA` fails on the raw run, and the cause is exactly one row.**
`max|orthrot_45 − common_displacement|` came out at `8.31e-02` on HateMM, four orders
above C01's `2e-6` bar. Traced and confirmed by measurement, not inference: `argmax` is
**row 355**, the registered structural null. Mechanism — the exact-zero input maps to the
*same* head key under both policies, so `displacement[355]` is exactly zero; but
`cos 45° − sin 45° = 1.11e-16 ≠ 0` in IEEE-754 double, so the rotation's second block at
`θ = 45°` is `1.11e-16 · v`, which `l2` renormalises into a **full-norm arbitrary
direction**. An exact zero becomes a unit vector. With the zero mask applied the guard
passes with enormous margin — `4.95e-15` (HateMM) and `5.24e-15` (ZH), against `1.39e-17`
and `2.08e-17` for `θ = 0`. **Design consequence:** C01's own frozen zero-mask discipline
(`displacement_registered_null_exclusion`) is carried into head space verbatim, and both
`GATE-ZEROMASK` and `GATE-DUALPATH` in §6 exist because of this measurement. Without it
the battery would have silently fed a garbage row into all nine paired arms.

**Finding 2 — the head contracts the prompt orbit by ~219×, and that is the single
largest threat to this falsifier.** Measured on unit-norm endpoints:

| space | median `‖e_ow − e_std‖` | median `cos(e_std, e_ow)` | rows below C01's `1e-3` epsilon |
|---|---|---|---|
| raw fused key (C01's arena) | **0.7016** | ≈ 0.754 | 1 / 744 |
| deployed head space (this arena) | **0.0032** | **0.999995** | 1 / 744 |

The two prompt endpoints are very nearly collinear after the head. **This is measured at
an untrained head and it is not known whether a trained head contracts more or less** —
training reshapes the space, and no measurement here bears on that. But the risk it names
is concrete: if the trained head pushes the orbit to the numerical floor, a
"rotations match displacement" result would be an artifact of the instrument rather than a
fact about C06, and it would look exactly like a clean closure. `GATE-ORBITSCALE` (§6) is
the pre-registered HALT that catches it, and it uses C01's own two frozen constants
(`tiny_displacement_epsilon = 1e-3`, `max_tiny_displacement_fraction = 0.05`) rather than
a threshold invented here. At the untrained head the gate would pass at `1/744 = 0.0013`
against a `0.05` bar — with only ~3.2× of headroom on the median, which is why it is a
gate and not a footnote.

### 7.5 What the dry check did **not** do

No trained head was applied to any ro cache; no arm accuracy was computed, printed or
recorded, on any dataset, at any layer. No `dev_seen` or `test_seen` file was opened. No
decision quantity of any kind exists yet.

### 7.6 Instruction overrun, disclosed

The brief capped the timing dry check at *"~5 CPU-minutes total"*. **One unit of the
dominant loop is itself 4.3 CPU-minutes** (40.39 s wall × 6.4 cores measured), so the cap
and R1's *"measure the per-unit cost on the real path at the real scale"* cannot both be
met. R1 was preferred. Total dry-check consumption: **≈ 3.5 wall-minutes, ≈ 22
CPU-minutes**, `$0`, zero GPU, on a 64-core node. This is stated rather than rounded
down; a reviewer who disagrees with the trade should say so.

---

## 8. Compute projection — measured unit cost × explicit count, multiplication shown

Every unit below is measured in §7.2/§7.3 at real scale on the real banked data. **No
figure is extrapolated from a reduced-scale run.**

### Phase 1 — head mints (the dominant loop), 36 units

| unit | count | measured unit | product |
|---|---|---|---|
| HateMM fitting-pool heads | 3 seeds × 5 folds = **15** | 40.39 s | `15 × 40.39 = 605.9 s` |
| HateMM full-train heads | 3 seeds × 1 = **3** | 49.30 s | `3 × 49.30 = 147.9 s` |
| MHC-ZH fitting-pool heads | 3 × 5 = **15** | 34.40 s | `15 × 34.40 = 516.0 s` |
| MHC-ZH full-train heads | 3 × 1 = **3** | 38.87 s | `3 × 38.87 = 116.6 s` |
| | **36** | | **1386.4 s** |

### Phase 1b — key extraction inside each mint

5 train forwards per mint (native + {std, ow} × {L24, L28}); the 6 full-train mints add 5
dev forwards, bounded above by the train unit.
`(30 × 5) + (6 × 10) = 210 forwards × 0.0461 s = **9.7 s**`

### Phase 2 — arm evaluation. Fold-cells = 5 folds × 3 seeds × 2 datasets = **30**

| | count | unit | product |
|---|---|---|---|
| 1024-d arms (`endpoint_std`, `endpoint_ow`, `common`, `displacement`) | `4 × 30 × 2 layers = 240` | 0.0042 s | `1.0 s` |
| 2048-d arms (`endpoint_concat`, `common_displacement`, `common_interaction`, 6 × `orthrot`) | `9 × 30 × 2 = 540` | 0.0218 s | `11.8 s` |
| `GATE-FLOOR` native vote | `1 × 30 = 30` | 0.0042 s | `0.1 s` |
| `avg_score` (derived from banked endpoint scores, no new vote) | — | — | `0 s` |
| | | | **12.9 s** |

### Phase 3 — shuffled-pair null, C01 frozen `n_id_hash_permutations = 256`

`256 draws × 3 seeds × 2 datasets × 2 layers = **3072** draw-units × 0.1241 s = **381.2 s**`
(the `U4` unit was measured at HateMM scale and applied to MHC-ZH too, which over-states
the smaller dataset — deliberately, in the conservative direction).

### Phase 4 — bootstrap, C01 frozen `n_bootstrap = 2000`

`2 real arms × 11 comparators (5 ordinary controls + 6 rotations) = 22 comparisons`
`22 × 3 seeds × 2 datasets × 2 layers = **264** comparison-cells × 0.126 s = **33.3 s**`

### Phase 5 — null-row sensitivity leg (HateMM only, `n = 743`, **no re-mint**)

Re-runs phases 2 + 3 + 4 restricted to HateMM:
`(12.9 + 381.2 + 33.3) / 2 = **213.7 s**`

### Phase 6 — `GATE-DEVFID`, `headspace_fidelity.py` × 2 datasets

C09 measured this at *"seconds"*; budgeted **30 s**.

### Phase 7 — sha256 of 8 caches + 11 modules, ledger aggregation, JSON emit

The caches total ≈ 152 MB; hashing is sub-second. Budgeted **60 s** as slack.

### Total

| phase | seconds |
|---|---|
| 1 mints | 1386.4 |
| 1b key extraction | 9.7 |
| 2 arm evaluation | 12.9 |
| 3 shuffled-pair null | 381.2 |
| 4 bootstrap | 33.3 |
| 5 null-row leg | 213.7 |
| 6 `GATE-DEVFID` | 30.0 |
| 7 hashing / IO / emit | 60.0 |
| **corroborating total** | **2127.2 s = 35.5 min** |
| **conservative total (× 1.25 for shared-node contention)** | **2659.0 s = 44.3 min** |

**Projected peak RSS ≈ 1.3 GiB** (measured 1.25 GiB for the dominant unit; the arm
matrices add `744 × 2048 × 8 B = 12 MB` each, ≤ 320 MB if every arm at both layers is held
simultaneously). Request 32 GB, as C02 and C09 did.

**Both totals are under one hour.** The two figures follow the C04 v8 pre-submit
projection-gate pattern (F116/F117: 3817.6 s conservative / 2597.8 s corroborating,
realized 2668 s — 1.7 % off), extended to a CPU job as `rule_1_compute_projection`
requires.

**Honest bound on the projection.** Its weakest element is Phase 3's `U4`, measured on
HateMM-scale keys with a `float64` rebuild; if the implementation stores the permuted arms
differently it could drift. Phase 3 is 17.9 % of the corroborating total, so a 2× miss
there moves the total to `2127.2 + 381.2 = 2508.4 s ≈ 42 min` and a 5× miss to
`2127.2 + 4 × 381.2 = 3652.0 s ≈ 61 min` — still the right order, which is the property
C09's projection lacked. **If the realized cost exceeds the conservative total by
more than 2×, that is itself a reportable process finding.**

---

## 9. Heartbeat specification (`rule_2_heartbeat`)

* A single progress file, `$BASE/progress/C06_PROGRESS.txt`, created by the sbatch
  driver before the first python process starts.
* **Every** python process of the job appends to it through a handle opened with
  `buffering=1` (line-buffered) — *not* a block-buffered stream flushed at exit, which
  `rule_2_heartbeat` explicitly rejects.
* One line per unit of the dominant loop and per phase transition, each carrying:
  ISO-8601 timestamp · phase name · units done / units total · elapsed seconds ·
  elapsed ÷ projected ratio.
* Emission granularity, chosen so no interval can exceed ~60 s:
  **one line per mint** (36 lines, ≤ 50 s apart), **one line per `(dataset, seed, layer)`
  arm block** (12), **one line per 32 null draws** (96), one per bootstrap block, one per
  gate evaluated, one per verdict field written.
* The **bash driver also** echoes a line per mint independently of python, so a phase
  boundary is visible even if a python process wedges before its first flush.
* An external observer can therefore distinguish progress from a hang **without attaching
  to the process** — the property whose absence produced F118's 51.6-hour blackout.

---

## 10. Scope of any verdict

### 10.1 The prompt/readout-span confound — declared, as the gate requires

`src/utils/generate_VideoMLLM_embedding_readout_HF.py:73-89` defines the `ow_` cells as
`("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID)`: the one-word cells
change the **readout span as well as the prompt**. The two orbit endpoints therefore
differ in *two* respects, not one. **No result of this battery can attribute an effect —
or the absence of one — to the prompt alone.** This is the same confound C01's own review
narrowed its claim for, and it is stated here, on the face of the verdict, because
`falsifier_design_constraints` (ii) requires it.

### 10.2 What a CLOSE would and would not close

A CLOSE closes **C06's first-order (tangent/chord) prompt-orbit route in the deployed head
space, under a head trained on the native deployed cache, at `ro_L24`, on
`HateMM (-LoRA-curric)` and `MHC_zh (-LoRA)`.** It executes the state file's rule as
written. It does **not** establish:

* anything about **curvature** — two banked prompt points give a chord; ≥ 3 points require
  extraction and are `$0`-impossible (A7);
* anything about a head **retrained** on orbit-geometry features — F66's caveat stands
  (A4, §3.3);
* anything about a **different readout span** or a prompt axis without the §10.1 confound;
* anything about the **test split** — no test artifact is opened at any point.

### 10.3 What a SURVIVE would license

Only what the gate says: that C06 *"has earned its extraction"* — i.e. the `1.7–2.5 GPU-h`
bounded extraction may be **proposed**, under
`iteration_8_stage0_bounded_extraction_amendment`, with its own preregistration, its own
independent design review, its own separate code/resource review lineage and its own
main-dialogue authorization. **A SURVIVE is not a Stage-0 PASS and authorizes no GPU by
itself.**

### 10.4 Bans checked, and why none reaches this

F80's object is prompt **language** (and its unconditional leg is scoped to `MHC_zh`);
F70's object is individual **readout cells**. Neither object is orbit geometry, which is
why the Gate-0 record gated C06 rather than striking it. The multi-prompt **ensembling**
carve-outs in both bans have C14 as their object, not C06, and are not relied on here.
This battery constructs **one trained representation per arm evaluation** and forms **no
ensemble of prompt predictions** — the C06 dedup boundary, respected by construction.

---

## 11. Frozen imports and inputs — sha256, measured 2026-08-04

**Imported unmodified, sha256 asserted at run time:**

| module | sha256 |
|---|---|
| `scripts/analysis/headspace_mint.py` | `cefdf8dc2f4a9aefa042ef7bec9b1d06c9721ae5b4a70ec117e9929ff0916612` |
| `scripts/analysis/mechfix_ops.py` | `635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d` |
| `scripts/analysis/mechnov_pairverify.py` | `77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d` |
| `scripts/analysis/headspace_fidelity.py` | `72fd8e0aab61b635b4421b87bdbccc8ef6c58bf28fe1ff64cab0671e08bf6598` |
| `scripts/analysis/c09_guard/c09guard.py` | `aed50842c232105f1b06182aa89512ee89dd050bdcaedec2706062c9d745f062` |
| `scripts/analysis/c09_guard/sitecustomize.py` | `b238789fd80076b0b890c4894fd8b69255792af51c80cd9fe2d6db6c53383850` |

**Read for definitions and thresholds, not imported as code:**

| file | sha256 | role |
|---|---|---|
| `configs/c01/c01_a0_v2.json` | `f3997bddb4788d451ae5f90d9d03d096df3de383f8133a6d3818d97a241563f5` | every arm, angle and threshold in §5 |
| `configs/c01/c01_a0_v3.json` | `4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5` | lineage |
| `scripts/analysis/c01_policy_contrast_a0.py` | `d2b9c2ff909c07518ae35526db9550df655fb4af395cc7a0899f83e48db1b855` | `orthogonal_blocks` / `paired_key` / `contrast_blocks` algebra |
| `scripts/analysis/c02_a0_mint.py` | `e6430b76b7ccdd831ddb9939500aa24ea70d9662b62b955a2a11273a3b00ac1b` | the thin-wrapper precedent (§3.3) |

**Input caches — `train` split only:**

| file | sha256 | leg |
|---|---|---|
| `HateMM/train_…-LoRA-curric_HF-ro_L24.pt` | `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f` | primary |
| `HateMM/train_…-LoRA-curric_HF-ro_ow_L24.pt` | `60054f3be1204ca7bf2ac55b9bae6a88dd84d9dda35b0225f1ca27ce61977f4e` | primary |
| `HateMM/train_…-LoRA-curric_HF-ro_L28.pt` | `d5a6796733f1fd56fc254a3e1c7f428541772a1a7b9be40f53013bc65dbd9ef1` | L28 (non-decisional) |
| `HateMM/train_…-LoRA-curric_HF-ro_ow_L28.pt` | `1747cb2750e757cfc0a472a8089457e452574d45f9c00debe1b5a86d561dd13f` | L28 (non-decisional) |
| `MHC_zh/train_…-LoRA_HF-ro_L24.pt` | `1d33fe5d69083479f0b6968a924578770364c00ca78c37cfef664bb4b6221c06` | primary |
| `MHC_zh/train_…-LoRA_HF-ro_ow_L24.pt` | `3ad1309dc75001820318e3e9a073b781d28ee0afc2b879571d063a276b8d2a23` | primary |
| `MHC_zh/train_…-LoRA_HF-ro_L28.pt` | `cf1dba903d22b96e106cc6033559258f868e33121454b208612d5eb58ca19009` | L28 (non-decisional) |
| `MHC_zh/train_…-LoRA_HF-ro_ow_L28.pt` | `203d2860fa62efd60f039c0b0014564d02c6a39e59b491aa1b67b137f510858c` | L28 (non-decisional) |

Plus the six banked `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`
(read for `GATE-FLOOR` anchors only) and the ten banked
`scripts/analysis/vsw_ckpt/{hatemm,zh}/f{0..4}.npz` (fold parity).

**New code, confined to the battery:** `scripts/analysis/c06_falsifier_mint.py` (the thin
wrapper, modelled on `c02_a0_mint.py`), `scripts/analysis/c06_falsifier_arena.py`,
`configs/c06/c06_falsifier.json`, `scripts/slurm/c06_falsifier_cpu.sbatch`. Nothing in the
frozen list above is edited, forked or copied.

---

## 12. Label and split discipline

* **Test contact: none, enforced in three independent layers.** (1) `headspace_mint.py`
  replaces `load_feats_from_CLIP` wholesale so only `train_*.pt` and `dev_seen_*.pt` can
  be opened, and installs a `torch.load` guard that raises on any path containing
  `test_seen` / `/test`. (2) The wrapper asserts `split == "train"` on every ro-cache load
  and refuses any non-`train_` path, exactly as `c02_a0_mint.py:64-66` does. (3) The
  frozen `c09_guard` `sitecustomize` is placed on `PYTHONPATH` by the sbatch driver, so an
  `open()`-level, component-wise, repo-scoped test predicate is installed at interpreter
  startup in **every** process of the job — mints, fidelity and arena alike — and each
  process appends its measured counts to a shared ledger directory.
  **`GATE-LEDGER` requires `test_path_opens == 0` and all expected processes reporting;
  the predicate's coverage is re-derived in-job rather than asserted in a comment.**
  The guard is imported **unmodified**, which means its ledger environment variable
  remains `C09_LEDGER_DIR` — renaming it would break the sha256 pin, so the name is
  retained deliberately and noted here so no reviewer reads it as a copy-paste slip.
* **Labels.** Train-split labels only. Every query's label is held out from the head that
  judges it. `dev_seen` labels are materialised only inside `headspace_fidelity.py`, which
  is reporting-only and gates nothing.
* **No selection anywhere.** No arm, angle, layer, seed, threshold or `k` is chosen by
  looking at a result. Every threshold in §5 is either C01's frozen value or fixed in §4
  before measurement.

---

## 13. Execution boundary

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** Identical envelope to C02 A0 (job `13847`, `00:29:49`) and C09
A0 (job `13885`).

**Why not a login-node `nohup`.** The brief asked whether F88's `$0` forensics establish
that precedent. **They do not.** `ERRPAT_HateMM_2026-07-26.md` §0.1 describes re-running
the command from `scripts/slurm/enc3seed_lora_curric.sbatch` CPU-side and prices it at
*"52 s wall per seed on 8 CPUs"*, naming no non-SLURM channel — and a 52-second process is
not a precedent for a 44-minute one. Against that, CLAUDE.md's standing rule
(*"所有 GPU / 计算任务必须通过 SLURM 提交…非 SLURM 的计算进程会被回收"*), C01's own frozen
`execution.require_slurm = true, cpu_only = true, required_cpus = 8, required_memory =
32G`, and the C02/C09 A0 precedents all point one way. **SLURM.**

**Queue routing.** CLAUDE.md's 2026-07-31 ruling sends lightweight work to the cloud when
the queue is crowded. It does not apply: (a) `squeue` must be read at submission time, not
now; (b) the cloud route would require the same-table-same-hardware precondition, and
`GATE-FLOOR` anchors this battery to six floors measured **locally** on
`foscsmlprd01` — re-running them on cloud hardware would cost more than the whole job;
(c) at 44 min this is not long-running. **Local SLURM CPU.**

**Not authorized by this document.** No prereg is frozen, no hash is frozen, no job is
submitted. Before anything runs, this design needs an independent design review to GO
(0C/0H/0I), then a **separate** code/resource review lineage over the battery script
(`feedback-separate-code-review-lineage` — on C09 that lineage ran 7 rounds after 17 clean
design rounds and caught two wrong-verdict paths), then main-dialogue authorization.

---

## 14. Open issues the reviewer must rule on

1. **The direction of "conservative" (§4).** Resolved as *hardest to deliver the closure*.
   A reviewer may reasonably hold that a gated candidate's default is death and the bar
   should be easier to close. The rule is pre-registered either way; what changes is A2's
   disjunction and possibly S3's `0.02`.
2. **`GATE-ORBITSCALE`'s threshold (§6, §7.4).** It borrows C01's two frozen constants
   rather than inventing one. At an untrained head the measured margin is ~3.2× on the
   median. Is `max_tiny_displacement_fraction = 0.05` the right bar in a space where the
   *typical* displacement is 219× smaller than in the space that constant was chosen for?
3. **A4's excluded reading.** Per-arm head retraining is C06's best shot and costs ≈ 4.4 h
   of CPU rather than 44 min. It is excluded on three grounds (§3.3). A reviewer who thinks
   a `$0` closure must survive C06's best shot should say so **now**, before freeze — the
   projection for that variant is `195 mints × 40.39 s + 195 × 34.40 s = 14 584 s = 4.05 h`
   of mints, plus this design's phases 1b–7 (≈ 741 s) and a proportionally larger
   `GATE-FLOOR` burden, ≈ **4.4 h** all in. It remains `$0` and GPU-free; the objection to
   it is cost and instrument-anchoring, not legality.
4. **The L28 leg (§5.6).** Declared non-decisional. A reviewer may prefer it dropped
   entirely rather than run-and-reported, to remove any chance of post-hoc leaning on it.

---

*No GPU, SLURM, Modal, teacher call, model load, training of any deployed arm, cache
write, test-split access, job submission or commit occurred in producing this document.
Login-node dry-check processes only, itemised in §7, total ≈ 3.5 wall-minutes.
`TARGET_STATE.json` was read and not modified.*
