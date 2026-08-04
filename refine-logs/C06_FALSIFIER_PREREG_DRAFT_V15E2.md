# C06 `$0` CPU falsifier — preregistration **DRAFT v15 + ERRATUM 1 + ERRATUM 2** (2026-08-05)

**THIS IS v15 PLUS ERRATUM 1, THE CODE-R1 §8 CORRECTION AND ERRATUM 2, AND NOTHING ELSE.**
ERRATUM 2 landed under seven independent adjudications; its obligations checklist is
`refine-logs/C06_FALSIFIER_ERRATUM2_LANDED.md` and its final review is
`refine-logs/C06_FALSIFIER_ERRATUM2_REVIEW_R7.md`. `C06_FALSIFIER_PREREG_DRAFT_V15E1.md`
(`8cde58aa…`) stays byte-unmodified. `C06_FALSIFIER_PREREG_DRAFT_V15.md` remains on
disk **unmodified** as the GO'd record (sha256 `75e3aa84a39f3c276be8b5b7fb1271e83c0c2e7e3b0ad0480b47465a81408228`).
Erratum 1 was raised by the implementation lineage, proposed at
`C06_FALSIFIER_ERRATUM1_PROPOSAL.md`, adjudicated at `C06_FALSIFIER_ERRATUM1_REVIEW.md`
(**REVISE — 1C/1H/2I/2M**, endorsing option (i) under **six binding obligations**), and landed at
`C06_FALSIFIER_ERRATUM1_LANDED.md`. **The defect:** §5.4 pre-registered a per-item accuracy
decomposition as *the* bootstrap statistic while S4 ranges over `holm_metrics = [accuracy,
macro_f1]` and §5.5 counts both — so 46 macro-F1 hypotheses per dataset had **no statistic**.
**The repair, in one line:** the accuracy leg's expression is retained **verbatim**, and a macro-F1
bullet is **added** defining C01's own recompute-per-resample form. Sections touched: **§5.4**
(one added bullet + one added sentence), **§5.9** (one added item 10), **§7.7** (`U3` retired and
two units registered), **§8** (Phase 4 re-priced and the totals re-derived). **§5.2, §5.5, §5.6,
§5.8, §6, §13 are untouched** — the S4 scope, the 92-family, the witness floor of 22/24, the
verdict combination, the CLOSE-attribution list, every gate and the whole handoff are exactly as
GO'd.

---

## v15's own header, retained

**SUPERSESSION.** Supersedes `C06_FALSIFIER_PREREG_DRAFT_V14.md` → V13 → V12 → V11 → V10 → V9 → V8 →
V7 → V6 → V5 → V4 → V3 → V2 → v1. All fourteen remain on disk **unmodified** as the record of what
each round reviewed. Reviews of record: `C06_FALSIFIER_PREREG_REVIEW{,_R2,…,_R14}.md` — REVISE
3C/6H/10I+4M, 3C/3H/7I+3M, 2C/1H/6I+4M, 3C/3H/8I+4M, 3C/3H/6I+5M, 2C/3H/5I+6M, 4C/2H/3I+4M,
2C/2H/4I+6M, **0C**/2H/2I+4M, **0C/0H**/3I+4M, **0C/0H/1I**+1M, **0C/0H/1I**+3M, **0C/0H/1I**+2M,
**0C/0H/1I**+2M. Complete standalone document.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is frozen,
`TARGET_STATE.json` is untouched, nothing is committed.

**Disposition: 3 round-14 findings, dispositioned at LIMB level — every limb quoted VERBATIM AND IN
FULL from round 14's review, with the line numbers of the sentence it was taken from. No blanket
adoption claim is made anywhere in this document (§14, §14.1).**

**Round 14 discharged round-13 I-1 and ruled the widening that discharged it warranted.** It built
the seven rungs itself and ran **56 timed starts in one command, 95.7 seconds of wall** against this
document's *"about 96 seconds"*; every run-count reconciles under its own arithmetic (`7 × 8 = 56`,
`8 + 3 + 24 = 35`, `32 + 20 = 52`, cumulative `12 / 36 / 136`), and **both pooled endpoints are
exact against the source reviews** at `R12:349` and `R13:237`. Its own rung 7 is `3.070–3.540 s`,
bounded by `3.8 s` with `0.26 s` to spare. Its ruling on the widening: **warranted**, and *"I would
have done the same"* — decisively because the re-measurement **pooled with** rounds 12 and 13 rather
than replacing them: *"A designer preferring its own instrument would have dropped the other two
rows."* It also ruled correcting v13's recorded spend **right**, and *"the campaign's discipline, not
an unusual step."*

**Everything else came back clean and re-derived rather than inherited:** 13/13 arms rebuilt from
§3.4's prose alone at `0.000e+00` on both datasets, `GATE-ROWSUBSET` at `0.000e+00`, 26/26 `ρ_raw` at
6 dp, trained-head `0/18` on row-renormalised keys, 16/16 C01 accuracies and 16/16 net-fix integers,
37/37 digests, the Holm counterexample **and its three-way equality** through C01's own `holm_adjust`,
all **23** §8 unit×count products re-multiplied with zero mismatches, all twenty gates unable to fire
on a warranted CLOSE, and 6/6 limbs verbatim, complete and inside their cited ranges. **Zero stale
totals.**

**What round 14 found, and it is the reason this version exists.** Not a number and not a count — a
**false factual claim about this document's own verification mechanism, refuted by its own embedded
transcript two paragraphs away**:

* §14.1 said *"The same vacuity holds for v14, whose limbs land elsewhere"* and §15 item 5 told
  round 15 that *"no row or limb cites §14.1"*. **Both false.** Round-13 M-2 lands **in** §14.1;
  v14's M-2 disposition row and M-2 limb both cite it; and the transcript prints
  `OK M-2 cites §14.1`.
* Round 14 ran the plain splice: it **exits `1`**, failing the M-2 row and the M-2 limb and reporting
  `named by a row but unchanged: ['14.1']`. **The mechanism is stronger than v14 claimed for it**,
  and the claim erred in the direction that understates the artifact.
* **The sentence was an unchecked inheritance from v12 and v13 — the precise defect round-13 M-2
  named — sitting in the paragraph that lands round-13 M-2's repair.**

**What v15 changes.**

* **I-1** — §14.1's vacuity sentence and §15 item 5 are replaced by what is **measured against this
  document as finalized**, in the ordering round 14's repair requires and §14.1 now states: the
  §14.2 fixed point first, then the splice, then the sentence. §15 item 5 now asks round 15 to run
  **both** forms and report both.
* **M-1** — round 13's framing sentence is quoted **in full** wherever it is quoted, *"Repair — one
  line, arithmetic only, no new measurement"*, so v14's widening is declared against **both** of its
  clauses: the method **and** the length.
* **M-2** — §7.9 now states the bridge from v13's printed `44` to its executed `52`: an eighth rung
  went unreported (`+4`) and rung 5's `(10 runs)` was an increment where rung 7's `(14 runs)` was a
  total (`+4`). `44 + 4 + 4 = 52`, checkable against v13's own table.
* **Recorded, not prescribed** — §13.1 gains **item 28**, carrying round 14's note to the separate
  code lineage that §12's three-layer test guard depends on the C06 sbatch exporting `PYTHONPATH` to
  reach `c09_guard`, whose `sitecustomize.py` docstring still names C09's sbatch.

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
| `common` | `0.8692` / `+3` | `0.8718` / `+1` |
| `endpoint_concat` | `0.8598` / `+2` | `0.8846` / `+2` |
| best rotation `orthrot_83p8` | `0.8692` / `+3` | `0.8974` / `+3` |
| `orthrot_72p7` | `0.8505` / `+1` | `0.8974` / `+3` |

`gain_over_strongest_control` **`−0.0093`** / `−0.0256`; `pass: false`; `decision.continue = false`.
**This table is load-bearing twice over** — it is the evidence the gate rests on, and it is
what refutes `GATE-ARMVIAB`'s escape branch (§6.2).

*(Round-5 M-1, adopted with an erratum. Earlier drafts printed `−0.0094`, inherited verbatim from
`TARGET_STATE.json::…rotation_family_precision_R14` and `GATE0_REOPEN_2026-07-31.md:756`. The
executed `C01_A0_OUT.json` stores `-0.009345794392523366`, which rounds to **`−0.0093`**; the
MHC-ZH figure `−0.0256` is exact (`-0.02564102564102566`). Nothing downstream moves — `pass: false`
either way and the figure enters no decision — but this document claims the table is recomputed
from the stored confusion matrices, and this one number was not. Corrected here under the
campaign's numeric-provenance discipline.)*

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
`dev_seen` **unconditionally on every mint**, before the `fold` branch, and `:322-324` writes `lab_dev` (the array is at `:323`)
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
arms, both datasets) — reproduced independently from this prose by rounds **2, 3, 4, 5 and 6**.

**The arm→formula map is pinned HERE, because the prose above does not determine it (round-6 C-1).**
Two of six independent reviewers rebuilt the battery from this section and both derived the *same*
arm wrongly: `common_interaction`. It is **not** `l2(std ⊙ ow)`; the correct definition, from
`contrast_blocks` (`c01_policy_contrast_a0.py:1246-1265`), is

> **`std[m]` and `ow[m]` denote the `l2_rows`-NORMALISED endpoint blocks.**
> `prepare_views:1296-1304` normalises **every policy/modality block before `contrast_blocks` is
> called**, so the contrast definitions below are on the normalised blocks:
> `std[m] := l2_rows(standard[m])`, `ow[m] := l2_rows(oneword[m])`. Then
> `common[m] = l2(std[m] + ow[m])`, `displacement[m] = l2(ow[m] − std[m])`,
> **`common_interaction[m] = l2(common[m] ⊙ displacement[m])`**, and the arm is
> `paired(common, common_interaction)` — *not* `fuse(common_interaction)` and *not* a product of the
> raw endpoints.

**The pre-normalisation step is stated because omitting it is a wrong-verdict path the anchor gate
did not catch (round-7 C-1).** v7's pin defined neither `std[m]` nor `ow[m]`, so its literal reading
takes the *input* blocks. Measured, that reading differs from `prepare_views` by **`1.878e-06`**
(HateMM) and **`1.609e-06`** (MHC-ZH) — the ro caches are unit-norm to `1.79e-07`, so raw space
nearly forgives it — and **both figures PASS the `2e-6` tolerance v7's `GATE-C01PARITY` row also
stated**, with 6 % of margin. In **head** space the head emits an unnormalised 1024-d vector
(`classifier.py:146`), and round 7 measured the same misreading moving every contrast arm by
`1.9e-02`–`1.3e-01` — both real arms and C01's primary — while the endpoint arms stay bit-identical,
the `θ = 0`/`θ = 45` identities still hold to `2.2e-08`, and `GATE-ALGEBRA`, `GATE-ZEROOP`,
`GATE-ARENA` and `GATE-ORBITDISP` all pass. **`GATE-C01PARITY` is therefore a single predicate —
bit-exact — and the `2e-6` clause is struck from it** (§6).

Round 6 measured the cost of getting it wrong: `max|diff| = 9.697e-01` across the 13 arms, versus
`0.000e+00` once corrected. **The full map is therefore pinned by `GATE-C01PARITY` against
`prepare_views` and by nothing else** — a code lineage that reimplements from this document rather
than testing against `prepare_views` will produce a wrong arm and a build that still looks like it
passes. §13 **item 23** carries the *"test against `prepare_views`, do not reimplement"* instruction and
names this arm as the concrete instance; **item 19** carries the one-construction claim it rests on
(round-7 M-2 corrects v7's reversed attribution). One
block (the fused head key) ⇒ the head-space arms, whose dimensions are **`4 × 1024-d` and
`9 × 2048-d`** (round-4 measurement γ, the first independent confirmation of §8 Phase 2's
`240 / 540` split).

**What the anchor buys.** Two-block parity pins `l2_rows`, concatenation order, the contrast
definitions, the Givens mixing and its angle convention, the arm-name→formula map, the `float32`
dtype and the θ=0/θ=45 identities. It does **not** pin the outer `fuse` normalisation at one
block, which re-normalises an already-unit vector: **what round 1's C-3 actually fixed for the
head-space arms is the dtype.** The block count is **forced by the head's architecture** —
`classifier_hateClipper` fuses internally at `classifier.py:140-141`
(`x = torch.mul(img_feats, text_feats)` under `fusion_mode == 'align'`) and emits a single fused
1024-d vector at `:146` (`embed = self.mlp[:-2](x)`). Round-5 M-2: v5 cited `:116-120`, which is
the **two-stream** projection-and-normalise block — the opposite of a fused vector. The claim was
true; the citation was not.

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
at **`src/model/classifier.py:81-82`** (`:80` is the comment line), so `head_f(0,0)`
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

**Population-derived constants — computed on the arena, never read.** Round 3 believed the majority
rate was the only one; round 4 found two more. Every constant in **this** block is frozen here,
computed on the **arena** population, and checked at run time rather than read (§13 item 5). *(A
second block below carries frozen C01 config constants, which are **read** and asserted equal —
round-8 H-2 found v8 quantifying the "computed" verb over both.)*

| constant | HateMM | MHC-ZH | consumed by |
|---|---|---|---|
| arena size `n_D` | **743** | **579** | `GATE-SELFTEST`, S6, the tie cap |
| arena class counts | **(297 pos, 446 neg)** | **(180, 399)** | `GATE-POP` |
| **arena majority** | **`446/743 = 0.600269 → 0.6003`** | **`399/579 = 0.689119 → 0.6891`** | `GATE-ARENA`'s lower bound |
| `GATE-ARENA` band | **`[0.6203, 0.98]`** | **`[0.7091, 0.98]`** | `GATE-ARENA` |
| **small-displacement quantile** (round-4 I-3) | `0.1` quantile of the **743** displacement norms | of the **579** | **S7** (§5.2) |
| **`GATE-DOMAIN` majorities** (round-4 I-2) | `maj_arena = 0.6003` with `acc_ro`; `maj_full = 0.5995` with the banked `acc_native` | `0.6891` for both (arena = full) | `GATE-DOMAIN` |
| tie cap | `⌊0.01 × 743⌋ = 7` | `⌊0.01 × 579⌋ = 5` | `GATE-ZEROOP` |

**Frozen C01 config constants — READ from the sha-gated config and asserted equal, NOT recomputed.**
Round-8 H-2: v8 placed these three under the population-derived preamble, which is false of all
three — they are `c01_a0_v2.json::transforms` values — and made §13 item 5's *"computed from the
arena"* instruction impossible to satisfy for the `<=` operator. They are their own block, with
their own verb:

| constant | value | source | consumed by |
|---|---|---|---|
| **small-set comparison operator** | **`<=`** | `displacement_audit:2036` | **S7** (§5.2.2) |
| **`tiny_displacement_epsilon`** | `0.001` | `c01_a0_v2.json::transforms` | **§5.2.3** — the bar `tiny_ok` is *not* carried against |
| **`max_tiny_displacement_fraction`** | `0.05` | `c01_a0_v2.json::transforms` | **§5.2.3** |
| **`normalization_epsilon`** (round-8 I-3) | **`1e-12`** | `c01_a0_v2.json::transforms` | **every `l2_rows` call in BOTH spaces.** The two-block build inherits it through `prepare_views`, which reads the config; **the one-block head-space build does not go through `prepare_views` and must be passed it explicitly.** `GATE-C01PARITY` cannot pin it — it compares outputs, which are epsilon-independent unless a row `die()`s — so §13 item 8 carries it. Registered nowhere in v1–v8 |

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
scored against **train-split** labels held out from the head that judged them. `acc_s(A,D,L)` is the
**per-seed** OOF accuracy, `acc(A,D,L)` its **mean over the three seeds** (round-5 I-1: v5 used
only `acc`, then asserted a per-seed identity with it), and `net_s(A)` the **per-seed integer**
net-fix count against the **reference arm of §5.2**.

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
| **S4** | for every comparator in `C ∪ Θ`: bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05`, with the statistic pre-registered at **§5.4** | `minimum_bootstrap_lower_bound`, `require_primary_bootstrap_holm_reject`, `require_rotation_bootstrap_holm_reject`, `n_bootstrap = 2000`, `statistics.seed = 20260728`, `statistics.holm_alpha = 0.05`, `statistics.holm_metrics` |
| **S5** | **both** real arms exceed the 95th percentile of their shuffled-pair null, **and** the shuffle comparison Holm-rejects | `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject` |
| **S6** | **`net_s(A) ≥ 3` (HateMM) / `≥ 2` (MHC-ZH) in 3/3 seeds**, against the **reference arm** of §5.2.1 | `minimum_net_fixes`; reference per `c01_policy_contrast_a0.py:2702-2714` |
| **S7** | **no small-displacement dominance** for `common_displacement`: with the small set defined by the `0.1` quantile of the head-space per-item displacement norms on the **arena**, the fraction of the arm's fixes falling in that set must not exceed **`0.5`**; per seed, in 3/3 seeds | `small_displacement_train_quantile = 0.1`, **`max_small_displacement_fix_fraction = 0.5`**, `require_no_small_displacement_dominance` |

### 5.2.1 The reference arm — a defect no review round caught (**D-1**)

**v5's S6 named the wrong reference, and rounds 3, 4 and 5 each verified it as correct.** C01
computes `fix_break` at **two** sites and they use **different** references:

* `c01_policy_contrast_a0.py:1725` uses `config["retrieval"]["fix_break_reference"]` =
  `endpoint_std`, and it produces the **reporting** field `fix_break_vs_endpoint_std` emitted for
  every arm;
* `:2702-2714` uses `select_strongest_ordinary_control(evaluations, gain_controls)` and produces
  the **decision** check `net_fixes`, which is the object carrying `minimum_net_fixes` — the
  condition S6 imports.

Read directly from the executed `C01_A0_OUT.json`:
`decision.datasets.HateMM.checks.net_fixes.reference = "common"` and
`decision.datasets.MHC_zh.checks.net_fixes.reference = "endpoint_concat"` — **not** `endpoint_std`
in either case. `endpoint_std` is the reference of the reporting field only.

**Correction, and it runs in the conservative direction.** S6, S7 and `GATE-SELFTEST` all use the
**strongest ordinary control**, selected by C01's own frozen rule:

> `reference(D, L) = argmax` over `decision.gain_controls` of
> `(accuracy, macro_f1, −frozen_order_index)` — `select_strongest_ordinary_control`,
> ranking at `c01_policy_contrast_a0.py:1955-1962` (guards at `:1940-1948`; round-7 M-3) — **evaluated within the `(dataset, lineage)` cell** and
> **recorded on the verdict face**.

This is a *rule*, not a researcher choice: it is deterministic given the run, so it introduces no
degree of freedom and is pre-registration-safe even though the arm it selects is not knowable in
advance. C01's own runs selected `common` (HateMM) and `endpoint_concat` (MHC-ZH).

Three consequences, all improvements:

1. **S6 gets harder.** `acc(strongest control) ≥ acc(endpoint_std)` by construction, so net fixes
   against it are fewer. v5's reference made S6 easier than C01's — anti-conservative.
2. **§5.9 item 4's arithmetic becomes coherent.** S3 is stated against `max_{c∈C} acc(c)`; with S6
   measured against the same arm, `mean_s net_s = n_D · (acc(A) − acc(reference))`, so S3 and S6
   are two statements about **one** quantity — which is what makes the mean-versus-minimum
   counterexample the exact content of S6.
3. **C01's own consistency assertion is inherited.** `:2724` **dies** if the small-displacement
   gate's reference differs from `strongest_control_name`. So **S6 and S7 must use the same
   reference within a cell**, and the battery asserts it. (C01 also runs a second,
   `endpoint_concat`-referenced small-displacement computation whose role is fixed as
   `diagnostic_only`; v6 carries it as a reported diagnostic and it gates nothing.)

### 5.2.2 S7's remaining parameters — frozen (**round-5 C-3**)

v5 pre-registered S7's quantile, population, arm scope and zero-fix convention and stopped. The
rest, all frozen here, none derived from any trained-head number:

* **Dominance threshold `0.5`** — `c01_a0_v2.json::transforms.max_small_displacement_fix_fraction`,
  which appeared **nowhere** in v5. `dominated = fixed > 0 and fixed_fraction > 0.5`
  (`:1993-1996`).
* **Reference** — §5.2.1, shared with S6 and asserted equal.
* **Space and statistic — a declared departure from C01.** C01's small set is the `0.1` quantile of
  `min(‖d_img‖, ‖d_text‖)`, a per-row minimum over **two modality** displacement norms on the raw
  features. **That object does not exist in head space**: the deployed head fuses internally
  (`classifier.py:140-141`, `x = torch.mul(img_feats, text_feats)` under `fusion_mode == 'align'`;
  emit at `:146`) and emits **one** fused 1024-d block, so there is no modality axis and no minimum
  to take. v6 therefore uses the **head-space one-block statistic**
  `d_i = ‖l2(h_ow,i) − l2(h_std,i)‖`, computed **per item under the head that scored it** (its own
  fold's head, matching the OOF discipline `GATE-NESTED` enforces). This is declared as a
  deliberate departure, and it keeps §3.6 true: **the raw leg remains non-decisional.** The
  alternative — reading S7 on the raw features — would have made the raw leg decisional and
  falsified §3.6, and would have tested a displacement that no arm in the head arena is built from.
* **Small-set comparison operator — frozen as `<=`** (round-6 I-4; round-7 C-4 found it frozen
  nowhere in the decision rule). C01 uses `small_mask = dev_min <= threshold` (**`:2036`**; round-8 M-1 corrects v8's `:2049`.
  Round-9 M-1: `:2049` is `"source_rows"`, and `"registered_null_rows_excluded"` is at **`:2050`** —
  the description was inherited verbatim from round 8's own text). Because the
  C06 arena **is** the population the quantile is taken from, `<` and `<=` differ on the boundary
  rows, so this is a **decision parameter of a binding SURVIVE conjunct**. It is frozen here, and
  carried as a row in §3.7's constant table so §13 item 5 resolves.
* **Seed axis** — per seed, required in **3/3** seeds, symmetric with S2 and S6.
* **Zero-fix convention** — `fixed_fraction := 0`, `dominated := false` when `fixed == 0`,
  mirroring `:1989-1996`. **The battery does not inherit it because it never *calls*
  `displacement_audit`** — §11 does import `c01_policy_contrast_a0.py`, which contains that function
  (round-7 M-1 corrects v7's "does not import").

**Two structural differences from C01, recorded rather than hidden.**

*(i) The population.* In C01 the quantile comes from the **train** split and the small mask is
applied to a **disjoint dev** split, so the small fraction on the scored population is a free
quantity. In C06 the arena **is** the train split, so the `0.1` quantile of the arena norms makes
the small set exactly the bottom decile of the scored population by construction. A different test
from C01's, defensible, and recorded.

*(ii) The dispersion (round-6 I-4, round-7 C-4).* The statistic's **shape** also differs, and the
difference runs in S7's favour. In the **raw** space round 6 measured
`min(‖d_img‖, ‖d_text‖)` over the arena rows at HateMM
`min 0.614645 / q₀.₁ 0.647340 / median 0.673298 / max 0.737694` and MHC-ZH
`0.620789 / 0.659981 / 0.686138 / 0.737210` — a full support of only ~20 %, so C01's bottom decile
separated rows differing by about **5 %** in displacement norm and the "small" set was barely small.
In **head** space §7.8 measures `q₀.₁` at `0.044`–`0.069` against medians of `0.18`–`0.23` and maxima
of `0.80`–`1.83` — a **genuinely separated lower tail**, not a narrow band. **S7 therefore tests
something sharper in head space than it did in C01**, which is a strengthening rather than drift,
and it is stated here so no later round reads the difference as an unnoticed change.

**S6 is BINDING (round-4 C-2).** v4 demoted it to *"reported, not screening"* on the strength of a
claimed implication `S3 ⇒ S6`. **That implication is false**, and the demotion deleted a conjunct
that can fail — the anti-conservative direction. See §5.9 item 4 for the corrected statement and
the counterexample.

**S7 is a SURVIVE condition, not a HALT gate (round-4 H-1).** C01 places
`require_no_small_displacement_dominance` in `decision`, and
**`output.decision_schema.required_halt_only_validity_guards`** (seven entries — round-6 M-4: the
list is under `output.decision_schema`, not under `decision`) does **not** contain it. Its **arm scope is
`common_displacement` only**, matching C01's hard-wired `evaluations["common_displacement"]`. Its
full parameter set is frozen in **§5.2.2**.

### 5.2.3 C01's `tiny_ok` limb — not carried, and why (**round-7 C-2**)

**v7 referenced this section from five places and never wrote it.** Round 7 found the pointers in
§7.8, §13 item 25, two §14 rows and §15 item 4, and correctly recorded that
`max_tiny_displacement_fraction` appeared nowhere in the document — so the bar the non-carriage was
being excused against was unregistered. It is written here.

**What C01's condition actually is.** From `displacement_audit:2047-2076`, the final boolean behind
`checks["displacement_stability"]` is

```
tiny_ok    = max_tiny_fraction <= transforms["max_tiny_displacement_fraction"]      # 0.05
final_bool = tiny_ok and (not require_no_small_displacement_dominance or not dominated)
```

where `max_tiny_fraction` is the largest per-modality `fraction(d ≤ tiny_displacement_epsilon)` and
`tiny_displacement_epsilon = 0.001`. **Both constants are frozen here** and both are carried into
§3.7's constant table:

| constant | value | source |
|---|---|---|
| `tiny_displacement_epsilon` | **`0.001`** | `c01_a0_v2.json::transforms` |
| `max_tiny_displacement_fraction` | **`0.05`** | `c01_a0_v2.json::transforms` |

**S7 carries the `dominated` limb only. `tiny_ok` is NOT carried.** The warrant is measured, not
argued:

* **Raw space** — the fraction at or below `0.001` is `0.0000` in every modality on both datasets
  against a minimum of `0.6146`, and `C01_A0_OUT.json` records `maximum_tiny_fraction = 0.0`.
  `tiny_ok`'s own margin was total — a measured fraction of `0.0` against a `0.05` bar — and the
  **minimum displacement norm** sat `~600×` above the epsilon (`0.6146 / 0.001 = 614.6`). Round-8
  M-5: v8 attached the `~600×` to `tiny_ok` itself, which compares fractions, not norms.
* **Head space** — §7.8 measures `min d_i` at **`0.018`–`0.038`** across four cells spanning both
  lineages and both datasets, i.e. **`18×`–`38×` above the epsilon**, with
  `frac(d_i ≤ 0.001) = 0.0000` in **every** cell. The limb would pass wherever it were applied.

**Direction, disclosed.** Dropping a conjunct that can only make S7 *fail* makes S7 easier, hence
SURVIVE easier, hence **CLOSE harder** — the **conservative** direction under §4. It is listed as
§5.9 item 9.

**Why no magnitude gate is reinstated in its place.** Round-1 C-1 ruled that a **magnitude** gate is
the wrong *instrument* discriminator for a **direction** artifact — the ruling that replaced v1's
`GATE-ORBITSCALE` with `GATE-ORBITDISP`. That ruling stands and is not reopened here.

**What bounds the four-cell evidence.** Four cells of sixty bound nothing formally about a
`min` over `743 × 60` rows. The control is **§13 item 25**, which requires every cell to record its
own `min_i d_i` and `frac(d_i ≤ 0.001)` at run time alongside the `GATE-ALGEBRA` residual, so the
assumption is auditable per cell rather than extrapolated from four.

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
* **Draw sharing:** the resample indices are shared across all comparators **and across both
  lineages** within a dataset, since the two lineages' comparisons live in one Holm family
  (round-5 §15.4's one-clause gap).

**ERRATUM 1 — the macro-F1 leg (added; the bullets above are unchanged).** The *"Per-resample
statistic"* bullet above is a **mean of a per-item quantity** and therefore has an instance for
**accuracy only**. Macro-F1 is a function of four confusion counts and admits no per-item
decomposition, so S4's macro-F1 leg — 46 hypotheses per dataset under §5.5 — had no statistic.
It is defined here, by **C01's own frozen form**, which `paired_bootstrap`
(`c01_policy_contrast_a0.py:1742-1772`) implements and which the executed `C01_A0_OUT.json`
carries for 30 macro-F1 blocks at `n = 2000`:

* **Per-resample statistic, macro-F1 leg:** on draw `b`, using the **same** shared resample indices
  as the accuracy leg, `Δ_b = mean_s mF1(gold[draw_b], pred_{A,s}[draw_b]) − mean_s mF1(gold[draw_b],
  pred_{c,s}[draw_b])`. The metric is **recomputed on each resample** from the resampled predictions
  and the resampled gold, as C01 does; the **seed mean is inside the statistic**, as the accuracy
  bullet requires and for the same reason.
* **The function is `mechfix_ops.macro_f1`** (`scripts/analysis/mechfix_ops.py:56-66`, sha-frozen at
  §11 as `635c1312…`) — **named here rather than only at a call site**, because the two same-named
  candidates differ: measured over all `68,915,480` confusion triples with `tp + fp + fn ≤ 743`,
  `mechfix_ops.macro_f1` and C01's `metric_bundle` are not bit-equal on `39.84 %` of them. This
  battery uses `mechfix_ops.macro_f1` for S1's strict `>`, S3's `≥ 0.02`, `GATE-FLOOR`'s mF1 anchor
  and S5's null, and S4 uses the same function so that one conjunct does not carry a different tie
  convention from its siblings. The direction this choice leans is disclosed at **§5.9 item 10**.
* **Predictions are resampled directly**, which is identical to C01's resample-scores-then-threshold
  because `retrieval.prediction_cutoff = 0.0` matches `deployed_vote`'s `votes >= 0` convention and
  item-wise thresholding commutes with resampling.
* **Degenerate draws.** A class absent from a draw, or never predicted in it, contributes per-class
  F1 `0.0`; **both candidate functions agree exactly there** and **neither returns `None`**. C01's
  `die("bootstrap … produced class-degenerate resamples")` guard (`:1760-1761`) has **no object in
  C06**, because its sole trigger is `roc_auc`'s `None` and `holm_metrics` excludes `roc_auc` — the
  same disposition §5.4.1 records for `shuffle_fixed_point_bijection`. At `n = 743` with 297
  positives a class-degenerate draw is not reachable in practice.

**Why the accuracy bullet is retained verbatim rather than restated in C01's form.** For accuracy
the two are **algebraically identical** — means commute,
`mean_{i∈draw}[mean_s c] = mean_s[mean_{i∈draw} c]` — but they are **not bit-identical**:
re-associating the summation changes the floating-point order, measured at
`max|Δ_frozen − Δ_C01form| = 5.55e-16`. S4's two predicates are the only ulp-sensitive predicates in
this design — a **strict** `lower > 0` and a **zero-count** threshold (`p = 1/2001` demands *zero*
adverse draws) — and on near-identical arm pairs at `n = 743`, `B = 2000` the adjudication measured
**`38.8 %`** of pairs getting a different `one_sided_raw_p` and **`1.2 %`** flipping the `lower > 0`
predicate, with **neither form dominating** on exactly-tied draws (`82.1 %` vs `88.7 %` adverse).
Retaining the frozen expression therefore departs from C01 by **nothing** (same real number) while
restating it would depart from fifteen reviewed rounds by up to one ulp on the two predicates that
decide S4. **The accuracy leg's expression is unchanged, therefore no accuracy quantity moves.**

### 5.4.1 S5's statistic — pre-registered (round-5 H-2)

Round 4 required S4's statistic to be written down; the identical requirement was not carried to
S5, which is also a binding conjunct and, unlike S4, applies to **both** real arms. Frozen here:

* **Null construction:** C01's `id_hash_permutation(ids, dataset, split, draw, seed,
  fixed_indices=())`, `n_id_hash_permutations = 256` draws, `permutation_hash = sha256`. The
  `fixed_indices` argument is **empty**, and this is the point at which C01's
  `shuffle_fixed_point_bijection` / `require_fixed_null_in_shuffle` guard is disposed of: it pins
  the registered null as a permutation fixed point, and in C06's arena **that row is physically
  removed, so the guard has no object** (§3.7). This is the same disposition `GATE-ROWSUBSET`
  records for `displacement_registered_null_exclusion`, now stated for this guard too.
* **Statistic:** for each draw `b`, the one-word endpoint rows are permuted, `displacement` and
  `common_displacement` are rebuilt in head space by the §3.4 builder, and each arm's **seed-mean**
  OOF accuracy and macro-F1 are computed — the seed axis inside the statistic, as in §5.4.
* **p95:** the observed statistic must exceed the `95`th percentile of the 256 draws
  (`bootstrap_upper_quantile = 0.95`), for **both** real arms.
* **p-value:** `p = (1 + #{b : null_b ≥ observed}) / (256 + 1)`, C01's `one_sided_raw_p` form.
* **Holm family:** **`2 real arms × 2 metrics = 4` per `(dataset, lineage)`**, separate from S4's
  family (§5.5 explains why S5's rejections sit outside it).
* **Feasibility, stated before freeze.** The smallest achievable p at 256 draws is
  `1/257 = 0.0038911`, so a shuffle family of `n` members can reject at rank 0 only if
  `n × 0.0038911 ≤ 0.05`, i.e. **`n ≤ 12`**. Measured: `n = 4` needs `0.01556` ✓ and `n = 8` needs
  `0.03113` ✓, while **`n ≥ 13` can never reject** — which would make S5, and therefore SURVIVE,
  unreachable and leave the battery able to publish only CLOSE or HALT. The frozen family of 4 is
  comfortably feasible, and this note exists so no later revision enlarges it past 12 unknowingly.

**Disposition of all seven of C01's `required_halt_only_validity_guards`**, since v5 accounted for
four: `probe_evidence_exact` and `raw_zero_allowlist_exact` → `GATE-ZEROMASK` (feature space);
`derived_zero_masks_exact` → enforced inside `prepare_views` on the only population where it holds
and asserted by `GATE-C01PARITY`; `with_null_remove_null_dtype_shape_bytes_equivalence` and
`displacement_null_exclusion_dual_path_exact` → superseded at the key level by `GATE-ROWSUBSET`;
`registered_null_absent_from_all_top20` → **enforced by construction** (removal) and checked by
`GATE-NULLREMOVED`; `shuffle_fixed_point_bijection` → **has no object**, per the bullet above.

### 5.5 Multiplicity, and its resolution floor

*(**ERRATUM 1 — no edit to this section.** The family stays `92` per dataset, the witness floor
stays `22 or 24` comparators, and the counterexample table below stays valid: the erratum changes
how a macro-F1 `p` is computed, not how many hypotheses exist or how Holm treats them. Stated
explicitly so a reader checks it rather than assumes it. Option (ii) — scoping S4's bootstrap to
accuracy — was rejected precisely because it would have moved all three.)*

**One Holm family per dataset spanning both lineages**, `α = 0.05`: `common_displacement`
6 comparators + 6 rotations = 12; `displacement` 5 + 6 = 11; `(12 + 11) × 2 metrics = 46` per
`(dataset, lineage)`; **× 2 lineages = 92 hypotheses per dataset**. The two **datasets** remain a
conjunction. S5's shuffle rejections are correctly outside the family: they are conjunctive within
each disjunct, so `P(∃ disjunct : all conditions) ≤ P(∃ disjunct : its bootstrap legs all reject)
≤ α`.

**The `B = 2000` resolution consequence, corrected (round-5 H-1).** v5 stated this as *"at least
42 of the 92 comparators must show zero adverse resamples"*. **That is the condition for all 92 to
reject, and S4 does not require that** — S4 is scoped *"for every comparator in `C ∪ Θ`"* of the
**witness** arm on the **witness** lineage: `12 × 2 = 24` hypotheses for `common_displacement`,
`11 × 2 = 22` for `displacement`. Executing C01's `holm_adjust` over the 92-family with the
witness's hypotheses at `1/2001` and the rest at `0.5`: **24/24 reject** (and 22/22 for the other
disjunct); degrade one witness hypothesis to `2/2001` and it becomes 23/24. So the true floor is:

> **every one of the witness's 22 or 24 comparators must show zero adverse resamples out of 2000
> (`p = 1/2001`)** for S4 to pass. This is a property of the frozen `B` and family size, not of the
> data.

v5's figure overstated the design's own statistical bar by roughly 2×, and §15 asked a reviewer to
decide whether `B` should rise on that basis. **`B = 2000` stays** — it is C01's frozen constant,
the true floor is attainable for a genuinely dominant arm, and raising it would depart from the
frozen source and re-price §8 Phase 4 for nothing.

**The family size is frozen at 92 on every path (round-5 C-2b).** v5 left undefined what happens
to a dropped lineage's 46 hypotheses. v6 fixes the family by **preregistration, not by the realised
run**: it is `23 comparisons × 2 metrics × 2 lineages = 92` per dataset **always**, and a dropped
lineage's hypotheses are recorded `NOT_TESTED` and assigned `p = 1`, which makes them
non-rejecting. **The warrant, corrected (round-6 H-1).** v6 claimed the freeze made the bar *"invariant to a drop
… engineering a drop changes S4's difficulty by nothing at all"*, on the measurement that the
witness rejects `24/24` identically under `m = 92` padded `0.5`, under `m = 92` padded `1.0`, and
under `m = 46`. **That three-way equality is real but the generalisation drawn from it was false**,
and its counterexample sits one paragraph above: the equality holds only when every witness
hypothesis is *at* the resolution floor `1/2001`, which is exactly the condition required for S4 to
pass at all. One step off the floor the family size does matter, and round 6 measured it:

| witness p-values | `m = 92` (this design) | `m = 46` (the alternative) |
|---|---|---|
| 24 × `1/2001` | 24/24 — **S4 PASS** | 24/24 — **S4 PASS** |
| 23 × `1/2001` + 1 × `2/2001` | **23/24 — S4 FAIL** | 24/24 — **S4 PASS** |
| 24 × `2/2001` | **0/24 — S4 FAIL** | 24/24 — **S4 PASS** |

`92 × 2/2001 = 0.091954 > 0.05` while `46 × 2/2001 = 0.045977 ≤ 0.05`, and the `B = 2000` grid
straddles the rank-0 bar between `α/46` and `α/92`.

**The freeze at 92 stands, on the honest warrant.** It is kept because **a preregistered family size
that cannot be moved by a realised run is what makes the design auditable**, and because the drop is
caused by instrument gates whose outcome no analyst controls. It is **not** kept because it is
without consequence: it makes S4 strictly no easier than the 46-alternative, therefore SURVIVE no
easier, therefore **CLOSE easier** — the anti-conservative direction under §4, disclosed at §5.9
item 8. Round 5 prescribed the 46-family and this design chooses otherwise; that is recorded as a
**partial rebuttal** in §14, not as an adoption. §8 Phase 4 prices 92 comparison-cells as an **upper
bound** (round-6 M-5: on a drop path only 46 execute, the dropped lineage's hypotheses being
`NOT_TESTED` and therefore uncomputed).

**The two `92`s are different products** (round-3 M-4): §5.5's is `23 comparisons × 2 metrics ×
2 lineages` per dataset; §8 Phase 4's is `23 × 2 datasets × 2 lineages` comparison-cells with both
metrics inside `U3`. They must not be reconciled.

### 5.6 Verdict combination, instrument failure, and per-lineage gate scoping

**Round-4 H-3, adopted in its stronger form.** v4 combined lineages **disjunctively for SURVIVE**
but **conjunctively for HALT**, so an instrument failure confined to Head-N — the lineage the
design itself marks *"in-domain: no"* — would void a verdict Head-R could have delivered cleanly.
v5 scopes the instrument gates:

* **Global gates** (`GATE-DET1`, `GATE-SHA`, `GATE-FOLD`, `GATE-FLOOR`, `GATE-POP`,
  `GATE-C01PARITY`, `GATE-ROWSUBSET`, `GATE-RHORAW`, `GATE-NULLREMOVED`, `GATE-IDPARITY`,
  `GATE-ZEROMASK`, `GATE-LEDGER` — **twelve**) govern provenance, population, algebra and
  bookkeeping shared by both lineages —
  and `GATE-FLOOR` anchors the shared driver. **Any failure HALTs the whole battery.**
* **Per-lineage gates** (`GATE-ARENA`, `GATE-ORBITDISP`, `GATE-NESTED`, `GATE-SELFTEST`,
  `GATE-ZEROOP`, `GATE-ALGEBRA`) are evaluated per **`(dataset, lineage)`** cell — every one of
  them carries per-dataset constants (`GATE-ARENA`'s bands `[0.6203, 0.98]` vs `[0.7091, 0.98]`,
  `ρ*` `0.968176` vs `0.977223`, `n_D` `743` vs `579`, `GATE-ZEROOP`'s cap `7` vs `5`).

**The dataset axis of the drop, named (round-5 C-1).** v5 scoped the drop on the lineage axis
alone and never said whether a failure on **one** dataset drops the lineage on **both**. On the
design's own most likely instrument-failure path — Head-N, the out-of-domain transplant, missing
`GATE-ARENA`'s lower bound on HateMM while clearing it on MHC-ZH — the two readings publish
**HALT** and **CLOSE** on the same event, and a CLOSE is terminal. v6 fixes the axis in the
direction that is conservative on the CLOSE lane:

> **A lineage that fails a per-lineage gate on ANY dataset is marked `INSTRUMENT_FAILED` and is
> dropped on BOTH datasets. A lineage "passed its per-lineage gates" only if it passed every
> per-lineage gate on BOTH datasets.**

**Worked, on the path that motivated it.** Head-N fails `GATE-ARENA`'s lower bound on HateMM only;
Head-R passes everywhere and does not clear S1–S7. Head-N is dropped on both datasets; rule 2's
*"both lineages passed"* is false; **the run HALTs**. It does not CLOSE. That is the intended
outcome: a CLOSE rests on two clean negatives, and a lineage that was clean on only one dataset
supplies neither.

**The combination rule, in the conservative direction:**

1. **SURVIVE** if any lineage that **passed** its per-lineage gates clears S1–S7 on both datasets.
2. **CLOSE** if **both** lineages passed their per-lineage gates and neither clears.
3. **HALT** (`INSTRUMENT_INCONCLUSIVE`) otherwise — i.e. whenever a global gate fails, or a
   lineage is dropped and no surviving lineage clears.

A clean Head-R SURVIVE is therefore not voided by the transplant lineage's failure, and a CLOSE
still requires **two** clean negatives, never one. The lineage(s) that ran, and the dataset(s) on
which each passed its gates, must be named in §10.2's scope sentence.

**Finiteness, absence, and crashes.**

* Every **gate and decision** quantity is asserted **finite and present** before comparison
  (round-4's extension of v4's gate-only clause), and every gate is written in **pass-condition**
  form. The concrete instance the extension covers is S7's `fixed_fraction` at zero fixes, now
  pre-registered in §5.2.
* An **absent** decision or gate quantity HALTs on the same footing as a non-finite one —
  **scoped, per round-5 C-2a, to the lineage(s) that passed their per-lineage gates.** v5 stated
  the absence rule unscoped, which made a dropped lineage's S1–S7 quantities absent decision
  quantities and HALTed on exactly the path the combination rule exists to rescue — reintroducing
  round-4 H-3 inside its own repair. The scoped form:

  > Quantities belonging to a **dropped** lineage are recorded `INSTRUMENT_FAILED`, are **not
  > required**, are excluded from the evaluation of S1–S7 and from the S5 family, and enter the S4
  > family **only** as `NOT_TESTED` with `p = 1` (§5.5). **Absence by declared drop is lawful;
  > absence by computation failure in a surviving lineage still HALTs.**

  The S4 family is *not* shrunk by a drop — it is frozen at 92 with the dropped lineage's
  hypotheses at `p = 1` (§5.5). `GATE-LEDGER`'s process count is binding (§12).
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
4. **S6 binds, and v4's vacuity claim was false (round-4 C-2).** The relationship is an
   **inequality**, not an identity (round-6 I-1): `reference ∈ gain_controls ⊆ C`, so
   `mean_s net_s = n_D · (acc(A) − acc(reference)) ≥ n_D · (acc(A) − max_{c∈C} acc(c)) ≥ 0.02 · n_D`,
   with equality iff the selected reference is also the strongest comparator in `C`. That holds by
   construction for the `displacement` disjunct (whose `C` **is** `gain_controls`) and for the
   `common_displacement` disjunct only when `displacement` is not the strongest of its six — not an
   exotic caveat, since in C01's executed MHC-ZH run `displacement` (`0.8846`) **ties** the selected
   `endpoint_concat` (`0.8846`) at the top. The conclusion is unaffected: S3 still bounds the seed
   **mean** and says nothing about the minimum. S3 is defined on the **seed
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
6. **S7's reclassification eases CLOSE relative to v4 (round-5 I-5).** Under v4, *"S1 fails **and**
   the small-displacement gate fires"* produced `INSTRUMENT_INCONCLUSIVE`; under v5/v6 the same
   state produces **CLOSE**. The design can therefore publish a `$0` closure in a state where v4
   could not. The reclassification is correct on C01's own placement — the condition lives in
   `decision`, not in `required_halt_only_validity_guards` — and the warrant is that a CLOSE is
   what S1's failure warrants, **not** that the change is direction-neutral. §4 binds this design
   to disclose what its lean buys, and this is the largest such item.
7. **S6's reference was corrected upward (D-1, §5.2.1).** Moving from `endpoint_std` to the
   strongest ordinary control makes S6 strictly harder, i.e. it moves against SURVIVE. Disclosed
   for symmetry with §5.9 item 6, since a change in the other direction is disclosed there.
8. **Freezing the S4 family at 92 on every path eases CLOSE (round-6 H-1).** Relative to
   recomputing at 46 on a drop path, the frozen family makes S4 strictly no easier, hence SURVIVE no
   easier, hence CLOSE easier. It is kept for auditability (§5.5), not because it is neutral, and
   it is the direction change round 6 found undisclosed. *(Round-7 M-4: §5.9 item 6 keeps the
   "largest such item" claim; this item drops its own competing superlative.)*
9. **`tiny_ok` is not carried (§5.2.3), which eases S7 and therefore hardens CLOSE** — the
   conservative direction. Listed for completeness, since §5.9 items 6 and 8 record changes the
   other way. *(Round-10 I-2: these three sibling references now carry their `§5.9` prefix, which is
   what makes round-9 M-2's **prescribed** prefix-exclusion mechanism exact — see §14.2.)*

10. **ERRATUM 1's macro-F1 tie convention eases S4, therefore hardens CLOSE — the conservative
    direction, and the mechanism is float association rather than design.** §5.4's macro-F1 leg uses
    `mechfix_ops.macro_f1`, which performs four rounded operations (`pr`, `rc`, then
    `2·pr·rc/(pr+rc)`), where C01's `metric_bundle` performs **one** correctly-rounded division of
    exactly-representable integers. On a resample where two arms have **mathematically equal**
    macro-F1, `metric_bundle` returns `Δ_b` exactly `0.0` — **adverse** — in `100 %` of sampled tied
    pairs, while `mechfix_ops` returns `0.0` or `±1.11e-16` and so escapes the adverse count in
    **`19.3 %`** of them. The bias is **strictly one-directional**: `mechfix_ops`' adverse count is
    always `≤` `metric_bundle`'s, never greater. Because S4's floor requires **zero** adverse draws
    out of 2000, a single tied draw decides a hypothesis, so this makes S4 **easier**, hence SURVIVE
    easier, hence **CLOSE harder** — conservative under §4. It is listed because §4 binds this design
    to disclose what its lean buys, and because the lean here is an artifact of floating-point
    association, not a design judgement. *(Exact ties are not hypothetical: two of the three
    macro-F1 rows quoted from `C01_A0_OUT.json` in the erratum record carry a bootstrap quantile of
    exactly `0.000000`.)*

### 5.10 What this design does not run

**No head-space null-row sensitivity leg**, **no L28 leg**, **no per-arm retrained head**, **no
vote at `n = 744`** (the sole full-`n` vote is `GATE-FLOOR`'s 30 native deployed-key votes, which
score no arm), and **no test-split read of any kind**. No gate, output field or record sentence
describes any of them.

---

## 6. Gates

**Twenty gates** — 12 `G`, 6 `L`, 2 `R`. (Round-5 M-5: v5's header said *eighteen* and its table
had nineteen rows. v6 adds `GATE-RHORAW` under round-5 I-2, making twenty; the count and the table
are re-derived together here so the miscount cannot recur.) `GATE-ARMVIAB` is **retired** (§6.2) and `GATE-SMALLDISP` has become **S7**
(§5.2). Every quantity is asserted finite and present before comparison. Scope column: **G** =
global (HALT the battery), **L** = per-lineage, evaluated per **`(dataset, lineage)`** cell and
dropping that lineage on **both** datasets when it fails on either (§5.6), **R** = reporting
only.

| gate | scope | asserts |
|---|---|---|
| `GATE-DET1` | G | thread env exported before any python starts |
| `GATE-SHA` | G | every frozen import, input cache **and the sixteen banked artifacts of §11** (the six arena OUT JSONs and the ten `vsw_ckpt` npz) matches its §11 digest, **and the design document itself (ERRATUM 2 §3)**; **twice — once in the sbatch driver before any other process, and again in the arena at the point of use, because a digest checked in another process is a TOCTOU claim rather than a guarantee**. Round-8 H-1: v8's §11 asserted this scope and this row did not contain it, leaving `GATE-FLOOR`'s anchor files and `GATE-FOLD`'s parity files unverified. Measured cost of the widening: the sixteen files total `1.2 MB` and hash in `5 ms` |
| `GATE-FOLD` | G | fold parity vs the banked `vsw_ckpt`, discharged **both** on executed mints (`headspace_mint.py:203-216`) **and** by re-reading `meta["fold_parity_vs_banked_vsw_ckpt"]` + `fold_of` from all 66 banked `.npz` — resume-safe (§3.2) |
| `GATE-FLOOR` | G | **Head-N through the shared driver**, native keys, **full `n`**, reproduces the banked floors at 4 dp on **both** metrics — acc HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946`; mF1 HateMM `0.8838/0.8811/0.8812`, ZH `0.8747/0.8710/0.8765`; every `fold_acc_deployed` entry. **Anchors the driver for both lineages**, hence global |
| `GATE-POP` | G | realised populations equal §3.7's table; head leg and raw leg on **identical row index sets**; realised arena class counts equal **`(297,446)` / `(180,399)`**; **and every population-derived constant in §3.7's table is recomputed from the arena, not read** |
| `GATE-C01PARITY` | G | **ONE predicate: `max\|diff\| == 0.0`.** The two-block builder reproduces `prepare_views` **bit-exactly** at `n = 744` one-hot `{355}` (HateMM) and `n = 579` all-False (ZH); **any non-zero residual HALTs.** Round-7 C-1: v7's row stated two criteria at once (*"bit-exactly"* **and** *"HALT above `2e-6`"*), and the `2e-6` reading admits a builder that omits endpoint pre-normalisation — wrong by `10⁻¹` in head space with every other gate passing. `2e-6` is **`GATE-ALGEBRA`'s** bar and does not belong here; §7.6 measures `0.000e+00`, so bit-exactness costs nothing |
| `GATE-ROWSUBSET` | G | **HateMM only** — the `n = 743` all-False build is **bit-identical** to the `n = 744` one-hot build restricted to the 743 surviving rows, all 13 arms. **Strictly stronger than C01's `displacement_registered_null_exclusion` at the key level; it is not C01's property**, which is vote-level and **has no object here** |
| `GATE-NULLREMOVED` | G | no arena population contains an exact-zero row; removed set `{355}` / `{}` |
| `GATE-IDPARITY` | G | every ro cache's `ids` order and `labels` identical to the native bank |
| `GATE-ZEROMASK` | G | **feature space only** — measured exact-zero row set equals `{355}` / `{}` on both policies |
| `GATE-LEDGER` | G | C09's full declared-count predicate set, process count **binding** (§12) |
| `GATE-ORBITDISP` | **L** | **per fold, all 60 head cells** (round-4 I-6): drops its lineage iff `ρ_head > ρ*_D ∧ ρ_raw ≤ ρ*_D` for any arm in any fold; `ρ*` per dataset at full precision (§6.1); `ρ` over **arena rows only** |
| `GATE-RHORAW` | **G** | **split out of `GATE-ORBITDISP` (round-5 I-2)**: `ρ_raw` reproduces §6.1's 26 frozen values at 4 dp. This is a property of the ro caches, the raw leg and the frozen table — **identical for both lineages** — so a drift would have dropped one lineage while the other proceeded on the same drifted data. It belongs beside `GATE-C01PARITY` as a shared-algebra global gate |
| `GATE-ARENA` | **L** | **lower** bound `arena majority + 0.02 ≤ acc` on **`endpoint_std` only**; **upper** bound `acc ≤ 0.98` on `endpoint_std` **and** both real arms. Bands `[0.6203, 0.98]` / `[0.7091, 0.98]` (§6.3) |
| `GATE-NESTED` | **L** | **per item**, the head that scored it excluded its fold; check count equals the item count |
| `GATE-SELFTEST` | **L** | **`net_s(A) = n_D · (acc_s(A) − acc_s(reference))`** exactly for **every one of the 14 arms** (round-4 I-7, including `avg_score`), **per seed**, dataset and lineage, with `n_D` pinned to §3.7 and `reference` per §5.2.1. Round-5 I-1: v5 wrote the seed-**mean** symbol `acc(A)` in a per-seed assertion, which is false in general and would have dropped both lineages on every run |
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

**Reduction order, frozen (round-7 I-1).** `ρ = ‖mean_i k_i‖` is computed with the **mean
accumulated in `float64` over the `float32` keys**. This matters at the sixth decimal of exactly one
of the 26 values: HateMM `orthrot_83p8` is `0.9568935731 → 0.956894` under `float64` accumulation
and `0.9568933249 → 0.956893` under `float32`. Both are honest measurements of one quantity under
different reduction orders, differing by `2.5e-07` (~2 `float32` eps); v6 printed the `float32`
value, v7 the `float64` one, and neither said which. **All 26 agree at 4 dp under both reductions,
so `GATE-RHORAW`'s 4-dp assertion is unaffected under either** — the freeze exists for
reproducibility under §13 item 7, not because a gate is at risk.

**`ρ_raw`, frozen at 6 dp under that reduction, measured on the arena population:**

| arm | HateMM | MHC-ZH |
|---|---|---|
| `endpoint_std` | 0.968176 | 0.977223 |
| `common` | 0.964446 | 0.969686 |
| `orthrot_83p8` | 0.956894 | 0.964384 |
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
matrices, taken over row-renormalised keys** (`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz::K_train`,
18 cells per dataset): HateMM min/median/max `0.447803 / 0.562434 / 0.632996`; MHC-ZH
`0.340179 / 0.574247 / 0.667326`; **0/18 above `ρ*` on both**. *(Round-12 M-2: `ρ` is defined at the
top of this section as `‖mean unit key‖`, but the banked `K_train` rows are **not** unit-norm as
stored — measured row norms run `0.027`–`0.56` across all 36 cells (round-13 M-1: v13's `0.04`–`0.27` was one cell's span read as if it were the corpus's) — so reading these six figures off the stored arrays
literally gives values `3.5×`–`7.6×` lower. Renormalising the rows reproduces all six to the digit, and
the three words above put the licence at the measurement instead of fifty lines earlier. Nothing
moves: §11 declares these 36 mints inputs to this **reference measurement only — no gate reads
them**.)* A trained deployed head sits at
roughly **half** the bar. Reproduced to the digit by rounds 3 and 4. Label-free, computes no
accuracy.

### 6.2 `GATE-ARMVIAB` is retired — round-4 C-1

v4's `GATE-ARMVIAB` escaped a one-sided HALT only through case 1: *head-space arm fails
`majority + 0.02` **and the raw counterpart also fails** ⇒ no HALT*. **That branch is
unreachable.** §1's table records C01's measured raw `displacement` at `0.8505` / `0.8846` and
`common_displacement` at `0.8598` / `0.8590`, against arena bars of `0.6203` / `0.7091` — clearing
by `0.15`–`0.24` (round-12 M-3: the four measured clearances are `0.1499 / 0.2302 / 0.2395 / 0.1755`,
so the upper end rounds to `0.24`, not `0.23`; every clearance clears and nothing reads the band). The raw leg here is the same features and the same operator on a **larger**
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
sentence — **scoped, per round-6 I-2, to runs in which Head-N's arena accuracy exists.** Head-N's
36 mints run regardless (they anchor `GATE-FLOOR`) and §5.6 rule 2 keeps Head-N on every CLOSE
path, so the quantity exists on every closure; on a Head-N-drop SURVIVE or HALT it is reported as
`INSTRUMENT_FAILED` rather than silently omitted.

### 6.5 `GATE-ZEROOP` and `GATE-ALGEBRA`

The two are **logically independent in both directions**; the value is their conjunction. The θ=45
identity is not exact, so a key perturbation can reorder a top-20 neighbourhood and `GATE-ZEROOP`
carries a real false-HALT probability on a correct run.

**The tie diagnostic, with round-4 I-4's unit correction.** v4 compared a **key-component**
residual against a **similarity** gap. `GATE-ALGEBRA`'s residual is a per-component max-abs on the
key matrices (`:1372-1377`); a key perturbation `Δk` with `max|Δk| = ε` changes an inner product
against a unit query by up to `‖Δk‖₂ ≤ √d · ε` — a factor `√2048 = 45.25` at the paired arms'
dimension.

**The residual is now measured on trained heads across four cells, and it is the head-space value
that binds** (§7.8). **`GATE-ALGEBRA`'s residual is a `np.max` over rows** (`:1372-1377`) and is
governed by the **smallest** `d_i` in the cell, not the median — so the headroom claim is a
statement about the tail (round-6 H-3a). Measured at `θ = 45` across both lineages and both
datasets: **`8.848e-08`–`2.682e-07`**, giving similarity windows of `4.00e-06`–`1.21e-05` and
`2e-6`-bar headroom of **`7.5×`–`22.6×`**. v5
illustrated the point with the **raw**-key `1.192e-07`, and round 5 measured `1.863e-07`–`1.974e-07`
on *untrained* heads while this document measured `1.175e-06`–`1.362e-06` on untrained heads at
different seeds — a `6×` spread between two honest measurements of the same quantity. **That spread
is itself the finding: at an untrained head the orbit is near-collinear (median `‖Δ‖ = 0.0032`), so
the residual is dominated by a near-zero denominator and is not a stable estimate of anything.** On
a trained head the orbit opens to median `‖Δ‖ = 0.2301` and the residual settles at `2.384e-07`.
Only the trained figure is quoted as load-bearing; the untrained ones are recorded in §7.8 as the
reason not to calibrate anything on them.

* **Ranking:** the **union** of the two arms' top-21 sets.
* **Residual:** `‖Δk‖₂` measured directly on the head-space key difference, or its bound
  `√d · max|Δk|`; the **maximum** over the two compared identities. **It is the head-space
  residual, measured at run time on the same lineage whose predictions are being compared** — not
  the raw-key figures v4 quoted.
* **Criterion:** an item is a **tie casualty** iff recomputing its rank-weighted vote leaves the
  two arms' predictions equal under the **worst case over all orderings** of every near-tie group
  (this is what "collapsing" means; v4 left it undefined, and different readings give different
  tie sets).
* **Aggregation (round-5 H-3):** mismatches are counted **per `(dataset, seed, lineage)`, pooling
  the five folds' held-out items**, so the denominator is `n_D` and the cap is the `1 %` it is
  described as. v5 left this undefined, and the same integer `7` would have been a `1 %` cap
  against a `(seed, lineage)` aggregate, a `4.7 %` cap against a single fold cell, or a `0.31 %`
  cap against a `3 seeds × 5 folds` aggregate — readings that differ by `15×` in HALT probability.
  The gate drops its lineage if **any** `(dataset, seed)` cell exceeds the cap.
* **Cap:** a mismatch on more than `⌊0.01 × n_D⌋` items (`7` HateMM, `5` MHC-ZH) HALTs regardless.
  The cap is **one-directional** — it can only convert REPORT → HALT (§5.9 item 5).
* **Recording:** the measured residual and the tie-casualty count are written to the **verdict
  face**, so a run-time quantity is auditable after the fact rather than a hidden free parameter
  (round-5 §15.6's first condition).
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
import cost is **already inside every one of the 66 mint units and inside `U9`** (§7.7): the same
run's internal timer reads `33.0 s`, a `7.4 s` gap, and measured startup alone is `3.05–3.18 s`.
**No separate interpreter line is added for those 72 processes, because adding one would
double-count** — confirmed correct by rounds 3 and 4. **The remaining two processes — the `--gate-sha-only` driver leg and the arena — are a different
case and are priced separately at §8 Phase 1g**: every arena-side unit in §7.7 is an
*internal-operation* timing two to five orders of magnitude below a python startup, so none of them
can contain one.

*(Round-11 I-1 and M-1, both in this paragraph. The scope clause is the repair: v11's *"already
inside every unit"* was true of the mint units this section is titled for and **false of the
arena's**, which is how the tenth uncounted item in §8 stayed invisible for eleven rounds. And v11's
*"No **Phase 1e** line is added"* — byte-stable since v4, when §8 had no Phase 1e — named a label
§8 has used since v6 for `GATE-FOLD`'s banked-`.npz` parity re-read, so one label carried two senses
two sections apart; the sentence now names what it actually excludes.)*

| lineage | unit | dataset | measured wall |
|---|---|---|---|
| Head-N | fitting-pool head | HateMM / MHC-ZH | **40.39 / 34.40 s** |
| Head-N | full-train head | HateMM / MHC-ZH | **49.30 / 38.87 s** |
| Head-R | scratchpad harness | HateMM / MHC-ZH | 37.46 / 27.54 s — **not used**, §3.3 |

Fold parity passed in all seven mints; peak RSS **1.25 GiB**.

### 7.3 Blindness

**The warrant is two-part from v6 onward (round-6 H-2), because v6 was the first draft in which a
trained head touched the ro caches.** (i) Every *untrained*-head leg is scientifically void by
construction — real operations at real scale, meaningless numbers. (ii) The **trained**-head legs
(§7.8) computed **only** the native deployed vote — a banked instrument anchor already published in
§6 — plus key-space algebra residuals and displacement-norm geometry. **No vote was taken on any
ro-derived arm at any point**, which is the assertion that actually secures blindness: it is a
statement about what was computed, not about how a head was initialised. §13 item 26 puts it where
the code lineage can check the freeze record against it. **No arm accuracy has been computed,
printed or recorded at any point in v1–v15.** Rounds 4 and 5 audited this by grepping every decimal
in the **closed** interval `[0.6, 0.99]` — a convention stated here once because it decides the
count (round-12 M-1); round 5 classified all **98** distinct values across v1–v5 and found each to be a
`ρ`, a `‖head_f(0,0)‖` magnitude, a cos/`‖Δ‖` geometry figure, a banked `GATE-FLOOR` anchor, a
published C01 dev-arena accuracy, a majority/band constant or a unit-time string. Round 10 repeated
the corpus grep across v1–v10 independently, found **116** distinct values and **exactly one new in
v10** — `0.8718`, ZH `common`'s published C01 dev-arena accuracy read out of `C01_A0_OUT.json` — and
confirmed the sentence true of v10, which is why its scope label reads `v1–v12` rather than
`v1–v9` (round-10 M-1). Round 11 repeated the grep across v1–v11 and found **exactly two new in
v11** — `0.615` and `0.66` — and verified **both are seconds** (Phase 1f's measured product and its
`0.0044 s` bound), so *"v11 adds no accuracy of any kind"* was checked rather than inherited.
**v12 added no accuracy either**, and round 12 verified it the hard way: it reclassified **all 81**
in-band values present in v12, found a verified non-arm provenance for every one, and reported the
**new-in-v12 in-band set EMPTY** — v12's twenty genuinely new decimals are all timings, second
totals, minutes and one share. **None of v13, v14 or v15 adds one**: v13's and v14's new measurements are interpreter and
full-process-wall **timings** (§7.7's arena decomposition — `52` timed starts in v13, `56` in v14's
single uniform re-measurement) plus arithmetic checks of round 12's and round 13's Minors, and
**v15 measures nothing new at all** — its work is two prose corrections, one bridge computation and
one execution of the §14.2 script against itself. Round 14 verified the corpus total is **unchanged
at `118` from v12 through v14**, so the new-in-v13 and new-in-v14 in-band sets are both **empty**. *(Round-12 M-1 also corrected the count above: `97` is obtainable only under a
half-open interval, which would have made round 10's figure `115` rather than the `116` it
reported. Under the closed interval this document now names, the consistent triple is
`98 / 116 / 118` for v1–v5 / v1–v10 / v1–v12 — reproduced here — and `96 / 114 / 116` if the two
self-referential endpoint tokens inside the literal `` `[0.6, 0.99]` `` are excluded.)*
v10's `0.8718` was **read** out of a banked OUT file, not measured, which is why the next sentence
still holds.
**v6–v15 add exactly one measured accuracy and it is not a battery arm**
(round-6 M-2 corrects v6's *"two"*): §7.8's
`0.8725`, the native deployed vote, an instrument anchor already published in §6. Round 6 confirmed
by corpus grep that this is the only new decimal in `[0.6, 0.99]` across v1–v6; the other §7.8
figures are geometry (`‖Δ‖`) and key-space residuals, not accuracies.

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
| `U3` | **RETIRED by ERRATUM 1.** Priced *"one comparison, both metrics"*, an object that no longer exists: after the erratum the accuracy leg is per **comparison** and the macro-F1 leg is per **(arm, seed)**. Round 14 measured `0.023`–`0.028 s` against its frozen `0.126 s` and ruled it conservative; that measurement was of the superseded object and is recorded as such | HateMM | ~~0.126 s~~ |
| `U_acc` | **ERRATUM 1.** Accuracy leg, §5.4's retained frozen expression, **per comparison** — `c̄_A[draws].mean(1) − c̄_c[draws].mean(1)` at `B = 2000` over the shared draw matrix. Timed region = the vectorised computation from the shared draws index matrix to the `(B,)` delta vector, warm; **the draws matrix is built once and is not in the unit** | HateMM, `n = 743` | **0.0049 s** |
| `U_mF1` | **ERRATUM 1.** Macro-F1 leg, C01's recompute-per-resample form via `mechfix_ops.macro_f1`, **per `(arm, seed)`** and vectorised over all `B = 2000` draws. Same timed region as `U_acc`. Verified **bit-identical** to a scalar `mechfix_ops.macro_f1` loop over 300 draws (`max|diff| = 0.000e+00`) | HateMM, `n = 743` | **0.0384 s** |
| `U4` | one shuffled-pair null draw (2 arms × 5 folds + rebuild) | HateMM, **head space, 1024/2048-d** (round-4: the space is now named) | ~~0.08908 s~~ → **`0.33 s`** (CODE-R1 H-4). §7.7 flagged `U4` as *"the single largest uncorroborated unit"* and rounds 13 and 14 left it the only substantial one uncorroborated end to end. The code lineage measured it against its **own stated object** — two arms, five folds' builds, ten votes, through the real `ArmBuilder` and the frozen `deployed_vote` at `n = 743`, 8 reps: min `0.3225` / **median `0.3241`** / max `0.3357 s`; MHC-ZH `0.1805`/`0.1811`/`0.1843 s`, applied at the HateMM figure per this table's convention. Carried `0.33 s`. The frozen value was low by **`3.64×`**. The cost is the frozen `l2_rows`' own — ~11 calls per build, each doing two full `O(n·d)` exact-zero scans — not an implementation inefficiency; the separate 10.8× defect (S5 building all 15 arms when it reads 2) is fixed in code and is **not** in this figure |
| `U5a` / `U5b` | two-block build + compare / builder-only | HateMM | 11.27 / 4.63 s |
| `U6` | `ρ` over 13 arms | HateMM, raw | 0.62 s |
| `U7` | `GATE-SHA` over **all 38 §11 artifacts** (8 caches + 13 modules/configs + 16 banked + **the design document, ERRATUM 2 §3**; round-8 H-1 — v8 said "8 caches + 6 modules" while §11 listed 37) | — | 0.12 s + `0.005 s` measured for the sixteen + `0.000164 s` measured for the design document (188 061 B, 7 reps; round 5 independently `0.000148 s`) = **0.13 s**, unchanged at two decimals |
| `U8` | ro cache `torch.load`, 2 files | HateMM | 0.033 s |
| `U9` | `GATE-DEVFID`, per `(dataset, seed)` — **timed region = the full `python headspace_fidelity.py …` process wall**, interpreter and imports included (round-11 I-1) | HateMM / MHC-ZH | 3.70 / 3.49 s |
| `U10` | head-space build of all 13 arms, one cell | HateMM, `n = 743` | 0.1873 s |
| `U11` | interpreter + imports — **class-dependent, and the class is the import set** (round-12 I-1) | — | **mint/fidelity class** `3.05–3.18 s` (`headspace_mint`'s set: numpy, torch, sklearn, `mechnov_pairverify`); **arena class** `3.094–3.717 s` over **35 arena-class runs by three parties** — this document `8`, round 12 `3`, round 13 `24`, split in §7.7's second table (`headspace_arena.py`'s actual top-level set — §13.1 item 27 — plus `c01_policy_contrast_a0` and a `runtime_block()` call). Both are **inside the mint units and inside `U9`**; the **two arena-class startups — the `--gate-sha-only` driver leg and the arena — are priced at §8 Phase 1g** |

**Convention:** every unit was measured on **HateMM**, the larger dataset, and applied to MHC-ZH
unchanged, so every such application **over-states** the MHC-ZH cost. Exceptions: `U9` (per
dataset), `U7` and `U11` (dataset-independent).

**The arena's startup, and why `U11` is class-dependent (round-12 I-1).** v12 reported an
*"arena-class"* interpreter+import of `1.82–1.85 s`, and round 11 independently reported
`1.84–1.91 s` for the same set. **Both measurements omitted `sklearn`**, which the arena cannot
avoid: `headspace_mint.py:68` imports `StratifiedKFold` at top level and is a §11 sha-frozen import,
and `headspace_arena.py:35-36` imports `sklearn.metrics` and `sklearn.model_selection` directly.
Round 12 caught it and decomposed the cost. **This document re-measured the whole decomposition
from scratch in v14 as a single uniform sample — `7` rungs × `8` runs = `56` timed interpreter
starts, one command, no rung run a different number of times** (round-13 I-1: v13's counts were
assembled from two separate commands and reported three ways that did not reconcile, so v14 replaces
reconstruction with one reproducible sample):

| # | import set | runs | measured wall |
|---|---|---|---|
| 1 | bare interpreter | 8 | `0.01 s` |
| 2 | `+ numpy` | 8 | `0.09–0.10 s` |
| 3 | `+ torch` | 8 | `1.75–1.98 s` |
| 4 | `+ faiss` | 8 | `1.79–1.95 s` |
| 5 | `+ c01_policy_contrast_a0 + mechfix_ops` — **the set v12 and round 11 timed** | 8 | `1.81–1.94 s` |
| 6 | **`headspace_arena.py`'s actual top-level set** (adds `sklearn.metrics`, `sklearn.model_selection`, `mechnov_pairverify`, `vsw_pregate`, `headspace_mint`) | 8 | **`3.02–3.21 s`** |
| 7 | **the arena class** — rung 6 `+ c01_policy_contrast_a0` `+ runtime_block()` | 8 | **`3.10–3.70 s`** |

Timed by `/usr/bin/time -f '%e'` around the `python -c` invocation under
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8` and `CUDA_VISIBLE_DEVICES=""`; rung 7 replicates §13.1
item 27's set exactly. **`sklearn` alone is `≈ 1.2 s`** (rung 6 − rung 5), about two-thirds of the
figure.

**The arena-class evidence, pooled across three parties and split so the count is checkable
(round-13 I-1):**

| party | arena-class runs | range |
|---|---|---|
| this document, rung 7 above | **8** | `3.10–3.70 s` |
| round 12, its `+ runtime_block()` rung (`R12:349`) | **3** | `3.12–3.27 s` |
| round 13, its rung 7 (`R13:237`) | **24** | `3.094–3.717 s` (median `3.172`, mean `3.254`) |
| **pooled** | **35** | **`3.094–3.717 s`** |

§8 Phase 1g carries **`2 × 3.8 s = 7.6 s`**; the **unit** `3.8 s` is above the pooled maximum `3.717 s` by `0.083 s`. *(Round 12 prescribed keeping
`3.2 s` and relabelling the basis as an approximation with the residual `≤ 0.2 s`. That prescription
was written against round 12's own three-run maximum of `3.27 s`; the pooled 35-run maximum is
`3.717 s`, so the stated residual would have been false and the row would have become the second in
§8 carried **below** its measurement — the first, Phase 1d, being one-decimal rounding that rounds 8
and 9 ruled acceptable and that round 13 independently confirmed conservative. **Round 13 ruled the
deviation `FORCED BY MEASUREMENT`** on its own 24 runs, ten of which exceed `3.2 s`. The deviation is
recorded at §14 with this measurement as its warrant.)*

*(Two endpoint corrections v13 got wrong, both from mis-attributing a rung. v13 wrote the pooled
arena-class range as `3.00–3.75 s`: the `3.00` was round 12's **sklearn-only** rung, not an
arena-class rung, and the `3.75` came from a v13 sample that mixed rung 6, rung 7 and an
intermediate set. Both endpoints above are now arena-class observations and each is attributable to
one party's stated rung.)*

**`U9`'s boundary, measured rather than inferred (round-11 I-1, second line).** Round 11 was right
that the count in §8 Phase 1g turns on this clause and that v11 nowhere stated it. It is settled by
measurement, not by the `--seeds 0` anecdote below: three timed runs of
`python scripts/analysis/headspace_fidelity.py --dataset hatemm --mintdir <banked C09 scratch>
--out <scratchpad> --seeds 0`, wall-clocked around the invocation, give **`3.13 / 3.46 / 3.12 s`**,
while the same process's interpreter and imports **alone** measure `3.06–3.16 s` on this node. The
frozen `U9 = 3.70 s` therefore **cannot** be an internal timing — an internal timing would have to
be under `0.4 s` — and it already contains the six fidelity processes' startup. **So §8 Phase 1g's
count is `2` — the `--gate-sha-only` driver leg and the arena — and the `U9` boundary that
excludes the six fidelity processes is determined by this measurement rather than inferred.**
(The `--out` target was the session scratchpad; nothing in the repository was written.)

**Corroboration status, carried to freeze.** `U5a`, `U5b`, `U6`, `U10`'s object and the mints are
independently reproduced by rounds 3 and/or 4; `U2a`–`U2d`, `U7`, `U8`, `U11` are not (rounds 13 and
14 subsequently corroborated `U2a`–`U2d`, `U7`, `U8` and the retired `U3`, leaving `U4` the only
substantial uncorroborated unit). **ERRATUM 1's `U_acc` and `U_mF1` are this document's own
measurements and are uncorroborated**; the erratum adjudication independently measured the same two
objects at `0.0030` and `0.0131 s`, i.e. **below** the values carried here, so the carried figures
bound its measurement as well as this one.
**`U4` is the single largest uncorroborated unit** (`273.7 s`, `9.3 %` of the total) and its space
is now named above. The freeze record will state the exit-status discipline under which each unit
was timed — the instrument that catches the `U9` defect class, where a crashed process's `echo`
status was briefly recorded as a measurement before being re-run correctly with `--seeds 0`.

### 7.8 `GATE-FLOOR` discharged, and the head-space displacement tail

**Round-5 I-4 was right: `GATE-FLOOR` is the design's only anchor, its failure is a global HALT,
and no dry check had ever exercised it.** v6 discharges it by measurement rather than by
declaration. One HateMM `s0 / fold 0` Head-N head was minted through the wrapper path, its native
deployed vote computed on that fold's held-out fifth, and compared against the banked anchor:

| quantity | value |
|---|---|
| re-minted `hatemm s0 fold0` native deployed-vote accuracy | **`0.8725`** |
| banked `headspace_arena_hatemm_s0_OUT.json::fold_acc_deployed[0]` | **`0.8725`** |
| match at 4 dp | **yes** |
| mint cost | `33.5 s`, inside the `40.39 s` unit |

**This is not a battery arm.** It is the native deployed vote — the instrument anchor whose value
is already published in §6 — so computing it discloses no arm accuracy and §7.3's blindness
discipline is untouched. What it establishes is that the anchor the entire two-lineage structure
hangs on **does** reproduce on a freshly minted head.

**The head-space displacement tail and the `GATE-ALGEBRA` residual — four cells, both lineages,
both datasets (round-6 H-3).** v6 priced a **max-over-rows** statistic from a **median**, on one
cell of sixty and one lineage of two. Round 6 was right that a median cannot bound a max: the
`θ = 45` identity fails only through the `cos45 − sin45 = 1.11e-16` asymmetry acting on the
difference vector, so the residual is governed by the **smallest** `d_i = ‖l2(h_ow,i) − l2(h_std,i)‖`
in the cell. v7 measures the tail directly, on four freshly minted `seed 0 / fold 0` heads spanning
**both lineages and both datasets**:

| cell | `min d_i` | `q₀.₁` | median | max | frac `d_i ≤ 1e-3` | `θ = 0` | `θ = 45` | headroom |
|---|---|---|---|---|---|---|---|---|
| HateMM · Head-N | **0.021052** | 0.061348 | 0.230101 | 0.802757 | **0.0000** | 1.490e-08 | 2.384e-07 | 8.4× |
| HateMM · Head-R | **0.018145** | 0.044342 | 0.179587 | 1.825995 | **0.0000** | 1.490e-08 | **2.682e-07** | **7.5×** |
| MHC-ZH · Head-N | 0.038435 | 0.060409 | 0.180500 | 1.092214 | **0.0000** | 1.118e-08 | 8.848e-08 | 22.6× |
| MHC-ZH · Head-R | 0.029817 | 0.069485 | 0.181580 | 1.772445 | **0.0000** | 1.490e-08 | 1.183e-07 | 16.9× |

**Three results, and the load-bearing claim is now a range, not a point.**

1. **`GATE-ALGEBRA`'s `2e-6` bar transfers with `7.5×`–`22.6×` headroom**, worst case HateMM ·
   Head-R. The `θ = 45` residual spans `8.848e-08`–`2.682e-07` across the four cells — a **`3×`**
   spread, not the `6×` the untrained measurements suggested, and **Head-R, the in-domain lineage
   round 6 flagged as entirely unmeasured, is measured and is the tightest cell.** Per §7.5's own
   discipline the range is what is load-bearing; the single trained value v6 quoted is not.
2. **`min d_i` is `0.018`–`0.038`, i.e. `18×`–`38×` above C01's `tiny_displacement_epsilon = 0.001`,
   and the fraction at or below it is `0.0000` in every cell.** So C01's `tiny_ok` limb (§5.2.3)
   would pass with large margin in head space, as it does in raw space, and the head-space lower
   tail is **not** near the numerical floor. v6's untrained median of `0.0032` — only `3.2×` above
   the epsilon — was the misleading figure; it is an initialisation artifact.
3. **The trained head does not collapse the prompt orbit.** Trained medians are `0.18`–`0.23`
   against `0.0032` untrained, a `~70×` difference, corroborating §6.1's trained-head `ρ` result and
   retiring the last remnant of v1's "219× contraction" worry, exactly as round 1's I-10 warned.

**Untrained residuals are recorded only as the reason not to calibrate on them:** round 5 measured
`1.863e-07`–`1.974e-07`, this document measured `1.175e-06`–`1.362e-06` at different seeds — two
honest measurements `6×` apart, because at an untrained head the endpoints are near-collinear and
the quantity is dominated by a near-zero denominator. No accuracy of any battery arm was computed in
producing any row of this table.

### 7.9 Dry-check cost

**Five CPU head mints are attributable to the v6–v7 rounds:** one trained in **v6** for §7.8's
`GATE-FLOOR` discharge (`33.5 s`) and four trained in **v7** for its four-cell displacement-tail
table, spanning both lineages and both datasets (`≈ 33–40 s` each). *(Round-9 H-1: v9's headline
said "five … trained in v7", which applied its own reported-versus-spent rule backwards and
contradicted the parenthetical beneath it, the sum below, and §7.8. The rule stands — §7.8 records
when each measurement was first **reported**, this section when the compute was **spent** — and the
headline now states it correctly, so there is nothing left to reconcile.)*

**v7's measurements** added ≈ **4 wall-minutes / ≈ 21 CPU-minutes**: the **four** tail mints
(`≈ 2.4` wall-min at §7.2's units) plus the round-6 C-1/C-2 source reads, the `holm_adjust`
executions and the arithmetic checks. *(Round-9 H-1's second limb: v9 attributed `≈ 5 / ≈ 25` to v7
while counting five mints there; with four mints the figure is `≈ 4 / ≈ 21`, and v6 carries the
discharge mint's `≈ 1 / ≈ 4`.)* v7 also ran the untrained-residual
comparison, the `holm_adjust` executions, the S5 feasibility arithmetic and the C01 source/OUT reads
behind D-1.

**v8's measurements** added ≈ **2 wall-minutes / ≈ 6 CPU-minutes** and **no mints**: the round-7 C-1
reproduction (both readings of the builder against `prepare_views` on both datasets), the I-1
reduction-order comparison, the I-2 loop timings, and the sixteen sha256 of I-3.

**v9's measurements** added ≈ **1 wall-minute / ≈ 3 CPU-minutes** and **no mints**: the round-8 C-1
byte-count reproduction, the `GATE-SHA` widening cost (`5 ms` for the sixteen banked artifacts), and
the audit re-runs.

**v10's measurements** added ≈ **1 wall-minute / ≈ 3 CPU-minutes** and **no mints**: the arena-side
`.npz` key-materialisation unit of round-9 I-1, the `U_tie` measurement of round-9 I-2, the two
`C01_A0_OUT.json` cells of round-9 M-3, and the audit re-runs.

**v11's measurements** added ≈ **1 wall-minute / ≈ 3 CPU-minutes** and **no mints**: a
re-measurement of the `.npz` key-materialisation unit on the **native** `K_train` array specifically
(round-10 I-1 — 20 timed loads over five banked mints, median `0.0042 s`, mean `0.0044 s`, against
the frozen `0.0041 s`), the exhaustive item-reference scan behind round-10 I-2, and the audit
re-runs.

**v12's measurements** added ≈ **1 wall-minute / ≈ 3 CPU-minutes** and **no mints**: three timings
of an arena-class interpreter+import (`1.82 / 1.85 / 1.82 s`) and three of a fidelity-class one
(`3.08 / 3.06 / 3.16 s`), **three full-process-wall runs of `headspace_fidelity.py --seeds 0`**
(`3.13 / 3.46 / 3.12 s`, `--out` into the session scratchpad) settling `U9`'s boundary for round-11
I-1, an independent re-sum of §8's printed column, and the audit re-runs.

**v13's measurements** added ≈ **2 wall-minutes / ≈ 4 CPU-minutes** and **no mints**: the arena
import decomposition of §7.7 — **`52` timed interpreter starts**, from two commands (`8` rungs × `4`
runs = `32`, then `10` runs each on two rungs = `20`) — a corpus re-grep of every decimal in the
closed `[0.6, 0.99]` across v1–v13 for round-12 M-1, a re-derivation of `ρ` on the 36 banked mints
with and without row renormalisation for round-12 M-2, the four clearance subtractions of round-12
M-3, and the audit re-runs. *(Round-13 I-1: v13 reported this as `≈ 1 / ≈ 3` and the starts as `24`
in total, `4` per rung with `10` each on two rungs — three figures that reconciled with neither its
own table nor each other. The `52` above is the sum of the two commands v13 actually ran. **The
bridge to v13's own printed table, which round-14 M-2 asked for so the correction is checkable and
not merely asserted:** that table showed seven rows at *"`4` runs per rung unless stated"* with rung
5 at `(10 runs)` and rung 7 at `(14 runs)`, i.e. `5×4 + 10 + 14 = 44`. It differs from the executed
`52` in two places, both v13's reporting rather than its execution — an **eighth** rung (the arena's
actual set plus `c01_policy_contrast_a0` but **without** `runtime_block()`) was run and never printed
as a row, contributing `4`; and rung 5's `(10 runs)` was the **second command's increment** where
rung 7's `(14 runs)` was a **total**, so rung 5's true total is also `14`, contributing another `4`.
`44 + 4 + 4 = 52`. Recording one rung by increment and its neighbour by total is exactly the
confusion round-13 I-1 diagnosed from the outside, and v14's uniform `7 × 8` sample exists so the
question cannot recur. The time is corrected upward accordingly. Correcting a recorded spend figure downstream of the round
that spent it is the campaign's numeric-provenance discipline, not a re-estimate.)*

**v14's measurements** added ≈ **2 wall-minutes / ≈ 4 CPU-minutes** and **no mints**: §7.7's
decomposition **re-measured from scratch as one uniform sample — `7` rungs × `8` runs = `56` timed
interpreter starts, one command** — a re-derivation of the row norms and gap factors of all 36
banked `K_train` matrices for round-13 M-1, a direct `§14.2` line diff for round-13 M-2, and the
audit re-runs.

**v15's measurements** added ≈ **1 wall-minute / ≈ 1 CPU-minute** and **no mints, and no timing of
the instrument at all**: the §14.2 script run against the finalized document, the plain and biting
splice counterfactuals of §14.1, and the arithmetic bridge from v13's printed `44` to its executed
`52`. **v8 through v15 each trained no heads.**

**Cumulative v1–v15, shown as a sum so it is checkable rather than asserted** (round-8 I-1):
mints `= 7 (§7.2's unit-timing and payload runs, v1–v6) + 1 (v6's GATE-FLOOR discharge) + 4 (v7's
tail cells) + 0 (v8) + 0 (v9) + 0 (v10) + 0 (v11) + 0 (v12) + 0 (v13) + 0 (v14) + 0 (v15) = `
**12 CPU head mints**; time `≈ 22 (v1–v6, incl. the discharge mint) + 4 (v7) + 2 (v8) + 1 (v9)
+ 1 (v10) + 1 (v11) + 1 (v12) + 2 (v13) + 2 (v14) + 1 (v15) = ` **≈ 37 wall-minutes**,
`≈ 89 + 21 + 6 + 3 + 3 + 3 + 3 + 4 + 4 + 1 = ` **≈ 137 CPU-minutes** *(the v13 terms are round-13
I-1's correction, from `1`/`3`; round 14 ruled the in-place correction right and this document
supplies the reconciliation clause its ruling made a condition — see the v13 paragraph above)*. All `$0`, zero GPU, no
battery-arm accuracy at any point. *(v8's "eleven" did not reconstruct: `7 + 5 = 12`. Round-9 H-1
moved the discharge mint's `≈ 1 / ≈ 4` from the v7 column to v6, where it was spent; the totals are
unchanged by the move. Round-10 I-3: v10's heading read "Cumulative v1–v9" over a sum whose own
terms already ran through v10 and over a footer that said v1–v10 — the heading is the version span
of the terms beneath it, which at v15 is `v1–v15`. Rounds 11 through 14 each verified heading, terms
and footer agree and re-derived all three sums.)* The CPU-cap conflict was knowable from C09's
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
| **1b** key forwards — `30` Head-N fold mints × {native, `ro_std`, `ro_ow`} + `6` Head-N full mints × {those three, native dev} + `30` Head-R fold mints × {`ro_std`, `ro_ow`} = `(30×3)+(6×4)+(30×2)` (round-5 I-3: the factors are now named) | `174` | `U1` | `8.0 s` |
| **1c** ro cache loads, per process — **66 mints + the arena process itself** (round-5 I-3) | `67` | `U8` | `2.2 s` |
| **1d** `GATE-SHA`, **twice** — the `--gate-sha-only` driver leg and the arena | **`2`** | `U7` | **`0.3 s`** (`2 × 0.13 = 0.26 → 0.3` at one decimal, by the Phase 1c worked example below; **not** `2 × 0.1`) |
| **1e** `GATE-FOLD`'s banked-`.npz` parity re-read (round-5 I-3), `66 × 0.5 ms = 0.033 s` — **metadata only** (`meta` + `fold_of`), re-measured this round at `0.0003 s/file`, so the frozen `0.5 ms` unit is conservative | `66` | measured | `0.1 s` |
| **1f** the arena's materialisation of head-key matrices from the banked mint `.npz` — **both streams** (**round-9 I-1** priced the ro-derived one, **round-10 I-1** the native one; Phase 1c prices the arena's *ro cache* load, Phase 1e a *metadata-only* read, Phase 2b a build on *in-memory* arrays; none prices `np.load(...)['K_*']`) | `60` cells × 2 ro-derived matrices (`h_std`, `h_ow`) **`+ 30` Head-N fold mints × 1 native matrix** for `GATE-FLOOR`'s 30 votes = **`60 × 2 + 30 × 1 = 150`** | **measured `0.0041 s` per matrix**; timed region = `np.load(...)` **plus `np.asarray` on the named array**, warm cache. Re-measured this round on the **native** `K_train` array itself (`(744, 1024)` float64 in a `6.1 MB` file, the same object class as `h_std`/`h_ow`): 20 loads over five banked mints, median `0.0042 s`, mean `0.0044 s`; round 10 independently measured `0.0043 s` | `1.3 s` (measured product `150 × 0.0041 = 0.615 s`; carried at the conservative bound for a cold cache, which also bounds `150 × 0.0044 = 0.66 s` at `≈ 2×`) |
| **1g** interpreter + imports for the **non-mint** side (**round-11 I-1**, the tenth uncounted item — not a payload loop but the per-process fixed cost the payload loops sit inside; every arena-side unit `U1`, `U2a`–`U2d`, `U3`–`U8`, `U10` is an *internal-operation* timing two to five orders of magnitude below a python startup, so none of them contains one) | **`2`** — the `--gate-sha-only` driver leg and the arena. The 66 mints carry it inside their full-process-wall units and the 6 fidelity processes inside `U9`, whose boundary §7.7 now states from measurement; `1 + 66 + 6 + 1 = 74` accounts for **every** process §13 declares | `U11`, **arena class** | **`7.6 s`** (`2 × 3.8`) (round-12 I-1. v12 carried `3.2 s` against an *“arena-class”* measurement of `1.82–1.85 s` that **omitted `sklearn`** — which `headspace_mint.py:68` and `headspace_arena.py:35-36` both require — and §14 wrongly called that a `≈ 1.7×` bound. §7.7 now decomposes the real startup over a single uniform sample of `7` rungs × `8` runs: the sklearn-less set measures `1.81–1.94 s` and the arena class `3.10–3.70 s`; pooled with round 12's `3` runs and round 13's `24` the arena class is **`3.094–3.717 s` over 35 runs**. the **unit** `3.8 s` is carried **above the pooled maximum** `3.717 s` by `0.083 s`, restoring the convention Phases 1e, 1f and 7z use; the ROW carries `2 ×` that unit. Round 13 ruled the departure from round 12's prescribed `3.2 s` **forced by measurement**. The import set is pinned for the code lineage at §13.1 item 27) |
| **2** head-space votes, 1024-d / 2048-d arms | `4×60 = 240` / `9×60 = 540` | `U2a` / `U2b` | `0.7` / `3.4 s` |
| **2** `GATE-FLOOR` native vote — computed **in the arena process**, so `U2a` is a vote-only timing and the 30 native key matrices it consumes are loaded in Phase 1f, not here (round-10 I-1; the placement is pinned by §13 item 22) | `30` | `U2a` | `0.1 s` |
| **2b** head-space arm construction — **`2 ds × 3 seeds × 5 folds × 2 lin`** | **`60`** *(was 12)* | `U10` | **`11.2 s`** |
| **2z** `GATE-ZEROOP` guard arms — votes `2 × 60`, construction `60 × (2/13)` | **`120` votes + `60` partial builds** *(was 0)* | `U2b` / `U10` | **`2.5 s`** |
| **2R** raw votes, 7168-d / 14336-d | `4×10 = 40` / `9×10 = 90` | `U2c` / `U2d` | `1.7` / `7.3 s` |
| **2Ra** raw arm construction | `2` datasets | `U5b` | `9.3 s` |
| **2C** `GATE-C01PARITY` | `2` datasets | `U5a` | `22.5 s` |
| **2C** `GATE-ROWSUBSET` (HateMM only) | `1` | `U5b + 0.21` | `4.8 s` |
| **2D** `ρ` — **raw `2` + head `60`** | **`62`** *(was 14)* | `U6` | **`38.4 s`** |
| **3** shuffled-pair null draws | `256 × 3 × 2 × 2 = 3072` | `U4` | `3072 × 0.33 = ` **`1013.8 s`** (CODE-R1 H-4; was `273.7 s` at the superseded `U4`) |
| **4** bootstrap comparison-cells (**re-priced by ERRATUM 1**: the accuracy leg stays per comparison on §5.4's retained expression, the macro-F1 leg is a per-`(arm, seed)` precompute that every comparison then differences — **bit-identical** to per-comparison work, because §5.4 shares the draw indices across all comparators and both lineages within a dataset) | macro-F1 `14 arms × 3 seeds × 2 ds × 2 lin = ` **`168`** (an upper bound: only 13 distinct arms enter S4's comparator sets); accuracy `23 × 2 ds × 2 lin = ` **`92`** | `U_mF1` / `U_acc` | `168 × 0.0384 + 92 × 0.0049 = 6.90 s`, carried **`7.0 s`** |
| **5** head-space null-row sensitivity | **0 — the leg does not exist** | — | `0.0 s` |
| **6** `GATE-DEVFID` | `3 + 3` | `U9` | `21.6 s` |
| **7z** `GATE-ALGEBRA`'s residual comparison — `np.max(np.abs(·))` over two `(n_D, 2048)` key-difference matrices per cell (**round-7 I-2**) | `2 × 60 = 120` reductions | measured `0.128`–`1.0 s` across three timers | `1.0 s` |
| **7z** §13 item 25's per-cell tail record — `min_i d_i` and `frac(d_i ≤ 0.001)` (**round-7 I-2**) | `60` cells | measured `0.034`–`0.619 s` across three timers | `0.7 s` |
| **7z** `GATE-ZEROOP`'s mismatch scan and tie-casualty evaluation (**round-8 I-2**; **round-9 I-2** supplied the missing measured unit and corrected the cell count) | scan `2 identities × 60` vectorised comparisons; tie work `≤ cap × cells × U_tie` with **`cells = 12`** — `3 seeds × 2 lineages × 2 datasets`, per §6.5's aggregation, which pools the five folds away — so `7×6 + 5×6 = 72` items worst case (v9's `360` used the fold-level `30`, contradicting the gate's own rule; the direction was conservative) | scan sub-`0.1 s`; **`U_tie` measured at `2.0e-05 s`/item** on one synthetic near-tie group of size `g = 5`, analytic worst-case-over-orderings, timed region = the vote recomputation alone ⇒ `72 × 2.0e-05 = 0.0014 s`; tie work is **zero on a clean run** | `0.1 s` |
| **7** per-gate arithmetic on materialised vectors — `GATE-SELFTEST` (**`14 × 3 × 2 × 2 = 168`**), `GATE-NESTED`, S7, `GATE-POP` (incl. class counts and constant recomputation), `GATE-NULLREMOVED`, `GATE-IDPARITY`, **`GATE-ZEROMASK`, `GATE-FOLD`'s in-process leg and `mints_present_before_arena`** (round-5 I-3) | all | sub-`0.1 s` class | `0.1 s` |

**Total, re-multiplied (ERRATUM 2):** `2642.5 + 1.0 + 0.7 + 0.1 (Phase 7z) + 1.3 (Phase 1f)
+ 7.6 (Phase 1g) + 7.0 (Phase 4, ERRATUM 1) + 1013.8 (Phase 3, CODE-R1 H-4) = ` **`3674.0 s =
61.2 min`** corroborating; **`× 1.25 = 4592.5 s = 76.5 min`** conservative, where `2642.5` is the sum
of every row except the seven named. *(ERRATUM 2 moved two summands. Phase 1g's `3.8 → 7.6` is a
named term. Phase 1d's `0.1 → 0.3` lives INSIDE the residue, which is why `2642.3 → 2642.5`: the
count rises `1 → 2` and the product is **re-multiplied**, `2 × 0.13 = 0.26 → 0.3`, not obtained by
doubling the already-rounded `0.1` — the convention §8 states at Phase 1c below. Round-7 H-1 caught
`0.2` here, carried unexamined from erratum-2 proposal v3 through four adjudications.)* *(Two corrections, both explicit. **ERRATUM 1** moved Phase 4
`11.6 → 7.0 s`. **CODE-R1 H-4** re-priced Phase 3 `273.7 → 1013.8 s` on the first end-to-end
measurement of `U4` against its own stated object: `2929.9 − 273.7 + 1013.8 = 3670.0`. The second
correction is **large and upward** — `U4` was the last substantial uncorroborated unit and it was low
by `3.64×`. It changes where the risk sits: mints fall from `85.6 %` to `68.3 %` of the budget and
Phase 3 rises from `9.3 %` to `27.6 %`.)*
(Round-9 I-1 added Phase 1f at `1.0 s` for `120` materialisations; round-10 I-1 extended it to `150`
and the row to `1.3 s`, moving the total `2930.4 → 2930.7`. Round-9 I-2 reduced the `GATE-ZEROOP`
row from `0.3` to `0.1 s` now that `U_tie` is measured and the cell count follows §6.5. **Round-11
I-1 added Phase 1g at `1 × U11`, moving the total `2930.7 → 2933.9` on the `count = 1` reading
§7.7 **then carried** (ERRATUM 2 moves that count to `2`); **round-12 I-1 corrects that row's unit from `3.2` to
`3.8 s` once `sklearn` is restored to the measured import set, moving the total
`2933.9 → 2934.5` and the conservative figure `3667.4 → 3668.1`.**)

**On the timing spread, and what it institutionalises (round-8 §15 item 5).** The same two loops have
now been timed by three parties with materially different results: `GATE-ALGEBRA`'s residual at
`0.128 s` (round 8), `0.160 s` (round 7) and `1.0 s` (v8); item 25's tail record at `0.034 s` (v8),
`0.122 s` (round 7) and `0.619 s` (round 8). **Rounds 7 and 8 bracket v8 in opposite directions**,
which round 8 correctly reads as decisive: the spread is about *what each timer enclosed*, not about
the machine. v9 freezes the **conservative** figure in each case — `1.0 s` and `0.7 s` — bounding all
three measurements of each loop, costing `0.06 %` of the total, and leaving every heartbeat interval
untouched. The lesson §7.7 institutionalised for the mint units applies here too: **state the timing
boundary, not just the number.** (v5's `2927.5` plus Phase 1e; Phase 1c's count
rises `66 → 67` and its product is `67 × 0.033 = 2.211 → 2.2`, unchanged at one decimal.)

*(Round 4's headline figure was `2925.0 s`; the `2.5 s` difference is the guard-arm
**construction**, which it offered as an option and v5 counts. The direction is conservative.)*

**M-2, adopted:** the printed product column now sums to the total directly, with **Phase 7
carried at its `0.1 s` upper bound**; v4's rounding note described v3's table and is retired.

**Declared slack, outside the projection:** `30 s` for ledger aggregation and JSON emit.

**Peak RSS ≈ 1.3 GiB.** Request 32 GB.

**Where the risk sits — re-derived after CODE-R1 H-4.** Mints are **`68.3 %`** of the total and are
measured directly. **Phase 3 is now `27.6 %`**, up from `9.3 %`, and it is no longer the small term
the earlier text treated it as: a 2× miss moves the total to `4687.8 s = 78.1 min` and a 5× miss to
`7729.2 s = 128.8 min`, the latter `1.68×` the conservative figure — still inside the `2×` clause
below, but with far less margin than before. `U4` is now measured end to end rather than
uncorroborated, which is what made the correction possible.
**If the realized cost exceeds the conservative total by more than 2×, that is itself a reportable
process finding.**

---

## 9. Heartbeat specification

* **The battery's output root is `artifacts/c06_falsifier/`**, repo-relative, and it is the only
  directory this battery writes to apart from SLURM's own job log: the 66 mint `.npz` (each naming
  its `(dataset, lineage, seed, fold)` quadruple, §12), the verdict JSON and the progress file all
  live under it. *(Round-10 M-3: v10 wrote the progress path as `$BASE/progress/…`, the document's
  only shell variable and the only path it never defined. §7's "zero write into `artifacts/`" is a
  statement about the **dry check**, whose outputs went only to the session scratchpad; this root is
  written by the **run**, which has not happened and is not authorized.)*
* One progress file `artifacts/c06_falsifier/progress/C06_PROGRESS.txt`, created by the sbatch
  driver before the first python process starts; every python process **that this lineage authors**
  appends through a handle opened `buffering=1` — the 66 mints, the `--gate-sha-only` driver leg and
  the arena, 68 of the 74. The six `headspace_fidelity.py` processes are sha-frozen and third-party,
  have no progress handle at all, and the **bash** driver writes their line (`sbatch:128-129`) with a
  literal `-` in the elapsed and ratio columns. The bash driver **also** echoes a line per mint,
  unbuffered.
* Each line: ISO-8601 timestamp · phase · units done / total · elapsed · elapsed ÷ **§8's frozen
  projected** value. *(ERRATUM 1 set it to `2929.9 s`; CODE-R1 H-4 to `3670.0 s`; **ERRATUM 2 sets
  it to `3674.0 s`**. It does NOT track §8 automatically — that claim was false and is what let
  `c06_falsifier_mint.py` divide 66 of 74 processes' heartbeat lines by a denominator `740.1 s` from
  the arena's. There is now ONE source, `configs/c06/c06_falsifier.json:"projected_seconds"`; the
  sbatch exports it as `C06_PROJECTED_SECONDS`; `c06_falsifier_arena.py` and `c06_falsifier_mint.py`
  read the export with that literal as a hand-run fallback only; and the arena asserts
  environment == module constant == config in the pre-`gate_sha_only` block, HALTing on mismatch in
  process 1 of 74.)*
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
  single `GATE-C01PARITY` dataset at `11.27 s` (`14.1 s` conservative). The `--gate-sha-only`
  driver leg's startup is the first span that precedes any python-side line and the arena's is the
  second; both are bounded by the same arena-class band — **`3.094–3.717 s` measured over 35
  arena-class runs by three parties** (§7.7), **`2 × 3.8 s = 7.6 s`** as carried at §8 Phase 1g (round-11 I-1,
  corrected by round-12 I-1, count reconciled by round-13 I-1; v12 quoted the sklearn-less
  `1.82–1.85 s` here and v13 mis-stated the sample size) — and the bash driver's unbuffered echo brackets it, so it is
  both bounded and observed. Round 12 ruled the `~15 s` interval untouched *"under either reading,
  with `~12 s` of headroom"*, and it is untouched under the corrected one too.

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
lineage(s) that passed their instrument gates, each named explicitly in the verdict together
with the dataset(s) on which it passed** — as §5.6 requires (round-6 I-2, unmade in v7, found by
round-7 C-3; markup unbalanced in v8, round-8 M-4) — together with the `GATE-DOMAIN` recovery fraction and the raw-vs-head `endpoint_std`
comparison. *(On a CLOSE the per-lineage dataset list is degenerate — §5.6 rule 2 requires both
lineages to have passed on both datasets — but the requirement binds the SURVIVE and HALT faces too,
where it is not degenerate, so it is stated unconditionally.)* It does **not**
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
  table lists only `orthrot_83p8` and `orthrot_72p7`, while `C01_A0_OUT.json` records **4 of 6**
  HateMM rotations and **2 of 6** ZH rotations below the primary — HateMM `17.6/29.1/60.4/72.7` at
  `0.8505 < 0.8598`, ZH `8.3/29.1` at `0.8462 < 0.8590` (round-8 M-2: the counts are attributed to
  the OUT file, which is where they can be checked).

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

**Banked artifacts read by gates — digests added (round-7 I-3).** v7 named these without digests
while `GATE-SHA` asserts that everything in §11 matches. `GATE-FLOOR`'s six anchor triples are read
out of the arena OUT files and `GATE-FOLD`'s parity is asserted against the `vsw_ckpt` npz
(`headspace_mint.py:209-216`), so a silent change to either was the one provenance failure the
design could not detect. Note the path prefix: `headspace_mint.py:209` resolves
**`scripts/analysis/vsw_ckpt/<ds>/`**, and that directory also contains `en`, `st_A`, `st_B` and
`st_B0` siblings, so the unqualified form was ambiguous to an operator.

| file | sha256 |
|---|---|
| `scripts/analysis/headspace_arena_hatemm_s0_OUT.json` | `d0352b5aa69c78cb5a8785655572f12771cca0a8f1ee44022b3c0080b87a8ca4` |
| `scripts/analysis/headspace_arena_hatemm_s1_OUT.json` | `9a435fe57004a537cf3810831ccf40e76f391cd8b721dc43b78df69b7ecdb3d2` |
| `scripts/analysis/headspace_arena_hatemm_s2_OUT.json` | `26c7a72c8291419648fc24ffea5a2ccf60485097b96fbcc0acd440ca9882b472` |
| `scripts/analysis/headspace_arena_zh_s0_OUT.json` | `3d91f51ed2b31334ebec21e36bdcac21e130df8978ed479b6bd9ad24cba45373` |
| `scripts/analysis/headspace_arena_zh_s1_OUT.json` | `690454e4dec0815b0beb7827d69def6cb65147a5dc121029b7fb1ed137805d88` |
| `scripts/analysis/headspace_arena_zh_s2_OUT.json` | `97c4cda590960d61ceb466126bb22a1dd9ed5fbaebee52983c5c8e1817e2b7d1` |
| `scripts/analysis/vsw_ckpt/hatemm/f0.npz` | `9f6957a548bc6f8eedb8cbdf59af203b9bee6d1d44e72fa4cd3a45e3898d03b4` |
| `scripts/analysis/vsw_ckpt/hatemm/f1.npz` | `d600b8b90026821e7e72bfc2461fadb505a071ef8fa5bff9bcaa37881311a0fe` |
| `scripts/analysis/vsw_ckpt/hatemm/f2.npz` | `58d526d1dce8266618b711b959af9c133f7882863216ebd19c6d74213e0d210a` |
| `scripts/analysis/vsw_ckpt/hatemm/f3.npz` | `8f2200bca467bdeefb08cf448f3cee38956f6b38d48f4b1d00ff2ea686c067fe` |
| `scripts/analysis/vsw_ckpt/hatemm/f4.npz` | `2675a056d1c5e6d328e13df89ff73056677476113fb91b954a08df50e2c97b9c` |
| `scripts/analysis/vsw_ckpt/zh/f0.npz` | `8d44cef7c1327631f1950e8267b95ed41c6e06c0a7182050d8604bb30e5960bb` |
| `scripts/analysis/vsw_ckpt/zh/f1.npz` | `52cd05eb324252ade2b78052276b125ed730dbeb4851acd698cb139e7c53268d` |
| `scripts/analysis/vsw_ckpt/zh/f2.npz` | `d0178cf929c41ee7c7be2e6642101000ac6d099d646784cb1edcaa0e2c839a76` |
| `scripts/analysis/vsw_ckpt/zh/f3.npz` | `9a68e23e4f22109ba09212f8ed9cc4234deadbce6ec04ee3f23e3c9fbc59a530` |
| `scripts/analysis/vsw_ckpt/zh/f4.npz` | `d9abd15eee49fc0f21c66781ee75df7c3446c8720ecba63ac64322f9bb85d4b6` |

**Digest-free by declaration:** the 36 banked
`artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz`. They are inputs to §6.1's **reference
measurement** only — no gate reads them — and `GATE-SHA`'s scope is stated in §6 as the frozen
imports and the input caches plus the sixteen banked artifacts above **and the design document
itself (ERRATUM 2 §3), whose digest `configs/c06/c06_falsifier.json:"design_sha256"` carries and
which the arena hashes on disk and compares, HALTing on mismatch in process 1 of 74**.

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
| `test_label_materialisations` | **by construction, not measured** — `_guarded_open` **raises** on a test path, so no test file is opened at all. The runtime `!= 0` assertion is RETAINED VERBATIM as defence-in-depth | **warranted string, never binding** |
| `mints_present_before_arena` | **66** `.npz` (36 Head-N + 30 Head-R) | yes |
| `dev_path_opens` | **`mints_executed + expected_sha_dev_opens`**, `expected_sha_dev_opens = 4` — round-4 I-5 established that `headspace_fidelity.py` opens **no** `dev_seen` file (it reads only `meta` out of the banked mint `.npz` at `:68` and references no label array at all), so *its* contribution is zero; but round-8 H-1 widened `GATE-SHA` to the input caches, two of which are `dev_seen_*.pt`, and `GATE-SHA` runs in **two** processes ⇒ `2 × 2 = 4`. The arena **derives** both factors from the digest tables and **asserts** them against this declaration rather than reading it (ERRATUM 2) | yes |
| `dev_label_materialisations_outside_decisions` | **by construction, not measured** — no code path increments it. `lab_dev` is written twice in the executed corpus (`headspace_mint.py:323` and `c06_falsifier_mint.py:336`, the latter into the banked `.npz`) and read by no path in the arena or `headspace_fidelity.py` | **warranted string, never binding** |
| `dev_or_test_labels_into_decision_quantities` | **by construction, not measured** — same writes, read by no decision path; the arena never iterates `.npz` keys generically. Assertion RETAINED VERBATIM | **warranted string, never binding** |
| `banked_trainlog_opens` | `GATE-DEVFID` only, `2 × 3` | reported |
| processes reporting | **1 `GATE-SHA` driver leg + 66 mints + 6 fidelity + 1 arena** | yes — HALT on any mismatch |
| predicate coverage | re-derived in-job | reported |

**Why `mints_executed` and not `66`.** `headspace_mint.py:192-194` returns **before** the
`dev_seen` load at `:199` whenever `--out` exists, so on a **resumed** job a skipped mint opens no
dev file. A binding `dev_path_opens == 66` would HALT a legitimate resume — the same class of
self-defeating gate this lineage has removed twice elsewhere. **ERRATUM 2 keeps this warrant and
makes it operational:** the sbatch increments `C06_MINTS_EXECUTED` on exactly the mint driver's own
skip condition — the `.npz` absent at call time (`c06_falsifier_mint.py:218-220` returns at
`MINT-SKIP` **before** the dev load whenever `--out` exists) — so the export counts EXECUTED mints,
never attempts. `MINT_N` counts attempts and is `66` on every run including a resume; exporting it
would HALT a legitimate resume, which is the failure this paragraph exists to prevent. Binding
against the measured `mints_executed`, plus the separate `mints_present_before_arena == 66` assertion and
`GATE-FOLD`'s banked-flag re-read (§3.2), is exactly predictable on fresh, resumed and
partially-resumed runs and after a HALT, and still closes the absence lane.

**Dev labels.** `headspace_mint.py:199` loads the native `dev_seen` unconditionally on every mint
and `:322-324` writes `lab_dev` (the array is at `:323`) into every `.npz`; `--train-cache` does not redirect it. Head-R opens
no `dev_seen_*-ro_*` file. None reaches a decision quantity: at `fold ≥ 0` `dev_sp` is a slice of
the fitting pool (`:223-226`) and only `dv[3]` is written to disk.

**No selection anywhere.** Every threshold in §5 and §6 is C01's frozen value, C09's banked
constant, a population-derived constant frozen in §3.7 from banked label-free arithmetic, or a
declared engineering choice disclosed in §5.9.

---

## 13. Execution boundary, and what the code lineage must verify

**SLURM CPU queue. One submission. 8 CPU / 32 GB. No `--gres`, no `--time`, no array, no
dependency, no requeue.** 74 processes in the order **1 `GATE-SHA` driver leg** → 66 mints → 6 fidelity → 1 arena, with
`GATE-SHA` in the driver leg before any of them **and again in the arena at the point of use** and `GATE-POP` before any population-consuming
gate. All six rounds confirmed the channel; the cloud route is inapplicable because `GATE-FLOOR`
anchors to six floors measured locally on `foscsmlprd01` — and round 6's remint reproducing the
banked keys **bit-exactly on this node** is direct evidence that the local anchor is the real one.

**Not authorized by this document.** Required before anything runs: an independent design review to
GO (0C/0H/0I), a **separate** code/resource review lineage over the executable, and main-dialogue
authorization.

### 13.1 The handoff — **28 items**

Round 6's C-1 found that v6 never edited this section: it was byte-identical to v5, still reading
*"round 3's twelve items plus round 4's six"* while the body cited items 19–22 that did not exist,
so round-5 I-6 was **not adopted in any form**. §13 is the **sole input to the mandatory separate
code/resource review lineage** (R4) — the lineage the campaign's record says caught two
wrong-verdict paths on C09 after seventeen clean design rounds. It is rebuilt here from scratch:
**twenty-six items**, with rounds 5 and 6's extensions folded into items 5, 10, 15 and 16 rather
than left as loose prose; round 12 adds a twenty-seventh and round 14 a twenty-eighth, so the list
now runs **1–28**. Every internal reference in this document to a `§13 item N` points into
this list.

**Rounds 3–4 (items 1–18), carried verbatim.**

*The shared mint driver.* **(1)** It imports `headspace_mint` with its sha256 asserted and **no**
behaviour outside `--train-cache` differs between lineages — fold-parity assertion, dummy
construction, `torch.load` guard, seeding and DET-1 are the frozen ones, not re-implemented.
**(2)** `--train-cache` overrides **only** the training cache and cannot reach `model_name`, the dev
load or the dataset table, and §12's declared counts match what the code does. **(3)** There is **no
branch conditional on the cache filename or suffix** — `GATE-FLOOR` exercises the native path only,
which is why `GATE-SHA` over the ro caches and `GATE-IDPARITY` are load-bearing. **(4)** The
`GATE-FLOOR` mints and the Head-R mints go through the *same* function, not two copies.

*Populations and constants.* **(5)** §3.7 now has **two** blocks with **two different verbs** (round-8 H-2). *(a)* Every
**population-derived** constant — arena size, class counts, majority, `GATE-ARENA` band,
`GATE-DOMAIN`'s two majorities, the tie cap, S7's quantile threshold — is **computed from the arena,
not read**. *(b)* Every **frozen C01 config constant** — the `<=` small-set operator,
`tiny_displacement_epsilon = 0.001`, `max_tiny_displacement_fraction = 0.05` and
`normalization_epsilon = 1e-12` — is **read from the sha-gated config and asserted equal to it**,
never recomputed. v8's item 5 listed the `<=` operator among things to compute from the arena, which
is not a thing that can be computed from anything. **(6)** `GATE-SELFTEST`'s `n` is the arena size and no banked
`744` leaks into any per-item denominator. **(7)** `ρ` is computed over the `743/579`-row matrices,
not a 744-row array with a masked row left in (a `1.301e-03` shift — fail-safe, but presenting as an
unexplained HALT).

*The mask convention and the epsilon.* **(8)** Every `prepare_views` call passes an explicit boolean
array and every `l2_rows` call's mask matches the population it is handed — **with an assertion, not
a comment**. **And every `l2_rows` call in the head-space builder is passed
`normalization_epsilon = 1e-12` read from the sha-gated config and no other value** (round-8 I-3):
the two-block build inherits it via `prepare_views`, the one-block build has no config to read it
from, and `GATE-C01PARITY` cannot detect a wrong epsilon because outputs are epsilon-independent
unless a row `die()`s.
**(9)** The `n = 744` build exists **only** inside `GATE-C01PARITY`/`GATE-ROWSUBSET` and nothing
votes on it.

*The tie diagnostic.* **(10)** Which ranking and which residual the implementation uses; that the
residual is the **head-space** `‖Δk‖₂` (or its `√d` bound); that "collapse" is the worst case over
orderings **and is computed ANALYTICALLY, not by enumerating the `g!` orderings of a near-tie group**
(round-8 I-2); that the report branch cannot be reached outside the tie set or above the cap; **and that
mismatches are aggregated per `(dataset, seed, lineage)` pooling the five folds, so the denominator
is `n_D`** (round-5 H-3).

*`GATE-POP` and heartbeat.* **(11)** `GATE-POP` runs **before** any gate consuming a
population-derived constant and asserts row identity by **index set**. **(12)** All six §9 items plus
the `RuntimeError` wrapper, the `buffering=1` handle never re-wrapped, the unbuffered driver echo,
append-without-interleaving across all 74 processes — of which the 68 this lineage authors append
through the `buffering=1` handle, the six sha-frozen `headspace_fidelity.py` processes having their
line written by the bash driver (`sbatch:128-129`) — and the frozen `elapsed ÷ projected`
denominator, which has ONE source and is asserted three ways.

*Round 4's six.* **(13)** **The fold axis** — the 13 arms and every `ρ` are rebuilt from **each** of
the 60 fold key matrices, and no arm built under head `f` is ever voted for a query outside fold
`f`'s held-out fifth. **(14)** **The guard arms** — `orthrot_0` and `orthrot_45` are built by the
*rotation* route and `endpoint_concat` / `common_displacement` by their own, never aliased, and all
four voted. **(15)** **S7** — its classification as a SURVIVE condition, its
`common_displacement`-only arm scope, the zero-fix convention, **and its full parameter set: the
dominance threshold `0.5`, the reference-selection rule of §5.2.1, the head-space one-block
statistic, the `<=` operator and the per-seed `3/3` axis** (round-5 C-3) — none of which the battery
inherits, because **it never *calls* `displacement_audit`** (round-7 M-1 limb 2, unlanded in v8:
§11 *does* import `c01_policy_contrast_a0.py`, which contains that function; what the battery does
not do is call it). **Also: `tiny_ok`'s non-carriage (§5.2.3) and its two constants.** **(16)** **The statistics** — that the
bootstrap statistic, the one-sided `p` and the Holm step-down match §5.4; **that S5's null
construction, statistic, p95 convention, p-value form and 4-member family match §5.4.1**; **that
`GATE-SELFTEST` is asserted per seed as `net_s(A) = n_D · (acc_s(A) − acc_s(reference))`**
(round-5 I-1); and that no **decision** quantity — not only no gate quantity — reaches a comparison
non-finite. **(17)** Population-derived constants, extended, as item 5 now states. **(18)**
**`GATE-FOLD` under resume** — fold parity verified for all 66 mints including skipped ones, by
reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of` from the banked `.npz`.

**Round 5's four (items 19–22) — absent from v6, added here.**

**(19) The one-construction claim, and the two properties that make it checkable.** The two-block
and one-block builds are the **same function** called with different block lists; **no separate
head-space builder exists**; and `GATE-C01PARITY` runs against that function, not a copy. This is
the sole warrant transferring the parity guarantee into head space — the head-space arms have **no
other anchor anywhere in the design**. Two properties the lineage must check explicitly
(round-7 C-1 limb 3, unlanded in v8 and found by round-8 C-2):

* **Endpoint pre-normalisation.** The contrast blocks are formed from `l2_rows`-**normalised**
  endpoints — `std[m] := l2_rows(standard[m])`, `ow[m] := l2_rows(oneword[m])`, matching
  `prepare_views:1296-1304` — in **both** the two-block and the one-block instantiation. A builder
  that omits this is wrong by `1.878e-06` / `1.609e-06` in raw space and by `10⁻²`–`10⁻¹` in head
  space (§3.4).
* **`GATE-C01PARITY` is asserted at bit-exactness**, `max|diff| == 0.0`, **never at a tolerance.**
  A `2e-6` tolerance admits exactly the builder above — wrong by `10⁻¹` in head space with every
  other gate passing.

**(20) The `(dataset, lineage)` cross.** Every per-lineage gate is evaluated per cell; the drop
propagates across datasets per §5.6 (fails on **any** dataset ⇒ dropped on **both**); and no verdict
path can be reached with a lineage that passed on one dataset only.

**(21) The dropped lineage's quantities.** Exempt from the absence rule, excluded from S1–S7 and
from the S5 family, and entering the S4 family **only** as `NOT_TESTED` with `p = 1`; the S4 family
size the code uses is the frozen 92 of §5.5 on every path.

**(22) The key-forward site, and the vote site.** **All three** key forwards — the **native** one
as well as the two ro-cache forwards producing `h_std`/`h_ow` — happen **inside the mint process**,
since `headspace_mint` suppresses state-dict saves and the head weights never leave it, and each
mint writes all of its key matrices into its own `.npz` (`headspace_mint.py:321-325`'s `np.savez`
pattern). §8 Phase 1b's `174` is priced against that placement. **`GATE-FLOOR`'s vote is computed
in the arena process, not in the mint** — so the arena must materialise the 30 Head-N fold mints'
native key matrices off disk, which is why §8 Phase 1f prices `60 × 2 + 30 × 1 = 150`
materialisations and why Phase 2's `30 × U2a` is a **vote-only** timing with no load inside it
(round-10 I-1). A lineage that moves the `GATE-FLOOR` vote into the mint must move those 30
materialisations out of Phase 1f with it; either placement is admissible, but the code and §8 must
agree on **one**, and §8 is written for this one.

**Round 6's four (items 23–26).**

**(23) The arm→formula map is not determined by §3.4's prose.** The map is pinned by
`GATE-C01PARITY` against `prepare_views` and **by nothing else**. The concrete instance two
independent reviewers mis-derived is **`common_interaction`**: it is `paired(common,
l2(common ⊙ displacement))`, **not** `l2(std ⊙ ow)` and **not** `fuse(·)`
(`c01_policy_contrast_a0.py:1246-1265`). Getting it wrong costs `max|diff| = 9.697e-01` against
`prepare_views` and still produces a build that looks like it passes. **A code lineage that
reimplements from this document rather than testing against `prepare_views` will produce a wrong
arm.**

**(24) The reference arm is a run-time decision quantity.** That
`select_strongest_ordinary_control` is called over `decision.gain_controls` within each
`(dataset, lineage)` cell; that **S6, S7 and `GATE-SELFTEST` all consume the same selection**; that
C01's `:2724`-equivalent consistency assertion (S7's reference must equal the selected control) is
implemented; and that the selected arm name is **written to the verdict face**.

**(25) The head-space displacement tail.** That `min_i d_i` and the fraction `d_i ≤ 0.001` are
computed and recorded per `(dataset, seed, fold, lineage)` cell alongside the `GATE-ALGEBRA`
residual, so the max-versus-median gap §7.8 measures is auditable at run time rather than assumed,
and so `tiny_ok`'s non-carriage (§5.2.3) rests on measurement in every cell rather than on four.

**(26) The trained-head blindness boundary.** That the battery computes **no vote on any ro-derived
arm outside the arena phase**, and that the `GATE-FLOOR` phase votes **only on native keys**. This
is the assertion that replaces v6's untrained-head warrant (§7.3), and it is the one a code lineage
can check the freeze record against.

**Round 12's one (item 27).**

**(27) The arena process's import set, because §8 Phase 1g's unit is undeterminable without it**
(round-12 I-1, second repair line — *"the exact analogue of round-11 I-1's second line"*). The arena
process imports, at top level: `numpy`; `sklearn.metrics.roc_auc_score` and
`sklearn.model_selection.StratifiedKFold`; `mechfix_ops`; `mechnov_pairverify`; `vsw_pregate`;
`headspace_mint` (for `det1_assert`, `runtime_block`, `sha256_of` — **which pulls `sklearn` in a
second time via `headspace_mint.py:68`, so `sklearn` is unavoidable even if the direct imports were
dropped**); `torch`; and `c01_policy_contrast_a0`, which this battery adds because §3.4's builder
calls its `l2_rows`. It also calls `runtime_block()`, whose deferred `threadpoolctl`/`scipy`/
`sklearn` work is cheap **only because `sklearn` is already resident**. A lineage that trims this set
must re-measure `U11`'s arena class and re-carry Phase 1g; a lineage that adds to it must do the
same. **Two independent reviewers measured a strict subset of this set and agreed with each other,
which is exactly how a subset measurement survives** — the set is written down here so the next
reader checks the number instead of reconstructing the set from three source files.

**Round 14's one (item 28) — recorded, not prescribed.**

**(28) The `PYTHONPATH` wiring the three-layer test guard depends on.** §12's third layer is the
frozen `c09_guard` `sitecustomize`, which installs an `open()`-level, repo-scoped predicate at
interpreter startup in **every** process. It only loads if the C06 sbatch exports `PYTHONPATH` so
that `c09_guard` is on it, and that module's docstring still names **C09's** sbatch. Round 14 raised
this in its guidance to the separate code/resource review lineage and was explicit that it is *"a
code-side wiring item, not a design defect, since both guard files are sha-frozen in §11"* — so it is
**not** a finding and nothing in §12 changes. It is recorded here because §13 is the sole input to
that lineage and a note living only in a review file would be lost. The lineage must verify the
export exists and that layer 3 is actually active in all 74 processes, not merely importable.


---

## 14. Cumulative disposition — all fourteen rounds

**Every row cites the v14→v15 section(s) implementing it; every limb is quoted VERBATIM AND IN FULL
from round 14's review, with the line numbers of the sentence it was taken from; §14.1 prints the
scripted audit that checks both.** No blanket adoption claim is made anywhere in this document.

### Round 14 (1 Important + 2 Minor) — row level

| finding | disposition | §(diffed) |
|---|---|---|
| **I-1** §14.1's *"The same vacuity holds for v14"* and §15 item 5's *"no row or limb cites §14.1"* are **false**, contradicted by §14.1's own embedded transcript, and understate the mechanism: the plain splice exits `1`, not `0` | **ADOPTED, both sentences, and derived from this document as finalized** — §14.1 now states what the plain construction does against **v15**, computed after the §14.2 fixed point rather than carried over, with the ordering stated in the section so the fix cannot itself become an unchecked inheritance; §15 item 5 asks round 15 to run **both** forms and report both | §14.1, §15 |
| **M-1** §14's widening paragraph and §15 item 2 quote round 13's framing sentence with *"one line"* dropped, which is a **second** dimension of the same deviation | **ADOPTED** — both sites now quote *"Repair — one line, arithmetic only, no new measurement"* in full, and the widening is declared against the length as well as the method: §7.7 grew `1583` characters and §7.9 `949`, both printed in this document's own transcript | §14, §15 |
| **M-2** §7.9's decomposition of v13's `52` cannot be checked against v13's own §7.7, which prints `44` | **ADOPTED** — the bridge is stated and is arithmetically checkable against v13's printed table: an **eighth** rung (arena's actual set `+ c01` but **without** `runtime_block()`) was run and never printed as a row (`+4`), and rung 5's `(10 runs)` was the second command's **increment** where rung 7's `(14 runs)` was a **total**, so rung 5's true total is `14` (`+4`). `44 + 4 + 4 = 52` | §7.9 |

**On the ordering, because round 14's repair turns on it.** The two corrected sentences are the last
substantive edits in this document. The sequence was: finish every other edit; run §14.2 against the
finished on-disk v15 to its byte-identical fixed point; **then** splice v14's §14.1 into a scratchpad
copy of the finalized v15 and run the unmodified script on it; **then** write §14.1's sentence and
§15 item 5 from that output; then re-run §14.2 and re-verify the fixed point. §14.1 states this
sequence in place. The point of the sequence is that the repair for an unchecked inheritance must
not be written from an expectation, and *"the limbs land elsewhere"* was an expectation that had been
true two versions earlier.

### Round 14 — LIMB level, quoted verbatim and in full

Each limb is round 14's own sentence, in quotation marks, with the line range in
`C06_FALSIFIER_PREREG_REVIEW_R14.md` it was taken from. Emphasis and quote-glyph style are
normalised; **no word is dropped, including qualifying clauses.**

LIMB-TABLE-BEGIN

| finding | limb, quoted verbatim from round 14 | landed in |
|---|---|---|
| **I-1** | *"In §14.1 (`v14:2003-2004`), replace "The same vacuity holds for v14, whose limbs land elsewhere; §15 carries the biting form forward" with the fact: "Against v14 the plain construction is no longer vacuous — round-13 M-2's row and limb both cite §14.1 — so splicing v13's §14.1 into v14 yields `UNCHANGED §14.1`, `FAIL M-2 cites §14.1 -- NOT DIFFED`, one failing limb, `named by a row but unchanged: ['14.1']` and exit `1`, with no synthetic row required. v14 is the first version since v11 for which that holds.""* — `R14:650-655`. **The fact is landed for v15 rather than v14, and measured rather than transcribed**: the sentence names round-14 I-1's own row and limb as the §14.1 citers, and reports the splice output this document actually produced against its own finalized text. Writing round 14's v14 sentence verbatim would have re-committed the defect it names — a claim about the artifact carried over instead of checked | §14.1 |
| **I-1** | *"In §15 item 5 (`v14:2293-2295`), strike "The plain splice is vacuous against v14 as it was against v12 and v13 — no row or limb cites §14.1" and ask round 15 to run **both** forms and report both, noting that the plain form is expected to exit `1`."* — `R14:656-658` | §15 |
| **M-1** | *"Repair: quote the framing sentence in full where it is quoted at all — "Repair — one line, arithmetic only, no new measurement" — so the widening is declared against both of its clauses."* — `R14:674-675`. Both sites; and the length clause is now answered explicitly with the two section growths | §14, §15 |
| **M-2** | *"Repair: one clause naming the bridge, e.g. "v13's table printed `44` across seven rungs; the executed sample was `52`, because an eighth rung went unreported and rung 5's `(10 runs)` was the second command's increment rather than its total of `14`." Or drop the decomposition and state the spend as `≈ 2 / ≈ 4` with the `52` unelaborated."* — `R14:691-695`. **First disjunct taken**, with round 14's diagnosis confirmed against this designer's own record of the two v13 commands: the eighth rung was the arena's actual set plus `c01_policy_contrast_a0` **without** `runtime_block()`, and the increment/total mismatch is exactly as round 14 reconstructed it from the outside | §7.9 |

LIMB-TABLE-END

**Round-14 rulings accepted without change.** The science layer is closed — 13/13 arms from the prose
alone at `0.000e+00`, the un-normalised misreading at `1.878e-06` / `1.609e-06` under `2e-6`,
`GATE-ROWSUBSET` at `0.000e+00`, 26/26 `ρ_raw` at 6 dp with the `1.301e-03` masked-row shift
re-derived as `0.968176 × (1 − 743/744)`, 16/16 accuracies and 16/16 net-fix integers, 37/37 digests,
Holm and its three-way equality, `0/18` trained heads, all twenty gates unable to fire on a warranted
CLOSE, and the verdict path total and mutually exclusive. Also accepted: **round-13 I-1 is
discharged** and **the widening warranted** (round-14 C.1, *"and I would have done the same"*);
**correcting v13's recorded spend was right** and is *"the campaign's discipline, not an unusual
step"* — with round 14's general rule adopted here in terms: *a prior version's figure may be
corrected in place when it feeds a live sum, provided the correction states what the earlier version
got wrong and how the two reconcile*, the second clause being what M-2 supplies; **no eleventh
uncounted item** on the newly-searched axis; and **§6.1's gap factor is the ratio of order
statistics**, which is what its own sentence asserts and the reading that reproduces — round 14's
caution that the per-cell ratio is a different quantity spanning `2.45×`–`9.45×` is recorded here so
a later round does not re-derive the wrong one.

**Round 14's corroborations, recorded because §7.7 asks for them.** `U4`'s one plausible unpriced
companion — C01's `id_hash_permutation`, which §7.7's description of `U4` does not name — is measured
at `1.216–1.273 ms` per draw, `3.82 s` over all `3072`, **`0.13 %` of the then-current `2934.5 s` total** *(round-14's own figure, historical: against ERRATUM 2's `3674.0 s` the fraction is `0.10 %`, and the conclusion below only strengthens)* and
inside the row's printed precision, with at most `1536` distinct permutations actually required.
`U8` measures `0.0106–0.0287 s` against its frozen `0.033 s`, and a `B = 2000` resample measures
`0.023–0.028 s` against `U3`'s frozen `0.126 s`. **Both frozen units are conservative.** Together
with round 13's `U2a`–`U2d` and `U7`, §7.7's uncorroborated list is now reduced to `U4` end-to-end,
whose only identified companion is priced above.

### Rounds 1–13 — carried

Round 14 audited v14's round-13 block at limb level by subtraction and found **6 of 6 limbs faithful
and complete**, with the two Minors' residues non-prescriptive and I-1's residue being the framing
sentence — *"not purely non-prescriptive"*, since it carries the method constraint the repair does
not satisfy. v14 quoted and ruled on that clause rather than hiding it; **round-14 M-1 is that the
quotation dropped *"one line"***, which this version restores.

Earlier rounds stand as recorded in v3/v4/v6/v7/v8/v9/v10/v11/v12/v13/v14 §14; where a later round
refined an earlier disposition, the later section governs. Round-9 M-2's second limb, restored in
full by v11 after round-10 I-2 found v10 had truncated it, reads: *"or widen the pattern to bare
`item N` with the §5.9/§15 item references excluded by their own prefixes"* — implemented, and
verified by rounds 11 through 14 against their own scans.

**Rulings carried without change across all rounds.** The direction of *"conservative"* (§4); A7 is
not an obstacle; per-arm retraining excluded; `max` as `ρ*`'s order statistic; SLURM and the
login-node dismissal; HALT semantics; §5.9 item 1's inapplicability reasoning; the tie cap's
one-directionality; `GATE-ROWSUBSET`'s renaming; §3.4's account of what two-block parity does and
does not buy; `B = 2000`; the 92-family's auditability warrant; and `GATE-C01PARITY` at
bit-exactness.

**Struck, and not carried:** *S6's net-fix reference* (overturned by D-1, §5.2.1) and *the
untrained-head blindness discipline* (re-scoped by round-6 H-2, §7.3).

### 14.1 Mechanical disposition verification

**The convention, and what it does and does not cover (round-9 H-2, verified by round 10).** §14.1
is **self-excluding for its SIZE ONLY**: the audit never prints §14.1's byte count, because pasting
the transcript changes the length of the section being measured. **It does not exempt §14.1 from the
changed/unchanged check.** v9's script contained a fallback that added §14.1 to the touched set when
it was *unchanged*, so a row citing §14.1 could pass without §14.1 having been edited. v10 replaced
it with a status computed from an actual comparison — `CHANGED`, `UNCHANGED` or `ADDED` — that adds
§14.1 to the touched set only on a genuine change. **Round 10 broke it in the direction that
matters and reported the result**: on a counterfactual whose §14.1 is byte-identical to v9's, the
script printed `UNCHANGED §14.1 (self, size not reported)`, **failed** the two §14.1-citing rows and
the one §14.1-citing limb, reported `named by a row but unchanged: ['14.1']` and exited `1`; against
the real v10 its output was byte-identical to the embedded transcript. Round 10 also ruled size-only
*"the right and minimal line"* and confirmed no other convention is load-bearing for another row's
verification. **Round 11 repeated the construction against v11** — splicing v10's §14.1 in — and got
`UNCHANGED`, two failing rows, two failing limbs, `named by a row but unchanged: ['14.1']` and exit
`1`. **Round 12 found the plain construction vacuous against v12** — no v12 row and no v12 limb
cites §14.1, so there was nothing for it to bite on — and proved the mechanism live instead by
splicing v11's §14.1 in **and** inserting one synthetic §14.1-citing row, which produced
`UNCHANGED`, `FAIL X-9 cites §14.1 -- NOT DIFFED`, `named by a row but unchanged: ['14.1']` and
exit `1`. **Round 13 ran both forms against v13 and reported both**: the plain splice vacuous at
exit `0`, the biting form failing one row at exit `1`. **Against v15 the plain construction is not vacuous, and this is measured against this document as finalized rather than carried forward (round-14 I-1).** Round-14 I-1's own disposition row and its own limb both land in §14.1 and cite it, so splicing **v14's** §14.1 into a copy of the finalized v15 and running the unmodified §14.2 script yields `UNCHANGED §14.1 (self, size not reported)`, `FAIL I-1 cites §14.1 -- NOT DIFFED`, `rows verified against diff hunks: 2 ; rows failing: 1`, one failing limb, `named by a row but unchanged: ['14.1']` and **exit `1`** — with no synthetic row required. Adding one anyway (the biting form) gives a second failing row `X-9` and changes nothing else: `rows failing: 2`, exit `1`. **v14 was the first version since v11 for which the plain form bit, and v15 is the second** — because in both, a finding about this section is dispositioned in this section. *(v14 asserted the opposite — that the plain form was vacuous — and its own transcript printed the citation that refuted it. That sentence was an unchecked inheritance from v12 and v13, where it had been true. The ordering §14 records exists so this sentence is not one: the fixed point was reached first, the splice run second, and this sentence written third, from that output.)* The logic is
unchanged in v15.

**How the transcript below was produced.** The script was executed against the **finished, on-disk
v15** as the last action before this document was closed, then re-executed against the document with
the transcript already embedded; the two outputs are byte-identical, so the transcript is a
**verified** fixed point — the seventh consecutive version for which that holds, and round 12 verified
v12's independently at 1733 bytes with a matching sha256 and round 13 verified v13's at 1971 bytes,
both having confirmed by direct diff that the `CHANGED §14.2 +0 chars` line is **four substitution
classes** — the version string, the diff label, `V_OLD`/`V_NEW`, and the two `print` headers — all
same-length. *(Round-13 M-2: those four classes occupy **six** lines, not the "exactly five" round 12
wrote and v13 restated; round 13 counted six in both the v11→v12 and v12→v13 diffs, and I reproduce
six for v13→v14. The class count is the checkable statement and the line count was inherited without
being checked.)* Round-10 M-2's note that moving
the printed labels moves the transcript is the reason the re-run comes after the label edits, not
before. v12's drafting also writes after every edit and verifies each write.

**What the audit does.** (1) Splits v14 and v15 into sections and diffs them, printing §14.1's
status without its size. (2) For every §14 row, checks each cited section appears in the diff.
(3) Flags any row whose cited section shows no diff. (4) Resolves references — emphasis-tolerant,
and for §13 items using **round-9 M-2's prescribed form**: bare `item N` is matched, including
comma/and lists and en-dash ranges, and a reference carrying a `§`-section prefix other than `§13`
is excluded **by that prefix** (round-10 I-2 — v10 used a six-verb whitelist instead, and printed a
scope narrower than what it scanned). Every reference line names its scope, and the scope is now
true of the pattern. Dotted `§N.N` and bare `§N` are reported separately. (5) **Limb level**: parses
the `LIMB-TABLE` and checks each limb's landing section diffed. (6) **Coverage both ways**: sections
that changed but no row or limb cites, and sections a row names that did not change.

**What the audit still cannot do, stated because rounds 9 and 10 each proved it by example.** It
verifies limbs against **this document's transcription** of the prescriptions. Verbatim quotation
with provenance makes a narrowing *visible* — round 10 found v10's one truncation by subtracting the
quotations from round 9's Repair paragraphs — but the check that a quotation is faithful and
complete remains a **reading** obligation and belongs to the reviewer, not to the script. Round 11
executed that obligation in full and reported *"13/13 limbs faithful and complete"*; round 12
repeated it on v12's four and reported both Repair paragraphs *"subtracting to bare scaffolding"*;
round 13 repeated it on v13's five and found them all faithful, noting that the limb v13 deviates
from was quoted **in full including the three clauses its own deviation contradicts** — *"quoting
the prescription that convicts you is the opposite of a narrowing."* Round 14 repeated it on v14's
six and found them all faithful too, and its one Important was **not** a limb defect but a false
sentence about this section. §15 puts the same obligation to round 15.

**Output, verbatim:**

```
=== (1) SECTION DIFF v14 -> v15 ===
  CHANGED  §14       -482 chars
  CHANGED  §15       +540 chars
  CHANGED  §7.3      +317 chars
  CHANGED  §7.9      +1403 chars
  CHANGED  §13.1     +968 chars
  CHANGED  §14.2     +0 chars
  CHANGED  §14.1     (self, size not reported)
  CHANGED  header    +396 chars
  UNCHANGED: 50 sections

=== (2)+(3) DISPOSITION ROWS vs DIFF ===
  OK    I-1   cites §14.1, §15
  OK    M-1   cites §14, §15
  OK    M-2   cites §7.9
  rows verified against diff hunks: 3 ; rows failing: 0

=== (4) REFERENCE RESOLUTION (emphasis-tolerant; dotted AND bare) ===
  dotted §N.N refs (scope: in-document sections): 33 ; unresolved: NONE
  bare   §N   refs (scope: top-level sections):   12 ; unresolved: NONE
  §13 item refs (scope: every item/items reference carrying a number, a comma/and list or an en-dash range; a reference prefixed by a § section other than §13 is excluded by that prefix)
    30 reference sites -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28] ; defined 1..28 ; unresolved: NONE

=== (5) LIMB-LEVEL DISPOSITION (round-14 prescriptions) ===
  OK    I-1   *"In §14.1 (`v14:2003-2004`), replace "The same vacu -> §14.1
  OK    I-1   *"In §15 item 5 (`v14:2293-2295`), strike "The plain -> §15
  OK    M-1   *"Repair: quote the framing sentence in full where i -> §14, §15
  OK    M-2   *"Repair: one clause naming the bridge, e.g. "v13's  -> §7.9
  limbs landed: 4 ; limbs open/failing: 0

=== (6) CHANGED-BUT-UNCITED / NAMED-BUT-UNCHANGED ===
  changed but cited by no row/limb: ['13.1', '14.2', '7.3', 'header']
  named by a row but unchanged:    NONE
```

**Reading it.** All three round-14 rows cite sections the diff shows as changed, and all four limbs
name sections that diffed. Step (6)'s *named-but-unchanged* list is **empty**. The row scan excludes
the limb-table region, so the two counts (**3 rows**, **4 limbs**) are disjoint.

**Two things in this transcript will look odd on a first read, and each is a measurement.**

* **`CHANGED §14.2 +0 chars`, for the fourth consecutive version.** Every §14.2 edit this round was
  again a same-length substitution: **four substitution classes** — the version string, the diff
  label, `V_OLD`/`V_NEW`, and the two `print` headers — occupying **six** lines. Rounds 12, 13 and 14
  each verified the equality by direct diff, and round 14 printed the six lines with their class
  labels at its A.2, confirming *"four classes, six lines, every line the same length."* The
  `CHANGED` verdict is computed from a content comparison (`SA[k] != SB[k]`), never from the size.
* **`defined 1..28`.** §13.1 gained item 28 this round — round 14's `PYTHONPATH`/`c09_guard` note to
  the code lineage, recorded rather than prescribed — so the contiguity check runs to 28.

**The audit earned its keep four rounds ago, and the record of that stands.** A first run against a
complete v11 reported `FAIL I-1 cites §13 -- NOT DIFFED` and
`named by a row but unchanged: ['13']`: round-10 I-1's second half had landed in **§13.1**, where
item 22 lives, while the row and the limb cited **§13**, whose preamble had not changed. The
citation was corrected to §13.1 — step (3) and step (6) catching a real mis-citation before a
reviewer had to.

*(The `§13 item` scan.* It uses round-9 M-2's **prescribed** mechanism: bare `item N` is matched,
comma/and lists and en-dash ranges are expanded, and a reference carrying a `§`-section prefix other
than `§13` is excluded by that prefix. Round 11 verified it by an exhaustive scan of all `107`
`item`/`items` occurrences and **ruled that range headings should keep being expanded**, because
that is what makes `unresolved: NONE` assert that every item the document's own structural headings
claim to exist is actually defined — *"precisely the check that would have caught round-6 C-1."*
Rounds 12, 13 and 14 each re-scanned and each confirmed `defined 1..N` contiguous. The count
includes this transcript's own previews and §14.2's reproduction of the script, because the scan
reads the whole document including this section.)*

### 14.2 The audit script, in full (round-8 M-6)

Reproduced so the claim *"re-runnable"* is one a reader can act on. It is a drafting instrument: it
reads two draft files, writes nothing, and touches no repository artifact. **The self-exclusion
logic is the two-line correction round-9 H-2 prescribed** — status computed, not asserted.

```python
#!/usr/bin/env python3
"""Mechanical disposition verification, C06 falsifier v15.
(1) section diff v14->v15 (2) every §14 row cites a diffed section   (3) rows failing that are flagged
(4) reference resolution (emphasis-tolerant, dotted AND bare)        (5) LIMB-LEVEL disposition
(6) changed-but-uncited / named-but-unchanged                        §14.1 is SELF-EXCLUDING for SIZE.
"""
import re, sys
V_OLD='/data/jehc223/RGCL/refine-logs/C06_FALSIFIER_PREREG_DRAFT_V14.md'
V_NEW='/data/jehc223/RGCL/refine-logs/C06_FALSIFIER_PREREG_DRAFT_V15.md'
a=open(V_OLD).read(); b=open(V_NEW).read()
SELF='14.1'   # self-excluding: its own size is never reported (round-8 C-1 fixed point)

def sections(t):
    idx=[(m.start(), m.group(1)) for m in re.finditer(r'(?m)^#{2,3} (\d+(?:\.\d+)*)[. ].*$', t)]
    return {num: t[pos:(idx[k+1][0] if k+1<len(idx) else len(t))] for k,(pos,num) in enumerate(idx)}
SA,SB=sections(a),sections(b)
def preamble(t):
    m=re.search(r'(?m)^## 1[. ]', t); return t[:m.start()]
HDR = preamble(a)!=preamble(b)

print('=== (1) SECTION DIFF v14 -> v15 ===')
changed=set(); added=set()
for k in sorted(SB, key=lambda x:(len(x),x)):
    if k==SELF: continue                      # printed separately below, size never reported
    if k not in SA: added.add(k); print('  ADDED    §%-8s %d chars'%(k,len(SB[k])))
    elif SA[k]!=SB[k]:
        changed.add(k); print('  CHANGED  §%-8s %+d chars'%(k,len(SB[k])-len(SA[k])))
# self-exclusion covers the SIZE ONLY, never the changed/unchanged fact (round-9 H-2)
if SELF in SB:
    st = 'CHANGED' if (SELF in SA and SA[SELF]!=SB[SELF]) else ('ADDED' if SELF not in SA else 'UNCHANGED')
    print('  %-8s §%-8s (self, size not reported)'%(st,SELF))
    if st=='CHANGED': changed.add(SELF)
    elif st=='ADDED': added.add(SELF)
print('  %-8s header    %+d chars'%('CHANGED' if HDR else 'UNCHANGED', len(preamble(b))-len(preamble(a))))
print('  UNCHANGED: %d sections'%len([k for k in SB if k in SA and SA[k]==SB[k] and k!=SELF]))
touched = changed|added|({'header'} if HDR else set())

def cited_secs(cell):
    s=re.findall(r'§(\d+(?:\.\d+)*)', cell)
    if 'header' in cell: s=s+['header']
    return s
def rows_of(sec, exclude_limbs=True):
    if exclude_limbs:
        m=re.search(r'(?s)LIMB-TABLE-BEGIN.*?LIMB-TABLE-END', sec)
        if m: sec=sec[:m.start()]+sec[m.end():]
    out=[]
    for line in sec.split('\n'):
        m=re.match(r'^\| \*\*([A-Z]-\d+|M-\d+|D-\d+)\*\*', line)
        if not m: continue
        cells=[c.strip() for c in re.split(r'(?<!\\)\|', line) if c.strip()]
        out.append((m.group(1), cells[-1] if cells else ''))
    return out

print()
print('=== (2)+(3) DISPOSITION ROWS vs DIFF ===')
ok=bad=0
for tag,cites in rows_of(SB.get('14','')):
    secs=cited_secs(cites)
    if not secs: print('  SKIP  %-5s (no section cited)'%tag); continue
    miss=[x for x in secs if x not in touched]
    if miss: print('  FAIL  %-5s cites §%s -- NOT DIFFED'%(tag,', §'.join(miss))); bad+=1
    else: print('  OK    %-5s cites §%s'%(tag,', §'.join(secs))); ok+=1
print('  rows verified against diff hunks: %d ; rows failing: %d'%(ok,bad))

print()
print('=== (4) REFERENCE RESOLUTION (emphasis-tolerant; dotted AND bare) ===')
allsec=set(SB)
def qualified(pos):
    pre=b[max(0,pos-45):pos]
    return ('round-' in pre[-12:]) or ('round ' in pre[-12:]) or ('.md' in pre[-45:])
dotted=sorted({m.group(1) for m in re.finditer(r'§(\d+\.\d+(?:\.\d+)*)', b) if not qualified(m.start())})
bare  =sorted({m.group(1) for m in re.finditer(r'§(\d+)(?!\.\d)', b) if not qualified(m.start())})
top={k.split('.')[0] for k in allsec}
un_d=[r for r in dotted if r not in allsec]; un_b=[r for r in bare if r not in top]
print('  dotted §N.N refs (scope: in-document sections): %d ; unresolved: %s'%(len(dotted), un_d or 'NONE'))
print('  bare   §N   refs (scope: top-level sections):   %d ; unresolved: %s'%(len(bare), un_b or 'NONE'))
# round-9 M-2 as prescribed (round-10 I-2): bare 'item N' IS matched, and a reference carrying a
# §-section prefix other than §13 is excluded BY THAT PREFIX.  No verb whitelist.
ITEM_RUN=re.compile(r'items?\s+((?:\*\*)?\d+(?:\*\*)?(?:\s*(?:,|and|–|—|-)\s*(?:\*\*)?\d+(?:\*\*)?)*)')
SEC_PRE =re.compile(r'§(\d+)(?:\.\d+)*[’\']?s?\s*$')
def run_nums(run):
    out=[]; prev=None; rng=False
    for tok in re.findall(r'\d+|[–—-]', run.replace('**','')):
        if not tok.isdigit(): rng=True; continue
        n=int(tok)
        out += list(range(prev+1,n+1)) if (rng and prev is not None) else [n]
        prev=n; rng=False
    return out
items=[]; sites=0
for m in ITEM_RUN.finditer(b):
    sm=SEC_PRE.search(b[max(0,m.start()-14):m.start()])
    if sm and sm.group(1)!='13': continue
    sites+=1; items+= run_nums(m.group(1))
s13=SB.get('13.1','')+SB.get('13','')
defined={int(x) for x in re.findall(r'\*\*\((\d+)\)', s13)}
un_i=sorted({i for i in items if i not in defined})
print('  §13 item refs (scope: every item/items reference carrying a number, a comma/and list or an '
      'en-dash range; a reference prefixed by a § section other than §13 is excluded by that prefix)')
print('    %d reference sites -> %s ; defined 1..%d ; unresolved: %s'
      %(sites, sorted(set(items)), max(defined) if defined else 0, un_i or 'NONE'))

print()
print('=== (5) LIMB-LEVEL DISPOSITION (round-14 prescriptions) ===')
lt=re.search(r'(?s)LIMB-TABLE-BEGIN(.*?)LIMB-TABLE-END', b)
lim_ok=lim_bad=0
if not lt: print('  FAIL  limb table not found')
else:
    for line in lt.group(1).split('\n'):
        m=re.match(r'^\| \*\*([A-Z]-\d+|M-\d+)\*\* \| (.+?) \| (.+?) \|\s*$', line)
        if not m: continue
        tag,limb,land=m.group(1),m.group(2),m.group(3).strip()
        secs=cited_secs(land)
        if not secs or 'NOT ADOPTED' in land.upper():
            print('  OPEN  %-5s %-52s -> %s'%(tag,limb[:52],land)); lim_bad+=1; continue
        miss=[x for x in secs if x not in touched]
        if miss: print('  FAIL  %-5s %-52s -> §%s NOT DIFFED'%(tag,limb[:52],', §'.join(miss))); lim_bad+=1
        else: print('  OK    %-5s %-52s -> §%s'%(tag,limb[:52],', §'.join(secs))); lim_ok+=1
print('  limbs landed: %d ; limbs open/failing: %d'%(lim_ok,lim_bad))

print()
print('=== (6) CHANGED-BUT-UNCITED / NAMED-BUT-UNCHANGED ===')
cited=set()
for _,c in rows_of(SB.get('14','')): cited|=set(cited_secs(c))
if lt:
    for line in lt.group(1).split('\n'):
        m=re.match(r'^\| \*\*[A-Z]-\d+|^\| \*\*M-\d+', line)
        if m: cited|=set(cited_secs(line.split('|')[-2] if line.count('|')>2 else ''))
cbu=sorted(touched-cited, key=str); print('  changed but cited by no row/limb: %s'%(cbu or 'NONE'))
named=set()
for _,c in rows_of(SB.get('14','')): named|=set(cited_secs(c))
nbu=sorted([s for s in named if s!='header' and s in SA and s in SB and SA[s]==SB[s]])
print('  named by a row but unchanged:    %s'%(nbu or 'NONE'))
sys.exit(1 if (bad or un_d or un_b or un_i or lim_bad) else 0)
```

## 15. Open issues for round 15

1. **The self-exclusion, in BOTH forms — and the premise is now stated rather than assumed.**
   Round-14 I-1 was that §14.1 and this item both asserted the plain construction was vacuous when it
   was not. **Run both**: splice v14's §14.1 into a copy of v15 with no synthetic row, then repeat
   with one synthetic §14.1-citing row added, and **report both outputs**. §14.1 states what this
   document measured for **both** forms against its own finalized text — plain: exit `1`, one failing
   row, one failing limb; biting: exit `1`, two failing rows — so **the plain form is expected to
   exit `1`, not `0`**. Check that rather than inherit it, which is the whole lesson of the round.
2. **The widening on round-13 I-1, now declared against both clauses.** Round 13 prescribed
   *"Repair — one line, arithmetic only, no new measurement"*; v14 re-measured, and §7.7 grew `1583`
   characters and §7.9 `949`. Round 14 ruled the widening **warranted** and said it would have done
   the same, decisively because the re-measurement **pooled with** rounds 12 and 13 rather than
   replacing them. Round-14 M-1 was that the quotation had dropped *"one line"*; it is restored.
   **Re-rule it against the full sentence**, length clause included.
3. **v13's `52`, and the bridge.** §7.9 now derives `52` from v13's printed `44`: an eighth unreported
   rung (`+4`) and rung 5's `(10 runs)` being an increment where rung 7's `(14 runs)` was a total
   (`+4`). Check the bridge against **v13's own §7.7 table**, and rule whether round 14's general
   principle — *a prior version's figure may be corrected in place when it feeds a live sum, provided
   the correction states what the earlier version got wrong and how the two reconcile* — is now
   satisfied in both clauses.
4. **The eleventh uncounted item.** Fifteen rounds, ten items. Round 12 searched three axes, round 13
   two, round 14 one (`U4`'s `id_hash_permutation` companion, measured at `0.13 %` of the total).
   Rounds 13 and 14 between them corroborated `U2a`–`U2d`, `U7`, `U8` and `U3`, leaving **`U4` as the
   only substantial unit uncorroborated end to end**. **Name the axis you search**, and consider `U4`
   directly: `273.7 s` is `9.3 %` of the projection and its object is described as *"2 arms × 5 folds
   + rebuild"*.
5. **§13.1 item 28, recorded not prescribed.** Round 14 raised the `PYTHONPATH`/`c09_guard` wiring in
   its guidance to the code lineage and was explicit it is *"not a design defect"*. v15 records it in
   §13 because that section is the lineage's sole input. **Rule whether recording a non-finding in
   §13 is right**, or whether it inflates the handoff list. The list is now `1–28`; confirm
   contiguity.
6. **Is the record still sound, and is the design freeze-ready?** Round 14 answered *"the record is
   sound at limb level"* and *"freeze-ready on everything except I-1"*, calling I-1 *"two sentences of
   prose in §14.1 and §15 that require no new measurement and move no quantity a reader would act
   on."* Both are above. Fifteen rounds is evidence of nothing; round 14 held both lines at once —
   it declined to grant a GO early **and** declined to inflate its finding, grading I-1 Important
   rather than High *"because nothing is narrowed and no quantity moves"*, and raising *"no second
   finding I could not measure."* If round 15 finds the design and the record clean, say **GO**
   plainly; if not, name the specific defect.

---

*No GPU, SLURM, Modal, teacher call, cache write, test-split access, job submission or commit
occurred in producing this document. **v15 trained no heads and took no new measurement of the
instrument**: its work was two prose corrections derived from running the §14.2 script and its
splice counterfactual against v15 itself, one arithmetic bridge from v13's printed `44` to its
executed `52`, and the audit re-runs — all read-only on the drafts themselves. **One
side-effect remains disclosed rather than described away:** **v13's** import timings caused CPython
to write one bytecode cache into the tree,
`scripts/analysis/__pycache__/vsw_pregate.cpython-311.pyc` — `.gitignore`d (`*.pyc`),
machine-generated rather than authored, in a directory that already held eight such files from
earlier rounds. **v14's `56` timed starts wrote no new one**, every module involved being already
cached: the directory holds **11** files before and after, which is what rounds 13 and 14 each
reported for their own timings, and **v15 ran no timings at all**. **No `.py` source moved** — all 38
§11 digests recompute **as of this document's own freeze; the 38th is this document, whose digest is
by construction the one `configs/c06/c06_falsifier.json:"design_sha256"` carries**. Nothing else was written outside `refine-logs/` and the session scratchpad.
The **twelve** CPU head mints across v1–v15 (§7.9's sum) wrote nothing outside the session scratchpad
and computed no battery-arm accuracy. `TARGET_STATE.json` was read and not modified. v1–v14 are
unmodified.*
