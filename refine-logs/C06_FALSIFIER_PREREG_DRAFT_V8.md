# C06 `$0` CPU falsifier — preregistration **DRAFT v8** (2026-08-04)

**SUPERSESSION.** Supersedes `C06_FALSIFIER_PREREG_DRAFT_V7.md` → V6 → V5 → V4 → V3 → V2 → v1. All
seven remain on disk **unmodified** as the record of what each round reviewed. Reviews of record:
`C06_FALSIFIER_PREREG_REVIEW{,_R2,_R3,_R4,_R5,_R6,_R7}.md` — REVISE 3C/6H/10I+4M, 3C/3H/7I+3M,
2C/1H/6I+4M, 3C/3H/8I+4M, 3C/3H/6I+5M, 2C/3H/5I+6M, 4C/2H/3I+4M. Complete standalone document.

**Status: DRAFT, NOT FROZEN, NOT AUTHORIZED, NOT SUBMITTED.** No job exists, no hash is frozen,
`TARGET_STATE.json` is untouched, nothing is committed.

**Disposition: all 13 round-7 findings ADOPTED, 0 rebutted — and this is the first version whose
disposition table is verified mechanically rather than asserted (§14.1).**

**Why the protocol changed.** Round 6 caught v6 claiming repairs in §13/§14 that were never made.
Round 7 caught v7 doing it again in three more places — §5.2.2, §10.2, and a **§5.2.3 that did not
exist** — and correctly recorded that v7's header claim of *"all 17 round-6 findings ADOPTED"* was
false (its audit: 11 adopted, 3 partial, 1 not adopted, 2 adopted-with-a-new-defect). Two
consecutive rounds of bookkeeping asserting edits that were not made is a pattern, not an accident.
**v8 therefore applies the F118 discipline to its own record: every disposition row in §14 cites the
v7→v8 diff hunk that implements it, a scripted audit checks every row and every internal reference,
and §14.1 prints that audit's output.** A row that cannot cite its diff does not ship.

**What v8 changes.** Round 7's four Criticals.

* **C-1 is a wrong-verdict path inside the pin v7 wrote to close one.** §3.4 pinned
  `common[m] = l2(std[m] + ow[m])` without ever defining `std[m]` — and `prepare_views:1296-1304`
  `l2_rows`-normalises every policy/modality block **first**. I reproduced round 7's measurement:
  the un-normalised reading differs from `prepare_views` by **`1.878e-06`** (HateMM) /
  **`1.609e-06`** (MHC-ZH), **both of which PASS** the `2e-6` clause §6's `GATE-C01PARITY` row also
  stated — while in head space, where the head emits an unnormalised 1024-d vector, the same
  misreading moves every contrast arm by `10⁻²`–`10⁻¹` with every other gate passing. §3.4 now
  defines the normalisation step and **`GATE-C01PARITY` is a single predicate: bit-exact
  (`max|diff| == 0.0`)**, with the `2e-6` language struck from that row.
* **C-2/C-3/C-4** — the three places §14 described edits that were not made: **§5.2.3 now exists**
  (`tiny_ok`, both constants), **§10.2** carries the dataset naming §5.6 demands, and **§5.2.2**
  freezes the `<=` operator and records the raw-versus-head dispersion contrast.
* **H-1/H-2** — this header no longer describes a previous version or quotes the retired
  single-cell residual; §7.9 accounts for this round's mints.

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
| **small-set comparison operator** (round-7 C-4) | **`<=`** | **`<=`** | **S7** (§5.2.2) |
| **`tiny_displacement_epsilon`** (round-7 C-2) | `0.001` | `0.001` | **§5.2.3** — the bar `tiny_ok` is *not* carried against |
| **`max_tiny_displacement_fraction`** (round-7 C-2) | `0.05` | `0.05` | **§5.2.3** — absent from v7 entirely |

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
  nowhere in the decision rule). C01 uses `small_mask = dev_min <= threshold` (`:2049`). Because the
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
  `tiny_ok` was slack by ~600× in C01's own run and could never bite.
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
   for symmetry with item 6, since a change in the other direction is disclosed there.
8. **Freezing the S4 family at 92 on every path eases CLOSE (round-6 H-1).** Relative to
   recomputing at 46 on a drop path, the frozen family makes S4 strictly no easier, hence SURVIVE no
   easier, hence CLOSE easier. It is kept for auditability (§5.5), not because it is neutral, and
   it is the direction change round 6 found undisclosed. *(Round-7 M-4: item 6 keeps the
   "largest such item" claim; this item drops its own competing superlative.)*
9. **`tiny_ok` is not carried (§5.2.3), which eases S7 and therefore hardens CLOSE** — the
   conservative direction. Listed for completeness, since items 6 and 8 record changes the other
   way.

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
| `GATE-SHA` | G | every frozen import and input cache matches §11; **once in the sbatch driver** |
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

**The warrant is two-part from v6 onward (round-6 H-2), because v6 was the first draft in which a
trained head touched the ro caches.** (i) Every *untrained*-head leg is scientifically void by
construction — real operations at real scale, meaningless numbers. (ii) The **trained**-head legs
(§7.8) computed **only** the native deployed vote — a banked instrument anchor already published in
§6 — plus key-space algebra residuals and displacement-norm geometry. **No vote was taken on any
ro-derived arm at any point**, which is the assertion that actually secures blindness: it is a
statement about what was computed, not about how a head was initialised. §13 item 26 puts it where
the code lineage can check the freeze record against it. **No arm accuracy has been computed,
printed or recorded at any point in v1–v7.** Rounds 4 and 5 audited this by grepping every decimal
in `[0.6, 0.99]`; round 5 classified all **97** distinct values across v1–v5 and found each to be a
`ρ`, a `‖head_f(0,0)‖` magnitude, a cos/`‖Δ‖` geometry figure, a banked `GATE-FLOOR` anchor, a
published C01 dev-arena accuracy, a majority/band constant or a unit-time string. **v6–v7 add exactly one
measured accuracy and it is not a battery arm** (round-6 M-2 corrects v6's *"two"*): §7.8's
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

**v7's measurements** added ≈ **5 wall-minutes / ≈ 25 CPU-minutes** — round-7 H-2 found v7's
paragraph still describing v6's burn and attributing to v6 a programme four-fifths of which happened
in v7. **Five CPU head mints were trained in v7, not one:** one for §7.8's `GATE-FLOOR` discharge
(`33.5 s`) and **four more** for its four-cell displacement-tail table, spanning both lineages and
both datasets (`≈ 33–40 s` each at §7.2's measured units). v7 also ran the untrained-residual
comparison, the `holm_adjust` executions, the S5 feasibility arithmetic and the C01 source/OUT reads
behind D-1.

**v8's measurements** added ≈ **2 wall-minutes / ≈ 6 CPU-minutes** and **no mints**: the round-7 C-1
reproduction (both readings of the builder against `prepare_views` on both datasets), the I-1
reduction-order comparison, the I-2 loop timings, and the sixteen sha256 of I-3.

**Cumulative v1–v8: ≈ 28 wall-minutes / ≈ 116 CPU-minutes, eleven CPU head mints, all `$0`, zero
GPU.** No battery-arm accuracy at any point. The CPU-cap conflict was knowable from C09's
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
| **1d** `GATE-SHA`, once in the driver | `1` | `U7` | `0.1 s` |
| **1e** `GATE-FOLD`'s banked-`.npz` parity re-read (round-5 I-3), `66 × 0.5 ms = 0.033 s` | `66` | measured | `0.1 s` |
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
| **7z** `GATE-ALGEBRA`'s residual comparison — `np.max(np.abs(·))` over two `(n_D, 2048)` key-difference matrices per cell (**round-7 I-2**, uncounted in v7) | `2 × 60 = 120` reductions | measured | `1.0 s` |
| **7z** §13 item 25's per-cell tail record — `min_i d_i` and `frac(d_i ≤ 0.001)` (**round-7 I-2**, newly required) | `60` cells | measured `0.034 s` | `0.1 s` |
| **7** per-gate arithmetic on materialised vectors — `GATE-SELFTEST` (**`14 × 3 × 2 × 2 = 168`**), `GATE-NESTED`, S7, `GATE-POP` (incl. class counts and constant recomputation), `GATE-NULLREMOVED`, `GATE-IDPARITY`, **`GATE-ZEROMASK`, `GATE-FOLD`'s in-process leg and `mints_present_before_arena`** (round-5 I-3) | all | sub-`0.1 s` class | `0.1 s` |

**Total, re-multiplied:** `2927.6 + 1.0 + 0.1 = ` **`2928.7 s = 48.8 min`** corroborating;
**`× 1.25 = 3660.9 s = 61.0 min`** conservative. (v7's `2927.6` plus Phase 7z's two rows, which
round-7 I-2 found uncounted; measured together at `1.017 s` against Phase 7's stated `0.1 s` bound,
so they are priced as their own line rather than absorbed.) (v5's `2927.5` plus Phase 1e; Phase 1c's count
rises `66 → 67` and its product is `67 × 0.033 = 2.211 → 2.2`, unchanged at one decimal.)

*(Round 4's headline figure was `2925.0 s`; the `2.5 s` difference is the guard-arm
**construction**, which it offered as an option and v5 counts. The direction is conservative.)*

**M-2, adopted:** the printed product column now sums to the total directly, with **Phase 7
carried at its `0.1 s` upper bound**; v4's rounding note described v3's table and is retired.

**Declared slack, outside the projection:** `30 s` for ledger aggregation and JSON emit.

**Peak RSS ≈ 1.3 GiB.** Request 32 GB.

**Where the risk sits.** Mints are `85.6 %` of the total and are measured directly. Phase 3 is
`9.3 %`; a 2× miss moves the total to `3202.4 s = 53.4 min`, a 5× miss to `4023.5 s = 67.1 min`.
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
lineage(s) that passed their instrument gates — each named explicitly in the verdict **together
with the dataset(s) on which it passed**, as §5.6 requires** (round-6 I-2, unmade in v7 and found by
round-7 C-3), together with the `GATE-DOMAIN` recovery fraction and the raw-vs-head `endpoint_std`
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
imports and the input caches plus the sixteen banked artifacts above.

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
and `:322-324` writes `lab_dev` (the array is at `:323`) into every `.npz`; `--train-cache` does not redirect it. Head-R opens
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
gate. All six rounds confirmed the channel; the cloud route is inapplicable because `GATE-FLOOR`
anchors to six floors measured locally on `foscsmlprd01` — and round 6's remint reproducing the
banked keys **bit-exactly on this node** is direct evidence that the local anchor is the real one.

**Not authorized by this document.** Required before anything runs: an independent design review to
GO (0C/0H/0I), a **separate** code/resource review lineage over the executable, and main-dialogue
authorization.

### 13.1 The handoff — **26 items**

Round 6's C-1 found that v6 never edited this section: it was byte-identical to v5, still reading
*"round 3's twelve items plus round 4's six"* while the body cited items 19–22 that did not exist,
so round-5 I-6 was **not adopted in any form**. §13 is the **sole input to the mandatory separate
code/resource review lineage** (R4) — the lineage the campaign's record says caught two
wrong-verdict paths on C09 after seventeen clean design rounds. It is rebuilt here from scratch:
**twenty-six items**, with rounds 5 and 6's extensions folded into items 5, 10, 15 and 16 rather
than left as loose prose. Every internal reference in this document to a `§13 item N` points into
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

*Populations and constants.* **(5)** **Every** population-derived constant in §3.7's table is
**computed from the arena, not read** — the arena size, class counts, majority, `GATE-ARENA` band,
`GATE-DOMAIN`'s two majorities, the tie cap, S7's quantile threshold **and S7's `<=` small-set
comparison operator** (round-6 I-4). **(6)** `GATE-SELFTEST`'s `n` is the arena size and no banked
`744` leaks into any per-item denominator. **(7)** `ρ` is computed over the `743/579`-row matrices,
not a 744-row array with a masked row left in (a `1.301e-03` shift — fail-safe, but presenting as an
unexplained HALT).

*The mask convention.* **(8)** Every `prepare_views` call passes an explicit boolean array and every
`l2_rows` call's mask matches the population it is handed — **with an assertion, not a comment**.
**(9)** The `n = 744` build exists **only** inside `GATE-C01PARITY`/`GATE-ROWSUBSET` and nothing
votes on it.

*The tie diagnostic.* **(10)** Which ranking and which residual the implementation uses; that the
residual is the **head-space** `‖Δk‖₂` (or its `√d` bound); that "collapse" is the worst case over
orderings; that the report branch cannot be reached outside the tie set or above the cap; **and that
mismatches are aggregated per `(dataset, seed, lineage)` pooling the five folds, so the denominator
is `n_D`** (round-5 H-3).

*`GATE-POP` and heartbeat.* **(11)** `GATE-POP` runs **before** any gate consuming a
population-derived constant and asserts row identity by **index set**. **(12)** All six §9 items plus
the `RuntimeError` wrapper, the `buffering=1` handle never re-wrapped, the unbuffered driver echo,
append-without-interleaving across all 73 processes, and the frozen `elapsed ÷ projected`
denominator.

*Round 4's six.* **(13)** **The fold axis** — the 13 arms and every `ρ` are rebuilt from **each** of
the 60 fold key matrices, and no arm built under head `f` is ever voted for a query outside fold
`f`'s held-out fifth. **(14)** **The guard arms** — `orthrot_0` and `orthrot_45` are built by the
*rotation* route and `endpoint_concat` / `common_displacement` by their own, never aliased, and all
four voted. **(15)** **S7** — its classification as a SURVIVE condition, its
`common_displacement`-only arm scope, the zero-fix convention, **and its full parameter set: the
dominance threshold `0.5`, the reference-selection rule of §5.2.1, the head-space one-block
statistic, the `<=` operator and the per-seed `3/3` axis** (round-5 C-3) — none of which the battery
inherits, because it does not import `displacement_audit`. **(16)** **The statistics** — that the
bootstrap statistic, the one-sided `p` and the Holm step-down match §5.4; **that S5's null
construction, statistic, p95 convention, p-value form and 4-member family match §5.4.1**; **that
`GATE-SELFTEST` is asserted per seed as `net_s(A) = n_D · (acc_s(A) − acc_s(reference))`**
(round-5 I-1); and that no **decision** quantity — not only no gate quantity — reaches a comparison
non-finite. **(17)** Population-derived constants, extended, as item 5 now states. **(18)**
**`GATE-FOLD` under resume** — fold parity verified for all 66 mints including skipped ones, by
reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of` from the banked `.npz`.

**Round 5's four (items 19–22) — absent from v6, added here.**

**(19) The one-construction claim.** The two-block and one-block builds are the **same function**
called with different block lists; **no separate head-space builder exists**; and `GATE-C01PARITY`
runs against that function, not a copy. This is the sole warrant transferring the parity guarantee
into head space — the head-space arms have **no other anchor anywhere in the design**.

**(20) The `(dataset, lineage)` cross.** Every per-lineage gate is evaluated per cell; the drop
propagates across datasets per §5.6 (fails on **any** dataset ⇒ dropped on **both**); and no verdict
path can be reached with a lineage that passed on one dataset only.

**(21) The dropped lineage's quantities.** Exempt from the absence rule, excluded from S1–S7 and
from the S5 family, and entering the S4 family **only** as `NOT_TESTED` with `p = 1`; the S4 family
size the code uses is the frozen 92 of §5.5 on every path.

**(22) The key-forward site.** The ro-cache forwards producing `h_std`/`h_ow` happen **inside the
mint process**, since `headspace_mint` suppresses state-dict saves and the head weights never leave
it, and §8 Phase 1b's `174` is priced against that placement.

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


---

## 14. Cumulative disposition — all seven rounds

**Every row below cites the v7→v8 section(s) that implement it, and §14.1 prints the scripted audit
that checks each citation against the actual diff.** Round 6 caught v6 claiming repairs never made;
round 7 caught v7 doing it in three more places. The rule adopted here: **a disposition row that
cannot cite a diffed section is deleted or marked NOT ADOPTED.**

### Round 7 (9 findings + 4 Minor) — all adopted

| finding | disposition | §(diffed) |
|---|---|---|
| **C-1** §3.4's pin omits endpoint pre-normalisation; `GATE-C01PARITY` states two criteria and the `2e-6` one admits the wrong builder | **ADOPTED** — §3.4 defines `std[m]`/`ow[m]` as the `l2_rows`-normalised blocks and cites `prepare_views:1296-1304`; the measured cost of the omission (`1.878e-06` / `1.609e-06`, both passing `2e-6`) is recorded; **`GATE-C01PARITY` is now ONE predicate, `max\|diff\| == 0.0`**, with the `2e-6` clause struck | §3.4, §6 |
| **C-2** §5.2.3 does not exist; `tiny_ok`'s bar registered nowhere | **ADOPTED** — **§5.2.3 written**: both constants frozen (`0.001`, `0.05`), the `dominated`-only carriage stated, the measured warrant given (raw `0.0000` at `600×` slack; head `min d_i` `0.018`–`0.038`, `frac = 0.0000` in all four cells), the direction disclosed, and §13 item 25 named as the run-time control | §5.2.3, §3.7 |
| **C-3** §10.2 byte-identical to v6 while §14 claimed round-6 I-2 adopted there | **ADOPTED** — §10.2's scope sentence now names the dataset(s) each surviving lineage passed on, with the CLOSE-path degeneracy noted and the requirement kept unconditional because it binds SURVIVE and HALT too | §10.2 |
| **C-4** §5.2.2 unedited except a citation fix; `<=` frozen nowhere; dispersion clause absent | **ADOPTED** — `<=` frozen **in the decision rule** and added to §3.7's table so §13 item 5 resolves; the raw-versus-head dispersion contrast recorded with both measurements | §5.2.2, §3.7 |
| **H-1** header describes v6 and quotes the retired single-cell residual | **ADOPTED** — the header block is rewritten as *"What v8 changes"* over round 7's findings; the residual is not quoted there at all, and §6.5/§7.8 carry the four-cell range `8.848e-08`–`2.682e-07`, `7.5×`–`22.6×` | header |
| **H-2** §7.9 accounts for one mint where five were trained | **ADOPTED** — §7.9 splits v7's and v8's burns, states **five** v7 mints (one discharge + four tail cells), adds v8's no-mint measurements, and updates the cumulative to **v1–v8: ≈ 28 wall-min / ≈ 116 CPU-min, eleven mints** | §7.9 |
| **I-1** `ρ` reduction order unspecified; one value's 6th digit depends on it | **ADOPTED** — §6.1 freezes **`float64` accumulation over `float32` keys**, records both readings (`0.956893` / `0.956894`) and notes all 26 agree at 4 dp so `GATE-RHORAW` is unaffected either way | §6.1 |
| **I-2** two run-time loops uncounted, exceeding Phase 7's bound | **ADOPTED** — new **Phase 7z** prices `GATE-ALGEBRA`'s residual comparison (120 reductions, measured `1.0 s`) and §13 item 25's per-cell tail record (60 cells, `0.034 s`); total re-multiplied to **`2928.7 s`** / `3660.9 s` | §8 |
| **I-3** sixteen load-bearing banked inputs without digests | **ADOPTED** — sha256 added for the six arena OUT JSONs and the ten `vsw_ckpt` npz; the path is written with its `scripts/analysis/` prefix (the directory has `en`/`st_A`/`st_B`/`st_B0` siblings); the 36 C09 mints are declared digest-free as reference-measurement inputs only | §11 |
| **M-1** "does not import `displacement_audit`" | **ADOPTED** — corrected to *does not **call*** it; §11 does import the module that contains it | §5.2.2 |
| **M-2** §13 item attribution reversed | **ADOPTED** — item 23 carries the test-against-`prepare_views` instruction, item 19 the one-construction claim | §3.4 |
| **M-3** `select_strongest_ordinary_control` line range | **ADOPTED** — ranking at `:1955-1962`, guards at `:1940-1948` | §5.2.1 |
| **M-4** two competing superlatives in §5.9 | **ADOPTED** — item 8 drops its own; item 6 keeps *"largest such item"* | §5.9 |

**Round-7 rulings accepted without change:** no gate can fire on a warranted CLOSE (all twenty,
re-tested); the verdict-path enumeration is total and mutually exclusive with one lawful absence
path; §13's 26 items are complete and actionable; the 92-family's auditability warrant justifies its
direction and §5.9 item 8 states it strongly enough; four cells are adequate for the residual range
*because* §13 item 25 is the run-time control, not because four bound sixty.

### Rounds 1–6 — carried

Round 7 audited v7's round-6 block by execution and returned **11 VERIFIED ADOPTED, 3 PARTIAL, 1 NOT
ADOPTED, 2 adopted-with-a-new-defect**; v7's header claim of *"all 17 ADOPTED, 0 rebutted"* was
false. The three PARTIAL/NOT-ADOPTED items are round-6 H-3(c), I-2 and I-4 — **completed in v8 as
C-2, C-3 and C-4 above**. The two adopted-with-a-new-defect items are round-6 M-1 (the §7.8/§7.9
reorder, which left §7.9's cost stale → H-2) and round-6 M-3 (the `ρ` correction on an
underspecified basis → I-1), **both closed above**. Everything else in rounds 1–6 stands as recorded
in v3/v4/v6/v7 §14 and is not restated; where a later round refined an earlier disposition, the
later section governs.

**Rulings carried without change across all rounds.** The direction of *"conservative"* (§4); A7 is
not an obstacle; per-arm retraining excluded; `max` as `ρ*`'s order statistic; SLURM and the
login-node dismissal; HALT semantics; §5.9 item 1's inapplicability reasoning; the tie cap's
one-directionality; `GATE-ROWSUBSET`'s renaming; §3.4's account of what two-block parity does and
does not buy; and `B = 2000`.

**Struck, and not carried:** *S6's net-fix reference* (overturned by D-1, §5.2.1) and *the
untrained-head blindness discipline* (re-scoped by round-6 H-2, §7.3).

### 14.1 Mechanical disposition verification (**new protocol, round-7 response**)

Two consecutive rounds found this document asserting edits that were not made. The cause was
mechanical: the drafting script staged several edits in memory and wrote once at the end, so an
abort part-way discarded the staged edits **while the disposition table had already been written to
describe them**. v8's drafting writes after **every** edit and verifies the write, and the audit
below is run against the finished artifact.

**What the audit does.** (1) Splits v7 and v8 into sections and diffs them. (2) For every §14 row,
extracts the cited section numbers and checks each appears in the diff. (3) Reports any row whose
cited section shows no diff. (4) Resolves every internal `§N.N` reference, every `§13 item N`, and
each §3.7 constant-table row that §13 item 5 depends on.

**Output, verbatim:**

```
=== (1) SECTION DIFF v7 -> v8 ===
  CHANGED  §6        +409 chars
  CHANGED  §8        +585 chars
  CHANGED  §11       +2605 chars
  CHANGED  §14       -3745 chars
  CHANGED  §15       -161 chars
  CHANGED  §3.4      +1505 chars
  CHANGED  §3.7      +332 chars
  CHANGED  §5.9      +297 chars
  CHANGED  §6.1      +756 chars
  CHANGED  §7.9      +620 chars
  CHANGED  §10.2     +376 chars
  ADDED    §14.1     2626 chars
  CHANGED  §5.2.1    +49 chars
  CHANGED  §5.2.2    +1463 chars
  ADDED    §5.2.3    2653 chars
  CHANGED  header    -326 chars
  UNCHANGED: 41 sections

=== (2)+(3) DISPOSITION ROWS vs DIFF ===
  OK    C-1   cites §3.4, §6
  OK    C-2   cites §5.2.3, §3.7
  OK    C-3   cites §10.2
  OK    C-4   cites §5.2.2, §3.7
  OK    H-1   cites §header
  OK    H-2   cites §7.9
  OK    I-1   cites §6.1
  OK    I-2   cites §8
  OK    I-3   cites §11
  OK    M-1   cites §5.2.2
  OK    M-2   cites §3.4
  OK    M-3   cites §5.2.1
  OK    M-4   cites §5.9
  rows verified against diff hunks: 13 ; rows failing: 0

=== (4) INTERNAL REFERENCE TARGETS ===
  §N.N refs: 30 distinct ; unresolved: NONE
  §13 item refs: [5, 7, 25, 26] ; defined 1..26 ; unresolved: NONE
  §3.7 table row small-set comparison operator    PRESENT
  §3.7 table row tiny_displacement_epsilon        PRESENT
  §3.7 table row max_tiny_displacement_fraction   PRESENT
  §5.2.3 exists: True
```

**Reading it.** Every §14 round-7 row cites a section the diff shows as CHANGED or ADDED. The three
sections round 7 named as falsely claimed in v7 — §5.2.2, §10.2 and the absent §5.2.3 — now appear
as `CHANGED +1463`, `CHANGED +376` and `ADDED 2653`. The audit script is
`scratchpad/v8_audit.py`; it is a drafting instrument, not part of the battery, and it touches no
repository file.

## 15. Open issues for round 8

1. **The §14.1 protocol itself (new).** Round 8 should re-run the audit independently — diff v7→v8
   by section, check every §14 row's citation, resolve every reference — and rule whether a
   self-audit embedded in the artifact is adequate, or whether the disposition record needs an
   external check. Two rounds found this document asserting edits never made; §14.1 is the response,
   and its own sufficiency is untested.
2. **`GATE-C01PARITY` as a single bit-exact predicate (§6, §3.4).** The `2e-6` clause is struck.
   Round 8 should confirm bit-exactness is achievable in practice (four reviewers have now measured
   `0.000e+00`) and that no other gate inherited the retired tolerance by reference.
3. **The §3.4 pin, second attempt.** v7's pin closed `common_interaction` and opened the endpoint
   pre-normalisation hole. Rebuild the arms from §3.4 **as now written**, both readings, and report
   whether anything is still unstated — that is the only evidence that settles a specification
   defect of this class.
4. **§5.2.3's `tiny_ok` non-carriage.** Newly written, so newly rulable. Four cells of sixty, a
   conservative direction, and §13 item 25 as the run-time control — is that sufficient?
5. **The Phase 7z pricing (§8).** `GATE-ALGEBRA`'s residual measured at `1.0 s` here against round
   7's `0.160 s` — a `6×` gap between two honest measurements of the same loop on the same node,
   probably array-shape dependent. Round 8 should rule whether the conservative figure is the right
   one to freeze, and hunt for the next uncounted loop (seven rounds, six found one each).
6. **Is the record now sound?** Round 7 answered *no* and named four specific defects. All four are
   closed above with diff citations. If round 8 agrees, say so plainly; if not, the specific
   remaining defect is what matters. Do not grade on trajectory.

---

*No GPU, SLURM, Modal, teacher call, cache write, test-split access, job submission or commit
occurred in producing this document. **v8 trained no heads**; its measurements were the round-7 C-1
reproduction, the `ρ` reduction-order comparison, the Phase 7z loop timings and sixteen `sha256`,
all read-only on banked train-split artifacts. The eleven CPU head mints across v1–v8 wrote nothing
outside the session scratchpad and computed no battery-arm accuracy (§7.9). `TARGET_STATE.json` was
read and not modified. v1–v7 are unmodified.*
