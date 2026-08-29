# C06 `$0` falsifier — independent design review, **ROUND 6**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V6.md` (DRAFT v6, 2026-08-04)
**Reviewer posture:** fresh, independent of rounds 1–5 and of the designer. Read-only on the
repository. No GPU, no SLURM, no Modal, no arena run, no cache write, no test-split access, no job
submission, no commit. `TARGET_STATE.json`, all six drafts, all configs and all five prior reviews
were read and not modified. Computation was `sha256sum`, file reads, numpy/torch-CPU re-derivation
on already-banked **train-split** caches and banked mint checkpoints, and **one** authorized CPU
fold-head remint (39.7 s wall, 1.24 GiB, into the session scratchpad). **No arm accuracy was
computed at any point.**

---

# VERDICT

## **REVISE (2C / 3H / 5I)** — plus 6 Minor

Not `GO (0C/0H/0I)`.

**Ceremony floor: clean, and re-derived rather than read.** All **21** sha256 digests reproduce
character-for-character against disk (7 modules, 6 read-for-definition files, 8 input caches). Both
provenance chains are sha-gated in source. All **26** `ρ_raw` values reproduce (one at the 6th
decimal, → M-3). `ρ*` reproduces at `0.968176` / `0.977223`; the trained-head reference reproduces
to the digit on all 36 banked mints, **0/18 above `ρ*` on both**. Every population constant, class
count, majority, band and tie cap re-derives exactly. The row-subset identity is `0.000e+00` on all
13 arms. I rebuilt the 13-arm algebra from §3.4's prose alone and matched `prepare_views`
**bit-exactly** on both datasets — the fifth independent reproduction, but only after correcting a
formula the prose does not pin (→ **C-1**). §8's printed column re-multiplies and sums exactly to
`2927.6`. Test-split non-contact is sound by construction. `TARGET_STATE.json`'s two quotations at
§1 are verbatim-correct. **Blindness is intact across v1–v6:** I grepped every decimal in
`[0.6, 0.99]` across all six drafts; v6 adds **exactly one** new value, `0.8725`, which I confirmed
is the banked `headspace_arena_hatemm_s0_OUT.json::fold_acc_deployed[0]` — an instrument anchor, not
a battery arm.

**Both new measurements verify, and one is stronger than v6 claims.** I re-minted HateMM `s0/fold0`
through `headspace_mint.py` and the re-minted `K_train` is **bit-identical to the banked mint**
(`max|diff| = 0.000e+00`), not merely accuracy-equal at 4 dp; the native deployed vote is
`0.872483 → 0.8725` on both. Separately I recomputed **all 36 banked accuracy quantities** (30
`fold_acc_deployed` + 6 pooled) from the banked mints and every one reproduces. `GATE-FLOOR` is in
much better shape than §7.9 asserts (→ **I-5**).

**D-1 is real, its repair is right, and its warrant is stronger than v6 gives.** I read both
`fix_break` sites from the frozen source and the executed `C01_A0_OUT.json`. v5's S6 reference was
wrong and rounds 3–5 verified it wrongly. See PART C, deliverable 7.

**Where the two Criticals sit — and it is not where five rounds have been looking.** Every prior
round found its Criticals in the seam the previous round's repair opened. v6's science layer is
clean: I could not manufacture a CLOSE anywhere on the new dataset-axis rule, the twenty gates hold
under the warranted-CLOSE test, and the verdict combination is total and mutually exclusive. **The
Criticals are that §13 and §14 were never edited.** Both are **byte-identical to v5** — I diffed
them, and the only difference in the whole of §14 is the string `round 5` → `round 6` in §15's
heading. Consequently:

* **Round-5 I-6 is NOT ADOPTED at all** (§13 item 19 does not exist), and six further round-5
  findings had explicit §13 limbs in their prescribed repairs which are all absent. §13 still reads
  *"round 3's twelve items plus round 4's six"* (v6:1265, v6:1296) — eighteen items — while v6's own
  §15.2 asks a reviewer to check **"§13 item 20"** (v6:1369) and the round-6 review request calls it
  the *"twenty-two-item code-lineage handoff"*. The document points at repairs it does not contain.
* **§14 has no round-5 disposition block whatsoever**, while the header claims *"all 17 round-5
  findings ADOPTED, 0 rebutted … Cumulative table in §14"*. Worse, §14 still asserts two things the
  body of v6 now refutes: the **"42 of 92 must show zero adverse resamples"** floor (v6:1321), which
  §5.5 corrects to 22/24; and **"S6's net-fix reference"** among *"rulings carried without change
  across all rounds"* (v6:1352), which is precisely what D-1 overturns. v6 states both readings of
  its own headline correction.

This is not a bookkeeping complaint. §13 is the **sole input to the mandatory separate code/resource
review lineage** (§2 rule R4), the lineage the campaign's own record says caught two wrong-verdict
paths on C09 after seventeen clean design rounds. Round-5 I-6 named the one claim that transfers
`GATE-C01PARITY`'s guarantee into head space, and I hit that rock myself: reconstructing the builder
from §3.4's prose I built `common_interaction` as `l2(std ⊙ ow)` and got `max|diff| = 9.697e-01`
across 13 arms; the correct form is `l2(common ⊙ displacement)` (`contrast_blocks`,
`c01_policy_contrast_a0.py:1260-1265`), after which parity is `0.000e+00`. Two of six reviewers have
now independently mis-derived the same arm from the same prose. The head-space arms have **no other
anchor**, and nothing in §13 tells the code lineage to check it.

**The three Highs are all repairs whose warrant is false, not repairs that are missing.** §5.5's
family-invariance guarantee is refuted by a case stated two paragraphs above it in the same
subsection (H-1); §7.3's blanket untrained-head sentence is falsified by §7.9's own trained-head leg
(H-2); and `GATE-ALGEBRA`'s trained-head discharge prices a **max-over-rows** statistic from a
**median**, on 1 of 60 cells and 1 of 2 lineages, over a lower tail that C01 guarded and v6 carries
no disposition of (H-3).

**None of the eleven findings requires a GPU, an extraction, new data or a redesign.** C-1 and C-2
are edits to two sections that were simply not touched. H-1 is one sentence plus a §5.9 item. H-2 is
one sentence. H-3 is one measurement the designer can take on the head already minted, plus one
disposition sentence. **The falsifier can still discharge its written condition at `$0`**, at the
projection §8 states, which I re-multiplied independently and which is right.

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All 21 digests recomputed against disk, character-for-character. Spot values: `headspace_mint.py` `cefdf8dc…0916612`; `mechfix_ops.py` `635c1312…c83fc8d`; `c01_policy_contrast_a0.py` `d2b9c2ff…8db1b855`; `c09guard.py` `aed50842…d745f062`; `c01_a0_v2.json` `f3997bdd…7a241563f5`; HateMM `ro_L24` `6a44cce4…0be045f`; HateMM `ro_ow_L24` `60054f3b…1c27ce61977f4e`; MHC-ZH `ro_L24` `1d33fe5d…6221c06`; MHC-ZH native `dev_seen` `4c07af75…7e4f5d3c`. Both chains sha-gated **in source** (`_v4.py:52-55` → `_v3.py:48-51` → base). |
| **V2** | **VERIFIED — D-1 is real, at both the code and the executed-output level** | `c01_policy_contrast_a0.py:1725` is exactly `reference = evaluations[config["retrieval"]["fix_break_reference"]]["predictions"]`, feeding the **reporting** field `fix_break_vs_endpoint_std`; `retrieval.fix_break_reference = "endpoint_std"` (`c01_a0_v2.json:145`). `:2702-2714` is `strongest_control_name = select_strongest_ordinary_control(evaluations, controls)` → `checks["net_fixes"] = {"reference": strongest_control_name, …}`. Executed: `decision.datasets.HateMM.checks.net_fixes.reference = "common"` (net `−1`, min `3`, pass `False`); `MHC_zh … = "endpoint_concat"` (net `−2`, min `2`, pass `False`). **v5's S6 reference was wrong.** |
| **V3** | **VERIFIED — and one constant more than v6 names** | `transforms.max_small_displacement_fix_fraction = 0.5` ✓; `small_displacement_train_quantile = 0.1` ✓; C01's statistic is `np.quantile(np.minimum(d_norm["train"]["img"], d_norm["train"]["text"]), 0.1)` on the raw features, where `d_norm[m] = ‖l2(ow_m) − l2(std_m)‖` (`contrast_blocks:1253-1259`) ✓; the consistency `die()` is guarded by `if small_gain_gate["reference"] != strongest_control_name:` at **`:2724`** ✓. **Also frozen in the same audit and carried nowhere in v6:** `tiny_displacement_epsilon = 0.001` and `max_tiny_displacement_fraction = 0.05`, whose conjunction `tiny_ok` is a limb of C01's `scientific_gate_final_bool` (`:2068-2076`) and therefore of `checks["displacement_stability"]["pass"]` (`:2750-2753`). → **H-3**. |
| **V4** | **VERIFIED — exact** | Executing `holm_adjust` over `m = 92` with 24 witness hypotheses at `1/2001 = 0.00049975` and 68 at `0.5`: **24/24 reject**; 22 at `1/2001`: **22/22**. Degrade one witness hypothesis to `2/2001 = 0.00099950`: **23/24**. Rank-0 arithmetic: `92 × 1/2001 = 0.045977 ≤ 0.05` ✓, `92 × 2/2001 = 0.091954 > 0.05` ✗. The floor is the witness's own 22 or 24 comparators, not 42 of 92. |
| **V5** | **VERIFIED as stated — and the generalisation v6 draws from it is FALSE** | The three configurations are identical: `m = 92` pad `0.5` → 24/24; `m = 92` pad `1.0` → 24/24; `m = 46` → 24/24. **But** with one witness hypothesis at `2/2001`: `m = 92` → **23/24 (S4 FAILS)**, `m = 46` → **24/24 (S4 PASSES)**. With all 24 at `2/2001`: `m = 92` → **0/24**, `m = 46` → **24/24**. → **H-1**. |
| **V6** | **VERIFIED — exact** | `1/257 = 0.0038911`; `n × 0.0038911 ≤ 0.05` ⇒ `n ≤ 12` (I enumerated: `n = 12` → `0.04669` ✓, `n = 13` → `0.05058` ✗). Frozen `n = 4` needs `0.01556` ✓; `n = 8` needs `0.03113` ✓. |
| **V7** | **VERIFIED — and stronger than claimed** | I re-minted HateMM `s0/fold0` through `headspace_mint.py` (39.69 s wall, internal 32.7 s, MAXRSS 1.24 GiB, `fold_parity_vs_banked_vsw_ckpt = [True]×5`, 38 state-dict saves suppressed). Native deployed vote on fold 0's held-out 149: **`0.872483 → 0.8725`**, banked `fold_acc_deployed[0] = 0.8725` ✓. **The re-minted `K_train` is bit-identical to the banked mint: `max|diff| = 0.000e+00`**, `lab`, `fit_idx` and `fold_of` all identical, `head_dim = 1024`. Separately, all **36** banked accuracy quantities (30 fold + 6 pooled) reproduce from the banked mints: HateMM `0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946`. → **I-5**. |
| **V8** | **NOT REPRODUCIBLE UNDER THIS AUTHORIZATION — mechanism verified, magnitude not** | `headspace_mint.py` suppresses state-dict saves and banks only `K_train` (native keys), so a trained head's `h_std` / `h_ow` ro-forwards cannot be recovered from any banked artifact, and reproducing `2.384e-07`, `1.490e-08` and `0.2301` would require a second mint plus code the review is not authorized to write. **The mechanism is verified from the source:** at `θ = 45`, `orthogonal_blocks:1272-1290` gives `first = l2(cos45·std + sin45·ow)` and `second = l2(−sin45·std + cos45·ow)`, while `contrast_blocks` gives `common = l2(std + ow)` and `displacement = l2(ow − std)` — identical up to the `cos45 − sin45 = 1.11e-16` asymmetry acting on the difference vector, so the residual does scale inversely with `‖l2(h_ow) − l2(h_std)‖` exactly as §6.5/§7.9 argue. **But the guard is `np.max(np.abs(...))` over rows (`prepare_views:1372-1377`), driven by the row of *smallest* displacement, and v6 prices it from a median.** → **H-3**. On the raw features I reproduced the published guards exactly: `θ = 0` `8.9407e-08` both datasets; `θ = 45` `1.1921e-07` (HateMM) / `8.9407e-08` (MHC-ZH). |
| **V9** | **VERIFIED — bit-exact, fifth independent reproduction** | I re-implemented `fuse` / `paired` / `build_views` from §3.4's prose alone, calling the imported `l2_rows`, and compared against `prepare_views` with §3.7's mask forms: **`max\|diff\| = 0.000e+00`, 13 arms, both datasets, `float32` both sides** — one-hot `{355}` at `n = 744` (HateMM), all-False at `n = 579` (ZH). `GATE-ROWSUBSET`: the `n = 743` all-False build vs the `n = 744` one-hot build restricted to the 743 survivors is **`0.000e+00`** on all 13 arms. The null-contract defect is real: `classifier.py:81-82` builds both projections with the default bias; row 355 = `hate_video_95`, label 1, exact-zero in both modalities of both ro caches and the native cache, and the only such row on either dataset. Two-block arm dims `{7168: 4 arms, 14336: 9}`, so the one-block head-space analogue is `{1024: 4, 2048: 9}` ✓, confirming §8 Phase 2's `240 / 540`. **Caveat: my first attempt failed at `max\|diff\| = 9.697e-01`** because §3.4's prose does not determine the arm→formula map. → **C-1**. |
| **V10** | **VERIFIED — all 26, and the trained-head reference** | `ρ* = 0.968176` (HateMM, `endpoint_std`) / `0.977223` (MHC-ZH, `endpoint_std`); runner-ups `0.964446` / `0.969686`, both `common`. Every `ρ_raw` reproduces at 6 dp **except** HateMM `orthrot_83p8`, which is `0.9568935731` → `0.956894`, printed `0.956893` (→ **M-3**). `ρ` over 744 rows with the masked zero row left in shifts by up to **`1.301e-03`** ✓. Trained-head reference over all 36 banked mints: HateMM min/median/max `0.447803 / 0.562434 / 0.632996`; MHC-ZH `0.340179 / 0.574247 / 0.667326`; **0/18 above `ρ*` on both** ✓. |
| **V11** | **VERIFIED — every constant, including all three new rows** | HateMM full `n = 744`, `pos 298 / neg 446`, `446/744 = 0.599462 → 0.5995`. **Arena `n = 743`, `pos 297 / neg 446`, `446/743 = 0.600269 → 0.6003`, band `[0.6203, 0.98]`, tie cap `⌊7.43⌋ = 7`.** MHC-ZH `n = 579`, `pos 180 / neg 399`, `399/579 = 0.689119 → 0.6891`, band `[0.7091, 0.98]`, tie cap `5`, full = arena. `GATE-IDPARITY` holds directly on both datasets: both ro caches' `ids` are order-identical and `labels` element-identical to the native bank. The three new rows: S7's quantile population is the arena `743 / 579` ✓; the `0.5` dominance threshold is `transforms.max_small_displacement_fix_fraction` ✓; the reference-arm rule is `select_strongest_ordinary_control` over `decision.gain_controls = ['endpoint_std','endpoint_ow','avg_score','endpoint_concat','common']` ✓ — five members, and `displacement` is **not** among them, so no self-comparison is reachable for either real arm. |
| **V12** | **VERIFIED** | I re-multiplied all 26 §8 rows independently from folds × seeds × lineages × datasets × arms × draws before reading the table; every product is right (unrounded sum `2927.517`). The **printed** column sums to **`2927.6`** exactly (mints subtotal `2508.3`) ✓; `× 1.25 = 3659.5` ✓; `48.8 / 61.0 min` ✓; mint share `2508.3/2927.6 = 85.68 → 85.7 %` ✓; Phase 3 `273.7/2927.6 = 9.35 → 9.3 %` ✓; sensitivities `3201.3` and `4022.4` ✓. §6's table has **twenty** rows — **12 `G`** (DET1, SHA, FOLD, FLOOR, POP, C01PARITY, ROWSUBSET, NULLREMOVED, IDPARITY, ZEROMASK, LEDGER, RHORAW), **6 `L`** (ORBITDISP, ARENA, NESTED, SELFTEST, ZEROOP, ALGEBRA), **2 `R`** (DOMAIN, DEVFID) — and both §5.6 lists match member-for-member. Round-5 M-5 fully discharged. |

## Additional measurements v6 does not report

**(α) The raw displacement-norm distribution is extraordinarily tight, and C01's tiny guard is slack
by ~600×.** On the arena rows, `min(‖d_img‖, ‖d_text‖)`: HateMM min `0.614645`, `q₀.₁ 0.647340`,
median `0.673298`, max `0.737694`; MHC-ZH min `0.620789`, `q₀.₁ 0.659981`, median `0.686138`, max
`0.737210`. The fraction at or below `tiny_displacement_epsilon = 0.001` is **`0.0000`** in every
modality on both datasets, and `C01_A0_OUT.json` records `maximum_tiny_fraction = 0.0` on both. Two
consequences: C01's `tiny_ok` never bit in the raw space, which is why nobody noticed it; and C01's
"small set" is the bottom decile of a distribution whose full support spans only ~20 %, so the
small/large split separates rows differing by about 5 % in displacement norm. Neither fact transfers
to head space, where v6's own §7.9 reports a **median** of `0.2301` (trained) and `0.0032`
(untrained) — the untrained median sitting only `3.2×` above the tiny epsilon. → **H-3**, **I-4**.

**(β) The two HateMM seeds that share a fold-accuracy vector are not degenerate.** `hatemm` s1 and
s2 have identical banked `fold_acc_deployed` `[0.8591, 0.8859, 0.9060, 0.8658, 0.9122]` and identical
pooled `0.8858`. I checked whether the seed axis is real: the banked `K_train` matrices differ at
`max|diff| = 3.68e-02` (`f0`) and `5.90e-02` (`ffull`), so the heads are genuinely distinct and the
coincidence is the `1/149` accuracy quantisation acting on the campaign's known ~90 % seed-invariant
error sets (F88). **Not a finding** — recorded so a later round does not re-open it. S6's and S2's
`3/3` seed requirements retain their content.

**(γ) The reference-selection rule is a frozen config constant, not only a code path.**
`transforms.small_displacement_gate_reference =
"strongest_ordinary_control_by_accuracy_then_macro_f1_then_frozen_gain_controls_order"`, and it is
written onto C01's own verdict face as `displacement_stability.selection_rule`. This is a materially
stronger pre-registration warrant for D-1 than §5.2.1 offers. → PART C, deliverable 7.

---

# PART B — FINDINGS

## CRITICAL

### C-1. **§13 is byte-identical to v5.** Round-5 I-6 is NOT ADOPTED, and six further round-5 findings' prescribed §13 limbs are absent — while the header claims all 17 adopted and §15.2 cites a §13 item that does not exist.
*Attaches to:* §13 (v6:1253-1307, unchanged from v5); §15 item 2 (v6:1369); §3.4; §7.6; the header's
disposition sentence (v6:11-13).

I extracted §13 from both drafts and diffed: **zero lines differ.** It still opens *"round 3's twelve
items plus round 4's six"* (v6:1265) and closes at item (18) (v6:1296-1307). Round 5's PART F
prescribed four additions and two extensions; none is present:

| round-5 finding | prescribed §13 change | in v6 |
|---|---|---|
| **I-6** | new item **19** — the two-block and one-block builds are the **same function** with different block lists, no separate head-space builder, and `GATE-C01PARITY` runs against that function | **ABSENT — this finding is NOT ADOPTED in any form** |
| **C-1** | new item **20** — the `(dataset, lineage)` cross and the cross-dataset drop | **ABSENT** (v6:1369 cites it by number) |
| **C-2** | new item **21** — a dropped lineage's quantities, their exemption and the family size | **ABSENT** |
| — | new item **22** — the key-forward site inside the mint process, against which Phase 1b's `174` is priced | **ABSENT** |
| **C-3** | extend item **15** to S7's threshold `0.5`, reference rule, space, reduction and seed axis | **ABSENT** — item 15 is verbatim v5 |
| **H-2 / I-1** | extend item **16** to S5's statistic and family, and to `GATE-SELFTEST`'s per-seed form | **ABSENT** — item 16 is verbatim v5 |
| **H-3** | add `GATE-ZEROOP`'s aggregation to item **10** | **ABSENT** |

The round-6 review request describes §13 as the *"twenty-two-item code-lineage handoff"*, which is
independent evidence that the additions were believed made. Under the severity definition in force —
*"any claimed repair the artifact does not contain"* — this is Critical on its face. But the reason it
is Critical in substance is **I-6 specifically**.

**I-6's prediction, reproduced.** §3.4's warrant for the head-space arms is *"**One construction**,
parameterised by an ordered list of blocks"*: two blocks reproduce `prepare_views` bit-exactly,
therefore the one-block instantiation inherits the arm→formula map, the normalisation order, the
Givens convention and the dtype. That inference holds **only if the code is one function with a
block-list parameter**. The head-space arms have no other anchor anywhere in the design. Building
from §3.4's prose alone I produced `common_interaction` as `l2(std ⊙ ow)` and got
`max|diff| = 9.697e-01` across the 13 arms; the correct definition is
`l2(common ⊙ displacement)` (`c01_policy_contrast_a0.py:1260-1265`), after which parity is
`0.000e+00`. Round 5 hit the identical arm from the identical prose. **Two of six independent
reviewers have now mis-derived the same arm**, which settles that the map is pinned by the parity
gate and not by the document — and §13, the only place a code lineage is told what to check, does
not say so.

**Repair.** Add items 19–22 and the three extensions exactly as round 5's PART F prescribes, and add
one clause to item 19 recording that the arm→formula map is **not** determined by §3.4's prose (with
`common_interaction` named as the concrete instance), so the code lineage tests the map against
`prepare_views` rather than against the document. Then re-number §15's cross-references.

### C-2. **§14 is byte-identical to v5.** There is no round-5 disposition block, and §14 asserts two claims the body of v6 refutes — including the one D-1 exists to correct.
*Attaches to:* §14 (v6:1311-1353, unchanged from v5); the header (v6:11-13); §5.5 (v6:519-526);
§5.2.1 (v6:379-389); §7.3 (v6:915).

I diffed §14 across the two drafts: the only difference in the entire section is `## 15. Open issues
for round 5` → `for round 6`. The header claims *"all 17 round-5 findings ADOPTED, 0 rebutted, plus
one defect the designer found … Cumulative table in **§14**"* — and §14 contains a round-4 block, a
rounds-1–3 paragraph, and nothing else. Three consequences, in ascending order of seriousness:

1. **The cumulative table is incomplete across the six versions**, which the round-6 freeze-readiness
   criterion requires it to be. A reader cannot audit round 5's dispositions from the artifact.
2. **§14:1321 still states the refuted floor.** The H-2 row reads *"§5.5 records the `B = 2000` /
   92-family resolution floor (**42 of 92 must show zero adverse resamples**)"*. §5.5 (v6:519-526)
   now says the opposite, in bold, with the correction attributed to round-5 H-1. **The document
   states two mutually contradictory versions of S4's statistical bar.** Round 4 and round 5 both
   rated "two rules in one document" Critical; this is the same defect in the disposition record.
   Round-4 measurement ζ is likewise still listed as *"folded into the record"* (v6:1339) although
   round 5 showed its inference was wrong.
3. **§14:1351-1352 lists as *"rulings carried without change across all rounds"* both `S6's net-fix
   reference` and `the untrained-head blindness discipline`.** The first is exactly what D-1
   overturns — v6's headline correction is contradicted, in the same document, by a line asserting
   the superseded ruling still stands. The second is falsified by §7.9's trained-head leg (→ **H-2**).

**Repair.** Add a round-5 block to §14 covering all 17 findings plus D-1 with honest dispositions
(several are PARTIAL — see PART D); correct the H-2 row's floor to *"every one of the witness's 22 or
24 comparators"*; strike `S6's net-fix reference` from the carried-rulings list and replace it with a
pointer to §5.2.1; strike or re-scope `the untrained-head blindness discipline`; and mark round-4
measurement ζ superseded by round-5 H-1.

---

## HIGH

### H-1. §5.5's family-invariance guarantee — *"engineering a drop changes S4's difficulty by nothing at all"* — is **false**, and its counterexample is stated two paragraphs above it in the same subsection. The freeze at 92 is the CLOSE-easing choice and §5.9 does not disclose it.
*Attaches to:* §5.5 (v6:534-541); §5.6 (v6:608-609); §5.9 (v6:636-671); §4; §8 Phase 4; §15 item 3.

v6 fixes the family at 92 on every path with a dropped lineage's hypotheses at `p = 1`, and warrants
the choice by measurement:

> **Measured, this makes the bar invariant to a drop:** … the witness's 24 hypotheses reject `24/24`
> identically under `m = 92` padded with `0.5`, under `m = 92` padded with `1.0`, and under `m = 46`
> … So the choice cannot be gamed in either direction — **engineering a drop changes S4's difficulty
> by nothing at all** …

I executed `holm_adjust` on all three configurations and the three-way equality is exactly right (V5).
**The generalisation is not.** The invariance holds only when every witness hypothesis sits at the
resolution floor `1/2001` — which is the very condition §5.5's *preceding* paragraph identifies as
required for S4 to pass. The argument is therefore circular: it establishes "if S4 passes at the
floor, the family size does not matter", and reports it as "the family size never matters."
Measured, one step off the floor:

| witness p-values | `m = 92` (v6's rule) | `m = 46` (round 5's prescription) |
|---|---|---|
| 24 × `1/2001` | 24/24 — **S4 PASS** | 24/24 — **S4 PASS** |
| 23 × `1/2001` + 1 × `2/2001` | 23/24 — **S4 FAIL** | 24/24 — **S4 PASS** |
| 24 × `2/2001` | 0/24 — **S4 FAIL** | 24/24 — **S4 PASS** |

`92 × 2/2001 = 0.091954 > 0.05` while `46 × 2/2001 = 0.045977 ≤ 0.05`; the rank-0 admissibility bar
moves from `α/46 = 0.00108696` to `α/92 = 0.00054348`, which the `B = 2000` grid straddles. So a drop
under the frozen-92 rule **can** convert a SURVIVE into a CLOSE relative to the recompute-at-46
alternative, and v6's own §5.5 says as much one paragraph earlier (*"degrade one witness hypothesis
to `2/2001` and it becomes 23/24"*). Round 5's C-2(b) measured the same gap and called it material.

Two further points make this High rather than Improvement:

* **The direction is the disclosed-lean direction, and it is undisclosed.** §4 fixes *"conservative"*
  as *hardest for the falsifier to deliver the `$0` CLOSURE*. Freezing at 92 makes S4 strictly harder
  ⇒ SURVIVE harder ⇒ **CLOSE easier**. That is the anti-conservative direction under the design's own
  definition, and §4's binding condition (round 1's ruling, upheld by rounds 2–4) is that the design
  disclose what its lean buys. §5.9's seven items do not include it — while item 7 discloses a change
  in the *opposite* direction "for symmetry".
* **Round 5's prescribed repair was the 46-family, and v6 chose the opposite while claiming
  "0 rebutted".** Choosing differently is legitimate; presenting it as an adoption is not.

**I am not asking for the 46-family.** Freezing at 92 is defensible — it is auditable, it cannot be
enlarged by a realised run, and it is the stricter multiplicity control. **Repair:** delete the
sentences claiming invariance and non-gameability; state instead that the family is frozen at 92
*because* a preregistered constant cannot be moved by a realised run, that this makes S4 strictly no
easier than the alternative, and that the invariance measured at `1/2001` is a property of the
resolution floor and not of the rule; and add an eighth §5.9 item disclosing that the choice runs in
the CLOSE-easing direction.

### H-2. §7.3's load-bearing blindness sentence — *"Every head used in any arm-building or voting dry check is **untrained**"* — is **false in v6**. §7.9 built arms and voted on a trained head. The blindness outcome is intact; the stated warrant is not.
*Attaches to:* §7.3 (v6:913-922); §7.9 (v6:979-1018); §14 (v6:1351); §15 item 5.

§7.3's safety argument is that every dry-check number is *"scientifically void"* because the heads are
untrained. v6's §7.9 breaks that in two places at once: it mints a **trained** HateMM `s0/fold0`
head, **votes** with it (the `0.8725` native deployed vote), and **builds arms** with it (the
`θ = 0` and `θ = 45` guard identities are `prepare_views`' arm constructions — I confirmed the
residual is `np.max(np.abs(views["common_displacement"] − endpoint45))` at
`c01_policy_contrast_a0.py:1372-1377`). §7.3 was not updated; §14:1351 still carries *"the
untrained-head blindness discipline"* among rulings unchanged across all rounds.

**The outcome is intact and I verified it independently.** Grepping every decimal in `[0.6, 0.99]`
across v1–v6, v6 adds exactly **one** new value, `0.8725`, which I confirmed against
`headspace_arena_hatemm_s0_OUT.json::fold_acc_deployed[0]`. No battery arm accuracy appears anywhere
in the six drafts. The correct warrant is available and stronger: **no arm derived from the ro caches
was ever voted**, so no arm accuracy exists to disclose — which is a statement about what was
computed, not about how the head was initialised.

This matters beyond wording because §7.9 is the **first** time in six drafts that a trained head
touched the ro caches. Until v6, arm accuracy was structurally unavailable; from v6 it is one
function call away. The section that governs the campaign's blindness discipline should say so.

**Repair.** Replace §7.3's first sentence with a two-part statement: (i) all *untrained*-head legs are
scientifically void by construction; (ii) the §7.9 trained-head leg computed **only** the native
deployed vote (a banked instrument anchor) and the two key-space algebra residuals, and **no vote was
taken on any ro-derived arm** — with that assertion added to §13 so the code lineage can check the
freeze record against it. Correct §7.3's *"v6 adds **two** measured accuracies"* (v6:920, one is
named and one exists — → M-2), and strike the stale ruling from §14.

### H-3. `GATE-ALGEBRA`'s trained-head discharge prices a **max-over-rows** statistic from a **median**, on 1 of 60 cells and 1 of 2 lineages — over a lower tail that C01 guarded with `tiny_ok` and that v6 carries no disposition of anywhere in v1–v6.
*Attaches to:* §6 `GATE-ALGEBRA` (v6:711); §6.5 (v6:829-873); §7.9 (v6:998-1018); §5.2.2 (v6:405-435);
§5.4.1's guard disposition (v6:498-504); §3.7; §8 Phase 7.

Three limbs, one object: **the lower tail of `d_i = ‖l2(h_ow,i) − l2(h_std,i)‖` in head space.**

**(a) The statistic is a max; the evidence is a median.** `prepare_views:1372-1377` computes the
algebra guard as `float(np.max(np.abs(views[arm] − endpointθ)))` and `die()`s above `2e-6`. Because
the `θ = 45` identity fails only through the `cos45 − sin45 = 1.11e-16` asymmetry acting on the
difference vector — which I verified from `orthogonal_blocks:1272-1290` against
`contrast_blocks:1246-1265` — the residual is governed by the **smallest** `d_i` in the cell, not the
median. §7.9 reports `median ‖l2(h_ow) − l2(h_std)‖ = 0.2301` and concludes *"`GATE-ALGEBRA`'s frozen
`2e-6` bar transfers to head space with `8.4×` headroom"*. A median cannot bound a max. This is
round-4 I-4's defect class (key components vs similarities) and round-5 H-3's (undefined denominator)
in the gate next door.

**(b) `8.4×` is thin against the document's own demonstrated spread.** §6.5 argues, correctly, that
the untrained residuals are unstable — v6 measured `1.175e-06`–`1.362e-06` where round 5 measured
`1.863e-07`–`1.974e-07`, *"a `6×` spread between two honest measurements of the same quantity"*. The
design then quotes a **single** trained measurement, from **one** `(dataset, seed, fold, lineage)`
cell of **sixty**, as load-bearing. `2.384e-07 × 8.4 = 2.0e-06`: a 6× excursion of the kind the
document has just demonstrated for this quantity lands at `1.4e-06`, inside the bar but not
comfortably. And the measured cell is **Head-N** — the native-trained transplant. **Head-R, the
in-domain lineage and the one most likely to carry a SURVIVE, has no measurement at all.** This
directly contradicts §7.4(a)'s and §7.5's own discipline, which the design applies to every other
weight-dependent quantity: state a range, load-bear only on the invariant.

**(c) C01's own lower-tail guard has no disposition in this lineage.** C01's decision-level check is
`checks["displacement_stability"]["pass"] = dual_path_null_exclusion_audit.pass AND
scientific_gate_final_bool` (`:2750-2753`), and
`scientific_gate_final_bool = tiny_ok AND (not dominated)` (`:2068-2076`), where
`tiny_ok = max_tiny_fraction ≤ max_tiny_displacement_fraction (0.05)` over
`fraction(d ≤ tiny_displacement_epsilon = 0.001)`. v6's S7 carries **only** the `dominated` limb.
The word `tiny` appears **nowhere in v6, and nowhere in v2–v5**. §5.4.1 gives a complete disposition
of C01's seven `required_halt_only_validity_guards`, but `tiny_ok` is not one of them — it sits
inside `decision`, in the same check S7 imports, and falls through the gap.

I am aware this is not a new gate proposal: v1 carried it as `GATE-ORBITSCALE`, round-1 C-1 correctly
ruled a **magnitude** gate the wrong *instrument* discriminator for a **direction** artifact, and v2
deleted it for `GATE-ORBITDISP`. That ruling stands and I do not reopen it. What changed is that v5
created **S7**, a binding SURVIVE conjunct that names C01's small-displacement condition as its
frozen source — and that condition has two limbs. The measured facts make the gap concrete rather
than formal: in the **raw** space the fraction below `0.001` is `0.0000` on both datasets with a
minimum of `0.6146` (measurement α), so `tiny_ok` was slack by ~600× and could never bite; in **head**
space v6's own untrained median is `0.0032`, only `3.2×` above the epsilon. The head-space lower tail
is where this quantity could bite for the first time, and it is unmeasured.

Direction: `GATE-ALGEBRA` is scope **L**, so a spurious excursion drops a lineage and — via §5.6 rule
2 — converts a warranted CLOSE into a **HALT**, never into a wrong verdict. I therefore rate this
High and not Critical, following round 5's explicit precedent on the structurally identical
`GATE-ZEROOP` granularity finding (*"the direction is safe … it can only cause the falsifier not to
publish"*). Dropping `tiny_ok` likewise runs in the conservative direction (it eases SURVIVE).

**Repair, all three limbs, on the head already minted.** (i) Report `min_i d_i`, the fraction
`d_i ≤ 0.001`, and the resulting residual for a handful of cells spanning **both lineages** and both
datasets, and state the residual as a **range** with only the invariant claim load-bearing, per §7.5's
own discipline; (ii) state in §6.5 that the residual is a max over rows and is governed by the
smallest `d_i`, so the headroom claim is a statement about the tail, not the median; (iii) add one
sentence to §5.2.2 or §5.4.1 disposing of `tiny_ok` — recording that C01's `displacement_stability`
has a second limb, that C06 does not carry it, that the omission eases SURVIVE, and that round-1 C-1
is the reason no magnitude *instrument* gate is reinstated.

---

## IMPORTANT

### I-1. §5.9 item 4's *"S3 and S6 are two statements about **one** quantity"* is over-strong for the `common_displacement` disjunct: S3's comparator set has six members and S6's reference is selected over five.
*Attaches to:* §5.9 item 4 (v6:646-657); §5.1 (v6:342-345); §5.2 S3/S6 (v6:356, 359); §5.2.1 (v6:396-398).

§5.1 sets `C = gain_controls ∪ {displacement}` (**six**) for `common_displacement` and
`C = gain_controls` (**five**) for `displacement`. §5.2.1's reference is
`argmax` over `decision.gain_controls` — **five**, verified: `['endpoint_std', 'endpoint_ow',
'avg_score', 'endpoint_concat', 'common']`. So for `A = common_displacement`, S3 binds against
`max_{c∈C} acc(c)` which may be `acc(displacement)`, while S6 measures net fixes against an arm
chosen from the five. Whenever `displacement` is the strongest of the six, S3 and S6 are statements
about **different** quantities, and §5.9 item 4's *"exactly"* is wrong. This is not exotic: in C01's
own executed run on MHC-ZH, `displacement` (`0.8846`) **ties** the selected reference `endpoint_concat`
(`0.8846`) at the top of the comparator set.

**The conclusion survives**, which is why this is Important and not High: since `reference ∈
gain_controls ⊆ C`, we have `acc(reference) ≤ max_C acc(c)`, so S3 still implies
`mean_s net_s ≥ 0.02 · n_D` and the `(2, 21, 22)` counterexample (mean `15.00 ≥ 14.86`,
`min = 2 < 3`) still shows S3 ⇏ S6. Only the identity claim is over-stated.

**Repair.** Rewrite item 4 as an inequality: `mean_s net_s = n_D · (acc(A) − acc(reference)) ≥
n_D · (acc(A) − max_{c∈C} acc(c)) ≥ 0.02 · n_D`, with equality iff the selected reference is also the
strongest comparator in `C` — which holds for the `displacement` disjunct by construction and for the
`common_displacement` disjunct only when `displacement` is not the strongest of the six.

### I-2. §5.6 imposes a scope requirement on §10.2 that §10.2 does not carry, and §6.4's unconditional reporting sentence was not re-checked as round-5 C-1 prescribed. Both sections are unchanged from v5.
*Attaches to:* §5.6 (v6:590); §10.2 (v6:1116-1121, unchanged); §6.4 (v6:812-827, unchanged); §13.

§5.6 now ends: *"The lineage(s) that ran, and the dataset(s) on which each passed its gates, **must be
named in §10.2's scope sentence**."* §10.2's scope sentence names the lineages (*"under the lineage(s)
that passed their instrument gates — named explicitly in the verdict"*) but says nothing about
datasets. The requirement and the section it binds disagree.

Round-5 C-1's repair also said *"Then re-check §6.4's unconditional reporting sentence"*: §6.4 requires
the `GATE-DOMAIN` recovery fraction and the raw-vs-head comparison *"on the verdict face"*
unconditionally, while §5.6 now permits Head-N — the only lineage `GATE-DOMAIN` is defined for — to be
dropped. §6.4 is byte-identical to v5. **The consequence is benign**: Head-N's 36 mints run
regardless (they anchor `GATE-FLOOR`), so `acc_ro` exists even on a drop, and rule 2 keeps Head-N on
every CLOSE path. But the re-check round 5 asked for was not performed and the sentence remains
literally unsatisfiable on a Head-N-drop SURVIVE.

**Repair.** Extend §10.2's scope sentence to name the dataset(s) on which each surviving lineage
passed, matching §5.6's requirement; and add one clause to §6.4 scoping the reporting duty to the runs
where Head-N's arena accuracy exists.

### I-3. §5.6's absence exemption says a dropped lineage's quantities are *"excluded from every decision rule"*, which the S4-family-at-`p = 1` rule contradicts; only the following sentence resolves it.
*Attaches to:* §5.6 (v6:604-609); §5.5 (v6:534-541); §13.

The exemption reads: *"Quantities belonging to a **dropped** lineage are recorded `INSTRUMENT_FAILED`,
are **not required**, and are excluded from every decision rule and from the S5 family."* S4 **is** a
decision rule, and the dropped lineage's 46 hypotheses are explicitly **in** its family at `p = 1`.
The next sentence resolves it correctly (*"The S4 family is not shrunk by a drop — it is frozen at 92
…"*), and the phrase *"and from the S5 family"* implies the S4 family is treated differently. So the
rule is determinate. But a context-free operator reading the exemption in isolation gets the wrong
answer, and this is the exact textual pattern round 4's C-2 and round 5's C-2 both punished.

**Repair.** Make the exemption self-contained: *"… are excluded from the evaluation of S1–S7 and from
the S5 family, and enter the S4 family only as `NOT_TESTED` with `p = 1` (§5.5)."*

### I-4. S7's head-space substitute is structurally faithful — but the statistic's **dispersion** differs qualitatively between the two spaces, and the small-set comparison operator is unregistered.
*Attaches to:* §5.2.2 (v6:415-435); §3.7 (v6:293); §15 item 4.

I verified the substitution is the right one. C01's `d_norm[split][m] = ‖l2(ow_m) − l2(std_m)‖`
(`contrast_blocks:1253-1259`) — each endpoint l2-normalised per modality, then the difference norm.
v6's `d_i = ‖l2(h_ow,i) − l2(h_std,i)‖` on the single fused block is the exact structural analogue with
the modality axis removed, and removing it is forced: `classifier.py:140-141` fuses by
`torch.mul(img_feats, text_feats)` under `fusion_mode == 'align'` and `:146` emits one 1024-d vector
(dims confirmed: head `head_dim = 1024` on the remint). Reading S7 on the raw features instead would
have made §3.6 false. **The departure is correct and correctly declared.** Two gaps remain:

* **Dispersion.** §5.2.2 records the structural difference (arena = train split ⇒ the small set is the
  bottom decile by construction) but not that the statistic's *shape* differs. Measurement α: in raw
  space the whole distribution spans `0.6146–0.7377` (HateMM), so C01's bottom decile separates rows
  differing by ~5 % in displacement norm — the "small displacement" set is barely small. In head space
  the distribution is unmeasured except for a trained median of `0.2301`, `2.9×` below the raw median.
  Whether the head-space bottom decile is a comparably narrow band or a genuinely distinct population
  is unknown, and it changes what S7 tests.
* **The comparison operator.** C01 uses `small_mask = dev_min <= threshold` (`:2049`). §5.2.2 says
  only *"the small set defined by the `0.1` quantile"*. Because the C06 arena is the population the
  quantile is taken from, `<` and `<=` can differ on the boundary rows; the operator is a decision
  parameter and should be frozen with the other five.

**Repair.** Freeze `<=` explicitly; and add one clause reporting the head-space `d_i` distribution
(min, `q₀.₁`, median, max) on the cells measured for **H-3**, so the record says what the bottom decile
actually is in the space S7 runs in.

### I-5. §7.9's `GATE-FLOOR` discharge is right for a much weaker reason than the evidence supports, and does not say which of the 42 banked quantities one cell covers.
*Attaches to:* §7.9 (v6:979-996); §6 `GATE-FLOOR` (v6:697); §15 item 5.

§7.9 claims *"a re-minted HateMM `s0/fold0` head reproduces the banked `fold_acc_deployed[0]` exactly
at 4 dp"*. I reproduced that and found it understates the result substantially: the re-minted
`K_train` is **bit-identical to the banked mint** (`max|diff| = 0.000e+00`, with `lab`, `fit_idx` and
`fold_of` identical). Accuracy agreement at 4 dp is a coarse consequence of an exact match; quoting the
coarse form invites the question §15 item 5 asks (*"is one exact match sufficient for a gate demanding
42 quantities?"*), which the exact form largely dissolves. I also recomputed **all 36** banked accuracy
quantities from the banked mints and every one reproduces, which separates the two things `GATE-FLOOR`
conflates: the *arena arithmetic* (36/36 verified here) and the *mint reproducibility* (1/66 verified,
bit-exactly).

**Repair.** State the bit-exactness rather than the 4-dp agreement; say explicitly that the discharge
covers one of 66 mints and that the arena arithmetic over all 36 banked quantities is separately
reproducible from banked artifacts at zero cost; and note that the remaining risk is confined to
per-mint nondeterminism, which DET-1 and the identical `script_sha256` control.

---

## MINOR

* **M-1.** §7.9 (v6:979) is placed **before** §7.8 (v6:1020). Reorder or renumber.
* **M-2.** §7.3 (v6:920) says *"**v6 adds two measured accuracies** and neither is a battery arm"* and
  then names one. My corpus grep finds exactly one new decimal in `[0.6, 0.99]` across v6. Either the
  count is wrong or a second accuracy is undisclosed; if the intended second is `0.2301`, it is a
  geometry figure, not an accuracy.
* **M-3.** §6.1's `ρ_raw` for HateMM `orthrot_83p8` prints `0.956893`; the value is `0.9568935731`,
  which is `0.956894` at 6 dp. One of 26; not `ρ*`; the sort order and every gate are unaffected. Round
  5 read it as exact, so the truncation is now twice-recorded — worth a one-digit erratum under the
  campaign's numeric-provenance discipline.
* **M-4.** §5.2.2 (v6:442-444) says C01 places the condition in `decision` *"and **its**
  `required_halt_only_validity_guards` (seven entries) does not contain it"*. The seven-entry list is at
  **`output.decision_schema.required_halt_only_validity_guards`**, not under `decision`. The substance is
  true — I read the list verbatim and it does not contain
  `require_no_small_displacement_dominance` — only the location is wrong. Same class as round-4 M-1 and
  round-5 M-2.
* **M-5.** §5.5 (v6:541) says §8 Phase 4's 92 comparison-cells is *"now an exact count rather than an
  upper bound"*. On the drop path the dropped lineage's hypotheses are `NOT_TESTED` and therefore not
  computed, so only 46 cells execute. It remains an upper bound; the direction is conservative.
* **M-6.** §5.2's S4 row (v6:357) lists `holm_alpha = 0.05` bare while explicitly prefixing
  `statistics.seed`. Both `holm_alpha` and `holm_metrics` live under **`statistics`**, not `decision`;
  `decision` has neither key. Prefix them for the code lineage.

---

# PART C — REQUIRED RULINGS

## Deliverable 6 — is there any gate that can fire on a **warranted CLOSE**? All twenty.

**NO for nineteen. One carries an under-evidenced false-HALT probability (`GATE-ALGEBRA`, H-3), and
like round 5's `GATE-ZEROOP` it converts CLOSE → HALT, never CLOSE → SURVIVE.** I re-tested all twenty
rows, including `GATE-RHORAW`, which is new and global.

| gate | scope | can it fire on a warranted CLOSE? |
|---|---|---|
| `GATE-DET1` | G | **No** — thread environment; science-independent. Verified: `det1_assert` checks four env vars at `headspace_mint.py:75-79`. |
| `GATE-SHA` | G | **No** — provenance; all 21 digests verified (V1). |
| `GATE-FOLD` | G | **No** — fold parity; my remint returned `[True]×5` and `:321-325` writes only after the assertion, so the banked-flag re-read is faithful on resume. |
| `GATE-FLOOR` | G | **No** — reproduces banked native floors, independent of every arm comparison. Discharged bit-exactly (V7). |
| `GATE-POP` | G | **No** — populations and constants; all re-derived exactly (V11). |
| `GATE-C01PARITY` | G | **No** — raw algebra; `0.000e+00` on 13 arms, both datasets (V9). |
| `GATE-ROWSUBSET` | G | **No** — key-level row-subset identity; `0.000e+00` (V9). |
| `GATE-NULLREMOVED` | G | **No** — population predicate; `{355}` / `{}` verified. |
| `GATE-IDPARITY` | G | **No** — verified to hold directly on both datasets and both policies. |
| `GATE-ZEROMASK` | G | **No** — feature-space measurement; `{355}` / `{}` verified. |
| `GATE-LEDGER` | G | **No** — bookkeeping; the process count is predictable on fresh, resumed and HALTed runs. |
| **`GATE-RHORAW`** | **G** | **No.** The new gate is a property of the ro caches, the raw builder and the frozen table — I reproduced all 26 values, so it fires only on cache or builder drift, which is exactly what should HALT. Making it global is right: as round-5 I-2 argued, under per-lineage scoping a drift would have dropped one lineage while the other ran on the same drifted data. **The split creates no new failure mode**: the raw leg is non-decisional (§3.6), so a `ρ_raw` failure cannot be correlated with the verdict. |
| `GATE-ORBITDISP` | L | **No** — measurement (V10) puts a trained deployed head at `0.34`–`0.67` against `ρ* ≈ 0.97`, a margin of ~0.30 across all 36 banked cells. The both-above branch is explicitly a warranted CLOSE. |
| `GATE-ARENA` | L | **No.** The lower bound is scoped to `endpoint_std`; the upper `≤ 0.98` catches a leak and cannot fire downward. I re-checked every §5/§6/§12 clause for a re-entering lower bound on a real arm and found none, so §6.3's invariant holds. |
| `GATE-NESTED` | L | **No** — OOF construction. |
| `GATE-SELFTEST` | L | **No.** Round-5 I-1 is fixed: v6 writes `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` per seed (v6:709), which is an identity given the definitions and cannot fail on a correct run. The reference change (D-1) does not affect it — the identity holds against *any* fixed reference arm. |
| `GATE-ZEROOP` | L | **No** on the science. Round-5 H-3 is fixed: aggregation is per `(dataset, seed, lineage)` pooling five folds, so the denominator is `n_D` and the cap is the `1 %` it is described as. Residual false-HALT probability is disclosed and one-directional (§5.9 item 5). |
| **`GATE-ALGEBRA`** | **L** | **No CLOSE→SURVIVE inversion, but a CLOSE→HALT probability that is asserted small on inadequate evidence (H-3).** Key-level `2e-6` on a max-over-rows residual, discharged from a median on 1 of 60 cells and 1 of 2 lineages. Direction is safe; magnitude is unquantified. |
| `GATE-DOMAIN` | R | **No** — reports, no bar (but see I-2 on its unconditional sentence). |
| `GATE-DEVFID` | R | **No** — reports. |

**Verdict-path enumeration (deliverable B.1).** I enumerated `(Head-N gates × Head-R gates × clears ×
drop state × global gates)` from §5.6's three rules alone. The rules are **total and mutually
exclusive**: rule 1 and rule 2 cannot co-fire (rule 2 requires *"neither clears"*), and rule 3 is
*"otherwise"*. Six reachable configurations, one published state each: both passed + ≥1 clears →
SURVIVE; both passed + neither clears → CLOSE; one passed + it clears → SURVIVE; one passed + it does
not clear → HALT; neither passed → HALT; any global gate fails → HALT. No configuration is unmapped, no
gate failure is reportable as closure, and the only absent quantity reaching a verdict does so via the
declared-drop exemption. **The drop × dataset × absence interaction I stressed hardest**: a lineage
whose gate *quantity* is absent has by construction not "passed its per-lineage gates", so it is
dropped rather than reaching the surviving-lineage absence rule; the outcome is then HALT unless the
clean lineage clears, in which case SURVIVE — never CLOSE. Verified consistent.

## §15.1 — D-1's dataset axis: trace the Head-N-fails-on-HateMM-only path

**Exactly one published state, and no reading manufactures a CLOSE.** The new sentence (v6:571-573) has
two clauses: *"A lineage that fails a per-lineage gate on **ANY** dataset is marked `INSTRUMENT_FAILED`
and is dropped on **BOTH** datasets"*, and — closing the definitional loophole round-5 C-1's reading B
exploited — *"A lineage 'passed its per-lineage gates' **only if** it passed every per-lineage gate on
**BOTH** datasets."* Traced: Head-N fails `GATE-ARENA`'s lower bound on HateMM only ⇒ `INSTRUMENT_FAILED`,
dropped on both. Head-R passes everywhere and does not clear S1–S7. Rule 1: no lineage that passed
clears ⇒ no. Rule 2: *"both lineages passed"* is false ⇒ no. Rule 3 ⇒ **HALT**. There is no reading
under which Head-N "passed", because the second clause is a definition and not a heuristic. **C-1 is
closed.**

**Is the conservative direction right here, given that it converts some CLOSEs into HALTs and a HALT
spends the falsifier without discharging it?** **Yes.** A CLOSE is terminal — by `falsifier_spec` it
means the `1.7–2.5 GPU-h` extraction *"is never queued"* — while a HALT is recoverable by
re-preregistration at the same `$0`. Asymmetric costs justify the asymmetric lean, and §5.6's own
warrant (*"a CLOSE rests on two clean negatives, and a lineage that was clean on only one dataset
supplies neither"*) is the right one. Residual: the `(dataset, lineage)` cross reaches §6's scope
column (v6:688-690) and §5.2/§5.3 by reference to §5.6, but **not** §10.2 and **not** §13 (→ I-2, C-1).

## §15.2 — the dataset axis: is the cross everywhere it must be?

Answered above. §5.2 ✓ (by reference), §5.3 ✓ (by reference), §6's scope column ✓ (explicit legend),
§10.2 ✗ (I-2), §13 item 20 ✗ — **the item does not exist** (C-1).

## §15.3 — the frozen 92-family: is `p = 1` padding the right convention?

**The convention is coherent; the justification for it is false (H-1); and 92 is the right choice for a
reason v6 does not give.** Padding at `p = 1` is statistically coherent — an untested hypothesis that
cannot reject is exactly a hypothesis at `p = 1`, and because `holm_adjust` sorts ascending with a
running max, padded values sort last and cannot affect any witness rejection (which is why the measured
three-way equality at `1/2001` holds). Recomputing at 46 would be **more powerful**, not more honest:
it would make S4 easier, SURVIVE easier and CLOSE harder, i.e. it is the *conservative* direction under
§4's own definition. **My ruling: keep 92**, because a preregistered family size that cannot be moved by
a realised run is the property that makes the design auditable, and because the drop is caused by an
instrument gate whose outcome the analyst does not control. But the design must stop claiming the choice
is free of consequence: it is strictly the CLOSE-easing option and §5.9 must say so.

## §15.4 — S7's head-space statistic

**The departure is the right one, and forced.** Verified in I-4: the substitute is the exact structural
analogue with the modality axis removed, the removal is architectural rather than chosen, and reading S7
on the raw features would have falsified §3.6. **The arena-is-train-split note is necessary but not
sufficient** — it records that the small set is the bottom decile by construction, but not that the
statistic's dispersion may differ qualitatively between the two spaces (in raw space the whole
distribution spans ~20 % and the bottom decile is a ~5 % band; the head-space shape is unmeasured). Add
the distribution summary and freeze the `<=` operator (I-4). And dispose of `tiny_ok`, the second limb of
the C01 condition S7 names as its source (H-3c).

## §15.5 — `GATE-FLOOR`'s discharge: is one cell sufficient?

**Yes — and by a stronger argument than the one offered.** One cell reproducing at 4 dp would be weak
evidence for a gate demanding 42 quantities. One cell reproducing **bit-exactly at the key level** is
strong evidence, because it demonstrates the mint path is deterministic rather than merely accurate,
and determinism is the property the other 65 mints need. I additionally verified that all 36 banked
accuracy quantities are recomputable from the banked mints, which discharges the arena-arithmetic half
of `GATE-FLOOR` completely and at zero cost. **No further mints are needed before freeze.** The design
should bank the stronger claim (I-5), and — because the same trained head is already on disk — should
spend the marginal zero cost on **H-3**'s tail measurements instead, which is where the real
uncertainty is.

## §15.6 — the enumeration: is there another uncounted loop, and is Phase 7 still honest?

**No new uncounted loop.** Round 5's two are counted (Phase 1c `66 → 67`, new Phase 1e at 66 × 0.5 ms),
Phase 1b's factors are named, and Phase 7's enumeration now includes `GATE-ZEROMASK`, `GATE-FOLD`'s
in-process leg and `mints_present_before_arena`. I re-derived every count independently before reading
the table and re-multiplied all 26 rows (V12). I checked the same places round 5 did and three it did
not: S5's null rebuild (arms and folds inside `U4` ✓), the `GATE-ORBITDISP` per-fold `ρ` loop (Phase 2D
at 62 ✓), and the `GATE-SELFTEST` per-arm-per-seed loop (Phase 7 at 168 ✓). Nothing iterates over folds,
seeds, lineages or datasets without appearing in §8.

**Is Phase 7's *"sub-`0.1 s` class"* still honest?** **Yes, but it is now the load-bearing rounding in
the table and should be stated as such.** Phase 7 carries nine named items at a declared `0.1 s` upper
bound, and Phase 1e carries a measured `0.033 s` at a printed `0.1`. Both round **up**, so the total
`2927.6` over-states the true `2927.517` — conservative. Because two rows are now upper bounds rather
than measurements, §8's *"the printed product column now sums to the total directly"* (round-4 M-2's
adoption) is true but no longer means "every row is a measured product". One clause.

## Deliverable 7 — **is C01's accuracy-based reference selector pre-registration-safe?**

**Yes — and on a stronger warrant than §5.2.1 offers. Adopt it.**

v6's argument is that `select_strongest_ordinary_control` is a deterministic *rule* evaluated at run
time, so it introduces no researcher degree of freedom even though the selected arm is unknowable in
advance. That is correct as far as it goes, and it is the standard test: a pre-registration is
contaminated when the analyst can *choose* after seeing the data, not when a frozen rule *computes* a
choice from the data. Three properties of this particular selector make it safe rather than merely
arguably safe, and I verified each:

1. **The rule is itself a frozen config constant**, not only a code path:
   `transforms.small_displacement_gate_reference =
   "strongest_ordinary_control_by_accuracy_then_macro_f1_then_frozen_gain_controls_order"`. It is also
   written onto C01's verdict face as `displacement_stability.selection_rule`. There is no version of
   this design in which the rule is inferred from source; it is read from a sha-gated JSON.
2. **The tie-break is total and label-independent in its final term:**
   `(accuracy, macro_f1, −frozen_order_index)` over a frozen five-member list, so no run can produce an
   ambiguous selection. `select_strongest_ordinary_control:1940-1948` additionally `die()`s on an empty
   family, on duplicates, on a missing evaluation, and — importantly for C06 — if `common_displacement`
   ever enters the control family. I confirmed `displacement` is **not** in `gain_controls`, so neither
   real arm can be selected as its own reference.
3. **The direction is conservative and the selection is adversarial to the hypothesis.** By
   construction `acc(strongest control) ≥ acc(endpoint_std)`, so S6 against the selected arm is
   strictly harder than v5's version. A selector that picks the *strongest* opponent cannot be gamed
   toward SURVIVE; the analyst's incentive runs the other way.

**What rides on it, and whether the coupling is a problem.** S3's `max_{c∈C}`, S6, S7 and
`GATE-SELFTEST` are now coupled to one arm per `(dataset, lineage)`. I checked each: `GATE-SELFTEST`'s
identity holds against *any* fixed reference, so the coupling is inert there; S6 and S7 are required by
C01's own `:2724` consistency `die()` to share a reference, and inheriting that assertion is right; S3
is coupled only partially and that partiality is I-1. **The coupling is a feature** — it is what makes
S3 and S6 statements about a common baseline rather than two unrelated bars.

**One caution for the freeze.** The selected arm is a *run-time* quantity that determines two SURVIVE
conjuncts, so it must be recorded on the verdict face — §5.2.1 says *"recorded on the verdict face"* and
that must survive into §13, which it currently does not (C-1). Round 5 imposed the identical condition
on `GATE-ZEROOP`'s run-time residual and was right to.

## Deliverable B.3 — cross-document consistency

I found no assertion in v6 that contradicts a repository source, other than the internal contradictions
recorded as C-2. Specifically verified: `TARGET_STATE.json`'s `falsifier_spec` and
`falsifier_design_constraints` are quoted **verbatim** at §1; `rotation_family_precision_R14`'s Givens
claim matches `orthogonal_blocks:1272` and the measured `θ = 0` / `θ = 45` guards
(`8.9407e-08` / `1.1921e-07`); §1's erratum is right (`-0.009345794392523366` → `−0.0093`, MHC-ZH
`-0.02564102564102566` → `−0.0256` exact), as are the primaries and references it implies
(`0.8598130841` vs `common 0.8691588785`; `0.8589743590` vs `endpoint_concat 0.8846153846`); every C01
constant §5 and §6 cite reproduces from `c01_a0_v2.json`; `require_no_small_displacement_dominance` is
in `decision` and not among the seven halt-only guards; `classifier.py:81-82`, `:140-141` and `:146`
are all correctly cited this round; the four items of new code are **absent** from the tree, as they
must be; all ten `vsw_ckpt` files, all six banked arena JSONs and exactly 36 `mint_*.npz` exist.
Hard constraints: none touched — I re-read `d_no_other_relaxation` and checked each condition, and
`avg_score` is confirmed in `decision.gain_controls`, so §10.4's no-ensemble reading holds.

## Deliverable B.2 — freeze-readiness

**Not freeze-ready, on five items a context-free operator cannot execute or audit as written:** (1) the
§13 handoff the separate code-review lineage consumes (**C-1**); (2) the §14 record it would audit
against, which contradicts the body twice (**C-2**); (3) §5.5's stated guarantee, which is false, and
the §5.9 disclosure it displaces (**H-1**); (4) §7.3's blindness warrant (**H-2**); (5)
`GATE-ALGEBRA`'s tail evidence and `tiny_ok`'s disposition (**H-3**).

**Everything else is freeze-ready.** The run boundary is unambiguous — one `sbatch`, 8 CPU / 32 GB, no
`--gres`/`--time`/array/dependency/requeue, 73 processes in the order 66 mints → 6 fidelity → 1 arena,
`GATE-SHA` once in the driver before any of them, `GATE-POP` before any population-consuming gate. The
cloud-routing dismissal is correct: `GATE-FLOOR` anchors to floors measured locally on `foscsmlprd01`,
so CLAUDE.md's same-table-same-hardware condition cannot be met off-node — and my remint reproducing
the banked keys **bit-exactly on this node** is direct evidence that the local anchor is the real one.
`rule_1_compute_projection` is satisfied (V12, §15.6). `rule_2_heartbeat`: **v6 changes no interval** —
the longest un-instrumented span remains one `GATE-C01PARITY` dataset at `11.27 s` (`14.1 s`
conservative); §7.9's leg adds no run-time phase, and Phase 1e is inside a per-mint line. All 21
sha256s recompute; all constants are pinned with values.

## Deliverable 9 — can the falsifier discharge its written condition at `$0`?

**Yes.** Nothing I found requires a GPU, an extraction, new data or a redesign. Two of the five
substantive findings (C-1, C-2) are edits to sections that were simply not touched this round; two
(H-1, H-2) are corrections to sentences whose repairs are already present; one (H-3) needs a handful of
tail measurements on a head that is already minted, at a few CPU-minutes. The projection —
`48.8` corroborating / `61.0` conservative CPU-minutes on 8 CPU / 32 GB — is right; I re-derived it
from the design's own structure. After six rounds the instrument is verified end to end by execution:
the three-population contract, the mask convention, the row-subset identity, the arena constants, the
`ρ*` calibration, the two-block anchor, the head-space build and its dimensions, the head-space
statistic's structural fidelity, and now the mint path's bit-level reproducibility. **The remaining
failure surface is narrower than any previous round and lies almost entirely in the record rather than
the science.**

---

# PART D — DISPOSITION AUDIT OF §14's ROUND-5 BLOCK, BY EXECUTION

**There is no round-5 block in §14 to audit (C-2), so I audited all 18 against the artifact and the
primary sources directly.**

**Result: 9 VERIFIED ADOPTED, 8 PARTIAL, 1 NOT ADOPTED.** Every PARTIAL has the same shape — the main
text carries the repair, the §13 handoff limb does not — and the single NOT ADOPTED is the finding whose
entire content *was* a §13 item.

| finding | audit result |
|---|---|
| **C-1** dataset axis | **PARTIAL.** §5.6's two-clause rule is present and airtight — I traced the motivating path and it publishes exactly HALT (§15.1). §6's scope-column legend carries the `(dataset, lineage)` cross. **But** §10.2 does not carry the naming §5.6:590 requires of it, §6.4's unconditional sentence was not re-checked, and §13 item 20 does not exist. → **I-2**, **C-1**. |
| **C-2** absence + family | **PARTIAL.** The exemption is present and correctly scoped (*"absence by declared drop is lawful; absence by computation failure in a surviving lineage still HALTs"*) — I confirmed an implementer cannot confuse the two, because a lineage with an absent gate quantity has not "passed" and is dropped rather than reaching the rule. The family is defined on every path. **But** round 5's prescribed 46-family was rejected without being recorded as a rebuttal, the invariance warrant offered instead is false, §14's H-3 row was not fixed, and §13 item 21 does not exist. → **H-1**, **C-2**, **C-1**, **I-3**. |
| **C-3** S7's five parameters | **PARTIAL.** All five are frozen and I verified each against `c01_a0_v2.json` and `displacement_audit`: threshold `0.5` ✓, reference via `select_strongest_ordinary_control` with the `:2724` consistency assertion inherited ✓, head-space one-block statistic ✓ (structurally faithful — I-4), seed axis `3/3` ✓, zero-fix convention ✓. **But** §13 item 15 was not extended, and C01's `tiny_ok` limb of the same check has no disposition. → **H-3**, **C-1**, **I-4**. |
| **H-1** the 42-of-92 floor | **PARTIAL.** §5.5 is corrected to the witness's 22 or 24 comparators and I reproduced both the corrected floor and the `2/2001` degradation. **But §14:1321 still states "42 of 92"**, so the document now carries both. → **C-2**. |
| **H-2** S5's statistic | **PARTIAL — substantively excellent.** §5.4.1 pre-registers the null construction, the statistic and its seed axis, the p95 convention, the p-value form, the 4-member family and the `n ≤ 12` feasibility bound (all verified, V6), and disposes of all seven of C01's halt-only guards rather than four. §5.4 adds the cross-lineage draw-sharing clause round 5's §15.4 flagged. **But** §13 item 16 was not extended. → **C-1**. |
| **H-3** `GATE-ZEROOP` granularity | **PARTIAL.** §6.5's aggregation bullet is exactly as prescribed — per `(dataset, seed, lineage)`, pooling five folds, denominator `n_D`, fail-if-any-cell — and the run-time residual is now recorded on the verdict face, discharging round-5 §15.6's first condition. **But** §13 item 10 was not extended. → **C-1**. |
| **I-1** `GATE-SELFTEST` symbol | **PARTIAL.** §6:709 now reads `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` per seed and §5.1 defines `acc_s`. The identity is now true and cannot fail on a correct run. **But** §13 item 16 was not extended. → **C-1**. |
| **I-2** `ρ_raw` global | **VERIFIED ADOPTED.** `GATE-RHORAW` split out, scope **G**, placed beside `GATE-C01PARITY` in §5.6's global list, with the reasoning round 5 gave. Gate count re-derived to twenty and the header, table and both §5.6 lists agree (V12). Creates no new failure mode (PART C). |
| **I-3** uncounted loops | **VERIFIED ADOPTED.** Phase 1c `66 → 67`, new Phase 1e, Phase 1b's factors named, Phase 7's enumeration extended. Total re-multiplies to `2927.6` (V12). |
| **I-4** `GATE-FLOOR` untested | **VERIFIED ADOPTED via option (i).** §7.9 discharges it by measurement, and I reproduced the measurement and found it stronger than stated (bit-exact keys). → **I-5** on the framing only. |
| **I-5** S7 eases CLOSE | **VERIFIED ADOPTED.** §5.9 item 6 states the direction change and the correct warrant (*"a CLOSE is what S1's failure warrants, not that the change is direction-neutral"*), and item 7 adds the symmetric disclosure for D-1. |
| **I-6** §13 item 19 | **NOT ADOPTED.** §13 is byte-identical to v5; item 19 does not exist and the one-construction claim appears in no handoff form. I hit the predicted failure independently (V9). → **C-1**. |
| **M-1** `−0.0094 → −0.0093` | **VERIFIED ADOPTED.** §1 carries the erratum with the exact stored value and the correct reasoning; I confirmed both figures from `C01_A0_OUT.json`. |
| **M-2** `classifier.py:116-120` | **VERIFIED ADOPTED.** §3.4 now cites `:140-141` (`torch.mul` under `fusion_mode == 'align'`) and `:146` (`embed = self.mlp[:-2](x)`), both correct, and records the correction. |
| **M-3** `:322` → `:322-324`/`:323` | **VERIFIED ADOPTED** in both §3.3 and §12; `np.savez` spans `:322-324` and `lab_dev` is on `:323`. |
| **M-4** stale residual illustration | **VERIFIED ADOPTED.** §6.5 is rewritten around a trained-head measurement with the untrained range recorded as the reason not to calibrate on it. **Residual: H-3** — the replacement is a median for a max, on one cell. |
| **M-5** gate count | **VERIFIED ADOPTED, both limbs.** Header and table both say twenty, `12 G / 6 L / 2 R`, matching §5.6 (V12). The supersession header's `0.18–0.23` is gone; only §6.2's correct `0.15–0.23` remains. |
| **D-1** (designer self-found) | **REAL, and the repair is SOUND.** Verified from the source and the executed output (V2), with a stronger warrant available (measurement γ). It opens **one** seam: §5.9 item 4's *"one quantity"* claim, because S3's comparator set for the primary has six members while the reference is selected over five (**I-1**). It also correctly inherits C01's `:2724` S6/S7 consistency assertion and correctly records `endpoint_concat`'s `diagnostic_only` role, both of which I verified. |

**Adoptions that could have broken each other** — the mechanism that produced round-4's C-2 and
round-5's C-2. I checked the three most likely collisions. (i) **D-1 × S3**: the reference change
coupled S6 and S7 to an arm selected over five controls while S3 ranges over six — collision found,
**I-1**. (ii) **C-2's family freeze × H-1's corrected floor**: both edit §5.5, and the freeze's warrant
is refuted by the floor paragraph immediately above it — collision found, **H-1**. (iii) **I-4's
discharge × §7.3's blindness discipline**: discharging `GATE-FLOOR` required training a head, which
falsifies §7.3's blanket untrained-head sentence — collision found, **H-2**. **All three of this round's
Highs are adoption collisions**, which is the same generative mechanism every previous round has
recorded, now operating between §7 and §5 rather than within §5.

---

# PART E — ADDITIONS TO §13's HANDOFF

§13 currently has **eighteen** items and must reach **twenty-two plus three extensions** before freeze.
Round 5's PART F list stands verbatim (items 19–22; extensions to items 10, 15, 16). Four further
additions from this round:

23. **The arm→formula map is not determined by §3.4's prose (C-1).** Item 19 must say that the map is
    pinned by `GATE-C01PARITY` against `prepare_views` and by nothing else, and must name
    `common_interaction = l2(common ⊙ displacement)` as the concrete instance two independent reviewers
    mis-derived. A code lineage that reimplements from the document will produce a wrong arm and a
    passing-looking build.
24. **The reference arm is a run-time decision quantity (D-1).** That `select_strongest_ordinary_control`
    is called over `decision.gain_controls` within each `(dataset, lineage)` cell, that S6, S7 and
    `GATE-SELFTEST` all consume the **same** selection, that C01's `:2724`-equivalent consistency
    assertion is implemented, and that the selected arm name is written to the verdict face.
25. **The head-space displacement tail (H-3).** That `min_i d_i` and the fraction `d_i ≤ 0.001` are
    computed and recorded per `(dataset, seed, fold, lineage)` cell alongside the `GATE-ALGEBRA`
    residual, so the max-vs-median gap is auditable at run time rather than assumed.
26. **The trained-head blindness boundary (H-2).** That the battery computes **no** vote on any
    ro-derived arm outside the arena phase, and that the `GATE-FLOOR` phase votes only on native keys —
    the assertion that replaces §7.3's untrained-head warrant.

*Extension to item 5:* the population-derived constant list must include S7's `<=` small-set operator
(I-4) once frozen.

---

*Read-only review. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission or
commit occurred. `TARGET_STATE.json`, all six drafts, all configs and all five prior reviews were read
and not modified. All computation was `sha256sum`, file reads, numpy/torch-CPU re-derivation on
already-banked **train-split** caches and banked mint checkpoints, and one authorized CPU fold-head
remint written only into the session scratchpad. No arm accuracy was computed at any point. A GO on
this lineage would authorise nothing to run: the design would still require freeze with hashes, a
**separate** independent code/resource review lineage over the executable reaching its own `0C/0H/0I`,
and main-dialogue authorization.*
