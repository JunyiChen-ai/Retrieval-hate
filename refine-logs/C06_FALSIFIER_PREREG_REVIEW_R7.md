# C06 `$0` falsifier — independent design review, **ROUND 7**

**Artifact:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V7.md` (unfrozen).
**Reviewer:** fresh, independent of rounds 1–6 and of the designer.
**Date:** 2026-08-04. **Read-only.** No GPU, SLURM, Modal, arena run, cache write, test-split
access, job submission, commit, or `TARGET_STATE.json` write occurred. No head was minted. All
computation was `sha256sum`, file reads, and numpy/torch-CPU re-derivation on already-banked
**train-split** caches and banked mint checkpoints, written only into the session scratchpad.
**No arm accuracy was computed at any point.**

---

# VERDICT

## **REVISE — 4C / 2H / 3I + 4M**

Round 6 judged the science layer clean and said the remaining failure surface lay *"almost entirely
in the record rather than the science."* **I do not agree that it is only the record.** §13 and §14
really were rebuilt this round — I verified that by diffing every section against v6, and §13's
twenty-six items are contiguous, complete against every limb the six rounds prescribed, and
internally consistent. But the round-6 repair that mattered most — pinning §3.4 so a code lineage
cannot rebuild the wrong battery — **fixed the arm two reviewers got wrong and left a second,
larger hole in the same sentence**, and I measured that the design's only anchor gate does not
close it. That is C-1, and it is a wrong-verdict path, not a bookkeeping defect.

Separately, three of §14's round-6 disposition rows claim repairs that are not in the artifact.
Round 6 caught v6 by diffing §13/§14 against v5; I did the same against v6, and the same defect
class has moved rather than disappeared: this time the sections **were** edited, but the disposition
table describes edits that were not made to §5.2.2, §10.2, and a §5.2.3 that does not exist.

---

# PART A — THE TWELVE §2 VERIFICATIONS

| # | claim | result |
|---|---|---|
| **V1** | 21 sha256 in §11 match disk; both provenance chains sha-gated in source | **VERIFIED.** All 21 recomputed and matched byte-for-byte (7 imported modules, 6 read-for-definition files, 8 input caches). Chain gating confirmed in source: `c01_policy_contrast_a0_v4.py:52-55` (`load_frozen_v3`, raises on `V3_SOURCE_SHA256` drift) → `_v3.py:48-51` (`load_frozen_base`, raises on `BASE_SOURCE_SHA256` drift). `scientific_thresholds_exact: true` present in both `c01_a0_v4.json` and `c01_a0_v3.json`. **See I-3** for load-bearing inputs §11 names *without* digests. |
| **V2** | §13 has exactly 26 items, contiguous; every `§13 item N` body reference resolves | **VERIFIED.** Items `(1)`…`(26)`, no gaps, no repeats. Twelve `§13 item N` references in the body (§3.4→19, §3.7→5, §6.1→7, §7.3→26, §13.1 preamble, §14 rows→20/21/15/16/10/19/26, §15→25); **all twelve resolve to the correct item.** One caveat: item 5's `<=` limb points into §3.7's constant table at a row that does not exist (**C-4**). |
| **V3** | §14's two stale assertions gone | **VERIFIED.** "42 of 92" survives only at §5.5:544 (quoting v5 in order to correct it) and in §14/§15 corrective rows — **no sentence asserts the floor**. "S6's net-fix reference" appears at §14:1558 as an explicitly **struck** ruling with its reason (*"overturned by D-1"*), never among rulings carried unchanged. Framing checked, not just the string. Both purges are real. |
| **V4** | the `common_interaction` pin | **VERIFIED FOR THAT ARM — AND THE PIN IS STILL INCOMPLETE.** I rebuilt all 13 arms from §3.4's prose and got `max|diff| = 0.000e+00` on **both** datasets, all 13 arms, **first attempt**, without consulting `contrast_blocks` for the arm→formula map. The `common_interaction` trap is closed: the pinned `paired(common, l2(common ⊙ displacement))` is exactly `prepare_views`. The 4×1024 / 9×2048 dimension split resolves the residual `fuse`-vs-`paired` ambiguity for the single-block arms. **But** I had to supply one step the prose never states — that the endpoint blocks are `l2_rows`-normalised *before* the contrast blocks are formed — and the reading that omits it is not caught by the anchor gate. → **C-1**. |
| **V5** | the four-cell tail | **NOT RE-DERIVABLE UNDER THIS AUTHORIZATION; INTERNALLY CONSISTENT.** No mint was taken (permitted compute excluded it). `headspace_mint.py` suppresses state-dict saves and banks only `K_train`/`K_dev` on **native** keys, so no banked artifact contains a trained head's ro-forwards; the four-cell table is reproducible only by re-minting. Its internals are consistent with what I could check independently: the `θ = 45` residual is governed by `min d_i` (mechanism verified from `orthogonal_blocks:1272` against `contrast_blocks:1246-1265`), and on the **raw** features I reproduced the published guards exactly — `θ = 0` `8.941e-08` both datasets, `θ = 45` `1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH), matching §1. **The table remains single-source: no round has reproduced any of its four cells.** |
| **V6** | `GATE-FLOOR`'s bit-exact discharge | **NOT RE-DERIVED (no mint).** Round 6 verified bit-identity directly; v7 records it as bit-exactness of the re-minted `K_train` per round-6 I-5, which is the stronger and correct framing. I verified the surrounding claims: all ten `vsw_ckpt/{hatemm,zh}/f{0..4}.npz` exist and are what `headspace_mint.py:209-216` reads; the six `headspace_arena_*_OUT.json` exist; `:321-325` writes the `.npz` only after the parity assertion at `:203-216` passes, so §3.2's resume-safety argument holds. |
| **V7** | H-1's counterexample table | **VERIFIED EXACTLY, by executing `holm_adjust`.** `m = 92`: 24×`1/2001` → **24/24**; 23×`1/2001` + 1×`2/2001` → **23/24**; 24×`2/2001` → **0/24**. `m = 46`: **24/24** in all three. Also confirmed: the 22-comparator `displacement` disjunct rejects 22/22 at `m = 92`; padding with `1.0` instead of `0.5` still gives 24/24; `92 × 2/2001 = 0.091954 > 0.05` and `46 × 2/2001 = 0.045977 ≤ 0.05`. §5.5's table is right to the cell. |
| **V8** | D-1 | **VERIFIED.** Two `fix_break` sites with different references: `:1725` reads `config["retrieval"]["fix_break_reference"]` (= `endpoint_std`, confirmed in `c01_a0_v2.json`) and feeds the **reporting** field at `:1731`; `:2705` uses `select_strongest_ordinary_control` and feeds `checks["net_fixes"]`. Executed `C01_A0_OUT.json`: `net_fixes.reference` = **`common`** (HateMM) / **`endpoint_concat`** (MHC_zh). `transforms.small_displacement_gate_reference = "strongest_ordinary_control_by_accuracy_then_macro_f1_then_frozen_gain_controls_order"` ✓. The selection rule at `:1955-1962` is `max` by `(accuracy, macro_f1, −control_order.index(name))` — exactly §5.2.1's statement. The consistency `die()` guard is `if small_gain_gate["reference"] != strongest_control_name:` at `:2724` ✓. **D-1 is real and the repair is correct.** |
| **V9** | S7's six parameters + `tiny_ok`'s two constants | **FIVE OF SIX VERIFIED; THE SIXTH IS NOT FROZEN; `tiny_ok` HAS NO SECTION.** `max_small_displacement_fix_fraction = 0.5` ✓, `small_displacement_train_quantile = 0.1` ✓, reference rule ✓ (V8), head-space one-block statistic ✓ (forced by `classifier.py:140-141`/`:146`), per-seed `3/3` ✓. **The `<=` operator is frozen nowhere in §5 or §3.7** — it appears only in §13 items 5 and 15 (**C-4**). `tiny_ok`'s constants: `tiny_displacement_epsilon = 0.001` appears once, in §7.8's dry-check prose; **`max_tiny_displacement_fraction = 0.05` appears nowhere in v7**, and §5.2.3 — the section five places say disposes of `tiny_ok` — **does not exist** (**C-2**). |
| **V10** | `ρ*` and all 26 `ρ_raw` at 6 dp | **25 OF 26 VERIFIED; `orthrot_83p8` (HateMM) DISAGREES.** `ρ*` `0.968176` / `0.977223` ✓ (`endpoint_std` supplies both; runners-up `0.964446` / `0.969686` = `common` ✓). Twenty-five values reproduce at 6 dp on the arena rows. **`orthrot_83p8` HateMM: I measure `0.9568933249` → `0.956893`, not v7's `0.956894`.** The discrepancy is a reduction-order artifact, not an error: float32 accumulation gives `0.9568933249`; float64 gives `0.9568935731` — round 6's exact figure. v6's value was right for one reduction and v7's for the other, and §6.1 does not say which. → **I-1**. `ρ` over 744 rows shifts by `1.301e-03` ✓. |
| **V11** | every population-derived constant in §3.7 | **VERIFIED, except the `<=` row, which does not exist.** HateMM full `n = 744` (298 pos / 446 neg, majority `0.599462 → 0.5995`); arena `n = 743` (**297** / 446, majority `0.600269 → 0.6003`); MHC-ZH `n = 579` (180 / 399, majority `0.689119 → 0.6891`), full = arena. Exact-zero rows: HateMM `{355}` in both modalities of both ro caches, MHC-ZH none. `ids` order-identical and `labels` identical across policies on every file. Bands `[0.6203, 0.98]` / `[0.7091, 0.98]` ✓. Tie caps `⌊0.01×743⌋ = 7` / `⌊0.01×579⌋ = 5` ✓. `GATE-ROWSUBSET` reproduced: the `n = 743` all-False build is **bit-identical** (`max|diff| = 0.000e+00`) to the `n = 744` one-hot build restricted to the surviving rows, all 13 arms. §3.7's table contains **no `<=` operator row**, which §13 item 5 asserts it does (**C-4**). |
| **V12** | §8 sums to `2927.6`; `×1.25 = 3659.5`; §6 has twenty rows `12 G / 6 L / 2 R` matching §5.6 | **VERIFIED — independent re-multiplication (F118).** The 26 printed products sum to **`2927.6`**; `× 1.25 = ` **`3659.5`**. Every product re-derived from its unit and count: `174×0.0461=8.02→8.0`; `67×0.033=2.211→2.2`; `240×0.00305=0.732→0.7`; `540×0.00629=3.397→3.4`; `60×0.1873=11.238→11.2`; `120×0.00629+60×(2/13)×0.1873=2.484→2.5`; `40×0.04239=1.696→1.7`; `90×0.08098=7.288→7.3`; `2×4.63=9.26→9.3`; `2×11.27=22.54→22.5`; `4.63+0.21=4.84→4.8`; `62×0.62=38.44→38.4`; `3072×0.08908=273.65→273.7`; `92×0.126=11.592→11.6`; `3×3.70+3×3.49=21.57→21.6`. Share claims check: mints `85.68 %` → "85.7 %" ✓; Phase 3 `9.349 %` → "9.3 %" ✓; 2× miss `2927.6+273.7=3201.3 s = 53.4 min` ✓; 5× miss `2927.6+4×273.7=4022.4 s = 67.0 min` ✓. §6's table has **twenty** rows; **12 G** (DET1, SHA, FOLD, FLOOR, POP, C01PARITY, ROWSUBSET, NULLREMOVED, IDPARITY, ZEROMASK, LEDGER, RHORAW), **6 L** (ORBITDISP, ARENA, NESTED, SELFTEST, ZEROOP, ALGEBRA), **2 R** (DOMAIN, DEVFID) — matching §5.6's two lists exactly. |

**Ceremony floor.** All listed sha256 recompute (V1). C01 constants verified against
`configs/c01/c01_a0_v2.json` (V8, V9, V11). **Blindness intact across v1–v7**: I grepped every
decimal in `[0.6, 0.99]` across all seven drafts. v7 adds exactly **three** values absent from
v1–v6 — `0.802757` (§7.8's `max d_i`, HateMM · Head-N, a `‖Δ‖` geometry figure), `0.9568935731`
and `0.956894` (both `ρ_raw`). **None is a battery-arm accuracy**, and §7.3's claim that no arm
accuracy has been computed, printed or recorded at any point in v1–v7 holds. Test-set non-contact
by construction: no phase opens a `test_*` path, the ro `test_seen` caches are opened by nothing,
and I opened no test file in this review.

---

# PART B — DISPOSITION AUDIT OF §14's ROUND-6 BLOCK, BY EXECUTION

Method, as prescribed: I extracted every section of v6 and v7 and diffed them, then resolved each
claim against the artifact rather than against the claim. Section-level diff result — **§13 and §14
are genuinely rebuilt this round** (§13.1 new at 7,806 chars; five new §14 sub-blocks); §3.4, §5.2,
§5.2.2, §5.5, §5.6, §5.9, §6.1, §6.4, §6.5, §7.3, §7.8, §7.9, §15 and the header edited; everything
else byte-identical to v6.

| # | claim in §14 | audit |
|---|---|---|
| **C-1** §13 rebuilt at 26 items; §3.4 pins the map | **VERIFIED ADOPTED.** 26 contiguous items; all references resolve; the `common_interaction` pin is correct against `contrast_blocks:1246-1265` and I rebuilt from it bit-exactly. |
| **C-2** §14 rebuilt; both stale assertions purged | **VERIFIED ADOPTED** for the purge (V3). The rebuilt table itself carries three false rows — I-2, I-4, and H-3's third limb, below. |
| **H-1** family-invariance warrant | **VERIFIED ADOPTED.** §5.5 keeps 92 on the auditability warrant, carries the three-row counterexample table (which I reproduced exactly), records round 5's 46-family as a **partial rebuttal**, and §5.9 item 8 discloses the CLOSE-easing direction. The warrant is now true. |
| **H-2** untrained-head warrant | **VERIFIED ADOPTED.** §7.3 replaced with the two-part warrant; the securing assertion is *"no vote was taken on any ro-derived arm at any point"* — a statement about what was computed, which is the right kind of claim; §13 item 26 carries it to the code lineage. The warrant is true and my blindness grep is consistent with it. |
| **H-3** max-from-median, one cell, one lineage | **PARTIAL.** (a) **ADOPTED** — §6.5 now states the residual is a `np.max` over rows governed by the smallest `d_i`. (b) **ADOPTED** — §7.8 measures four cells spanning both lineages and both datasets and the load-bearing claim is a range `7.5×`–`22.6×`; **but the header still quotes the retired single-cell `2.384e-07` / `8.4×` as the measurement** (→ **H-1** below). (c) **NOT ADOPTED** — `tiny_ok`'s disposition is claimed at **§5.2.3, which does not exist** (→ **C-2**). |
| **I-1** §5.9 item 4 over-strong | **VERIFIED ADOPTED.** Rewritten as an inequality with the equality condition named and C01's MHC-ZH `0.8846` tie recorded. |
| **I-2** §10.2 dataset naming; §6.4 re-check | **PARTIAL — and §14 claims ADOPTED.** The §6.4 limb landed (§6.4 edited, reporting duty now scoped to runs where Head-N's arena accuracy exists). **The §10.2 limb did not: §10.2 is byte-identical to v6**, the version round 6 audited as failing this exact requirement. → **C-3**. |
| **I-3** absence exemption self-containment | **VERIFIED ADOPTED.** §5.6's exemption now reads *"excluded from the evaluation of S1–S7 and from the S5 family, and enter the S4 family only as `NOT_TESTED` with `p = 1` (§5.5)"* — verbatim the prescribed repair. |
| **I-4** S7's dispersion and `<=` operator | **NOT ADOPTED, and §14 claims ADOPTED at §5.2.2.** §5.2.2's **only** change from v6 is round-6 M-4's citation fix. The `<=` operator is frozen nowhere in §5 or §3.7; the raw-versus-head dispersion comparison (round 6's `0.6146–0.7377` / "~5 % band" measurement) appears nowhere in v7's body — its only occurrence is inside §14's own claim that it was recorded. → **C-4**. |
| **I-5** `GATE-FLOOR` discharge understated | **VERIFIED ADOPTED.** §7.8 states bit-exactness of the re-minted `K_train`, says the discharge covers 1 of 66 mints, and records that the arena arithmetic over all 36 banked quantities is separately reproducible at zero cost. |
| **M-1** §7.9 before §7.8 | **ADOPTED for the ordering, and it introduced a defect.** §7.8 is now the discharge and §7.9 the cost; every cross-reference repointed (I checked all of them). **But §7.9's content was carried unchanged and now understates this round's burn** → **H-2**. |
| **M-2** "two measured accuracies" | **VERIFIED ADOPTED** — corrected to exactly one, and my grep confirms one. |
| **M-3** `ρ_raw` `orthrot_83p8` | **ADOPTED AS WRITTEN, ON AN UNDERSPECIFIED BASIS.** `0.956894` is the float64-reduction value; the float32 reduction gives `0.956893` (v6's value). Neither is wrong and §6.1 does not say which reduction it freezes. → **I-1**. |
| **M-4** halt-only guard list location | **VERIFIED ADOPTED** — `output.decision_schema.required_halt_only_validity_guards`, confirmed in the config; seven entries; `require_no_small_displacement_dominance` correctly absent from it. |
| **M-5** Phase 4's 92 as upper bound | **VERIFIED ADOPTED** — §5.5 restates it as an upper bound with the drop path costing 46. |
| **M-6** `holm_alpha` unprefixed | **VERIFIED ADOPTED** — `statistics.holm_alpha`, `statistics.holm_metrics` in §5.2's S4 row. |

**Tally: 11 VERIFIED ADOPTED, 3 PARTIAL, 1 NOT ADOPTED, 2 adopted-with-a-new-defect.** v7's header
claim — *"all 17 round-6 findings ADOPTED, 0 rebutted"* — is **not true as stated**.

---

# FINDINGS

## CRITICAL

### C-1. §3.4's pin fixes the arm two reviewers got wrong and leaves a larger hole in the same sentence — and I measured that `GATE-C01PARITY`, the design's sole anchor, does not close it.
*Attaches to:* §3.4 (v7:198-224); §6 `GATE-C01PARITY` (v7:758); §7.6 (v7:1022); §13 items 19, 23.

§3.4 now pins the arm→formula map:

> `common[m] = l2(std[m] + ow[m])`, `displacement[m] = l2(ow[m] − std[m])`,
> `common_interaction[m] = l2(common[m] ⊙ displacement[m])`

**`std[m]` and `ow[m]` are never defined.** The only signature given is
`build_views(std_blocks, ow_blocks, angles)`, so the literal reading is that `std[m]` is the input
block. `prepare_views:1296-1304` does something else: it l2-normalises **every policy/modality block
first**, and `contrast_blocks` then operates on the normalised endpoints. The correct formula is
`common[m] = l2(l2(std[m]) + l2(ow[m]))`. As written, the pin is literally the wrong formula unless
the reader silently supplies a step the document never states — which is the same defect class the
pin was written to close.

**I measured what the omitted step costs, and where it is caught.**

*In raw space*, the ro caches are unit-norm to `1.79e-07`, so the two readings nearly agree:

| build | HateMM | MHC-ZH |
|---|---|---|
| pre-normalised (correct) | `0.000e+00` | `0.000e+00` |
| **not pre-normalised** | **`1.878e-06`** | **`1.609e-06`** |

§6's `GATE-C01PARITY` row says the builder *"reproduces `prepare_views` **bit-exactly** … **HALT
above C01's `2e-6`**"* — two different criteria in one row. **Under the operative clause the row
actually states, both wrong-reading figures pass**, with 6 % of margin on HateMM.

*In head space* — the space every decision quantity comes from — the head emits an unnormalised
1024-d MLP output (`classifier.py:146`), so the same misreading is catastrophic. On synthetic
one-block inputs with realistic norm dispersion:

| endpoint-norm spread | worst arm `max\|diff\|` between the two readings |
|---|---|
| none (natural chi dispersion) | `1.912e-02` |
| ×2 | `1.310e-01` (`common_interaction`; `common` `4.41e-02`, `displacement` `3.80e-02`, `common_displacement` `3.12e-02`) |

**And no other gate fires.** I executed the check: `endpoint_std`, `endpoint_ow` and
`endpoint_concat` are **identical to `1.49e-08` under both readings**, because `fuse` and `paired`
normalise internally — so `GATE-ARENA` (whose lower bound is `endpoint_std`-only) sees nothing. The
`θ = 0` and `θ = 45` identities **both still hold** under the wrong reading (`2.235e-08`), because
the Givens combination is a scalar multiple of the contrast blocks and `l2` kills the scale — so
`GATE-ALGEBRA` passes and `GATE-ZEROOP` compares identical keys and passes. `GATE-ORBITDISP` sees a
healthy `ρ`. `GATE-SELFTEST` and `GATE-NESTED` are identities that hold on any arm set.

So a code lineage that builds from §3.4's prose under its literal reading produces head-space arms
displaced by `10⁻²`–`10⁻¹` on **every contrast arm — including both real arms and C01's primary —
and every gate in the design passes.** The battery would render a verdict on a different battery.
§3.4's own sentence anticipates the shape of this (*"a wrong arm and a build that still looks like
it passes"*) but prices it only for `common_interaction`, whose error is loud (`9.697e-01`); this
one is quiet in the only space the anchor tests.

**Repair.** Three lines, all cheap:
1. State the missing step in §3.4: *"`std[m]` and `ow[m]` denote the **`l2_rows`-normalised**
   endpoint blocks — `prepare_views:1296-1304` normalises every policy/modality block before
   `contrast_blocks` is called, and the contrast definitions below are on the normalised blocks."*
2. Restate `GATE-C01PARITY` as **bit-exact** (`max|diff| == 0.0`, all 13 arms, both datasets) and
   **strike the `2e-6` clause from that row** — `2e-6` is `GATE-ALGEBRA`'s bar and does not belong
   to the parity anchor. §7.6 already reports `0.000e+00`, so this costs nothing and closes the gap.
3. Extend §13 item 19 (or 23) to name endpoint pre-normalisation as a checked property, and to
   state that `GATE-C01PARITY` must be asserted at bit-exactness because a `2e-6` tolerance admits a
   builder that is wrong by `10⁻¹` in head space.

### C-2. §5.2.3 does not exist. Five references point into it, two of them §14 rows claiming round-6 H-3(c) and round-5 C-3 COMPLETE by it; `tiny_ok`'s bar is registered nowhere.
*Attaches to:* §5.2.2/§5.3 boundary (v7:474-476); §7.8 (v7:1095); §13 item 25 (v7:1458); §14
(v7:1483, 1506); §15 item 4 (v7:1580).

§5 runs `5.2 → 5.2.1 → 5.2.2 → 5.3`. **There is no §5.2.3.** Five places assert its content:

* §7.8: *"So C01's `tiny_ok` limb (§5.2.3) would pass with large margin"*
* §13 item 25: *"`tiny_ok`'s non-carriage (§5.2.3) rests on measurement in every cell"*
* §14 round-6 H-3: *"and **§5.2.3 disposes of `tiny_ok`**"* — cited as the third limb of an ADOPTED
* §14 round-5 C-3: *"**COMPLETE** — §13 item 15 extended, §5.2.3 disposes of `tiny_ok` by measurement"*
* §15 item 4 asks round 7 to rule on *"`tiny_ok`'s non-carriage (§5.2.3)"*

This is not a cross-reference typo, because the substance is missing too. From
`displacement_audit:2047-2076`, C01's own final boolean for the condition S7 imports is

```
tiny_ok    = max_tiny_fraction <= transforms["max_tiny_displacement_fraction"]   # 0.05
final_bool = tiny_ok and (not require_no_small_displacement_dominance or not dominated)
```

`tiny_ok` is a **conjunct of the frozen condition**, and S7 carries only the `dominated` limb.
**`max_tiny_displacement_fraction = 0.05` appears nowhere in v7**, so the bar the non-carriage is
being excused against is unregistered; `tiny_displacement_epsilon = 0.001` appears once, in §7.8's
dry-check prose, not in §5. §5.4.1 gives a complete disposition of C01's seven
`required_halt_only_validity_guards` and correctly notes `tiny_ok` is not among them — it sits
inside `decision`, in the same check S7 names as its source, and still falls through the gap round 6
identified.

The direction is conservative — dropping a conjunct that can only make S7 **fail** makes S7 easier,
SURVIVE easier, CLOSE harder — so this is not a wrong-verdict path. It is Critical under the
standing bar (*"any claimed repair the artifact does not contain"*), and it leaves round 7 with
nothing to rule on where §15 asks for a ruling.

**Repair.** Write §5.2.3 (*"`tiny_ok` — carried or not, and why"*): name both constants
(`tiny_displacement_epsilon = 0.001`, `max_tiny_displacement_fraction = 0.05`), state that the
battery carries the `dominated` limb only, give the measured warrant (`min d_i` `0.018`–`0.038`,
`frac(d_i ≤ 1e-3) = 0.0000` in all four measured cells, and `0.0000` against a `0.05` bar in raw
space), state the direction (eases S7, hence hardens CLOSE — the conservative direction under §4),
and add it to §5.9's disclosure list. Then the five pointers resolve and §15 item 4 becomes
rulable.

### C-3. §14 claims round-6 I-2 ADOPTED at §10.2; §10.2 is byte-identical to v6, and §5.6's requirement on it survives verbatim unsatisfied.
*Attaches to:* §5.6 (v7:638); §10.2 (v7:1205-1227, unchanged from v6); §14 (v7:1485, 1504).

Round 6's I-2 repair had two limbs. The §6.4 limb landed. The §10.2 limb reads: *"Extend §10.2's
scope sentence to name the dataset(s) on which each surviving lineage passed, matching §5.6's
requirement."* **My section diff shows §10.2 is byte-identical to v6** — the version round 6 audited
as failing exactly this. §5.6 still ends:

> The lineage(s) that ran, and **the dataset(s) on which each passed its gates**, must be named in
> §10.2's scope sentence.

and §10.2's scope sentence still names both datasets unconditionally plus *"under the lineage(s)
that passed their instrument gates — named explicitly in the verdict"*, saying nothing about which
datasets each lineage passed on. The requirement and the section it binds still disagree, which is
round 6's finding word for word.

§14 asserts the opposite in two places: the I-2 row (*"**ADOPTED** — §10.2 names the dataset(s) per
surviving lineage"*) and the round-5 C-1 row (*"**COMPLETE** — §10.2 (I-2)"*). Both are false.

The consequence is bounded — under §5.6 rule 2 a CLOSE requires both lineages to have passed on
**both** datasets, so on the CLOSE path the per-lineage dataset list is degenerate — but the
requirement is stated unconditionally, it binds SURVIVE and HALT publications too, and the record
now claims a repair that was not made.

**Repair.** Either extend §10.2's scope sentence as round 6 prescribed, or — if the degeneracy on
the CLOSE path is the reason it was not extended — say so in §5.6 and narrow the requirement to the
SURVIVE/HALT faces. Then correct both §14 rows.

### C-4. §14 claims round-6 I-4 ADOPTED at §5.2.2 on two limbs; §5.2.2's only edit is an unrelated citation fix, and S7's `<=` operator is frozen nowhere in the decision rule.
*Attaches to:* §5.2.2 (v7:432-474); §3.7's constant table (v7:314-322); §13 item 5 (v7:1377-1380);
§14 (v7:1487).

§14 states: *"**I-4** S7's dispersion and `<=` operator | **ADOPTED** — the operator is frozen as
`<=`; the raw-versus-head dispersion difference is recorded, with the head-space bottom decile shown
to be a genuinely distinct population rather than a ~5 % band | §5.2.2"*.

My diff of §5.2.2 against v6 returns exactly one change — round-6 M-4's relocation of
`required_halt_only_validity_guards` under `output.decision_schema`. Neither I-4 limb is there.

* **The `<=` operator.** C01 uses `small_mask = dev_min <= threshold`
  (`displacement_audit`, the line round 6 cited as `:2049`). Because the C06 arena **is** the
  population the quantile is taken from, `<` and `<=` differ on the boundary rows and the operator
  is a decision parameter of a binding SURVIVE conjunct. It appears in v7 only in §13 items 5 and
  15 — the **code-lineage handoff**, not the decision rule. Worse, round 6's prescription was
  explicit that item 5's list should include it *"once frozen"*, and item 5 now reads *"Every
  population-derived constant in §3.7's table … S7's quantile threshold **and S7's `<=` small-set
  comparison operator**"* — **pointing at a §3.7 table row that does not exist.** A code lineage
  told to verify that every constant in a table is recomputed from the arena will not find the
  operator in that table.
* **The dispersion clause.** Round 6 prescribed one clause reporting the head-space `d_i`
  distribution so the record says what the bottom decile actually is. §7.8's table does give min,
  `q₀.₁`, median and max — that data exists. But §5.2.2 draws no conclusion from it, and the
  raw-space comparison the §14 row claims (*"rather than a ~5 % band"* — round 6's
  `0.6146–0.7377` measurement) **appears nowhere in v7**; grep returns it only inside §14's own
  claim that it was recorded.

**Repair.** Freeze `<=` in §5.2.2's parameter list and add the row to §3.7's constant table (so
item 5 resolves). Add one sentence to §5.2.2 contrasting the head-space decile (`q₀.₁` `0.044`–`0.069`
against medians `0.18`–`0.23`, i.e. a genuinely separated lower tail) with the raw-space band, citing
§7.8. Then correct the §14 row.

---

## HIGH

### H-1. v7's header, above the fold, describes v6's changes and asserts the single-cell `GATE-ALGEBRA` figure that §7.8 explicitly retires.
*Attaches to:* header (v7:26-42); §7.8 result 1 (v7:1089-1093); §6.5 (v7:903-916).

Two defects in one block, both inherited verbatim from v6's header:

**(a) Wrong version, wrong round.** The paragraph opens *"**What v6 changes.** Round 5's three
Criticals sit in the two structures v5 introduced…"* and then describes round-5's C-1/C-2/C-3. This
is v7; its changes are round 6's. The paragraph sits directly beneath *"Disposition: all 17
**round-6** findings ADOPTED"*, so a reader is told what round 6 found and then handed round 5's
repair list as this version's content. This is precisely the defect class round 6 raised to Critical
in §13/§14 — text asserting a stale frame in a document that supersedes it — surviving in the one
place every reader sees first.

**(b) A retired number, asserted as current, on a gate quantity.** The next paragraph says:

> the head-space `GATE-ALGEBRA` residual … is measured here on a **trained** head at
> **`2.384e-07`** — `8.4×` inside the frozen `2e-6` bar (§7.8).

§7.8 result 1 says the opposite about that figure's status: *"the load-bearing claim is now a range,
not a point … Per §7.5's own discipline the range is what is load-bearing; **the single trained
value v6 quoted is not**."* The binding cell is HateMM · Head-R at `2.682e-07`, **`7.5×`** — and
Head-R is the in-domain lineage, the one most likely to carry a SURVIVE. The header therefore
publishes the more comfortable of two numbers for the same gate, in the exact form round-6 H-3
ruled unsound, in a document whose body has already withdrawn it.

**Repair.** Rewrite the block as *"What v7 changes"* over round 6's findings, and restate the
residual as the four-cell range `8.848e-08`–`2.682e-07`, headroom `7.5×`–`22.6×`, worst cell
HateMM · Head-R — matching §7.8 and §6.5.

### H-2. §7.9's dry-check cost record is v6's and contradicts v7's own footer: it accounts for one mint where five were trained.
*Attaches to:* §7.9 (v7:1109-1117); the closing footer (v7:1594-1596); §7.8 (v7:1072-1085).

§7.9 reads: *"**v6's measurements** added ≈ 3 wall-minutes / ≈ 8 CPU-minutes — **one real head
mint** for §7.8's `GATE-FLOOR` discharge and trained-head residual … Cumulative **v1–v6**: ≈ 21
wall-minutes / ≈ 85 CPU-minutes."* The document's own closing footer says: *"**Five CPU head mints**
were trained on the login node for §7.8's `GATE-FLOOR` discharge **and its four-cell displacement-tail
measurement**."* §7.8's four-cell table is new in v7 and required four further mints spanning both
lineages and both datasets — at §7.2's measured units that is roughly `2.5` additional wall-minutes
and a comparable CPU charge, none of it recorded.

So the section whose job is accounting for login-node compute (a) omits this round's burn entirely,
(b) leaves the cumulative figure stale at v1–v6, and (c) attributes to v6 a measurement programme
four-fifths of which happened in v7. This is the F118 lesson's own class, mirrored: §2 binds the
design to *"never let boilerplate describe a leg that did not run"*, and here boilerplate fails to
describe a leg that did.

**Repair.** Retitle to *"v7's measurements added …"*, state the four additional mints and their
wall/CPU cost, and update the cumulative to v1–v7. One paragraph.

---

## IMPORTANT

### I-1. §6.1 freezes 26 `ρ_raw` values at 6 dp without specifying the accumulation dtype, and the sixth digit of one of them depends on it.
*Attaches to:* §6.1 (v7:801-815); `GATE-RHORAW` (v7:765); §13 item 7; §14 M-3 (v7:1491).

I recomputed all 26 values under both reductions. **Exactly one disagrees at 6 dp and none at
4 dp:**

| | float32 accumulation | float64 accumulation |
|---|---|---|
| HateMM `orthrot_83p8` | `0.9568933249` → **`0.956893`** | `0.9568935731` → **`0.956894`** |

v6 printed `0.956893`; v7 "corrects" it to `0.956894`, and §14's M-3 row calls this a
re-measurement. Both are honest measurements of the same quantity under different reduction orders
— the gap is `2.5e-07`, about 2 float32 eps — and §6.1 says nothing about which reduction it
freezes. Since `ρ` is computed on `float32` keys, the float32 reduction is the natural default and
the "correction" moved away from it.

`GATE-RHORAW` asserts reproduction at **4 dp**, and all 26 agree at 4 dp under both reductions, so
**no gate is at risk**. The defect is reproducibility of a frozen table under §13 item 7, and it
touches this campaign's numeric-provenance discipline directly.

**Repair.** One clause in §6.1: state the reduction (*"`ρ = ‖mean_i k_i‖` with the mean accumulated
in `float64` over `float32` keys"*), and note in §14's M-3 row that the earlier figure was the
float32 reduction rather than an error.

### I-2. §8 Phase 7's enumeration omits two run-time loops, and they exceed the row's stated `0.1 s` bound between them.
*Attaches to:* §8 Phase 7 (v7:1151); §6 `GATE-ALGEBRA` (v7:770); §13 item 25 (v7:1455-1458).

Six rounds have found five uncounted loops on `rule_1_compute_projection`'s axis. Two more are
unnamed:

* **`GATE-ALGEBRA`'s residual** — a `np.max(np.abs(·))` over two `(n_D, 2048)` key-difference
  matrices per `(dataset, seed, fold, lineage)` cell, i.e. `120` reductions. The guard **arms** are
  counted (Phase 2z); the **comparison** is not, and `GATE-ALGEBRA` is absent from Phase 7's list.
* **§13 item 25's per-cell tail record** — `min_i d_i` and `frac(d_i ≤ 0.001)` over 60 cells,
  newly required by this draft.

Measured on this node: `0.160 s` and `0.122 s` respectively, **`0.282 s` against Phase 7's stated
`0.1 s` upper bound**. The projection is unaffected in substance (`2927.6 → ≈2927.9 s`, `0.01 %`),
and the heartbeat is unaffected (both sit far under the `~15 s` interval). But Phase 7's count
column reads *"all"* against an explicit list, and a list that omits the two largest members of its
own class is not auditable.

**Repair.** Name `GATE-ALGEBRA`'s residual and item 25's tail record in Phase 7's list and raise the
row's bound to `0.5 s` (or price them as their own line). Re-multiply the total.

### I-3. §11 names three groups of load-bearing inputs without sha256, while `GATE-SHA` asserts that everything in §11 matches.
*Attaches to:* §11 (v7:1291-1293); `GATE-SHA` (v7:754); `GATE-FLOOR` (v7:756); `GATE-FOLD` (v7:755).

§11's digest tables cover 21 files, all of which recompute. Its closing sentence then adds, with no
digests: the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`, the ten banked
`vsw_ckpt/{hatemm,zh}/f{0..4}.npz`, and the 36 banked `c09_topo` `mint_*.npz`. The first two are
directly load-bearing — **`GATE-FLOOR`'s six anchor triples are read out of the arena OUT files**,
and **`GATE-FOLD`'s parity is asserted against the `vsw_ckpt` npz** (`headspace_mint.py:209-216`).
`GATE-SHA`'s stated scope is *"every frozen import and input cache matches §11"*, which these
cannot satisfy. §7.3's blindness argument and §5.6's global-HALT structure both rest on
`GATE-FLOOR`, so a silent change to an arena OUT file is the one provenance failure the design
cannot currently detect.

I confirmed all sixteen files exist and are the ones the code reads. Also: §11 writes the path as
`vsw_ckpt/{hatemm,zh}/f{0..4}.npz` while `headspace_mint.py:209` resolves
`scripts/analysis/vsw_ckpt/<ds>/` — and `scripts/analysis/vsw_ckpt/` also contains `en`, `st_A`,
`st_B`, `st_B0` siblings, so the unqualified path is genuinely ambiguous to an operator.

**Repair.** Add sha256 rows for the six arena OUT JSONs and the ten `vsw_ckpt` npz (the 36 C09
mints are reference-measurement inputs only and may stay digest-free if §11 says so explicitly), and
write the `vsw_ckpt` path with its `scripts/analysis/` prefix.

---

## MINOR (each non-blocking, none on the verdict path)

* **M-1.** §5.2.2's zero-fix bullet says the convention *"the battery does not inherit, because §11
  does not import `displacement_audit`"*. §11 **does** import `c01_policy_contrast_a0.py`, which
  contains `displacement_audit`; the true statement is that the battery does not **call** it. Same
  for §13 item 15's *"none of which the battery inherits"*.
* **M-2.** §3.4 attributes the *"test against `prepare_views`, do not reimplement"* instruction to
  §13 item **19**; item 19 carries the one-construction claim and item **23** carries that
  instruction. The sentence names both, so nothing is lost, but the attribution is backwards.
* **M-3.** §5.2.1 cites `select_strongest_ordinary_control:1940-1948` for the argmax rule; `:1940-1948`
  is the function's guard block and the ranking is at `:1955-1962`. The rule as quoted is exactly
  right — only the line range is short.
* **M-4.** §5.9 item 6 calls S7's reclassification *"the largest such item"* on the disclosure list
  while item 8 (added this round) calls itself *"the largest undisclosed direction change round 6
  found"*. The two superlatives are scoped differently and do not strictly collide, but one of the
  two adjectives should go.

---

# REQUIRED RULINGS

## 1. §3.D — can any gate fire on a warranted CLOSE? **NO, for all twenty.**

I re-tested each rather than inheriting round 6's answer. A *warranted CLOSE* is the state the
falsifier exists to publish: the head space is alive on its controls and the real arms fail to beat
the rotations.

**Twelve global gates** (`DET1`, `SHA`, `FOLD`, `FLOOR`, `POP`, `C01PARITY`, `ROWSUBSET`,
`NULLREMOVED`, `IDPARITY`, `ZEROMASK`, `LEDGER`, `RHORAW`) are all provenance, population, algebra
or bookkeeping predicates whose values are **independent of how the real arms score**. None can be
triggered by a real arm losing. `RHORAW` is a property of the ro caches and the raw builder — I
reproduced all 26 values, so it fires only on cache or builder drift.

**Six per-lineage gates:**
* `GATE-ARENA` — lower bound is `endpoint_std`-**only** (a control), so real-arm collapse cannot
  trip it; the `≤ 0.98` upper bound is a leak catcher and cannot fire downward. **Safe.** This is
  what `GATE-ARMVIAB`'s retirement bought, and §6.2's argument holds: §1's raw arms clear the arena
  bars by `0.15`–`0.23`, so the escape branch really was unreachable.
* `GATE-ORBITDISP` — fires only when `ρ_head > ρ*` while `ρ_raw ≤ ρ*`. **I reproduced §6.1's
  defence exactly** from the 36 banked C09 mints: HateMM `0.447803 / 0.562434 / 0.632996`, MHC-ZH
  `0.340179 / 0.574247 / 0.667326`, **0/18 above `ρ*` on both**. A trained head sits at roughly half
  the bar. **Safe.**
* `GATE-NESTED`, `GATE-SELFTEST` — identities that hold by construction on any arm set. **Safe.**
* `GATE-ZEROOP` — a disclosed, **one-directional** REPORT→HALT probability, capped at `1 %` of
  `n_D` and aggregated per `(dataset, seed, lineage)`. Cannot invert a verdict. **Safe.**
* `GATE-ALGEBRA` — the one round 6 left under-evidenced. v7 now measures four cells: `θ = 45`
  residual `8.848e-08`–`2.682e-07`, headroom `7.5×`–`22.6×`, worst cell HateMM · Head-R. **I rule
  the direction adequately evidenced and the magnitude adequately *bounded for its purpose*, with
  one qualification the design should state.** The residual scales inversely with `min d_i`;
  measured `min d_i` is `0.018`–`0.038` and `frac(d_i ≤ 1e-3) = 0.0000` in all four cells, so
  firing would need an unmeasured cell whose minimum is ~`0.0024`, an order of magnitude below any
  measured minimum and well inside the measured `q₀.₁` of `0.044`–`0.069`. **But four cells of
  sixty formally bound nothing about a min-over-`743 × 60`**, and the honest control is not the four
  measurements — it is §13 item 25's requirement that every cell record its own `min d_i` at run
  time. The design should say that. The direction is CLOSE→HALT, i.e. refusal to publish, never a
  wrong verdict. **Safe.**

**Two reporting gates** carry no bar. **Ruling: no gate can fire on a warranted CLOSE, and no gate
failure is reportable as a closure** — a per-lineage failure drops that lineage on both datasets and
§5.6 rule 2 requires **both** lineages to have passed before a CLOSE can be published; a global
failure HALTs outright.

## 2. Full verdict-path re-enumeration (mine, not inherited): **total, mutually exclusive, one lawful absence path.**

Let `G` = all twelve global gates pass; each lineage is `P` (passed every per-lineage gate on
**both** datasets) or `F` (`INSTRUMENT_FAILED`); `X` = clears S1–S7 on both datasets.

| state | rule 1 | rule 2 | rule 3 | published |
|---|---|---|---|---|
| `¬G` | — | — | fires | **HALT** |
| `G`, (P,P), either clears | fires | — | — | **SURVIVE** |
| `G`, (P,P), neither clears | no | fires | — | **CLOSE** |
| `G`, (P,F)/(F,P), passed lineage clears | fires | — | — | **SURVIVE** |
| `G`, (P,F)/(F,P), passed lineage does not clear | no | no (not both passed) | fires | **HALT** |
| `G`, (F,F) | no (no passed lineage) | no | fires | **HALT** |

Six states, each terminating in exactly one published verdict. Rules 1 and 2 are disjoint by rule
2's *"neither clears"*; rule 3 is the exact complement. **I could not manufacture a CLOSE on one
clean lineage, or on a lineage clean on only one dataset** — §5.6's dataset-axis clause forecloses
both, and the worked example (Head-N fails `GATE-ARENA` on HateMM only) publishes HALT, as stated.

**The declared-drop exemption is the only lawful absent-quantity path.** §5.6 asserts every gate and
decision quantity finite and present, HALTs on absence, and carves out exactly one class — a dropped
lineage's quantities, `INSTRUMENT_FAILED`, excluded from S1–S7 and the S5 family, entering S4 only
as `NOT_TESTED` at `p = 1` — closed by the sentence *"absence by declared drop is lawful; absence by
computation failure in a surviving lineage still HALTs."* Round-6 I-3's repair makes the exemption
self-contained, and a lineage with an absent gate quantity has not "passed", so it is dropped rather
than reaching the rule. **No second exemption exists.** `RuntimeError` from the imported C01 algebra
is a crash recorded as `INSTRUMENT_INCONCLUSIVE`, not a gate result, and is wrapped at every call.

## 3. §13's completeness for the separate code/resource review lineage: **sufficient in structure, with two additions required.**

The 26 items cover every limb the six rounds prescribed. I traced each: round 5's PART F items
19–22 present and faithful; round 6's PART E items 23–26 present and faithful (item 23's formula is
correct against `contrast_blocks:1246-1265`); the extensions to items 5, 10, 15 and 16 all landed.
The §3.4 parity-anchor handoff is explicit and in the right place — item 19 (*"`GATE-C01PARITY`
runs against that function, not a copy … the head-space arms have **no other anchor anywhere in the
design**"*) plus item 23 (*"a code lineage that reimplements from this document rather than testing
against `prepare_views` will produce a wrong arm"*). Every item is actionable without context.

**What I would add — and C-1 makes the first non-optional:**

1. **Endpoint pre-normalisation, and `GATE-C01PARITY` at bit-exactness.** Item 19 or 23 must state
   that the contrast blocks are formed from `l2_rows`-normalised endpoints and that the parity
   assertion is `max|diff| == 0.0`, **not** `≤ 2e-6` — because I measured a wrong reading that
   passes at `2e-6` in raw space and is wrong by `10⁻¹` in head space, with every other gate
   passing.
2. **The `<=` operator's actual location.** Item 5 currently sends the lineage to a §3.7 table row
   that does not exist (C-4). Once the operator is frozen in §5.2.2/§3.7, item 5 resolves.

I would also fold C-2's §5.2.3 into item 15 once it exists, so `tiny_ok`'s non-carriage reaches the
lineage as a checkable property rather than a dangling pointer.

## 4. Rulings on §15's six open issues

1. **§13/§14 rebuilt.** §13: **YES** — 26 items, contiguous, every reference resolves, every
   prescribed limb present (V2, ruling 3). §14: **PARTIALLY** — both stale assertions are genuinely
   purged (V3), but three rows of the new table assert repairs the artifact does not contain
   (C-2, C-3, C-4).
2. **The `common_interaction` pin.** **YES for that arm** — I rebuilt bit-exactly at first attempt
   without consulting the source for the map, which is the evidence asked for. **But the pin is not
   sufficient for the section's stated purpose**, because it leaves the endpoint pre-normalisation
   unstated and that omission is quieter and larger (C-1).
3. **The 92-family as a declared partial rebuttal.** **The auditability warrant justifies the
   direction, and §5.9 item 8 states it strongly enough.** A preregistered family size that a
   realised run cannot move is the right property for a design whose drop is caused by instrument
   gates no analyst controls; the alternative lets the family size be a function of the outcome. The
   cost is correctly identified (S4 no easier ⇒ SURVIVE no easier ⇒ CLOSE easier) and correctly
   disclosed. I verified §5.5's counterexample table exactly (V7). No change needed.
4. **`tiny_ok`'s non-carriage.** **I cannot rule: the section §15 points at does not exist** (C-2).
   On the substance, had it been written: four cells of sixty **would** be sufficient, because the
   direction is conservative (dropping a conjunct that can only fail makes CLOSE harder), the
   measured margin is `18×`–`38×` on `min d_i` with a `0.0000` fraction in every cell, C01's own bar
   is a slack `0.05`, and §13 item 25 converts the four-cell evidence into a per-cell run-time
   record. Write §5.2.3 and this becomes a straightforward accept.
5. **The head-space residual range.** **Four cells is adequate for the decision it supports, and
   `7.5×` is enough margin — but not because four cells bound sixty.** The four cells span both
   lineages and both datasets, which is the axis that mattered (round 6's objection was that Head-R
   was entirely unmeasured; it now is, and it is the tightest cell). The real control is item 25's
   per-cell record plus the gate's one-directional CLOSE→HALT direction. The design should say that
   four cells bound nothing formally and that the run-time record is the control (ruling 1).
6. **Is the record now sound?** **No.** One referenced section does not exist, three §14 rows claim
   repairs that were not made, the header describes the wrong version and asserts a retired number,
   and §7.9 accounts for one mint where five were trained. I say this plainly rather than waving it
   through, and equally I record that §13 — the section that most needed rebuilding — **is** sound.

## 5. Process rules

**`rule_1_compute_projection`.** §8 is byte-identical to v6 and its arithmetic is correct under
independent re-multiplication (V12, satisfying F118). §7.8's four-cell measurement implies one new
run-time loop — item 25's per-cell tail record — which §8 does not name, alongside
`GATE-ALGEBRA`'s residual comparison; measured together at `0.282 s` against Phase 7's `0.1 s`
bound (**I-2**). Immaterial to the total; the enumeration should still name them.

**`rule_2_heartbeat`.** §9 is byte-identical to v6 and **nothing in v7 changes an interval.** I
re-derived the longest un-instrumented span from §8's corrected counts: `GATE-C01PARITY` at
`11.27 s` per dataset (`14.1 s` conservative), with Phase 3 at `2.85 s` per 32-draw line, Phase 2D
at `0.62 s` per `ρ` cell, Phase 4 per bootstrap block and Phase 6 at `3.6 s` per `(dataset, seed)`.
`~15 s` holds. Line-buffered per-phase discipline, the unbuffered driver echo, the HALT path naming
its gate, and the `RuntimeError` `context` string on the final line are all specified.

**Run boundary.** Unambiguous: one SLURM CPU submission, 8 CPU / 32 GB, no `--gres`/`--time`/array/
dependency/requeue, 73 processes in the order 66 mints → 6 fidelity → 1 arena, `GATE-SHA` once in
the driver, `GATE-POP` before any population-consuming gate. The local-only justification is sound
(`GATE-FLOOR` anchors to floors measured on `foscsmlprd01`).

**Freeze-readiness.** Operator-executable as written **except** for C-1 (an operator implementing
`GATE-C01PARITY` from §6's row would use the wrong tolerance), C-2 (§5.2.3 unwritten), and I-3
(sixteen load-bearing inputs without digests). All constants otherwise pinned.

## 6. Can the falsifier discharge the written condition at `$0`?

**Yes.** Nothing I found calls the instrument into question. The two-block anchor is bit-exact and I
reproduced it independently; the row-subset bridge is bit-exact; the population constants are right;
the trained-head `ρ` defence reproduces to the digit; the Holm arithmetic is exact; the verdict
combination is total and exclusive; no gate can fire on a warranted CLOSE; and the projection is
`≈ 61 minutes` of CPU on one node. C-1 is a specification defect with a three-line repair, not a
design flaw — the correct battery is well defined and already demonstrated; what is missing is the
sentence that forces a code lineage to build it.

---

# CLOSING

**REVISE — 4C / 2H / 3I + 4M.**

The most severe finding is **C-1**. Round 6's central lesson was that §3.4's prose does not
determine the battery, and it proved it by mis-deriving `common_interaction` at a cost of
`max|diff| = 9.697e-01`. v7 pins that arm, and the pin works — I rebuilt all thirteen arms from the
prose and matched `prepare_views` bit-exactly on both datasets at the first attempt, which is the
evidence §15 asked for. But the same pinned sentence writes `common[m] = l2(std[m] + ow[m])` without
ever saying that `std[m]` and `ow[m]` are the **l2-normalised** endpoint blocks, and
`prepare_views:1296-1304` normalises them first. I built the arms both ways and measured the
consequence. In raw space the two readings differ by `1.878e-06` (HateMM) and `1.609e-06` (MHC-ZH)
— **both inside the `2e-6` HALT tolerance that §6's `GATE-C01PARITY` row states**, so the anchor
that §3.4 and §13 items 19 and 23 identify as the head-space arms' *only* protection does not fire.
In head space, where the head emits an unnormalised 1024-d vector, the same misreading moves every
contrast arm by `1.9e-02` to `1.3e-01` — both real arms and C01's primary among them — while the
endpoint arms stay bit-identical, the `θ = 0` and `θ = 45` identities still hold to `2.2e-08`, and
`GATE-ALGEBRA`, `GATE-ZEROOP`, `GATE-ARENA` and `GATE-ORBITDISP` all pass. That is a wrong verdict
published by a build that looks clean on every instrument the design owns. The repair is three
lines and costs nothing: state the pre-normalisation, assert `GATE-C01PARITY` at bit-exactness
(§7.6 already measures `0.000e+00`), and strike the `2e-6` clause from that row, where it never
belonged.

The three remaining Criticals are one defect wearing three faces: §14's rebuilt table describes
edits that were not made — to §5.2.2 (the `<=` operator and the dispersion clause, C-4), to §10.2
(the dataset naming §5.6 still demands of it, C-3), and to a §5.2.3 that does not exist at all
(`tiny_ok`, C-2). Round 6 caught v6 by diffing §13/§14 against v5; I diffed v7 against v6 and found
that the sections were genuinely rebuilt this time — §13 in particular is sound, complete and
actionable, which matters because it is what the separate code/resource review lineage will execute
against. What did not change is the habit of the disposition table running one step ahead of the
document.

*Read-only review. No GPU, SLURM, Modal, arena run, cache write, test-split access, job submission
or commit occurred, and no head was minted. `TARGET_STATE.json`, all seven drafts, all configs and
all six prior reviews were read and not modified. No arm accuracy was computed at any point. A GO on
this lineage would authorise nothing to run: the design would still require freeze with hashes, a
**separate** independent code/resource review lineage over the executable reaching its own
`0C/0H/0I`, and main-dialogue authorization.*
