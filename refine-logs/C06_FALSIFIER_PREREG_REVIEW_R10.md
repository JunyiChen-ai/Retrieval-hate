# C06 `$0` falsifier — independent design review, **ROUND 10**

*Artifact:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V10.md` (unfrozen, sha256
`f515764760638a24a940ff2bc0932be03f4ffa04910b7e9bfec682b227036a75`, 145901 bytes, 2046 lines).
*Reviewer:* fresh, independent of rounds 1–9 and of the designer.
*Compute used:* `sha256sum`; read-only numpy/torch-CPU re-derivations on banked **train-split**
caches, banked `vsw_ckpt` and C09 mint `.npz`, banked arena OUT JSON and `C01_A0_OUT.json`;
execution of the audit script **as reproduced in §14.2** against the on-disk v10 and against a
counterfactual of my own construction; an independent line-based section splitter of my own. No
mint, no arena run, no GPU, no SLURM, no Modal, no job, no cache write, no test-split open, no
commit. The draft, the configs and all repository files are unmodified. `TARGET_STATE.json` read
only. I declined the four permitted CPU mints: §7.8 is byte-identical v7→v10, rounds 7–9 have each
verified it, and this round's obligation is the v9→v10 delta plus the record. Where a head-space
quantity was needed I used the banked C09 mints, and I say so.

---

# VERDICT

## **REVISE — 0C / 0H / 3I + 4M**

**The science layer is closed and I confirm it a fourth time, independently and by measurement.**
I rebuilt all thirteen arms from §3.4's prose alone and got `max|diff| = 0.000e+00` against
`prepare_views` on **both** datasets, **at the first attempt with nothing silently supplied**, at
`n = 744` one-hot and at the arena `n = 743` / `579` all-False. The `GATE-ROWSUBSET` bridge is
`0.000e+00` on all 13 arms. Every wrong reading I constructed is caught: omitting endpoint
pre-normalisation costs `1.878e-06` (HateMM) / `1.609e-06` (MHC-ZH) — **both under the `2e-6` v7's
row would have allowed**, which is round-7 C-1's wrong-verdict path reproduced exactly;
`common_interaction = l2(std ⊙ ow)` costs `9.697e-01` / `9.558e-01`. All **37** sha256 recompute
against disk with zero mismatches. All **26** `ρ_raw` reproduce at **6 dp** under §6.1's frozen
float64 reduction, `ρ*` included. The 36 banked C09 mints give **0/18 above `ρ*` on both datasets**
at `0.447803 / 0.562434 / 0.632996` and `0.340179 / 0.574247 / 0.667326`, reproducing §6.1 to the
digit. All six `GATE-FLOOR` triples match §6 on **both** metrics, read out of the banked arena OUT
JSONs. §8's printed product column re-multiplies to **`2930.4`** exactly. The Holm counterexample
table reproduces cell for cell under C01's own `holm_adjust`, and the S5 bound is `n ≤ 12`.
**No gate can fire on a warranted CLOSE, all twenty, re-derived from the gate texts and from
measurement.**

**Both round-9 Highs are genuinely closed, and I verified the harder one by construction.** The
corrected self-exclusion **checks**: I built a counterfactual whose §14.1 is byte-identical to v9's
and ran the §14.2 script on it — it prints `UNCHANGED §14.1 (self, size not reported)`, **fails**
the two §14.1-citing rows and the one §14.1-citing limb, reports `named by a row but unchanged:
['14.1']`, and exits `1`. Against the real v10 the same script's output is **byte-identical** to
the embedded transcript (2803 bytes, sha `927e31db9321aba0…`, both). My own line-based splitter
reproduces **every** printed delta exactly, including `UNCHANGED: 49` and the true §14.1 size
(`6143 → 6660`) the transcript correctly declines to print — and confirms §14.1 genuinely changed,
so v10's `CHANGED §14.1` line is a measurement, not a convention.

**The verbatim protocol works, and it let me find the one thing it was built to expose.** All 20
limb quotations are verbatim substrings of round 9's review — I checked every one programmatically
and then by reading. **Nineteen are faithful and complete. One is truncated**: round-9 M-2's limb
drops *"with the §5.9/§15 item references excluded by their own prefixes"*, and v10 implemented a
different constraint while recording *"both adopted"*. That is exactly the failure round 9 said no
machinery could catch, and the quotation protocol is what made it visible in seconds rather than
invisible.

**Three Importants, none of which can move a verdict.** A **ninth uncounted loop** (the arena must
materialise 30 native key matrices for `GATE-FLOOR`'s votes; Phase 1f counts only the 120 ro-derived
ones — I measured the unit myself at `0.0043 s`, so ≈ `0.13 s`); the truncated M-2 limb and the
audit line whose printed scope is false of its own pattern; and §7.9's *"Cumulative v1–v9"* heading
sitting over a sum whose own terms run through v10 and contradicting the footer's *"v1–v10"*.

I have not graded on trajectory in either direction. Ten rounds is evidence of nothing. Two of the
three Importants I found by measurement, and I grade each at the severity rounds 7, 8 and 9 gave
the same defect class.

---

# PART A — THE TWELVE §3 VERIFICATIONS

| # | claim | result |
|---|---|---|
| **V1** | all 37 sha256 match disk | **PASS — 37/37 recompute, zero mismatches.** 7 imported modules + 6 read-for-definitions + 8 input caches + 16 banked = **37**, matching `U7`'s "8 caches + 13 modules/configs + 16 banked" (7+6 = 13). Verified programmatically against `data/CLIP_Embedding/…`, `scripts/analysis/…`, `configs/c01/…`. |
| **V2** | break the corrected self-exclusion | **PASS.** Counterfactual built (v10 with v9's §14.1 spliced in under the document's own splitter; splice verified byte-identical to v9's section). Output: `UNCHANGED §14.1 (self, size not reported)`; `FAIL H-2 cites §14.1 -- NOT DIFFED`; `FAIL M-2 cites §14.1 -- NOT DIFFED`; `rows verified 6 ; rows failing 2`; `FAIL H-2 *"Then say in §14.1 …"* -> §14.1 NOT DIFFED`; `limbs landed 19 ; open/failing 1`; `named by a row but unchanged: ['14.1']`; exit `1`. Every clause of §14.1's claim at v10:1787-1792 is exactly what I observed. |
| **V3** | re-run the audit; transcript byte-identical; §14.1's size never printed | **PASS on both.** My run of the extracted §14.2 script against the finished on-disk v10 is **byte-identical** to the embedded transcript — 2803 bytes each, identical sha256. §14.1's size is never printed; the line reads `(self, size not reported)` while its changed status **is** printed. §14.1 is a verified fixed point for the second consecutive version. |
| **V4** | the 20 limb quotations, complete and faithful | **19 FAITHFUL / 1 TRUNCATED / 0 NARROWED.** All 20 are verbatim substrings of round 9's text (checked under a normalisation that strips only backticks, emphasis and quote-glyph style). Limb 18 (**M-2**, v10:1743) stops at *"or widen the pattern to bare `item N`"* and drops round 9's qualifying clause. Full table in Part B. **I-2.** |
| **V5** | §7.9's headline, the dropped sentence, `≈ 4 / ≈ 21`, the sum | **PASS on the prescription; see I-3 on the sum's heading.** The headline now reads *"Five CPU head mints are attributable to the v6–v7 rounds: one trained in **v6** … four trained in **v7** …"* — round 9's own words. *"Both are now stated so the two cannot be read as contradicting"* is **gone** (zero occurrences). v7 re-derived at `≈ 4 / ≈ 21` with four mints, the discharge mint's `≈ 1 / ≈ 4` moved to v6. Sums re-derive: `7+1+4+0+0+0 = 12`; `22+4+2+1+1 = 30`; `89+21+6+3+3 = 122`; and the move is net-zero against v9's `29`/`119` (`+1`/`+3` is v10's own row), exactly as the parenthetical claims. §7.8 and §7.9 now agree everywhere. |
| **V6** | Phase 1f and the corrected Phase 7z | **PASS on every figure.** Phase 1f: `60` cells × 2 matrices, unit `0.0083 s`/cell, **timing boundary stated** (*"`np.load(...)` plus `np.asarray` on the named array, warm cache"*), carried `1.0 s` against measured `0.50 s`. I measured the same operation independently on banked mints at **`0.0043 s` per matrix** — corroborating the unit. Phase 7z `GATE-ZEROOP`: `U_tie = 2.0e-05 s`/item, `cells = 12` = `3 seeds × 2 lineages × 2 datasets` per §6.5's aggregation, `7×6 + 5×6 = 72` items, `72 × 2.0e-05 = 0.0014 s`, row reduced `0.3 → 0.1 s`. Printed product column sums to **`2930.4`** exactly; `× 1.25 = 3663.0`; `48.84 / 61.05 min`; mint share `85.60 %`; Phase 3 share `9.34 %`; `2×` miss `3204.1 s = 53.4 min`; `5×` miss `4025.2 s = 67.1 min`; freeze cost `(1.0+0.7)/2930.4 = 0.058 %`. Every explicit count re-derives: `(30×3)+(6×4)+(30×2)=174`, `4×60/9×60=240/540`, `2×60=120`, `2+60=62`, `256×3×2×2=3072`, `14×3×2×2=168`, `23×2×2=92`, `13×60=780`. No stale total survives anywhere (`2929.6`, `3662.0`, `3203.3`, `4024.4`, `2930.6`: zero occurrences). |
| **V7** | §1's two previously dashed cells | **PASS, recomputed from the confusion matrices.** ZH `common` = `0.8717948718 → 0.8718`, net `+1`; HateMM `endpoint_concat` = `0.8598130841 → 0.8598`, net `+2`. **Every** cell of §1's table recomputes from `C01_A0_OUT.json`'s stored confusion matrices, and all 16 net-fix integers match. `gain_over_strongest_control` reads `-0.009345794392523366 → −0.0093` and `-0.02564102564102566 → −0.0256`; `decision.continue = false`. §1 now contains **zero** `—` cells, and §6.2's argument is untouched by the addition (it uses `displacement` and `common_displacement` only). |
| **V8** | round-7 C-1's measurement; `GATE-C01PARITY` one predicate; **rebuild the arms yourself** | **PASS, measured.** Rebuilt from §3.4's prose alone: `0.000e+00` on both datasets, first attempt, nothing supplied, 13/13 arms, arm sets identical. Dims `4 × 7168-d` + `9 × 14336-d` in raw space ⇒ `4 × 1024-d` + `9 × 2048-d` in head space. The un-normalised misreading measures `1.878e-06` / `1.609e-06`, **both under `2e-6`** — so v7's row would indeed have passed a builder wrong by `10⁻¹` in head space. §6's row states exactly one predicate, `max|diff| == 0.0`; `2e-6` survives only as `GATE-ALGEBRA`'s bar and in the narrative. |
| **V9** | `ρ*`; all 26 `ρ_raw` at 6 dp; trained-head `0/18` | **PASS, exactly.** Under the float64-over-float32 reduction all 26 values reproduce at **6 dp**, including `orthrot_83p8` at `0.956894`. `ρ*` `0.968176` / `0.977223` (`endpoint_std`), runners-up `0.964446` / `0.969686` (`common`). Trained-head `ρ` on the 36 banked C09 mints: HateMM `0.447803 / 0.562434 / 0.632996`, ZH `0.340179 / 0.574247 / 0.667326`; **0/18 above `ρ*` on both**. Masked-zero-row shift `1.3013e-03` ✓. |
| **V10** | Holm counterexample table; `n ≤ 12` | **PASS, cell for cell, under C01's own `holm_adjust`.** `m = 92`: **24/24**, **23/24**, **0/24**. `m = 46`: 24/24 in all three rows. `displacement` disjunct **22/22** at `m = 92`. Padding `1.0` (the drop path's `NOT_TESTED, p = 1`) still gives 24/24, so the drop rule is non-rejecting as §5.5 requires. `92×1/2001 = 0.045977 ≤ 0.05`; `92×2/2001 = 0.091954 > 0.05`; `46×2/2001 = 0.045977 ≤ 0.05`. `1/257 = 0.0038911`; `12/257 = 0.046693 ≤ 0.05`, `13/257 = 0.050584 > 0.05` ⇒ **`n ≤ 12`**. |
| **V11** | §3.7's two blocks, each with the right verb | **PASS.** Population block (7 rows) re-derived on the arena from the caches: `n_D` **743 / 579**; class counts **(297, 446)** / **(180, 399)**; majorities `446/743 = 0.600269 → 0.6003` and `399/579 = 0.689119 → 0.6891`; full-population `0.5995`; bands `[0.6203, 0.98]` / `[0.7091, 0.98]`; tie caps `⌊0.01×743⌋ = 7` / `⌊0.01×579⌋ = 5`. All **computed**, none readable from a config. Config block (4 rows) verified **read** at `c01_a0_v2.json::transforms`: `<=` at `displacement_audit:2036` (source-verified), `tiny_displacement_epsilon 0.001`, `max_tiny_displacement_fraction 0.05`, `normalization_epsilon 1e-12`. Verbs correct in both directions. |
| **V12** | 20 gate rows, `12 G / 6 L / 2 R`; §13's 26 contiguous items; items 10 and 15 | **PASS.** §6 has exactly **20** rows; scope column counts **12 G / 6 L / 2 R**; the G-set and L-set match §5.6's two lists **name for name** (set-symmetric-difference empty in both directions). §13.1 defines `**(1)**…**(26)**` contiguously, no gap, no repeat. **Item 10** still carries *"computed ANALYTICALLY, not by enumerating the `g!` orderings"* (round-8 I-2) and the per-`(dataset, seed, lineage)` pooling; **item 15** still carries *"it never **calls** `displacement_audit`"* with the import/call distinction and `tiny_ok`'s non-carriage; **item 19** still carries both endpoint pre-normalisation and bit-exactness. Nothing was lost when §14's earlier blocks were compressed. |

**Ceremony floor.**

* **All 37 sha256 recompute**, measured today against §11's `2026-08-04` heading.
* **C01 constants verified against `configs/c01/c01_a0_v2.json`**: the five `transforms` values above,
  `minimum_gain_over_strongest_control 0.02`, `minimum_net_fixes {HateMM: 3, MHC_zh: 2}`,
  `gain_controls = ['endpoint_std','endpoint_ow','avg_score','endpoint_concat','common']` (the five
  §5.1 names, `avg_score` among them), `n_bootstrap 2000`, `statistics.seed 20260728`,
  `holm_alpha 0.05`, `holm_metrics ['accuracy','macro_f1']`, `n_id_hash_permutations 256`,
  `permutation_hash sha256`, `bootstrap_lower_quantile 0.05` / `upper 0.95`,
  `small_displacement_gate_reference = 'strongest_ordinary_control_by_accuracy_then_macro_f1_then_frozen_gain_controls_order'`
  (**under `transforms`** — D-1 confirmed at the source),
  `small_displacement_endpoint_concat_role = 'diagnostic_only'`, and
  `required_halt_only_validity_guards` = **7 entries under `output.decision_schema`**, not under
  `decision`, matching §5.4.1's disposition list name for name. `retrieval.fix_break_reference =
  'endpoint_std'` confirms §5.2.1's two-site account.
  Every cited source line resolves at the digit: `:2036` (`small_mask = dev_min <= threshold`),
  `:2049` (`"source_rows"`), `:2050` (`"registered_null_rows_excluded"`), `:2055-2058` (`tiny_ok`),
  `:1725`, `:2724`, `:1769`, `:1272`, `:1193-1194`, `:1202`, `classifier.py:81-82/140-141/146`,
  `mechfix_ops.py:94`, `headspace_mint.py:192-194/199/209/322-324` (`lab_dev` at `:323`).
* **Blindness grep across v1–v10**, every decimal in `[0.6, 0.99]`: 116 distinct across the corpus,
  **exactly one new in v10** — `0.8718`, which I read out of `C01_A0_OUT.json` as
  `0.8717948718` and which is a **published C01 raw dev-arena** accuracy for ZH `common`. It is not
  a battery-arm accuracy. **No arm accuracy has been computed, printed or recorded at any point in
  v1–v10** — I verified this by my own corpus grep rather than inheriting it. (§7.3's sentence
  states the scope as *v1–v9*; the claim is **true** of v10, the label is one version short — **M-1**.)
* **Test-set non-contact by construction.** `test_seen` occurs once in v10, in a negative assertion
  (*"the `test_seen` ro caches are opened by nothing"*); every other test mention is a prohibition
  or a scope exclusion. No §8 phase opens a `test_*` path. Every quantity I recomputed came from
  `train_*` caches, banked train-split mints, or banked OUT JSON.
* The four new-code artifacts (`c06_falsifier_mint.py`, `c06_falsifier_arena.py`,
  `configs/c06/c06_falsifier.json`, `c06_falsifier_cpu.sbatch`) are confirmed **absent** from the
  tree, as §11 states, so the code lineage starts from zero.
* **`GATE-IDPARITY`'s property verified directly**: on both datasets the native, `ro_L24` and
  `ro_ow_L24` caches carry identical `ids` order and identical `labels`. HateMM row 355 is
  `hate_video_95`, **label 1**, exactly zero in both modalities of both ro caches **and** the native
  cache, and is the only such row on either dataset.

---

# PART B — MY OWN LIMB-LEVEL DISPOSITION AUDIT OF ROUND 9's EIGHT FINDINGS

**Method.** I read round 9's four findings and four Minors in full, extracted each **Repair**
paragraph verbatim, and checked the corresponding v10 limb against **that text** — never against
v10's row table. I then subtracted the quoted limbs from each Repair paragraph and inspected the
residue, which is how the one truncation surfaced. Where a repair had a measurable consequence I
measured it. My independent line-based splitter agrees with §14.1's printed transcript in every
line:

| section | my delta | printed | | section | my delta | printed |
|---|---|---|---|---|---|---|
| §1 | `+28` | `+28` | | §14.1 | `6143 → 6660` | `(self, size not reported)` |
| §8 | `+1335` | `+1335` | | §14.2 | `+419` | `+419` |
| §14 | `−84` | `−84` | | §5.2.2 | `+90` | `+90` |
| §15 | `+103` | `+103` | | header | `+943` | `+943` |
| §7.9 | `+822` | `+822` | | UNCHANGED | 49 | 49 |
| | | | | added / removed | none | — |

I also recomputed step (6)'s two sets independently: *changed-but-uncited* = `{15, header}` and
*named-but-unchanged* = `∅`, both agreeing with the transcript.

## Result: 19 FAITHFUL / 1 TRUNCATED / 0 NARROWED / 0 unlanded

| # | finding | limb, checked against round 9's Repair text | verdict | evidence |
|---|---|---|---|---|
| 1 | **H-1** | *"Rewrite the headline as 'Five CPU head mints are attributable to the v6–v7 rounds: …'"* | **FAITHFUL** | v10:1235-1237 carries round 9's sentence word for word, with v9's *"spanning both lineages and both datasets"* tail retained |
| 2 | **H-1** | *"and drop 'Both are now stated so the two cannot be read as contradicting'"* | **FAITHFUL** | zero occurrences in v10; the parenthetical ends on round 9's own *"there is nothing left to reconcile"* |
| 3 | **H-1** | *"Then check the ≈ 5 wall-minutes / ≈ 25 CPU-minutes attributed to v7 still holds with four mints rather than five, or say what else is inside it"* | **FAITHFUL — and it did both** | v10:1243-1249 re-derives `≈ 4 / ≈ 21` **and** enumerates what else is inside it. Round 9's justification clause *"— with the headline corrected there is nothing left to reconcile"* is an explanation, not an action, and is correctly not a limb |
| 4 | **H-2** | *"Delete the `changed.add(SELF)` fallback"* | **FAITHFUL** | absent from v10:1893-2011 |
| 5 | **H-2** | *"and print §14.1's status honestly without its size"* | **FAITHFUL** | round 9's three-line block appears **byte-identical** at v10:1922-1924, plus two lines (`if st=='CHANGED': changed.add(SELF)` / `elif st=='ADDED': added.add(SELF)`). Those two are **necessary**: round 9's literal three lines would have left §14.1 permanently outside `touched`, so every §14.1-citing row would fail *always*, not *exactly when unchanged*. The completion is printed in full and stated in prose at v10:1786. This is the designer reasoning through a prescription rather than pasting it, and it is disclosed — not a deviation |
| 6 | **H-2** | *"so a row citing §14.1 fails exactly when §14.1 did not change, and the printed line is a measurement rather than a convention"* | **FAITHFUL — verified by construction** | my counterfactual: `UNCHANGED`, 2 rows and 1 limb fail, exit 1 |
| 7 | **H-2** | *"Then say in §14.1 that self-exclusion covers the size only, never the changed/unchanged fact"* | **FAITHFUL** | v10:1780-1783, in those words |
| 8 | **I-1** | *"One Phase 1f row: 60 arena-side cell loads (or 66, if the full-fold mints are also read)"* | **FAITHFUL** | v10:1294; `60` chosen, the `66` alternative correctly declined (the 6 full mints are read by the fidelity processes, priced inside `U9`) |
| 9 | **I-1** | *"unit measured on one banked `.npz` with the key arrays materialised"* | **FAITHFUL** | `0.0041 s`/matrix; I measured `0.0043 s` independently on the same objects |
| 10 | **I-1** | *"state whether the timed region includes `np.load` alone or `np.asarray` on the arrays too …"* | **FAITHFUL** | *"timed region = `np.load(...)` **plus `np.asarray` on the named array**, warm cache"* |
| 11 | **I-1** | *"and re-multiply"* | **FAITHFUL** | `2930.4` / `3663.0`, which I re-summed from the printed column |
| 12 | **I-2** | *"Measure `U_tie` on one synthetic near-tie group of realistic size (a few milliseconds of CPU, no mint)"* | **FAITHFUL** | `2.0e-05 s`/item, group `g = 5`, no mint |
| 13 | **I-2** | *"price the row as `≤ cap × cells × U_tie` with the cell count taken from §6.5's aggregation (6 per dataset, 12 total)"* | **FAITHFUL** | `cells = 12`, `72` items, `0.0014 s`. Round 9's conditional alternative (*"if the designer prefers to keep the fold-level 30…"*) has a false antecedent and correctly is not a limb |
| 14 | **I-2** | *"state the timing boundary"* | **FAITHFUL** | *"timed region = the vote recomputation alone"* |
| 15 | **I-2** | *"and re-multiply"* | **FAITHFUL** | row **reduced** `0.3 → 0.1 s`; total re-derived |
| 16 | **M-1** | *"`:2049` → `:2050`, or drop the description and keep the line number"* | **FAITHFUL** | v10:496-497; I verified `:2049` is `"source_rows"` and `:2050` is `"registered_null_rows_excluded"` at the source |
| 17 | **M-2** | *"name the scope on that line too"* | **FAITHFUL in text** | the line now names a scope. Round 9's parenthetical model wording was followed. But the scope named is **false of the implemented pattern** — see I-2 |
| 18 | **M-2** | *"or widen the pattern to bare `item N`"* — annotated *"both adopted"* | **TRUNCATED** | round 9 wrote *"or widen the pattern to bare `item N` **with the §5.9/§15 item references excluded by their own prefixes**"*. The quotation stops at `item N`. v10 implemented a different exclusion (a six-verb whitelist), and the annotation *"both adopted"* over-states it. **I-2** |
| 19 | **M-3** | *"add the two cells, or one clause saying what the dash means"* | **FAITHFUL** | both cells added and independently verified from `C01_A0_OUT.json`; zero `—` remain in §1 |
| 20 | **M-4** | *"quote each prescribed limb verbatim from the previous round's review, with its location in that review, so the enumeration is checkable against a source rather than trusted"* | **FAITHFUL** | 20/20 are verbatim substrings of round 9. The *"location"* supplied is the finding tag rather than a line reference; I checked all twenty using only the tag with no ambiguity, so the prescription's stated purpose is met — **M-4** records the sharper form |

---

# FINDINGS

## CRITICAL

**None.** No finding this round can publish a wrong verdict or block execution on the verdict path.
I state this rather than imply it by omission: I looked in the three places the last four rounds
found theirs — inside this round's own repairs (§7.9, §8, §14.1/§14.2), in §13, and in the gate
set — and the three Importants below are the strongest things I could construct. I also attempted
to manufacture a CLOSE anywhere in the combination space and could not.

## HIGH

**None.** Nothing I found weakens the verdict's authority or scope, and no repair landed narrower
than prescribed in a way that touches a decision quantity. The one truncated quotation (I-2) is in
a drafting-audit line whose measured effect on v10 is nil.

---

## IMPORTANT

### I-1. The ninth uncounted loop: the arena must materialise **30 native key matrices** for `GATE-FLOOR`'s 30 votes, and Phase 1f counts only the 120 ro-derived ones. §13 item 22 pins the placement for the `h_std`/`h_ow` stream only, so Phase 1f's count cannot be checked against any stated placement.

*Attaches to:* §8 Phase 1b (v10:1290), Phase 1f (v10:1294), Phase 2's `GATE-FLOOR` row (v10:1296);
§13.1 item 22 (v10:1664-1666); §15 item 4.

§15 item 4 asks round 10 to hunt. Here it is, and it is the same kind as the eighth at roughly a
third the size.

Phase 1b prices the **forwards**, and its own factorisation names three streams for Head-N fold
mints: *"`30` Head-N fold mints × {**native**, `ro_std`, `ro_ow`}"*. So each of the 30 Head-N fold
mints produces **three** key matrices and writes them to its `.npz` — this is the pattern
`headspace_mint.py:322-324` already implements (`np.savez(..., K_train=K_train, ...)`), and I
confirmed a banked mint carries `K_train (744, 1024)` in a `6.1 MB` file.

Phase 1f then prices the arena's **loads** as *"`60` cells × 2 matrices (`h_std`, `h_ow`)"* = 120
materialisations, and explicitly rules out the neighbours: *"Phase 1c prices the arena's ro cache
load, Phase 1e a metadata-only read, Phase 2b a build on in-memory arrays; none prices
`np.load(...)['K_*']`."*

But §8 places `GATE-FLOOR`'s **30 native votes** in **Phase 2**, the arena's vote phase, at
`30 × U2a` — and `U2a` is defined in §7.7 as *"vote, 1024-d / 2048-d, per fold-cell"*, a vote
timing with no load inside it. To vote on native keys the arena must materialise the native key
matrix from each of the 30 Head-N fold `.npz`. **That stream is on no list.**

**Measured, not inferred.** On the banked C09 mints (`744 × 1024` in `6.1 MB` files) I measured
`np.load` + `np.asarray` on the named key array at **`0.0043 s` per matrix** warm, against
`0.0002 s` metadata-only — corroborating Phase 1f's own `0.0041 s`. The missing stream is therefore
`30 × 0.0043 ≈ **0.13 s**` warm, or `≈ 0.26 s` at the cold-cache convention Phase 1f itself adopts.

**The second half is the part a code lineage needs.** §13 item 22 says only *"The **ro-cache**
forwards producing `h_std`/`h_ow` happen inside the mint process"*. It says nothing about the native
forward and nothing about where `GATE-FLOOR`'s vote is computed. Under one reading the arena votes
and the 30 loads are real; under the other the mint votes and Phase 2's `30 × U2a` is itself
misplaced (conservatively). **Either way Phase 1f's `60 × 2` rests on a placement the document does
not state**, which is the checkability §8's discipline exists to provide.

**On severity.** The brief's Critical column admits *"any un-counted loop in §8"*, and I considered
it. Rounds 7, 8 and 9 found the fifth through eighth under the same language and graded each
Important, reasoning that a sub-second loop against a projection carrying a `× 1.25` margin and
`30 s` of declared slack cannot misrepresent anything material. That reasoning holds here with more
force — this loop is the smallest of the nine. Grading it differently would be grading on
trajectory, which the brief forbids in both directions.

**Repair.** Extend Phase 1f's count to `60 × 2 + 30 × 1 = 150` materialisations (or add a Phase 1g
row for the native stream), re-multiply — `150 × 0.0083/2 ≈ 0.62 s` measured, `1.3 s` at the
cold-cache bound, so the total moves `2930.4 → ≈ 2930.7`, still `3663` at one decimal after
`× 1.25`. **And extend §13 item 22 by one clause** naming where the native forward happens and where
`GATE-FLOOR`'s vote is computed, so the count has a stated placement to be checked against.

### I-2. Round-9 M-2's second limb is **truncated** — round 9's qualifying clause *"with the §5.9/§15 item references excluded by their own prefixes"* is dropped — and the repair landed with a different exclusion mechanism. The scope the audit line now prints is **false of the pattern it describes**, and the annotation *"both adopted"* over-states what landed.

*Attaches to:* the limb table (v10:1743); §14's M-2 row (v10:1713); the transcript's `§13 item`
line (v10:1843); the pattern (v10:1970-1977); §14.1's parenthetical (v10:1881-1885).

Round 9 wrote, in full: *"Repair: name the scope on that line too (…), **or** widen the pattern to
bare `item N` **with the §5.9/§15 item references excluded by their own prefixes**."* v10's limb
reads *"or widen the pattern to bare `item N`"* — faithful to the clause, a silent deletion of its
qualifier. This is the identical mechanism round 9 demonstrated on round-8 I-2, and the verbatim
protocol is what made it visible: I found it by subtracting the quotations from round 9's Repair
paragraph, in seconds.

**Three consequences, all measured.**

1. **The exclusion mechanism is not the prescribed one.** v10's third pattern arm is
   `(?<![§\w.])item (\d+) (?:requires|carries|now|must|is|reads)`. The lookbehind is a
   prefix exclusion, but it inspects only the character immediately before `item` — for
   `§5.9 item 4` that character is a **space**, so the lookbehind does **not** fire. What actually
   keeps §5.9/§15 references out is the six-verb whitelist. I confirmed by exhaustive scan that
   **today no §5.9 or §15 item reference is swept in** — all eleven of them (`§5.9 item 1/4/5/6/8/9`,
   `§15 item 4/5`, and three bare `item 6` references) are excluded. So the printed list
   `[5, 7, 8, 19, 23, 25, 26]` is correct and `unresolved: NONE` is true. But the mechanism is
   incidental: a future sentence such as *"§5.9 item 6 is disclosed"* would match the whitelist and
   silently inflate the list, which is precisely what round 9's qualifier was for.

2. **The printed scope is false of the pattern.** The line reads
   `(scope: '§13 item N' or bare '**item N**')`, but the third arm matches bare `item N` that is
   **not** bold — v10:1634's *"as item 5 now states"* is reached by it and is neither form. So the
   one line round-9 M-2 asked to have its scope named now names a scope narrower than what it scans.

3. **Round 9's underlying condition is not closed.** Round 9's M-2 was that *"one more genuine
   reference the pattern cannot reach"* exists. My exhaustive scan finds v10 still contains three:
   `` `item 10` `` at v10:1713 (in §14's own M-2 row), `items 15 and 19` at v10:1752, and
   `items 5, 10, 15 and 16` at v10:1569 — all genuine §13-item references, none reachable. §14.1's
   parenthetical (v10:1881-1885) explains the list shrank *"because v9's round-5 and round-8 blocks
   were compressed"*, which is true but reads as though the unreachable references went away. They
   did not; three remain. Nothing is wrong as a result — I verified against §13 directly that items
   10 and 15 keep their round-7/round-8 repairs (V12) — but the parenthetical's account is
   incomplete.

**On severity.** The brief pre-commits that *"a truncated quotation or a dropped clause is the
finding"*, and this is the round's central obligation. I grade Important rather than High because
the object is a drafting-audit line, not a verdict quantity: the printed output is correct today,
`unresolved: NONE` is true, and round 9 graded the parent Minor and non-blocking. I grade it above
Minor because round 9's repair was disjunctive and **neither** disjunct landed as written — the
first with a scope statement that misdescribes its own pattern, the second with a substituted
mechanism — while v10 records *"both adopted"*.

**Repair, three lines.** Restore the full quotation in the limb table. Either implement round 9's
prescribed exclusion (require the reference to be preceded by `§13 ` or by nothing that is a section
number — e.g. add `(?<!§5\.9 )(?<!§15 )` or drop the verb whitelist in favour of a
`§13`-anchored form), or state in the annotation that a different constraint was chosen and why.
Then correct the printed scope to describe all three arms, and soften the §14.1 parenthetical to say
that three unreachable references remain and are verified against §13 by reading.

### I-3. §7.9's sum is headed *"Cumulative **v1–v9**"* while the sum's own terms run through **v10**, and the footer states the same quantity as *"v1–v10"*. §7.9 contradicts itself and the footer, in the section round-9 H-1 was about.

*Attaches to:* §7.9 (v10:1264-1268, the heading and the sum); the footer (v10:2044).

v10:1264 reads:

> **Cumulative v1–v9, shown as a sum so it is checkable rather than asserted** (round-8 I-1):

and the sum immediately beneath it is:

> mints `= 7 (…v1–v6) + 1 (v6's GATE-FLOOR discharge) + 4 (v7's tail cells) + 0 (v8) + 0 (v9) +
> **0 (v10)** = ` **12**; time `≈ 22 + 4 + 2 + 1 + **1 (v10)** = ` **≈ 30 wall-minutes**,
> `≈ 89 + 21 + 6 + 3 + **3** = ` **≈ 122 CPU-minutes**

while v10:2044 reads *"The **twelve** CPU head mints across **v1–v10** (§7.9's sum)"*.

**The arithmetic is right and I re-derived every term of it** — `7+1+4 = 12`, `22+4+2+1+1 = 30`,
`89+21+6+3+3 = 122`, and the move of the discharge mint is net-zero against v9's `29`/`119`, with
the `+1`/`+3` being v10's own row, exactly as the parenthetical claims. Nothing is mis-stated about
any quantity. What is wrong is the label: a heading that excludes the version whose terms it
contains, disagreeing with the sum beneath it and with the footer.

**On severity.** Round 8 graded a structurally identical under-scoped label (its M-3, *"the scope
should read v1–v8"*) as **Minor**, and I keep that calibration for the §7.3 instance below (**M-1**).
This one is different in two respects that I weighed rather than assumed: the label is contradicted
by the very sum it heads and by the document's own footer, so the document disagrees with itself
about one quantity's scope; and §7.9 is the section `rule_1` and the F118 erratum lesson bind by
name and where round-9 H-1 found the same *shape* of defect — a heading at odds with the sum beneath
it — one version ago. It is not High: no reader is misled about a number, the sum is checkable and
checks, and nothing downstream moves.

**Repair, one word.** *"Cumulative v1–v10"*.

---

## MINOR (each non-blocking; none touches the verdict path)

* **M-1.** §7.3 (v10:1112-1113) states *"No arm accuracy has been computed, printed or recorded at
  any point in **v1–v9**."* **I verified the claim is true of v10** by my own corpus grep — the one
  new decimal in `[0.6, 0.99]` is `0.8718`, a published C01 dev-arena accuracy I read out of
  `C01_A0_OUT.json`. So this is a true claim one version short of its own scope, the same class
  round 8 graded Minor as its M-3. Non-blocking: the assertion is correct and independently
  verified. Repair: `v1–v9` → `v1–v10`. (v10:2046's *"v1–v9 are unmodified"* is **correct** as
  written and must not be changed — v10 is the live draft.)
* **M-2.** Stale round/version labels inside §14.1's transcript and §14.2's script, in a section
  whose whole purpose is honest mechanical verification: the transcript prints
  `=== (5) LIMB-LEVEL DISPOSITION (**round-8** prescription) ===` (v10:1845, from v10:1980) over a
  table headed *"Round 9 — LIMB level"*, and the script docstring reads *"C06 falsifier **v9**"* and
  *"(1) section diff **v8->v9**"* (v10:1895-1896) while `V_OLD`/`V_NEW` point at v9/v10 and the
  printed header says `v9 -> v10`. Non-blocking: labels only; the transcript I reproduced is
  byte-identical and every check it performs is the right one. Repair: `round-8` → `round-9`, `v9` →
  `v10`, `v8->v9` → `v9->v10`. Changing the printed header will move the transcript, so re-run and
  re-embed — the fixed-point discipline already covers this.
* **M-3.** §9's progress file is `` `$BASE/progress/C06_PROGRESS.txt` `` (v10:1348) and `$BASE` is
  defined nowhere in the document — it is the only shell variable in a document whose every other
  path is repo-relative or absolute. Non-blocking: the sentence names who creates it (*"the sbatch
  driver"*) and when (*"before the first python process starts"*), and the driver is the separate
  code lineage's object under R4. Repair: name the base directory, or write the path repo-relative
  as every other path in the document is.
* **M-4.** The limb table supplies each quotation's provenance as the **finding tag** (`H-1`, `I-2`,
  …) where round-9 M-4 asked for *"its location in that review"*, and the header (v10:12-14, 1700)
  says *"with its location there"* while §14's own explanatory sentence (v10:1719) says *"with the
  finding it came from"*. Non-blocking, and I record it having tested it: I checked all twenty
  quotations against round 9 using only the tag, with no ambiguity, because each finding has exactly
  one **Repair** paragraph — so M-4's stated purpose is met. Repair: either cite round 9's line
  numbers (its own reviews use the `v9:1222-1224` form), or make the header say *"with the finding
  it came from"* so the two sentences agree.

---

# REQUIRED RULINGS

## 1. §4.B — is the corrected self-exclusion sound, and is *size only* the right line? **Yes to both, and I broke it to check.**

**Sound, demonstrated.** The two lines round 9 asked to be deleted are gone, and the replacement
computes rather than asserts. My counterfactual — v10 with v9's §14.1 spliced in, so §14.1 is
byte-identical to its predecessor's — makes the script print `UNCHANGED §14.1 (self, size not
reported)`, **fail** `H-2` and `M-2` (the two §14.1-citing rows) and the one §14.1-citing limb,
report `named by a row but unchanged: ['14.1']`, and exit `1`. Under v9's script round 9 showed the
same construction printed `CHANGED` and passed everything. The branch that certified a repair claim
without checking it is closed, and it is closed in the direction that fails.

**Size-only is the right line, and it is the only line available.** Pasting the transcript changes
the length of the section being measured, so a printed byte count for §14.1 is a fixed-point
impossibility — that is round-8 C-1's original defect. The changed/unchanged **fact**, by contrast,
is a fixed point: appending a transcript to §14.1 leaves §14.1 changed, and the second execution
against the document with the transcript already embedded confirms it (I verified the two runs are
byte-identical). So the convention exempts exactly the quantity that cannot be self-measured and
nothing else. That is the minimal exemption, and it is now non-load-bearing: nothing else in the
audit depends on it.

**Is any other convention load-bearing for another row's verification?** I looked and found one
class, and it is disclosed rather than hidden. The `qualified()` filter (v10:1961-1963) suppresses a
`§`-reference when the preceding 12 characters contain `round-`/`round ` or the preceding 45 contain
`.md`. I audited every suppression: exactly **three** dotted references are hidden — `§4.4`
(`GATE0_REOPEN_2026-07-31.md`), `§15.4` and `§15.6` (round 5's review) — and all three are
references to **other documents**, which is what the filter is for. It hides **no** in-document
unresolved reference, so `unresolved: NONE` is a real result. The remaining convention is the
`cited_secs`/`rows_of` pair, which reads the **last** table cell; I confirmed the limb scan reads
only the *landed in* column and the row scan excludes the `LIMB-TABLE` region, so the two counts
(**8 rows**, **20 limbs**) are genuinely disjoint as v10:1879 claims.

**One thing the convention still cannot do, and v10 says so.** §14.1:1808-1812 states plainly that
the audit verifies limbs against *this document's transcription* and that faithfulness remains a
reading obligation. That is correct, and I-2 is the proof: the machinery rated limb 18 `OK` and the
truncation is visible only against round 9's sentence.

## 2. §4.D — can any gate fire on a warranted CLOSE? **No, for all twenty. Re-derived from the gate texts and from measurement, not inherited.**

A *warranted CLOSE* is: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction, and I measured seven of them.**
`GATE-DET1` (thread env), `GATE-SHA` (37 digests over files no phase writes — all 37 recomputed
today), `GATE-FOLD` (banked parity flags + `fold_of` from 66 `.npz`), `GATE-POP` (populations, class
counts, index-set identity, recomputed constants — I re-derived `743/579`, `(297,446)/(180,399)`,
`0.6003/0.6891/0.5995`), `GATE-NULLREMOVED` / `GATE-ZEROMASK` (exact-zero row sets — I measured
`{355}` / `{}`), `GATE-IDPARITY` (ids order and labels — verified identical across native/ro/ro_ow
on both datasets), and `GATE-LEDGER` (process and path counts) touch no arm score.
`GATE-FLOOR` votes **native** deployed keys against banked anchors — I read all six triples out of
the arena OUT JSONs and they match §6 to the digit on **both** metrics (`acc_deployed`
`0.8884/0.8858/0.8858`, `0.8929/0.8895/0.8946`; `mF1_deployed` `0.8838/0.8811/0.8812`,
`0.8747/0.8710/0.8765`) — and no battery arm enters it. `GATE-C01PARITY`, `GATE-ROWSUBSET` and
`GATE-RHORAW` are properties of the **raw two-block build** and the frozen `ρ_raw` table, fixed
before any head exists; I measured all three at `0.000e+00`, `0.000e+00` and 26/26 at 6 dp.

**Bit-exact `GATE-C01PARITY` and the false-HALT question.** The comparison is between the battery's
builder and `prepare_views` **in the same process, over the same arrays, through the same
`l2_rows`**, so both sides execute identical operations in identical order and agree bitwise by
construction. Mine is the eighth independent reconstruction to measure `0.000e+00`, and it holds on
the `n = 743` bridge too. A failure is a builder defect ⇒ HALT, never a CLOSE.

**The six per-lineage gates.** `GATE-ARENA`'s lower bound is on `endpoint_std` **only** — a control
— so real arms losing cannot entail it; its `≤ 0.98` upper bound catches leaks and cannot fire
downward, and a warranted CLOSE puts the real arms *low*, not high. `GATE-ORBITDISP` fires on
`ρ_head > ρ* ∧ ρ_raw ≤ ρ*`; trained heads measure `0.34`–`0.67` against bars of `0.968`/`0.977` —
**0/18 on both datasets**, my own measurement reproducing §6.1 to the digit, i.e. roughly half the
bar — and nothing about a real arm losing raises `ρ_head`. `GATE-NESTED` and `GATE-SELFTEST` are
identities that hold for any arm set: `net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is arithmetic
between two quantities the battery itself computes, not performance. `GATE-ALGEBRA` bounds the
θ=0/θ=45 identity residuals — I measured the raw-key values at `8.941e-08` (θ=0, both datasets) and
`1.192e-07`/`8.941e-08` (θ=45), and §7.8's trained-head measurements sit `7.5×`–`22.6×` inside the
`2e-6` bar. **`GATE-DOMAIN` and `GATE-DEVFID` carry no bar.**

**`GATE-ZEROOP` is the one gate that can HALT a correct run, and that is disclosed rather than
denied.** §6.5 states it: the θ=45 identity is not exact, so a key perturbation can reorder a top-20
neighbourhood. But its firing is a **numerical-tie event uncorrelated with the arm outcome** — it
compares two *guard* arms against their algebraic counterparts, never a real arm against a control —
so it cannot fire *because* the CLOSE is warranted. It is one-directional (REPORT → HALT only,
§5.9 item 5), capped at `⌊0.01 n_D⌋` = `7`/`5`, and a HALT is a non-publication, not a wrong verdict.

**`GATE-ARMVIAB`'s retirement, verified at the source.** C01's raw `displacement` `0.8505`/`0.8846`
and `common_displacement` `0.8598`/`0.8590` against arena bars `0.6203`/`0.7091` clear by
`0.1499`–`0.2395` (*"0.15–0.23"* ✓), and `GATE-FLOOR`'s native OOF `0.8884`/`0.8929` runs above
C01's dev-arena `0.8411`/`0.8590`. The escape branch was unreachable and the gate would have fired
on the warranted CLOSE. Retirement, not restriction, is right.

## 3. Verdict-path enumeration — mine, not inherited: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.**

Let `G` = all twelve globals pass; `P_N, P_R ∈ {passed, dropped}` after the per-lineage gates, where
*passed* means every per-lineage gate cleared on **both** datasets (a failure on **any** dataset
drops the lineage on **both**, §5.6); and for each passed lineage `C_L ∈ {clears S1–S7 on both
datasets, does not}`, clearing being the disjunction over `A ∈ {displacement, common_displacement}`.

* **¬G** → §5.6's global bullet is categorical (*"Any failure HALTs the whole battery"*) and rule 3
  names it → **HALT**. (1 state.)
* **G, both passed**: some lineage clears → rule 1 **SURVIVE** (3 combinations); neither clears →
  rule 2 **CLOSE** (1). (4 states.)
* **G, exactly one passed**: it clears → rule 1 **SURVIVE**; it does not → rule 2's antecedent
  (*"both lineages passed"*) is false → rule 3 **HALT**. (4 states across the two symmetric cases.)
* **G, both dropped**: rule 1 has no passed lineage, rule 2 false → rule 3 **HALT**. (1 state.)

Ten outcome classes, each mapped to exactly one published state. **Totality** holds by construction
because rule 3 is the catch-all *"otherwise"*. **Exclusivity** holds because rule 2 requires the
negation of rule 1's antecedent under *"both passed"*. The **dataset axis adds nothing**: the drop
rule collapses per-dataset gate outcomes into a single lineage pass/fail before combination, and
S1–S7 are required on **both** datasets conjunctively, so any pattern of per-dataset S-failures maps
to ¬clears. The **S1–S7 axis adds nothing**: every failure pattern maps to ¬clears; note S5 and S7
are conjuncts that do not vary with the choice of `A` (S5 requires *both* real arms; S7's arm scope
is `common_displacement` only), which makes SURVIVE strictly harder — the conservative direction —
and both are stated as such.

**The declared-drop exemption is the only lawful path to an absent quantity**, and it is scoped
correctly: a dropped lineage's quantities are `INSTRUMENT_FAILED`, excluded from S1–S7 and the S5
family, and enter the S4 family **only** as `NOT_TESTED` with `p = 1` — which I verified is
non-rejecting by executing C01's `holm_adjust` with `1.0` padding (adjusted `p = 1 > α` at every
rank, and the witness still rejects 24/24). *"Absence by computation failure in a surviving lineage
still HALTs."* **No gate failure is reportable as a closure**: rule 2 requires all globals passed
**and** both lineages passed, and every gate failure routes to rule 3 or drops a lineage.

I also re-derived the resolution floor the drop rule depends on: `92 × 1/2001 = 0.045977 ≤ 0.05`
and `92 × 2/2001 = 0.091954 > 0.05`, so every witness comparator must sit at `p = 1/2001`. That
makes S4's separate *"bootstrap lower bound `> 0`"* leg non-binding rather than ambiguous — zero
adverse resamples forces `Δ_b > 0` for all `b`, hence a strictly positive `5 %` quantile. There is
no `>` versus `≥` seam.

## 4. Rulings on §15's six open issues

1. **Narrowing, not absence — did the verbatim protocol work?** **Yes, and I can date the
   improvement to the minute.** All 20 quotations are verbatim substrings of round 9; I verified
   that programmatically and then read every Repair paragraph and subtracted the quotations from it.
   That subtraction is what surfaced the single truncation (I-2), and it took seconds — where round
   9 had to reconstruct round 8's intent from a paraphrase. **Adopt the protocol permanently.** Two
   refinements: quote the *whole* prescription including qualifying clauses (I-2), and record the
   location at line granularity rather than finding granularity (M-4).
2. **The corrected self-exclusion.** Ruled in §4.B above: **sound, broken by construction,
   size-only is the right and minimal line, and no other convention is load-bearing for another
   row's verification.**
3. **Phase 1f and the corrected Phase 7z.** Both re-derived (V6), both with timing boundaries
   stated. **The cold-cache convention is right and consistent**, and I checked it against every
   other carried row rather than assuming: Phase 1e carries `0.1 s` against a measured `66 × 0.0003
   = 0.02 s`; Phase 7z's algebra row carries `1.0 s` against a `0.128`–`1.0 s` spread; the tail
   record carries `0.7 s` above its `0.619 s` maximum; Phase 7 carries its `sub-0.1 s` class at the
   `0.1 s` upper bound. Carrying `1.0 s` against a warm-measured `0.50 s` is the same discipline,
   it is **disclosed in the row itself**, and it lands on round 9's own independent estimate of
   `≈ 1.0 s`. The only row carried *below* its measurement is Phase 1d (`U7 = 0.13 s` at `0.1 s`),
   which is one-decimal rounding and which rounds 8 and 9 both ruled acceptable. No inconsistency.
   The Phase 7z reduction leaves §6.5 and §8 consistent: `cells = 12` is exactly §6.5's
   `(dataset, seed, lineage)` aggregation, and the row is now the only §8 row whose count *derives*
   from a gate's own stated rule.
4. **The ninth uncounted loop.** **Found — I-1 above**: the arena's materialisation of the 30 native
   key matrices `GATE-FLOOR`'s votes consume, priced in no phase, measured by me at `0.0043 s` per
   matrix ⇒ `≈ 0.13 s` warm. It carries a second half worth more than the seconds: §13 item 22 pins
   the placement for the `h_std`/`h_ow` stream only, so the count has no stated placement to be
   checked against.
5. **§1's table after M-3.** **Confirmed clean.** Zero `—` cells remain; both added values recompute
   from `C01_A0_OUT.json`'s confusion matrices; all sixteen net-fix integers match. §1's
   *"load-bearing twice over"* role is **strengthened**, not disturbed: the full table now shows
   that C01's selected strongest ordinary control differs by dataset (`common` `0.8692` on HateMM,
   `endpoint_concat` `0.8846` on ZH), which is exactly D-1's point and was previously inferable only
   from §5.2.1. §6.2's argument is untouched — it rests on `displacement` and `common_displacement`
   against the arena bars, neither of which moved.
6. **Is the record now sound?** **Substantially, and for the first time the residue is not about
   whether a repair landed.** All eight round-9 findings landed; nineteen of twenty limbs are
   faithful and complete; the one truncation is in a drafting-audit line with a measured effect of
   zero; the self-audit now fails in the direction that matters and I proved it by construction. The
   three Importants are one sub-second loop, one truncated quotation, and one stale heading — repairs
   measured in words and lines, not sections. **The science is sound**: I rebuilt the arms from the
   prose without help, could not manufacture a CLOSE anywhere in the combination space, and every
   number I checked reproduced.

## 5. Process rules

**`rule_1_compute_projection`.** §8 re-multiplies exactly — I re-derived the printed column sum
(`2930.4`), the `× 1.25`, both minute figures, both shares, both sensitivity figures, the freeze
cost and every explicit count. **Ten rounds, nine uncounted loops: the ninth is the arena's native
key materialisation for `GATE-FLOOR` (I-1).** On the timing spread, ruled at §15 item 3: the
conservative carry is right, consistent, and disclosed per row.

**`rule_2_heartbeat`.** **Nothing in v10 changes an interval.** Phase 7z falls `2.0 → 1.8 s`, Phase
1f adds `1.0 s` spread across 60 cells already covered by the per-`(dataset, seed, lineage, fold)`
line, and I-1's missing row would add `≈ 0.13 s`. The longest un-instrumented span remains one
`GATE-C01PARITY` dataset at `11.27 s` (`14.1 s` conservative), under the `~15 s` bound. Per-cell
lines cover Phase 2D's `38.4 s`; per-32-draw lines cover Phase 3's `273.7 s` at `≈ 2.9 s` each
(`3072/32 = 96` lines); per-epoch lines cover the `40 s` mints. The `buffering=1` per-phase handle,
the unbuffered driver echo and the `elapsed ÷ §8's frozen projected` denominator are unchanged and
adequate. The denominator tracks §8 automatically, so the `2929.6 → 2930.4` move needs no separate
edit.

## 6. Freeze-readiness, in the operational sense

**Ready on everything except the three Importants, and I judged it as the document that will be
hash-frozen and executed.** All 37 sha256 recompute today. Every constant is pinned and verified
against `configs/c01/c01_a0_v2.json` at the source. The run boundary is unambiguous for a
context-free operator: **one submission, SLURM CPU queue, 8 CPU / 32 GB, no `--gres`, no `--time`,
no array, no dependency, no requeue; 73 processes** in the stated order **66 mints → 6 fidelity →
1 arena** (`66 + 6 + 1 = 73`, agreeing with §12's process-count predicate), with `GATE-SHA` once in
the driver before any of them and `GATE-POP` before any population-consuming gate. Preconditions are
checkable: `mints_present_before_arena == 66` is binding and separate from the resume-safe
`dev_path_opens == mints_executed`, and `GATE-FOLD`'s banked-flag re-read makes the fold contract
exactly predictable on fresh, resumed and partially-resumed runs. Exit semantics are defined: the
HALT path names the failing gate in its final line, and a `RuntimeError` from the imported C01
algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE` with `l2_rows`' `context` string, and written
to that line before exit. §8 is independently re-multiplied per F118 — measured units × explicit
counts, no extrapolation. The four new-code artifacts are absent from the tree.

Two operational residues: the heartbeat path's `$BASE` is undefined in the document (**M-3**), and
§13 item 22 does not pin the native forward or `GATE-FLOOR`'s vote site (**I-1**, second half) —
the latter is the only thing a code lineage would have to guess at.

## 7. Can the falsifier discharge the written condition at `$0`? **Yes.**

The instrument does exactly what the registry asks: re-run C01's real-displacement-versus-rotation
battery in the fold-head arena on already-banked caches, on CPU, with no extraction. I confirmed the
head-space arms are buildable and their dimensions forced (`4 × 1024-d` + `9 × 2048-d`, which falls
out of the raw `4 × 7168` + `9 × 14336` I measured); the anchor reproduces bit-exactly and I
reproduced it myself from the prose; the arena is alive (`GATE-FLOOR`'s native OOF accuracies run
above C01's dev arena, verified from the banked OUT JSONs); and the decision rule reaches CLOSE,
SURVIVE and HALT on distinguishable states with no unmapped combination. Nothing in my three
findings threatens the `$0` character, the verdict path, or the `1.7–2.5 GPU-h` this falsifier
exists to avoid spending. **The blocker is the record, and it is now three line-level repairs
deep.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Not applicable this round. For the record: a GO authorizes **nothing to run**. Before any job this
design still needs (1) a freeze with hashes, (2) a **separate** independent code/resource review
lineage over the executable reaching its own `0C/0H/0I`, and (3) main-dialogue authorization. A GO
is not authority to write `TARGET_STATE.json`.

---

# CLOSING

The most severe finding is **I-1**, and it is severe only in the sense that it is the ninth
consecutive round in which §8's enumeration has been found one loop short — never in what it costs.
The arena votes `GATE-FLOOR`'s 30 native cells in Phase 2, and to do that it must pull 30 native key
matrices off disk that Phase 1f, which prices exactly this class of operation, counts at `60 × 2`
and not `60 × 2 + 30`. I measured the unit rather than estimating it (`0.0043 s` per matrix, warm,
against v10's own `0.0041 s`), so the omission is `≈ 0.13 s` against a `2930.4 s` projection
carrying a `× 1.25` margin and `30 s` of declared slack. It changes nothing. What matters more than
the seconds is that §13 item 22 pins the mint-side placement for the `h_std`/`h_ow` stream and for
no other, so the count Phase 1f prints cannot be checked against any statement in the document — and
§13 is the sole input to the code lineage that has to build this.

The rest is the smallest residue any round has reported. **The verbatim-quotation protocol works**,
and the evidence is that the one thing it was built to expose is the one thing I found: round-9
M-2's limb stops before its qualifier, and v10 implemented a different exclusion while recording
*"both adopted"*. Round 9 said no machinery could catch that; the protocol did not catch it either,
but it made a reader's check take seconds instead of a reconstruction. Nineteen of twenty limbs are
faithful and complete, and the H-2 limb is the interesting one in the other direction — the designer
adopted round 9's three lines byte-identically and then added the two lines round 9's block was
missing, without which §14.1 would have failed *always* rather than *exactly when unchanged*. That
is a prescription read rather than pasted, and it is disclosed in both the script and the prose.

And the science reproduces under independent measurement at every point I tested it — the arms
rebuilt from prose at `0.000e+00` on the first attempt, all 26 `ρ` at six decimals, all 37 digests,
all six floor triples on both metrics, the Holm table cell for cell, the whole §8 column to
`2930.4`, and twenty gates none of which can fire on the outcome this falsifier exists to publish.
That is now the fourth consecutive round for which that is true.

---

*Read-only review. No GPU, SLURM, Modal, arena run, mint, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json` was read and not modified. The draft, the configs
and all repository files are unmodified; my scripts, my counterfactual copy and all outputs live in
the session scratchpad only.*
