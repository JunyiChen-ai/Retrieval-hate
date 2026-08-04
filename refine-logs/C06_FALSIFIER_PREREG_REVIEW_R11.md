# C06 `$0` falsifier — independent design review, **ROUND 11**

*Artifact:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V11.md` (unfrozen, sha256
`94699ac37c4a06b800cde384d67d5b2da6b04ca45f6c9808011f7304aa98fc32`, 157460 bytes, 2136 lines) —
recomputed by me against disk.
*Reviewer:* fresh, independent of rounds 1–10 and of the designer.
*Compute used:* `sha256sum`; read-only numpy/torch-CPU re-derivations on banked **train-split**
caches, banked `vsw_ckpt` and C09 mint `.npz`, banked arena OUT JSON and `C01_A0_OUT.json`;
execution of the audit script **as reproduced in §14.2** against the on-disk v11 and against a
counterfactual of my own construction; my own line-based section splitter; my own exhaustive
item-reference scanner; three timings of an arena-class interpreter+import. No mint, no arena run,
no GPU, no SLURM, no Modal, no job, no cache write, no test-split open, no commit. The draft, the
configs and all repository files are unmodified. `TARGET_STATE.json` read only. **I declined the
four permitted CPU mints**, as rounds 8, 9 and 10 did: §7.8 is byte-identical v7→v11 and has been
verified four times; this round's obligation is the record and the v10→v11 delta, and every
head-space quantity I needed was available from the banked C09 mints, which I say so wherever I
used them.

---

# VERDICT

## **REVISE — 0C / 0H / 1I + 1M**

**The science layer is closed and I confirm it a fifth time, independently and by measurement.** I
rebuilt all thirteen arms from §3.4's prose alone and got `max|diff| = 0.000e+00` against
`prepare_views` on **both** datasets, **at the first attempt with nothing silently supplied**, at
`n = 744` one-hot `{355}` and at the arena `n = 743` / `579` all-False. The `GATE-ROWSUBSET` bridge
is `0.000e+00` on all 13 arms on both datasets. Every wrong reading I constructed is caught:
omitting endpoint pre-normalisation costs `1.878e-06` / `1.609e-06` — **both under the `2e-6` v7's
row would have allowed**, which is round-7 C-1's wrong-verdict path reproduced exactly;
`common_interaction = l2(std ⊙ ow)` costs `9.697e-01` / `9.558e-01`. All **37** sha256 recompute
against disk with zero mismatches. All **26** `ρ_raw` reproduce at **6 dp** under §6.1's frozen
float64 reduction, `ρ*` included. The 36 banked C09 mints give **0/18 above `ρ*` on both** at
`0.447803 / 0.562434 / 0.632996` and `0.340179 / 0.574247 / 0.667326`. All six `GATE-FLOOR` triples
match §6 on **both** metrics. §1's table recomputes **cell for cell** from `C01_A0_OUT.json`'s
stored confusion matrices — 16/16 accuracies, 16/16 net-fix integers. The Holm counterexample table
reproduces cell for cell under C01's own `holm_adjust`, and the S5 bound is `n ≤ 12`.
**No gate can fire on a warranted CLOSE, all twenty, re-derived from the gate texts and from
measurement.**

**The record is now sound, and this is the first round at which I can say that without a
qualification.** All 13 limbs are verbatim **and complete** substrings of round 10's review, each
inside the `R10:NNN-NNN` line range it cites. I subtracted them from all seven Repair paragraphs:
**six subtract to nothing but connective punctuation**, and the seventh leaves a single clause
which is a directive about **v10's own file** that was literally obeyed (v10 on disk is
byte-identical to round 10's recorded sha). **All four disclosed deviations are warranted**, and two
of them are not merely defensible but mechanically required by round 10's own stated rules. §14.1's
transcript is a byte-identical fixed point at the first attempt, and the self-exclusion still fails
in the direction that matters when I break it by construction.

**One Important, and it is the tenth uncounted item in §8 — the one §15 item 3 asked me to hunt.**
The battery runs **73 processes**, and §8 prices interpreter+import for **66** of them. The 66
mints carry it inside their full-process-wall units, exactly as §7.2 explains; the arena — a 67th
process §8 Phase 1c names explicitly — carries it in no unit and in no row, because every arena-side
unit (`U1`, `U2a`–`U2d`, `U3`–`U8`, `U10`) is an internal operation timing. I measured the cost
rather than inferring it: **1.84–1.91 s** on this node for an arena-class import set, against the
document's own `U11` of `3.05–3.18 s`. §7.2's sentence *"Interpreter and import cost is **already
inside every unit**"* is true of the mint units the section is about and false of the arena's.
The same sentence carries the Minor.

I have not graded on trajectory in either direction. Eleven rounds is evidence of nothing. I found
the Important by measurement after failing to find anything in the record, and I grade it at exactly
the severity rounds 7, 8, 9 and 10 gave the fifth through ninth items of the same class — which is
also why I decline to grade it Critical, and why I decline to grade it Minor.

---

# PART A — THE TWELVE §3 VERIFICATIONS

| # | claim | result |
|---|---|---|
| **V1** | all 37 sha256 match disk; resolve the four elided cache stems | **PASS — 37/37 recompute, zero mismatches.** 7 imported modules + 6 read-for-definitions + 8 input caches + 16 banked = **37**, matching `U7`'s *"8 caches + 13 modules/configs + 16 banked"*. The four elided stems resolve to `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` (HateMM) and `Qwen2.5-VL-7B-Instruct-LoRA_HF` (MHC_zh) under `data/CLIP_Embedding/<ds>/`. **Independent cross-check:** `c01_a0_v2.json::inputs.datasets.*.expected` stores `standard_provenance_sha16` / `oneword_provenance_sha16` — `6a44cce4f65d4a60`, `60054f3be1204ca7`, `1d33fe5d69083479`, `3ad1309dc7500182` — all four are the 16-hex prefixes of §11's four ro-cache digests, so the caches are pinned twice over. |
| **V2** | break the self-exclusion as round 10 broke v9's | **PASS.** I built a counterfactual: v11 with **v10's §14.1 spliced in**, splice verified byte-identical to v10's section under my own splitter. The §14.2 script then prints `UNCHANGED §14.1 (self, size not reported)`; **fails** `I-2` and `M-2` (`rows verified 5 ; rows failing 2`); **fails** the two §14.1-citing limbs (`limbs landed 11 ; open/failing 2`); reports `named by a row but unchanged: ['14.1']`; exits `1`. The logic is unchanged from v10 and it still fails in the direction that matters. |
| **V3** | re-run the audit; transcript byte-identical; §14.1's size never printed | **PASS on both.** My run of the extracted §14.2 script against the finished on-disk v11 is **byte-identical** to the embedded transcript — **2610 bytes**, sha256 `bd28df91b0e3eb98e6819b7662b75dd059763f51011d4b1098d557dea4f296d7`, both — exit `0`. §14.1's size is never printed; its **changed status is**. My own line-based splitter reproduces **every** printed delta exactly (`+786 / +686 / +3844 / +905 / +193 / +600 / +644 / +751 / +721`, header `+348`, `UNCHANGED: 47`) and independently measures the §14.1 size the transcript correctly declines to print (`6659 → 8528`), confirming `CHANGED §14.1` is a measurement and not a convention. §14.1 is a verified fixed point for the **third** consecutive version. |
| **V4** | the 13 limb quotations, by subtraction | **13 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded.** All 13 are verbatim substrings of round 10 under the declared normalisation (backticks, emphasis, quote-glyph style), **and every one is inside the `R10:NNN-NNN` range it cites** — I checked the range independently, line by line. Full table and the residues in Part B. |
| **V5** | Phase 1f at the extended count, and every derived figure | **PASS on every figure, re-multiplied by my own parser of the printed column.** `60 × 2 + 30 × 1 = 150`; unit `0.0041 s` with the timed region stated; **I re-measured the native `K_train` array myself** on five banked mints, 20 timed `np.load` + `np.asarray`: **median `0.0041 s`, mean `0.0043 s`** — bracketing v11's `0.0042 / 0.0044` and round 10's `0.0043`, and confirming the file description (`(744, 1024)` float64, `6.1 MB`). Measured product `150 × 0.0041 = 0.615 s`; carried `1.3 s` bounds both that and `150 × 0.0044 = 0.66 s`. Printed column sums to **`2930.7`** exactly; `× 1.25 = 3663.375 → 3663.4`; `48.845 / 61.056 min → 48.8 / 61.1`; mint share `2508.3/2930.7 = 85.587 % → 85.6 %`; Phase 3 `9.339 % → 9.3 %`; `2×` miss `3204.4`; `5×` miss `4025.5`; and the stated decomposition `2927.6 + 1.0 + 0.7 + 0.1 + 1.3 = 2930.7` closes. Every explicit count re-derives: `(30×3)+(6×4)+(30×2)=174`, `67`, `4×60/9×60=240/540`, `30`, `60`, `120+60×(2/13)`, `40/90`, `62`, `256×3×2×2=3072`, `23×2×2=92`, `14×3×2×2=168`, `13×60=780`. **No stale total survives:** `3663.0`, `2929.6`, `3662.0`, `3203.3`, `4024.4`, `2930.6`, `3204.1`, `4025.2`, `48.84`, `61.05`, `85.60`, `9.34` — zero occurrences each. The five surviving `2930.4` are all historical or quotations of round 10's own text. |
| **V6** | item 22's placement, checked against the scripts **as they actually are** | **PASS, at the source, and the count follows from it.** `headspace_mint.py:306` `K_train = keys_of(tr)` and `:307` `K_dev = keys_of(dv) if a.fold < 0 else …` — the key forwards happen **inside the mint**, and `:307`'s `fold < 0` branch is exactly why Phase 1b prices the native dev forward for the **6 full mints only** (`6 × 4`). `:322` `np.savez(tmp, K_train=…, K_dev=…, …)` — each mint writes its key matrices into its own `.npz`. On the arena side, the fold loop opens at `headspace_arena.py:75`, `load_mint` (`:59` `np.load(mint_{ds}_s{seed}_f{fold}.npz)`) is called at `:85` **inside the loop**, `:89` materialises `X = P.l2n(zf["K_train"])`, and `:92-93` computes `M.deployed_vote(X[fit_idx], lab[fit_idx], X[ho_idx], topk=K)`. **So `GATE-FLOOR`'s vote is computed in the arena process and the arena must pull 30 native key matrices off disk** — item 22's clause is true of the code, and Phase 2's `30 × U2a` is correctly a vote-only timing. I re-derived the count from the design's own structure: `GATE-FLOOR`'s 30 votes are `2 ds × 3 seeds × 5 folds` Head-N fold mints ⇒ 30 native matrices; the 6 full mints are read by the fidelity processes inside `U9`; Head-R has no native matrix; Head-R's `h_std` **is** its `K_train`, counted once inside the 120. `60 × 2 + 30 × 1 = 150`, with no double count. |
| **V7** | the item-scan mechanism, by my own exhaustive scan | **PASS on all three questions.** (a) **No genuine §13-item reference is unreachable.** I enumerated all **107** `item`/`items` occurrences in v11: 41 are matched numbered runs, 13 are excluded by the `§`-prefix filter, and all **53** remaining are non-references (`per item`, `item count`, `72 items`, `bare item N`, `§9 items`, the script's own `items` identifier, …). I inspected all 41 matched sites: **every one is a genuine §13-item reference**. (b) **No non-§13 reference is swept in** — the 13 excluded sites are exactly the §5.9 items (1/4/5/6/6/6+8/8/9) and §15 items (2/4/5), all correctly excluded by their prefix. (c) **The printed scope is true of the pattern**: `items?\s+((\*\*)?\d+…)` matches *"every item/items reference carrying a number, a comma/and list or an en-dash range"*, and `SEC_PRE` implements *"a reference prefixed by a § section other than §13 is excluded by that prefix"* — including `§13.1 item 22`, which captures `13` and is correctly **not** excluded. Round-10 I-2's defect (a scope line false of its own pattern) is repaired. **I also reproduced v11's own new measurement**: running v11's pattern against v10's text under v10's **full three-arm** pattern, I find **12** sites v10 could not reach — the **nine** genuine §13 sites v11 enumerates (`v10:1320, 1565, 1569, 1573, 1590, 1638, 1668, 1713, 1752`, exact line-for-line match) **plus the three §5.9 siblings v11 separately says it prefixed** (`v10:837, 841, 844`). Both groups are accounted, correctly and separately. |
| **V8** | §7.9's sum, heading, footer, and the v11 spend row | **PASS.** Heading reads **`Cumulative v1–v11`**; the terms are `7 + 1 + 4 + 0 + 0 + 0 + 0 = ` **12**, `≈ 22 + 4 + 2 + 1 + 1 + 1 = ` **31**, `≈ 89 + 21 + 6 + 3 + 3 + 3 = ` **125** — all three re-derive, all three run through v11, and the footer says *"twelve CPU head mints across **v1–v11**"*. Heading, terms and footer agree for the first time. **v11's own round-compute is accounted per the section's own spend rule**: `≈ 1 wall-minute / ≈ 3 CPU-minutes and no mints`, itemised as the native `K_train` re-measurement, the exhaustive item scan and the audit re-runs — which is exactly the work v11 claims elsewhere, and nothing else. |
| **V9** | rebuild the arms from §3.4; `GATE-C01PARITY` one predicate | **PASS, measured, first attempt.** `max|diff| = 0.000e+00`, both datasets, 13/13 arms, arm sets identical, `float32`, dims **`4 × 7168-d` + `9 × 14336-d`** raw ⇒ `4 × 1024-d` + `9 × 2048-d` in head space. The prose determines the Givens sign convention uniquely: the θ=45 identity forces `second = −sinθ·std + cosθ·ow`, and I measured the identities at `8.941e-08` (θ=0, both) and `1.192e-07` / `8.941e-08` (θ=45) — §1's figures to the digit. §6's row states exactly one predicate, `max|diff| == 0.0`; `2e-6` survives only as `GATE-ALGEBRA`'s bar. My un-normalised misreading measures `1.8775e-06` / `1.6093e-06`, **both under `2e-6`**, confirming round-7 C-1's path is real and that bit-exactness is the only safe bar. |
| **V10** | `ρ*`; all 26 `ρ_raw` at 6 dp; trained-head `0/18` | **PASS, exactly, 26/26.** Under the float64-over-float32 reduction all 26 reproduce at **6 dp**, `orthrot_83p8` at `0.956894` included. `ρ*` `0.968176` / `0.977223` (`endpoint_std`); runners-up `0.964446` / `0.969686` (`common`) — I confirm `common` is the runner-up on ZH too, since ZH's `common_interaction` (`0.968188`) sits just below it. Trained-head `ρ` on the 36 banked C09 mints: **0/18 above `ρ*` on both**, at roughly half the bar. Masked-zero-row shift `1.3013e-03` ✓. |
| **V11** | Holm counterexample; `n ≤ 12`; §3.7's two blocks | **PASS.** Under C01's own `holm_adjust`: `m = 92` gives **24/24**, **23/24**, **0/24**; `m = 46` gives 24/24 in all three rows; the `displacement` disjunct is **22/22** at `m = 92`; padding `1.0` (the drop path) still gives 24/24, so `NOT_TESTED, p = 1` is non-rejecting as §5.5 requires. `92×1/2001 = 0.045977 ≤ 0.05`; `92×2/2001 = 0.091954 > 0.05`; `46×2/2001 = 0.045977 ≤ 0.05`. S5: `1/257 = 0.0038911`, `12/257 = 0.046693 ≤ 0.05`, `13/257 = 0.050584 > 0.05` ⇒ **`n ≤ 12`**. §3.7's **two blocks with two verbs**: the 7 population rows I **computed** from the arena (`743/579`, `(297,446)/(180,399)`, `0.600269→0.6003` / `0.689119→0.6891`, full `0.599462→0.5995`, bands `[0.6203, 0.98]`/`[0.7091, 0.98]`, caps `⌊0.01×743⌋=7` / `⌊0.01×579⌋=5`), none readable from a config; the 4 config rows I **read** at `c01_a0_v2.json::transforms` (`<=` at `displacement_audit:2036`, `0.001`, `0.05`, `1e-12`). Verbs correct in both directions. §5.9 item 4's counterexample also verifies: `net = (2,21,22)`, mean `15.00 ≥ 14.86`, `min 2 < 3`, spread `20/743 = 2.69` accuracy points. |
| **V12** | 20 gate rows, `12 G / 6 L / 2 R`; 26 contiguous items; items 10/15/19/22 | **PASS.** §6 has exactly **20** rows; the scope column counts **12 G / 6 L / 2 R**; the G-set and L-set match §5.6's two lists **name for name** (set-symmetric-difference empty in both directions). §13.1 defines `**(1)**…**(26)**` contiguously, no gap, no repeat, no duplicate. **Item 10** carries *"computed ANALYTICALLY"* and the per-`(dataset, seed, lineage)` pooling; **item 15** carries the import/call distinction and `tiny_ok`'s non-carriage; **item 19** carries both endpoint pre-normalisation and bit-exactness; **item 22** now carries *"**All three** key forwards"* **and** *"`GATE-FLOOR`'s vote is computed in the arena process, not in the mint"* — round-10 I-1's second half, landed. |

**Ceremony floor.**

* **All 37 sha256 recompute** today against §11's `2026-08-04` heading.
* **C01 constants verified at the source in `configs/c01/c01_a0_v2.json`**: the five `transforms`
  values; `minimum_gain_over_strongest_control 0.02`; `minimum_net_fixes {HateMM: 3, MHC_zh: 2}`;
  `gain_controls = ['endpoint_std','endpoint_ow','avg_score','endpoint_concat','common']` (**five**,
  `avg_score` among them, so §5.1's *six* and *five* comparator counts are right);
  `n_bootstrap 2000`; `statistics.seed 20260728`; `holm_alpha 0.05`;
  `holm_metrics ['accuracy','macro_f1']`; `n_id_hash_permutations 256`; `permutation_hash sha256`;
  `bootstrap_lower_quantile 0.05` / `upper 0.95`; `small_displacement_gate_reference` and
  `small_displacement_endpoint_concat_role = 'diagnostic_only'` **under `transforms`**;
  `angles_degrees = [8.3, 17.6, 29.1, 60.4, 72.7, 83.8]` with `same_block_l2: true`;
  `required_halt_only_validity_guards` = **7 entries under `output.decision_schema`**, matching
  §5.4.1's disposition list **name for name**; `retrieval.fix_break_reference = 'endpoint_std'`,
  and `decision.datasets.*.checks.net_fixes.reference` = `'common'` / `'endpoint_concat'` —
  **D-1's two-site account confirmed at the source in both directions**.
  `inputs.feature_dim 3584`, `standard_suffix ro_L24`, `oneword_suffix ro_ow_L24`,
  `expected.train.n` `744` / `579` — §3.1 exact.
  Every cited source line resolves at the digit: `l2_rows:1183`, `:1187-1188` (the `None`→zeros
  normalisation), `:1193-1194` (fail-closed), `contrast_blocks:1242`, `orthogonal_blocks:1272`,
  `prepare_views:1294` with the normalisation loop at `:1296-1304`, the algebra guard `:1372-1377`,
  `displacement_audit:2036/2049/2050`, `holm_adjust:1777-1784`, `mechfix_ops.py:94`
  (`votes = ((lab*2−1)·sim·w).sum(1)/w.sum()`), `classifier.py:81-82` (both projections with default
  bias, `:80` the comment line), `:140-141`, `:146`, `headspace_mint.py:192-194/199/203-216/209/
  306-307/321-325` (`lab_dev` at `:323`), `headspace_fidelity.py:31/33/66`.
* **`headspace_fidelity.py` opens no `dev_seen` file at all** — I grepped the whole module for
  `dev_seen`, `load_split` and `torch.load` and it contains none, reading `lab_dev` out of
  `mint_*_ffull.npz` at `:66`. §12's `dev_path_opens == mints_executed + 0` is correct and §7.4's
  row (l) is accurate.
* **Blindness grep across v1–v11**, every decimal in `[0.6, 0.99]`: **116 distinct** across the
  corpus under my regex, with **exactly two new in v11** — `0.615` and `0.66`, and **both are
  seconds**, from Phase 1f's `150 × 0.0041 = 0.615 s` and its `150 × 0.0044 = 0.66 s` bound.
  **Neither is an accuracy**, so §7.3's *"v11 adds no accuracy of any kind"* is verified rather than
  inherited, and the `v1–v11` scope label is earned. I classified every other in-range decimal in
  v11: published C01 dev-arena accuracies (§1's table, all recomputed by me from the OUT file),
  banked `GATE-FLOOR` anchors, majority/band constants, `ρ` values, displacement-norm geometry, and
  §7.8's `0.8725` native deployed vote — an instrument anchor already published in §6.
  **No battery-arm accuracy exists anywhere in v1–v11.**
* **Test-set non-contact by construction.** `test_seen` occurs **once** in v11, in a negative
  assertion (*"the `test_seen` ro caches are opened by nothing"*); every other test mention is a
  prohibition or a scope exclusion. No §8 phase opens a `test_*` path. Every quantity I recomputed
  came from `train_*` caches, banked train-split mints, or banked OUT JSON.
* **`GATE-IDPARITY` verified directly**: on both datasets the native, `ro_L24` and `ro_ow_L24`
  caches carry identical `ids` order and identical `labels`. HateMM row 355 is `hate_video_95`,
  **label 1**, exactly zero in both modalities of both ro caches **and** the native cache, the only
  such row on either dataset, and `h_std[355] == h_ow[355]` bit-identically at the feature level.
* The four new-code artifacts are confirmed **absent** from the tree, and so is
  `artifacts/c06_falsifier/`, so the code lineage starts from zero.
* **v10 on disk is byte-identical to what round 10 reviewed** (`f515764760638a24…`), and every
  draft's mtime precedes its own review's. v1–v10 are unmodified, as v11's footer states.

---

# PART B — MY LIMB-LEVEL DISPOSITION AUDIT OF ROUND 10's SEVEN FINDINGS, BY SUBTRACTION

**Method.** I extracted each **Repair** paragraph from round 10 verbatim by line range, normalised
only backticks, emphasis and quote-glyph style, checked each v11 limb is a substring of round 10's
whole text **and** of the `R10:NNN-NNN` range it cites, then **subtracted the limbs from the Repair
paragraph and read the residue**. Nothing here was checked against v11's row table.

## Result: 13 FAITHFUL / 0 TRUNCATED / 0 NARROWED / 0 unlanded

| # | finding | limb | cited range | verbatim | in range | verdict |
|---|---|---|---|---|---|---|
| 1 | I-1 | *"Extend Phase 1f's count to `60 × 2 + 30 × 1 = 150` materialisations (or add a Phase 1g row for the native stream)"* | `R10:236-237` | ✓ | ✓ | **FAITHFUL** |
| 2 | I-1 | *"re-multiply — `150 × 0.0083/2 ≈ 0.62 s` measured, `1.3 s` at the cold-cache bound, so the total moves `2930.4 → ≈ 2930.7`, still `3663` at one decimal after `× 1.25`"* | `R10:237-239` | ✓ | ✓ | **FAITHFUL** — and the annotation is honest about the one place it differs: round 10 wrote `3663`, v11 prints `3663.4`, and the limb says so |
| 3 | I-1 | *"And extend §13 item 22 by one clause naming where the native forward happens and where `GATE-FLOOR`'s vote is computed, so the count has a stated placement to be checked against"* | `R10:239-240` | ✓ | ✓ | **FAITHFUL** |
| 4 | I-2 | *"Restore the full quotation in the limb table"* | `R10:290` | ✓ | ✓ | **FAITHFUL** — and it is restored in full in the *Rounds 1–9 — carried* block, which I checked against round 9's own sentence |
| 5 | I-2 | *"Either implement round 9's prescribed exclusion (…) or state in the annotation that a different constraint was chosen and why"* (full, including the `(?<!§5\.9 )(?<!§15 )` example and the `§13`-anchored alternative) | `R10:290-293` | ✓ | ✓ | **FAITHFUL** — first disjunct taken, and I verified by running the pattern that it **is** round 9's prefix mechanism with no verb whitelist |
| 6 | I-2 | *"Then correct the printed scope to describe all three arms"* | `R10:294` | ✓ | ✓ | **FAITHFUL** (deviation ruled below) |
| 7 | I-2 | *"and soften the §14.1 parenthetical to say that three unreachable references remain and are verified against §13 by reading"* | `R10:294-295` | ✓ | ✓ | **FAITHFUL** (deviation ruled below) |
| 8 | I-3 | *"Cumulative v1–v10"* | `R10:328` | ✓ | ✓ | **FAITHFUL** (deviation ruled below) |
| 9 | M-1 | *"`v1–v9` → `v1–v10`"* | `R10:339` | ✓ | ✓ | **FAITHFUL** (deviation ruled below) |
| 10 | M-2 | *"`round-8` → `round-9`, `v9` → `v10`, `v8->v9` → `v9->v10`"* | `R10:347-348` | ✓ | ✓ | **FAITHFUL** (deviation ruled below) |
| 11 | M-2 | *"Changing the printed header will move the transcript, so re-run and re-embed — the fixed-point discipline already covers this"* | `R10:348-349` | ✓ | ✓ | **FAITHFUL** — and I confirmed the fixed point holds |
| 12 | M-3 | *"name the base directory, or write the path repo-relative as every other path in the document is"* | `R10:354-355` | ✓ | ✓ | **FAITHFUL** — and v11 does **both**, as its annotation claims |
| 13 | M-4 | *"either cite round 9's line numbers (its own reviews use the `v9:1222-1224` form), or make the header say "with the finding it came from" so the two sentences agree"* | `R10:361-363` | ✓ | ✓ | **FAITHFUL** — first disjunct taken, the form round 10 called sharper, and **the line ranges are real**: I checked all 13 |

## The residues

| finding | residue after subtracting the limbs |
|---|---|
| **I-1** | `Repair. ⟦LIMB⟧ , ⟦LIMB⟧ . ⟦LIMB⟧ .` — **nothing but connective punctuation** |
| **I-2** | `Repair, three lines. ⟦LIMB⟧ . ⟦LIMB⟧ . ⟦LIMB⟧ , ⟦LIMB⟧ .` — **nothing but connective punctuation** |
| **I-3** | `Repair, one word. " ⟦LIMB⟧ ".` — **nothing** |
| **M-1** | `Repair: ⟦LIMB⟧ . (v10:2046's "v1–v9 are unmodified" is correct as written and must not be changed — v10 is the live draft.)` — **one clause remains** |
| **M-2** | `Repair: ⟦LIMB⟧ . ⟦LIMB⟧ .` — **nothing** |
| **M-3** | `Repair: ⟦LIMB⟧ .` — **nothing** |
| **M-4** | `Repair paragraph — so M-4's stated purpose is met. Repair: ⟦LIMB⟧ .` — **nothing but the sentence tail that precedes the word "Repair"** |

**The one residue, ruled: not a finding, and I checked it rather than assuming it.** Round-10 M-1's
parenthetical is a directive about **v10's own file** — *"v10:2046's `v1–v9 are unmodified` is
correct as written and **must not be changed**"* — carrying its own condition, *"v10 is the live
draft"*. Two things are true. First, the directive was **literally obeyed**: v10 on disk hashes to
`f515764760638a24a940ff2bc0932be03f4ffa04910b7e9bfec682b227036a75`, byte-identical to the artifact
round 10 reviewed, so v10:2046 was not changed. Second, the condition is explicitly false at v11 —
v11 is now the live draft — and v11's own M-1 row says exactly that: *"The footer's `vN are
unmodified` clause advances to **v1–v10**, since v11 is now the live draft."* v11's footer reads
*"v1–v10 are unmodified"*, which is the true statement for v11 and which I verified. Structurally
this clause **cannot** be a limb: the limb table's third column is *landed in*, and a directive to
leave a different file alone has no landing section. Round 10's own standard — its limb 3 ruled a
justification clause *"an explanation, not an action, and correctly not a limb"* — points the same
way. The residue is disclosed, reasoned and correct.

---

# THE FOUR DISCLOSED DEVIATIONS — RULED

**All four warranted. None is narrowing wearing a disclosure.** Two are consequences of a disjunct
round 10 itself offered; two are mechanically required by round 10's own stated rule.

**1. *"correct the printed scope to describe all three arms"* → v11 has one scanner and one filter.
WARRANTED, and it is entailed by the limb immediately above it.** Round 10's preceding limb offered
*"drop the verb whitelist in favour of a `§13`-anchored form"* as an explicit disjunct; taking it
necessarily collapses v10's three arms to one scanner plus one prefix filter. Round 10's *"all
three arms"* was written on the assumption the three-arm pattern would survive, which the other
disjunct forecloses. What round 10 actually diagnosed — *"the printed scope is false of the pattern
it describes"* — is the defect, and it is repaired: I ran the pattern against the document and the
printed scope is **true of it**, arm for arm. Adopting the letter here would have required printing
a scope for arms that no longer exist.

**2. *"soften the §14.1 parenthetical to say that three unreachable references remain"* → v11
reports zero remain. WARRANTED, and v11's measurement is the stronger one.** The prescription is
executable only if three references remain unreachable; after the pattern is widened as round 10's
other limb directed, **none does** — I verified this by my own exhaustive scan of all 107
`item`/`items` occurrences. Softening to *"three remain"* would have put a false statement into the
document. More than that: v11's replacement measurement is **more accurate than round 10's own**.
Round 10 called its scan exhaustive and reported three unreachable sites in v10; running v10's full
three-arm pattern myself I find **nine** genuine §13-item sites it could not reach — exactly the
nine v11 enumerates, line for line — plus the three §5.9 siblings v11 separately says it prefixed.
v11 discloses the relationship plainly (*"Round 10 enumerated three of them"*) rather than quietly
correcting a reviewer. This is a condition closed, not a condition softened, and the record says so.

**3. I-3's *"Cumulative v1–v10"* and M-1's *"`v1–v9` → `v1–v10`"* → landed at v1–v11. WARRANTED,
and mechanically required.** Round 10 wrote both against v10 and stated the governing rule in I-3
itself: *"the heading is the version span of the terms beneath it."* At v11 the terms beneath it
run through v11 (`+ 0 (v11)`, `+ 1 (v11)`, `+ 3 (v11)`), so v1–v10 would have reproduced the exact
defect round 10 found — a heading excluding the version whose terms it contains. I checked heading,
terms and footer and all three now agree. The §7.3 case is the same shape with one extra
obligation, and v11 discharges it: the widened blindness scope is only honest if v11's own
contribution is stated, and v11 states it (*"no accuracy of any kind"*), which I verified
independently by corpus grep. Both are claims, not extrapolations.

**4. M-2's `round-8`/`v9`/`v8->v9` → landed at `round-10`/`v11`/`v10->v11`. WARRANTED, and
mechanically required.** v11 dispositions round 10 and diffs v10→v11, so those are the only labels
that are true. I confirmed the docstring reads *"C06 falsifier v11"* and *"section diff v10->v11"*,
`V_OLD`/`V_NEW` point at v10/v11, the printed header says `v10 -> v11`, and step (5) reads
*"(round-10 prescriptions)"* over a table headed *"Round 10 — LIMB level"*. The stale-label defect
round 10 named is gone in every instance, and the transcript was re-run after the labels moved,
exactly as limb 11 required.

---

# FINDINGS

## CRITICAL

**None.** I state this rather than imply it by omission. I looked where the last five rounds found
theirs — inside this round's own repairs (§8, §9, §13.1, §14/§14.1/§14.2), in §13, and in the gate
set — and I could not construct a wrong-verdict path, could not manufacture a CLOSE anywhere in the
combination space, and could not find a claimed repair the artifact does not contain. The one
Important below is an enumeration gap in a compute projection, which touches no decision quantity.

## HIGH

**None.** Nothing weakens the verdict's authority or scope. **No repair landed narrower than
prescribed** — this is the first round for which that sentence needs no qualification: 13 of 13
limbs are faithful *and* complete, the subtraction leaves nothing unaccounted, and all four
deviations are warranted on inspection rather than on assertion.

---

## IMPORTANT

### I-1. The tenth uncounted item in §8: the battery runs **73 processes** and §8 prices interpreter+import for **66**. The arena — the 67th process §8 Phase 1c names by name — carries its startup in no unit and in no row, and §7.2 asserts the cost away in a sentence that is true of the mint units and false of the arena's.

*Attaches to:* §8's phase table (v11:1305-1331, no startup row); §7.2 (v11:1095-1099); §7.7's `U11`
row (v11:1177) and its `U9` row (v11:1175); §15 item 3.

§15 item 3 asks round 11 to *"hunt the tenth. Ten rounds, nine found."* Here it is. It is not a
payload loop — it is the per-process fixed cost that the nine payload loops all sit inside, and it
is the one axis of `rule_1`'s *"measured unit × explicit count"* form where a measured unit
(`U11 = 3.05–3.18 s`) is carried with **no count** outside the mints.

**Where it is correctly priced, and where it stops.** §7.2 is titled *"Mint units — full-process
wall"* and it is right about them: every mint figure is *"measured around the `python …`
invocation"*, the `40.39 s` unit contains the `33.0 s` internal timer plus the `7.4 s` gap, and
rounds 3 and 4 correctly confirmed that adding a separate interpreter row for the mints would
double-count. `U11`'s own row records where the cost lands: **"(inside the mint units)"**. That
statement is exact, and it is also the boundary of the accounting.

**The arena is a process and §8 says so.** §13 pins *"73 processes in the order 66 mints → 6
fidelity → 1 arena"*, and §8 Phase 1c prices its ro-cache load explicitly as *"66 mints **+ the
arena process itself**"*. But every arena-side unit is an internal operation timing — `U1`
(`0.0461 s`), `U2a`–`U2d`, `U3` (`0.126 s`), `U4`, `U5a`/`U5b`, `U6`, `U7` (`0.13 s`), `U8`
(`0.033 s`), `U10` — each two to five orders of magnitude below a python startup, so none of them
can contain one. I read all 24 rows of §8's table: **there is no startup row.**

**Measured, not inferred.** I timed an arena-class interpreter+import (numpy, torch, faiss, the
frozen `c01_policy_contrast_a0` and `mechfix_ops` modules) three times on this node:
**`1.91 / 1.84 / 1.84 s`**, against a bare interpreter at `0.01 s`. The document's own `U11` puts
the same quantity at `3.05–3.18 s`. So the unpriced cost is **`1.8–3.2 s`**, which is larger than
the ninth loop by roughly `15×`–`25×` and larger than seven of §8's priced rows.

**And one clause decides whether it is `1.8–3.2 s` or `13–22 s`.** §7.7 gives `U9` as
`3.70 / 3.49 s` per `(dataset, seed)` with **no timing boundary stated** — in the section whose own
institutionalised lesson is *"state the timing boundary, not just the number"*. If `U9` is a
full-process wall the six fidelity processes are priced and the gap is the arena alone; if it is an
internal timing, six more startups (`≈ 11–19 s`) are unpriced too. The evidence favours the former —
§7.7's own `U9` anecdote describes *"a crashed process's `echo` status … re-run correctly with
`--seeds 0`"*, which is a CLI invocation — but the document nowhere says it, and the count cannot be
derived from what is written.

**On severity.** The brief's Critical column admits *"any un-counted loop in §8"*, and I weighed it.
Rounds 7, 8, 9 and 10 found the fifth through eighth and ninth under the same language and graded
each **Important**, reasoning that a cost sitting inside a `× 1.25` margin (`732.7 s`) and `30 s` of
declared slack cannot misrepresent anything material. That reasoning holds here unchanged: at the
worst reading the total moves `2930.7 → 2953.0` and `3663.4 → 3691.3`, no heartbeat interval
changes, and no verdict quantity is touched. I also decline to grade it **Minor**: it is the largest
of the ten by a wide margin, and softening it because a GO is one finding away would be grading on
trajectory, which the brief forbids in both directions.

**Repair, three lines.** (1) Add one §8 row — *"1g interpreter + imports, non-mint processes"* —
counted as `1` (arena) or `7` (arena + the six fidelity processes) × `U11`, and re-multiply:
`2930.7 → 2933.9` / `2953.0`, `× 1.25 = 3667.4` / `3691.3`, with the mint share and Phase 3 share
recomputed. (2) State `U9`'s **timing boundary** in §7.7, one clause, so the count in (1) is
determined by the document rather than inferred from an anecdote. (3) Amend §7.2's *"already inside
every unit"* to name the scope it is true of — the mint units — so the sentence stops covering the
arena.

---

## MINOR (non-blocking; does not touch the verdict path)

* **M-1.** §7.2 (v11:1098) reads *"**No Phase 1e line is added**, because adding one would
  double-count"*, naming a phase label that §8 has used since v6 for something else: `**1e**` is
  `GATE-FOLD`'s banked-`.npz` parity re-read (v11:1313), added by round-5 I-3 *after* §7.2's
  sentence was written — the sentence is byte-stable from v4, where no §8 Phase 1e existed. §8's own
  note at v11:1347 then uses *"Phase 1e"* in the **new** sense (*"v5's `2927.5` plus Phase 1e"*), so
  the document uses one label in two senses two sections apart. Non-blocking: the claim §7.2 makes
  is substantively right (interpreter cost genuinely is inside the mint units and must not be
  double-counted there), Phase 1e's own row is correct and correctly priced at `0.1 s`, and no
  quantity moves either way. It is a stale cross-reference, the class round 8 graded Minor as its
  M-3. Repair: *"No separate interpreter line is added"*, or cite the label the row now carries.
  (This is one sentence away from I-1's third line and is naturally fixed with it.)

---

# REQUIRED RULINGS

## 1. §4.D — can any gate fire on a warranted CLOSE? **No, for all twenty. Re-derived from the gate texts and from measurement, not inherited.**

A *warranted CLOSE* is: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals are arm-outcome-independent by construction, and I measured eight of them
today.** `GATE-DET1` (thread env — and I hit its CPU-only guard myself: `import_compute_modules`
dies with *"CUDA is visible despite the CPU-only guard"*, so the guard is live). `GATE-SHA`
(37 digests over files no phase writes — all 37 recomputed). `GATE-FOLD` (banked parity flags +
`fold_of`). `GATE-POP` (populations, class counts, index-set identity, recomputed constants — I
re-derived `743/579`, `(297,446)/(180,399)`, `0.600269/0.689119/0.599462`). `GATE-NULLREMOVED` /
`GATE-ZEROMASK` (exact-zero row sets — I measured `{355}` / `{}`). `GATE-IDPARITY` (ids order and
labels — verified identical across native/ro/ro_ow on both datasets). `GATE-LEDGER` (process and
path counts). `GATE-FLOOR` votes **native** deployed keys against banked anchors — I read all six
triples out of the arena OUT JSONs and they match §6 on **both** metrics — and no battery arm enters
it. `GATE-C01PARITY`, `GATE-ROWSUBSET` and `GATE-RHORAW` are properties of the **raw two-block
build** and the frozen `ρ_raw` table, fixed before any head exists; I measured all three at
`0.000e+00`, `0.000e+00` and 26/26 at 6 dp. A `GATE-C01PARITY` failure is a builder defect ⇒ HALT,
never a CLOSE — the comparison runs in one process over the same arrays through the same `l2_rows`,
so it agrees bitwise by construction, which my own reconstruction confirms for the ninth time.

**The six per-lineage gates.** `GATE-ARENA`'s lower bound is on `endpoint_std` **only** — a control —
so real arms losing cannot entail it; the `≤ 0.98` upper bound catches a leak and cannot fire
downward, and a warranted CLOSE puts the real arms *low*. `GATE-ORBITDISP` fires on
`ρ_head > ρ* ∧ ρ_raw ≤ ρ*`; trained heads measure `0.34`–`0.67` against bars `0.968`/`0.977` —
**0/18 on both, my own measurement** — and nothing about a real arm losing raises `ρ_head`, which is
a direction-concentration statistic on the key matrix computed before any accuracy exists.
`GATE-NESTED` and `GATE-SELFTEST` are identities that hold for any arm set:
`net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is the definition of net fixes, not a performance
claim, so it can fail only on an implementation bug ⇒ HALT. `GATE-ALGEBRA` bounds the θ=0/θ=45
residuals — I measured the raw-key values at `8.941e-08` and `1.192e-07` / `8.941e-08`, and §7.8's
trained-head figures sit `7.5×`–`22.6×` inside the `2e-6` bar (I re-derived every headroom and every
similarity window: `√2048 = 45.2548`, windows `4.00e-06`–`1.21e-05`). **`GATE-DOMAIN` and
`GATE-DEVFID` carry no bar.**

**`GATE-ZEROOP` is the one gate that can HALT a correct run, and the design says so rather than
denying it.** Its firing is a numerical-tie event **uncorrelated with the arm outcome**: it compares
two *guard* arms against their algebraic counterparts, never a real arm against a control, so it
cannot fire *because* the CLOSE is warranted. It is one-directional (REPORT → HALT only), capped at
`⌊0.01 n_D⌋` = `7`/`5`, and a HALT is a non-publication, not a wrong verdict. I re-derived the three
alternative cap readings §6.5 rejects — `7/743 = 0.94 %`, `7/149 = 4.70 %`, `7/2229 = 0.31 %`, a
`15.0×` spread — confirming that pinning the aggregation was load-bearing.

**`GATE-ARMVIAB`'s retirement, verified at the source.** C01's raw `displacement` `0.8505`/`0.8846`
and `common_displacement` `0.8598`/`0.8590` clear the arena bars `0.6203`/`0.7091` by
`0.1499`–`0.2395`, and `GATE-FLOOR`'s native OOF `0.8884`/`0.8929` runs above C01's dev-arena
`0.8411`/`0.8590`. The escape branch was unreachable and the gate would have fired on the warranted
CLOSE. Retirement, not restriction, is right.

## 2. Verdict-path enumeration — mine, not inherited: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.**

Let `G` = all twelve globals pass; `P_N, P_R ∈ {passed, dropped}`, where *passed* means every
per-lineage gate cleared on **both** datasets (a failure on **any** dataset drops the lineage on
**both**, §5.6); and for each passed lineage `C_L ∈ {clears S1–S7 on both datasets, does not}`.

* **¬G** → §5.6's global bullet is categorical (*"Any failure HALTs the whole battery"*), §6's scope
  legend repeats it (*"**G** = global (HALT the battery)"*), and rule 3's own gloss names it
  (*"whenever a global gate fails"*) → **HALT**. (1 state.)
* **G, both passed**: some lineage clears → rule 1 **SURVIVE** (3); neither clears → rule 2
  **CLOSE** (1). (4 states.)
* **G, exactly one passed**: it clears → **SURVIVE**; it does not → rule 2's antecedent (*"both
  lineages passed"*) is false → rule 3 **HALT**. (4 states across two symmetric cases.)
* **G, both dropped**: rule 1 has no passed lineage, rule 2 false → rule 3 **HALT**. (1 state.)

Ten outcome classes, each mapped to exactly one published state. **Totality** holds because rule 3
is the catch-all. **Exclusivity** holds because rule 2 requires the negation of rule 1's antecedent
under *"both passed"*. I checked the one seam a hostile reader would try: whether rule 1 could fire
under `¬G`, since rule 1 speaks only of *"a lineage that passed its **per-lineage** gates"* and a
lineage can do that while a global fails. It cannot — the global bullet is categorical about the
**battery**, not a term in the combination, so a battery that has HALTed publishes HALT; and the
design states this in three places (§5.6's bullet, §6's scope legend, rule 3's gloss). The
**dataset axis adds nothing**: the drop rule collapses per-dataset outcomes into one lineage
pass/fail before combination, and S1–S7 are conjunctive across datasets. The **S1–S7 axis adds
nothing**: every failure pattern maps to ¬clears, and S5 (both real arms) and S7
(`common_displacement` only) do not vary with the witness `A`, which makes SURVIVE strictly
harder — the conservative direction, and stated as such. **The declared-drop exemption is the only
lawful path to an absent quantity**, correctly scoped, and I verified `NOT_TESTED, p = 1` is
non-rejecting by executing C01's own `holm_adjust` with `1.0` padding (witness still 24/24).
**No gate failure is reportable as a closure**: rule 2 requires all globals passed **and** both
lineages passed, and every gate failure routes to rule 3 or drops a lineage.

## 3. Rulings on §15's five open issues

1. **The subtraction technique, turned on this document.** Executed, and it is the strongest part of
   this round's evidence: **13/13 limbs faithful and complete**, verified verbatim *and* against
   their cited line ranges; six of seven Repair paragraphs subtract to connective punctuation; the
   seventh leaves a directive about v10's own file that was literally obeyed and is correctly not a
   limb. **All four deviations warranted**, ruled individually above. The protocol has now caught
   what it was built to catch and produced a version with nothing left in the residue.
2. **The item-scan mechanism.** Verified by my own exhaustive scan of all 107 occurrences: no
   genuine §13-item reference is unreachable, no non-§13 reference is swept in, and the printed
   scope is true of the pattern. **On the range-headings question I rule: keep expanding them.**
   Expanding *"items 1–18"*, *"items 19–22"*, *"items 23–26"* is what makes `unresolved: NONE`
   assert that every item the document's **own structural headings** claim to exist is actually
   defined — and that is precisely the check that would have caught round-6 C-1, where §13's body
   cited *"items 19–22 that did not exist"*. Excluding range headings would trade a real check for a
   prettier list. The cost is that the printed list is uninformative as a "which items are
   individually cited" signal, and §14.1 already discloses exactly that, including that the site
   count of `41` contains the transcript's own previews and the script's reproduction. No change
   needed.
3. **The ninth loop's placement, not its seconds.** The stated placement is **true of the code as it
   actually is** — verified line by line at `headspace_mint.py:306-307/322` and
   `headspace_arena.py:75/85/89/92-93` (V6) — and item 22's *"either placement is admissible, but
   the code and §8 must agree on one, and §8 is written for this one"* is the right instruction for
   the code lineage. §8 re-multiplies to `2930.7`. **The tenth is found — I-1 above**, and it is not
   a payload loop but the per-process cost the payload loops sit inside.
4. **§9's output root.** **No conflict, and I checked all three places.** §7's *"zero write into
   `data/`, `artifacts/` or `logging/`"* sits under the heading *"## 7. Dry-check — what was
   executed"* and is scoped by that heading and by §9's own parenthetical to the **dry check**,
   whose outputs went to the session scratchpad; `artifacts/c06_falsifier/` is written by the
   **run**, which has not happened. §11 declares no `artifacts/` output and all four new-code
   artifacts plus `artifacts/c06_falsifier/` are absent from the tree. §12 names no path. The only
   other `artifacts/` path in the document is `artifacts/c09_topo/…/scratch/mint_*.npz`, read-only,
   digest-free by declaration, read by no gate. Correct as written.
5. **Is the record now sound?** **Yes — and for the first time the residue is not about the record
   at all.** Every round from 6 to 10 found its blocker in §14: absence, then narrowing, then
   truncation. This round the record survives the sharpest instrument anyone has applied to it, and
   the one finding is in §8's enumeration, where nine previous rounds also found theirs. That is a
   different failure mode, and it is the one this campaign has never fully closed.

## 4. Process rules

**`rule_1_compute_projection`.** §8 re-multiplies exactly — I re-derived the printed column sum
(`2930.7`), the `× 1.25`, both minute figures, both shares, both sensitivity figures and every
explicit count, from the measured units in §7.7 rather than from the printed products. **Eleven
rounds, ten uncounted items: the tenth is interpreter+import for the non-mint processes (I-1).**
The cold-cache carrying convention is right and consistent — Phase 1e carries `0.1 s` against
`0.033 s`, Phase 1f `1.3 s` against `0.615 s`, the 7z rows `1.0`/`0.7 s` above their measured
maxima, Phase 7 at its `0.1 s` upper bound — with Phase 1d (`U7 = 0.13 s` at `0.1 s`) the only row
carried below its measurement, which is one-decimal rounding.

**`rule_2_heartbeat`.** **Nothing in v11 changes an interval.** Phase 1f rises `1.0 → 1.3 s` spread
over 90 materialisations already covered by the per-`(dataset, seed, lineage, fold)` line; Phase 2's
`GATE-FLOOR` row is `0.1 s`. The longest un-instrumented span remains one `GATE-C01PARITY` dataset
at `11.27 s` (`14.1 s` conservative), under the `~15 s` bound. I-1's repair would add `1.8–3.2 s`
once, at the arena's start, still far under the bound. §9 now states the progress-file path
(`artifacts/c06_falsifier/progress/C06_PROGRESS.txt`), names its creator and its creation point,
keeps the `buffering=1` per-phase handle and the unbuffered driver echo, and pins the denominator to
*"§8's frozen projected"* value, so it tracks the `2930.4 → 2930.7` move without a separate edit.

## 5. Freeze-readiness, in the operational sense

**Ready on everything except I-1, and I judged it as the document that will be hash-frozen and
executed.** All 37 sha256 recompute today; every constant is pinned and verified against
`configs/c01/c01_a0_v2.json` at the source. The run boundary is unambiguous for a context-free
operator: **one submission, SLURM CPU queue, 8 CPU / 32 GB, no `--gres`, no `--time`, no array, no
dependency, no requeue; 73 processes** in the stated order **66 mints → 6 fidelity → 1 arena**
(`66 + 6 + 1 = 73`, agreeing with §12's process-count predicate), `GATE-SHA` once in the driver
before any of them, `GATE-POP` before any population-consuming gate. Paths are now exact and
repo-relative throughout — round-10 M-3's `$BASE` is gone and the output root is declared.
Preconditions are checkable: `mints_present_before_arena == 66` is binding and separate from the
resume-safe `dev_path_opens == mints_executed`, and `GATE-FOLD`'s banked-flag re-read makes the fold
contract exactly predictable on fresh, resumed and partially-resumed runs — I confirmed at
`headspace_mint.py:192-194` that a resumed mint returns **before** the `:199` dev load, which is
what makes the resume-safe form necessary rather than merely convenient. Exit semantics are defined:
the HALT path names the failing gate in its final line, and a `RuntimeError` from the imported C01
algebra is caught, recorded `INSTRUMENT_INCONCLUSIVE` with `l2_rows`' `context` string, and written
to that line before exit. §8 is independently re-multiplied per F118 — measured units × explicit
counts, no extrapolation — with the single exception that is I-1.

## 6. Can the falsifier discharge the written condition at `$0`? **Yes.**

The instrument does exactly what the registry asks: re-run C01's real-displacement-versus-rotation
battery in the fold-head arena on already-banked caches, on CPU, with no extraction. I confirmed the
head-space arms are buildable and their dimensions forced (`4 × 1024-d` + `9 × 2048-d`, falling out
of the `4 × 7168` + `9 × 14336` I measured in raw space); the anchor reproduces and I read all six
triples out of the banked OUT JSONs myself; the arena is alive (`GATE-FLOOR`'s native OOF
accuracies run above C01's dev arena); and the decision rule reaches CLOSE, SURVIVE and HALT on
distinguishable states with no unmapped combination. The one live risk is named in the design rather
than by me: Head-N is an out-of-domain transplant and may miss `GATE-ARENA`'s lower bound, which
under §5.6 drops it and can turn a would-be closure into a HALT. §5.7 prices that risk honestly and
the bar is low — the transplanted `endpoint_std` needs a recovery fraction of
`0.02/(0.8884 − 0.6003) = 6.94 %`, which I re-derived. Neither my Important nor my Minor threatens
the `$0` character, the verdict path, or the `1.7–2.5 GPU-h` this falsifier exists to avoid
spending.

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Not applicable this round. For the record: a GO authorizes **nothing to run**. Before any job this
design still needs (1) a freeze with hashes, (2) a **separate** independent code/resource review
lineage over the executable reaching its own `0C/0H/0I`, and (3) main-dialogue authorization. A GO
is not authority to write `TARGET_STATE.json`.

---

# CLOSING

The most severe finding is **I-1**, and its severity is entirely in what it says about §8's
enumeration rather than in what it costs. The battery is 73 processes. §8 accounts for the
interpreter and import cost of 66 of them — correctly, inside the mint units, with §7.2 explaining
exactly why no separate row belongs there. The arena is the 67th, §8 Phase 1c names it as a process
in so many words when pricing its cache load, and its startup — which I measured at `1.84–1.91 s`,
against the document's own `U11` of `3.05–3.18 s` — appears in no row and inside no unit, because
every arena-side unit is an internal operation timing two to five orders of magnitude smaller. The
sentence that would otherwise catch this is §7.2's *"Interpreter and import cost is already inside
every unit"*, which is true of the mint units the section is about and not of the arena's; and that
same sentence still names *"Phase 1e"* for a row §8 has used since v6 for the fold-parity re-read,
which is the Minor. Whether the fidelity processes are also affected turns on one unstated clause:
`U9`'s timing boundary, in the section whose own institutionalised lesson is to state the boundary
and not just the number. At the worst reading the total moves `2930.7 → 2953.0`, no interval
changes, and no verdict quantity is touched — which is why this is Important and not Critical, on
exactly the reasoning rounds 7 through 10 applied to the five items before it.

Everything else in this round came back clean, and several things came back better than clean. The
record, which has been the blocker for five consecutive rounds, survives the sharpest instrument
anyone has turned on it: all thirteen limbs verbatim **and** complete, each inside the line range it
cites, six of seven Repair paragraphs subtracting to bare punctuation, and the seventh leaving a
directive about v10's own file that was literally obeyed — I checked v10's sha256. All four
disclosed deviations are warranted, and two of them are not concessions but corrections: v11's
count of **nine** previously unreachable item references is more accurate than round 10's **three**,
and I reproduced both numbers to confirm it, with v11 saying plainly which reviewer's figure it is
improving on. That is a document correcting its reviewer in the open, which is the opposite of the
failure mode five rounds were spent chasing.

And the science reproduces under independent measurement at every point I tested it — the thirteen
arms rebuilt from prose at `0.000e+00` on the first attempt, the row-subset bridge at `0.000e+00`,
all 26 `ρ` at six decimals, all 37 digests, all six floor triples on both metrics, all sixteen cells
of §1's table recomputed from stored confusion matrices, the Holm table cell for cell, the whole §8
column to `2930.7`, and twenty gates none of which can fire on the outcome this falsifier exists to
publish. That is now the fifth consecutive round for which that is true.

---

*Read-only review. No GPU, SLURM, Modal, arena run, mint, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json` was read and not modified. The draft, the configs
and all repository files are unmodified; my scripts, my counterfactual copy and all outputs live in
the session scratchpad only.*
