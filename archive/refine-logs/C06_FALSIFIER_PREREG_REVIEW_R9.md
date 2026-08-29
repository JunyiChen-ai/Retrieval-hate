# C06 `$0` falsifier — independent design review, **ROUND 9**

*Artifact:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V9.md` (unfrozen, sha256
`cdac0a63c7cf2b139a0c553370f722cd538ba0e0fe9163d782c455dcdd7bebbe`, 141693 bytes, 2021 lines).
*Reviewer:* fresh, independent of rounds 1–8 and of the designer.
*Compute used:* `sha256sum`; read-only numpy/torch-CPU re-derivations on banked **train-split**
caches, banked mint `.npz`, banked arena OUT JSON and `C01_A0_OUT.json`; execution of the audit
script **as reproduced in §14.2** and of two independent re-implementations of my own. No mint, no
arena run, no GPU, no SLURM, no Modal, no job, no cache write, no test-split open, no commit. The
draft, the configs and all repository files are unmodified. `TARGET_STATE.json` read only.
I declined the four permitted CPU mints: §7.8 is **byte-identical v8→v9**, round 7 measured it and
round 8 re-derived its arithmetic, and this round's obligation is the v8→v9 delta. I verified
§7.8's raw-space leg instead, from `C01_A0_OUT.json`, and say so under V7 and in the §5.2.3 audit.

---

# VERDICT

## **REVISE — 0C / 2H / 2I + 4M**

**The science layer is closed and I confirm it a third time, independently.** I rebuilt all
thirteen arms from §3.4's prose alone and got `max|diff| = 0.000e+00` against `prepare_views` on
**both** datasets **at the first attempt with nothing silently supplied**; every wrong reading I
could construct is caught (`1.878e-06` / `1.609e-06` un-normalised — **both under the `2e-6` v7
would have allowed**; `9.697e-01` / `9.558e-01` for the `common_interaction` misreading;
`9.701e-01` / `9.660e-01` for a flipped displacement sign). All **37** sha256 recompute. All **26**
`ρ_raw` values reproduce at 6 dp under the frozen float64 reduction — including the one value
round-7 I-1 was about, `orthrot_83p8` at `0.9568935731 → 0.956894` (float64) versus `0.9568933249 →
0.956893` (float32), both to the digit — and all 26 agree at 4 dp under both reductions. The
`GATE-ROWSUBSET` bridge is `0.000e+00` on all 13 arms. §8's product column re-multiplies to
`2929.6` exactly. The Holm counterexample table reproduces cell for cell under C01's own
`holm_adjust`. **No gate can fire on a warranted CLOSE, all twenty, re-derived.**

**Round-8 C-1 and C-2 are genuinely closed**, and I verified C-1 the only way that settles it: the
embedded transcript is **byte-identical** to my run of the §14.2 script against the finished
on-disk v9 (3588 bytes, both). §14.1 is a verified fixed point. §13.1 is edited for the first time
since v7 (`+1904`), items 15 and 19 carry what round 7 asked for and round 8 found missing, and my
own splitter reproduces **every** printed section delta exactly, including the `UNCHANGED: 43`
count and the true `§14.1` size (`2658 → 6143`) the transcript correctly declines to print.

**The record is not yet clean, and the failure has changed shape.** It is no longer *absence* — no
limb is unlanded, and I checked all 26 against round 8's prescription text rather than against the
limb table's paraphrase. It is **narrowing**: two limbs land a weaker repair than the one
prescribed, and the limb table's paraphrase is where the weakening happens. §7.9 still contains a
bolded sentence that contradicts §7.9's own parenthetical, §7.9's own sum, and §7.8 — the very
disagreement round-8 I-1 asked to end. And §14.1's self-exclusion, the repair for round-8 C-1,
**asserts rather than checks** that §14.1 diffed: I demonstrated by construction that a draft whose
§14.1 is byte-identical to its predecessor's still prints `CHANGED §14.1` and still rates all three
§14.1-citing rows and all five §14.1-citing limbs `OK`.

I have not graded on trajectory in either direction. Nine rounds is evidence of nothing; the four
findings below are the argument, and two of them I found by measurement rather than by reading.

---

# PART A — THE TWELVE §3 VERIFICATIONS

| # | claim | result |
|---|---|---|
| **V1** | 37 sha256 | **PASS — all 37 recompute, zero mismatches.** 7 imported modules + 6 read-for-definitions + 16 banked (6 arena OUT JSON, 10 `vsw_ckpt` npz) verified programmatically; the 8 input caches verified by `sha256sum` against `data/CLIP_Embedding/{HateMM,MHC_zh}/`. 7+6+16+8 = **37**, matching `U7`'s new "8 caches + 13 modules/configs + 16 banked". |
| **V2** | re-run the audit; transcript byte-identical; §14.1 never reports its own size | **PASS on both, with H-2.** I extracted the script from §14.2 and ran it: output is **byte-identical** to the embedded transcript (3588 chars each). §14.1's size is never printed; the line reads `(self, size not reported)`. Round-8 C-1's `2626`-vs-`2658` is gone and I confirm `2658` was the true v8 value. **But** the `CHANGED §14.1` line is unconditioned — see H-2. |
| **V3** | 26 limbs, each naming a diffed section; then check each against round 8's own prescription text | **PASS mechanically; 24 LANDED / 2 NARROWED / 0 NOT LANDED against the prescription text.** Every limb names a section my independent diff shows as changed or added. Against round 8's words rather than the table's paraphrase, I-1's first limb and I-2's first limb are narrowed (H-1, I-2). Full limb table in Part B. |
| **V4** | §13 items 19 and 15; §13.1 diffs `+1904` | **PASS.** §13.1 `7843 → 9747 = +1904` under my own splitter. Item 19 now carries two explicit bullets — endpoint pre-normalisation as a checked property in **both** instantiations, and `GATE-C01PARITY` asserted at `max\|diff\| == 0.0` "never at a tolerance". Item 15 reads *"because **it never *calls* `displacement_audit`**"* with the import/call distinction spelled out. Both are what round 7 prescribed and round 8 found unlanded. |
| **V5** | `GATE-SHA`'s widened scope; `U7` re-described and re-priced | **PASS.** §6's row now reads *"every frozen import, input cache **and the sixteen banked artifacts of §11**…"*. I grepped all **16** `GATE-SHA` sites: **none** still describes the narrower scope, and §11's closing sentence is now true of §6. `U7` = `0.12 s + 0.005 s = 0.13 s`; §8 Phase 1d's `1 × U7 = 0.1 s` is unchanged and correct at one decimal, exactly as round 8 said it needed to be. §12's ledger predicates are untouched by the widening — the 16 files are neither test nor `dev_seen` paths. |
| **V6** | §3.7's two blocks; item 5's verbs; `normalization_epsilon` registered | **PASS.** The table is split, the population preamble's *"the full list"* is gone and it now quantifies over *"this block"* only. §13 item 5 is scoped by its own opening clause (*"§3.7 now has two blocks with two different verbs"*) and both enumerations are **exhaustive against §3.7's two tables** — 7 population rows / 7 named, 4 config rows / 4 named. I verified `normalization_epsilon = 1e-12` lives under `c01_a0_v2.json::transforms` exactly as claimed, and confirmed the design's rationale at the source: `l2_rows:1202` divides by `norms`, never by `max(norms, ε)`, so outputs **are** exactly epsilon-independent unless `:1195-1199` fires — `GATE-C01PARITY` genuinely cannot pin it. |
| **V7** | §8 sums to `2929.6`; `× 1.25 = 3662.0`; the Phase 7z `GATE-ZEROOP` row; `85.6 %` / `9.3 %`; `3203.3` / `4024.4` | **PASS, every figure.** The 29-row printed product column sums to **`2929.6`** exactly; `2927.6 + 1.0 + 0.7 + 0.3 = 2929.6`; `× 1.25 = 3662.00`; `48.83 / 61.03 min`. Mint share `85.62 %`, Phase 3 share `9.34 %`, `2×` miss `3203.3 s = 53.4 min`, `5×` miss `4024.4 s = 67.1 min`. The `0.06 %` freeze cost is `(1.0+0.7)/2929.6 = 0.058 %` ✓. Every count re-derives: `(30×3)+(6×4)+(30×2)=174`, `4×60/9×60 = 240/540`, `23×2×2=92`, `256×3×2×2=3072`, `14×3×2×2=168`, `2+60=62`, `13×60=780`, `36+30=66`, `66+1=67`. No stale `2928.7`/`3660.9`/`3202.4`/`4023.5` survives anywhere. |
| **V8** | §7.9's mint count as a sum; the reported-versus-spent distinction | **PARTIAL — PASS on the sum, FAIL on the distinction as applied.** `7 + 1 + 4 + 0 + 0 = 12` ✓, the footer's *"twelve"* agrees, `≈ 21+5+2+1 = 29` wall-min and `85+25+6+3 = 119` CPU-min ✓, and v8's *"eleven"* is explicitly retired. But §7.9's own bolded headline still says *"**Five CPU head mints were trained in v7:** one for §7.8's `GATE-FLOOR` discharge"* while the parenthetical two lines below says the discharge mint was trained in **v6**, and the sum attributes it to v6. **H-1.** |
| **V9** | correct build `0.000e+00`; un-normalised `1.878e-06` / `1.609e-06`, both under `2e-6`; §6 states exactly one predicate | **PASS, measured myself.** Rebuilt from §3.4's prose alone — `0.000e+00` on both datasets, first attempt, nothing supplied. Un-normalised: **`1.878e-06`** (HateMM) / **`1.609e-06`** (MHC-ZH), both under `2e-6` with margin, so v7's row would indeed have passed a builder wrong by `10⁻¹` in head space. `common_interaction = l2(std ⊙ ow)`: `9.697e-01` / `9.558e-01`. Flipped displacement sign: `9.701e-01` / `9.660e-01`. §6's row states exactly one predicate, `max\|diff\| == 0.0`, and `2e-6` survives in v9 only as `GATE-ALGEBRA`'s bar and in the narrative explaining the strike. |
| **V10** | `ρ*`; all 26 `ρ_raw` at 6 dp under float64; trained-head `0/18` | **PASS, exactly.** Under §6.1's stated reduction (mean accumulated in `float64` over the `float32` keys) all 26 values reproduce at 6 dp, and all 26 agree at 4 dp under `float32` accumulation — so `GATE-RHORAW`'s 4-dp assertion is reduction-order-safe as claimed. `ρ*` `0.968176` / `0.977223` (`endpoint_std`), runners-up `0.964446` / `0.969686` (`common`). The 36 banked C09 mints: HateMM `0.447803 / 0.562434 / 0.632996`, ZH `0.340179 / 0.574247 / 0.667326`, **0/18 above `ρ*` on both**. The masked-zero-row shift is `1.3013e-03` ✓. |
| **V11** | the Holm counterexample table; S5's `n ≤ 12` | **PASS, cell for cell, under C01's own `holm_adjust`.** `m = 92`: 24/24, **23/24**, **0/24**. `m = 46`: 24/24 in all three rows. `displacement` disjunct 22/22 at `m = 92`. Padding `1.0` instead of `0.5` (the drop path's `NOT_TESTED, p = 1`) still gives 24/24, so the drop rule is non-rejecting as §5.5 requires. `92×2/2001 = 0.091954 > 0.05`; `46×2/2001 = 0.045977 ≤ 0.05`; `12/257 = 0.046693 ≤ 0.05` and `13/257 = 0.050584 > 0.05`, so `n ≤ 12`. |
| **V12** | 20 gate rows, `12 G / 6 L / 2 R`, matching §5.6; §13's 26 contiguous items | **PASS.** §6's table has exactly 20 rows; the scope column counts **12 G / 6 L / 2 R**; the G-set and L-set match §5.6's two lists **name for name** (set-symmetric-difference empty in both directions). §13.1 defines `**(1)**…**(26)**` contiguously with no gap and no repeat. |

**Ceremony floor.**

* **All 37 sha256 recompute**, measured today against the `2026-08-04` claim in §11's heading.
* **C01 constants verified against `configs/c01/c01_a0_v2.json`**: `normalization_epsilon 1e-12`,
  `tiny_displacement_epsilon 0.001`, `max_tiny_displacement_fraction 0.05`,
  `small_displacement_train_quantile 0.1`, `max_small_displacement_fix_fraction 0.5`,
  `minimum_gain_over_strongest_control 0.02`, `minimum_net_fixes {HateMM: 3, MHC_zh: 2}`,
  `n_bootstrap 2000`, `statistics.seed 20260728`, `holm_alpha 0.05`,
  `holm_metrics ['accuracy','macro_f1']`, `n_id_hash_permutations 256`, `permutation_hash sha256`,
  `bootstrap_lower_quantile 0.05`, `bootstrap_upper_quantile 0.95`, `gain_controls` = the five arms
  §5.1 names, `small_displacement_gate_reference =
  'strongest_ordinary_control_by_accuracy_then_macro_f1_then_frozen_gain_controls_order'` (D-1
  confirmed at the source), `small_displacement_endpoint_concat_role = 'diagnostic_only'`, and
  `required_halt_only_validity_guards` = **7 entries under `output.decision_schema`**, not under
  `decision` (round-6 M-4 correct). Every cited source line resolves: `:2036`, `:1955-1962`,
  `:1246-1265`, `:1272`, `:1296-1304`, `:1372-1377`, `:1379`, `:1769`, `:1775-1784`, `:2724`,
  `:1725`, `:1989-1996`, `:1381-1386`, `:387`, `:392-393`, `classifier.py:81-82/140-141/146`,
  `mechfix_ops.py:94`, `headspace_mint.py:192-194/199/203-216/209/321-325`.
* **Blindness grep across v1–v9**, every decimal in `[0.6, 0.99]`: 113 distinct across the corpus,
  **2 new in v9** — `0.619` (round 8's item-25 tail-record *timing in seconds*, quoted in §8's new
  spread paragraph) and `0.8462` (a **published C01 raw dev-arena** accuracy for ZH `orthrot_8p3` /
  `orthrot_29p1`, which I read out of `C01_A0_OUT.json` as `0.8461538461538461`). Neither is a
  battery-arm accuracy. **No arm accuracy has been computed, printed or recorded at any point in
  v1–v9** — §7.3's rescoped claim is true, and I verified it rather than inheriting it.
* **Test-set non-contact by construction.** `test_seen` occurs once in v9, in a negative assertion
  (*"the `test_seen` ro caches are opened by nothing"*); every other test mention is a prohibition.
  No §8 phase opens a `test_*` path. Every quantity I recomputed came from `train_*` caches, banked
  train-split mints, or banked OUT JSON.
* The four new-code artifacts (`c06_falsifier_mint.py`, `c06_falsifier_arena.py`,
  `configs/c06/c06_falsifier.json`, `c06_falsifier_cpu.sbatch`) are confirmed **absent** from the
  tree, as §11 states.

---

# PART B — MY OWN LIMB-LEVEL DISPOSITION AUDIT OF ROUND 8's FOURTEEN FINDINGS

**Method.** I split v8 and v9 with my own line-based splitter (a different construction from the
designer's regex-index one), diffed section by section, then checked each limb against **round 8's
prescription text quoted from `C06_FALSIFIER_PREREG_REVIEW_R8.md`** — never against the limb
table's paraphrase. Where a repair had a measurable consequence I measured it.

**My independent section diff agrees with §14.1's printed transcript in every line:**

| section | my delta | printed | | section | my delta | printed |
|---|---|---|---|---|---|---|
| §6 | `+343` | `+343` | | §13.1 | `+1904` | `+1904` |
| §8 | `+1005` | `+1005` | | §14.1 | `2658 → 6143` | `(self, size not reported)` |
| §14 | `+1053` | `+1053` | | §14.2 | `ADDED 6107` | `ADDED 6107` |
| §15 | `−91` | `−91` | | §5.2.2 | `+116` | `+116` |
| §3.7 | `+1146` | `+1146` | | §5.2.3 | `+214` | `+214` |
| §7.3 | `+0` | `+0` | | header | `−48` | `−48` |
| §7.7 | `+164` | `+164` | | UNCHANGED | 43 | 43 |
| §7.9 | `+840` | `+840` | | removed | none | — |

I also recomputed step (6)'s two sets independently: *changed-but-uncited* = `{14, 15}` and
*named-but-unchanged* = `∅`, both agreeing with the transcript.

**Result: 24 LANDED, 2 NARROWED, 0 NOT LANDED, 0 rebutted.**

| # | limb, as **round 8** prescribed it | verdict | evidence |
|---|---|---|---|
| **C-1**.1 | *"make §14.1 self-excluding: print `ADDED §14.1 (self, size not reported)` so no future edit can invalidate it"* | **LANDED** | script `SELF='14.1'`; transcript byte-identical to my run; true v8 size `2658` confirmed |
| **C-1**.2 | *"add one sentence saying which of the two conventions is in force"* | **LANDED** | §14.1:1762-1767 states it explicitly (but see H-2 for what the convention actually does) |
| **C-2**.1 | *"Extend item 19 or 23 with one clause on endpoint pre-normalisation"* | **LANDED** | §13.1 item 19, bullet 1, naming `std[m] := l2_rows(standard[m])` and `prepare_views:1296-1304`, with the `1.878e-06`/`1.609e-06` cost |
| **C-2**.2 | *"…and the bit-exact predicate"* | **LANDED** | §13.1 item 19, bullet 2: *"asserted at bit-exactness, `max\|diff\| == 0.0`, **never at a tolerance**"* |
| **C-2**.3 | *"fix item 15's *import*→*call*"* | **LANDED** | §13.1 item 15 now distinguishes importing the module from calling the function |
| **C-2**.4 | *"drop the blanket 'all 13 ADOPTED' from the header"* | **LANDED** | header is limb-level; *"no blanket adoption claim"*. The residual claim *"every one landed"* is what Part B exists to test, and it is over-stated for the two NARROWED limbs |
| *(C-2.5)* | *"add to §14.1 a step (5): for every repair with more than one prescribed limb, list the limbs and the section each landed in"* | **LANDED — not in the limb table** | §14.1 step (5) exists and runs. Round 8 prescribed **five** limbs for C-2; the table lists four. Nothing is concealed, but this is the table paraphrasing rather than quoting (**M-4**) |
| **H-1**.1 | *"One clause in §6's `GATE-SHA` row — '…and the sixteen banked artifacts of §11'"* | **LANDED** | §6:856; all 16 `GATE-SHA` sites checked, none narrower |
| **H-1**.2 | *"update `U7`'s description… Phase 1d's `0.1 s` needs no change"* | **LANDED** | §7.7:1147, `0.13 s`; Phase 1d unchanged at `0.1 s`, correct at one decimal |
| **H-2**.1 | *"Split §3.7's table into two blocks… or add a per-row provenance column"* | **LANDED** | two blocks with distinct preambles and verbs |
| **H-2**.2 | *"correct the preamble to quantify over the first block only"* | **LANDED** | *"Every constant in **this** block"*; v8's *"the full list"* removed |
| **H-2**.3 | *"rewrite item 5 to name the `<=` operator and the two tiny constants as read from the frozen config and asserted, not computed"* | **LANDED** | item 5(b) names all four, verb *"read from the sha-gated config and asserted equal to it"*; both enumerations exhaustive against §3.7 |
| **I-1**.1 | *"Decide which version trained the discharge mint, **make §7.8 and §7.9 agree**"* | **NARROWED** | the parenthetical states v6; the **bolded headline it was appended to still says v7** (§7.9:1222-1224 vs :1225-1229 vs :1242). §7.8 is byte-identical. **H-1** |
| **I-1**.2 | *"show the mint count as a sum so it is checkable"* | **LANDED** | `7+1+4+0+0 = 12`; footer agrees; v8's *"eleven"* explicitly retired |
| **I-2**.1 | *"One Phase 7z row: the mismatch scan… **and the tie-casualty evaluation priced as `≤ cap × cells × U_tie` with `U_tie` measured on one synthetic near-tie group**"* | **NARROWED** | the row exists (§8:1283) but carries **no measured unit** for the tie work, and its count `7×30+5×30=360` does not follow from §6.5's own aggregation. **I-2** |
| **I-2**.2 | *"a sentence in §13 item 10 requiring the worst-case-over-orderings computation to be analytic, not enumerative over `g!` orderings"* | **LANDED** | §13.1 item 10, in those words |
| **I-3**.1 | *"One row in §3.7's frozen C01 constants block: `normalization_epsilon = 1e-12`, source `c01_a0_v2.json::transforms`, consumed by every `l2_rows` call in both spaces"* | **LANDED** | §3.7:352, all three attributes present; value and location verified in the config |
| **I-3**.2 | *"one clause in §13 item 8 requiring the head-space builder to pass that value and no other"* | **LANDED** | §13.1 item 8, in those words, with the reason `GATE-C01PARITY` cannot detect a wrong epsilon |
| **I-4**.1 | *"Make the pattern emphasis-tolerant"* | **LANDED** | items 19 and 23 — the two round 8 showed were missed — now appear in the printed list |
| **I-4**.2 | *"include bare `§N` references"* | **LANDED** | printed separately, 12; I reproduce 12 |
| **I-4**.3 | *"print both counts with their scope named"* | **LANDED** | *"scope: in-document sections"* / *"scope: top-level sections"*; I verified the `qualified()` filter excludes exactly three references (`§15.4`, `§15.6` — round 5's review; `§4.4` — `GATE0_REOPEN`) and hides no in-document unresolved reference |
| **M-1** | *"`:2049` → `:2036`"* | **LANDED** | `:2036` is `small_mask = dev_min <= threshold` — verified at the source. (The parenthetical's description of `:2049` is off by one — **M-1** below) |
| **M-2** | *"attribute the counts to `C01_A0_OUT.json`"* | **LANDED** | and I verified from that file: HateMM `17.6/29.1/60.4/72.7` all `0.8504672897 < 0.8598130841`; ZH `8.3/29.1` both `0.8461538462 < 0.8589743590`. 4 of 6 and 2 of 6 exactly |
| **M-3** | *"the scope should read v1–v8"* | **LANDED** (as v1–v9) | and I verified the claim is **true** by my own corpus grep |
| **M-4** | *"balance §10.2's emphasis markers"* | **LANDED** | `**` count in the paragraph is even |
| **M-5** | *"attach `~600×` to the minimum norm, not to `tiny_ok`"* | **LANDED** | `0.6146/0.001 = 614.6`; HateMM train-text min `0.6146448850631714` read from `C01_A0_OUT.json` |
| **M-6** | *"the path should be one a reader can follow — or the script should be committed"* | **LANDED** | §14.2 reproduces it in full; I extracted and ran **that** copy and got the embedded transcript byte-for-byte |

---

# FINDINGS

## CRITICAL

**None.** No finding this round can publish a wrong verdict or block execution on the verdict path.
I state this plainly rather than by omission: I looked for one in the two places the last three
rounds found theirs — inside this round's repairs, and in §13 — and the two Highs below are the
strongest things I could construct.

---

## HIGH

### H-1. §7.9's bolded headline still assigns the `GATE-FLOOR` discharge mint to v7, contradicting the parenthetical appended directly beneath it, §7.9's own sum, and §7.8. Round-8 I-1's first limb is narrowed, and the limb table records it as landed.

*Attaches to:* §7.9 (v9:1222-1224, the headline; v9:1225-1229, the parenthetical; v9:1242, the
sum); §7.8 (v9:1167, byte-identical to v8); §14's I-1 row (v9:1684); the limb table (v9:1712).

Round 8's prescription was three words long and unambiguous: *"Decide which version trained the
discharge mint, **make §7.8 and §7.9 agree**, and show the mint count as a sum."* The second limb
did not land. §7.9 now reads, in this order:

> **Five CPU head mints were trained in v7:** one for §7.8's `GATE-FLOOR` discharge (`33.5 s`) and
> **four more** for its four-cell displacement-tail table…
>
> *(Round-8 I-1: … The discharge mint was trained in the **v6** drafting round and re-verified by
> round 6; the four tail mints were trained in the **v7** round. §7.8 describes when each
> measurement was first **reported**; this section accounts for when the compute was **spent**.
> Both are now stated so the two cannot be read as contradicting.)*

and then, eighteen lines later:

> mints `= 7 (…v1–v6) + 1 (**v6's** GATE-FLOOR discharge) + 4 (v7's tail cells) + 0 + 0 = ` **12**

The parenthetical announces a rule — §7.8 reports, §7.9 spends — and the sum applies it correctly.
The bolded sentence the parenthetical was appended to is a **spend** statement and applies it
backwards. So §7.9 now contradicts itself where v8 contradicted §7.8, and the sentence that does it
is the one a reader's eye lands on. The claim *"Both are now stated so the two cannot be read as
contradicting"* is false of the section that contains it.

I grade this High rather than Critical for three reasons and against two: the arithmetic that
matters is coherent and I re-derived every term of it (`7+1+4+0+0 = 12`, `≈21+5+2+1 = 29` wall-min,
`85+25+6+3 = 119` CPU-min, footer agrees), no gate, constant or verdict quantity depends on it, and
it cannot move the falsifier's `$0` character. Against: §7.9 is the section `rule_1` and the F118
erratum lesson bind by name, this is the **third consecutive version** in which it carries an
internal inconsistency, and the limb table asserts the repair landed — which is the honesty axis
the last three rounds all graded at the top. It is not Critical because the artifact *does* contain
the prescribed repair; it contains it alongside the sentence it was supposed to fix.

**Repair, one sentence.** Rewrite the headline as *"**Five CPU head mints are attributable to the
v6–v7 rounds:** one trained in v6 for §7.8's `GATE-FLOOR` discharge (`33.5 s`) and four trained in
v7 for its four-cell displacement-tail table"*, and drop *"Both are now stated so the two cannot be
read as contradicting"* — with the headline corrected there is nothing left to reconcile. Then check
the `≈ 5 wall-minutes / ≈ 25 CPU-minutes` attributed to v7 still holds with four mints rather than
five, or say what else is inside it.

### H-2. §14.1's self-exclusion **asserts** that §14.1 diffed instead of checking it, exempting 3 of 14 rows and 5 of 26 limbs from the only mechanical check the header cites — and printing an unconditioned `CHANGED` line inside a transcript labelled *"Output, verbatim"*.

*Attaches to:* §14.1's convention paragraph (v9:1762-1767); §14.2 (v9:1904); the transcript's
`CHANGED §14.1` line (v9:1798); §14 rows C-1, I-4, M-6; limb-table rows C-1×2, I-4×3.

§14.1 states the convention as: *"§14.1 is **self-excluding**: the audit never reports §14.1's own
byte count, **only that it changed**."* The script implements it at v9:1904:

```python
if SELF in SB and SELF in SA and SA[SELF]==SB[SELF]: changed.add(SELF)
```

That line fires **only when §14.1 is unchanged**, and it adds §14.1 to `touched` anyway. Combined
with the print branch above it, the effect is that §14.1 is reported and treated as changed
unconditionally.

**I demonstrated it rather than inferring it.** I constructed a counterfactual pair — a draft
identical to v9 except that its §14.1 is byte-identical to its predecessor's — and ran the §14.2
script on it. Output:

```
  CHANGED  §14.1     (self, size not reported)
  OK    C-1   cites §14.1
  OK    I-4   cites §14.1
  OK    C-1   make §14.1 self-excluding so the transcript is a fix -> §14.1
  OK    C-1   state which convention is in force                   -> §14.1
  OK    I-4   make the `§13 item N` pattern emphasis-tolerant      -> §14.1
  OK    I-4   include bare `§N` references                         -> §14.1
  OK    I-4   print both counts with their scope named             -> §14.1
```

Twelve of the fourteen rows and twenty-one of the twenty-six limbs correctly `FAIL` in that
counterfactual. The five limbs and two rows that cite §14.1 pass, and the transcript prints
`CHANGED §14.1` about a section that did not change.

Nothing in v9 is false as a result — §14.1 genuinely changed by `+3485` (I measured it), and I
verified all five §14.1-citing limbs by reading rather than by the script. The finding is that the
instrument round 8 built to end unverified repair claims contains a branch that **certifies a
repair claim without checking it**, in the one section the convention exempts, and that §14.1's own
step (3) — *"Flags any row whose cited section shows no diff"* — is stated as a universal and is
not one. A self-measuring section needs an explicit convention, which v9 supplies; it also needs the
convention not to be load-bearing for other rows' verification, which v9 does not.

**Repair, two lines.** Delete the `changed.add(SELF)` fallback, and print §14.1's status honestly
without its size:

```python
if SELF in SB:
    st = 'CHANGED' if (SELF in SA and SA[SELF]!=SB[SELF]) else ('ADDED' if SELF not in SA else 'UNCHANGED')
    print('  %-8s §%-8s (self, size not reported)'%(st,SELF))
```

so a row citing §14.1 fails exactly when §14.1 did not change, and the printed line is a
measurement rather than a convention. Then say in §14.1 that self-exclusion covers the **size
only**, never the changed/unchanged fact.

---

## IMPORTANT

### I-1. The eighth uncounted loop: the arena process's materialisation of the 60 per-cell head-key matrices from the banked mint `.npz` is priced in no §8 phase.

*Attaches to:* §8 Phase 1c (v9:1265), Phase 1e (v9:1267), Phase 2b (v9:1270); §13.1 item 22
(v9:1635-1637); §15 item 5.

§15 item 5 asks round 9 to hunt. Here it is, and it is a loop of the same kind and magnitude as the
seven before it.

The battery's 66 mint processes each forward the ro caches through their head **inside the mint
process** (§13 item 22) and write the resulting key matrices to `.npz`. The arena process then
rebuilds 13 arms and votes on **60 fold cells** — which requires materialising those key matrices
off disk. §8 prices:

* **Phase 1c** — *"**ro cache** loads, per process — 66 mints + the arena process itself"*,
  `67 × U8`, where `U8` is *"**ro cache** `torch.load`, 2 files"*. That is the arena loading its two
  `ro_L24` caches for the **raw** leg. It is not the mint `.npz`.
* **Phase 1e** — *"`GATE-FOLD`'s banked-`.npz` parity re-read, `66 × 0.5 ms`"*. That is a
  **metadata-only** read of `meta` and `fold_of`. I measured the same operation on banked mint
  `.npz` at **`0.2 ms`** per file, so `0.5 ms` is a fair conservative unit **for that object** — and
  it is not the object that materialises `K_*`.
* **Phase 2b** — head-space arm construction, `60 × U10`, where `U10` is *"head-space build of all
  13 arms, one cell"*, a build on in-memory arrays.

Nothing prices `np.load(...)['K_*']` in the arena. I measured it on the banked C09 mints (the
closest available analogue, `744 × 1024` matrices in `6.1 MB` files): metadata-only `0.0002 s` per
file against full key materialisation **`0.0055 s`** per file, i.e. **`≈ 0.33 s` for 60 cells** on
a warm cache — and C06's per-cell file will carry the `h_std` and `h_ow` key matrices rather than
one, so the realistic figure is `0.7–1.0 s`, with cold-cache reads higher.

That is the same order as the three Phase 7z rows v9 has just added (`1.0`, `0.7`, `0.3 s`), and
larger than Phase 1e, Phase 1d, Phase 1b and Phase 7 combined. It changes no conclusion — `2929.6 s`
carries a `× 1.25` margin and `30 s` of declared slack — but §8's discipline is *measured unit ×
explicit count against an explicit list*, and this loop is on no list.

**On severity.** The round-9 brief's Critical column admits *"any un-counted loop in §8"*, and I
considered it. I grade Important because rounds 7 and 8 found the fifth, sixth and seventh
uncounted loops under the same brief language and graded each Important, on the reasoning that a
sub-second loop against a projection with a 25 % margin cannot misrepresent anything material.
Grading the eighth differently would be grading on trajectory, which the brief forbids in both
directions.

**Repair.** One Phase 1f row: `60` arena-side cell loads (or `66`, if the full-fold mints are also
read), unit measured on one banked `.npz` with the key arrays materialised — state whether the
timed region includes `np.load` alone or `np.asarray` on the arrays too, per the timing-boundary
lesson §8 has just institutionalised — and re-multiply. `0.017 s` per cell (two matrices, warm)
gives `≈ 1.0 s` and a new total of `2930.6 s`.

### I-2. Round-8 I-2's Phase 7z row carries **no measured unit** for the tie-casualty work, and its worst-case count does not follow from `GATE-ZEROOP`'s own aggregation rule. The limb table's paraphrase is where the measurement requirement was dropped.

*Attaches to:* §8 Phase 7z, third row (v9:1283); §6.5's aggregation bullet (v9:1038-1043); §2's R1
(v9:114); the limb table (v9:1714); §14's I-2 row (v9:1685).

Round 8 prescribed, in full: *"One Phase 7z row: the mismatch scan (`120` vectorised comparisons,
sub-`0.1 s`) and **the tie-casualty evaluation priced as `≤ cap × cells × U_tie` with `U_tie`
measured on one synthetic near-tie group**."*

v9's row:

> `2 identities × 60` vectorised comparisons, plus tie work bounded by `cap × cells` =
> `7×30 + 5×30 = 360` items worst case | vectorised scan sub-`0.1 s`; tie work is **zero on a clean
> run** and capped before HALT | `0.3 s`

Two problems, one substantive and one arithmetic.

**No `U_tie` was measured.** The unit column contains a scan bound and two qualitative statements;
the product `0.3 s` follows from neither. §7.9 confirms nothing was measured: *"v9's measurements…
the round-8 C-1 byte-count reproduction, the `GATE-SHA` widening cost, and the audit re-runs."*
§2's R1 binds this document to *"measured-unit-cost × explicit-count"*, and this is the only row in
§8 that has an explicit count and no measured unit. It is also, by §6.5's own description, the one
loop whose cost is largest precisely when the run is going wrong — which is why round 8 asked for
a synthetic-group measurement rather than an estimate.

**The count does not follow from §6.5.** §6.5 fixes the aggregation: *"mismatches are counted per
`(dataset, seed, lineage)`, pooling the five folds' held-out items, so the denominator is `n_D` and
the cap is the `1 %` it is described as."* Under that rule there are `3 seeds × 2 lineages = 6`
capped cells per dataset, so the worst case is `7×6 + 5×6 = 72` items, not `7×30 + 5×30 = 360`. The
`30` is the fold-level cell count, which §6.5 explicitly pools away. The direction is conservative
— `5×` over, so the projection is safe — but a number that contradicts the gate's own aggregation
rule is exactly the class of defect round 5's H-3 was about, and §8 is where `rule_1` is
discharged.

**And this is the demonstrable instance of the limb table narrowing a prescription.** The limb
reads *"add a Phase 7z row for `GATE-ZEROOP`'s scan and tie-casualty work"* — a faithful summary of
round 8's first clause and a silent deletion of its second. §14.1 step (5) then verified the limb
against the paraphrase and passed it. Nothing in the machinery could have caught this; I caught it
by reading round 8's sentence.

**Repair.** Measure `U_tie` on one synthetic near-tie group of realistic size (a few milliseconds
of CPU, no mint), price the row as `≤ cap × cells × U_tie` with the cell count taken from §6.5's
aggregation (`6` per dataset, `12` total), state the timing boundary, and re-multiply. If the
designer prefers to keep the fold-level `30` as a deliberate over-count, say so in the row rather
than leaving `cap × cells` to be read against a `cells` §6.5 defines differently.

---

## MINOR (each non-blocking; none touches the verdict path)

* **M-1.** §5.2.2 (v9:483-484) now correctly cites `small_mask = dev_min <= threshold` at **`:2036`**
  — I verified it at the source — but the parenthetical explaining v8's error says `:2049` *"is
  `"registered_null_rows_excluded"` inside the tiny-fraction audit"*. It is not: `:2049` is
  `"source_rows": int(len(d_norm[split][modality])),` and `"registered_null_rows_excluded"` is at
  **`:2050`**. The error is inherited verbatim from round 8's own M-1 text. Non-blocking: the
  load-bearing correction is right and the operator is frozen correctly. Repair: `:2049` → `:2050`,
  or drop the description and keep the line number.
* **M-2.** §14.1's `§13 item N` scan prints `[5, 7, 8, 15, 19, 23, 25, 26]`. My own exhaustive scan
  finds one more genuine reference the pattern cannot reach: **`item 10`** in §14's own I-2 row
  (v9:1685, *"item 10 requires the worst-case-over-orderings computation to be analytic"*) — text
  v9 wrote **this round**. Round 8's prescribed regex was implemented to the letter and beyond, and
  `unresolved: NONE` remains true because item 10 is defined, so this is not an unlanded limb. It is
  the same shape as round-8 I-4 recurring on the round's own new text, and the `§13 item` line is
  the only one of the three reference lines that does **not** name its scope. Repair: name the
  scope on that line too (*"scope: references of the form `§13 item N` or `**item N**`"*), or widen
  the pattern to bare `item N` with the §5.9/§15 item references excluded by their own prefixes.
* **M-3.** §1's table — which §1 calls *"load-bearing twice over"* — prints `—` for `common` on
  MHC-ZH and for `endpoint_concat` on HateMM. Both exist in `C01_A0_OUT.json` and I read them out:
  ZH `common` = `0.8718 / +1`, HateMM `endpoint_concat` = `0.8598 / +2`. The convention appears to
  be *"show each dataset's selected strongest ordinary control"*, which is defensible but unstated,
  and a `—` in an accuracy table reads as *not measured*. Same class as round-8 M-2. Repair: add the
  two cells, or one clause saying what the dash means.
* **M-4.** The limb table lists **four** limbs for round-8 C-2; round 8 prescribed **five** (its
  fifth was *"Then add to §14.1 a step (5)…"*). The fifth landed, so nothing is concealed. It is
  recorded because it is direct evidence that the limb enumeration is the designer's paraphrase of
  the prescriptions rather than a transcription of them — the same mechanism that produced I-2
  above. Repair: quote each prescribed limb verbatim from the previous round's review, with its
  location in that review, so the enumeration is checkable against a source rather than trusted.

---

# REQUIRED RULINGS

## 1. §4.A — does the limb protocol close the gap, or move it? **It narrows it substantially and moves the residue, and the residue is now *narrowing* rather than *absence*.**

**What it closed, and I can measure it.** §13.1 — byte-identical v7→v8, cited by no v8 row, and the
location of every unlanded limb round 8 found — is now cited by **six** limbs and diffs `+1904`.
All four of round 8's partials are closed. Step (6)'s *named-but-unchanged* list is empty and its
*changed-but-uncited* list is `{14, 15}`, both of which I recomputed independently. A limb that
names no section, or names a section that did not change, can no longer ship: I verified that by
counterfactual, where 21 of 26 limbs correctly `FAIL`. That is real, and it is the check round 8
asked for.

**Three ways the residue survives, each demonstrated rather than asserted.**

1. **A limb can still pass by citing a section that diffed for another reason.** Three C-2 limbs
   cite §13.1, which diffed for five separate repairs; both H-2 limbs and one I-3 limb cite §3.7,
   which diffed for two; M-3's limb cites §7.3, which diffed by **`+0` characters** — a pure digit
   substitution that the check reads as a diff. All of them did in fact land, and I established
   that by reading, not by the audit.
2. **The limb table is a paraphrase, and the paraphrase is where prescriptions get narrowed.**
   I-2's limb dropped *"with `U_tie` measured on one synthetic near-tie group"*; I-1's limb kept
   *"make §7.8 and §7.9 agree"* but the artifact only added a parenthetical; and C-2's fifth limb is
   absent from the table entirely. **Both of this round's Highs are narrowings, and the machinery
   rated both `OK`.** A protocol that verifies limbs against its own restatement of the
   prescriptions cannot detect that the restatement is weaker than the prescription.
3. **The self-excluded section is exempt from its own audit** (H-2), and it carries three rows and
   five limbs.

**So: the gap moved from *"did an uncited section need to change?"* to *"is the limb table a
faithful transcription, and did the change do what was asked?"*** That is a strictly smaller gap —
the first question was invisible from inside the document, the second is answerable by any reader
holding the previous review — but it is the same gap in kind, and it will keep producing findings
until the enumeration is sourced rather than authored.

**Round 8's ruling stands and I confirm it from the other side of it:** an embedded self-audit is
necessary and not sufficient; the mechanism that actually caught both Highs was an independent
reader diffing v8→v9 against **round 8's prescription text**. **What should move from the script
into the review request:** it already has — this round's request made limb-level checking against
the prescription text an explicit obligation, and that obligation is what produced H-1, I-2 and
M-4. What should move into the *drafting* instruction is the converse: the limb table must **quote**
each prescribed limb verbatim with its location in the previous review, so a reader can verify the
enumeration is complete before verifying that each entry landed. Two of my four findings would have
been impossible to write if the table had quoted rather than summarised — because the designer
would have seen the deletion while making it.

## 2. §4.D — can any gate fire on a warranted CLOSE? **No, for all twenty. Re-derived from the gate texts and from measurement, not inherited.**

A *warranted CLOSE* is: the instrument is sound, and the real arms fail to beat the rotation family.

**The twelve globals** are arm-outcome-independent by construction. `GATE-DET1` (thread env),
`GATE-FOLD` (banked parity flags), `GATE-POP` (populations, class counts, recomputed constants),
`GATE-NULLREMOVED` / `GATE-ZEROMASK` (exact-zero row sets — I measured `{355}` / `{}`),
`GATE-IDPARITY` (`ids`/`labels` order), `GATE-LEDGER` (process and path counts) touch no arm score.
`GATE-FLOOR` votes **native** deployed keys against banked anchors — I read all six triples out of
the arena OUT JSONs and they match §6 to the digit on **both** metrics, and no battery arm enters
it. `GATE-C01PARITY`, `GATE-ROWSUBSET` and `GATE-RHORAW` are properties of the **raw two-block
build** and the frozen `ρ_raw` table, fixed before any head exists; I measured all three at
`0.000e+00`, `0.000e+00` and 26/26 at 4 dp under both reduction orders.

**The widened `GATE-SHA` specifically**, since it is this round's change. Its enlarged scope is 16
additional **banked, pre-existing, read-only** artifacts whose digests I recomputed today and which
all match §11. The gate's predicate is a digest comparison over files no phase of the battery
writes, so its firing is entailed by tampering or corruption, never by an arm outcome. The widening
therefore cannot fire on a warranted CLOSE, and it closes the provenance hole round 7 opened and
round 8 kept open: without it a conforming implementation would hash 21 files and leave
`GATE-FLOOR`'s six anchors and `GATE-FOLD`'s ten parity files unverified. It also does not interact
with §12's ledger — none of the 16 is a test path, a `dev_seen` path or a banked trainlog — and
Phase 1d's `0.1 s` still bounds `U7`'s `0.13 s` at one decimal.

**The six per-lineage gates.** `GATE-ARENA`'s lower bound is on `endpoint_std` **only** — a
control — so real arms losing cannot entail it; its `≤ 0.98` upper bound catches leaks and cannot
fire downward. `GATE-ORBITDISP` fires on `ρ_head > ρ* ∧ ρ_raw ≤ ρ*`; trained heads measure
`0.34`–`0.67` against bars of `0.968` / `0.977`, **0/18 on both datasets** (my measurement,
reproducing §6.1 to the digit), i.e. roughly half the bar, and nothing about a real arm losing moves
it. `GATE-NESTED` and `GATE-SELFTEST` are identities that hold for any arm set — `GATE-SELFTEST`'s
`net_s(A) = n_D · (acc_s(A) − acc_s(reference))` is arithmetic, not performance. `GATE-ZEROOP`
compares guard arms to their counterparts and is explicitly one-directional (REPORT → HALT only).
`GATE-ALGEBRA` bounds the θ=0/θ=45 identity residuals, measured `7.5×`–`22.6×` inside `2e-6` on
trained heads. **`GATE-DOMAIN` and `GATE-DEVFID` carry no bar.**

**Bit-exact `GATE-C01PARITY` and the false-HALT question.** The comparison is between the battery's
builder and `prepare_views` **in the same process, over the same arrays, through the same
`l2_rows`**, so both sides execute identical operations in identical order and agree bitwise by
construction. Seven independent reconstructions have now measured `0.000e+00`; mine is the seventh,
and it holds for the `n = 743` bridge as well. A failure is a HALT, never a CLOSE.

`GATE-ARMVIAB` remains correctly retired, and I verified its retirement argument at the source: C01's
raw `displacement` `0.8505` / `0.8846` and `common_displacement` `0.8598` / `0.8590` against arena
bars `0.6203` / `0.7091`, and `GATE-FLOOR`'s native OOF `0.8884` / `0.8929` above C01's dev-arena
`0.8411` / `0.8590`.

## 3. Verdict-path enumeration — mine, not inherited: **total, mutually exclusive, one lawful absence path, no gate failure reportable as a closure.**

Let `G` = all globals pass, `P_N, P_R ∈ {passed, dropped}` after the per-lineage gates (a failure on
**any** dataset drops the lineage on **both**, §5.6), and for each passed lineage `C_L ∈ {clears
S1–S7 on both datasets, does not}`, clearing being the disjunction over
`A ∈ {displacement, common_displacement}`.

* **¬G** → §5.6's global bullet is categorical (*"Any failure HALTs the whole battery"*) and rule 3
  names it explicitly → **HALT**.
* **both passed**: some lineage clears → rule 1 **SURVIVE**; neither clears → rule 2 **CLOSE**.
* **exactly one passed**: it clears → rule 1 **SURVIVE**; it does not → rule 2's antecedent
  (*"both lineages passed"*) is false → rule 3 **HALT**.
* **both dropped**: rule 1 has no passed lineage, rule 2 false → rule 3 **HALT**.

Rule 3 is the catch-all *"otherwise"*, so totality holds by construction; rules 1 and 2 are
exclusive because rule 2 requires the negation of rule 1's antecedent under *"both passed"*. The
dataset axis adds nothing: the drop rule collapses per-dataset outcomes into the lineage's
pass/fail before combination, and S1–S7 are required on **both** datasets conjunctively.

**The declared-drop exemption is the only lawful path to an absent quantity**, and it is scoped
correctly: a dropped lineage's quantities are `INSTRUMENT_FAILED`, excluded from S1–S7 and the S5
family, and enter the S4 family **only** as `NOT_TESTED` with `p = 1` — which I verified is
non-rejecting at any rank by executing C01's `holm_adjust` with `1.0` padding and getting the same
24/24 for a witness at the resolution floor. *"Absence by computation failure in a surviving lineage
still HALTs."* **No gate failure is reportable as a closure**: rule 2 requires both lineages passed
and all globals passed, and every gate failure routes to rule 3 or drops a lineage.

I also re-derived the resolution floor the drop rule depends on: `92 × 1/2001 = 0.045977 ≤ 0.05` and
`92 × 2/2001 = 0.091954 > 0.05`, so every witness comparator must sit at `p = 1/2001` — and note
this makes S4's separate *"bootstrap lower bound `> 0`"* leg non-binding rather than ambiguous:
zero adverse resamples forces `Δ_b > 0` for all `b`, hence a strictly positive `5 %` quantile. There
is no `>` versus `≥` seam.

## 4. Rulings on §15's six open issues

1. **The limb-level protocol.** Ruled in §4.A above: it closes the *absence* gap and moves the
   residue to *narrowing*. Adopt the "quote, don't paraphrase" change (M-4) and H-2's two-line
   script fix.
2. **`GATE-SHA`'s widened scope.** **Confirmed clean.** All 16 `GATE-SHA` sites checked — none
   describes the narrower scope; §11's closing assertion is now true of §6; §12's ledger is
   unaffected; `U7`'s `0.13 s` is consistent with Phase 1d's `0.1 s` at one decimal, and I verified
   the widening's own cost claim is of the right order (the 16 files are small and hash in
   milliseconds).
3. **§3.7's two blocks.** **Every consumer uses the right verb.** Population-derived: `n_D`
   (`GATE-SELFTEST`, S6, tie cap), class counts (`GATE-POP`), majority (`GATE-ARENA` lower bound),
   band, S7's quantile threshold, `GATE-DOMAIN`'s two majorities, tie cap — all computed. Config:
   `<=` (S7), the two tiny constants (§5.2.3), `normalization_epsilon` (every `l2_rows` call) — all
   read and asserted. `GATE-POP`'s row is **still correct** after the split because its quantifier is
   *"every **population-derived** constant in §3.7's table"*, which now picks out exactly the first
   block. §12's *"No selection anywhere"* paragraph also survives: the four config constants fall
   under its *"C01's frozen value"* disjunct, which is what round-8 I-3 required.
4. **The frozen timing figures.** **Freezing the conservative bound is right** — it bounds all
   three measurements of each loop, costs `0.058 %` of the total, and leaves every heartbeat
   interval untouched. On the second half of the question — should the loops be re-specified so the
   boundary is unambiguous — **yes, and v9 records the lesson without applying it**: §8 says *"state
   the timing boundary, not just the number"* and then states neither boundary for the two frozen
   rows. For the third 7z row there is no measurement at all (**I-2**). Repair: one clause per row
   naming what the timed region enclosed.
5. **The eighth uncounted loop.** **Found — I-1 above**: the arena's materialisation of the 60
   per-cell head-key matrices from the banked mint `.npz`, priced in no phase, measured at `≈ 0.33 s`
   for 60 single-matrix loads on a warm cache and realistically `0.7–1.0 s` for C06's two matrices
   per cell.
6. **Is the record now sound?** **Not yet, but the diagnosis has changed and that matters.** Round 8
   answered *no* and named §13 as the common cause of three of its four record findings; §13 is
   fixed, and I confirmed it limb by limb. Nothing this round is an unlanded repair. What remains is
   two narrowed limbs, one uncounted loop, and one self-exempting branch in the audit — four
   findings, none of which can move a verdict, and all four with repairs measured in lines rather
   than sections. The **science** is sound: I could not manufacture a CLOSE anywhere in the
   combination space, all twenty gates hold under the warranted-CLOSE test, the arm builder is
   pinned and I rebuilt it from the prose without help, and every number I checked reproduces.

## 5. Process rules

**`rule_1_compute_projection`.** §8 re-multiplies exactly — I re-derived the total, the `× 1.25`,
both minute figures, both shares, both sensitivity figures and every explicit count. **Nine rounds,
eight uncounted loops: the eighth is the arena's key materialisation (I-1).** On the timing spread,
ruled at §15 item 4: freeze the max, and state the boundary.

**`rule_2_heartbeat`.** **Nothing in v9 changes an interval.** Phase 7z rises from `1.1 s` to
`2.0 s`, `GATE-SHA` adds `5 ms`, and I-1's missing row would add `≈ 1 s` — all far under the
`~15 s` bound. The longest un-instrumented span remains one `GATE-C01PARITY` dataset at `11.27 s`
(`14.1 s` conservative). Per-cell lines cover Phase 2D's `38.4 s`, per-32-draw lines cover Phase
3's `273.7 s` at `≈ 2.9 s` each, and per-epoch lines cover the `40 s` mints. The `buffering=1`
per-phase handle and the unbuffered driver echo are unchanged and adequate.

## 6. Freeze-readiness

**Ready on everything except the four findings.** All 37 sha256 recompute today; all constants are
pinned and I verified each against `configs/c01/c01_a0_v2.json`; the run boundary is unambiguous
(one submission, 8 CPU / 32 GB, no `--gres`, `--time`, array, dependency or requeue; 73 processes
in a stated order with `GATE-SHA` before all of them and `GATE-POP` before any
population-consuming gate); §8 is independently re-multiplied per F118; the heartbeat is
line-buffered per phase with no interval above `~15 s`; the four new-code artifacts are absent from
the tree so the code lineage starts from zero. §13's 26 items are contiguous, actionable, and — for
the five v9 touched — correct: item 5 is coherent again with two verbs and two exhaustive
enumerations, item 8 carries the epsilon, item 10 carries the analytic requirement, item 15 carries
the import/call distinction and `tiny_ok`'s non-carriage, item 19 carries both properties round 7
asked for. **The only thing a code lineage still needs that §13 does not give it** is I-2's `U_tie`
question, which is a projection matter rather than a correctness one.

## 7. §5.2.3 — `tiny_ok` "not carried, with a measured warrant", audited hard

The round-9 request asks three questions about this section. My answers:

**Does "not carried, with a measured warrant" satisfy round-7 C-2's prescription?** **Yes.** C-2
required `tiny_ok`'s constants frozen and its disposition registered. Both constants are frozen
twice — in §5.2.3's own table and in §3.7's config block — with the correct source
(`c01_a0_v2.json::transforms`, which I verified). The disposition is registered as non-carriage,
with the direction disclosed (§5.9 item 9), the mechanism stated (`final_bool = tiny_ok and (not
require_… or not dominated)`, which I checked against `displacement_audit:2055-2058`), and a
run-time control attached (§13 item 25). A prescription to freeze and register does not require
carriage.

**Is the warrant's measurement sound?** **The raw leg is, and I re-verified it without a mint.**
`C01_A0_OUT.json` records `maximum_tiny_fraction = 0.0` with `tiny_epsilon = 0.001` on both
datasets, `tiny_count = 0` and `tiny_fraction = 0.0` in every modality of every split, and a
minimum displacement norm of `0.6146448850631714` on HateMM train text — so `0.6146 / 0.001 = 614.6`
and round-8 M-5's re-attachment of the `~600×` to the **minimum norm** rather than to `tiny_ok`
(which compares fractions) is correct. The head-space leg is four cells of sixty, measured in v7 and
verified by round 7; §7.8 is byte-identical since then and §5.2.3 says plainly that four cells
*"bound nothing formally"*.

**Does dropping `tiny_ok` open a wrong-verdict path — an S7 that passes where C01's full
`displacement_audit` would have failed?** **It opens one, and it opens it on the safe side.** Because
`tiny_ok` is a **conjunct**, dropping it can only make the composite easier: S7 easier ⇒ SURVIVE
easier ⇒ **CLOSE harder**, which is the conservative direction under §4 for the verdict this
falsifier exists to deliver. The state it admits is *"S7 passes with `frac(d_i ≤ 0.001) > 0.05` in
some cell"* — i.e. a SURVIVE reachable where C01's own stability check would have failed. Three
things bound that. It is a **SURVIVE** risk, and §10.3 makes a SURVIVE a licence only to *propose* a
bounded extraction under its own preregistration, review and authorization — no GPU is spent on the
strength of it. The measured head-space tail sits `18×`–`38×` above the epsilon with `frac = 0.0000`
in all four cells, so the state is not near. And **§13 item 25 makes it auditable in every cell**,
not extrapolated from four: `min_i d_i` and `frac(d_i ≤ 0.001)` are recorded per
`(dataset, seed, fold, lineage)` alongside the `GATE-ALGEBRA` residual. Item 25 *records* rather
than *gates*, which is the honest structure and is stated as such. **I would not reopen it either**,
and I reach that conclusion from the mechanism rather than by inheriting round 8's ruling.

## 8. Can the falsifier discharge the written condition at `$0`? **Yes.**

The instrument does exactly what the registry asks: re-run C01's real-displacement-versus-rotation
battery in the fold-head arena on already-banked caches, on CPU, with no extraction. The head-space
arms are buildable (`4 × 1024-d` + `9 × 2048-d`, an arithmetic identity I confirmed from the
one-block instantiation: 4 `fuse` arms and 9 `paired` arms), the anchor reproduces bit-exactly, the
arena is alive (`GATE-FLOOR`'s native OOF accuracies run above C01's dev arena), and the decision
rule reaches CLOSE, SURVIVE and HALT on distinguishable states. Nothing in my four findings
threatens the `$0` character, the verdict path, or the `1.7–2.5 GPU-h` this falsifier exists to
avoid spending. **The blocker remains the record, and it is now four line-level repairs deep rather
than a section deep.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Not applicable this round. For the record: a GO authorizes **nothing to run**. Before any job this
design still needs (1) a freeze with hashes, (2) a **separate** independent code/resource review
lineage over the executable reaching its own `0C/0H/0I`, and (3) main-dialogue authorization. A GO
is not authority to write `TARGET_STATE.json`.

---

# CLOSING

The most severe finding is **H-2**, and it is severe for what it says about the instrument rather
than for anything it breaks in v9. Round 8's C-1 was a transcript line that had gone stale; v9's
repair — self-exclusion — is the better of the two engineering options round 8 offered, and the
transcript is now a fixed point I verified byte-for-byte against my own execution of the script as
printed. But the implementation buys that fixed point by making §14.1's *changed* status an
assertion rather than a measurement, and I showed by construction that a draft whose §14.1 never
moved still prints `CHANGED §14.1` and still passes every row and limb that cites it. Three rows and
five limbs — including both C-1 limbs, which is to say the repair certifies itself — sit behind that
branch. Nothing in v9 is false as a result, because §14.1 genuinely changed and I checked all five
limbs by hand; the defect is that the document's central honesty mechanism contains, in the one
section exempt from it, precisely the pattern it was built to end. Two lines close it.

The rest is smaller than any previous round's. Every limb landed; two landed narrower than they were
asked to, and in both cases the narrowing happened in the limb table's *paraphrase* of round 8's
sentence rather than in the artifact — which is the one structural lesson worth carrying into v10:
**quote the prescription, do not summarise it**, and the enumeration becomes checkable before its
entries are. The eighth uncounted loop is real and sub-second. §7.9 needs one sentence rewritten.
And the science — the arms, the gates, the verdict paths, the multiplicity, the populations, the
digests, the constants — reproduces under independent measurement at every point I tested it, which
is now the third consecutive round that has been true.

---

*Read-only review. No GPU, SLURM, Modal, arena run, mint, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json` was read and not modified. The draft, the
configs and all repository files are unmodified; my scripts, my counterfactual copies and all
outputs live in the session scratchpad only.*
