# C06 `$0` falsifier — independent design review, **ROUND 8**

*Artifact:* `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V8.md` (unfrozen).
*Reviewer:* fresh, independent of rounds 1–7 and of the designer.
*Compute used:* `sha256sum`; read-only numpy/torch-CPU re-derivations on banked **train-split**
caches, banked mint `.npz` and banked arena OUT JSON; no mint, no arena run, no GPU, no SLURM, no
Modal, no job, no cache write, no test-split open, no commit. `TARGET_STATE.json` read only.
I declined the four permitted CPU mints — §7.8 is **byte-identical** to v7, round 7 measured it, and
this round's obligation is the v7→v8 delta; I verified §7.8's derived arithmetic instead and say so
under V7.

---

# VERDICT

## **REVISE — 2C / 2H / 4I + 6M**

The science layer is clean and I confirm it independently: rebuilding all thirteen arms from v8's
§3.4 prose alone gave `max|diff| = 0.000e+00` against `prepare_views` on **both** datasets at the
**first attempt with nothing silently supplied**, and every wrong reading I could construct from the
text is now caught by the tightened `GATE-C01PARITY`. Round-7 C-1 is genuinely closed. All 37 sha256
recompute, §8's product column re-multiplies to `2928.7` exactly, the Holm counterexample table
reproduces cell for cell under C01's own `holm_adjust`, and no gate can fire on a warranted CLOSE.

The record is not clean, and it fails in the same family as rounds 6 and 7 — the document asserting
more than it contains. **The §14.1 protocol did not prevent it, and I can show why structurally: the
audit verifies that a *cited* section diffed, never that an *uncited* repair limb landed.** Every
unlanded limb this round is in §13, the one section v8 never touched, and §13 is cited by no §14
row. The protocol also fails its own fidelity test: its printed transcript reports
`ADDED §14.1 2626 chars` where the script it cites prints `2658` — reproduced independently twice.

I have not graded on trajectory in either direction. Eight rounds is not evidence of anything; the
findings below are.

---

# PART A — THE TWELVE §3 VERIFICATIONS

| # | claim | result |
|---|---|---|
| **V1** | 37 sha256 | **PASS — all 37 recompute.** 7 imported modules + 6 read-for-definitions + 8 caches = 21 as before, **plus all sixteen** new banked digests (6 arena OUT JSON, 10 `vsw_ckpt` npz). Zero mismatches. |
| **V2** | re-run §14.1 independently | **FAIL on one line.** 15 of 16 section deltas and the `UNCHANGED: 41` count reproduce exactly under my own splitter; **`§14.1 2626` is wrong — the true value is `2658`** (C-1). All 13 disposition rows do cite a diffed section. Reference resolution is correct in conclusion but under-scans (I-4). |
| **V3** | round-7 C-1's measurement + one predicate | **PASS.** Correct build `0.000e+00` / `0.000e+00`; un-normalised build **`1.878e-06`** (HateMM) / **`1.609e-06`** (MHC-ZH) — **both under `2e-6`**, so v7's row would indeed have passed the wrong builder. §6's row now states exactly one predicate, `max\|diff\| == 0.0`. |
| **V4** | §5.2.3 exists; §3.7's three rows; §10.2 names datasets | **PASS.** §5.2.3 present with both constants (`0.001`, `0.05`) frozen and sourced; §3.7 carries all three new rows; §10.2's scope sentence now names the per-lineage dataset(s). (Semantics of the new rows: H-2. Emphasis markup: M-4.) |
| **V5** | ρ reduction order | **PASS, exactly.** HateMM `orthrot_83p8` = `0.9568933249 → 0.956893` (float32) and `0.9568935731 → 0.956894` (float64); §6.1 freezes float64 and prints `0.956894`. I recomputed **all 26** values under both reductions: every one agrees at 4 dp, and all 26 match §6.1's 6-dp table under float64. |
| **V6** | I-2's two loops; §8 totals | **PASS on the totals; the timing gap is real.** The printed product column sums to **`2928.7`** exactly and `× 1.25 = 3660.9`; `48.8`/`61.0 min`, the `85.6 %` mint share (85.65 %), the `9.3 %` Phase-3 share (9.35 %) and the 2×/5× miss figures (`3202.4`/`4023.5 s`) all re-derive. My own timings: 120 residual reductions **`0.128 s`** (close to round 7's `0.160`, far from v8's `1.0`); item-25 tail record over 60 cells **`0.619 s`** *including* materialising the norms (v8: `0.034`, round 7: `0.122`). Ruling under §15 item 5 below. |
| **V7** | the four-cell tail table | **NOT RE-MEASURED — declined by choice.** §7.8 is byte-identical to v7 and round 7 measured it. Its derived arithmetic is self-consistent: `2e-6 / 2.682e-07 = 7.46 → 7.5×`, `2e-6 / 8.848e-08 = 22.6×`, `2e-6 / 2.384e-07 = 8.4×`, `2e-6 / 1.183e-07 = 16.9×`; `0.018145`–`0.038435` is `18×`–`38×` above `0.001`. **The four cells remain four of sixty and no round has bounded the other 56.** |
| **V8** | `GATE-FLOOR`, `GATE-ROWSUBSET`, `ρ*`, `0/18` | **PASS, all four.** Banked anchors read from the six OUT JSONs: acc HateMM `0.8884/0.8858/0.8858`, ZH `0.8929/0.8895/0.8946`, mF1 HateMM s0 `0.8838`, `fold_acc_deployed[0] = 0.8725`. `GATE-ROWSUBSET` bridge re-measured: **`0.000e+00`** across all 13 arms. `ρ*` `0.968176` / `0.977223` with runners-up `0.964446` / `0.969686`. The 36 banked C09 mints: HateMM `0.447803 / 0.562434 / 0.632996`, ZH `0.340179 / 0.574247 / 0.667326`, **0/18 above `ρ*` on both**. |
| **V9** | D-1's two `fix_break` sites | **PASS.** `C01_A0_OUT.json` gives `net_fixes.reference = "common"` (HateMM) and `"endpoint_concat"` (MHC-ZH); `:1725` is the `endpoint_std`-referenced reporting field, `:2702-2714` the decision check. |
| **V10** | H-1's Holm counterexample | **PASS, exactly, under C01's own `holm_adjust`.** `m = 92`: 24/24, **23/24**, **0/24**. `m = 46`: 24/24 in all three rows. The `displacement` disjunct gives 22/22 at `m = 92`. `92×2/2001 = 0.091954 > 0.05`, `46×2/2001 = 0.045977 ≤ 0.05`. |
| **V11** | §3.7's population constants | **PASS on values, FAIL on the table's own semantics (H-2).** Arena `743`/`579`, `(297,446)`/`(180,399)`, majorities `0.600269 → 0.6003` / `0.689119 → 0.6891`, full-HateMM `0.599462 → 0.5995`, bands `[0.6203,0.98]`/`[0.7091,0.98]`, caps `7`/`5`, and every C01 constant (`0.1`, `0.5`, `0.001`, `0.05`, `0.02`, `2`/`3`, `20260728`, `2000`, `0.05`, `256`, the six angles) verified against `configs/c01/c01_a0_v2.json`. |
| **V12** | 20 gates, 12 G / 6 L / 2 R; 26 items | **PASS.** §6's table has exactly 20 rows and the scope column counts 12/6/2; §5.6's two lists match it name for name. §13 defines items **(1)…(26)** contiguously. |

**Ceremony floor.** Blindness grep across v1–v8 for every decimal in `[0.6, 0.99]`: 99 distinct in
v1–v7, 71 in v8, **10 new**. All ten (`0.6146`, `0.61464`, `0.62078`, `0.64734`, `0.65998`,
`0.67329`, `0.68613`, `0.73721`, `0.73769`, `0.80`) are raw- or head-space **displacement-norm
geometry** from §5.2.2's new dispersion paragraph and §7.8's max — **no battery-arm accuracy anywhere
in v1–v8.** I reproduced the raw dispersion figures exactly: HateMM `0.614645 / 0.647340 / 0.673298
/ 0.737694`, MHC-ZH `0.620789 / 0.659981 / 0.686138 / 0.737210`. Test-set non-contact holds by
construction — every quantity I recomputed came from `train_*` caches, and no phase in §8 opens a
`test_*` path.

---

# PART B — MY OWN DISPOSITION AUDIT OF §14's ROUND-7 BLOCK

Method: I diffed v7→v8 section by section with my own splitter, then checked each finding against
**round 7's prescribed repair**, not against the §14 row's self-description.

**Result: 9 VERIFIED ADOPTED, 4 PARTIAL, 0 NOT ADOPTED, 0 rebutted.** Against the header's
*"all 13 round-7 findings ADOPTED, 0 rebutted."*

| # | round-7 repair | my verdict | diff evidence |
|---|---|---|---|
| **C-1** | (1) state the pre-normalisation step in §3.4; (2) restate `GATE-C01PARITY` bit-exact and strike `2e-6`; (3) **extend §13 item 19 or 23** to name endpoint pre-normalisation and the bit-exact requirement | **PARTIAL** — (1) ✓ (§3.4 `+1505`, the `std[m] := l2_rows(standard[m])` block), (2) ✓ (§6 `+409`), **(3) not made — §13 and §13.1 are byte-identical v7→v8 (848 and 7843 chars each)** | §3.4, §6 changed; §13 UNCHANGED |
| **C-2** | write §5.2.3 with both constants, the `dominated`-only carriage, the measured warrant, the direction, plus §5.9 | **VERIFIED ADOPTED** — all six limbs present | §5.2.3 ADDED 2653; §3.7 `+332`; §5.9 `+297` (item 9) |
| **C-3** | extend §10.2's scope sentence or narrow §5.6, then correct both §14 rows | **VERIFIED ADOPTED** — §10.2 now names the dataset(s) per lineage, records the CLOSE-path degeneracy, keeps the requirement unconditional | §10.2 `+376` |
| **C-4** | freeze `<=` in §5.2.2, add the §3.7 row, add the dispersion sentence citing §7.8 | **VERIFIED ADOPTED** — and I reproduced round 6's raw dispersion numbers exactly | §5.2.2 `+1463`; §3.7 `+332` |
| **H-1** | rewrite the header over round 7's findings; restate the residual as the four-cell range | **VERIFIED ADOPTED** — header is *"What v8 changes"*, the retired `2.384e-07` is not quoted there | header `−326` |
| **H-2** | retitle §7.9 to this round's burn, state the extra mints, update the cumulative | **PARTIAL / adopted-with-a-new-defect** — §7.9 is rewritten and its arithmetic (21+5+2 = 28, 85+25+6 = 116) is coherent, but it now **contradicts §7.8** on which version trained the `GATE-FLOOR` mint, and *"eleven CPU head mints"* does not reconstruct (I-1) | §7.9 `+620`; §7.8 UNCHANGED |
| **I-1** | freeze the reduction order; note the earlier figure was float32 | **VERIFIED ADOPTED** — §6.1 freezes float64 and records both readings; the note is carried in §14's rounds-1–6 summary | §6.1 `+756` |
| **I-2** | name the two loops and re-multiply | **VERIFIED ADOPTED** — Phase 7z, two rows, total `2928.7`/`3660.9`, which re-multiplies exactly | §8 `+585` |
| **I-3** | add the sixteen digests; write the `vsw_ckpt` path with its prefix | **PARTIAL** — all sixteen digests present and correct, path prefixed; but `GATE-SHA`'s scope in §6 was not widened and §11 asserts that it was (H-1) | §11 `+2605`; §6's `GATE-SHA` row UNCHANGED |
| **M-1** | correct *"does not import"* → *"does not call"*, **in §5.2.2 and §13 item 15** | **PARTIAL** — §5.2.2 corrected; **§13 item 15 still reads *"because it does not import `displacement_audit`"*** | §5.2.2 changed; §13 UNCHANGED |
| **M-2** | swap the item 19 / item 23 attribution | **VERIFIED ADOPTED** | §3.4 |
| **M-3** | ranking at `:1955-1962`, guards at `:1940-1948` | **VERIFIED ADOPTED** — I checked the source: `def` at 1940, guard `die()`s through 1954, `return max(` at 1955 closing at 1962 | §5.2.1 `+49` |
| **M-4** | drop one of the two superlatives | **VERIFIED ADOPTED** | §5.9 |

**Does §14.1's printed audit match my independent run?** **No, in one place.** Section deltas,
touched-section set, `UNCHANGED: 41`, and all 13 row citations reproduce. `ADDED §14.1 2626 chars`
does not: the true figure is `2658`, and I obtained it twice — once by executing the designer's own
`v8_audit.py`, once from a splitter I wrote without reference to theirs.

---

# FINDINGS

## CRITICAL

### C-1. §14.1's transcript is labelled *"Output, verbatim"* and *"run against the finished artifact"*, and one of its sixteen lines is neither.

*Attaches to:* §14.1 (v8:1659-1724), the block at v8:1697.

The embedded output reports:

```
  ADDED    §14.1     2626 chars
```

Executing the cited script against the finished v8 prints `2658`. My own independent
re-implementation prints `2658`. Fifteen of the sixteen section lines, the touched-section set and
`UNCHANGED: 41` all reproduce exactly; this line does not.

The cause is self-reference: the transcript was captured before §14.1's text was final, and pasting
it changed the length of the very section it measures. That makes §14.1's two framing sentences
false as written — *"the audit below is run against the finished artifact"* and *"Output,
verbatim"* — in the one section whose entire value is that its assertions can be trusted without
re-execution. Two rounds raised **exactly this class** to Critical, the protocol exists to end it,
and it recurs inside the protocol's own transcript on its first outing.

Nothing downstream moves: the number is §14.1's own byte count, it enters no gate, no constant and
no verdict. It is Critical because the brief for this round binds it as such and because the
document's own bar covers a false factual claim in the record.

**Repair, four characters, and it converges.** Replace `2626` with `2658`. Because the digit count
is unchanged, §14.1's length stays `2658` and the transcript becomes a fixed point — I checked. The
alternative, and the better engineering, is to make §14.1 self-excluding: print
`ADDED §14.1 (self, size not reported)` so no future edit can invalidate it. Either way, add one
sentence saying which of the two conventions is in force, so round 9 does not have to rediscover
that a self-measuring section cannot be verbatim by accident.

### C-2. The header claims *"all 13 round-7 findings ADOPTED, 0 rebutted"*; two are partial, both unlanded in §13, and §14.1 cannot see either.

*Attaches to:* header (v8:11-12); §14's round-7 block header (v8:1615); §13.1 items 15, 19, 23
(v8:1546-1587); §14.1 (v8:1659-1724).

**§13 and §13.1 are byte-identical v7→v8** — 848 and 7843 characters in both files. Two round-7
repairs had a §13 limb and neither landed:

* **C-1 limb 3.** Round 7's repair was three numbered lines; the third was *"Extend §13 item 19 (or
  23) to name endpoint pre-normalisation as a checked property, and to state that
  `GATE-C01PARITY` must be asserted at bit-exactness because a `2e-6` tolerance admits a builder
  that is wrong by `10⁻¹` in head space."* Item 19 still carries only the one-construction claim;
  item 23 still names only `common_interaction` and the `9.697e-01` cost. Neither mentions endpoint
  pre-normalisation, and neither states the predicate. The section that exists to tell the code
  lineage what to check does not carry the property this round's Critical was about.
* **M-1 limb 2.** Round 7: *"Same for §13 item 15's 'none of which the battery inherits'."* §5.2.2
  was corrected to *"does not **call**"*; **§13 item 15 still reads *"because it does not import
  `displacement_audit`"*** — a statement a code lineage checking §11 will find false, since §11
  imports `c01_policy_contrast_a0.py`, which contains that function.

Neither omission can invert a verdict. C-1's science is closed by other means — item 23's
*"test against `prepare_views`, do not reimplement"* plus §6's bit-exact predicate is sufficient,
and I demonstrated it — and M-1 is a wording error. The Critical is the **claim**: the header and
§14's block header both assert unqualified adoption, and this document has machinery for saying
otherwise (§5.5 records round 5's 46-family prescription as *"a partial rebuttal"*) which was not
used. Round 7 graded v7 on precisely this basis, and grading v8 differently would be grading on
trajectory.

**The structural point, and it is the answer to §4.A.** §14.1 rates C-1 `OK` because C-1 cites §3.4
and §6 and both diffed. It has no way to notice a limb the row does not claim, and it never asks
whether §13 — cited by **no** §14 row — should have changed. The audit's blind spot and the
document's failure sit in the same place.

**Repair.** Extend item 19 or 23 with one clause on endpoint pre-normalisation and the bit-exact
predicate; fix item 15's *import*→*call*; and either mark C-1 and M-1 `PARTIAL` in §14 or drop the
blanket *"all 13 ADOPTED"* from the header. Then add to §14.1 a step (5): **for every round-7 repair
with more than one prescribed limb, list the limbs and the section each landed in** — that is the
check the current four steps cannot perform.

---

## HIGH

### H-1. §11 states that §6 gives `GATE-SHA` a scope covering the sixteen new digests. §6 does not, and round-7 I-3's provenance hole is only half closed.

*Attaches to:* §11 (v8:1430-1433); §6 `GATE-SHA` (v8:840); §7.7 `U7` (v8:1131).

§11's new closing paragraph reads:

> …and `GATE-SHA`'s scope is stated in §6 as the frozen imports and the input caches **plus the
> sixteen banked artifacts above**.

§6's `GATE-SHA` row, in full and unchanged from v7:

> `GATE-SHA` | G | every frozen import and input cache matches §11; **once in the sbatch driver**

There is no third clause. I grepped every `GATE-SHA` occurrence in v8 — nine sites — and no other
one widens the scope either. So §11 asserts a fact about §6 that §6 does not contain, and the gate
that must check the sixteen digests still describes itself as covering *"every frozen import and
input cache"*, which they are not.

The consequence is the exact hole round 7 identified: `GATE-FLOOR`'s six anchor triples are read out
of the arena OUT files and `GATE-FOLD`'s parity is asserted against the `vsw_ckpt` npz
(`headspace_mint.py:209-216`), and a code lineage implementing §6's table verbatim would hash 21
files and leave both unverified. `GATE-FLOOR` is the design's only anchor and a global HALT; §7.3's
blindness argument and §5.6's whole structure rest on it.

Cost is not the obstacle: I measured the sixteen files at **1.2 MB total, hashing in 5 ms**, against
`U7`'s `0.12 s`. The repair is free.

**Repair.** One clause in §6's `GATE-SHA` row — *"…and the sixteen banked artifacts of §11"* — and
update `U7`'s description in §7.7, which still reads *"over 8 caches + 6 modules"* (14 files) while
§11 now lists 37. Phase 1d's `0.1 s` needs no change.

### H-2. v8's three new §3.7 rows are C01 config constants, and they falsify §3.7's own preamble and make §13 item 5 incoherent.

*Attaches to:* §3.7 preamble (v8:324-326) and table rows (v8:337-339); §13.1 item 5 (v8:1517-1520);
§6 `GATE-POP` (v8:843).

§3.7's table is introduced as:

> **Population-derived constants — the full list.** … **All** are frozen here, **all computed on
> the arena population**, and **all are checked at run time rather than read** (§13 item 5)

v8 added three rows under that preamble: the **`<=` small-set comparison operator**,
**`tiny_displacement_epsilon = 0.001`** and **`max_tiny_displacement_fraction = 0.05`**. None is
population-derived. All three are `c01_a0_v2.json::transforms` values — the table says so itself in
the source column — and all three **must** be read, not recomputed. The preamble is now false of
three of its own rows.

The handoff consequence is sharper, because §13 was not touched. Item 5 reads:

> **(5)** **Every** population-derived constant in §3.7's table is **computed from the arena, not
> read** — the arena size, class counts, majority, `GATE-ARENA` band, `GATE-DOMAIN`'s two
> majorities, the tie cap, S7's quantile threshold **and S7's `<=` small-set comparison operator**

It **explicitly enumerates the `<=` operator among the things to compute from the arena**. That is
not a thing that can be computed from anything, and the code lineage — for which §13 is the sole
input — is handed an impossible checklist item. The two tiny constants are in the table but in
neither item 5's enumeration nor anywhere else in §13.

§6's `GATE-POP` row is written correctly (*"every **population-derived** constant in §3.7's table"*
restricts to the right subset), so I do **not** find that a conforming implementation halts on a
warranted CLOSE — the qualifier saves it. But that safety depends on a reader preferring the gate
row's grammar over §3.7's preamble and §13 item 5's explicit list, both of which say otherwise.

This is a seam v8's own C-2 and C-4 repairs opened, and it is the second finding this round whose
whole content is that §13 was never revisited.

**Repair.** Split §3.7's table into two blocks — *population-derived (recomputed)* and *frozen C01
constants (read and asserted equal to the config)* — or add a per-row provenance column; correct the
preamble to quantify over the first block only; and rewrite item 5 to name the `<=` operator and the
two tiny constants as **read from the frozen config and asserted**, not computed.

---

## IMPORTANT

### I-1. §7.9's repair contradicts §7.8 on which version trained the `GATE-FLOOR` mint, and the cumulative mint count does not reconstruct.

*Attaches to:* §7.9 (v8:1206-1221); §7.8 (v8:1148-1173, unchanged); §7.2 (v8:1072); the footer
(v8:1751-1755).

§7.9, new this round: *"**Five CPU head mints were trained in v7, not one:** one for §7.8's
`GATE-FLOOR` discharge (`33.5 s`) and **four more** for its four-cell displacement-tail table."*

§7.8, byte-identical to v7: *"**v6** discharges it by measurement rather than by declaration. One
HateMM `s0 / fold 0` Head-N head was minted…"* and then *"**v7** measures the tail directly, on four
freshly minted `seed 0 / fold 0` heads."*

The two sections assign the same mint to different versions. And the headline does not add up:
§7.2 records *"Fold parity passed in all seven mints"*, §7.9 adds five, and the footer states
**eleven** — `7 + 5 = 12`. Whichever number is right, the document supplies two.

Round 7 raised the stale version of this paragraph to High because §7.9 is the compute-accounting
section that `rule_1` and the F118 lesson bind. The repair improved it and left a contradiction, so
I grade it Important rather than High: §7.9 is now accurate about **this** round, its wall/CPU
arithmetic is coherent (`21+5+2 = 28`, `85+25+6 = 116`), and no gate or verdict depends on it.

**Repair.** Decide which version trained the discharge mint, make §7.8 and §7.9 agree, and show the
mint count as a sum (`§7.2's N + v7's five + v8's zero = …`) so it is checkable rather than asserted.

### I-2. The seventh uncounted loop: `GATE-ZEROOP`'s mismatch detection and tie-casualty evaluation are priced in no phase.

*Attaches to:* §8 Phase 2z and Phase 7 (v8:1245, 1257); §6.5 (v8:1013-1034); §13.1 item 10.

§15 item 5 asks round 8 to hunt, so I hunted. Phase 2z prices `GATE-ZEROOP`'s **guard arms** — 120
votes and 60 partial builds, `2.5 s`. Phase 7's list is explicit and names `GATE-SELFTEST`,
`GATE-NESTED`, S7, `GATE-POP`, `GATE-NULLREMOVED`, `GATE-IDPARITY`, `GATE-ZEROMASK`, `GATE-FOLD`'s
in-process leg and `mints_present_before_arena`. **`GATE-ZEROOP` appears in neither**, yet §6.5
specifies a substantial per-item computation for it:

> an item is a **tie casualty** iff recomputing its rank-weighted vote leaves the two arms'
> predictions equal under the **worst case over all orderings** of every near-tie group

over the **union of the two arms' top-21 sets**, for two compared identities across 60 cells. The
prediction-equality comparison itself is vectorised and genuinely sub-`0.1 s`; the tie-casualty
recomputation is not, and its count is data-dependent — zero on a clean run, up to the `1 %` cap per
`(dataset, seed)` before HALT, and in the worst case bounded only by `n_D × 60`.

Zero expected cost is exactly why six rounds missed it, but §8's discipline is measured-unit ×
explicit-count against an explicit list, and this loop has neither. It is also the one loop whose
cost is largest precisely when the run is going wrong.

**Repair.** One Phase 7z row: the mismatch scan (`120` vectorised comparisons, sub-`0.1 s`) and the
tie-casualty evaluation priced as `≤ cap × cells × U_tie` with `U_tie` measured on one synthetic
near-tie group, plus a sentence in §13 item 10 requiring the worst-case-over-orderings computation to
be **analytic, not enumerative over `g!` orderings**.

### I-3. `normalization_epsilon` is registered nowhere in v8, and no gate pins it.

*Attaches to:* §3.4 (v8:189-252); §3.7's table; §12's *"no selection anywhere"* paragraph
(v8:1476-1478); §13.1.

`1e-12` and `normalization_epsilon` occur **zero times** in v8 — I grepped. Yet every call in §3.4's
builder is `C01's l2_rows`, whose signature is `l2_rows(array, epsilon, context, zero_mask)`, and
`l2_rows:1195-1199` **dies** on any row whose norm is at or below it. The two-block build inherits
the value because it goes through `prepare_views`, which reads the config; **the head-space build
does not go through `prepare_views` and has no config to read it from**, and §13's 26 items never
name it.

`GATE-C01PARITY` does not pin it either: it compares outputs, and the outputs are
epsilon-independent unless a row dies, so a builder passing a different epsilon still measures
`0.000e+00`. I verified this is not idle — §7.8's own head-space `min d_i` is `0.018`, only `18×`
above `tiny_displacement_epsilon = 0.001`, which v8 has now placed in §3.7's table one row away and
which is literally named *"epsilon"*. Head-space `common_interaction` blocks, being Hadamard
products of unit vectors in 1024-d, sit lower still.

This cannot invert a verdict — the failure mode is fail-closed `die()` → `INSTRUMENT_INCONCLUSIVE`,
which §5.9 item 5 already establishes as the harmless direction. It is Important because §12 asserts
*"Every threshold in §5 and §6 is C01's frozen value, C09's banked constant, a population-derived
constant frozen in §3.7 …, or a declared engineering choice disclosed in §5.9"*, and this one is in
no category, and because a document whose purpose is a code-lineage handoff must state the constant
the handoff needs. Seven rounds have not raised it.

**Repair.** One row in §3.7's *frozen C01 constants* block (per H-2's split):
`normalization_epsilon = 1e-12`, source `c01_a0_v2.json::transforms`, consumed by every `l2_rows`
call in both spaces; and one clause in §13 item 8 requiring the head-space builder to pass that
value and no other.

### I-4. §14.1's reference-resolution step misses the two references v8's own M-2 repair created.

*Attaches to:* §14.1 step (4) (v8:1713); §3.4 (v8:233-235).

The transcript prints `§13 item refs: [5, 7, 25, 26]`. The document contains at least five distinct
`§13` item references: **`§13 **item 23**`** at v8:233 and a bare **`**item 19**`** at v8:234 are
both missed, because the script's pattern is the literal `§13 item (\d+)` and markdown emphasis
breaks it. My markdown-tolerant scan returns `[5, 7, 23, 25, 26]` plus the bare item 19.

Its **conclusion** survives — items 19 and 23 are both defined, so `unresolved: NONE` is true — but
the printed enumeration is a false statement about the document, and the two references it misses
are precisely the text **round-7 M-2 asked v8 to write**. The step that is supposed to prove the
round's edits resolve does not scan the round's edits.

The same step also restricts to dotted `§N.N` forms: 30 distinct, against 42 when bare `§N`
references are included. Nothing is unresolved either way, but the printed `30` is a scope figure,
not a coverage figure, and the transcript does not say so.

**Repair.** Make the pattern emphasis-tolerant (`§13\s*(?:\*\*)?\s*items?\s*(?:\*\*)?\s*(\d+)`),
include bare `§N` references, and print both counts with their scope named.

---

## MINOR (each non-blocking; none touches the verdict path)

* **M-1.** §5.2.2 cites *"`small_mask = dev_min <= threshold` (`:2049`)"*. The site is **`:2036`**;
  `:2049` is `"registered_null_rows_excluded"` inside the tiny-fraction audit. The operator and the
  freeze are right; the line number is not. Inherited from round 6's citation, but v8 states it as
  fact. (§5.2.3's `displacement_audit:2047-2076` **is** correct — `tiny_ok` is at `:2054-2057`.)
* **M-2.** §10.2's sixth bullet says *"§1's table shows 4 of 6 HateMM rotations and 2 of 6 ZH
  rotations sitting **below** the primary."* **The counts are correct** — I verified from
  `C01_A0_OUT.json`: HateMM `17.6/29.1/60.4/72.7` all `0.8505 < 0.8598`, ZH `8.3/29.1` at `0.8462 <
  0.8590` — but §1's table lists only `orthrot_83p8` and `orthrot_72p7`, so it shows no such thing.
  Repair: add the four missing rotation rows to §1's table, or attribute the counts to
  `C01_A0_OUT.json`.
* **M-3.** §7.3 is byte-identical to v7 and still asserts *"**No arm accuracy has been computed,
  printed or recorded at any point in v1–v7**"* and *"v6–v7 add exactly one measured accuracy"*, in
  a v8 document. The footer covers v8 and the claim remains true; the scope should read v1–v8.
* **M-4.** §10.2's new clause leaves `**` markers unbalanced — *"…named explicitly in the verdict
  **together with the dataset(s) on which it passed**, as §5.6 requires**"* — nested inside the bold
  span opened at *"C06's first-order"*. The content is unambiguous in plain text; the rendering is
  not.
* **M-5.** §5.2.3 says *"`tiny_ok` was slack by ~600× in C01's own run"*. `tiny_ok` compares a
  **fraction** (`0.0`) against `0.05`; the `~600×` is the margin of the **minimum displacement
  norm** over the epsilon (`0.6146 / 0.001 = 614.6`). Both numbers are correct and I verified both;
  the ratio is attached to the wrong test.
* **M-6.** §14.1 cites the audit script as `scratchpad/v8_audit.py`. The repository's `scratchpad/`
  contains `g2_score_summary.json` and `verdict_parse.py` only; the script lives in the session
  scratchpad. Since §14.1's claim is that the audit is re-runnable, the path should be one a reader
  can follow — or the script should be committed alongside the freeze.

---

# REQUIRED RULINGS

## 1. §4.A — is the §14.1 protocol adequate? **No, and I can name the gap precisely.**

It is a real improvement and it did real work: 15 of its 16 section deltas, its touched-section set
and all 13 row citations reproduce independently, and the three sections round 7 named as falsely
claimed do appear as `CHANGED +1463`, `CHANGED +376` and `ADDED 2653`. A row that cannot cite a
diffed section can no longer ship. That closes the exact defect rounds 6 and 7 found.

It is not adequate, for three reasons I can demonstrate rather than assert:

1. **It checks the wrong direction.** It asks *"did the section this row cites change?"* and never
   *"did every limb of the finding land?"* Round-7 C-1 had three prescribed limbs; two landed, C-1
   cites the two sections they landed in, and the audit rates it `OK`. Every unlanded limb this
   round is in **§13, which no §14 row cites**, so the audit is structurally blind to the one
   section v8 never opened. C-2 above is that blindness realised on its first run.
2. **The reviewer's own question is well founded.** Yes — a row can cite a section that diffed for
   an unrelated reason and pass. It happens here: M-2 cites §3.4, which diffed overwhelmingly for
   C-1; C-4 cites §3.7, which diffed for C-2 as well. Both repairs did in fact land, so nothing is
   concealed this round, but the audit did not establish that — I did, by reading. This matters
   whenever a section is edited for two findings at once, which is the common case.
3. **It cannot audit itself.** §14.1 reports its own byte count, and pasting the report changes it
   (C-1). A self-measuring instrument needs an explicit convention, and none is stated.

**What I would add**, in order of value:
* **Step (5): limb-level disposition.** For each finding, list the prescribed repair limbs and the
  section each landed in; a limb with no section is `NOT ADOPTED`, and the header may not say
  *"all ADOPTED"* while any limb is open. This is the check that would have caught C-2, and it is
  the only one of my suggestions that is load-bearing.
* **Step (6): sections that changed but no row cites, and sections a finding names that did not
  change.** §13 would have appeared under the second list this round.
* **Make §14.1 self-excluding**, so its transcript is a fixed point by construction.
* **Commit the script with the freeze.** An audit that cannot be re-run from the repository is a
  claim, which is what the protocol exists to replace.

On the framing question — *is an embedded self-audit an adequate response to two rounds of false
disposition claims?* An embedded self-audit is necessary and not sufficient. The mechanism that
actually caught both C-2 and the four PARTIALs is an independent reader diffing v7→v8 against
**round 7's prescriptions**, not against §14's self-description. That is a review obligation, not a
drafting one, and it should be written into the round-9 request rather than delegated to a script.

## 2. §4.D — can any gate fire on a warranted CLOSE? **No, for all twenty. Re-derived, not inherited.**

A *warranted CLOSE* is: the instrument is sound, and the real arms fail to beat the rotation family.
I checked each gate for whether its firing predicate can be entailed by that state.

**The twelve globals** are all arm-outcome-independent by construction. `GATE-DET1` (env),
`GATE-SHA` (digests), `GATE-FOLD` (banked parity flags), `GATE-POP` (populations and class counts),
`GATE-NULLREMOVED` / `GATE-ZEROMASK` (exact-zero row sets), `GATE-IDPARITY` (`ids`/`labels` order),
`GATE-LEDGER` (process and path counts) touch no arm score at all. `GATE-FLOOR` votes **native**
deployed keys against banked anchors — no battery arm enters it. `GATE-C01PARITY`, `GATE-ROWSUBSET`
and `GATE-RHORAW` are properties of the **raw two-block build** and the frozen `ρ_raw` table, which
are fixed before any head exists; I recomputed all three at `0.000e+00`, `0.000e+00` and 26/26 at
4 dp.

**The six per-lineage gates.** `GATE-ARENA`'s lower bound is on `endpoint_std` **only** — a control,
not a real arm — so a CLOSE state (real arms losing) cannot entail it; a failing `endpoint_std`
means the head space is dead, which is an instrument failure and correctly a HALT. Its upper bound
`≤ 0.98` catches leaks and cannot fire downward. `GATE-ORBITDISP` fires on
`ρ_head > ρ* ∧ ρ_raw ≤ ρ*` — direction degeneracy, measured at `0.34`–`0.67` against bars of
`0.968`/`0.977`, **0/18**, i.e. a trained head sits at roughly half the bar; nothing about a real arm
losing moves it. `GATE-ALGEBRA` bounds the θ=0/θ=45 identity residuals, `7.5×`–`22.6×` inside `2e-6`
on trained heads. `GATE-ZEROOP` compares guard arms to their counterparts and is explicitly
one-directional (REPORT → HALT only). `GATE-SELFTEST` and `GATE-NESTED` are identities that hold for
any arm set. **`GATE-DOMAIN` and `GATE-DEVFID` carry no bar.**

**The tightened `GATE-C01PARITY` specifically.** Bit-exactness raises the false-HALT question, and
the answer is that it is attainable and robust: the comparison is between the battery's builder and
`prepare_views` **in the same process, over the same arrays, through the same `l2_rows`**, so the two
sides execute identical operations in identical order and agree bitwise by construction. Six
independent reconstructions have now measured `0.000e+00`; mine is the sixth, and it also holds for
the `n = 743` `GATE-ROWSUBSET` bridge. A failure would be a HALT, never a CLOSE.

`GATE-ARMVIAB` was the one gate that could fire on a warranted CLOSE and §6.2 retires it with a
correct argument, including the demonstration that a restricted version would be strictly redundant
with `GATE-ARENA`'s lower bound.

## 3. Verdict-path enumeration — mine, not inherited: **total, mutually exclusive, one lawful absence path.**

Let `P_N, P_R ∈ {passed, dropped}` after the per-lineage gates (drop propagates across datasets by
§5.6's rule: fail on **any** dataset ⇒ dropped on **both**), and for each passed lineage let
`C_L ∈ {clears S1–S7 on both datasets, does not}`, where clearing is the disjunction over
`A ∈ {displacement, common_displacement}`.

* **any global gate fails** → rule 3 names this explicitly → **HALT**.
* **both passed**: some lineage clears → rule 1 **SURVIVE**; neither clears → rule 2 **CLOSE**.
* **exactly one passed**: it clears → rule 1 **SURVIVE**; it does not → rule 2 false (not both
  passed) → rule 3 **HALT**.
* **both dropped**: rule 1 has no passed lineage, rule 2 false → rule 3 **HALT**.

Rule 3 is written as the catch-all *"otherwise"*, so totality holds by construction and the three
rules are pairwise exclusive because rule 2 requires the negation of rule 1's antecedent under
*"both passed"*. Crossing this with the dataset axis adds nothing: the drop rule collapses the
per-dataset outcome into the lineage's pass/fail before combination, and S1–S7 are required on
**both** datasets conjunctively.

**The declared-drop exemption is the only lawful path to an absent quantity**, and it is scoped
correctly: §5.6 exempts a dropped lineage's quantities, excludes them from S1–S7 and the S5 family,
and admits them to the S4 family only as `NOT_TESTED` with `p = 1`; *"absence by computation failure
in a surviving lineage still HALTs"*. **No gate failure is reportable as a closure** — rule 2 requires
both lineages to have passed, and every gate failure routes to rule 3 or drops a lineage.

I verified the S4 family arithmetic that the drop rule depends on: `92 × 1/2001 = 0.045977 ≤ 0.05`
and `92 × 2/2001 = 0.091954 > 0.05`, so the resolution floor §5.5 states — every witness comparator
at `p = 1/2001` — is exactly right, and `NOT_TESTED` at `p = 1` is non-rejecting at any rank.
S5's feasibility bound also checks: `12 × 1/257 = 0.0467 ≤ 0.05`, `13 × 1/257 = 0.0506 > 0.05`, so
`n ≤ 12` and the frozen family of 4 is comfortable.

## 4. Rulings on §15's six open issues

1. **The §14.1 protocol.** Re-run; one line disagrees (C-1); the protocol is necessary and not
   sufficient (ruling 1 above). Adopt step (5).
2. **`GATE-C01PARITY` as a single bit-exact predicate.** **Correct and settled.** Bit-exactness is
   attainable — I measured `0.000e+00` on 13 arms × 2 datasets and on the `n = 743` bridge — and
   **no other gate inherited the retired tolerance**: `2e-6` now appears in v8 only as
   `GATE-ALGEBRA`'s bar (§6:856, §6.5, §7.8) and in the narrative explaining the strike. Keep it.
3. **The §3.4 pin, second attempt.** **Sufficient.** I rebuilt all thirteen arms from §3.4's prose
   plus §1's two identities and got `0.000e+00` on both datasets **at the first attempt, supplying
   nothing** — the modality order is stated (*"Two blocks `[img, text]`"*), the Givens sign
   convention is forced by *"θ = 45° **is** `common_displacement`, θ = 0 **is**
   `endpoint_concat`"*, and the pre-normalisation is now explicit. I then ran the wrong readings the
   prose might still permit: dropping pre-normalisation gives `1.878e-06` / `1.609e-06`,
   `common_interaction = l2(std ⊙ ow)` gives `9.697e-01` / `9.558e-01`, and flipping the
   displacement sign gives `9.701e-01` / `9.660e-01`. **Every one is now caught**, where the first
   was not under v7's row. The one thing still unstated is the epsilon (I-3), and it cannot produce
   a wrong arm — only a `die()`.
4. **§5.2.3's `tiny_ok` non-carriage.** **Sufficient, and I would not reopen it.** The direction is
   conservative (dropping a conjunct that can only make S7 fail hardens CLOSE), C01's own run
   records `maximum_tiny_fraction = 0.0` with `tiny_epsilon = 0.001` on both datasets — I read both
   out of `C01_A0_OUT.json` — and the four head-space cells sit `18×`–`38×` above the epsilon with
   `frac = 0.0000`. Four cells of sixty bound nothing formally, and the document says so; what makes
   it rulable is **§13 item 25**, which converts the assumption into a per-cell run-time record. That
   is the right structure: the warrant is not *"four cells generalise"* but *"every cell is audited
   and the four say what to expect"*.
5. **Phase 7z's `1.0 s` versus round 7's `0.160 s`.** **Freeze the conservative figure — it is the
   right one — and say what was inside the timed region.** My independent measurement of the same
   loop is `0.128 s` (120 `np.max(np.abs(·))` reductions, 60 cells at `n = 743` and 60 at
   `n = 579`), close to round 7 and `8×` under v8. For item 25's tail record I measured `0.619 s`
   *including* materialising the 1024-d norms, against v8's `0.034 s` and round 7's `0.122 s` — so
   the two of us bracket v8 in **opposite directions**, which is decisive: the spread is about what
   each timer enclosed, not about the machine. Freezing `1.1 s` bounds all three measurements, costs
   `0.04 %` of the total, and leaves the heartbeat interval untouched. The lesson to record is the
   one §7.7 already institutionalised for the mint units: **state the timing boundary**, not just the
   number. The next uncounted loop is I-2 above.
6. **Is the record now sound?** **No** — two Criticals, two Highs, and four of the thirteen round-7
   dispositions are partial rather than adopted. The **science** is sound: I could not manufacture a
   CLOSE anywhere in the combination space, all twenty gates hold under the warranted-CLOSE test, the
   arm builder is now pinned, and every number I checked reproduces. What remains is bookkeeping, and
   it is concentrated in one place: **§13 has not been edited since v7, and three of this round's
   four record findings are consequences of that.**

## 5. Process rules

**`rule_1_compute_projection`.** §8 re-multiplies exactly — the printed product column sums to
`2928.7`, `× 1.25 = 3660.9`, `48.8`/`61.0 min`, and the mint share, Phase-3 share and 2×/5× miss
figures all re-derive. Every count I could check is right: `36 + 30 = 66` mints,
`(30×3)+(6×4)+(30×2) = 174` key forwards, `13 × 60 = 780` head cells matching §6.1, `256×3×2×2 =
3072` null draws, `23×2×2 = 92` comparison-cells, `14×3×2×2 = 168` selftests. **Seven rounds, seven
uncounted loops: the seventh is `GATE-ZEROOP` (I-2).**

**`rule_2_heartbeat`.** Nothing in v8 changes an interval. Phase 7z's `1.1 s` and the sixteen extra
digests (5 ms, measured) sit far under the `~15 s` bound, and the longest un-instrumented span
remains one `GATE-C01PARITY` dataset at `11.27 s` (`14.1 s` conservative). The `buffering=1`
per-phase line-buffered handle and the unbuffered driver echo are unchanged and adequate.

## 6. Can the falsifier discharge the written condition at `$0`? **Yes.**

The instrument does what the registry asks: it re-runs C01's real-displacement-versus-rotation
battery in the fold-head arena on already-banked caches, on CPU, with no extraction. The head-space
arms are buildable (§7.4(g), 13 arms at `n = 743`), the anchor reproduces bit-exactly, the arena is
alive (`GATE-FLOOR`'s native OOF accuracies run above C01's dev arena), and the decision rule can
reach CLOSE, SURVIVE and HALT on distinguishable states. Nothing in my findings threatens the `$0`
character or the verdict path. **The blocker is the record, not the science.**

---

# WHAT A GO WOULD AND WOULD NOT AUTHORIZE

Not applicable this round. For the record: a GO authorizes **nothing to run**. Before any job this
design still needs (1) a freeze with hashes, (2) a **separate** independent code/resource review
lineage over the executable reaching its own `0C/0H/0I` — and on the evidence of H-2 and I-3 that
lineage has real work waiting in §13 — and (3) main-dialogue authorization. A GO is not authority to
write `TARGET_STATE.json`.

---

# CLOSING

Nine of thirteen round-7 dispositions are fully adopted and I verified each by measurement where a
measurement existed. Round-7 C-1 — the wrong-verdict path — is genuinely closed, and I confirmed it
the only way that settles a specification defect: by building the battery from the prose and getting
the right answer without help. §5.2.3, §10.2, §5.2.2, §6.1 and §8 are all substantively repaired,
and the raw-space dispersion figures, the Holm counterexample, the `ρ` tables, the C01 constants and
all 37 digests reproduce exactly.

The most severe finding is **C-2**, and it is severe because of what it reveals rather than what it
breaks. Neither unlanded limb can produce a wrong verdict: round-7 C-1's third limb is redundant
with §13 item 23 plus §6's new predicate, and M-1's second limb is a wrong verb. What matters is
that the header once again says *"all ADOPTED"* over an artifact that is partially adopted, for the
third consecutive version, and that the mechanism built to prevent exactly that rated all thirteen
rows `OK`. §14.1 asks whether each cited section changed; it cannot ask whether an *uncited* section
should have. Every unlanded limb this round, and both Highs, trace to **§13 — byte-identical between
v7 and v8 at 848 + 7843 characters** — which is cited by no disposition row and is simultaneously the
sole input to the mandatory code-review lineage. A protocol whose blind spot coincides with the
document's weakest section has not yet earned the confidence its framing claims. Adding limb-level
disposition (step 5) and a *"changed-but-uncited / named-but-unchanged"* report (step 6) would close
it, and both are a few lines of the same script.

---

*Read-only review. No GPU, SLURM, Modal, arena run, mint, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json` was read and not modified. The draft, the
configs and all repository files are unmodified; my scripts and outputs live in the session
scratchpad only.*
