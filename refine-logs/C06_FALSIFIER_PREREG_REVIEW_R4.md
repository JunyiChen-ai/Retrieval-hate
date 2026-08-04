# C06 `$0` falsifier — independent design review, **ROUND 4**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V4.md` (DRAFT v4, 2026-08-04)
**Reviewer posture:** fresh, independent of rounds 1–3 and of the designer. Read-only. No GPU,
SLURM, Modal, model load, head training, arena run, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json`, all four drafts, all configs and all prior
reviews were read and not modified. Nothing heavier than `sha256sum`, file reads, and
numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint
checkpoints was executed. **No arm accuracy was computed at any point.**

---

# VERDICT

## **REVISE (3C / 3H / 8I)** — plus 4 Minor

Not `GO (0C/0H/0I)`.

**Ceremony floor: clean, and re-derived rather than read.** All **21** sha256 digests reproduce
character-for-character. Both provenance chains are sha-gated in source. All **26** `ρ_raw`
values reproduce at 6 dp. The six `GATE-FLOOR` anchors match the banked JSONs on both metrics.
The trained-head `ρ` reference reproduces to the digit on all 36 banked mints (**0/18 above `ρ*`**
on both datasets). Every §8 product re-multiplies. `GATE-IDPARITY`'s property holds directly.
Test-split non-contact is sound by construction. Blindness discipline is intact: I grepped every
number in `[0.6, 0.99]` in v4 and every one is a `ρ`, a banked floor, a published C01 dev-arena
figure or a majority rate — **no arm accuracy from this battery appears anywhere in v1–v4**.

**Disposition audit: 12 of the 13 round-3 findings and all 4 Minors are genuinely adopted; one
(I-3) is adopted in text but its arithmetic is now false — broken by the adoption of I-4.**
Both round-3 Criticals landed and I verified each **by executing C01's frozen code**, not by
reading. The `None` sweep is complete: I grepped the whole draft and no `prepare_views` call site
is specified with `None` anywhere. The arena majority is `446/743 = 0.600269 → 0.6003` exactly as
claimed, and `GATE-POP`'s class-count clause does make it checkable.

**All three of my Criticals sit in the seam a round-3 repair opened — as every round before has
found.**

**C-1 is the recurring wrong-verdict class, in the gate that was rewritten to prevent it.**
`GATE-ARMVIAB`'s two-case form escapes a one-sided HALT only via case 1, which requires the
**raw** counterpart of a real arm to *also* fail `arena majority + 0.02`. v4's own §1 table
records C01's measured raw `displacement` at **`0.8505 / 0.8846`** against bars of
**`0.6203 / 0.7091`** — clearing by 0.18–0.23. Case 1 is therefore unreachable, and
`GATE-ARMVIAB` degenerates into exactly the one-sided majority-rate HALT that §6.2's own opening
sentence identifies as *"convert[ing] a **warranted CLOSE** into a HALT, leaving C06 gated
forever on an instrument that can never close it."* Round 3 blessed the repair on the assertion
that *"if C06's premise is false the raw counterpart will not [clear]"* — a claim the document's
own §1 table refutes.

**C-2 is round 3's I-3 and I-4 destroying each other.** I-4 pinned S6 to the **per-seed** integer
net in 3/3 seeds. I-3's vacuity argument requires S3 ⇒ S6, but S3 is defined on the **seed mean**
(§5.1). A seed mean of `15` is consistent with per-seed nets `(2, 21, 22)`, which fails S6 on
HateMM. So §5.8 item 4's *"S3 implies S6 by arithmetic"* is **false**, and §5.2's instruction on
the strength of it — *"S6 is reported, not screening"* — deletes a conjunct that **can** fail from
the SURVIVE conjunction. That relaxation runs **anti-conservatively**, against §4's declared lean
and against round 1's binding condition that the lean *"must not be allowed to excuse an
arithmetic error."*

**C-3 is an uncounted loop, and it is the one the process rule names by name.**
`rule_1_compute_projection` requires the count to be multiplied through *"draws x **folds** x
seeds x taus x spaces x datasets."* §8 Phase 2b prices head-space arm construction at **12** cells
(`2 ds × 3 seeds × 2 lineages`) and Phase 2D at **12** head `ρ` cells. The head key matrix is
**per fold** — §8's own Phase 1b decomposition `(30×3)+(6×4)+(30×2)` proves there are 60 fold
mints, each with its own `(h_std, h_ow)` pair — so both counts are short by ×5 (60 and 60).
Separately, `GATE-ZEROOP`'s two guard arms (`orthrot_0`, `orthrot_45`) are outside the 13 and are
neither built nor voted anywhere in §8: 120 uncounted votes. Corrected total `2925.0 s`
(`× 1.25 = 3656.3 s`).

**None of the three requires a GPU, an extraction, or a redesign.** C-1 is a discriminator swap
(the right one already exists, in `GATE-ARENA`); C-2 is a choice between two sentences; C-3 is
three integers and one row. The instrument, once they are fixed, does measure what
`falsifier_spec` asks. **The falsifier can still discharge its written condition at `$0`.**

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All 21 digests recomputed and matched: 7 imported modules, 6 read-for-definition files, 8 input caches. Spot values: `headspace_mint.py` `cefdf8dc…0916612`; `c01_policy_contrast_a0.py` `d2b9c2ff…8db1b855`; HateMM `ro_L24` `6a44cce4…0be045f`; MHC-ZH `ro_ow_L24` `3ad1309d…276b8d2a3`. Both provenance chains sha-gated **in source** (`_v4.py`→`_v3.py`→base), as round 3 recorded. |
| **V2** | **VERIFIED — I executed all four legs** | `prepare_views(…, None)` **DIES on both datasets**: MHC-ZH at `n = 579` with *"derived exact-zero mask preservation failed"*; HateMM at `n = 744` earlier, at `standard/img`, with *"exact-zero mask diverged from authorized mask"*; and — the leg v4 does not report — **HateMM at the arena `n = 743` also dies with `None`**, at the derived-mask check, confirming the failure is unconditional on the presence of a null row. `np.zeros(579, bool)` ⇒ **OK** on MHC-ZH. `np.zeros(744, bool)` ⇒ **DIES** on HateMM (one-hot `{355}` required). **`np.zeros(743, bool)` on the HateMM arena ⇒ OK**, and `np.zeros(579, bool)` on the MHC-ZH arena ⇒ OK. `l2_rows:1187-1188` does normalise `None` (source-verified), so the two functions genuinely differ. **The `None` sweep is complete** — I grepped the draft; the only surviving occurrences are the explanation of *why* `None` is inadmissible, the unrelated deferred-import note at §3.4, and the §14 disposition row. |
| **V3** | **VERIFIED — exact** | HateMM full `n = 744`, `pos 298 / neg 446`, majority `446/744 = 0.599462 → 0.5995`. **Arena `n = 743`, `pos 297 / neg 446`, majority `446/743 = 0.600269 → 0.6003`, band `[0.6203, 0.98]`.** MHC-ZH `n = 579`, `pos 180 / neg 399`, majority `399/579 = 0.689119 → 0.6891`, band `[0.7091, 0.98]`. Row 355 = `hate_video_95`, **label 1**, the only exact-zero row on either dataset in either policy. `0.5995` appears in v4 only at §3.7 and §7.1, both scoped to the 744-row population, and is consumed by no gate. **But see I-2 and I-3: two *other* population-dependent quantities still carry no population.** |
| **V4** | **VERIFIED** | `‖head_f(0,0)‖` non-zero at torch seeds 0/1/2. `h_std[355] == h_ow[355]` **exactly** and not zero; **zero** exact-zero head rows on either dataset; row 355 is the only such row. Per block, HateMM: one-hot `{355}` ⇒ endpoint **DIES**, common **DIES**, displacement OK; all-False at `n = 744` ⇒ endpoint OK, common OK, displacement **DIES** ⇒ **`common_displacement` unbuildable under either mask** ✓. **The repair: all 13 head-space arms BUILD at `n = 743` with the all-False array**, through the imported `l2_rows`, `float32`, key dims `{1024: 4 arms, 2048: 9 arms}` — which independently confirms §8 Phase 2's `240 / 540` split. Min displacement-block row norm `1.0000` against `ε = 1e-12`: no epsilon risk. See **M-3** on the `0.58–0.65` range. |
| **V5** | **VERIFIED — exact** | Raw arms at `n = 743` vs the `n = 744` one-hot build restricted to the 743 surviving rows: **`max\|diff\| = 0.000e+00` on all 13 arms**. Algebra guards **bit-identical** (`8.940696716308594e-08` / `1.1920928955078125e-07` on both). Every `ρ` unchanged. |
| **V6** | **VERIFIED** | I re-implemented §3.4's `fuse` / `paired` / `build_views` from the prose alone, calling the imported `l2_rows`, and compared against `prepare_views` with §3.7's mask forms: **`max\|diff\| = 0.000e+00`, all 13 arms, both datasets**, `float32` on both sides. Third independent reproduction from the document text. |
| **V7** | **VERIFIED — all 26, exact** | `ρ* = 0.968176` (HateMM, `endpoint_std`) / `0.977223` (MHC-ZH, `endpoint_std`); runner-ups `0.964446` / `0.969686`, both `common`. Every one of the 26 `ρ_raw` entries reproduces at 6 dp exactly as tabulated, including `common_interaction`'s cross-dataset asymmetry and `displacement` as least-concentrated on both. **The full-precision freeze does remove the self-exemption**: `ρ_raw(endpoint_std) ≤ ρ*` now holds by equality, and both §6.1 tables quote the same digits. `ρ` over 744 rows with the masked zero left in: `0.966874` vs `0.968176` — shift `1.301e-03` ✓. |
| **V8** | **VERIFIED — to the digit** | Over all 36 banked `mint_*.npz`, `ρ = ‖mean_i l2n(K_i)‖`: HateMM `0.447803 / 0.562434 / 0.632996`; MHC-ZH `0.340179 / 0.574247 / 0.667326`. **0/18 above `ρ*` on both.** The bar has a measurement behind it and a trained head sits at roughly half of it. |
| **V9** | **VERIFIED** | From the six banked `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`: `acc_deployed` HateMM `0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946`; `mF1_deployed` HateMM `0.8838 / 0.8811 / 0.8812`, ZH `0.8747 / 0.8710 / 0.8765`. Identical to §6's anchors. |
| **V10** | **VERIFIED** | From `c01_a0_v2.json`: `gain_controls` = 5; `bootstrap_comparisons.primary_vs_controls` = 6. `(6+6) + (5+6) = 23`; `× 2 metrics = 46` per `(dataset, lineage)`; `× 2 lineages = 92` per dataset. §8 Phase 4's `23 × 2 ds × 2 lineages = 92` is a **different product** that coincides. M-4 is correct. |
| **V11** | **VERIFIED, with M-1's wording now stale** | Every product re-multiplied; none is wrong. Totals `2886.3` / `3607.9 s`; `48.1` / `60.1 min`; mint share `2508.3/2886.3 = 86.90 %`; Phase 3 share `9.48 % → 9.5 %`; sensitivities `3160.0` and (on the rounded `273.7`) `3981.1`. **But v4's Phase 7 row now prints `0.1 s` in the product column, so the printed column sums to `2886.3`, not `2886.2` — §8's M-1 sentence describes v3's table, not v4's** (**M-2**). And the enumeration itself is incomplete (**C-3**). |
| **V12** | **MISMATCH** | The arithmetic is right on the **seed-mean** axis (`0.02 × 743 = 14.86 ⇒ ≥ 15`; `0.02 × 579 = 11.58 ⇒ ≥ 12`, against frozen minima 3 and 2). It is **wrong on the axis v4 actually pins S6 to**. §5.2 (round-3 I-4) defines S6 as the **per-seed** integer net required in **3/3 seeds**; §5.1 defines `acc(A)` as the **seed mean**. S3 therefore bounds only `mean_s net_s ≥ 14.86`, which is satisfied by `net = (2, 21, 22)` — and that fails S6. **S6 is not implied by S3 and can bind.** → **C-2**. |

## Additional measurements v4 does not report

**(α) `prepare_views(…, None)` dies on the HateMM *arena* too.** v4 §7.4(i) records the failure
at `n = 744` and `n = 579`. I add the third cell: at `n = 743`, where no null row exists, `None`
still dies at `:1381-1386`. This strengthens §3.7's paragraph — the inadmissibility is a property
of the function, not of the data — and it is worth one clause in the record.

**(β) `GATE-IDPARITY` holds directly.** Both ro caches' `ids` are order-identical and `labels`
element-identical to the native bank, on both datasets.

**(γ) The head-space key dimensions are `4 × 1024-d` and `9 × 2048-d`.** This is the first
independent confirmation of §8 Phase 2's `240 / 540` split, and it also shows the two
`GATE-ZEROOP` guard arms are 2048-d objects outside the 13 (→ **C-3**).

**(δ) `deployed_vote` confirms §3.7's neighbour-displacement argument.**
`mechfix_ops.py:94` is `votes = ((lab*2-1) * sim * w).sum(1) / w.sum()` over `_flat_ip`
neighbours: a zero key contributes exactly `0` to the vote while occupying a top-20 slot ahead of
any negatively-similar candidate. §3.7's sharpened argument is right and is confirmed at the
operator.

**(ε) `headspace_fidelity.py` opens no `dev_seen_*.pt` at all.** It reads
`mint_{ds}_s{seed}_ffull.npz` (`:66`) and the banked floor trainlog (`:42`). So §12's
`dev_path_opens` expectation *"`mints_executed` + `GATE-DEVFID` reads"* has a second term of
**zero**, which the design leaves unquantified in a **binding** predicate (→ **I-5**).

**(ζ) Holm feasibility at `B = 2000` over 92 hypotheses.** C01's `one_sided_raw_p` is
`(1 + #{Δ ≤ 0}) / (B + 1)` (`:1769`), floor `1/2001 = 0.00049975`, against `α/92 = 0.00054348`.
The next achievable level, `2/2001 = 0.00099950`, only clears Holm from rank 42 onward. So **at
least 42 of the 92 comparators must show zero adverse resamples out of 2000.** Feasible, but the
design never states the p-value definition at all (→ **H-2**).

---

# PART B — FINDINGS

## CRITICAL

### C-1. `GATE-ARMVIAB`'s escape case is unreachable on the real arms, so the gate **fires on a warranted CLOSE** — the exact failure §6.2's own opening sentence exists to prevent.
*Attaches to:* §6 `GATE-ARMVIAB`; §6.2 both bullets; §6.3's closing invariant; §14 row for round-2 C-1's companion.

§6.2 opens correctly:

> A one-sided majority-rate HALT on the real arms would fire on exactly the outcome the falsifier
> exists to detect: if C06's premise is false, `displacement` in head space *should* sit near the
> majority rate, and a one-sided gate would convert a **warranted CLOSE** into a HALT, leaving C06
> gated forever on an instrument that can never close it.

The repair is a two-case form whose only non-HALT branch is:

> head-space arm fails `arena majority + 0.02` **and the raw counterpart also fails** ⇒ genuine
> negative ⇒ **no HALT**.

**That branch cannot be reached.** v4's own §1 table — the Gate-0 adjudicator's re-verification
against `C01_A0_OUT.json` — records the raw `displacement` arm at **`0.8505`** (HateMM) and
**`0.8846`** (MHC-ZH), and `common_displacement` at `0.8598` / `0.8590`. The arena bars are
**`0.6203`** and **`0.7091`**. Every real arm clears its raw bar by `0.15`–`0.23`. The raw leg
here is the *same features and the same operator* on a **larger** arena (743/579 OOF train items
rather than 107/78 dev queries), and this campaign's OOF train arenas run *higher* than its dev
arenas, not lower — `GATE-FLOOR`'s own native OOF accuracies are `0.8884 / 0.8929` against C01's
dev-arena `0.8411 / 0.8590`. There is no credible path on which raw `displacement` falls below
`0.6203`.

So on the real arms `GATE-ARMVIAB` reduces to: *head-space real arm fails `majority + 0.02`
⇒ **HALT***. That is the one-sided gate §6.2 rejects, restored verbatim, and it fires on the
outcome §6.2's own sentence names as the warranted CLOSE.

**Why the discriminator is the wrong one.** The question `GATE-ARMVIAB` needs to answer is *"is
the head space alive?"* — and that is answered by the **controls in the same space**, never by the
same arm in a different space. A real arm that collapses in head space while the head-space
controls (`endpoint_std`, `common`, the rotation family) stay healthy is not an instrument
failure; it is the **strongest possible negative for C06**, and the design cannot publish it. Only
the case where the controls collapse *too* is an instrument failure — and that case is already
caught, by `GATE-ARENA`'s lower bound on `endpoint_std`, which round 2 ruled substantively
discharges round 1's C-2 and which §6.3 keeps.

**Round 3 verified this repair on a false premise.** Its Part C §3.B reads *"`GATE-ARMVIAB` fires
only if the **raw** counterpart clears the same bar, and if C06's premise is false the raw
counterpart will not."* C01's measured `0.8505 / 0.8846` — printed in the same document — says it
will. C06's premise being false never meant *`displacement` carries no signal*; it meant
*`displacement` carries no more signal than an arbitrary angle on its own Givens family*, which
§1's round-14 sharpening states explicitly.

**Repair (one of two, both free).** Either (i) **restrict `GATE-ARMVIAB` to `endpoint_std`**,
where the raw-vs-head discriminator is meaningful and where it harmlessly duplicates
`GATE-ARENA`'s lower bound, and state that no viability HALT is applied to a real arm — §6.3's
invariant *"real arms lose badly is a reportable scientific outcome, never an instrument HALT"*
then becomes literally true; or (ii) **replace the discriminator**: HALT on a real arm's
head-space viability failure **only if `endpoint_std` and the strongest ordinary control also fail
in head space** (i.e. the space is dead), and otherwise report. (i) is simpler and is what §6.3
already claims the design does.

### C-2. §5.8 item 4's *"S3 implies S6 by arithmetic"* is **false** once round-3 I-4 pins S6 to the per-seed axis, and §5.2 deletes a conjunct that can fail on the strength of it — an **anti-conservative** relaxation of the SURVIVE rule.
*Attaches to:* §5.2 S3/S6 and the sentence *"S6 is reported, not screening"*; §5.1; §5.8 item 4; §6 `GATE-SELFTEST`; §14 rows **I-3** and **I-4**; §15.4.

The two round-3 adoptions are individually correct and jointly inconsistent.

* **I-4 adopted:** *"the count is the **per-seed** integer net, required in `3/3` seeds. A
  seed-mean net is non-integer and would be ill-typed against `minimum_net_fixes`."* (§5.2)
* **§5.1:** *"`acc(A,D,L)` is the mean over the three seeds."*
* **S3:** `acc(A) − max_{c∈C} acc(c) ≥ 0.02` — therefore on the **seed mean**.
* **`GATE-SELFTEST`:** `net(A) = n_D · (acc(A) − acc(endpoint_std))` *"for every arm, **seed**,
  dataset and lineage"* — therefore per seed.

S3 with `endpoint_std ∈ C` gives `mean_s net_s(A) ≥ 0.02 × 743 = 14.86`. It gives **nothing at
all** about `min_s net_s(A)`. Counterexample on HateMM: `net = (2, 21, 22)` has mean `15 ≥ 14.86`,
so S3 is satisfiable, and `net_0 = 2 < 3`, so **S6 fails**. The required across-seed spread is
`20` net items on 743 = `2.7` accuracy points, which is inside this campaign's ordinary seed
noise, not an exotic construction.

**Why this is Critical rather than a correction.** §5.2 does not merely disclose the (claimed)
vacuity; it acts on it: *"**S6 is reported, not screening — see §5.8 item 4.**"* An implementer
following that sentence drops S6 from the SURVIVE conjunction. Because S6 **can** fail, dropping
it makes SURVIVE strictly **easier** — the anti-conservative direction, against §4's fixed
meaning of *"conservative"* (*hardest for the falsifier to deliver the `$0` closure*) and against
round 1's binding condition that the lean *"must not be allowed to excuse an arithmetic error."*
It is also, structurally, the round-3 C-2 pattern recurring: a constant/axis change applied in one
place and not propagated to the argument that consumed it.

Secondarily, §5.2's table and its own footnote **contradict each other on the verdict path**: the
table says *"C06 SURVIVES iff … S1–S6 all hold"*, the footnote says S6 does not screen. Those are
different decision rules. A freeze cannot carry both.

**Repair — pick one, and only one.**
(a) **Keep S6 binding.** Delete the *"reported, not screening"* sentence and rewrite §5.8 item 4
to say what is true: at arena scale the frozen minima are far **below** what S3 implies *on the
seed mean*, so S6 binds only through **across-seed dispersion**, which is a real and intended
tightening. This is the conservative option and costs nothing.
(b) **Move S6 out.** Delete it from the S1–S6 conjunction, re-letter, and carry the per-seed net
purely as a reported quantity in the Gate-0 currency — but then §5.8 must say plainly that this
**relaxes** the rule relative to v3, and must justify the relaxation on its own terms rather than
on a vacuity claim.
I recommend (a): `GATE-SELFTEST` keeps its object either way, and (a) is the only option that does
not loosen a frozen decision rule.

### C-3. §8's enumeration is not exhaustive: two phases omit the **fold** axis (×5) and `GATE-ZEROOP`'s two guard arms are counted nowhere. `rule_1_compute_projection` names "folds" in its own enumeration list.
*Attaches to:* §8 Phases **2b**, **2D**, **2**; §6 `GATE-ZEROOP`, `GATE-ALGEBRA`; §6.1; §9's interval claim; §2 rule R1.

**(a) The fold axis.** The head-space key matrices are per fold — `headspace_arena.py:75-89`
loads `mint_{ds}_s{seed}_f{fold}.npz` and takes `X = P.l2n(zf["K_train"])` inside the fold loop,
and `GATE-NESTED` requires *"per item, the head that scored it excluded its fold."* §8's **own
Phase 1b** decomposition `(30×3) + (6×4) + (30×2) = 174` names **60 fold mints** (30 Head-N + 30
Head-R), each producing its own `(h_std, h_ow)` pair. There are therefore **60** distinct 13-arm
head-space builds, not 12, and **60** head `ρ` cells, not 12.

| phase | v4 | correct | v4 product | correct product |
|---|---|---|---|---|
| **2b** head-space arm construction | `12` cells | `60` = `2 ds × 3 seeds × 5 folds × 2 lin` | `2.2 s` | `11.2 s` |
| **2D** `ρ`, raw + head | `14` = `2 + 12` | `62` = `2 + 60` | `8.7 s` | `38.4 s` |

**(b) The `GATE-ZEROOP` guard arms.** §3.5's thirteen arms contain `endpoint_concat` and
`common_displacement` but **not** `orthrot_0` and `orthrot_45` — the rotation family is the six
frozen angles `{8.3, 17.6, 29.1, 60.4, 72.7, 83.8}`. `GATE-ZEROOP` compares *"`orthrot_0` vs
`endpoint_concat` and `orthrot_45` vs `common_displacement`"*, which requires those two objects to
be **built by the other route and voted**. In C01 they are `prepare_views`' internal
`guard_rot0`/`guard_rot45` builds (`:1357-1370`); in head space the battery must build them
itself. §8 Phase 2's `240 / 540` decomposes exactly as `4 × 5 × 3 × 2 × 2` and `9 × 5 × 3 × 2 × 2`
— **the 13 arms and nothing else** — which I confirmed independently by measuring the head-space
key dims (`4 × 1024-d`, `9 × 2048-d`). The guard arms add `2 × 5 × 3 × 2 × 2 = 120` votes at
`U2b` (`0.755 s`) plus their construction, none of it counted.

**Corrected total `2925.0 s` (`48.8 min`); `× 1.25 = 3656.3 s` (`60.9 min`).** The cost delta is
`+1.3 %` and changes no decision — which is precisely why it must be fixed rather than waved
through. `rule_1_compute_projection`'s motivating incident is *"the realised counts … were never
re-multiplied through"*, and its own enumeration list reads *"draws x **folds** x seeds x taus x
spaces x datasets."* An enumeration that drops the named axis is the defect the rule exists to
catch, at small scale.

**Consequence for §9.** Under the corrected Phase 2D and §9's *"one per gate"* granularity, the
`ρ` computation becomes a **38.4 s** un-instrumented span at §8's own conservative unit —
above the *"nothing exceeds ~15 s"* claim. §9 needs a per-cell line for Phase 2D (and see **M-4**
for `GATE-C01PARITY`).

**Repair.** Set Phase 2b to `60`, Phase 2D to `2 + 60 = 62`, add a Phase 2z row for the guard
arms (`120 × U2b` plus `12 × U10`-class construction, or fold them into 2/2b at the corrected
counts), re-multiply the total, the `× 1.25`, the mint share and the two sensitivities, and add a
per-cell heartbeat line to Phase 2D.

---

## HIGH

### H-1. `GATE-SMALLDISP` is a **SURVIVE condition in §5.2 and an unannotated HALT gate in §6**, and the two readings give opposite outcomes on the same event. Under the §6 reading it fires on a warranted CLOSE.
*Attaches to:* §5.2's closing line; §6's gate table; §5.6.

§5.2 places it correctly: *"Plus `GATE-SMALLDISP` (§6) for C01's
`require_no_small_displacement_dominance`"* — appended to the SURVIVE conditions. C01 agrees:
it lives in `decision.require_no_small_displacement_dominance`, and
`output.decision_schema.required_halt_only_validity_guards` (seven entries, which I read verbatim)
**does not contain it**. But §6's table lists it among the gates with no *"reporting only, does not
gate"* annotation — the annotation §6 does give to `GATE-DOMAIN` and `GATE-DEVFID` — and §5.6
makes *"any HALT gate failing on either dataset in either lineage ⇒ **HALT**: no verdict, in
either direction."*

Under the §6 reading, a real arm whose few fixes happen to concentrate in the bottom displacement-
norm decile produces `INSTRUMENT_INCONCLUSIVE` instead of the CLOSE that S1's failure warrants.
That is the same class as **C-1**, in a gate no round has examined.

There is a second, narrower hazard the design does not pre-register: C01's implementation guards
the degenerate case explicitly (`c01_policy_contrast_a0.py:1989-1996`:
`fixed_fraction = 0.0 if full["fixed"] == 0 else …` and `dominated = full["fixed"] > 0 and …`).
The battery is **new code** and §11 does not list `displacement_audit` among the imports, so that
guard is not inherited by construction. A CLOSE with zero fixes would otherwise divide by zero —
the C09-lineage undefined-quantity class.

Third, C01's `displacement_audit` is hard-wired to `evaluations["common_displacement"]`. v4 runs
**two** real arms and never states whether `GATE-SMALLDISP` applies to `displacement` as well.
Applying it to a second arm is an un-preregistered extension of a frozen C01 condition.

**Repair.** State in §6's table that `GATE-SMALLDISP` is a **SURVIVE condition, not a HALT gate**,
matching §5.2 and C01's own classification; name its arm scope (`common_displacement` only, per
C01, or both real arms with that extension declared); and pre-register the zero-fix convention
(`fixed_fraction := 0`, `dominated := false` when `fixed == 0`) rather than leaving it to the
implementer. Add it to §13's list.

### H-2. **S4's statistic is not pre-registered.** The map from 2000 resamples to a Holm-testable p-value is nowhere in the document, and §5.4's own seed-averaging makes C01's `paired_bootstrap` non-reusable.
*Attaches to:* §5.2 S4; §5.4; §5.5; §11; §13.

S4 requires *"paired item-level bootstrap lower bound `> 0` **and** Holm rejects at `α = 0.05`"*
and cites four C01 constants. It never says **what p is**. §5.4 defines the resampling —
*"Resample items once (`B = 2000`); inside each resample, average the three seeds' per-item
correctness"* — which is a **different statistic from C01's**: `paired_bootstrap`
(`c01_policy_contrast_a0.py:1742-1772`) resamples items and evaluates `metric_value` on
`candidate["scores"][sampled]`, with no seed axis and no per-item correctness. The C06 battery
therefore **cannot import C01's function**, and the design does not say what replaces it. Nor does
it name the tail, the Holm ordering, or the step-down form.

This is load-bearing, not pedantic, because the natural inheritance has a hard resolution floor.
With C01's `one_sided_raw_p = (1 + #{Δ ≤ 0}) / (B + 1)` (`:1769`) the smallest achievable p is
`1/2001 = 0.00049975`, against `α/92 = 0.00054348`; the next level, `2/2001 = 0.00099950`, clears
Holm only from rank 42 of 92. **At least 42 of the 92 comparators must show zero adverse resamples
out of 2000** for S4 to pass. That is a property of the pre-registered `B` and family size that
the design should know and state before freezing, and a code lineage cannot check a statistic that
was never written down.

**Repair.** Pre-register the statistic in §5.4: the per-resample delta (seed-averaged per-item
correctness difference), the lower bound (`5 %` quantile, `> 0`), the one-sided p
(`(1 + #{Δ ≤ 0})/(B+1)`, if that is the intent), and the Holm form (C01's `holm_adjust:1775-1784`
step-down over the 92, if that is the intent). State the `B = 2000` resolution consequence for a
92-member family in §5.5. Add an item to §13.

### H-3. HALT is **global across lineages** while SURVIVE is **per-lineage disjunctive**, so an instrument failure confined to Head-N — the lineage the design itself marks *"in-domain: **no**"* — voids the verdict Head-R would have delivered cleanly. Nothing in the design prices this.
*Attaches to:* §5.2; §5.3; §5.6; §3.3's lineage table; §6 `GATE-ARENA`, `GATE-ARMVIAB`; §6.4; §5.7; §8.

§5.3 combines the lineages **disjunctively** for SURVIVE (*"C06 survives if it clears S1–S6 on
**either** lineage"*), but §5.6 combines them **conjunctively** for HALT (*"Any HALT gate failing
on either dataset **in either lineage** ⇒ HALT"*). The asymmetry matters because Head-N is, by
construction, an **out-of-domain transplant**: §3.3 measures the native and `ro_L24` image streams
as near-orthogonal (`median cos = 0.0234 / 0.0373`) and marks Head-N `in-domain: no`.
`GATE-ARENA`'s lower bound then asks Head-N's `endpoint_std` to clear `majority + 0.02` — a
recovery fraction of `0.02/(0.8884 − 0.6003) ≈ 6.9 %` — and §6.4 explicitly **refuses to invent**
a recovery-fraction bar because the quantity is unknown. So the design simultaneously declares the
quantity unknown and stakes the whole two-lineage battery on it clearing a bar.

The outcome is fail-safe (no wrong verdict) and is arguably the conservative direction under §4.
But it is a live path on which the `$0` falsifier spends itself and discharges nothing, and
neither §5.7's pre-declared expectation nor §8's risk paragraph mentions it.

**Repair.** Either (i) declare it: add to §5.7 that a Head-N-only instrument HALT is a recognised
outcome, state that it produces `INSTRUMENT_INCONCLUSIVE` for the whole battery, and say what
happens next (re-run Head-R alone under a fresh preregistration, or accept C06 remains gated); or
(ii) **scope the instrument gates per lineage**: a lineage that fails its own instrument gates is
dropped, `SURVIVE` is evaluated on the surviving lineage(s), and the battery HALTs only if **no**
lineage survives its gates — with the lineage that ran named in §10.2's scope sentence. (ii) is
the stronger design and costs nothing; (i) is the honest minimum.

---

## IMPORTANT

### I-1. `GATE-FOLD` is discharged by code that a **resumed** mint returns before reaching — the one code path §12 now blesses.
*Attaches to:* §3.2; §6 `GATE-FOLD`; §12's resume paragraph; §15.5.

§3.2 attributes the fold contract to *"asserted against the banked `vsw_ckpt/<ds>/f{0..4}.npz` by
`headspace_mint.py:203-216`."* §12's own repair rests on `headspace_mint.py:192-194` returning
**before** `:199` when `--out` exists — and `:192-194` is also before `:203-216`. On a resumed
job every skipped mint runs **no** fold-parity assertion. §12 legitimises resume explicitly
(*"The resume path is real and is the design's own"*), so `GATE-FOLD` is undefined on a path the
design now permits, and §5.6's absence rule would HALT the legitimate resume — the same
self-defeat §12 says it repaired one table over.

The mitigation exists but is unstated: a `.npz` is written at `:321-325` **only after** the
assertion at `:215-216` passes, and `:315` banks
`meta.fold_parity_vs_banked_vsw_ckpt` inside every file.

**Repair.** State that `GATE-FOLD` is discharged by `headspace_mint.py:203-216` on executed mints
**and** by re-reading `meta["fold_parity_vs_banked_vsw_ckpt"]` plus `fold_of` from all 66 banked
`.npz` before the arena — which is exactly predictable on fresh, resumed and partially-resumed
runs alike, and free.

### I-2. `GATE-DOMAIN`'s recovery fraction **mixes populations** and names neither majority — the one gate round-3 C-2's repair did not touch, and its number goes on the verdict face.
*Attaches to:* §6 `GATE-DOMAIN`; §6.4; §10.2; §3.7's constant table.

`(acc_ro − maj)/(acc_native − maj)` for `endpoint_std` under Head-N. `acc_ro` is a head-space
accuracy on the **arena** (`n = 743`, majority `0.6003`). `acc_native` is `GATE-FLOOR`'s banked
`acc_deployed` (`0.8884 / 0.8858 / 0.8858`), measured on the **full** population (`n = 744`,
majority `0.5995`) with native keys — §3.7's own table says so. One `maj` cannot be right for both
terms, and the design specifies neither. The arithmetic difference is small
(`0.2881` vs `0.2889` in the denominator, `0.3 %`), but §6.4 requires the figure *"on the verdict
face and in §10.2's scope sentence"*, so it is a published number.

The request asks whether any other population-dependent quantity still carries a full-population
value. Round 3 said the majority rate was the only one. **It was not:** this is the second, and
I-3 is the third.

**Repair.** Name both: `maj_arena = 0.6003` in the numerator with `acc_ro`, `maj_full = 0.5995`
in the denominator with the banked `acc_native`; or re-measure `acc_native` on the arena rows so
one constant serves both. Add the choice to §3.7's table and to §13 item 5.

### I-3. `GATE-SMALLDISP`'s quantile threshold is population-dependent and its population is unnamed — and the removed row's displacement norm is exactly `0`, i.e. it would sit **inside** the bottom decile.
*Attaches to:* §6 `GATE-SMALLDISP`; §3.7's contract table.

`small_displacement_train_quantile = 0.1` defines a threshold on the train displacement norms.
Computed over 744 rows it includes row 355, whose displacement block is exactly zero, so the
`0.1` quantile of the 744-row vector is strictly below the `0.1` quantile of the 743-row vector.
The design names the population for the majority rate, `n_D`, `ρ` and the tie cap, but not for
this one.

**Repair.** Add a line to §3.7's constant table: the small-displacement quantile threshold is
computed on the **arena** displacement norms (`743 / 579`). Add it to §13 item 5's list of
population-derived constants the code lineage must confirm are computed, not read.

### I-4. §6.5's tie criterion compares a **key-component** residual against a **similarity** gap. The units do not match, and the correct bound is up to `√d ≈ 45×` larger — so the sharpened diagnostic is still narrower than the mechanism it was added for.
*Attaches to:* §6.5 all four bullets; §6 `GATE-ALGEBRA`; §13 item 10.

`GATE-ALGEBRA`'s residual is `float(np.max(np.abs(views[arm] − guard)))` — a **per-component**
max-abs on the key matrices (`c01_policy_contrast_a0.py:1372-1377`). §6.5's criterion is *"every
pair of neighbours whose signed **similarities** differ by less than the residual."* A key
perturbation `Δk` with `max|Δk| = ε` changes an inner product against a unit query by up to
`‖Δk‖₂ ≤ √d · ε`. At `d = 2048` and the measured `ε = 1.192e-07` that is `5.4e-06`, **45× the
threshold the design applies**. The tie set is therefore ~45× too narrow, which is fail-safe
(more HALTs, never fewer) — but it means round-3 I-5's repair still does not cover the flip
mechanism it was written to excuse, one round later.

Two further under-specifications in the same bullet list: (a) **"collapsing" is undefined** —
whether it means averaging the pair's rank weights, taking the worst-case permutation within each
near-tie group, or something else, and those give different tie sets; this is a preregistration
item, not an implementation choice, so §13 item 10 does not dispose of it. (b) §6.5 quotes the
residuals *"on the raw keys"* (`1.192e-07 / 8.941e-08`), but `GATE-ZEROOP` compares **head-space**
predictions, whose algebra residual is a different, unmeasured number.

**Repair.** Use `‖Δk‖₂` directly (measure it alongside the max-abs residual), or bound the
similarity perturbation as `√d · max|Δk|` and say so. Define "collapse" as *the worst case over
all orderings of the near-tie group*. State that the residual is the head-space one, measured at
run time on the same lineage whose predictions are being compared.

### I-5. §12's `dev_path_opens` is **binding** with an unquantified term. Measured, that term is `0`.
*Attaches to:* §12's `GATE-LEDGER` table; §15.5.

The row reads: expected *"`mints_executed` + `GATE-DEVFID` reads"*, binding *"yes, against
`mints_executed`"*. The second term is never given a number, and a binding predicate with a free
term is not exactly predictable — the property §12 and §15.5 both claim for it. Measured:
`headspace_fidelity.py` opens `mint_{ds}_s{seed}_ffull.npz` (`:66`) and the banked floor trainlog
(`:42`) and **no `dev_seen_*.pt` at all**. It reads `lab_dev` out of the mint `.npz`.

**Repair.** Write the term: `dev_path_opens == mints_executed + 0`, with one clause recording that
`GATE-DEVFID` reads dev labels only via the banked mint `.npz` and opens no `dev_seen` file — and
keep `banked_trainlog_opens` where it is.

### I-6. `GATE-ORBITDISP` has no stated cell granularity or fold aggregator for `ρ_head`, so the gate's own input is undefined.
*Attaches to:* §6 `GATE-ORBITDISP`; §6.1; §8 Phase 2D.

*"Applied to all 13 arms"* and *"`ρ` is computed over the arena rows only"* fix the row set but
not the cell. There are **60** head-space key matrices (C-3), giving `60 × 13 = 780` values of
`ρ_head`. Whether the gate compares each, the per-`(ds, seed, lineage)` maximum, or a fold mean
against `ρ*` is not stated, and the three give different HALT probabilities. §8's `12` implies the
designer has one value per `(ds, seed, lineage)` in mind but names no aggregator.

**Repair.** State it. The conservative and structurally correct choice is **per fold, all 60
cells, HALT if any fires** — a degenerate head space in any fold destroys that fold's OOF
predictions, which enter every decision quantity.

### I-7. `GATE-SELFTEST` says *"every arm"*; §8 prices it at `156 = 13 × 3 × 2 × 2`, which excludes `avg_score` — the 14th arm, and a member of **both** real arms' comparator sets.
*Attaches to:* §6 `GATE-SELFTEST`; §8 Phase 7; §3.5; §5.1.

§3.5 defines *"Thirteen key-space arms **plus one score-derived arm**"*. `avg_score` yields a
prediction vector (§5.1's *"each arm `A` yields one OOF prediction vector"*), is one of C01's five
frozen `gain_controls`, and is therefore a comparator in `C` for both `common_displacement` and
`displacement` and a member of the Holm family. `14 × 3 × 2 × 2 = 168`, not `156`. Either
`GATE-SELFTEST`'s identity is asserted on `avg_score` too (and Phase 7's count is wrong) or it is
not (and *"every arm"* is wrong).

**Repair.** One word in §6 (*"every key-space arm"* or *"every arm including `avg_score`"*) and
the matching integer in Phase 7.

### I-8. §10.2 omits the **strongest** narrowing available to it — that the rotation family is a one-parameter Givens family that **contains the primary**.
*Attaches to:* §10.2; §1's round-14 sharpening.

§1 states it: `orthogonal_blocks()` is a Givens mixing of the two endpoint blocks, `θ = 45°` **is**
`common_displacement`, `θ = 0` **is** `endpoint_concat`. A CLOSE therefore establishes *"the real
displacement is not the best angle on its own one-parameter family"* — not *"the prompt-orbit
tangent carries nothing"*. §10.2's five exclusions are all real and all weaker than this one, and a
CLOSE **will** be read in the stronger sense unless the scope sentence forecloses it. The Gate-0
record itself insists on the same point (*"the adverse reading is that a matched-norm random
direction **can** reach or exceed the real displacement, not that every one does"*), and §1's own
table shows 4 of 6 HateMM rotations and 2 of 6 ZH rotations sitting **below** the primary.

**Repair.** Add a sixth bullet to §10.2 carrying §1's round-14 sharpening verbatim, and state that
the six angles are controls on the primary's **own** family, not independent directions.

---

## MINOR

* **M-1.** §3.7 cites *"`src/model/classifier.py:80-81`"* for the biased projections. They are at
  **`:81-82`** (`:80` is the comment line `# Projection layers prior to modality fusion`). Round 3
  cited `:81-82` correctly; v4 shifted it by one. `:115-124` for the forward is correct
  (`def forward(self, img_feats, text_feats, …)` is `:115`).
* **M-2.** §8's M-1 note (*"the printed product column sums to `2886.2 s`"*) describes v3's table.
  v4's Phase 7 row prints `0.1 s` in the product column, so v4's printed column sums to
  **`2886.3`**. The convention statement is still worth keeping; the sentence needs re-pointing.
* **M-3.** §7.4(a)'s *"`0.58–0.65` observed"* is specific to the `mlp[:-2](l2n(img_proj(·)) *
  l2n(text_proj(·)))` emitter. Under the pre-MLP Hadamard convention I measure
  `0.031354 / 0.030729 / 0.030842` at seeds 0/1/2 — non-zero at every seed, an order of magnitude
  outside the quoted band. Only *"non-zero"* is invariant across emitter conventions, which v4
  almost says; the band should be scoped to the emitter it was measured on.
* **M-4.** §9's granularity list gives *"one per gate"*. `GATE-C01PARITY` runs on two datasets at
  `11.27 s` each, so one line per **gate** leaves a `22.5 s` span (`28.2 s` conservative) — above
  the *"nothing exceeds ~15 s"* claim. Round 3's re-check assumed one line per `(gate, dataset)`.
  Say so. (Also: §3.7's contract table row 3 assigns `GATE-ROWSUBSET` the ZH population, but §8
  Phase 2C correctly marks it **HateMM only**.)

---

# PART C — REQUIRED RULINGS

## Deliverable 6 — **is there any gate that can fire on a warranted CLOSE?**

**YES — two.** I tested all twenty, not only the previously flagged ones.

| gate | can it fire on a warranted CLOSE? |
|---|---|
| `GATE-DET1`, `GATE-SHA`, `GATE-FOLD`, `GATE-FLOOR`, `GATE-POP`, `GATE-C01PARITY`, `GATE-ROWSUBSET`, `GATE-NULLREMOVED`, `GATE-IDPARITY`, `GATE-ZEROMASK`, `GATE-NESTED`, `GATE-SELFTEST`, `GATE-ALGEBRA`, `GATE-LEDGER` | **No** — provenance, population, algebra or bookkeeping; all independent of the science |
| `GATE-ORBITDISP` | **No** — measurement (V8) puts a trained head at `≈ 0.45–0.67` against `ρ* ≈ 0.97`, and the both-above branch is explicitly a warranted CLOSE |
| `GATE-ARENA` | **No** for the `≤ 0.98` upper bound (catches a leak only); **no** for the lower bound, which is scoped to `endpoint_std` and fires only when the head space itself is dead — but see **H-3** on its lineage scope |
| `GATE-ZEROOP` | **No** on the science, though it carries a disclosed false-HALT probability (**I-4**) |
| **`GATE-ARMVIAB`** | **YES — C-1.** Its escape branch requires the raw real arm to fail a bar C01 measured it clearing by `0.18–0.23` |
| **`GATE-SMALLDISP`** | **YES under the §6 reading — H-1.** §5.2 classifies it as a SURVIVE condition and C01 agrees; §6 lists it unannotated among the HALT gates |
| `GATE-DOMAIN`, `GATE-DEVFID` | **No** — they do not gate |

## §15.1 — the mask convention

**Confirmed, and the all-False mask at `n = 743` is right for the stated reason, not by
coincidence.** I executed all six cells (V2, plus measurement (α)). The reason the design gives —
*"the null is physically absent, so the all-False mask is correct"* — is the operative one:
`l2_rows:1192-1194` derives `exact_zero = np.all(array == 0, axis=1)` from the data and demands
`array_equal(exact_zero, zero_mask)`, so at `n = 743` the derived mask *is* all-False and the
argument must match it. Each of §3.7's four objects carries a mask argument correct for its
population. **No `None` survives at any specified `prepare_views` call site.** One addition worth
making: `None` dies at the arena population too (measurement α), which shows the inadmissibility
is a property of the function rather than of the null row.

## §15.2 — the arena majority constant

**Re-derived independently: `446/743 = 0.600269 → 0.6003` and `399/579 = 0.689119 → 0.6891`.**
Both bands are right. `GATE-POP`'s class-count clause **is** sufficient to make the constant
checkable at run time — asserting `(297, 446)` / `(180, 399)` pins exactly the two integers the
majority is a ratio of, so a population change cannot leave the constant behind. **But round 3's
claim that the majority rate was the *only* population-dependent quantity was wrong:** the
`GATE-DOMAIN` recovery fraction (**I-2**) and the small-displacement quantile threshold (**I-3**)
are both population-dependent and both unnamed. `GATE-POP`'s clause does not reach either.

## §15.3 — `GATE-ZEROOP`'s `1 %` cap

**A cap is needed, and `1 %` is defensible. I do not raise it as a finding.** v3 had no cap, so a
systematic defect that happened to leave near-ties would have been *reported* rather than HALTed —
the cap closes that. Its direction is what makes it safe: the cap can only ever convert
REPORT → HALT, never HALT → REPORT, so it cannot cause a wrong verdict in either the SURVIVE or
the CLOSE direction; it can only cause the falsifier not to publish. That is a one-sided,
disclosed engineering choice and it does not need a banked derivation. `≤ 7` and `≤ 5` are the
correct floors of `0.01 × 743` and `0.01 × 579`. **One clause should be added to §5.8 item 5**
making the one-directional property explicit, so a future round does not mistake the cap for a
threshold touching a decision. The real defect in §6.5 is the residual's units (**I-4**), not the
cap.

## §15.4 — S6's retention

**Ruled against v4's disposition. S6 is not vacuous, and the document must stop saying it is.**
See **C-2**. The generic question the request poses — *does a condition that cannot fail belong in
a SURVIVE conjunction?* — is moot here, because I-4's per-seed pinning made S6 a condition that
**can** fail. My ruling on the general principle, for the record: a genuinely non-binding condition
should be reported, not conjoined, because a conjunction that a reader believes is tightening and
is not misstates the rule's strength. But the tension the request names (`GATE-SELFTEST` needs an
object; the Gate-0 record demands the net-item currency) dissolves — `GATE-SELFTEST` asserts its
identity on every arm regardless of whether S6 screens, and the net figure is reported either way.
Take repair (a): keep S6 binding, and rewrite §5.8 item 4 to state the true relationship (S3
bounds the seed **mean**; S6 binds through across-seed dispersion).

## §15.5 — §12's binding dev counts and the resume path

**The pair is *not* exactly predictable under every legal path, on two counts, and one other thing
in the design does assume a fresh run.**

* The `dev_path_opens` expectation has an unquantified second term; measured, it is `0`
  (**I-5**).
* `GATE-FOLD`'s discharge is attributed to code a resumed mint returns before reaching
  (**I-1**). This is the direct answer to *"does anything **else** in the design assume a fresh
  run?"* — yes, `GATE-FOLD` does, via §3.2.
* The three the request names are fine: §8's projection explicitly assumes fresh and a resume
  costs strictly less; `GATE-LEDGER`'s `66 + 6 + 1` holds because a skipped mint still runs as a
  process and can still emit its heartbeat line; §5.6's absence rule is satisfied by the separate
  `mints_present_before_arena == 66` assertion.

**Is binding these counts the right call, given C09 merely reported them?** **Yes.** §5.6's
absence rule needs a binding process/artifact count to close the lane where a silently missing
lineage makes SURVIVE vacuously false and thereby supplies half of CLOSE. C09 had no lineage
disjunction and so had no such lane. Binding is the right strengthening; the defect is only that
two of the bound quantities are not yet exactly predictable.

## §3.B — the design's self-caught §12 defect

**The `mints_executed` + `66 .npz` pair is the right instrument, and the reasoning is sound.**
`headspace_mint.py:192-194` does return before `:199`, so a literal `== 66` would HALT a resume;
binding against the measured `mints_executed` is exactly predictable on fresh, resumed and
partially-resumed runs, and a re-run after a HALT is the resumed case. The self-catch is the kind
of finding a design lineage exists to produce and it is disclosed in the right place. It is
incomplete only in the two respects above.

## §3.C — the decision rule

**§5.5's multiplicity is right and I re-derived it** (V10): the false-positive event is a
disjunction over 4 `(arm, lineage)` disjuncts, one Holm family per dataset of
`23 × 2 metrics × 2 lineages = 92` covers exactly those disjuncts' bootstrap legs, the datasets
stay a conjunction (tightening), and S5's shuffle rejections are correctly outside the family
because they are conjunctive within each disjunct. **M-4's two coincident `92`s are two genuinely
different products** and must not be reconciled.

**§5.6's finiteness, absence and `RuntimeError` rules are sound as far as they go, with one gap.**
The first bullet asserts finiteness of every **gate** quantity; the second makes **absence** of a
decision-or-gate quantity a HALT. **Non-finiteness of a *decision* quantity is covered by
neither** — which is the C09-lineage class the request names. The concrete instance is
`GATE-SMALLDISP`'s `fixed_fraction` at zero fixes (**H-1**); with S4's statistic unwritten
(**H-2**) there may be others. Extend the first bullet to *"every gate **and decision**
quantity."*

**The sharpened tie criterion does cover the in-set reordering mechanism round 3 identified** —
collapsing near-tied adjacent pairs and recomputing the rank-weighted vote matches
`mechfix_ops.py:94`'s operator rather than a boundary, which is the right move. It is defeated by
the units error (**I-4**), not by the concept. The report-not-HALT branch cannot be reached by
anything outside the tie set: §6.5's last bullet is written as a closed condition
(*"any mismatch outside them HALTs"*) and the cap is one-directional.

## §3.D — gates and scope

Answered in the table above. `GATE-ROWSUBSET`'s renaming **is** an honest resolution of round-3
I-1: the gate is cited as strictly stronger at the key level, the over-claim on C01's name is
gone, and the statement that C01's vote-level property *has no object here* is correct — I
confirmed that none of the 13 arms is ever voted at `n = 744`: §8 Phase 2C runs
`GATE-C01PARITY`/`GATE-ROWSUBSET` as key comparisons only, and Phase 2's `240 / 540` arm votes are
all at arena scale — the sole full-`n` vote is `GATE-FLOOR`'s 30 native deployed-key votes, which
score no arm. §10.2 is nearly complete;
**I-8** names what it is missing. **Hard constraints: none touched** — I re-read
`iteration_8_stage0_bounded_extraction_amendment.amended_rule.conditions.d_no_other_relaxation`
verbatim and checked each: no OCR; no cross-dataset mixing (the two-dataset requirement is a
conjunction of independently computed verdicts and no pooled object remains); no external API;
single-dataset train split; parent-video binary label only; no ensemble (`avg_score` is C01's own
frozen `gain_control`); no size scaling; SLURM-only, no `--time`. §10.4's ban analysis is unchanged
and correct against the Gate-0 record's own `why_gated_not_struck` text, which I read verbatim.
§10.3's statement that a SURVIVE authorises no GPU is correct against condition `a` of the
amendment.

## §3.E — the process rules

**`rule_1_compute_projection`: NOT discharged — C-3.** The round-3 fixes *are* compute-neutral, as
v4 claims: a calling convention and a constant add no unit, and I re-multiplied every row to
confirm the total is unchanged at `2886.3`. The new `GATE-POP` class-count clause, the
`RuntimeError` wrapper and the sharpened tie diagnostic all carry sub-`0.1 s` cost inside Phase 7's
class, correctly. **But the hunt for an uncounted loop succeeds this round**: the fold axis is
missing from Phases 2b and 2D, and `GATE-ZEROOP`'s guard arms are missing entirely. Corrected
total `2925.0 s` / `3656.3 s`.

**`rule_2_heartbeat`: adequate, with two gaps.** The `RuntimeError` wrapper does not change any
interval — it fires once, at exit, and §5.6/§9/§13 item 12 specify it consistently and completely.
The gaps are consequences of C-3 and of §9's own granularity list: Phase 2D at the corrected count
is a `38.4 s` un-instrumented span under *"one per gate"*, and `GATE-C01PARITY` is `22.5 s` across
its two datasets (**M-4**). §13 item 12 is otherwise complete — it carries all six §9 items plus
the wrapper, the `buffering=1` handle, the unbuffered driver echo, the append-without-interleaving
requirement across all 73 processes, and the frozen denominator.

**§7.7's `U9` disclosure and the residual question.** The correction is sound and in the right
place. On *"could any remaining unit carry the same defect?"*: I independently reproduced the
outputs of `U5a`, `U5b`, `U6` and `U10`'s object (V4, V5, V6, V7), so those processes certainly ran
and produced what is claimed, and fold parity is banked inside all 36 mint `.npz` I opened.
`U2a`–`U2d`, `U3`, `U4`, `U7`, `U8` and `U11` remain uncorroborated, exactly as round 3 found.
v4's commitment that *"the freeze record will state the exit-status discipline under which each
was timed"* is the right instrument and should be held to at freeze. One addition: `U4` is the
single largest uncorroborated unit (`273.7 s`, `9.5 %` of the total) **and** the one whose space
is ambiguous — §7.7 does not say whether a shuffled-pair draw was timed in the 1024/2048-d head
space or the 7168/14336-d raw space, and Phase 3 applies it only to head cells. Name the space.

## §3.F — honesty

**Does v4 claim any repair the artifact does not contain?** **No.** Every one of the 13 round-3
findings and all 4 Minors is physically present in the text. The single failure is different in
kind and worse: **I-3's repair is present but its stated arithmetic is false**, broken by the
adoption of I-4 in the same round (**C-2**). That is not a claimed-but-absent repair; it is two
adoptions that were not checked against each other.

**M-2 and M-3 are now stated as ranges rather than as unreproducible digits** — §7.4(a) gives
`0.58–0.65` with the initialisation-dependence stated, and §7.5 records all three rounds' values
for the one-block `paired` and rests the claim only on the invariant. Both adoptions are real.
See **M-3** on the emitter scope of the `0.58–0.65` band.

**§7.3's blindness discipline is intact.** I grepped every decimal in `[0.6, 0.99]` across the
draft: every one is a `ρ`, a banked `GATE-FLOOR` anchor, a C01 published dev-arena accuracy from
§1's table, or a majority/band constant. No accuracy produced by this battery exists anywhere in
v1–v4, and §6.1's trained-head `ρ` reference reads `K_train` only — I reproduced it and it
computes no accuracy.

---

# PART D — DISPOSITION AUDIT OF §14 (round 3: 13 findings + 4 Minor)

Each row checked against the primary source or by execution, never against §14.

| finding | v4 claim | audit result |
|---|---|---|
| **C-1** `zero_mask = None` inadmissible | ADOPTED | **VERIFIED ADOPTED.** I executed all six mask cells (V2) plus a seventh v4 does not report (α). The convention is stated once at §3.7 and used at every call site; §6's `GATE-C01PARITY` and `GATE-ROWSUBSET` both carry explicit arrays; §7.4(i)(j) record what was executed; the `l2_rows`-vs-`prepare_views` asymmetry is cited to the right lines and is true in source. I grepped the whole draft: **no residual `None` at any `prepare_views` call site.** |
| **C-2** 744-majority on the 743 arena | ADOPTED | **VERIFIED ADOPTED.** `0.6003 / 0.6891` re-derived exactly; bands `[0.6203, 0.98]` / `[0.7091, 0.98]`; `GATE-ARMVIAB` cites the arena constant; `GATE-POP`'s class-count clause `(297,446)`/`(180,399)` is present and is sufficient for the constant it gates. `0.5995` survives only in population-scoped statements. **Two *other* population-dependent quantities were missed by the sweep: I-2, I-3.** |
| **H-1** §12's Head-R dev sentence false | ADOPTED | **VERIFIED ADOPTED.** §12 now says all 66 mints open the native `dev_seen`, substitutes the true statement (*Head-R opens no `dev_seen_*-ro_*`*), and makes both counts binding. I confirmed `:192-194`, `:199`, `:203-216`, `:223-226` and `:322-324` in source. §3.3's *"only variable"* sentence is retained and is now consistent. **Residual: I-5 (unquantified term) and I-1 (`GATE-FOLD` under resume).** |
| **I-1** `GATE-DUALPATH` wears C01's name | ADOPTED | **VERIFIED ADOPTED.** Renamed `GATE-ROWSUBSET`, cited as strictly stronger at the key level, with the explicit statement that C01's vote-level property has no object here — which I confirmed (no arm is voted at `n = 744`). |
| **I-2** `ρ*` truncation exempts `endpoint_std` | ADOPTED | **VERIFIED ADOPTED, by execution.** `ρ*` frozen at `0.968176 / 0.977223`; `ρ_raw ≤ ρ*` now holds by **equality** for `endpoint_std` on both datasets, so the structural exemption is gone; both §6.1 tables quote the same digits; all 26 values reproduce at 6 dp. |
| **I-3** S6 vacuous at arena scale | ADOPTED | **ADOPTED IN TEXT, ARITHMETIC FALSE.** §5.8 item 4 exists and carries the scale-transfer story, but its central claim (*"S3 implies S6 by arithmetic"*) is untrue on the per-seed axis I-4 pinned in the same round, and §5.2 acts on it. → **C-2**. |
| **I-4** `GATE-SELFTEST`'s `n`, S6's axis | ADOPTED | **VERIFIED ADOPTED** — `n_D = \|arena(D)\|` pinned to §3.7's table in both §5.1 and §6; S6 defined on the per-seed integer net in 3/3 seeds. This is the adoption that falsifies I-3's argument. Residual: **I-7** (`avg_score` and the `156`). |
| **I-5** tie diagnostic wrong boundary | ADOPTED in full | **VERIFIED ADOPTED.** All four prescriptions are present: union of the two arms' top-21, max of the two residuals, collapse-and-recompute against the operator, and a cap. The operator match is right — `mechfix_ops.py:94` weights by `sim × w`, so an in-set reorder does move the vote. **Residual: I-4 (units, "collapse" semantics, raw-vs-head residual).** |
| **I-6** `die()` is a crash, not a gate result | ADOPTED | **VERIFIED ADOPTED.** §5.6's third bullet, §9's HALT-line clause and §13 item 12 all specify the wrapper, the `INSTRUMENT_INCONCLUSIVE` record, the `context` string and both destinations. `die:392-393` is `raise RuntimeError(message)` as cited, and `l2_rows`' `context` does carry arm and block name. |
| **M-1** Phase 7 rounding | ADOPTED | **VERIFIED ADOPTED**, but the adopting sentence is now stale against its own table (**M-2**). |
| **M-2** `‖head(0,0)‖` six digits | ADOPTED | **VERIFIED ADOPTED** — stated as non-zero at every seed and emitter convention tested with a range, not six digits. See **M-3** on the range's emitter scope. |
| **M-3** one-block `paired` digits | ADOPTED | **VERIFIED ADOPTED** — all three rounds' values recorded (`1.118e-08 / 7.451e-09 / 3.725e-09`) with the weight-dependence stated and only the invariant claim load-bearing. |
| **M-4** the two `92`s | ADOPTED | **VERIFIED ADOPTED** — §5.5 distinguishes them explicitly and forbids reconciliation. I verified both products independently (V10). |

**Round-3's extra recommendation** (cite the trained-head `ρ` measurement in §6.1) is adopted, and
I reproduced it to the digit on all 36 banked mints (V8).

**Rounds 1 and 2, spot-checked rather than re-audited** (round 3's sweep was clean and this round
found no reason to reopen it): the direction of *"conservative"* (§4), A7, per-arm retraining
excluded, `max` as `ρ*`'s order statistic, SLURM and the login-node dismissal, HALT semantics,
§5.8 item 1's inapplicability reasoning, and §3.4's account of what two-block parity does and does
not buy — all still present and still sound. §3.4's claim that the block count is *"forced by the
head's architecture"* is confirmed at `classifier.py:116-120` (a single fused 1024-d vector), and
I confirmed the head-space key dims directly.

---

# PART E — FREEZE-READINESS AND THE RUN BOUNDARY

**Every file the document says exists, exists**, and every digest recomputes. `vsw_ckpt/{hatemm,
zh}/f{0..4}.npz` — all ten present. `headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json` — all six
present, at `scripts/analysis/`. `artifacts/c09_topo/v1/a0/C09-A0-v1/scratch/mint_*.npz` — exactly
**36**. All four items of new code (`c06_falsifier_mint.py`, `c06_falsifier_arena.py`,
`configs/c06/c06_falsifier.json`, `scripts/slurm/c06_falsifier_cpu.sbatch`) are **absent**, as they
must be. `TARGET_STATE.json` quotations at §1 (`falsifier_spec`, `falsifier_design_constraints`)
and §1's evidence table are verbatim-correct against `gate0_reopen_2026_07_31.dispositions.gated[0]`
and `GATE0_REOPEN_2026-07-31.md` §4.4. Every C01 constant §5 and §6 cite reproduces from
`configs/c01/c01_a0_v2.json`: `minimum_gain_over_strongest_control 0.02`, `gain_controls` (5),
`primary_vs_controls` (6), `minimum_net_fixes {HateMM: 3, MHC_zh: 2}`, `holm_alpha 0.05`,
`n_bootstrap 2000`, `statistics.seed 20260728`, `bootstrap_lower_quantile 0.05`,
`small_displacement_train_quantile 0.1`, the six angles, `topk 20`,
`rank_weights descending_integer`, `fix_break_reference endpoint_std`, `normalization_epsilon
1e-12`, `feature_dim 3584`, `standard_suffix ro_L24`, `oneword_suffix ro_ow_L24`,
`expected.train.n 744/579`, and `execution {require_slurm, cpu_only, required_cpus 8}`.

**What is not yet freeze-ready.** Three things an operator with no context cannot execute
unambiguously as written: (1) whether `GATE-SMALLDISP` HALTs or merely denies SURVIVE (**H-1**);
(2) what S4's p-value is (**H-2**); (3) whether S6 screens (**C-2**). Two more that an implementer
would have to invent: the `ρ_head` cell/aggregator (**I-6**) and the tie-collapse semantics
(**I-4**). The **run boundary** — one `sbatch`, 8 CPU / 32 GB, no `--gres`/`--time`/array/
dependency/requeue, 73 processes in the order 66 mints → 6 fidelity → 1 arena, `GATE-SHA` once in
the driver before any of them, `GATE-POP` before any population-consuming gate — is otherwise
unambiguous, and the cloud-routing dismissal is correct (`GATE-FLOOR` anchors to six floors
measured locally on `foscsmlprd01`).

# PART F — ADDITIONS TO §13's CODE-LINEAGE HANDOFF

The twelve items are good and should be kept verbatim. Six to add:

13. **The fold axis.** That the 13 head-space arms and every `ρ` are rebuilt from **each** of the
    60 fold key matrices, and that no arm built under head `f` is ever voted for a query outside
    fold `f`'s held-out fifth (`GATE-NESTED` is the run-time check; this is the code check).
14. **The `GATE-ZEROOP` guard arms.** That `orthrot_0` and `orthrot_45` are built by the *rotation*
    route and `endpoint_concat` / `common_displacement` by their own, that the two are never
    aliased to one object, and that all four are voted.
15. **`GATE-SMALLDISP`.** Its classification (SURVIVE condition, not HALT), its arm scope, and the
    zero-fix convention (`fixed_fraction := 0`, `dominated := false` when `fixed == 0`), which the
    battery does not inherit from `c01_policy_contrast_a0.py:1989-1996` because it does not import
    `displacement_audit`.
16. **The statistics.** That the bootstrap statistic, the one-sided p and the Holm step-down match
    what §5.4/§5.5 pre-register once they say it, and that no decision quantity — not only no gate
    quantity — reaches a comparison non-finite.
17. **Population-derived constants, extended.** Item 5's list must also cover the `GATE-DOMAIN`
    recovery fraction's two majorities and the small-displacement quantile threshold's population.
18. **`GATE-FOLD` under resume.** That fold parity is verified for every one of the 66 mints,
    including skipped ones, by reading `meta["fold_parity_vs_banked_vsw_ckpt"]` and `fold_of` from
    the banked `.npz`.

---

# PART G — MINIMAL SET OF CHANGES THAT WOULD EARN GO

1. **C-1** — restrict `GATE-ARMVIAB` to `endpoint_std`, or replace its discriminator with the
   head-space controls. One paragraph in §6.2.
2. **C-2** — keep S6 binding and rewrite §5.8 item 4 to state the true seed-mean/per-seed
   relationship; delete *"S6 is reported, not screening"* from §5.2.
3. **C-3** — Phase 2b `12 → 60`, Phase 2D `14 → 62`, add the guard-arm row, re-multiply, and add a
   Phase 2D heartbeat line.
4. **H-1 … H-3** — classify `GATE-SMALLDISP`; pre-register S4's statistic; declare or repair the
   global-HALT/per-lineage-SURVIVE asymmetry.
5. **I-1 … I-8** as written, and **M-1 … M-4**.

**Can the falsifier still discharge the written condition at `$0`? Yes.** Nothing I found requires
a GPU, an extraction, new data or a redesign, and the corrected projection is `48.8` corroborating
/ `60.9` conservative CPU-minutes on 8 CPU / 32 GB. The three-population contract, the mask
convention, the row-subset identity, the arena majority, the `ρ*` calibration and the two-block
anchor are all correct and all verified by execution this round. What is not yet right is one
gate's discriminator, one arithmetic claim about the decision rule, and one enumeration — and
after four rounds the remaining defects are, for the first time, all in the *decision layer* and
the *accounting layer* rather than in the instrument.

---

*Read-only review. No GPU, SLURM, Modal, model load, head training, arena run, cache write,
test-split access, job submission or commit occurred. `TARGET_STATE.json`, all four drafts, all
configs and all prior reviews were read and not modified. All computation was `sha256sum`, file
reads, and numpy/torch-CPU re-derivation on already-banked **train-split** caches and banked mint
checkpoints. No arm accuracy was computed at any point. A GO on this lineage would authorise
nothing to run: the design would still require freeze with hashes, a **separate** independent
code/resource review lineage over the executable reaching its own `0C/0H/0I`, and main-dialogue
authorization.*
