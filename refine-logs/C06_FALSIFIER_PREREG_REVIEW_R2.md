# C06 `$0` falsifier — independent design review, **ROUND 2**

**Target:** `refine-logs/C06_FALSIFIER_PREREG_DRAFT_V2.md` (DRAFT v2, 2026-08-04)
**Reviewer posture:** fresh, independent of round 1 and of the designer. Read-only. No GPU,
SLURM, Modal, model load, head training, arena run, cache write, test-split access, job
submission or commit occurred. `TARGET_STATE.json`, the drafts and all configs were read and
not modified. Nothing heavier than `sha256sum`, file reads, and small numpy re-derivations on
the already-banked **train-split** caches was executed.

---

# VERDICT

## **REVISE (3C / 3H / 7I)** — plus 3 Minor

Not `GO (0C/0H/0I)`.

**Disposition audit: 20 of 23 round-1 findings verified as truly adopted. Three are not** —
**C-3**, the **C-1 companion**, and **I-3** (details in Part D).

v2 is a substantially better document than v1 and the ceremony floor is clean: **all 21
sha256 digests reproduce**, every state-file quotation is verbatim, the config and algebra
provenance chains are real, the ρ table reproduces to 4 dp on all 26 values, the `GATE-FLOOR`
macro-F1 anchors match the banked files exactly, the Holm family arithmetic is right, and
`GATE-C01PARITY`'s bit-exactness claim is **true** — I re-implemented §3.4's builder spec from
the document alone and obtained `max|diff| = 0.000e+00` on all 13 arms on both datasets.

The three Criticals are not ceremony, and none of them is a round-1 finding recurring. All
three are **new failure modes created by v2's own repairs**, and I found them by executing
C01's frozen code rather than by reading the document. Two share one root cause: **C01's
zero contract is defined on a property the head destroys.** `nn.Linear` carries a bias
(`src/model/classifier.py:80-81`), so `head(0,0)` is a **non-zero constant**, identical under
both policies. In head space the registered null row 355 therefore makes the *displacement*
block exactly zero while every *endpoint / common / rotation* block is non-zero — and C01's
`l2_rows` has a fail-closed assertion (`:1193-1194`) that no single `zero_mask` can satisfy
across those arms. I measured it: **C01's primary arm `common_displacement` cannot be built
in head space on HateMM under either mask choice.** The third Critical is a gate contradiction:
`GATE-ARENA`'s lower bound, applied to the two real arms, fires on precisely the outcome
§6.2 argues must not trigger a HALT — so v2 refines `GATE-ARMVIAB` to avoid a self-defeating
gate and then re-introduces the identical self-defeat one row above it in the same table.

---

# PART A — INDEPENDENT VERIFICATION OF ALL TWELVE §2 ITEMS

| # | result | what I obtained |
|---|---|---|
| **V1** | **VERIFIED** | All **21** digests recomputed with `sha256sum` and matched character-for-character: 7 imported modules, 6 read-for-definition files, 8 input caches. No mismatch. |
| **V2** | **VERIFIED** | First-16 hex `6a44cce4f65d4a60` / `60054f3be1204ca7` (HateMM) and `1d33fe5d69083479` / `3ad1309dc7500182` (MHC_zh) equal `c01_a0_v2.json::inputs.datasets.<ds>.expected.train.{standard,oneword}_provenance_sha16`. HateMM `ro_L24` full 64-hex `6a44cce4f65d4a60b8863969a2ad9ed72731658cea49e75f487a911000be045f` equals `c01_a0_v3.json::lineage_evidence.diagnostic_train_cache_sha256` **in full**. |
| **V3** | **VERIFIED** | Config chain: `v4.frozen_v3` → `configs/c01/c01_a0_v3.json` (`4ddb0f6f…`, matches disk) with `reuse_policy: sha256_exact_import_then_v4_audit_schema_override_only`, `scientific_thresholds_exact: true`; `v3.scientific_base` → `configs/c01/c01_a0_v2.json` (`f3997bdd…`, matches disk), `scientific_thresholds_exact: true`. Algebra chain: `c01_policy_contrast_a0_v4.py:44-52` `load_frozen_v3()` sha-checks `_v3.py` against `40b35eee…`; `_v3.py:29-50` `load_frozen_base()` sha-checks `c01_policy_contrast_a0.py` against `d2b9c2ff…`. Both hops sha-gated in code, not just in config. |
| **V4** | **VERIFIED (one truncation)** | Every product re-multiplied: `15×40.39=605.85→605.9`; `3×49.30=147.9`; `15×34.40=516.0`; `3×38.87=116.61→116.6`; Σ **1386.4** ✓. `15×37.46=561.9`; `15×27.54=413.1`; Σ **975.0** ✓. `(30×3)+(6×4)+(30×2)=174`; `174×0.0461=8.02→8.0` ✓. `66×0.033=2.178→2.2` ✓. `240×0.00305=0.732→0.7`; `540×0.00629=3.397→3.4`; `30×0.00305=0.0915→0.1`; Σ **4.2** ✓. `40×0.04239=1.696→1.7`; `90×0.08098=7.288→7.3`; Σ **9.0** ✓. `2×11.27=22.54→22.5` ✓. `1.24+2.48=3.72→3.7` ✓. `3072×0.08908=`**`273.654`** — §8 records `273.6`, a **truncation** where every other product is rounded (→ M-1). `92×0.126=11.592→11.6` ✓. `(4.2+9.0+273.6+11.6)/2=149.2` ✓ (arithmetic correct; the *method* is I-2). `3×3.70=11.1`, `3×3.49=10.47→10.5`, Σ **21.6** ✓. **Total 2867.1 s = 47.785 min → 47.8** ✓; `×1.25 = 3583.875 → 3583.9 s = 59.73 min → 59.7` ✓. Sensitivity `2867.1+273.6=3140.7 s=52.3 min ≈ 52` ✓; `2867.1+4×273.6=3961.5 s=66.0 min` ✓. Phase-3 share `273.6/2867.1=9.54 %` ✓. Mint share `2361.4/2867.1=82.4 %` ✓. Excluded variant `195×40.39+195×34.40=14 584.05 s=4.051 h` ✓. |
| **V5** | **VERIFIED** | `decision.gain_controls` = 5 `[endpoint_std, endpoint_ow, avg_score, endpoint_concat, common]`; `statistics.bootstrap_comparisons.primary_vs_controls` = 6 (those plus `displacement`). Union for `common_displacement` = **6**; for `displacement` = **5**. `6+6=12`, `5+6=11`, `23 × 2 metrics = **46**` per `(dataset, lineage)`; §8 Phase 4's `23 × 2 ds × 2 lineages = 92` follows. |
| **V6** | **VERIFIED** | Re-read from all six `scripts/analysis/headspace_arena_{hatemm,zh}_s{0,1,2}_OUT.json`: `acc_deployed` HateMM `0.8884 / 0.8858 / 0.8858`, ZH `0.8929 / 0.8895 / 0.8946`; `mF1_deployed` HateMM `0.8838 / 0.8811 / 0.8812`, ZH `0.8747 / 0.8710 / 0.8765`. Identical to C09's banked table (`C09_A0_V17_RECORD.md:1511-1522`). |
| **V7** | **VERIFIED** | From the train labels: HateMM `posrate 0.4005` ⇒ majority **`0.5995`**; MHC-ZH `posrate 0.3109` ⇒ majority **`0.6891`**. |
| **V8** | **VERIFIED — all 26 values reproduce at 4 dp** | Computed by importing `c01_policy_contrast_a0`, calling `prepare_views` on the real raw L24 caches, masking the exact-zero row, and taking `ρ = ‖mean_i k_i‖` over unit keys. `endpoint_std` `0.968176/0.977223`; `common` `0.964446/0.969686`; `orthrot_83p8` `0.956893/0.964384`; `orthrot_72p7` `0.956491/0.965058`; `endpoint_concat` `0.955291/0.962418`; `orthrot_8p3` `0.951438/0.958355`; `orthrot_60p4` `0.948430/0.958728`; `orthrot_17p6` `0.944759/0.951882`; `endpoint_ow` `0.942230/0.947382`; `orthrot_29p1` `0.933575/0.941849`; `common_displacement` `0.928799/0.939863`; `common_interaction` `0.913840/0.968188`; `displacement` `0.891728/0.909063`. **Max = 0.977223 → 0.9772** (`endpoint_std`, MHC-ZH), 13 arms, unit-norm verified on every arm. §6.1's table is exact. |
| **V9** | **VERIFIED** | `c01_a0_v2.json::decision` contains all six: `minimum_net_fixes {MHC_zh: 2, HateMM: 3}`, `require_primary_and_displacement_above_shuffle_p95`, `require_shuffle_holm_reject`, `require_rotation_bootstrap_holm_reject`, `require_no_small_displacement_dominance`, `require_accuracy_gain_over_deployed_r0_context`. |
| **V10** | **VERIFIED — v2's withdrawal is correct and complete** | `c02_a0_mint.py:214` is `keys[view] = keys_of(tr[1], view_text[view])` — `tr[1]` is the **native** `img_feats`, reused on every view. `:68` is `assert "img_feats" not in d, "view file must not carry img_feats"`. Docstring `:21-23` states it. `:69-71` is the `ids` unwrapping and `:72-76` the parity assertions, so **M-4** is correctly repaired too. |
| **V11** | **VERIFIED** | Median `cos(native_img, ro_L24_img)` = **`0.0234`** (HateMM) / **`0.0373`** (MHC-ZH); text `0.2300` / `0.2495`. Both caches unit-norm. Reproduces round 1's measurement (d) exactly. |
| **V12** | **VERIFIED — v2's withdrawal is correct** | `headspace_mint.py:199` `dv = load_split(cache_dir, "dev_seen", model_name)` — unconditional, every mint. `:322` `lab_dev=dv[3].numpy().astype(int)` inside `np.savez` — every `.npz`. `:229` the `else` branch (`a.fold < 0`) sets `dev_sp = dv`, so at `fold = −1` the real dev split **is** the training dev set. |

## Additional measurements I made, which v2 does not report

**(α) `GATE-C01PARITY` reproduces — independently.** I implemented §3.4's three formulas
(`fuse`, `paired`, `build_views`) from the document text alone, calling C01's imported
`l2_rows`, and compared against `prepare_views` on the real raw L24 features:
`max|diff| = 0.000e+00`, **bit-identical on all 13 arms, both datasets**, `dtype float32`.
That the spec is reproducible by a third party from the prose is meaningful evidence the
builder is fully specified.

**(β) C01's own algebra guard, re-measured on the same call:**
`endpoint_concat_vs_theta0_max_abs = 8.941e-08` (both datasets);
`common_displacement_vs_theta45_max_abs = 1.192e-07` (HateMM) / `8.941e-08` (MHC-ZH).
Matches `rotation_family_precision_R14`'s *"8.9e-08 to 1.2e-07"* and §1's quotation.

**(γ) The one-block instantiation, measured** (`n = 744`, `d = 1024`, float32):
`fuse([b])` differs from `l2(b)` by `1.490e-08`; one-block `paired(A,B)` differs from v1's
rejected `pair(A,B) = l2(concat(l2 A, l2 B))` by **`7.451e-09`** — one sixteenth of a float32
eps (`1.192e-07`). The outer normalisation C-3 restored is a **re-normalisation of an
already-unit vector**, i.e. numerically vacuous at one block. → Part C, sharpest question.

**(δ) The head-space zero-mask, measured against C01's `l2_rows`.** With a row whose value is
a non-zero constant identical under both policies (what `head(0,0)` is):

| block | `zero_mask = {null}` | `zero_mask = None` |
|---|---|---|
| endpoint (`h_std`) | **DIES** — *"exact-zero mask diverged from authorized mask"* | OK |
| common (`h_std + h_ow`) | **DIES** | OK |
| displacement (`h_ow − h_std`) | OK | **DIES** |
| **`common_displacement = paired(common, displacement)`** | **DIES** | **DIES** |

→ **C-1** below. `nn.Linear(image_dim, map_dim)` (`src/model/classifier.py:80-81`) carries the
default bias, so `img_proj(0) = b_img ≠ 0` and `head(0,0)` is a non-zero constant.

**(ε) Per-process interpreter + import cost, measured on this node:**
`python -c "import torch, numpy, faiss, sklearn.model_selection"` = **3.35 / 3.40 / 3.02 s**
over three runs. → **I-1**.

---

# PART B — FINDINGS

## CRITICAL

### C-1. The head-space arm builder **cannot be executed** through the imported `c01_policy_contrast_a0.l2_rows` on HateMM. C01's **primary arm is unbuildable** under either mask choice, so C-3's repair does not reach the path that renders the verdict — and §3.4 claims it does.
*Attaches to:* §3.4; §6 `GATE-C01PARITY`, `GATE-ZEROMASK`; §14 row **C-3**; §15.

§3.4 states that the battery *"defines **one** builder … in which every normalisation is
C01's `l2_rows` called through the **imported** `c01_policy_contrast_a0` module"*, and that
*"the one-block answer is now **derived by the same code path**, not chosen"*. §14 records
**C-3** as `ADOPTED`. That is the whole content of the C-3 repair: putting the
verdict-rendering code path under a fidelity anchor.

**It cannot run.** `l2_rows` (`:1193-1194`) is fail-closed on the mask:

```python
exact_zero = np.all(array == 0, axis=1)
if not np.array_equal(exact_zero, zero_mask):
    die("{} exact-zero mask diverged from authorized mask".format(context))
```

`classifier_hateClipper.__init__` (`src/model/classifier.py:80-81`) builds
`img_proj = nn.Sequential(nn.Linear(image_dim, map_dim), …)` — **with the default bias** — so
`head(0, 0) = mlp(normalize(b_img) ⊙ normalize(b_text))` is a **non-zero constant**, and
because HateMM row 355 is bit-identically zero in both modalities of **both** ro caches,
`h_std[355] == h_ow[355]` exactly. Therefore, in head space:

* every **endpoint / common / rotation** block is **non-zero** at row 355 → requires
  `zero_mask = None`;
* the **displacement** block is **exactly zero** at row 355 → requires `zero_mask = {355}`.

Measurement (δ): each choice kills the other, and **`common_displacement` — C01's primary,
and one of the two arms `R` whose comparison renders the verdict — dies under both.** So does
every `orthrot_θ` arm, which shares `paired`'s internal normalisations.

Both branches are unacceptable:

* **implemented as written** ⇒ the run HALTs on HateMM at the first head-space arm ⇒
  `INSTRUMENT_INCONCLUSIVE` ⇒ the `$0` falsifier can never discharge its written condition;
* **worked around** by a local normaliser without `l2_rows`' assertion ⇒ the one-block path is
  no longer the gated path, `GATE-C01PARITY` anchors only the two-block raw instantiation, and
  C-3 silently reverts to v1's state while §3.4 and §14 assert the opposite. By this review's
  own bar — *"any claimed repair the artifact does not contain"* — that is Critical.

Note that §6's `GATE-ZEROMASK` shows the designer half-saw this: its second clause is *"those
rows give identical head keys under both policies"*, which is exactly the observation above.
The consequence for `l2_rows`, for `derived_masks`, and for which population the head-space
arena runs on was not followed through.

**Repair.** Pre-register the head-space null handling explicitly, and re-scope what depends
on it. The cleanest option, and the one that keeps C01's semantics rather than approximating
them: make the **physically-removed** path (`n = 743`) the **main** head-space path on HateMM
rather than a Phase-5 sensitivity leg, so `zero_mask = None` is correct for every head-space
block and the imported `l2_rows` runs unmodified on all thirteen arms. Then state which
population each of the raw leg, the head leg and the six banked floors is computed on, so
`ρ_raw` vs `ρ_head` and `GATE-ARMVIAB`'s raw-vs-head accuracy comparison are on the **same
rows**. If instead the masked path is to remain primary, the design must pre-register the
exact wrapper used, say why it is not `l2_rows`, and say what anchors it — at which point
C-3 needs a different repair, not this one.

### C-2. `GATE-ARENA`'s lower bound is applied to the two real arms, so the **expected, warranted CLOSE fires a HALT** — and it silently overrides the `GATE-ARMVIAB` refinement v2 asks round 2 to rule on.
*Attaches to:* §6 `GATE-ARENA` and `GATE-ARMVIAB`; §6.2; §5.6; §15.5; §14 rows **C-1 companion** and **I-6**.

§6 defines `GATE-ARENA` as *"two-sided band `majority + 0.02 ≤ acc ≤ 0.98` on `endpoint_std`
**and both real arms**"*. §5.6: *"Any HALT gate failing on either dataset in either lineage ⇒
**HALT**: no verdict, in either direction."*

§6.2 then argues, correctly and in the document's own words, that

> a plain one-sided HALT on the real arms would fire on exactly the outcome the falsifier
> exists to detect. If C06's premise is false, `displacement` in head space *should* sit near
> the majority rate — that is a **warranted CLOSE**, and a one-sided gate would convert it
> into a HALT, leaving C06 gated forever on an instrument that can never close it.

That argument applies verbatim to `GATE-ARENA`. The refinement in `GATE-ARMVIAB` is therefore
**inoperative**: on any run where a real arm falls below `majority + 0.02`, `GATE-ARENA`
HALTs unconditionally before `GATE-ARMVIAB`'s permissive branch can be reached. v2 avoids a
self-defeating gate in §6.2 and re-introduces the identical self-defeat one row above it in
the same table. §15.5 asks round 2 to rule on a refinement that, as the document stands, has
no effect.

This is also a **scope error against the precedent it cites**. C09's `GATE-ARENA`
(`C09_A0_V17_RECORD.md:1569-1572`) reads *"**Pooled native accuracy** must satisfy
`majority_rate + 0.02 ≤ acc ≤ 0.98` on both datasets"* — the **floor arm only**, never a
treatment arm. Round 1's **I-6** asked only that the **upper** bound be added (*"C06's
`GATE-VIABILITY` is one-sided; the **upper** bound is what catches a leak"*). Extending the
**lower** bound to the real arms is v2's own addition, unasked for and unpriced.

**Repair (one line).** Restrict `GATE-ARENA`'s **lower** bound to `endpoint_std` — C09's scope,
and the arm that measures instrument health rather than the hypothesis. Keep the `≤ 0.98`
**upper** bound on `endpoint_std` and both real arms, which is the leak catcher round 1 asked
for and which cannot fire on a warranted CLOSE. Leave the real arms' lower side entirely to
`GATE-ARMVIAB`'s two-case form.

### C-3. C01's **zero contract is not portable to the head space**, and four instruments that v2 imports from it — `GATE-ZEROMASK`, `GATE-DUALPATH`, `GATE-SHUFFLEFIX` and the Phase-5 leg — inherit a premise the head falsifies. The registered null becomes a **live retrieval neighbour in the control arms and a dead key in the real arm**, an asymmetry no gate detects.
*Attaches to:* §6 `GATE-ZEROMASK`, `GATE-DUALPATH`, `GATE-SHUFFLEFIX`; §8 Phase 5; §3.6.

`c01_a0_v2.json::zero_contract_v2` is explicit about what it assumes:
`input_state: exact_numeric_zero`, **`normalization_output_state: exact_numeric_zero`**,
`require_derived_mask_preservation: true`, **`require_null_absent_from_all_top20: true`**, and
`require_displacement_null_exclusion_dual_path_exact: true`. `prepare_views:1381-1386` enforces
the third by checking every arm's exact-zero rows equal `zero_mask`. All of that holds in the
raw space because a zero row survives `l2_rows` as a zero row.

In head space it does not hold, for the reason given in C-1. Concretely, on HateMM:

* the **derived-mask preservation** property is false by construction — the endpoint, common
  and rotation arms are non-zero at row 355 and the displacement arm is zero there;
* `require_null_absent_from_all_top20` fails as a *property*, not as a check: row 355's key in
  the control arms is an ordinary unit vector, so faiss `IndexFlatIP` will return it as a
  genuine top-20 neighbour for other queries, and `deployed_vote` (`mechfix_ops.py:74-95`)
  will weight its label into their votes. In the **`displacement`** arm the same row has a
  zero key, inner product `0` with everything, and contributes nothing;
* so an item whose features are a **known extraction failure** votes in every control arm and
  in no real arm — an asymmetry lying **exactly along the comparison that renders the
  verdict**, on the dataset that carries the null;
* `GATE-ZEROMASK` as written checks the **feature-space** exact-zero row set and that the two
  policies give identical head keys. Neither clause detects any of the above;
* `GATE-DUALPATH` requires the masked and physically-removed paths to *"agree exactly on every
  arm's predictions"*. They cannot: the masked path (if it could run at all, cf. C-1) has row
  355 in the bank for eleven arms; the removed path has it in none. The gate is therefore
  guaranteed to fail, or is being applied to something other than what it says.

**Repair.** Adopt C-1's repair — physically remove the registered null from the head-space
arena on HateMM, making `n = 743` the head-space population — and then: re-scope
`GATE-DUALPATH` to the **raw** leg only, where C01's dual-path equivalence is actually defined
and where I have no objection to it; restate `GATE-ZEROMASK` as a **feature-space** gate with
its head-space clause dropped (its second clause becomes vacuous once the row is removed);
re-derive Phase 5's role, since the removed path is now the main path and the sensitivity
question becomes its converse. `GATE-SHUFFLEFIX` remains correct and necessary on the raw leg.

---

## HIGH

### H-1. The **two-lineage disjunction** is a fourth multiplicity axis, left uncorrected, and defended with the exact argument round 1 rejected as **H-2**.
*Attaches to:* §5.3, §5.5; §14 row **H-2**.

§5.5 folds the two real arms into one Holm family of 46 hypotheses **per `(dataset, lineage)`**
— correct, and I verified the arithmetic (V5). It then writes: *"The two lineages are also a
conjunction for CLOSE (§5.3), which is conservative in the same direction."*

CLOSE is a conjunction; **SURVIVE is a disjunction**. §5.3 states it plainly: *"C06 survives if
it clears S1–S6 on **either** Head-N **or** Head-R."* Round 1's H-2 ruling was that a
conjunction *"controls error within an arm and says nothing about testing two arms and
reporting SURVIVE if either passes"*, and its ruling on the direction of "conservative"
attached the condition that it *"must not be allowed to excuse H-2: an uncorrected disjunction
is an arithmetic error, not a considered lean."* v2 accepted that at the arm level and then
committed the same error at the lineage level it introduced.

The direction favours SURVIVE, not CLOSE, so this cannot publish a wrong closure — but a false
SURVIVE is not free: it queues a `1.7–2.5 GPU-h` extraction proposal, and the document claims
a correction it has not applied on this axis.

**Repair.** Pick one and state it before freeze: (a) `α = 0.025` per lineage; (b) one Holm
family of `46 × 2 = 92` hypotheses per dataset spanning both lineages; or (c) declare the
lineage disjunction deliberately uncorrected, name the inflation (`≈ 2α` on the SURVIVE
event), and place it in §5.8's disclosure list rather than describing it as conservative in
§5.5.

### H-2. **Head-R has no banked anchor *and* does not run the banked mint script.** §3.3's *"shares every other component"* is contradicted by §7.2, and the missing anchor is worse than §15.6 states.
*Attaches to:* §3.3, §7.2, §11, §15.6.

§3.3: *"**Both lineages share every other component** — same wrapper, same fold contract, same
recipe CLI, same arm builder, same vote, same gates. **The only variable is the training
cache.**"*

§7.2 says otherwise: Head-N was minted with *"`headspace_mint.py` run **unmodified**, sha256
verified"*; Head-R with *"a **scratchpad harness** mirroring `c02_a0_mint.py`'s monkeypatch
structure"*. §11 lists `scripts/analysis/c06_falsifier_mint.py` as new code. So the **mint
driver is a second variable**, and on the lineage that has no banked output to reproduce it is
**new code**. `headspace_mint.py`'s fold-parity assertion (`:203-216`), its `dummy` construction
from the fitting pool (`:219-227`) and its `torch.load` test guard (`:106-116`) are properties
of the *frozen* script; on Head-R they are properties of a script nobody has reviewed.

This matters because of the **direction of the CLOSE conjunction**. CLOSE requires SURVIVE
false on **both** lineages. A degenerate Head-R yields SURVIVE-false for free and thereby
**satisfies half of CLOSE**, while a genuinely surviving Head-R is the only way it can block
one. A broken second lineage is therefore indistinguishable from a genuine negative and helps
close. `GATE-ARENA`/`GATE-ARMVIAB` catch a Head-R that *collapses toward the majority rate*;
they do not catch one that is healthy-looking but wrong (wrong cache, fold slip, wrong seed
semantics, different early-stop behaviour because it opens no dev file — §12).

**Ruling on §15.6: not sufficient**, and for a sharper reason than v2 gives — the claim it
rests on ("shares every component except the training cache") is false as written.

**Repair, and it supplies the anchor §15.6 asks for.** Make the Head-R mint a
**parameterisation of `headspace_mint.py`** — one driver, the training-cache path as its only
new argument — so that `GATE-FLOOR` anchors the *driver* for both lineages. If the driver must
stay separate, pre-register **6 additional native-cache mints through the Head-R driver** and
require them to reproduce the six banked `GATE-FLOOR` anchors at 4 dp on both metrics. That
anchors Head-R's harness on banked numbers without inventing a bar for the ro-trained head's
science, and at the measured full-train unit costs (`49.30`/`38.87 s`) it prices at
`3×49.30 + 3×38.87 = 264.5 s`, under 10 % of the projection.

### H-3. §10.2 does not disclose that the head-space arms are a **post-fusion, one-block analogue** of C01's per-modality two-block contrast — a different transform, forced by the head's architecture.
*Attaches to:* §3.4, §10.2.

C01 takes the contrast **per modality, before fusion**: `contrast_blocks` (`:1242-1270`)
computes `common[m] = l2(std[m] + ow[m])` and `displacement[m] = l2(ow[m] − std[m])` for
`m ∈ {img, text}`, and only then `fuse_modalities` / `paired_key`. The deployed head fuses
internally — `mlp(normalize(img_proj(·)) ⊙ normalize(text_proj(·)))`,
`src/model/classifier.py:115-124` — so a head-space arm can only be
`l2(h_ow ± h_std)`: a **post-fusion** contrast on a single 1024-d block.

Round 1's V9 established that the paired arms cannot be fed to the head at all, so the
one-block reading is **forced, not chosen** — this is a scope limit, not an error, and I am
not raising it as Critical. But §10.2 enumerates five things a CLOSE does not establish and
this is not among them, while §3.4's *"Instantiated with **two** blocks it *is* C01's
`prepare_views`"* points the reader the other way. A CLOSE will be read as "C01's battery,
re-run in the fold-head arena"; what it will actually have measured is a *post-fusion*
displacement, which is not the transform C01 scored `0.8505 / 0.8846`.

**Repair.** One bullet in §10.2 (*"…nothing about the per-modality contrast C01 measured: the
deployed head fuses image and text internally, so every head-space arm is a post-fusion
one-block analogue"*) and one clause in §3.4 acknowledging that the two-block and one-block
instantiations compute structurally different objects even though they share a code path.

---

## IMPORTANT

### I-1. Disposition audit: **round-1 I-3 is claimed `ADOPTED` but only two of its three legs are repaired.**
*Attaches to:* §8 Phase 1c/1d; §7.2; §14 row **I-3**.

Round 1's I-3 named three per-process loops: **(a)** the ro-cache `torch.load`, **(b)** the
per-process `GATE-SHA`, and **(c)** *"wrapper import cost above `headspace_mint.py`'s measured
`40.39 s`"*. v2 counts (a) as Phase 1c and moves (b) to the driver as Phase 1d. **(c) is not
enumerated anywhere**, and §7.2 never says whether its *"measured wall"* figures are
full-process wall or in-process elapsed. Measurement (ε): interpreter plus
`torch/numpy/faiss/sklearn` import costs **3.02–3.40 s** on this node. If not already inside
the mint unit, `66 × ≈3.2 s ≈ 211 s` = **7.4 %** of the 2867.1 s total — the F118 under-count
pattern at material scale.
**Repair:** state in §7.2 that the units are full-process wall and how they were timed, or add
a Phase 1e line at a measured per-process startup unit × 66.

### I-2. §8 Phase 5 substitutes a **ratio for an enumeration**, which `rule_1_compute_projection` forbids in terms, and it under-counts.
*Attaches to:* §8 Phase 5; §2 R1.

`rule_1_compute_projection` requires *"the per-unit cost measured on the real path at the real
scale … multiplied through an **enumerated count of every unit the run will actually
execute**"*. Phase 5 instead writes `(4.2 + 9.0 + 273.6 + 11.6) / 2 = 149.2 s`, which assumes
HateMM is exactly half the two-dataset cost. HateMM is the **larger** dataset (`n = 744` vs
`579`) and its own measured mint unit is larger (`40.39` vs `34.40 s`), so its share exceeds
half; at the `744 / 1323 = 56.2 %` row-count ratio the line is ≈ `167.7 s`, an under-count of
≈ `19 s`. Immaterial to the total, categorical against the rule, and every constituent count
is enumerable (HateMM-only: `4 × 30` and `9 × 30` at `U2a`/`U2b`; `4 × 5` and `9 × 5` at
`U2c`/`U2d`; `256 × 3 × 2` at `U4`; `23 × 2` at `U3`).
**Repair:** enumerate Phase 5 the way every other phase is enumerated.

### I-3. §7's unit table does not attribute units to a **dataset**, so §8's per-dataset phases cannot be audited for scale.
*Attaches to:* §7.6, §7.7; §8 Phases 2, 2R, 3, 4, 5.

`U2a`–`U2d`, `U3`, `U4` and `U6` are given as single numbers with no dataset named, while
`U1` and `U9` are explicitly attributed (§8 Phase 1b: *"the `U1` unit is HateMM-scale and
applied to MHC-ZH too — conservative"*; `U9` is split `3.70 / 3.49`). Every vote and null-draw
unit scales with `n`, and I-2's Phase 5 turns on exactly this attribution.
**Repair:** label each unit with the dataset it was measured on, or state the
conservative-application convention once and apply it uniformly to all of them.

### I-4. §8 counts head-space arm **votes** but not head-space arm **construction**.
*Attaches to:* §8 Phases 1b, 2.

Phase 1b counts head *forwards*; Phase 2 counts *votes* at `U2a`/`U2b`; the raw arm build is
inside `U5` (Phase 2C) and the null-draw rebuild inside `U4` (Phase 3). Nothing counts building
the **13 head-space arms once per `(dataset, seed, lineage)`** = 12 cells = **156 builds**.
From `U4`'s own residual (`0.04234 s` for 2 arms at `n = 744`) the magnitude is ≈ `3 s` —
immaterial to the total, but it is an uncounted loop in §8 and round 1 was told to hunt for
exactly this class.
**Repair:** add a Phase 2b line at a measured per-arm build unit × 156.

### I-5. `GATE-ZEROOP`'s *identical predictions* form can fail on a healthy run, and it is **not** "strictly stronger" than `GATE-ALGEBRA` — the two are logically independent.
*Attaches to:* §6 `GATE-ZEROOP`, `GATE-ALGEBRA`.

The θ=45 identity is **not exact**: measurement (β) gives `1.192e-07` (HateMM) / `8.941e-08`
(MHC-ZH) on the raw keys, and the same `cos45 − sin45 = 1.11e-16` asymmetry (§7.4) recurs on
the head keys. A `~1e-7` key perturbation can reorder a top-20 neighbourhood whenever two
neighbours' signed cosines lie closer than that — which is precisely why C01 gated keys at
`2e-6` rather than asserting equality. So `GATE-ZEROOP` has a real false-HALT probability on a
correct run. Separately, identical predictions does not imply `≤ 2e-6` keys and `≤ 2e-6` keys
does not imply identical predictions; the value of v2's arrangement is the **conjunction** of
the two, not that one dominates.
**Repair:** keep both gates; correct the *"strictly stronger"* wording; and pre-register a tie
diagnostic — on a `GATE-ZEROOP` mismatch, emit the number of affected items whose 20th/21st
neighbour similarities differ by less than the measured `GATE-ALGEBRA` residual, and
pre-register that a mismatch confined to such items is **reported**, not HALTed.

### I-6. **Absence**, as distinct from non-finiteness, is not named as a HALT trigger, and `GATE-LEDGER`'s process-count leg is *declared* rather than binding.
*Attaches to:* §5.6; §6 `GATE-LEDGER`; §12.

§5.6's finiteness mandate closes the NaN path and is the right response to round 1's note. The
residual path is a lineage, dataset or process that produces **no value at all**: SURVIVE is
then vacuously false for it, and CLOSE requires only that SURVIVE be false on both lineages —
so a silently missing Head-R contributes half of CLOSE (cf. H-2). §12's ledger books
*"processes reporting: **66 mints + 6 fidelity + 1 arena**, all reporting"*, but §6 names only
*"dev-or-test labels into any decision quantity `= 0`"* as `GATE-LEDGER`'s **binding** leg.
**Repair:** make the process-count predicate binding (HALT on any mismatch against the declared
66 / 6 / 1), and add one sentence to §5.6: an **absent** decision or gate quantity HALTs on the
same footing as a non-finite one.

### I-7. `ρ*` is a **single cross-dataset bar**, which contradicts §3.1's stated within-dataset discipline and is looser on HateMM than the data supports.
*Attaches to:* §3.1, §6.1.

§3.1: *"**No cross-dataset comparison of absolute numbers is made**; every decision quantity is
a within-dataset, within-seed, within-lineage arm comparison."* §6.1 then sets one global
`ρ* = 0.9772` as the max over **13 arms × 2 datasets**, and `0.9772` is a **MHC-ZH**
measurement (`endpoint_std`, `0.977223`; V8). HateMM's own maximum is `0.968176`. So HateMM's
HALT bar is set by the other dataset, and is `0.0090` looser than its own data supports — on
the dataset that carries the registered null and the whole Phase-5 leg.
A HALT bar is not literally a "decision quantity", but it decides whether a verdict publishes.
**Repair:** freeze **per-dataset** `ρ*` — HateMM `0.9681`, MHC-ZH `0.9772` (truncated down at
4 dp from the values I reproduce in V8) — which costs nothing, keeps the calibration
label-free and banked, and restores §3.1's discipline. See also the §15.2 ruling in Part C.

---

## MINOR

* **M-1.** §8 Phase 3 records `273.6 s` where `3072 × 0.08908 = 273.654`; every other product
  in §8 is rounded, not truncated. Use `273.7` and re-add (total `2867.2 s`; nothing else
  moves at the stated precision).
* **M-2.** §5.1 gives `displacement` five comparators and omits `common_displacement` from its
  set, while `common_displacement` is required to beat `displacement` (H-1's repair). C01
  froze a comparator list only for its primary, so this is not a violation — but the asymmetry
  eases SURVIVE for one of the two disjuncts and belongs in §5.8's disclosure list rather than
  being silent.
* **M-3.** §7.8 says the dry-check overrun is *"raised now rather than afterwards"*, but the
  ≈ 30 CPU-minutes were already spent when the sentence was written; what changed from v1 is
  the framing, not the sequence. The disclosure is honest and the burn is `$0`, so this is
  wording only — say "disclosed at the same time as the result" rather than "raised rather
  than resolved silently".

---

# PART C — REQUIRED RULINGS

## The sharpest question (§3.A / deliverable 5): does bit-exact parity on the **two-block** instantiation constrain the **one-block** path that renders the verdict?

**Partially — more than round 1 could have hoped, less than §3.4 claims, and the residual is
not the one the question anticipates.**

**What it does constrain, and this is most of the surface.** The two-block parity pins
`l2_rows` itself, concatenation order within and across blocks, the per-modality contrast
definitions (`common = l2(std + ow)`, `displacement = l2(ow − std)`,
`common_interaction = l2(common ⊙ displacement)`), the Givens mixing and its angle convention,
the arm-name→formula mapping, the `float32` dtype, and the θ=0 / θ=45 identities — all of
which are **shared, block-count-independent code**. That is a real anchor and a large advance
on v1, where nothing outside the re-implementation checked the re-implementation. My own
independent reconstruction of §3.4's spec landing bit-exact (measurement α) is further
evidence the shared helpers are fully and unambiguously specified.

**What it does not constrain: the one operation the repair restored.** Measurement (γ): at one
block, `fuse([b]) = l2(l2(b))` and `paired(A,B) = l2(l2(concat[l2 A, l2 B]))`. The outer
`fuse` normalisation — the exact defect round 1's C-3 named, *"missing `fuse_modalities`' outer
per-block normalisation"* — is a re-normalisation of an already-unit vector, differing from
v1's rejected `pair` by `7.451e-09`, one sixteenth of a float32 eps. **In the instantiation
that renders the verdict, C-3's numerical repair is a no-op.** What v2 actually fixed for the
head-space arms is the **dtype** (`float64` → `float32`), which is real but is not what §3.4
says was wrong.

**Is the load-bearing choice "relocated into the shared helper"?** No — it is *eliminated*, by
the head rather than by the design. `classifier_hateClipper` emits a single fused 1024-d
vector (round 1's V9), so there is no two-block head reading available to get wrong. The block
count is forced. To that extent §3.4's *"derived, not chosen"* is right, though for a reason it
does not give.

**But a different load-bearing choice is relocated, and the gate does not touch it: *where the
contrast is taken*.** C01 contrasts per modality before fusion; the head-space battery
contrasts after fusion. Bit-exactness on two blocks says nothing about that, because the two
objects are not defined on a common domain. That is H-3, and it belongs on the verdict face.

**Net ruling.** `GATE-C01PARITY` is a genuine and worthwhile fidelity anchor and should be
kept exactly as specified for the raw leg. It does **not** discharge C-3 as §3.4 and §14 state,
for two independent reasons: its distinguishing content is vacuous at one block, and — C-1 —
the one-block path cannot execute through the anchored code at all on HateMM. Two sentences fix
the overstatement; C-1 needs a design change.

## §15.1 — the arena reading

**Agreed, no open question on the arena itself.** Round 1's ruling stands and v2's
two-lineage construction removes the training-set half of it. One consequence is reopened,
though not about the arena's *definition*: **which rows constitute it**. C-1 and C-3 mean the
head-space arena's population on HateMM is currently unspecified (744 masked, or 743 removed),
and the raw leg, the head leg and `GATE-ARMVIAB`'s raw-vs-head comparison must all be on the
same rows for the gate discriminators to mean anything. Fix that in §3.6 and §5.1.

## §15.2 — is the max the right calibration statistic for `ρ*`?

**The max is correct as an order statistic; the pooling across datasets is not.**

`ρ* = 0.977223 → 0.9772` reproduces exactly (V8), and it is the maximum over 13 arms × 2
datasets. A max is the most permissive choice, and v2 is right to make it — but the reason it
gives (that these arms bound "what still informative looks like") is weaker than the one
available. The decisive reason is **structural**: `GATE-ORBITDISP` is an *instrument* gate,
not a decision gate. Its job is to fire only when the head space is **more degenerate than
anything the raw feature family produces**. A quantile bar would HALT on head-space arms whose
concentration sits inside the observed raw range — converting ordinary head-induced
concentration into `INSTRUMENT_INCONCLUSIVE`, which is the same self-defeat as C-2. Under a
gate whose only job is to separate "instrument destroyed the object" from "the object is not
there", the max is the right statistic and a per-arm bar would be worse still (it would let a
head-space arm be arbitrarily concentrated so long as its raw counterpart was).

Two conditions on that ruling. First, **record the runner-up** — `0.9697` (`common`, MHC-ZH) —
so the permissiveness is on the face of the document rather than implicit. Second, **freeze
`ρ*` per dataset**, not pooled: **I-7**.

## §15.3 — per-arm retraining stays excluded

**Agreed, and the condition round 1 attached is met.** Round 1 made the exclusion conditional
on C-2 being repaired by a gate; v2 repairs it by a gate (`GATE-ARENA` on `endpoint_std`,
`GATE-ARMVIAB`) **and** by an in-domain lineage. Head-R is not per-arm retraining — one head
per lineage, arms built after — so the excluded variant remains genuinely excluded, and its
price re-multiplies correctly at `195 × 40.39 + 195 × 34.40 = 14 584.05 s = 4.051 h` (V4).

## §15.4 — L28 dropped

**Agreed and verified clean.** I grepped v2 for every occurrence of `L28`: nine hits, all of
them statements that the leg is dropped, the manifest entry removed, the scope exclusion, the
A5 row, or the §14 disposition row. **No decision rule, gate, projection line, arm or ledger
predicate references L28.** No dangling reference remains.

## §15.5 — `GATE-ARMVIAB`'s two-case form (the one refinement)

**I endorse the refinement over the one-sided form, on exactly the reasoning §6.2 gives**, and
I would have made the same call. A one-sided majority-rate HALT on the real arms fires on the
falsifier's own predicted outcome and would leave C06 gated on an instrument that can never
close it. The two-case form keeps round 1's intent — the real arms are no longer unwatched —
while using the raw space as the discriminator, which is where round 1's own C-1 argument put
the reference. That it rests on a logical argument rather than on measured raw accuracies is
**correct discipline**, not a weakness: those accuracies are decision-relevant inputs and
measuring them before freeze is exactly what §7.3's blindness rule forbids. A gate that
discriminates between two instrument states does not need its own outcome measured in advance.

**However, the refinement is currently inoperative**, because `GATE-ARENA` HALTs
unconditionally on the same condition one row above it — **C-2**. The refinement is not the
problem; its sibling is. Keep §6.2 verbatim and fix `GATE-ARENA`.

## §15.6 — Head-R's missing floor anchor

**Not sufficient — see H-2.** The claim the sufficiency argument rests on ("shares every
component except the training cache") is contradicted by §7.2, which shows Head-N running the
sha-verified frozen `headspace_mint.py` and Head-R running new code. And the CLOSE conjunction
points the wrong way for instrument failure: a degenerate Head-R yields SURVIVE-false for free
and thereby **supplies half of CLOSE**, so a broken second lineage is indistinguishable from a
genuine negative and *helps* close.

**A banked object is available.** Head-R's *science* cannot be anchored — there is no banked
ro-trained head, and inventing an accuracy bar would be the un-preregistered threshold §6.3
rightly refuses. But Head-R's *harness* can be: require the Head-R driver, handed the **native**
cache, to reproduce the six banked `GATE-FLOOR` anchors at 4 dp on both metrics. Best done by
making Head-R a parameterisation of `headspace_mint.py` so `GATE-FLOOR` anchors one driver for
both lineages; failing that, six extra native-cache mints through the Head-R driver, priced at
`264.5 s` from v2's own measured full-train units.

## §5.8 — the one C01 condition deliberately not carried

**The reasoning is sound and I endorse it.** I verified the comparator: `c01_a0_v2.json`
carries `deployed_r0_accuracy_context_only` = `0.8504672897196262` (HateMM) /
`0.8589743589743589` (MHC-ZH) inside a block literally named
**`inputs.datasets.<ds>.historical_strict_devtrain`**, sourced from
`refine-logs/READOUT_SCREEN_OUT.json`. It is a raw dev-arena figure at `n_dev` 107/78 from a
different protocol, and importing it as a bar for a CPU fold-head train-OOF arena would breach
F88's *"a CPU-trained arm must be paired against a CPU-TRAINED FLOOR"*. **Inapplicable across
arenas, not waived for convenience** — v2's characterisation is exact, and the `GATE-FLOOR`
anchors are the right in-arena substitute. The disclosure is also correctly located, in the
section round 1's I-5 asked for.

## §5.2 S6 — the net-fixes direction

**Correct.** `c01_a0_v2.json::retrieval.fix_break_reference = "endpoint_std"`, and
`c01_policy_contrast_a0.py:1725` reads
`reference = evaluations[config["retrieval"]["fix_break_reference"]]["predictions"]`. S6's
*"net fixes ≥ 3 (HateMM) / ≥ 2 (MHC-ZH) **against `endpoint_std`**"* is C01's own reference arm
and its own frozen minima (`decision.minimum_net_fixes`, V9). Importing it also brings the
battery into the currency the Gate-0 record demands, as §5.2 argues.

## §3.D — gates and scope

**Count.** §6 lists **19** rows, of which `GATE-DOMAIN` and `GATE-DEVFID` do not gate, so **17
HALT gates** against C09's nine. `GATE-DOMAIN` is a gate in name only — but v2 says so
explicitly in §6.3, and I rule that **acceptable**: C-2(a) is substantively discharged not by
`GATE-DOMAIN` but by `GATE-ARENA`'s lower bound on **`endpoint_std`**, which *is* a
head-space input-domain fidelity HALT gate anchored on a banked number (the majority rate).
Refusing to invent a recovery-fraction bar is the right call and consistent with how this
house treats un-preregistered thresholds. What is **not** acceptable is extending that same
lower bound to the real arms — C-2.

**Against C09's nine:** `GATE-FLOOR` ✓ (both metrics, H-5); `GATE-PARITY-FOLD` folded into
`GATE-FLOOR`'s *"every `fold_acc_deployed` entry"* ✓; `GATE-FIXK20` has no object (`k = 20`
throughout, and `mechfix_ops.deployed_vote`'s `topk` default is the frozen `TOPK`);
`GATE-BLIND` has no object (no learned features); `GATE-LEDGER` ✓ (H-4 repair verified against
C09's `:1549-1554` shape); `GATE-NESTED` ✓; `GATE-SELFTEST` — C09's is `net_s == n · Δacc_s`,
and now that S6 imports net fixes (I-5 repair) **C06 does have an object for it**: the
identity `net(A) = n · (acc(A) − acc(endpoint_std))` should hold exactly for every arm, seed,
dataset and lineage, and is free. Round 1 correctly said C06 had no analogue *because* it
dropped net items; adopting S6 restores the object. Worth adding, but I am not raising it
separately — it is a free strengthening, not a defect. `GATE-ZEROOP` ✓ with I-5's caveat;
`GATE-ARENA` ✓ in existence, wrong in scope (C-2).

**Unfalsifiable or redundant gates:** none unfalsifiable. `GATE-ALGEBRA` is subsumed in
practice by nothing (I-5: the two are independent, keep both). `GATE-DUALPATH` as written is
unsatisfiable in head space (C-3).

**Able to fire on a warranted CLOSE:** `GATE-ARENA` (C-2), and `GATE-ZEROOP` on a float32 tie
(I-5).

**§10.2's scope sentence** now says more than v1's and correctly names both lineages and the
`GATE-DOMAIN` recovery fraction. It is still missing the post-fusion contrast (H-3). With H-3
added and C-1/C-3 resolved, it would say everything a CLOSE is scoped to.

**Hard constraints: none touched.** No OCR, no cross-dataset mixing of training data
(§3.1's conjunction-of-independent-verdicts structure holds — note that `ρ*` is nonetheless a
cross-dataset *bar*, I-7), no external API, single-dataset train split, parent-video binary
label only, no ensemble (`avg_score` is C01's frozen `gain_control` serving as a control), no
size scaling, SLURM-only. §10.4's ban analysis and M-2's addition stand as round 1 ruled.

## §3.C — the process rules

**`rule_1_compute_projection`.** The **H-6 repair is real and reconciles**: `5 × 0.00305 +
5 × 0.00629 = 0.04670 s` of votes against `U4 = 0.08908 s`, leaving `0.04238 s` for the arm
rebuild — positive and plausible (v2 records `0.04674` and `0.04234`; my recomputation differs
in the fourth decimal by rounding only). The stated cause is **consistent with the source**:
`mechfix_ops._norm32:39` is `np.ascontiguousarray(np.asarray(X, dtype="float32"))`, so a
`float64` arm matrix pays a conversion copy on every `deployed_vote` call, and
`c01_policy_contrast_a0.l2_rows:1200-1202` returns `astype("float32")`, so the real battery's
arms are `float32` by construction. The withdrawal-and-re-measurement is the right response.

**Hunting for a loop §8 still does not count**, as instructed: I found **one uncounted loop**
(head-space arm construction, **I-4**), **one leg of a round-1 finding still uncounted**
(per-process startup, **I-1**), **one phase derived by ratio rather than enumeration**
(**I-2**), and **one auditability gap that blocks checking the rest** (unattributed unit scale,
**I-3**). None of them moves the total by more than ~8 %, and the projection's shape —
two figures, sensitivity at 2×/5×, the self-imposed "2× over conservative is itself a
reportable finding" — is right.

**`rule_2_heartbeat`.** §9 closes round 1's `61.6 s` gap correctly, and I checked for any
remaining interval longer than the stated bound: the longest un-instrumented span is
`GATE-C01PARITY` at `11.27 s` per dataset (`14.1 s` under the `×1.25` factor), well inside;
Phase 3 emits every 32 draws = `2.85 s`; Phase 2's whole 12-block span is `4.2 s`. With the
epoch line inside the mints, **no interval exceeds ~15 s**. §9's list of what the code lineage
must verify is **complete** as far as a design review can determine — it carries all five items
round 1 named. One free addition worth making: that the HALT path names *which* gate failed in
its final line, so a HALT is distinguishable from a crash without reading the JSON.

**§7.7's `U9` correction.** **Sound.** Recording a crashed process's `echo` exit status as a
measurement is the F118 failure mode in miniature, and re-running with `--seeds 0` and
confirming both datasets exit `0` and write their JSON is the right fix; multiplying the
per-`(dataset, seed)` unit by 3 is unit × count, as the rule requires. **Could another §7 unit
have the same defect?** The mints are corroborated by fold parity passing in all seven; the
parity, algebra-guard and `ρ` units are corroborated because **I independently reproduced their
outputs** (measurements α, β, V8), so those processes certainly ran. `U2a`–`U2d`, `U3`, `U4`
and `U8` have no independent corroboration and no stated exit-status discipline — that is
**I-3**'s territory rather than a separate finding, but the freeze should record how each unit
was timed.

## §3.E — execution and honesty

**§13 is sufficient without the withdrawn reason.** Two independent legs remain and either
would decide it: CLAUDE.md's standing SLURM rule together with C01's own frozen
`execution.require_slurm = true, cpu_only = true, required_cpus = 8`, and the cloud route's
inapplicability because `GATE-FLOOR` anchors to six floors measured locally on `foscsmlprd01`
— under the standing same-table-same-hardware ruling, moving to cloud would require re-minting
all six there, costing more than the job. Withdrawing *"at 44 min this is not long-running"* is
correct: it is exactly the kind of projection F118 says may not be trusted for routing until
measured.

**§7.8's disclosure.** The trade is right — a standing `TARGET_STATE.json` rule beats a task
brief's CPU cap, and the overrun is `$0`, zero-GPU, on a 64-core node. The seven mint units
were unavoidable under R1 and the Head-R units did not exist before v2 required them. Only the
framing is off (**M-3**).

**Does v2 claim a repair the artifact does not contain?** **Yes, in two places, both named
above:** §3.4/§14's C-3 adoption (the anchored code path cannot execute on the verdict path —
**C-1** — and its distinguishing content is vacuous at one block), and §14's I-3 adoption
(leg (c) unrepaired — **I-1**). A third, §14's "C-1 companion ADOPTED, REFINED", is true as
written but **inoperative** because of C-2.

---

# PART D — DISPOSITION AUDIT

**20 of 23 verified as truly adopted.** Each verified row was checked against the primary
source, not against §14.

| finding | v2 claim | audit result |
|---|---|---|
| **C-1** dispersion not magnitude | ADOPTED | **VERIFIED** — `GATE-ORBITSCALE` deleted; `GATE-ORBITDISP` present; bar `0.9772` reproduces (V8); two-case split matches round 1's prescription; applied to all 13 arms. Bar pooling is I-7. |
| **C-1 companion** both real arms clear majority | ADOPTED, REFINED | **NOT ADOPTED IN EFFECT** — the refinement is sound and I endorse it (§15.5 ruling), but `GATE-ARENA`'s lower bound on the real arms HALTs unconditionally on the same condition, making it unreachable. **C-2.** |
| **C-2** OOD transplant; C02 precedent miscited | ADOPTED (a,b,c) | **VERIFIED** — (a) `GATE-ARENA` on `endpoint_std` is a banked-anchored head-space fidelity HALT gate; (b) Head-R on **both** datasets, exceeding round 1's "at least one"; (c) §10.2 rescoped. Precedent withdrawal is correct and complete (V10). New defect in *how* Head-R is built: **H-2**. |
| **C-3** unanchored arm algebra | ADOPTED | **NOT ADOPTED ON THE VERDICT PATH** — two-block parity is real and bit-exact (α), but the one-block path cannot execute through the anchored code (**C-1**) and the restored normalisation is vacuous there (γ). |
| **H-1** `displacement` missing as comparator | ADOPTED | **VERIFIED** — 6 for the primary, 5 for `displacement`, from `gain_controls` ∪ `primary_vs_controls` (V5). |
| **H-2** disjunction uncorrected | ADOPTED | **VERIFIED at the arm level** — 46 hypotheses, arithmetic checked. Re-committed on the new lineage axis: **H-1**. |
| **H-3** bootstrap unit unstated | ADOPTED | **VERIFIED** — §5.4 states the unit; Phase 4's 92 re-derives from it and re-multiplies (V4). |
| **H-4** dev-label sentence false | ADOPTED | **VERIFIED** — sentence withdrawn; ledger has C09's declared-count shape; code claims confirmed (V12). |
| **H-5** `GATE-FLOOR` accuracy only | ADOPTED | **VERIFIED** — all six macro-F1 anchors present and equal to the banked `mF1_deployed` (V6). |
| **H-6** `U4` below its constituents | ADOPTED | **VERIFIED** — reconciles with a positive residual; cause consistent with `_norm32` and `l2_rows` (Part C). |
| **I-1** dev ro-forwards counted, never opened | ADOPTED | **VERIFIED** — 174 re-derived; native `dev_seen` in §11 with matching digests, covered by `GATE-SHA`. |
| **I-2** Phase 6 scope; budgets inside a projection | ADOPTED | **VERIFIED** — `U9` measured per `(dataset, seed)` × 3; the `30 s` slack is explicitly outside the sum. |
| **I-3** three uncounted per-process loops | ADOPTED | **PARTIAL — 2 of 3.** (a) and (b) repaired; **(c) wrapper/interpreter import cost is not enumerated and §7.2 does not say the units include it.** Measured 3.02–3.40 s × 66 ≈ 211 s. **I-1.** |
| **I-4** gate value on an excluded population | ADOPTED | **VERIFIED** — §7.5 restated with the mask applied (`0/743`), consistent with round 1's own measurement (b). |
| **I-5** five C01 conditions dropped silently | ADOPTED | **VERIFIED** — four restored (S5 both arms, shuffle Holm, S6 net fixes against the correct `fix_break_reference`, `GATE-SMALLDISP`); the fifth disclosed with reasoning I endorse (Part C). |
| **I-6** three C09 gates missing | ADOPTED | **VERIFIED in existence** — `GATE-NESTED`, `GATE-ZEROOP`, `GATE-ARENA` all added. `GATE-ARENA` over-scoped beyond both C09 and round 1's request: **C-2**. `GATE-ZEROOP` caveat: **I-5**. |
| **I-7** shuffle fixed-point guard | ADOPTED | **VERIFIED** — `GATE-SHUFFLEFIX` present; `zero_contract_v2.require_fixed_null_in_shuffle: true` and `shuffle_fixed_point_bijection` confirmed in `required_halt_only_validity_guards`. |
| **I-8** L28 not a sibling orbit | ADOPTED | **VERIFIED** — leg dropped; grep clean, no dangling reference (§15.4 ruling). |
| **I-9** C01 v4 artifacts absent | ADOPTED | **VERIFIED** — both v4 digests present and matching disk; both chains traced in config **and** in source sha-gates (V3). |
| **I-10** untrained contraction as evidence | ADOPTED | **VERIFIED** — §5.7 withdraws the clause and says why. |
| **M-1** `n_train` header | ADOPTED | **VERIFIED** — §3.1 names `expected.train.n`. |
| **M-2** F70's two cells enter as controls | ADOPTED | **VERIFIED** — §10.4. |
| **M-3** wrong constant for shuffle p95 | ADOPTED | **VERIFIED** — S5 cites `require_primary_and_displacement_above_shuffle_p95`. |
| **M-4** `c02_a0_mint.py` line numbers | ADOPTED | **VERIFIED** — `:69-71` unwrapping, `:72-76` assertions, confirmed in source. |

---

# PART E — MINIMAL SET OF CHANGES THAT WOULD EARN GO

1. **C-1** — pre-register the head-space null handling. Recommended: make the physically-removed
   path (`n = 743`) the **main** head-space path on HateMM, so the imported `l2_rows` runs
   unmodified on all thirteen arms; state the population of the raw leg, the head leg and the
   floors explicitly so all gate comparisons are on the same rows.
2. **C-2** — restrict `GATE-ARENA`'s **lower** bound to `endpoint_std`; keep `≤ 0.98` on all
   three; leave the real arms' lower side to `GATE-ARMVIAB`.
3. **C-3** — re-scope `GATE-DUALPATH` to the raw leg; drop `GATE-ZEROMASK`'s head-space clause;
   restate Phase 5's role once the removed path is primary.
4. **H-1** — correct or explicitly declare the lineage-disjunction multiplicity.
5. **H-2** — anchor Head-R's harness on the banked floors (parameterise `headspace_mint.py`, or
   six native-cache mints through the Head-R driver); correct §3.3's "only variable" sentence.
6. **H-3** — add the post-fusion-contrast scope bullet to §10.2 and the matching clause to §3.4.
7. **I-1 … I-7** as written, including per-dataset `ρ*` and the binding process-count ledger leg.
8. **M-1 … M-3** — arithmetic and wording.

**Can the falsifier still discharge the written condition at `$0`?** **Yes.** Nothing I found
requires GPU, extraction or new data. C-1 and C-3 are resolved by a choice of population that
the battery already computes for other reasons; C-2 is a one-line scope restriction; H-2's
anchor costs `264.5 s` inside a `47.8`-minute projection. The instrument, once these are fixed,
does measure what `falsifier_spec` asks — with H-3's scope caveat on the face of the verdict.

---

*Read-only review. No GPU, SLURM, Modal, model load, head training, arena run, cache write,
test-split access, job submission or commit occurred. `TARGET_STATE.json`, both drafts and all
configs were read and not modified. All computation was numpy/torch-CPU re-derivation on the
already-banked **train-split** caches plus `sha256sum`. A GO on this lineage would authorise
nothing to run: the design would still require freeze with hashes, a **separate** independent
code/resource review lineage over the executable reaching its own `0C/0H/0I`, and
main-dialogue authorization.*
